package main

import (
	"container/heap"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"sync"

	"gonum.org/v1/gonum/stat/distuv"
)

type Model struct {
	ArvRates      []float64 `json:"arv_rates"`
	ServTimes     []float64 `json:"serv_times"`
	ServCVs       []float64 `json:"serv_cvs"`
	Deadlines     []float64 `json:"deadlines"`
	MemDemands    []int     `json:"mem_demands"`
	Memory        int       `json:"memory"`
	QueueCapacity int       `json:"queue_capacity"`
	QueuePolicy   string    `json:"queue_policy"`
	InitTimes     []float64 `json:"init_times"`
	MAPs          []string  `json:"markovian_arrival_processes"`
	NetOverhead   float64   `json:"net_overhead"`
}

const (
	QUEUE_FIFO = "fifo"
	QUEUE_EDF  = "edf"
	QUEUE_SJF  = "sjf"
)

type Results struct {
	Utility     []float64
	AvgRT       []float64
	Completions []int64
	Arrivals    []int64
	ColdStarts  []int64
	Time        float64
}

const (
	EV_ARRIVAL    = iota
	EV_COMPLETION = iota
)

type Event struct {
	t      float64
	evType int
	fun    int
}

type QueuedReq struct {
	arrivalT float64
	fun      int
	priority float64
}

func (e Event) String() string {
	var descr string
	if e.evType == EV_ARRIVAL {
		descr = "arv"
	} else {
		descr = "cmpl"
	}
	return fmt.Sprintf("[%f](%s-%d)", e.t, descr, e.fun)
}

type containerPool struct {
	Busy int
	Warm int
}

func totalReclaimableMemory(pools []containerPool, model *Model) int {
	sum := 0
	for i, m := range model.MemDemands {
		sum += pools[i].Warm * m
	}
	return sum
}

func reclaimMemory(pools []containerPool, toReclaim int, model *Model, sortedFunIndices []int) int {
	reclaimed := 0
	for reclaimed < toReclaim {
		for _, i := range sortedFunIndices {
			if pools[i].Warm == 0 {
				continue
			}
			pools[i].Warm--
			reclaimed += model.MemDemands[i]
			break
		}
	}
	return reclaimed
}

func canExecute(pools []containerPool, f int, availMem int, model *Model) (bool, bool, bool) {
	// warm
	if pools[f].Warm > 0 {
		return true, false, false
	}
	// cold
	if availMem > model.MemDemands[f] {
		return true, true, false
	}
	// cold, reclaiming
	if totalReclaimableMemory(pools, model)+availMem > model.MemDemands[f] {
		return true, true, true
	}
	return false, false, false
}

func simulate(model Model, maxArrivals int64, utilityStoppingPercDelta float64, seed int64) (*Results, error) {
	t := 0.0
	h := &EventHeap{}
	heap.Init(h)
	n := len(model.MemDemands)

	// MAP Parsing
	MAPs := make([]*MAP, 0)
	MAPstates := make([]int, len(model.ArvRates))
	if model.MAPs != nil && len(model.MAPs) > 0 {
		for i, mapstring := range model.MAPs {
			m, err := NewMAPFromString(mapstring)
			if err != nil {
				log.Fatal(err)
			}

			// Rescale rate
			m.Rescale(model.ArvRates[i])

			MAPs = append(MAPs, m)
			MAPstates[i] = -1
		}
	}

	sortedFunIndices := make([]int, len(model.MemDemands))
	for i := range model.MemDemands {
		sortedFunIndices[i] = i
	}
	sort.Slice(sortedFunIndices, func(i, j int) bool {
		return model.MemDemands[sortedFunIndices[i]] > model.MemDemands[sortedFunIndices[j]]
	})

	var queue Queue
	switch model.QueuePolicy {
	case QUEUE_FIFO:
		queue = NewFCFSQueue()
	case QUEUE_EDF, QUEUE_SJF:
		queue = NewPriorityQueue()
	default:
		return nil, fmt.Errorf("unsupported queue policy: %s", model.QueuePolicy)
	}

	availMem := model.Memory
	containerPools := make([]containerPool, n, n)
	utilityProb := make([]float64, n, n)
	completions_within_deadline := make([]int64, n, n)
	completions := make([]int64, n, n)
	arrivals := make([]int64, n, n)
	meanRT := make([]float64, n, n)
	coldStarts := make([]int64, n, n)

	var stoppingDelta []float64
	nextStopCheckT := -1.0
	if utilityStoppingPercDelta > 0.0 {
		nextStopCheckT = -1.0
		stoppingDelta = make([]float64, n, n)
	}

	var totArrivals int64 = 0

	r := rand.New(rand.NewSource(seed))

	// Schedule first arrivals
	for f := 0; f < n; f++ {
		if model.ArvRates[f] <= 0.0 {
			continue // no arrivals
		}
		var at float64
		if len(MAPs) == 0 {
			at = t + r.ExpFloat64()/model.ArvRates[f]
		} else {
			iats, state, err := MAPs[f].Sample(1, MAPstates[f], r)
			if err != nil {
				log.Fatal(err)
			}
			MAPstates[f] = state
			at = t + iats[0]
		}
		heap.Push(h, Event{at, EV_ARRIVAL, f})
	}

	for h.Len() > 0 {
		// Stopping check
		if t > nextStopCheckT {
			nextStopCheckT += 10000
			// Compute utility diff
			stop := true

			for i := 0; i < n; i++ {
				uNew := max(0.000001, float64(completions_within_deadline[i])/float64(arrivals[i]))
				stoppingDelta[i] = math.Abs(uNew-utilityProb[i]) / utilityProb[i]
				utilityProb[i] = uNew
			}
			for i := 0; i < n; i++ {
				if math.IsNaN(stoppingDelta[i]) || stoppingDelta[i] > utilityStoppingPercDelta {
					stop = false
					break
				}
			}

			if stop {
				break
			}
		}

		event := heap.Pop(h).(Event)
		t = event.t
		f := event.fun

		if math.IsInf(t, 0) {
			fmt.Println(event)
			panic("infinite time")
		}

		if event.evType == EV_ARRIVAL {
			totArrivals++
			arrivals[f]++
			canExec, cold, mustReclaim := canExecute(containerPools, f, availMem, &model)
			//fmt.Printf("Pool: %v, Can Ex: %v, cold: %v, recl: %v\n", containerPools[f], canExec, cold, mustReclaim)
			if canExec && !cold {
				containerPools[f].Warm--
				containerPools[f].Busy++
				// Immediate execution

				// Gamma distribution
				mean := model.ServTimes[f]
				cv := model.ServCVs[f]
				k := 1.0 / (cv * cv) // shape
				theta := mean / k    // scale
				beta := 1.0 / theta  // rate

				gammaDist := distuv.Gamma{
					Alpha: k,
					Beta:  beta,
					Src:   r,
				}

				servTime := gammaDist.Rand()

				compl_t := t + servTime + model.NetOverhead

				heap.Push(h, Event{compl_t, EV_COMPLETION, f})
				rt := compl_t - t
				if rt < 0 {
					panic("negative RT")
				}
				completions[f]++
				meanRT[f] += (rt - meanRT[f]) / float64(completions[f])
				if rt < model.Deadlines[f] {
					completions_within_deadline[f] += 1
				}
			} else if canExec {
				// cold start
				coldStarts[f]++
				if mustReclaim {
					toReclaim := model.MemDemands[f] - availMem
					availMem += reclaimMemory(containerPools, toReclaim, &model, sortedFunIndices)
				}
				availMem -= model.MemDemands[f]
				containerPools[f].Busy++

				// Gamma distribution
				mean := model.ServTimes[f]
				cv := model.ServCVs[f]
				k := 1.0 / (cv * cv) // shape
				theta := mean / k    // scale
				beta := 1.0 / theta  // rate

				gammaDist := distuv.Gamma{
					Alpha: k,
					Beta:  beta,
					Src:   r,
				}

				servTime := gammaDist.Rand()
				compl_t := t + servTime + model.InitTimes[f] + model.NetOverhead

				heap.Push(h, Event{compl_t, EV_COMPLETION, f})
				rt := compl_t - t
				if rt < 0 {
					panic("negative RT")
				}
				completions[f]++
				meanRT[f] += (rt - meanRT[f]) / float64(completions[f])
				if rt < model.Deadlines[f] {
					completions_within_deadline[f] += 1
				}
			} else if queue.Len() < model.QueueCapacity {
				// ADD TO QUEUE
				var priority float64
				switch model.QueuePolicy {
				case QUEUE_FIFO:
					priority = t
				case QUEUE_EDF:
					priority = t + model.Deadlines[f]
				case QUEUE_SJF:
					priority = model.ServTimes[f]
				}
				queue.Push(QueuedReq{arrivalT: t, fun: f, priority: priority})
			} else {
				// BLOCKED
			}

			// New arrival?
			if totArrivals < maxArrivals {
				if len(MAPs) == 0 {
					event.t = t + r.ExpFloat64()/model.ArvRates[f]
				} else {
					iats, state, err := MAPs[f].Sample(1, MAPstates[f], r)
					if err != nil {
						log.Fatal(err)
					}
					MAPstates[f] = state
					event.t = t + iats[0]
				}
				heap.Push(h, event)
			}
		} else {
			// COMPLETION
			containerPools[f].Busy--
			containerPools[f].Warm++

			for queue.Len() > 0 {
				// try dequeue
				dequeued := queue.Front()

				f = dequeued.fun
				canExec, cold, mustReclaim := canExecute(containerPools, f, availMem, &model)
				if !canExec {
					break
					// TODO: we assume no backfilling
				}

				if cold {
					coldStarts[f]++
					if mustReclaim {
						toReclaim := model.MemDemands[f] - availMem
						availMem += reclaimMemory(containerPools, toReclaim, &model, sortedFunIndices)
					}
					availMem -= model.MemDemands[f]
				} else {
					containerPools[f].Warm--
				}
				containerPools[f].Busy++

				// Gamma distribution
				mean := model.ServTimes[f]
				cv := model.ServCVs[f]
				k := 1.0 / (cv * cv) // shape
				theta := mean / k    // scale
				beta := 1.0 / theta  // rate

				gammaDist := distuv.Gamma{
					Alpha: k,
					Beta:  beta,
					Src:   r,
				}

				servTime := gammaDist.Rand()
				compl_t := t + servTime + model.NetOverhead

				if cold {
					compl_t += model.InitTimes[f]
				}
				heap.Push(h, Event{compl_t, EV_COMPLETION, f})
				rt := compl_t - dequeued.arrivalT
				if rt < 0 {
					fmt.Printf("Current time: %f\n", t)
					fmt.Printf("Compl: %.3f, deq: %.3f\n", compl_t, dequeued.arrivalT)
					panic("negative RT")
				}
				completions[f]++
				meanRT[f] += (rt - meanRT[f]) / float64(completions[f])
				if rt < model.Deadlines[f] {
					completions_within_deadline[f] += 1
				}
				queue.Pop()
			}
		}
	}

	for i := 0; i < n; i++ {
		if arrivals[i] > 0 {
			utilityProb[i] = float64(completions_within_deadline[i]) / float64(arrivals[i])
		} else {
			utilityProb[i] = 1.0
		}
	}

	return &Results{Utility: utilityProb, AvgRT: meanRT, Time: t, Completions: completions, ColdStarts: coldStarts, Arrivals: arrivals}, nil
}

func main() {
	args := os.Args[1:]
	if len(args) < 1 {
		fmt.Println("Usage: simulator <models json file> [<max arrivals> <seed> <parallelism=1>]")
		os.Exit(1)
	}

	jsonFile, err := os.Open(args[0])
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	defer jsonFile.Close()

	var maxArrivals int64
	if len(args) > 1 {
		maxArrivals, err = strconv.ParseInt(args[1], 10, 64)
		if err != nil {
			fmt.Printf("Invalid maxArrivals: %s", args[1])
			os.Exit(2)
		}
	} else {
		maxArrivals = 100000
	}

	var seed int64
	if len(args) > 2 {
		seed, err = strconv.ParseInt(args[2], 10, 64)
		if err != nil {
			fmt.Printf("Invalid seed: %s", args[2])
			os.Exit(2)
		}
	} else {
		seed = 1
	}

	var maxParallelism int64 = 1
	if len(args) > 3 {
		maxParallelism, err = strconv.ParseInt(args[3], 10, 64)
		if err != nil {
			fmt.Printf("Invalid parallelism level: %s", args[3])
			os.Exit(2)
		}
	}
	runtime.GOMAXPROCS(int(maxParallelism))

	byteValue, _ := ioutil.ReadAll(jsonFile)

	var models []Model
	err = json.Unmarshal(byteValue, &models)
	if err != nil {
		fmt.Printf("Failed parsing: %v\n", err)
		os.Exit(2)
	}

	results := make([]*Results, len(models), len(models))

	//model := Model{
	//	arvRates:      []float64{1.0, 2.0},
	//	servTimes:     []float64{0.5, 0.5},
	//	deadlines:     []float64{0.7, 0.7},
	//	memDemands:    []int{128, 128},
	//	memory:        1024,
	//	queueCapacity: 3,
	//}
	var wg = sync.WaitGroup{}
	guard := make(chan struct{}, maxParallelism)

	for i, _ := range models {
		guard <- struct{}{} // would block if guard channel is already filled
		wg.Add(1)
		go func(index int) {
			res, err := simulate(models[index], maxArrivals, 0.01, seed)
			if err != nil {
				fmt.Println(err)
				os.Exit(3)
			}
			results[index] = res
			<-guard
			wg.Done()
		}(i)
	}

	wg.Wait()

	b, err := json.MarshalIndent(results, "", " ")
	if err != nil {
		fmt.Println(err)
		fmt.Println(results[0])
		os.Exit(3)
	}
	fmt.Println(string(b))
}
