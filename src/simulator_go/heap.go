package main

type EventHeap []Event

func (h EventHeap) Len() int           { return len(h) }
func (h EventHeap) Less(i, j int) bool { return h[i].t < h[j].t }
func (h EventHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *EventHeap) Push(x any) {
	// Push and Pop use pointer receivers because they modify the slice's length,
	// not just its contents.
	*h = append(*h, x.(Event))
}

func (h *EventHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func (h *EventHeap) Front() any {
	old := *h
	return old[0]
}

type DeadlineHeap []QueuedReq

func (h DeadlineHeap) Len() int           { return len(h) }
func (h DeadlineHeap) Less(i, j int) bool { return h[i].priority < h[j].priority }
func (h DeadlineHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *DeadlineHeap) Push(x any) {
	// Push and Pop use pointer receivers because they modify the slice's length,
	// not just its contents.
	*h = append(*h, x.(QueuedReq))
}

func (h *DeadlineHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func (h *DeadlineHeap) Front() any {
	old := *h
	return old[0]
}
