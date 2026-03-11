package main

import (
	"container/heap"
	"fmt"
	"sort"
)

type Queue interface {
	Push(QueuedReq)
	Front() QueuedReq
	Pop() QueuedReq
	Len() int
}

type FCFSQueue []QueuedReq

type PriorityQueue []QueuedReq

func NewFCFSQueue() Queue {
	return &FCFSQueue{}
}

func (q *FCFSQueue) Pop() QueuedReq {
	old := *q
	x := old[0]
	*q = old[1:]
	return x
}

func (q *FCFSQueue) Front() QueuedReq {
	old := *q
	x := old[0]
	return x
}

func (q *FCFSQueue) Push(e QueuedReq) {
	*q = append(*q, e)
}

func (q *FCFSQueue) Len() int {
	return len(*q)
}

func (q *FCFSQueue) String() string {
	return fmt.Sprintf("%v", *q)
}

func NewPriorityQueue() Queue {
	return &PriorityQueue{}
}

func (q *PriorityQueue) Pop() QueuedReq {
	old := *q
	x := old[0]
	*q = old[1:]
	return x
}

func (q *PriorityQueue) Front() QueuedReq {
	old := *q
	x := old[0]
	return x
}

func (q *PriorityQueue) Push(e QueuedReq) {
	*q = append(*q, e)
	i := sort.Search(len(*q)-1, func(i int) bool {
		return (*q)[i].priority >= e.priority
	})

	copy((*q)[i+1:], (*q)[i:])
	(*q)[i] = e
}

func (q *PriorityQueue) Len() int {
	return len(*q)
}

func (q *PriorityQueue) String() string {
	return fmt.Sprintf("%v", *q)
}

type EDFQueue struct {
	h *DeadlineHeap
}

func NewEDFQueue() Queue {
	q := EDFQueue{}
	q.h = &DeadlineHeap{}
	heap.Init(q.h)
	return &q
}

func (q *EDFQueue) Push(e QueuedReq) {
	heap.Push(q.h, e)
}

func (q *EDFQueue) Front() QueuedReq {
	return q.h.Front().(QueuedReq)
}

func (q *EDFQueue) Pop() QueuedReq {
	return heap.Pop(q.h).(QueuedReq)
}

func (q *EDFQueue) Len() int {
	return q.h.Len()
}

func (q EDFQueue) String() string {
	return fmt.Sprintf("%v", q.h)
}
