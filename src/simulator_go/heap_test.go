package main

import (
	"container/heap"
	"fmt"
	"testing"
)

func TestHeap(t *testing.T) {
	q := &EventHeap{}
	heap.Init(q)
	fmt.Printf("%p\n", q)
	e := Event{t: 2}
	heap.Push(q, e)
	fmt.Printf("%p\n", q)
	e = Event{t: 2.2}
	heap.Push(q, e)
	fmt.Printf("%p\n", q)
	e = Event{t: 1.5}
	heap.Push(q, e)
	fmt.Printf("%p\n", q)

	for q.Len() > 0 {
		fmt.Printf("elem: %v\n", q.Front())
		fmt.Printf("elem: %v\n", heap.Pop(q))
	}
}
