package main

import (
	"fmt"
	"testing"
)

func TestQueue(t *testing.T) {
	q := NewFCFSQueue()
	e := QueuedReq{}
	q.Push(e)
	q.Push(e)
	q.Push(e)
	if q.Len() != 3 {
		t.Fail()
	}
	for q.Len() > 0 {
		fmt.Printf("elem: %v\n", q.Pop())
	}
}

func TestPrioQueue(t *testing.T) {
	q := NewPriorityQueue()
	e := QueuedReq{priority: 2.1}
	q.Push(e)
	e = QueuedReq{priority: 2.5}
	q.Push(e)
	e = QueuedReq{priority: 0.5}
	q.Push(e)
	e = QueuedReq{priority: 2.7}
	q.Push(e)
	e = QueuedReq{priority: 0.9}
	q.Push(e)
	if q.Len() != 3 {
		t.Fail()
	}
	for q.Len() > 0 {
		fmt.Printf("elem: %v\n", q.Front())
		fmt.Printf("elem: %v\n", q.Pop())
	}
}
