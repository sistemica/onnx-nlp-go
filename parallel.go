package onnxnlp

import (
	"runtime"
	"sync"
	"sync/atomic"
)

const minParallelWork = 1 << 15

type parallelJob struct {
	fn    func(lo, hi int)
	n     int
	chunk int
	next  atomic.Int64
	wg    sync.WaitGroup
}

// work claims chunks until none are left, so any number of goroutines may share one job.
func (j *parallelJob) work() {
	for {
		lo := int(j.next.Add(int64(j.chunk))) - j.chunk
		if lo >= j.n {
			return
		}
		j.fn(lo, min(lo+j.chunk, j.n))
		j.wg.Done()
	}
}

var pool struct {
	mu      sync.Mutex
	workers int
	tasks   chan *parallelJob
}

// ensureWorkers starts persistent helpers until n exist and returns the task queue.
// Helpers park on the queue instead of spinning: with hyperthreads, a spinning helper
// slows the sibling thread that is doing real work.
func ensureWorkers(n int) chan<- *parallelJob {
	pool.mu.Lock()
	defer pool.mu.Unlock()
	if pool.tasks == nil {
		pool.tasks = make(chan *parallelJob, 256)
	}
	for ; pool.workers < n; pool.workers++ {
		go func(tasks <-chan *parallelJob) {
			for job := range tasks {
				job.work()
			}
		}(pool.tasks)
	}
	return pool.tasks
}

// parallelFor runs fn over [0, n) in chunks shared by the calling goroutine and the pool's
// helpers; each item costs about cost elementary operations, and small totals run inline.
func parallelFor(n, cost int, fn func(lo, hi int)) {
	workers := min(runtime.GOMAXPROCS(0), n, max(1, n*cost/minParallelWork))
	if workers <= 1 {
		fn(0, n)
		return
	}
	chunk := (n + 2*workers - 1) / (2 * workers)
	job := &parallelJob{fn: fn, n: n, chunk: chunk}
	job.wg.Add((n + chunk - 1) / chunk)
	tasks := ensureWorkers(workers - 1)
	for i := 1; i < workers; i++ {
		select {
		case tasks <- job:
		default:
		}
	}
	job.work()
	job.wg.Wait()
}
