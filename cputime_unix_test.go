//go:build unix

package onnxnlp

import (
	"syscall"
	"time"
)

// processCPUTime is the CPU time this process has used, over all its threads.
func processCPUTime() time.Duration {
	var u syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &u)
	return time.Duration(u.Utime.Nano() + u.Stime.Nano())
}
