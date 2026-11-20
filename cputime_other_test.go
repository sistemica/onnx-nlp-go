//go:build !unix

package onnxnlp

import "time"

// processCPUTime is not available on this platform; the speed tests report zero cores busy.
func processCPUTime() time.Duration { return 0 }
