//go:build darwin

package onnxnlp

import (
	"github.com/ebitengine/purego"
)

const accelerateLibrary = "/System/Library/Frameworks/Accelerate.framework/Accelerate"

func init() { registerPreferredBackend("accelerate", openAccelerate) }

func openAccelerate() (gemmDevice, error) {
	handle, err := purego.Dlopen(accelerateLibrary, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return nil, err
	}
	api := &accelerateAPI{}
	purego.RegisterLibFunc(&api.sgemm, handle, "cblas_sgemm")
	return newAccelerateDevice(api)
}
