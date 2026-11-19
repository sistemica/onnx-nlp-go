//go:build !amd64

package onnxnlp

var (
	useAVX2   = false
	useAVX512 = false
)

func kernel6x16(a *float32, lda int, b *float32, ldb int, c *float32, ldc int, k int) {
	panic("kernel6x16 is only available on amd64")
}

func kernel12x16(a *float32, lda int, b *float32, ldb int, c *float32, ldc int, k int) {
	panic("kernel12x16 is only available on amd64")
}
