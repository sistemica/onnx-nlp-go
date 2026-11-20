package onnxnlp

import (
	"math"
	"os"
	"sync"
	"sync/atomic"
)

const (
	tileCols    = 16
	maxTileRows = 12
)

// The micro-kernel and its height are chosen once. AVX2 takes six rows at a time. The AVX-512
// kernel takes twelve and is opt-in (ONNXNLP_AVX512=on): it was slower on Zen 4, where 512-bit
// operations run as two halves, and within a few percent either way on Zen 5.
var (
	tileRows   int
	rowBlock   int
	tileKernel func(a *float32, lda int, b *float32, ldb int, c *float32, ldc int, k int)
)

func init() { selectKernel() }

func selectKernel() {
	if useAVX512 && useAVX2 && os.Getenv("ONNXNLP_AVX512") == "on" {
		tileRows, tileKernel = 12, kernel12x16
	} else {
		tileRows, tileKernel = 6, kernel6x16
	}
	rowBlock = 96
}

// singleRowKernel routes products with a single input row to the vector kernel. It is a variable
// so that tests and benchmarks can compare it with the matrix kernel.
var singleRowKernel = true

type gemmJob struct {
	out, a, b []float32
	m, k, n   int
	bStatic   bool
	// pack, when set, is the pack cache of b, which must be a single static matrix.
	pack *packCache
}

// runGemms accumulates out += a*b for every job, which must all share the same n,
// spreading column tiles of all jobs over the available cores.
func runGemms(jobs []gemmJob) {
	if len(jobs) == 0 {
		return
	}
	if jobs = offloadGemms(jobs); len(jobs) == 0 {
		return
	}
	runGemmsCPU(jobs)
}

// prepack builds the packed weights a job will read, once. It runs before the parallel loop: inside
// it, the other workers would be waiting for the packing and none would be free to share it.
func (j gemmJob) prepack() {
	if j.pack == nil || !useAVX2 || j.k == 0 {
		return
	}
	if j.m == 1 && singleRowKernel && bf16Weights && j.pack.panels16(j.b, j.k, j.n) != nil {
		return
	}
	j.pack.panels(j.b, j.k, j.n)
}

func runGemmsCPU(jobs []gemmJob) {
	for _, j := range jobs {
		j.prepack()
	}
	first := jobs[0]
	tiles := (first.n + tileCols - 1) / tileCols
	blocks := (first.m + rowBlock - 1) / rowBlock
	perJob := tiles * blocks
	parallelFor(len(jobs)*perJob, min(first.m, rowBlock)*first.k*tileCols, func(lo, hi int) {
		for u := lo; u < hi; u++ {
			job, r := jobs[u/perJob], u%perJob
			rowLo := r % blocks * rowBlock
			job.columnTile(r/blocks*tileCols, rowLo, min(rowBlock, first.m-rowLo))
		}
	})
}

var panelPool = sync.Pool{New: func() any { return new([]float32) }}

func pooledBuffer(n int) (*[]float32, []float32) {
	buf := panelPool.Get().(*[]float32)
	if cap(*buf) < n {
		*buf = make([]float32, n)
	}
	return buf, (*buf)[:n]
}

// packCache holds the column panels of one static weight matrix, packed once.
type packCache struct {
	once sync.Once
	data []float32

	once16 sync.Once
	data16 []uint16
}

// bf16Weights lets single-row products read weights as 16-bit bfloat16 values when that loses
// nothing. It is a variable so that tests and benchmarks can compare it with float32 panels.
var bf16Weights = true

// panels16 returns the packed panels of b as bfloat16, or nil when some weight has bits below the
// top sixteen set and the conversion would change it. Models trained in bfloat16 and exported to
// float32 qualify, and read half as much memory.
func (c *packCache) panels16(b []float32, k, n int) []uint16 {
	c.once16.Do(func() {
		var inexact atomic.Bool
		parallelFor(k, n, func(lo, hi int) {
			for _, v := range b[lo*n : hi*n] {
				if math.Float32bits(v)&0xffff != 0 {
					inexact.Store(true)
					return
				}
			}
		})
		if inexact.Load() {
			return
		}
		tiles := (n + tileCols - 1) / tileCols
		data := make([]uint16, tiles*k*tileCols)
		parallelFor(tiles, k*tileCols, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				col := t * tileCols
				end := min(col+tileCols, n)
				panel := data[t*k*tileCols : (t+1)*k*tileCols]
				for p := 0; p < k; p++ {
					for i, v := range b[p*n+col : p*n+end] {
						panel[p*tileCols+i] = uint16(math.Float32bits(v) >> 16)
					}
				}
			}
		})
		c.data16 = data
	})
	return c.data16
}

// packPanel copies the column tile starting at col into dst as k contiguous rows of tileCols,
// zero-padding the columns past n.
func packPanel(dst, b []float32, k, n, col int) {
	end := min(col+tileCols, n)
	for p := 0; p < k; p++ {
		row := dst[p*tileCols : (p+1)*tileCols]
		copy(row, b[p*n+col:p*n+end])
		clear(row[end-col:])
	}
}

func (c *packCache) panels(b []float32, k, n int) []float32 {
	c.once.Do(func() {
		tiles := (n + tileCols - 1) / tileCols
		c.data = make([]float32, tiles*k*tileCols)
		parallelFor(tiles, k*tileCols, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				packPanel(c.data[t*k*tileCols:(t+1)*k*tileCols], b, k, n, t*tileCols)
			}
		})
	})
	return c.data
}

// columnTile accumulates rows [rowLo, rowLo+rows) of the tileCols-wide column tile starting at col.
func (j gemmJob) columnTile(col, rowLo, rows int) {
	end := min(col+tileCols, j.n)
	if !useAVX2 || j.k == 0 {
		scalarBlock(j.out, j.a, j.b, j.k, j.n, rowLo, rowLo+rows, col, end)
		return
	}
	if j.m == 1 && singleRowKernel && j.pack != nil && bf16Weights {
		if p16 := j.pack.panels16(j.b, j.k, j.n); p16 != nil {
			offset := col / tileCols * j.k * tileCols
			j.vectorTile16(p16[offset:offset+j.k*tileCols], col, end-col)
			return
		}
	}
	var panel []float32
	if j.pack != nil {
		offset := col / tileCols * j.k * tileCols
		panel = j.pack.panels(j.b, j.k, j.n)[offset : offset+j.k*tileCols]
	} else {
		buf, scratch := pooledBuffer(j.k * tileCols)
		defer panelPool.Put(buf)
		packPanel(scratch, j.b, j.k, j.n, col)
		panel = scratch
	}
	if j.m == 1 && singleRowKernel {
		j.vectorTile(panel, col, end-col)
		return
	}
	for row := rowLo; row < rowLo+rows; row += tileRows {
		j.block(panel, row, min(tileRows, rowLo+rows-row), col, end-col)
	}
}

// vectorTile16 is vectorTile for a bfloat16 panel.
func (j gemmJob) vectorTile16(panel []uint16, col, width int) {
	if width == tileCols {
		gemv16bf16(&j.a[0], &panel[0], &j.out[col], j.k)
		return
	}
	var tile [tileCols]float32
	copy(tile[:width], j.out[col:col+width])
	gemv16bf16(&j.a[0], &panel[0], &tile[0], j.k)
	copy(j.out[col:col+width], tile[:width])
}

// vectorTile accumulates one input row against a packed column tile. The matrix kernel would pad
// the row to six and waste five sixths of its work.
func (j gemmJob) vectorTile(panel []float32, col, width int) {
	if width == tileCols {
		gemv16(&j.a[0], &panel[0], &j.out[col], j.k)
		return
	}
	var tile [tileCols]float32
	copy(tile[:width], j.out[col:col+width])
	gemv16(&j.a[0], &panel[0], &tile[0], j.k)
	copy(j.out[col:col+width], tile[:width])
}

// block accumulates a tile of up to tileRows x tileCols; ragged edges go through small
// zero-padded temporaries so the SIMD kernel handles every shape.
func (j gemmJob) block(panel []float32, row, rows, col, width int) {
	a := j.a[row*j.k:]
	if rows < tileRows {
		buf, padded := pooledBuffer(tileRows * j.k)
		defer panelPool.Put(buf)
		clear(padded)
		copy(padded, j.a[row*j.k:(row+rows)*j.k])
		a = padded
	}
	if rows == tileRows && width == tileCols {
		tileKernel(&a[0], j.k, &panel[0], tileCols, &j.out[row*j.n+col], j.n, j.k)
		return
	}
	var tile [maxTileRows * tileCols]float32
	for r := 0; r < rows; r++ {
		copy(tile[r*tileCols:r*tileCols+width], j.out[(row+r)*j.n+col:(row+r)*j.n+col+width])
	}
	tileKernel(&a[0], j.k, &panel[0], tileCols, &tile[0], tileCols, j.k)
	for r := 0; r < rows; r++ {
		copy(j.out[(row+r)*j.n+col:(row+r)*j.n+col+width], tile[r*tileCols:r*tileCols+width])
	}
}

func scalarBlock(out, a, b []float32, k, n, rowLo, rowHi, colLo, colHi int) {
	for i := rowLo; i < rowHi; i++ {
		crow := out[i*n+colLo : i*n+colHi]
		arow := a[i*k : (i+1)*k]
		p := 0
		for ; p+4 <= k; p += 4 {
			a0, a1, a2, a3 := arow[p], arow[p+1], arow[p+2], arow[p+3]
			b0 := b[p*n+colLo : p*n+colHi]
			b1 := b[(p+1)*n+colLo : (p+1)*n+colHi]
			b2 := b[(p+2)*n+colLo : (p+2)*n+colHi]
			b3 := b[(p+3)*n+colLo : (p+3)*n+colHi]
			for j := range crow {
				crow[j] += a0*b0[j] + a1*b1[j] + a2*b2[j] + a3*b3[j]
			}
		}
		for ; p < k; p++ {
			av := arow[p]
			bp := b[p*n+colLo : p*n+colHi]
			for j := range crow {
				crow[j] += av * bp[j]
			}
		}
	}
}
