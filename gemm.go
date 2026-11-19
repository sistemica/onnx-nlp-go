package onnxnlp

import (
	"os"
	"sync"
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

func runGemmsCPU(jobs []gemmJob) {
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
		for t := 0; t < tiles; t++ {
			packPanel(c.data[t*k*tileCols:(t+1)*k*tileCols], b, k, n, t*tileCols)
		}
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
	for row := rowLo; row < rowLo+rows; row += tileRows {
		j.block(panel, row, min(tileRows, rowLo+rows-row), col, end-col)
	}
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
