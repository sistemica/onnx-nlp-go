package onnxnlp

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

type Session struct {
	graph *Graph
	prof  *opProfile
}

type opProfile struct {
	mu    sync.Mutex
	total map[string]time.Duration
	count map[string]int
}

func (p *opProfile) add(op string, d time.Duration) {
	p.mu.Lock()
	p.total[op] += d
	p.count[op]++
	p.mu.Unlock()
}

// OpStat is the accumulated wall time of one operator type.
type OpStat struct {
	Op    string
	Calls int
	Total time.Duration
}

// EnableProfiling starts accumulating wall time per operator type across Run calls.
func (s *Session) EnableProfiling() {
	s.prof = &opProfile{total: map[string]time.Duration{}, count: map[string]int{}}
}

// OpStats returns the profile gathered since EnableProfiling, slowest first.
func (s *Session) OpStats() []OpStat {
	if s.prof == nil {
		return nil
	}
	s.prof.mu.Lock()
	defer s.prof.mu.Unlock()
	stats := make([]OpStat, 0, len(s.prof.total))
	for op, d := range s.prof.total {
		stats = append(stats, OpStat{op, s.prof.count[op], d})
	}
	sort.Slice(stats, func(a, b int) bool { return stats[a].Total > stats[b].Total })
	return stats
}

func Load(path string) (*Session, error) {
	g, err := ParseGraphFile(path)
	if err != nil {
		return nil, err
	}
	return NewSession(g)
}

func NewSession(g *Graph) (*Session, error) {
	if err := checkOps(g); err != nil {
		return nil, err
	}
	return &Session{graph: fuseNodes(foldConstants(g))}, nil
}

// foldConstants evaluates, once, every top-level node whose inputs are all constants and
// turns its outputs into initializers. The graph it returns shares tensors with g.
func foldConstants(g *Graph) *Graph {
	folded := &Graph{Name: g.Name, Inputs: g.Inputs, Outputs: g.Outputs, Initializers: make(map[string]*Tensor, len(g.Initializers))}
	for name, t := range g.Initializers {
		folded.Initializers[name] = t
	}
	for _, n := range g.Nodes {
		if outs, ok := evalConstantNode(n, folded.Initializers); ok {
			for i, name := range n.Out {
				if name != "" && i < len(outs) && outs[i] != nil {
					t := outs[i]
					t.static, t.pack = true, &packCache{}
					folded.Initializers[name] = t
				}
			}
			continue
		}
		folded.Nodes = append(folded.Nodes, n)
	}
	return folded
}

func evalConstantNode(n *Node, consts map[string]*Tensor) ([]*Tensor, bool) {
	for _, a := range n.Attr {
		if a.G != nil {
			return nil, false
		}
	}
	ctx := &opCtx{node: n, in: make([]*Tensor, len(n.In)), scope: &scope{vals: consts}}
	for i, name := range n.In {
		if name == "" {
			continue
		}
		t, ok := consts[name]
		if !ok {
			return nil, false
		}
		ctx.in[i] = t
	}
	outs, err := ops[n.Op](ctx)
	return outs, err == nil
}

func (s *Session) InputNames() []string  { return s.graph.Inputs }
func (s *Session) OutputNames() []string { return s.graph.Outputs }

func checkOps(g *Graph) error {
	for _, n := range g.Nodes {
		if n.Domain != "" && n.Domain != "ai.onnx" {
			return fmt.Errorf("node %q: unsupported domain %q", n.Name, n.Domain)
		}
		if _, ok := ops[n.Op]; !ok {
			return fmt.Errorf("node %q: unsupported operator %s", n.Name, n.Op)
		}
		for _, a := range n.Attr {
			if a.G != nil {
				if err := checkOps(a.G); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

type scope struct {
	vals   map[string]*Tensor
	consts map[string]*Tensor
	parent *scope
	prof   *opProfile
}

func (sc *scope) lookup(name string) (*Tensor, bool) {
	for s := sc; s != nil; s = s.parent {
		if t, ok := s.vals[name]; ok {
			return t, true
		}
		if t, ok := s.consts[name]; ok {
			return t, true
		}
	}
	return nil, false
}

type opCtx struct {
	node  *Node
	in    []*Tensor
	scope *scope
}

type opFunc func(c *opCtx) ([]*Tensor, error)

var ops map[string]opFunc

func (s *Session) Run(inputs map[string]*Tensor) (map[string]*Tensor, error) {
	for _, name := range s.graph.Inputs {
		if _, ok := inputs[name]; !ok {
			return nil, fmt.Errorf("missing input %q", name)
		}
	}
	outs, err := runGraph(s.graph, &scope{vals: inputs, prof: s.prof})
	if err != nil {
		return nil, err
	}
	res := make(map[string]*Tensor, len(outs))
	for i, name := range s.graph.Outputs {
		res[name] = outs[i]
	}
	return res, nil
}

func runGraph(g *Graph, parent *scope) ([]*Tensor, error) {
	sc := &scope{vals: make(map[string]*Tensor, len(g.Nodes)), consts: g.Initializers, parent: parent}
	if parent != nil {
		sc.prof = parent.prof
	}
	for _, n := range g.Nodes {
		ctx := &opCtx{node: n, in: make([]*Tensor, len(n.In)), scope: sc}
		for i, name := range n.In {
			if name == "" {
				continue
			}
			t, ok := sc.lookup(name)
			if !ok {
				return nil, fmt.Errorf("node %q (%s): input %q not computed", n.Name, n.Op, name)
			}
			ctx.in[i] = t
		}
		var started time.Time
		if sc.prof != nil {
			started = time.Now()
		}
		outs, err := ops[n.Op](ctx)
		if sc.prof != nil {
			sc.prof.add(n.Op, time.Since(started))
		}
		if err != nil {
			return nil, fmt.Errorf("node %q (%s): %w", n.Name, n.Op, err)
		}
		for i, name := range n.Out {
			if name != "" && i < len(outs) && outs[i] != nil {
				sc.vals[name] = outs[i]
			}
		}
	}
	outs := make([]*Tensor, len(g.Outputs))
	for i, name := range g.Outputs {
		t, ok := sc.lookup(name)
		if !ok {
			return nil, fmt.Errorf("graph output %q not computed", name)
		}
		outs[i] = t
	}
	return outs, nil
}

func (c *opCtx) need(n int) error {
	if len(c.in) < n {
		return fmt.Errorf("expected at least %d inputs, got %d", n, len(c.in))
	}
	for i := 0; i < n; i++ {
		if c.in[i] == nil {
			return fmt.Errorf("input %d is empty", i)
		}
	}
	return nil
}
