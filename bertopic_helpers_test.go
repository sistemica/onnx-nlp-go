package onnxnlp

// renumberBySize gives the clusters the IDs 0, 1, ... in order of decreasing size. Noise stays -1.
func renumberBySize(labels []int) []int {
	ranking := sizeRanking(labels)
	out := make([]int, len(labels))
	for i, l := range labels {
		out[i] = rankedTopic(ranking, l)
	}
	return out
}
