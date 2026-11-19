"""Records the stages of BERTopic's default pipeline for a fixed corpus so the Go port can be checked stage by stage.

usage: reference.py OUTDIR NDOCS

Needs bertopic, umap-learn, hdbscan, sentence-transformers, scikit-learn and scipy, and downloads
the 20 newsgroups corpus and all-MiniLM-L6-v2. Point ONNXGO_BERTOPIC_REF at OUTDIR to run the
model-backed tests (and ONNXGO_MINILM at the MiniLM ONNX export for the end-to-end one).
"""
import json
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from umap import UMAP
from umap.layouts import optimize_layout_euclidean

out, n_docs = sys.argv[1], int(sys.argv[2])
data = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), data_home=out + "/sk")
docs = [d for d in data.data if len(d.split()) >= 5][:n_docs]
json.dump(docs, open(out + "/docs.json", "w"))

emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu").encode(docs, batch_size=64, normalize_embeddings=False)
np.save(out + "/emb.npy", emb.astype(np.float32))

umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
low = umap.fit_transform(emb)
np.save(out + "/umap5.npy", low.astype(np.float64))
graph = umap.graph_.tocoo()
np.savez(out + "/umap_graph.npz", row=graph.row, col=graph.col, val=graph.data, a=umap._a, b=umap._b)


def pruned(graph, epochs):
    g = graph.tocsr().copy()
    g.sort_indices()
    g.data[g.data < g.data.max() / float(epochs)] = 0.0
    g.eliminate_zeros()
    return g


from umap.spectral import spectral_layout
from umap.umap_ import noisy_scale_coords
from umap.utils import make_epochs_per_sample

n = len(docs)
full = pruned(umap.graph_, 500)
spectral = spectral_layout(emb, full, 5, np.random.RandomState(42), metric="cosine")
np.save(out + "/spectral_vectors.npy", np.asarray(spectral, dtype=np.float64))
sd = np.sqrt(np.asarray(full.sum(axis=0)).squeeze())
laplacian = sp.identity(n) - sp.spdiags(1.0 / sd, 0, n, n) @ full @ sp.spdiags(1.0 / sd, 0, n, n)
values = spla.eigsh(laplacian, 6, which="SM", ncv=max(13, int(np.sqrt(n))), tol=1e-4, v0=np.ones(n), maxiter=n * 5)[0]
np.save(out + "/spectral_values.npy", np.sort(values))
np.save(out + "/pruned_edges.npy", np.stack([full.tocoo().row, full.tocoo().col], 1).astype(np.float64))
np.save(out + "/pruned_weights.npy", full.tocoo().data.astype(np.float64))

sgd_epochs, sgd_run = 30, 2
sgd_graph = pruned(umap.graph_, sgd_epochs).tocoo()
init = noisy_scale_coords(spectral, np.random.RandomState(42), max_coord=10, noise=0.0001)
embedding = init.copy()
optimize_layout_euclidean(
    embedding, embedding, sgd_graph.row.astype(np.int32), sgd_graph.col.astype(np.int32), sgd_run, n,
    make_epochs_per_sample(sgd_graph.data, sgd_epochs), umap._a, umap._b, np.array([1, 2, 3], dtype=np.int64),
    0.0, 1.0, 5, parallel=False, verbose=False, densmap_kwds={}, tqdm_kwds={}, move_other=True, optimizer="compatibility",
)
np.save(out + "/sgd_init.npy", init.astype(np.float32))
np.save(out + "/sgd_edges.npy", np.stack([sgd_graph.row, sgd_graph.col], 1).astype(np.float64))
np.save(out + "/sgd_weights.npy", sgd_graph.data.astype(np.float32))
np.save(out + "/sgd_out.npy", embedding.astype(np.float32))

hdb = HDBSCAN(min_cluster_size=10, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
hdb.fit(low)
np.save(out + "/hdb_labels.npy", hdb.labels_)
np.save(out + "/hdb_probs.npy", hdb.probabilities_)
exact = HDBSCAN(min_cluster_size=10, metric="euclidean", cluster_selection_method="eom", approx_min_span_tree=False).fit(low)
np.save(out + "/hdb_exact_labels.npy", exact.labels_)
np.save(out + "/hdb_exact_probs.npy", exact.probabilities_)
np.save(out + "/hdb_exact_linkage.npy", np.asarray(exact.single_linkage_tree_.to_numpy(), dtype=np.float64))

for seed in (1, 2, 3, 4):
    other = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed).fit_transform(emb)
    np.save(out + "/umap5_seed%d.npy" % seed, other.astype(np.float64))
    np.save(out + "/hdb_exact_labels_seed%d.npy" % seed, HDBSCAN(min_cluster_size=10, approx_min_span_tree=False).fit(other).labels_)

topic_model = BERTopic(umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42),
                       hdbscan_model=HDBSCAN(min_cluster_size=10, metric="euclidean", cluster_selection_method="eom", prediction_data=True))
topics, probs = topic_model.fit_transform(docs, emb)
stopped = BERTopic(umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42),
                   hdbscan_model=HDBSCAN(min_cluster_size=10, metric="euclidean", cluster_selection_method="eom", prediction_data=True),
                   vectorizer_model=CountVectorizer(stop_words="english"))
stopped_topics, _ = stopped.fit_transform(docs, emb)
assert list(stopped_topics) == list(topics)
json.dump({
    "words_stop": {str(k): [[w, float(s)] for w, s in v] for k, v in stopped.get_topics().items()},
    "topics": [int(t) for t in topics],
    "probs": [float(p) for p in probs],
    "words": {str(k): [[w, float(s)] for w, s in v] for k, v in topic_model.get_topics().items()},
    "sizes": {str(k): int(v) for k, v in topic_model.topic_sizes_.items()},
}, open(out + "/bertopic.json", "w"))
print(len(docs), "docs;", len(set(topics)) - (1 if -1 in topics else 0), "topics;", sum(t == -1 for t in topics), "outliers")
print("hdbscan labels equal BERTopic topics:", bool((np.array(topics) == hdb.labels_).all()))
print("umap a,b:", umap._a, umap._b)
