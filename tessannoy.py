import annoy


class TessAnnoyIndex:
    def __init__(self, vectors, labels, citations):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype("float32")
        self.labels = labels
        self.citations = citations
        self.search_in_x_trees = 8

    def build(self, number_of_trees=5):
        self.index = annoy.AnnoyIndex(self.dimension, metric="angular")
        for i, vec in enumerate(self.vectors):
            self.index.add_item(i, vec.tolist())
        self.index.build(number_of_trees)

    def load(self, filename):
        self.index = annoy.AnnoyIndex(self.dimension, metric="angular")
        self.index.load(filename)

    def save(self, filename):
        self.index.save(filename)

    def query(self, vector, k=10):
        indices, dists = self.index.get_nns_by_vector(
            vector.tolist(), k, search_k=self.search_in_x_trees, include_distances=True
        )
        return [
            (self.citations[i], self.labels[i], 1 - dist)
            for i, dist in zip(indices, dists)
        ]
