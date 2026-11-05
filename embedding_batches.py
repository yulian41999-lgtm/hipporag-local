"""Per-store batching over one shared embedding model."""
class BatchSizeEmbedding:
    def __init__(self, embedding, batch_size):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("Embedding batch size must be a positive integer")
        self.embedding = embedding
        self.batch_size = batch_size

    def __getattr__(self, name):
        return getattr(self.embedding, name)

    def batch_encode(self, texts, **kwargs):
        kwargs.setdefault("batch_size", self.batch_size)
        return self.embedding.batch_encode(texts, **kwargs)
