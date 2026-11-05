"""BGE-M3 dense vectors from existing local files; never download weights."""
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from hipporag.embedding_model.base import BaseEmbeddingModel, EmbeddingConfig


class LocalBGEEmbedding(BaseEmbeddingModel):
    query_instruction_mode = "ignored"

    def __init__(self, global_config, settings):
        super().__init__(global_config)
        model_path = Path(settings["embedding_path"])
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"Missing local model config: {model_path}")
        device = settings["device"]
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = settings["batch_size"]
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = SentenceTransformer(
            str(model_path), device=device, local_files_only=True,
            model_kwargs={"torch_dtype": dtype},
        )
        self.model.max_seq_length = settings["max_seq_length"]
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.embedding_config = EmbeddingConfig.from_dict({
            "embedding_model_name": self.embedding_model_name,
            "norm": True,
            "encode_params": {"batch_size": self.batch_size},
        })

    def batch_encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        with torch.inference_mode():
            vectors = self.model.encode(
                texts, batch_size=kwargs.get("batch_size", self.batch_size),
                normalize_embeddings=False, convert_to_numpy=True,
                show_progress_bar=False,
            )
        return self._normalize_embeddings(vectors, normalize=kwargs.get("norm"))

    def close(self):
        if hasattr(self, "model"):
            del self.model
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
