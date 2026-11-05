"""Prepare per-store embedding batch sizes without touching a running task."""
from pathlib import Path
from datetime import datetime
import ast,json
root=Path('/home/ailab/data/HippoRAG/hipporag_local')
module=root/'embedding_batches.py'
assert not module.exists(), 'Inspect existing batch-size adapter first'
module_text='''"""Per-store batching over one shared embedding model."""
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
'''
business=root/'business.py'
old=business.read_text()
needle='''        return HippoRAG(global_config=config, embedding_model=embedding, index_identity=digest), embedding'''
replacement='''        from embedding_batches import BatchSizeEmbedding
        rag = HippoRAG(global_config=config, embedding_model=embedding, index_identity=digest)
        rag.entity_embedding_store.embedding_model = BatchSizeEmbedding(
            embedding, cfg.get("entity_batch_size", cfg["batch_size"]))
        rag.fact_embedding_store.embedding_model = BatchSizeEmbedding(
            embedding, cfg.get("fact_batch_size", cfg["batch_size"]))
        return rag, embedding'''
assert old.count(needle)==1
new=old.replace(needle,replacement)
settings=root/'settings.py'
old_settings=settings.read_text()
needle_settings='{"batch_size": "HIPPO_BATCH_SIZE", "max_seq_length": "HIPPO_MAX_SEQ_LENGTH"}'
replacement_settings='{"batch_size": "HIPPO_BATCH_SIZE", "max_seq_length": "HIPPO_MAX_SEQ_LENGTH", "entity_batch_size": "HIPPO_ENTITY_BATCH_SIZE", "fact_batch_size": "HIPPO_FACT_BATCH_SIZE"}'
assert old_settings.count(needle_settings)==1
new_settings=old_settings.replace(needle_settings,replacement_settings)
config_path=root/'config.json'
config=json.loads(config_path.read_text())
config.update(entity_batch_size=64,fact_batch_size=32)
for code in (module_text,new,new_settings): ast.parse(code)
backup=root/'tools/backups'/('embedding-batches-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
backup.mkdir(parents=True,exist_ok=False)
for p in (business,settings,config_path): (backup/p.name).write_bytes(p.read_bytes())
module.write_text(module_text)
business.write_text(new)
settings.write_text(new_settings)
config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
print('BACKUP',backup)
print('PREPARED: entity=64, fact=32, existing chunk batch unchanged:',config['batch_size'])
