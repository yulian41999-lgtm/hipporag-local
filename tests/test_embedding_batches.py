"""Verify per-store batching without loading a model or making network calls."""
import ast
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from embedding_batches import BatchSizeEmbedding

class FakeEmbedding:
    def __init__(self,*args):
        self.device='cuda'; self.embedding_dim=1024; self.calls=[]; self.closed=False
    def batch_encode(self,texts,**kwargs):
        self.calls.append((texts,kwargs)); return ['vector']
    def close(self): self.closed=True

class Tests(unittest.TestCase):
    def test_views_share_model_and_route_batches(self):
        model=FakeEmbedding(); entity=BatchSizeEmbedding(model,64); fact=BatchSizeEmbedding(model,32)
        self.assertEqual(entity.batch_encode(['entity'],norm=True),['vector'])
        fact.batch_encode(['fact'])
        self.assertEqual([x[1]['batch_size'] for x in model.calls],[64,32])
        self.assertTrue(model.calls[0][1]['norm'])
        self.assertIs(entity.embedding,fact.embedding)
        self.assertEqual(entity.embedding_dim,1024)
    def test_explicit_override(self):
        model=FakeEmbedding(); BatchSizeEmbedding(model,64).batch_encode(['x'],batch_size=8)
        self.assertEqual(model.calls[0][1]['batch_size'],8)
    def test_invalid_sizes(self):
        for value in (0,-1,True,1.5,'64',None):
            with self.subTest(value=value),self.assertRaises(ValueError): BatchSizeEmbedding(FakeEmbedding(),value)
    def test_business_store_integration(self):
        config=types.ModuleType('hipporag.utils.config_utils')
        config.BaseConfig=lambda **kw:types.SimpleNamespace(**kw)
        hippo=types.ModuleType('hipporag')
        def fake_rag(**kw):
            model=kw['embedding_model']
            return types.SimpleNamespace(**{name:types.SimpleNamespace(embedding_model=model) for name in ['chunk_embedding_store','entity_embedding_store','fact_embedding_store']})
        hippo.HippoRAG=fake_rag
        local=types.ModuleType('local_embedding'); local.LocalBGEEmbedding=FakeEmbedding
        tree=ast.parse((ROOT/'business.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create_rag')
        isolated=ast.Module(body=[fn],type_ignores=[])
        namespace={'json':json,'hashlib':hashlib,'os':types.SimpleNamespace(environ={})}
        modules={'hipporag':hippo,'hipporag.utils':types.ModuleType('hipporag.utils'),'hipporag.utils.config_utils':config,'local_embedding':local}
        cfg={'llm_model':'test','llm_base_url':'http://unused','llm_api_key_env':'TEST_KEY','embedding_model_id':'test','batch_size':1,'max_seq_length':512,'embedding_revision':'test','entity_batch_size':64,'fact_batch_size':32}
        options={'retrieval_top_k':5,'qa_top_k':5}
        with patch.dict(sys.modules,modules):
            exec(compile(isolated,str(ROOT/'business.py'),'exec'),namespace)
            rag,model=namespace['create_rag'](cfg,options,Path('unused'))
            self.assertIs(rag.chunk_embedding_store.embedding_model,model)
            rag.entity_embedding_store.embedding_model.batch_encode(['entity'])
            rag.fact_embedding_store.embedding_model.batch_encode(['fact'])
            self.assertEqual([x[1]['batch_size'] for x in model.calls],[64,32])
            cfg.pop('entity_batch_size'); cfg.pop('fact_batch_size')
            rag,_=namespace['create_rag'](cfg,options,Path('unused'))
            self.assertEqual(rag.entity_embedding_store.embedding_model.batch_size,1)

if __name__=='__main__': unittest.main()
