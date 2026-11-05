"""Benchmark local embeddings using cached triples; never modify the index."""
from pathlib import Path
import sys
ROOT=Path('/home/ailab/data/HippoRAG/hipporag_local').resolve()
sys.path.insert(0,str(ROOT))
import json,sqlite3,time,statistics,random,hashlib,os
from datetime import datetime
import numpy as np
from settings import load_settings
cfg=load_settings()
from hipporag.utils.config_utils import BaseConfig
from local_embedding import LocalBGEEmbedding
import torch
os.chdir(ROOT)
OUT=ROOT/'outputs'/('batch_benchmark_'+datetime.now().strftime('%Y%m%d-%H%M%S'))
OUT.mkdir(exist_ok=False)
SOURCE=ROOT/'outputs/server_native_full/banking/index/llm_cache'
db=next(SOURCE.glob('*.sqlite'))
decoder=json.JSONDecoder()
entities=set(); facts=set()
with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as conn:
    for (message,) in conn.execute('SELECT message FROM cache'):
        for pos,ch in enumerate(message):
            if ch!='{': continue
            try: obj,_=decoder.raw_decode(message[pos:])
            except ValueError: continue
            if not isinstance(obj,dict) or not isinstance(obj.get('triples'),list): continue
            for triple in obj['triples']:
                if isinstance(triple,list) and len(triple)==3 and all(isinstance(s,str) for s in triple):
                    facts.add(str(triple)); entities.update((triple[0],triple[2]))
            break
assert len(entities)>=256 and len(facts)>=256, (len(entities),len(facts))

def sample(values):
    values=sorted(values,key=lambda s:(len(s),s))
    chosen=[values[round(i*(len(values)-1)/255)] for i in range(256)]
    random.Random(42).shuffle(chosen)
    return chosen

groups={'entities':sample(entities),'facts':sample(facts)}
cfg.update(device='cuda',batch_size=1)
torch.cuda.set_per_process_memory_fraction(0.35,0)
base=BaseConfig(embedding_model_name=cfg['embedding_model_id'],embedding_return_as_normalized=True)
encoder=LocalBGEEmbedding(base,cfg)
report={'created_at':datetime.now().astimezone().isoformat(),'source_database':str(db),'model':cfg['embedding_model_id'],'max_seq_length':cfg['max_seq_length'],'precision':'float16','samples_per_group':256,'repeats':2,'memory_cap_fraction':0.35,'rows':[],'sample_stats':{}}
(OUT/'sample_hashes.json').write_text(json.dumps({k:[hashlib.sha256(s.encode()).hexdigest() for s in v] for k,v in groups.items()},indent=2))
print('BENCHMARK_OUTPUT',OUT,flush=True)
try:
    encoder.batch_encode(['编码预热测试'],batch_size=1)
    torch.cuda.synchronize()
    for kind,texts in groups.items():
        lengths=[len(encoder.model.tokenizer.encode(s,truncation=False)) for s in texts]
        report['sample_stats'][kind]={'available_unique':len(entities if kind=='entities' else facts),'token_min':min(lengths),'token_median':statistics.median(lengths),'token_p95':float(np.percentile(lengths,95)),'token_max':max(lengths)}
        baseline=None
        for batch in (1,4,8,16,32,64):
            if not Path('/proc/2755556').exists():
                print('STOP_BENCHMARK: index process no longer present',flush=True)
                raise SystemExit(2)
            free,total=torch.cuda.mem_get_info()
            if free<8*1024**3:
                print('STOP_BENCHMARK: insufficient GPU headroom',flush=True)
                raise SystemExit(2)
            try:
                encoder.batch_encode(texts[:batch],batch_size=batch)
                torch.cuda.synchronize(); torch.cuda.empty_cache()
                durations=[]; peak_alloc=[]; peak_reserved=[]; vectors=None
                for repeat in range(2):
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize(); start=time.perf_counter()
                    vectors=encoder.batch_encode(texts,batch_size=batch)
                    torch.cuda.synchronize(); durations.append(time.perf_counter()-start)
                    peak_alloc.append(torch.cuda.max_memory_allocated()/1024**2)
                    peak_reserved.append(torch.cuda.max_memory_reserved()/1024**2)
                elapsed=statistics.median(durations)
                assert np.isfinite(vectors).all()
                if baseline is None: baseline=vectors.copy()
                cosine=np.sum(baseline*vectors,axis=1)/(np.linalg.norm(baseline,axis=1)*np.linalg.norm(vectors,axis=1))
                row={'group':kind,'batch_size':batch,'texts_per_second':round(len(texts)/elapsed,2),'seconds':list(map(lambda x:round(x,4),durations)),'peak_allocated_mib':round(max(peak_alloc),1),'peak_reserved_mib':round(max(peak_reserved),1),'min_cosine_vs_batch1':round(float(cosine.min()),7),'mean_cosine_vs_batch1':round(float(cosine.mean()),7)}
                report['rows'].append(row)
                print(json.dumps(row),flush=True)
                (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print('BATCH_OOM_STOP',kind,batch,flush=True)
                break
finally:
    encoder.close()
print('BENCHMARK_COMPLETE',OUT,flush=True)
