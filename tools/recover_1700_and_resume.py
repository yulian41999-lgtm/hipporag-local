"""Recover only failed triples, audit cache reuse, then start one fresh index."""
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
import os,sys,json,ast,re,hashlib,sqlite3,subprocess,shutil
ROOT=Path('/home/ailab/data/HippoRAG/hipporag_local'); os.chdir(ROOT); sys.path.insert(0,str(ROOT))
setup=json.loads(Path('outputs/current_1700_recovery.json').read_text())
TARGET=Path(setup['target']); SOURCE=Path(setup['source'])
assert not Path('/proc/2755556').exists()
os.environ.update(HIPPO_OUTPUT_DIR=str(TARGET.relative_to(ROOT)),HIPPO_DEVICE='cuda',HIPPO_BATCH_SIZE='1',HIPPO_ENTITY_BATCH_SIZE='64',HIPPO_FACT_BATCH_SIZE='32',HIPPO_OPENIE_MAX_WORKERS='4')
from settings import load_settings
cfg=load_settings(); options=json.loads(Path('business_config.json').read_text()); index_dir=TARGET/'banking/index'
from hipporag.utils.config_utils import BaseConfig
from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.information_extraction.openie_openai import OpenIE
fn=next(n for n in ast.parse(Path('business.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='create_rag')
call=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='BaseConfig')
config=eval(compile(ast.Expression(body=call),'business.py','eval'))
llm=CacheOpenAI.from_experiment_config(config)
openie=OpenIE(llm,max_workers=4,ner_max_tokens=config.openie_ner_max_tokens,triple_max_tokens=config.openie_triple_max_tokens)
source_log=(SOURCE/'index.log').read_text(errors='replace')
failed=ast.literal_eval(re.search(r'RuntimeError: Triple extraction failed for \d+ chunk\(s\): (\[.*\])',source_log).group(1))
corpus=json.loads((TARGET/'banking/corpus.json').read_text())
by_id={'chunk-'+hashlib.md5(x['text'].encode()).hexdigest():x for x in corpus}
assert len(by_id)==5337 and len(failed)==5
original_create=llm.openai_client.chat.completions.create
ner_http_calls=[0]
def forbid_ner_http(*args,**kwargs):
 ner_http_calls[0]+=1; raise RuntimeError('NER must reuse existing cache')
llm.openai_client.chat.completions.create=forbid_ner_http
entities={}
try:
 for key in failed:
  ner=openie.ner(key,by_id[key]['text'])
  assert not ner.metadata.get('error') and ner.metadata['cache_hit']
  entities[key]=ner.unique_entities
finally: llm.openai_client.chat.completions.create=original_create
assert ner_http_calls[0]==0
report={**setup,'started_at':datetime.now().astimezone().isoformat(),'failed_recovery':[]}
def save(): (TARGET/'recovery_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
def repair(key):
 result=openie.triple_extraction(key,by_id[key]['text'],entities[key])
 return {'chunk_id':key,'source_id':by_id[key]['source_id'],'triples':result.triples,'metadata':result.metadata}
try:
 with ThreadPoolExecutor(max_workers=2) as pool:
  for future in as_completed([pool.submit(repair,key) for key in failed]):
   row=future.result(); report['failed_recovery'].append(row); save()
   print('REPAIRED',row['source_id'],'TRIPLES',len(row['triples']),'ERROR',row['metadata'].get('error'),'NEW_PROMPT_TOKENS',row['metadata'].get('billable_prompt_tokens'),flush=True)
 assert all(not x['metadata'].get('error') for x in report['failed_recovery']), 'Some failed chunks still need repair; index not started'
 assert all(len(x['triples'])>0 for x in report['failed_recovery']), 'Empty repaired output requires manual review; index not started'
 audit_http=[0]
 def forbid_audit_http(*args,**kwargs):
  audit_http[0]+=1; raise RuntimeError('Cache-only audit forbids uncached inference')
 llm.openai_client.chat.completions.create=forbid_audit_http
 print('CACHE_ONLY_AUDIT_START',flush=True)
 chunks={key:{'content':value['text']} for key,value in by_id.items()}
 ner_results,triple_results=openie.batch_openie(chunks)
 assert len(ner_results)==len(triple_results)==5337 and audit_http[0]==0
 assert all(x.metadata.get('cache_hit') and not x.metadata.get('error') for x in triple_results.values())
 report['cache_audit']={'chunks':5337,'ner_cache_hits':sum(bool(x.metadata.get('cache_hit')) for x in ner_results.values()),'triple_cache_hits':sum(bool(x.metadata.get('cache_hit')) for x in triple_results.values()),'uncached_inference_attempts':audit_http[0],'parsed_triples_before_deduplication':sum(len(x.triples) for x in triple_results.values())}
 print('CACHE_ONLY_AUDIT_PASS',json.dumps(report['cache_audit']),flush=True)
 with sqlite3.connect(Path(llm.cache_file_name).resolve().as_uri()+'?mode=ro',uri=True) as conn:
  assert conn.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
  report['recovered_cache_rows']=conn.execute('SELECT COUNT(*) FROM cache').fetchone()[0]
 report['status']='validated'; save()
finally: llm.close()
# No signal is sent to any process. Refuse a second index task.
check=subprocess.run(['pgrep','-f','[p]ython.*business.py index'],capture_output=True,text=True)
assert check.returncode==1, 'An index task is already running; replacement not started'
for name in ('business.py','settings.py','embedding_batches.py','config.json','business_config.json'):
 shutil.copy2(ROOT/name,Path(setup['backup'])/('launched_'+name))
env=os.environ.copy()
with (TARGET/'index.log').open('xb') as log:
 log.write(b'Recovery: documents=1700, chunks=5337, OpenIE=4, entity_batch=64, fact_batch=32; all extraction requests verified cached.\n'); log.flush()
 child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','business.py','index'],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
report['new_pid']=child.pid; report['launched_at']=datetime.now().astimezone().isoformat(); report['status']='index_started'; save()
(TARGET/'index.pid').write_text(str(child.pid)+'\n')
print('INDEX_STARTED',child.pid,'OUTPUT',TARGET,flush=True)
