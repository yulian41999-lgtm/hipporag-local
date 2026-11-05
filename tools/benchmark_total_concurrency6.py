"""Bounded prefetch experiment: main task stays at four workers."""
from pathlib import Path
import os,sys,time,json,ast,hashlib,sqlite3,random,re,urllib.request,statistics
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
ROOT=Path('/home/ailab/data/HippoRAG/hipporag_local').resolve(); os.chdir(ROOT); sys.path.insert(0,str(ROOT))
PID=2755556
ACTIVE=ROOT/'outputs/server_native_full_c4_20261030-105942'
assert Path(f'/proc/{PID}/cwd').resolve()==ROOT
assert b'business.py' in Path(f'/proc/{PID}/cmdline').read_bytes().split(b'\0')
parent_env=dict(x.split('=',1) for x in Path(f'/proc/{PID}/environ').read_bytes().decode().split('\0') if '=' in x)
assert parent_env['HIPPO_OUTPUT_DIR']==str(ACTIVE.relative_to(ROOT))
os.environ.update(parent_env)
from settings import load_settings
cfg=load_settings(); options=json.loads((ROOT/'business_config.json').read_text()); index_dir=ACTIVE/'banking/index'
from hipporag.utils.config_utils import BaseConfig
from hipporag.llm.openai_gpt import CacheOpenAI,normalize_generation_token_params
from hipporag.information_extraction.openie_openai import OpenIE,_extract_ner_from_response
from hipporag.utils.llm_utils import fix_broken_generated_json
fn=next(n for n in ast.parse((ROOT/'business.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='create_rag')
call=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='BaseConfig')
config=eval(compile(ast.Expression(body=call),'business.py','eval'),{'BaseConfig':BaseConfig,'cfg':cfg,'options':options,'index_dir':index_dir,'os':os})
assert config.openie_max_workers==4
llm=CacheOpenAI.from_experiment_config(config)
openie=OpenIE(llm,max_workers=2,ner_max_tokens=config.openie_ner_max_tokens,triple_max_tokens=config.openie_triple_max_tokens)
db=Path(llm.cache_file_name)
assert db.parent==index_dir/'llm_cache'
OUT=ROOT/'outputs'/('concurrency6_benchmark_'+datetime.now().strftime('%Y%m%d-%H%M%S')); OUT.mkdir(exist_ok=False)
report={'started_at':datetime.now().astimezone().isoformat(),'main_pid':PID,'main_concurrency_unchanged':4,'auxiliary_max_workers':2,'source_output':str(ACTIVE),'output':str(OUT),'stages':[],'auxiliary_results':[]}
metrics_url=cfg['llm_base_url'].rstrip('/').removesuffix('/v1')+'/metrics'
def key_for(messages,limit):
 d=normalize_generation_token_params(llm.llm_config.generate_params,{'messages':messages,'max_new_tokens':limit},'_max_tokens')
 d.update(messages=messages,_cache_schema=3,_provider_class=f'{llm.__class__.__module__}.{llm.__class__.__qualname__}',_endpoint=config.azure_endpoint or config.llm_base_url,_azure_api_version=config.azure_api_version,_azure_chat_deployment=config.azure_chat_deployment)
 return hashlib.sha256(json.dumps(d,sort_keys=True,default=str).encode()).hexdigest()
def connection(): return sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True,timeout=5)
with connection() as conn: cache={k:(m,json.loads(md)) for k,m,md in conn.execute('SELECT key,message,metadata FROM cache')}
corpus=json.loads((ACTIVE/'banking/corpus.json').read_text())
candidates=[]; matched=[]; ner_hits=0
for chunk in corpus:
 text=chunk['text']; chunk_id='chunk-'+hashlib.md5(text.encode()).hexdigest()
 messages=openie.prompt_template_manager.render(name='ner',passage=text)
 ner_key=key_for(messages,openie.ner_max_tokens)
 if ner_key not in cache: continue
 raw,md=cache[ner_key]
 try:
  parsed=_extract_ner_from_response(fix_broken_generated_json(raw) if md.get('finish_reason')=='length' else raw)
  if not all(isinstance(x,str) for x in parsed): continue
  entities=list(dict.fromkeys(parsed))
 except (ValueError,TypeError): continue
 ner_hits+=1
 messages=openie.prompt_template_manager.render(name='triple_extraction',passage=text,named_entity_json=json.dumps({'named_entities':entities}))
 key=key_for(messages,openie.triple_max_tokens); row=(key,chunk_id,text,entities,messages)
 if key in cache: matched.append(row)
 else: candidates.append(row)
assert len(matched)>=10 and len(candidates)>=100, 'Cache compatibility preflight failed'
original_create=llm.openai_client.chat.completions.create
network_calls=[0]
def forbid_http(*args,**kwargs):
 network_calls[0]+=1
 raise RuntimeError('Preflight attempted uncached request')
llm.openai_client.chat.completions.create=forbid_http
try:
 for key,chunk_id,text,entities,messages in matched[:10]:
  response,md,hit=llm.infer(messages=messages,max_new_tokens=openie.triple_max_tokens)
  assert hit and response==cache[key][0]
finally: llm.openai_client.chat.completions.create=original_create
assert network_calls[0]==0
report['preflight']={'cached_ner_matches':ner_hits,'cached_triple_matches':len(matched),'uncached_candidates':len(candidates),'verified_cache_hits':10,'preflight_network_calls':0}
random.Random(20261030).shuffle(candidates); iterator=iter(candidates)
print('PREFLIGHT_OK',json.dumps(report['preflight']),flush=True); print('OUTPUT',OUT,flush=True)
def save(): (OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
def snapshot():
 with urllib.request.urlopen(metrics_url,timeout=5) as response: raw=response.read(2_000_000).decode()
 metrics={}
 wanted=('num_requests_running','num_requests_waiting','kv_cache_usage_perc','generation_tokens_total','prompt_tokens_total','request_success_total','e2e_request_latency_seconds_count','e2e_request_latency_seconds_sum','request_queue_time_seconds_count','request_queue_time_seconds_sum')
 for line in raw.splitlines():
  if line.startswith('#') or not any('vllm:'+x+'{' in line for x in wanted): continue
  if 'model_name="'+cfg['llm_model']+'"' not in line: continue
  name=line.split('{',1)[0].removeprefix('vllm:')
  try: metrics[name]=metrics.get(name,0)+float(line.rsplit(' ',1)[-1])
  except ValueError: pass
 with connection() as conn: rows=conn.execute('SELECT COUNT(*) FROM cache').fetchone()[0]
 return {'time':time.time(),'at':datetime.now().astimezone().isoformat(),'cache_rows':rows,'metrics':metrics}
def summarize(start,end,samples):
 dt=end['time']-start['time']; a=start['metrics']; b=end['metrics']
 delta=lambda name:b.get(name,0)-a.get(name,0)
 n=delta('e2e_request_latency_seconds_count'); qn=delta('request_queue_time_seconds_count')
 return {'seconds':round(dt,2),'new_cache_rows':end['cache_rows']-start['cache_rows'],'new_cache_rows_per_minute':round((end['cache_rows']-start['cache_rows'])*60/dt,3),'server_requests_per_minute':round(delta('request_success_total')*60/dt,3),'server_output_tokens_per_second':round(delta('generation_tokens_total')/dt,3),'server_mean_request_seconds':round(delta('e2e_request_latency_seconds_sum')/n,3) if n else None,'server_mean_queue_seconds':round(delta('request_queue_time_seconds_sum')/qn,3) if qn else None,'observed_running_mean':round(statistics.mean(s['metrics'].get('num_requests_running',0) for s in samples),3),'observed_running_max':max(s['metrics'].get('num_requests_running',0) for s in samples),'observed_waiting_max':max(s['metrics'].get('num_requests_waiting',0) for s in samples)}
def observe(name,seconds):
 stage={'name':name,'auxiliary_workers':0,'samples':[]}; report['stages'].append(stage)
 start=snapshot(); stage['samples'].append(start); save(); deadline=time.monotonic()+seconds
 print('STAGE_START',name,flush=True)
 while time.monotonic()<deadline:
  time.sleep(min(5,max(0,deadline-time.monotonic())))
  stage['samples'].append(snapshot()); save()
  if not Path(f'/proc/{PID}').exists(): raise RuntimeError('Main process ended')
 stage['summary']=summarize(start,stage['samples'][-1],stage['samples']); save()
 print('STAGE_COMPLETE',name,json.dumps(stage['summary']),flush=True)
def worker(row):
 key,chunk_id,text,entities,messages=row
 start=time.monotonic(); result=openie.triple_extraction(chunk_id,text,entities); md=result.metadata
 return {'chunk_id':chunk_id,'cache_key':key,'seconds':round(time.monotonic()-start,3),'cache_hit':bool(md.get('cache_hit')),'status':'error' if md.get('error') else 'ok','finish_reason':md.get('finish_reason'),'prompt_tokens':md.get('prompt_tokens',0),'completion_tokens':md.get('completion_tokens',0),'triple_count':len(result.triples)}
try:
 observe('baseline4_before',120)
 stage={'name':'total6','auxiliary_workers':2,'samples':[]}; report['stages'].append(stage)
 start=snapshot(); stage['samples'].append(start); save(); deadline=time.monotonic()+300; next_sample=time.monotonic()+5
 pending=set(); submitted=0; errors=0; reason='duration_complete'
 print('STAGE_START total6',flush=True)
 with ThreadPoolExecutor(max_workers=2) as pool:
  while time.monotonic()<deadline:
   if not Path(f'/proc/{PID}').exists(): reason='main_process_ended'; break
   if errors>=2: reason='two_auxiliary_errors'; break
   if stage['samples'][-1]['metrics'].get('num_requests_waiting',0)>=4: reason='queue_guard'; break
   while len(pending)<2 and submitted<200:
    row=next(iterator,None)
    if row is None: break
    with connection() as conn: exists=conn.execute('SELECT 1 FROM cache WHERE key=?',(row[0],)).fetchone()
    if exists: continue
    pending.add(pool.submit(worker,row)); submitted+=1
   done,pending=wait(pending,timeout=min(1,max(0,deadline-time.monotonic())),return_when=FIRST_COMPLETED)
   for future in done:
    result=future.result(); report['auxiliary_results'].append(result); errors+=result['status']=='error'
   if time.monotonic()>=next_sample:
    stage['samples'].append(snapshot()); save(); next_sample=time.monotonic()+5
  stage['samples'].append(snapshot())
  stage['summary']=summarize(start,stage['samples'][-1],stage['samples']); stage['submission_end_reason']=reason; stage['submitted']=submitted; save()
  print('SUBMISSIONS_FINISHED',json.dumps(stage['summary']),flush=True)
  for future in pending: report['auxiliary_results'].append(future.result())
 stage['drained_at']=datetime.now().astimezone().isoformat(); save()
 print('AUXILIARY_REQUESTS_DRAINED',len(report['auxiliary_results']),flush=True)
 if Path(f'/proc/{PID}').exists(): observe('baseline4_after',120)
 report['completed_at']=datetime.now().astimezone().isoformat(); report['status']='complete'; save()
 print('BENCHMARK_COMPLETE',OUT,flush=True)
finally: llm.close()
