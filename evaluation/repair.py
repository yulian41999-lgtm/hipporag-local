"""Repair failed metrics only; preserve successful scores and original artifacts."""
from pathlib import Path
from datetime import datetime
import copy,hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from settings import load_settings
from evaluation.judge import JSONJudge,parse_object
from evaluation.metrics import answer_correctness
from evaluation.run import validate_input,summarize

def atomic_json(path,value):
 temporary=path.with_name(path.name+'.new')
 temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 os.replace(temporary,path)

class ReusingJudge:
 def __init__(self,original_calls,delegate,reuse_log):
  self.delegate=delegate;self.reuse_log=reuse_log;self.reused=0;self.new=0;self.cache={}
  for row in original_calls:
   if row.get('status')=='ok':
    key=(row['task'],json.dumps(row['input'],sort_keys=True,ensure_ascii=False))
    self.cache[key]=row
 def call(self,task,instruction,payload,validator):
  key=(task,json.dumps(payload,sort_keys=True,ensure_ascii=False))
  if key in self.cache:
   row=self.cache[key];result=parse_object(row['raw_response']);validator(result)
   with self.reuse_log.open('a',encoding='utf-8') as f:
    f.write(json.dumps({'task':task,'status':'reused','original_attempt':row['attempt'],'payload_sha256':hashlib.sha256(key[1].encode()).hexdigest()},ensure_ascii=False)+'\n')
   self.reused+=1
   return result
  self.new+=1
  return self.delegate.call(task,instruction,payload,validator)

def main():
 os.chdir(ROOT);settings=load_settings()
 root=Path(json.loads((ROOT/'outputs/current_1700_recovery.json').read_text())['target']).resolve();root.relative_to(ROOT)
 state_path=root/'testset_status.json';state=json.loads(state_path.read_text());source=Path(state['evaluation_output']).resolve();source.relative_to(ROOT)
 assert not Path('/proc/'+str(state['coordinator_pid'])).exists(), 'Original evaluation is still running'
 original=json.loads((source/'scores.json').read_text());rows=json.loads((source/'eval_input.json').read_text())
 validate_input(rows)
 assert len(original['scores'])==15 and original['run_scope']=='formal_15'
 failed=[(r['id'],k) for r in original['scores'] for k,v in r['metrics'].items() if v['status']!='ok']
 assert failed and all(k=='answer_correctness' for _,k in failed), 'This repair handles correctness only'
 out=source.parent/('formal_15_repair_'+datetime.now().strftime('%Y%m%d-%H%M%S'));out.mkdir(exist_ok=False)
 import shutil
 for name in ('scores.json','eval_input.json','judge_calls.jsonl'):
  shutil.copy2(source/name,out/('original_'+name if name!='eval_input.json' else name))
 shutil.copy2(state_path,out/'testset_status.before.json')
 original_hashes={str(source/name):hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('scores.json','eval_input.json','judge_calls.jsonl')}
 config=copy.deepcopy(original['profile']);config.update(judge_max_tokens=8192,judge_timeout_seconds=240,judge_retries=1)
 calls=[json.loads(x) for x in (source/'judge_calls.jsonl').read_text().splitlines() if x.strip()]
 recovered=copy.deepcopy(original);recovered.pop('summary',None)
 recovered['repair']={'source_scores':str(source/'scores.json'),'original_artifact_sha256':original_hashes,'failed_metric_targets':[{'id':i,'metric':k} for i,k in failed],'output_limit':8192,'timeout_seconds':240,'formula_and_prompts_unchanged':True,'successful_scores_preserved':True,'started_at':datetime.now().astimezone().isoformat()}
 atomic_json(out/'scores.partial.json',recovered['scores'])
 state['previous_evaluations']=state.get('previous_evaluations',[])+[{'output':str(source),'status':state['status'],'failed_stage':state.get('failed_stage')}]
 state.update(status='repair',coordinator_pid=os.getpid(),evaluation_output=str(out),repair_targets=[i for i,k in failed])
 state.pop('failed_stage',None)
 state['stages'].append({'stage':'repair','pid':os.getpid(),'started_at':datetime.now().astimezone().isoformat(),'log':str(out/'repair.log')})
 atomic_json(state_path,state)
 log=(out/'repair.log').open('a',encoding='utf-8',buffering=1)
 def progress(*args):
  message=' '.join(map(str,args));print(message,flush=True);log.write(message+'\n')
 progress('REPAIR_STARTED',out,'TARGETS',failed)
 from hipporag.utils.config_utils import BaseConfig
 from local_embedding import LocalBGEEmbedding
 encoder_settings=dict(settings,max_seq_length=config['embedding_max_seq_length'],batch_size=1)
 encoder_config=BaseConfig(embedding_model_name=settings['embedding_model_id'],embedding_return_as_normalized=True)
 embedding=None;delegate=None
 try:
  embedding=LocalBGEEmbedding(encoder_config,encoder_settings)
  delegate=JSONJudge(config,out/'judge_calls.jsonl')
  judge=ReusingJudge(calls,delegate,out/'judge_reused.jsonl')
  inputs={r['id']:r for r in rows};score_rows={r['id']:r for r in recovered['scores']}
  for ident,key in failed:
   row=inputs[ident]
   for text in (row['generated_answer'],row['ground_truth']):
    length=len(embedding.model.tokenizer(text,truncation=False)['input_ids'])
    if length>embedding.model.max_seq_length: raise ValueError('Scoring text exceeds original scoring length; do not change protocol silently')
   progress('REPAIR_METRIC_START',ident,key)
   try:
    score,details=answer_correctness(row['question'],row['generated_answer'],row['ground_truth'],judge,embedding,ident)
    score_rows[ident]['metrics'][key]={'score':float(score),'status':'ok','details':details}
    progress('REPAIR_METRIC_OK',ident,round(float(score),6))
   except Exception as exc:
    score_rows[ident]['metrics'][key]={'score':None,'status':'error','error':type(exc).__name__}
    if isinstance(exc,ValueError):score_rows[ident]['metrics'][key]['reason']=str(exc)
    progress('REPAIR_METRIC_FAILED',ident,type(exc).__name__)
   atomic_json(out/'scores.partial.json',recovered['scores'])
  for before,after in zip(original['scores'],recovered['scores']):
   assert before['id']==after['id']
   for key,value in before['metrics'].items():
    if value['status']=='ok': assert value==after['metrics'][key], 'An original successful score changed'
  for path,digest in original_hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
  recovered['summary']=summarize(recovered['scores'])
  recovered['repair'].update(finished_at=datetime.now().astimezone().isoformat(),reused_judge_results=judge.reused,new_judge_tasks=judge.new,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'evaluation/repair.py',ROOT/'evaluation/metrics.py',ROOT/'evaluation/judge.py']})
  recovered['repair_runtime_profile']=config
  atomic_json(out/'scores.json',recovered)
  errors=[(r['id'],k) for r in recovered['scores'] for k,v in r['metrics'].items() if v['status']!='ok']
  state.update(status='failed' if errors else 'complete',scores_file=str(out/'scores.json'),summary=recovered['summary'],same_model_judge=recovered['same_model_judge'])
  state['stages'][-1].update(exit_code=2 if errors else 0,finished_at=datetime.now().astimezone().isoformat())
  if errors:state['failed_stage']='repair'
  atomic_json(state_path,state);progress('REPAIR_FINISHED','ERRORS',errors,'REUSED',judge.reused,'NEW_TASKS',judge.new)
  if errors:raise SystemExit(2)
 except BaseException:
  if state['status']=='repair':
   state['status']='failed';state['failed_stage']='repair';state['stages'][-1]['exit_code']=1;atomic_json(state_path,state)
  raise
 finally:
  if delegate:delegate.close()
  if embedding:embedding.close()
  log.close()
if __name__=='__main__':main()
