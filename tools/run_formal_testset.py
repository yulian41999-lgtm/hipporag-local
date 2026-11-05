"""Run the frozen 15 questions and existing evaluation, preserving outputs."""
from pathlib import Path
from datetime import datetime
import os,json,subprocess,hashlib,sys
ROOT=Path('/home/ailab/data/HippoRAG/hipporag_local').resolve(); os.chdir(ROOT)
setup=json.loads((ROOT/'outputs/current_1700_recovery.json').read_text()); output=Path(setup['target']).resolve(); output.relative_to(ROOT)
manifest=json.loads((output/'banking/index/business_corpus_manifest.json').read_text())
assert manifest['state']=='ready'
assert hashlib.sha256((output/'banking/corpus.json').read_bytes()).hexdigest()==manifest['corpus_sha256']
assert not (output/'banking/predictions.json').exists(), 'Predictions already exist; preserve them'
assert subprocess.run(['pgrep','-f','[p]ython.*business.py answer'],capture_output=True).returncode==1, 'Answer process already running'
stamp=datetime.now().strftime('%Y%m%d-%H%M%S')
run=output/'banking/evaluation'/('formal_15_'+stamp); run.mkdir(parents=True,exist_ok=False)
status_path=output/'testset_status.json'
assert not status_path.exists(), 'A previous test attempt exists'
env=os.environ.copy()
env.update(HIPPO_OUTPUT_DIR=str(output.relative_to(ROOT)),HIPPO_DEVICE='cuda',HIPPO_BATCH_SIZE='1',HIPPO_ENTITY_BATCH_SIZE='64',HIPPO_FACT_BATCH_SIZE='32',HIPPO_OPENIE_MAX_WORKERS='4')
report={'status':'starting','started_at':datetime.now().astimezone().isoformat(),'coordinator_pid':os.getpid(),'output':str(output),'evaluation_output':str(run),'expected_questions':15,'stages':[]}
def save():
 status_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
def stage(name,command,log_path):
 report['status']=name; item={'stage':name,'started_at':datetime.now().astimezone().isoformat(),'log':str(log_path)};report['stages'].append(item)
 with log_path.open('xb') as log:
  child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  item['pid']=child.pid; save(); print('STAGE_STARTED',name,child.pid,flush=True)
  code=child.wait()
 item['exit_code']=code; item['finished_at']=datetime.now().astimezone().isoformat(); save()
 if code:
  report['status']='failed'; report['failed_stage']=name;save();print('STAGE_FAILED',name,code,flush=True);raise SystemExit(code)
 print('STAGE_COMPLETE',name,flush=True)
save()
python=str(ROOT/'.venv/bin/python')
stage('answer',[python,'-u','business.py','answer'],output/'answer_testset.log')
predictions=json.loads((output/'banking/predictions.json').read_text())
assert len(predictions)==15 and all(p['status']=='ok' and p['generated_answer'].strip() and p['context'] for p in predictions)
report['prediction_count']=len(predictions);save()
eval_input=run/'eval_input.json'
stage('export',[python,'data/banking/银行监管测试集/tools/export_eval_input.py','--predictions',str(output/'banking/predictions.json'),'--output',str(eval_input)],run/'export.log')
stage('evaluate',[python,'-u','-m','evaluation.run','--input',str(eval_input),'--output-dir',str(run)],run/'evaluation.log')
scores=json.loads((run/'scores.json').read_text())
assert scores['run_scope']=='formal_15' and len(scores['scores'])==15
report.update(status='complete',finished_at=datetime.now().astimezone().isoformat(),scores_file=str(run/'scores.json'),summary=scores['summary'],same_model_judge=scores['same_model_judge'])
save();print('TESTSET_COMPLETE',str(status_path),flush=True)
