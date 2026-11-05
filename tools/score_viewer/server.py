"""Read-only live evaluation status; no inference and no modification of results."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from datetime import datetime
import json,statistics,time
ROOT=Path(__file__).resolve().parents[2]
def read_json(path):
 for attempt in range(3):
  try: return json.loads(path.read_text(encoding='utf-8'))
  except json.JSONDecodeError:
   if attempt==2: raise
   time.sleep(.08)
def snapshot():
 root=Path(read_json(ROOT/'outputs/current_1700_recovery.json')['target']).resolve();root.relative_to(ROOT)
 status=read_json(root/'testset_status.json');run=Path(status['evaluation_output']).resolve();run.relative_to(ROOT)
 final=run/'scores.json';partial=run/'scores.partial.json'
 report=read_json(final) if final.exists() else None
 scores=report['scores'] if report else (read_json(partial) if partial.exists() else [])
 questions=read_json(run/'eval_input.json'); by_id={x['id']:x for x in scores}
 predictions=read_json(root/'banking/predictions.json')
 calls=[];callpath=run/'judge_calls.jsonl'
 if callpath.exists():
  for line in callpath.read_text().splitlines():
   try: calls.append(json.loads(line))
   except json.JSONDecodeError: continue
 coordinator_alive=Path('/proc/'+str(status['coordinator_pid'])).exists()
 pending=[q['id'] for q in questions if q['id'] not in by_id]
 current=pending[0] if pending and status['status']=='evaluate' and coordinator_alive else None
 keys=['context_relevancy','evidence_recall','rouge_l','answer_correctness']
 rows=[]
 for q in questions:
  ident=q['id'];scored=by_id.get(ident)
  metrics={k:{field:value for field,value in v.items() if field in ['score','status','error','reason']} for k,v in scored['metrics'].items()} if scored else {}
  rows.append({'id':ident,'question':q['question'],'question_type':q['question_type'],'state':('error' if any(v['status']!='ok' for v in metrics.values()) else 'complete') if scored else ('running' if ident==current else 'waiting'),'metrics':metrics,'judge_ok':sum(x.get('status')=='ok' and x['task'].split('/')[0]==ident for x in calls),'judge_errors':sum(x.get('status')=='error' and x['task'].split('/')[0]==ident for x in calls)})
 aggregates={}
 for key in keys:
  valid=[x['metrics'][key]['score'] for x in scores if key in x['metrics'] and x['metrics'][key]['status']=='ok']
  aggregates[key]={'mean':statistics.mean(valid) if valid else None,'valid_n':len(valid),'failed_n':sum(key in x['metrics'] and x['metrics'][key]['status']!='ok' for x in scores)}
 return {'checked_at':datetime.now().astimezone().isoformat(),'status':status['status'],'coordinator_alive':coordinator_alive,'expected':len(questions),'answered':len(predictions),'answer_ok':sum(x['status']=='ok' for x in predictions),'evaluated':len(scores),'current':current,'final_report_available':bool(report),'same_model_judge':report.get('same_model_judge') if report else True,'judge_ok':sum(x.get('status')=='ok' for x in calls),'judge_errors':sum(x.get('status')=='error' for x in calls),'judge_last_update':datetime.fromtimestamp(callpath.stat().st_mtime).astimezone().isoformat() if callpath.exists() else None,'rows':rows,'aggregates':aggregates,'recent_calls':[{k:x.get(k) for k in ['task','attempt','status','error']} for x in calls[-10:]][::-1],'evaluation_path':str(run)}
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args): pass
 def do_GET(self):
  try:
   if self.path.split('?')[0]=='/api/status':data=json.dumps(snapshot(),ensure_ascii=False).encode();ctype='application/json; charset=utf-8';code=200
   elif self.path=='/':data=(Path(__file__).parent/'index.html').read_bytes();ctype='text/html; charset=utf-8';code=200
   else:data=b'Not found';ctype='text/plain';code=404
   self.send_response(code);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(data)
  except (BrokenPipeError,ConnectionResetError):pass
  except Exception as e:
   print('STATUS_READ_ERROR',type(e).__name__,flush=True)
   self.send_response(503);self.end_headers()
if __name__=='__main__':
 server=ThreadingHTTPServer(('127.0.0.1',8791),Handler)
 print('SCORE_VIEWER_READY http://127.0.0.1:8791/',flush=True);server.serve_forever()
