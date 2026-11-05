"""Read-only diagnosis of the completed extraction attempt."""
from pathlib import Path
from datetime import datetime
import ast,json,os,hashlib,sqlite3,re
ROOT=Path('/home/ailab/data/HippoRAG/hipporag_local'); os.chdir(ROOT)
import sys
sys.path.insert(0,str(ROOT))
from settings import load_settings
cfg=load_settings(); options=json.loads(Path('business_config.json').read_text())
active=ROOT/'outputs/server_native_full_c4_20261030-105942'; index_dir=active/'banking/index'
from hipporag.utils.config_utils import BaseConfig
from hipporag.llm.openai_gpt import CacheOpenAI,normalize_generation_token_params
from hipporag.information_extraction.openie_openai import OpenIE,_extract_json_list_field
from hipporag.utils.llm_utils import fix_broken_generated_json
fn=next(n for n in ast.parse(Path('business.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='create_rag')
call=next(n for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='BaseConfig')
config=eval(compile(ast.Expression(body=call),'business.py','eval'))
llm=CacheOpenAI.from_experiment_config(config)
network_calls=[0]
def forbid_http(*args,**kwargs):
 network_calls[0]+=1; raise RuntimeError('Read-only audit forbids network inference')
llm.openai_client.chat.completions.create=forbid_http
openie=OpenIE(llm,ner_max_tokens=config.openie_ner_max_tokens,triple_max_tokens=config.openie_triple_max_tokens)
key_fn=next(n for n in ast.parse(Path('tools/benchmark_total_concurrency6.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='key_for')
exec(compile(ast.Module(body=[key_fn],type_ignores=[]),'key_for','exec'))
log=(active/'index.log').read_text(errors='replace')
failed=ast.literal_eval(re.search(r'RuntimeError: Triple extraction failed for \d+ chunk\(s\): (\[.*\])',log).group(1))
corpus=json.loads((active/'banking/corpus.json').read_text())
corpus_by_id={'chunk-'+hashlib.md5(x['text'].encode()).hexdigest():x for x in corpus}
report={'checked_at':datetime.now().astimezone().isoformat(),'main_pid_running':Path('/proc/2755556').exists(),'source_output':str(active),'failed_count':len(failed),'failed':[]}
db=Path(llm.cache_file_name)
try:
 with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as conn:
  report['cache_rows']=conn.execute('SELECT COUNT(*) FROM cache').fetchone()[0]
  report['sqlite_quick_check']=conn.execute('PRAGMA quick_check').fetchone()[0]
  for chunk_id in failed:
   chunk=corpus_by_id[chunk_id]
   ner=openie.ner(chunk_id,chunk['text'])
   assert not ner.metadata.get('error'), 'NER cache unavailable; no repair performed'
   messages=openie.prompt_template_manager.render(name='triple_extraction',passage=chunk['text'],named_entity_json=json.dumps({'named_entities':ner.unique_entities}))
   key=key_for(messages,openie.triple_max_tokens)
   row=conn.execute('SELECT message,metadata FROM cache WHERE key=?',(key,)).fetchone()
   result={'chunk_id':chunk_id,'source_id':chunk.get('source_id'),'title':chunk.get('metadata',{}).get('title'),'cache_key':key,'cached_response_present':row is not None}
   if row:
    raw,md=row[0],json.loads(row[1])
    result.update(finish_reason=md.get('finish_reason'),completion_tokens=md.get('completion_tokens'),response_characters=len(raw))
    try:
     parsed=_extract_json_list_field(fix_broken_generated_json(raw) if md.get('finish_reason')=='length' else raw,'triples')
     result['current_parse']='valid'; result['triple_count']=len(parsed)
    except ValueError as e:
     result['current_parse']='invalid'; result['parse_error']=str(e)
     decoder=json.JSONDecoder(); fields=[]
     for i,ch in enumerate(raw):
      if ch!='{': continue
      try:
       obj,_=decoder.raw_decode(raw[i:])
       if isinstance(obj,dict): fields=list(obj.keys())[:10]; break
      except ValueError: continue
     result['first_json_object_fields']=fields
   report['failed'].append(result)
 report['corpus_manifest']=json.loads((active/'banking/corpus.manifest.json').read_text())
 report['inference_network_calls']=network_calls[0]
 out=ROOT/'outputs'/('extraction_diagnosis_'+datetime.now().strftime('%Y%m%d-%H%M%S')); out.mkdir()
 (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(report,ensure_ascii=False,indent=2)); print('REPORT',out/'report.json')
finally: llm.close()
