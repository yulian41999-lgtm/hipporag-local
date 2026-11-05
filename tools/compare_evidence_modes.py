"""Compare existing passage QA with graph-fact QA; never build an index."""
import argparse, hashlib, json, os, sys, time, subprocess
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from business import create_rag, dump
from business_qa import SYSTEM_PROMPT, final_answer
from settings import load_settings

def now(): return datetime.now(timezone.utc).isoformat()
def append(path,row):
    with Path(path).open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')

def install_meter(path,state):
    from openai.resources.chat.completions import Completions
    original=Completions.create
    def measured(self,*args,**kwargs):
        t=time.perf_counter(); row={'started_at':now(),**state,'messages':kwargs.get('messages'),'model':kwargs.get('model')}
        try:
            response=original(self,*args,**kwargs)
            row.update(status='response',response_id=response.id,usage=response.usage.model_dump() if response.usage else None,
                       raw_response=response.model_dump())
            return response
        except Exception as e:
            row.update(status='error',error=type(e).__name__)
            raise
        finally:
            row.update(elapsed_seconds=time.perf_counter()-t,finished_at=now()); append(path,row)
    Completions.create=measured

def usage_summary(path):
    rows=[json.loads(s) for s in Path(path).read_text().splitlines()] if Path(path).exists() else []
    responses=[r for r in rows if r.get('status')=='response']
    valid=[r for r in responses if r.get('usage') is not None]
    duration=sum(r['elapsed_seconds'] for r in rows)
    return {'api_attempts':len(rows),'saved_api_responses':len(responses),'responses_missing_usage':len(responses)-len(valid),
            'input_tokens':sum(r['usage']['prompt_tokens'] for r in valid),
            'output_tokens':sum(r['usage']['completion_tokens'] for r in valid),
            'total_tokens':sum(r['usage']['total_tokens'] for r in valid),
            'api_attempt_seconds_sum':duration,'api_response_seconds_sum':sum(r['elapsed_seconds'] for r in responses),
            'api_response_seconds_mean':sum(r['elapsed_seconds'] for r in responses)/len(responses) if responses else None,
            'accounting':'service usage; SDK retries disabled; errors without usage have unknown consumption'}

def snapshot(index):
    excluded={'llm_cache','__pycache__'}
    return {str(p.relative_to(index)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(index.rglob('*'))
            if p.is_file() and not any(x in excluded for x in p.relative_to(index).parts) and p.suffix!='.lock'}

def answer(mode,run,index):
    overall=time.perf_counter(); state={'stage':'initialization','question_id':None}; install_meter(run/'api_calls.jsonl',state)
    cfg=load_settings(); options=json.loads((ROOT/'business_config.json').read_text()); options['index_dir']=str(index)
    from hipporag import HippoRAG
    def forbidden(*a,**k): raise RuntimeError('Index/graph rebuild forbidden in evaluation')
    HippoRAG.index=forbidden
    manifest=json.loads((index/'business_corpus_manifest.json').read_text()); assert manifest['state']=='ready'
    core=time.perf_counter(); rag,embedding=create_rag(cfg,options,index)
    try:
        # Separate empty response caches preserve historical caches and give both modes cold local LLM caches.
        seen=set()
        for name in ('llm_model','extraction_llm','qa_llm'):
            llm=getattr(rag,name,None)
            if llm is None or id(llm) in seen: continue
            seen.add(id(llm)); llm.cache_file_name=str(run/f'qa_cache_{len(seen)}.sqlite')
            llm.openai_client.max_retries=0
        prep=time.perf_counter(); rag.prepare_retrieval_objects(); preparation=time.perf_counter()-prep
        questions=json.loads((ROOT/options['questions_file']).read_text()); assert len(questions)==15
        predictions=[]
        for q in questions:
            start=time.perf_counter(); state.update(stage='retrieval',question_id=q['id'])
            candidate=rag.retrieve([q['question']])[0]; retrieval=time.perf_counter()-start
            values=candidate.docs[:options['qa_top_k']] if mode=='original' else [json.dumps(list(t),ensure_ascii=False) for t in candidate.graph_seeds]
            selected=[]; tokenizer=embedding.model.tokenizer
            for text in values:
                proposal=selected+[text]; rendered='\n\n'.join(f'[资料{i}]\n{v}' for i,v in enumerate(proposal,1))
                if len(tokenizer(rendered,truncation=False)['input_ids'])>options['context_bge_token_budget']: break
                selected.append(text)
            system=SYSTEM_PROMPT if mode=='original' else SYSTEM_PROMPT+'\n本轮资料仅为图谱三元组，依次为主体、关系、客体。引用[资料编号]支持结论；没有原文条款，不得编造原文引文。路径不自动代表法律权益可传递。'
            messages=[{'role':'system','content':system},{'role':'user','content':'\n\n'.join(f'[资料{i}]\n{v}' for i,v in enumerate(selected,1))+'\n\n用户问题：'+q['question']}]
            row={**q,'mode':mode,'context':selected,'generation_messages':messages,'graph_seeds':candidate.graph_seeds,
                 'context_origin':'retrieved original passages' if mode=='original' else 'existing fact-store extracted triples; no source body',
                 'retrieval_seconds':retrieval,'generated_answer':'','generation_model':cfg['llm_model'],'context_budget_tokenizer':'BGE-M3',
                 'context_budget':options['context_bge_token_budget'],'max_new_tokens':2048}
            state['stage']='generation'; g=time.perf_counter()
            try:
                if not selected: row.update(status='insufficient_evidence',generated_answer='图中信息不足，无法支持回答。' if mode=='graph' else '资料不足，无法支持回答。')
                else:
                    raw,usage,hit=rag.qa_llm.infer(messages=messages,max_new_tokens=2048)
                    row.update(raw_response=raw,generation_usage=usage,cache_hit=hit,status='ok',generated_answer=final_answer(raw))
            except Exception as e: row.update(status='generation_error',error=type(e).__name__)
            row.update(generation_seconds=time.perf_counter()-g,latency_seconds=time.perf_counter()-start)
            predictions.append(row); dump(run/'predictions.json',predictions); print(mode,len(predictions),row['status'],flush=True)
        dump(run/'answer_cost.json',{'mode':mode,'core_qa_seconds':time.perf_counter()-core,'overall_calendar_seconds':time.perf_counter()-overall,
             'retrieval_preparation_seconds':preparation,'preflight_calls':0,'cache_condition':'fresh process and empty local LLM response cache; provider/OS cache uncontrolled',
             **usage_summary(run/'api_calls.jsonl')})
    finally: rag.close(); embedding.close()

def evaluate(run):
    from evaluation import run as evaluator
    state={'stage':'judge','question_id':None}; install_meter(run/'judge_api_calls.jsonl',state)
    sys.argv=['evaluation.run','--input',str(run/'eval_input.json'),'--output-dir',str(run/'evaluation')]
    started=time.perf_counter()
    try: evaluator.main()
    finally: dump(run/'judge_cost.json',{'overall_calendar_seconds':time.perf_counter()-started,**usage_summary(run/'judge_api_calls.jsonl')})

def coordinator(output):
    output.mkdir(parents=True,exist_ok=False)
    setup=json.loads((ROOT/'outputs/current_1700_recovery.json').read_text()); index=Path(setup['target'])/'banking/index'
    index.resolve().relative_to(ROOT); before=snapshot(index); dump(output/'index_before.json',before)
    dump(output/'protocol.json',{'index':str(index),'started_at':now(),'rebuild':False,'modes':['original','graph'],'historical_audit':{
        'generation_usage':'available','actual_context':'available','per_question_total_time':'available',
        'all_retrieval_api_usage':'not isolated','api_response_times':'not recorded','decision':'rerun both modes for consistent metering'},
        'historical_build_cost':{'status':'not recorded in comparison; no rebuild','input_tokens':None,'output_tokens':None,'total_tokens':None},
        'quality_metrics':['context_relevancy','evidence_recall','rouge_l','answer_correctness'],
        'graph_scope':'reranked existing fact triples used as retrieval seeds; no new extraction or graph expansion',
        'citation_check':'context references checked; graph triples are not verbatim source quotations'})
    commands=[]
    try:
        for mode in ('original','graph'):
            run=output/mode; run.mkdir()
            for phase in ('answer','export','evaluate'):
                command=([sys.executable,str(Path(__file__).resolve()),'--worker',phase,'--mode',mode,'--output',str(run),'--index',str(index)] if phase!='export' else
                    [sys.executable,'data/banking/银行监管测试集/tools/export_eval_input.py','--predictions',str(run/'predictions.json'),'--output',str(run/'eval_input.json')])
                with (run/f'{phase}.log').open('w') as log: code=subprocess.call(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                commands.append({'mode':mode,'phase':phase,'exit_code':code}); dump(output/'progress.json',commands)
                if code and phase!='evaluate': raise RuntimeError(f'{mode} {phase} failed')
        report={}
        for mode in ('original','graph'):
            r=output/mode; scores=json.loads((r/'evaluation/scores.json').read_text()); preds=json.loads((r/'predictions.json').read_text())
            import re
            checks=[{'id':p['id'],'status':p['status'],'empty_answer':not p['generated_answer'].strip(),
                     'references_valid':all(1<=int(i)<=len(p['context']) for i in re.findall(r'\[资料(\d+)\]',p['generated_answer'])),
                     'has_reference':bool(re.search(r'\[资料\d+\]',p['generated_answer']))} for p in preds]
            dump(r/'answer_checks.json',checks)
            report[mode]={'answer_cost':json.loads((r/'answer_cost.json').read_text()),'judge_cost':json.loads((r/'judge_cost.json').read_text()),'quality':scores['summary'],'answer_checks':checks}
        a=report['original']['answer_cost']; b=report['graph']['answer_cost']
        report['graph_savings']={k:1-b[k]/a[k] if a[k] else None for k in ('total_tokens','overall_calendar_seconds','core_qa_seconds')}
        dump(output/'comparison.json',report)
        text=['# 原文与仅图回答测评','', '复用现有图谱与索引。质量为现有四指标；费用统计限服务端LLM usage，非人民币总成本。','']
        for mode in ('original','graph'):
            text += ['## '+mode,'','```json',json.dumps(report[mode],ensure_ascii=False,indent=2),'```','']
        text += ['## 仅图模式节省比例','','```json',json.dumps(report['graph_savings'],indent=2),'```','', '缓存条件：各模式新进程及独立空LLM缓存；无法清除服务端、操作系统缓存。历史构图投入未补跑。']
        (output/'report.md').write_text('\n'.join(text),encoding='utf-8')
    finally:
        after=snapshot(index); dump(output/'index_after.json',after); dump(output/'index_integrity.json',{'unchanged':before==after,'before_file_count':len(before),'after_file_count':len(after)})
        if before!=after: raise RuntimeError('Existing index changed')
    subprocess.check_call([sys.executable,str(ROOT/'tools/repair_evidence_comparison.py'),str(output)],cwd=ROOT)
    subprocess.check_call([sys.executable,str(ROOT/'tools/summarize_evidence_comparison.py'),str(output)],cwd=ROOT)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--worker',choices=['answer','evaluate']); p.add_argument('--mode',choices=['original','graph']); p.add_argument('--index',type=Path); args=p.parse_args()
    os.chdir(ROOT); os.environ['HIPPO_DEVICE']='cuda'
    if args.worker=='answer': answer(args.mode,args.output,args.index)
    elif args.worker=='evaluate': evaluate(args.output)
    else: coordinator(args.output)
