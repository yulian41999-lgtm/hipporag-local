"""Produce concise, auditable comparison without any model calls."""
import argparse,json,re
from pathlib import Path

def read(p): return json.loads(p.read_text())
def totals(rows):
    responses=[x for x in rows if x['status']=='response']; valid=[x for x in responses if x.get('usage')]
    return {'responses':len(responses),'input_tokens':sum(x['usage']['prompt_tokens'] for x in valid),'output_tokens':sum(x['usage']['completion_tokens'] for x in valid),'total_tokens':sum(x['usage']['total_tokens'] for x in valid),'api_response_seconds_sum':sum(x['elapsed_seconds'] for x in responses),'usage_missing':len(responses)-len(valid)}

def main(root):
    report=read(root/'comparison.json'); detail={}
    for mode in ['original','graph']:
        r=root/mode
        repaired=r/'evaluation_repaired/scores.json'
        if repaired.exists(): report[mode]['quality']=read(repaired)['summary']
        report[mode]['judge_cost']=read(r/'judge_cost.json')
        calls=[json.loads(x) for x in (r/'api_calls.jsonl').read_text().splitlines()]
        stages={stage:totals([x for x in calls if x['stage']==stage]) for stage in sorted({x['stage'] for x in calls})}
        pred=read(r/'predictions.json'); summary=report[mode]
        checks={'question_count':len(pred),'all_actual_messages_saved':all(p.get('generation_messages') for p in pred),
                'generation_errors':sum(p['status']=='generation_error' for p in pred),
                'empty_context_count':sum(not p['context'] for p in pred),
                'refusal_phrase_count':sum(any(s in p['generated_answer'] for s in ['无法回答','无法支持回答','资料不足','信息不足','依据不足']) for p in pred),
                'refusal_count_method':'heuristic phrase detection, not human adjudication',
                'graph_context_is_only_selected_triples':all(set(p['context']).issubset({json.dumps(list(t),ensure_ascii=False) for t in p['graph_seeds']}) for p in pred) if mode=='graph' else None}
        if mode=='graph': assert checks['graph_context_is_only_selected_triples']
        detail[mode]={'by_stage':stages,'verification':checks}
    historical=read(root/'historical_build_cost.json') if (root/'historical_build_cost.json').exists() else {'status':'not recorded'}
    report['historical_build_cost']=historical; report['stage_details']=detail
    report['index_integrity']=read(root/'index_integrity.json'); assert report['index_integrity']['unchanged']
    (root/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    lines=['# 原文与仅图回答：质量及开销对比','','两模式均复用现有1700份文档、5337分块的索引，各15题；未重新构图或重建索引。索引运行前后文件哈希一致。','',
           '## 问答成本','']
    for mode,label in [('original','依据原文'),('graph','仅图三元组')]:
        a=report[mode]['answer_cost']; d=detail[mode]
        lines += [f'- {label}：输入 {a["input_tokens"]:,}、输出 {a["output_tokens"]:,}、合计 {a["total_tokens"]:,} token；实际响应 {a["saved_api_responses"]} 次。',
                  f'  核心问答 {a["core_qa_seconds"]:.2f} 秒，整体日历 {a["overall_calendar_seconds"]:.2f} 秒；API响应耗时合计 {a["api_response_seconds_sum"]:.2f} 秒，平均 {a["api_response_seconds_mean"]:.2f} 秒。']
        for stage,v in d['by_stage'].items(): lines += [f'  {stage}：{v["responses"]} 次响应，{v["total_tokens"]:,} token。']
    savings=report['graph_savings']; lines += ['',f'仅图模式问答总token变化：节省 {savings["total_tokens"]:.2%}；本轮整体耗时变化：节省 {savings["overall_calendar_seconds"]:.2%}。负值表示增加。','', '## 质量','']
    for mode,label in [('original','依据原文'),('graph','仅图三元组')]:
        metrics=report[mode]['quality']['overall']['metrics']
        lines += [f'- {label}：'+ '；'.join(f'{k}={v["mean"]:.4f}（有效{v["valid_n"]}/{v["expected_n"]}）' if v['mean'] is not None else f'{k}=未获得有效值' for k,v in metrics.items())+'。']
        c=report[mode]['answer_checks']; v=detail[mode]['verification']
        lines += [f'  引用编号越界 {sum(not x["references_valid"] for x in c)} 题，无资料编号引用 {sum(not x["has_reference"] for x in c)} 题；空上下文 {v["empty_context_count"]} 题，含拒答相关短语 {v["refusal_phrase_count"]} 题（启发式检测）。']
    lines += ['', '## 裁判消耗（独立于业务问答）','']
    for mode,label in [('original','依据原文'),('graph','仅图三元组')]:
        a=report[mode]['judge_cost']; lines += [f'- {label}：{a["saved_api_responses"]} 次响应，输入 {a["input_tokens"]:,}、输出 {a["output_tokens"]:,}、合计 {a["total_tokens"]:,} token；评分经过时间 {a["overall_calendar_seconds"]:.2f} 秒。']
    lines += ['', '## 口径与限制','',
              '- 问答token包含检索重排及答案生成；裁判格式失败及重试保留。缺失usage或无响应失败不能视为零用量。无预检调用。',
              '- 各模式使用新进程和独立空LLM缓存；系统及服务端缓存、共享负载不可控。固定顺序单轮耗时不能解释为稳定加速比。',
              '- 图模式只提交最多5条现有事实三元组，不提交原文正文；两模式上下文预算相同，但实际证据量不同。不能将差异全部归因于表示格式。',
              '- 图模式增加三元组解释提示；生成模型及输出上限一致。质量为AI裁判初评；引用编号有效不代表原文证据已经人工核实。',
              '- 裁判截断失败按既有恢复规则补评：输出上限8192、超时240秒，成功评分保持不变；全部失败及恢复消耗并入裁判阶段。',
              '- 正确性是事实F1和语义相似度组合分，不是答对比例。',
              '- 历史恢复日志中NER和关系抽取全部命中缓存，新增抽取token为0；逻辑用量见 historical_build_cost.json。最初完整构图实际成本及完整耗时未核实，未记录项不补跑。',
              '- 不含BGE、本地NER、CPU/GPU及人民币费用；构图不重复计入两组问答。','',
              '## 可追溯输出','',
              '- comparison.json：完整质量、成本和分阶段汇总。',
              '- historical_metric_audit.json：旧测评指标覆盖审计。',
              '- historical_build_cost.json：历史构图日志引用及限制。',
              '- original/ 与 graph/：逐题答案、实际消息、API响应、评分及引用检查。',
              '- index_before.json、index_after.json、index_integrity.json：索引校验。']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'savings':savings,'integrity':report['index_integrity']},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);main(p.parse_args().output)
