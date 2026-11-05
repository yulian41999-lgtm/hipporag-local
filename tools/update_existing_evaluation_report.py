"""Update the existing report using saved results only; no model calls."""
import json,hashlib,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'outputs/evidence_mode_comparison_20261105-095518'
report=ROOT/'reports/单跳与多跳检索问答评测报告_2026-10-30.md'
backup=report.with_suffix('.md.before-20261105.bak')
if not backup.exists():shutil.copy2(report,backup)
old=backup.read_text();data=json.loads((run/'comparison.json').read_text())
assert data['index_integrity']['unchanged']
for mode in ['original','graph']:
 assert all(x['valid_n']==15 and x['failed_n']==0 for x in data[mode]['quality']['overall']['metrics'].values())
notice='> 更新日期：2026年11月5日。下方原有正文为2026年10月30日历史测评；文末新增本次“依据原文／仅图信息”质量与成本对比。两轮结果分别保留，不混算均值。\n\n'
first,body=old.split('\n',1)
text=first+'\n\n'+notice+body.lstrip('\n')
new=(run/'report.md').read_text().splitlines()[1:]
text+='\n\n## 2026年11月5日更新：原文与仅图回答对比\n\n'
for line in new:
 text+=(('#'+line) if line.startswith('## ') else line)+'\n'
names={'context_relevancy':'上下文相关性','evidence_recall':'证据召回','rouge_l':'ROUGE-L','answer_correctness':'答案正确性组合分'}
text+='\n### 本轮分组结果\n\n分组沿用历史题型标注，事实题10道、复杂推理题5道，不代表实际图跳数。每项均按题等权统计。\n\n'
for mode,label in [('original','依据原文'),('graph','仅图信息')]:
 text+=f'#### {label}\n\n'
 for group,values in data[mode]['quality'].items():
  if group=='overall':continue
  text+='- '+group+'：'+'；'.join(f'{names[k]} {v["mean"]:.4f}（有效{v["valid_n"]}/{v["expected_n"]}）' for k,v in values['metrics'].items())+'。\n'
 text+='\n'
text+='### 本轮逐题评分\n\n每题按“上下文相关性／证据召回／ROUGE-L／答案正确性组合分”顺序列示；均为补评后有效值。\n\n'
scores={}
for mode in ['original','graph']:
 p=run/mode/'evaluation_repaired/scores.json'
 if not p.exists():p=run/mode/'evaluation/scores.json'
 scores[mode]={x['id']:x for x in json.loads(p.read_text())['scores']}
for ident in scores['original']:
 vals={m:'／'.join(f'{scores[m][ident]["metrics"][k]["score"]:.4f}' for k in names) for m in scores}
 text+=f'- {ident}：原文 {vals["original"]}；仅图 {vals["graph"]}。\n'
text+='\n### 本轮审计与统计边界\n\n'
audit=json.loads((run/'historical_metric_audit.json').read_text())
for metric,status in audit['coverage'].items():text+=f'- {metric}：{status}。\n'
text+='\n原文模式在R001、R003、R004的正确性裁判输出达到3072 token上限后失败，按既有恢复流程将输出上限调至8192、超时240秒，仅补这3项。原成功评分保持不变；原始失败与恢复响应计入裁判成本。仅图模式无需补评。\n\n'
text+='两种模式问答与裁判合计实际消耗 '+format(sum(data[m][s]['total_tokens'] for m in ['original','graph'] for s in ['answer_cost','judge_cost']),',')+' token。此合计不含历史构图、BGE及本地资源费用。不能将裁判成本当作每次业务回答的额外成本。\n\n'
text+='本轮结果支持：当前最多5条图事实种子的配置能减少生成输入与问答token，但四项质量均下降。该结论不代表所有仅图方法均有同样表现；原文与图的实际证据量、提示词和一题检索种子存在差异，且同模型AI评分仍需人工复核。\n\n'
text+='### 本轮结果定位\n\n服务器本轮输出目录：\n\n```text\n'+str(run)+'\n```\n\n'
text+='原文最终评分位于 original/evaluation_repaired/scores.json；原始评分保留于 original/evaluation/scores.json。仅图最终评分位于 graph/evaluation/scores.json。逐题答案、实际上下文、API响应和耗时分别保存在两模式目录。报告更新不新增模型调用。\n'
report.write_text(text,encoding='utf-8')
manifest={'report':str(report),'historical_backup':str(backup),'backup_sha256':hashlib.sha256(backup.read_bytes()).hexdigest(),'updated_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'source_comparison':str(run/'comparison.json'),'new_model_calls':0}
(run/'report_update_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
assert body.strip() in text
assert '\x00' not in text
print(json.dumps(manifest,ensure_ascii=False))
