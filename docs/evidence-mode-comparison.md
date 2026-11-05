# 原文与仅图回答测评

入口为 `tools/compare_evidence_modes.py`，复用 `outputs/current_1700_recovery.json` 指向的完整索引。不得执行 prepare/index；索引不存在或不完整时直接失败。

```bash
.venv/bin/python -u tools/compare_evidence_modes.py --output outputs/evidence_mode_comparison_NEW_RUN
```

输出目录必须不存在。脚本依次运行原文问答、原文评分、仅图问答和仅图评分，保留历史结果。`HippoRAG.index` 在问答工作进程中被禁止调用，运行前后保存索引文件哈希。

## 模式与预算

原文模式沿用业务提示词，提交检索所得前5个原文候选中预算允许的内容。仅图模式提交相同检索流程选出的图事实种子三元组，不提交原文正文，不进行新抽取或子图扩展。两者均使用现有配置中的 BGE token 预算，生成输出上限2048。图模式附加三元组解释和防止编造原文引用的提示。

图事实来自建图阶段LLM抽取的事实索引，不是只有节点连通性的无语义拓扑。当前最多5个检索种子事实；结论只适用于这个适配配置，不能外推为所有图检索方法。

## 指标

`api_calls.jsonl` 记录检索重排和生成的实际服务端响应、usage、耗时、题号、阶段与入模消息。`answer_cost.json` 汇总输入、输出、总token，响应数，API响应耗时合计和平均，核心问答时间与整体日历时间。核心问答包含模型、索引加载和检索对象准备；逐题时间另存。当前没有预检调用。

`judge_api_calls.jsonl` 与 `judge_cost.json` 单独记录裁判开销；原始裁判重试及格式失败响应保留。SDK自动重试关闭，应用层裁判重试继续保留。网络失败无usage时用量未知，不当作零；usage缺失数量单独输出。

两模式使用新进程与独立空LLM响应缓存，不改变历史缓存。操作系统、服务端缓存及共享服务器负载无法控制，固定顺序单轮时间只作本次记录，不能视为稳定加速比。

`comparison.json`、`report.md` 汇总现有四指标和成本。`answer_checks.json` 检查空答案、资料编号范围和是否包含资料引用；这不是完整法律证据有效性判断。实际上下文与生成消息均保留供人工核验。

构图投入与业务问答、裁判成本分列。历史构图日志有记录则引用，无法确定完整历史用量时标注未记录。缓存恢复的逻辑token不可重复算作实际新增消耗。人民币、本地NER、BGE、CPU/GPU资源费用未纳入。

## 评分恢复

主入口在两轮完成并校验索引后，自动调用 `tools/repair_evidence_comparison.py`，仅补失败指标。沿用现有恢复策略，将裁判输出上限调至8192、超时240秒；成功评分复用，原始失败评分保留于 evaluation，补评保存于 evaluation_repaired。两模式使用同一恢复规则。补评消耗并入裁判成本，修复前汇总保存在 judge_cost.before_repair.json。最后调用 summarize_evidence_comparison.py 生成简明报告。
