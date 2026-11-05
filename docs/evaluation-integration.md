# 六指标评测接入分析

历史方案：用户后续已将生成为 ROUGE-L 和 Correctness、检索为 Context Relevancy 和 Evidence Recall，共四项。当前实现与运行说明以 [四指标与中文业务问答](evaluation-four-metrics.md) 为准；本文保留早期分析。

分析日期：2026-10-29。范围为 GraphRAG-Bench 的两项检索指标和四项生成指标；本文是接入方案，尚未实现正式评测，也未构建 Docker 镜像。

## 接入结论

在 HippoRAG 外增加独立的运行结果记录层和评分层，复用 GraphRAG-Bench 的指标定义、公式及提示词，并显式记录中文适配。正式 15 道题全部计算六项指标。回答生成只读取问题及检索内容，评分阶段再合并标准答案、标准证据和评分点。

不直接把上游整个 requirements.txt 安装进现有环境：上游固定 datasets 3.3.2、ragas 0.2.15、LangChain 等依赖，而当前项目已有验证过的依赖组合。建议使用现有 OpenAI SDK、本地 BGE-M3 和独立 ROUGE 依赖实现薄适配。实施时固定上游 commit，并保留来源、许可证和差异清单；本轮查看的是官方 main 页面，尚未将其作为固定源码版本引入。

## 六项指标与计算口径

所有得分以 0—1 存储，展示时可乘 100，但不得称 Answer Correctness 为答对题数比例。

| 指标 | 上游实际实现 | 我们的接入方式 |
| --- | --- | --- |
| Context Relevancy | 将上下文拼接，LLM 两次按 0、1、2 分打分，归一化后取平均 | 首轮保留默认实现语义，输入实际入模上下文；不把它解释成相关块占比 |
| Evidence Recall | 判断每条标准证据能否从上下文得到支持，取命中比例 | 使用 evidence_sources[].quote，保留每条证据的判定及原因 |
| ROUGE-L | 最长公共子序列的 F1，使用默认英文 tokenizer | 使用自定义中文字符 tokenizer，英文和数字按连续串处理，关闭词干化；标注为中文适配版本 |
| Answer Correctness | 0.75 × 事实陈述 F1 + 0.25 × 归一化余弦相似度 | 保留组合公式；LLM 判定 TP/FP/FN，BGE-M3 计算语义相似度，两个分量分别保存 |
| Coverage Score | 从标准答案抽取事实，再判断生成答案覆盖比例 | 首轮保留从完整标准答案抽取事实的流程；固定并缓存参考事实表 |
| Faithfulness | 从生成答案抽取事实陈述，判断其中受实际上下文支持的比例 | 不向该判定提供标准答案；保存逐陈述支持与不支持的理由 |

Answer Correctness 中事实陈述 F1 为 `2TP / (2TP + FP + FN)`，语义项为 `(cosine + 1) / 2`。两者应分开显示，避免总体相似度掩盖事实错误。

Faithfulness 是对检索依据的忠实程度，不等同于外部事实核查。答案可以忠实地复述一段错误或不适用的材料，因此应与 Correctness 同时解释。

官方指标入口从 context_relevance.py 导入默认相关性函数；另有 context_relevance_v2.py，后者使用标准证据并按块评分。两者不混用；如后续选择 v2，应固定命名、版本及汇总方式。

## 必须处理的实现差异

### 题型与指标分配

现有题库有 10 道 Fact Retrieval、5 道 Complex Reasoning。上游 generation_eval.py 对这两类默认只计算 ROUGE-L 和 Answer Correctness；Coverage 默认分配给总结和创作题，Faithfulness 默认分配给创作题。

我们保持题型不变，新增明确的六指标配置。向事实题和推理题补充 Coverage、Faithfulness 属于扩展设置，报告中应说明，而不是为了触发上游分支改写题型。

### 中文 ROUGE

上游 rouge.py 使用 rouge_score 的默认 tokenizer，其正则仅保留英文及数字，不能直接用于中文法规答案。仅在汉字间插入空格仍会被过滤。必须传入能够保留中文字符的 tokenizer 对象。

以完整标准答案和最终答案计算中文 ROUGE-L，保留完整参考文本，不擅自改用 answer_summary。标点、空白、数字和英文的处理规则固定，并使用“完全相同的中文文本得到 1 分”等对照样例验证实现。

### 真实上下文与最终答案

当前 HippoRAG 的 qa() 仅将 query_solution.docs[:qa_top_k] 送入回答模型，因此必须从同一条推理记录保存实际入模段落及其顺序；不能把更多检索候选作为 Faithfulness 的依据。若另行报告候选检索表现，应明确标为 candidate 指标并保留 k 值。

当前 MuSiQue 模板要求最终答案简短，并由 Answer: 后提取答案。业务题标准答案长约 433—1384 字，包含结论、依据和适用分析。正式评测前应加入中文业务问答模板，要求依据检索资料形成完整答复；仅向模型提供通用结构要求，不提供本题标准答案、评分点或标准证据。

保存原始响应、最终用户可见答案及解析状态。评分针对最终答案，不把额外推理文本擅自拼回答案以改善分数。

### 长度与 Embedding

用现有 BGE-M3 tokenizer 实测 15 份完整标准答案：最大 890 tokens，4 份超过当前运行配置的 512 tokens。评测语义相似度不能直接复用该 512-token 上限，否则部分参考答案会被截断。

建议评测编码器独立配置 max_seq_length=2048、batch_size=1。实际运行前检查标准答案及生成答案的 token 数；若仍超限，显式调整并记录，不能静默截断。该评分配置不改变检索索引的 Embedding 配置或向量。

上游 Context Relevancy、Evidence Recall、Coverage、Faithfulness 存在按字符截断的实现。适配时改为明确的输入长度检查及必要的逐事实分批判断，保存实际评分输入和截断状态。任何分批或超长处理都进入适配版本说明。

### 完整性与失败状态

上游有的分支返回 NaN、有的异常返回 0，汇总还会跳过部分无效分数。我们应保留 15 条题目记录，并为每个指标单独记录状态。

- 系统无检索结果：保留空 context，检索指标计 0；非空答案的 Faithfulness 计 0。
- 系统无答案：生成指标计 0，并记录 empty_answer；避免沿用上游空答案 Faithfulness=1 的边界设置。
- 裁判请求失败或 JSON 无效：指标设为 null，记录 error，可重试；不伪装成模型回答错误，也不默默从报告中消失。
- 标准证据和参考事实的分类输出须一一对应、无遗漏、无重复；分母来自固定输入清单，不能来自裁判实际返回的条目数。

上述空答案及错误处理属于明确的本地适配，不称为上游逐字复现。

## 数据接线

```text
原始全量法规 → 清洗、去重、切块 → HippoRAG 索引
questions.json → 检索 → 保存实际入模上下文 → 生成最终答案
                                          ↓
                                     predictions.json
                                          ↓
                 与 banking_qa_15.json 按 id 合并
                                          ↓
                              eval_input.json
                                          ↓
                       六指标评分 → 逐题表与汇总表
```

现有 tools/export_eval_input.py 已实现基础合并：question、question_type、ground_truth、generated_answer、context、evidence。正式推理仅使用 questions.json；标准答案在推理结束后才进入评测阶段。

| 字段 | 来源 | 用途 |
| --- | --- | --- |
| question | questions.json | 检索、生成及评分 |
| ground_truth | 正式题库 answer | ROUGE-L、Correctness、Coverage |
| generated_answer | 实际最终答案 | 四项生成指标 |
| context | 实际送入回答模型的段落 | 两项检索指标、Faithfulness |
| evidence | evidence_sources[].quote，共 22 处 | Evidence Recall |
| answer_key_points | 正式题库已有评分点 | 人工复核；可扩展 keypoint_coverage 辅助项 |

answer_key_points 不直接替换官方 Coverage 的参考事实抽取；如果后续选择按标注要点评分，应作为独立版本，而不是无说明改变 Coverage 定义。evidence_relations 不等同于图三元组真值，也不是这六项指标的必需输入。

每条系统结果还应带入 retrieved_candidates、chunk_id、doc_id、源文件、段落顺序、检索得分、延迟、模型版本和配置。裁判原始响应与解析结果单独存储。

## 模块与配置建议

建议后续新增 evaluation/，含 runner、judge、metrics、aggregation 与独立的评测配置；业务建库和问答增加独立入口，不继续把正式业务逻辑塞进样例运行器。

裁判使用独立配置：judge_model、judge_base_url、judge_api_key_env、temperature、并发、超时、重试和缓存。优先使用独立于生成模型的裁判；若首轮复用 Qwen，应记录为同模型裁判，并人工复核全部 15 题。模型名称、接口地址相同不意味着应共用推理与裁判提示词或缓存键。

评分语义 Embedding 使用项目内 BGE-M3；跨实验保持模型 revision、归一化及长度处理一致。BGE-M3 替换上游示例的英文 BGE 模型属于适配配置，应记录。

缓存键至少覆盖数据哈希、答案哈希、上下文哈希、指标版本、提示词版本、裁判模型与参数。并发首轮设为 1；不改变已冻结的标准答案。按原逻辑逐题计算六项指标，未计重试和缓存时约需 10 次裁判调用，15 题约 150 次，另有语义向量编码；这是调用量估计，不是已发生的用量。

## 报告与验收

逐题记录六项指标、判定理由及失败状态，汇总分别输出：整体 15 题、事实题 10 题、跨文档推理题 5 题。主表采用逐题宏平均；每项同时显示 expected_n、valid_n、failed_n 和 empty_answer_n。补充证据条目级召回时独立命名，不与逐题宏平均混合。

不把六项指标再平均为一个总分。Correctness=0.8 表示连续评分，不表示“答对 80% 的问题”。15 题仅支持小样本诊断，重点保留逐题结果和人工复核记录。

建议验收顺序：

1. 固定指标口径及上游 commit，补充中文问答模板、真实上下文记录和结果导出。
2. 用人工构造的固定输出验证：相同中文答案、漏掉报告时限、颠倒否定条件、混入错误规则、空答案、无检索、裁判异常和证据分类遗漏。此类夹具与真实实验结果隔离。
3. 用少量真实问题检查接口、费用及裁判输出，再运行全部 15 题并逐题复核。
4. 完成评分模块及配置的迁移验证后，再更新 Docker 依赖与入口并统一打包。

## 核对来源

- [官方评测说明](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Evaluation/README.md)
- [题型与指标映射](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Evaluation/generation_eval.py)
- [Context Relevancy 默认实现](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Evaluation/metrics/context_relevance.py)
- [Evidence Recall](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Evaluation/metrics/evidence_recall.py)
- [Answer Correctness](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Evaluation/metrics/answer_accuracy.py)
- [Coverage](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Evaluation/metrics/coverage.py)
- [Faithfulness](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Evaluation/metrics/faithfulness.py)
- [ROUGE-L 调用](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Evaluation/metrics/rouge.py)
- [ROUGE 默认 tokenizer](https://github.com/google-research/google-research/blob/master/rouge/tokenize.py)
- [上游依赖](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/requirements.txt)
