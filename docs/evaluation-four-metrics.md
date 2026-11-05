# 四指标与中文业务问答

当前有效设置：检索评测为 Context Relevancy、Evidence Recall；生成评测为 ROUGE-L、Answer Correctness。不计算 Coverage 或 Faithfulness。先完成业务评测验证，再统一更新容器入口并打包；本轮未构建镜像。

## 两项检索指标

两项均不依赖生成质量指标，可独立计算：

- Context Relevancy 读取问题及实际入模上下文，裁判按 0/1/2 评两次，归一化后平均。
- Evidence Recall 读取实际入模上下文及标准证据，逐条判定是否得到支持，以输入证据条数为固定分母。

数据已有 15 道题和 22 处标准证据。标准答案与标准证据仅在评分阶段合并；检索和回答生成使用只含 id/question 的 questions.json。

生成请求失败时保留已成功检索的上下文，仍可计算检索指标。检索本身无结果则两项为 0。裁判失败记为 null/error，报告同时显示 expected_n、valid_n、failed_n，不与系统失败混淆。

Context Relevancy 是整体上下文信息充分性的粗粒度判断，不解释为相关块占比；Evidence Recall 是证据内容覆盖，不是仅命中文档 ID。

## 分词器选择

评估了以下设计选项，选择依据是中文保留、数字区分、可复现性及现有环境适配，未用正式题目分数调参：

| 方案 | 适配判断 |
| --- | --- |
| ROUGE 默认英文 tokenizer | 会过滤汉字，不适用 |
| Jieba 词级分词 | 支持中文，但领域词边界及用户词典会影响分数，还需引入并冻结词典；本轮不采用 |
| 中文按字、英文及数字按串 | 不依赖分词词典，法律术语不会因词边界变化丢失；本轮采用 |
| BGE-M3 自带 tokenizer | 保留用于 Embedding 和 token 预算；不作为 ROUGE 的分词标准 |

采用 evaluation/tokenization.py 的 `ChineseRougeTokenizer`，版本 `zh-char-alnum-v1`。规则为 Unicode NFKC 规范化、英文 casefold、保留每个汉字、保留连续英文及数字串、忽略其他标点与空白。例如“资本１０％，BGE-M3，１５日”得到“资、本、10%、bge-m3、15、日”。不移除停用词和否定词，不做词干化。

ROUGE-L 使用标准最长公共子序列 F1 公式：`2 × LCS / (预测 token 数 + 标准 token 数)`。相同中文答案为 1，空答案为 0；5 与 15 为不同 token。它衡量文本重合，否定和数字差异仍主要由 Correctness 判别。

这只改变 ROUGE 评分，不替换预训练 BGE-M3 自带的 SentencePiece tokenizer。后者同时用于文档切块及向量编码。

## Correctness

保留 GraphRAG-Bench 的组合公式：`0.75 × 事实陈述 F1 + 0.25 × ((余弦相似度 + 1) / 2)`。

中文提示词提取生成答案与标准答案的原子事实，逐项判定支持、遗漏及错误，计算 TP/FP/FN；输出分类须覆盖所有输入 ID 且无重复。保存两个评分分量和原始裁判结果。

语义向量复用项目内 BGE-M3，但评分长度独立设为 2048、batch size 1。15 份标准答案中有 4 份超过检索编码的 512-token 上限，最长为 890，因此不能直接复用检索长度。超长评分输入报错，不静默截断；检索索引配置不变。

裁判配置独立保存在 evaluation/config.json，可通过 HIPPO_JUDGE_MODEL、HIPPO_JUDGE_BASE_URL、HIPPO_JUDGE_API_KEY_ENV 覆盖。当前首轮使用同一 Qwen 模型，报告标记 same_model_judge=true；正式报告应结合逐题人工复核，后续可换独立裁判而无需重跑检索。

## 业务问答替换

business_qa.py 新增独立的中文业务问答模块，调用 HippoRAG.retrieve() 后使用自己的回答模板，不修改上游代码或样例入口。

- 生成完整的“结论、法规依据、适用分析”，标记资料编号。
- 只基于检索内容回答，保留主体、适用范围、版本、例外、否定、金额比例及期限；资料不足时说明不足。
- 不沿用要求简短答案的 MuSiQue 提示词；不把额外推理文本拼入最终答案。
- 分别保存检索候选、实际入模段落、顺序、元数据、完整请求、原始响应、最终答案、状态和用量。
- 默认候选 top_k=20，入模最多 5 段，BGE tokenizer 上下文预算为 3072；该预算不是托管 Qwen tokenizer 的精确计数。

business_config.json 配置原始语料、预处理输出、索引、问题及预测结果路径。推理入口拒绝包含标准答案等字段的问题输入，并保留失败题记录。

## 运行顺序

在项目根目录、corecoder 环境下运行：

```powershell
python business.py prepare
python business.py index
python business.py answer
python 'data/banking/银行监管测试集/tools/export_eval_input.py' --predictions outputs/banking/predictions.json --output outputs/banking/eval_input.json
python -m evaluation.run --input outputs/banking/eval_input.json --output-dir outputs/banking/evaluation/run_01
```

prepare 只做本地解析与切块，不调用 LLM。index 会对语料执行真实 OpenIE、向量编码和建图，应在实际运行前确定资源与索引目录。answer 从完成的索引回答全部 15 题；未完成的索引、语料指纹不匹配或已有预测文件会被明确拒绝。

正式评分要求 15 个题号、问题、题型、标准答案和证据与冻结题库完全一致。`--allow-partial` 只用于隔离的 smoke/partial 检查，输出会标记，不能作为正式 15 题报告。

## 本次验证及剩余工作

- 16 项针对性测试通过，包括中文一致性、已知 LCS、不同数字、空输入、完整证据分类、公式、统计分母、生成异常时保留检索上下文、切块覆盖及长度检查。
- 新业务问答模板通过一条真实 Qwen 调用的中文合成跨文档样例；实际入模 3 段，均记录在 generation_messages 中。
- 四项评分真实 smoke 检查通过，报告位于 outputs/business_smoke/evaluation_four_metrics_v1/scores.json；该样例仅验证链路，不能用于估计正式业务效果。
- 全部 1711 份原始文件已预处理为 5387 块，最长 508 个 BGE tokens；仅扫描原始语料目录，没有把题库或标准证据作为检索库。预处理文件位于 outputs/banking/corpus.json。
- 尚未对这 5387 块执行正式建图，也未运行正式 15 题的问答和评分。
- Docker 配置已补入业务和评测入口；目前只通过本机入口和配置检查，镜像构建及容器内运行尚未验证，服务器方案待环境信息确认。

## 参考实现

- [GraphRAG-Bench 检索评测](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Evaluation/retrieval_eval.py)
- [GraphRAG-Bench Correctness](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark/blob/main/Evaluation/metrics/answer_accuracy.py)
- [ROUGE scorer 的 tokenizer 接口](https://github.com/google-research/google-research/blob/master/rouge/rouge_scorer.py)
- [Jieba 官方说明](https://github.com/fxsjy/jieba)

本项目保持上述指标思想和公式，采用中文提示词、自定义中文 ROUGE、严格分类完整性校验及显式错误处理；不标为上游未改动的逐字复现。源码哈希、评分输入哈希、裁判配置及模型 revision 随报告保存。
