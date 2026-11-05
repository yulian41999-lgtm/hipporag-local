# 银行监管问答测试集

版本：1.2；核验日期：2026-10-29。包含10道事实型和5道跨文档推理型，覆盖12份文档，每份最多用于3道题。正式题库的内容和核验哈希保持不变。

## 保留文件

- [questions.json](questions.json)：实验推理输入，只含 `id` 和 `question`，共15题。
- [banking_qa_15.json](banking_qa_15.json)：正式标准题库，包含完整标准答案、证据、评分点和推理标注，供评测使用。
- [source_manifest.json](source_manifest.json)：12份来源的原始文件路径、版本日期、文件哈希及对应题号。
- [evidence_corpus](evidence_corpus)：12份证据来源的清洗文本，供证据定位和复核；这是来源子集，不是全量检索库。
- [content_audit.json](content_audit.json)：逐题内容核验、独立HTML解析核对、适用范围、跨文档必要性及外部核查记录，包含已核验题库的SHA-256。
- [validation_report.json](validation_report.json)：最近一次本地结构、来源、输入及适配检查结果。
- [tools/export_eval_input.py](tools/export_eval_input.py)：合并真实系统输出与标准题库，导出评测输入。
- [tools/validate_dataset.py](tools/validate_dataset.py)：实验前一致性检查。
- [tools/inspect_corpus.py](tools/inspect_corpus.py)：校验脚本使用的原始HTML解析模块。

原始语料与本测试集统一存放在 HippoRAG 项目的 `data/banking/` 数据根目录中：`D01_银行业务与审慎监管` 与 `银行监管测试集` 为同级目录。保留此相对目录结构，以支持来源校验；来源清单、正式题库及其核验哈希保持不变。生成脚本、风格分析、重复问答视图、旧版数据及缓存已从数据集目录移除。

## 实验流程

1. 在项目的 `data/banking/` 数据根目录运行一致性校验：

   ```powershell
   python -B "银行监管测试集/tools/validate_dataset.py"
   ```

2. 使用同一版本的原始全量语料或统一清洗后的正文构建各被测系统索引。记录语料范围、清洗方法、切块参数和模型版本；若只使用12份证据子集，必须单独注明该实验设置。
3. 用 `questions.json` 逐题执行检索和问答，保存真实结果为 `predictions.json`。每条记录格式如下；实际文件需覆盖全部15个题号：

   ```json
   {
     "id": "Compliance-F001",
     "generated_answer": "系统实际生成的答案",
     "context": ["系统实际检索到的文本"]
   }
   ```

   `predictions.json` 顶层为记录数组。无检索结果使用 `[]`，空答案使用 `""`，保留失败记录。
4. 合并并导出 GraphRAG-Bench 评测输入：

   ```powershell
   python -B "银行监管测试集/tools/export_eval_input.py" --predictions "predictions.json" --output "银行监管测试集/results/eval_input.json"
   ```

5. 将生成的 `eval_input.json` 传给框架评测脚本的 `--data_file`，其他模型、API和运行环境参数按所使用的框架版本配置。实验结果可保存到 `results/`，该目录在有实际结果时创建。

题库、README、核验记录及结果文件均包含答案或实验信息，不应进入检索索引。不得把标准证据当作实际检索结果，也不得把标准答案当作系统输出。

## 数据与字段

文件使用UTF-8编码。`question_type` 取值为 `Fact Retrieval` 或 `Complex Reasoning`，两类题分别统计结果。

- `answer` 是“结论—法规依据及原文—适用分析—总结”样式的完整标准答案；导出时映射为 `ground_truth`。
- `answer_summary` 仅供快速查看，不替代完整标准答案。
- `evidence` 是标准原文证据字符串；`evidence_sources` 保存来源、条款、摘录、哈希及清洗文本行号。行号从1开始，包含两端。
- `evidence_relations` 是事实或规则关联描述，不是已验证的知识图谱三元组。
- `answer_key_points` 是评分要点；`reasoning_steps` 和 `cross_document_dependencies` 分别记录推理步骤及各文档的必要贡献。
- `applicability` 记录地域、主体及本地版本。`question_style` 和 `style_reference_examples` 记录咨询类型及原30条样例的风格映射。
- `source` 是语料集合标识，具体路径在 `evidence_sources` 和来源清单中。

导出文件使用 `ground_truth`、`generated_answer`、`context` 等字段，并把 `evidence_sources[].quote` 转为 `evidence` 字符串列表。`context` 始终来自真实系统检索。

中文ROUGE-L应在评测端统一中文分词或字粒度tokenizer并记录口径。官方默认ROUGE实现未做中文分词适配，仅在汉字间插空格不能保证默认tokenizer正确保留中文。

## 核验范围

已完成本地原文、元数据、文件哈希及22处证据摘录核对，并保存逐题模型语义复核。校验脚本还检查15个推理输入与标准题库一致，以及正式题库哈希与内容核验记录一致；它不会自动重新进行语义或外部法规效力核查。

本地来源均标示“现行有效”，但尚未穷尽确认全部文件截至核验日的外部现行法律效力。1.2版已替换存在版本或主体适用风险的F009、R004、R005，具体依据保留在 `content_audit.json`。本集适用于指定本地语料版本的RAG实验，不应标记为“全部外部现行效力已核实”。
