# 1700 篇语料三元组恢复记录

本轮语料范围按用户要求固定为 1700 篇、5337 块，不补入额外 11 篇。

原建图在三元组阶段结束时因 5 块回复不含 triples JSON 而退出。原输出目录、原数据库和原始失败回复均保留。

## 本次恢复

- 先使用 SQLite backup 备份并校验原有 10676 条缓存，再复制到新的输出目录。
- 正常抽取请求及其缓存键保持不变；格式失败时最多追加两次严格 JSON 请求，每次使用独立缓存键。
- 5 块均在第一次严格重试后恢复，新增加 5 条缓存；不将解析失败替换为空三元组。
- 全部 5337 块的 NER 和三元组缓存验证通过，验证期间禁止新模型请求，实际新请求为 0。
- 解析得到 79808 条跨块去重前的三元组记录。
- 新进程使用 OpenIE 并发 4、文档 batch 1、实体 batch 64、事实 batch 32。

## 路径

- 原输出：`/home/ailab/data/HippoRAG/hipporag_local/outputs/server_native_full_c4_20261030-105942`。
- 新输出：`/home/ailab/data/HippoRAG/hipporag_local/outputs/server_1700_recovered_20261030-163200`。
- 备份：`/home/ailab/data/HippoRAG/hipporag_local/outputs/recovery_backups/triples_20261030-163200`。
- 验证报告：`/home/ailab/data/HippoRAG/hipporag_local/outputs/server_1700_recovered_20261030-163200/recovery_validation.json`。
- 新建图 PID：`3104606`。

## 验证范围

7 项三元组恢复测试和 4 项批大小接入测试通过。恢复仅修改本项目源码和新输出目录，未终止任何其他任务。最终构图成功仍须同时出现 Business corpus indexed. 和 business_corpus_manifest.json 中的 ready 状态。

## 最终验证

构图完成，日志包含 Business corpus indexed.，索引状态为 ready。完整性检查通过，原 10676 条缓存内容全部保留，仅新增 5 条严格重试回复。

- 图节点：54210。
- 图边：235028。
- vdb_chunk：5337 条向量。
- vdb_fact：73586 条向量。
- vdb_entity：48873 条向量。
- 完成报告：/home/ailab/data/HippoRAG/hipporag_local/outputs/server_1700_recovered_20261030-163200/recovery_completed.json。

本轮未启动问答或评测。
