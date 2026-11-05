# 失败指标补评

原正式评测已处理全部 15 题，三题的 answer_correctness 失败，其余 57 项评分成功。失败题为 Compliance-R001、Compliance-R002、Compliance-R003。

日志显示复杂题的分类回复达到 3072 输出 token 上限，JSON 未闭合；R003 另有一次 90 秒超时。补评以原公式、原提示词、原生成答案及标准答案执行，仅在补评进程中使用 8192 输出 token、240 秒超时、最多一次重试。未修改 evaluation/config.json 中的全局设置。

通过任务名称和完整输入载荷精确匹配原有成功裁判记录，重新运行原校验器后复用事实拆分；原有失败回复不复用。正常的其他指标不重新评分。输出采用新的 formal_15_repair 时间目录，原 scores.json、输入及裁判日志的 SHA-256 在补评结束时核验不变。修复结果在 repair 字段中保留来源、参数、复用数、新调用数及实现哈希，汇总重新计算有效题数。

执行入口为 evaluation/repair.py。它只接受已结束评测中的失败正确性指标，若发现原协调进程仍在运行则拒绝启动。补评前有四项测试，验证输入精确匹配、输入变化不复用、旧结果再次校验、失败尝试不复用。

当前评测目录由输出根目录 testset_status.json 的 evaluation_output 指向。评分看板自动读取这个目录，原评测位置保存在 previous_evaluations 中。

## 补评完成

三题均成功，仅新增 3 次裁判调用、复用 6 份事实拆分。原 57 项评分及原始文件哈希核验不变。全部 60 项指标有效，看板状态为 complete。

最终结果：/home/ailab/data/HippoRAG/hipporag_local/outputs/server_1700_recovered_20261030-163200/banking/evaluation/formal_15_repair_20261030-184926/scores.json。
