# OpenIE 请求并发配置

所有操作限于 `/home/ailab/data/HippoRAG` 及其子目录。未经用户明确同意，不得中断或重启现有任务。

`business.py` 支持环境变量 `HIPPO_OPENIE_MAX_WORKERS`；其次读取 `config.json` 的 `openie_max_workers`，缺省为 1。值必须为正整数，由 BaseConfig 校验。该参数控制 NER 和三元组抽取线程数，与 `HIPPO_BATCH_SIZE` 相互独立。

仅后续启动的进程生效。现有进程已创建线程池，修改文件或在另一终端 export 不会改变它。

后续启动时可设置：

```bash
export HIPPO_OPENIE_MAX_WORKERS=4
```

建议先尝试 4，但未验证远程模型服务的并发容量或实际加速比。不得在旧任务仍运行时启动第二个正式建图任务。若用户授权中断后恢复，先核实旧进程退出、快照备份缓存，再使用新的输出目录复用 LLM 缓存；不要直接复用缺图的旧向量目录。
