# HippoRAG2 银行监管问答

本仓库为服务器 `/home/ailab/data/HippoRAG/hipporag_local` 的源码、语料与测评产物快照（2026-11-05），私有保存。

## 上传内容

保留项目源码、依赖与配置模板、业务语料／题库、报告、当前正式测评的实际上下文与API用量；HippoRAG的vendor源码保留且移除其嵌套Git元数据。详见 `UPLOAD_MANIFEST.json`。

模型权重、Python虚拟环境、密钥、LLM缓存、原始大型图／向量索引、运行日志及上游基准数据未上传。模型与现有索引继续保留在服务器；本仓库不是完整磁盘备份，克隆后不会自动恢复大型索引。

## 保持原项目相对路径

两个仓库在同一父目录克隆，并分别命名为 `hipporag_local` 和 `Microsoft_GraphRAG`，以保留项目间引用。配置中原有服务器绝对路径按实际部署目录调整。API密钥在环境变量或本机 `.env` 设置，不提交到Git。

## 原项目文档

以下保留原文，其中部署阶段与旧结果以实际运行清单及最新正式测评产物为准。

# HippoRAG 本地部署

使用 `corecoder` 环境，在 Windows 本机运行 HippoRAG 2；大规模实验后续迁移至服务器。

## 组件

- Python：`D:\Anaconda3\envs\corecoder\python.exe`，Python 3.10.22。
- 官方源码：项目内的 `vendor/HippoRAG`，固定 commit 为 `d5c8329422e0a0b834a15874545cb6a74b4f9b26`，同时记录在 `upstream_revision.txt`。
- Embedding：项目内的 `models/bge-m3`，从已有模型复制，原目录保留。
- 模型标识：`BAAI/bge-m3`，本地下载清单的 revision 为 `5617a9f61b028005a4858fdac845db406aefb181`。
- LLM：`qwen3.8-27B-raw`，base URL 为 `http://10.253.170.81:8006/v1`。
- 密钥：当前文件夹的 `.env`，由 `.gitignore` 忽略，代码及报告不包含密钥。

本地只计算 BGE-M3 的 1024 维稠密向量；实体关系抽取、事实筛选及答案生成调用用户提供的 LLM 服务。未接入 `bge-reranker-v2-m3`。这是替换了模型的功能复现，评测应与论文原配置分开报告。

## 业务数据位置

原始银行监管资料和正式 15 道问答已移动到项目内的 `data/banking/`，保留二者原有的相对目录层级：

```text
hipporag_local/
  data/
    banking/
      D01_银行业务与审慎监管/
      银行监管测试集/
        questions.json
        banking_qa_15.json
        source_manifest.json
        evidence_corpus/
        tools/
    banking_migration_manifest.json
```

后续原始法规文件放入 `data/banking/D01_银行业务与审慎监管/`。正式建库仅扫描该原始语料目录；测试集、标准答案、证据标注、校验报告及实验输出不进入检索索引。`evidence_corpus/` 仅为 12 份来源子集，不是全量检索库。业务入口现为 `business.py`，`run.py` 保留官方及合成样例用途。

移动了 1732 个文件，并逐一核对 SHA-256；移动清单保存在 `data/banking_migration_manifest.json`。正式题库与证据内容未改写，15 道题的一致性校验使用以下命令：

```powershell
conda activate corecoder
python -B 'D:\Workspace\jsd\project_working\hipporag_local\data\banking\银行监管测试集\tools\validate_dataset.py'
```

将整个项目复制到服务器时一并保留 `data/banking/`、`models/bge-m3/` 和 `vendor/HippoRAG/`。模型及数据引用按项目所在位置解析，不依赖本机盘符。BGE-M3 共复制 11 个文件、2,293,317,620 字节，逐个核对 SHA-256 均一致，记录在 `models/bge-m3.copy-verification.json`。权重未纳入 Git 跟踪，复制目录时须包含 `models/`。

## 样例运行

在 PowerShell 中运行：

```powershell
conda activate corecoder
Set-Location 'D:\Workspace\jsd\project_working\hipporag_local'
python probe_llm.py
python run.py embedding
python run.py sample
python run.py reload
python run.py sample --dataset chinese_demo
python run.py reload --dataset chinese_demo
```

也可以使用启动脚本，优先使用已激活的 `corecoder` 或 `HIPPO_PYTHON` 指定的解释器：

```powershell
& 'D:\Workspace\jsd\project_working\hipporag_local\run.ps1' -Mode sample
```

`embedding` 验证维度、数值有效性、归一化和重复编码一致性；`sample` 使用官方三段语料及一道问题，验证真实 LLM 抽取、构图、检索和问答；`reload` 在新进程中加载已有图并再次检索问答，不调用索引构建。

`chinese_demo` 是另设的合成测试：虚构银行、城市与省份，跨两段语料回答行政归属问题。它只验证中文跨文档流程，不用于估算真实业务效果。官方样例与合成测试分开保存，不修改官方题目或标准答案。

## 配置

编辑 `config.json` 修改模型目录、服务地址、batch size、输入长度和输出目录。Embedding 以 `local_files_only=True` 加载，启用 Hugging Face 离线环境变量，复用已有权重。模型目录本身只读。

默认 `embedding_path` 为项目相对路径 `models/bge-m3`。服务器可用 `HIPPO_EMBEDDING_PATH`、`HIPPO_LLM_BASE_URL`、`HIPPO_LLM_MODEL`、`HIPPO_OUTPUT_DIR`、`HIPPO_DEVICE`、`HIPPO_BATCH_SIZE` 和 `HIPPO_MAX_SEQ_LENGTH` 覆盖配置；环境变量优先。密钥保留在运行环境或本地 `.env`，模板为 `.env.example`。

首轮使用 batch size 1、最大输入长度 512 tokens。输入超过该长度会截断；业务文档接入前必须按条款或段落切分，并检查实际 token 长度，避免遗漏证据。官方样例中的短段落不需要额外切分。

设备为 `auto`：CUDA 可用时使用 GPU FP16，否则使用 CPU FP32。两者的索引身份不同，切换精度时应设置新的 `output_dir`，重新构建索引。

本次已安装 PyTorch 2.5.1+cpu，当前实际使用 CPU FP32。GPU 推理需另行安装 CUDA 构建后再验证。与官方 Transformers 4.45.2 对齐时，将现有环境中的 Hugging Face Hub、Datasets 等依赖调整为兼容版本；`requirements-local.txt` 记录直接依赖约束，`corecoder-before.json` 保存变更前的版本清单。

```powershell
python run.py embedding --device cpu
```

BGE-M3 使用独立适配器注入 HippoRAG，保持稳定模型标识，避免将含盘符的 Windows 绝对模型路径用于索引文件夹名。适配器按 BGE-M3 的要求不添加查询指令，输出归一化的 float32 数组。

## 结果

- LLM 连通性报告：`outputs/llm_probe.json`。
- Embedding 报告：`outputs/smoke/embedding_report.json`。
- 官方样例报告：`outputs/smoke/sample_report.json`。
- 索引重新加载报告：`outputs/smoke/reload_report.json`。
- 持久化图、向量及抽取状态：`outputs/smoke/index/`。
- 中文合成测试及其重新加载报告：`outputs/smoke_chinese_demo/`。

启动脚本打印 `PASS` 表示相应流程检查完成；问答效果以报告中的 EM/F1 为准。单道样例不能代表数据集表现。

本次官方样例实际得到 EM=0、F1=0，答案为 `Karl Deisseroth`，标准答案为 `Thomas C. Sudhof`。样例的三段语料未明确给出问题中的 Alzheimer's 关联；保留原始结果，作为问答失败案例，不据此宣称论文指标复现成功。

中文合成跨文档测试答出“临江省”，EM=1、F1=1，检索 Recall@2=1；该测试仅有一道题。构图得到 9 个节点、11 条边。Embedding 检查得到 1024 维向量，相关句相似度为 0.855，无关句为 0.443。官方 SDK 兼容性测试 18 项通过，`pip check` 无依赖冲突。

安装后完整版本清单在 `corecoder-installed.json`，环境变更在 `environment-changes.json`。下载中遇到大文件传输停滞，`download_wheels.py` 支持校验 PyPI SHA-256 并分段下载大型安装包；已缓存的 Windows 安装包位于 `cache/wheels/`，不能用于 Linux 服务器。

## 迁移到服务器

当前已选择服务器构建，按 [服务器构建操作手册](docs/server-build-runbook.md) 执行。`tools/prepare_delivery.py` 默认仅检查交付集合，指定 `--output` 才生成不含密钥/缓存/旧结果的源码包；它不是 Docker 镜像。`compose.server-build.yaml` 配合 `compose.server.yaml` 在服务器构建 CUDA 版镜像。

两种构建位置的完整步骤见 [两条 Docker 部署路线](docs/docker-two-routes.md)：本地构建后 save/load，或上传项目后在服务器 build；两者共用同一套服务器建图和评测流程。

已根据用户提供的 Ubuntu 20.04 / RTX 3090 环境准备 [服务器部署方案](docs/server-deployment.md) 和离线使用的 `compose.server.yaml`。当前 Docker Hub 直连超时、NVIDIA runtime 未在 Docker 查询中出现，实际镜像构建及 GPU 容器验证尚未完成。

1. 复制整个项目，包含模型、业务数据及 `vendor/HippoRAG` 源码；在项目根目录使用服务器的 `corecoder` 执行 `python -m pip install -r requirements-local.txt`。
2. 核对 `upstream_revision.txt` 和模型复制校验记录；默认使用项目内 BGE-M3，相对路径无需修改。
3. 在服务器本地设置密钥环境变量，或建立被忽略的 `.env`。
4. 重新运行 Embedding、官方样例与重新加载验证，再扩大语料规模。
5. 复用已有索引前核对模型、归一化、输入长度、精度和索引身份；不匹配时使用新输出目录重建。

## 容器化准备与当前边界

四指标模块及中文业务问答已接入，Dockerfile 已包含新的业务、结果导出与评测入口。正式全量建图和 15 题运行尚未执行；当前尚未构建或运行容器，也未生成镜像归档。服务器部署方式等待系统、GPU、Docker 与网络信息确认。

已准备 `Dockerfile`、`compose.yaml`、`compose.gpu.yaml` 和 `.dockerignore`，供后续评测接入后使用；CPU 与 GPU 两组 Compose 配置已通过静态配置校验。构建与容器内运行尚未验证，本机 Docker Linux 引擎当前未运行。

`runtime` 构建阶段通过 Compose 挂载项目内模型及数据，适合迭代；`bundled` 阶段可把模型和数据复制进镜像，供最终交付使用。CPU 默认安装 PyTorch 2.5.1 CPU 构建，GPU 覆盖使用 CUDA 12.1 构建，服务器需具备相应 NVIDIA 容器运行条件。两者使用不同输出目录。

容器使用 `container_cli.py`，支持 `embedding`、`probe`、`prepare`、`index`、`answer`、`export`、`evaluate` 及样例命令。默认只验证 Embedding，不自动触发大规模建图。业务预处理、索引和预测随 HIPPO_OUTPUT_DIR 写入对应 CPU/GPU 输出目录；容器内使用固定 Python 3.10 环境，不依赖宿主机 corecoder 的安装路径。

`.env`、本机 Windows 安装包缓存、历史输出、Python 缓存及 Git 元数据不进入镜像构建上下文；密钥仅在运行时配置，不能随镜像或项目共享。

## 扩展计划

当前生效的四指标和业务模块见 [四指标与中文业务问答](docs/evaluation-four-metrics.md)：检索为 Context Relevancy、Evidence Recall，生成为 ROUGE-L、Correctness。模块已接入并通过隔离样例验证，正式 15 题尚未运行。原 [六指标分析](docs/evaluation-integration.md) 保留为历史方案，其中 Coverage、Faithfulness 不纳入当前评测。

业务评测集以用户确认的 **15 道问答**为准；文件名中的“30”不作为题目数量依据。后续逐题保存检索证据、预测答案与评分，同时报告答对题数及总题数（分母为 15）。业务问答与独立的中文合成测试分开统计。

1. 完成上述四项验证并保存依赖版本。
2. 在固定公开数据上运行 BGE-M3 向量 RAG 基线与 HippoRAG，使用相同语料、模型和上下文预算。
3. 为 15 道银行监管问答补充支持条款、文档版本和页码，再逐题进行中文业务验证。
4. 记录抽取成本、索引耗时、查询延迟、Recall@k 与 EM/F1，确认收益后迁移服务器做大规模实验。
