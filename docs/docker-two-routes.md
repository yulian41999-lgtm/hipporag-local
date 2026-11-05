# 两条 Docker 部署路线

两条路线使用同一 Dockerfile 和同一个 linux/amd64 GPU 镜像，只选择一个构建位置。本地 corecoder 可继续使用 CPU PyTorch；镜像内单独安装 CUDA 12.1 版 PyTorch 2.5.1。

Qwen 始终使用远程服务，RTX 3090 主要加速 BGE-M3。模型和数据采用项目目录只读挂载，索引及结果写入宿主机 outputs，不把密钥放入镜像。

## 两条路线共同的前提

- 固定源码、业务问答、四指标评测、BGE-M3 及 15 道题均保留在项目中。
- 镜像目标为 linux/amd64，服务器为 Ubuntu x86_64。
- 服务器容器 GPU 支持仍需实际验证，不能仅凭 Toolkit 已安装作结论。
- 服务器已有 5 个运行中的服务。不要为部署测试随意重启 Docker、清理镜像或改写 daemon 配置。
- 服务器项目目标目录为 `/home/ailab/data/HippoRAG/hipporag_local`，与现有目录内其他内容隔离。
- 服务器另行设置 `.env`；不需要安装 corecoder 或重新安装宿主机 Python 依赖。

## 路线一：本地构建，服务器导入

### 第一步：启用本地构建环境

打开 Windows Docker Desktop，使用 Linux containers，等引擎就绪后在 PowerShell 检查 `docker version`。必须同时有 Client 和 Server 信息。当前已知本机 Linux 引擎未运行，尚未满足此条件。

本地需要能下载基础镜像、Debian 系统依赖和 Python/PyTorch 包；不需要把 corecoder 改为 CUDA 版，也不要求本地有与服务器相同的 GPU。

### 第二步：在本地构建 GPU 镜像

```powershell
Set-Location 'D:\Workspace\jsd\project_working\hipporag_local'
docker build --platform linux/amd64 --target runtime --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 -t hipporag-local:gpu .
```

必须指定 runtime 和 cu121 参数；当前 Dockerfile 的 PyTorch 下载源默认是 CPU。runtime 不把业务数据和模型权重复制进镜像。

### 第三步：检查并导出

```powershell
docker image inspect hipporag-local:gpu --format '{{.Os}}/{{.Architecture}}'
docker run --rm --entrypoint python hipporag-local:gpu -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.version.cuda is not None"
New-Item -ItemType Directory -Force -Path dist | Out-Null
docker save --output dist/hipporag-gpu-linux-amd64.tar hipporag-local:gpu
Get-FileHash -Algorithm SHA256 -LiteralPath dist/hipporag-gpu-linux-amd64.tar
```

预期架构为 linux/amd64，torch.version.cuda 为 12.1。此检查不要求本地 GPU，不能替代服务器实际 CUDA 运算验证。

### 第四步：上传文件

上传镜像 tar、models/、data/、compose.server.yaml 和 .env.example 至服务器项目目录。为保留完整项目，也可上传整个项目，但排除 .env、cache/、旧 outputs/、Python 缓存及 vendor 下的 .git。确保 models/bge-m3 存在，单传镜像 tar 不含该模型。

镜像 tar 的 SHA-256 用于传输后核对，不在聊天或共享文件中包含真实密钥。

### 第五步：服务器载入

```bash
cd /home/ailab/data/HippoRAG/hipporag_local
sha256sum hipporag-gpu-linux-amd64.tar
docker load --input hipporag-gpu-linux-amd64.tar
```

核对哈希后继续下方共同运行流程。该路线服务器不再次执行 docker build，也不重新 pip install。

## 路线二：上传项目，服务器构建

### 第一步：上传完整项目

把项目传到服务器目标目录，包含 Dockerfile、requirements.txt、入口脚本、evaluation/、vendor/HippoRAG/、models/、data/ 和 Compose 文件。排除真实 .env、Windows wheel 缓存、旧 outputs/、Python 缓存及 Git 元数据。

模型和源码使用项目相对路径，保留目录结构。现有 python:3.12-slim 只可用作基础 GPU 注入测试，不能直接替换项目已验证的 Python 3.10 环境。

### 第二步：测试服务器构建所需网络与磁盘

```bash
docker pull python:3.10-slim-bookworm
curl -I --connect-timeout 5 --max-time 15 https://download.pytorch.org/whl/cu121/torch/
df -h '/home/cuiyuan/桌面/CEDD/20260706_unlimited_ocr_exp/API/docker_data'
```

服务器配置了镜像源，但尚未验证该基础镜像能实际下载。PyPI 索引 200 也不代表 wheel、PyTorch 及 Debian 软件源全部可达。Docker 缓存位于上述目录，不应把项目数据盘的 425 GB 可用空间视为 Docker 缓存盘的空间。

### 第三步：构建同一个 GPU 镜像

```bash
cd /home/ailab/data/HippoRAG/hipporag_local
docker build --platform linux/amd64 --target runtime --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 -t hipporag-local:gpu .
```

如果管理员指定其他基础镜像来源，使用 BASE_IMAGE 构建参数；不要未经确认改 Docker 全局镜像源。

### 第四步：检查后直接运行

```bash
docker image inspect hipporag-local:gpu --format '{{.Os}}/{{.Architecture}}'
```

服务器上已经有镜像，不需要 docker load。可选择 docker save 留存备份，然后进入共同运行流程。

## 两条路线共同的验证和正式执行

### 运行配置

在服务器项目目录建立本机配置文件。已有 .env 时编辑现有文件，不覆盖：

```bash
test -f .env || cp .env.example .env
nano .env
```

填入实际 HIPPO_LLM_API_KEY，核对模型和 base URL。当前是 qwen3.8-27B-raw、http://10.253.170.81:8006/v1；401 只证明 HTTP 可达，仍需通过下一步的 probe 验证鉴权与返回内容。

### GPU、Embedding 与模型接口验证

```bash
docker run --rm --pull=never --gpus all --entrypoint python hipporag-local:gpu -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(128,128,device='cuda'); print((x@x).shape)"
docker compose -f compose.server.yaml run --rm hipporag embedding
docker compose -f compose.server.yaml run --rm hipporag probe
docker compose -f compose.server.yaml run --rm hipporag sample --dataset chinese_demo
```

预期能识别 RTX 3090、完成实际 CUDA 矩阵计算，Embedding 报告显示 cuda、1024 维向量，probe 返回有效模型内容和 token 用量，小型跨文档样例完成索引及问答。如果 GPU 注入失败，先保留错误并确认现有服务与维护安排，再决定 runtime 配置；本方案不要求立即重启 daemon。

### 全量准备与建图

```bash
docker compose -f compose.server.yaml run --rm hipporag prepare
docker compose -f compose.server.yaml run --rm hipporag index
```

prepare 重新在服务器检查原始文件并切块；当前本地预处理结果是 1711 份文件、5387 块，但不复制历史 CPU 索引作为 GPU 正式结果。index 会执行真实 OpenIE 和向量编码，产生实际时间与服务调用量。

### 15 题问答与四指标评分

```bash
docker compose -f compose.server.yaml run --rm hipporag answer
docker compose -f compose.server.yaml run --rm hipporag export --predictions /app/outputs/server_gpu/banking/predictions.json --output /app/outputs/server_gpu/banking/eval_input.json
docker compose -f compose.server.yaml run --rm hipporag evaluate --input /app/outputs/server_gpu/banking/eval_input.json --output-dir /app/outputs/server_gpu/banking/evaluation/run_01
```

检索为 Context Relevancy 和 Evidence Recall；生成为 ROUGE-L 和 Correctness。运行失败的题目或评分保留状态，报告需核对 15 道题和各指标有效数量。

索引、预测结果和评测报告均保存在服务器项目的 outputs/server_gpu/banking/ 中。重复实验先保存上一轮结果，使用新的输出目录，避免覆盖。

## 如何选择

| 比较项 | 本地构建再导入 | 服务器构建 |
| --- | --- | --- |
| 构建依赖下载 | 发生在本地 | 发生在服务器 |
| 传输内容 | 镜像 tar + 模型、数据及运行配置 | 完整源码项目 + 模型、数据 |
| 服务器是否 build | 否，只 load | 是 |
| GPU 验证 | 在服务器执行 | 在服务器执行 |
| 当前待确认项 | 本地 Docker Linux 引擎和构建网络 | 服务器镜像源、依赖下载、Docker 缓存盘空间 |

根据目前信息，优先验证服务器构建路线；若服务器下载链路不通，则改为本地或其他构建机生成镜像后离线导入。两条路线均尚未实际构建成功或完成服务器容器验证。

## 官方命令参考

- [docker save](https://docs.docker.com/reference/cli/docker/image/save/)
- [docker load](https://docs.docker.com/reference/cli/docker/image/load/)
- [Compose GPU 配置](https://docs.docker.com/compose/how-tos/gpu-support/)
