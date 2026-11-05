# 服务器构建：本地整理及操作手册

当前方案只在服务器构建一次镜像，本地 corecoder 保持现有环境。所有服务器命令由用户执行，尚未实际构建或运行服务器容器。

## 本地整理状态

项目内的源码、BGE-M3、1711 份原始资料及 15 道题已齐备。现有样例验证和四指标测试不代表已完成正式建图和 15 题评测。

本次交付采用 tools/prepare_delivery.py 的白名单，不直接压缩整个工作目录。.dockerignore 只约束 Docker 构建上下文，不会替普通上传工具排除文件。

| 交付内容 | 处理方式 |
| --- | --- |
| Dockerfile、requirements、Compose、配置和入口 | 必须包含 |
| vendor/HippoRAG、evaluation、业务模块 | 包含源码，剔除内部 .git、egg-info 和缓存 |
| models/bge-m3 | 包含已有 11 个模型文件并核对哈希，原外部模型目录保留 |
| data/banking | 包含完整原始资料和正式 15 题，二者目录分开，题库不参与索引 |
| docs、tests、tools | 包含操作说明、测试和校验脚本 |
| .env | 排除，服务器单独设置；仅交付 .env.example |
| cache、outputs、dist、__pycache__ | 排除，服务器重新产生，不混入 CPU 索引、旧预测或 Windows wheels |
| corecoder 环境清单、Windows 启动脚本 | 本地保留，不作为服务器运行依赖 |

交付集合已检查约 2.197 GiB、1895 个文件，后续文档及测试新增会使文件数略增。未生成归档文件。生成命令在本地 PowerShell：

```powershell
Set-Location 'D:\Workspace\jsd\project_working\hipporag_local'
& 'D:\Anaconda3\envs\corecoder\python.exe' tools/prepare_delivery.py
& 'D:\Anaconda3\envs\corecoder\python.exe' tools/prepare_delivery.py --output dist/hipporag-server-source.tar
```

第一条仅检查，第二条才生成源码 tar 与 .sha256 文件，拒绝覆盖旧包。包里包含 delivery_manifest.json，逐文件保存 SHA-256。此 tar 是源码交付包，不能 docker load；后续在服务器执行 docker build。

## 阶段一：服务器前置验证

先运行以下命令，不重启 daemon、不修改已有服务：

```bash
docker run --rm --pull=never --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility python:3.12-slim nvidia-smi
docker pull python:3.10-slim-bookworm
curl -I --connect-timeout 5 --max-time 15 https://download.pytorch.org/whl/cu121/torch/
df -h '/home/cuiyuan/桌面/CEDD/20260706_unlimited_ocr_exp/API/docker_data'
```

GPU 命令使用已存在的 3.12 镜像，仅检查设备注入，不替换本项目 Python 3.10。实际 CUDA 运算仍在项目镜像构建后验证。

如果 GPU 设备注入失败，记录错误；NVIDIA runtime 配置可能涉及 daemon 更新和重启，应先与现有 5 个服务的维护安排协调，不在本流程中自动执行。Docker 数据盘独立核查，不能把项目盘的 425 GB 当作 Docker 缓存盘的剩余空间。

## 阶段二：上传和解包

将本地生成的 tar 和 .sha256 上传到 `/home/ailab/data/HippoRAG`，可使用 SFTP 客户端。不要上传本地真实 .env，也不要覆盖已有项目目录。

在服务器终端：

```bash
cd /home/ailab/data/HippoRAG
sha256sum -c hipporag-server-source.tar.sha256
```

确认显示 OK，且目标 `hipporag_local` 目录尚不存在，再解包。若目录已存在，先选新的发布目录，勿直接覆盖旧文件：

```bash
test ! -e hipporag_local && tar -xf hipporag-server-source.tar
cd hipporag_local
python3 tools/verify_delivery.py
```

verify_delivery.py 仅用标准库，兼容 Ubuntu 20.04 常见的 Python 3.8，不需要服务器 corecoder。它只校验交付文件完整性，不改写题库。

## 阶段三：配置及构建

在项目目录设置密钥，已有文件时保留并编辑，不覆盖。密钥不发回聊天：

```bash
test -f .env || cp .env.example .env
chmod 600 .env
nano .env
mkdir -p outputs cache
```

保持 HIPPO_LLM_MODEL=qwen3.8-27B-raw、HIPPO_LLM_BASE_URL=http://10.253.170.81:8006/v1，填写实际 HIPPO_LLM_API_KEY。需要独立裁判时添加 HIPPO_JUDGE_MODEL、HIPPO_JUDGE_BASE_URL 和 HIPPO_JUDGE_API_KEY_ENV，否则首轮为同模型裁判。

校验 Compose 并构建：

```bash
docker compose -p hipporag-project -f compose.server.yaml -f compose.server-build.yaml config --quiet
set -o pipefail
docker compose -p hipporag-project -f compose.server.yaml -f compose.server-build.yaml build hipporag 2>&1 | tee outputs/build.log
```

compose.server-build.yaml 将 build 添加到原来的运行配置，指定 runtime 阶段、Python 3.10 基础镜像及 CUDA 12.1 的 PyTorch 源。构建完成得到 hipporag-local:gpu，不需要 docker load。

构建会访问基础镜像源、Debian 软件源、PyPI 及 PyTorch wheel 下载源；某一索引 HTTP 200 不代表所有依赖下载都通过。错误保留在 build.log，根据具体失败源处理，不盲目更换 Python 版本或修改全局 Docker 配置。

镜像包含 Python、CUDA PyTorch、源码和业务及四指标模块；模型与数据通过只读挂载读取。宿主机无需复制本地 Conda 环境，也无需另行安装匹配的 CUDA Toolkit。镜像构建成功并不证明 GPU 容器运行成功。

## 阶段四：最小运行验证

以下命令均在项目根目录执行。使用固定的独立项目名 hipporag-project，不连接已有 Milvus/MySQL，也不需要为本项目开放额外端口。

```bash
docker run --rm --pull=never --gpus all --entrypoint python hipporag-local:gpu -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0)); x=torch.randn(128,128,device='cuda'); print((x@x).shape)"
docker compose -p hipporag-project -f compose.server.yaml run --rm hipporag embedding
docker compose -p hipporag-project -f compose.server.yaml run --rm hipporag probe
docker compose -p hipporag-project -f compose.server.yaml run --rm hipporag sample --dataset chinese_demo
```

验收条件：真实 CUDA 计算成功、Embedding 显示 cuda 和 1024 维、Qwen POST 返回有效内容及用量、小型索引与问答跑通。若容器访问内网接口失败，先检查容器路由/代理，不直接改用 host 网络。

## 阶段五：正式建图与 15 题评测

```bash
docker compose -p hipporag-project -f compose.server.yaml run --rm hipporag prepare
set -o pipefail
docker compose -p hipporag-project -f compose.server.yaml run --rm hipporag index 2>&1 | tee outputs/index.log
docker compose -p hipporag-project -f compose.server.yaml run --rm hipporag answer
docker compose -p hipporag-project -f compose.server.yaml run --rm hipporag export --predictions /app/outputs/server_gpu/banking/predictions.json --output /app/outputs/server_gpu/banking/eval_input.json
docker compose -p hipporag-project -f compose.server.yaml run --rm hipporag evaluate --input /app/outputs/server_gpu/banking/eval_input.json --output-dir /app/outputs/server_gpu/banking/evaluation/run_01
```

每一步退出码为 0 且报告通过后再继续下一步，特别是 index 成功才运行 answer。耗时操作在稳定终端或已配置的 tmux 会话运行，避免 SSH 断开导致中断；本项目不会自动创建后台任务。

首轮 batch size=1、OpenIE 并发=1，不因显存空闲就立即增大并发。构图仍有远程 Qwen 调用，GPU 不能消除 OpenIE 网络耗时。正式峰值内存、显存、耗时及调用量由本次运行记录，不能从样例时间线性保证。

## 输出及重复运行

| 内容 | 服务器项目内路径 |
| --- | --- |
| 原始语料清洗和切块结果 | outputs/server_gpu/banking/corpus.json |
| 图、向量与抽取状态 | outputs/server_gpu/banking/index/ |
| 15 题系统答案与实际上下文 | outputs/server_gpu/banking/predictions.json |
| 合并后的评分输入 | outputs/server_gpu/banking/eval_input.json |
| 四指标逐题和汇总结果 | outputs/server_gpu/banking/evaluation/run_01/scores.json |
| 裁判原始输出及用量 | outputs/server_gpu/banking/evaluation/run_01/judge_calls.jsonl |

检查 overall 的 expected_n=15，每项 valid_n/failed_n；同时核对 10 道事实题和 5 道推理题的汇总。ROUGE-L 和 Correctness 是连续指标，不等同于答对题数比例。同模型裁判需要逐题复核。

容器退出后结果仍在宿主机挂载目录中。重新问答或评测前保留旧结果，选择新输出目录；程序会拒绝覆盖已有预测或正式 scores.json。模型精度、输入长度或索引身份变化时，使用新的索引目录重建，不直接混用本地 CPU 索引。

## 当前完成边界

本地已完成代码、模型和数据整理，新增白名单源码交付工具、哈希校验工具及服务器 build 覆盖配置。已执行本地测试和 Compose 静态校验；尚未生成源码归档、上传、服务器构建或完成正式 15 题实验。
