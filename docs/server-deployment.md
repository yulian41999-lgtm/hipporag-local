# 服务器环境核对与部署方案

记录日期：2026-10-29。事实来自用户粘贴的服务器查询结果，尚未远程登录或修改服务器。

## 已确认信息

| 项目 | 查询结果 |
| --- | --- |
| 系统与架构 | Ubuntu 20.04.6，x86_64 |
| CPU | i9-10900K，10 核 20 线程 |
| 内存 | 总计 31 GiB，查询时可用约 18 GiB |
| GPU | RTX 3090，24576 MiB；最新查询已用 39 MiB、空闲 24097 MiB、利用率 0% |
| NVIDIA 驱动 | 570.133.07 |
| Docker | 客户端 20.10.12；服务端 28.0.0；当前用户能查询 daemon |
| Compose | 2.35.1 |
| NVIDIA Container Toolkit | CLI 1.17.4，Docker Runtimes 未列出 nvidia |
| 数据盘空间 | /home/ailab/data，约 425 GB 可用 |
| 当前目录 | /home/ailab/data/HippoRAG |
| Docker Hub | curl 直连超时；daemon 配有 https://x1ofrvkv.mirror.aliyuncs.com/，实际 pull 尚未测试 |
| PyPI | 索引 HTTP 200；尚未验证安装包下载主机和 PyTorch 下载源 |
| Qwen | 最后一次干净的请求为 HTTP 401，证明 HTTP 可达；尚未验证服务器凭据及容器内调用 |

粘贴记录中夹杂的 curl 000 和域名解析错误由命令与中文注释黏连造成，不据此判定 Qwen 不可达。交换分区已使用较多，但仅凭此不能断定当前仍在持续换页。

## 确定的运行方向

采用 linux/amd64 GPU 镜像，PyTorch 2.5.1 CUDA 12.1，GPU 运行本地 BGE-M3；Qwen 仍通过已有服务调用，不在该服务器额外部署 27B 模型。初始 batch size 保持 1，先验证 GPU 可用性再根据空闲显存调整。

驱动 570 满足 CUDA 12.x 驱动版本要求。宿主机无需匹配容器内 Python 安装目录。正式峰值内存、显存及吞吐仍需从实际运行测量。

补充信息确认 daemon 为本机 unix:///var/run/docker.sock，未显示 rootless。已有 5 个容器正在运行，涵盖 MinIO、Milvus、etcd 和 MySQL，不能为测试随意重启 Docker。镜像缓存包含 python:3.12-slim，但不包含本项目使用的 3.10-slim-bookworm；不据此改动已经验证过的 Python 3.10 依赖组合。

镜像实际根目录为 /home/cuiyuan/桌面/CEDD/20260706_unlimited_ocr_exp/API/docker_data；先前的 425 GB 可用空间属于 /home/ailab/data，不能直接推定 Docker 缓存盘也有同样的剩余空间。

目前优先测试服务器上的 GPU 容器和现有镜像源。如果 pull 与依赖下载正常，可在服务器构建，避免依赖当前未启动的本机 Docker Engine；若不通，再采用其他构建机 docker save、服务器 docker load 的离线方案。项目模型和数据以只读挂载方式供容器使用，索引与结果写入 outputs。

compose.server.yaml 不包含 build，且 pull_policy=never，避免离线部署时隐式联网。它只启动 Embedding 检查，不自动开始大量建图。

## 下一步验证（不重启 daemon）

```bash
docker run --rm --pull=never --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility python:3.12-slim nvidia-smi
docker pull python:3.10-slim-bookworm
curl -I --connect-timeout 5 --max-time 15 https://download.pytorch.org/whl/cu121/torch/
df -h /home/cuiyuan/桌面/CEDD/20260706_unlimited_ocr_exp/API/docker_data
```

第一条只启动一个使用现有镜像的临时容器，不下载镜像，不改动 daemon，也不修改已有服务；成功显示 nvidia-smi 才证明基础 GPU 设备注入可用，后续仍需验证容器内 PyTorch CUDA。runtime 列表缺少 nvidia 本身不能替代实际测试。第二条会经现有 daemon 配置尝试下载基础镜像，不改注册表配置、不删除已有镜像。第三条只验证 PyTorch 包源索引连接，不证明大型 wheel 已成功下载。第四条查询 Docker 缓存盘空间。

Dockerfile 支持 BASE_IMAGE 参数，且不再要求额外拉取 Dockerfile frontend。默认 python:3.10-slim-bookworm；如需改用管理员认可的镜像转发地址，可通过构建参数指定，不能只因缓存有 3.12 就直接换 Python 版本。

对标准 rootful Docker，NVIDIA 官方配置方式为 `sudo nvidia-ctk runtime configure --runtime=docker` 后重启 Docker；rootless 配置不同。此操作会修改 daemon 配置，重启可能影响已有容器，故目前只记录方法，尚未执行或要求用户立即执行。

## 镜像生成与部署后的验证顺序

以下命令是后续步骤，不表示镜像已生成。

构建机在项目根目录：

```bash
docker build --platform linux/amd64 --target runtime --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 -t hipporag-local:gpu .
docker save -o hipporag-gpu-linux-amd64.tar hipporag-local:gpu
```

项目复制到服务器的新子目录 `/home/ailab/data/HippoRAG/hipporag_local`，不覆盖目录中已有内容；密钥在服务器单独设置 `.env`，不进入镜像或共享项目包。NVIDIA runtime 配置并验证后，在服务器项目根目录：

```bash
docker load -i hipporag-gpu-linux-amd64.tar
docker compose -f compose.server.yaml run --rm hipporag embedding
docker compose -f compose.server.yaml run --rm hipporag probe
docker compose -f compose.server.yaml run --rm hipporag prepare
```

Embedding 必须显示 cuda 且通过向量检查。Qwen probe 必须返回有效内容及 token 用量。随后才运行 index、answer、export、evaluate：

```bash
docker compose -f compose.server.yaml run --rm hipporag index
docker compose -f compose.server.yaml run --rm hipporag answer
docker compose -f compose.server.yaml run --rm hipporag export --predictions /app/outputs/server_gpu/banking/predictions.json --output /app/outputs/server_gpu/banking/eval_input.json
docker compose -f compose.server.yaml run --rm hipporag evaluate --input /app/outputs/server_gpu/banking/eval_input.json --output-dir /app/outputs/server_gpu/banking/evaluation/run_01
```

当前本机 Docker Linux Engine 未运行，尚不能实际构建或测试镜像；项目源码、模型复制、数据预处理、业务问答及四指标已完成本机/样例层面的验证，不能替代容器验证。

## 官方参考

- [NVIDIA Docker runtime 配置](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)
- [CUDA 驱动兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [Docker 离线载入镜像](https://docs.docker.com/reference/cli/docker/image/load/)
