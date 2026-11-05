# 实时评分看板

入口为 <http://127.0.0.1:8791/>，每 10 秒读取现有评分记录。看板只读，不触发模型调用，不重启评分任务。

展示生成答案、四指标评分和汇总报告三个阶段；逐题显示已评分、评分中、等待或含评分错误。题目四项分数在该题完整结果写入后显示，未评分不计为零。阶段均值仅包含有效评分，同时显示有效题数和失败题数。最近裁判调用列出任务名称、尝试数和状态，不显示标准答案或模型密钥。

## 服务与连接

所有服务器文件均位于 /home/ailab/data/HippoRAG 内。服务只监听服务器回环地址 127.0.0.1:8791，不开放公网。

先检查端口是否已监听，不要重复启动：

```bash
cd /home/ailab/data/HippoRAG/hipporag_local
ss -ltn '( sport = :8791 )'
cat outputs/graph_visualization/score_viewer.pid
```

只有服务未运行且端口空闲时启动：

```bash
nohup .venv/bin/python -u tools/score_viewer/server.py > outputs/graph_visualization/score_viewer.log 2>&1 < /dev/null &
echo "$!" > outputs/graph_visualization/score_viewer.pid
```

本机转发命令，密码仅在 SSH 提示中输入，不保存到文件：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:8791:127.0.0.1:8791 ailab@10.254.2.76
```

当前本机已建立转发。若断开，重新建立后再打开入口。不得未经用户明确同意中断任何任务。

## 验证

已完成页面脚本语法检查、服务器与本机 HTTP 200 检查、15 题条数和状态一致性检查、四指标均值范围及有效样本数检查。Work 设置禁用了浏览器控制，未完成浏览器截图与交互验收；用户可以点击本机入口查看。
