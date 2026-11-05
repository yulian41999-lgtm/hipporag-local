# 图谱浏览器

本工具读取当前已完成的 1700 篇语料索引，提供实体搜索、一跳局部图、节点详情、来源片段和切换中心节点。原图、向量和 LLM 缓存均不改动，不调用模型。

## 当前入口

服务器服务只监听 127.0.0.1:8790。通过 SSH 转发后，本机浏览器打开 <http://127.0.0.1:8790/>。

实体检索覆盖全部非空实体；每次显示最多 30 个搜索结果。局部图默认最多选取 16 个邻居，优先事实边和较大权重，保留所选节点之间的边。点击节点查看内容，再选择“以此为中心”继续浏览。图上的事实来自抽取结果，不等同于经人工核实的法律结论。

## 再次连接

先在服务器项目内检查服务，不要重复启动：

```bash
cd /home/ailab/data/HippoRAG/hipporag_local
cat outputs/graph_visualization/viewer.pid
ss -ltn '( sport = :8790 )'
```

仅当服务未运行、端口空闲时启动：

```bash
nohup .venv/bin/python -u tools/graph_viewer/server.py > outputs/graph_visualization/viewer.log 2>&1 < /dev/null &
echo "$!" > outputs/graph_visualization/viewer.pid
```

在本机终端建立转发，密码只在 SSH 提示中输入：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -L 127.0.0.1:8790:127.0.0.1:8790 ailab@10.254.2.76
```

如果本机 8790 已有转发，不要重复创建。转发断开后页面无法读取数据，重新连接即可。服务绑定本机回环地址，不开放公网或修改防火墙。

## 验证

已通过服务器与本机 HTTP 检查、中文实体检索、邻域边端点和去重检查、来源片段检查、再次切换中心节点以及无效节点拒绝检查。页面初次加载和默认局部图已在浏览器中显示；后续浏览器自动化因 Work 设置禁用而未继续完成。

所有服务器文件均位于 /home/ailab/data/HippoRAG 内。未经用户明确同意，不中断任何任务。
