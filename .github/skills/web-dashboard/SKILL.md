---
name: web-dashboard
description: "启动或管理 Web 可视化面板。Use when: 启动面板, web dashboard, 启动服务, start web, 停止服务, stop web, 可视化, 网页界面, Flask, 策略执行界面。"
argument-hint: "操作类型，例如：启动、停止、查看状态"
---

# Web 可视化面板 (Web Dashboard)

Flask 驱动的策略执行与结果展示面板，自动发现 `src/strategy/` 下所有策略。

## 功能

- 自动发现策略目录，读取每个策略的 `README.md` 生成描述
- 从 CLI 参数自动生成 Web 表单（文本框、数字、日期、复选框）
- 在线执行策略，实时查看日志输出
- 展示选股结果、候选列表、回测汇总

## 使用步骤

### 启动面板

```bash
cd /nvme5/xtang/gp-workspace/gp-quant

# 方式一：直接启动
python web/app.py --port 5050

# 方式二：使用脚本启动（后台运行）
./web/start_web.sh

# 访问地址
# http://localhost:5050
```

### 停止面板

```bash
./web/stop_web.sh
```

### 面板使用流程

1. 打开浏览器访问 `http://localhost:5050`
2. 首页展示所有已注册策略的卡片
3. 点击策略卡片进入策略页面
4. 填写参数表单（扫描日期、Top N、持有天数等）
5. 点击执行按钮，等待策略运行
6. 查看结果：选股列表、汇总指标、执行日志

### 添加新策略到面板

面板自动发现 `src/strategy/*/` 下的策略。新策略需要：

1. 在 `src/strategy/<name>/` 目录下创建策略代码
2. 提供 `README.md`（面板从中提取描述和参数说明）
3. 提供 CLI 入口脚本（面板解析 argparse 参数生成表单）

面板会在启动时自动扫描并注册。
