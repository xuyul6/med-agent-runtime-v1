# MedAgent Runtime

面向个人项目的医疗信息 Multi-Agent 运行时，重点展示多 Agent 编排、风险控制、知识检索、监控与评测。支持整理症状、查询有限的公开药品资料、阅读报告文字、准备就医材料，也能回答普通生活问题。

这是**工程演示原型，不是经过临床验证的医疗服务**：不诊断、不处方、不决定个人剂量；没有接入医院、病历、号源、真实预约、急救调度或人工转接。演示请使用虚构信息，不要输入真实健康隐私。使用 AI 辅助开发。

## 当前状态

- 本地离线测试（测试文件不随仓库发布）：后端 916 项通过、1 项跳过；前端 40 项通过，生产构建通过。离线测试不能证明真实模型回答质量。
- 真实模型验证只做过小范围定向回归；当前版本没有重跑历史上的 39 项完整回归。
- 还没有做公平的多 Agent 对单 Agent、知识库对无检索的对照实验，不能宣称这两项效果更好；也没有证据支持任何整体 Token 节省率。
- `.github/workflows/ci.yml` 只做后端导入检查和前端构建，远端 CI 结果以 GitHub 上的实际运行为准。

## 请求链路

```text
对象解析 / 风险门 → 隔离记忆 → 三路意图融合 → 结构化路由与任务分工
       → 受控工具 / 多路检索 → 角色并行执行 → 引用与输出校验
       → 协作汇总 → Redis 会话 / ChromaDB 摘要 → 脱敏 Trace
```

风险命中时直接返回紧急求助提示，不等待模型、检索或存储。并非每次请求都会调用所有角色、检索或回答模型；风险规则是工程筛查，不保证识别所有急症。

| 角色 | 职责与边界 |
| --- | --- |
| `general` | 回答普通生活问题，介绍能力范围并澄清需求。 |
| `symptom_guide` | 整理用户原述、时间线与信息缺口，不作诊断。 |
| `drug_reference` | 查找有限的公开药品资料，说明产品、人群、地区等适用范围，不生成个人用药方案。 |
| `report_explainer` | 保留用户给出的项目、数值、单位和缺口，按可用资料解释通用概念。 |
| `visit_guide` | 整理待问问题和就医材料，不能查询医院流程或办理预约。 |
| `risk_handoff` | 风险命中时输出确定性的紧急求助建议，明确未实际转接。 |

## 项目结构

```text
backend/
  medagent/
    app.py  runtime.py      FastAPI 入口与总编排
    memory.py               按用户/会话/咨询对象隔离的 Redis 工作记忆与 Chroma 摘要
    monitoring.py           单进程内存监控、告警、路由降权与恢复探测
    evaluate.py             评测命令行入口
    core/                   配置、公共契约、模型网关、请求 Trace
    understanding/          风险规则、意图融合、路由评分、多轮对象与字段跟踪
    agents/                 角色 Profile 与契约、Skills、执行流程、回答校验、协作汇总
    knowledge/              参考资料检索、重排、TTL 缓存与药品资料适用性
    tools/                  注册工具、白名单、预算、超时与熔断（内部工具，不是 MCP 客户端）
    evaluation/             隔离评测、指标、基线、LLM-as-Judge 与资料快照
    benchmark/              预注册的上下文配对实验（默认只生成计划）
    smoke/                  对运行中后端的联网冒烟检查
  resources/                资料摘要、角色 Skills、合成评测与回归数据集
frontend/src/               Vue 工作台：api/ 客户端、components/、utils/、styles/
monitoring/prometheus.yml   可选 Prometheus 抓取配置
```

`docs/`（设计、计划与验收记录）、`backend/artifacts/`（合成评测报告）以及 `backend/tests/`、`frontend/tests/`（离线测试）只保留在本地，不纳入版本库；CI 只做导入检查和前端构建。

## 本地启动

需要 Python 3.12、Node.js 20.19+ 或 22.12+、Docker。

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env.local
# 设置模型密钥、模型 ID、Redis 密码及对应 Redis URL。
cd ..
docker compose --env-file backend/.env.local up -d redis chroma
cd backend
python -m uvicorn medagent.app:app --host 127.0.0.1 --port 8000
```

另开终端：

```bash
cd frontend
npm ci
npm run dev
```

默认工作台 `http://localhost:5173`，API 文档 `http://localhost:8000/docs`。前端通过 `/api` 代理请求，可用 `VITE_API_TARGET` 修改代理目标。密钥只放在后端被忽略的配置文件中。

离线预览：后端设置 `MAR_PREVIEW=true`，使用固定预览输出、字符哈希向量和内存会话，界面会标明预览，不代表真实模型效果。线上向量模式为 `MAR_EMBEDDINGS=onnx`，首次运行需要下载 Chroma 官方模型缓存。

全栈容器：`docker compose --env-file backend/.env.local up -d --build`，API 与工作台端口只绑定回环地址。

## 验证

```bash
# backend/
python -m pytest -q
# 以下命令会调用真实模型并产生费用，仅在明确授权后运行。
python -m medagent.evaluate --url http://127.0.0.1:8000 --output artifacts/evaluation.json
python -m medagent.smoke.roles --url http://127.0.0.1:8000 --output artifacts/role-smoke.json
# 仅生成受控上下文实验计划，不连接后端或模型；加 --execute 才会真实运行。
python -m medagent.benchmark --output artifacts/paired-plan.json
# frontend/
npm test
npm run build
```

对话评测在隔离状态下调用与工作台相同的 `runtime.chat`、记忆和工具。规则检查、Judge 分数或缺测、未运行项和 Token 用量分别记录；CLI 拒绝覆盖已有报告。Judge 是辅助指标，不能代替专业事实核验；不同代码版本的分数不合并比较。

常用接口：`GET /health`、`POST /chat`、`GET /agents`、`GET /skills`、`GET /tools`、`GET /knowledge`、`GET /monitor`、`GET /metrics`、`GET /traces/{request_id}`、`POST /eval/run`。

## 边界与资料

没有完整账户认证、医疗级权限控制、数据删除界面、医院接口、OCR 或临床验证。Redis 会话 TTL 为 24 小时，压缩摘要持久化在 ChromaDB；摘要带用户原话出处，不应当成确诊事实。监控统计只在单进程内存中，重启后清空。

药品目录只包含英国成人概览和美国 Advil 200 mg 口服包衣片的有限标签摘要，按产品、规格、剂型、人群和地区过滤，没有完整说明书或相互作用数据库。公开资料默认为草稿，核对后才允许检索；允许检索不等于医学审核。资料来源与使用条件见 [THIRD_PARTY.md](THIRD_PARTY.md)，贡献约定见 [AGENTS.md](AGENTS.md)。
