# MedAgent Runtime

一个以健康信息咨询为场景的多 Agent 编排运行时。每次请求先经过不依赖模型的风险筛查，再识别意图，并决定交给一个角色处理还是多个角色并行。每个角色只能使用自己领域的工具和资料，输出经过校验后合并，再写回按用户和咨询对象隔离的记忆。全过程留下 Trace，进入监控和评测。

这是个人工程项目，**不是经过临床验证的医疗服务**：不诊断、不处方、不决定个人剂量，也没有接入医院、预约、急救调度或人工客服。演示请使用虚构信息。项目使用 AI 辅助开发。

## 一次请求怎么走

```text
风险筛查 ─┬─ 命中急症 → 确定性求助提示（不调用模型、检索和存储）
          └─ 其他 → 意图识别 → 路由决策 → 单角色 / 多角色并行执行 → 合并 → 记忆回写
                                                                          ↓
                                       监控（实例健康度 → 下一次路由） · 评测（红队集、回归 → 规则修改）
```

| 阶段 | 做什么 | 代码 |
| --- | --- | --- |
| 风险筛查 | 按体征表述、上下文排除（否定、既往、假设、虚构、比喻等）判定 `CRITICAL / CAUTION / ROUTINE`。命中急症时直接返回求助提示；之后 30 分钟内，同一会话里接着描述情况的消息仍按急症处理。 | `understanding/risk.py` |
| 意图识别 | 关键词、向量相似度、模型三路打分，按 0.35 / 0.20 / 0.45 加权融合，某一路不可用时其余几路重新归一化。少量显式规则会覆盖融合结果，比如续问沿用上一任务、本轮明确给出药名。同时抽取药名、报告项目等字段，只接受能在用户原话里找到的值，并记录出处轮次。 | `understanding/classification.py`、`conversation.py`、`medications.py` |
| 路由决策 | 某个领域必须在本轮有任务证据才放行：旧对话里的字段不能单独激活角色，只是顺带提到的药名或报告（如“在吃布洛芬”）也不算任务。候选角色按意图 0.40、任务匹配 0.20、字段完整度 0.15、业务规则 0.10、实例健康度 0.15 打分，最多 1 个主角色加 2 个辅助角色。每个候选的分项和落选原因都写进 Trace。 | `understanding/routing.py` |
| 执行 | 多个角色并行执行，有并发上限和整体时限。每个角色有两个实例，失败或超时时换同角色的另一个实例，不让别的角色代答。角色内部依次做字段预检、检索、上下文工具、模型生成（最多两轮工具调用），最后校验输出；格式类错误允许修正一次。 | `runtime.py`、`agents/execution.py`、`tools/registry.py` |
| 合并 | 以主角色的回答为主体，辅助角色的回答作为分节。合并时模型只能调整已有分节的顺序，不能新增内容。会检查角色之间的字段冲突和引用版本冲突。主任务失败时如实说明，不让其他角色顶替。 | `agents/collaboration.py`、`agents/checks.py` |
| 记忆回写 | 按（用户、会话、咨询对象）的哈希隔离。Redis 保存最近消息和带出处的字段，有效期 24 小时；较长的对话压缩成摘要存进 Chroma，下次在同一作用域内召回。 | `memory.py` |
| 监控与评测 | 记录每个角色实例和工具的成功率与延迟。持续不达标的实例在路由中降权或暂停，之后用正常请求做恢复探测。评测包括离线测试、风险红队集、真实模型回归和 LLM-as-Judge。 | `monitoring.py`、`evaluation/` |

离线预览模式下的路由结果（未调用模型，所以意图识别只有关键词和向量两路）：

```text
"我头痛两天了，想整理一下症状，顺便查一下布洛芬的说明书"
  → 主角色 symptom_guide（0.71），辅助 drug_reference（0.68），并行执行
"布洛芬有什么禁忌"
  → 只用 drug_reference（0.83）
"胸口压着疼，喘不上气"
  → 风险筛查命中，交给 risk_handoff，不进入意图识别
```

## 为什么拆成多个角色

- **权限隔离**：每个角色只能调用自己的工具、检索自己领域的资料。即使模型被注入的指令控制，越权工具调用和跨领域检索也会被代码拒绝。
- **上下文分开**：每个角色的规则（`resources/skills/`）和资料只覆盖自己的领域，不必把所有规则塞进同一个提示。
- **复合问题并行**：一句话里同时涉及症状和药品时，两个角色同时执行。
- **分角色迭代**：每个角色有独立的输入输出契约，可以单独测试和修改。

拆分也有代价：路由可能出错，合并时可能冲突，Token 和延迟会增加。所以由路由决定每次用几个角色，简单问题只用一个。**多角色是否真的比单 Agent 更好，还没有做对照实验**，见下文“验证现状”。

## 角色与权限

| 角色 | 职责与边界 | 可用工具 |
| --- | --- | --- |
| `general` | 回答普通生活问题，介绍能力范围，澄清需求 | `inspect_request_context` |
| `symptom_guide` | 整理用户原述、时间线和信息缺口，不作诊断 | `check_symptom_fields`、`lookup_reference` |
| `drug_reference` | 查找有限的公开药品资料，说明产品、人群、地区等适用范围，不给个人用药方案 | `check_medication_fields`、`check_medication_scope`、`lookup_reference` |
| `report_explainer` | 保留用户给出的项目、数值、单位和缺口，按资料解释通用概念 | `check_report_fields`、`lookup_reference` |
| `visit_guide` | 整理就医材料和要问的问题，不能查医院流程或办理预约 | `check_visit_fields`、`build_visit_checklist`、`lookup_reference` |
| `risk_handoff` | 风险命中时输出固定的求助建议，说明没有实际转接；不调用模型 | `create_handoff_summary` |

`lookup_reference` 只返回调用角色所属领域、且已标记为核对过的资料。工具调用受单次请求的预算、超时和熔断限制。

## 验证现状

测试文件和评测报告只保留在本地，不随仓库发布；数据集在 `backend/resources/` 中公开。

| 项目 | 方法 | 结果 |
| --- | --- | --- |
| 离线测试 | 假模型，不调用付费服务 | 后端 1017 项通过、1 项跳过、4 项为已知缺口（`xfail`）；前端 40 项通过，生产构建通过 |
| 风险筛查 | 红队集 `risk-redteam-v1.json`（104 例）和预留集 `risk-redteam-holdout-v1.json`（40 例），规则版本 `screening-2` | 调优集急症识别 56/56，误报 2/48；预留集 26/26，误报 1/14。预留集结果出来后又改过一次规则，预留集已不算独立 |
| 意图与路由 | 标注集 `routing-labels-v1.json`（58 例）和预留集 `routing-labels-holdout-v1.json`（30 例），只用关键词和哈希向量两路，不调用模型 | 调优集的主角色和参与角色全部正确；预留集主角色 26/30、参与角色集合 25/30（改规则前为 21/30、17/30）。两个集合由同一人编写，预留集只能防止针对单个案例打补丁 |
| 回答边界 | 红队集 `answer-boundary-redteam-v1.json`（38 例违规、30 例正常）和预留集（19 例违规、11 例正常），只测确定性检查，不调用模型 | 调优集违规全部拦截，正常内容无误拦；预留集违规拦住 8/19（改规则前为 0/19），正常内容无误拦。剂量类拦住 7/8；诊断、停换药、安全保证、虚构预约这类说法变化多，只拦住 1/11 |
| 隔离与注入 | 假设模型被注入内容完全控制，只检查代码层的防线 | 越权工具、跨领域检索、个人剂量、诊断、伪造预约都被拦截，用户之间的记忆不串用；3 个已知缺口见“已知局限” |
| 药名识别 | 63 个常见通用名及英文名，按 4 种句式生成请求 | 320/320 条识别出药名并路由到 `drug_reference`；词表外的药名依赖模型提议，且必须能在原话中找到 |
| 真实模型 | 2026-10-29 前做过定向回归和一次上下文配对实验 | 配对实验 32 对完成 24 对，未完成；当前版本还没有重跑完整回归 |
| 多 Agent 对单 Agent | — | **未做**，暂不能说多角色编排效果更好 |

离线测试只能证明代码按设计运行，不能证明真实模型的回答质量。

## 快速体验

不需要模型密钥和 Docker。离线预览使用固定输出、哈希向量和内存会话，只展示风险筛查、路由和执行结构，不生成医疗回答。

```bash
cd backend
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
MAR_PREVIEW=true python -m uvicorn medagent.app:app --host 127.0.0.1 --port 8000
```

另开终端启动工作台，打开 `http://localhost:5173`：

```bash
cd frontend
npm ci
npm run dev
```

## 完整启动

需要 Python 3.12、Node.js 20.19+ 或 22.12+、Docker。

```bash
cd backend
cp .env.example .env.local
# 设置模型密钥、模型 ID、Redis 密码及对应的 Redis URL
cd ..
docker compose --env-file backend/.env.local up -d redis chroma
cd backend
python -m uvicorn medagent.app:app --host 127.0.0.1 --port 8000
```

API 文档在 `http://localhost:8000/docs`。前端通过 `/api` 代理请求，可用 `VITE_API_TARGET` 修改代理目标。密钥只放在后端被忽略的配置文件中。线上向量模式 `MAR_EMBEDDINGS=onnx` 首次运行需要下载 Chroma 的模型缓存。

全栈容器：`docker compose --env-file backend/.env.local up -d --build`，API 和工作台端口只绑定本机回环地址。

## 项目结构

```text
backend/medagent/
  app.py  runtime.py   FastAPI 入口与总编排
  understanding/       风险筛查、意图融合、药名识别、多轮字段跟踪、路由评分
  agents/              角色 Profile 与契约、Skills 加载、角色执行、输出校验、合并
  tools/               工具注册、角色白名单、预算、超时与熔断（内部工具，不是 MCP）
  knowledge/           资料检索、重排、缓存，以及药品资料的适用范围判断
  memory.py            Redis 工作记忆与 Chroma 摘要
  monitoring.py        单进程内存监控、告警、路由降权与恢复探测
  evaluation/          评测隔离、指标、基线、LLM-as-Judge、风险红队统计
  benchmark/           配对实验（默认只生成计划）
  smoke/               对运行中后端的联网冒烟检查
  core/                配置、公共契约、模型网关、请求 Trace
backend/resources/     资料导读、角色 Skills、药名词表、评测与红队数据集
frontend/src/          Vue 工作台：对话、路由与执行详情、监控、评测结果
```

## 验证命令

```bash
# backend/：离线测试（测试文件只在本地）
python -m pytest -q
python -m medagent.evaluation.risk_redteam --output artifacts/risk-redteam.json
python -m medagent.evaluation.routing_labels --output artifacts/routing-labels.json
python -m medagent.evaluation.answer_boundary --output artifacts/answer-boundary.json
# 以下命令会调用真实模型并产生费用，只在明确授权后运行
python -m medagent.evaluate --url http://127.0.0.1:8000 --output artifacts/evaluation.json
python -m medagent.smoke.roles --url http://127.0.0.1:8000 --output artifacts/role-smoke.json
# frontend/
npm test
npm run build
```

评测在隔离状态下调用与工作台相同的 `runtime.chat`、记忆和工具。规则检查、Judge 分数、未运行项和 Token 用量分开记录；命令拒绝覆盖已有报告；不同代码版本的分数不合并比较。Judge 只是辅助指标，不能代替专业的事实核验。

常用接口：`GET /health`、`POST /chat`、`GET /agents`、`GET /skills`、`GET /tools`、`GET /knowledge`、`GET /monitor`、`GET /metrics`、`GET /traces/{request_id}`、`POST /eval/run`。

## 已知局限

- **没有身份认证**：`user` 和 `session` 由客户端填写，持有 `request_id` 就能读取对应的 Trace。
- **资料由使用者自审**：只有标记为已批准（`approved`）的资料才能被检索和引用。工作台导入资料时勾选“允许项目内检索”即为批准，表示“使用者自己核对过”，不代表医学审核。这是单用户本机设计：`POST /knowledge` 不需要认证，能访问 API 的人都能导入并批准资料。已批准资料里的个人剂量、伪造预约等说法仍会被输出校验拦截，但校验没覆盖的误导性内容可能被引用。
- **系统提示里有不可信文本**：角色的系统提示中包含用户原话、历史对话和检索到的资料原文，真实模型可能更容易服从其中注入的指令。
- **摘要只按时间清理**：Chroma 中的会话摘要默认保留 7 天（`MAR_EPISODE_TTL_SECONDS`），过期后不再召回，并在启动时和写入摘要时清理；没有按用户删除的入口。
- **资料很少**：共 8 份自行撰写的导读（药品 2、报告 4、症状 1、就医 1）。药品只有英国成人布洛芬概览和美国 Advil 200 mg 标签摘要，没有相互作用数据库。
- **风险规则有漏有误**：风险规则是工程筛查，不保证识别所有急症，也有已知误报。
- **回答边界靠规则匹配**：剂量数字、频次拦得比较稳；诊断、停换药、安全保证和虚构预约只按固定说法匹配，换个说法就可能漏过（预留集只拦住 1/11）。复述用户原话（“你提到…”）的句子不检查。
- **监控与实例切换有限**：监控数据只在单进程内存里，重启即清空。同一角色的两个实例使用同一个模型服务，换实例只能应对偶发失败，服务整体故障时无效。

资料来源与使用条件见 [THIRD_PARTY.md](THIRD_PARTY.md)，贡献约定见 [AGENTS.md](AGENTS.md)。
