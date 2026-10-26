# 独立实现验证记录

本文件只记录 `independent-v1` 的验证，日期为2026-10-26。其他实现的成绩不沿用。

## 已验证

- 后端29项测试通过，涵盖风险语境、作用域隔离、记忆压缩顺序、并发消息、角色工具隔离、熔断恢复、超时、引用与输出边界、实例重试、API、模型用量和 Judge 失败。
- 其中资料更新测试使用真实本地 ChromaDB，注入测试向量，不下载模型。
- 前端4项测试通过，覆盖不明用量、危险链接、请求契约及 HTTP 失败。
- 前端依赖更新后 npm 审计为0告警，生产构建通过。
- 全新固定数据集 `independent-medical-v1` 的6个案例，规则及在线 Judge 综合通过6/6；模型为 `claude-haiku-5-5`，使用服务商 JSON Schema 结构化输出，真实 ONNX 向量、Redis 和 ChromaDB。原始结果位于 `backend/artifacts/live-evaluation.json`，Judge 用量与运行时用量分别记录。
- 长会话实测：预置10条虚构历史后调用真实 `/chat`，触发模型摘要，Redis保留4条近期消息，ChromaDB情景回读命中。五项检查全部通过，记录为 `backend/artifacts/memory-check.json`。
- Compose 配置解析通过。`docs/source-audit.json` 的精确文件比对未发现与先前应用源码完全相同的文件；此检查不是法律权属认定。
- 保留初轮及修复前原始报告。联调修复了 Redis 配置不匹配、模型短 JSON 截断、就医信息整理路由与药名元数据迁移；没有调低评测阈值来消除失败。
- 浏览器实测：`http://127.0.0.1:5174/` 已切换为独立实现，普通就医准备请求耗时7047ms、服务商输入2269 Token，storage=available；引用指向新仓库的原创沟通模板。截图为 `docs/workbench.png`。

## 待完成

- Docker 全栈镜像构建及真实 Token 对比实验。

本地语义向量、模型质量与医学准确性是不同的验证对象。任何测试通过都不等同于临床验证。
