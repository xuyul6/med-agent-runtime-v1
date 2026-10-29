# 第三方依赖与参考资料

项目应用代码、界面、角色规则及测试在本版本中按需求规格重新编写。第三方库与公开医学资料有各自的权利和使用条件；本文件不将它们声明为项目作者原创。

## 运行依赖

依赖版本见 `backend/requirements.txt`、`frontend/package-lock.json`，许可证和版权信息以各依赖发行包为准：

- [FastAPI](https://github.com/fastapi/fastapi)、[Pydantic](https://github.com/pydantic/pydantic)、[HTTPX](https://github.com/encode/httpx)、[Uvicorn](https://github.com/encode/uvicorn)。
- [redis-py](https://github.com/redis/redis-py)、[Chroma](https://github.com/chroma-core/chroma)、[python-dotenv](https://github.com/theskumar/python-dotenv)、[PostHog Python](https://github.com/PostHog/posthog-python)。
- [Vue](https://github.com/vuejs/core)、[Vite](https://github.com/vitejs/vite)、[esbuild](https://github.com/evanw/esbuild)。
- 本地语义向量通过 Chroma 的 ONNX 实现加载 [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)，模型文件不随源码提交。
- Redis、Chroma、Nginx 和可选的 [Prometheus](https://github.com/prometheus/prometheus) 容器镜像由各发行方分发，未纳入应用源码；版本见 `compose.yaml`。

## 医学资料

仅保存自行撰写的短导读，并保留原页面链接、版本与适用范围，不复制整篇正文。

- [FAO：Family Nutrition Guide（2004），Topic 10](https://www.fao.org/4/y5740e/y5740e0d.htm)：2026-10-29 查看生病时饮食和饮水的信息，自行撰写有限导读，不宣称 FAO 背书。该资料较早，不是个人饮食处方；[出版物版权说明](https://www.fao.org/4/y5740e/y5740e00.htm)及 [FAO 网站条款](https://www.fao.org/contact-us/terms/)区分非商业用途和其他使用，后续传播或商用须自行核实相应权利。

- [MedlinePlus：How to Understand Your Lab Results](https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/)：报告阅读和参考范围的一般背景。2026-10-29 核对 “What is a reference range?” 及不同计量单位的说明，新增独立中文导读 `lab-reference-range-definition`、`lab-unit-overview`；原 `lab-context` 和历史快照保留。导读不提供单位换算或个人参考值，不替代来源原文及专业解释。
- [美国退伍军人事务部：Understand Your Kidney Lab Tests](https://www.va.gov/EKIDNEYCLINIC/rooms/laboratory/topics/understand-your-kidney-lab-tests/index.asp)：2026-10-29 核对血液检验报告的项目名称和结果阅读示例，整理独立中文导读 `lab-structure-overview`；示例不覆盖所有报告格式或个人诊断。
- [NHS：Ibuprofen for adults](https://www.nhs.uk/medicines/ibuprofen-for-adults/)：英国成人药品概览。不是本项目编写的制剂说明书，不能外推到儿童、其他剂型或地区。
- [DailyMed：Advil 布洛芬包衣片](https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=1a665e64-9f30-be37-4a83-38789f1f1e89)：Haleon US Holdings LLC 提交的在用标签，SPL v38；项目只写美国 Advil 200 mg 口服包衣片的成人资料摘要。并核对厂商 LBL-00000027 v38.0，不镜像 PDF、包装图或全文。
- `consultation-record` 和 `symptom-record` 是本项目的信息整理模板；没有医学指南或医院业务来源的含义。
- `backend/resources/medication-names.json` 是项目作者整理的常见药品通用名（国际非专利名）及英文名列表，只用于识别用户提到的药名；不含剂量或临床内容，不是药品数据库，也不表示已有对应资料。

资料更新时核对来源、文本使用条件及适用范围，不把来源页面的权利转授给使用者。

2026-10-27 实查范围、版本、日期和章节见目录的 `provenance`。NHS 材料遵循[网站条款](https://www.nhs.uk/our-policies/terms-and-conditions/)及适用的 OGL 例外：Contains public sector information licensed under the Open Government Licence v3.0. 项目导读与原文区分，不使用 NHS 标识或宣称背书。NLM [版权说明](https://www.nlm.nih.gov/web_policies.html#copyright)区分政府作品和第三方材料；企业药品标签不会因托管于 NLM 就自动成为公有领域，不承诺原文再分发权。DailyMed 在用标签不等同 FDA 最新核准标签。
