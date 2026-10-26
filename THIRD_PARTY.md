# 第三方依赖与参考资料

项目应用代码、界面、角色规则及测试在本版本中按需求规格重新编写。第三方库与公开医学资料有各自的权利和使用条件；本文件不将它们声明为项目作者原创。

## 运行依赖

依赖版本见 `backend/requirements.txt`、`frontend/package-lock.json`，许可证和版权信息以各依赖发行包为准：

- [FastAPI](https://github.com/fastapi/fastapi)、[Pydantic](https://github.com/pydantic/pydantic)、[HTTPX](https://github.com/encode/httpx)、[Uvicorn](https://github.com/encode/uvicorn)。
- [redis-py](https://github.com/redis/redis-py)、[Chroma](https://github.com/chroma-core/chroma)、[python-dotenv](https://github.com/theskumar/python-dotenv)、[PostHog Python](https://github.com/PostHog/posthog-python)。
- [Vue](https://github.com/vuejs/core)、[Vite](https://github.com/vitejs/vite)、[esbuild](https://github.com/evanw/esbuild)。
- 本地语义向量通过 Chroma 的 ONNX 实现加载 [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)，模型文件不随源码提交。
- Redis、Chroma 和 Nginx 容器镜像由各发行方分发，未纳入应用源码。

## 医学资料

仅保存自行撰写的短导读，并保留原页面链接、版本与适用范围，不复制整篇正文。

- [MedlinePlus：How to Understand Your Lab Results](https://medlineplus.gov/lab-tests/how-to-understand-your-lab-results/)：报告阅读和参考范围的一般背景。
- [NHS：Ibuprofen for adults](https://www.nhs.uk/medicines/ibuprofen-for-adults/)：英国成人药品概览。不是本项目编写的制剂说明书，不能外推到儿童、其他剂型或地区。
- `consultation-record` 和 `symptom-record` 是本项目的信息整理模板；没有医学指南或医院业务来源的含义。

资料更新时核对来源、文本使用条件及适用范围，不把来源页面的权利转授给使用者。
