# iChat Backends 设计文档

本文档旨在详细阐述 iChat `backends` 模块的设计理念、架构和具体实现。该模块是 iChat 与底层推理引擎（如大语言模型、嵌入模型等）之间的核心桥梁。

## 1. 概述

iChat 的 `backends` 目录负责封装和管理底层的推理引擎。每个后端都是一个独立的适配器，将特定推理引擎（如 vLLM, SGLang, SentenceTransformer）的启动、配置和生命周期管理逻辑封装成统一、可编程的接口。

核心优势：
- **解耦**：主服务 (`serve.py`) 无需关心底层推理引擎的实现细节，只需根据配置选择并实例化对应后端，调用其 `run()` 方法。
- **可扩展性**：添加新推理引擎支持只需实现一个新的后端类，遵循 `BaseBackend` 接口。
- **精细化控制**：后端类可深入控制推理引擎生命周期，实现健康检查、无缝重启、动态配置等高级功能。

## 2. `BaseBackend` 接口

所有后端都继承自 `BaseBackend` 类 (`ichat/backends/base_backend.py`)，该类定义了后端必须实现的核心接口，确保所有后端行为一致：
- `__init__(self, framework_args, backend_argv, backend_ready_event)`
- `async run(self)`
- `cleanup(self)`
- `get_backend_args(self)`
- `wait_for_server_ready(self)`

## 3. 支持的后端类型

### 3.1 vLLM Backend

详见 [vllm/README.md](./vllm/README.md)

- 封装 vLLM 服务，独立参数解析与转换，精细化生命周期管理。
- 支持服务就绪通知、服务预热、健康检查、无缝集成。
- 适用于大语言模型推理。

### 3.2 SGLang Backend

详见 [sglang/README.md](./sglang/README.md)

- 封装 SGLang 服务，独立参数解析，精细化生命周期管理。
- 支持子进程健康检查、无缝集成、鲁棒性监控。
- 适用于大语言模型推理。

### 3.3 Sentence Transformer Backend

详见 [sentence_transformer/README.md](./sentence_transformer/README.md)

- 封装 `sentence-transformers` 库，提供文本嵌入（embedding）和重排序（rerank）服务。
- 独立参数解析，支持 embedding/rerank 两种任务。
- 以 FastAPI+Uvicorn 方式服务化，支持 OpenAI 兼容 API（如 `/v1/embeddings`、`/v1/rerank`）。
- 适用于文本向量化、检索增强等场景。

## 4. 设计对比

| 后端类型         | 典型用途         | 启动方式         | 健康检查/监控方式         | 主要接口/端点           |
|----------------|----------------|----------------|-------------------------|------------------------|
| vLLM           | LLM推理         | 直接集成API/Socket | /health + RPC存活检查    | OpenAI兼容             |
| SGLang         | LLM推理         | 直接集成API/多子进程 | /get_model_info + 子进程监控 | OpenAI兼容             |
| Sentence       | 嵌入/重排序     | FastAPI子进程      | /health + 进程监控        | /v1/embeddings, /v1/rerank |

## 5. 配置与扩展

所有后端均可通过 `configs.yaml` 配置，支持多模型、多实例部署。详见各自 README。

---

> vLLM、SGLang、SentenceTransformer 的详细设计与实现请分别参考各自子目录下的 README 文档。
