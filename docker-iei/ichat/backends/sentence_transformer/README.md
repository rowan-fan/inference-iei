# iChat Sentence Transformer Backend 设计文档

本文档旨在详细阐述 iChat `sentence_backend` 模块的设计理念、架构和具体实现。该模块是 iChat 与 `sentence-transformers` 推理库之间的核心桥梁，专门用于提供文本嵌入 (Embedding) 和重排序 (Reranking) 功能。

## 1. 概述

`sentence-transformers` 是一个广泛用于计算句子、文本和图像嵌入的 Python 库，并提供了强大的重排序（Cross-Encoder）模型。然而，它本身只是一个库，不提供开箱即用的网络服务。

`sentence_backend` 的核心目标就是将 `sentence-transformers` 的强大功能封装成一个符合 iChat 架构标准的、可通过 API 访问的独立服务。

这种设计的核心优势在于：

- **服务化 (Serving)**: 将一个纯粹的 Python 库转化为一个健壮的、可独立部署的推理服务，使其能够处理网络请求。
- **标准化 (Standardization)**: 提供与 OpenAI 兼容的 API 端点，如 `/v1/embeddings` 和 `/v1/rerank`，使得上层应用可以无缝切换和使用。
- **解耦 (Decoupling)**: iChat 的主服务 (`serve.py`) 无需关心 `sentence-transformers` 的模型加载、设备管理和 API 实现细节。它只需根据配置选择并实例化 `SentenceBackend`。
- **可扩展性 (Extensibility)**: 设计支持同时部署多种嵌入和重排序模型。每个 `SentenceBackend` 实例运行一个模型，而 Gateway 可以管理多个不同的 `SentenceBackend` 实例。
- **专注性 (Focus)**: 该后端专注于 CPU 环境下的高效推理，为 embedding 和 rerank 等任务提供轻量级的部署方案。

## 2. 设计目标

1.  **封装 `sentence-transformers`**: 将模型加载和推理逻辑封装在一个可通过 `uvicorn` 运行的 FastAPI 应用中。
2.  **独立的参数解析**: 后端内部拥有独立的参数解析逻辑，能够解析 `model-path`, `task-type` (embedding/rerank) 等专用参数。
3.  **支持双重任务**: 明确支持两种核心任务：
    -   `embedding`: 使用 `SentenceTransformer` (Bi-Encoder) 模型生成文本嵌入向量。
    -   `rerank`: 使用 `CrossEncoder` 模型对查询和文档列表进行重排序。
4.  **生命周期管理**: 遵循 `BaseBackend` 接口，实现服务的启动、健康检查、预热和优雅关闭。
5.  **无缝集成**: 使得 `serve.py` 只需实例化 `SentenceBackend` 并调用其 `run()` 方法即可启动服务，完全无需了解其内部实现。
6.  **配置驱动**: 完全通过 `configs.yaml` 进行配置，支持 Gateway 自动拉起和管理多个 embedding 和 rerank worker。

## 3. `sentence_backend` 架构

`sentence_backend` 包含三个核心组件，共同构成一个完整的、自包含的服务单元。

-   **`sentence_backend.py`**:
    -   定义了 `SentenceBackend` 类，继承自 `BaseBackend`。
    -   作为服务的“编排者”，它不执行具体的推理逻辑。
    -   负责解析从 `serve.py` 传递过来的参数，并启动 `api_server`。
    -   管理 `api_server` 的生命周期，包括启动、预热、健康检查和关闭。

-   **`api_server.py`**:
    -   一个独立的 FastAPI 应用。
    -   在启动时，根据命令行参数加载一个 `sentence-transformers` 模型（`SentenceTransformer` 或 `CrossEncoder`）。
    -   根据加载的模型类型，暴露 `/v1/embeddings` 或 `/v1/rerank` 端点。
    -   包含一个 `/health` 端点，用于健康检查和服务预热。

-   **`args.py`**:
    -   定义并解析 `sentence_backend` 所需的全部命令行参数，如 `--model-path`, `--task-type`, `--device` 等。

这种分层设计将服务管理（`SentenceBackend`）与业务逻辑（`api_server.py`）清晰地分离开来。

## 4. `SentenceBackend` 类架构

`SentenceBackend` 的实现比 `VLLMBackend` 或 `SGLangBackend` 更为简洁，因为它管理的 `api_server` 是一个纯 Python 应用，不涉及复杂的外部子进程。

### 4.1. `__init__` 和 `_parse_args`

-   构造函数接收 `framework_args` 和 `backend_argv`。
-   调用私有的 `_parse_args(backend_argv)` 方法，该方法会使用 `sentence_backend/args.py` 中定义的解析器来处理所有后端专属参数。
-   为了兼容性，它会将 iChat 的标准参数（如 `--model-path`）映射到后端的对应参数。

#### 参数映射关系

| iChat 标准参数      | `sentence_backend` 对应参数 | 描述                               |
| :------------------ | :-------------------------- | :--------------------------------- |
| `--model-path`      | `--model-path`              | 模型权重的路径或HuggingFace ID     |
| `--served-model-name`| `--model-name`              | 在Gateway中注册的模型名称          |
| `--host`            | `--host`                    | 服务监听的主机地址                 |
| `--port`            | `--port`                    | 服务监听的端口                     |
| (无)                | `--task-type`               | **必需**, `embedding` 或 `rerank` |
| (无)                | `--device`                  | 推理设备，默认为 `cpu`             |

### 4.2. `async run(self)`

-   `run` 方法是启动和管理服务的主入口。
-   它通过 `subprocess.Popen` 启动一个新的 Python 进程，运行 `api_server.py` 脚本。
-   将所有解析后的参数传递给子进程。
-   创建并 `await` 两个 `asyncio.Task`:
    1.  `self._wait_and_warmup()`: 等待 `api_server` 启动并完成预热。
    2.  `self._health_check_monitor()`: 持续监控 `api_server` 子进程的存活状态。
-   在 `finally` 块中调用 `self.cleanup()` 确保子进程被正确终止。

### 4.3. `async _wait_and_warmup(self)`

-   在一个循环中，定期（例如每 2 秒）尝试通过 `aiohttp` 访问 `api_server` 的 `/health` 接口。
-   一旦 `/health` 接口返回 200 状态码，意味着模型已加载，服务已就绪。
-   此时，调用 `self.server_ready.set()` 通知 iChat 框架服务可用，并正常返回。

### 4.4. `async _health_check_monitor(self)`

-   在一个循环中，定期（例如每 5 秒）检查由 `subprocess.Popen` 创建的子进程是否仍在运行 (`self.process.poll() is None`)。
-   如果子进程意外退出，此方法将抛出 `RuntimeError`，这将导致 `run` 方法终止，并触发整个 worker 的关闭和重启流程。

### 4.5. `cleanup(self)`

-   负责在服务停止时进行资源清理。
-   如果子进程 `self.process` 仍在运行，则向其发送 `SIGTERM` 信号以请求其优雅关闭。
-   等待一小段时间后，如果进程仍未终止，则发送 `SIGKILL` 信号强制终止。

## 5. API Server (`api_server.py`) 架构

这是一个独立的 FastAPI 应用，是实际执行推理的地方。

### 5.1. 启动流程

1.  使用 `args.py` 中定义的 `ArgumentParser` 解析命令行参数。
2.  根据 `--device` 参数设置推理设备。
3.  根据 `--task-type` 参数加载相应的模型：
    -   如果 `task-type` 为 `embedding`，则加载 `sentence_transformers.SentenceTransformer(model_path, device=device)`。
    -   如果 `task-type` 为 `rerank`，则加载 `sentence_transformers.cross_encoder.CrossEncoder(model_path, device=device)`。
4.  将加载的模型和参数存储在全局状态中。
5.  根据 `--task-type` 动态添加对应的 API 路由。
6.  使用 `uvicorn` 启动 FastAPI 应用。

### 5.2. API 端点

#### 5.2.1. `GET /health`

-   一个简单的健康检查端点。
-   直接返回 `{"status": "ok"}`，用于服务预热和存活探测。

#### 5.2.2. `POST /v1/embeddings` (当 `task-type` 为 `embedding` 时)

-   **Request Body**:
    ```json
    {
      "input": ["sentence 1", "sentence 2"],
      "model": "model-name" // 可选，由Gateway转发而来
    }
    ```
-   **Logic**:
    -   从请求体中获取 `input` 列表。
    -   调用 `model.encode(input, normalize_embeddings=True)` 来计算嵌入。
    -   将结果封装成 OpenAI 兼容的格式返回。
-   **Response Body**:
    ```json
    {
      "object": "list",
      "data": [
        {
          "object": "embedding",
          "index": 0,
          "embedding": [0.1, 0.2, ...]
        },
        {
          "object": "embedding",
          "index": 1,
          "embedding": [0.3, 0.4, ...]
        }
      ],
      "model": "model-name",
      "usage": { "prompt_tokens": 0, "total_tokens": 0 }
    }
    ```

#### 5.2.3. `POST /v1/rerank` (当 `task-type` 为 `rerank` 时)

-   **Request Body**:
    ```json
    {
      "query": "The user's query.",
      "documents": ["doc 1", "doc 2", "doc 3"],
      "model": "model-name", // 可选
      "top_n": 3 // 可选
    }
    ```
-   **Logic**:
    -   从请求体中获取 `query` 和 `documents`。
    -   构造输入对：`[[query, doc1], [query, doc2], ...]`。
    -   调用 `model.predict(sentence_pairs)` 获取每个文档对的相关性分数。
    -   根据分数对文档进行排序，并可选择返回 `top_n` 个结果。
-   **Response Body**:
    ```json
    {
      "id": "rerank-id",
      "results": [
        {
          "index": 1,
          "relevance_score": 0.98,
          "document": { "text": "doc 2" }
        },
        {
          "index": 0,
          "relevance_score": 0.95,
          "document": { "text": "doc 1" }
        },
        {
          "index": 2,
          "relevance_score": 0.12,
          "document": { "text": "doc 3" }
        }
      ],
      "model": "model-name",
      "usage": { "total_tokens": 0 }
    }
    ```

## 6. `configs.yaml` 配置示例

通过在 `managed_workers` 列表中添加配置，iChat Gateway 可以自动启动和管理多个 `sentence_backend` 实例。

```yaml
# docker-iei/ichat/configs.yaml

# ... (server_settings)

managed_workers:
  # ... (other workers like vLLM)

  - model_name: bge-m3-embedding
    model_path: /mnt/models/bge-m3
    backend: sentence           # <--- 指定使用 sentence 后端
    gpu_ids: []                 # <--- CPU部署，留空或不填
    port: 8002
    # --- 以下为透传给 sentence_backend 的参数 ---
    task_type: embedding        # <--- 核心参数：指定任务类型
    device: cpu                 # <--- 核心参数：指定推理设备

  - model_name: bge-reranker-v2
    model_path: /mnt/models/bge-reranker-v2-m3
    backend: sentence
    gpu_ids: []
    port: 8003
    task_type: rerank           # <--- 核心参数：指定任务类型
    device: cpu
```

**注意**: 为了让 Gateway 能识别 `backend: sentence`，需要在 `ichat/worker/__main__.py` 中添加对 `SentenceBackend` 的导入和实例化逻辑。

## 7. `sentence_backend` 核心代码架构 (伪代码)

*本章节仅包含核心逻辑的伪代码，以保持文档的简洁性。*

```python
# docker-iei/ichat/backends/sentence_backend/sentence_backend.py

class SentenceBackend(BaseBackend):
    def __init__(...):
        # ...
        self.backend_args = self._parse_args(backend_argv)
        self.process = None
        # ...

    async def run(self):
        try:
            # Command to run api_server.py as a separate process
            cmd = ["python3", "-m", "ichat.backends.sentence_backend.api_server"]
            # ... convert self.backend_args to command line flags ...
            
            self.process = subprocess.Popen(cmd)

            await self._wait_and_warmup()
            
            health_check_task = asyncio.create_task(self._health_check_monitor())
            await health_check_task

        except asyncio.CancelledError:
            logger.info("SentenceBackend run task cancelled.")
        finally:
            self.cleanup()

    async def _wait_and_warmup(self):
        # ... poll http://host:port/health ...
        self.server_ready.set()

    async def _health_check_monitor(self):
        while self.process.poll() is None:
            await asyncio.sleep(5)
        raise RuntimeError("Sentence API server process has exited unexpectedly.")

    def cleanup(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            # ... handle graceful shutdown with timeout and SIGKILL ...
```

```python
# docker-iei/ichat/backends/sentence_backend/api_server.py

# 1. Parse args from args.py
args = parse_sentence_server_args()

# 2. Load model based on args.task_type
if args.task_type == "embedding":
    model = SentenceTransformer(args.model_path, device=args.device)
    # Add /v1/embeddings endpoint to app
elif args.task_type == "rerank":
    model = CrossEncoder(args.model_path, device=args.device)
    # Add /v1/rerank endpoint to app

# 3. Define FastAPI app and endpoints

@app.post("/v1/embeddings")
async def create_embeddings(...):
    # ... call model.encode() ...
    pass

@app.post("/v1/rerank")
async def create_rerank(...):
    # ... call model.predict() ...
    pass

# 4. Start uvicorn server
uvicorn.run(app, host=args.host, port=args.port)

```

