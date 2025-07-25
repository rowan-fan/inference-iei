# iChat Backends 设计文档

本文档旨在详细阐述 iChat `backends` 模块的设计理念、架构和具体实现。该模块是 iChat 与底层大语言模型（LLM）推理引擎之间的核心桥梁。

## 1. 概述

iChat 的 `backends` 目录负责封装和管理底层的大语言模型推理引擎。每个后端都是一个独立的适配器，它将特定推理引擎（如 vLLM, SGLang）的启动、配置和生命周期管理逻辑封装成一个统一的、可编程的接口。

这种设计的核心优势在于：

- **解耦**: iChat 的主服务 (`serve.py`) 无需关心底层推理引擎的具体实现细节。它只需根据配置选择并实例化对应的后端，然后调用其 `run()` 方法。
- **可扩展性**: 添加对新推理引擎的支持变得非常简单，只需实现一个新的后端类，遵循 `BaseBackend` 定义的接口即可。
- **精细化控制**: 后端类不仅仅是启动一个子进程，而是深入到推理引擎的内部启动逻辑，以编程方式控制其生命周期。这使得实现健康检查、无缝重启、动态配置等高级功能成为可能。

## 2. `BaseBackend` 接口

所有后端都继承自 `BaseBackend` 类 (`ichat/backends/base_backend.py`)，该类定义了后端必须实现的核心接口，确保了所有后端行为的一致性。

- `__init__(self, framework_args, backend_argv, backend_ready_event)`: 构造函数接收 iChat 框架解析后的参数 (`framework_args`)、所有未被框架解析的原始参数列表 (`backend_argv`)，以及一个 `asyncio.Event` (`backend_ready_event`) 用于通知上层服务后端已就绪。
- `async run(self)`: 启动和管理后端服务的主入口点。这是一个阻塞方法，会一直运行直到服务停止。
- `cleanup(self)`: 在服务停止后，负责清理后端使用的所有资源（如子进程、网络套接字等）。
- `get_backend_args(self)`: 返回后端经过内部解析和处理后的最终参数对象。
- `wait_for_server_ready(self)`: 一个 `async` 方法，它会 `await` `backend_ready_event`，直到后端完成初始化、模型加载和预热。

---

## 3. VLLM Backend (`vllm_backend.py`)

`vllm_backend.py` 定义了 `VLLMBackend` 类，它是 iChat Worker 与 vLLM 推理引擎之间的核心适配层。

### 3.1. 目标

1.  **封装 vLLM 服务**: 将 vLLM 的 `api_server.py` 启动和管理逻辑封装成一个可编程的 Python 类。
2.  **独立的参数解析与转换**: 接收来自 `serve.py` 的原始参数列表 (`backend_argv`)，并独立完成所有 vLLM 相关参数的解析。这包括将 iChat 的统一参数名（如 `--model-path`）转换为 vLLM `api_server` 所能理解的参数格式（如 `--model`）。
3.  **精细化生命周期管理**: 以编程方式对 vLLM 服务器的启动、运行和停止进行精细化控制，而不仅仅是简单地调用一个顶层函数。这为实现高级功能（如无缝重启、动态配置更新）奠定了基础。
4.  **实现服务就绪通知**: 集成 `BaseBackend` 的 `server_ready` 事件，在 vLLM 服务器完成启动和预热后，精确地通知上层应用服务已可用。
5.  **增加服务预热 (Warmup)**: 在服务器正式对外服务前，通过健康检查和可选的预热请求，确保模型已加载并准备好处理请求，减少首个请求的延迟。
6.  **无缝集成**: 使得 `serve.py` 无需关心 vLLM 的内部实现细节，只需实例化 `VLLMBackend` 并调用其 `run()` 方法即可。
7.  **支持未来扩展**: 为未来可能需要与 vLLM 服务器内部状态（如模型配置、引擎状态）进行更深度交互的功能提供扩展点。

### 3.2. 设计原则

- **非侵入式 (Non-invasive)**: 这是最高原则。**即使进行了更深层次的集成，也绝不修改任何 vLLM 源代码**。所有集成工作都在 `VLLMBackend` 类中完成，通过调用和组合 vLLM 自身提供的函数和类（如 `build_app`, `build_async_engine_client`）来实现。这确保了与 vLLM 版本的解耦，便于未来升级。

- **渐进式集成 (Progressive Integration)**: `VLLMBackend` 的设计遵循渐进式优化的思想。
    - **第一阶段（旧版实现）**: 最初的实现可以只调用 `vllm.entrypoints.openai.api_server.run` (或其前身)，用最少的代码快速启动服务。
    - **当前阶段（优化后）**: 为了获得更强的控制力，我们“深入”一层，将 `run` 的内部逻辑（包括Socket创建、FastAPI应用构建、Uvicorn服务启动等）在 `VLLMBackend` 中重现。这种渐进式的开发方式，使得我们可以根据需求，逐步增强对底层服务的控制力，而无需一开始就实现最复杂的版本。

- **代码重用 (Code Reuse)**: 最大限度地重用 `vllm.entrypoints.openai.api_server.py` 中的函数，而不是通过 `subprocess` 来启动服务。这避免了进程间通信的复杂性，并提供了更好的控制和集成能力。

- **配置驱动**: `VLLMBackend` 的行为完全由传入的参数决定。所有 vLLM 支持的参数都应该能被 `serve.py` 接收并传递给该后端。

### 3.3. `VLLMBackend` 类架构

`VLLMBackend` 的架构已经从一个简单的包装器演变为一个精细的服务控制器。

#### 3.3.1. `__init__(self, framework_args, backend_argv, backend_ready_event)`

构造函数负责初始化和参数转换。
-   **接收参数**: 接收来自 `serve.py` 的 `framework_args`（iChat框架解析后的参数）和 `backend_argv`（所有未被框架解析的原始参数列表）。
-   **调用解析器**: 调用新的 `self._parse_vllm_args(backend_argv)` 方法，将原始参数列表转换为 vLLM 所需的 `Namespace` 对象。
-   **状态初始化**: 初始化 `self.app`, `self.engine_client`, `self.sock` 等实例变量，用于在服务生命周期中持有 FastAPI 应用、vLLM 引擎客户端、服务器套接字等关键组件的引用。

#### 3.3.2. `_parse_vllm_args(self, backend_argv)`

这个新的私有方法是后端参数处理的核心。它将参数解析的逻辑从 `serve.py` 完全移交给了 `VLLMBackend`。

-   **创建vLLM解析器**: 调用 vLLM 自己的 `make_arg_parser()` 函数来创建一个标准的 `ArgumentParser`，该解析器包含了所有 vLLM 支持的命令行参数。
-   **解析原始参数**: 使用创建的解析器来解析 `backend_argv` 列表。
-   **参数名映射**: 为了保持与 iChat 通用配置的兼容性，该方法会检查是否存在 iChat 的标准参数，并将其映射到 vLLM 的对应参数上。
-   **合并框架参数**: 将 `framework_args` 中的一些通用配置（如 `host`, `port`）合并到最终的 `vllm_args` 中，确保框架层面的配置能够生效。
-   **返回结果**: 返回一个完全配置好的 `Namespace` 对象 (`self.vllm_args`)，该对象可直接被 vLLM 的其他函数使用。

##### 参数映射关系

为了提供一致的用户体验，`VLLMBackend` 内部维护了以下从 iChat 标准参数到 vLLM 原生参数的映射关系：

| iChat 标准参数      | vLLM 对应参数     | 描述                           |
| :------------------ | :---------------- | :----------------------------- |
| `--model-path`      | `--model`         | 模型权重的路径或HuggingFace ID |
| `--tokenizer-path`  | `--tokenizer`     | Tokenizer的路径或ID            |
| `--context-length`  | `--max-model-len` | 模型的最大上下文长度           |
| `--served-model-name`| `--served-model-name`| 在Gateway中注册的模型名称 |
| `--trust-remote-code`| `--trust-remote-code`| 是否信任远程代码 |
| `--host` | `--host` | 服务监听的主机地址 |
| `--port` | `--port` | 服务监听的端口 |

#### 3.3.3. `async run(self)`

`run` 方法是启动和管理 vLLM 服务的核心入口点。它现在负责编排整个启动流程。
-   调用 `self._setup_server()` 来准备服务器环境并创建监听套接字。
-   创建并运行一个 `asyncio.Task` 来执行 `self._run_server_worker()`，这是服务运行的主体。
-   `await` 这个 task，从而阻塞 `run` 方法，直到服务停止。
-   在 `finally` 块中调用 `self.cleanup()` 确保资源被正确释放。

#### 3.3.4. `_setup_server(self)`

这个私有方法负责服务器启动前的准备工作，它重现了 `vllm.entrypoints.openai.api_server` 中部分启动逻辑。
-   验证 API 服务器参数 (调用 `validate_parsed_serve_args`)。
-   创建并绑定服务器 `socket`。
-   设置系统资源限制 (`ulimit`)。
-   返回监听地址和创建好的 `socket` 对象。

#### 3.3.5. `async _run_server_worker(self, ...)`

这是 vLLM 服务运行的核心工作函数，改编自 vLLM 的同名函数。它负责编排服务器的启动、预热和健康监控。
-   使用 `build_async_engine_client` 上下文管理器来创建和管理 vLLM 引擎客户端。
-   调用 `build_app` 创建 FastAPI 应用实例。
-   调用 `init_app_state` 将引擎客户端、配置等状态信息填充到 FastAPI 应用中。
-   **任务编排**:
    1.  创建并启动 Uvicorn 服务器任务 (`server_task`)。
    2.  `await` `self._wait_and_warmup()`，等待服务器完成模型加载并响应健康检查。这确保了服务在进入下一步前已基本就绪。
    3.  **在预热成功后**，创建并启动持续健康检查任务 (`health_check_task`)。
    4.  使用 `asyncio.wait` 同时监控 `server_task` 和 `health_check_task`。任何一个任务的意外退出（例如，引擎崩溃或服务无响应）都会导致另一个任务被取消，从而触发整个后端的优雅关闭。

#### 3.3.6. `_serve_http(self, ...)`

该方法负责配置和创建 Uvicorn 服务器实例。
-   创建一个 `uvicorn.Config` 对象。
-   基于该配置创建一个 `uvicorn.Server` 对象。
-   **关键**: 通过 `server.install_signal_handlers = lambda: {}` 禁用了 Uvicorn 默认的信号处理器，这样 iChat 主进程就可以通过 `asyncio.Task.cancel()` 来控制服务的启停，而不是响应 `SIGINT` 或 `SIGTERM` 信号。
-   **新增**: 保存 `server.should_exit` 事件到 `self.server_shutdown_event`，用于后续的优雅关停。
-   返回 `server.serve(...)` 协程。

#### 3.3.7. `async _wait_and_warmup(self)`

该方法负责**等待 vLLM 服务器完成初始加载**。对于大型模型，这个过程可能需要数分钟。
-   **等待服务启动**: 在一个无限循环中，使用 `aiohttp` 异步地、定期（每5秒）尝试访问 vLLM 服务的 `/health` 健康检查接口。
-   **成功返回**: 一旦 `/health` 接口返回 200 状态码，意味着模型已加载，服务器已准备好接收请求。此时，方法将打印成功信息并正常返回。
-   **持续等待**: 如果连接失败或返回非 200 状态码，它会静默地继续等待，直到服务就绪。这取代了旧的固定超时机制，以适应不同规模模型的加载时间。
-   **通知就绪**: 在成功返回前，调用 `self.server_ready.set()` 通知 iChat 框架服务已就绪。

#### 3.3.8. `async _health_check_monitor(self)`

这个协程在服务器**初始预热成功后**启动，负责对 vLLM 引擎进行**持续的健康监控**。
-   **启动时机**: 在 `_wait_and_warmup` 成功返回后才启动，避免在模型加载期间进行不必要的检查。
-   **移除了初始等待**: 之前版本中固定的15秒等待 (`asyncio.sleep(15)`) 已被移除，监控会立即开始。
-   **监控机制**: 在一个无限循环中，每 5 秒执行一次检查。
-   **RPC 存活检查**: 通过调用 `self.engine_client.is_sleeping()` 方法进行 RPC 调用。这是一个轻量级的检查，用于确认 vLLM 的工作进程是否仍在运行且能够响应请求。
-   **超时处理**: 如果 RPC 调用在 10 秒内没有响应 (`asyncio.TimeoutError`)，则认为引擎已无响应。
-   **异常处理**: 任何来自存活检查的异常（包括超时）都会被捕获，并作为 `RuntimeError` 重新抛出。这会触发 `_run_server_worker` 中的 `asyncio.wait` 机制，导致整个服务关闭和重启。
-   **日志降噪**: 为了避免在正常运行时产生过多日志，健康检查成功的 `print` 语句已被注释掉。

#### 3.3.9. `cleanup(self)`

负责在服务停止后进行优雅的资源清理。
-   **触发关闭事件**: 如果 `self.server_shutdown_event` (即 `uvicorn.Server.should_exit`) 存在且未被设置，则调用 `.set()` 来通知 Uvicorn 服务器开始关闭流程。
-   关闭服务器 `socket`。
-   取消仍在运行的 `server_task`，以防万一。

### 3.4. `vllm_backend.py` 核心代码架构

*本章节仅包含核心逻辑的伪代码，以保持文档的简洁性。完整的实现请直接参考 `ichat/backends/vllm_backend.py` 源文件。*

```python
# docker-iei/ichat/backends/vllm_backend.py
class VLLMBackend(BaseBackend):
    def __init__(self, framework_args: Namespace, backend_argv: List[str], backend_ready_event: asyncio.Event):
        # ...
        self.vllm_args = self._parse_vllm_args(backend_argv)
        # ...

    def _parse_vllm_args(self, backend_argv: List[str]) -> Namespace:
        # 1. Get the standard vLLM argument parser.
        # 2. Parse the raw argument list passed to the backend.
        # 3. Apply mappings from iChat's unified args to vLLM's native args.
        # 4. Merge relevant arguments from the framework args.
        pass

    async def run(self):
        try:
            listen_address, self.sock = self._setup_server()
            self.server_task = asyncio.create_task(
                self._run_server_worker(listen_address, self.sock)
            )
            await self.server_task
        finally:
            self.cleanup()

    async def _run_server_worker(self, listen_address: str, sock: socket.socket):
        async with build_async_engine_client(self.vllm_args) as engine_client:
            # ... setup app and state ...
            
            # 1. Create a task for the Uvicorn server.
            server_task = asyncio.create_task(self._serve_http(sock=sock))
            
            # 2. Wait for the model to load and the server to be ready.
            await self._wait_and_warmup()
            
            # 3. Once ready, start the continuous health check monitor.
            health_check_task = asyncio.create_task(self._health_check_monitor())
            
            # 4. Monitor both tasks. If one fails, the other is cancelled.
            done, pending = await asyncio.wait(
                {server_task, health_check_task}, 
                return_when=asyncio.FIRST_COMPLETED
            )
            # ... exception handling and cleanup ...

    async def _wait_and_warmup(self):
        # ... poll /health endpoint ...
        self.server_ready.set()
        return

    async def _health_check_monitor(self):
        # ... periodically call self.engine_client.is_sleeping() ...
        pass

    def cleanup(self):
        # ... set shutdown event, close socket, cancel task ...
        pass
```

---

## 4. SGLang Backend (`sglang_backend.py`)

`sglang_backend.py` 定义了 `SGLangBackend` 类，它是 iChat Worker 与 SGLang 推理引擎之间的核心适配层。

### 4.1. 目标

1.  **封装 SGLang 服务**: 将 SGLang 的 `launch_server` 启动和管理逻辑封装成一个可编程的 Python 类。
2.  **独立的参数解析**: 接收来自 `serve.py` 的原始参数列表 (`backend_argv`)，并独立完成所有 SGLang 相关参数的解析，生成 `ServerArgs` 配置对象。
3.  **精细化生命周期管理**: 以编程方式对 SGLang 服务器（包括 Tokenizer、Scheduler、Detokenizer 等核心组件）的启动、运行和停止进行精si化控制，而不仅仅是简单地调用一个顶层函数。这为实现高级功能（如无缝重启、动态配置更新）奠定了基础。
4.  **无缝集成**: 使得 `serve.py` 无需关心 SGLang 的内部实现细节，只需实例化 `SGLangBackend` 并调用其 `run()` 方法即可。
5.  **鲁棒性与监控**: 内置了对 SGLang 核心子进程的健康检查，确保在关键组件失效时能够快速失败并重启，提高了服务的稳定性。

### 4.2. 设计原则

- **非侵入式 (Non-invasive)**: 这是最高原则。**即使进行了更深层次的集成，也绝不修改任何 SGLang 源代码**。所有集成工作都在 `SGLangBackend` 类中完成，通过调用和组合 SGLang 自身提供的函数和类（如 `_launch_subprocesses`）来实现。这确保了与 SGLang 版本的解耦，便于未来升级。

- **渐进式集成 (Progressive Integration)**: `SGLangBackend` 的设计遵循渐进式优化的思想。
    - **第一阶段（旧版实现）**: 最初的实现可以只调用 `sglang.srt.entrypoints.http_server.launch_server`，用最少的代码快速启动服务。
    - **当前阶段（优化后）**: 为了获得更强的控制力，我们“深入”一层，将 `launch_server` 的内部逻辑（包括引擎子进程启动、FastAPI 应用构建、Uvicorn 服务启动等）在 `SGLangBackend` 中重现。这种渐进式的开发方式，使得我们可以根据需求，逐步增强对底层服务的控制力，而无需一开始就实现最复杂的版本。

- **代码重用 (Code Reuse)**: 最大限度地重用 `sglang.srt.entrypoints.http_server.py` 中的函数，而不是通过 `subprocess` 来启动服务。这避免了进程间通信的复杂性，并提供了更好的控制和集成能力。

- **配置驱动**: `SGLangBackend` 的行为完全由传入的参数决定。所有 SGLang 支持的参数都应该能被 `serve.py` 接收并传递给该后端。

### 4.3. `SGLangBackend` 类架构

`SGLangBackend` 的架构旨在成为一个精细的服务控制器，而非简单的包装器。

#### 4.3.1. `__init__(self, framework_args, backend_argv)`

构造函数负责初始化和参数解析。
-   **接收参数**: 接收来自 `serve.py` 的 `framework_args`（iChat框架解析后的参数）和 `backend_argv`（所有未被框架解析的原始参数列表）。
-   **调用解析器**: 调用 `self._parse_sglang_args(backend_argv)` 方法，将原始参数列表转换为 SGLang 所需的 `ServerArgs` 对象。
-   **状态初始化**: 初始化 `self.app`, `self.tokenizer_manager`, `self.server_task`, `self.server_shutdown_event`, `self.subprocesses` 等实例变量，用于在服务生命周期中持有 FastAPI 应用、SGLang 组件、`asyncio.Task`、Uvicorn 关闭事件以及子进程列表等关键组件的引用。

#### 4.3.2. `_parse_sglang_args(self, backend_argv)`

这个私有方法是后端参数处理的核心，将参数解析逻辑完全封装在 `SGLangBackend` 内部。

-   **创建SGLang解析器**: 调用 SGLang 提供的 `ServerArgs.add_cli_args(parser)` 静态方法，来构建一个包含所有 SGLang 支持的命令行参数的 `ArgumentParser`。
-   **解析原始参数**: 使用创建的解析器来解析 `backend_argv` 列表。
-   **合并框架参数**: 将 `framework_args` 中的一些通用配置（如 `host`, `port`, `log_level`）合并到最终的参数中，确保框架层面的配置能够生效。
-   **生成配置对象**: 调用 `ServerArgs.from_cli_args()` 和 `check_server_args()` 来创建并验证最终的 `ServerArgs` 配置对象 (`self.sglang_args`)。

#### 4.3.3. `async run(self)`

`run` 方法是启动和管理 SGLang 服务的核心入口点。它负责编排整个启动与关闭流程。
-   创建并运行一个 `asyncio.Task` 来执行 `self._run_server_worker()`，这是服务运行的主体。
-   `await` 这个 task，从而阻塞 `run` 方法，直到服务停止。
-   在 `finally` 块中调用 `self.cleanup()` 确保资源被正确释放。

#### 4.3.4. `async _run_server_worker(self)`

这是 SGLang 服务运行的核心工作函数，它重构并增强了 `sglang.srt.entrypoints.http_server.launch_server` 的逻辑。
-   **子进程监控**: 在启动 SGLang 引擎前记录当前子进程列表，启动后再记录一次，通过对比两次列表来精确识别并存储 SGLang 引擎的子进程 (`self.subprocesses`)，以便后续进行健康检查。
-   **启动引擎**: 调用 `sglang.srt.entrypoints.engine._launch_subprocesses` 来启动引擎的各个组件（Tokenizer, Scheduler, Detokenizer）。
-   **设置全局状态**: 调用 `sglang.srt.entrypoints.http_server.set_global_state` 来设置 SGLang API 端点运行所需的全局状态。
-   **配置FastAPI**: 获取在 `http_server.py` 中定义的 FastAPI `app` 实例，并为其配置 API Key 中间件等。
-   **任务编排**:
    1.  创建 `server_task`，用于运行 `self._serve_http()` 返回的 Uvicorn 服务器协程。
    2.  `await self._wait_and_warmup()` 等待服务器启动并完成预热。如果预热失败，它将抛出异常，`run` 方法会捕获该异常并终止启动流程。
    3.  预热成功后，创建 `health_check_task`，用于运行 `self._health_check_monitor()` 进行持续的子进程健康检查。
    4.  使用 `asyncio.wait` 同时监控 `server_task` 和 `health_check_task`。`return_when=asyncio.FIRST_COMPLETED` 确保任何一个任务首先完成（无论是正常结束还是异常退出），`_run_server_worker` 都会继续执行。
    5.  检查已完成任务的结果，如果存在异常，则重新抛出，从而触发整个后端的关闭和清理逻辑。
-   **优雅关闭**: 在 `finally` 块中，取消所有仍在运行的待处理任务，确保在退出时不会有挂起的协程。

#### 4.3.5. `_serve_http(self)`

该方法负责配置和创建 Uvicorn 服务器实例。
-   创建一个 `uvicorn.Config` 对象。
-   基于该配置创建一个 `uvicorn.Server` 对象。
-   **获取关闭事件**: `self.server_shutdown_event = server.should_exit` 获取 Uvicorn 内部用于触发关闭的 `asyncio.Event`。这使得 `cleanup` 方法可以从外部命令 Uvicorn 服务器关闭。
-   **禁用信号处理**: 通过 `server.install_signal_handlers = lambda: {}` 禁用了 Uvicorn 默认的信号处理器。这确保了 iChat 主进程可以通过 `asyncio.Task.cancel()` 和设置 `server_shutdown_event` 来完全控制服务的启停，而不是被 `SIGINT` 或 `SIGTERM` 信号中断。
-   返回 `server.serve()` 协程。

#### 4.3.6. `_wait_and_warmup(self)`

该方法改编自 SGLang 的同名函数，负责在服务启动后进行健康检查和预热。
-   通过轮询 `/get_model_info` 端点来等待服务器就绪。
-   发送一个预热请求（`/generate` 或 `/encode`）来确保模型被加载并准备好处理流量。
-   预热成功后调用 `self.server_ready.set()`，通过 `asyncio.Event` 通知 iChat 框架，服务器已完全就绪。

#### 4.3.7. `_health_check_monitor(self)`

此方法提供了一种比 HTTP 健康检查更强大的服务监控机制。
- **直接监控子进程**: 它不依赖于网络端点，而是直接、定期地（每5秒）检查 `self.subprocesses` 列表中的每个 SGLang 子进程的状态。
- **快速失败**: 通过调用 `proc.is_running()`，它可以立即检测到是否有任何关键子进程（如 Scheduler 或 Detokenizer）已经崩溃或退出。
- **触发关闭**: 一旦检测到有子进程终止，它会记录一条致命错误日志，并抛出一个 `RuntimeError`。这个异常会被 `_run_server_worker` 中的 `asyncio.wait` 捕获，从而立即触发整个后端的关闭和清理流程，确保服务不会在部分组件失效的情况下继续运行。

#### 4.3.8. `cleanup(self)`

负责在服务停止后进行优雅的资源清理。
- **请求Uvicorn关闭**: 首先检查 `self.server_shutdown_event` 是否存在并且尚未被设置。如果服务器仍在运行，就调用 `self.server_shutdown_event.set()` 来请求 Uvicorn 服务器优雅地停止。
- **取消主任务**: 接着，取消 `self.server_task` 以确保 `_run_server_worker` 协程能够退出。
- **强制清理**: 最后，在 `finally` 块中，调用 SGLang 提供的 `kill_process_tree` 来确保所有 SGLang 的子进程都被彻底终止，防止出现僵尸进程。这个调用是保证资源完全释放的最后一道防线。

### 4.4. `sglang_backend.py` 核心代码架构

*本章节仅包含核心逻辑的伪代码，以保持文档的简洁性。完整的实现请直接参考 `ichat/backends/sglang_backend.py` 源文件。*

```python
# docker-iei/ichat/backends/sglang_backend.py
class SGLangBackend(BaseBackend):
    def __init__(self, framework_args: Namespace, backend_argv: List[str], backend_ready_event: asyncio.Event):
        # ...
        self.sglang_args: ServerArgs = self._parse_sglang_args(backend_argv)
        # ...

    def _parse_sglang_args(self, backend_argv: List[str]) -> ServerArgs:
        # 1. Create parser
        # 2. Parse argv
        # 3. Merge framework args
        # 4. Return ServerArgs instance
        pass

    async def run(self):
        try:
            self.server_task = asyncio.create_task(self._run_server_worker())
            await self.server_task
        finally:
            self.cleanup()

    async def _run_server_worker(self):
        # ... Detect subprocesses before and after launch ...
        self.subprocesses = [p for p in post_launch_children if p not in pre_launch_children]

        # ... Launch engine and set global state ...

        server_task = asyncio.create_task(self._serve_http())
        
        await self._wait_and_warmup()

        health_check_task = asyncio.create_task(self._health_check_monitor())

        # ... await asyncio.wait on server_task and health_check_task ...
    
    async def _wait_and_warmup(self):
        # 1. Wait for http server to be ready by polling /get_model_info
        # 2. Send a warmup /generate or /encode request.
        self.server_ready.set()

    async def _health_check_monitor(self):
        while True:
            await asyncio.sleep(5)
            for proc in self.subprocesses:
                if not proc.is_running():
                    raise RuntimeError("A critical SGLang subprocess has failed.")

    def cleanup(self):
        # ... set shutdown event, cancel task ...
        kill_process_tree(os.getpid(), include_parent=False)

```
