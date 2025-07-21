# iChat SGLang Backend 设计文档

## 1. 目标

`sglang_backend.py` 定义了 `SGLangBackend` 类，它是 iChat Worker 与 SGLang 推理引擎之间的核心适配层。其主要目标是：

1.  **封装 SGLang 服务**: 将 SGLang 的 `launch_server` 启动和管理逻辑封装成一个可编程的 Python 类。
2.  **独立的参数解析**: 接收来自 `serve.py` 的原始参数列表 (`backend_argv`)，并独立完成所有 SGLang 相关参数的解析，生成 `ServerArgs` 配置对象。
3.  **精细化生命周期管理**: 以编程方式对 SGLang 服务器（包括 Tokenizer、Scheduler、Detokenizer 等核心组件）的启动、运行和停止进行精细化控制，而不仅仅是简单地调用一个顶层函数。这为实现高级功能（如无缝重启、动态配置更新）奠定了基础。
4.  **无缝集成**: 使得 `serve.py` 无需关心 SGLang 的内部实现细节，只需实例化 `SGLangBackend` 并调用其 `run()` 方法即可。
5.  **支持未来扩展**: 为未来可能需要与 SGLang 服务器内部状态进行更深度交互的功能提供扩展点。

## 2. 设计原则

- **非侵入式 (Non-invasive)**: 这是最高原则。**即使进行了更深层次的集成，也绝不修改任何 SGLang 源代码**。所有集成工作都在 `SGLangBackend` 类中完成，通过调用和组合 SGLang 自身提供的函数和类（如 `_launch_subprocesses`）来实现。这确保了与 SGLang 版本的解耦，便于未来升级。

- **渐进式集成 (Progressive Integration)**: `SGLangBackend` 的设计遵循渐进式优化的思想。
    - **第一阶段（旧版实现）**: 最初的实现可以只调用 `sglang.srt.entrypoints.http_server.launch_server`，用最少的代码快速启动服务。
    - **当前阶段（优化后）**: 为了获得更强的控制力，我们“深入”一层，将 `launch_server` 的内部逻辑（包括引擎子进程启动、FastAPI 应用构建、Uvicorn 服务启动等）在 `SGLangBackend` 中重现。这种渐进式的开发方式，使得我们可以根据需求，逐步增强对底层服务的控制力，而无需一开始就实现最复杂的版本。

- **代码重用 (Code Reuse)**: 最大限度地重用 `sglang.srt.entrypoints.http_server.py` 中的函数，而不是通过 `subprocess` 来启动服务。这避免了进程间通信的复杂性，并提供了更好的控制和集成能力。

- **配置驱动**: `SGLangBackend` 的行为完全由传入的参数决定。所有 SGLang 支持的参数都应该能被 `serve.py` 接收并传递给该后端。

## 3. `SGLangBackend` 类架构

`SGLangBackend` 的架构旨在成为一个精细的服务控制器，而非简单的包装器。

### 3.1. `__init__(self, framework_args, backend_argv)`

构造函数负责初始化和参数解析。
-   **接收参数**: 接收来自 `serve.py` 的 `framework_args`（iChat框架解析后的参数）和 `backend_argv`（所有未被框架解析的原始参数列表）。
-   **调用解析器**: 调用 `self._parse_sglang_args(backend_argv)` 方法，将原始参数列表转换为 SGLang 所需的 `ServerArgs` 对象。
-   **状态初始化**: 初始化 `self.app`, `self.tokenizer_manager`, `self.server_task` 等实例变量，用于在服务生命周期中持有 FastAPI 应用、SGLang Tokenizer 管理器和 `asyncio.Task` 等关键组件的引用。

### 3.2. `_parse_sglang_args(self, backend_argv)` (新增)

这个新的私有方法是后端参数处理的核心，将参数解析逻辑完全封装在 `SGLangBackend` 内部。

-   **创建SGLang解析器**: 调用 SGLang 提供的 `ServerArgs.add_cli_args(parser)` 静态方法，来构建一个包含所有 SGLang 支持的命令行参数的 `ArgumentParser`。
-   **解析原始参数**: 使用创建的解析器来解析 `backend_argv` 列表。
-   **合并框架参数**: 将 `framework_args` 中的一些通用配置（如 `host`, `port`, `log_level`）合并到最终的参数中，确保框架层面的配置能够生效。
-   **生成配置对象**: 调用 `ServerArgs.from_cli_args()` 和 `check_server_args()` 来创建并验证最终的 `ServerArgs` 配置对象 (`self.sglang_args`)。


### 3.3. `async run(self)`

`run` 方法是启动和管理 SGLang 服务的核心入口点。它负责编排整个启动与关闭流程。
-   创建并运行一个 `asyncio.Task` 来执行 `self._run_server_worker()`，这是服务运行的主体。
-   `await` 这个 task，从而阻塞 `run` 方法，直到服务停止。
-   在 `finally` 块中调用 `self.cleanup()` 确保资源被正确释放。

### 3.4. `async _run_server_worker(self, ...)` (新增)

这是 SGLang 服务运行的核心工作函数，它重构了 `sglang.srt.entrypoints.http_server.launch_server` 的逻辑。
-   调用 `sglang.srt.entrypoints.engine._launch_subprocesses` 来启动引擎的各个组件（Tokenizer, Scheduler, Detokenizer）。
-   调用 `sglang.srt.entrypoints.http_server.set_global_state` 来设置 SGLang 运行所需的全局状态。
-   获取在 `http_server.py` 中定义的 FastAPI `app` 实例，并为其配置中间件。
-   调用 `self._serve_http()` 创建 Uvicorn 服务器的协程。
-   创建并 `await` 服务器协程，使其开始接受请求。
-   在服务启动后，异步执行 `_wait_and_warmup` 逻辑，以确保服务完全就绪。

### 3.5. `_serve_http(self, ...)` (新增)

该方法负责配置和创建 Uvicorn 服务器实例。
-   创建一个 `uvicorn.Config` 对象。
-   基于该配置创建一个 `uvicorn.Server` 对象。
-   **关键**: 通过 `server.install_signal_handlers = lambda: {}` 禁用了 Uvicorn 默认的信号处理器，这样 iChat 主进程就可以通过 `asyncio.Task.cancel()` 来控制服务的启停，而不是响应 `SIGINT` 或 `SIGTERM` 信号。
-   返回 `server.serve(...)` 协程。

### 3.6. `_wait_and_warmup(self, ...)` (新增)
该方法改编自 SGLang 的同名函数，负责在服务启动后进行健康检查和预热。
-   通过轮询 `/get_model_info` 端点来等待服务器就绪。
-   发送一个预热请求（`/generate` 或 `/encode`）来确保模型被加载并准备好处理流量。
-   通过 `asyncio.Event` 通知 `run` 方法，服务器已完全就绪。

### 3.7. `cleanup(self)` (新增)

负责在服务停止后进行优雅的资源清理。
-   取消仍在运行的 `server_task`。
-   调用 SGLang 提供的 `kill_process_tree` 来确保所有相关的子进程都被正确终止。

## 4. `sglang_backend.py` 优化后的代码架构

*本章节仅包含核心逻辑的伪代码，以保持文档的简洁性。完整的实现请直接参考 `ichat/backends/sglang_backend.py` 源文件。*

```python
# docker-iei/ichat/backends/sglang_backend.py

# --- Import necessary components from asyncio, sglang, fastapi, etc. ---
# ...

class SGLangBackend(BaseBackend):
    """
    An adapter class that provides fine-grained control over the SGLang server lifecycle.
    """

    def __init__(self, framework_args: Namespace, backend_argv: List[str]):
        """
        Initializes the backend and prepares SGLang arguments by parsing them internally.
        """
        super().__init__(framework_args, backend_argv)
        # Parse all SGLang specific arguments internally.
        self.sglang_args = self._parse_sglang_args(backend_argv)
        
        # Initialize placeholders for server components.
        self.app: Optional[FastAPI] = None
        self.tokenizer_manager: Optional[TokenizerManager] = None
        self.server_task: Optional[asyncio.Task] = None
        # ... other placeholders ...
        
    def _parse_sglang_args(self, backend_argv: List[str]) -> ServerArgs:
        """
        Creates a dedicated parser for SGLang, parses the backend-specific
        arguments, and merges them with framework-level settings.
        """
        # 1. Get the standard SGLang argument parser.
        parser = ArgumentParser()
        ServerArgs.add_cli_args(parser)

        # 2. Parse the raw argument list passed to the backend.
        parsed_args = parser.parse_args(backend_argv)

        # 3. Merge relevant arguments from the framework args.
        for key, value in vars(self.framework_args).items():
            if hasattr(parsed_args, key) and value is not None:
                setattr(parsed_args, key, value)
        
        # 4. Create and validate the final ServerArgs object.
        server_args = ServerArgs.from_cli_args(parsed_args)
        server_args.check_server_args()

        return server_args

    async def run(self):
        """
        Orchestrates the startup and shutdown of the SGLang server.
        """
        print("INFO:     Starting SGLang backend server...")
        try:
            # Create a task to run the main server worker.
            self.server_task = asyncio.create_task(
                self._run_server_worker()
            )
            await self.server_task
            
        except asyncio.CancelledError:
            print("INFO:     SGLang backend server task was cancelled.")
        except Exception as e:
            # Log and re-raise other exceptions.
            raise
        finally:
            # Ensure resources are cleaned up on exit.
            self.cleanup()

    async def _run_server_worker(self):
        """
        The core logic to build and run the SGLang API server worker.
        This is an adaptation of SGLang's `launch_server` function.
        """
        # 1. Launch SGLang engine subprocesses (Tokenizer, Scheduler, Detokenizer).
        self.tokenizer_manager, scheduler_info = _launch_subprocesses(
            server_args=self.sglang_args
        )
        
        # 2. Set the global state required by SGLang's API endpoints.
        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                scheduler_info=scheduler_info,
            )
        )
        
        # 3. Get and configure the FastAPI app.
        from sglang.srt.entrypoints.http_server import app, lifespan
        self.app = app
        self.app.server_args = self.sglang_args
        # ... (Add middleware for API key, metrics, etc.) ...
        
        print(f"INFO:     Starting SGLang server on {self.sglang_args.host}:{self.sglang_args.port}")
        
        # 4. Create and run the Uvicorn server coroutine.
        server_coro = self._serve_http()
        
        # 5. Concurrently run the server and the warmup process.
        await asyncio.gather(
            server_coro,
            self._wait_and_warmup()
        )

    def _serve_http(self) -> Coroutine:
        """
        Configures and creates the Uvicorn server instance as a coroutine.
        """
        # Create a uvicorn.Config object with all necessary settings.
        config = uvicorn.Config(
            self.app,
            host=self.sglang_args.host,
            port=self.sglang_args.port,
            # ... other configs ...
        )
        
        server = uvicorn.Server(config)
        
        # KEY STEP: Disable default signal handlers to allow programmatic control.
        server.install_signal_handlers = lambda: {}
        
        # Return the server's 'serve' coroutine.
        return server.serve()
        
    async def _wait_and_warmup(self):
        """
        Waits for the server to be healthy and then sends a warmup request.
        """
        # 1. Wait for the http server to be ready by polling an endpoint.
        # ... (Implementation omitted for brevity) ...

        # 2. Send a warmup /generate or /encode request.
        # ... (Implementation omitted for brevity) ...

        print("INFO:     SGLang server is warmed up and ready to roll!")


    def cleanup(self):
        """
        Gracefully cleans up all server resources.
        """
        print("INFO:     Cleaning up SGLang backend resources...")
        # 1. Cancel the main server task if it's still running.
        if self.server_task and not self.server_task.done():
            self.server_task.cancel()
        
        # 2. Terminate the process tree to ensure all SGLang subprocesses are closed.
        kill_process_tree(os.getpid(), include_parent=False)
```
