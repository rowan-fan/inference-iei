# iChat SGLang Backend 设计文档

## 1. 目标

`sglang_backend.py` 定义了 `SGLangBackend` 类，它是 iChat Worker 与 SGLang 推理引擎之间的核心适配层。其主要目标是：

1.  **封装 SGLang 服务**: 将 SGLang 的 `launch_server` 启动和管理逻辑封装成一个可编程的 Python 类。
2.  **独立的参数解析**: 接收来自 `serve.py` 的原始参数列表 (`backend_argv`)，并独立完成所有 SGLang 相关参数的解析，生成 `ServerArgs` 配置对象。
3.  **精细化生命周期管理**: 以编程方式对 SGLang 服务器（包括 Tokenizer、Scheduler、Detokenizer 等核心组件）的启动、运行和停止进行精细化控制，而不仅仅是简单地调用一个顶层函数。这为实现高级功能（如无缝重启、动态配置更新）奠定了基础。
4.  **无缝集成**: 使得 `serve.py` 无需关心 SGLang 的内部实现细节，只需实例化 `SGLangBackend` 并调用其 `run()` 方法即可。
5.  **鲁棒性与监控**: 内置了对 SGLang 核心子进程的健康检查，确保在关键组件失效时能够快速失败并重启，提高了服务的稳定性。

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
-   **状态初始化**: 初始化 `self.app`, `self.tokenizer_manager`, `self.server_task`, `self.server_shutdown_event`, `self.subprocesses` 等实例变量，用于在服务生命周期中持有 FastAPI 应用、SGLang 组件、`asyncio.Task`、Uvicorn 关闭事件以及子进程列表等关键组件的引用。

### 3.2. `_parse_sglang_args(self, backend_argv)`

这个私有方法是后端参数处理的核心，将参数解析逻辑完全封装在 `SGLangBackend` 内部。

-   **创建SGLang解析器**: 调用 SGLang 提供的 `ServerArgs.add_cli_args(parser)` 静态方法，来构建一个包含所有 SGLang 支持的命令行参数的 `ArgumentParser`。
-   **解析原始参数**: 使用创建的解析器来解析 `backend_argv` 列表。
-   **合并框架参数**: 将 `framework_args` 中的一些通用配置（如 `host`, `port`, `log_level`）合并到最终的参数中，确保框架层面的配置能够生效。
-   **生成配置对象**: 调用 `ServerArgs.from_cli_args()` 和 `check_server_args()` 来创建并验证最终的 `ServerArgs` 配置对象 (`self.sglang_args`)。


### 3.3. `async run(self)`

`run` 方法是启动和管理 SGLang 服务的核心入口点。它负责编排整个启动与关闭流程。
-   创建并运行一个 `asyncio.Task` 来执行 `self._run_server_worker()`，这是服务运行的主体。
-   `await` 这个 task，从而阻塞 `run` 方法，直到服务停止。
-   在 `finally` 块中调用 `self.cleanup()` 确保资源被正确释放。

### 3.4. `async _run_server_worker(self)`

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

### 3.5. `_serve_http(self)`

该方法负责配置和创建 Uvicorn 服务器实例。
-   创建一个 `uvicorn.Config` 对象。
-   基于该配置创建一个 `uvicorn.Server` 对象。
-   **获取关闭事件**: `self.server_shutdown_event = server.should_exit` 获取 Uvicorn 内部用于触发关闭的 `asyncio.Event`。这使得 `cleanup` 方法可以从外部命令 Uvicorn 服务器关闭。
-   **禁用信号处理**: 通过 `server.install_signal_handlers = lambda: {}` 禁用了 Uvicorn 默认的信号处理器。这确保了 iChat 主进程可以通过 `asyncio.Task.cancel()` 和设置 `server_shutdown_event` 来完全控制服务的启停，而不是被 `SIGINT` 或 `SIGTERM` 信号中断。
-   返回 `server.serve()` 协程。

### 3.6. `_wait_and_warmup(self)`
该方法改编自 SGLang 的同名函数，负责在服务启动后进行健康检查和预热。
-   通过轮询 `/get_model_info` 端点来等待服务器就绪。
-   发送一个预热请求（`/generate` 或 `/encode`）来确保模型被加载并准备好处理流量。
-   预热成功后调用 `self.server_ready.set()`，通过 `asyncio.Event` 通知 iChat 框架，服务器已完全就绪。

### 3.7. `_health_check_monitor(self)` (新增)
此方法提供了一种比 HTTP 健康检查更强大的服务监控机制。
- **直接监控子进程**: 它不依赖于网络端点，而是直接、定期地（每5秒）检查 `self.subprocesses` 列表中的每个 SGLang 子进程的状态。
- **快速失败**: 通过调用 `proc.is_running()`，它可以立即检测到是否有任何关键子进程（如 Scheduler 或 Detokenizer）已经崩溃或退出。
- **触发关闭**: 一旦检测到有子进程终止，它会记录一条致命错误日志，并抛出一个 `RuntimeError`。这个异常会被 `_run_server_worker` 中的 `asyncio.wait` 捕获，从而立即触发整个后端的关闭和清理流程，确保服务不会在部分组件失效的情况下继续运行。

### 3.8. `cleanup(self)`
负责在服务停止后进行优雅的资源清理。
- **请求Uvicorn关闭**: 首先检查 `self.server_shutdown_event` 是否存在并且尚未被设置。如果服务器仍在运行，就调用 `self.server_shutdown_event.set()` 来请求 Uvicorn 服务器优雅地停止。
- **取消主任务**: 接着，取消 `self.server_task` 以确保 `_run_server_worker` 协程能够退出。
- **强制清理**: 最后，在 `finally` 块中，调用 SGLang 提供的 `kill_process_tree` 来确保所有 SGLang 的子进程都被彻底终止，防止出现僵尸进程。这个调用是保证资源完全释放的最后一道防线。

## 4. `sglang_backend.py` 优化后的代码架构

*本章节仅包含核心逻辑的伪代码，以保持文档的简洁性。完整的实现请直接参考 `ichat/backends/sglang_backend.py` 源文件。*

```python
# docker-iei/ichat/backends/sglang_backend.py
import asyncio
import os
import psutil
from typing import Optional, List
# --- Import SGLang components ---

class SGLangBackend(BaseBackend):
    def __init__(self, framework_args: Namespace, backend_argv: List[str]):
        super().__init__(framework_args, backend_argv)
        self.sglang_args: ServerArgs = self._parse_sglang_args(backend_argv)
        
        self.app = None
        self.tokenizer_manager = None
        self.server_task: Optional[asyncio.Task] = None
        self.server_shutdown_event: Optional[asyncio.Event] = None
        self.subprocesses: List[psutil.Process] = []

    def _parse_sglang_args(self, backend_argv: List[str]) -> ServerArgs:
        # ... (Implementation is mostly unchanged) ...
        # 1. Create parser
        # 2. Parse argv
        # 3. Merge framework args
        # 4. Return ServerArgs instance
        pass

    async def run(self):
        logger.info("Starting SGLang backend server...")
        try:
            self.server_task = asyncio.create_task(self._run_server_worker())
            await self.server_task
        except asyncio.CancelledError:
            logger.info("SGLang backend server task was cancelled.")
        except Exception:
            logger.error(f"SGLang backend server failed: {get_exception_traceback()}")
            raise
        finally:
            self.cleanup()

    async def _run_server_worker(self):
        pre_launch_children = psutil.Process(os.getpid()).children()

        (
            tokenizer_manager,
            template_manager,
            scheduler_info,
            *_,
        ) = _launch_subprocesses(server_args=self.sglang_args)
        self.tokenizer_manager = tokenizer_manager

        post_launch_children = psutil.Process(os.getpid()).children()
        self.subprocesses = [p for p in post_launch_children if p not in pre_launch_children]

        set_global_state(
            _GlobalState(
                tokenizer_manager=self.tokenizer_manager,
                scheduler_info=scheduler_info,
                template_manager=template_manager,
            )
        )

        self.app = fastapi_app
        if self.sglang_args.api_key:
            add_api_key_middleware(self.app, self.sglang_args.api_key)

        server_task = asyncio.create_task(self._serve_http())
        
        try:
            await self._wait_and_warmup()
        except Exception as e:
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)
            raise

        health_check_task = asyncio.create_task(self._health_check_monitor())

        pending = {server_task, health_check_task}
        try:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.exception():
                    raise task.exception()
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    def _serve_http(self) -> Coroutine:
        config = uvicorn.Config(
            self.app,
            host=self.sglang_args.host,
            port=self.sglang_args.port,
            # ... other configs ...
        )
        server = uvicorn.Server(config)
        self.server_shutdown_event = server.should_exit
        server.install_signal_handlers = lambda: {}
        return server.serve()

    async def _wait_and_warmup(self):
        # 1. Wait for http server to be ready by polling /get_model_info
        # ...
        # 2. Send a warmup /generate or /encode request.
        # ...
        logger.info("SGLang backend server is warmed up and ready to roll!")
        self.server_ready.set()

    async def _health_check_monitor(self):
        while True:
            await asyncio.sleep(5)
            for proc in self.subprocesses:
                if not proc.is_running():
                    logger.error(f"SGLang subprocess with PID {proc.pid} has terminated unexpectedly.")
                    raise RuntimeError("A critical SGLang subprocess has failed.")

    def cleanup(self):
        logger.info("Cleaning up SGLang backend resources...")
        try:
            if self.server_shutdown_event and not self.server_shutdown_event.is_set():
                self.server_shutdown_event.set()
            if self.server_task and not self.server_task.done():
                self.server_task.cancel()
        finally:
            logger.info("Killing SGLang process tree...")
            kill_process_tree(os.getpid(), include_parent=False)

```
