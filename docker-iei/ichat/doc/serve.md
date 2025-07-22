# iChat Worker (`serve.py`) 设计文档

## 1. 目标

`serve.py` 是 iChat Worker 的主入口程序。它作为一个通用的服务运行器，其核心职责是：

1.  **框架与后端参数分离解析**: `serve.py` 负责解析 iChat 框架自身的特定参数（如 `--gateway-address`），并将所有其他参数作为原始列表“透传”给指定的后端。
2.  **后端的动态加载与运行**: 根据 `--backend` 参数，动态地实例化并加载指定的推理后端（`VLLMBackend` 或 `SGLangBackend`）。后端接收透传的参数列表，并负责完成自身的参数解析。
3.  **服务生命周期的统一管理**: 启动并协调整个 Worker 服务的生命周期，包括推理后端服务和所有后台任务（如心跳）。它通过信号处理和 `asyncio` 事件机制，实现对所有组件的优雅启动与关闭。
4.  **与 Gateway 的无缝集成**: 如果配置了 Gateway 地址，`serve.py` 负责启动与 Gateway 的所有交互，包括服务注册、周期性心跳和（可选的）日志流上报。
5.  **提供一个独立的、可执行的服务单元**: 使得每个 Worker 都是一个自包含的进程，可以独立部署和管理。

`serve.py` 的设计旨在将 Worker 的框架逻辑（如服务发现、监控、生命周期管理）与业务逻辑（即模型的实际推理，由后端负责）彻底解耦，使得整个系统更具模块化、健壮性和可扩展性。

## 2. 设计原则

- **后端无关性 (Backend Agnostic)**: `serve.py` 被设计成一个通用的后端运行器。它不关心具体后端（vLLM, SGLang）的内部实现，只需后端类继承自 `backends.base.BaseBackend` 并实现了其 `run()` 方法。参数解析的职责也被下放到各个后端，`serve.py` 只需传递原始参数列表。
- **集中式生命周期控制 (Centralized Lifecycle Control)**: `serve.py` 是整个 Worker 进程的“大脑”，负责捕获操作系统信号（`SIGINT`, `SIGTERM`）并协调所有异步任务（推理服务、心跳任务等）的有序关闭，防止资源泄露。
- **配置驱动 (Configuration-Driven)**: Worker 的核心行为，如选择后端和与 Gateway 的集成，完全由命令行参数驱动。iChat 框架的参数在 `config/args.py` 中定义。
- **无缝网关集成 (Seamless Gateway Integration)**: `serve.py` 的设计原生支持与 iChat Gateway 的集成。通过简单的参数配置，即可启用服务自动注册、心跳和日志上报功能。同时，它也能在无 Gateway 的情况下独立运行，便于开发和测试。
- **关注点分离 (Separation of Concerns)**: 新的参数处理策略强化了关注点分离原则。`serve.py` 只关心框架级别的配置，而将所有与模型推理相关的配置完全委托给后端处理。这使得添加新后端变得更加简单，因为无需修改核心的 `serve.py` 逻辑。

## 3. `serve.py` 核心架构

`serve.py` 的架构围绕 `asyncio` 构建，通过几个关键函数和组件的协作，实现了对复杂异步服务的优雅管理。

### 3.1. `main()` 主函数

`main()` 是 Worker 的异步入口点，负责编排整个应用程序的启动和关闭序列。

1.  **解析框架参数**: 调用 `config.args.parse_worker_args()`。该函数只解析 iChat 框架定义的参数（如 `--backend`），并返回一个包含这些已解析参数的 `Namespace` 对象，以及一个包含所有未解析的、将传递给后端的原始参数字符串列表 (`backend_argv`)。
2.  **设置日志**: 调用 `utils.logger.setup_logging()` 初始化日志系统。根据参数，日志可以被配置为输出到控制台，并可选地通过 SSE 流式传输到 Gateway。
3.  **实例化后端**: 根据 `args.backend` 的值，选择并实例化对应的后端类 (`VLLMBackend` 或 `SGLangBackend`)。将框架参数对象和后端原始参数列表都传递给后端构造函数。后端将负责解析它所需要的参数。
4.  **管理后台任务**: 使用 `lifespan` 异步上下文管理器来启动和停止所有后台服务。目前主要是 `HeartbeatManager`。
5.  **运行主服务**: 在 `lifespan` 上下文内，创建 `asyncio.Task` 来运行 `backend.run()` 方法。这是推理服务的主任务。
6.  **等待与关闭**: `main` 函数会创建一个 `server_task` (用于运行 `backend.run()`) 和一个 `shutdown_task` (用于等待 `shutdown_event`)。它使用 `asyncio.wait` 等待这两个任务中的任何一个首先完成。
    *   **如果收到外部信号 (如 `Ctrl+C`)**: `shutdown_event` 被设置，`shutdown_task` 完成。程序会取消仍在运行的 `server_task`。
    *   **如果后端服务自行退出 (正常或崩溃)**: `server_task` 完成。程序会主动设置 `shutdown_event`，以确保所有其他后台任务（如心跳）也能收到关闭信号，然后取消 `shutdown_task`。
    这种双向的关闭机制确保了无论是由外部信号还是内部服务故障触发的关闭，整个 Worker 都能被优雅地清理和终止。

### 3.2. `lifespan()` 上下文管理器

`@asynccontextmanager` 定义的 `lifespan` 函数，为需要长时间运行的后台任务提供了一个优雅的生命周期管理方案。

- **启动**: 在 `yield` 之前，它会检查是否配置了 `gateway_address`。如果是，则创建并启动 `HeartbeatManager` 任务。这种设计将后台任务的启动逻辑与主业务逻辑分离。
- **停止**: 在 `finally` 块中，它确保无论程序是正常退出还是因异常/信号中断，所有启动的后台任务（如 `heartbeat_manager`）都会被安全地停止 (`await heartbeat_manager.stop()`)。

### 3.3. `_signal_handler()` 信号处理器

这是一个标准的 Python 信号处理函数，通过 `signal.signal()` 注册来捕获 `SIGINT` (Ctrl+C) 和 `SIGTERM` (kill 命令)。

- 当接收到信号时，它唯一的职责就是调用 `shutdown_event.set()`。`shutdown_event` 是一个 `asyncio.Event`，在整个应用程序中共享。
- 多个 `asyncio` 任务（如此处的 `main` 函数）可以 `await shutdown_event.wait()`。一旦事件被设置，所有等待的任务都会被唤醒，从而触发它们各自的清理和关闭逻辑。这是在异步程序中实现优雅关闭的最佳实践。

### 3.4. 参数处理 - 框架与后端分离策略

为了在最大化灵活性的同时保持清晰的架构，参数处理采用了**框架与后端分离**的策略。`serve.py` 不再尝试理解或统一任何后端的参数，而是将这个责任完全下放。

`parse_worker_args()` 函数 (`config/args.py`) 的核心逻辑如下：
1.  **定义框架参数**: 使用 `argparse` 定义 iChat 框架自身需要的参数，例如：`--backend`, `--gateway-address`, `--heartbeat-interval`, `--log-level`, `--log-streaming` 等。这些参数控制着 Worker 的元数据和行为，与具体的推理引擎无关。
2.  **解析与分离**: 使用 `parser.parse_known_args()` 来执行解析。此方法会将命令行参数分割成两部分：
    *   一个 `Namespace` 对象，其中包含所有已定义的框架参数。
    *   一个字符串列表 (`backend_argv`)，其中包含所有未被框架解析器识别的参数。这个列表被视为后端专属的参数。

**后端的参数处理责任**:
-   `serve.py` 在实例化后端时，会将 `framework_args` 和 `backend_argv` 这两个对象都传递给后端的 `__init__` 方法。
-   每个后端（如 `VLLMBackend`, `SGLangBackend`）内部都有自己的参数解析逻辑。它会定义一个自己的 `ArgumentParser`，用来解析 `backend_argv` 列表。
-   后端可以自由地定义它所支持的所有参数，包括像 `--model-path`, `--tensor-parallel-size` 等。
-   后端还可以在内部实现 iChat 统一参数名到其原生参数名的映射（例如，vLLM 后端可以将接收到的 `--model-path` 转换为 vLLM 所需的 `--model`），从而为用户提供便利。

| iChat 标准参数 | vLLM 对应参数 | SGLang 对应参数 | 描述 |
|:---|:---|:---|:---|
| `--host` | `--host` | `--host` | 服务监听的主机地址 |
| `--port` | `--port` | `--port` | 服务监听的端口 |
| `--model-path` | `--model` | `--model-path`| 模型权重的路径或HuggingFace ID |
| `--tokenizer-path` | `--tokenizer` | `--tokenizer-path` | Tokenizer的路径或HuggingFace ID |
| `--trust-remote-code`| `--trust-remote-code`| `--trust-remote-code`| 是否信任远程代码 |
| `--served-model-name`| `--served-model-name`| `--served-model-name`| 在Gateway中注册的模型名称 |
| `--context-length` | `--max-model-len` | `--context-length`| 模型的最大上下文长度 |

2.  **解析与分离**: 使用 `parser.parse_known_args()` 来执行解析。此方法会返回两个结果：
    *   一个包含所有已定义参数（iChat特定 + 统一通用）的 `Namespace` 对象。
    *   一个包含所有命令行中出现但未在解析器中定义的未知参数的列表（例如 `--tensor-parallel-size`, `--gpu-memory-utilization` 等）。

3.  **参数传递**: 解析出的已知参数和未知参数列表，都会被传递给所选后端的构造函数 (`__init__`)。
    *   后端实现（如 `VLLMBackend`）负责处理这些参数：它会优先使用统一参数（如 `args.model_path`），并将其转换为自己的内部格式（如 `vllm_args.model`），同时将所有透传的未知参数也应用到自己的参数配置中。

这种设计实现了灵活性和标准化的平衡：用户可以立即使用任何后端支持的最新参数，而无需等待 iChat 框架进行更新；同时，对于最核心的配置，又提供了一致和简化的体验。


## 4. 执行流程

`serve.py` 的 `main` 函数遵循以下执行流程：

```mermaid
graph TD
    A[开始] --> B{解析框架参数<br>(其他参数透传)};
    B --> C{设置日志系统};
    C --> D{进入 lifespan 上下文};
    D -- gateway-address 已配置 --> E[启动心跳后台任务];
    E --> F[实例化推理后端];
    D -- gateway-address 未配置 --> F;
    F --> G[创建后端服务Task和关闭信号Task];
    G --> H{等待任一Task首先完成};
    H --> I{无论哪个Task先完成, 都触发优雅关闭};
    I --> J[设置关闭事件(若未设置)并取消其他任务];
    J --> K[等待所有任务完成清理];
    K --> L[退出 lifespan 上下文];
    L --> M[停止心跳后台任务];
    M --> N[结束];
```

## 5. `serve.py` 完整代码架构

*本章节代码为高级伪代码，旨在展示核心架构和逻辑流程。完整的实现请参考 `ichat/serve.py` 源文件。*

```python
# docker-iei/ichat/serve.py

import asyncio
import signal
from contextlib import asynccontextmanager

# 导入后端实现、参数解析、日志和心跳管理器
from backends.vllm_backend import VLLMBackend
from backends.sglang_backend import SGLangBackend
from config.args import parse_worker_args
from utils.logger import setup_logging
from monitor.heartbeat import HeartbeatManager

# 全局事件，用于在整个应用程序中协调优雅关闭
shutdown_event = asyncio.Event()

def _signal_handler(*_):
    """
    信号处理函数。当接收到 SIGINT 或 SIGTERM 时，设置全局关闭事件。
    这是触发所有异步任务开始其清理程序的信标。
    """
    print("INFO:     Shutdown signal received. Initiating graceful shutdown...")
    shutdown_event.set()

@asynccontextmanager
async def lifespan(args):
    """
    一个异步上下文管理器，负责管理所有后台任务的生命周期。
    """
    heartbeat_manager = None
    # 仅当提供了 Gateway 地址时，才启动心跳服务
    if args.gateway_address:
        heartbeat_manager = HeartbeatManager(args)
        # 在后台启动心跳任务
        asyncio.create_task(heartbeat_manager.start())
    
    try:
        # yield 将控制权交还给主逻辑
        yield
    finally:
        # 确保在程序退出时，所有后台任务都被优雅地停止
        if heartbeat_manager:
            print("INFO:     Stopping heartbeat manager...")
            await heartbeat_manager.stop()
        print("INFO:     Worker has shut down.")

async def main():
    """iChat Worker 的主入口点。"""
    
    # 1. 解析框架参数，并将后端参数分离出来
    framework_args, backend_argv = parse_worker_args()

    # 2. 设置日志系统
    setup_logging(
        log_level=getattr(framework_args, "log_level", "INFO"),
        stream_to_gateway=getattr(framework_args, "log_streaming", False),
        gateway_address=getattr(framework_args, "gateway_address", None),
    )

    # 3. 根据参数动态实例化推理后端
    if framework_args.backend == 'vllm':
        from .backends.vllm_backend import VLLMBackend
        backend = VLLMBackend(framework_args, backend_argv)
    elif framework_args.backend == 'sglang':
        from .backends.sglang_backend import SGLangBackend
        backend = SGLangBackend(framework_args, backend_argv)
    else:
        raise ValueError(f"Unsupported backend: {framework_args.backend}")

    # 4. 使用 lifespan 上下文管理器来运行服务
    async with lifespan(framework_args):
        # 创建后端服务任务和关闭信号监听任务
        server_task = asyncio.create_task(backend.run())
        shutdown_task = asyncio.create_task(shutdown_event.wait())

        # 等待任一任务首先完成
        done, pending = await asyncio.wait(
            {server_task, shutdown_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        # 如果是后端服务Task完成（例如崩溃），则主动触发全局关闭
        if server_task in done and not shutdown_event.is_set():
            print("INFO:     Backend task completed. Initiating graceful shutdown...")
            shutdown_event.set()
        
        # 此刻，关闭信号已发出，取消所有仍在运行的挂起任务
        for task in pending:
            task.cancel()
            
        # 等待所有原始任务（包括已完成和刚被取消的）完成其清理过程
        await asyncio.gather(server_task, shutdown_task, return_exceptions=True)

        # （可选）记录关闭原因
        if server_task.done() and not shutdown_task.done():
             # ... 检查并记录后端任务的退出状态 ...
             pass
        else:
             print("INFO:     Shutdown signal received, server has stopped.")

if __name__ == "__main__":
    # 注册信号处理程序以实现优雅关闭
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, asyncio.CancelledError):
        # 优雅地处理最终的取消异常
        print("INFO:     Main task cancelled. Exiting.")
        pass
