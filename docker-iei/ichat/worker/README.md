# iChat Worker (`ichat.worker`) 设计文档

## 1. 目标

`ichat.worker` 是 iChat Worker 的主入口模块。它作为一个通用的服务运行器，其核心职责是：

1.  **框架与后端参数分离解析**: `ichat.worker` 负责解析 iChat 框架自身的特定参数（如 `--gateway-address`），并将所有其他参数作为原始列表“透传”给指定的后端。
2.  **后端的动态加载与运行**: 根据 `--backend` 参数，动态地实例化并加载指定的推理后端（`VLLMBackend` 或 `SGLangBackend`）。后端接收透传的参数列表，并负责完成自身的参数解析。
3.  **服务生命周期的统一管理**: 启动并协调整个 Worker 服务的生命周期，包括推理后端服务和所有后台任务（如心跳）。它通过信号处理和 `asyncio` 事件机制，实现对所有组件的优雅启动与关闭。
4.  **与 Gateway 的无缝集成**: 如果配置了 Gateway 地址，`ichat.worker` 负责启动与 Gateway 的所有交互，包括服务注册、周期性心跳和（可选的）日志流上报。
5.  **提供一个独立的、可执行的服务单元**: 使得每个 Worker 都是一个自包含的进程，可以独立部署和管理。

`ichat.worker` 的设计旨在将 Worker 的框架逻辑（如服务发现、监控、生命周期管理）与业务逻辑（即模型的实际推理，由后端负责）彻底解耦，使得整个系统更具模块化、健壮性和可扩展性。

## 2. 设计原则

- **后端无关性 (Backend Agnostic)**: `ichat.worker` 被设计成一个通用的后端运行器。它不关心具体后端（vLLM, SGLang）的内部实现，只需后端类继承自 `backends.base.BaseBackend` 并实现了其 `run()` 方法。参数解析的职责也被下放到各个后端，`ichat.worker` 只需传递原始参数列表。
- **集中式生命周期控制 (Centralized Lifecycle Control)**: `ichat.worker` 是整个 Worker 进程的“大脑”，负责捕获操作系统信号（`SIGINT`, `SIGTERM`）并协调所有异步任务（推理服务、心跳任务等）的有序关闭，防止资源泄露。
- **配置驱动 (Configuration-Driven)**: Worker 的核心行为，如选择后端和与 Gateway 的集成，完全由命令行参数驱动。iChat 框架的参数在 `worker/args.py` 中定义。
- **无缝网关集成 (Seamless Gateway Integration)**: `ichat.worker` 的设计原生支持与 iChat Gateway 的集成。通过简单的参数配置，即可启用服务自动注册、心跳和日志上报功能。同时，它也能在无 Gateway 的情况下独立运行，便于开发和测试。
- **关注点分离 (Separation of Concerns)**: 新的参数处理策略强化了关注点分离原则。`ichat.worker` 只关心框架级别的配置，而将所有与模型推理相关的配置完全委托给后端处理。这使得添加新后端变得更加简单，因为无需修改核心的 `ichat.worker` 逻辑。

## 3. `ichat.worker` 核心架构

`ichat.worker` 的架构围绕 `asyncio` 构建，通过几个关键函数和组件的协作，实现了对复杂异步服务的优雅管理。

### 3.1. `main()` 主函数

`main()` 是 Worker 的异步入口点，负责编排整个应用程序的启动和关闭序列。

1.  **解析框架参数**: 调用 `worker/args.py` 的 `parse_worker_args()`。该函数只解析 iChat 框架定义的参数（如 `--backend`），并返回一个包含这些已解析参数的 `Namespace` 对象，以及一个包含所有未解析的、将传递给后端的原始参数字符串列表 (`backend_argv`)。
2.  **设置日志**: 调用 `utils/logger.py` 的 `setup_logging()` 初始化日志系统。根据参数，日志可以被配置为输出到控制台，并可选地通过 SSE 流式传输到 Gateway。
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

参数处理采用**框架与后端分离**的策略。`parse_worker_args()` 只解析 iChat 框架自身参数，未识别参数全部透传给后端。

### 3.5. 心跳检测机制实现细节

心跳检测由 `worker/heartbeat.py` 的 `HeartbeatManager` 类实现，主要负责 Worker 与 Gateway 的注册与健康状态上报，实现了完整的生命周期管理。

## 4. 执行流程

`ichat.worker` 的 `main` 函数遵循以下执行流程：

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

## 5. 启动方式

### a. 目录结构

```
/app/
└── ichat/
    ├── worker/
    │   ├── __main__.py
    │   └── ...
    ├── gateway/
    │   ├── __main__.py
    │   └── ...
    ├── config.yaml
    └── ...
```

### b. 启动命令

在 `ichat` 的父目录中（例如 `/app`），使用以下命令启动 Worker：

```bash
python3 -m ichat.worker --backend vllm --model-path /path/to/model --port 8001 ...
```

- **`python3 -m ichat.worker`**: 这会告诉 Python 将 `ichat` 目录当作一个包，并执行其中的 `worker` 模块（即 `worker/__main__.py`）。
- 其余参数根据后端和业务需求传递。

### c. 配置文件与参数

Worker 的所有行为均由命令行参数驱动，通常由 Gateway 自动生成并启动。