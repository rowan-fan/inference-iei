# iChat Gateway (`gateway.py`) 设计文档

## 1. 目标

`gateway.py` 是 iChat 框架的中央网关服务，其核心职责是：

1.  **统一 API 入口**: 作为所有客户端请求的唯一入口，遵循 OpenAI API 标准。它利用 `LiteLLM` 提供统一的 `/v1/chat/completions` 等接口，将底层异构的 Worker 服务抽象化。
2.  **动态请求路由**: 根据客户端请求中的 `model` 字段，智能地将流量动态路由到后端正确的 iChat Worker 实例。
3.  **Worker 生命周期管理**: 能够根据配置文件 (`config.yaml`) 自动启动、监控和管理一组 Worker 进程（`managed_workers`），确保服务的可用性。
4.  **服务发现与健康检查**: 维护一个动态的服务注册中心。它接收来自所有 Worker（包括手动启动的）的注册和周期性心跳，并基于此实时更新可用模型路由，自动剔除不健康的节点。
5.  **集中式管理与监控**: 提供一套管理 API，用于查询集群状态、查看所有已注册的 Worker、手动启动或停止 Worker 实例，并集中收集来自所有 Worker 的日志。
6.  **混合部署模式支持**: 无缝集成两种 Worker 部署模式：由 Gateway 自动管理的进程和外部独立启动并向 Gateway 动态注册的进程。

`gateway.py` 的设计旨在将客户端与后端模型服务完全解耦，提供一个稳定、可扩展、易于管理的中心控制平面，从而简化复杂的多模型、多节点推理服务的部署和运维。

## 2. 设计原则

- **关注点分离 (Separation of Concerns)**: Gateway 专注于网络路由、服务管理和 API 聚合，而 Worker 专注于模型推理。这种分离使得系统各组件职责清晰，易于独立扩展和维护。
- **动态配置驱动 (Dynamic Configuration-Driven)**: Gateway 的核心行为，特别是模型路由表，是动态的。它通过服务注册中心实时更新，支持在不重启 Gateway 的情况下动态增删 Worker 和模型。
- **高可用性与自愈 (High Availability & Self-healing)**: 通过心跳机制，Gateway 能够持续监控 Worker 的健康状况。当 Worker 心跳超时，Gateway 会自动将其从路由表中移除，防止请求被发送到故障节点。
- **混合部署的灵活性 (Hybrid Deployment Flexibility)**: Gateway 的设计原生支持“自动管理”和“动态注册”两种模式的 Worker，为不同的部署场景（如开发测试、生产环境、分布式部署）提供了最大的灵活性。
- **基于 LiteLLM (Built on LiteLLM)**: 充分利用 LiteLLM 强大的 API 兼容性和路由能力作为数据平面的基础，确保了对 OpenAI 生态系统的广泛兼容性。
- **中心化控制 (Centralized Control)**: 为管理员提供了一套统一的 RESTful API，使其能够从一个中心点查看和控制整个 LLM 服务集群。

## 3. `gateway.py` 核心架构

Gateway 的架构是围绕一个中心化的 FastAPI 应用构建的，集成了服务注册、Worker 管理和 LiteLLM 路由等多个核心组件。

```mermaid
graph TD
    subgraph "外部交互"
        Client[客户端]
        Admin[管理员]
        Worker[iChat Worker]
    end

    subgraph "iChat Gateway (gateway.py)"
        FastAPI["FastAPI 应用"]
        
        subgraph "控制平面 (Control Plane)"
            Registry[("服务注册中心<br>ServiceRegistry")]
            WorkerManager["Worker管理器<br>(管理 managed_workers)"]
            AdminAPI["管理 API<br>(/v1/admin/...)"]
            WorkerAPI["Worker 交互 API<br>(/v1/workers/...)"]
        end
        
        subgraph "数据平面 (Data Plane)"
            LiteLLMRouter["LiteLLM 路由器"]
            OpenAI_API["OpenAI 兼容 API<br>(/v1/chat/completions, etc.)"]
        end
        
        LogCollector["日志收集器"]
        Config["config.yaml"]
    end

    Client --> |请求 (model='x')| OpenAI_API
    Admin --> |管理操作| AdminAPI
    
    OpenAI_API -- "查询路由" --> LiteLLMRouter
    LiteLLMRouter -- "获取可用模型/Worker" --> Registry
    LiteLLMRouter --> |转发请求| Worker
    
    Worker --> |注册/心跳| WorkerAPI
    WorkerAPI -- "更新状态" --> Registry
    
    Worker -.-> |SSE 日志流| LogCollector
    
    AdminAPI -- "查询/控制" --> Registry
    AdminAPI -- "启动/停止" --> WorkerManager
    
    WorkerManager -- "管理子进程" --> Worker
    
    FastAPI -- "读取配置" --> Config
```

### 3.1. 主程序与配置加载

- Gateway 由 `gateway.py` 启动，它需要一个 `--config` 参数指定 `config.yaml` 配置文件的路径。
- 启动时，程序首先解析 YAML 文件，获取 `server_settings`, `managed_workers` 和 `litellm_settings` 等配置。

### 3.2. 服务注册中心 (`monitor.registry.ServiceRegistry`)

- 这是 Gateway 的核心组件之一，一个内存中的数据库，用于跟踪所有 Worker 的状态。
- **功能**:
    - **注册 (Register)**: 当 Worker 首次发送心跳时，将其元数据（模型名称、地址、端口、后端类型等）添加到注册表。
    - **心跳 (Heartbeat)**: 更新现有 Worker 的最后活跃时间戳。
    - **状态检查 (Health Check)**: 定期扫描所有 Worker，将心跳超时的 Worker 标记为 `unhealthy`。
    - **路由表提供 (Provide Routing Table)**: 为 LiteLLM 动态提供一份健康的、可用的模型部署列表。

### 3.3. Worker 管理器 (`WorkerManager`)

- 该组件仅在 `config.yaml` 中定义了 `managed_workers` 时才被激活。
- **功能**:
    - **启动 (Launch)**: 在 Gateway 启动时，根据配置列表，为每个 Worker 创建并启动一个子进程 (`serve.py`)。它会负责构建命令行参数（如 `--model-path`, `--port`）和设置 `CUDA_VISIBLE_DEVICES` 环境变量。
    - **监控 (Monitor)**: 监控它所创建的子进程的生命周期。
    - **终止 (Terminate)**: 在 Gateway 关闭时，优雅地终止所有由它管理的 Worker 子进程。

### 3.4. API 服务器 (FastAPI)

Gateway 的所有网络交互都通过一个 FastAPI 应用提供。

- **数据平面 (Data Plane)**:
    - 由 `LiteLLM` 提供核心的 OpenAI 兼容 API (`/v1/chat/completions`, `/v1/models` 等)。
    - LiteLLM 的 `router` 被配置为动态模式，其模型列表 (`litellm.model_list`) 会由一个后台任务定期从 `ServiceRegistry` 中同步，确保路由信息总是最新的。

- **控制平面 (Control Plane)**:
    - **Worker API (`/v1/workers/heartbeat`)**: 接收 Worker 的注册和心跳请求，并调用 `ServiceRegistry` 进行处理。
    - **管理 API (`/v1/admin/*`)**: 提供给管理员的一系列接口，用于查询 Worker 状态（从 `ServiceRegistry` 读取）、启动新 Worker（通过 `WorkerManager`）等。
    - **日志流 API (`/v1/logs/stream/{worker_id}`)**: （设计）可能是一个代理端点，当管理员访问时，Gateway 会连接到对应 Worker 的 `/logs/stream` SSE 端点，并实时将日志流转发给管理员。

## 4. 执行流程

`gateway.py` 的 `main` 函数遵循以下执行流程：

```mermaid
graph TD
    A[开始] --> B{解析 --config 参数};
    B --> C{读取并解析 config.yaml};
    C --> D{初始化服务注册中心和Worker管理器};
    D --> E{检查 config.yaml 中是否有 managed_workers};
    E -- 是 --> F[Worker管理器启动所有 managed_workers 子进程];
    E -- 否 --> G[跳过子进程启动];
    F --> G;
    G --> H{配置并初始化 LiteLLM 路由器};
    H --> I[启动后台任务: 定期从注册中心同步模型列表到LiteLLM];
    I --> J[启动后台任务: 定期检查Worker心跳超时];
    J --> K{启动 FastAPI/Uvicorn 服务器};
    K --> L{Gateway 开始监听端口 (e.g., 4000)};
    L -- "接收 /v1/workers/heartbeat" --> M[更新服务注册中心];
    L -- "接收 /v1/chat/completions" --> N[LiteLLM 根据注册中心信息路由请求];
    L -- "接收 /v1/admin/workers" --> O[从注册中心查询并返回Worker列表];
    L -- 收到 SIGINT/SIGTERM --> P[触发优雅关闭];
    P --> Q[停止所有后台任务];
    Q --> R[Worker管理器终止所有子进程];
    R --> S[停止Web服务器];
    S --> T[结束];
```

## 5. API 规范

Gateway API 分为数据平面和控制平面。

### a. 数据平面 API (OpenAI 兼容, 面向客户端)

由 LiteLLM 提供，与 OpenAI API 完全兼容。

- `POST /v1/chat/completions`: 接收聊天补全请求，根据 `model` 字段路由到合适的 Worker。
- `POST /v1/completions`: (传统接口) 接收文本补全请求并路由。
- `POST /v1/embeddings`: 接收文本嵌入请求并路由。
- `GET /v1/models`: 聚合所有已注册的、健康的 Worker 的模型信息，并返回一个统一的模型列表。
- `GET /v1/models/{model_name}`: 获取特定模型的详细信息。

### b. 控制平面 API (面向 Worker 和管理员)

#### i. Worker-Gateway 交互接口

- `POST /v1/workers/heartbeat`
  - **功能**: Worker 使用此接口进行首次注册和后续的周期性心跳。Gateway 采用 "upsert" 逻辑处理请求。
  - **请求体**: 包含 Worker 的完整元数据。
    ```json
    {
      "worker_id": "unique-worker-id-123",
      "model_name": "qwen-7b-chat",
      "model_path": "/path/to/qwen-7b-chat",
      "backend": "vllm",
      "host": "192.168.1.10",
      "port": 8001,
      "status": "healthy"
    }
    ```

#### ii. 管理员接口

- `GET /v1/admin/workers`
  - **功能**: 列出所有已注册的 Worker（包括 `managed` 和 `dynamic` 类型）及其详细状态。

- `GET /v1/admin/workers/{worker_id}`
  - **功能**: 获取指定 Worker 的详细信息。

- `POST /v1/admin/workers/launch`
  - **功能**: 动态启动一个新的 Worker 进程（仅限在 Gateway 本机）。
  - **请求体**: 定义新 Worker 的完整配置。

- `DELETE /v1/admin/workers/{worker_id}`
  - **功能**: 停止并移除一个由 Gateway 启动的 (`managed`) Worker 实例。

- `GET /v1/admin/cluster/status`
  - **功能**: 获取 iChat 集群的总体状态概览。

- `GET /v1/admin/cluster/version`
  - **功能**: 获取 iChat Gateway 的版本信息。

## 6. 配置文件 (`config.yaml`) 详解

`config.yaml` 是 Gateway 的核心配置文件，控制其所有行为。

**`config.yaml.example`:**
```yaml
# Gateway服务配置
server_settings:
  host: 0.0.0.0
  port: 4000
  log_level: info

# Gateway自动管理的Worker配置
# Gateway会根据此列表自动启动和管理Worker进程
managed_workers:
  - model_name: qwen-7b-chat
    model_path: /path/to/qwen-7b-chat
    backend: vllm
    gpu_ids: [0]
    port: 8001
    heartbeat_interval: 10
    # 以下为透传给 vLLM 的参数
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    
  - model_name: qwen-14b-chat
    model_path: /path/to/qwen-14b-chat
    backend: vllm
    gpu_ids: [1, 2]
    port: 8002
    heartbeat_interval: 10
    tensor_parallel_size: 2

# LiteLLM设置
litellm_settings:
  drop_params: True
```

- **`server_settings`**:
    - `host`: Gateway 监听的主机地址。
    - `port`: Gateway 监听的端口。
    - `log_level`: Gateway 自身的日志级别。

- **`managed_workers`**:
    - 这是一个列表，每一项定义一个由 Gateway 直接管理的 Worker。
    - `model_name`: 在 Gateway 中注册的模型名称，也是客户端请求时使用的名称。
    - `model_path`: 模型的本地路径或 HuggingFace ID。
    - `backend`: 指定推理后端，如 `vllm` 或 `sglang`。
    - `gpu_ids`: 一个整数列表，指定分配给此 Worker 的物理 GPU 索引。Gateway 会自动设置 `CUDA_VISIBLE_DEVICES` 环境变量。
    - `port`: 分配给此 Worker 监听的端口。
    - `heartbeat_interval`: Worker 向 Gateway 发送心跳的间隔秒数。
    - **其他参数**: 任何未被 Gateway 明确定义的参数 (如 `tensor_parallel_size`) 都会被自动转换为命令行参数（如 `--tensor-parallel-size 2`）并传递给 `serve.py` 子进程。

- **`litellm_settings`**:
    - 存放所有应用于 LiteLLM 模块的全局配置。
    - Gateway 启动时，会遍历此部分下的所有键值对，并执行类似 `litellm.<key> = <value>` 的操作。
    - 例如，设置 `drop_params: True` 会执行 `litellm.drop_params = True`。
    - 更多可用配置，请参考 LiteLLM 的官方文档。

## 7. 启动方式

要正确启动 iChat Gateway，请遵循以下步骤。关键在于需要将 `ichat` 目录作为 Python 的一个模块来运行，而不是直接执行 `gateway.py` 文件。

### a. 目录结构

确保您的工作目录是 `ichat` 的**父目录**。例如，如果您的项目结构如下：

```
/app/
└── ichat/
    ├── gateway.py
    ├── serve.py
    ├── config.yaml
    └── ...
```

那么您应该在 `/app` 目录下执行启动命令。

### b. 启动命令

在 `ichat` 的父目录中（例如 `/app`），使用以下命令启动 Gateway：

```bash
python3 -m ichat.gateway --config ichat/config.yaml
```

- **`python3 -m ichat.gateway`**: 这会告诉 Python 将 `ichat` 目录当作一个包，并执行其中的 `gateway` 模块。这是解决 `ImportError` 的正确方法。
- **`--config ichat/config.yaml`**: 指定配置文件的路径。请注意，路径是相对于您当前所在的父目录（`/app`）而言的。

### c. 配置文件

确保 `ichat/config.yaml` 文件已根据您的环境正确配置，特别是 `managed_workers` 部分的模型路径和 GPU 分配。
