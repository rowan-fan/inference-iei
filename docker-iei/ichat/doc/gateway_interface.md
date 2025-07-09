# iChat Gateway API 接口文档

本文档定义了 iChat Gateway API 的核心交互接口。

## 1. Worker 注册与心跳接口

Worker 启动后，定期向 Gateway 发送心跳请求，以更新状态。此接口同时处理首次注册和后续心跳，采用 "upsert"（更新或插入）模式。

- 如果 Gateway 中不存在 `worker_id`，则视为 **注册**，并记录 Worker 的完整信息。
- 如果 `worker_id` 已存在，则视为 **心跳**，并更新其健康状态和元数据。

这种设计简化了 Worker 的实现，使其无需区分初次连接和后续通信，只需周期性地报告自身状态。

- **URL**: `/v1/workers/heartbeat`
- **Method**: `POST`
- **Content-Type**: `application/json`

### 请求体 (Request Body)

每次请求都应包含 Worker 的完整配置信息。

```json
{
  "worker_id": "a-uuid-generated-by-worker",
  "model_name": "large-model-b-dynamic",
  "backend": "vllm",
  "host": "10.0.1.23",
  "port": 8002,
  "model_path": "/path/to/large_model_B",
  "gpu_ids": "1,2",
  "heartbeat_interval": 30,
  "backend_args": {
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.90
  }
}
```

### 参数详解

| 字段 | 类型 | 描述 | 是否必须 |
|---|---|---|---|
| `worker_id` | string | Worker 的唯一标识符 (UUID)，由 Worker 实例在启动时自行生成。 | 是 |
| `model_name` | string | Worker 的唯一标识名称，用于 Gateway 进行路由和管理。对应 `serve.py` 的 `--served-model-name` 参数。 | 推荐 |
| `backend` | string | 使用的推理引擎，可选值为 `"vllm"` 或 `"sglang"`。对应 `--backend` 参数。 | 是 |
| `host` | string | Worker 服务的 IP 地址或主机名。对应 `--host` 参数。 | 是 |
| `port` | integer | Worker 服务的端口号。对应 `--port` 参数。 | 是 |
| `model_path` | string | 加载的模型路径或 Hugging Face Hub ID。对应 `--model` 参数。 | 是 |
| `gpu_ids` | string | Worker 使用的 GPU 设备 ID，源于 `CUDA_VISIBLE_DEVICES` 环境变量。 | 是 |
| `heartbeat_interval` | integer | Worker 发送心跳的间隔秒数。 | 是 |
| `backend_args` | object | 一个包含所有其他引擎启动参数的键值对对象。这些参数将直接传递给 `vllm` 或 `sglang` 引擎。例如 `{"tensor_parallel_size": 2}`。 | 否 |

### 响应 (Response)

Gateway 收到心跳后，返回一个确认响应，未来可扩展为包含需要 Worker 执行的指令（例如，`graceful_shutdown`）。

**成功响应 (`200 OK`)**

```json
{
  "success": true,
  "action": "none"
}
```

**失败响应 (`400 Bad Request` 或 `500 Internal Server Error`)**

如果请求失败（例如，参数缺失或无效），Gateway 返回错误信息。

```json
{
  "success": false,
  "message": "Error message detailing what went wrong."
}
```

---

## 2. Worker 日志流接口

Worker 通过 SSE (Server-Sent Events) 将日志实时流式传输到 Gateway，以实现集中式日志管理。

- **URL**: `/v1/logs/stream/{worker_id}`
- **Method**: `GET`

### 流程描述

1.  **建立连接**: Worker 使用启动时自行生成的 `worker_id`，向 `/v1/logs/stream/{worker_id}` 端点发起一个 `GET` 请求。
2.  **流式传输**: Gateway 接受请求后，该连接将转为 SSE 长连接。Worker 捕获自身的 `stdout` 和 `stderr`，并将每行日志作为独立的 SSE 事件发送。
3.  **日志格式**: 每个 SSE 事件包含 `event` 类型（`stdout` 或 `stderr`）和 `data` 负载（原始日志字符串）。

### SSE 事件流 (SSE Event Stream)

连接建立后，Worker 开始发送日志事件。

```
event: stdout
data: INFO:     Uvicorn running on http://0.0.0.0:8001 (Press CTRL+C to quit)

event: stderr
data: WARNING:  High GPU memory usage detected on device 0: 95.0%
```

---

## 3. 数据平面 API (OpenAI 兼容)

Gateway 聚合了所有 Worker 的能力，并提供一个统一的 OpenAI 兼容入口。客户端的所有推理请求都应发往 Gateway。

### a. `POST /v1/chat/completions`

- **功能**: 接收客户端的聊天补全请求，并根据请求体中的 `model` 字段路由到合适的 Worker。
- **请求体/响应体**: 完全兼容 OpenAI `chat/completions` API 规范。

### b. `POST /v1/completions`

- **功能**: (传统接口) 接收文本补全请求并路由到合适的 Worker。
- **请求体/响应体**: 完全兼容 OpenAI `completions` API 规范。

### c. `POST /v1/embeddings`

- **功能**: 接收文本嵌入请求并路由到合适的 Worker。
- **请求体/响应体**: 完全兼容 OpenAI `embeddings` API 规范。

### d. `GET /v1/models`

- **功能**: 聚合所有已注册的、健康的 Worker 的模型信息，并返回一个统一的模型列表。
- **响应体示例**:
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen-7b-chat",
      "object": "model",
      "created": 1677610600,
      "owned_by": "organization-owner"
    },
    {
      "id": "qwen-14b-chat",
      "object": "model",
      "created": 1677610600,
      "owned_by": "organization-owner"
    },
    {
      "id": "model-a-dynamic",
      "object": "model",
      "created": 1677610600,
      "owned_by": "organization-owner"
    }
  ]
}
```

### e. `GET /v1/models/{model_name}`

- **功能**: 获取特定模型的详细信息。
- **响应体示例**:
```json
{
  "id": "qwen-7b-chat",
  "object": "model",
  "created": 1677610600,
  "owned_by": "organization-owner"
}
```
---

## 4. 管理员接口 (Admin API)

这些接口用于监控和管理整个 iChat 集群。

### a. `GET /v1/admin/workers`

- **功能**: 列出所有已注册的 Worker（包括由 Gateway 管理的和动态注册的），并显示其状态（如 `healthy`, `unhealthy`）、模型、地址等信息。
- **响应体示例**:
```json
{
  "success": true,
  "workers": [
    {
      "worker_id": "worker-id-1",
      "model_name": "qwen-7b-chat",
      "status": "healthy",
      "backend": "vllm",
      "host": "127.0.0.1",
      "port": 8001,
      "registered_at": "2023-10-27T10:00:00Z",
      "last_heartbeat": "2023-10-27T10:05:00Z"
    },
    {
      "worker_id": "worker-id-2",
      "model_name": "qwen-14b-chat",
      "status": "unhealthy",
      "backend": "sglang",
      "host": "127.0.0.1",
      "port": 8002,
      "registered_at": "2023-10-27T10:01:00Z",
      "last_heartbeat": "2023-10-27T10:03:00Z"
    }
  ]
}
```

### b. `GET /v1/admin/workers/{worker_id}`

- **功能**: 获取指定 Worker 的详细信息，包括元数据和健康状态。
- **响应体示例**:
```json
{
  "success": true,
  "worker": {
    "worker_id": "worker-id-1",
    "model_name": "qwen-7b-chat",
    "model_path": "/path/to/qwen-7b-chat",
    "status": "healthy",
    "backend": "vllm",
    "host": "127.0.0.1",
    "port": 8001,
    "gpu_ids": "0",
    "heartbeat_interval": 30,
    "backend_args": {
      "tensor_parallel_size": 1,
      "gpu_memory_utilization": 0.9
    },
    "registered_at": "2023-10-27T10:00:00Z",
    "last_heartbeat": "2023-10-27T10:05:00Z"
  }
}
```

### c. `POST /v1/admin/workers/launch`

- **功能**: 动态启动一个新的 Worker 进程。此功能适用于通过 Gateway 按需扩展 Worker 的场景。
- **请求体**: 包含启动 Worker 所需的完整配置，与 `config.yaml` 中 `managed_workers` 的条目结构类似。
```json
{
  "model_name": "new-model-from-api",
  "model_path": "/path/to/new-model",
  "backend": "vllm",
  "gpu_ids": [3],
  "port": 8003,
  "heartbeat_interval": 15,
  "backend_args": {
    "tensor_parallel_size": 1
  }
}
```
- **响应体**:
```json
{
  "success": true,
  "message": "Worker launch command issued.",
  "worker_id": "newly-launched-worker-id"
}
```

### d. `DELETE /v1/admin/workers/{worker_id}`

- **功能**: 停止并移除一个由 Gateway 启动的 Worker 实例。
- **响应体**:
```json
{
  "success": true,
  "message": "Worker shutdown command issued."
}
```

### e. `GET /v1/admin/cluster/status`

- **功能**: 获取整个 iChat 集群的总体状态，包括 Gateway 状态、资源使用情况和已注册 Worker 概览。
- **响应体示例**:
```json
{
  "success": true,
  "gateway_status": "running",
  "total_workers": 2,
  "healthy_workers": 1,
  "unhealthy_workers": 1,
  "models": ["qwen-7b-chat", "qwen-14b-chat"]
}
```

### f. `GET /v1/admin/cluster/version`

- **功能**: 获取 iChat Gateway 的版本信息。
- **响应体示例**:
```json
{
  "success": true,
  "version": "0.1.0"
}
```
