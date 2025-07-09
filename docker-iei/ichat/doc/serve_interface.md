# iChat Worker API 接口文档 (`serve.py`)

本文档定义了 iChat Worker (`serve.py`) 的核心 API 接口。Worker 是实际运行并提供模型推理服务的实例，其接口主要由 Gateway 调用。

## 1. 数据平面 API (OpenAI 标准兼容)

Worker 包装了底层推理引擎（如 vLLM, SGLang），并向上层（Gateway）暴露标准的 OpenAI API 接口。Gateway 会将客户端的推理请求直接转发到 Worker 的这些端点上。

### a. `POST /v1/chat/completions`

- **功能**: 接收聊天补全请求。Gateway 会将客户端的请求直接转发至此。
- **请求体/响应体**: 完全兼容 OpenAI `chat/completions` API 规范。

### b. `POST /v1/completions`

- **功能**: 接收文本补全请求（传统接口）。
- **请求体/响应体**: 完全兼容 OpenAI `completions` API 规范。

### c. `POST /v1/embeddings`

- **功能**: 接收文本嵌入请求。
- **请求体/响应体**: 完全兼容 OpenAI `embeddings` API 规范。

### d. `GET /v1/models`

- **功能**: 返回此 Worker 加载的模型的详细信息。
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
      }
    ]
  }
  ```

---

## 2. 控制平面 API

这些接口供 Gateway 用于监控和管理 Worker。

### a. `GET /health`

- **功能**: 健康检查接口。Gateway 使用此接口进行存活性和就绪性探测，以确认 Worker 是否可以正常接收流量。
- **响应 (`200 OK`)**:
  ```json
  {
    "status": "ok"
  }
  ```

### b. `GET /metrics`

- **功能**: 暴露 Prometheus 格式的性能指标。可用于监控推理延迟、吞吐量、GPU 使用率等。
- **响应**: 返回 Prometheus 格式的文本数据。

### c. 控制通信

除了暴露服务接口外，Worker 还需要主动与 Gateway 进行通信，以完成注册和日志上报。

- **向 Gateway 注册/心跳**: Worker 启动后，会定期向 Gateway 的 `/v1/workers/heartbeat` 接口发送 `POST` 请求，以报告自身状态。
- **向 Gateway 推送日志**: Worker 会主动与 Gateway 的 `/v1/logs/stream/{worker_id}` 接口建立 SSE 连接，将实时日志推送至 Gateway。
