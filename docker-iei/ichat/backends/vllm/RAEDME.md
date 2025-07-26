# iChat vLLM Backend 设计文档

本文件详细阐述 iChat `VLLMBackend` 的设计理念、架构和实现。该后端是 iChat Worker 与 vLLM 推理引擎之间的核心适配层。

## 1. 目标

1.  封装 vLLM 服务，将 vLLM 的 `api_server.py` 启动和管理逻辑封装成可编程的 Python 类。
2.  独立参数解析与转换，支持 iChat 标准参数到 vLLM 参数的映射。
3.  精细化生命周期管理，支持服务就绪通知、预热、健康检查。
4.  无缝集成，主服务只需实例化 `VLLMBackend` 并调用 `run()`。
5.  支持未来与 vLLM 服务器内部状态的深度交互。

## 2. 设计原则

- **非侵入式**：绝不修改 vLLM 源码，所有集成在 `VLLMBackend` 内完成。
- **渐进式集成**：可先用顶层 run，后逐步深入控制 socket、FastAPI、Uvicorn。
- **代码重用**：最大限度重用 vLLM 内部函数，避免 subprocess。
- **配置驱动**：所有 vLLM 支持参数均可配置。

## 3. 类架构

### 3.1 `__init__`
- 接收 `framework_args` 和 `backend_argv`，调用 `_parse_vllm_args` 生成 vLLM 参数。
- 初始化 FastAPI 应用、engine_client、sock 等。

### 3.2 `_parse_vllm_args`
- 调用 vLLM 的 `make_arg_parser()` 创建 ArgumentParser。
- 解析 `backend_argv`，映射 iChat 标准参数到 vLLM 参数。
- 合并 `framework_args`（如 host、port）。

### 3.3 `async run(self)`
- 调用 `_setup_server()` 创建 socket。
- 创建并运行 `_run_server_worker()` 协程。
- finally 块调用 `cleanup()`。

### 3.4 `_setup_server(self)`
- 验证参数，创建并绑定 socket，设置 ulimit。

### 3.5 `async _run_server_worker(self, ...)`
- 使用 `build_async_engine_client` 创建 engine_client。
- 调用 `build_app` 创建 FastAPI 应用。
- 调用 `init_app_state` 填充应用状态。
- 创建 server_task（Uvicorn），await `_wait_and_warmup()`。
- 预热成功后，创建 health_check_task（RPC存活检查）。
- await asyncio.wait 监控 server_task 和 health_check_task。

### 3.6 `_serve_http(self, ...)`
- 配置 Uvicorn，禁用信号处理，保存 shutdown_event。

### 3.7 `async _wait_and_warmup(self)`
- 轮询 `/health` 等待服务就绪。
- 成功后 set server_ready。

### 3.8 `async _health_check_monitor(self)`
- 定期调用 `engine_client.is_sleeping()` 检查存活。
- 超时或异常则抛出 RuntimeError，触发优雅关闭。

### 3.9 `cleanup(self)`
- 触发关闭事件，关闭 socket，取消 server_task。

## 4. 伪代码架构

```python
class VLLMBackend(BaseBackend):
    def __init__(self, framework_args, backend_argv, backend_ready_event):
        self.vllm_args = self._parse_vllm_args(backend_argv)
        # ...

    def _parse_vllm_args(self, backend_argv):
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

    async def _run_server_worker(self, listen_address, sock):
        async with build_async_engine_client(self.vllm_args) as engine_client:
            # ... setup app and state ...
            server_task = asyncio.create_task(self._serve_http(sock=sock))
            await self._wait_and_warmup()
            health_check_task = asyncio.create_task(self._health_check_monitor())
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

## 5. 配置与集成

- 支持通过 `configs.yaml` 配置多实例、多模型。
- 详见主项目文档和示例。
