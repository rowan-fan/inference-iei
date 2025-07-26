# iChat SGLang Backend 设计文档

本文件详细阐述 iChat `SGLangBackend` 的设计理念、架构和实现。该后端是 iChat Worker 与 SGLang 推理引擎之间的核心适配层。

## 1. 目标

1.  封装 SGLang 服务，将 SGLang 的 `launch_server` 启动和管理逻辑封装成可编程的 Python 类。
2.  独立参数解析，生成 `ServerArgs` 配置对象。
3.  精细化生命周期管理，支持 Tokenizer、Scheduler、Detokenizer 等核心组件的启动、运行和停止。
4.  无缝集成，主服务只需实例化 `SGLangBackend` 并调用 `run()`。
5.  内置子进程健康检查，关键组件失效时可快速失败并重启。

## 2. 设计原则

- **非侵入式**：绝不修改 SGLang 源码，所有集成在 `SGLangBackend` 内完成。
- **渐进式集成**：可先用顶层 `launch_server`，后逐步深入控制子进程、FastAPI、Uvicorn。
- **代码重用**：最大限度重用 SGLang 内部函数，避免 subprocess。
- **配置驱动**：所有 SGLang 支持参数均可配置。

## 3. 类架构

### 3.1 `__init__`
- 接收 `framework_args` 和 `backend_argv`，调用 `_parse_sglang_args` 生成 `ServerArgs`。
- 初始化 FastAPI 应用、Tokenzier 管理器、server_task、server_shutdown_event、subprocesses 等。

### 3.2 `_parse_sglang_args`
- 构建 ArgumentParser，解析 `backend_argv`。
- 合并 `framework_args`（如 host、port、log_level）。
- 生成并校验 `ServerArgs`。

### 3.3 `async run(self)`
- 创建并运行 `_run_server_worker()` 协程。
- finally 块调用 `cleanup()`。

### 3.4 `async _run_server_worker(self)`
- 启动前后记录子进程，识别 SGLang 引擎子进程。
- 启动引擎、设置全局状态、配置 FastAPI。
- 创建 server_task（Uvicorn），await `_wait_and_warmup()`。
- 预热成功后，创建 health_check_task（子进程监控）。
- await asyncio.wait 监控 server_task 和 health_check_task。
- finally 块取消所有待处理任务。

### 3.5 `_serve_http(self)`
- 配置 Uvicorn，获取关闭事件，禁用信号处理。

### 3.6 `_wait_and_warmup(self)`
- 轮询 `/get_model_info` 等待服务就绪。
- 发送预热请求（/generate 或 /encode）。
- 预热成功后 set server_ready。

### 3.7 `_health_check_monitor(self)`
- 定期检查所有 SGLang 子进程存活。
- 任一子进程终止则抛出 RuntimeError，触发优雅关闭。

### 3.8 `cleanup(self)`
- 请求 Uvicorn 关闭，取消主任务。
- 强制 kill 所有 SGLang 子进程，防止僵尸进程。

## 4. 伪代码架构

```python
class SGLangBackend(BaseBackend):
    def __init__(self, framework_args, backend_argv, backend_ready_event):
        self.sglang_args = self._parse_sglang_args(backend_argv)
        # ...

    def _parse_sglang_args(self, backend_argv):
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

## 5. 配置与集成

- 支持通过 `configs.yaml` 配置多实例、多模型。
- 详见主项目文档和示例。
