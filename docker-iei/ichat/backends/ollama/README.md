# iChat Ollama Backend 设计文档

本文件详细阐述 iChat `OllamaBackend` 的设计理念、架构和实现。该后端是 iChat Worker 与 [Ollama](https://ollama.com/) 推理引擎之间的核心适配层，通过 `ollama-python` 库进行交互。

## 1. 目标

1.  **封装 Ollama 交互**：基于 `ollama-python` 库，将与 Ollama 服务的交互（如模型检查、拉取、创建）封装成可编程的 Python 类。
2.  **模型缓存优先**：在启动模型前，优先检查 Ollama 本地是否已存在该模型，如果存在则直接使用，避免重复下载。
3.  **本地模型导入**：支持从本地文件路径导入模型，兼容 GGUF 和 SafeTensors 格式（通过 Ollama 的 `Modelfile` 机制）。
4.  **无缝集成**：遵循 `BaseBackend` 接口，使得主服务可以像对待其他后端（vLLM, SGLang）一样，通过配置无缝切换和使用 Ollama 后端。
5.  **配置驱动**：支持通过 `configs.yaml` 灵活配置 Ollama 服务地址、模型名称、本地模型路径等参数。

## 2. 设计原则

- **非侵入式**：完全通过 `ollama-python` API 与 Ollama 服务通信，不依赖任何 Ollama 内部实现或修改其源码。
- **服务解耦**：`OllamaBackend` 假定 Ollama 服务已独立运行。后端的职责是作为客户端，管理和确保模型在 Ollama 中处于就绪状态，而非启动或管理 Ollama 服务本身。
- **鲁棒性**：内置连接检查和模型准备逻辑，确保在与上游服务交互前，模型已成功加载。
- **代码重用**：最大限度重用 `ollama-python` 库提供的功能，简化交互逻辑。

## 3. 类架构

### 3.1 `__init__`
- 接收 `framework_args` 和 `backend_argv`。
- 解析参数，生成 Ollama 专用的配置，例如 `model`, `host`, `model_path` 等。
- 初始化 `ollama.AsyncClient`，用于与 Ollama 服务进行异步通信。

### 3.2 `async run(self)`
- 调用 `_ensure_model_ready()` 确保模型在 Ollama 中可用。
- 成功后，设置 `backend_ready_event`，通知上层服务后端已就绪。
- 进入等待状态，直到接收到清理信号。
- `finally` 块中调用 `cleanup()`。

### 3.3 `async _ensure_model_ready(self)`
- **检查缓存**：调用 `self.client.list()` 获取 Ollama 中所有可用模型的列表。
- **匹配模型**：检查所需模型（由 `self.ollama_args.model` 定义）是否存在于列表中。如果存在，则直接返回。
- **导入或拉取**：
    - 如果模型不存在，检查是否配置了本地模型路径 (`self.ollama_args.model_path`)。
    - 如果配置了本地路径，则调用 `_create_from_file()` 从本地文件创建模型。
    - 如果未配置本地路径，则调用 `self.client.pull()` 从 Ollama Hub 拉取模型。

### 3.4 `async _create_from_file(self, model_name, model_path)`
- 根据 `model_path` 动态生成一个 `Modelfile` 的内容。
- `Modelfile` 的核心指令是 `FROM {model_path}`，它指示 Ollama 从指定的本地文件系统路径加载模型。
- 调用 `self.client.create(model=model_name, modelfile=...)` 来根据 `Modelfile` 创建并注册一个新模型。

### 3.5 `cleanup(self)`
- 该方法用于在服务关闭时执行必要的清理工作。对于 `OllamaBackend`，由于不直接管理 Ollama 进程，此方法可能为空或只包含取消内部任务的逻辑。

## 4. 伪代码架构

```python
import ollama
import asyncio

class OllamaBackend(BaseBackend):
    def __init__(self, framework_args, backend_argv, backend_ready_event):
        self.ollama_args = self._parse_ollama_args(backend_argv)
        self.client = ollama.AsyncClient(host=self.ollama_args.host)
        self.backend_ready_event = backend_ready_event
        self.shutdown_event = asyncio.Event()
        # ...

    def _parse_ollama_args(self, backend_argv):
        # 解析 backend_argv 和 framework_args
        # 返回包含 model, host, model_path 等的配置对象
        pass

    async def run(self):
        try:
            await self._ensure_model_ready()
            self.backend_ready_event.set()
            await self.shutdown_event.wait()
        except Exception as e:
            print(f"Ollama backend failed: {e}")
        finally:
            self.cleanup()

    async def _ensure_model_ready(self):
        model_name = self.ollama_args.model
        try:
            response = await self.client.list()
            if any(m['name'] == model_name for m in response.get('models', [])):
                print(f"Model '{model_name}' already exists in Ollama.")
                return
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Ollama at {self.ollama_args.host}: {e}")

        if hasattr(self.ollama_args, 'model_path') and self.ollama_args.model_path:
            print(f"Creating model '{model_name}' from path '{self.ollama_args.model_path}'...")
            await self._create_from_file(model_name, self.ollama_args.model_path)
        else:
            print(f"Pulling model '{model_name}' from Ollama Hub...")
            await self.client.pull(model_name)

    async def _create_from_file(self, model_name, model_path):
        modelfile = f'FROM {model_path}'
        await self.client.create(model=model_name, modelfile=modelfile)
        print(f"Successfully created model '{model_name}'.")

    def cleanup(self):
        print("Cleaning up Ollama backend.")
        # No process to kill, just cancel tasks if any.
```

## 5. 配置与集成

Ollama 后端通过 `configs.yaml` 进行配置。一个典型的配置示例如下：

```yaml
- name: "ollama-backend"
  framework: "ollama"
  # ... other framework args ...
  backend_argv:
    - "--model=llama3"
    - "--host=http://127.0.0.1:11434"
    # - "--model-path=/path/to/models/llama3.gguf" # (可选) 如果使用本地模型文件
```

- **`framework: "ollama"`**：指定使用 Ollama 后端。
- **`--model`**: 指定要在 Ollama 中使用的模型名称。
- **`--host`**: Ollama 服务的地址。
- **`--model-path`**: (可选) 如果要从本地 GGUF 或 SafeTensors 文件创建模型，请提供此路径。如果提供此项，后端将忽略从远端拉取。

关于 GPU/CPU 部署，Ollama 服务自身会管理硬件资源。`OllamaBackend` 不直接控制设备分配。请参考 Ollama 官方文档来配置 Ollama 服务以使用特定的 GPU。