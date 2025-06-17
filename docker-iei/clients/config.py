"""
Configuration file for internal RAG service client.
Contains model configurations and YAML templates.
"""

rag_svc_host = "internal-rag-service-predictor-default.inais-inference-inner.svc.cluster.local"
vl_svc_host = "internal-vl-service-predictor-default.inais-inference-inner.svc.cluster.local"
ollama_svc_host = "internal-ollama-service-predictor-default.inais-inference-inner.svc.cluster.local"
custom_svc_host = "internal-custom-service-predictor-default.inais-inference-inner.svc.cluster.local"

# Model configurations for registration with Epaichat service
# The last two must be vl-cpu and vl-gpu models.
request_list = [
  {
  "name":"bce-embedding-base_v1",                             
  "type":3,            
  "model_key":"bce-embedding-base_v1",
  "api_key":"xxxxxxxx", 
  "access_path":"http://{}:8080/v1", 
  "logo":"",
  "protocol":"OpenAI",
  "all_user": False,  
  "usernames":[],
  "manufacturer":"有道翻译", 
  "parameter_size":"768维",
  "expert_fields":["专业领域"],
  "language":"中英", 
  "task_type":"embedding",
  "status_message":"Not Ready", 
  "default_settings":[]
},{
  "name":"bge-large-zh-v1.5",                             
  "type":3,            
  "model_key":"bge-large-zh-v1.5",
  "api_key":"xxxxxxxx", 
  "access_path":"http://{}:8080/v1", 
  "logo":"",
  "protocol":"OpenAI",
  "all_user": False,  
  "usernames":[],
  "manufacturer":"智源", 
  "parameter_size":"1024维",
  "expert_fields":["通用领域"],
  "language":"中文", 
  "task_type":"embedding",
  "status_message":"Not Ready", 
  "default_settings":[]
},{
  "name":"Yuan-embedding-1.0",                             
  "type":3,            
  "model_key":"Yuan-embedding-1.0",
  "api_key":"xxxxxxxx", 
  "access_path":"http://{}:8080/v1", 
  "logo":"",
  "protocol":"OpenAI",
  "all_user": False,  
  "usernames":[],
  "manufacturer":"浪潮信息", 
  "parameter_size":"1792维",
  "expert_fields":["语义搜索"],
  "language":"中文", 
  "task_type":"embedding",
  "status_message":"Not Ready", 
  "default_settings":[]
},{
  "name":"text2vec-base-chinese",                             
  "type":3,            
  "model_key":"text2vec-base-chinese",
  "api_key":"xxxxxxxx", 
  "access_path":"http://{}:8080/v1", 
  "logo":"",
  "protocol":"OpenAI",
  "all_user": False,  
  "usernames":[],
  "manufacturer":"个人", 
  "parameter_size":"768维",
  "expert_fields":["语义搜索"],
  "language":"中文", 
  "task_type":"embedding",
  "status_message":"Not Ready", 
  "default_settings":[]
},{
  "name":"bge-reranker-v2-m3",                             
  "type":4,            
  "model_key":"bge-reranker-v2-m3",
  "api_key":"xxxxxxxx", 
  "access_path":"http://{}:8080/v1", 
  "logo":"",
  "protocol":"OpenAI",
  "all_user": False,  
  "usernames":[],
  "manufacturer":"智源", 
  "parameter_size":"1024维",
  "expert_fields":["通用领域"],
  "language":"中文", 
  "task_type":"rerank",
  "status_message":"Not Ready", 
  "default_settings":[]
},{
  "name":"security-semantic-filtering",                             
  "type":2,            
  "model_key":"sensitive_filter",
  "api_key":"", 
  "access_path":"http://{}:39998/sensitive/evaluate", 
  "logo":"",
  "protocol":"OpenAI",
  "all_user": False,  
  "usernames":[],
  "manufacturer":"智源", 
  "parameter_size":"768维",
  "expert_fields":["通用领域"],
  "language":"中文", 
  "task_type":"classification",
  "status_message":"Not Ready", 
  "default_settings":[]
},{
  "name":"qwen2-vl-instruct",                             
  "type":5,            
  "model_key":"qwen2-vl-instruct",
  "api_key":"", 
  "access_path":"http://{}:8080/v1", 
  "logo":"",
  "protocol":"OpenAI",
  "all_user": False,  
  "usernames":[],
  "manufacturer":"通义千问", 
  "parameter_size":"32768维",
  "expert_fields":["通用领域"],
  "language":"中文,英文", 
  "task_type":"vision",
  "status_message":"Not Ready", 
  "default_settings":[]
},{
  "name":"minicpm-v",                             
  "type":5,            
  "model_key":"minicpm-v:8b-2.6-q4_0",
  "api_key":"ollama", 
  "access_path":"http://{}:11434/v1", 
  "logo":"",
  "protocol":"OpenAI",
  "all_user": False,  
  "usernames":[],
  "manufacturer":"OpenBMB", 
  "parameter_size":"32768维",
  "expert_fields":["通用领域"],
  "language":"中文,英文", 
  "task_type":"vision",
  "status_message":"Not Ready", 
  "default_settings":[]
}
]

full_request_list = [dict(item) for item in request_list]

union_request_list = []
for item in request_list:
    model = dict(item)
    if model.get("api_key") == "ollama":
        model["access_path"] = model["access_path"].format(ollama_svc_host)
    else:
        model["access_path"] = model["access_path"].format(custom_svc_host)
    union_request_list.append(model)

separate_request_list = []
for item in request_list:
    model = dict(item)
    if model.get("api_key") == "ollama":
        model["access_path"] = model["access_path"].format(ollama_svc_host)
    else:
        model["access_path"] = model["access_path"].format(custom_svc_host)
    separate_request_list.append(model)


# Configuration template for RAG service
rag_config_yaml = """
node-affinity: []
use-any-nodes: "false"
replicas: 1
num-cpus: 8
memory: "32G"
env:
  NVIDIA_VISIBLE_DEVICES: none
"""

# Configuration template for vision-language service
vl_config_yaml = """
node-affinity: []
use-any-nodes: "false"
replicas: 1
num-cpus: 6
memory: "16G"
env:
  NVIDIA_VISIBLE_DEVICES: none
"""

# Configuration template for ollama service
ollama_config_yaml = """
node-affinity: []
use-any-nodes: "false"
replicas: 1
num-cpus: 6
memory: "16G"
env:
  NVIDIA_VISIBLE_DEVICES: none
"""

custom_config_yaml = """
node-affinity: []
use-any-nodes: "false"
replicas: 1
num-cpus: 8
memory: "16G"
env:
  NVIDIA_VISIBLE_DEVICES: none
  pdf_parser_device: cpu
"""

# Configuration template for RAG service
rag_config_gpu_yaml = """
node-affinity: []
use-any-nodes: "false"
replicas: 1
num-cpus: 8
memory: "32G"
env:
  NVIDIA_VISIBLE_DEVICES: none
"""

# Configuration template for vision-language service
vl_config_gpu_yaml = """
node-affinity: []
use-any-nodes: "false"
replicas: 1
num-cpus: 6
num-gpus: 1
gpu-type: metax-tech.com/gpu
memory: "16G"
env:
  NVIDIA_VISIBLE_DEVICES: none
"""

# Configuration template for ollama service
ollama_config_gpu_yaml = """
node-affinity: []
use-any-nodes: "false"
replicas: 1
num-cpus: 6
memory: "16G"
env:
  NVIDIA_VISIBLE_DEVICES: none
"""

custom_config_gpu_yaml = """
node-affinity: []
use-any-nodes: "false"
replicas: 1
num-cpus: 8
num-gpus: 1
memory: "16G"
env:
  NVIDIA_VISIBLE_DEVICES: none
  pdf_parser_device: gpu
"""

custom_unoin_config_gpu_yaml = """
node-affinity: []
use-any-nodes: "false"
replicas: 1
num-cpus: 8
num-gpus: 1
memory: "16G"
env:
  NVIDIA_VISIBLE_DEVICES: none
  pdf_parser_device: gpu
  SENSITIVE_MODEL_ENABLE: "true"
"""

# Namespace configuration template
ns_yaml = """
apiVersion: v1
kind: Namespace
metadata:
  labels:
    istio-injection: enabled
  name: inais-inference-inner
"""

# YAML template for RAG service deployment
rag_svc_yaml = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    serving.kserve.io/autoscalerClass: hpa
    serving.kserve.io/deploymentMode: RawDeployment
    serving.kserve.io/isMultiPorts: "false"
    serving.kserve.io/metric: cpu
    serving.kserve.io/targetUtilizationPercentage: "90"
    autoscaling.knative.dev/target: "1"
    sidecar.istio.io/inject: "true"
  name: internal-rag-service
  namespace: inais-inference-inner
spec:
  predictor:
    affinity:
      nodeAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          preference:
            matchExpressions:
            - key: kubernetes.io/hostname
              operator: In
              values: AIS_NODE_AFFINITY
    nodeSelector:
      inf_use: "1"
    maxReplicas: 1
    minReplicas: 1
    model:
      storageUri: hostpath://{{data_loki_dir}}/loki/bussiness/embedding-models
      imagePullPolicy: Always
      modelFormat:
        name: llm
      env:
        - name: MODEL_CONFIG
          value: {{data_loki_dir}}/loki/bussiness/embedding-models/Param-rags.json
        - name: SENSITIVE_MODEL_ENABLE
          value: "true"
        - name: SENSITIVE_MODEL_PATH
          value: {{data_loki_dir}}/loki/bussiness/embedding-models/Security_semantic_filtering
      livenessProbe:
        exec:
          command:
          - python3
          - /workspace/probe.py
        failureThreshold: 3
        initialDelaySeconds: 1200
        periodSeconds: 60
        successThreshold: 1
        timeoutSeconds: 180
      name: kserve-container
      ports:
      - containerPort: 8080
        name: http1
        protocol: TCP
      - containerPort: 39998
        name: http8080
        protocol: TCP
      readinessProbe:
        exec:
          command:
          - python3
          - /workspace/probe.py
        failureThreshold: 20
        initialDelaySeconds: 60
        periodSeconds: 60
        successThreshold: 1
        timeoutSeconds: 180
      resources:
        limits:
          cpu: "8"
          memory: 16G
        requests:
          cpu: "8"
          memory: 16G
"""

# YAML template for vision-language service deployment
vl_cpu_svc_yaml = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    serving.kserve.io/autoscalerClass: hpa
    serving.kserve.io/deploymentMode: RawDeployment
    serving.kserve.io/isMultiPorts: "false"
    serving.kserve.io/metric: cpu
    serving.kserve.io/targetUtilizationPercentage: "90"
    autoscaling.knative.dev/target: "1"
    sidecar.istio.io/inject: "true"
  name: internal-vl-service
  namespace: inais-inference-inner
spec:
  predictor:
    affinity:
      nodeAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          preference:
            matchExpressions:
            - key: kubernetes.io/hostname
              operator: In
              values: AIS_NODE_AFFINITY
    nodeSelector:
      inf_use: "1"
    maxReplicas: 1
    minReplicas: 1
    containers:
    - image: {{keepalived_vip}}:{{harbor_host_port}}/inais/ollama:0.5.1-minicpmv-2.6
      imagePullPolicy: Always
      name: kserve-container
      ports:
      - containerPort: 11434
        name: http1
        protocol: TCP
      livenessProbe:
        httpGet:
          path: /
          port: 11434
        failureThreshold: 2
        initialDelaySeconds: 60
        periodSeconds: 20
        successThreshold: 1
        timeoutSeconds: 3
      readinessProbe:
        httpGet:
          path: /
          port: 11434
        failureThreshold: 5
        initialDelaySeconds: 30
        periodSeconds: 20
        successThreshold: 1
        timeoutSeconds: 3
      resources:
        limits:
          cpu: "8"
          memory: 16G
        requests:
          cpu: "8"
          memory: 16G
"""

# YAML template for vision-language service deployment
vl_gpu_svc_yaml = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    serving.kserve.io/autoscalerClass: hpa
    serving.kserve.io/deploymentMode: RawDeployment
    serving.kserve.io/isMultiPorts: "false"
    serving.kserve.io/metric: cpu
    serving.kserve.io/targetUtilizationPercentage: "90"
    autoscaling.knative.dev/target: "1"
    sidecar.istio.io/inject: "true"
  name: internal-vl-service
  namespace: inais-inference-inner
spec:
  predictor:
    affinity:
      nodeAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          preference:
            matchExpressions:
            - key: kubernetes.io/hostname
              operator: In
              values: AIS_NODE_AFFINITY
    nodeSelector:
      inf_use: "1"
    maxReplicas: 1
    minReplicas: 1
    model:
      storageUri: hostpath://{{data_loki_dir}}/loki/bussiness/vl-models
      imagePullPolicy: Always
      modelFormat:
        name: llm
      env:
        - name: MODEL_CONFIG
          value: {{data_loki_dir}}/loki/bussiness/vl-models/Param-vl.json
        - name: open_acceleration
          value: "True"
      ports:
      - containerPort: 8080
        name: http1
        protocol: TCP
      livenessProbe:
        exec:
          command:
          - python3
          - /workspace/probe.py
        failureThreshold: 3
        initialDelaySeconds: 1200
        periodSeconds: 60
        successThreshold: 1
        timeoutSeconds: 180
      readinessProbe:
        exec:
          command:
          - python3
          - /workspace/probe.py
        failureThreshold: 20
        initialDelaySeconds: 60
        periodSeconds: 60
        successThreshold: 1
        timeoutSeconds: 180
      resources:
        limits:
          cpu: "8"
          memory: 16G
        requests:
          cpu: "8"
          memory: 16G
"""

# YAML template for vision-language service deployment
vl_svc_yaml = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    serving.kserve.io/autoscalerClass: hpa
    serving.kserve.io/deploymentMode: RawDeployment
    serving.kserve.io/isMultiPorts: "false"
    serving.kserve.io/metric: cpu
    serving.kserve.io/targetUtilizationPercentage: "90"
    autoscaling.knative.dev/target: "1"
    sidecar.istio.io/inject: "true"
  name: internal-vl-service
  namespace: inais-inference-inner
spec:
  predictor:
    affinity:
      nodeAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          preference:
            matchExpressions:
            - key: kubernetes.io/hostname
              operator: In
              values: AIS_NODE_AFFINITY
    nodeSelector:
      inf_use: "1"
    maxReplicas: 1
    minReplicas: 1
    model:
      storageUri: hostpath://{{data_loki_dir}}/loki/bussiness/vl-models
      imagePullPolicy: Always
      modelFormat:
        name: llm
      env:
        - name: MODEL_CONFIG
          value: {{data_loki_dir}}/loki/bussiness/vl-models/Param-vl.json
        - name: open_acceleration
          value: "True"
      ports:
      - containerPort: 8080
        name: http1
        protocol: TCP
      livenessProbe:
        exec:
          command:
          - python3
          - /workspace/probe.py
        failureThreshold: 3
        initialDelaySeconds: 1200
        periodSeconds: 60
        successThreshold: 1
        timeoutSeconds: 180
      readinessProbe:
        exec:
          command:
          - python3
          - /workspace/probe.py
        failureThreshold: 20
        initialDelaySeconds: 60
        periodSeconds: 60
        successThreshold: 1
        timeoutSeconds: 180
      resources:
        limits:
          cpu: "8"
          memory: 16G
        requests:
          cpu: "8"
          memory: 16G
"""

# YAML template for Ollama service deployment
ollama_svc_yaml = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    serving.kserve.io/autoscalerClass: hpa
    serving.kserve.io/deploymentMode: RawDeployment
    serving.kserve.io/isMultiPorts: "false"
    serving.kserve.io/metric: cpu
    serving.kserve.io/targetUtilizationPercentage: "90"
    autoscaling.knative.dev/target: "1"
    sidecar.istio.io/inject: "true"
  name: internal-ollama-service
  namespace: inais-inference-inner
spec:
  predictor:
    affinity:
      nodeAffinity:
        requiredDuringSchedulingIgnoredDuringExecution:
        - weight: 100
          preference:
            matchExpressions:
            - key: kubernetes.io/hostname
              operator: In
              values: AIS_NODE_AFFINITY
    nodeSelector:
      inf_use: "1"
    maxReplicas: 1
    minReplicas: 1
    containers:
    - image: {{keepalived_vip}}:{{harbor_host_port}}/inais/ollama:0.3.14
      imagePullPolicy: Always
      name: kserve-container
      ports:
      - containerPort: 11434
        name: http1
        protocol: TCP
      livenessProbe:
        httpGet:
          path: /
          port: 11434
        failureThreshold: 2
        initialDelaySeconds: 60
        periodSeconds: 20
        successThreshold: 1
        timeoutSeconds: 3
      readinessProbe:
        httpGet:
          path: /
          port: 11434
        failureThreshold: 5
        initialDelaySeconds: 30
        periodSeconds: 20
        successThreshold: 1
        timeoutSeconds: 3
      resources:
        limits:
          cpu: "8"
          memory: 16G
        requests:
          cpu: "8"
          memory: 16G
"""
# embedding-models/Param-rags.json
custom_svc_yaml_template = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  annotations:
    autoscaling.knative.dev/target: '1'
    serving.kserve.io/autoscalerClass: hpa
    serving.kserve.io/deploymentMode: RawDeployment
    serving.kserve.io/isMultiPorts: 'false'
    serving.kserve.io/metric: cpu
    serving.kserve.io/targetUtilizationPercentage: '90'
    sidecar.istio.io/inject: 'true'
  name: internal-custom-service
  namespace: inais-inference-inner
spec:
  predictor:
    maxReplicas: 1
    minReplicas: 1
    model:
      env:
      - name: MODEL_CONFIG
        value: {{data_loki_dir}}/loki/bussiness/{}
      - name: PDF_PARSER_ENABLE
        value: 'true'
      - name: SENSITIVE_MODEL_PATH
        value: {{data_loki_dir}}/loki/bussiness/embedding-models/Security_semantic_filtering
      - name: pdf_parser_port
        value: "8877"
      - name: pdf_parser_device
        value: "gpu"
      - name: pdf_parser_model_path
        value: "{{data_loki_dir}}/loki/bussiness/custom-models"
      - name: YOLO_LAYOUT_BASE_BATCH_SIZE
        value: "1"
      - name: MFD_BASE_BATCH_SIZE
        value: "1"
      - name: MFR_BASE_BATCH_SIZE
        value: "16"
      - name: MAX_GPU_CUR_MEM
        value: "12"
      - name: BATCH_RATIO
        value: "4"
      - name: MINERU_MIN_EPOCH_SIZE
        value: "200"  
      imagePullPolicy: Always
      livenessProbe:
        exec:
          command:
          - python3
          - /workspace/probe.py
        failureThreshold: 3
        initialDelaySeconds: 1200
        periodSeconds: 60
        successThreshold: 1
        timeoutSeconds: 180
      modelFormat:
        name: llm
      name: kserve-container
      ports:
      - containerPort: 8080
        name: http1
        protocol: TCP
      - containerPort: 39998
        name: http8080
        protocol: TCP
      - containerPort: 8877
        name: http8877
        protocol: TCP
      readinessProbe:
        exec:
          command:
          - python3
          - /workspace/probe.py
        failureThreshold: 20
        initialDelaySeconds: 60
        periodSeconds: 60
        successThreshold: 1
        timeoutSeconds: 180
      resources:
        limits:
          cpu: "8"
          memory: 16G
        requests:
          cpu: "8"
          memory: 16G
      runtime: llmserver
      storageUri: hostpath://{{data_loki_dir}}/loki/bussiness
""" 

custom_svc_yaml = custom_svc_yaml_template.format("custom-models/Param-custom.json")
custom_unoin_svc_yaml = custom_svc_yaml_template.format("Param-unoin.json")

param_rags = [
    {
        "modelType":"embedding",
        "modelName": "bce-embedding-base_v1",
        "modelUid": "bce-embedding-base_v1",
        "dimensions": 768,
        "maxToken": 512,
        "modelLang": [
            "中文",
            "英文"
        ],
        "modelUri": "/mnt/inaisfs/loki/bussiness/embedding-models/bce-embedding-base_v1",
        "modelPath": "/mnt/inaisfs/loki/bussiness/embedding-models/bce-embedding-base_v1",
        "configVersion": "epai1230"
    },
    {
        "modelType":"embedding",
        "modelName": "bge-large-zh-v1.5",
        "modelUid": "bge-large-zh-v1.5",
        "dimensions": 768,
        "maxToken": 512,
        "modelLang": [
            "中文",
            "英文"
        ],
        "modelUri": "/mnt/inaisfs/loki/bussiness/embedding-models/bge-large-zh-v1.5",
        "modelPath": "/mnt/inaisfs/loki/bussiness/embedding-models/bge-large-zh-v1.5",
        "configVersion": "epai1230"
    },
    {
        "modelType":"embedding",
        "modelName": "text2vec-base-chinese",
        "modelUid": "text2vec-base-chinese",
        "dimensions": 768,
        "maxToken": 512,
        "modelLang": [
            "中文",
            "英文"
        ],
        "modelUri": "/mnt/inaisfs/loki/bussiness/embedding-models/text2vec-base-chinese",
        "modelPath": "/mnt/inaisfs/loki/bussiness/embedding-models/text2vec-base-chinese",
        "configVersion": "epai1230"
    },
    {
        "modelType":"embedding",
        "modelName": "Yuan-embedding-1.0",
        "modelUid": "Yuan-embedding-1.0",
        "dimensions": 1792,
        "maxToken": 512,
        "modelLang": [
            "中文"
        ],
        "modelUri": "/mnt/inaisfs/loki/bussiness/embedding-models/Yuan-embedding-1.0",
        "modelPath": "/mnt/inaisfs/loki/bussiness/embedding-models/Yuan-embedding-1.0",
        "configVersion": "epai1230"
    },
    {
        "modelType":"rerank",
        "modelName": "bge-reranker-v2-m3",
        "modelUid": "bge-reranker-v2-m3",
        "type": "normal",
        "modelLang": [
            "中文",
            "英文",
            "多语言"
        ],
        "modelUri": "/mnt/inaisfs/loki/bussiness/embedding-models/bge-reranker-v2-m3",
        "modelPath": "/mnt/inaisfs/loki/bussiness/embedding-models/bge-reranker-v2-m3",
        "configVersion": "epai1230"
    }
]


param_vl = [
    {
        "modelType":"LLM",
        "modelFormat":"pytorch",
        "modelSizeInBillions": 7,
        "modelName": "qwen2-vl-instruct",
        "modelUid": "qwen2-vl-instruct",
        "contextLength": 8192,
        "quantizations": ["None"],
        "version": 1,
        "modelFamily":"qwen2-vl-instruct",
        "modelAbility":"chat",
        "gpu_idxs": [0],
        "configVersion": "epai1230",
        "modelUri": "/mnt/inaisfs/loki/bussiness/vl-models/Qwen2-VL-7B-Instruct",
        "modelPath": "/mnt/inaisfs/loki/bussiness/vl-models/Qwen2-VL-7B-Instruct",
        "kwargs": {"enforce_eager":"True"}
    }
]


param_union = [
    {
        "modelType":"LLM",
        "modelFormat":"pytorch",
        "modelSizeInBillions": 7,
        "modelName": "qwen2-vl-instruct",
        "modelUid": "qwen2-vl-instruct",
        "contextLength": 8192,
        "quantizations": ["None"],
        "version": 1,
        "modelFamily":"qwen2-vl-instruct",
        "modelAbility":"chat",
        "gpu_idxs": [0],
        "configVersion": "epai1230",
        "modelUri": "/mnt/inaisfs/loki/bussiness/vl-models/Qwen2-VL-7B-Instruct",
        "modelPath": "/mnt/inaisfs/loki/bussiness/vl-models/Qwen2-VL-7B-Instruct",
        "kwargs": {"enforce_eager":"True", "gpu_memory_utilization": 0.5}
    }
]

for item in param_rags:
    param_union.append(item)


rags_output_path = "{{data_loki_dir}}/loki/bussiness/embedding-models/Param-rags.json"
union_output_path = "{{data_loki_dir}}/loki/bussiness/Param-unoin.json"