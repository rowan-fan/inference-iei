#!/bin/bash

# Parse command line arguments and set environment variables
# Command line arguments have higher priority than environment variables
# Arguments:
#   $@ - All command line arguments
# Returns:
#   Sets environment variables based on command line arguments
replace_environments() {
  local param_name
  local param_value
  local supported_params=(port gateway-config)
  
  while [ -n "$1" ]; do
    param_name="${1#--}"  # Remove leading --
    
    # Check if parameter is supported
    local is_supported=0
    for p in "${supported_params[@]}"; do
      if [ "$param_name" = "$p" ]; then
        is_supported=1
        break
      fi
    done
    
    if [ $is_supported -eq 1 ]; then
      shift
      param_value="$1"
      # Convert param name to env var format (lowercase to uppercase, dash to underscore)
      param_name=$(echo "$param_name" | tr '[:lower:]-' '[:upper:]_')
      export "${param_name}"="$param_value"
      echo "Setting $param_name=$param_value"
    else
      echo "Error: Unknown parameter '$1'"
      echo "Supported parameters:"
      printf "  --%s\n" "${supported_params[@]}"
      exit 1
    fi
    shift
  done
}


# Initialize MACA and CUDA environment
init_maca_environment() {
    echo "Initializing MACA and CUDA environment..."

    # 1. MACA related paths
    export MACA_PATH="${MACA_PATH:-/opt/maca}"
    export MACA_CLANG_PATH="$MACA_PATH/mxgpu_llvm/bin"

    # 2. CUDA related paths
    export CUDA_PATH="$MACA_PATH/tools/cu-bridge"
    export CUCC_PATH="$MACA_PATH/tools/cu-bridge"

    # 3. Update PATH
    export PATH="$MACA_PATH/bin:\
$MACA_PATH/mxgpu_llvm/bin:\
$MACA_PATH/ompi/bin:\
$MACA_PATH/ucx/bin:\
$CUDA_PATH/bin:\
$CUDA_PATH/tools:\
$PATH"

    # 4. Update LD_LIBRARY_PATH
    export LD_LIBRARY_PATH="$MACA_PATH/mxgpu_llvm/lib:\
$MACA_PATH/ompi/lib:\
$MACA_PATH/lib:\
$MACA_PATH/ucx/lib:\
/opt/conda/envs/pytorch/lib/python3.8/site-packages/torch_tensorglue/csrc/lib:\
/opt/conda/envs/pytorch/lib/python3.8/site-packages/torch/lib:\
$LD_LIBRARY_PATH"

    # 5. Validate critical paths
    local required_dirs=(
        "$MACA_PATH"
        "$CUDA_PATH"
        "$MACA_PATH/mxgpu_llvm"
        "$MACA_PATH/ompi"
        "$MACA_PATH/ucx"
    )

    for dir in "${required_dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            echo "Error: Required directory not found: $dir"
            return 1
        fi
    done

    # 6. Source TensorGlue backend configuration
    if [ -d "/etc/tensorglue_backend" ]; then
        local current_dir=$(pwd)
        cd /etc/tensorglue_backend
        if [ -f "env.sh" ]; then
            source env.sh
        else
            echo "Warning: TensorGlue env.sh not found"
        fi
        cd "$current_dir"
    else
        echo "Warning: TensorGlue backend directory not found"
    fi

    echo "MACA and CUDA environment initialized successfully"
    return 0
}


# Initialize and activate conda environment if it exists
setup_conda_environment() {
  # Return early if CONDA_ENV is not set
  if [ -z "$CONDA_ENV" ]; then
    echo "No conda environment specified, skipping conda activation"
    return 0
  fi

  # Get conda initialization script path
  local CONDA_PATH="${CONDA_PATH:-/opt/conda}"
  local CONDA_SETUP="$CONDA_PATH/etc/profile.d/conda.sh"

  # Skip if conda setup script doesn't exist
  if [ ! -f "$CONDA_SETUP" ]; then
    echo "Conda setup script not found at $CONDA_SETUP, skipping conda activation"
    return 0
  fi

  echo "Setting up conda environment..."

  # Initialize conda
  source "$CONDA_SETUP"

  # Try to activate the specified environment
  echo "Activating conda environment: $CONDA_ENV"
  if ! conda activate "$CONDA_ENV" 2>/dev/null; then
    echo "Warning: Failed to activate conda environment $CONDA_ENV, continuing without conda environment"
    return 0
  fi

  echo "Successfully activated conda environment: $CONDA_ENV"
}


set_mx_cuda_visible_deivces() {
  # Get GPU count and set CUDA_VISIBLE_DEVICES
  local gpu_count=0
  if [ -d "/dev/dri" ]; then
    gpu_count=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)
  fi

  if [ "$gpu_count" -gt 0 ]; then
    # Create comma-separated list of GPU indices (0 to gpu_count-1)
    gpu_list=$(seq -s ',' 0 $((gpu_count-1)))
    export CUDA_VISIBLE_DEVICES=$gpu_list
    echo "Setting CUDA_VISIBLE_DEVICES=$gpu_list"
  # else
  #   echo "No MX GPUs found, running without GPU acceleration"
  #   export CUDA_VISIBLE_DEVICES=""
  fi
}

set_mx_cuda_visible_deivces

# Start ichat.gateway
# Args:
#   LOG_FILE: The log file path for server output, default is gateway.log
# Returns: None
start_ichat_gateway() {
  # Construct arguments for ichat.gateway
  if [ -z "$GATEWAY_CONFIG" ]; then
    echo "Warning: --gateway-config not be provided. Using default config file: /app/ichat/config.yaml"
    GATEWAY_CONFIG="/app/ichat/config.yaml"
  fi

  local gateway_args="--config ${GATEWAY_CONFIG}"

  local server_command="python3 -m ichat.gateway $gateway_args"

  echo "Starting ichat.gateway..."
  echo "Gateway log file: ${1:-gateway.log}"
  echo "Executing command: $server_command"

  $server_command 2>&1 | tee "${1:-gateway.log}" | sed 's/^/[ICHAT_GATEWAY] /' &

  local server_pid=$!

  sleep 2
  if ps -p $server_pid > /dev/null; then
    echo "ichat.gateway started successfully with PID $server_pid"
  else
    echo "Failed to start ichat.gateway"
    exit 1
  fi
}

# Wait for ichat.gateway server to be ready
# Args:
#   PORT: The port number for the server, default is 4000
#   MAX_RETRIES: Maximum number of retry attempts, default is 120
#   RETRY_INTERVAL: Time interval between retries in seconds, default is 2
# Returns: None
wait_for_ichat_gateway() {
  local port=${PORT:-4000}
  local URL="http://localhost:${port}/v1/models"
  local retries=0
  local max_retries=${MAX_RETRIES:-120}
  local retry_interval=${RETRY_INTERVAL:-2}

  echo "Checking ichat.gateway server status at $URL"
  while [ $retries -lt $max_retries ]; do
    if response=$(curl --max-time 2 -s -o /dev/null -w "%{http_code}" "$URL" 2>/dev/null); then
      if [ "$response" -eq 200 ]; then
        echo "Successfully connected to ichat.gateway server at $URL"
        return 0
      fi
    fi

    retries=$((retries + 1))
    echo "Waiting for ichat.gateway server to be ready (attempt $retries/$max_retries)..."
    sleep $retry_interval
  done

  echo "Error: Failed to connect to ichat.gateway server after $max_retries attempts"
  exit 1
}

# Helper function to add argument if environment variable exists
add_arg() {
  local var_name=$1
  local arg_name=$2
  local var_value=${!var_name}
  
  if [ -n "$var_value" ]; then
    echo "Parameter found: $arg_name"
    args="$args --$arg_name $var_value"
  else
    echo "Parameter missing: $arg_name" 
  fi
}

# Step 1: Parse command-line arguments and replace environments
replace_environments "$@"

if [ -n "$CONDA_ENV" ]; then
  # Step 2: Setup conda environment
  setup_conda_environment

  # Step 3: Initialize MACA environment
  init_maca_environment
fi

# Step 2: Start the ichat gateway server
start_ichat_gateway

# Step 3: Wait for the ichat gateway server to be ready
wait_for_ichat_gateway

# The following services are assumed to be managed by the gateway or run separately.
# If they need to be started here, ensure the logic is correct.

# Step 4: Check if the sensitive model service should be started and start it if enabled
if [ "${SENSITIVE_MODEL_ENABLE:-false}" = true ]; then
  echo "Starting sensitive service"
  python3 sensitive_server/sensitive_models_filtering_api.py --port "${SENSITIVE_SVC_PORT:-39998}" --model_path "${SENSITIVE_MODEL_PATH:-/mnt/inaisfs/loki/bussiness/embedding-models/Security_semantic_filtering}"  2>&1 | while IFS= read -r line; do
    echo "[SENSITIVE] $line"
  done &
fi

# Step 5: Check if the PDF parser service should be started and start it if enabled
if [ "${PDF_PARSER_ENABLE:-false}" = true ]; then
  echo "Starting PDF parser service"
  python3 epaipdfparser/batch_analyze.py --port "${pdf_parser_port:-8877}" --device-type "${pdf_parser_device:-gpu}" --model-path "${pdf_parser_model_path:-/mnt/inaisfs/loki/bussiness/custom-models}"  2>&1 | while IFS= read -r line; do
    echo "[PDF_PARSER] $line"
  done &
fi

# Step 6: Wait for all background processes to complete
wait

# engine can be llama.cpp, vLLM, transformers, SGLang
