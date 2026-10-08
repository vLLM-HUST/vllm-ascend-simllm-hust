#!/usr/bin/env bash
set -euo pipefail

# Recreates the Ascend service configuration used by reproduce_qwen35_latency.sh.
# Requires the compatible vLLM Ascend launcher and the local Qwen W8A8 weights.
server_launcher="${SERVER_LAUNCHER:-/data/jxd/quant-Qwen3.5-35B-A3B/run_swe_server.sh}"
model_dir="${MODEL_DIR:-/data/jxd/Qwen3.5-35B-A3B-W8A8}"
runtime_dir="${PERF_ENV:-/data/jxd/envs/vllm-hust-v1-cann91}"
cann_env="${CANN_SET_ENV:-/data/jxd/Ascend-9.1.0/cann-9.1.0/set_env.sh}"
output_dir="${1:?usage: $0 OUTPUT_DIR}"
baseline_port="${BASELINE_PORT:-8107}"
plugin_port="${PLUGIN_PORT:-8106}"
baseline_device="${BASELINE_DEVICE:-7}"
plugin_device="${PLUGIN_DEVICE:-6}"
extra_args='--block-size 128 --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-seqs 4 --compilation-config {"cudagraph_mode":"PIECEWISE","max_cudagraph_capture_size":4}'

[[ -f "$server_launcher" ]] || { echo "Launcher missing: $server_launcher" >&2; exit 1; }
[[ -d "$model_dir" ]] || { echo "Model missing: $model_dir" >&2; exit 1; }
[[ -x "$runtime_dir/bin/vllm" ]] || { echo "vLLM missing: $runtime_dir" >&2; exit 1; }
[[ -f "$cann_env" ]] || { echo "CANN setup missing: $cann_env" >&2; exit 1; }
for port in "$baseline_port" "$plugin_port"; do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        echo "Port $port already has a service; use free ports" >&2
        exit 1
    fi
done
mkdir -p "$output_dir/baseline" "$output_dir/plugin"

wait_ready() {
    local port="$1"
    for _ in $(seq 1 480); do
        if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    echo "Service on port $port did not become ready" >&2
    return 1
}

# Load one model at a time to avoid shared-checkpoint I/O contention.
env -u VLLM_PLUGINS \
    ASCEND_RT_VISIBLE_DEVICES="$baseline_device" PORT="$baseline_port" \
    PERF_ENV="$runtime_dir" MODEL_PATH="$model_dir" CANN_SET_ENV="$cann_env" \
    LABEL=w8a8 SERVED_NAME=qwen35-a3b-w8a8 TP=1 MAX_MODEL_LEN=8192 \
    GPU_MEM_UTIL=0.85 PERF_FIX_KERNELS_SO=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
    VLLM_CACHE_ROOT="${output_dir}/baseline-cache" \
    VLLM_ASCEND_SIMLLM_ENABLED=0 LOG_DIR="${output_dir}/baseline" \
    EXTRA_SERVER_ARGS="$extra_args" \
    nohup bash "$server_launcher" > "${output_dir}/baseline-launch.log" 2>&1 &
wait_ready "$baseline_port"

env -u VLLM_PLUGINS \
    ASCEND_RT_VISIBLE_DEVICES="$plugin_device" PORT="$plugin_port" \
    PERF_ENV="$runtime_dir" MODEL_PATH="$model_dir" CANN_SET_ENV="$cann_env" \
    LABEL=w8a8 SERVED_NAME=qwen35-a3b-w8a8 TP=1 MAX_MODEL_LEN=8192 \
    GPU_MEM_UTIL=0.85 PERF_FIX_KERNELS_SO=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
    VLLM_CACHE_ROOT="${output_dir}/plugin-cache" \
    VLLM_ASCEND_SIMLLM_ENABLED=1 \
    VLLM_ASCEND_SIMLLM_HYBRID_REUSE_MODE=tail_anchor \
    VLLM_ASCEND_SIMLLM_HYBRID_RECOMPUTE_TOKENS=512 \
    LOG_DIR="${output_dir}/plugin" EXTRA_SERVER_ARGS="$extra_args" \
    nohup bash "$server_launcher" > "${output_dir}/plugin-launch.log" 2>&1 &
wait_ready "$plugin_port"

echo "Baseline: http://127.0.0.1:${baseline_port} (device ${baseline_device})"
echo "SimLLM:   http://127.0.0.1:${plugin_port} (device ${plugin_device})"
echo "Server PID files and logs: $output_dir/{baseline,plugin}/"
