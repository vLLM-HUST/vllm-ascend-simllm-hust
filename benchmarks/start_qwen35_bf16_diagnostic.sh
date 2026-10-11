#!/usr/bin/env bash
set -euo pipefail

# Launch one arm at a time on the same two Ascend cards. This is a 910B3
# compatibility diagnostic, not the benchmark program's 910B2 acceptance run.
arm="${1:?usage: $0 baseline|plugin OUTPUT_DIR}"
output_dir="${2:?usage: $0 baseline|plugin OUTPUT_DIR}"
[[ "$arm" == baseline || "$arm" == plugin ]] || { echo "Invalid arm: $arm" >&2; exit 2; }

launcher="${SERVER_LAUNCHER:-/data/jxd/quant-Qwen3.5-35B-A3B/run_swe_server.sh}"
model="${MODEL_DIR:-/data/jxd/Qwen3.5-35B-A3B}"
runtime="${PERF_ENV:-/data/jxd/envs/vllm-hust-v1-cann91}"
cann="${CANN_SET_ENV:-/data/jxd/Ascend-9.1.0/cann-9.1.0/set_env.sh}"
atb="${ATB_SET_ENV:-/data/jxd/Ascend-9.1.0/nnal/atb/set_env.sh}"
cards="${ASCEND_RT_VISIBLE_DEVICES:-0,1}"
port="${PORT:-8117}"
extra_args='--block-size 128 --no-enable-prefix-caching --no-enable-chunked-prefill --max-num-seqs 4 --compilation-config {"cudagraph_mode":"PIECEWISE","max_cudagraph_capture_size":4}'

for path in "$launcher" "$cann" "$atb" "$runtime/bin/vllm"; do
    [[ -f "$path" ]] || { echo "Missing dependency: $path" >&2; exit 1; }
done
[[ -d "$model" ]] || { echo "Missing model: $model" >&2; exit 1; }
if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
    echo "Port $port already serves a model" >&2
    exit 1
fi
mkdir -p "$output_dir/$arm"

enabled=0
if [[ "$arm" == plugin ]]; then enabled=1; fi
env -u VLLM_PLUGINS \
    ASCEND_RT_VISIBLE_DEVICES="$cards" PORT="$port" \
    PERF_ENV="$runtime" MODEL_PATH="$model" CANN_SET_ENV="$cann" ATB_SET_ENV="$atb" \
    LABEL=bf16 SERVED_NAME=qwen35-a3b-bf16 TP=2 MAX_MODEL_LEN=8192 \
    GPU_MEM_UTIL=0.85 PERF_FIX_KERNELS_SO=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0 \
    VLLM_CACHE_ROOT="$output_dir/$arm-cache" \
    VLLM_ASCEND_SIMLLM_ENABLED="$enabled" \
    VLLM_ASCEND_SIMLLM_HYBRID_REUSE_MODE=tail_anchor \
    VLLM_ASCEND_SIMLLM_HYBRID_RECOMPUTE_TOKENS=512 \
    LOG_DIR="$output_dir/$arm" EXTRA_SERVER_ARGS="$extra_args" \
    nohup bash "$launcher" > "$output_dir/$arm-launch.log" 2>&1 < /dev/null &

for _ in $(seq 1 480); do
    if curl -fsS "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
        echo "$arm ready at http://127.0.0.1:$port on $cards"
        exit 0
    fi
    sleep 5
done
echo "$arm failed to become ready; inspect $output_dir/$arm-launch.log" >&2
exit 1
