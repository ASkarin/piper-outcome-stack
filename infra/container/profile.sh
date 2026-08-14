export PIPER_SHARED_PYTHON_ENV="/workspace/piper/python-env"
export PATH="/workspace/piper/bin:${PIPER_SHARED_PYTHON_ENV}/bin:${PATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export HF_HOME="/workspace/piper/cache/huggingface"
export TORCH_HOME="/workspace/piper/cache/torch"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
unset PYTHONNOUSERSITE

if [ -n "${USER:-}" ]; then
    export PIPER_STAGING_ROOT="/workspace/piper/staging/${USER}"
    export PIPER_RUN_ROOT="/workspace/piper/runs/${USER}"
fi
