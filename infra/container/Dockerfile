# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.11.32@sha256:df4cae8f3a96d175e2e5f992e597550000edbe78fdc2594d5cd8de1a217f504c AS uv

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85

ARG PYTHON_VERSION=3.12.13

LABEL org.opencontainers.image.title="PiPER OutcomeStack remote training environment" \
      org.opencontainers.image.source="https://github.com/ASkarin/piper-outcome-stack"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/piper/python \
    UV_PROJECT_ENVIRONMENT=/opt/piper/.venv \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/workspace/piper/bin:/workspace/piper/python-env/bin:/opt/piper/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        acl \
        bash-completion \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        jq \
        libavcodec-dev \
        libavdevice-dev \
        libavfilter-dev \
        libavformat-dev \
        libavutil-dev \
        libgl1 \
        libglib2.0-0 \
        libswresample-dev \
        libswscale-dev \
        numactl \
        openssh-client \
        openssh-server \
        pkg-config \
        procps \
        ripgrep \
        rsync \
        sudo \
        tini \
        tmux \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_CONCURRENT_DOWNLOADS=4 \
    UV_HTTP_RETRIES=5 \
    UV_HTTP_TIMEOUT=300 \
    UV_PYTHON_CACHE_DIR=/root/.cache/uv/python

WORKDIR /opt/piper/build
COPY pyproject.toml uv.lock .python-version ./
COPY packages/lerobot_robot_outcome_piper/pyproject.toml \
    packages/lerobot_robot_outcome_piper/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install "${PYTHON_VERSION}" \
    && export GIT_CONFIG_COUNT=1 \
        GIT_CONFIG_KEY_0=http.version \
        GIT_CONFIG_VALUE_0=HTTP/1.1 \
    && sync_args=( \
        --frozen \
        --no-install-project \
        --extra remote-training \
        --group dev \
    ) \
    && for attempt in 1 2 3; do \
        if uv sync "${sync_args[@]}"; then \
            break; \
        fi; \
        if (( attempt == 3 )); then \
            exit 1; \
        fi; \
        sleep "$((attempt * 5))"; \
    done \
    && /opt/piper/.venv/bin/python -c \
        "import importlib.metadata as m; import tensorboard, torch, torchvision; assert m.version('tensorboard') == '2.21.0'; assert torch.__version__ == '2.11.0+cu128'; assert torchvision.__version__ == '0.26.0+cu128'"

COPY infra/container/bin/ /usr/local/bin/
COPY infra/container/lib/ /usr/local/lib/piper-container/
COPY infra/container/entrypoint.sh /usr/local/sbin/piper-entrypoint
COPY infra/container/init-shared-python.sh /usr/local/sbin/piper-init-shared-python
COPY infra/container/profile.sh /etc/profile.d/piper.sh
COPY infra/container/sshd_config /etc/ssh/sshd_config.piper

RUN chmod 0755 \
        /usr/local/bin/piper-artifact-fetch \
        /usr/local/bin/piper-artifact-promote \
        /usr/local/bin/piper-env-doctor \
        /usr/local/bin/piper-gpu-run \
        /usr/local/bin/piper-python \
        /usr/local/sbin/piper-entrypoint \
        /usr/local/sbin/piper-init-shared-python \
    && chmod 0644 /etc/profile.d/piper.sh /etc/ssh/sshd_config.piper \
    && mkdir -p /run/sshd /workspace \
    && chown -R root:root /opt/piper /usr/local/lib/piper-container

EXPOSE 22

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/sbin/piper-entrypoint"]
