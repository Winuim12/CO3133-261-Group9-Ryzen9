FROM python:3.12.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /workspace/assignment3

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements-cpu.txt requirements-gpu.txt ./

FROM base AS cpu
RUN python -m pip install --no-cache-dir -r requirements-cpu.txt
COPY . .
CMD ["python", "-m", "experiments.check_environment", "--device", "cpu", "--smoke-test"]

FROM base AS gpu
RUN python -m pip install --no-cache-dir -r requirements-gpu.txt
COPY . .
CMD ["python", "-m", "experiments.check_environment", "--device", "cuda", "--smoke-test"]
