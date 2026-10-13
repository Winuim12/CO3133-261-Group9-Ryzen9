FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace/assignment2

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "--version"]