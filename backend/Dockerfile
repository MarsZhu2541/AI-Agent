FROM docker.1ms.run/python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "agent.api:app", "--host", "0.0.0.0", "--port", "8000"]
