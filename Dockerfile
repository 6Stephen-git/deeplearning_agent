FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# 可通过 build-arg 覆盖（例如清华镜像）以提升国内网络稳定性
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=pypi.org
ENV PIP_DEFAULT_TIMEOUT=120
RUN pip install --no-cache-dir --retries 8 --timeout 120 \
    --index-url ${PIP_INDEX_URL} \
    --trusted-host ${PIP_TRUSTED_HOST} \
    --trusted-host files.pythonhosted.org \
    -r /app/requirements.txt

COPY . /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "src.backend.main"]
