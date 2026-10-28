FROM python:3.11-slim

WORKDIR /srv
COPY pyproject.toml ./
COPY app ./app
RUN pip install --no-cache-dir .

EXPOSE 8080
CMD ["mini-llm-gateway", "--config", "gateway.yaml"]
