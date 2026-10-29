# Mini LLM Gateway

一个面向单进程部署的轻量 LLM 网关：用 Gateway Key 调用逻辑模型，由网关完成鉴权、准入、能力路由、重试与 fallback，并把调用和费用记录到 SQLite。

支持 Chat Completions 字段子集、SSE、工具调用、JSON Schema 校验与有限次修复、版本化 Prompt。内部协议独立于供应商 JSON，当前提供 OpenAI 兼容适配器。

## 快速开始

在项目根目录操作。先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)；仓库默认使用 Python 3.11，uv 可以自动安装该版本。

```bash
uv sync --locked --extra dev
cp gateway.example.yaml gateway.yaml
cp .env.example .env
```

编辑 `.env`，填写 `OPENAI_API_KEY` 和独立的 `GATEWAY_DEMO_KEY`。后者是你自己生成的网关访问密钥，例如使用 `uv run --locked --extra dev python -c "import secrets; print(secrets.token_urlsafe(32))"` 生成。

编辑 `gateway.yaml`，确认供应商 `base_url`、`allowed_hosts`、`resolved_model`、能力和价格配置与你使用的上游一致。示例中的模型和价格仅作配置演示；逻辑模型名由 `logical_model` 决定。

```bash
uv run --locked --extra dev --env-file .env mini-llm-gateway --config gateway.yaml
```

启动后可通过 `http://localhost:8080` 访问。网关自身只读进程环境变量，上面的 uv 命令负责加载 `.env`。

新开一个终端，在项目根目录发送请求。这个示例会实际调用上游，可能产生费用；若修改过逻辑模型名，同步修改 `model`。

```bash
uv run --locked --extra dev --env-file .env python - <<'PY'
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key=os.environ["GATEWAY_DEMO_KEY"],
    max_retries=0,
)
result = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Reply with OK."}],
    max_tokens=16,
)
print(result.choices[0].message.content)
PY
```

## 本地检查与 CI

安装开发依赖后即可检查，无需创建 `.env`、配置上游或提供真实密钥：

```bash
uv lock --check
uv run --locked --extra dev ruff check app tests scripts
uv run --locked --extra dev mypy
uv run --locked --extra dev pytest -q
```

[GitHub Actions](.github/workflows/ci.yml) 在 push、pull request 和手动触发时运行：Python 3.11 / 3.12 / 3.13 下检查锁文件、Ruff、mypy 和测试；另一个任务构建 Docker 镜像、检查 CLI 和 Compose 配置。测试使用合成凭据和受控上游，无需配置仓库 Secrets。Ruff 规则固定在 `pyproject.toml`；mypy 当前覆盖 8 个核心及组装模块。

## Docker Compose

准备好上述 `gateway.yaml`、`.env` 后，在根目录运行（需要 Docker Compose v2）：

```bash
docker compose up --build -d
docker compose logs -f gateway
# 停止服务
docker compose down
```

Compose 读取 `.env`，把两个密钥传入容器。数据库位于 `gateway-data` 命名卷的 `/srv/data/gateway.db`；本地启动默认写入根目录 `gateway.db`，可用 `GATEWAY_DATABASE` 覆盖。普通 `down` 保留卷，`down -v` 会删除账本。已有数据库的迁移方法见[设计说明](docs/architecture.md#17-官方-sdk-验证持久化与性能测量)。

## 依赖管理

运行与开发依赖及其传递依赖统一记录在 `uv.lock`。本地、CI 和 Docker 都使用 `--locked`，依赖声明与锁文件不一致时会失败。CI 和 Docker 的 uv 工具版本固定为 `0.11.11`。

有意升级依赖时运行以下命令，完成上述检查后，一并提交依赖声明和锁文件：

```bash
uv lock --upgrade-package httpx
uv sync --locked --extra dev
# 修改 pyproject.toml 后，用 uv lock 更新锁文件
```

`pyproject.toml` 保留兼容版本范围；不要手工编辑锁文件。锁文件固定 Python 包依赖，操作系统和容器基础镜像仍需单独维护。

## 接口与能力边界

| 接口 | 用途 |
| --- | --- |
| `GET /v1/models` | 当前 Key 可见的逻辑模型 |
| `POST /v1/chat/completions` | 普通 / 流式补全、工具调用、结构化输出 |
| `POST /v1/prompts/render` | 校验并预览 Prompt 版本 |

所有接口使用 `Authorization: Bearer <Gateway Key>`。支持文本消息、`n=1`；不支持的请求字段会返回 400。当前范围不包含 Responses、原生 Anthropic 协议、图像 / 音频、分布式限流和多进程共享熔断状态。并发与 RPM 限制按 Key 在当前进程内维护。

## 进一步验证与文档

```bash
# 本机受控 HTTP 上游，测网关开销及数据库写入；不调用外部模型
uv run --locked --extra dev python scripts/benchmark.py --requests 100 --db-pairs 500
# 真实上游：额外填写 .env 中 UPSTREAM_BASE_URL 和 UPSTREAM_MODEL
uv run --locked --extra dev python scripts/verify_live.py --env-file .env
```

真实验证默认最多 8 次上游请求、最多 2 并发，可能产生费用。基准与真实验证各自写入 `docs/*-results.json`；本机受控结果不能代表真实模型性能。已有报告仍将外部上游验证标记为缺少配置；后续以实际生成的报告为准。

- [架构、异常处理和账本设计](docs/architecture.md)
- [基准原始数据](docs/benchmark-results.json)
- [真实上游验证状态](docs/live-validation-results.json)
- [配置示例](gateway.example.yaml) · [环境变量模板](.env.example) · [Prompt 示例](prompts/demo.yaml)

常见问题：启动提示缺少环境变量时检查 `.env` 并使用带 `--env-file` 的命令；401 检查 Gateway Key，404 检查逻辑模型及能力，429 检查 RPM / 并发限制。健康探针、指标导出和长期运行治理仍待补齐。
