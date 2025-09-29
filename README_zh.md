# Anthropic API 代理 - 支持 Gemini 和 OpenAI 模型 🔄

**让 Anthropic 客户端（如 Claude Code）使用 Gemini、OpenAI 或直接使用 Anthropic 后端。** 🤝

一个代理服务器，让您可以通过 LiteLLM 使用 Anthropic 客户端连接 Gemini、OpenAI 或 Anthropic 模型本身（透明代理）。🌉

![Anthropic API 代理](pic.png)

## 快速开始 ⚡

### 前置要求

- OpenAI API 密钥 🔑
- Google AI Studio (Gemini) API 密钥（如果使用 Google 提供商）🔑
- 已安装 [uv](https://github.com/astral-sh/uv)

### 安装配置 🛠️

#### 从源码安装

1. **克隆此仓库**：
   ```bash
   git clone https://github.com/1rgs/claude-code-proxy.git
   cd claude-code-proxy
   ```

2. **安装 uv**（如果尚未安装）：
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   *（当您运行服务器时，`uv` 会根据 `pyproject.toml` 处理依赖项）*

3. **配置环境变量**：
   复制示例环境文件：
   ```bash
   cp .env.example .env
   ```
   编辑 `.env` 并填入您的 API 密钥和模型配置：

   *   `ANTHROPIC_API_KEY`：（可选）仅在代理到 Anthropic 模型时需要。
   *   `OPENAI_API_KEY`：您的 OpenAI API 密钥（如果使用默认 OpenAI 偏好或作为后备则必需）。
   *   `GEMINI_API_KEY`：您的 Google AI Studio (Gemini) API 密钥（如果 PREFERRED_PROVIDER=google 则必需）。
   *   `PREFERRED_PROVIDER`（可选）：设置为 `openai`（默认）、`google` 或 `anthropic`。这决定了 `haiku`/`sonnet` 映射的主要后端。
   *   `BIG_MODEL`（可选）：映射 `sonnet` 请求的模型。默认为 `gpt-4.1`（如果 `PREFERRED_PROVIDER=openai`）或 `gemini-2.5-pro-preview-03-25`。当 `PREFERRED_PROVIDER=anthropic` 时忽略。
   *   `SMALL_MODEL`（可选）：映射 `haiku` 请求的模型。默认为 `gpt-4.1-mini`（如果 `PREFERRED_PROVIDER=openai`）或 `gemini-2.0-flash`。当 `PREFERRED_PROVIDER=anthropic` 时忽略。

   **映射逻辑：**
   - 如果 `PREFERRED_PROVIDER=openai`（默认），`haiku`/`sonnet` 映射到带有 `openai/` 前缀的 `SMALL_MODEL`/`BIG_MODEL`。
   - 如果 `PREFERRED_PROVIDER=google`，`haiku`/`sonnet` 映射到带有 `gemini/` 前缀的 `SMALL_MODEL`/`BIG_MODEL`（如果这些模型在服务器的已知 `GEMINI_MODELS` 列表中，否则回退到 OpenAI 映射）。
   - 如果 `PREFERRED_PROVIDER=anthropic`，`haiku`/`sonnet` 请求直接传递给带有 `anthropic/` 前缀的 Anthropic，不重新映射到不同模型。

4. **运行服务器**：
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082 --reload
   ```
   *（`--reload` 是可选的，用于开发）*

#### Docker

如果使用 docker，下载示例环境文件到 `.env` 并按上述说明编辑。
```bash
curl -O .env https://raw.githubusercontent.com/1rgs/claude-code-proxy/refs/heads/main/.env.example
```

然后，您可以使用 [docker compose](https://docs.docker.com/compose/) 启动容器（推荐）：

```yml
services:
  proxy:
    image: ghcr.io/1rgs/claude-code-proxy:latest
    restart: unless-stopped
    env_file: .env
    ports:
      - 8082:8082
```

或使用命令：

```bash
docker run -d --env-file .env -p 8082:8082 ghcr.io/1rgs/claude-code-proxy:latest
```

### 与 Claude Code 一起使用 🎮

1. **安装 Claude Code**（如果尚未安装）：
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **连接到您的代理**：
   ```bash
   ANTHROPIC_BASE_URL=http://localhost:8082 claude
   ```

3. **完成！** 您的 Claude Code 客户端现在将通过代理使用配置的后端模型（默认为 Gemini）。🎯

## 模型映射 🗺️

代理根据配置的模型自动将 Claude 模型映射到 OpenAI 或 Gemini 模型：

| Claude 模型 | 默认映射 | 当 BIG_MODEL/SMALL_MODEL 是 Gemini 模型时 |
|--------------|--------------|---------------------------|
| haiku | openai/gpt-4o-mini | gemini/[model-name] |
| sonnet | openai/gpt-4o | gemini/[model-name] |

### 支持的模型

#### OpenAI 模型
以下 OpenAI 模型支持自动 `openai/` 前缀处理：
- o3-mini
- o1
- o1-mini
- o1-pro
- gpt-4.5-preview
- gpt-4o
- gpt-4o-audio-preview
- chatgpt-4o-latest
- gpt-4o-mini
- gpt-4o-mini-audio-preview
- gpt-4.1
- gpt-4.1-mini

#### Gemini 模型
以下 Gemini 模型支持自动 `gemini/` 前缀处理：
- gemini-2.5-pro-preview-03-25
- gemini-2.0-flash

### 模型前缀处理
代理自动为模型名称添加适当的前缀：
- OpenAI 模型获得 `openai/` 前缀
- Gemini 模型获得 `gemini/` 前缀
- BIG_MODEL 和 SMALL_MODEL 将根据它们是否在 OpenAI 或 Gemini 模型列表中获得适当的前缀

例如：
- `gpt-4o` 变为 `openai/gpt-4o`
- `gemini-2.5-pro-preview-03-25` 变为 `gemini/gemini-2.5-pro-preview-03-25`
- 当 BIG_MODEL 设置为 Gemini 模型时，Claude Sonnet 将映射到 `gemini/[model-name]`

### 自定义模型映射

使用 `.env` 文件中的环境变量或直接控制映射：

**示例 1：默认（使用 OpenAI）**
除了 API 密钥外，`.env` 中无需更改，或确保：
```dotenv
OPENAI_API_KEY="your-openai-key"
GEMINI_API_KEY="your-google-key" # 如果 PREFERRED_PROVIDER=google 则需要
# PREFERRED_PROVIDER="openai" # 可选，这是默认值
# BIG_MODEL="gpt-4.1" # 可选，这是默认值
# SMALL_MODEL="gpt-4.1-mini" # 可选，这是默认值
```

**示例 2：偏好 Google**
```dotenv
GEMINI_API_KEY="your-google-key"
OPENAI_API_KEY="your-openai-key" # 后备需要
PREFERRED_PROVIDER="google"
# BIG_MODEL="gemini-2.5-pro-preview-03-25" # 可选，这是 Google 偏好的默认值
# SMALL_MODEL="gemini-2.0-flash" # 可选，这是 Google 偏好的默认值
```

**示例 3：使用直接 Anthropic（"仅 Anthropic 代理"模式）**
```dotenv
ANTHROPIC_API_KEY="sk-ant-..."
PREFERRED_PROVIDER="anthropic"
# BIG_MODEL 和 SMALL_MODEL 在此模式下被忽略
# haiku/sonnet 请求直接传递给 Anthropic 模型
```

*用例：此模式使您能够使用代理基础设施（用于日志记录、中间件、请求/响应处理等），同时仍使用实际的 Anthropic 模型，而不是被迫重新映射到 OpenAI 或 Gemini。*

**示例 4：使用特定 OpenAI 模型**
```dotenv
OPENAI_API_KEY="your-openai-key"
GEMINI_API_KEY="your-google-key"
PREFERRED_PROVIDER="openai"
BIG_MODEL="gpt-4o" # 示例特定模型
SMALL_MODEL="gpt-4o-mini" # 示例特定模型
```

## 工作原理 🧩

此代理通过以下方式工作：

1. **接收请求** 以 Anthropic 的 API 格式 📥
2. **转换** 请求通过 LiteLLM 转换为 OpenAI 格式 🔄
3. **发送** 转换后的请求到 OpenAI 📤
4. **转换** 响应回 Anthropic 格式 🔄
5. **返回** 格式化的响应给客户端 ✅

代理处理流式和非流式响应，保持与所有 Claude 客户端的兼容性。🌊

## 贡献 🤝

欢迎贡献！请随时提交拉取请求。🎁


