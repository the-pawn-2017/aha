# API 参考

AHA REST API 的完整参考。

## 概述

AHA 提供了 OpenAI 兼容的 REST API 用于运行 AI 模型推理。所有端点遵循标准 HTTP 约定并返回 JSON 响应。

### 基础 URL

默认情况下，API 服务器运行在：
```
http://127.0.0.1:10100
```

您可以在启动服务时自定义：
```bash
aha cli -m qwen3-0.6b -a 0.0.0.0 -p 8080
```

### 身份验证

目前，AHA 不需要身份验证。所有端点在配置的地址/端口上公开访问。

**安全提示**：如果您将 API 暴露到外部网络，请考虑通过反向代理（如 nginx、traefik）实现身份验证。

### 内容类型

所有请求应使用：
```
Content-Type: application/json
```

### 响应格式

成功响应遵循此结构：
```json
{
  "data": { ... },
  "model": "model-name",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

错误响应：
```json
{
  "error": {
    "message": "错误描述",
    "type": "error_type",
    "code": "error_code"
  }
}
```

## 端点

### 对话补全

生成对话补全或文本响应。

#### 端点
```
POST /chat/completions
```

#### 请求体

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `model` | string | 是 | 模型标识符（如 "qwen3-0.6b"） |
| `messages` | array | 是 | 消息对象数组 |
| `temperature` | number | 否 | 采样温度（0-2，默认：1） |
| `top_p` | number | 否 | 核采样（0-1，默认：1） |
| `max_tokens` | integer | 否 | 要生成的最大令牌数 |
| `stream` | boolean | 否 | 启用流式传输（默认：false） |

#### 消息对象

| 字段 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `role` | string | 是 | "system"、"user" 或 "assistant" |
| `content` | string/array | 是 | 消息内容（字符串或多模态数组） |

#### 多模态内容

对于视觉/音频模型，内容可以是数组：

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "描述这张图片"},
    {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}}
  ]
}
```

支持的内容类型：
- `text` - 文本内容
- `image_url` - 图像文件（file://、base64:// 或 http://）
- `audio_url` - 音频文件（file:// 或 base64://）

#### 示例

**简单对话：**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "你好！"}
    ]
  }'
```

**带系统消息：**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "system", "content": "你是一个有用的助手。"},
      {"role": "user", "content": "用一句话解释 Rust。"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**视觉理解：**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3vl-2b",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "这张图片里有什么？"},
          {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}}
        ]
      }
    ]
  }'
```

**OCR（文本提取）：**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ocr",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "提取所有文本"},
          {"type": "image_url", "image_url": {"url": "file:///path/to/document.png"}}
        ]
      }
    ]
  }'
```

**ASR（语音识别）：**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-asr-nano-2512",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "转写这段音频"},
          {"type": "audio_url", "audio_url": {"url": "file:///path/to/audio.wav"}}
        ]
      }
    ]
  }'
```

**流式响应：**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {"role": "user", "content": "给我讲个故事"}
    ],
    "stream": true
  }'
```

流式响应作为服务器发送事件（SSE）发送：
```
data: {"id": "1", "choices": [{"delta": {"content": "从前"}}]}

data: {"id": "1", "choices": [{"delta": {"content": "有"}}]}

data: [DONE]
```

#### 响应

**非流式：**

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "qwen3-0.6b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好！今天我能帮你什么？"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 9,
    "total_tokens": 19
  }
}
```

#### 支持的模型

- 文本：`qwen3-0.6b`、`minicpm4-0.5b`
- 视觉：`qwen2.5vl-3b`、`qwen2.5vl-7b`、`qwen3vl-2b`、`qwen3vl-4b`、`qwen3vl-8b`、`qwen3vl-32b`
- OCR：`deepseek-ocr`、`hunyuan-ocr`、`paddleocr-vl`
- ASR：`glm-asr-nano-2512`、`fun-asr-nano-2512`、`qwen3asr-0.6b`、`qwen3asr-1.7b`

### 语音生成

从文本生成语音（文本转语音）。

#### 端点
```
POST /audio/speech
```

#### 请求体

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `model` | string | 是 | 模型标识符（如 "voxcpm1.5"） |
| `input` | string | 是 | 要转换为语音的文本 |
| `voice` | string | 否 | 语音选择（默认："default"） |

#### 示例

```bash
curl http://127.0.0.1:10100/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "voxcpm1.5",
    "input": "你好，世界！",
    "voice": "default"
  }' \
  --output speech.wav
```

#### 响应

以 WAV 格式返回音频数据。

#### 支持的模型

- `voxcpm`、`voxcpm1.5`

### 图像背景移除

从图像中移除背景。

#### 端点
```
POST /images/remove_background
```

#### 请求体

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `model` | string | 是 | 模型标识符（如 "rmbg2.0"） |
| `image` | string | 是 | 图像文件路径（file://）或 base64 数据 |

#### 示例

**从文件：**

```bash
curl http://127.0.0.1:10100/images/remove_background \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rmbg2.0",
    "image": "file:///path/to/photo.png"
  }' \
  --output no-background.png
```

**从 Base64：**

```bash
curl http://127.0.0.1:10100/images/remove_background \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rmbg2.0",
    "image": "base64://$(base64 -w 0 photo.png)"
  }' \
  --output no-background.png
```

#### 响应

以 PNG 格式返回处理后的图像。

#### 支持的模型

- `rmbg2.0`

## 错误处理

### 错误代码

| 代码 | 描述 |
|------|------|
| 400 | 错误请求 - 无效参数 |
| 404 | 未找到 - 模型或端点未找到 |
| 500 | 内部服务器错误 - 模型推理错误 |
| 503 | 服务不可用 - 模型未加载 |

### 错误响应格式

```json
{
  "error": {
    "message": "未找到模型 'unknown-model'",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

## 速率限制

目前，AHA 不实现速率限制。服务器可以处理并发请求，仅受系统资源限制。

## 文件上传限制

- 字符串数据：5 MB
- 文件上传：100 MB

## OpenAI 兼容性

AHA 的 API 设计为与 OpenAI 的 API 格式兼容。这意味着您可以使用现有的 OpenAI 客户端库，只需最少的更改：

### Python 示例

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:10100",
    api_key="dummy"  # 不使用但库需要
)

response = client.chat.completions.create(
    model="qwen3-0.6b",
    messages=[
        {"role": "user", "content": "你好！"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript 示例

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://127.0.0.1:10100',
  apiKey: 'dummy'  // 不使用但需要
});

const response = await client.chat.completions.create({
  model: 'qwen3-0.6b',
  messages: [{ role: 'user', content: '你好！' }]
});

console.log(response.choices[0].message.content);
```

## 最佳实践

### 1. 对长响应使用流式传输

对于长文本生成，使用流式传输增量获取响应：

```bash
curl ... -d '{"stream": true, ...}'
```

### 2. 设置适当的令牌限制

防止过长的响应：

```json
{
  "max_tokens": 500
}
```

### 3. 调整温度

控制响应创造性：
- `0.0-0.3`：确定性、专注
- `0.4-0.7`：平衡（默认：1.0）
- `0.8-2.0`：创造性、多样

### 4. 使用系统消息

使用系统消息设置行为：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个技术写作助手。"},
    {"role": "user", "content": "..."}
  ]
}
```

## 另见

- [快速入门](./getting-started.zh-CN.md) - 快速入门指南
- [CLI 参考](./cli.zh-CN.md) - 命令行使用
- [安装](./installation.zh-CN.md) - 安装指南
- [开发](./development.zh-CN.md) - 贡献指南
