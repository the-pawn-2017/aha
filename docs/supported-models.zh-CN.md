# 支持的模型

aha 支持多个领域的最先进 AI 模型集合。

## 文本生成

| 模型 | 参数量 | 描述 | 使用场景 |
|------|--------|------|----------|
| **Qwen2.5-7B** | 7B | 通用大语言模型 | 对话、推理、代码 |
| **Qwen3** | 多种 | 最新一代 | 高级推理 |
| **MiniCPM4** | 4B | 高效轻量级 | 边缘部署 |

## 视觉与多模态

| 模型 | 类型 | 描述 | 分辨率 |
|------|------|------|--------|
| **Qwen2.5-VL** | 视觉语言 | 图像理解 | 最高 1024x1024 |
| **Qwen3-VL** | 视觉语言 | 增强多模态 | 最高 1536x1536 |
| **MiniCPM-V** | 视觉语言 | 轻量视觉 | 最高 768x768 |

## 语音识别 (ASR)

| 模型 | 语言 | 实时 | 速度 |
|------|------|------|------|
| **FunASR-Nano** | 中/英 | 是 | 16x 实时 |
| **GLM-ASR-Nano** | 中/英 | 是 | 32x 实时 |

## OCR

| 模型 | 语言 | 类型 | 优势 |
|------|------|------|------|
| **PaddleOCR-VL** | 80+ | 轻量级 | 通用文档 |
| **Hunyuan-OCR** | 中文 | 深度学习 | 复杂布局 |
| **DeepSeek-OCR** | 多语言 | 场景文字 | 自然图像 |

## 音频处理

| 模型 | 类型 | 描述 |
|------|------|------|
| **VoxCPM** | 语音编解码器 | 神经音频编解码 |
| **RMBG-2.0** | 背景移除 | 语音隔离 |

## 模型格式

所有模型均采用优化的 ONNX 格式，提供：

- **跨平台兼容** - Windows、macOS、Linux
- **CPU 加速** - AVX2、NEON、SIMD
- **边缘部署** - 无需 GPU
- **快速推理** - 优化运行时

## 模型选择

aha 会自动为每个任务选择最佳模型。如需覆盖：

```bash
aha chat "你好" --model qwen2.5-7b
aha vision --model qwen2.5-vl "描述这个" --image img.jpg
aha asr --model fun-asr-nano audio.wav
```

## 模型来源

模型来源：

- [Hugging Face](https://huggingface.co) - 主模型中心
- [ModelScope](https://modelscope.cn) - 中文模型中心
- [GitHub Releases](https://github.com) - 备份发布

## 添加新模型

参见 [开发指南](./development.zh-CN.md) 了解添加新模型集成的说明。

## 模型更新

模型定期更新。查看 [releases](https://github.com/jhqxxx/aha/releases) 获取最新版本。

## 许可证

每个模型都有自己的许可证。在生产中使用前请查看模型许可证。

## 性能基准

CPU (M1 Pro) 上的近似推理速度：

| 模型 | 任务 | Tokens/秒 |
|------|------|-----------|
| Qwen2.5-7B | 文本 | 25-35 |
| Qwen2.5-VL | 视觉 | 20-30 |
| FunASR-Nano | ASR | 200-500x |

*基准测试因硬件和输入大小而异。*
