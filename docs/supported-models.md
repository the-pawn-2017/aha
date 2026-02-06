# Supported Models

aha supports a growing collection of state-of-the-art AI models across multiple domains.

## Text Generation

| Model | Parameters | Description | Use Case |
|-------|-----------|-------------|----------|
| **Qwen2.5-7B** | 7B | General-purpose LLM | Chat, reasoning, code |
| **Qwen3** | Various | Latest generation | Advanced reasoning |
| **MiniCPM4** | 4B | Efficient lightweight | Edge deployment |

## Vision & Multimodal

| Model | Type | Description | Resolution |
|-------|------|-------------|------------|
| **Qwen2.5-VL** | Vision-Language | Image understanding | Up to 1024x1024 |
| **Qwen3-VL** | Vision-Language | Enhanced multimodal | Up to 1536x1536 |
| **MiniCPM-V** | Vision-Language | Lightweight vision | Up to 768x768 |

## Speech Recognition (ASR)

| Model | Language | Real-time | Speed |
|-------|----------|-----------|-------|
| **FunASR-Nano** | Chinese/English | Yes | 16x realtime |
| **GLM-ASR-Nano** | Chinese/English | Yes | 32x realtime |

## OCR

| Model | Languages | Type | Strength |
|-------|-----------|------|----------|
| **PaddleOCR-VL** | 80+ | Lightweight | General documents |
| **Hunyuan-OCR** | Chinese | Deep learning | Complex layouts |
| **DeepSeek-OCR** | Multi | Scene text | Natural images |

## Audio Processing

| Model | Type | Description |
|-------|------|-------------|
| **VoxCPM** | Voice Codec | Neural audio codec |
| **RMBG-2.0** | Background Removal | Voice isolation |

## Model Formats

All models are served in optimized ONNX format for:

- **Cross-platform compatibility** - Windows, macOS, Linux
- **CPU acceleration** - AVX2, NEON, SIMD
- **Edge deployment** - No GPU required
- **Fast inference** - Optimized runtime

## Model Selection

aha automatically selects the best model for each task. To override:

```bash
aha chat "Hello" --model qwen2.5-7b
aha vision --model qwen2.5-vl "Describe this" --image img.jpg
aha asr --model fun-asr-nano audio.wav
```

## Model Sources

Models are sourced from:

- [Hugging Face](https://huggingface.co) - Primary model hub
- [ModelScope](https://modelscope.cn) - Chinese model hub
- [GitHub Releases](https://github.com) - Backup releases

## Adding New Models

See [Development Guide](./development.md) for instructions on adding new model integrations.

## Model Updates

Models are regularly updated. Check the [releases](https://github.com/yourusername/aha/releases) for the latest versions.

## License

Each model has its own license. Please review the model's license before use in production.

## Performance Benchmarks

Approximate inference speeds on CPU (M1 Pro):

| Model | Task | Tokens/sec |
|-------|------|------------|
| Qwen2.5-7B | Text | 25-35 |
| Qwen2.5-VL | Vision | 20-30 |
| FunASR-Nano | ASR | 200-500x |

*Benchmarks vary by hardware and input size.*
