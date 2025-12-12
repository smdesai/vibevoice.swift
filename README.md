# VibeVoice.swift

Swift / MLX implementation of **Microsoft VibeVoice‑Realtime‑0.5B** with a macOS CLI for fast, low‑latency text‑to‑speech.  
Includes streaming inference and optional 8‑bit / 4‑bit quantization.

This repo does **not** ship model weights. The CLI downloads models from Hugging Face on first use and caches them locally.

## Features

- **Realtime TTS** on Apple Silicon via MLX.
- **CLI modes**
  - `generate`: offline generation to WAV
  - `speak`: realtime streaming playback (and optional save)
  - `stream`: read text from stdin and stream audio
  - `quantize`: produce 8‑bit / 4‑bit weight packs + manifest
- **Quantized inference** using `QuantizedLinear` layers (MLX).
- **Hugging Face model resolution & caching** (respects `HF_HUB_CACHE` / `HF_HOME`).

## Requirements

- macOS **14+** (Sonoma)
- **Apple Silicon** (M‑series)
- Xcode **15+** (recommended for runnable Release builds)
- Swift **5.9**

## Build

```bash
xcodebuild -scheme vibevoiceCLI -configuration Release \
  -destination 'platform=macOS' \
  -derivedDataPath .build build
```

The Release binary will be at:

```bash
.build/Build/Products/Release/vibevoiceCLI
```

## Quick Start

You need a **voice cache** (`.safetensors`) for speaker conditioning.  
This repo includes an example cache in `voice_cache/`.

### Offline generation

```bash
.build/Build/Products/Release/vibevoiceCLI generate "Hello world!" \
  --voice-cache voice_cache/en-Mike_man.safetensors \
  -o hello.wav
```

### Use the 8‑bit model from Hugging Face

```bash
.build/Build/Products/Release/vibevoiceCLI generate "Hello world!" \
  --voice-cache voice_cache/en-Mike_man.safetensors \
  --model mzbac/VibeVoice-Realtime-0.5B-8bit \
  -o hello-8bit.wav
```

### Realtime playback

```bash
.build/Build/Products/Release/vibevoiceCLI speak "Streaming TTS demo." \
  --voice-cache voice_cache/en-Mike_man.safetensors \
  -o speak.wav
```

### Stream from stdin

```bash
echo "Hello from stdin." | \
  .build/Build/Products/Release/vibevoiceCLI stream \
    --voice-cache voice_cache/en-Mike_man.safetensors \
    -o stream.wav
```

All audio is 24 kHz mono WAV.

## CLI Reference

Run `vibevoiceCLI -h` or `vibevoiceCLI <subcommand> -h` for full usage and options.

## Quantization

Quantize a model (default: `microsoft/VibeVoice-Realtime-0.5B`) to 8‑bit affine weights:

```bash
.build/Build/Products/Release/vibevoiceCLI quantize \
  --input microsoft/VibeVoice-Realtime-0.5B \
  --output vibevoice-8bit \
  --bits 8 \
  --group-size 32 \
  --mode affine
```

Then run inference from the quantized folder:

```bash
.build/Build/Products/Release/vibevoiceCLI generate "Hello!" \
  --voice-cache voice_cache/en-Mike_man.safetensors \
  --model vibevoice-8bit
```

## Using as a Library

VibeVoice expects **Qwen2.5 token IDs** internally. The library includes the same tokenizer path as the CLI via Swift Transformers:

```swift
import VibeVoice
import Transformers
import MLX

// Resolve/download model + tokenizer from Hugging Face (cached after first run)
let modelDir = try await ModelResolution.resolve(modelSpec: "microsoft/VibeVoice-Realtime-0.5B")
let tokenizerDir = try await ModelResolution.resolveTokenizer(
    modelSpec: Qwen2TokenizerRepository.id
)
let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

let model = try loadVibeVoiceStreamModel(from: modelDir)
let inference = VibeVoiceStreamInference(model: model, numInferenceSteps: 20, cfgScale: 1.3)
try inference.loadVoiceCache(from: "voice_cache/en-Mike_man.safetensors")

let text = "Hello world!\n"
let ids = tokenizer.encode(text: text, addSpecialTokens: false).map { Int32($0) }
let tokenIds = MLXArray(ids).reshaped([1, ids.count])

let audio = try inference.generateWithVoiceCache(tokenIds: tokenIds, maxSpeechTokens: 500)
// `audio` is an MLXArray shaped [1, 1, samples]
```

## Troubleshooting

- **`Failed to load the default metallib`**  
  Build and run the Xcode Release binary (`xcodebuild ...`).  
  The SPM `swift build` binary may not bundle Metal libraries.
- **Model not found / download issues**  
  Models are fetched from Hugging Face and cached in `~/.cache/huggingface/hub` (or `HF_HUB_CACHE` / `HF_HOME`).

## Credits

- Microsoft **VibeVoice‑Realtime‑0.5B** model: https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B  
- MLX Swift: https://github.com/ml-explore/mlx-swift  
- Swift Transformers: https://github.com/huggingface/swift-transformers

## License

MIT. See `LICENSE`.  
Model weights and voice caches are covered by their respective upstream licenses.
