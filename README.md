# MeloTTS RK3588 Server

Device-local text-to-speech service for Rockchip RK3588 using the published MeloTTS RKNN conversion.

## Features

- Loopback-only HTTP API for single-user local clients
- Session-based incremental playback queue for low-latency local clients
- Multi-runtime language routing: Chinese bundle for `zh`, English-v3 bundle for `en`, Japanese bundle for `ja`
- CLI for `serve`, `speak`, `synthesize`, `status`, `stop`, and `devices`
- Hybrid inference path: CPU preprocessing/encoder plus RKNN decoder on the NPU
- Server-owned playback to the analog audio output through ALSA via `aplay`
- Auto-selection of the ES8388 analog output path on RK3588 when available
- Artifact bootstrap from Hugging Face for the published RKNN conversion

## Requirements

- RK3588 target with Rockchip NPU runtime available
- Python 3.10+
- ALSA `aplay`
- Access to the Rockchip `rknn-toolkit-lite2` Python wheel or an environment where `pip install rknn-toolkit-lite2` works

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e ".[dev,english,japanese]"
pip install rknn-toolkit-lite2
melotts-cli bootstrap
melotts-cli serve
```

In another shell:

```bash
melotts-cli speak "你好，this is a bilingual MeloTTS test."
melotts-cli status
```

If you already have the Rockchip wheel on-device instead of using PyPI:

```bash
. .venv/bin/activate
pip install /path/to/rknn_toolkit_lite2-*.whl
```

## Bootstrap Script

The repo includes a helper script that sets up the virtualenv, installs Python dependencies, optionally installs the RKNN wheel, and downloads model assets.

```bash
RKNN_LITE_WHEEL=/path/to/rknn_toolkit_lite2-*.whl ./scripts/bootstrap.sh
```

If `RKNN_LITE_WHEEL` is omitted, the script only installs the Python package set and downloads the model artifacts.

Language extras:

- `.[english]` installs the English runtime dependencies
- `.[japanese]` installs the Japanese frontend dependencies: `fugashi`, `mecab-python3`, `num2words`, `pykakasi`, and `unidic-lite`

## Model assets

All model assets live under the repo-local `./models/` directory.
The server now uses this in-repo layout by default:

- `models/chinese` for the Chinese RKNN bundle
- `models/english-v3` for the English-v3 bundle
- `models/japanese` for the Japanese hybrid bundle

These model files are local runtime assets and are not committed to Git.

By default the bootstrap step downloads `happyme531/MeloTTS-RKNN2` into `./models/chinese`.
If enabled, the English-v3 bundle is expected at `./models/english-v3`.
If enabled, the Japanese bundle is expected at `./models/japanese`.

The service expects these files:

- `encoder.onnx`
- `decoder.rknn`
- `decoder.onnx` optional reference artifact from the upstream conversion bundle
- `g.bin`
- `lexicon.txt`
- `tokens.txt`
- `utils.py`
- `text/`

The implementation is validated against the published split pipeline where:

- `encoder.onnx` runs on CPU through ONNX Runtime
- `decoder.rknn` runs on the RK3588 NPU through RKNN Runtime
- Chinese text and mixed Chinese/English text both use the Chinese mixed-language model path

## Audio Output

The service uses ALSA and `aplay` for playback. With the default `MELOTTS_ALSA_DEVICE=auto`, it prefers:

- `sysdefault:CARD=rockchipes8388`
- `plughw:CARD=rockchipes8388,DEV=0`
- `hw:CARD=rockchipes8388,DEV=0`
- `default`

On the validated RK3588 host, the resolved playback device was:

```text
sysdefault:CARD=rockchipes8388
```

List devices with:

```bash
melotts-cli devices
```

Override the output device if needed:

```bash
MELOTTS_ALSA_DEVICE=plughw:CARD=rockchipes8388,DEV=0 melotts-cli serve
```

## Configuration

Environment variables:

- `MELOTTS_HOST` default `127.0.0.1`
- `MELOTTS_PORT` default `8123`
- `MELOTTS_SAMPLE_RATE` default `44100`
- `MELOTTS_MODEL_REPO_ID` default `happyme531/MeloTTS-RKNN2`
- `MELOTTS_MODEL_DIR` default `models/chinese`
- `MELOTTS_ENGLISH_ENABLED` default auto-enable when `MELOTTS_ENGLISH_BUNDLE_DIR` exists
- `MELOTTS_ENGLISH_BUNDLE_DIR` default `models/english-v3`
- `MELOTTS_ENGLISH_LATENT_FRAMES` default `256`
- `MELOTTS_JAPANESE_ENABLED` default auto-enable when `MELOTTS_JAPANESE_BUNDLE_DIR` exists
- `MELOTTS_JAPANESE_BUNDLE_DIR` default `models/japanese`
- `MELOTTS_JAPANESE_LATENT_FRAMES` default `256`
- `MELOTTS_RUNTIME_LIB_DIR` default `lib`
- `MELOTTS_ALSA_DEVICE` default `auto`
- `MELOTTS_PREBUFFER_MS` default `200`
- `MELOTTS_MAX_TEXT_CHARS` default `1000`

## CLI

Start the server:

```bash
melotts-cli serve
```

Speak through the local server:

```bash
melotts-cli speak "这是一个测试。This is a test."
echo "你好，来自 stdin 的请求。" | melotts-cli speak
```

Generate a WAV file without playback:

```bash
melotts-cli synthesize "你好，生成 wav 文件。" --out sample.wav
```

Inspect or control the server:

```bash
melotts-cli status
melotts-cli job <request-id>
melotts-cli devices
melotts-cli stop
melotts-cli bootstrap
MELOTTS_ENGLISH_BUNDLE_SRC=~/repos/melotts-converter-2-rk3588-test/artifacts/bundle melotts-cli bootstrap
```

## API

Status and legacy playback:

- `POST /v1/speak`
- `POST /v1/synthesize`
- `GET /v1/status`
- `GET /v1/jobs/{request_id}`
- `POST /v1/stop`
- `GET /v1/audio/devices`

Session streaming:

- `POST /v1/sessions`
- `POST /v1/sessions/{session_id}/chunks`
- `POST /v1/sessions/{session_id}/commit`
- `POST /v1/sessions/{session_id}/cancel`
- `GET /v1/sessions/{session_id}`

The server binds to `127.0.0.1` by default.

Language routing:

- `language=en` uses the converted English-v3 hybrid runtime
- `language=ja` uses the Japanese hybrid runtime
- `language=zh` and `language=yue` use the existing Chinese RKNN bundle

Example:

```bash
curl -s http://127.0.0.1:8123/v1/status

curl -s -X POST http://127.0.0.1:8123/v1/speak \
  -H 'content-type: application/json' \
  -d '{"text":"This is a local English-v3 test.","speed":1.0,"volume":1.0,"language":"en"}'
```

Poll a submitted playback job:

```bash
melotts-cli speak "作业状态测试"
melotts-cli job <request-id>
```

Queue an incremental playback session:

```bash
SESSION_ID=$(curl -s -X POST http://127.0.0.1:8123/v1/sessions \
  -H 'content-type: application/json' \
  -d '{"language":"zh","speed":1.0,"volume":1.0}' | python -c 'import json,sys; print(json.load(sys.stdin)["session_id"])')

curl -s -X POST "http://127.0.0.1:8123/v1/sessions/${SESSION_ID}/chunks" \
  -H 'content-type: application/json' \
  -d '{"text":"你好，这是第一段。","speed":1.0,"volume":1.0}'

curl -s -X POST "http://127.0.0.1:8123/v1/sessions/${SESSION_ID}/chunks" \
  -H 'content-type: application/json' \
  -d '{"text":"这是第二段。","speed":1.0,"volume":1.0}'

curl -s -X POST "http://127.0.0.1:8123/v1/sessions/${SESSION_ID}/commit"
curl -s "http://127.0.0.1:8123/v1/sessions/${SESSION_ID}"
```

## Voice Assistant Integration

This server is now the expected TTS backend for `~/repos/voice-assistant`.

Current integration contract:

- `voice-assistant` checks readiness with `GET /v1/status`
- preferred path: `voice-assistant` creates a playback session with `POST /v1/sessions`, enqueues stable chunks, commits, and polls `GET /v1/sessions/{session_id}`
- fallback path: `voice-assistant` submits whole text with `POST /v1/speak` and polls `GET /v1/jobs/{request_id}`
- `voice-assistant` can keep sending `language=en` / `language=zh`; the TTS server routes to the correct runtime internally
- audio playback remains server-owned on the local RK3588 device

Default integrated port:

```text
127.0.0.1:8123
```

## systemd User Service

An example user unit is included at `deploy/melotts-rk3588.service`.

Install it with:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/melotts-rk3588.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now melotts-rk3588.service
```

Check status:

```bash
systemctl --user status melotts-rk3588.service
journalctl --user -u melotts-rk3588.service -f
```

## Validation Status

Validated on the current RK3588 host:

- Python environment created successfully
- Model bundle downloaded successfully
- `rknn-toolkit-lite2` installed successfully
- RKNN warmup completed successfully
- Real synthesis produced a WAV output successfully
- Loopback HTTP server responded successfully
- Live playback request entered `playing` state on `sysdefault:CARD=rockchipes8388`
- Japanese live playback request completed successfully on `sysdefault:CARD=rockchipes8388`

Notes:

- ONNX Runtime logs a GPU discovery warning on this host. It does not block inference because the encoder is forced onto `CPUExecutionProvider`.
- RKNN emits a static-shape dynamic-range warning for the published model bundle. This does not block inference on the validated host.
