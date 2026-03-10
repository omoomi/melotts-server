# MeloTTS RK3588 Server

Device-local text-to-speech service for Rockchip RK3588 using the published MeloTTS RKNN conversion.

## Features

- Loopback-only HTTP API for single-user local clients
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
pip install -e ".[dev]"
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

## Model assets

By default the bootstrap step downloads `happyme531/MeloTTS-RKNN2` into `./models/melotts-rknn2`.

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
- `MELOTTS_MODEL_DIR` default `models/melotts-rknn2`
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
melotts-cli devices
melotts-cli stop
melotts-cli bootstrap
```

## API

- `POST /v1/speak`
- `POST /v1/synthesize`
- `GET /v1/status`
- `POST /v1/stop`
- `GET /v1/audio/devices`

The server binds to `127.0.0.1` by default.

Example:

```bash
curl -s http://127.0.0.1:8123/v1/status

curl -s -X POST http://127.0.0.1:8123/v1/speak \
  -H 'content-type: application/json' \
  -d '{"text":"你好，this is a local HTTP test.","speed":1.0,"volume":1.0}'
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

Notes:

- ONNX Runtime logs a GPU discovery warning on this host. It does not block inference because the encoder is forced onto `CPUExecutionProvider`.
- RKNN emits a static-shape dynamic-range warning for the published model bundle. This does not block inference on the validated host.
