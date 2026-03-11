from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx
import uvicorn

from .app import create_app
from .artifacts import bootstrap_artifacts, install_english_bundle, validate_artifacts, validate_english_bundle
from .config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(prog="melotts-cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve")

    bootstrap = subparsers.add_parser("bootstrap")
    bootstrap.add_argument("--repo-id", default=None)
    bootstrap.add_argument("--model-dir", default=None)
    bootstrap.add_argument("--english-bundle-src", default=None)
    bootstrap.add_argument("--english-bundle-dir", default=None)

    speak = subparsers.add_parser("speak")
    speak.add_argument("text", nargs="?")
    speak.add_argument("--speed", type=float, default=1.0)
    speak.add_argument("--volume", type=float, default=1.0)
    speak.add_argument("--language", default="en")
    speak.add_argument("--device", default=None)

    synthesize = subparsers.add_parser("synthesize")
    synthesize.add_argument("text", nargs="?")
    synthesize.add_argument("--speed", type=float, default=1.0)
    synthesize.add_argument("--volume", type=float, default=1.0)
    synthesize.add_argument("--language", default="en")
    synthesize.add_argument("--out", required=True)

    job = subparsers.add_parser("job")
    job.add_argument("request_id")

    subparsers.add_parser("status")
    subparsers.add_parser("stop")
    subparsers.add_parser("devices")

    args = parser.parse_args()
    settings = Settings.from_env()

    if args.command == "serve":
        validate_artifacts(settings.model_dir)
        if settings.english_enabled:
            validate_english_bundle(settings.english_bundle_dir)
        uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")
        return

    if args.command == "bootstrap":
        repo_id = args.repo_id or settings.model_repo_id
        model_dir = Path(args.model_dir) if args.model_dir else settings.model_dir
        zh_path = bootstrap_artifacts(repo_id, model_dir)
        print(zh_path)
        english_src = args.english_bundle_src
        if english_src is not None:
            english_target = Path(args.english_bundle_dir) if args.english_bundle_dir else settings.english_bundle_dir
            en_path = install_english_bundle(Path(english_src), english_target)
            print(en_path)
        return

    if args.command in {"speak", "synthesize"}:
        text = args.text if args.text is not None else sys.stdin.read().strip()
        if not text:
            raise SystemExit("text is required")
        with httpx.Client(base_url=f"http://{settings.host}:{settings.port}", timeout=120.0) as client:
            if args.command == "speak":
                response = client.post(
                    "/v1/speak",
                    json={
                        "text": text,
                        "speed": args.speed,
                        "volume": args.volume,
                        "language": args.language,
                        "playback_device": args.device,
                    },
                )
                response.raise_for_status()
                print(json.dumps(response.json(), ensure_ascii=False, indent=2))
                return
            response = client.post(
                "/v1/synthesize",
                json={"text": text, "speed": args.speed, "volume": args.volume, "language": args.language},
            )
            response.raise_for_status()
            Path(args.out).write_bytes(response.content)
            print(args.out)
            return

    with httpx.Client(base_url=f"http://{settings.host}:{settings.port}", timeout=30.0) as client:
        if args.command == "status":
            response = client.get("/v1/status")
        elif args.command == "job":
            response = client.get(f"/v1/jobs/{args.request_id}")
        elif args.command == "stop":
            response = client.post("/v1/stop")
        elif args.command == "devices":
            response = client.get("/v1/audio/devices")
        else:
            raise SystemExit(f"unknown command {args.command}")
        response.raise_for_status()
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
