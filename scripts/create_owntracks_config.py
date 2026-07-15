"""Create an importable OwnTracks iOS configuration without printing secrets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def build_config(host: str, password: str) -> dict[str, object]:
    return {
        "_type": "configuration",
        "mode": 0,
        "monitoring": 1,
        "host": host,
        "port": 1883,
        "mqttProtocolLevel": 4,
        "tls": False,
        "ws": False,
        "auth": True,
        "username": "smart_room",
        "password": password,
        "deviceId": "iphone",
        "tid": "SI",
        "pubTopicBase": "owntracks/%u/%d",
        "pubQos": 1,
        "pubRetain": True,
        "sub": False,
        "cmd": False,
        "remoteConfiguration": False,
        "allowRemoteLocation": False,
        "allowinvalidcerts": False,
        "extendedData": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True, help="PC Tailscale IPv4 address")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes")) / ".env",
    )
    args = parser.parse_args()
    password = _read_env(args.env_file).get("SMART_ROOM_MQTT_PASSWORD", "")
    if not password:
        raise SystemExit("SMART_ROOM_MQTT_PASSWORD is missing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_config(args.host, password), indent=2) + "\n", encoding="utf-8")
    print(f"OwnTracks configuration created: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
