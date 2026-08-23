# Marvi Smart Room Plugin

> Native Windows smart room engine for Marvi — replaces Home Assistant Docker/WSL for the smart room.

## What It Does

- **Presence fusion**: BLE (ESP32/ESPresense) + mmWave (HE20) + OwnTracks geofence → room-level presence with iPhone deep-sleep handling
- **Tuya LAN control**: Direct control of RGBCW bulb + HE20 sensor via tinytuya — no cloud
- **Room automations**: Adaptive light, sleep/alarm behavior, work-return settle/cancel, evening sleep, and daily resets
- **Local vision**: The sidecar is the single camera owner for owner/visitor recognition, presence, gestures, and posture; only bounded facts and events leave the plugin
- **World-awareness**: Marvi knows where you are, what mode the room is in, light state — as ambient context, not memory writes
- **Marvi integration**: Authenticated loopback RPC, plugin tools (`smart_room_state`, `smart_room_set_mode`, etc.), bounded session context, and structured events

## Architecture

```
ESP32 (ESPresense) ──MQTT──┐
OwnTracks (iPhone) ──MQTT──┼──→ Smart Room sidecar ──authenticated RPC/events──→ Marvi Gateway
Windows camera ────────────┘             │
                                         └── Tuya LAN (bulb + HE20 sensor)
```

## Installation

### 1. Install dependencies
```bash
pip install tinytuya paho-mqtt pyyaml python-dotenv insightface onnxruntime opencv-python-headless mediapipe
```

### 2. Install Mosquitto MQTT broker
Download from https://mosquitto.org/download/ and install as Windows service.

### 3. Enable in config.yaml
```yaml
smart_room:
  enabled: true
  mqtt:
    broker: "127.0.0.1"
    port: 1883
  automations:
    adaptive_light:
      enabled: true
      auto_off: true           # all modes except Focus
  vision:
    enabled: true
    camera_index: 0
    inference_fps: 1.0
    motion_threshold: 6.0
    gesture_confidence: 0.65
    auto_download_models: true
  # ... see NEEDS_YOU_AT_HOME.md for full config
```

Marvi installs this repository as an independent sidecar plugin. Runtime state,
credentials, logs, and the RPC token live under
`%LOCALAPPDATA%\Marvi-OS\plugin-data\smart_room`; the plugin does not import the
host's constants or internal modules.

Vision model files, the face library, unknown-visitor thumbnails, and camera
state stay in that same plugin-owned data directory. Frames and embeddings are
never sent through RPC or placed in Marvi's prompt. On first enable, the
sidecar downloads its official local model files; readiness remains available
over RPC while that happens.

### 4. Hardware setup
See **NEEDS_YOU_AT_HOME.md** for optional hardware calibration and soak checks.

To regenerate the password-bearing iPhone configuration without printing the
password, run:

```powershell
python scripts/create_owntracks_config.py `
  --host <PC_TAILSCALE_IP> `
  --env-file "$env:LOCALAPPDATA\Marvi-OS\plugin-data\smart_room\secrets.env" `
  --output "$env:USERPROFILE\Downloads\marvi-owntracks.otrc"
```

## Tools

| Tool | Description |
|------|-------------|
| `smart_room_state` | Full room snapshot |
| `smart_room_set_mode` | Set mode (normal/reading/focus/relax/night/sleep/alarm/off) |
| `smart_room_set_light` | Direct light control |
| `smart_room_cancel_sleep` | Cancel sleep mode |
| `smart_room_override` | Keep presence automation on, hold light on, or hold light off |
| `smart_room_health` | Device health check |
| `smart_room_diagnostic` | Full diagnostic dump |
| `smart_room_alarm` | Create/update/list/delete one-day or daily alarms; acknowledge active alarm |
| `smart_room_vision` | Read current observations, a bounded description, enrolled people, or pending visitors |
| `smart_room_vision_identity` | Enrol the owner or approve/reject a visitor identity (confirmation-sensitive in Marvi) |

## License
MIT — xRetro Labs
