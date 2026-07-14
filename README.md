# Marvi Smart Room Plugin

> Native Windows smart room engine for Marvi — replaces Home Assistant Docker/WSL for the smart room.

## What It Does

- **Presence fusion**: BLE (ESP32/ESPresense) + mmWave (HE20) + geofence (OwnTracks) → room-level presence with iPhone deep-sleep handling
- **Tuya LAN control**: Direct control of RGBCW bulb + HE20 sensor via tinytuya — no cloud
- **14 automations**: All ported from Home Assistant (adaptive light, sleep mode, alarm, work return, evening sleep, etc.)
- **World-awareness**: Marvi knows where you are, what mode the room is in, light state — as ambient context, not memory writes
- **Marvi integration**: Plugin tools (`smart_room_state`, `smart_room_set_mode`, etc.) + session context line + subconscious transitions

## Architecture

```
ESP32 (ESPresense) ──MQTT──┐
                           ├──→ Mosquitto MQTT ──→ Runtime (Python) ──→ Marvi Plugin
OwnTracks (iPhone)  ──MQTT──┘                          ↑
                                                  Tuya Controller (tinytuya)
                                                  (RGBCW bulb + HE20 sensor)
```

## Installation

### 1. Install dependencies
```bash
pip install tinytuya paho-mqtt pyyaml apscheduler
```

### 2. Install Mosquitto MQTT broker
Download from https://mosquitto.org/download/ and install as Windows service.

### 3. Link plugin into Marvi
```bash
# In the Marvi repo (D:\hermes-agent):
mklink /D plugins\smart_room D:\smart-room-plugin
# Or copy the directory:
xcopy D:\smart-room-plugin D:\hermes-agent\plugins\smart_room\ /E /I
```

### 4. Enable in config.yaml
```yaml
plugins:
  enabled:
    - smart_room

smart_room:
  enabled: true
  mqtt:
    broker: "127.0.0.1"
    port: 1883
  # ... see NEEDS_YOU_AT_HOME.md for full config
```

### 5. Hardware setup
See **NEEDS_YOU_AT_HOME.md** for the full checklist (Tuya keys, ESP32 flash, OwnTracks setup).

## Tools

| Tool | Description |
|------|-------------|
| `smart_room_state` | Full room snapshot |
| `smart_room_set_mode` | Set mode (reading/focus/relax/sleep/alarm/off) |
| `smart_room_set_light` | Direct light control |
| `smart_room_cancel_sleep` | Cancel sleep mode |
| `smart_room_override` | Toggle manual override |
| `smart_room_health` | Device health check |
| `smart_room_diagnostic` | Full diagnostic dump |

## Spec
See `D:\hermes-agent\docs\superpowers\specs\2026-07-14-marvi-smart-room-plugin-v0.3.md` for the full architecture spec.

## License
MIT — xRetro Labs
