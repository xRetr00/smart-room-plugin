# Marvi Smart Room Plugin

> Native Windows smart room engine for Marvi — replaces Home Assistant Docker/WSL for the smart room.

## What It Does

- **Presence fusion**: BLE (ESP32/ESPresense) + mmWave (HE20) + OwnTracks geofence → room-level presence with iPhone deep-sleep handling
- **Tuya LAN control**: Direct control of RGBCW bulb + HE20 sensor via tinytuya — no cloud
- **Room automations**: Adaptive light, sleep/alarm behavior, work-return settle/cancel, evening sleep, and daily resets
- **Sound controls**: Plugin-local quantized YAMNet detection — double clap toggles the light; triple clap enters Sleep; a lone clap is ignored
- **World-awareness**: Marvi knows where you are, what mode the room is in, light state — as ambient context, not memory writes
- **Marvi integration**: Plugin tools (`smart_room_state`, `smart_room_set_mode`, etc.) + session context line + subconscious transitions

## Architecture

```
ESP32 (ESPresense) ──MQTT──┐
                           ├──→ Mosquitto MQTT ──→ Runtime (Python) ──→ Marvi Plugin
OwnTracks (iPhone) ──MQTT───┘
                                                  Tuya Controller (tinytuya)
                                                  (RGBCW bulb + HE20 sensor)
Microphone ──transient gate──YAMNet──clap sequence──────────┘
```

## Installation

### 1. Install dependencies
```bash
pip install tinytuya paho-mqtt pyyaml ai-edge-litert==2.1.6 sounddevice==0.5.5
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
  sound_events:
    enabled: true
    # input_device: null       # default Windows recording device
    confidence: 0.45
    min_peak: 0.12
    noise_multiplier: 8.0
    min_crest: 3.0
    candidate_refractory_ms: 250
    model_delay_ms: 150
    max_gap_ms: 900
    decision_ms: 650
    cooldown_ms: 3000
    require_occupancy: true
  # ... see NEEDS_YOU_AT_HOME.md for full config
```

The first enabled start downloads the official 4 MB quantized YAMNet model to
the Marvi profile and verifies its pinned SHA-256 before loading it. Audio never
leaves the PC and is not sent to STT or an LLM. If claps are missed, first lower
`min_peak` slightly; if other sharp sounds trigger candidates, raise it. The
YAMNet confidence check remains the final decision.

The plugin is bundled with Marvi; no symlink or second repository is needed.

### 4. Hardware setup
See **NEEDS_YOU_AT_HOME.md** for the remaining OwnTracks import and Home-region steps.

To regenerate the password-bearing iPhone configuration without printing the
password, run:

```powershell
python plugins/smart_room/scripts/create_owntracks_config.py `
  --host <PC_TAILSCALE_IP> `
  --env-file "$env:LOCALAPPDATA\hermes\.env" `
  --output "$env:USERPROFILE\Downloads\marvi-owntracks.otrc"
```

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
See `D:\hermes-agent\docs\superpowers\specs\2026-07-14-marvi-smart-room-plugin-v0.3.md` for the current v0.4 revision (the original path is retained for existing links).

## License
MIT — xRetro Labs
