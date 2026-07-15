---
name: smart_room
description: "Control and monitor the smart room — lights, presence, modes, automations."
version: 0.4.0
author: xRetro Labs
---

# Smart Room Skill

## When to use

Use smart_room tools when the user asks about:
- Room state ("am I home?", "is the light on?", "what mode is the room in?")
- Light control ("turn on lights", "dim to 40%", "reading mode")
- Mode changes ("sleep mode", "focus mode", "alarm off")
- Presence ("did I leave?", "when did I get home?")
- Device health ("is the bulb online?", "is ESP32 connected?")
- Diagnostics ("why is the light not working?", "run diagnostics")

## Tools

| Tool | What it does |
|------|-------------|
| `smart_room_state` | Full room snapshot — presence, light, modes, devices, location |
| `smart_room_set_mode` | Set mode: reading, focus, relax, sleep, alarm, off |
| `smart_room_set_light` | Direct light control — on/off, brightness, color temp, RGB |
| `smart_room_cancel_sleep` | Cancel sleep mode and restore previous state |
| `smart_room_override` | Toggle manual override (disables presence-based automations) |
| `smart_room_health` | Device health check — online/offline status, last seen |
| `smart_room_diagnostic` | Full diagnostic dump for troubleshooting |

## Context

The smart room plugin provides ambient world-context to Marvi:
- One compact context line injected at session start
- Subconscious diff on meaningful transitions (mode changes, arrivals/departures)
- The runtime NEVER writes memory — the subconscious proposes, runtime supplies raw events

## Fusion Model

Presence is determined by fusing three signals:
1. **BLE** (ESP32/ESPresense) — strong evidence FOR identity, weak evidence AGAINST
2. **mmWave** (HE20 sensor) — room occupancy, no identity
3. **Geofence** (OwnTracks over authenticated MQTT) — home/away/work/uni zones

Key axiom: iPhone BLE silence is NOT absence (deep-sleep). Identity is sticky:
once BLE establishes presence, mmWave occupancy holds it even if BLE goes silent.
Confidence decays over time (0.95 → floor 0.6 over 2h).

## Automations

The runtime provides:
- Adaptive light on/off based on presence
- Sleep mode with darkness enforcement
- Alarm mode with bright flash (configurable time, default disabled)
- Reading/Focus/Relax scene modes
- Work return sleep (geofence + time window)
- Evening sleep schedule
- Daily mode flag reset

## Device Setup

### Tuya devices (one-time key extraction)
1. Run `python -m tinytuya scan` to discover devices on local network
2. Get local keys from Tuya IoT Portal (free account, one-time)
3. Save the keys in Desktop Settings → Smart Room; they are stored as secrets, not in config.yaml
4. After keys are set, all control is LAN-only — no cloud dependency

### ESP32 (ESPresense)
1. Flash ESPresense via https://espresense.com/install
2. Configure via web UI: Wi-Fi, MQTT broker IP, room name "smart_room"
3. Enroll iPhone IRK for stable identity tracking
4. Copy the enrolled device ID into Desktop Settings → Smart Room → Owner Device ID

### iPhone location
1. Configure OwnTracks in MQTT mode with the Smart Room broker credentials
2. Set the device/user topic to match `smart_room.owntracks.topic`
3. Create the `home` region and enable background location with Always permission
4. Use a private VPN such as Tailscale for reliable transitions away from home; never expose plaintext MQTT port 1883 to the internet
