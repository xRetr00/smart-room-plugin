# Smart Room — Current Setup Status

## Already complete

- Tuya plan active; bulb and HE20 keys extracted and stored as secrets.
- Bulb `192.168.1.104` and HE20 `192.168.1.182` verified over local Tuya protocol 3.5.
- ESP32-D0WD-V3 flashed with ESPresense v4.0.6 at `192.168.1.172`.
- iPhone securely enrolled as `phone:shereef`; Marvi receives it and establishes owner identity.
- ESPresense connected to authenticated Mosquitto and publishing telemetry/BLE data.
- Mosquitto running at `192.168.1.151:1883`; anonymous clients rejected.
- Tailscale connected on the PC and iPhone. The private MQTT endpoint is `100.118.1.55:1883`.
- Tailscale Serve forwards that endpoint to local Mosquitto without a public port or router forwarding.
- Python dependencies and discovery scripts verified.
- OwnTracks installed on the iPhone.
- Complete OwnTracks configuration generated at `C:\Users\xRetro\Downloads\marvi-owntracks.otrc` and sent to the iPhone with Taildrop.
- OwnTracks configuration imported, Home region created, and real Home/Bakery transitions received by Marvi.
- HE20 false-clear handling fixed: light-off waits for 60 seconds of continuous clear and is emitted once.
- Desktop Smart Room scrolling, switches, and separate owner/HE20 status verified in the packaged app.

## What is still useful to measure

No setup step is blocking normal use. The remaining work is calibration and soak testing:

- Measure locked-iPhone BLE gaps for 60 minutes at desk, bed, door, and outside the room.
- Run a real four-hour overnight presence test.
- Confirm OwnTracks background transitions on Wi-Fi and through Tailscale cellular access.
- Run the HE20 continuously for one hour and the complete room for seven days.

Shortcuts is cancelled. OwnTracks is the only phone geofence path.

## BLE choice

Keep ESPresense. It is the correct standalone MQTT firmware for this ESP32, secure IRK enrollment is working, and Marvi now receives the enrolled iPhone reliably.

A dedicated carried BLE beacon advertises more continuously than an iPhone and can be added later if a real-world soak test finds gaps. It is not needed now: ESPresense + HE20 + OwnTracks gives identity, room occupancy, and away-from-home confirmation without buying more hardware.
