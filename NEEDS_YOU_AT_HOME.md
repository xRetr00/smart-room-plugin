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
- HE20 false-clear handling fixed: light-off waits for 60 seconds of continuous clear and is emitted once.
- Desktop Smart Room scrolling, switches, and separate owner/HE20 status verified in the packaged app.

## What you still need to do

### 1. Import the OwnTracks configuration on iPhone

OwnTracks 26.2.3 disables external configuration files by default as a security precaution.

1. Open OwnTracks → Settings and enable external configuration-file loading.
2. Open the Taildrop file `marvi-owntracks.otrc` from the iPhone Files app and choose OwnTracks.
3. Review the configuration diff and approve the import.
4. The imported values should show MQTT mode, host `100.118.1.55`, port `1883`, User ID `smart_room`, Device ID `iphone`, TLS off, and authentication on.

The private Tailscale tunnel provides transport encryption, so TLS is intentionally off at the local Mosquitto endpoint. Port 1883 is not exposed publicly.

### 2. Grant location permissions and add Home

1. Give OwnTracks **Always** location permission, Precise Location, Background App Refresh, and motion access when requested.
2. In OwnTracks Regions, create a circular region named exactly `home` around the house (about 150–200 m radius).
3. Use Significant monitoring mode for normal battery-friendly operation.
4. Publish the current location once.
5. Tell Marvi that the OwnTracks import is complete so the MQTT location and region transition can be verified live.

Shortcuts is cancelled. OwnTracks is the only phone geofence path.

## BLE choice

Keep ESPresense. It is the correct standalone MQTT firmware for this ESP32, secure IRK enrollment is working, and Marvi now receives the enrolled iPhone reliably.

A dedicated carried BLE beacon advertises more continuously than an iPhone and can be added later if a real-world soak test finds gaps. It is not needed now: ESPresense + HE20 + OwnTracks gives identity, room occupancy, and away-from-home confirmation without buying more hardware.
