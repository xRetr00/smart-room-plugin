# Smart Room — What Needs You At Home

> These tasks require physical access to the devices or your phone.
> Marvi can't do these remotely. Do them when you're back.

---

## 1. Tuya Device Key Extraction (10 minutes, one-time)

**Why:** tinytuya needs the `local_key` for each Tuya device to control it over LAN. This is a one-time extraction — after this, all control is local, no cloud.

**Steps:**
1. On your phone, open the **Smart Life** or **Tuya** app — make sure your devices are registered there
2. On your PC, go to https://iot.tuya.com/ (Tuya IoT Developer Portal)
3. Create a free account (use your email)
4. Create a new **Cloud Development** project:
   - Name: `smart_room`
   - Industry: Smart Home
   - Data Center: Central Europe (or whichever is closest)
5. Go to **Device Management** → **Link Tuya App Account**
6. Scan the QR code with your Smart Life app
7. After linking, your devices appear in the device list
8. For each device (RGBCW bulb, HE20 sensor), click to expand and copy:
   - **Device ID** (already known from `discover_tuya.py`)
   - **Local Key** (16-char hex string)
9. Add them to your config.yaml:
   ```yaml
   smart_room:
     tuya:
       bulb:
         ip: "192.168.1.x"        # from discover_tuya.py
         device_id: "xxxxxx"       # from Tuya portal
         local_key: "xxxxxxxxxxxx" # from Tuya portal
         protocol: "3.3"
       he20:
         ip: "192.168.1.x"
         device_id: "xxxxxx"
         local_key: "xxxxxxxxxxxx"
         protocol: "3.3"
   ```

**Alternative:** Use https://github.com/redphx/tuya-local-key-extractor — Python script, uses your Tuya credentials once.

---

## 2. Confirm ESP32 Model (30 seconds)

**Why:** Need to know if it's ESP32, ESP32-S2, ESP32-S3, or ESP32-C3 for ESPresense compatibility.

**How:** Look at the chip markings on the ESP32 board. The model is printed on the metal can.

---

## 3. Flash ESPresense to ESP32 (15 minutes)

**Why:** BLE presence detection for room-level tracking.

**Steps:**
1. Connect ESP32 to PC via USB cable
2. Open https://espresense.com/install in Chrome (or Edge)
3. Select your ESP32 board type
4. Click "Connect" and select the USB serial port
5. Flash the firmware (web flasher, no install needed)
6. After flashing, the ESP32 creates a WiFi AP called "espresense-xxxx"
7. Connect to that AP on your phone/PC
8. Configure:
   - WiFi SSID: your home network
   - WiFi Password: your home password
   - MQTT Broker: `192.168.1.x` (your PC's IP on the local network)
   - MQTT Port: `1883`
   - Room: `smart_room`
9. Save and reboot — ESP32 connects to your WiFi and starts scanning

---

## 4. Enroll iPhone IRK (5 minutes)

**Why:** Stable BLE identity tracking that survives MAC randomization.

**Steps:**
1. On your iPhone: Settings → Bluetooth → make sure Bluetooth is on
2. On any device with a browser, open the ESPresense web UI (find ESP32 IP from your router's DHCP list)
3. Go to **Enrollment** tab
4. Follow the instructions to enroll your iPhone's IRK
5. This uses Apple's Identity Resolving Key for stable tracking

---

## 5. Install OwnTracks on iPhone (5 minutes)

**Why:** Geofence-based location tracking (home/work/uni/bakery) without cloud.

**Steps:**
1. Install **OwnTracks** from the App Store (free, open source)
2. Open the app → Settings (gear icon)
3. Configure:
   - Mode: **MQTT** (not HTTP)
   - Broker: `192.168.1.x` (your PC IP, or Tailscale IP if remote)
   - Port: `1883`
   - Username: (leave blank for local)
   - Password: (leave blank)
   - DeviceID: `shereef`
   - TrackerID: `sh`
4. Set up **Waypoints** (geofence zones):
   - Home: your home coordinates, radius 100m
   - University: Duzce University coordinates, radius 200m
   - Bakery: bakery coordinates, radius 100m
5. Enable **Background tracking** in iOS Settings → OwnTracks → Location → Always

---

## 6. Install Mosquitto MQTT Broker (5 minutes, if not done yet)

**Why:** Message bus for ESP32, OwnTracks, and the runtime.

**Steps:**
1. Download from https://mosquitto.org/download/
2. Run the Windows installer
3. Open `mosquitto.conf` (usually in `C:\Program Files\mosquitto\`)
4. Add:
   ```
   listener 1883
   allow_anonymous true
   ```
5. Start the service: `net start Mosquitto` (or it auto-starts)
6. Verify: `mosquitto_sub -t test -h 127.0.0.1` in one terminal, `mosquitto_pub -t test -m hello` in another

---

## 7. Install Python Dependencies (2 minutes)

```bash
pip install tinytuya paho-mqtt pyyaml apscheduler
```

---

## 8. Run Discovery Scripts (5 minutes)

```bash
# 1. Discover Tuya devices on your network
cd D:\smart-room-plugin
python scripts/discover_tuya.py

# 2. Inspect device data points (after getting keys)
python scripts/inspect_dps.py --ip 192.168.1.50 --id <device_id> --key <local_key>

# 3. Inspect MQTT topics (after ESPresense + OwnTracks configured)
python scripts/inspect_mqtt.py
```

---

## Checklist

- [ ] Tuya device keys extracted (one-time)
- [ ] ESP32 model confirmed
- [ ] ESPresense flashed to ESP32
- [ ] iPhone IRK enrolled
- [ ] OwnTracks installed and configured
- [ ] Mosquitto installed and running
- [ ] Python deps installed
- [ ] Discovery scripts run
- [ ] Config.yaml filled with device IPs, keys, thresholds

Once all checked, tell Marvi "smart room setup done" and the runtime will start.
