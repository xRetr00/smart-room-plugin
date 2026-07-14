#!/usr/bin/env python3
"""Phase 0: Discover Tuya devices on the local network.

Run this to find all Tuya devices on your Wi-Fi. It uses tinytuya's
built-in network scanner (UDP broadcast on ports 6666/6667/7000).

Usage:
    python scripts/discover_tuya.py

NOTE: This discovers devices but does NOT get local keys.
      To get local keys, you need to use the Tuya IoT Developer Portal
      (free account, one-time) or the tuya-local-key-extractor tool.

Install tinytuya first:
    pip install tinytuya
"""

import json
import sys

def main():
    try:
        import tinytuya
    except ImportError:
        print("ERROR: tinytuya not installed. Run: pip install tinytuya")
        sys.exit(1)

    print("Scanning for Tuya devices on local network (10 seconds)...")
    print("(Make sure your Tuya devices are powered on and connected to Wi-Fi)")
    print()

    devices = tinytuya.scan(maxretry=3, timeout=10)

    if not devices:
        print("No Tuya devices found. Make sure:")
        print("  - Devices are powered on")
        print("  - Devices are connected to the same Wi-Fi network as this PC")
        print("  - Your firewall allows UDP ports 6666, 6667, 7000")
        sys.exit(0)

    print(f"Found {len(devices)} device(s):\n")

    output = []
    for dev in devices:
        ip = dev.get("ip", "?")
        dev_id = dev.get("gwId", "?")
        version = dev.get("version", "?")
        product = dev.get("productKey", "?")

        entry = {
            "ip": ip,
            "device_id": dev_id,
            "protocol_version": version,
            "product_key": product,
        }
        output.append(entry)

        print(f"  IP: {ip}")
        print(f"  Device ID: {dev_id}")
        print(f"  Protocol: v{version}")
        print(f"  Product Key: {product}")
        print()

    print("\n--- NEXT STEPS ---")
    print("1. Note the IP addresses and Device IDs above")
    print("2. Get local keys from Tuya IoT Portal:")
    print("   a. Go to https://iot.tuya.com/ (free account)")
    print("   b. Create a Cloud Development project")
    print("   c. Link your Smart Life app (scan QR code)")
    print("   d. Go to Device Management → find your devices")
    print("   e. Copy the 'local_key' for each device")
    print("3. Add to config.yaml under smart_room.tuya:")
    print("   bulb:")
    print(f"     ip: \"{output[0]['ip']}\"")
    print(f"     device_id: \"{output[0]['device_id']}\"")
    print("     local_key: \"<from Tuya portal>\"")
    print("     protocol: \"3.3\"")

    # Save to file for reference
    with open("tuya_devices_found.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to tuya_devices_found.json")


if __name__ == "__main__":
    main()
