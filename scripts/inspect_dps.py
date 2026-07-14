#!/usr/bin/env python3
"""Phase 0: Inspect Tuya device DPS (data points).

Once you have the local key, this script connects to a Tuya device
and lists all available data points (DPS). Use this to determine
which DPS control on/off, brightness, color, etc.

Usage:
    python scripts/inspect_dps.py --ip 192.168.1.50 --id <device_id> --key <local_key> [--protocol 3.3]

Install tinytuya first:
    pip install tinytuya
"""

import argparse
import json
import sys

def main():
    parser = argparse.ArgumentParser(description="Inspect Tuya device data points")
    parser.add_argument("--ip", required=True, help="Device IP address")
    parser.add_argument("--id", required=True, dest="device_id", help="Device ID")
    parser.add_argument("--key", required=True, help="Local key")
    parser.add_argument("--protocol", default="3.3", help="Protocol version (3.1, 3.2, 3.3, 3.4)")
    args = parser.parse_args()

    try:
        import tinytuya
    except ImportError:
        print("ERROR: tinytuya not installed. Run: pip install tinytuya")
        sys.exit(1)

    print(f"Connecting to Tuya device at {args.ip} (ID: {args.device_id}, protocol: {args.protocol})...")

    dev = tinytuya.Device(
        dev_id=args.device_id,
        address=args.ip,
    )
    dev.set_version(float(args.protocol))

    try:
        status = dev.status()
        dps = status.get("dps", {})

        print(f"\nDevice online: {bool(dps)}")
        print(f"Data points ({len(dps)} found):\n")

        for dp_id, dp_value in sorted(dps.items(), key=lambda x: int(x[0])):
            print(f"  DP {dp_id}: {dp_value!r} ({type(dp_value).__name__})")

        print("\n--- COMMON DPS MAPPING ---")
        print("  DP 1:  Switch (on/off)")
        print("  DP 2:  Brightness (0-255 or 0-1000)")
        print("  DP 3:  Color temperature (0-255, warm to cool)")
        print("  DP 4:  Color mode (0=white, 1=color, 2=scene)")
        print("  DP 5:  Color (hex: rrggbb or hhhhssvv)")
        print("  DP 6:  Scene")
        print()

        # Save to file
        out = {"ip": args.ip, "device_id": args.device_id, "dps": dps}
        with open("tuya_dps_inspect.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("Saved to tuya_dps_inspect.json")

    except Exception as e:
        print(f"ERROR: {e}")
        print("\nTroubleshooting:")
        print("  - Check the IP, device ID, and local key are correct")
        print("  - Make sure the device is powered on and on the same network")
        print("  - Try protocol version 3.4 if 3.3 doesn't work")
        sys.exit(1)


if __name__ == "__main__":
    main()
