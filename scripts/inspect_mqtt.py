#!/usr/bin/env python3
"""Phase 0: Inspect MQTT topics on the local broker.

Connects to the local Mosquitto broker and listens for messages on
relevant topics for 30 seconds. Helps verify ESPresense and OwnTracks
are publishing correctly.

Usage:
    python scripts/inspect_mqtt.py [--broker 127.0.0.1] [--port 1883]

Install paho-mqtt first:
    pip install paho-mqtt
"""

import argparse
import json
import os
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Inspect MQTT topics")
    parser.add_argument("--broker", default="127.0.0.1", help="MQTT broker IP")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--duration", type=int, default=30, help="Listen duration (seconds)")
    parser.add_argument("--username", default=os.getenv("SMART_ROOM_MQTT_USERNAME", ""), help="MQTT username (defaults to SMART_ROOM_MQTT_USERNAME)")
    parser.add_argument("--password", default=os.getenv("SMART_ROOM_MQTT_PASSWORD", ""), help="MQTT password (defaults to SMART_ROOM_MQTT_PASSWORD)")
    args = parser.parse_args()

    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("ERROR: paho-mqtt not installed. Run: pip install paho-mqtt")
        sys.exit(1)

    topics = [
        ("owntracks/#", 0),
        ("espresense/#", 0),
        ("smart_room/#", 0),
    ]

    messages = []

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print(f"Connected to MQTT broker at {args.broker}:{args.port}")
            for topic, qos in topics:
                client.subscribe(topic, qos)
                print(f"  Subscribed to: {topic}")
        else:
            print(f"Connection failed (rc={rc})")
            sys.exit(1)

    def on_message(client, userdata, msg):
        timestamp = time.strftime("%H:%M:%S")
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            payload_str = json.dumps(payload)[:200]
        except Exception:
            payload_str = msg.payload.decode("utf-8", errors="replace")[:200]

        print(f"  [{timestamp}] {msg.topic}: {payload_str}")
        messages.append({"topic": msg.topic, "payload": payload_str, "timestamp": timestamp})

    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, client_id="smart_room_inspector")
    else:
        client = mqtt.Client(client_id="smart_room_inspector")
    if args.username:
        client.username_pw_set(args.username, args.password)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(args.broker, args.port, 60)
    except Exception as e:
        print(f"ERROR: Cannot connect to MQTT broker: {e}")
        print("Make sure Mosquitto is running:")
        print("  - Check service: sc query Mosquitto (Windows)")
        print("  - Or install: https://mosquitto.org/download/")
        sys.exit(1)

    print(f"\nListening for {args.duration} seconds...\n")
    client.loop_start()
    time.sleep(args.duration)
    client.loop_stop()
    client.disconnect()

    print(f"\n--- SUMMARY ---")
    print(f"Received {len(messages)} messages in {args.duration}s")

    if not messages:
        print("\nNo messages received. Check:")
        print("  - ESPresense: is ESP32 configured and publishing?")
        print("  - OwnTracks: is the app configured with broker IP?")
        print("  - Mosquitto: is the service running?")
    else:
        topics_seen = set(m["topic"] for m in messages)
        print(f"Topics seen: {topics_seen}")

    # Save to file
    with open("mqtt_inspect.json", "w") as f:
        json.dump(messages, f, indent=2)
    print("\nSaved to mqtt_inspect.json")


if __name__ == "__main__":
    main()
