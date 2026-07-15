"""MQTT contracts for OwnTracks and ESPresense v4 topics."""

from __future__ import annotations

from unittest.mock import MagicMock

from plugins.smart_room.runtime.mqtt.client import MQTTClient


def _client(owner_device_id: str = "irk:iphone") -> tuple[MQTTClient, MagicMock, MagicMock, MagicMock]:
    presence = MagicMock()
    geofence = MagicMock()
    node_status = MagicMock()
    client = MQTTClient(
        {
            "esp32": {
                "room_id": "smart_room",
                "owner_device_id": owner_device_id,
                "rssi_enter_threshold": -70,
                "rssi_exit_threshold": -85,
            }
        },
        on_presence=presence,
        on_geofence=geofence,
        on_command=MagicMock(),
        on_node_status=node_status,
    )
    return client, presence, geofence, node_status


def test_espresense_v4_owner_topic_drives_presence_and_ignores_fingerprints():
    client, presence, _, _ = _client()
    client._handle_espresense(
        "espresense/devices/irk:iphone/smart_room",
        {"id": "irk:iphone", "rssi": -64.6, "distance": 1.1},
    )
    presence.assert_called_once_with(True, -65, "irk:iphone")

    client._handle_espresense(
        "espresense/devices/apple:1006:10-12/smart_room",
        {"id": "apple:1006:10-12", "rssi": -55},
    )
    assert presence.call_count == 1


def test_espresense_room_topics_update_health_not_presence():
    client, presence, _, node_status = _client()
    client._handle_espresense(
        "espresense/rooms/smart_room/status", {"value": "online"}
    )
    client._handle_espresense(
        "espresense/rooms/smart_room/telemetry", {"ip": "192.168.1.172"}
    )
    assert node_status.call_args_list[0].args == (True, None)
    assert node_status.call_args_list[1].args == (True, "192.168.1.172")
    presence.assert_not_called()


def test_unenrolled_node_never_guesses_owner_from_generic_apple_id():
    client, presence, _, _ = _client(owner_device_id="")
    client._handle_espresense(
        "espresense/devices/apple:1006:10-12/smart_room",
        {"id": "apple:1006:10-12", "rssi": -50},
    )
    presence.assert_not_called()


def test_owntracks_transition_normalizes_zone():
    client, _, geofence, _ = _client()
    client._handle_owntracks(
        {"_type": "transition", "event": "enter", "desc": " Home "}
    )
    geofence.assert_called_once_with("enter", "home")


def test_owntracks_location_initializes_current_region():
    client, _, geofence, _ = _client()
    client._handle_owntracks({"_type": "location", "inregions": [" Home "]})
    geofence.assert_called_once_with("sync", "home")
