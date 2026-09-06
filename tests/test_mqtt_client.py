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
                "enter_debounce_seconds": 0,
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


def test_owntracks_location_outside_every_region_clears_the_zone():
    """An empty `inregions` is a statement, not a silence.

    It says "in no region at all", and discarding it meant leaving home
    without a `transition` left the state reading `home` until something else
    corrected it -- and nothing else does. The owner's own report log has this
    exact payload in it, with the battery at 10%.
    """
    client, _, geofence, _ = _client()
    client._handle_owntracks({"_type": "location", "inregions": [], "batt": 10, "bs": 1})
    geofence.assert_called_once_with("sync", "")


def test_owntracks_location_without_regions_says_nothing():
    # No `inregions` key at all is genuinely no news -- an older client, or a
    # payload that carries something else -- and must not clear the zone.
    client, _, geofence, _ = _client()
    client._handle_owntracks({"_type": "location", "lat": 41.1, "lon": 29.2})
    geofence.assert_not_called()


def test_every_owntracks_message_is_forwarded_for_history():
    report = MagicMock()
    client, _, _, _ = _client()
    client._on_owntracks = report
    payload = {"_type": "location", "lat": 41.1, "lon": 29.2, "tst": 1784358126}

    client._handle_owntracks(payload, "owntracks/smart_room/iphone")

    report.assert_called_once_with("owntracks/smart_room/iphone", payload)


def test_a_failing_handler_does_not_kill_the_client():
    """paho runs callbacks on its network thread, and an escape kills it.

    A `PermissionError` from an ordinary Windows file lock, raised inside the
    ESPresense handler, unsubscribed everything at once -- OwnTracks reports
    stopped arriving for a fortnight and nothing reported a fault, because
    from the room's point of view nothing had failed.
    """

    class _Message:
        topic = "espresense/rooms/smart_room/telemetry"
        payload = b'{"ip": "192.168.1.172"}'

    client, _, _, node_status = _client()
    node_status.side_effect = PermissionError("state.json is busy")

    # Must not raise: the network thread has to survive this.
    client._on_message(None, None, _Message())

    # And the next message still gets through.
    node_status.side_effect = None
    client._on_message(None, None, _Message())
    assert node_status.call_count == 2


def test_a_message_that_is_not_json_is_still_routed():
    class _Message:
        topic = "espresense/rooms/smart_room/status"
        payload = b"online"

    client, _, _, node_status = _client()
    client._on_message(None, None, _Message())
    node_status.assert_called_once_with(True, None)


def test_marvi_can_ask_the_phone_where_it_is():
    """The one thing missing from a system built on the phone volunteering."""
    client, _, _, _ = _client()
    client._owntracks_topic = "owntracks/smart_room/#"
    client._client = MagicMock()
    client._connected = True

    assert client.ask_phone_to_report() is True
    topic, body = client._client.publish.call_args[0]
    assert topic == "owntracks/smart_room/iphone/cmd"
    import json as _json
    assert _json.loads(body) == {"_type": "cmd", "action": "reportLocation"}


def test_it_does_not_pretend_to_ask_while_disconnected():
    client, _, _, _ = _client()
    client._connected = False
    assert client.ask_phone_to_report() is False
