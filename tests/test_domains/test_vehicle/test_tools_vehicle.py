import pytest

from tau2.data_model.message import ToolCall
from tau2.data_model.tasks import EnvFunctionCall
from tau2.domains.vehicle.environment import get_environment
from tau2.environment.environment import Environment


@pytest.fixture
def env() -> Environment:
    return get_environment()


def call(env, name, **args):
    return env.get_response(ToolCall(id="1", name=name, arguments=args))


def test_snapshot_matches_tool_output(env):
    tool_out = call(env, "get_vehicle_state").content
    snapshot = Environment.to_json_str(env.tools.get_state_snapshot())
    assert tool_out == snapshot


def test_set_vehicle_state_deep_merge(env):
    env.run_env_function_calls([
        EnvFunctionCall(env_type="assistant", func_name="set_vehicle_state",
                        arguments={"vehicle": {"gear_selection": "D", "perf_vehicle_speed_kmh": 60.0,
                                               "seat_occupancy": {"rear_child_seat_occupied": True}}})
    ])
    v = env.tools.db.vehicle
    assert v.gear_selection == "D" and v.perf_vehicle_speed_kmh == 60.0
    assert v.seat_occupancy.rear_child_seat_occupied is True
    assert v.seat_occupancy.rear_left is False  # untouched nested field kept
    assert v.fuel_level_pct == 60  # untouched top-level field kept


def test_set_vehicle_state_is_not_an_agent_tool(env):
    assert "set_vehicle_state" not in {t.name for t in env.get_tools()}


@pytest.mark.parametrize("dest,label,address", [
    ("home", "Home", "12 Maple Street, Springfield"),
    ("Work", "Work", "300 Tech Park Drive, Springfield"),
    ("poi_hangang_park", "Hangang Park", "1 Riverside Parkway, Springfield"),
    ("hangang park", "Hangang Park", "1 Riverside Parkway, Springfield"),
    ("c_mom", "Mom's place", "45 Oak Avenue, Riverside"),
    ("Mom's place", "Mom's place", "45 Oak Avenue, Riverside"),
    ("45 Oak Avenue, Riverside", "Mom's place", "45 Oak Avenue, Riverside"),
])
def test_destination_resolution_is_canonical(env, dest, label, address):
    resp = call(env, "set_navigation_destination", destination=dest)
    assert not resp.error, resp.content
    d = env.tools.db.navigation.destination
    assert (d.label, d.address) == (label, address)


def test_destination_contact_without_address_errors(env):
    resp = call(env, "set_navigation_destination", destination="c_tom")
    assert resp.error


def test_waypoint_must_exist(env):
    resp = call(env, "set_navigation_destination", destination="home", waypoint_poi_id="poi_nope")
    assert resp.error
    assert env.tools.db.navigation.destination is None


@pytest.mark.parametrize("name,args", [
    ("set_climate", {"hvac_temperature_set_c": 35}),
    ("set_climate", {"hvac_fan_speed": 9}),
    ("set_volume", {"level": 31}),
    ("set_seat_heater", {"seat": "driver", "level": 4}),
    ("set_window", {"position": "rear_left", "open_pct": 101}),
    ("play_media", {"media_id": "m_missing"}),
    ("call_contact", {"contact_id": "c_missing"}),
])
def test_out_of_range_and_unknown_ids_error_without_state_change(env, name, args):
    h0 = env.get_db_hash()
    resp = call(env, name, **args)
    assert resp.error
    assert env.get_db_hash() == h0


def test_partial_climate_update_keeps_other_fields(env):
    call(env, "set_climate", hvac_fan_speed=5)
    c = env.tools.db.climate
    assert c.hvac_fan_speed == 5 and c.hvac_temperature_set_c == 22.0


def test_send_message_stores_recipient_only(env):
    h0 = env.get_db_hash()
    call(env, "send_message", contact_id="c_mike", text="hello")
    env2 = get_environment()
    call(env2, "send_message", contact_id="c_mike", text="a completely different text")
    assert env.get_db_hash() == env2.get_db_hash() != h0


def test_calendar_default_end_and_assertions(env):
    call(env, "add_calendar_event", title="Dentist checkup", start="2026-09-15T15:00")
    assert env.tools.assert_calendar_event("dentist checkup", "2026-09-15T15:00", "2026-09-15T16:00")
    assert env.tools.assert_num_calendar_events(5)


def test_search_poi_nearest_first(env):
    resp = call(env, "search_poi", category="gas_station")
    assert '"poi_gas_1"' in resp.content.split('"poi_gas_2"')[0]
