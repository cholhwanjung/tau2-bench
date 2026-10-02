"""Data model for the `vehicle` domain.

Field names follow Android Automotive OS `VehiclePropertyIds` where a
property exists (IGNITION_STATE, GEAR_SELECTION, PERF_VEHICLE_SPEED,
DOOR_LOCK, DOOR_CHILD_LOCK_ENABLED, WINDOW_POS, SEAT_OCCUPANCY, FUEL_LEVEL,
EV_BATTERY_LEVEL, RANGE_REMAINING, ENV_OUTSIDE_TEMPERATURE, HVAC_*). Units are
converted to human-readable ones (km/h, %, km, °C).
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tau2.domains.vehicle.utils import VEHICLE_DB_PATH
from tau2.environment.db import DB

Gear = Literal["P", "R", "N", "D"]
IgnitionState = Literal["ON", "OFF", "ACC"]
FuelType = Literal["gasoline", "ev"]
DriveMode = Literal["eco", "normal", "sport"]
FanDirection = Literal["face", "feet", "face_feet", "defrost"]
WindowPosition = Literal["front_left", "front_right", "rear_left", "rear_right"]
HeaterSeat = Literal["driver", "passenger"]
MediaType = Literal["audio", "video"]
Rating = Literal["G", "PG", "R"]
PlaybackState = Literal["playing", "paused", "stopped"]


class Location(BaseModel):
    label: str = Field(description="Human-readable place name")
    address: str = Field(description="Street address")


class WindowPositions(BaseModel):
    front_left: int = Field(0, description="Driver window, percent open (0-100)")
    front_right: int = Field(0, description="Front passenger window, percent open")
    rear_left: int = Field(0, description="Rear left window, percent open")
    rear_right: int = Field(0, description="Rear right window, percent open")


class SeatOccupancy(BaseModel):
    front_passenger: bool = Field(False, description="Front passenger seat occupied")
    rear_left: bool = Field(False, description="Rear left seat occupied")
    rear_right: bool = Field(False, description="Rear right seat occupied")
    rear_child_seat_occupied: bool = Field(
        False, description="A child is detected in the rear child seat"
    )


class VehicleState(BaseModel):
    ignition_state: IgnitionState = Field("ON", description="IGNITION_STATE")
    gear_selection: Gear = Field("P", description="GEAR_SELECTION: P, R, N or D")
    perf_vehicle_speed_kmh: float = Field(0.0, description="PERF_VEHICLE_SPEED in km/h")
    parking_brake_on: bool = Field(True, description="PARKING_BRAKE_ON")
    drive_mode: DriveMode = Field("normal", description="eco, normal or sport")
    fuel_type: FuelType = Field("gasoline", description="INFO_FUEL_TYPE")
    fuel_level_pct: Optional[int] = Field(
        60, description="FUEL_LEVEL as percent (gasoline cars only)"
    )
    ev_battery_level_pct: Optional[int] = Field(
        None, description="EV_BATTERY_LEVEL as percent (EVs only)"
    )
    range_remaining_km: int = Field(420, description="RANGE_REMAINING in km")
    env_outside_temperature_c: float = Field(
        24.0, description="ENV_OUTSIDE_TEMPERATURE in °C"
    )
    location: Location = Field(description="Current vehicle location")
    door_lock: bool = Field(True, description="DOOR_LOCK: True if all doors are locked")
    door_child_lock_enabled: bool = Field(
        False, description="DOOR_CHILD_LOCK_ENABLED for the rear doors and windows"
    )
    window_pos_pct: WindowPositions = Field(description="WINDOW_POS per window")
    seat_occupancy: SeatOccupancy = Field(description="SEAT_OCCUPANCY per seat")


class ClimateState(BaseModel):
    hvac_power_on: bool = Field(True, description="HVAC_POWER_ON")
    hvac_ac_on: bool = Field(True, description="HVAC_AC_ON")
    hvac_auto_on: bool = Field(True, description="HVAC_AUTO_ON")
    hvac_temperature_set_c: float = Field(
        22.0, description="HVAC_TEMPERATURE_SET target in °C (16-30)"
    )
    hvac_fan_speed: int = Field(3, description="HVAC_FAN_SPEED (0-7)")
    hvac_fan_direction: FanDirection = Field("face", description="HVAC_FAN_DIRECTION")
    hvac_seat_temperature: Dict[HeaterSeat, int] = Field(
        description="HVAC_SEAT_TEMPERATURE: seat heater level per seat (0-3)"
    )


class MediaState(BaseModel):
    playback_state: PlaybackState = Field("stopped", description="Playback state")
    now_playing_id: Optional[str] = Field(None, description="media_id being played")
    volume: int = Field(12, description="Media volume (0-30)")


class NavigationState(BaseModel):
    destination: Optional[Location] = Field(None, description="Active destination")
    waypoints: List[str] = Field(
        default_factory=list, description="poi_ids visited before the destination"
    )


class PhoneState(BaseModel):
    active_call_contact_id: Optional[str] = Field(
        None, description="contact_id of the active call"
    )
    sent_message_recipients: List[str] = Field(
        default_factory=list,
        description="contact_ids that received a message this drive (text not stored)",
    )


class Contact(BaseModel):
    contact_id: str
    name: str
    phone: str
    favorite: bool = Field(description="Favorite contacts may be called while driving")
    relation: str
    address: Optional[str] = Field(None, description="Home address, if known")


class CalendarEvent(BaseModel):
    event_id: str
    title: str
    start: str = Field(description="ISO datetime YYYY-MM-DDTHH:MM")
    end: str = Field(description="ISO datetime YYYY-MM-DDTHH:MM")
    location: Optional[str] = None


class MediaItem(BaseModel):
    media_id: str
    title: str
    artist: str
    media_type: MediaType
    rating: Rating


class POI(BaseModel):
    poi_id: str
    name: str
    category: str
    address: str
    distance_km: float


class UserProfile(BaseModel):
    name: str
    home_address: str
    work_address: str
    contacts: Dict[str, Contact]
    media_library: Dict[str, MediaItem]


class VehicleStateSnapshot(BaseModel):
    """Everything `get_vehicle_state` returns (and condition B injects)."""

    vehicle: VehicleState
    climate: ClimateState
    media: MediaState
    navigation: NavigationState
    phone: PhoneState


class VehicleDB(DB):
    """Single-vehicle database: live state + static user data."""

    vehicle: VehicleState
    climate: ClimateState
    media: MediaState
    navigation: NavigationState
    phone: PhoneState
    calendar: Dict[str, CalendarEvent]
    user_profile: UserProfile
    pois: Dict[str, POI]


def get_db() -> VehicleDB:
    return VehicleDB.load(VEHICLE_DB_PATH)
