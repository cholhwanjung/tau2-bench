"""Agent tools for the `vehicle` domain.

READ tools query state; WRITE tools change it. Tools are deliberately
permissive (no built-in safety interlock): the agent must apply the policy.
`set_vehicle_state` is not a tool; it is only used by task
`initialization_actions` to set up the initial vehicle state.
"""

import re
from datetime import datetime, timedelta
from typing import Optional

from tau2.domains.vehicle.data_model import (
    CalendarEvent,
    ClimateState,
    Contact,
    DriveMode,
    FanDirection,
    HeaterSeat,
    Location,
    MediaItem,
    MediaState,
    NavigationState,
    PhoneState,
    POI,
    VehicleDB,
    VehicleState,
    VehicleStateSnapshot,
    WindowPosition,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _deep_update(base: dict, upd: dict) -> dict:
    for key, value in upd.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


class VehicleTools(ToolKitBase):
    """Tools of the in-vehicle assistant."""

    db: VehicleDB

    def __init__(self, db: VehicleDB) -> None:
        super().__init__(db)

    # ------------------------------------------------------------------ READ
    @is_tool(ToolType.READ)
    def get_vehicle_state(self) -> VehicleStateSnapshot:
        """
        Get the live vehicle state: ignition, gear, speed, parking brake, drive mode,
        fuel type, fuel or battery level, remaining range, outside temperature,
        location, door lock, child lock, window positions and seat occupancy,
        plus the current climate, media, navigation and phone status.

        Returns:
            A snapshot of the vehicle, climate, media, navigation and phone state.
        """
        return self.get_state_snapshot()

    def get_state_snapshot(self) -> VehicleStateSnapshot:
        """Same object `get_vehicle_state` returns (used by the context agent)."""
        return VehicleStateSnapshot(
            vehicle=self.db.vehicle,
            climate=self.db.climate,
            media=self.db.media,
            navigation=self.db.navigation,
            phone=self.db.phone,
        )

    @is_tool(ToolType.READ)
    def get_user_profile(self) -> dict:
        """
        Get the driver's profile: name, home and work addresses, contacts and media library.

        Returns:
            The profile with `contacts` and `media_library` as lists.
        """
        profile = self.db.user_profile
        return {
            "name": profile.name,
            "home_address": profile.home_address,
            "work_address": profile.work_address,
            "contacts": list(profile.contacts.values()),
            "media_library": list(profile.media_library.values()),
        }

    @is_tool(ToolType.READ)
    def find_contact(self, name: str) -> list[Contact]:
        """
        Find contacts whose name contains the given text (case-insensitive).

        Args:
            name: Full or partial contact name.

        Returns:
            Matching contacts (empty list if none).
        """
        needle = name.strip().lower()
        return [
            c for c in self.db.user_profile.contacts.values() if needle in c.name.lower()
        ]

    @is_tool(ToolType.READ)
    def get_calendar_events(self, date: str) -> list[CalendarEvent]:
        """
        Get the calendar events on a date.

        Args:
            date: Date in YYYY-MM-DD format.

        Returns:
            Events starting on that date, ordered by start time.
        """
        events = [e for e in self.db.calendar.values() if e.start.startswith(date)]
        return sorted(events, key=lambda e: e.start)

    @is_tool(ToolType.READ)
    def search_poi(self, category: str, max_distance_km: float = 25.0) -> list[POI]:
        """
        Search points of interest near the vehicle, nearest first.

        Args:
            category: One of gas_station, ev_charging, italian_restaurant,
                korean_restaurant, cafe, parking, park, pharmacy, hospital.
            max_distance_km: Maximum distance from the vehicle in km (default 25).

        Returns:
            Matching POIs sorted by distance (nearest first).
        """
        cat = category.strip().lower()
        pois = [
            p
            for p in self.db.pois.values()
            if p.category == cat and p.distance_km <= max_distance_km
        ]
        return sorted(pois, key=lambda p: p.distance_km)

    @is_tool(ToolType.READ)
    def search_media(self, query: str) -> list[MediaItem]:
        """
        Search the media library by keywords in the title or artist. Returns the
        best matches first (items sharing the most words with the query).

        Args:
            query: Words to search for, e.g. "dune trailer" or "chill playlist".

        Returns:
            Up to 5 matching media items with their media_id, type (audio/video) and rating.
        """
        q = _tokens(query)
        scored = []
        for m in self.db.user_profile.media_library.values():
            overlap = len(q & _tokens(f"{m.title} {m.artist}"))
            if overlap:
                scored.append((-overlap, m.title, m))
        return [m for _, _, m in sorted(scored, key=lambda x: (x[0], x[1]))[:5]]

    # ----------------------------------------------------------------- WRITE
    @is_tool(ToolType.WRITE)
    def set_climate(
        self,
        hvac_temperature_set_c: Optional[float] = None,
        hvac_fan_speed: Optional[int] = None,
        hvac_ac_on: Optional[bool] = None,
        hvac_fan_direction: Optional[FanDirection] = None,
    ) -> ClimateState:
        """
        Change climate settings. Only the given fields change.

        Args:
            hvac_temperature_set_c: Target temperature in °C (16-30).
            hvac_fan_speed: Fan speed (0-7).
            hvac_ac_on: Turn the air conditioner on or off.
            hvac_fan_direction: face, feet, face_feet or defrost.

        Returns:
            The updated climate state.

        Raises:
            ValueError: If a value is out of range.
        """
        climate = self.db.climate
        if hvac_temperature_set_c is not None:
            if not 16 <= hvac_temperature_set_c <= 30:
                raise ValueError("Temperature must be between 16 and 30 °C")
            climate.hvac_temperature_set_c = float(hvac_temperature_set_c)
        if hvac_fan_speed is not None:
            if not 0 <= hvac_fan_speed <= 7:
                raise ValueError("Fan speed must be between 0 and 7")
            climate.hvac_fan_speed = int(hvac_fan_speed)
        if hvac_ac_on is not None:
            climate.hvac_ac_on = bool(hvac_ac_on)
        if hvac_fan_direction is not None:
            climate.hvac_fan_direction = hvac_fan_direction
        climate.hvac_power_on = True
        return climate

    @is_tool(ToolType.WRITE)
    def set_seat_heater(self, seat: HeaterSeat, level: int) -> ClimateState:
        """
        Set a seat heater level.

        Args:
            seat: driver or passenger.
            level: Heater level (0 = off, 3 = max).

        Returns:
            The updated climate state.

        Raises:
            ValueError: If the level is out of range.
        """
        if not 0 <= level <= 3:
            raise ValueError("Seat heater level must be between 0 and 3")
        self.db.climate.hvac_seat_temperature[seat] = int(level)
        return self.db.climate

    @is_tool(ToolType.WRITE)
    def set_navigation_destination(
        self, destination: str, waypoint_poi_id: Optional[str] = None
    ) -> NavigationState:
        """
        Start navigation to a destination, optionally via one waypoint.

        Args:
            destination: "home", "work", a poi_id from search_poi, a contact_id from
                find_contact, or a full street address.
            waypoint_poi_id: Optional poi_id to stop at on the way.

        Returns:
            The updated navigation state.

        Raises:
            ValueError: If the destination or waypoint cannot be resolved.
        """
        location = self._resolve_destination(destination)
        waypoints: list[str] = []
        if waypoint_poi_id is not None:
            if waypoint_poi_id not in self.db.pois:
                raise ValueError(f"Unknown waypoint poi_id: {waypoint_poi_id}")
            waypoints = [waypoint_poi_id]
        self.db.navigation.destination = location
        self.db.navigation.waypoints = waypoints
        return self.db.navigation

    @is_tool(ToolType.WRITE)
    def cancel_navigation(self) -> NavigationState:
        """
        Cancel the active navigation.

        Returns:
            The updated navigation state.
        """
        self.db.navigation.destination = None
        self.db.navigation.waypoints = []
        return self.db.navigation

    @is_tool(ToolType.WRITE)
    def play_media(self, media_id: str) -> MediaState:
        """
        Play a media item from the library.

        Args:
            media_id: The media_id (see search_media or get_user_profile).

        Returns:
            The updated media state.

        Raises:
            ValueError: If the media_id does not exist.
        """
        if media_id not in self.db.user_profile.media_library:
            raise ValueError(f"Unknown media_id: {media_id}")
        self.db.media.now_playing_id = media_id
        self.db.media.playback_state = "playing"
        return self.db.media

    @is_tool(ToolType.WRITE)
    def stop_media(self) -> MediaState:
        """
        Stop media playback.

        Returns:
            The updated media state.
        """
        self.db.media.now_playing_id = None
        self.db.media.playback_state = "stopped"
        return self.db.media

    @is_tool(ToolType.WRITE)
    def set_volume(self, level: int) -> MediaState:
        """
        Set the media volume.

        Args:
            level: Volume (0-30).

        Returns:
            The updated media state.

        Raises:
            ValueError: If the level is out of range.
        """
        if not 0 <= level <= 30:
            raise ValueError("Volume must be between 0 and 30")
        self.db.media.volume = int(level)
        return self.db.media

    @is_tool(ToolType.WRITE)
    def call_contact(self, contact_id: str) -> PhoneState:
        """
        Start a hands-free call to a contact.

        Args:
            contact_id: The contact_id (see find_contact).

        Returns:
            The updated phone state.

        Raises:
            ValueError: If the contact does not exist.
        """
        if contact_id not in self.db.user_profile.contacts:
            raise ValueError(f"Unknown contact_id: {contact_id}")
        self.db.phone.active_call_contact_id = contact_id
        return self.db.phone

    @is_tool(ToolType.WRITE)
    def send_message(self, contact_id: str, text: str) -> str:
        """
        Send a text message to a contact. The message text is not stored.

        Args:
            contact_id: The contact_id (see find_contact).
            text: The message text.

        Returns:
            A confirmation string.

        Raises:
            ValueError: If the contact does not exist.
        """
        if contact_id not in self.db.user_profile.contacts:
            raise ValueError(f"Unknown contact_id: {contact_id}")
        self.db.phone.sent_message_recipients.append(contact_id)
        name = self.db.user_profile.contacts[contact_id].name
        return f"Message sent to {name}."

    @is_tool(ToolType.WRITE)
    def add_calendar_event(
        self,
        title: str,
        start: str,
        end: Optional[str] = None,
        location: Optional[str] = None,
    ) -> CalendarEvent:
        """
        Add a calendar event.

        Args:
            title: Event title.
            start: Start datetime, YYYY-MM-DDTHH:MM.
            end: End datetime, YYYY-MM-DDTHH:MM. Defaults to one hour after start.
            location: Optional location.

        Returns:
            The created event.

        Raises:
            ValueError: If the datetime format is invalid.
        """
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%dT%H:%M")
        except ValueError as exc:
            raise ValueError("start must be in YYYY-MM-DDTHH:MM format") from exc
        if end is None:
            end = (start_dt + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")
        else:
            try:
                datetime.strptime(end, "%Y-%m-%dT%H:%M")
            except ValueError as exc:
                raise ValueError("end must be in YYYY-MM-DDTHH:MM format") from exc
        event_id = f"evt_{len(self.db.calendar) + 1}"
        event = CalendarEvent(
            event_id=event_id,
            title=title.strip(),
            start=start,
            end=end,
            location=location,
        )
        self.db.calendar[event_id] = event
        return event

    @is_tool(ToolType.WRITE)
    def set_drive_mode(self, mode: DriveMode) -> VehicleState:
        """
        Set the drive mode.

        Args:
            mode: eco, normal or sport.

        Returns:
            The updated vehicle state.
        """
        self.db.vehicle.drive_mode = mode
        return self.db.vehicle

    @is_tool(ToolType.WRITE)
    def set_window(self, position: WindowPosition, open_pct: int) -> VehicleState:
        """
        Open or close one window. The driver's window is front_left.

        Args:
            position: front_left, front_right, rear_left or rear_right.
            open_pct: Percent open (0 = closed, 100 = fully open).

        Returns:
            The updated vehicle state.

        Raises:
            ValueError: If the percentage is out of range.
        """
        if not 0 <= open_pct <= 100:
            raise ValueError("open_pct must be between 0 and 100")
        setattr(self.db.vehicle.window_pos_pct, position, int(open_pct))
        return self.db.vehicle

    @is_tool(ToolType.WRITE)
    def set_door_lock(self, locked: bool) -> VehicleState:
        """
        Lock or unlock all doors.

        Args:
            locked: True to lock, False to unlock.

        Returns:
            The updated vehicle state.
        """
        self.db.vehicle.door_lock = bool(locked)
        return self.db.vehicle

    @is_tool(ToolType.WRITE)
    def set_child_lock(self, enabled: bool) -> VehicleState:
        """
        Enable or disable the rear child lock (rear doors and windows).

        Args:
            enabled: True to enable, False to disable.

        Returns:
            The updated vehicle state.
        """
        self.db.vehicle.door_child_lock_enabled = bool(enabled)
        return self.db.vehicle

    # -------------------------------------------------------- init only (no tool)
    def set_vehicle_state(
        self,
        vehicle: Optional[dict] = None,
        climate: Optional[dict] = None,
        media: Optional[dict] = None,
        navigation: Optional[dict] = None,
        phone: Optional[dict] = None,
    ) -> str:
        """Deep-merge partial updates into the live state. Used by task
        initialization only; not exposed to the agent."""
        updates = {
            "vehicle": (vehicle, VehicleState),
            "climate": (climate, ClimateState),
            "media": (media, MediaState),
            "navigation": (navigation, NavigationState),
            "phone": (phone, PhoneState),
        }
        for section, (upd, model_cls) in updates.items():
            if not upd:
                continue
            data = getattr(self.db, section).model_dump()
            setattr(self.db, section, model_cls.model_validate(_deep_update(data, upd)))
        return "ok"

    # ----------------------------------------------------------------- helpers
    def _resolve_destination(self, destination: str) -> Location:
        key = " ".join(destination.strip().split())
        low = key.lower()
        profile = self.db.user_profile
        if low in {"home", "my home", "my house", "house"}:
            return Location(label="Home", address=profile.home_address)
        if low in {"work", "my work", "office", "my office", "the office"}:
            return Location(label="Work", address=profile.work_address)
        if key in self.db.pois:
            poi = self.db.pois[key]
            return Location(label=poi.name, address=poi.address)
        for poi in self.db.pois.values():
            if poi.name.lower() == low:
                return Location(label=poi.name, address=poi.address)
        contact = profile.contacts.get(key)
        if contact is None:
            for c in profile.contacts.values():
                if low in {c.name.lower(), f"{c.name.lower()}'s", f"{c.name.lower()}'s place",
                           f"{c.name.lower()}'s home", f"{c.name.lower()}'s house"}:
                    contact = c
                    break
        if contact is not None:
            if contact.address is None:
                raise ValueError(f"No address is known for {contact.name}")
            return Location(label=f"{contact.name}'s place", address=contact.address)
        if low == profile.home_address.lower():
            return Location(label="Home", address=profile.home_address)
        if low == profile.work_address.lower():
            return Location(label="Work", address=profile.work_address)
        for poi in self.db.pois.values():
            if poi.address.lower() == low:
                return Location(label=poi.name, address=poi.address)
        for c in profile.contacts.values():
            if c.address is not None and c.address.lower() == low:
                return Location(label=f"{c.name}'s place", address=c.address)
        return Location(label=key, address=key)

    # ------------------------------------------------ assertions (no tool)
    def assert_calendar_event(
        self, title_contains: str, start: str, end: str
    ) -> bool:
        """True if an event whose title contains the text (case-insensitive)
        exists with exactly this start and end."""
        needle = title_contains.lower()
        return any(
            needle in e.title.lower() and e.start == start and e.end == end
            for e in self.db.calendar.values()
        )

    def assert_num_calendar_events(self, expected_number: int) -> bool:
        """True if the calendar holds exactly `expected_number` events."""
        return len(self.db.calendar) == expected_number
