# -*- coding: utf-8 -*-
"""
core/models.py

Main functionality:
This module defines the core data models for scenario generation, including
orbital parameters, mission distribution settings, satellite capabilities,
window data structures, and the main entity classes such as Satellite,
Mission, and GroundStation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


# =============================================================================
# 1) Enumerations
# =============================================================================


class OrbitalType(Enum):
    """Orbital type."""

    LEO = "LEO"
    MEO = "MEO"
    GEO = "GEO"


class ManeuverabilityType(Enum):
    """Maneuverability level."""

    NON_AGILE = "non_agile"
    AGILE = "agile"
    SUPER_AGILE = "super_agile"


class SensorResolution(Enum):
    """Sensor resolution."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class SensorMode(Enum):
    """Sensor mode."""
    VISIBLE = "visible"
    INFRARED = "infrared"
    SAR = "sar"
    OTHER = "other"


class AntennaFrequency(Enum):
    """Antenna frequency."""
    S_BAND = "s_band"
    X_BAND = "x_band"
    KU_BAND = "ku_band"
    KA_BAND = "ka_band"


class AntennaType(Enum):
    """Antenna type."""
    OMNIDIRECTIONAL = "omnidirectional"
    DIRECTIONAL = "directional"


# =============================================================================
# 2) Orbit / Location / Mission Distribution
# =============================================================================


@dataclass
class MissionInfo:
    """Distribution parameters for random mission point generation."""

    distribution_type: int
    latitude_range: List[List[float]]
    longitude_range: List[List[float]]


@dataclass
class OrbitalParameters:
    """Six orbital elements plus epoch string."""

    semi_major_axis: float  # km
    eccentricity: float
    inclination: float  # deg
    argument_of_perigee: float  # deg
    right_ascension_of_ascending_node: float  # deg
    mean_anomaly: float  # deg
    initial_representation_Epoch: str

    def validate_with_orbital_type(self, orbital_type: OrbitalType) -> None:
        """Perform a rough range check based on the orbital type."""
        a = float(self.semi_major_axis)
        if orbital_type == OrbitalType.LEO and not (6378 <= a <= 9000):
            raise ValueError(f"LEO semi_major_axis {a} km is not in the valid range [6378, 9000]")
        if orbital_type == OrbitalType.MEO and not (9000 < a <= 30000):
            raise ValueError(f"MEO semi_major_axis {a} km is not in the valid range (9000, 30000]")
        if orbital_type == OrbitalType.GEO and not (40000 <= a <= 45000):
            raise ValueError(f"GEO semi_major_axis {a} km is not in the valid range [40000, 45000]")


@dataclass
class GroundStationLocation:
    """Ground station location."""
    latitude: float
    longitude: float
    altitude: float = 0.0  # km


@dataclass
class TargetLocation:
    """Target location."""
    latitude: float
    longitude: float
    altitude: float = 0.0  # km


# =============================================================================
# 3) Capabilities and Requirements
# =============================================================================


@dataclass
class ObservationRequirement:
    """Observation requirement. During benchmark generation, only duration is typically used."""

    duration: float  # s
    required_resolution: Optional[SensorResolution] = None
    required_mode: Optional[SensorMode] = None
    min_elevation_angle: Optional[float] = None  # deg


@dataclass
class ObservationSensor:
    """Observation sensor properties."""
    sensor_id: str
    resolution: SensorResolution
    sensor_mode: SensorMode
    field_of_view: float  # deg
    observation_swath_width: Optional[float] = None
    data_rate: Optional[float] = None
    power_consumption: Optional[float] = None
    min_elevation_angle: Optional[float] = None  # deg

    @property
    def view_angle_deg(self) -> float:
        return float(self.field_of_view)


@dataclass
class ObservationCapability:
    """Observation capability."""
    sensors: List[ObservationSensor] = field(default_factory=list)

    @property
    def has_observation(self) -> bool:
        return bool(self.sensors)

    def get_default_sensor(self) -> Optional[ObservationSensor]:
        """Get the default sensor."""
        if not self.sensors:
            return None
        # Simple strategy: prefer visible mode
        for s in self.sensors:
            if s.sensor_mode == SensorMode.VISIBLE:
                return s
        return self.sensors[0]


@dataclass
class ManeuverabilityCapability:
    """Maneuverability capability."""
    maneuverability_type: ManeuverabilityType
    slew_rate: float  # deg/s
    max_pitch_angle: float = 45.0
    max_yaw_angle: float = 45.0
    max_roll_angle: float = 45.0
    stabilization_time: float = 0.0


@dataclass
class CommunicationAntenna:
    """Communication antenna properties."""
    antenna_id: str
    data_rate: float
    antenna_type: Optional[AntennaType] = None
    frequency: Optional[AntennaFrequency] = None
    power_consumption: Optional[float] = None
    can_concurrent: Optional[bool] = None


@dataclass
class CommunicationCapability:
    """Communication capability."""
    antennas: List[CommunicationAntenna] = field(default_factory=list)

    @property
    def has_transmission(self) -> bool:
        return bool(self.antennas)


@dataclass
class SatelliteSpecs:
    """Satellite platform specifications."""
    max_data_storage: float = 1000.0
    max_operation_time: float = 4500.0
    max_power: Optional[float] = 500.0
    max_battery_capacity: Optional[float] = 2000.0
    max_fuel: Optional[float] = 100.0


# =============================================================================
# 4) Window Data Structures (needed for JSON output)
# =============================================================================


@dataclass
class AgileData:
    """Agile satellite attitude data."""
    pitch_angles: List[float]
    yaw_angles: List[float]
    roll_angles: List[float]


@dataclass
class NonAgileData:
    """Non-agile satellite attitude data."""
    pitch_angle: float
    yaw_angle: float
    roll_angle: float


@dataclass
class TimeWindow:
    """Base time window class."""
    start_time: datetime
    end_time: datetime
    agile_data: Optional[AgileData] = None
    non_agile_data: Optional[NonAgileData] = None
    # Orbit number counted from the scenario start time as orbit 1.
    # It is filled during visibility calculation based on the Orekit orbital period.
    orbit_number: Optional[int] = None


@dataclass
class ObservationWindow:
    """Observation window."""
    satellite_id: str
    sensor_id: str
    mission_id: str
    time_window: List[TimeWindow] = field(default_factory=list)

    def get_attitude_at_time(self, target_time: datetime, time_step: float) -> Tuple[float, float, float]:
        """Return (pitch, yaw, roll) for a given time."""
        for tw in self.time_window:
            if tw.start_time <= target_time <= tw.end_time:
                if tw.agile_data and time_step > 0:
                    dt = (target_time - tw.start_time).total_seconds()
                    idx = int(round(dt / time_step))
                    idx = max(0, min(idx, len(tw.agile_data.pitch_angles) - 1))
                    return (
                        tw.agile_data.pitch_angles[idx],
                        tw.agile_data.yaw_angles[idx],
                        tw.agile_data.roll_angles[idx],
                    )
                if tw.non_agile_data:
                    na = tw.non_agile_data
                    return (na.pitch_angle, na.yaw_angle, na.roll_angle)
        return (0.0, 0.0, 0.0)


@dataclass
class CommunicationWindow:
    """Communication window."""
    satellite_id: str
    ground_station_id: str
    time_window: List[TimeWindow] = field(default_factory=list)


# =============================================================================
# 5) Main Static Models: Satellite / Mission / GroundStation
# =============================================================================


class Satellite:
    """Satellite class."""
    def __init__(
        self,
        satellite_id: str,
        orbital_type: OrbitalType,
        orbital_params: OrbitalParameters,
        satellite_specs: Optional[SatelliteSpecs] = None,
        observation_capability: Optional[ObservationCapability] = None,
        maneuverability_capability: Optional[ManeuverabilityCapability] = None,
        communication_capability: Optional[CommunicationCapability] = None,
    ) -> None:
        self.id = satellite_id
        self.orbital_type = orbital_type
        self.orbital_params = orbital_params
        self.orbital_params.validate_with_orbital_type(self.orbital_type)

        self.satellite_specs = satellite_specs or SatelliteSpecs()
        self.observation_capability = observation_capability or ObservationCapability()
        self.maneuverability_capability = maneuverability_capability or ManeuverabilityCapability(
            maneuverability_type=ManeuverabilityType.AGILE,
            slew_rate=1.0,
            max_pitch_angle=45.0,
            max_yaw_angle=45.0,
            max_roll_angle=45.0,
            stabilization_time=10.0,
        )
        self.communication_capability = communication_capability or CommunicationCapability()

    def get_default_sensor(self) -> Optional[ObservationSensor]:
        """Get the default sensor."""
        return self.observation_capability.get_default_sensor() if self.observation_capability else None

    def get_attitude_limits_deg(self, default: Tuple[float, float, float] = (45.0, 45.0, 45.0)) -> Tuple[float, float, float]:
        """Get attitude limit angles in degrees."""
        mc = self.maneuverability_capability
        if mc is None:
            return default
        return (float(mc.max_yaw_angle), float(mc.max_pitch_angle), float(mc.max_roll_angle))

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        oc = self.observation_capability
        mc = self.maneuverability_capability
        cc = self.communication_capability
        sp = self.satellite_specs

        return {
            "id": self.id,
            "orbital_type": self.orbital_type.value,
            "orbital_params": {
                "semi_major_axis_km": self.orbital_params.semi_major_axis,
                "eccentricity": self.orbital_params.eccentricity,
                "inclination_deg": self.orbital_params.inclination,
                "argument_of_perigee_deg": self.orbital_params.argument_of_perigee,
                "right_ascension_of_ascending_node_deg": self.orbital_params.right_ascension_of_ascending_node,
                "mean_anomaly_deg": self.orbital_params.mean_anomaly,
                "epoch": self.orbital_params.initial_representation_Epoch,
            },
            "satellite_specs": {
                "max_data_storage_GB": sp.max_data_storage,
                "max_operation_time_s": sp.max_operation_time,
                "max_power_W": sp.max_power,
                "max_battery_capacity_Wh": sp.max_battery_capacity,
                "max_fuel_kg": sp.max_fuel,
            },
            "observation_capability": {
                "sensors": [
                    {
                        "sensor_id": s.sensor_id,
                        "resolution": s.resolution.value,
                        "sensor_mode": s.sensor_mode.value,
                        "field_of_view_deg": s.field_of_view,
                        "observation_swath_width_km": s.observation_swath_width,
                        "data_rate_Mbps": s.data_rate,
                        "power_consumption_W": s.power_consumption,
                        "min_elevation_angle_deg": s.min_elevation_angle,
                    }
                    for s in (oc.sensors if oc else [])
                ]
            },
            "maneuverability_capability": {
                "maneuverability_type": mc.maneuverability_type.value if mc else None,
                "slew_rate_deg_per_s": mc.slew_rate if mc else None,
                "max_pitch_angle_deg": mc.max_pitch_angle if mc else None,
                "max_yaw_angle_deg": mc.max_yaw_angle if mc else None,
                "max_roll_angle_deg": mc.max_roll_angle if mc else None,
                "stabilization_time_s": mc.stabilization_time if mc else None,
            },
            "communication_capability": {
                "antennas": [
                    {
                        "antenna_id": a.antenna_id,
                        "data_rate_Mbps": a.data_rate,
                        "antenna_type": a.antenna_type.value if a.antenna_type else None,
                        "frequency": a.frequency.value if a.frequency else None,
                        "power_consumption_W": a.power_consumption,
                        "can_concurrent": a.can_concurrent,
                    }
                    for a in (cc.antennas if cc else [])
                ]
            },
        }


class GroundStation:
    """Ground station class."""
    def __init__(self, station_id: str, location: GroundStationLocation, communication_capability=None) -> None:
        self.id = station_id
        self.location = location
        self.communication_capability = communication_capability

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "location": {
                "latitude": self.location.latitude,
                "longitude": self.location.longitude,
                "altitude_km": self.location.altitude,
            },
        }


class Mission:
    """Mission class."""
    def __init__(
        self,
        mission_id: str,
        target_location: TargetLocation,
        priority: float = 1.0,
        observation_requirement: Optional[ObservationRequirement] = None,
    ) -> None:
        self.id = mission_id
        self.target_location = target_location
        self.priority = priority
        self.observation_requirement = observation_requirement

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        data: Dict = {
            "id": self.id,
            "target_location": {
                "latitude": self.target_location.latitude,
                "longitude": self.target_location.longitude,
                "altitude_km": self.target_location.altitude,
            },
            "priority": self.priority,
        }
        if self.observation_requirement is not None:
            orq = self.observation_requirement
            data["observation_requirement"] = {
                "duration_s": orq.duration,
                "required_resolution": orq.required_resolution.value if orq.required_resolution else None,
                "required_mode": orq.required_mode.value if orq.required_mode else None,
                "min_elevation_angle_deg": orq.min_elevation_angle,
            }
        return data