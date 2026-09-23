import math
import datetime
from typing import Dict, Any, List, Optional

class DriftPredictionService:
    """
    Physics-based Lagrangian Oil Spill Trajectory & Drift Forecasting Engine.
    Implements standard hydrodynamic slick advection:
        V_drift_vector = V_current_vector + alpha_windage * V_wind_vector
        Where alpha_windage = 0.030 (standard empirical wind drift factor for marine oil slicks).
    Computes 0-Hour (Current), 1-Hour, and 2-Hour spatial coordinates, expansion area, and trajectory path.
    """

    def __init__(self, windage_factor: float = 0.030, spreading_rate_per_hour: float = 0.12):
        self.windage_factor = windage_factor
        self.spreading_rate_per_hour = spreading_rate_per_hour

    def calculate_drift(
        self,
        lat: float,
        lon: float,
        current_speed_kn: float,
        current_dir_deg: float,
        wind_speed_kn: float,
        wind_dir_deg: float,
        initial_area_km2: float = 14.7,
        hours: int = 2
    ) -> Dict[str, Any]:
        """
        Calculates forward drift trajectory for 0h, 1h, and 2h horizons.
        """
        # 1. Convert directions to nautical math convention (Cartesian vector coordinates)
        # 0° = North (+Y), 90° = East (+X), 180° = South (-Y), 270° = West (-X)
        curr_rad = math.radians(current_dir_deg)
        wind_rad = math.radians(wind_dir_deg)

        # Ocean current velocity vector components (knots)
        curr_vx = current_speed_kn * math.sin(curr_rad)
        curr_vy = current_speed_kn * math.cos(curr_rad)

        # Wind velocity vector components (knots)
        wind_vx = wind_speed_kn * math.sin(wind_rad)
        wind_vy = wind_speed_kn * math.cos(wind_rad)

        # Combined Lagrangian surface drift velocity vector
        drift_vx = curr_vx + (self.windage_factor * wind_vx)
        drift_vy = curr_vy + (self.windage_factor * wind_vy)

        # Resultant drift speed (knots) and direction (degrees)
        resultant_speed_kn = math.sqrt(drift_vx**2 + drift_vy**2)
        resultant_dir_deg = (math.degrees(math.atan2(drift_vx, drift_vy)) + 360) % 360

        # Convert knots to km/h (1 knot = 1.852 km/h)
        speed_kmh = resultant_speed_kn * 1.852

        # 1 degree of latitude approx 111.139 km
        # 1 degree of longitude approx 111.139 * cos(lat) km
        km_per_lat = 111.139
        km_per_lon = 111.139 * math.cos(math.radians(lat))

        # Hourly step points
        steps = []
        for h in range(hours + 1):
            dist_km = speed_kmh * h
            delta_lat = (drift_vy * 1.852 * h) / km_per_lat
            delta_lon = (drift_vx * 1.852 * h) / km_per_lon

            pt_lat = round(lat + delta_lat, 6)
            pt_lon = round(lon + delta_lon, 6)

            # Fay's surface spreading model expansion (+12% area per hour)
            area_h = round(initial_area_km2 * (1.0 + (self.spreading_rate_per_hour * h)), 2)

            steps.append({
                "hour": h,
                "label": "Current Position" if h == 0 else f"+{h} Hour{'s' if h > 1 else ''} Prediction",
                "latitude": pt_lat,
                "longitude": pt_lon,
                "drift_distance_km": round(dist_km, 2),
                "slick_area_km2": area_h,
                "timestamp_offset_hours": h
            })

        # Generate 2-Hour predicted polygon (projected from 2h center)
        p2_lat = steps[-1]["latitude"]
        p2_lon = steps[-1]["longitude"]
        p2_poly = [
            [round(p2_lon - 0.045, 6), round(p2_lat - 0.025, 6)],
            [round(p2_lon + 0.035, 6), round(p2_lat - 0.035, 6)],
            [round(p2_lon + 0.055, 6), round(p2_lat + 0.025, 6)],
            [round(p2_lon - 0.025, 6), round(p2_lat + 0.045, 6)],
            [round(p2_lon - 0.045, 6), round(p2_lat - 0.025, 6)]
        ]

        cardinal = self._deg_to_cardinal(resultant_dir_deg)

        return {
            "initial_position": {"latitude": lat, "longitude": lon},
            "drift_velocity_kn": round(resultant_speed_kn, 2),
            "drift_direction_deg": round(resultant_dir_deg, 1),
            "drift_cardinal": cardinal,
            "total_2h_distance_km": round(speed_kmh * 2.0, 2),
            "predicted_2h_area_km2": steps[-1]["slick_area_km2"],
            "area_expansion_pct": round(((steps[-1]["slick_area_km2"] - initial_area_km2) / initial_area_km2) * 100, 1) if initial_area_km2 > 0 else 0.0,
            "trajectory_steps": steps,
            "predicted_2h_polygon": p2_poly,
            "weather_input": {
                "wind_speed_kn": wind_speed_kn,
                "wind_direction_deg": wind_dir_deg,
                "current_speed_kn": current_speed_kn,
                "current_direction_deg": current_dir_deg
            },
            "formula": "Lagrangian Advection: V_drift = V_current + 0.030 * V_wind"
        }

    def _deg_to_cardinal(self, deg: float) -> str:
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        ix = int((deg + 11.25) / 22.5) % 16
        return dirs[ix]

drift_service = DriftPredictionService()
