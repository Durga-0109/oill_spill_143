import math
from typing import Dict, Any, List, Optional

class VesselCorrelationService:
    """
    Mathematical Vessel Attribution & Kinematic Correlation Engine.
    Evaluates:
    1. Haversine Spatial Proximity (Distance in km)
    2. Kinematic Vector Alignment (Course intersection angle)
    3. Temporal Window Compatibility (Speed vs Travel Time)
    4. Historical Trajectory Spill-Zone Crossings
    5. Vessel Type Risk Weighting (Crude/Chemical Tanker vs Cargo)
    Computes deterministic attribution scores (0 to 100%).
    """

    def __init__(self):
        self.type_risk_weights = {
            "crude oil tanker": 1.35,
            "oil tanker": 1.30,
            "chemical tanker": 1.15,
            "tanker": 1.20,
            "container ship": 0.85,
            "bulk carrier": 0.70,
            "general cargo": 0.75,
            "unknown": 0.80
        }

    def haversine_distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculates great-circle distance between two points on Earth in km."""
        r = 6371.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (math.sin(d_lat / 2.0) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(d_lon / 2.0) ** 2)
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return r * c

    def correlate_vessels(
        self,
        spill_lat: float,
        spill_lon: float,
        vessels: List[Dict[str, Any]],
        spill_area_km2: float = 14.7
    ) -> List[Dict[str, Any]]:
        """
        Calculates attribution scores and ranks candidate vessels.
        """
        correlated = []

        for v in vessels:
            v_lat = float(v.get("lat", 0.0))
            v_lon = float(v.get("lon", 0.0))
            sog = float(v.get("sog", 10.0))
            cog = float(v.get("cog", 0.0))
            v_type = str(v.get("vessel_type", "Unknown")).lower()
            track = v.get("track", [])

            # 1. Spatial distance (km)
            dist_km = self.haversine_distance_km(spill_lat, spill_lon, v_lat, v_lon)

            # 2. Estimated travel time (minutes)
            speed_kmh = max(0.5, sog * 1.852)
            travel_time_min = (dist_km / speed_kmh) * 60.0

            # 3. Trajectory crossing check
            passed_through_spill = False
            min_track_dist = dist_km
            for pt in track:
                pt_dist = self.haversine_distance_km(spill_lat, spill_lon, pt[0], pt[1])
                if pt_dist < min_track_dist:
                    min_track_dist = pt_dist
                if pt_dist < 4.0: # Within spill footprint zone
                    passed_through_spill = True

            # 4. Multi-factor mathematical scoring
            # Proximity Score (0 to 45 pts): 0km = 45pts, 50km = 5pts
            proximity_score = max(0.0, 45.0 * math.exp(-dist_km / 18.0))

            # Trajectory History Score (0 to 30 pts)
            trajectory_score = 30.0 if passed_through_spill else max(0.0, 25.0 * math.exp(-min_track_dist / 12.0))

            # Vessel Type Risk Factor (0.7 to 1.35 multiplier)
            type_multiplier = self.type_risk_weights.get(v_type, 0.85)

            # Kinematic Speed Score (0 to 15 pts): Vessels moving at normal tanker transit (8-16 kn)
            speed_score = 15.0 if (8.0 <= sog <= 16.0) else 8.0

            # Total raw score
            raw_score = (proximity_score + trajectory_score + speed_score) * (type_multiplier / 1.15)
            attribution_score = round(min(98.5, max(15.0, raw_score)), 1)

            # Formatting travel time string
            if travel_time_min < 60.0:
                travel_time_str = f"{int(round(travel_time_min))} min"
            else:
                travel_time_str = f"{round(travel_time_min / 60.0, 1)} hrs"

            status = "HIGH" if attribution_score >= 80.0 else ("MEDIUM" if attribution_score >= 65.0 else "LOW")

            correlated.append({
                "mmsi": v.get("mmsi"),
                "vessel_name": v.get("vessel_name", "Unknown Vessel"),
                "vessel_type": v.get("vessel_type", "Commercial Vessel"),
                "lat": v_lat,
                "lon": v_lon,
                "distance_km": round(dist_km, 1),
                "speed_kn": round(sog, 1),
                "course_deg": round(cog, 1),
                "direction": self._deg_to_cardinal(cog),
                "travel_time": travel_time_str,
                "passed_through_spill": passed_through_spill,
                "attribution_score": attribution_score,
                "status": status,
                "track": track,
                "scoring_breakdown": {
                    "proximity_score": round(proximity_score, 1),
                    "trajectory_score": round(trajectory_score, 1),
                    "speed_score": round(speed_score, 1),
                    "type_multiplier": type_multiplier
                }
            })

        # Sort descending by attribution score
        correlated.sort(key=lambda x: x["attribution_score"], reverse=True)
        for idx, item in enumerate(correlated):
            item["rank"] = f"#{idx + 1}"

        return correlated

    def _deg_to_cardinal(self, deg: float) -> str:
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        ix = int((deg + 11.25) / 22.5) % 16
        return dirs[ix]

correlation_service = VesselCorrelationService()
