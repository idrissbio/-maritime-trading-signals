import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
import logging
from .data_fetcher import VesselPosition, PortCongestion

logger = logging.getLogger(__name__)

@dataclass
class MaritimeEvent:
    event_type: str
    severity: float  # 0-1 scale
    affected_commodity: str
    estimated_price_impact: float  # percentage
    confidence_level: float  # 0-1 scale
    location: str
    timestamp: datetime
    details: Dict[str, Any]

@dataclass
class CongestionEvent(MaritimeEvent):
    waiting_vessels: int
    average_wait_time: float
    congestion_multiplier: float

@dataclass
class ClusterEvent(MaritimeEvent):
    cluster_size: int
    cluster_center: Tuple[float, float]
    vessels_in_cluster: List[str]

@dataclass
class RouteDisruptionEvent(MaritimeEvent):
    vessel_id: str
    planned_route: List[Tuple[float, float]]
    actual_route: List[Tuple[float, float]]
    deviation_distance: float

@dataclass
class SupplyPressureEvent(MaritimeEvent):
    loading_imbalance: float
    discharge_imbalance: float
    net_flow_change: float

class MaritimeAnalyzer:
    def __init__(self):
        # Historical baselines for comparison
        self.port_baselines = {
            "singapore": {"avg_wait": 8.5, "avg_vessels": 25},
            "houston": {"avg_wait": 12.0, "avg_vessels": 35},
            "rotterdam": {"avg_wait": 6.5, "avg_vessels": 20},
            "fujairah": {"avg_wait": 10.0, "avg_vessels": 30}
        }
        
        # Commodity mappings
        self.port_commodities = {
            "singapore": "crude_oil",
            "houston": "crude_oil", 
            "rotterdam": "crude_oil",
            "fujairah": "crude_oil",
            "freeport": "lng",
            "sabine_pass": "lng"
        }
        
        logger.info("MaritimeAnalyzer initialized")

    def analyze_port_congestion(self, port_data: PortCongestion) -> Optional[CongestionEvent]:
        """Analyze port congestion and return event if significant"""
        port_name = port_data.port_name.lower()
        
        # Get baseline data
        baseline = self.port_baselines.get(port_name, {"avg_wait": 10.0, "avg_vessels": 25})
        
        # Calculate congestion multipliers
        wait_multiplier = port_data.average_wait_time / baseline["avg_wait"]
        vessel_multiplier = port_data.waiting_vessels / baseline["avg_vessels"]
        
        # Overall congestion score
        congestion_multiplier = (wait_multiplier + vessel_multiplier) / 2
        
        # Threshold for significant congestion
        if congestion_multiplier < 1.5:
            return None
        
        # Determine severity (0-1 scale)
        severity = min(1.0, (congestion_multiplier - 1.0) / 2.0)
        
        # Estimate price impact
        base_impact = 0.005  # 0.5% base impact
        price_impact = base_impact * congestion_multiplier * severity
        
        # Confidence based on data quality and historical patterns
        confidence = min(0.95, 0.6 + (severity * 0.35))
        
        # Determine affected commodity
        commodity = self.port_commodities.get(port_name, "crude_oil")
        
        event = CongestionEvent(
            event_type="port_congestion",
            severity=severity,
            affected_commodity=commodity,
            estimated_price_impact=price_impact,
            confidence_level=confidence,
            location=port_data.port_name,
            timestamp=port_data.timestamp,
            details={
                "baseline_wait": baseline["avg_wait"],
                "baseline_vessels": baseline["avg_vessels"],
                "current_wait": port_data.average_wait_time,
                "current_vessels": port_data.waiting_vessels,
                "cargo_types": port_data.cargo_types
            },
            waiting_vessels=port_data.waiting_vessels,
            average_wait_time=port_data.average_wait_time,
            congestion_multiplier=congestion_multiplier
        )
        
        logger.info(f"Port congestion detected: {port_name} - Severity: {severity:.2f}")
        return event

    def detect_vessel_clusters(self, vessel_positions: List[VesselPosition]) -> List[ClusterEvent]:
        """Detect vessel clustering using DBSCAN algorithm"""
        if len(vessel_positions) < 5:
            return []
        
        # Prepare data for clustering
        coordinates = np.array([[v.lat, v.lon] for v in vessel_positions])
        
        # DBSCAN clustering - eps in degrees (roughly 10 nautical miles)
        eps = 0.15  # degrees
        min_samples = 5
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coordinates)
        labels = clustering.labels_
        
        cluster_events = []
        
        # Analyze each cluster
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            # Get vessels in this cluster
            cluster_mask = labels == label
            cluster_vessels = [vessel_positions[i] for i in range(len(vessel_positions)) if cluster_mask[i]]
            
            if len(cluster_vessels) < 8:  # Only significant clusters
                continue
            
            # Calculate cluster center
            cluster_lats = [v.lat for v in cluster_vessels]
            cluster_lons = [v.lon for v in cluster_vessels]
            center_lat = np.mean(cluster_lats)
            center_lon = np.mean(cluster_lons)
            
            # Analyze cluster composition
            cargo_types = [v.cargo_type for v in cluster_vessels]
            cargo_counts = pd.Series(cargo_types).value_counts()
            dominant_cargo = cargo_counts.index[0]
            
            # Calculate severity based on cluster size and cargo type
            base_severity = min(1.0, len(cluster_vessels) / 50.0)
            
            # Higher severity for oil/gas clusters
            if dominant_cargo in ["crude_oil", "lng"]:
                severity = min(1.0, base_severity * 1.5)
            else:
                severity = base_severity
            
            # Estimate price impact
            price_impact = 0.003 * len(cluster_vessels) * severity
            
            # Confidence based on cluster density and composition
            cargo_dominance = cargo_counts.iloc[0] / len(cluster_vessels)
            confidence = min(0.9, 0.5 + (cargo_dominance * 0.4))
            
            # Determine location context
            location = self._get_location_context(center_lat, center_lon)
            
            event = ClusterEvent(
                event_type="vessel_clustering",
                severity=severity,
                affected_commodity=dominant_cargo,
                estimated_price_impact=price_impact,
                confidence_level=confidence,
                location=location,
                timestamp=datetime.now(),
                details={
                    "cargo_composition": cargo_counts.to_dict(),
                    "avg_speed": np.mean([v.speed for v in cluster_vessels]),
                    "speed_variance": np.var([v.speed for v in cluster_vessels])
                },
                cluster_size=len(cluster_vessels),
                cluster_center=(center_lat, center_lon),
                vessels_in_cluster=[v.vessel_id for v in cluster_vessels]
            )
            
            cluster_events.append(event)
            logger.info(f"Vessel cluster detected: {len(cluster_vessels)} vessels at {location}")
        
        return cluster_events

    def check_route_disruptions(self, vessel_routes: Dict[str, List[Dict]]) -> List[RouteDisruptionEvent]:
        """Analyze vessel routes for significant disruptions"""
        disruption_events = []
        
        for vessel_id, route_points in vessel_routes.items():
            if len(route_points) < 3:
                continue
            
            # Calculate route deviations
            deviations = []
            for point in route_points:
                deviation = point.get("deviation_from_planned", 0)
                deviations.append(deviation)
            
            max_deviation = max(deviations)
            avg_deviation = np.mean(deviations)
            
            # Threshold for significant disruption (100+ nautical miles)
            if max_deviation < 100:
                continue
            
            # Calculate severity
            severity = min(1.0, max_deviation / 500.0)  # Max at 500 nm
            
            # Estimate impact based on cargo type and route importance
            # This would need more sophisticated route analysis
            price_impact = 0.002 * severity
            
            # Confidence based on data quality
            confidence = 0.7 if len(route_points) > 5 else 0.5
            
            # Get vessel cargo type (would need to join with vessel data)
            commodity = "crude_oil"  # Default assumption
            
            event = RouteDisruptionEvent(
                event_type="route_disruption",
                severity=severity,
                affected_commodity=commodity,
                estimated_price_impact=price_impact,
                confidence_level=confidence,
                location="Maritime Route",
                timestamp=datetime.now(),
                details={
                    "max_deviation_nm": max_deviation,
                    "avg_deviation_nm": avg_deviation,
                    "route_points": len(route_points)
                },
                vessel_id=vessel_id,
                planned_route=[],  # Would need planned route data
                actual_route=[(p.get("lat", 0), p.get("lon", 0)) for p in route_points],
                deviation_distance=max_deviation
            )
            
            disruption_events.append(event)
            logger.info(f"Route disruption detected: Vessel {vessel_id}, Max deviation: {max_deviation:.1f} nm")
        
        return disruption_events

    def calculate_supply_pressure(self, loading_data: Dict[str, Any]) -> Optional[SupplyPressureEvent]:
        """Calculate supply pressure based on loading/discharge data"""
        
        # Mock implementation - would need real loading/discharge terminal data
        loading_terminals = loading_data.get("loading_terminals", {})
        discharge_terminals = loading_data.get("discharge_terminals", {})
        
        if not loading_terminals and not discharge_terminals:
            # Generate mock data for demonstration
            loading_terminals = {
                "singapore": {"capacity": 100, "current_load": 85, "vessels_waiting": 12},
                "houston": {"capacity": 150, "current_load": 120, "vessels_waiting": 8},
                "rotterdam": {"capacity": 80, "current_load": 65, "vessels_waiting": 15}
            }
            
            discharge_terminals = {
                "singapore": {"capacity": 90, "current_load": 75, "vessels_waiting": 10},
                "houston": {"capacity": 140, "current_load": 100, "vessels_waiting": 6},
                "rotterdam": {"capacity": 85, "current_load": 70, "vessels_waiting": 12}
            }
        
        # Calculate loading vs discharge imbalances
        total_loading_pressure = 0
        total_discharge_pressure = 0
        
        for terminal, data in loading_terminals.items():
            utilization = data["current_load"] / data["capacity"]
            wait_pressure = data["vessels_waiting"] / 10.0  # Normalize
            loading_pressure = (utilization + wait_pressure) / 2
            total_loading_pressure += loading_pressure
        
        for terminal, data in discharge_terminals.items():
            utilization = data["current_load"] / data["capacity"]
            wait_pressure = data["vessels_waiting"] / 10.0
            discharge_pressure = (utilization + wait_pressure) / 2
            total_discharge_pressure += discharge_pressure
        
        loading_imbalance = total_loading_pressure / len(loading_terminals)
        discharge_imbalance = total_discharge_pressure / len(discharge_terminals)
        
        net_flow_change = loading_imbalance - discharge_imbalance
        
        # Only create event if significant imbalance
        if abs(net_flow_change) < 0.3:
            return None
        
        severity = min(1.0, abs(net_flow_change))
        price_impact = 0.01 * severity * (1 if net_flow_change > 0 else -1)
        
        event = SupplyPressureEvent(
            event_type="supply_pressure",
            severity=severity,
            affected_commodity="crude_oil",
            estimated_price_impact=price_impact,
            confidence_level=0.6,
            location="Global",
            timestamp=datetime.now(),
            details={
                "loading_terminals": loading_terminals,
                "discharge_terminals": discharge_terminals,
                "flow_direction": "supply_increase" if net_flow_change > 0 else "supply_decrease"
            },
            loading_imbalance=loading_imbalance,
            discharge_imbalance=discharge_imbalance,
            net_flow_change=net_flow_change
        )
        
        logger.info(f"Supply pressure detected: Net flow change: {net_flow_change:.2f}")
        return event

    def analyze_speed_anomalies(self, vessel_positions: List[VesselPosition]) -> List[MaritimeEvent]:
        """Detect vessels with unusual speed patterns"""
        speed_events = []
        
        # Group by cargo type for baseline comparison
        cargo_speeds = {}
        for vessel in vessel_positions:
            if vessel.cargo_type not in cargo_speeds:
                cargo_speeds[vessel.cargo_type] = []
            cargo_speeds[vessel.cargo_type].append(vessel.speed)
        
        # Analyze each vessel against its cargo type baseline
        for vessel in vessel_positions:
            cargo_type_speeds = cargo_speeds[vessel.cargo_type]
            avg_speed = np.mean(cargo_type_speeds)
            std_speed = np.std(cargo_type_speeds)
            
            # Check for significant speed anomaly
            if std_speed == 0:  # Avoid division by zero
                continue
            
            z_score = abs(vessel.speed - avg_speed) / std_speed
            
            if z_score > 2.5:  # Significant anomaly
                # Determine if slow or fast
                anomaly_type = "slow_steaming" if vessel.speed < avg_speed else "high_speed"
                
                severity = min(1.0, z_score / 5.0)
                
                # Slow steaming often indicates demand issues
                price_impact = -0.002 * severity if anomaly_type == "slow_steaming" else 0.001 * severity
                
                event = MaritimeEvent(
                    event_type=f"speed_anomaly_{anomaly_type}",
                    severity=severity,
                    affected_commodity=vessel.cargo_type,
                    estimated_price_impact=price_impact,
                    confidence_level=0.5,
                    location=f"Vessel {vessel.vessel_id}",
                    timestamp=vessel.timestamp,
                    details={
                        "vessel_speed": vessel.speed,
                        "average_speed": avg_speed,
                        "z_score": z_score,
                        "position": (vessel.lat, vessel.lon)
                    }
                )
                
                speed_events.append(event)
        
        if speed_events:
            logger.info(f"Speed anomalies detected: {len(speed_events)} vessels")
        
        return speed_events

    def _get_location_context(self, lat: float, lon: float) -> str:
        """Determine geographic context for coordinates"""
        # Major shipping regions
        regions = [
            {"name": "Singapore Strait", "lat_range": (0.5, 2.0), "lon_range": (103.0, 105.0)},
            {"name": "Gulf of Mexico", "lat_range": (28.0, 31.0), "lon_range": (-98.0, -93.0)},
            {"name": "North Sea", "lat_range": (51.0, 53.0), "lon_range": (2.0, 6.0)},
            {"name": "Persian Gulf", "lat_range": (24.0, 27.0), "lon_range": (50.0, 56.0)},
            {"name": "South China Sea", "lat_range": (5.0, 25.0), "lon_range": (105.0, 120.0)}
        ]
        
        for region in regions:
            if (region["lat_range"][0] <= lat <= region["lat_range"][1] and
                region["lon_range"][0] <= lon <= region["lon_range"][1]):
                return region["name"]
        
        return f"Maritime ({lat:.2f}, {lon:.2f})"

    def analyze_all_maritime_factors(self, 
                                   port_congestion_data: List[PortCongestion],
                                   vessel_positions: List[VesselPosition],
                                   vessel_routes: Dict[str, List[Dict]],
                                   loading_data: Dict[str, Any] = None) -> List[MaritimeEvent]:
        """Comprehensive analysis of all maritime factors"""
        
        all_events = []
        
        # Analyze port congestion
        for port_data in port_congestion_data:
            congestion_event = self.analyze_port_congestion(port_data)
            if congestion_event:
                all_events.append(congestion_event)
        
        # Detect vessel clusters
        cluster_events = self.detect_vessel_clusters(vessel_positions)
        all_events.extend(cluster_events)
        
        # Check route disruptions
        disruption_events = self.check_route_disruptions(vessel_routes)
        all_events.extend(disruption_events)
        
        # Analyze supply pressure
        if loading_data:
            supply_event = self.calculate_supply_pressure(loading_data)
            if supply_event:
                all_events.append(supply_event)
        
        # Check speed anomalies
        speed_events = self.analyze_speed_anomalies(vessel_positions)
        all_events.extend(speed_events)
        
        # Sort by severity and confidence
        all_events.sort(key=lambda x: x.severity * x.confidence_level, reverse=True)
        
        logger.info(f"Maritime analysis complete: {len(all_events)} events detected")
        return all_events