import requests
import time
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class VesselPosition:
    vessel_id: str
    name: str
    lat: float
    lon: float
    speed: float
    course: float
    cargo_type: str
    timestamp: datetime

@dataclass
class PortCongestion:
    port_name: str
    waiting_vessels: int
    average_wait_time: float
    cargo_types: List[str]
    congestion_score: float
    timestamp: datetime

@dataclass
class MarketData:
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float

class DataFetcher:
    def __init__(self, datalastic_key: str = None, twelve_data_key: str = None, mock_mode: bool = True):
        self.datalastic_key = datalastic_key
        self.twelve_data_key = twelve_data_key
        self.mock_mode = mock_mode or not (datalastic_key and twelve_data_key)
        
        # Rate limiting
        self.datalastic_last_request = 0
        self.twelve_data_last_request = 0
        self.datalastic_rate_limit = 0.6  # 100 req/min = 0.6 seconds
        self.twelve_data_rate_limit = 1.0  # 60 req/min = 1 second
        
        # Cache with priority-based durations
        self.cache = {}
        self.cache_duration = {
            'critical_ports': 180,    # 3 minutes for critical chokepoints
            'standard_ports': 600,    # 10 minutes for standard ports
            'vessel_positions': 300,  # 5 minutes for vessel data
            'market_data': 60        # 1 minute for market data
        }
        
        # Port scanning priority system with significance multipliers
        self.port_priorities = {
            "strait_of_hormuz": {
                "significance_multiplier": 2.5,
                "strategic_importance": "critical",
                "scan_frequency_minutes": 3,
                "notes": "20% of global oil transit"
            },
            "suez_canal": {
                "significance_multiplier": 2.0,
                "strategic_importance": "critical",
                "scan_frequency_minutes": 3,
                "notes": "76% tonnage reduction in 2024"
            },
            "panama_canal": {
                "significance_multiplier": 1.8,
                "strategic_importance": "critical",
                "scan_frequency_minutes": 5,
                "notes": "LNG bottleneck affects NG futures"
            },
            "singapore": {
                "significance_multiplier": 1.5,
                "strategic_importance": "high",
                "scan_frequency_minutes": 10,
                "notes": "Major Asian trading hub"
            },
            "fujairah": {
                "significance_multiplier": 1.4,
                "strategic_importance": "high",
                "scan_frequency_minutes": 10,
                "notes": "Persian Gulf access point"
            },
            "houston": {
                "significance_multiplier": 1.3,
                "strategic_importance": "medium",
                "scan_frequency_minutes": 15,
                "notes": "US Gulf Coast refining hub"
            },
            "rotterdam": {
                "significance_multiplier": 1.2,
                "strategic_importance": "medium",
                "scan_frequency_minutes": 15,
                "notes": "European gateway port"
            },
            "sabine_pass": {
                "significance_multiplier": 1.6,
                "strategic_importance": "high",
                "scan_frequency_minutes": 8,
                "notes": "Largest US LNG export terminal"
            },
            "freeport_lng": {
                "significance_multiplier": 1.4,
                "strategic_importance": "medium",
                "scan_frequency_minutes": 12,
                "notes": "Major US LNG export facility"
            },
            "cameron_lng": {
                "significance_multiplier": 1.3,
                "strategic_importance": "medium",
                "scan_frequency_minutes": 12,
                "notes": "Gulf Coast LNG export hub"
            }
        }
        
        # Priority scanning order
        self.port_scan_priority = [
            "strait_of_hormuz",
            "suez_canal", 
            "panama_canal",
            "sabine_pass",
            "singapore",
            "fujairah",
            "freeport_lng",
            "cameron_lng",
            "houston",
            "rotterdam"
        ]
        
        logger.info(f"DataFetcher initialized - Mock mode: {self.mock_mode}")
        logger.info(f"Priority ports configured: {len(self.port_scan_priority)}")

    def _rate_limit_datalastic(self):
        """Enforce rate limiting for Datalastic API"""
        elapsed = time.time() - self.datalastic_last_request
        if elapsed < self.datalastic_rate_limit:
            time.sleep(self.datalastic_rate_limit - elapsed)
        self.datalastic_last_request = time.time()

    def _rate_limit_twelve_data(self):
        """Enforce rate limiting for 12Data API"""
        elapsed = time.time() - self.twelve_data_last_request
        if elapsed < self.twelve_data_rate_limit:
            time.sleep(self.twelve_data_rate_limit - elapsed)
        self.twelve_data_last_request = time.time()

    def _get_cache_key(self, method: str, *args) -> str:
        """Generate cache key"""
        return f"{method}_{hash(str(args))}"

    def _get_from_cache(self, key: str, cache_type: str = 'standard') -> Optional[Any]:
        """Get data from cache if not expired"""
        if key in self.cache:
            data, timestamp = self.cache[key]
            
            # Use appropriate cache duration based on data type
            if cache_type == 'critical_ports':
                duration = self.cache_duration['critical_ports']
            elif cache_type == 'standard_ports':
                duration = self.cache_duration['standard_ports']
            elif cache_type == 'vessel_positions':
                duration = self.cache_duration['vessel_positions']
            elif cache_type == 'market_data':
                duration = self.cache_duration['market_data']
            else:
                duration = self.cache_duration['standard_ports']
                
            if time.time() - timestamp < duration:
                return data
        return None

    def _set_cache(self, key: str, data: Any):
        """Set data in cache"""
        self.cache[key] = (data, time.time())

    def get_vessel_positions(self, region: str = "global", cargo_type: str = "all") -> List[VesselPosition]:
        """Get vessel positions for a region and cargo type with priority-based caching"""
        cache_key = self._get_cache_key("vessel_positions", region, cargo_type)
        cached_data = self._get_from_cache(cache_key, 'vessel_positions')
        if cached_data:
            return cached_data

        if self.mock_mode:
            vessels = self._mock_vessel_positions(region, cargo_type)
        else:
            vessels = self._fetch_vessel_positions(region, cargo_type)
        
        self._set_cache(cache_key, vessels)
        return vessels

    def _mock_vessel_positions(self, region: str, cargo_type: str) -> List[VesselPosition]:
        """Generate mock vessel position data"""
        vessels = []
        
        # Define region coordinates
        regions = {
            "singapore": {"lat_range": (1.0, 1.5), "lon_range": (103.5, 104.0)},
            "houston": {"lat_range": (29.5, 30.0), "lon_range": (-95.5, -95.0)},
            "rotterdam": {"lat_range": (51.5, 52.0), "lon_range": (3.5, 4.5)},
            "global": {"lat_range": (-90, 90), "lon_range": (-180, 180)}
        }
        
        region_coords = regions.get(region.lower(), regions["global"])
        cargo_types = ["crude_oil", "lng", "container", "dry_bulk", "chemicals"] if cargo_type == "all" else [cargo_type]
        
        num_vessels = random.randint(10, 50)
        for i in range(num_vessels):
            vessels.append(VesselPosition(
                vessel_id=f"VESSEL_{i:04d}",
                name=f"Test Vessel {i}",
                lat=random.uniform(*region_coords["lat_range"]),
                lon=random.uniform(*region_coords["lon_range"]),
                speed=random.uniform(0, 25),
                course=random.randint(0, 360),
                cargo_type=random.choice(cargo_types),
                timestamp=datetime.now()
            ))
        
        logger.info(f"Generated {len(vessels)} mock vessel positions for {region}")
        return vessels

    def _fetch_vessel_positions(self, region: str, cargo_type: str) -> List[VesselPosition]:
        """Fetch real vessel positions from Datalastic API"""
        self._rate_limit_datalastic()
        
        try:
            # Use the correct working endpoint: vessel_inradius
            url = "https://api.datalastic.com/api/v0/vessel_inradius"
            
            # Map regions to coordinates (all strategic ports including chokepoints)
            region_coords = {
                "singapore": {"lat": 1.35, "lon": 103.8, "radius": 50},
                "houston": {"lat": 29.7604, "lon": -95.3698, "radius": 50}, 
                "rotterdam": {"lat": 51.9225, "lon": 4.47917, "radius": 50},
                "fujairah": {"lat": 25.2048, "lon": 55.2708, "radius": 50},
                "strait_of_hormuz": {"lat": 26.5667, "lon": 56.25, "radius": 60},
                "suez_canal": {"lat": 30.0444, "lon": 32.3499, "radius": 40},
                "panama_canal": {"lat": 9.0820, "lon": -79.6749, "radius": 30},
                "sabine_pass": {"lat": 29.7278, "lon": -93.8700, "radius": 25},
                "freeport_lng": {"lat": 28.9544, "lon": -95.3656, "radius": 25},
                "cameron_lng": {"lat": 29.7964, "lon": -93.3132, "radius": 25},
            }
            
            coords = region_coords.get(region.lower(), region_coords["singapore"])
            
            headers = {
                "Accept": "application/json",
                "User-Agent": "Maritime-Trading-System/1.0"
            }
            
            params = {
                "api-key": self.datalastic_key,
                "lat": coords["lat"],
                "lon": coords["lon"],
                "radius": coords["radius"]
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            vessels = []
            
            # Parse Datalastic response structure: {"data": {"vessels": [...]}}
            vessel_list = data.get("data", {}).get("vessels", [])
            
            for vessel_data in vessel_list:
                # Map vessel type to cargo type
                vessel_type = vessel_data.get("type", "Unknown")
                cargo_type_mapped = self._map_vessel_type_to_cargo(vessel_type)
                
                # Skip if filtering by cargo type and doesn't match
                if cargo_type != "all" and cargo_type_mapped != cargo_type:
                    continue
                
                # Parse timestamp
                timestamp_str = vessel_data.get("last_position_UTC")
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')) if timestamp_str else datetime.now()
                except:
                    timestamp = datetime.now()
                
                vessels.append(VesselPosition(
                    vessel_id=vessel_data.get("mmsi", vessel_data.get("uuid", "UNKNOWN")),
                    name=vessel_data.get("name", "Unknown Vessel"),
                    lat=float(vessel_data.get("lat", 0)),
                    lon=float(vessel_data.get("lon", 0)),
                    speed=float(vessel_data.get("speed", 0) or 0),
                    course=float(vessel_data.get("course", 0) or 0),
                    cargo_type=cargo_type_mapped,
                    timestamp=timestamp
                ))
            
            logger.info(f"Fetched {len(vessels)} vessel positions from Datalastic for {region}")
            return vessels
            
        except Exception as e:
            logger.error(f"Error fetching vessel positions from Datalastic: {e}")
            return self._mock_vessel_positions(region, cargo_type)
    
    def _map_vessel_type_to_cargo(self, vessel_type: str) -> str:
        """Map Datalastic vessel type to our cargo type categories"""
        if not vessel_type:
            return "other"
        
        vessel_type = vessel_type.lower()
        
        if "tanker" in vessel_type or "oil" in vessel_type:
            return "crude_oil"
        elif "lng" in vessel_type or "gas" in vessel_type:
            return "lng"
        elif "container" in vessel_type:
            return "container"
        elif "bulk" in vessel_type or "cargo" in vessel_type:
            return "dry_bulk"
        elif "chemical" in vessel_type:
            return "chemicals"
        else:
            return "other"

    def get_port_congestion(self, port_name: str) -> PortCongestion:
        """Get port congestion data with priority-based caching"""
        cache_key = self._get_cache_key("port_congestion", port_name)
        
        # Determine cache type based on port priority
        cache_type = 'critical_ports' if self._is_critical_port(port_name) else 'standard_ports'
        cached_data = self._get_from_cache(cache_key, cache_type)
        if cached_data:
            return cached_data

        if self.mock_mode:
            congestion = self._mock_port_congestion(port_name)
        else:
            congestion = self._fetch_port_congestion(port_name)
        
        # Apply significance multiplier to congestion score
        if port_name in self.port_priorities:
            multiplier = self.port_priorities[port_name]["significance_multiplier"]
            congestion.congestion_score = min(1.0, congestion.congestion_score * multiplier)
            logger.debug(f"Applied {multiplier}x multiplier to {port_name} congestion score")
        
        self._set_cache(cache_key, congestion)
        return congestion

    def _mock_port_congestion(self, port_name: str) -> PortCongestion:
        """Generate mock port congestion data"""
        waiting_vessels = random.randint(5, 50)
        average_wait = random.uniform(2, 48)  # hours
        
        # Simulate higher congestion for major ports
        if port_name.lower() in ["singapore", "houston", "rotterdam"]:
            waiting_vessels = random.randint(15, 80)
            average_wait = random.uniform(6, 72)
        
        congestion_score = min(1.0, (waiting_vessels / 30) * (average_wait / 24))
        
        return PortCongestion(
            port_name=port_name,
            waiting_vessels=waiting_vessels,
            average_wait_time=average_wait,
            cargo_types=["crude_oil", "lng", "container", "chemicals"],
            congestion_score=congestion_score,
            timestamp=datetime.now()
        )

    def _fetch_port_congestion(self, port_name: str) -> PortCongestion:
        """Fetch real port congestion from Datalastic API"""
        self._rate_limit_datalastic()
        
        try:
            url = f"https://api.datalastic.com/api/v0/port_congestion/{port_name}"
            headers = {"X-API-Key": self.datalastic_key}
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            return PortCongestion(
                port_name=port_name,
                waiting_vessels=data["waiting_vessels"],
                average_wait_time=data["average_wait_time"],
                cargo_types=data["cargo_types"],
                congestion_score=data["congestion_score"],
                timestamp=datetime.fromisoformat(data["timestamp"])
            )
            
        except Exception as e:
            logger.error(f"Error fetching port congestion for {port_name}: {e}")
            return self._mock_port_congestion(port_name)

    def get_market_data(self, symbol: str, interval: str = "5min") -> List[MarketData]:
        """Get market data for a symbol with optimized caching"""
        cache_key = self._get_cache_key("market_data", symbol, interval)
        cached_data = self._get_from_cache(cache_key, 'market_data')
        if cached_data:
            return cached_data

        if self.mock_mode:
            market_data = self._mock_market_data(symbol, interval)
        else:
            market_data = self._fetch_market_data(symbol, interval)
        
        self._set_cache(cache_key, market_data)
        return market_data

    def _mock_market_data(self, symbol: str, interval: str) -> List[MarketData]:
        """Generate mock market data"""
        data_points = []
        base_price = {"CL": 75.0, "NG": 3.5, "GC": 2000.0}.get(symbol, 100.0)
        
        start_time = datetime.now() - timedelta(hours=24)
        
        for i in range(288):  # 24 hours of 5-minute data
            timestamp = start_time + timedelta(minutes=i * 5)
            
            # Random walk with some volatility
            price_change = random.uniform(-0.5, 0.5)
            price = base_price + price_change
            
            volume = random.randint(1000, 50000)
            
            data_points.append(MarketData(
                symbol=symbol,
                price=price,
                volume=volume,
                timestamp=timestamp,
                open=price - random.uniform(-0.2, 0.2),
                high=price + random.uniform(0, 0.5),
                low=price - random.uniform(0, 0.5),
                close=price
            ))
            
            base_price = price  # For next iteration
        
        logger.info(f"Generated {len(data_points)} mock market data points for {symbol}")
        return data_points

    def _fetch_market_data(self, symbol: str, interval: str) -> List[MarketData]:
        """Fetch real market data from 12Data API"""
        self._rate_limit_twelve_data()
        
        try:
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": symbol,
                "interval": interval,
                "apikey": self.twelve_data_key,
                "outputsize": 288
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            market_data = []
            
            for point in data.get("values", []):
                market_data.append(MarketData(
                    symbol=symbol,
                    price=float(point["close"]),
                    volume=int(point.get("volume", 0)),
                    timestamp=datetime.fromisoformat(point["datetime"]),
                    open=float(point["open"]),
                    high=float(point["high"]),
                    low=float(point["low"]),
                    close=float(point["close"])
                ))
            
            logger.info(f"Fetched {len(market_data)} market data points for {symbol}")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return self._mock_market_data(symbol, interval)

    def get_vessel_routes(self, vessel_ids: List[str]) -> Dict[str, List[Dict]]:
        """Get vessel route data"""
        if self.mock_mode:
            return self._mock_vessel_routes(vessel_ids)
        else:
            return self._fetch_vessel_routes(vessel_ids)

    def _mock_vessel_routes(self, vessel_ids: List[str]) -> Dict[str, List[Dict]]:
        """Generate mock vessel route data"""
        routes = {}
        
        for vessel_id in vessel_ids:
            route_points = []
            
            # Generate 5-10 route points
            num_points = random.randint(5, 10)
            start_lat, start_lon = random.uniform(-90, 90), random.uniform(-180, 180)
            
            for i in range(num_points):
                route_points.append({
                    "lat": start_lat + random.uniform(-5, 5),
                    "lon": start_lon + random.uniform(-5, 5),
                    "timestamp": datetime.now() - timedelta(hours=i),
                    "speed": random.uniform(10, 20),
                    "deviation_from_planned": random.uniform(0, 200)  # nautical miles
                })
            
            routes[vessel_id] = route_points
        
        return routes

    def _fetch_vessel_routes(self, vessel_ids: List[str]) -> Dict[str, List[Dict]]:
        """Fetch real vessel routes from Datalastic API"""
        routes = {}
        
        for vessel_id in vessel_ids:
            try:
                self._rate_limit_datalastic()
                
                url = f"https://api.datalastic.com/api/v0/vessel_route/{vessel_id}"
                headers = {"X-API-Key": self.datalastic_key}
                
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                routes[vessel_id] = data.get("route_points", [])
                
            except Exception as e:
                logger.error(f"Error fetching route for vessel {vessel_id}: {e}")
                routes[vessel_id] = self._mock_vessel_routes([vessel_id])[vessel_id]
        
        return routes

    def get_volume_profile(self, symbol: str) -> Dict[str, Any]:
        """Get volume profile data for a symbol"""
        if self.mock_mode:
            return self._mock_volume_profile(symbol)
        else:
            return self._fetch_volume_profile(symbol)

    def _mock_volume_profile(self, symbol: str) -> Dict[str, Any]:
        """Generate mock volume profile data"""
        base_price = {"CL": 75.0, "NG": 3.5, "GC": 2000.0}.get(symbol, 100.0)
        
        # Generate volume profile
        price_levels = []
        volumes = []
        
        for i in range(20):  # 20 price levels
            price = base_price + (i - 10) * 0.5
            volume = random.randint(5000, 50000)
            price_levels.append(price)
            volumes.append(volume)
        
        max_volume_idx = volumes.index(max(volumes))
        poc = price_levels[max_volume_idx]  # Point of Control
        
        # Value Area High/Low (70% of volume)
        total_volume = sum(volumes)
        target_volume = total_volume * 0.7
        
        vah = poc + 2.0  # Simplified
        val = poc - 2.0
        
        return {
            "symbol": symbol,
            "poc": poc,
            "vah": vah,
            "val": val,
            "price_levels": price_levels,
            "volumes": volumes,
            "total_volume": total_volume,
            "timestamp": datetime.now()
        }

    def _fetch_volume_profile(self, symbol: str) -> Dict[str, Any]:
        """Fetch real volume profile data"""
        # This would require a specialized API or calculation
        # For now, return mock data
        return self._mock_volume_profile(symbol)

    def get_economic_calendar(self) -> List[Dict[str, Any]]:
        """Get economic calendar events"""
        if self.mock_mode:
            return self._mock_economic_calendar()
        else:
            return self._fetch_economic_calendar()

    def _mock_economic_calendar(self) -> List[Dict[str, Any]]:
        """Generate mock economic calendar events"""
        events = [
            {
                "event": "EIA Crude Oil Inventories",
                "time": datetime.now() + timedelta(hours=2),
                "impact": "high",
                "forecast": "-2.5M",
                "previous": "-1.2M"
            },
            {
                "event": "API Weekly Crude Oil Stock",
                "time": datetime.now() + timedelta(hours=18),
                "impact": "medium",
                "forecast": "1.5M",
                "previous": "0.8M"
            },
            {
                "event": "Natural Gas Storage",
                "time": datetime.now() + timedelta(days=1),
                "impact": "high",
                "forecast": "45 Bcf",
                "previous": "52 Bcf"
            }
        ]
        
        return events

    def _fetch_economic_calendar(self) -> List[Dict[str, Any]]:
        """Fetch real economic calendar data"""
        # This would require an economic calendar API
        return self._mock_economic_calendar()

    def health_check(self) -> Dict[str, bool]:
        """Check the health of all data sources"""
        status = {
            "datalastic_api": False,
            "twelve_data_api": False,
            "cache_system": True
        }
        
        if self.mock_mode:
            status["mock_mode"] = True
            status["datalastic_api"] = True
            status["twelve_data_api"] = True
        else:
            # Test actual API connections
            try:
                if self.datalastic_key:
                    # Test Datalastic connection
                    pass
                if self.twelve_data_key:
                    # Test 12Data connection
                    pass
            except Exception as e:
                logger.error(f"Health check failed: {e}")
        
        return status

    def _is_critical_port(self, port_name: str) -> bool:
        """Check if port is classified as critical chokepoint"""
        if port_name in self.port_priorities:
            return self.port_priorities[port_name]["strategic_importance"] == "critical"
        return False

    def get_port_priority_info(self, port_name: str) -> Dict[str, Any]:
        """Get priority information for a port"""
        return self.port_priorities.get(port_name, {
            "significance_multiplier": 1.0,
            "strategic_importance": "standard",
            "scan_frequency_minutes": 15,
            "notes": "Standard port monitoring"
        })

    def get_priority_scanning_order(self) -> List[str]:
        """Get the priority order for port scanning"""
        return self.port_scan_priority.copy()

    def apply_chokepoint_multipliers(self, events: List[Any]) -> List[Any]:
        """Apply chokepoint multipliers to maritime events"""
        for event in events:
            if hasattr(event, 'location') and event.location in self.port_priorities:
                multiplier = self.port_priorities[event.location]["significance_multiplier"]
                if hasattr(event, 'severity'):
                    event.severity = min(1.0, event.severity * multiplier)
                if hasattr(event, 'estimated_price_impact'):
                    event.estimated_price_impact *= multiplier
                logger.debug(f"Applied {multiplier}x chokepoint multiplier to {event.location}")
        return events

    def get_api_optimization_settings(self) -> Dict[str, Any]:
        """Get API usage optimization settings"""
        return {
            "priority_ports_frequency": 3,      # Every 3 minutes for critical ports
            "secondary_ports_frequency": 15,    # Every 15 minutes for standard ports
            "batch_size": 5,                   # 5 ports per batch
            "rate_limit_delay": 1,             # 1 second between calls
            "max_concurrent_requests": 3       # Maximum parallel requests
        }