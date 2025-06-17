import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging
from ..core.data_fetcher import PortCongestion, VesselPosition, MarketData
from ..core.maritime_analyzer import MaritimeEvent, CongestionEvent
from ..core.signal_generator import TradingSignal, SignalDirection, SignalTier

logger = logging.getLogger(__name__)

class PortCongestionStrategy:
    def __init__(self):
        # Major ports and their primary commodities
        self.major_ports = {
            "singapore": {
                "primary_commodity": "crude_oil",
                "normal_wait_time": 8.5,  # hours
                "normal_vessel_count": 25,
                "significance_multiplier": 1.5  # congestion threshold
            },
            "houston": {
                "primary_commodity": "crude_oil",
                "normal_wait_time": 12.0,
                "normal_vessel_count": 35,
                "significance_multiplier": 1.5
            },
            "rotterdam": {
                "primary_commodity": "crude_oil",
                "normal_wait_time": 6.5,
                "normal_vessel_count": 20,
                "significance_multiplier": 1.5
            },
            "fujairah": {
                "primary_commodity": "crude_oil",
                "normal_wait_time": 10.0,
                "normal_vessel_count": 30,
                "significance_multiplier": 1.5
            },
            "freeport": {
                "primary_commodity": "lng",
                "normal_wait_time": 15.0,
                "normal_vessel_count": 12,
                "significance_multiplier": 1.3
            },
            "sabine_pass": {
                "primary_commodity": "lng",
                "normal_wait_time": 18.0,
                "normal_vessel_count": 10,
                "significance_multiplier": 1.3
            }
        }
        
        # Commodity symbols mapping
        self.commodity_symbols = {
            "crude_oil": "CL",
            "lng": "NG",
            "natural_gas": "NG",
            "gasoline": "RB",
            "heating_oil": "HO"
        }
        
        logger.info("PortCongestionStrategy initialized")

    def analyze_port_congestion(self, port_data: PortCongestion, 
                               vessel_positions: List[VesselPosition],
                               market_data: Dict[str, List[MarketData]]) -> Optional[TradingSignal]:
        """Analyze port congestion and generate trading signal if conditions are met"""
        
        port_name = port_data.port_name.lower()
        
        if port_name not in self.major_ports:
            logger.debug(f"Port {port_name} not in major ports list")
            return None
        
        port_config = self.major_ports[port_name]
        commodity = port_config["primary_commodity"]
        symbol = self.commodity_symbols.get(commodity)
        
        if not symbol or symbol not in market_data:
            logger.warning(f"No market data available for {symbol}")
            return None
        
        # Calculate congestion severity
        congestion_severity = self._calculate_congestion_severity(port_data, port_config)
        
        if congestion_severity < 0.3:  # Minimum threshold
            return None
        
        # Analyze vessel composition at the port
        vessel_analysis = self._analyze_port_vessels(port_name, vessel_positions, commodity)
        
        # Get market context
        market_context = self._analyze_market_context(market_data[symbol])
        
        # Determine signal direction and strength
        signal_direction, signal_strength = self._determine_signal_direction(
            congestion_severity, vessel_analysis, market_context, commodity
        )
        
        if not signal_direction:
            return None
        
        # Calculate entry and exit levels
        current_price = market_data[symbol][-1].close
        entry_price, stop_loss, targets = self._calculate_levels(
            current_price, signal_direction, congestion_severity, market_context
        )
        
        # Calculate confidence score
        confidence = self._calculate_confidence(
            congestion_severity, vessel_analysis, market_context, signal_strength
        )
        
        if confidence < 0.65:  # Minimum confidence threshold
            return None
        
        # Determine signal tier
        tier = self._determine_signal_tier(congestion_severity, confidence, vessel_analysis)
        
        # Generate reason string
        reason = self._generate_reason(port_data, congestion_severity, vessel_analysis)
        
        # Create maritime events for context
        maritime_events = [
            CongestionEvent(
                event_type="port_congestion",
                severity=congestion_severity,
                affected_commodity=commodity,
                estimated_price_impact=signal_strength * 0.01,  # Convert to percentage
                confidence_level=confidence,
                location=port_data.port_name,
                timestamp=port_data.timestamp,
                details={
                    "vessel_analysis": vessel_analysis,
                    "market_context": market_context
                },
                waiting_vessels=port_data.waiting_vessels,
                average_wait_time=port_data.average_wait_time,
                congestion_multiplier=congestion_severity
            )
        ]
        
        # Create trading signal
        signal = TradingSignal(
            symbol=symbol,
            direction=signal_direction,
            tier=tier,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_1=targets[0],
            target_2=targets[1],
            position_size=0,  # Will be calculated by risk manager
            confidence_score=confidence,
            reason=reason,
            maritime_events=maritime_events,
            timestamp=datetime.now(),
            expiry_time=datetime.now() + timedelta(hours=48),  # Port congestion signals last longer
            risk_amount=abs(entry_price - stop_loss),
            reward_ratio=abs(targets[0] - entry_price) / abs(entry_price - stop_loss),
            volume_confirmation=market_context.get("volume_breakout", False),
            technical_score=market_context.get("technical_score", 0.5),
            maritime_score=congestion_severity,
            volume_score=market_context.get("volume_score", 0.5)
        )
        
        logger.info(f"Port congestion signal generated: {port_name} -> {symbol} {signal_direction.value}")
        return signal

    def _calculate_congestion_severity(self, port_data: PortCongestion, 
                                     port_config: Dict) -> float:
        """Calculate congestion severity score (0-1)"""
        
        # Wait time factor
        wait_multiplier = port_data.average_wait_time / port_config["normal_wait_time"]
        
        # Vessel count factor
        vessel_multiplier = port_data.waiting_vessels / port_config["normal_vessel_count"]
        
        # Combined congestion score
        congestion_score = (wait_multiplier + vessel_multiplier) / 2
        
        # Normalize to 0-1 scale, with significance threshold
        significance_threshold = port_config["significance_multiplier"]
        
        if congestion_score < significance_threshold:
            return 0.0
        
        # Scale above threshold
        severity = (congestion_score - significance_threshold) / (3.0 - significance_threshold)
        return min(1.0, max(0.0, severity))

    def _analyze_port_vessels(self, port_name: str, vessel_positions: List[VesselPosition],
                            expected_commodity: str) -> Dict[str, float]:
        """Analyze vessels near the port"""
        
        # This is a simplified analysis - in reality you'd need port coordinates
        # and calculate distances to vessels
        
        port_vessels = [v for v in vessel_positions if v.speed < 2.0]  # Assume slow/stopped vessels are waiting
        
        if not port_vessels:
            return {"relevant_vessels": 0, "commodity_match": 0.0, "size_factor": 0.0}
        
        # Count vessels with matching commodity
        commodity_vessels = [v for v in port_vessels if v.cargo_type == expected_commodity]
        commodity_match_ratio = len(commodity_vessels) / len(port_vessels)
        
        # Analyze vessel sizes (simplified - using speed as proxy for size/urgency)
        avg_speed = np.mean([v.speed for v in port_vessels])
        size_factor = 1.0 - (avg_speed / 5.0)  # Lower speed = larger/more significant vessels
        
        return {
            "relevant_vessels": len(port_vessels),
            "commodity_match": commodity_match_ratio,
            "size_factor": max(0.0, min(1.0, size_factor)),
            "total_waiting": len(port_vessels)
        }

    def _analyze_market_context(self, market_data: List[MarketData]) -> Dict[str, float]:
        """Analyze market conditions for context"""
        
        if len(market_data) < 20:
            return {"technical_score": 0.5, "volume_score": 0.5, "volume_breakout": False}
        
        prices = [d.close for d in market_data[-20:]]
        volumes = [d.volume for d in market_data[-20:]]
        
        # Technical analysis
        ma_5 = np.mean(prices[-5:])
        ma_20 = np.mean(prices[-20:])
        current_price = prices[-1]
        
        # Trend score
        if current_price > ma_5 > ma_20:
            trend_score = 0.8
        elif current_price > ma_5:
            trend_score = 0.6
        elif current_price < ma_5 < ma_20:
            trend_score = 0.2
        else:
            trend_score = 0.4
        
        # Volume analysis
        recent_volume = np.mean(volumes[-5:])
        baseline_volume = np.mean(volumes[-20:-5])
        volume_ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
        
        volume_breakout = volume_ratio > 1.5
        volume_score = min(1.0, volume_ratio / 2.0)
        
        # Volatility (helps determine if market is ready to move)
        returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
        volatility = np.std(returns)
        volatility_score = min(1.0, volatility / 0.03)  # Normalize around 3% daily volatility
        
        return {
            "technical_score": trend_score,
            "volume_score": volume_score,
            "volume_breakout": volume_breakout,
            "volatility_score": volatility_score,
            "volume_ratio": volume_ratio
        }

    def _determine_signal_direction(self, congestion_severity: float,
                                  vessel_analysis: Dict, market_context: Dict,
                                  commodity: str) -> Tuple[Optional[SignalDirection], float]:
        """Determine signal direction based on congestion analysis"""
        
        # Port congestion typically leads to higher prices (bullish for commodities)
        base_strength = congestion_severity
        
        # Adjust based on vessel analysis
        if vessel_analysis["commodity_match"] > 0.7:  # Relevant vessels waiting
            base_strength *= 1.3
        
        if vessel_analysis["size_factor"] > 0.6:  # Large vessels waiting
            base_strength *= 1.2
        
        # Adjust based on market conditions
        technical_boost = market_context["technical_score"]
        if technical_boost > 0.6:  # Already in uptrend
            base_strength *= 1.1
        elif technical_boost < 0.4:  # In downtrend, might need more strength to reverse
            base_strength *= 0.9
        
        # Volume confirmation
        if market_context["volume_breakout"]:
            base_strength *= 1.2
        
        # Minimum threshold for signal generation
        if base_strength < 0.4:
            return None, 0.0
        
        # Port congestion is generally bullish for commodity prices
        return SignalDirection.LONG, base_strength

    def _calculate_levels(self, current_price: float, direction: SignalDirection,
                         congestion_severity: float, market_context: Dict) -> Tuple[float, float, List[float]]:
        """Calculate entry, stop loss, and target levels"""
        
        volatility = market_context.get("volatility_score", 0.5) * 0.02  # Convert to price volatility
        
        # Entry price (slightly above current for breakout)
        if direction == SignalDirection.LONG:
            entry_price = current_price * 1.002  # 0.2% above current
        else:
            entry_price = current_price * 0.998  # 0.2% below current
        
        # Stop loss based on volatility and congestion strength
        stop_distance = current_price * (0.01 + volatility)  # 1% base + volatility
        
        # Tighter stops for higher confidence signals
        if congestion_severity > 0.8:
            stop_distance *= 0.8
        
        if direction == SignalDirection.LONG:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        # Targets based on congestion severity
        risk = abs(entry_price - stop_loss)
        
        # Target 1: 2:1 risk/reward, adjusted by congestion strength
        target_1_multiplier = 2.0 + congestion_severity  # 2-3x risk
        
        # Target 2: 3:1 risk/reward, adjusted by congestion strength  
        target_2_multiplier = 3.0 + (congestion_severity * 1.5)  # 3-4.5x risk
        
        if direction == SignalDirection.LONG:
            target_1 = entry_price + (risk * target_1_multiplier)
            target_2 = entry_price + (risk * target_2_multiplier)
        else:
            target_1 = entry_price - (risk * target_1_multiplier)
            target_2 = entry_price - (risk * target_2_multiplier)
        
        return entry_price, stop_loss, [target_1, target_2]

    def _calculate_confidence(self, congestion_severity: float, vessel_analysis: Dict,
                            market_context: Dict, signal_strength: float) -> float:
        """Calculate overall confidence score"""
        
        # Base confidence from congestion severity
        base_confidence = 0.4 + (congestion_severity * 0.4)  # 0.4-0.8 range
        
        # Boost from vessel analysis
        if vessel_analysis["commodity_match"] > 0.8:
            base_confidence += 0.1
        
        if vessel_analysis["relevant_vessels"] > 15:  # Significant vessel count
            base_confidence += 0.05
        
        # Boost from market conditions
        if market_context["volume_breakout"]:
            base_confidence += 0.1
        
        if market_context["technical_score"] > 0.7:
            base_confidence += 0.05
        
        # Penalize if market conditions are poor
        if market_context["technical_score"] < 0.3:
            base_confidence -= 0.1
        
        return min(1.0, max(0.0, base_confidence))

    def _determine_signal_tier(self, congestion_severity: float, confidence: float,
                             vessel_analysis: Dict) -> SignalTier:
        """Determine signal tier based on criteria"""
        
        # Tier 1: High congestion with high confidence
        if (congestion_severity >= 0.8 and confidence >= 0.85 and 
            vessel_analysis["commodity_match"] > 0.8):
            return SignalTier.TIER_1
        
        # Tier 2: Moderate to high congestion with good confidence
        elif (congestion_severity >= 0.6 and confidence >= 0.7):
            return SignalTier.TIER_2
        
        # Tier 3: Lower congestion or confidence
        else:
            return SignalTier.TIER_3

    def _generate_reason(self, port_data: PortCongestion, congestion_severity: float,
                        vessel_analysis: Dict) -> str:
        """Generate human-readable reason for the signal"""
        
        port_name = port_data.port_name.title()
        
        # Main congestion description
        if congestion_severity > 0.8:
            congestion_desc = "severe congestion"
        elif congestion_severity > 0.6:
            congestion_desc = "significant congestion"
        else:
            congestion_desc = "moderate congestion"
        
        reason = f"{port_name} {congestion_desc} "
        
        # Add specific metrics
        wait_time = port_data.average_wait_time
        vessel_count = port_data.waiting_vessels
        
        reason += f"({vessel_count} vessels, {wait_time:.1f}h avg wait)"
        
        # Add vessel analysis if significant
        if vessel_analysis["commodity_match"] > 0.7:
            reason += f" + {vessel_analysis['commodity_match']*100:.0f}% commodity match"
        
        return reason

    def analyze_multiple_ports(self, port_data_list: List[PortCongestion],
                              vessel_positions: List[VesselPosition],
                              market_data: Dict[str, List[MarketData]]) -> List[TradingSignal]:
        """Analyze multiple ports and generate signals"""
        
        signals = []
        
        for port_data in port_data_list:
            try:
                signal = self.analyze_port_congestion(port_data, vessel_positions, market_data)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error analyzing port {port_data.port_name}: {e}")
        
        # Sort by confidence and congestion severity
        signals.sort(key=lambda x: x.confidence_score * x.maritime_score, reverse=True)
        
        return signals

    def get_port_status_summary(self, port_data_list: List[PortCongestion]) -> Dict[str, Any]:
        """Get summary of all port statuses"""
        
        summary = {
            "total_ports": len(port_data_list),
            "congested_ports": 0,
            "severely_congested": 0,
            "total_waiting_vessels": 0,
            "avg_wait_time": 0.0,
            "port_details": []
        }
        
        if not port_data_list:
            return summary
        
        total_wait_time = 0
        
        for port_data in port_data_list:
            port_name = port_data.port_name.lower()
            
            if port_name in self.major_ports:
                port_config = self.major_ports[port_name]
                severity = self._calculate_congestion_severity(port_data, port_config)
                
                if severity > 0.3:
                    summary["congested_ports"] += 1
                
                if severity > 0.7:
                    summary["severely_congested"] += 1
                
                summary["port_details"].append({
                    "name": port_data.port_name,
                    "waiting_vessels": port_data.waiting_vessels,
                    "avg_wait_time": port_data.average_wait_time,
                    "congestion_severity": severity,
                    "commodity": port_config["primary_commodity"]
                })
            
            summary["total_waiting_vessels"] += port_data.waiting_vessels
            total_wait_time += port_data.average_wait_time
        
        summary["avg_wait_time"] = total_wait_time / len(port_data_list)
        
        return summary