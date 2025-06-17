import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from .maritime_analyzer import MaritimeEvent
from .data_fetcher import MarketData

logger = logging.getLogger(__name__)

class SignalDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"

class SignalTier(Enum):
    TIER_1 = 1  # Pure maritime signals (85% win rate)
    TIER_2 = 2  # Maritime + Volume confirmation (70% win rate)  
    TIER_3 = 3  # Technical with maritime context (60% win rate)

@dataclass
class TradingSignal:
    symbol: str
    direction: SignalDirection
    tier: SignalTier
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    position_size: int
    confidence_score: float
    reason: str
    maritime_events: List[MaritimeEvent]
    timestamp: datetime
    expiry_time: datetime
    risk_amount: float
    reward_ratio: float
    volume_confirmation: bool = False
    technical_score: float = 0.0
    maritime_score: float = 0.0
    volume_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction.value,
            "tier": self.tier.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "target_1": self.target_1,
            "target_2": self.target_2,
            "position_size": self.position_size,
            "confidence_score": self.confidence_score,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "expiry_time": self.expiry_time.isoformat(),
            "risk_amount": self.risk_amount,
            "reward_ratio": self.reward_ratio,
            "volume_confirmation": self.volume_confirmation,
            "technical_score": self.technical_score,
            "maritime_score": self.maritime_score,
            "volume_score": self.volume_score
        }

class SignalGenerator:
    def __init__(self, risk_per_trade: float = 0.01, min_confidence: float = 0.65):
        self.risk_per_trade = risk_per_trade
        self.min_confidence = min_confidence
        
        # Symbol mappings for commodities
        self.commodity_symbols = {
            "crude_oil": "CL",
            "natural_gas": "NG", 
            "lng": "NG",
            "gasoline": "RB",
            "heating_oil": "HO",
            "gold": "GC",
            "silver": "SI",
            "copper": "HG"
        }
        
        # Tier criteria
        self.tier_criteria = {
            SignalTier.TIER_1: {
                "min_maritime_score": 0.8,
                "min_confidence": 0.85,
                "expected_win_rate": 0.85
            },
            SignalTier.TIER_2: {
                "min_maritime_score": 0.6,
                "min_volume_confirmation": True,
                "min_confidence": 0.70,
                "expected_win_rate": 0.70
            },
            SignalTier.TIER_3: {
                "min_technical_score": 0.5,
                "min_confidence": 0.60,
                "expected_win_rate": 0.60
            }
        }
        
        logger.info("SignalGenerator initialized")

    def generate_signals(self, 
                        maritime_events: List[MaritimeEvent],
                        market_data: Dict[str, List[MarketData]],
                        volume_profiles: Dict[str, Dict[str, Any]]) -> List[TradingSignal]:
        """Generate trading signals from maritime events and market data"""
        
        signals = []
        
        # Group events by commodity
        events_by_commodity = {}
        for event in maritime_events:
            commodity = event.affected_commodity
            if commodity not in events_by_commodity:
                events_by_commodity[commodity] = []
            events_by_commodity[commodity].append(event)
        
        # Generate signals for each commodity
        for commodity, events in events_by_commodity.items():
            symbol = self.commodity_symbols.get(commodity)
            if not symbol or symbol not in market_data:
                continue
            
            # Get market data for this symbol
            symbol_market_data = market_data[symbol]
            volume_profile = volume_profiles.get(symbol, {})
            
            # Calculate composite maritime score
            maritime_score = self._calculate_maritime_score(events)
            
            # Calculate volume score
            volume_score = self._calculate_volume_score(symbol_market_data, volume_profile)
            
            # Calculate technical score
            technical_score = self._calculate_technical_score(symbol_market_data)
            
            # Determine signal direction and strength
            direction, signal_strength = self._determine_signal_direction(events, symbol_market_data)
            
            if not direction:
                continue
            
            # Calculate entry price and levels
            current_price = symbol_market_data[-1].close
            entry_price = self._calculate_entry_price(direction, current_price, symbol_market_data)
            
            # Calculate stop loss and targets
            volatility = self._calculate_volatility(symbol_market_data)
            stop_loss = self._calculate_stop_loss(entry_price, direction, volatility)
            target_1, target_2 = self._calculate_targets(entry_price, stop_loss, direction)
            
            # Calculate overall confidence
            confidence = self._calculate_signal_confidence(maritime_score, volume_score, technical_score)
            
            if confidence < self.min_confidence:
                continue
            
            # Determine signal tier
            tier = self._determine_signal_tier(maritime_score, volume_score, technical_score, confidence)
            
            # Calculate risk metrics
            risk_amount = abs(entry_price - stop_loss)
            reward_1 = abs(target_1 - entry_price)
            reward_ratio = reward_1 / risk_amount if risk_amount > 0 else 0
            
            # Generate reason string
            reason = self._generate_reason(events, maritime_score, volume_score, technical_score)
            
            # Create signal
            signal = TradingSignal(
                symbol=symbol,
                direction=direction,
                tier=tier,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_1=target_1,
                target_2=target_2,
                position_size=0,  # Will be calculated by risk manager
                confidence_score=confidence,
                reason=reason,
                maritime_events=events,
                timestamp=datetime.now(),
                expiry_time=datetime.now() + timedelta(hours=24),
                risk_amount=risk_amount,
                reward_ratio=reward_ratio,
                volume_confirmation=volume_score > 0.6,
                technical_score=technical_score,
                maritime_score=maritime_score,
                volume_score=volume_score
            )
            
            signals.append(signal)
        
        # Sort by confidence and tier
        signals.sort(key=lambda x: (x.tier.value, -x.confidence_score))
        
        logger.info(f"Generated {len(signals)} trading signals")
        return signals

    def _calculate_maritime_score(self, events: List[MaritimeEvent]) -> float:
        """Calculate composite maritime score from events"""
        if not events:
            return 0.0
        
        # Weight events by severity and confidence
        weighted_scores = []
        total_weight = 0
        
        for event in events:
            weight = event.confidence_level
            score = event.severity
            
            # Boost score for high-impact event types
            if event.event_type in ["port_congestion", "vessel_clustering"]:
                score *= 1.2
            elif event.event_type == "supply_pressure":
                score *= 1.1
            
            weighted_scores.append(score * weight)
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        maritime_score = sum(weighted_scores) / total_weight
        return min(1.0, maritime_score)

    def _calculate_volume_score(self, market_data: List[MarketData], volume_profile: Dict[str, Any]) -> float:
        """Calculate volume-based score"""
        if len(market_data) < 20:
            return 0.0
        
        recent_data = market_data[-20:]
        volumes = [d.volume for d in recent_data]
        prices = [d.close for d in recent_data]
        
        # Volume trend
        recent_volume = np.mean(volumes[-5:])
        baseline_volume = np.mean(volumes[:-5])
        volume_ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
        
        # Volume breakout detection
        volume_breakout = 0.0
        if volume_ratio > 1.5:  # 50% above baseline
            volume_breakout = min(1.0, (volume_ratio - 1.0) / 2.0)
        
        # Price-volume correlation
        correlation = 0.0
        if len(volumes) > 10:
            try:
                correlation = abs(np.corrcoef(prices, volumes)[0, 1])
            except:
                correlation = 0.0
        
        # Volume profile analysis
        poc_score = 0.0
        if volume_profile and "poc" in volume_profile:
            current_price = prices[-1]
            poc = volume_profile["poc"]
            vah = volume_profile.get("vah", poc + 1)
            val = volume_profile.get("val", poc - 1)
            
            # Score based on price position relative to value area
            if val <= current_price <= vah:
                poc_score = 0.8  # In value area
            elif current_price > vah:
                poc_score = 0.9  # Above value area (bullish)
            else:
                poc_score = 0.7  # Below value area
        
        # Combine scores
        volume_score = (volume_breakout * 0.4 + correlation * 0.3 + poc_score * 0.3)
        return min(1.0, volume_score)

    def _calculate_technical_score(self, market_data: List[MarketData]) -> float:
        """Calculate technical analysis score"""
        if len(market_data) < 50:
            return 0.0
        
        prices = [d.close for d in market_data]
        volumes = [d.volume for d in market_data]
        
        # Moving averages
        ma_20 = np.mean(prices[-20:])
        ma_50 = np.mean(prices[-50:])
        current_price = prices[-1]
        
        # Trend score
        trend_score = 0.0
        if current_price > ma_20 > ma_50:
            trend_score = 0.8  # Strong uptrend
        elif current_price > ma_20:
            trend_score = 0.6  # Mild uptrend
        elif current_price < ma_20 < ma_50:
            trend_score = 0.2  # Downtrend
        else:
            trend_score = 0.4  # Sideways
        
        # Momentum (RSI-like calculation)
        momentum_score = self._calculate_momentum_score(prices)
        
        # Support/Resistance
        sr_score = self._calculate_support_resistance_score(prices)
        
        # Combine technical factors
        technical_score = (trend_score * 0.4 + momentum_score * 0.3 + sr_score * 0.3)
        return min(1.0, technical_score)

    def _calculate_momentum_score(self, prices: List[float]) -> float:
        """Calculate momentum-based score (simplified RSI)"""
        if len(prices) < 14:
            return 0.5
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        # Separate gains and losses
        gains = [max(0, change) for change in changes[-14:]]
        losses = [abs(min(0, change)) for change in changes[-14:]]
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 0.5
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Convert RSI to score (favor oversold/overbought conditions)
        if rsi < 30:
            return 0.8  # Oversold - potential bounce
        elif rsi > 70:
            return 0.3  # Overbought - potential decline
        else:
            return 0.5  # Neutral

    def _calculate_support_resistance_score(self, prices: List[float]) -> float:
        """Calculate support/resistance score"""
        if len(prices) < 20:
            return 0.5
        
        current_price = prices[-1]
        recent_high = max(prices[-20:])
        recent_low = min(prices[-20:])
        
        # Distance from recent high/low
        high_distance = (recent_high - current_price) / recent_high
        low_distance = (current_price - recent_low) / current_price
        
        # Score based on position
        if low_distance < 0.02:  # Near support
            return 0.8
        elif high_distance < 0.02:  # Near resistance
            return 0.3
        else:
            return 0.5

    def _determine_signal_direction(self, events: List[MaritimeEvent], 
                                  market_data: List[MarketData]) -> Tuple[Optional[SignalDirection], float]:
        """Determine signal direction and strength from events"""
        
        bullish_strength = 0.0
        bearish_strength = 0.0
        
        for event in events:
            impact = event.estimated_price_impact
            confidence = event.confidence_level
            
            # Weight by confidence
            weighted_impact = impact * confidence
            
            if weighted_impact > 0:
                bullish_strength += weighted_impact
            else:
                bearish_strength += abs(weighted_impact)
        
        # Net strength
        net_strength = bullish_strength - bearish_strength
        total_strength = bullish_strength + bearish_strength
        
        if total_strength < 0.005:  # Minimum threshold
            return None, 0.0
        
        # Determine direction
        if net_strength > 0.003:
            return SignalDirection.LONG, bullish_strength
        elif net_strength < -0.003:
            return SignalDirection.SHORT, bearish_strength
        else:
            return None, 0.0

    def _calculate_entry_price(self, direction: SignalDirection, current_price: float, 
                             market_data: List[MarketData]) -> float:
        """Calculate optimal entry price"""
        
        # For now, use current price with small buffer
        if direction == SignalDirection.LONG:
            return current_price * 1.001  # Slightly above current
        else:
            return current_price * 0.999  # Slightly below current

    def _calculate_volatility(self, market_data: List[MarketData]) -> float:
        """Calculate price volatility"""
        if len(market_data) < 20:
            return 0.02  # Default 2%
        
        prices = [d.close for d in market_data[-20:]]
        returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
        
        return np.std(returns) if returns else 0.02

    def _calculate_stop_loss(self, entry_price: float, direction: SignalDirection, volatility: float) -> float:
        """Calculate stop loss level"""
        
        # Use 2x volatility for stop distance
        stop_distance = entry_price * (volatility * 2)
        stop_distance = max(stop_distance, entry_price * 0.005)  # Minimum 0.5%
        stop_distance = min(stop_distance, entry_price * 0.03)   # Maximum 3%
        
        if direction == SignalDirection.LONG:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def _calculate_targets(self, entry_price: float, stop_loss: float, 
                         direction: SignalDirection) -> Tuple[float, float]:
        """Calculate target levels"""
        
        risk = abs(entry_price - stop_loss)
        
        # Target 1: 2:1 risk/reward
        # Target 2: 3:1 risk/reward
        if direction == SignalDirection.LONG:
            target_1 = entry_price + (risk * 2)
            target_2 = entry_price + (risk * 3)
        else:
            target_1 = entry_price - (risk * 2)
            target_2 = entry_price - (risk * 3)
        
        return target_1, target_2

    def _calculate_signal_confidence(self, maritime_score: float, volume_score: float, 
                                   technical_score: float) -> float:
        """Calculate overall signal confidence"""
        
        # Weighted combination
        confidence = (
            maritime_score * 0.5 +      # Maritime events are primary
            volume_score * 0.3 +        # Volume confirmation important
            technical_score * 0.2       # Technical as supporting evidence
        )
        
        # Boost confidence if multiple factors align
        if maritime_score > 0.7 and volume_score > 0.6:
            confidence *= 1.1
        
        if maritime_score > 0.8 and volume_score > 0.7 and technical_score > 0.6:
            confidence *= 1.2
        
        return min(1.0, confidence)

    def _determine_signal_tier(self, maritime_score: float, volume_score: float, 
                             technical_score: float, confidence: float) -> SignalTier:
        """Determine signal tier based on criteria"""
        
        # Tier 1: Pure maritime signals with high confidence
        if (maritime_score >= 0.8 and confidence >= 0.85):
            return SignalTier.TIER_1
        
        # Tier 2: Maritime + Volume confirmation
        elif (maritime_score >= 0.6 and volume_score >= 0.6 and confidence >= 0.70):
            return SignalTier.TIER_2
        
        # Tier 3: Technical with maritime context
        else:
            return SignalTier.TIER_3

    def _generate_reason(self, events: List[MaritimeEvent], maritime_score: float, 
                        volume_score: float, technical_score: float) -> str:
        """Generate human-readable reason for the signal"""
        
        reasons = []
        
        # Maritime events
        event_types = [e.event_type for e in events]
        if "port_congestion" in event_types:
            congestion_events = [e for e in events if e.event_type == "port_congestion"]
            port_names = [e.location for e in congestion_events]
            reasons.append(f"Port congestion at {', '.join(port_names)}")
        
        if "vessel_clustering" in event_types:
            cluster_events = [e for e in events if e.event_type == "vessel_clustering"]
            reasons.append(f"Vessel clustering detected ({len(cluster_events)} clusters)")
        
        if "supply_pressure" in event_types:
            reasons.append("Supply/demand imbalance")
        
        # Volume confirmation
        if volume_score > 0.6:
            reasons.append("Volume breakout confirmation")
        
        # Technical support
        if technical_score > 0.6:
            reasons.append("Technical alignment")
        
        # Combine reasons
        if reasons:
            return " + ".join(reasons)
        else:
            return "Maritime signal analysis"

    def filter_signals_by_tier(self, signals: List[TradingSignal], max_tier: SignalTier) -> List[TradingSignal]:
        """Filter signals by maximum tier"""
        return [s for s in signals if s.tier.value <= max_tier.value]

    def get_daily_signal_summary(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """Get summary statistics for daily signals"""
        
        if not signals:
            return {"total": 0, "by_tier": {}, "by_symbol": {}, "avg_confidence": 0.0}
        
        by_tier = {}
        by_symbol = {}
        
        for signal in signals:
            # By tier
            tier_key = f"tier_{signal.tier.value}"
            by_tier[tier_key] = by_tier.get(tier_key, 0) + 1
            
            # By symbol
            by_symbol[signal.symbol] = by_symbol.get(signal.symbol, 0) + 1
        
        avg_confidence = np.mean([s.confidence_score for s in signals])
        
        return {
            "total": len(signals),
            "by_tier": by_tier,
            "by_symbol": by_symbol,
            "avg_confidence": avg_confidence,
            "long_signals": len([s for s in signals if s.direction == SignalDirection.LONG]),
            "short_signals": len([s for s in signals if s.direction == SignalDirection.SHORT])
        }