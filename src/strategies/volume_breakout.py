import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging
from ..core.data_fetcher import MarketData
from ..core.maritime_analyzer import MaritimeEvent
from ..core.signal_generator import TradingSignal, SignalDirection, SignalTier

logger = logging.getLogger(__name__)

class VolumeBreakoutStrategy:
    def __init__(self):
        # Volume analysis parameters
        self.volume_lookback = 20  # periods for baseline volume
        self.volume_spike_threshold = 1.5  # 50% above average
        self.volume_surge_threshold = 2.0  # 100% above average
        self.min_price_move = 0.002  # 0.2% minimum price movement
        
        # Time-based volume analysis
        self.session_hours = {
            "asian": (22, 6),     # 22:00-06:00 UTC
            "european": (6, 14),  # 06:00-14:00 UTC  
            "american": (14, 22)  # 14:00-22:00 UTC
        }
        
        logger.info("VolumeBreakoutStrategy initialized")

    def analyze_volume_breakout(self, symbol: str, market_data: List[MarketData],
                               volume_profile: Dict[str, Any],
                               maritime_context: List[MaritimeEvent] = None) -> Optional[TradingSignal]:
        """Analyze volume breakout patterns and generate signals"""
        
        if len(market_data) < self.volume_lookback + 5:
            logger.debug(f"Insufficient data for {symbol} volume analysis")
            return None
        
        # Calculate volume metrics
        volume_analysis = self._analyze_volume_patterns(market_data)
        
        if not volume_analysis["breakout_detected"]:
            return None
        
        # Analyze price action during volume breakout
        price_analysis = self._analyze_price_action(market_data, volume_analysis)
        
        # Volume profile analysis
        vp_analysis = self._analyze_volume_profile(market_data[-1].close, volume_profile)
        
        # Determine signal direction and strength
        signal_direction, signal_strength = self._determine_breakout_direction(
            volume_analysis, price_analysis, vp_analysis
        )
        
        if not signal_direction:
            return None
        
        # Calculate entry and exit levels
        current_price = market_data[-1].close
        entry_price, stop_loss, targets = self._calculate_breakout_levels(
            current_price, signal_direction, volume_analysis, price_analysis
        )
        
        # Calculate confidence with maritime context
        confidence = self._calculate_breakout_confidence(
            volume_analysis, price_analysis, vp_analysis, maritime_context
        )
        
        if confidence < 0.60:  # Minimum confidence for volume breakouts
            return None
        
        # Determine signal tier
        tier = self._determine_breakout_tier(volume_analysis, confidence, maritime_context)
        
        # Generate reason string
        reason = self._generate_breakout_reason(volume_analysis, price_analysis, vp_analysis)
        
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
            maritime_events=maritime_context or [],
            timestamp=datetime.now(),
            expiry_time=datetime.now() + timedelta(hours=12),  # Volume breakouts are shorter-term
            risk_amount=abs(entry_price - stop_loss),
            reward_ratio=abs(targets[0] - entry_price) / abs(entry_price - stop_loss),
            volume_confirmation=True,
            technical_score=price_analysis.get("technical_score", 0.7),
            maritime_score=self._calculate_maritime_score(maritime_context),
            volume_score=volume_analysis["strength"]
        )
        
        logger.info(f"Volume breakout signal generated: {symbol} {signal_direction.value}")
        return signal

    def _analyze_volume_patterns(self, market_data: List[MarketData]) -> Dict[str, Any]:
        """Analyze volume patterns for breakout detection"""
        
        volumes = [d.volume for d in market_data]
        prices = [d.close for d in market_data]
        
        # Baseline volume calculation
        baseline_volume = np.mean(volumes[-self.volume_lookback:-5])
        recent_volume = np.mean(volumes[-5:])
        current_volume = volumes[-1]
        
        # Volume ratios
        volume_ratio = recent_volume / baseline_volume if baseline_volume > 0 else 1.0
        current_ratio = current_volume / baseline_volume if baseline_volume > 0 else 1.0
        
        # Detect different types of volume events
        volume_spike = current_ratio > self.volume_spike_threshold
        volume_surge = current_ratio > self.volume_surge_threshold
        sustained_volume = volume_ratio > self.volume_spike_threshold
        
        # Volume trend analysis
        volume_trend = self._calculate_volume_trend(volumes[-10:])
        
        # Institutional accumulation detection
        accumulation_score = self._detect_accumulation(volumes, prices)
        
        # Time-based volume analysis
        session_analysis = self._analyze_session_volume(market_data[-24:] if len(market_data) >= 24 else market_data)
        
        breakout_detected = volume_spike or (sustained_volume and volume_trend > 0.3)
        
        strength = min(1.0, max(
            current_ratio / 3.0,  # Normalize to 3x as max
            volume_ratio / 2.5,   # Or 2.5x sustained
            accumulation_score
        ))
        
        return {
            "breakout_detected": breakout_detected,
            "strength": strength,
            "volume_ratio": volume_ratio,
            "current_ratio": current_ratio,
            "volume_spike": volume_spike,
            "volume_surge": volume_surge,
            "sustained_volume": sustained_volume,
            "volume_trend": volume_trend,
            "accumulation_score": accumulation_score,
            "session_analysis": session_analysis,
            "baseline_volume": baseline_volume,
            "current_volume": current_volume
        }

    def _analyze_price_action(self, market_data: List[MarketData], 
                            volume_analysis: Dict) -> Dict[str, Any]:
        """Analyze price action during volume events"""
        
        prices = [d.close for d in market_data]
        highs = [d.high for d in market_data]
        lows = [d.low for d in market_data]
        
        # Price movement during volume event
        price_change = (prices[-1] - prices[-5]) / prices[-5]
        
        # Price-volume correlation
        recent_prices = prices[-10:]
        recent_volumes = [d.volume for d in market_data[-10:]]
        
        correlation = 0.0
        try:
            correlation = np.corrcoef(recent_prices, recent_volumes)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0
        
        # Breakout strength (price movement with volume)
        if volume_analysis["volume_spike"] and abs(price_change) > self.min_price_move:
            breakout_strength = min(1.0, abs(price_change) / 0.02)  # Normalize to 2% move
        else:
            breakout_strength = 0.0
        
        # Support/resistance analysis
        sr_analysis = self._analyze_support_resistance(prices, highs, lows)
        
        # Momentum indicators
        momentum = self._calculate_momentum(prices)
        
        # Technical score combining factors
        technical_score = (
            (abs(correlation) * 0.3) +
            (breakout_strength * 0.4) +
            (momentum * 0.3)
        )
        
        return {
            "price_change": price_change,
            "correlation": correlation,
            "breakout_strength": breakout_strength,
            "support_resistance": sr_analysis,
            "momentum": momentum,
            "technical_score": min(1.0, technical_score)
        }

    def _analyze_volume_profile(self, current_price: float, 
                              volume_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume profile context"""
        
        if not volume_profile:
            return {"poc_distance": 0.0, "value_area_position": "unknown", "profile_score": 0.5}
        
        poc = volume_profile.get("poc", current_price)
        vah = volume_profile.get("vah", current_price + 1)
        val = volume_profile.get("val", current_price - 1)
        
        # Distance from Point of Control
        poc_distance = abs(current_price - poc) / poc if poc > 0 else 0.0
        
        # Value area position
        if val <= current_price <= vah:
            value_area_position = "inside"
            position_score = 0.5  # Neutral
        elif current_price > vah:
            value_area_position = "above"
            position_score = 0.8  # Bullish breakout
        else:
            value_area_position = "below"
            position_score = 0.2  # Bearish breakout
        
        # Profile score based on position and distance
        if value_area_position != "inside":
            profile_score = 0.7 + min(0.3, poc_distance * 10)  # Higher score for larger moves
        else:
            profile_score = 0.5
        
        return {
            "poc": poc,
            "vah": vah,
            "val": val,
            "poc_distance": poc_distance,
            "value_area_position": value_area_position,
            "position_score": position_score,
            "profile_score": profile_score
        }

    def _calculate_volume_trend(self, volumes: List[int]) -> float:
        """Calculate volume trend over recent periods"""
        
        if len(volumes) < 5:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(volumes))
        try:
            slope, _ = np.polyfit(x, volumes, 1)
            # Normalize slope
            avg_volume = np.mean(volumes)
            trend = slope / avg_volume if avg_volume > 0 else 0.0
            return max(-1.0, min(1.0, trend * 10))  # Scale and clamp
        except:
            return 0.0

    def _detect_accumulation(self, volumes: List[int], prices: List[float]) -> float:
        """Detect institutional accumulation patterns"""
        
        if len(volumes) < 10 or len(prices) < 10:
            return 0.0
        
        # Look for sustained higher volume with minimal price movement
        recent_volumes = volumes[-10:]
        recent_prices = prices[-10:]
        
        avg_volume = np.mean(recent_volumes)
        baseline_volume = np.mean(volumes[-20:-10]) if len(volumes) >= 20 else avg_volume
        
        volume_increase = avg_volume / baseline_volume if baseline_volume > 0 else 1.0
        
        # Price stability during volume increase
        price_volatility = np.std(recent_prices) / np.mean(recent_prices)
        
        # High volume, low volatility suggests accumulation
        if volume_increase > 1.3 and price_volatility < 0.02:  # Volume up 30%, volatility < 2%
            accumulation_score = min(1.0, (volume_increase - 1.0) / 1.0)  # 0-1 scale
        else:
            accumulation_score = 0.0
        
        return accumulation_score

    def _analyze_session_volume(self, market_data: List[MarketData]) -> Dict[str, float]:
        """Analyze volume patterns by trading session"""
        
        if not market_data:
            return {"asian": 0.0, "european": 0.0, "american": 0.0}
        
        session_volumes = {"asian": [], "european": [], "american": []}
        
        for data in market_data:
            hour = data.timestamp.hour
            
            # Determine session
            if 22 <= hour or hour < 6:
                session = "asian"
            elif 6 <= hour < 14:
                session = "european"
            else:
                session = "american"
            
            session_volumes[session].append(data.volume)
        
        # Calculate average volume per session
        session_averages = {}
        for session, volumes in session_volumes.items():
            session_averages[session] = np.mean(volumes) if volumes else 0.0
        
        return session_averages

    def _analyze_support_resistance(self, prices: List[float], highs: List[float], 
                                  lows: List[float]) -> Dict[str, float]:
        """Analyze support and resistance levels"""
        
        if len(prices) < 10:
            return {"support": 0.0, "resistance": 0.0, "breakout_potential": 0.0}
        
        current_price = prices[-1]
        
        # Simple S/R calculation using recent highs/lows
        recent_highs = highs[-10:]
        recent_lows = lows[-10:]
        
        resistance = max(recent_highs)
        support = min(recent_lows)
        
        # Breakout potential
        resistance_distance = (resistance - current_price) / current_price
        support_distance = (current_price - support) / current_price
        
        # Higher score if near breakout levels
        if resistance_distance < 0.01:  # Within 1% of resistance
            breakout_potential = 0.8
        elif support_distance < 0.01:  # Within 1% of support
            breakout_potential = 0.8
        else:
            breakout_potential = 0.3
        
        return {
            "support": support,
            "resistance": resistance,
            "breakout_potential": breakout_potential,
            "resistance_distance": resistance_distance,
            "support_distance": support_distance
        }

    def _calculate_momentum(self, prices: List[float]) -> float:
        """Calculate price momentum"""
        
        if len(prices) < 10:
            return 0.5
        
        # Rate of change
        roc = (prices[-1] - prices[-10]) / prices[-10]
        
        # RSI-like calculation
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, change) for change in changes[-14:]]
        losses = [abs(min(0, change)) for change in changes[-14:]]
        
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 0.01
        
        rs = avg_gain / avg_loss if avg_loss > 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # Combine ROC and RSI
        momentum_score = (abs(roc) * 5 + (abs(rsi - 50) / 50)) / 2
        return min(1.0, momentum_score)

    def _determine_breakout_direction(self, volume_analysis: Dict, price_analysis: Dict,
                                    vp_analysis: Dict) -> Tuple[Optional[SignalDirection], float]:
        """Determine breakout direction and strength"""
        
        price_change = price_analysis["price_change"]
        correlation = price_analysis["correlation"]
        vp_position = vp_analysis["value_area_position"]
        
        # Direction based on price movement and volume profile
        if price_change > self.min_price_move and correlation > 0:
            direction = SignalDirection.LONG
            strength = volume_analysis["strength"] * abs(price_change) / 0.02
        elif price_change < -self.min_price_move and correlation > 0:
            direction = SignalDirection.SHORT
            strength = volume_analysis["strength"] * abs(price_change) / 0.02
        elif vp_position == "above":
            direction = SignalDirection.LONG
            strength = volume_analysis["strength"] * 0.8
        elif vp_position == "below":
            direction = SignalDirection.SHORT
            strength = volume_analysis["strength"] * 0.8
        else:
            return None, 0.0
        
        # Minimum strength threshold
        if strength < 0.4:
            return None, 0.0
        
        return direction, min(1.0, strength)

    def _calculate_breakout_levels(self, current_price: float, direction: SignalDirection,
                                 volume_analysis: Dict, price_analysis: Dict) -> Tuple[float, float, List[float]]:
        """Calculate entry, stop loss, and target levels for breakout"""
        
        volatility = abs(price_analysis["price_change"]) * 2  # Estimate volatility
        volatility = max(0.005, min(0.03, volatility))  # Clamp between 0.5% and 3%
        
        # Entry price (immediate for volume breakouts)
        entry_price = current_price
        
        # Stop loss based on volume strength (tighter for stronger volume)
        volume_strength = volume_analysis["strength"]
        stop_multiplier = 1.5 - (volume_strength * 0.5)  # 1.0-1.5x volatility
        stop_distance = current_price * volatility * stop_multiplier
        
        if direction == SignalDirection.LONG:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance
        
        # Targets based on volume strength and breakout potential
        risk = abs(entry_price - stop_loss)
        
        # More aggressive targets for stronger volume breakouts
        target_1_multiplier = 2.0 + volume_strength  # 2-3x risk
        target_2_multiplier = 3.5 + (volume_strength * 1.5)  # 3.5-5x risk
        
        if direction == SignalDirection.LONG:
            target_1 = entry_price + (risk * target_1_multiplier)
            target_2 = entry_price + (risk * target_2_multiplier)
        else:
            target_1 = entry_price - (risk * target_1_multiplier)
            target_2 = entry_price - (risk * target_2_multiplier)
        
        return entry_price, stop_loss, [target_1, target_2]

    def _calculate_breakout_confidence(self, volume_analysis: Dict, price_analysis: Dict,
                                     vp_analysis: Dict, maritime_context: List[MaritimeEvent]) -> float:
        """Calculate confidence for volume breakout signal"""
        
        # Base confidence from volume strength
        base_confidence = 0.5 + (volume_analysis["strength"] * 0.3)
        
        # Price action confirmation
        if price_analysis["correlation"] > 0.5:
            base_confidence += 0.1
        
        if price_analysis["breakout_strength"] > 0.5:
            base_confidence += 0.1
        
        # Volume profile confirmation
        if vp_analysis["value_area_position"] != "inside":
            base_confidence += 0.05
        
        # Volume surge bonus
        if volume_analysis["volume_surge"]:
            base_confidence += 0.1
        
        # Maritime context boost
        if maritime_context:
            maritime_score = self._calculate_maritime_score(maritime_context)
            if maritime_score > 0.6:
                base_confidence += 0.05
        
        # Accumulation bonus
        if volume_analysis["accumulation_score"] > 0.5:
            base_confidence += 0.05
        
        return min(1.0, max(0.0, base_confidence))

    def _determine_breakout_tier(self, volume_analysis: Dict, confidence: float,
                               maritime_context: List[MaritimeEvent]) -> SignalTier:
        """Determine signal tier for volume breakout"""
        
        # Volume breakouts are typically Tier 2 or 3
        maritime_score = self._calculate_maritime_score(maritime_context)
        
        # Tier 2: Strong volume with maritime confirmation
        if (volume_analysis["volume_surge"] and confidence >= 0.75 and maritime_score > 0.6):
            return SignalTier.TIER_2
        
        # Tier 2: Exceptional volume without maritime
        elif volume_analysis["current_ratio"] > 3.0 and confidence >= 0.8:
            return SignalTier.TIER_2
        
        # Otherwise Tier 3
        else:
            return SignalTier.TIER_3

    def _calculate_maritime_score(self, maritime_context: List[MaritimeEvent]) -> float:
        """Calculate maritime context score"""
        
        if not maritime_context:
            return 0.0
        
        scores = []
        for event in maritime_context:
            score = event.severity * event.confidence_level
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0

    def _generate_breakout_reason(self, volume_analysis: Dict, price_analysis: Dict,
                                vp_analysis: Dict) -> str:
        """Generate reason string for volume breakout signal"""
        
        reason_parts = []
        
        # Volume description
        if volume_analysis["volume_surge"]:
            reason_parts.append(f"Volume surge ({volume_analysis['current_ratio']:.1f}x normal)")
        elif volume_analysis["volume_spike"]:
            reason_parts.append(f"Volume spike ({volume_analysis['current_ratio']:.1f}x normal)")
        else:
            reason_parts.append(f"Volume breakout ({volume_analysis['volume_ratio']:.1f}x sustained)")
        
        # Price action
        if abs(price_analysis["price_change"]) > 0.01:
            direction = "up" if price_analysis["price_change"] > 0 else "down"
            reason_parts.append(f"Price moving {direction} {abs(price_analysis['price_change'])*100:.1f}%")
        
        # Volume profile context
        if vp_analysis["value_area_position"] == "above":
            reason_parts.append("Above value area")
        elif vp_analysis["value_area_position"] == "below":
            reason_parts.append("Below value area")
        
        # Accumulation
        if volume_analysis["accumulation_score"] > 0.5:
            reason_parts.append("Institutional accumulation detected")
        
        return " + ".join(reason_parts)