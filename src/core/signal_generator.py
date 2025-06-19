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
        
        # Enhanced symbol mappings for expanded commodity coverage
        self.commodity_symbols = {
            "crude_oil": "CL",
            "natural_gas": "NG", 
            "lng": "NG",
            "gasoline": "RB",
            "reformulated_gasoline": "RB",
            "rbob": "RB",
            "heating_oil": "HO",
            "diesel": "HO",
            "ulsd": "HO",
            "jet_fuel": "CL",  # Jet fuel typically tracks crude oil
            "gold": "GC",
            "silver": "SI",
            "copper": "HG"
        }
        
        # LNG terminal to Natural Gas futures mapping
        self.lng_terminals = {
            "sabine_pass": {
                "primary_symbol": "NG",
                "capacity_multiplier": 2.2,  # Largest US LNG export terminal
                "impact_radius_nm": 50,
                "notes": "Largest US LNG export terminal - major NG price impact"
            },
            "freeport_lng": {
                "primary_symbol": "NG",
                "capacity_multiplier": 1.8,
                "impact_radius_nm": 40,
                "notes": "Major US export facility with global impact"
            },
            "cameron_lng": {
                "primary_symbol": "NG",
                "capacity_multiplier": 1.6,
                "impact_radius_nm": 35,
                "notes": "Gulf Coast export hub"
            }
        }
        
        # Commodity-specific volatility and characteristics
        self.commodity_characteristics = {
            "CL": {"base_volatility": 0.025, "tick_size": 0.01, "contract_size": 1000},
            "NG": {"base_volatility": 0.045, "tick_size": 0.001, "contract_size": 10000},
            "RB": {"base_volatility": 0.030, "tick_size": 0.0001, "contract_size": 42000},
            "HO": {"base_volatility": 0.028, "tick_size": 0.0001, "contract_size": 42000},
            "GC": {"base_volatility": 0.020, "tick_size": 0.10, "contract_size": 100},
            "SI": {"base_volatility": 0.035, "tick_size": 0.005, "contract_size": 5000},
            "HG": {"base_volatility": 0.032, "tick_size": 0.0005, "contract_size": 25000}
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
        
        logger.info(f"SignalGenerator initialized with {len(self.commodity_symbols)} commodity mappings")
        logger.info(f"LNG terminals configured: {list(self.lng_terminals.keys())}")

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
            
            # Calculate composite maritime score with LNG terminal enhancement
            maritime_score = self._calculate_maritime_score(events, symbol)
            
            # Calculate volume score
            volume_score = self._calculate_volume_score(symbol_market_data, volume_profile)
            
            # Calculate technical score with commodity-specific parameters
            technical_score = self._calculate_technical_score(symbol_market_data, symbol)
            
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
        
        # Phase 3: Apply multi-commodity signal correlation validation
        validated_signals = self._validate_cross_commodity_signals(signals)
        
        logger.info(f"Generated {len(validated_signals)} validated trading signals (from {len(signals)} initial)")
        return validated_signals

    def _calculate_maritime_score(self, events: List[MaritimeEvent], symbol: str = None) -> float:
        """Calculate composite maritime score from events with LNG terminal enhancement"""
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
            
            # LNG terminal specific enhancements for Natural Gas futures
            if symbol == "NG" and hasattr(event, 'location'):
                if event.location in self.lng_terminals:
                    lng_info = self.lng_terminals[event.location]
                    capacity_multiplier = lng_info["capacity_multiplier"]
                    score *= capacity_multiplier
                    logger.debug(f"Applied LNG terminal multiplier {capacity_multiplier}x for {event.location}")
            
            # Enhanced scoring for refined products (RB, HO) from crude oil events
            elif symbol in ["RB", "HO"] and event.affected_commodity == "crude_oil":
                # Refined products get boosted score from crude oil maritime events
                refinement_multiplier = 1.3 if symbol == "RB" else 1.2  # Gasoline more sensitive
                score *= refinement_multiplier
                logger.debug(f"Applied refinement multiplier {refinement_multiplier}x for {symbol}")
            
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

    def _calculate_technical_score(self, market_data: List[MarketData], symbol: str = None) -> float:
        """Calculate technical analysis score with commodity-specific parameters"""
        if len(market_data) < 50:
            return 0.0
        
        prices = [d.close for d in market_data]
        volumes = [d.volume for d in market_data]
        
        # Get commodity-specific characteristics
        commodity_chars = self.commodity_characteristics.get(symbol, {
            "base_volatility": 0.025, "tick_size": 0.01, "contract_size": 1000
        })
        
        # Moving averages (adjust periods based on commodity volatility)
        short_period = 20 if commodity_chars["base_volatility"] < 0.03 else 15
        long_period = 50 if commodity_chars["base_volatility"] < 0.03 else 40
        
        ma_short = np.mean(prices[-short_period:])
        ma_long = np.mean(prices[-long_period:]) if len(prices) >= long_period else np.mean(prices)
        current_price = prices[-1]
        
        # Trend score with commodity-specific sensitivity
        trend_score = 0.0
        if current_price > ma_short > ma_long:
            trend_score = 0.8  # Strong uptrend
        elif current_price > ma_short:
            trend_score = 0.6  # Mild uptrend
        elif current_price < ma_short < ma_long:
            trend_score = 0.2  # Downtrend
        else:
            trend_score = 0.4  # Sideways
        
        # Volatility-adjusted momentum
        momentum_score = self._calculate_momentum_score(prices)
        
        # Support/Resistance with tick size consideration
        sr_score = self._calculate_support_resistance_score(prices, commodity_chars["tick_size"])
        
        # Commodity-specific weighting
        if symbol == "NG":
            # Natural gas is more momentum-driven
            technical_score = (trend_score * 0.3 + momentum_score * 0.5 + sr_score * 0.2)
        elif symbol in ["RB", "HO"]:
            # Refined products more trend-following
            technical_score = (trend_score * 0.5 + momentum_score * 0.3 + sr_score * 0.2)
        else:
            # Default weighting
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

    def _calculate_support_resistance_score(self, prices: List[float], tick_size: float = 0.01) -> float:
        """Calculate support/resistance score with tick size consideration"""
        if len(prices) < 20:
            return 0.5
        
        current_price = prices[-1]
        recent_high = max(prices[-20:])
        recent_low = min(prices[-20:])
        
        # Distance from recent high/low
        high_distance = (recent_high - current_price) / recent_high
        low_distance = (current_price - recent_low) / current_price
        
        # Adjust thresholds based on tick size and price level
        base_threshold = 0.02
        tick_adjustment = (tick_size * 10) / current_price  # Relative tick size impact
        threshold = max(base_threshold, tick_adjustment * 2)
        
        # Score based on position with tick-adjusted sensitivity
        if low_distance < threshold:  # Near support
            return 0.8
        elif high_distance < threshold:  # Near resistance
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

    def _validate_cross_commodity_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Phase 3: Multi-commodity signal correlation validation and enhancement"""
        if len(signals) < 2:
            return signals
        
        validated_signals = []
        
        # Define commodity correlation groups
        oil_complex = ["CL", "RB", "HO"]  # Oil, Gasoline, Heating Oil
        metals_group = ["GC", "SI", "HG"]  # Gold, Silver, Copper
        energy_group = ["CL", "NG", "RB", "HO"]  # All energy commodities
        
        for signal in signals:
            enhanced_signal = signal
            original_confidence = signal.confidence_score
            
            # Find correlated signals in the same direction
            correlated_signals = self._find_correlated_signals(signal, signals, oil_complex, energy_group)
            
            if len(correlated_signals) > 0:
                # Boost confidence for correlated signals
                correlation_boost = min(0.15, len(correlated_signals) * 0.05)  # Max 15% boost
                enhanced_signal.confidence_score = min(1.0, original_confidence + correlation_boost)
                
                # Update tier if confidence improvement is significant
                if enhanced_signal.confidence_score >= 0.85 and signal.tier != SignalTier.TIER_1:
                    if signal.maritime_score > 0.7:  # Must have strong maritime component
                        enhanced_signal.tier = SignalTier.TIER_1
                        enhanced_signal.reason += f" [UPGRADED: Cross-commodity correlation with {len(correlated_signals)} signals]"
                        logger.info(f"Upgraded {signal.symbol} to Tier 1 due to cross-commodity correlation")
                
                enhanced_signal.reason += f" [CORRELATED: {len(correlated_signals)} supporting signals]"
                logger.debug(f"Enhanced {signal.symbol} confidence: {original_confidence:.2%} → {enhanced_signal.confidence_score:.2%}")
            
            # Validate against conflicting signals (same commodity group, opposite direction)
            conflicting_signals = self._find_conflicting_signals(signal, signals, oil_complex, energy_group)
            
            if len(conflicting_signals) > 0:
                # Reduce confidence for conflicting signals
                conflict_penalty = min(0.20, len(conflicting_signals) * 0.08)  # Max 20% penalty
                enhanced_signal.confidence_score = max(0.0, enhanced_signal.confidence_score - conflict_penalty)
                
                # Downgrade tier if confidence drops significantly
                if enhanced_signal.confidence_score < 0.60:
                    logger.warning(f"Rejected {signal.symbol} due to conflicting cross-commodity signals")
                    continue  # Skip this signal
                
                enhanced_signal.reason += f" [CONFLICT: {len(conflicting_signals)} opposing signals]"
                logger.debug(f"Reduced {signal.symbol} confidence due to conflicts: {original_confidence:.2%} → {enhanced_signal.confidence_score:.2%}")
            
            # Only include signals that still meet minimum confidence after validation
            if enhanced_signal.confidence_score >= self.min_confidence:
                validated_signals.append(enhanced_signal)
            else:
                logger.debug(f"Rejected {signal.symbol} - confidence {enhanced_signal.confidence_score:.2%} below threshold {self.min_confidence:.2%}")
        
        return validated_signals
    
    def _find_correlated_signals(self, target_signal: TradingSignal, all_signals: List[TradingSignal], 
                                oil_complex: List[str], energy_group: List[str]) -> List[TradingSignal]:
        """Find signals that should be positively correlated with the target signal"""
        correlated = []
        
        for signal in all_signals:
            if signal.symbol == target_signal.symbol:
                continue
                
            # Same direction is required for correlation
            if signal.direction != target_signal.direction:
                continue
            
            # Check if signals are in correlated commodity groups
            is_correlated = False
            
            # Oil complex correlation (CL, RB, HO)
            if target_signal.symbol in oil_complex and signal.symbol in oil_complex:
                is_correlated = True
            
            # Energy group broader correlation during supply disruptions
            elif (target_signal.symbol in energy_group and signal.symbol in energy_group and
                  target_signal.maritime_score > 0.6 and signal.maritime_score > 0.6):
                is_correlated = True
            
            # Natural gas and LNG terminals correlation
            elif ((target_signal.symbol == "NG" and signal.symbol in ["CL"]) or 
                  (target_signal.symbol in ["CL"] and signal.symbol == "NG")):
                # Only correlate if one involves LNG terminal events
                if any("lng" in event.location.lower() if hasattr(event, 'location') else False 
                       for event in target_signal.maritime_events + signal.maritime_events):
                    is_correlated = True
            
            if is_correlated and signal.confidence_score >= 0.60:  # Only consider decent quality signals
                correlated.append(signal)
        
        return correlated
    
    def _find_conflicting_signals(self, target_signal: TradingSignal, all_signals: List[TradingSignal],
                                 oil_complex: List[str], energy_group: List[str]) -> List[TradingSignal]:
        """Find signals that conflict with the target signal (same group, opposite direction)"""
        conflicting = []
        
        for signal in all_signals:
            if signal.symbol == target_signal.symbol:
                continue
                
            # Opposite direction creates conflict
            if signal.direction == target_signal.direction:
                continue
            
            # Check for conflicting commodity relationships
            is_conflicting = False
            
            # Oil complex internal conflicts
            if target_signal.symbol in oil_complex and signal.symbol in oil_complex:
                is_conflicting = True
            
            # Energy group conflicts during high confidence events
            elif (target_signal.symbol in energy_group and signal.symbol in energy_group and
                  target_signal.confidence_score > 0.7 and signal.confidence_score > 0.7):
                is_conflicting = True
            
            if is_conflicting:
                conflicting.append(signal)
        
        return conflicting