import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from .signal_generator import TradingSignal, SignalTier, SignalDirection

logger = logging.getLogger(__name__)

@dataclass
class RiskParameters:
    account_balance: float
    risk_per_trade: float = 0.01  # 1% default
    max_daily_trades: int = 15
    max_positions: int = 10
    max_correlated_positions: int = 3
    max_tier3_allocation: float = 0.3  # 30% of risk budget for Tier 3
    position_size_limits: Dict[str, int] = None  # Symbol-specific limits

@dataclass
class Position:
    symbol: str
    direction: SignalDirection
    entry_price: float
    position_size: int
    stop_loss: float
    target_1: float
    target_2: float
    entry_time: datetime
    tier: SignalTier
    risk_amount: float
    
class RiskManager:
    def __init__(self, risk_params: RiskParameters):
        self.risk_params = risk_params
        self.active_positions: List[Position] = []
        self.daily_trades: List[TradingSignal] = []
        self.risk_budget_used = 0.0
        self.last_reset = datetime.now().date()
        
        # Symbol correlation matrix (simplified)
        self.correlations = {
            ("CL", "HO"): 0.85,  # Crude Oil - Heating Oil
            ("CL", "RB"): 0.80,  # Crude Oil - Gasoline
            ("NG", "HO"): 0.45,  # Natural Gas - Heating Oil
            ("GC", "SI"): 0.75,  # Gold - Silver
            ("CL", "NG"): 0.25,  # Crude Oil - Natural Gas
        }
        
        # Default position size limits by symbol
        self.default_limits = {
            "CL": 10,  # Crude Oil
            "NG": 5,   # Natural Gas
            "GC": 3,   # Gold
            "SI": 10,  # Silver
            "HG": 5,   # Copper
            "RB": 5,   # Gasoline
            "HO": 5    # Heating Oil
        }
        
        logger.info("RiskManager initialized")

    def calculate_position_size(self, signal: TradingSignal) -> int:
        """Calculate appropriate position size for a trading signal"""
        
        # Check daily limits first
        if not self._check_daily_limits():
            logger.warning("Daily trade limit reached")
            return 0
        
        # Check correlation limits
        if not self._check_correlation_limits(signal):
            logger.warning(f"Correlation limit exceeded for {signal.symbol}")
            return 0
        
        # Calculate base position size
        base_size = self._calculate_base_position_size(signal)
        
        if base_size <= 0:
            return 0
        
        # Apply tier-based adjustments
        tier_adjusted_size = self._apply_tier_adjustments(base_size, signal)
        
        # Apply symbol-specific limits
        final_size = self._apply_symbol_limits(tier_adjusted_size, signal.symbol)
        
        # Check total risk budget
        final_size = self._check_risk_budget(final_size, signal)
        
        logger.info(f"Position size calculated for {signal.symbol}: {final_size} contracts")
        return final_size

    def _calculate_base_position_size(self, signal: TradingSignal) -> int:
        """Calculate base position size using risk amount"""
        
        # Risk per trade in dollars
        risk_dollars = self.risk_params.account_balance * self.risk_params.risk_per_trade
        
        # Adjust risk based on confidence
        confidence_multiplier = 0.5 + (signal.confidence_score * 0.5)  # 0.5x to 1.0x
        adjusted_risk = risk_dollars * confidence_multiplier
        
        # Position size based on stop loss distance
        risk_per_contract = signal.risk_amount
        
        if risk_per_contract <= 0:
            logger.warning(f"Invalid risk per contract for {signal.symbol}: {risk_per_contract}")
            return 0
        
        # Calculate number of contracts
        position_size = int(adjusted_risk / risk_per_contract)
        
        return max(0, position_size)

    def _apply_tier_adjustments(self, base_size: int, signal: TradingSignal) -> int:
        """Apply tier-based position size adjustments"""
        
        tier_multipliers = {
            SignalTier.TIER_1: 1.0,   # Full size
            SignalTier.TIER_2: 0.8,   # 80% size
            SignalTier.TIER_3: 0.5    # 50% size
        }
        
        multiplier = tier_multipliers.get(signal.tier, 0.5)
        adjusted_size = int(base_size * multiplier)
        
        # Additional Tier 3 budget check
        if signal.tier == SignalTier.TIER_3:
            tier3_used = sum(pos.risk_amount for pos in self.active_positions 
                           if pos.tier == SignalTier.TIER_3)
            tier3_budget = self.risk_params.account_balance * self.risk_params.max_tier3_allocation
            
            if tier3_used >= tier3_budget:
                logger.warning("Tier 3 risk budget exhausted")
                return 0
        
        return adjusted_size

    def _apply_symbol_limits(self, size: int, symbol: str) -> int:
        """Apply symbol-specific position size limits"""
        
        # Get symbol limit
        symbol_limits = self.risk_params.position_size_limits or self.default_limits
        max_size = symbol_limits.get(symbol, 5)  # Default 5 contracts
        
        # Check current exposure to this symbol
        current_exposure = sum(pos.position_size for pos in self.active_positions 
                             if pos.symbol == symbol)
        
        available_size = max_size - current_exposure
        
        return min(size, max(0, available_size))

    def _check_risk_budget(self, size: int, signal: TradingSignal) -> int:
        """Check against total risk budget"""
        
        trade_risk = size * signal.risk_amount
        total_risk_budget = self.risk_params.account_balance * 0.1  # 10% max total risk
        
        if self.risk_budget_used + trade_risk > total_risk_budget:
            # Reduce size to fit budget
            available_budget = total_risk_budget - self.risk_budget_used
            max_affordable_size = int(available_budget / signal.risk_amount)
            
            return min(size, max(0, max_affordable_size))
        
        return size

    def _check_daily_limits(self) -> bool:
        """Check if daily trade limits are exceeded"""
        
        today = datetime.now().date()
        
        # Reset daily counters if new day
        if self.last_reset != today:
            self.daily_trades = []
            self.last_reset = today
        
        return len(self.daily_trades) < self.risk_params.max_daily_trades

    def _check_correlation_limits(self, signal: TradingSignal) -> bool:
        """Check correlation limits with existing positions"""
        
        correlated_positions = 0
        
        for position in self.active_positions:
            correlation = self._get_correlation(signal.symbol, position.symbol)
            
            # Count as correlated if correlation > 0.6 and same direction
            if correlation > 0.6 and signal.direction == position.direction:
                correlated_positions += 1
        
        return correlated_positions < self.risk_params.max_correlated_positions

    def _get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols"""
        
        if symbol1 == symbol2:
            return 1.0
        
        # Check both directions in correlation matrix
        key1 = (symbol1, symbol2)
        key2 = (symbol2, symbol1)
        
        return self.correlations.get(key1, self.correlations.get(key2, 0.0))

    def add_position(self, signal: TradingSignal) -> bool:
        """Add a new position to the portfolio"""
        
        if len(self.active_positions) >= self.risk_params.max_positions:
            logger.warning("Maximum positions limit reached")
            return False
        
        position_size = signal.position_size
        if position_size <= 0:
            logger.warning("Invalid position size")
            return False
        
        position = Position(
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            position_size=position_size,
            stop_loss=signal.stop_loss,
            target_1=signal.target_1,
            target_2=signal.target_2,
            entry_time=signal.timestamp,
            tier=signal.tier,
            risk_amount=signal.risk_amount * position_size
        )
        
        self.active_positions.append(position)
        self.daily_trades.append(signal)
        self.risk_budget_used += position.risk_amount
        
        logger.info(f"Position added: {signal.symbol} {position_size} contracts")
        return True

    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove a position from the portfolio"""
        
        for i, position in enumerate(self.active_positions):
            if position.symbol == symbol:
                removed_position = self.active_positions.pop(i)
                self.risk_budget_used -= removed_position.risk_amount
                logger.info(f"Position removed: {symbol}")
                return removed_position
        
        return None

    def update_stop_loss(self, symbol: str, new_stop: float) -> bool:
        """Update stop loss for an existing position"""
        
        for position in self.active_positions:
            if position.symbol == symbol:
                old_stop = position.stop_loss
                position.stop_loss = new_stop
                
                # Recalculate risk
                new_risk = abs(position.entry_price - new_stop) * position.position_size
                old_risk = position.risk_amount
                
                position.risk_amount = new_risk
                self.risk_budget_used += (new_risk - old_risk)
                
                logger.info(f"Stop loss updated for {symbol}: {old_stop} -> {new_stop}")
                return True
        
        return False

    def check_position_exits(self, current_prices: Dict[str, float]) -> List[Dict[str, any]]:
        """Check if any positions should be exited"""
        
        exits = []
        
        for position in self.active_positions:
            if position.symbol not in current_prices:
                continue
            
            current_price = current_prices[position.symbol]
            exit_reason = None
            exit_price = current_price
            
            # Check stop loss
            if position.direction == SignalDirection.LONG:
                if current_price <= position.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price = position.stop_loss
                elif current_price >= position.target_1:
                    exit_reason = "target_1"
                    exit_price = position.target_1
            else:  # SHORT
                if current_price >= position.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price = position.stop_loss
                elif current_price <= position.target_1:
                    exit_reason = "target_1"
                    exit_price = position.target_1
            
            # Check time-based exits (positions older than 7 days)
            if (datetime.now() - position.entry_time).days > 7:
                exit_reason = "time_exit"
                exit_price = current_price
            
            if exit_reason:
                exits.append({
                    "position": position,
                    "exit_reason": exit_reason,
                    "exit_price": exit_price,
                    "pnl": self._calculate_pnl(position, exit_price)
                })
        
        return exits

    def _calculate_pnl(self, position: Position, exit_price: float) -> float:
        """Calculate P&L for a position"""
        
        if position.direction == SignalDirection.LONG:
            pnl_per_contract = exit_price - position.entry_price
        else:
            pnl_per_contract = position.entry_price - exit_price
        
        return pnl_per_contract * position.position_size

    def get_portfolio_summary(self) -> Dict[str, any]:
        """Get portfolio risk summary"""
        
        if not self.active_positions:
            return {
                "total_positions": 0,
                "total_risk": 0.0,
                "risk_percentage": 0.0,
                "positions_by_tier": {},
                "positions_by_symbol": {},
                "daily_trades_used": len(self.daily_trades),
                "daily_trades_remaining": self.risk_params.max_daily_trades - len(self.daily_trades)
            }
        
        # Count by tier
        by_tier = {}
        for position in self.active_positions:
            tier_key = f"tier_{position.tier.value}"
            by_tier[tier_key] = by_tier.get(tier_key, 0) + 1
        
        # Count by symbol
        by_symbol = {}
        for position in self.active_positions:
            by_symbol[position.symbol] = by_symbol.get(position.symbol, 0) + position.position_size
        
        return {
            "total_positions": len(self.active_positions),
            "total_risk": self.risk_budget_used,
            "risk_percentage": (self.risk_budget_used / self.risk_params.account_balance) * 100,
            "positions_by_tier": by_tier,
            "positions_by_symbol": by_symbol,
            "daily_trades_used": len(self.daily_trades),
            "daily_trades_remaining": self.risk_params.max_daily_trades - len(self.daily_trades),
            "max_positions_used": f"{len(self.active_positions)}/{self.risk_params.max_positions}"
        }

    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Get correlation matrix for current positions"""
        
        symbols = list(set(pos.symbol for pos in self.active_positions))
        matrix = {}
        
        for symbol1 in symbols:
            matrix[symbol1] = {}
            for symbol2 in symbols:
                matrix[symbol1][symbol2] = self._get_correlation(symbol1, symbol2)
        
        return matrix

    def suggest_position_adjustments(self) -> List[str]:
        """Suggest portfolio adjustments"""
        
        suggestions = []
        
        # Check risk concentration
        risk_by_symbol = {}
        for position in self.active_positions:
            risk_by_symbol[position.symbol] = risk_by_symbol.get(position.symbol, 0) + position.risk_amount
        
        total_risk = sum(risk_by_symbol.values())
        
        for symbol, risk in risk_by_symbol.items():
            concentration = risk / total_risk if total_risk > 0 else 0
            if concentration > 0.4:  # 40% concentration in one symbol
                suggestions.append(f"High concentration in {symbol} ({concentration*100:.1f}%)")
        
        # Check correlation risks
        correlation_risks = []
        symbols = list(risk_by_symbol.keys())
        
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                correlation = self._get_correlation(symbol1, symbol2)
                if correlation > 0.7:
                    # Check if same direction
                    pos1_direction = next(pos.direction for pos in self.active_positions if pos.symbol == symbol1)
                    pos2_direction = next(pos.direction for pos in self.active_positions if pos.symbol == symbol2)
                    
                    if pos1_direction == pos2_direction:
                        correlation_risks.append(f"High correlation: {symbol1}-{symbol2} ({correlation:.2f})")
        
        suggestions.extend(correlation_risks)
        
        # Check Tier 3 allocation
        tier3_risk = sum(pos.risk_amount for pos in self.active_positions if pos.tier == SignalTier.TIER_3)
        tier3_percentage = tier3_risk / self.risk_params.account_balance
        
        if tier3_percentage > self.risk_params.max_tier3_allocation:
            suggestions.append(f"Tier 3 allocation exceeds limit ({tier3_percentage*100:.1f}%)")
        
        return suggestions

    def emergency_risk_reduction(self, reduction_percentage: float = 0.5) -> List[str]:
        """Emergency risk reduction - close most risky positions"""
        
        if not self.active_positions:
            return []
        
        # Sort positions by risk amount (highest first)
        sorted_positions = sorted(self.active_positions, key=lambda x: x.risk_amount, reverse=True)
        
        target_reduction = self.risk_budget_used * reduction_percentage
        current_reduction = 0.0
        positions_to_close = []
        
        for position in sorted_positions:
            if current_reduction >= target_reduction:
                break
            
            positions_to_close.append(position.symbol)
            current_reduction += position.risk_amount
        
        # Remove positions
        for symbol in positions_to_close:
            self.remove_position(symbol)
        
        logger.warning(f"Emergency risk reduction: Closed {len(positions_to_close)} positions")
        return positions_to_close

    def validate_signal(self, signal: TradingSignal) -> Tuple[bool, str]:
        """Validate if a signal can be traded based on risk rules"""
        
        # Check basic signal validity
        if signal.entry_price <= 0 or signal.risk_amount <= 0:
            return False, "Invalid signal parameters"
        
        # Check daily limits
        if not self._check_daily_limits():
            return False, "Daily trade limit exceeded"
        
        # Check position limits
        if len(self.active_positions) >= self.risk_params.max_positions:
            return False, "Maximum positions limit reached"
        
        # Check correlation limits
        if not self._check_correlation_limits(signal):
            return False, "Correlation limits exceeded"
        
        # Check if we can calculate a meaningful position size
        test_size = self.calculate_position_size(signal)
        if test_size <= 0:
            return False, "Cannot determine valid position size"
        
        return True, "Signal validated successfully"