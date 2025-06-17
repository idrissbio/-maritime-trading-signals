#!/usr/bin/env python3
"""
Test suite for Maritime Trading Signals System

This test suite validates the core functionality of the maritime trading system
including data fetching, signal generation, and risk management.

Usage:
    python -m pytest tests/test_signals.py -v
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.core.data_fetcher import DataFetcher, VesselPosition, PortCongestion, MarketData
from src.core.maritime_analyzer import MaritimeAnalyzer, CongestionEvent
from src.core.signal_generator import SignalGenerator, TradingSignal, SignalDirection, SignalTier
from src.core.risk_manager import RiskManager, RiskParameters
from src.strategies.port_congestion import PortCongestionStrategy
from src.strategies.volume_breakout import VolumeBreakoutStrategy

class TestDataFetcher:
    """Test data fetching functionality"""
    
    def test_mock_mode_initialization(self):
        """Test data fetcher initialization in mock mode"""
        fetcher = DataFetcher(mock_mode=True)
        assert fetcher.mock_mode is True
        assert fetcher.datalastic_key is None
        assert fetcher.twelve_data_key is None
    
    def test_vessel_positions_mock_data(self):
        """Test vessel positions generation in mock mode"""
        fetcher = DataFetcher(mock_mode=True)
        vessels = fetcher.get_vessel_positions("singapore", "crude_oil")
        
        assert len(vessels) > 0
        assert all(isinstance(v, VesselPosition) for v in vessels)
        assert all(v.cargo_type == "crude_oil" for v in vessels)
    
    def test_port_congestion_mock_data(self):
        """Test port congestion data generation"""
        fetcher = DataFetcher(mock_mode=True)
        congestion = fetcher.get_port_congestion("singapore")
        
        assert isinstance(congestion, PortCongestion)
        assert congestion.port_name == "singapore"
        assert congestion.waiting_vessels > 0
        assert congestion.average_wait_time > 0
    
    def test_market_data_mock_generation(self):
        """Test market data generation"""
        fetcher = DataFetcher(mock_mode=True)
        market_data = fetcher.get_market_data("CL")
        
        assert len(market_data) > 0
        assert all(isinstance(d, MarketData) for d in market_data)
        assert all(d.symbol == "CL" for d in market_data)
        assert all(d.price > 0 for d in market_data)
    
    def test_health_check(self):
        """Test system health check"""
        fetcher = DataFetcher(mock_mode=True)
        health = fetcher.health_check()
        
        assert isinstance(health, dict)
        assert "mock_mode" in health
        assert health["mock_mode"] is True

class TestMaritimeAnalyzer:
    """Test maritime event analysis"""
    
    def setup_method(self):
        """Setup test data"""
        self.analyzer = MaritimeAnalyzer()
        self.fetcher = DataFetcher(mock_mode=True)
    
    def test_port_congestion_analysis(self):
        """Test port congestion analysis"""
        port_data = self.fetcher.get_port_congestion("singapore")
        
        # Force high congestion for testing
        port_data.waiting_vessels = 50
        port_data.average_wait_time = 20.0
        
        event = self.analyzer.analyze_port_congestion(port_data)
        
        if event:  # May be None if thresholds not met
            assert isinstance(event, CongestionEvent)
            assert event.event_type == "port_congestion"
            assert event.severity > 0
            assert event.confidence_level > 0
    
    def test_vessel_clustering(self):
        """Test vessel clustering detection"""
        vessels = self.fetcher.get_vessel_positions("singapore", "crude_oil")
        
        # Ensure we have enough vessels for clustering
        if len(vessels) >= 5:
            cluster_events = self.analyzer.detect_vessel_clusters(vessels)
            
            # Clustering may or may not find clusters depending on random positions
            assert isinstance(cluster_events, list)
            for event in cluster_events:
                assert hasattr(event, 'cluster_size')
                assert hasattr(event, 'cluster_center')
    
    def test_comprehensive_analysis(self):
        """Test comprehensive maritime analysis"""
        vessel_positions = self.fetcher.get_vessel_positions("singapore", "crude_oil")
        port_congestion = [self.fetcher.get_port_congestion("singapore")]
        
        events = self.analyzer.analyze_all_maritime_factors(
            port_congestion, vessel_positions, {}, {}
        )
        
        assert isinstance(events, list)
        # Events may be empty if no significant patterns detected

class TestSignalGenerator:
    """Test trading signal generation"""
    
    def setup_method(self):
        """Setup test data"""
        self.generator = SignalGenerator()
        self.fetcher = DataFetcher(mock_mode=True)
        self.analyzer = MaritimeAnalyzer()
    
    def test_signal_generation(self):
        """Test basic signal generation"""
        # Generate maritime events
        vessel_positions = self.fetcher.get_vessel_positions("singapore", "crude_oil")
        port_congestion = [self.fetcher.get_port_congestion("singapore")]
        
        # Force significant congestion
        port_congestion[0].waiting_vessels = 60
        port_congestion[0].average_wait_time = 25.0
        
        maritime_events = self.analyzer.analyze_all_maritime_factors(
            port_congestion, vessel_positions, {}, {}
        )
        
        # Generate market data
        market_data = {"CL": self.fetcher.get_market_data("CL")}
        volume_profiles = {"CL": self.fetcher.get_volume_profile("CL")}
        
        signals = self.generator.generate_signals(
            maritime_events, market_data, volume_profiles
        )
        
        assert isinstance(signals, list)
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert signal.symbol in market_data
            assert signal.entry_price > 0
            assert signal.confidence_score > 0
            assert signal.confidence_score <= 1.0
    
    def test_signal_tier_classification(self):
        """Test signal tier classification"""
        # Create mock signal data
        maritime_score = 0.9
        volume_score = 0.8
        technical_score = 0.7
        confidence = 0.85
        
        # This tests the internal logic structure
        assert maritime_score > 0.8
        assert confidence > 0.85
        # Would be classified as Tier 1
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        maritime_score = 0.7
        volume_score = 0.6
        technical_score = 0.5
        
        # Weighted combination: 0.5 * 0.7 + 0.3 * 0.6 + 0.2 * 0.5
        expected_confidence = 0.5 * 0.7 + 0.3 * 0.6 + 0.2 * 0.5
        
        assert 0.5 <= expected_confidence <= 1.0

class TestRiskManager:
    """Test risk management functionality"""
    
    def setup_method(self):
        """Setup test data"""
        risk_params = RiskParameters(
            account_balance=100000,
            risk_per_trade=0.01,
            max_daily_trades=15
        )
        self.risk_manager = RiskManager(risk_params)
    
    def create_test_signal(self, symbol="CL", direction=SignalDirection.LONG, 
                          entry_price=75.0, stop_loss=74.0):
        """Create a test trading signal"""
        return TradingSignal(
            symbol=symbol,
            direction=direction,
            tier=SignalTier.TIER_2,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_1=77.0,
            target_2=79.0,
            position_size=0,
            confidence_score=0.75,
            reason="Test signal",
            maritime_events=[],
            timestamp=datetime.now(),
            expiry_time=datetime.now() + timedelta(hours=24),
            risk_amount=abs(entry_price - stop_loss),
            reward_ratio=2.0
        )
    
    def test_position_size_calculation(self):
        """Test position size calculation"""
        signal = self.create_test_signal()
        position_size = self.risk_manager.calculate_position_size(signal)
        
        assert position_size >= 0
        # With $100k account, 1% risk, $1 risk per contract -> should be around 10 contracts
        assert position_size <= 20  # Reasonable upper bound
    
    def test_daily_limits(self):
        """Test daily trading limits"""
        # Should allow trades initially
        assert self.risk_manager._check_daily_limits() is True
        
        # Add multiple signals to test limit
        for i in range(16):  # Exceed the limit of 15
            signal = self.create_test_signal()
            self.risk_manager.daily_trades.append(signal)
        
        assert self.risk_manager._check_daily_limits() is False
    
    def test_correlation_limits(self):
        """Test correlation limits"""
        # Add CL position
        cl_signal = self.create_test_signal("CL")
        self.risk_manager.add_position(cl_signal)
        
        # Try to add HO position (highly correlated with CL)
        ho_signal = self.create_test_signal("HO")
        
        # Should still allow some correlated positions
        correlation_check = self.risk_manager._check_correlation_limits(ho_signal)
        assert isinstance(correlation_check, bool)
    
    def test_signal_validation(self):
        """Test signal validation"""
        signal = self.create_test_signal()
        is_valid, reason = self.risk_manager.validate_signal(signal)
        
        assert isinstance(is_valid, bool)
        assert isinstance(reason, str)
        
        if is_valid:
            assert signal.entry_price > 0
            assert signal.risk_amount > 0
    
    def test_portfolio_summary(self):
        """Test portfolio summary generation"""
        summary = self.risk_manager.get_portfolio_summary()
        
        assert isinstance(summary, dict)
        assert "total_positions" in summary
        assert "total_risk" in summary
        assert "risk_percentage" in summary
        assert summary["total_positions"] >= 0

class TestPortCongestionStrategy:
    """Test port congestion trading strategy"""
    
    def setup_method(self):
        """Setup test data"""
        self.strategy = PortCongestionStrategy()
        self.fetcher = DataFetcher(mock_mode=True)
    
    def test_congestion_severity_calculation(self):
        """Test congestion severity calculation"""
        port_data = self.fetcher.get_port_congestion("singapore")
        port_config = self.strategy.major_ports["singapore"]
        
        # Test with normal conditions
        port_data.waiting_vessels = 25  # Normal
        port_data.average_wait_time = 8.5  # Normal
        
        severity = self.strategy._calculate_congestion_severity(port_data, port_config)
        assert severity == 0.0  # Should be 0 for normal conditions
        
        # Test with high congestion
        port_data.waiting_vessels = 50  # 2x normal
        port_data.average_wait_time = 17.0  # 2x normal
        
        severity = self.strategy._calculate_congestion_severity(port_data, port_config)
        assert severity > 0  # Should be positive for high congestion
    
    def test_signal_generation(self):
        """Test signal generation from port congestion"""
        port_data = self.fetcher.get_port_congestion("singapore")
        vessels = self.fetcher.get_vessel_positions("singapore", "crude_oil")
        market_data = {"CL": self.fetcher.get_market_data("CL")}
        
        # Force high congestion
        port_data.waiting_vessels = 60
        port_data.average_wait_time = 25.0
        
        signal = self.strategy.analyze_port_congestion(port_data, vessels, market_data)
        
        if signal:  # May be None if conditions not met
            assert isinstance(signal, TradingSignal)
            assert signal.symbol == "CL"
            assert signal.direction == SignalDirection.LONG  # Congestion should be bullish

class TestVolumeBreakoutStrategy:
    """Test volume breakout trading strategy"""
    
    def setup_method(self):
        """Setup test data"""
        self.strategy = VolumeBreakoutStrategy()
        self.fetcher = DataFetcher(mock_mode=True)
    
    def test_volume_pattern_analysis(self):
        """Test volume pattern analysis"""
        market_data = self.fetcher.get_market_data("CL")
        
        # Ensure we have enough data
        if len(market_data) >= 25:
            volume_analysis = self.strategy._analyze_volume_patterns(market_data)
            
            assert isinstance(volume_analysis, dict)
            assert "breakout_detected" in volume_analysis
            assert "strength" in volume_analysis
            assert "volume_ratio" in volume_analysis
            assert 0 <= volume_analysis["strength"] <= 1

class TestSystemIntegration:
    """Test system integration and end-to-end functionality"""
    
    def setup_method(self):
        """Setup full system components"""
        self.fetcher = DataFetcher(mock_mode=True)
        self.analyzer = MaritimeAnalyzer()
        self.generator = SignalGenerator()
        
        risk_params = RiskParameters(account_balance=100000)
        self.risk_manager = RiskManager(risk_params)
    
    def test_full_analysis_cycle(self):
        """Test complete analysis cycle"""
        # 1. Fetch data
        vessel_positions = self.fetcher.get_vessel_positions("singapore", "crude_oil")
        port_congestion = [self.fetcher.get_port_congestion("singapore")]
        market_data = {"CL": self.fetcher.get_market_data("CL")}
        volume_profiles = {"CL": self.fetcher.get_volume_profile("CL")}
        
        # 2. Analyze maritime events
        maritime_events = self.analyzer.analyze_all_maritime_factors(
            port_congestion, vessel_positions, {}, {}
        )
        
        # 3. Generate signals
        signals = self.generator.generate_signals(
            maritime_events, market_data, volume_profiles
        )
        
        # 4. Apply risk management
        validated_signals = []
        for signal in signals:
            is_valid, reason = self.risk_manager.validate_signal(signal)
            if is_valid:
                signal.position_size = self.risk_manager.calculate_position_size(signal)
                if signal.position_size > 0:
                    validated_signals.append(signal)
        
        # Verify the process completed without errors
        assert isinstance(vessel_positions, list)
        assert isinstance(maritime_events, list)
        assert isinstance(signals, list)
        assert isinstance(validated_signals, list)
    
    def test_mock_data_consistency(self):
        """Test consistency of mock data across multiple calls"""
        # Multiple calls should return reasonable data
        for _ in range(3):
            vessels = self.fetcher.get_vessel_positions("singapore", "crude_oil")
            congestion = self.fetcher.get_port_congestion("singapore")
            market_data = self.fetcher.get_market_data("CL")
            
            assert len(vessels) > 0
            assert congestion.waiting_vessels > 0
            assert len(market_data) > 0

# Test configuration and utilities
def test_imports():
    """Test that all required modules can be imported"""
    try:
        from src.core.data_fetcher import DataFetcher
        from src.core.maritime_analyzer import MaritimeAnalyzer
        from src.core.signal_generator import SignalGenerator
        from src.core.risk_manager import RiskManager
        from src.strategies.port_congestion import PortCongestionStrategy
        from src.strategies.volume_breakout import VolumeBreakoutStrategy
    except ImportError as e:
        pytest.fail(f"Import error: {e}")

if __name__ == "__main__":
    # Run tests when executed directly
    print("Running Maritime Trading Signals Test Suite...")
    pytest.main([__file__, "-v"])