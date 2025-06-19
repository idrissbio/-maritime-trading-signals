#!/usr/bin/env python3
"""
Maritime Trading Signals MVP Deployment Script

This script prepares and runs the MVP version for live trading:
- Validates live API connections
- Runs system health checks
- Starts the trading system with real data
- Provides real-time monitoring
"""

import os
import sys
import logging
import time
from datetime import datetime
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv
from src.core.data_fetcher import DataFetcher
from src.core.maritime_analyzer import MaritimeAnalyzer
from src.core.signal_generator import SignalGenerator
from src.core.risk_manager import RiskManager, RiskParameters
from src.utils.logger import setup_logging

class MVPDeployment:
    def __init__(self):
        self.logger = setup_logging()
        load_dotenv()
        
        # Validate required environment variables
        self.validate_environment()
        
        # Initialize components
        self.initialize_components()
        
    def validate_environment(self):
        """Validate all required environment variables for MVP"""
        required_vars = {
            'TWELVEDATA_API_KEY': os.getenv('TWELVEDATA_API_KEY'),
            'DATALASTIC_API_KEY': os.getenv('DATALASTIC_API_KEY'),
            'MOCK_MODE': os.getenv('MOCK_MODE', 'false')
        }
        
        missing_vars = [var for var, value in required_vars.items() if not value]
        
        if missing_vars:
            self.logger.error(f"Missing required environment variables: {missing_vars}")
            raise Exception(f"MVP deployment failed - missing: {missing_vars}")
        
        if required_vars['MOCK_MODE'].lower() == 'true':
            self.logger.warning("MOCK_MODE is enabled - this is not suitable for MVP trading")
            
        self.logger.info("✅ Environment validation passed")
        
    def initialize_components(self):
        """Initialize all system components for MVP"""
        try:
            # Data fetcher with live APIs
            self.data_fetcher = DataFetcher(
                datalastic_key=os.getenv('DATALASTIC_API_KEY'),
                twelve_data_key=os.getenv('TWELVEDATA_API_KEY'),
                mock_mode=False  # Force live mode for MVP
            )
            
            # Maritime analyzer
            self.maritime_analyzer = MaritimeAnalyzer()
            
            # Signal generator
            self.signal_generator = SignalGenerator(
                risk_per_trade=float(os.getenv('RISK_PER_TRADE', 0.01)),
                min_confidence=float(os.getenv('MIN_SIGNAL_CONFIDENCE', 0.65))
            )
            
            # Risk manager
            risk_params = RiskParameters(
                account_balance=float(os.getenv('ACCOUNT_BALANCE', 100000)),
                risk_per_trade=float(os.getenv('RISK_PER_TRADE', 0.01)),
                max_daily_trades=int(os.getenv('MAX_DAILY_TRADES', 15)),
                max_positions=10,
                max_correlated_positions=3
            )
            self.risk_manager = RiskManager(risk_params)
            
            self.logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise
            
    def run_api_health_check(self):
        """Comprehensive API health check for MVP readiness"""
        self.logger.info("🔍 Running API health checks...")
        
        health_status = {}
        
        # Test 12Data API
        try:
            market_data = self.data_fetcher.get_market_data("CL", "5min")
            if len(market_data) > 0:
                health_status['12data'] = True
                self.logger.info("✅ 12Data API: Connected and functional")
            else:
                health_status['12data'] = False
                self.logger.warning("⚠️ 12Data API: Connected but no data returned")
        except Exception as e:
            health_status['12data'] = False
            self.logger.error(f"❌ 12Data API: Failed - {e}")
        
        # Test Datalastic API
        try:
            vessels = self.data_fetcher.get_vessel_positions("singapore", "crude_oil")
            if len(vessels) > 0:
                health_status['datalastic'] = True
                self.logger.info(f"✅ Datalastic API: Connected - {len(vessels)} vessels found")
            else:
                health_status['datalastic'] = False
                self.logger.warning("⚠️ Datalastic API: Connected but no vessels returned")
        except Exception as e:
            health_status['datalastic'] = False
            self.logger.error(f"❌ Datalastic API: Failed - {e}")
        
        # Overall health assessment
        if all(health_status.values()):
            self.logger.info("🎯 MVP READY: All APIs functional")
            return True
        else:
            failed_apis = [api for api, status in health_status.items() if not status]
            self.logger.error(f"🚨 MVP NOT READY: Failed APIs - {failed_apis}")
            return False
    
    def run_live_data_test(self):
        """Test live data generation and signal creation"""
        self.logger.info("🧪 Running live data test...")
        
        try:
            # Test priority port scanning
            priority_ports = self.data_fetcher.get_priority_scanning_order()[:3]  # Test top 3
            self.logger.info(f"Testing priority ports: {priority_ports}")
            
            # Collect live maritime data
            vessel_positions = []
            port_congestion_data = []
            
            for port in priority_ports:
                self.logger.info(f"Scanning {port}...")
                
                # Get vessels
                vessels = self.data_fetcher.get_vessel_positions(port, "all")
                vessel_positions.extend(vessels)
                
                # Get congestion
                congestion = self.data_fetcher.get_port_congestion(port)
                port_congestion_data.append(congestion)
                
                time.sleep(1)  # Rate limiting
            
            # Test market data for key symbols
            key_symbols = ["CL", "NG", "RB"]
            market_data = {}
            volume_profiles = {}
            
            for symbol in key_symbols:
                self.logger.info(f"Fetching market data for {symbol}...")
                market_data[symbol] = self.data_fetcher.get_market_data(symbol)
                volume_profiles[symbol] = self.data_fetcher.get_volume_profile(symbol)
            
            # Generate maritime events
            maritime_events = self.maritime_analyzer.analyze_all_maritime_factors(
                port_congestion_data, vessel_positions, {}, {}
            )
            
            # Apply chokepoint multipliers
            maritime_events = self.data_fetcher.apply_chokepoint_multipliers(maritime_events)
            
            # Generate signals
            signals = self.signal_generator.generate_signals(
                maritime_events, market_data, volume_profiles
            )
            
            # Test results
            self.logger.info(f"✅ Live test results:")
            self.logger.info(f"   - Vessels found: {len(vessel_positions)}")
            self.logger.info(f"   - Ports analyzed: {len(port_congestion_data)}")
            self.logger.info(f"   - Maritime events: {len(maritime_events)}")
            self.logger.info(f"   - Trading signals: {len(signals)}")
            
            if len(signals) > 0:
                for signal in signals[:3]:  # Show first 3 signals
                    self.logger.info(f"   - Signal: {signal.symbol} {signal.direction.value} "
                                   f"(Tier {signal.tier.value}, {signal.confidence_score:.1%} confidence)")
            
            return len(signals) > 0
            
        except Exception as e:
            self.logger.error(f"Live data test failed: {e}")
            return False
    
    def start_mvp_trading_system(self):
        """Start the MVP trading system"""
        self.logger.info("🚀 STARTING MVP TRADING SYSTEM")
        
        try:
            # Import and start the main system
            from main import MaritimeTradingSystem
            
            trading_system = MaritimeTradingSystem(
                config_path="config/settings.json",
                mock_mode=False,  # Force live mode
                test_mode=False
            )
            
            self.logger.info("💼 Trading system initialized - starting continuous operation...")
            
            # Run continuous trading cycle
            cycle_count = 0
            while True:
                cycle_count += 1
                self.logger.info(f"📊 Starting analysis cycle #{cycle_count}")
                
                signals, events = trading_system.run_analysis_cycle()
                
                self.logger.info(f"✅ Cycle #{cycle_count} completed: "
                               f"{len(signals)} signals, {len(events)} events")
                
                # Wait for next cycle
                update_interval = int(os.getenv('UPDATE_INTERVAL_MINUTES', 5))
                self.logger.info(f"⏱️ Waiting {update_interval} minutes for next cycle...")
                time.sleep(update_interval * 60)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Trading system stopped by user")
        except Exception as e:
            self.logger.error(f"🚨 Trading system error: {e}")
            raise

def main():
    """Main MVP deployment function"""
    print("🚢 Maritime Trading Signals - MVP Deployment")
    print("=" * 50)
    
    try:
        # Initialize MVP deployment
        mvp = MVPDeployment()
        
        # Run health checks
        if not mvp.run_api_health_check():
            print("❌ MVP deployment failed - API health check failed")
            return False
        
        # Run live data test
        if not mvp.run_live_data_test():
            print("❌ MVP deployment failed - live data test failed")
            return False
        
        print("✅ MVP READY FOR TRADING!")
        print("\nStarting live trading system...")
        
        # Start the trading system
        mvp.start_mvp_trading_system()
        
    except Exception as e:
        print(f"❌ MVP deployment failed: {e}")
        return False

if __name__ == "__main__":
    main()