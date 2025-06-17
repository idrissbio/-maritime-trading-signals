#!/usr/bin/env python3
"""
Maritime Trading Signals - Main Entry Point

This is the main application that coordinates all components:
- Data fetching from maritime and market APIs
- Maritime event analysis
- Signal generation
- Risk management
- Alert distribution

Usage:
    python main.py [options]

Options:
    --mock-mode          Use mock data instead of live APIs
    --config FILE        Path to configuration file
    --dashboard-only     Run only the Streamlit dashboard
    --test-mode          Run in test mode with sample data
    --log-level LEVEL    Set logging level (DEBUG, INFO, WARNING, ERROR)
"""

import os
import sys
import argparse
import logging
import time
import json
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Any
import asyncio
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dotenv import load_dotenv
    from src.core.data_fetcher import DataFetcher
    from src.core.maritime_analyzer import MaritimeAnalyzer
    from src.core.signal_generator import SignalGenerator, SignalDirection, SignalTier
    from src.core.risk_manager import RiskManager, RiskParameters
    from src.alerts.alert_manager import AlertManager, AlertConfig
    from src.alerts.email_sender import EmailSender
    from src.alerts.sms_sender import SMSSender
    from src.alerts.discord_webhook import DiscordWebhook
    from src.strategies.port_congestion import PortCongestionStrategy
    from src.strategies.volume_breakout import VolumeBreakoutStrategy
    from src.utils.logger import setup_logging
    from src.utils.config import load_config
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)

class MaritimeTradingSystem:
    """Main trading system coordinator"""
    
    def __init__(self, config_path: str = None, mock_mode: bool = False, test_mode: bool = False):
        self.config_path = config_path or "config/settings.json"
        self.mock_mode = mock_mode
        self.test_mode = test_mode
        self.running = False
        
        # Load environment variables
        load_dotenv()
        
        # Setup logging
        self.logger = setup_logging()
        self.logger.info("Initializing Maritime Trading System")
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.stats = {
            "signals_generated": 0,
            "events_detected": 0,
            "alerts_sent": 0,
            "start_time": datetime.now(),
            "last_update": None
        }

    def _load_configuration(self) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            config = load_config(self.config_path)
            self.logger.info(f"Configuration loaded from {self.config_path}")
        except Exception as e:
            self.logger.warning(f"Could not load config file: {e}, using defaults")
            config = self._get_default_config()
        
        # Override with environment variables where available
        if os.getenv("DATALASTIC_API_KEY"):
            config["data_sources"]["datalastic_api_key"] = os.getenv("DATALASTIC_API_KEY")
        if os.getenv("TWELVEDATA_API_KEY"):
            config["data_sources"]["twelve_data_api_key"] = os.getenv("TWELVEDATA_API_KEY")
        if os.getenv("MOCK_MODE"):
            config["data_sources"]["mock_mode"] = os.getenv("MOCK_MODE", "true").lower() == "true"
        
        return config

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "data_sources": {
                "mock_mode": os.getenv("MOCK_MODE", "true").lower() == "true",
                "datalastic_api_key": os.getenv("DATALASTIC_API_KEY"),
                "twelve_data_api_key": os.getenv("TWELVEDATA_API_KEY"),
                "update_interval_minutes": 5
            },
            "risk_management": {
                "account_balance": float(os.getenv("ACCOUNT_BALANCE", 100000)),
                "risk_per_trade": float(os.getenv("RISK_PER_TRADE", 0.01)),
                "max_daily_trades": int(os.getenv("MAX_DAILY_TRADES", 15)),
                "max_positions": 10,
                "max_correlated_positions": 3
            },
            "alerts": {
                "email_enabled": True,
                "sms_enabled": True,
                "discord_enabled": True,
                "min_tier_for_sms": 2
            },
            "trading": {
                "min_signal_confidence": float(os.getenv("MIN_SIGNAL_CONFIDENCE", 0.65)),
                "symbols": ["CL", "NG", "GC", "SI", "HG"],
                "ports": ["singapore", "houston", "rotterdam", "fujairah"]
            }
        }

    def _initialize_components(self):
        """Initialize all system components"""
        
        # Data fetcher
        self.data_fetcher = DataFetcher(
            datalastic_key=self.config["data_sources"].get("datalastic_api_key"),
            twelve_data_key=self.config["data_sources"].get("twelve_data_api_key"),
            mock_mode=self.config["data_sources"]["mock_mode"]
        )
        
        # Maritime analyzer
        self.maritime_analyzer = MaritimeAnalyzer()
        
        # Signal generator
        self.signal_generator = SignalGenerator(
            risk_per_trade=self.config["risk_management"]["risk_per_trade"],
            min_confidence=self.config["trading"]["min_signal_confidence"]
        )
        
        # Risk manager
        risk_params = RiskParameters(
            account_balance=self.config["risk_management"]["account_balance"],
            risk_per_trade=self.config["risk_management"]["risk_per_trade"],
            max_daily_trades=self.config["risk_management"]["max_daily_trades"],
            max_positions=self.config["risk_management"]["max_positions"],
            max_correlated_positions=self.config["risk_management"]["max_correlated_positions"]
        )
        self.risk_manager = RiskManager(risk_params)
        
        # Alert components
        self._initialize_alert_system()
        
        # Trading strategies
        self.port_congestion_strategy = PortCongestionStrategy()
        self.volume_breakout_strategy = VolumeBreakoutStrategy()
        
        self.logger.info("All components initialized successfully")

    def _initialize_alert_system(self):
        """Initialize alert system"""
        
        # Email sender
        email_sender = None
        if self.config["alerts"]["email_enabled"]:
            email_sender = EmailSender(
                api_key=os.getenv("EMAIL_API_KEY"),
                from_email=os.getenv("EMAIL_FROM")
            )
        
        # SMS sender
        sms_sender = None
        if self.config["alerts"]["sms_enabled"]:
            sms_sender = SMSSender(
                account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
                auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
                from_number=os.getenv("TWILIO_FROM_NUMBER")
            )
        
        # Discord webhook
        discord_webhook = None
        if self.config["alerts"]["discord_enabled"]:
            discord_webhook = DiscordWebhook(
                webhook_url=os.getenv("DISCORD_WEBHOOK_URL")
            )
        
        # Alert config
        alert_config = AlertConfig(
            email_enabled=self.config["alerts"]["email_enabled"],
            sms_enabled=self.config["alerts"]["sms_enabled"],
            discord_enabled=self.config["alerts"]["discord_enabled"],
            email_recipients=[os.getenv("EMAIL_TO")] if os.getenv("EMAIL_TO") else [],
            sms_recipients=[os.getenv("TWILIO_TO_NUMBER")] if os.getenv("TWILIO_TO_NUMBER") else [],
            min_tier_for_sms=self.config["alerts"]["min_tier_for_sms"]
        )
        
        # Alert manager
        self.alert_manager = AlertManager(
            config=alert_config,
            email_sender=email_sender,
            sms_sender=sms_sender,
            discord_webhook=discord_webhook
        )

    def run_analysis_cycle(self):
        """Run one complete analysis cycle"""
        
        try:
            self.logger.info("Starting analysis cycle")
            
            # 1. Fetch maritime data
            vessel_positions = []
            port_congestion_data = []
            
            for port in self.config["trading"]["ports"]:
                # Get vessel positions near port
                vessels = self.data_fetcher.get_vessel_positions(port, "all")
                vessel_positions.extend(vessels)
                
                # Get port congestion
                congestion = self.data_fetcher.get_port_congestion(port)
                port_congestion_data.append(congestion)
            
            # 2. Fetch market data
            market_data = {}
            volume_profiles = {}
            
            for symbol in self.config["trading"]["symbols"]:
                market_data[symbol] = self.data_fetcher.get_market_data(symbol)
                volume_profiles[symbol] = self.data_fetcher.get_volume_profile(symbol)
            
            # 3. Analyze maritime events
            maritime_events = self.maritime_analyzer.analyze_all_maritime_factors(
                port_congestion_data, vessel_positions, {}, {}
            )
            
            self.stats["events_detected"] += len(maritime_events)
            self.logger.info(f"Detected {len(maritime_events)} maritime events")
            
            # 4. Generate trading signals
            signals = self.signal_generator.generate_signals(
                maritime_events, market_data, volume_profiles
            )
            
            # 5. Apply risk management
            validated_signals = []
            for signal in signals:
                is_valid, reason = self.risk_manager.validate_signal(signal)
                
                if is_valid:
                    # Calculate position size
                    signal.position_size = self.risk_manager.calculate_position_size(signal)
                    
                    if signal.position_size > 0:
                        validated_signals.append(signal)
                        self.logger.info(f"Signal validated: {signal.symbol} {signal.direction.value}")
                    else:
                        self.logger.warning(f"Signal rejected - no position size: {signal.symbol}")
                else:
                    self.logger.warning(f"Signal rejected: {reason}")
            
            self.stats["signals_generated"] += len(validated_signals)
            
            # 6. Send alerts for valid signals
            for signal in validated_signals:
                try:
                    results = self.alert_manager.send_signal_alert(signal)
                    if any(results.values()):
                        self.stats["alerts_sent"] += 1
                        self.logger.info(f"Alert sent for {signal.symbol}")
                    else:
                        self.logger.warning(f"Alert failed for {signal.symbol}")
                except Exception as e:
                    self.logger.error(f"Alert error for {signal.symbol}: {e}")
            
            # 7. Update statistics
            self.stats["last_update"] = datetime.now()
            
            self.logger.info(f"Analysis cycle completed: {len(validated_signals)} signals generated")
            
            return validated_signals, maritime_events
            
        except Exception as e:
            self.logger.error(f"Analysis cycle failed: {e}")
            return [], []

    def run_continuous(self):
        """Run the system continuously"""
        
        self.logger.info("Starting continuous operation")
        self.running = True
        
        # Schedule regular analysis
        update_interval = self.config["data_sources"]["update_interval_minutes"]
        schedule.every(update_interval).minutes.do(self.run_analysis_cycle)
        
        # Schedule daily summary
        schedule.every().day.at("17:00").do(self.send_daily_summary)
        
        # Schedule system health check
        schedule.every(30).minutes.do(self.health_check)
        
        # Initial run
        self.run_analysis_cycle()
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Received shutdown signal")
            self.shutdown()

    def send_daily_summary(self):
        """Send daily performance summary"""
        
        try:
            # Calculate performance metrics
            uptime = datetime.now() - self.stats["start_time"]
            
            performance_data = {
                "signals_generated": self.stats["signals_generated"],
                "events_detected": self.stats["events_detected"],
                "alerts_sent": self.stats["alerts_sent"],
                "uptime_hours": uptime.total_seconds() / 3600,
                "last_update": self.stats["last_update"].isoformat() if self.stats["last_update"] else None
            }
            
            # Mock signals for summary (in production, would track actual signals)
            mock_signals = []
            
            results = self.alert_manager.send_daily_summary(mock_signals, performance_data)
            
            if any(results.values()):
                self.logger.info("Daily summary sent successfully")
            else:
                self.logger.warning("Daily summary failed to send")
                
        except Exception as e:
            self.logger.error(f"Daily summary error: {e}")

    def health_check(self):
        """Perform system health check"""
        
        try:
            # Check data sources
            data_health = self.data_fetcher.health_check()
            
            # Check alert system
            alert_health = self.alert_manager.health_check()
            
            # Check system performance
            if self.stats["last_update"]:
                time_since_update = datetime.now() - self.stats["last_update"]
                if time_since_update > timedelta(minutes=30):
                    self.alert_manager.send_system_alert(
                        "Data Update Delay",
                        f"No data updates for {time_since_update.total_seconds()/60:.0f} minutes",
                        "warning"
                    )
            
            self.logger.info("Health check completed")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")

    def run_test_mode(self):
        """Run in test mode with sample data"""
        
        self.logger.info("Running in test mode")
        
        # Run single analysis cycle
        signals, events = self.run_analysis_cycle()
        
        # Print results
        print("\n" + "="*60)
        print("MARITIME TRADING SIGNALS - TEST MODE RESULTS")
        print("="*60)
        
        print(f"\nMaritime Events Detected: {len(events)}")
        for event in events:
            print(f"  - {event.event_type}: {event.location} (Severity: {event.severity:.2f})")
        
        print(f"\nTrading Signals Generated: {len(signals)}")
        for signal in signals:
            print(f"  - {signal.symbol} {signal.direction.value} @ ${signal.entry_price:.2f}")
            print(f"    Tier: {signal.tier.value}, Confidence: {signal.confidence_score:.1%}")
            print(f"    Reason: {signal.reason}")
        
        print(f"\nSystem Statistics:")
        print(f"  - Runtime: {datetime.now() - self.stats['start_time']}")
        print(f"  - Mock Mode: {self.data_fetcher.mock_mode}")
        print(f"  - Risk per Trade: {self.config['risk_management']['risk_per_trade']:.1%}")
        
        # Test alert system
        if signals:
            print(f"\nTesting alert system with first signal...")
            try:
                test_results = self.alert_manager.send_test_alert()
                print(f"  - Email: {'' if test_results.get('email') else ''}")
                print(f"  - SMS: {'' if test_results.get('sms') else ''}")
                print(f"  - Discord: {'' if test_results.get('discord') else ''}")
            except Exception as e:
                print(f"  - Alert test failed: {e}")
        
        print("\n" + "="*60)

    def shutdown(self):
        """Gracefully shutdown the system"""
        
        self.logger.info("Shutting down Maritime Trading System")
        self.running = False
        
        # Send shutdown notification
        try:
            self.alert_manager.send_system_alert(
                "System Shutdown",
                "Maritime Trading System is shutting down",
                "info"
            )
        except:
            pass  # Don't fail shutdown on alert error
        
        # Final statistics
        uptime = datetime.now() - self.stats["start_time"]
        self.logger.info(f"Final stats - Uptime: {uptime}, Signals: {self.stats['signals_generated']}")

def run_dashboard():
    """Run the Streamlit dashboard"""
    
    import subprocess
    import sys
    
    dashboard_path = os.path.join(os.path.dirname(__file__), "src", "dashboard", "app.py")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("Dashboard stopped")

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Maritime Trading Signals System")
    parser.add_argument("--mock-mode", action="store_true", help="Use mock data")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--dashboard-only", action="store_true", help="Run dashboard only")
    parser.add_argument("--test-mode", action="store_true", help="Run single test cycle")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Dashboard only mode
    if args.dashboard_only:
        print("Starting Maritime Trading Dashboard...")
        print("Access the dashboard at: http://localhost:8501")
        run_dashboard()
        return
    
    # Initialize system
    try:
        system = MaritimeTradingSystem(
            config_path=args.config,
            mock_mode=args.mock_mode,
            test_mode=args.test_mode
        )
        
        if args.test_mode:
            system.run_test_mode()
        else:
            system.run_continuous()
            
    except Exception as e:
        print(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()