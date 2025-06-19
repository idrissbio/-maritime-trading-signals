#!/usr/bin/env python3
"""
Maritime Trading Signals - Start MVP

Production MVP launcher with live data validation.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def validate_mvp_readiness():
    """Validate MVP is ready for live trading"""
    print("🚢 Maritime Trading Signals - MVP Startup")
    print("=" * 50)
    
    # Check environment variables
    required_vars = ['TWELVEDATA_API_KEY', 'DATALASTIC_API_KEY']
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing required environment variables: {missing}")
        return False
    
    print("✅ Environment variables configured")
    
    # Check if MOCK_MODE is disabled
    mock_mode = os.getenv('MOCK_MODE', 'false').lower()
    if mock_mode == 'true':
        print("⚠️  MOCK_MODE is enabled - switching to live mode for MVP")
        os.environ['MOCK_MODE'] = 'false'
    
    print("✅ Live data mode confirmed")
    
    # Test API connectivity
    print("🔍 Testing API connectivity...")
    
    try:
        # Quick API test
        sys.path.append('src')
        from dotenv import load_dotenv
        from src.core.data_fetcher import DataFetcher
        
        load_dotenv()
        
        data_fetcher = DataFetcher(
            datalastic_key=os.getenv('DATALASTIC_API_KEY'),
            twelve_data_key=os.getenv('TWELVEDATA_API_KEY'),
            mock_mode=False
        )
        
        # Test 12Data
        market_data = data_fetcher.get_market_data("CL", "5min")
        print(f"✅ 12Data API: {len(market_data)} market data points")
        
        # Test Datalastic
        vessels = data_fetcher.get_vessel_positions("singapore", "crude_oil")
        print(f"✅ Datalastic API: {len(vessels)} vessels found")
        
        print("🎯 MVP READY FOR LIVE TRADING!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def start_mvp_services():
    """Start MVP services"""
    
    print("\n🚀 Starting MVP Services...")
    
    # Start dashboard
    print("📊 Starting MVP Dashboard on http://localhost:8503")
    dashboard_cmd = [sys.executable, "-m", "streamlit", "run", "mvp_dashboard.py", "--server.port", "8503"]
    
    try:
        subprocess.Popen(dashboard_cmd)
        time.sleep(3)
        print("✅ MVP Dashboard started successfully")
        
        print("\n" + "="*50)
        print("🎯 MVP MARITIME TRADING SYSTEM ACTIVE")
        print("="*50)
        print(f"📊 Dashboard: http://localhost:8503")
        print(f"🔴 Status: LIVE DATA MODE")
        print(f"💰 Account: ${os.getenv('ACCOUNT_BALANCE', '100000')}")
        print(f"⚖️  Risk per Trade: {float(os.getenv('RISK_PER_TRADE', '0.01'))*100:.1f}%")
        print(f"🎯 Min Confidence: {float(os.getenv('MIN_SIGNAL_CONFIDENCE', '0.65'))*100:.0f}%")
        print("="*50)
        print("🚨 LIVE TRADING SYSTEM - TRADE AT YOUR OWN RISK")
        print("📱 Click 'Run Live Analysis' in dashboard to generate signals")
        print("💡 Signals are based on live maritime and market data")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        return False

def main():
    """Main MVP startup function"""
    
    # Validate readiness
    if not validate_mvp_readiness():
        print("❌ MVP startup failed - system not ready")
        return False
    
    # Start services
    if not start_mvp_services():
        print("❌ MVP startup failed - could not start services")
        return False
    
    print("\n✅ MVP startup complete!")
    print("🔗 Open http://localhost:8503 to start trading")
    
    # Keep script running
    try:
        print("\n⏱️  MVP running... Press Ctrl+C to stop")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 MVP stopped by user")

if __name__ == "__main__":
    main()