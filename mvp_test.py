#!/usr/bin/env python3
"""
Quick MVP test to validate signal generation
"""

import sys
import os
sys.path.append('src')

from dotenv import load_dotenv
load_dotenv()

from src.core.data_fetcher import DataFetcher
from src.core.maritime_analyzer import MaritimeAnalyzer
from src.core.signal_generator import SignalGenerator

def test_mvp():
    print("🧪 MVP Signal Generation Test")
    print("=" * 40)
    
    # Initialize with live APIs
    data_fetcher = DataFetcher(
        datalastic_key=os.getenv('DATALASTIC_API_KEY'),
        twelve_data_key=os.getenv('TWELVEDATA_API_KEY'),
        mock_mode=False
    )
    
    maritime_analyzer = MaritimeAnalyzer()
    
    # Use more relaxed criteria for MVP
    signal_generator = SignalGenerator(
        risk_per_trade=0.01,
        min_confidence=0.50  # Lower threshold for MVP testing
    )
    
    # Test with known working ports
    test_ports = ["singapore", "houston"]
    
    # Collect data
    vessel_positions = []
    port_congestion_data = []
    
    for port in test_ports:
        print(f"📍 Testing {port}...")
        
        vessels = data_fetcher.get_vessel_positions(port, "all")
        vessel_positions.extend(vessels)
        print(f"   Found {len(vessels)} vessels")
        
        congestion = data_fetcher.get_port_congestion(port)
        port_congestion_data.append(congestion)
        print(f"   Congestion score: {congestion.congestion_score:.2f}")
    
    # Get market data
    market_data = {}
    volume_profiles = {}
    
    symbols = ["CL", "NG"]
    for symbol in symbols:
        print(f"📈 Getting market data for {symbol}...")
        market_data[symbol] = data_fetcher.get_market_data(symbol)
        volume_profiles[symbol] = data_fetcher.get_volume_profile(symbol)
        print(f"   {len(market_data[symbol])} data points")
    
    # Analyze maritime events
    print("🌊 Analyzing maritime events...")
    maritime_events = maritime_analyzer.analyze_all_maritime_factors(
        port_congestion_data, vessel_positions, {}, {}
    )
    
    print(f"Found {len(maritime_events)} maritime events")
    for event in maritime_events[:5]:  # Show first 5
        print(f"   - {event.event_type} at {event.location}: severity {event.severity:.2f}")
    
    # Apply chokepoint multipliers
    enhanced_events = data_fetcher.apply_chokepoint_multipliers(maritime_events)
    
    # Generate signals
    print("🎯 Generating signals...")
    signals = signal_generator.generate_signals(enhanced_events, market_data, volume_profiles)
    
    print(f"Generated {len(signals)} signals")
    
    if signals:
        for signal in signals:
            print(f"✅ Signal: {signal.symbol} {signal.direction.value} "
                  f"(Tier {signal.tier.value}, {signal.confidence_score:.1%} confidence)")
            print(f"   Entry: ${signal.entry_price:.2f}, Stop: ${signal.stop_loss:.2f}")
            print(f"   Reason: {signal.reason}")
    else:
        print("❌ No signals generated")
        
        # Debug information
        print("\n🔍 Debug Info:")
        print(f"Maritime events: {len(maritime_events)}")
        print(f"Market data symbols: {list(market_data.keys())}")
        
        # Check event details
        if maritime_events:
            event = maritime_events[0]
            print(f"Sample event: {event.event_type}, severity: {event.severity:.2f}, confidence: {event.confidence_level:.2f}")
            print(f"Affected commodity: {event.affected_commodity}")
        
        # Check maritime scores
        for commodity in ["crude_oil", "lng"]:
            commodity_events = [e for e in maritime_events if e.affected_commodity == commodity]
            if commodity_events:
                maritime_score = signal_generator._calculate_maritime_score(commodity_events, "CL" if commodity == "crude_oil" else "NG")
                print(f"Maritime score for {commodity}: {maritime_score:.2f}")

if __name__ == "__main__":
    test_mvp()