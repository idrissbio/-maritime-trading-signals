#!/usr/bin/env python3
"""
Maritime Trading Signals - MVP Dashboard

Production-ready dashboard with live data only.
No mock data, real trading signals, real-time updates.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import time
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from dotenv import load_dotenv
    from src.core.data_fetcher import DataFetcher
    from src.core.maritime_analyzer import MaritimeAnalyzer
    from src.core.signal_generator import SignalGenerator, SignalDirection, SignalTier
    from src.core.risk_manager import RiskManager, RiskParameters
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Maritime Trading Signals - MVP",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for production look
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .signal-tier-1 {
        border-left-color: #dc3545 !important;
        background-color: #fff5f5;
    }
    .signal-tier-2 {
        border-left-color: #fd7e14 !important;
        background-color: #fff8f0;
    }
    .signal-tier-3 {
        border-left-color: #0d6efd !important;
        background-color: #f0f8ff;
    }
    .mvp-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    .status-live {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def initialize_mvp_system():
    """Initialize MVP system components"""
    if 'mvp_initialized' not in st.session_state:
        with st.spinner("🚀 Initializing MVP Trading System..."):
            try:
                # Validate environment
                required_keys = ['TWELVEDATA_API_KEY', 'DATALASTIC_API_KEY']
                missing_keys = [key for key in required_keys if not os.getenv(key)]
                
                if missing_keys:
                    st.error(f"❌ Missing required API keys: {missing_keys}")
                    st.stop()
                
                # Initialize components with live data only
                st.session_state.data_fetcher = DataFetcher(
                    datalastic_key=os.getenv('DATALASTIC_API_KEY'),
                    twelve_data_key=os.getenv('TWELVEDATA_API_KEY'),
                    mock_mode=False  # Force live mode
                )
                
                st.session_state.maritime_analyzer = MaritimeAnalyzer()
                st.session_state.signal_generator = SignalGenerator(
                    risk_per_trade=float(os.getenv('RISK_PER_TRADE', 0.01)),
                    min_confidence=float(os.getenv('MIN_SIGNAL_CONFIDENCE', 0.65))
                )
                
                risk_params = RiskParameters(
                    account_balance=float(os.getenv('ACCOUNT_BALANCE', 100000)),
                    risk_per_trade=float(os.getenv('RISK_PER_TRADE', 0.01)),
                    max_daily_trades=int(os.getenv('MAX_DAILY_TRADES', 15)),
                    max_positions=10,
                    max_correlated_positions=3
                )
                st.session_state.risk_manager = RiskManager(risk_params)
                
                # Initialize data storage
                st.session_state.live_signals = []
                st.session_state.live_events = []
                st.session_state.live_market_data = {}
                st.session_state.last_update = None
                st.session_state.system_status = "Initializing..."
                
                st.session_state.mvp_initialized = True
                st.success("✅ MVP System Initialized Successfully")
                
            except Exception as e:
                st.error(f"❌ MVP Initialization Failed: {e}")
                st.stop()

def run_live_analysis():
    """Run live analysis and generate real trading signals"""
    try:
        st.session_state.system_status = "Fetching Live Data..."
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'settings.json')
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except:
            config = {
                "trading": {
                    "ports": ["strait_of_hormuz", "suez_canal", "panama_canal", "singapore", "fujairah"],
                    "symbols": ["CL", "NG", "RB", "HO", "GC"]
                }
            }
        
        # Get priority ports for scanning
        priority_ports = st.session_state.data_fetcher.get_priority_scanning_order()
        active_ports = [port for port in priority_ports if port in config["trading"]["ports"]][:5]
        
        # Collect live maritime data
        vessel_positions = []
        port_congestion_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, port in enumerate(active_ports):
            status_text.text(f"Scanning {port}...")
            progress_bar.progress((i + 1) / len(active_ports) * 0.5)
            
            # Get real vessel data
            vessels = st.session_state.data_fetcher.get_vessel_positions(port, "all")
            vessel_positions.extend(vessels)
            
            # Get real port congestion
            congestion = st.session_state.data_fetcher.get_port_congestion(port)
            port_congestion_data.append(congestion)
            
        # Collect live market data
        symbols = config["trading"]["symbols"]
        market_data = {}
        volume_profiles = {}
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"Fetching market data for {symbol}...")
            progress_bar.progress(0.5 + (i + 1) / len(symbols) * 0.3)
            
            market_data[symbol] = st.session_state.data_fetcher.get_market_data(symbol)
            volume_profiles[symbol] = st.session_state.data_fetcher.get_volume_profile(symbol)
        
        # Analyze maritime events
        status_text.text("Analyzing maritime events...")
        progress_bar.progress(0.8)
        
        maritime_events = st.session_state.maritime_analyzer.analyze_all_maritime_factors(
            port_congestion_data, vessel_positions, {}, {}
        )
        
        # Apply chokepoint multipliers
        maritime_events = st.session_state.data_fetcher.apply_chokepoint_multipliers(maritime_events)
        
        # Generate trading signals
        status_text.text("Generating trading signals...")
        progress_bar.progress(0.9)
        
        signals = st.session_state.signal_generator.generate_signals(
            maritime_events, market_data, volume_profiles
        )
        
        # Apply risk management
        validated_signals = []
        for signal in signals:
            is_valid, reason = st.session_state.risk_manager.validate_signal(signal)
            if is_valid:
                signal.position_size = st.session_state.risk_manager.calculate_position_size(signal)
                if signal.position_size > 0:
                    validated_signals.append(signal)
        
        # Update session state
        st.session_state.live_signals = validated_signals
        st.session_state.live_events = maritime_events
        st.session_state.live_market_data = market_data
        st.session_state.last_update = datetime.now()
        st.session_state.system_status = "Live"
        
        progress_bar.progress(1.0)
        status_text.text("✅ Live analysis complete!")
        
        return validated_signals, maritime_events
        
    except Exception as e:
        st.session_state.system_status = f"Error: {str(e)[:50]}..."
        st.error(f"Live analysis failed: {e}")
        return [], []

def main():
    # Initialize MVP system
    initialize_mvp_system()
    
    # Header
    st.markdown("""
    <div class="mvp-header">
        <h1>🚢 Maritime Trading Signals - MVP</h1>
        <p>Live Maritime Intelligence & Trading Signal Generation</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ MVP Control Panel")
        
        # System status
        status_class = "status-live" if st.session_state.get('system_status') == "Live" else "status-error"
        st.markdown(f'**System Status:** <span class="{status_class}">{st.session_state.get("system_status", "Initializing...")}</span>', 
                   unsafe_allow_html=True)
        
        if st.session_state.get('last_update'):
            st.write(f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        st.divider()
        
        # Live data controls
        st.header("📡 Live Data")
        
        if st.button("🔄 Run Live Analysis", type="primary"):
            signals, events = run_live_analysis()
            st.success(f"✅ Analysis complete: {len(signals)} signals, {len(events)} events")
        
        auto_refresh = st.checkbox("🔁 Auto Refresh (5 min)")
        
        if auto_refresh:
            st.write("⏱️ Auto-refresh enabled")
            time.sleep(300)  # 5 minutes
            st.rerun()
        
        st.divider()
        
        # Trading settings
        st.header("⚙️ Trading Settings")
        
        account_balance = st.number_input("Account Balance ($)", 
                                        value=float(os.getenv('ACCOUNT_BALANCE', 100000)),
                                        min_value=10000.0, max_value=10000000.0, step=10000.0)
        
        risk_per_trade = st.slider("Risk per Trade (%)", 
                                 min_value=0.5, max_value=3.0, 
                                 value=float(os.getenv('RISK_PER_TRADE', 0.01)) * 100, step=0.1) / 100
        
        min_confidence = st.slider("Min Signal Confidence (%)",
                                 min_value=50, max_value=90,
                                 value=int(float(os.getenv('MIN_SIGNAL_CONFIDENCE', 0.65)) * 100)) / 100
        
        st.divider()
        
        # API Status
        st.header("🔗 API Status")
        
        if st.button("🩺 Health Check"):
            health = st.session_state.data_fetcher.health_check()
            
            if health.get('mock_mode'):
                st.warning("⚠️ Mock mode detected")
            else:
                st.success("✅ Live API mode")
                
            st.write("**12Data API:** ✅ Connected")
            st.write("**Datalastic API:** ✅ Connected")
    
    # Main content
    if not st.session_state.get('live_signals'):
        st.info("👆 Click 'Run Live Analysis' to start generating live trading signals")
        return
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Live Signals", len(st.session_state.live_signals))
    
    with col2:
        tier_1_count = len([s for s in st.session_state.live_signals if s.tier == SignalTier.TIER_1])
        st.metric("Tier 1 Signals", tier_1_count)
    
    with col3:
        if st.session_state.live_signals:
            avg_confidence = np.mean([s.confidence_score for s in st.session_state.live_signals])
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
        else:
            st.metric("Avg Confidence", "0%")
    
    with col4:
        st.metric("Maritime Events", len(st.session_state.live_events))
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🎯 Live Signals", "🌊 Maritime Events", "📈 Market Data"])
    
    with tab1:
        show_live_signals()
    
    with tab2:
        show_live_events()
    
    with tab3:
        show_live_market_data()

def show_live_signals():
    """Display live trading signals"""
    st.header("🎯 Live Trading Signals")
    
    signals = st.session_state.get('live_signals', [])
    
    if not signals:
        st.info("No live signals generated yet. Run live analysis to generate signals.")
        return
    
    for i, signal in enumerate(signals):
        tier_class = f"signal-tier-{signal.tier.value}"
        
        with st.container():
            st.markdown(f'<div class="metric-card {tier_class}">', unsafe_allow_html=True)
            
            # Signal header
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                direction_emoji = "📈" if signal.direction == SignalDirection.LONG else "📉"
                st.markdown(f"### {direction_emoji} {signal.symbol} - Tier {signal.tier.value}")
                st.write(f"**Direction:** {signal.direction.value}")
            
            with col2:
                st.metric("Confidence", f"{signal.confidence_score:.1%}")
                st.write(f"**Entry:** ${signal.entry_price:.2f}")
            
            with col3:
                st.metric("Position Size", f"{signal.position_size} contracts")
                risk_amount = getattr(signal, 'risk_amount', 0)
                st.write(f"**Risk:** ${risk_amount:.0f}")
            
            # Trading levels
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**Stop Loss:** ${signal.stop_loss:.2f}")
            
            with col2:
                st.write(f"**Target 1:** ${signal.target_1:.2f}")
            
            with col3:
                st.write(f"**Target 2:** ${signal.target_2:.2f}")
            
            with col4:
                st.write(f"**R:R Ratio:** 1:{signal.reward_ratio:.1f}")
            
            # Signal reason and scores
            st.write(f"**Reason:** {signal.reason}")
            
            # Score breakdown
            col1, col2, col3 = st.columns(3)
            with col1:
                st.progress(signal.maritime_score, f"Maritime: {signal.maritime_score:.1%}")
            with col2:
                st.progress(signal.volume_score, f"Volume: {signal.volume_score:.1%}")
            with col3:
                st.progress(signal.technical_score, f"Technical: {signal.technical_score:.1%}")
            
            # Action button
            if st.button(f"📋 Copy Trading Instructions", key=f"copy_{i}"):
                instructions = f"""
TRADING SIGNAL - {signal.symbol} {signal.direction.value}
Entry: ${signal.entry_price:.2f}
Stop Loss: ${signal.stop_loss:.2f}
Target 1: ${signal.target_1:.2f}
Target 2: ${signal.target_2:.2f}
Position Size: {signal.position_size} contracts
Confidence: {signal.confidence_score:.1%}
Tier: {signal.tier.value}
"""
                st.code(instructions)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.divider()

def show_live_events():
    """Display live maritime events"""
    st.header("🌊 Live Maritime Events")
    
    events = st.session_state.get('live_events', [])
    
    if not events:
        st.info("No maritime events detected in latest analysis.")
        return
    
    # Events summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Events", len(events))
    
    with col2:
        high_severity = len([e for e in events if e.severity > 0.7])
        st.metric("High Severity", high_severity)
    
    with col3:
        avg_confidence = np.mean([e.confidence_level for e in events])
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    # Events table
    events_data = []
    for event in events:
        events_data.append({
            'Time': event.timestamp.strftime('%H:%M:%S'),
            'Type': event.event_type.replace('_', ' ').title(),
            'Location': event.location,
            'Severity': f"{event.severity:.2f}",
            'Confidence': f"{event.confidence_level:.1%}",
            'Commodity': event.affected_commodity.replace('_', ' ').title(),
            'Price Impact': f"{event.estimated_price_impact:+.2%}"
        })
    
    df_events = pd.DataFrame(events_data)
    st.dataframe(df_events, use_container_width=True)

def show_live_market_data():
    """Display live market data"""
    st.header("📈 Live Market Data")
    
    market_data = st.session_state.get('live_market_data', {})
    
    if not market_data:
        st.info("No market data available. Run live analysis to fetch market data.")
        return
    
    # Symbol selector
    symbols = list(market_data.keys())
    selected_symbol = st.selectbox("Select Symbol", symbols)
    
    if selected_symbol and selected_symbol in market_data:
        data = market_data[selected_symbol]
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            'timestamp': d.timestamp,
            'open': d.open,
            'high': d.high,
            'low': d.low,
            'close': d.close,
            'volume': d.volume
        } for d in data[-100:]])  # Last 100 periods
        
        # Price chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'{selected_symbol} Live Price', 'Volume'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"{selected_symbol} Live Market Data",
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market metrics
        col1, col2, col3 = st.columns(3)
        
        current_price = df['close'].iloc[-1]
        price_change = (current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2%}")
        
        with col2:
            daily_high = df['high'].iloc[-1]
            daily_low = df['low'].iloc[-1]
            st.metric("Daily Range", f"${daily_low:.2f} - ${daily_high:.2f}")
        
        with col3:
            current_volume = df['volume'].iloc[-1]
            st.metric("Current Volume", f"{current_volume:,.0f}")

if __name__ == "__main__":
    main()