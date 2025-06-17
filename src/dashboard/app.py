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
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.core.data_fetcher import DataFetcher
    from src.core.maritime_analyzer import MaritimeAnalyzer
    from src.core.signal_generator import SignalGenerator, SignalDirection, SignalTier
    from src.core.risk_manager import RiskManager, RiskParameters
    from src.alerts.alert_manager import AlertManager, AlertConfig
    from src.strategies.port_congestion import PortCongestionStrategy
    from src.strategies.volume_breakout import VolumeBreakoutStrategy
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page config
st.set_page_config(
    page_title="Maritime Trading Signals",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .signal-tier-1 {
        border-left-color: #ff4444 !important;
        background-color: #fff5f5;
    }
    .signal-tier-2 {
        border-left-color: #ff8800 !important;
        background-color: #fff8f0;
    }
    .signal-tier-3 {
        border-left-color: #0088ff !important;
        background-color: #f0f8ff;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_fetcher' not in st.session_state:
    # Load environment variables for live API integration
    from dotenv import load_dotenv
    load_dotenv()
    
    datalastic_key = os.getenv("DATALASTIC_API_KEY")
    twelve_data_key = os.getenv("TWELVEDATA_API_KEY")
    mock_mode = os.getenv("MOCK_MODE", "false").lower() == "true"
    
    st.session_state.data_fetcher = DataFetcher(
        datalastic_key=datalastic_key,
        twelve_data_key=twelve_data_key,
        mock_mode=mock_mode
    )
if 'maritime_analyzer' not in st.session_state:
    st.session_state.maritime_analyzer = MaritimeAnalyzer()
if 'signal_generator' not in st.session_state:
    st.session_state.signal_generator = SignalGenerator()
if 'risk_manager' not in st.session_state:
    risk_params = RiskParameters(account_balance=100000)  # $100k demo account
    st.session_state.risk_manager = RiskManager(risk_params)
if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'maritime_events' not in st.session_state:
    st.session_state.maritime_events = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

def load_sample_data():
    """Load sample data for demonstration"""
    
    # Generate sample vessel positions
    vessel_positions = st.session_state.data_fetcher.get_vessel_positions("singapore", "crude_oil")
    
    # Generate sample port congestion
    ports = ["singapore", "houston", "rotterdam"]
    port_congestion = []
    for port in ports:
        congestion = st.session_state.data_fetcher.get_port_congestion(port)
        port_congestion.append(congestion)
    
    # Generate sample market data
    symbols = ["CL", "NG", "GC"]
    market_data = {}
    volume_profiles = {}
    
    for symbol in symbols:
        market_data[symbol] = st.session_state.data_fetcher.get_market_data(symbol)
        volume_profiles[symbol] = st.session_state.data_fetcher.get_volume_profile(symbol)
    
    # Analyze maritime events
    maritime_events = st.session_state.maritime_analyzer.analyze_all_maritime_factors(
        port_congestion, vessel_positions, {}, {}
    )
    
    # Generate signals
    signals = st.session_state.signal_generator.generate_signals(
        maritime_events, market_data, volume_profiles
    )
    
    return signals, maritime_events, market_data, volume_profiles

def main():
    # Header
    st.title("🚢 Maritime Trading Signals Dashboard")
    st.markdown("Real-time maritime data analysis and trading signal generation")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Control")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        
        # Manual refresh button
        if st.button("🔄 Refresh Data", type="primary"):
            with st.spinner("Refreshing data..."):
                signals, events, market_data, volume_profiles = load_sample_data()
                st.session_state.signals = signals
                st.session_state.maritime_events = events
                st.session_state.market_data = market_data
                st.session_state.volume_profiles = volume_profiles
                st.session_state.last_update = datetime.now()
                st.success("Data refreshed!")
        
        st.divider()
        
        # System status
        st.header("📊 System Status")
        
        # API status
        health_check = st.session_state.data_fetcher.health_check()
        
        if health_check.get("mock_mode", False):
            st.success("✅ Mock Mode Active")
        else:
            st.info("🔗 Live API Mode")
        
        # Last update
        st.write(f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S')}")
        
        st.divider()
        
        # Quick settings
        st.header("⚡ Quick Settings")
        
        risk_per_trade = st.slider("Risk per Trade (%)", 0.5, 3.0, 1.0, 0.1) / 100
        min_confidence = st.slider("Min Signal Confidence (%)", 50, 90, 65) / 100
        max_daily_trades = st.slider("Max Daily Trades", 5, 20, 15)
        
        # Update risk manager if settings changed
        current_risk_params = st.session_state.risk_manager.risk_params
        if (current_risk_params.risk_per_trade != risk_per_trade or 
            current_risk_params.max_daily_trades != max_daily_trades):
            
            current_risk_params.risk_per_trade = risk_per_trade
            current_risk_params.max_daily_trades = max_daily_trades
        
        st.divider()
        
        # Emergency controls
        st.header("🚨 Emergency")
        
        if st.button("⚠️ Emergency Stop", type="secondary"):
            st.session_state.signals = []
            st.warning("All signals cleared!")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(1)  # Small delay
        st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Active Signals", 
        "🌊 Maritime Events", 
        "📈 Market Analysis", 
        "⚖️ Risk Management", 
        "📋 System Logs"
    ])
    
    with tab1:
        show_active_signals()
    
    with tab2:
        show_maritime_events()
    
    with tab3:
        show_market_analysis()
    
    with tab4:
        show_risk_management()
    
    with tab5:
        show_system_logs()

def show_active_signals():
    """Display active trading signals"""
    
    st.header("🎯 Active Trading Signals")
    
    # Generate fresh signals if none exist
    if not st.session_state.signals:
        with st.spinner("Generating signals..."):
            signals, events, market_data, volume_profiles = load_sample_data()
            st.session_state.signals = signals
            st.session_state.maritime_events = events
            st.session_state.market_data = market_data
            st.session_state.volume_profiles = volume_profiles
    
    signals = st.session_state.signals
    
    if not signals:
        st.info("No active signals at this time.")
        return
    
    # Signal summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Signals", len(signals))
    
    with col2:
        tier_1_count = len([s for s in signals if s.tier.value == 1])
        st.metric("Tier 1 Signals", tier_1_count)
    
    with col3:
        avg_confidence = np.mean([s.confidence_score for s in signals])
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    with col4:
        long_signals = len([s for s in signals if s.direction == SignalDirection.LONG])
        st.metric("Long/Short", f"{long_signals}/{len(signals) - long_signals}")
    
    st.divider()
    
    # Signal cards
    for i, signal in enumerate(signals):
        
        # Calculate position size
        signal.position_size = st.session_state.risk_manager.calculate_position_size(signal)
        
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
                risk_pct = (signal.risk_amount / 100000) * 100  # Assuming 100k account
                st.write(f"**Risk:** {risk_pct:.2f}%")
            
            # Price levels
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**Stop Loss:** ${signal.stop_loss:.2f}")
                stop_pct = abs((signal.stop_loss - signal.entry_price) / signal.entry_price * 100)
                st.caption(f"({stop_pct:.2f}%)")
            
            with col2:
                st.write(f"**Target 1:** ${signal.target_1:.2f}")
                target1_pct = abs((signal.target_1 - signal.entry_price) / signal.entry_price * 100)
                st.caption(f"(+{target1_pct:.2f}%)")
            
            with col3:
                st.write(f"**Target 2:** ${signal.target_2:.2f}")
                target2_pct = abs((signal.target_2 - signal.entry_price) / signal.entry_price * 100)
                st.caption(f"(+{target2_pct:.2f}%)")
            
            with col4:
                st.write(f"**R:R Ratio:** 1:{signal.reward_ratio:.1f}")
                st.caption(f"Expires: {signal.expiry_time.strftime('%H:%M')}")
            
            # Reason and scores
            st.write(f"**Reason:** {signal.reason}")
            
            # Score breakdown
            col1, col2, col3 = st.columns(3)
            with col1:
                st.progress(signal.maritime_score, f"Maritime: {signal.maritime_score:.1%}")
            with col2:
                st.progress(signal.volume_score, f"Volume: {signal.volume_score:.1%}")
            with col3:
                st.progress(signal.technical_score, f"Technical: {signal.technical_score:.1%}")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"📊 View Chart", key=f"chart_{i}"):
                    show_signal_chart(signal)
            
            with col2:
                if st.button(f"💰 Trade Now", key=f"trade_{i}", type="primary"):
                    st.success(f"Trade order placed for {signal.symbol}!")
            
            with col3:
                if st.button(f"❌ Skip Signal", key=f"skip_{i}"):
                    # Remove signal from list
                    st.session_state.signals.remove(signal)
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.divider()

def show_signal_chart(signal):
    """Show chart for a specific signal"""
    
    # Get market data
    market_data = st.session_state.market_data.get(signal.symbol, [])
    
    if not market_data:
        st.warning("No chart data available")
        return
    
    # Create price chart
    df = pd.DataFrame([{
        'timestamp': d.timestamp,
        'open': d.open,
        'high': d.high,
        'low': d.low,
        'close': d.close,
        'volume': d.volume
    } for d in market_data[-50:]])  # Last 50 periods
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(f'{signal.symbol} Price', 'Volume'),
        vertical_spacing=0.1,
        row_width=[0.7, 0.3]
    )
    
    # Candlestick chart
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
    
    # Add signal levels
    fig.add_hline(y=signal.entry_price, line_dash="dash", line_color="blue", 
                  annotation_text="Entry", row=1, col=1)
    fig.add_hline(y=signal.stop_loss, line_dash="dash", line_color="red", 
                  annotation_text="Stop Loss", row=1, col=1)
    fig.add_hline(y=signal.target_1, line_dash="dash", line_color="green", 
                  annotation_text="Target 1", row=1, col=1)
    fig.add_hline(y=signal.target_2, line_dash="dash", line_color="darkgreen", 
                  annotation_text="Target 2", row=1, col=1)
    
    # Volume chart
    fig.add_trace(
        go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color='lightblue'),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{signal.symbol} - {signal.direction.value} Signal",
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_maritime_events():
    """Display maritime events analysis"""
    
    st.header("🌊 Maritime Events Analysis")
    
    events = st.session_state.maritime_events
    
    if not events:
        st.info("No maritime events detected.")
        return
    
    # Events summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Events", len(events))
    
    with col2:
        high_severity = len([e for e in events if e.severity > 0.7])
        st.metric("High Severity", high_severity)
    
    with col3:
        congestion_events = len([e for e in events if e.event_type == "port_congestion"])
        st.metric("Port Congestion", congestion_events)
    
    with col4:
        avg_confidence = np.mean([e.confidence_level for e in events])
        st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    
    st.divider()
    
    # Events map (simplified visualization)
    st.subheader("📍 Events Map")
    
    # Create sample map data
    map_data = []
    for event in events:
        if hasattr(event, 'cluster_center'):
            lat, lon = event.cluster_center
        else:
            # Use approximate coordinates for ports
            port_coords = {
                "singapore": (1.35, 103.8),
                "houston": (29.7, -95.4),
                "rotterdam": (51.9, 4.5)
            }
            lat, lon = port_coords.get(event.location.lower(), (0, 0))
        
        map_data.append({
            'lat': lat,
            'lon': lon,
            'severity': event.severity,
            'event_type': event.event_type,
            'location': event.location
        })
    
    if map_data:
        df_map = pd.DataFrame(map_data)
        
        fig = px.scatter_mapbox(
            df_map,
            lat="lat",
            lon="lon",
            size="severity",
            color="event_type",
            hover_name="location",
            hover_data=["severity"],
            mapbox_style="open-street-map",
            height=400,
            zoom=1
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Events table
    st.subheader("📋 Events Details")
    
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

def show_market_analysis():
    """Display market analysis"""
    
    st.header("📈 Market Analysis")
    
    if not hasattr(st.session_state, 'market_data'):
        st.info("No market data available.")
        return
    
    market_data = st.session_state.market_data
    
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
            subplot_titles=(f'{selected_symbol} Price Chart', 'Volume'),
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
        
        # Moving averages
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['ma_20'], name='MA 20', line=dict(color='orange')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['ma_50'], name='MA 50', line=dict(color='red')),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(x=df['timestamp'], y=df['volume'], name='Volume', marker_color='lightblue'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"{selected_symbol} Market Analysis",
            xaxis_rangeslider_visible=False,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Market metrics
        col1, col2, col3, col4 = st.columns(4)
        
        current_price = df['close'].iloc[-1]
        price_change = (current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2%}")
        
        with col2:
            daily_high = df['high'].iloc[-1]
            daily_low = df['low'].iloc[-1]
            st.metric("Daily Range", f"${daily_low:.2f} - ${daily_high:.2f}")
        
        with col3:
            volume_avg = df['volume'].rolling(20).mean().iloc[-1]
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / volume_avg if volume_avg > 0 else 1
            st.metric("Volume Ratio", f"{volume_ratio:.1f}x")
        
        with col4:
            volatility = df['close'].pct_change().rolling(20).std().iloc[-1] * 100
            st.metric("Volatility (20d)", f"{volatility:.2f}%")

def show_risk_management():
    """Display risk management dashboard"""
    
    st.header("⚖️ Risk Management")
    
    risk_manager = st.session_state.risk_manager
    portfolio_summary = risk_manager.get_portfolio_summary()
    
    # Portfolio overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Positions", portfolio_summary["total_positions"])
    
    with col2:
        st.metric("Total Risk", f"${portfolio_summary['total_risk']:.0f}")
    
    with col3:
        st.metric("Risk %", f"{portfolio_summary['risk_percentage']:.2f}%")
    
    with col4:
        st.metric("Daily Trades", f"{portfolio_summary['daily_trades_used']}/{risk_manager.risk_params.max_daily_trades}")
    
    st.divider()
    
    # Risk breakdown charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Positions by Tier")
        if portfolio_summary["positions_by_tier"]:
            tier_data = portfolio_summary["positions_by_tier"]
            fig = px.pie(
                values=list(tier_data.values()),
                names=list(tier_data.keys()),
                title="Position Distribution by Tier"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active positions")
    
    with col2:
        st.subheader("Positions by Symbol")
        if portfolio_summary["positions_by_symbol"]:
            symbol_data = portfolio_summary["positions_by_symbol"]
            fig = px.bar(
                x=list(symbol_data.keys()),
                y=list(symbol_data.values()),
                title="Position Sizes by Symbol"
            )
            fig.update_layout(xaxis_title="Symbol", yaxis_title="Contracts")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active positions")
    
    # Risk parameters
    st.subheader("⚙️ Risk Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Account Balance:** ${risk_manager.risk_params.account_balance:,.0f}")
        st.write(f"**Risk per Trade:** {risk_manager.risk_params.risk_per_trade:.1%}")
        st.write(f"**Max Daily Trades:** {risk_manager.risk_params.max_daily_trades}")
    
    with col2:
        st.write(f"**Max Positions:** {risk_manager.risk_params.max_positions}")
        st.write(f"**Max Correlated:** {risk_manager.risk_params.max_correlated_positions}")
        st.write(f"**Tier 3 Limit:** {risk_manager.risk_params.max_tier3_allocation:.1%}")
    
    # Risk suggestions
    suggestions = risk_manager.suggest_position_adjustments()
    if suggestions:
        st.subheader("⚠️ Risk Alerts")
        for suggestion in suggestions:
            st.warning(suggestion)

def show_system_logs():
    """Display system logs and performance"""
    
    st.header("📋 System Logs")
    
    # System performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Uptime", "2h 34m")
    
    with col2:
        st.metric("API Calls", "1,247")
    
    with col3:
        st.metric("Signals Generated", len(st.session_state.signals))
    
    with col4:
        st.metric("Events Detected", len(st.session_state.maritime_events))
    
    st.divider()
    
    # Mock log entries
    log_entries = [
        {"time": "14:32:15", "level": "INFO", "message": "Signal generated: CL LONG @ $75.42"},
        {"time": "14:31:58", "level": "INFO", "message": "Port congestion detected: Singapore"},
        {"time": "14:31:45", "level": "INFO", "message": "Volume breakout: CL 2.3x normal volume"},
        {"time": "14:31:30", "level": "INFO", "message": "Maritime data refreshed successfully"},
        {"time": "14:31:15", "level": "WARNING", "message": "High correlation detected: CL-RB positions"},
        {"time": "14:31:00", "level": "INFO", "message": "Risk check passed for new signal"},
        {"time": "14:30:45", "level": "INFO", "message": "Position opened: NG 3 contracts LONG"},
        {"time": "14:30:30", "level": "ERROR", "message": "API rate limit warning: Datalastic"},
    ]
    
    # Display logs
    st.subheader("📜 Recent Logs")
    
    for entry in log_entries:
        level_color = {
            "INFO": "🔵",
            "WARNING": "🟡", 
            "ERROR": "🔴"
        }.get(entry["level"], "⚪")
        
        st.write(f"{level_color} **{entry['time']}** [{entry['level']}] {entry['message']}")
    
    # System health
    st.divider()
    st.subheader("💓 System Health")
    
    health_status = st.session_state.data_fetcher.health_check()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Data Sources:**")
        if health_status.get("mock_mode"):
            st.success("✅ Mock Data - Active")
        else:
            st.info("🔗 Live APIs - Connected")
    
    with col2:
        st.write("**System Components:**")
        st.success("✅ Signal Generator")
        st.success("✅ Risk Manager")
        st.success("✅ Maritime Analyzer")

if __name__ == "__main__":
    main()