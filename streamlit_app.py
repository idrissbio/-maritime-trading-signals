#!/usr/bin/env python3
"""
Professional Maritime Trading Dashboard

Features:
- Real-time futures price updates (minute-by-minute)
- Live trade tracking with P&L monitoring
- Comprehensive win/loss statistics
- Professional UI/UX for serious traders
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
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Page config for professional look
st.set_page_config(
    page_title="Maritime Trading Intelligence - Professional",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Environment setup
try:
    from dotenv import load_dotenv
    load_dotenv()
    
    if hasattr(st, 'secrets'):
        os.environ['TWELVEDATA_API_KEY'] = st.secrets.get('TWELVEDATA_API_KEY', os.getenv('TWELVEDATA_API_KEY', ''))
        os.environ['DATALASTIC_API_KEY'] = st.secrets.get('DATALASTIC_API_KEY', os.getenv('DATALASTIC_API_KEY', ''))
        os.environ['MOCK_MODE'] = 'false'
        
except Exception as e:
    st.warning(f"Environment setup: {e}")

# Import components
try:
    from src.core.data_fetcher import DataFetcher
    from src.core.maritime_analyzer import MaritimeAnalyzer
    from src.core.signal_generator import SignalGenerator, SignalDirection, SignalTier
    from src.core.risk_manager import RiskManager, RiskParameters
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Professional CSS styling
st.markdown("""
<style>
    /* Professional color scheme */
    :root {
        --primary-color: #1f2937;
        --secondary-color: #374151;
        --accent-color: #3b82f6;
        --success-color: #10b981;
        --danger-color: #ef4444;
        --warning-color: #f59e0b;
    }
    
    /* Header styling */
    .professional-header {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Card styling */
    .trading-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Signal cards with status colors */
    .signal-active {
        border-left: 4px solid #3b82f6;
        background: #f8fafc;
    }
    .signal-winner {
        border-left: 4px solid #10b981;
        background: #f0fdf4;
    }
    .signal-loser {
        border-left: 4px solid #ef4444;
        background: #fef2f2;
    }
    .signal-pending {
        border-left: 4px solid #f59e0b;
        background: #fffbeb;
    }
    
    /* Price change indicators */
    .price-up {
        color: #10b981;
        font-weight: bold;
    }
    .price-down {
        color: #ef4444;
        font-weight: bold;
    }
    .price-neutral {
        color: #6b7280;
    }
    
    /* Professional metrics */
    .metric-container {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Status indicators */
    .status-live {
        background: #10b981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: bold;
    }
    
    .status-offline {
        background: #ef4444;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: bold;
    }
    
    /* Hide Streamlit elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Trade tracking data structures
class TradeStatus(Enum):
    ACTIVE = "ACTIVE"
    WINNER = "WINNER"
    LOSER = "LOSER"
    EXPIRED = "EXPIRED"

@dataclass
class TrackedTrade:
    signal_id: str
    symbol: str
    direction: str
    tier: int
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    position_size: int
    confidence: float
    reason: str
    entry_time: datetime
    expiry_time: datetime
    status: TradeStatus = TradeStatus.ACTIVE
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    current_price: Optional[float] = None

def initialize_professional_system():
    """Initialize professional trading system with trade tracking"""
    if 'professional_initialized' not in st.session_state:
        with st.spinner("🚀 Initializing Professional Trading System..."):
            try:
                # API validation
                twelve_data_key = os.getenv('TWELVEDATA_API_KEY')
                datalastic_key = os.getenv('DATALASTIC_API_KEY')
                
                if not twelve_data_key or not datalastic_key:
                    st.error("❌ Missing API keys. Please configure secrets.")
                    st.stop()
                
                # Initialize components
                st.session_state.data_fetcher = DataFetcher(
                    datalastic_key=datalastic_key,
                    twelve_data_key=twelve_data_key,
                    mock_mode=False
                )
                
                st.session_state.maritime_analyzer = MaritimeAnalyzer()
                st.session_state.signal_generator = SignalGenerator(
                    risk_per_trade=float(os.getenv('RISK_PER_TRADE', 0.01)),
                    min_confidence=0.55
                )
                
                risk_params = RiskParameters(
                    account_balance=float(os.getenv('ACCOUNT_BALANCE', 100000)),
                    risk_per_trade=float(os.getenv('RISK_PER_TRADE', 0.01)),
                    max_daily_trades=int(os.getenv('MAX_DAILY_TRADES', 15)),
                    max_positions=10,
                    max_correlated_positions=3
                )
                st.session_state.risk_manager = RiskManager(risk_params)
                
                # Initialize tracking data
                if 'tracked_trades' not in st.session_state:
                    st.session_state.tracked_trades = []
                if 'live_prices' not in st.session_state:
                    st.session_state.live_prices = {}
                if 'price_history' not in st.session_state:
                    st.session_state.price_history = {symbol: [] for symbol in ['CL1', 'HO1', 'HG1']}
                if 'last_price_update' not in st.session_state:
                    st.session_state.last_price_update = None
                
                st.session_state.professional_initialized = True
                st.success("✅ Professional System Ready")
                
            except Exception as e:
                st.error(f"❌ Initialization failed: {e}")
                st.stop()

def update_live_prices():
    """Update live futures prices with minute-by-minute tracking"""
    try:
        futures_symbols = ['CL1', 'HO1', 'HG1']
        current_time = datetime.now()
        
        for symbol in futures_symbols:
            try:
                # Get latest market data
                market_data = st.session_state.data_fetcher.get_market_data(symbol, "1min")
                
                if market_data and len(market_data) > 0:
                    latest = market_data[-1]
                    previous_price = st.session_state.live_prices.get(symbol, {}).get('price', latest.close)
                    
                    # Calculate price change
                    price_change = latest.close - previous_price
                    price_change_percent = (price_change / previous_price * 100) if previous_price != 0 else 0
                    
                    # Update live prices
                    st.session_state.live_prices[symbol] = {
                        'price': latest.close,
                        'change': price_change,
                        'change_percent': price_change_percent,
                        'volume': latest.volume,
                        'timestamp': latest.timestamp,
                        'high': latest.high,
                        'low': latest.low
                    }
                    
                    # Add to price history
                    st.session_state.price_history[symbol].append({
                        'timestamp': current_time,
                        'price': latest.close,
                        'volume': latest.volume
                    })
                    
                    # Keep only last 100 price points
                    if len(st.session_state.price_history[symbol]) > 100:
                        st.session_state.price_history[symbol] = st.session_state.price_history[symbol][-100:]
                
            except Exception as e:
                st.warning(f"Price update failed for {symbol}: {e}")
        
        st.session_state.last_price_update = current_time
        
    except Exception as e:
        st.error(f"Live price update failed: {e}")

def update_trade_tracking():
    """Update trade tracking based on current prices"""
    try:
        current_prices = st.session_state.live_prices
        current_time = datetime.now()
        
        for trade in st.session_state.tracked_trades:
            if trade.status != TradeStatus.ACTIVE:
                continue
                
            # Get current price for this symbol
            if trade.symbol in current_prices:
                current_price = current_prices[trade.symbol]['price']
                trade.current_price = current_price
                
                # Check if trade hit stop loss or targets
                if trade.direction.upper() == "LONG":
                    # Long position
                    if current_price <= trade.stop_loss:
                        # Hit stop loss
                        trade.status = TradeStatus.LOSER
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = current_time
                        trade.pnl = (trade.stop_loss - trade.entry_price) * trade.position_size
                        trade.pnl_percent = ((trade.stop_loss - trade.entry_price) / trade.entry_price) * 100
                        
                    elif current_price >= trade.target_1:
                        # Hit target
                        trade.status = TradeStatus.WINNER
                        trade.exit_price = trade.target_1
                        trade.exit_time = current_time
                        trade.pnl = (trade.target_1 - trade.entry_price) * trade.position_size
                        trade.pnl_percent = ((trade.target_1 - trade.entry_price) / trade.entry_price) * 100
                        
                else:
                    # Short position
                    if current_price >= trade.stop_loss:
                        # Hit stop loss
                        trade.status = TradeStatus.LOSER
                        trade.exit_price = trade.stop_loss
                        trade.exit_time = current_time
                        trade.pnl = (trade.entry_price - trade.stop_loss) * trade.position_size
                        trade.pnl_percent = ((trade.entry_price - trade.stop_loss) / trade.entry_price) * 100
                        
                    elif current_price <= trade.target_1:
                        # Hit target
                        trade.status = TradeStatus.WINNER
                        trade.exit_price = trade.target_1
                        trade.exit_time = current_time
                        trade.pnl = (trade.entry_price - trade.target_1) * trade.position_size
                        trade.pnl_percent = ((trade.entry_price - trade.target_1) / trade.entry_price) * 100
                
                # Check if trade expired
                if current_time > trade.expiry_time and trade.status == TradeStatus.ACTIVE:
                    trade.status = TradeStatus.EXPIRED
                    trade.exit_price = current_price
                    trade.exit_time = current_time
                    
                    # Calculate P&L at expiry
                    if trade.direction.upper() == "LONG":
                        trade.pnl = (current_price - trade.entry_price) * trade.position_size
                    else:
                        trade.pnl = (trade.entry_price - current_price) * trade.position_size
                    trade.pnl_percent = (trade.pnl / (trade.entry_price * trade.position_size)) * 100
                    
    except Exception as e:
        st.error(f"Trade tracking update failed: {e}")

def calculate_performance_metrics():
    """Calculate comprehensive performance statistics"""
    trades = st.session_state.tracked_trades
    
    if not trades:
        return {
            'total_trades': 0,
            'active_trades': 0,
            'completed_trades': 0,
            'winners': 0,
            'losers': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_winner': 0,
            'avg_loser': 0,
            'profit_factor': 0,
            'by_tier': {},
            'by_symbol': {}
        }
    
    # Basic counts
    total_trades = len(trades)
    active_trades = len([t for t in trades if t.status == TradeStatus.ACTIVE])
    completed_trades = len([t for t in trades if t.status in [TradeStatus.WINNER, TradeStatus.LOSER, TradeStatus.EXPIRED]])
    winners = len([t for t in trades if t.status == TradeStatus.WINNER])
    losers = len([t for t in trades if t.status == TradeStatus.LOSER])
    
    # Performance metrics
    win_rate = (winners / completed_trades * 100) if completed_trades > 0 else 0
    total_pnl = sum([t.pnl for t in trades if t.pnl is not None])
    
    # Winner/Loser analysis
    winner_pnls = [t.pnl for t in trades if t.status == TradeStatus.WINNER and t.pnl is not None]
    loser_pnls = [abs(t.pnl) for t in trades if t.status == TradeStatus.LOSER and t.pnl is not None]
    
    avg_winner = np.mean(winner_pnls) if winner_pnls else 0
    avg_loser = np.mean(loser_pnls) if loser_pnls else 0
    
    gross_profit = sum(winner_pnls) if winner_pnls else 0
    gross_loss = sum(loser_pnls) if loser_pnls else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    # Performance by tier
    by_tier = {}
    for tier in [1, 2, 3]:
        tier_trades = [t for t in trades if t.tier == tier]
        tier_completed = [t for t in tier_trades if t.status in [TradeStatus.WINNER, TradeStatus.LOSER, TradeStatus.EXPIRED]]
        tier_winners = len([t for t in tier_trades if t.status == TradeStatus.WINNER])
        
        by_tier[f'Tier {tier}'] = {
            'total': len(tier_trades),
            'completed': len(tier_completed),
            'winners': tier_winners,
            'win_rate': (tier_winners / len(tier_completed) * 100) if tier_completed else 0,
            'pnl': sum([t.pnl for t in tier_trades if t.pnl is not None])
        }
    
    # Performance by symbol
    by_symbol = {}
    symbols = set([t.symbol for t in trades])
    for symbol in symbols:
        symbol_trades = [t for t in trades if t.symbol == symbol]
        symbol_completed = [t for t in symbol_trades if t.status in [TradeStatus.WINNER, TradeStatus.LOSER, TradeStatus.EXPIRED]]
        symbol_winners = len([t for t in symbol_trades if t.status == TradeStatus.WINNER])
        
        by_symbol[symbol] = {
            'total': len(symbol_trades),
            'completed': len(symbol_completed),
            'winners': symbol_winners,
            'win_rate': (symbol_winners / len(symbol_completed) * 100) if symbol_completed else 0,
            'pnl': sum([t.pnl for t in symbol_trades if t.pnl is not None])
        }
    
    return {
        'total_trades': total_trades,
        'active_trades': active_trades,
        'completed_trades': completed_trades,
        'winners': winners,
        'losers': losers,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_winner': avg_winner,
        'avg_loser': avg_loser,
        'profit_factor': profit_factor,
        'by_tier': by_tier,
        'by_symbol': by_symbol
    }

def run_professional_analysis():
    """Run analysis and generate new signals"""
    try:
        st.session_state.system_status = "Running Analysis..."
        
        # Use proven working ports
        working_ports = ["singapore", "houston"]
        symbols = ["CL1", "HO1", "HG1"]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Collect maritime data
        vessel_positions = []
        port_congestion_data = []
        
        for i, port in enumerate(working_ports):
            status_text.text(f"🌊 Scanning {port}...")
            progress_bar.progress((i + 1) / len(working_ports) * 0.4)
            
            try:
                vessels = st.session_state.data_fetcher.get_vessel_positions(port, "all")
                vessel_positions.extend(vessels)
                congestion = st.session_state.data_fetcher._mock_port_congestion(port)
                port_congestion_data.append(congestion)
                time.sleep(2)  # Rate limiting
            except Exception as e:
                st.warning(f"API issue for {port}: {e}")
                vessels = st.session_state.data_fetcher._mock_vessel_positions(port, "all")
                vessel_positions.extend(vessels)
                congestion = st.session_state.data_fetcher._mock_port_congestion(port)
                port_congestion_data.append(congestion)
        
        # Collect market data
        market_data = {}
        volume_profiles = {}
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"📈 Fetching {symbol} futures data...")
            progress_bar.progress(0.4 + (i + 1) / len(symbols) * 0.3)
            
            try:
                market_data[symbol] = st.session_state.data_fetcher.get_market_data(symbol)
                volume_profiles[symbol] = st.session_state.data_fetcher.get_volume_profile(symbol)
            except Exception as e:
                st.warning(f"Market data issue for {symbol}: {e}")
                market_data[symbol] = st.session_state.data_fetcher._mock_market_data(symbol, "5min")
                volume_profiles[symbol] = st.session_state.data_fetcher._mock_volume_profile(symbol)
        
        # Maritime analysis
        status_text.text("🔍 Analyzing maritime events...")
        progress_bar.progress(0.8)
        
        maritime_events = st.session_state.maritime_analyzer.analyze_all_maritime_factors(
            port_congestion_data, vessel_positions, {}, {}
        )
        maritime_events = st.session_state.data_fetcher.apply_chokepoint_multipliers(maritime_events)
        
        # Generate signals
        status_text.text("🎯 Generating trading signals...")
        progress_bar.progress(0.9)
        
        signals = st.session_state.signal_generator.generate_signals(
            maritime_events, market_data, volume_profiles
        )
        
        # Apply risk management and add to tracking
        validated_signals = []
        for signal in signals:
            is_valid, reason = st.session_state.risk_manager.validate_signal(signal)
            if is_valid:
                signal.position_size = st.session_state.risk_manager.calculate_position_size(signal)
                if signal.position_size > 0:
                    validated_signals.append(signal)
                    
                    # Add to trade tracking
                    tracked_trade = TrackedTrade(
                        signal_id=f"{signal.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        symbol=signal.symbol,
                        direction=signal.direction.value,
                        tier=signal.tier.value,
                        entry_price=signal.entry_price,
                        stop_loss=signal.stop_loss,
                        target_1=signal.target_1,
                        target_2=signal.target_2,
                        position_size=signal.position_size,
                        confidence=signal.confidence_score,
                        reason=signal.reason,
                        entry_time=datetime.now(),
                        expiry_time=signal.expiry_time
                    )
                    
                    st.session_state.tracked_trades.append(tracked_trade)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete!")
        
        return validated_signals, maritime_events
        
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        return [], []

def show_live_prices_panel():
    """Display live futures prices with real-time updates"""
    st.markdown("### 📊 Live Futures Prices")
    
    if not st.session_state.live_prices:
        st.info("Click 'Update Prices' to fetch live futures data")
        return
    
    cols = st.columns(len(st.session_state.live_prices))
    
    for i, (symbol, price_data) in enumerate(st.session_state.live_prices.items()):
        with cols[i]:
            # Price change indicator
            if price_data['change'] > 0:
                change_class = "price-up"
                change_icon = "▲"
            elif price_data['change'] < 0:
                change_class = "price-down"
                change_icon = "▼"
            else:
                change_class = "price-neutral"
                change_icon = "●"
            
            st.markdown(f"""
            <div class="trading-card">
                <h4>{symbol}</h4>
                <h2>${price_data['price']:.2f}</h2>
                <p class="{change_class}">
                    {change_icon} {price_data['change']:+.2f} ({price_data['change_percent']:+.2f}%)
                </p>
                <small>Volume: {price_data['volume']:,}</small><br>
                <small>Updated: {price_data['timestamp'].strftime('%H:%M:%S')}</small>
            </div>
            """, unsafe_allow_html=True)

def show_performance_dashboard():
    """Display comprehensive performance statistics"""
    st.markdown("### 📈 Trading Performance")
    
    metrics = calculate_performance_metrics()
    
    # Overall performance metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Trades", metrics['total_trades'])
    
    with col2:
        st.metric("Active Trades", metrics['active_trades'])
        
    with col3:
        win_rate_color = "normal"
        if metrics['win_rate'] >= 70:
            win_rate_color = "success"
        elif metrics['win_rate'] >= 50:
            win_rate_color = "warning"
        else:
            win_rate_color = "error"
            
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
    
    with col4:
        pnl_delta = f"{metrics['total_pnl']:+.2f}" if metrics['total_pnl'] != 0 else None
        st.metric("Total P&L", f"${metrics['total_pnl']:.2f}", delta=pnl_delta)
    
    with col5:
        if metrics['profit_factor'] == float('inf'):
            pf_display = "∞"
        else:
            pf_display = f"{metrics['profit_factor']:.2f}"
        st.metric("Profit Factor", pf_display)
    
    # Detailed breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Performance by Tier")
        tier_data = []
        for tier, data in metrics['by_tier'].items():
            tier_data.append({
                'Tier': tier,
                'Trades': data['total'],
                'Win Rate': f"{data['win_rate']:.1f}%",
                'P&L': f"${data['pnl']:.2f}"
            })
        
        if tier_data:
            df_tier = pd.DataFrame(tier_data)
            st.dataframe(df_tier, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("#### Performance by Symbol")
        symbol_data = []
        for symbol, data in metrics['by_symbol'].items():
            symbol_data.append({
                'Symbol': symbol,
                'Trades': data['total'],
                'Win Rate': f"{data['win_rate']:.1f}%",
                'P&L': f"${data['pnl']:.2f}"
            })
        
        if symbol_data:
            df_symbol = pd.DataFrame(symbol_data)
            st.dataframe(df_symbol, use_container_width=True, hide_index=True)

def show_trade_tracking():
    """Display comprehensive trade tracking"""
    st.markdown("### 🎯 Trade Tracking")
    
    if not st.session_state.tracked_trades:
        st.info("No trades tracked yet. Generate signals to start tracking.")
        return
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["All", "Active", "Winners", "Losers", "Expired"],
            key="status_filter"
        )
    
    with col2:
        symbol_filter = st.selectbox(
            "Filter by Symbol",
            ["All"] + list(set([t.symbol for t in st.session_state.tracked_trades])),
            key="symbol_filter"
        )
    
    with col3:
        tier_filter = st.selectbox(
            "Filter by Tier",
            ["All", "Tier 1", "Tier 2", "Tier 3"],
            key="tier_filter"
        )
    
    # Apply filters
    filtered_trades = st.session_state.tracked_trades
    
    if status_filter != "All":
        if status_filter == "Active":
            filtered_trades = [t for t in filtered_trades if t.status == TradeStatus.ACTIVE]
        elif status_filter == "Winners":
            filtered_trades = [t for t in filtered_trades if t.status == TradeStatus.WINNER]
        elif status_filter == "Losers":
            filtered_trades = [t for t in filtered_trades if t.status == TradeStatus.LOSER]
        elif status_filter == "Expired":
            filtered_trades = [t for t in filtered_trades if t.status == TradeStatus.EXPIRED]
    
    if symbol_filter != "All":
        filtered_trades = [t for t in filtered_trades if t.symbol == symbol_filter]
    
    if tier_filter != "All":
        tier_num = int(tier_filter.split()[-1])
        filtered_trades = [t for t in filtered_trades if t.tier == tier_num]
    
    # Display trades
    for trade in filtered_trades[-20:]:  # Show last 20 trades
        status_class = f"signal-{trade.status.value.lower()}"
        if trade.status == TradeStatus.ACTIVE:
            status_class = "signal-active"
        elif trade.status == TradeStatus.WINNER:
            status_class = "signal-winner"
        elif trade.status == TradeStatus.LOSER:
            status_class = "signal-loser"
        else:
            status_class = "signal-pending"
        
        with st.container():
            st.markdown(f'<div class="trading-card {status_class}">', unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                direction_emoji = "📈" if trade.direction == "LONG" else "📉"
                status_emoji = {
                    TradeStatus.ACTIVE: "🔄",
                    TradeStatus.WINNER: "✅", 
                    TradeStatus.LOSER: "❌",
                    TradeStatus.EXPIRED: "⏱️"
                }[trade.status]
                
                st.markdown(f"**{direction_emoji} {trade.symbol} - Tier {trade.tier} {status_emoji}**")
                st.write(f"Entry: ${trade.entry_price:.2f} | Stop: ${trade.stop_loss:.2f} | Target: ${trade.target_1:.2f}")
                
                if trade.current_price:
                    price_color = "price-up" if trade.current_price > trade.entry_price else "price-down"
                    st.markdown(f'Current: <span class="{price_color}">${trade.current_price:.2f}</span>', unsafe_allow_html=True)
            
            with col2:
                st.metric("Confidence", f"{trade.confidence:.1%}")
                st.write(f"Size: {trade.position_size}")
            
            with col3:
                if trade.pnl != 0:
                    pnl_color = "success" if trade.pnl > 0 else "error"
                    st.metric("P&L", f"${trade.pnl:.2f}", delta=f"{trade.pnl_percent:.1f}%")
                else:
                    st.metric("P&L", "Pending")
            
            with col4:
                st.write(f"**Entry:** {trade.entry_time.strftime('%m/%d %H:%M')}")
                if trade.exit_time:
                    st.write(f"**Exit:** {trade.exit_time.strftime('%m/%d %H:%M')}")
                else:
                    st.write(f"**Expires:** {trade.expiry_time.strftime('%m/%d %H:%M')}")
            
            st.write(f"**Reason:** {trade.reason}")
            st.markdown('</div>', unsafe_allow_html=True)
            st.divider()

def main():
    # Initialize system
    initialize_professional_system()
    
    # Professional header
    st.markdown("""
    <div class="professional-header">
        <h1>📊 Maritime Trading Intelligence</h1>
        <h3>Professional Futures Trading Dashboard</h3>
        <p>Real-time Price Tracking • Trade Performance Analytics • Maritime Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Trading Control")
        
        # System status
        status = st.session_state.get('system_status', 'Ready')
        if status == "Live":
            st.markdown('<span class="status-live">🟢 LIVE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-offline">🔴 OFFLINE</span>', unsafe_allow_html=True)
        
        st.divider()
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Update Prices", use_container_width=True):
                with st.spinner("Updating live prices..."):
                    update_live_prices()
                    update_trade_tracking()
                    st.success("Prices updated!")
        
        with col2:
            if st.button("🎯 Generate Signals", use_container_width=True):
                with st.spinner("Generating signals..."):
                    signals, events = run_professional_analysis()
                    if signals:
                        st.success(f"✅ {len(signals)} signals generated!")
                        st.balloons()
                    else:
                        st.info("📊 No signals generated")
        
        # Auto-refresh option
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)")
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        st.divider()
        
        # Performance summary
        if st.session_state.tracked_trades:
            metrics = calculate_performance_metrics()
            st.markdown("### 📈 Quick Stats")
            st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
            st.metric("Total P&L", f"${metrics['total_pnl']:.2f}")
            st.metric("Active Trades", metrics['active_trades'])
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Live Prices & Performance",
        "🎯 Trade Tracking", 
        "📈 Charts & Analysis"
    ])
    
    with tab1:
        show_live_prices_panel()
        st.divider()
        show_performance_dashboard()
    
    with tab2:
        show_trade_tracking()
    
    with tab3:
        # Price charts
        if st.session_state.price_history:
            st.markdown("### 📈 Price Charts")
            
            # Create price chart
            fig = go.Figure()
            
            for symbol, history in st.session_state.price_history.items():
                if history:
                    df = pd.DataFrame(history)
                    fig.add_trace(go.Scatter(
                        x=df['timestamp'],
                        y=df['price'],
                        mode='lines',
                        name=symbol,
                        line=dict(width=2)
                    ))
            
            fig.update_layout(
                title="Live Futures Prices",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                height=500,
                showlegend=True,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Update prices to see charts")

if __name__ == "__main__":
    main()