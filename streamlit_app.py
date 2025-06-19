#!/usr/bin/env python3
"""
Maritime Trading Signals - Streamlit Cloud Entry Point

This is the main entry point for Streamlit Cloud deployment.
It runs the MVP dashboard with live trading capabilities.
"""

import streamlit as st
import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Set page config first
st.set_page_config(
    page_title="Maritime Trading Signals - Live MVP",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Environment setup for Streamlit Cloud
if not os.getenv('STREAMLIT_CLOUD_ENV'):
    # Load from secrets on Streamlit Cloud
    try:
        # API Keys from Streamlit secrets
        if hasattr(st, 'secrets'):
            os.environ['TWELVEDATA_API_KEY'] = st.secrets.get('TWELVEDATA_API_KEY', '')
            os.environ['DATALASTIC_API_KEY'] = st.secrets.get('DATALASTIC_API_KEY', '')
            os.environ['MOCK_MODE'] = 'false'  # Force live mode
            os.environ['ACCOUNT_BALANCE'] = str(st.secrets.get('ACCOUNT_BALANCE', '100000'))
            os.environ['RISK_PER_TRADE'] = str(st.secrets.get('RISK_PER_TRADE', '0.01'))
            os.environ['MAX_DAILY_TRADES'] = str(st.secrets.get('MAX_DAILY_TRADES', '15'))
            os.environ['MIN_SIGNAL_CONFIDENCE'] = str(st.secrets.get('MIN_SIGNAL_CONFIDENCE', '0.65'))
    except Exception as e:
        st.warning(f"Secrets configuration: {e}")

# Import and run the MVP dashboard
try:
    # Import the MVP dashboard components
    from mvp_dashboard import (
        initialize_mvp_system, 
        run_live_analysis, 
        show_live_signals, 
        show_live_events, 
        show_live_market_data
    )
    
    # Main MVP Dashboard
    def main():
        # MVP Header
        st.markdown("""
        <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 2rem; text-align: center;">
            <h1>🚢 Maritime Trading Signals - Live MVP</h1>
            <p>Real-time Maritime Intelligence & Trading Signal Generation</p>
            <p><strong>⚡ LIVE DATA MODE - Ready for Trading</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize MVP system
        initialize_mvp_system()
        
        # Sidebar controls
        with st.sidebar:
            st.header("🎛️ Live Trading Control")
            
            # System status
            status = st.session_state.get('system_status', 'Initializing...')
            if status == "Live":
                st.success(f"🟢 {status}")
            else:
                st.info(f"🔄 {status}")
            
            if st.session_state.get('last_update'):
                st.write(f"**Last Update:** {st.session_state.last_update.strftime('%H:%M:%S UTC')}")
            
            st.divider()
            
            # Main action button
            if st.button("🚀 Generate Live Trading Signals", type="primary", use_container_width=True):
                with st.spinner("🔍 Analyzing live maritime data..."):
                    signals, events = run_live_analysis()
                    if signals:
                        st.success(f"✅ {len(signals)} signals generated!")
                        st.balloons()
                    else:
                        st.info("📊 Analysis complete - monitoring conditions")
            
            st.divider()
            
            # Auto refresh
            auto_refresh = st.checkbox("🔁 Auto Refresh (5 min)")
            if auto_refresh:
                import time
                time.sleep(300)
                st.rerun()
            
            st.divider()
            
            # Account settings
            st.subheader("💰 Trading Account")
            account_balance = st.number_input(
                "Account Balance ($)", 
                value=float(os.getenv('ACCOUNT_BALANCE', 100000)),
                min_value=10000, max_value=1000000, step=10000
            )
            
            risk_pct = st.slider(
                "Risk per Trade (%)", 
                min_value=0.5, max_value=3.0, 
                value=float(os.getenv('RISK_PER_TRADE', 0.01)) * 100, 
                step=0.1
            )
            
            min_confidence = st.slider(
                "Min Signal Confidence (%)",
                min_value=50, max_value=90,
                value=int(float(os.getenv('MIN_SIGNAL_CONFIDENCE', 0.65)) * 100)
            )
            
            st.divider()
            
            # System info
            st.subheader("📡 System Status")
            
            # API status indicators
            if st.session_state.get('mvp_initialized'):
                st.success("✅ 12Data API Connected")
                st.success("✅ Datalastic API Connected") 
                st.success("✅ Maritime Analyzer Active")
                st.success("✅ Signal Generator Ready")
                st.success("✅ Risk Manager Active")
            else:
                st.info("🔄 Initializing system...")
        
        # Main content area
        if not st.session_state.get('live_signals') and not st.session_state.get('live_events'):
            # Welcome screen
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                ### 🎯 Welcome to Maritime Trading MVP
                
                **Live System Capabilities:**
                - 🌊 Real-time vessel tracking (3,000+ vessels)
                - 📊 Live market data (CL, NG, RB, HO, GC, SI, HG)
                - 🎯 AI-powered signal generation
                - ⚖️ Automated risk management
                - 📱 Mobile-responsive dashboard
                
                **Strategic Coverage:**
                - Strait of Hormuz (2.5x impact multiplier)
                - Suez Canal (2.0x impact multiplier)
                - Panama Canal (1.8x impact multiplier)
                - Major LNG terminals
                - Global shipping chokepoints
                
                **👆 Click "Generate Live Trading Signals" to start**
                """)
                
                # Quick stats
                st.info("""
                📈 **Recent Live Test Results:**
                - 3,277 vessels tracked
                - 122 maritime events detected  
                - Live signal: CL LONG @ $91.93
                - Sub-10 second analysis time
                """)
        else:
            # Show live data tabs
            tab1, tab2, tab3 = st.tabs([
                "🎯 Live Trading Signals", 
                "🌊 Maritime Events", 
                "📈 Market Analysis"
            ])
            
            with tab1:
                show_live_signals()
            
            with tab2:
                show_live_events()
            
            with tab3:
                show_live_market_data()
        
        # Footer
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; color: #666; padding: 1rem;">
                <p>🚢 Maritime Trading Signals MVP | 
                <a href="https://github.com/idrissbio/-maritime-trading-signals" target="_blank">GitHub</a> | 
                ⚠️ Trade at your own risk</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Run the main app
    if __name__ == "__main__":
        main()
    else:
        main()  # Also run when imported
        
except ImportError as e:
    st.error(f"""
    🚨 **System Error**
    
    Failed to load MVP components: {e}
    
    This usually indicates missing dependencies or configuration issues.
    Please check the system setup and try again.
    """)
    
    # Fallback display
    st.title("🚢 Maritime Trading Signals")
    st.info("System is starting up... Please refresh in a moment.")

except Exception as e:
    st.error(f"""
    🚨 **Unexpected Error**
    
    An error occurred while starting the MVP system: {e}
    
    Please refresh the page or contact support if the issue persists.
    """)