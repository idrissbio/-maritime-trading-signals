#!/usr/bin/env python3
"""
Maritime Trading Signals - Streamlit App Entry Point
This is the main entry point for Streamlit Cloud deployment
"""

import streamlit as st
import sys
import os

# Add src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

# Handle missing dependencies gracefully
try:
    # Import and run the dashboard
    from dashboard.app import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    st.error(f"""
    🚨 **Dependency Error**
    
    Missing required packages for the maritime trading system.
    
    Error: {e}
    
    This usually happens when optional dependencies are missing.
    The system will continue with limited functionality.
    """)
    
    # Fallback simple dashboard
    st.title("🚢 Maritime Trading Signals")
    st.info("System is starting up... Please refresh in a moment.")
    
except Exception as e:
    st.error(f"""
    🚨 **System Error**
    
    An error occurred while starting the maritime trading system.
    
    Error: {e}
    
    Please try refreshing the page or contact support.
    """)