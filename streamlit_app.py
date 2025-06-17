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

# Import and run the dashboard
from dashboard.app import main

if __name__ == "__main__":
    main()