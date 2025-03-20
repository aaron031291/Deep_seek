#!/usr/bin/env python3
import streamlit as st
import json

class ControlPanel:
    def __init__(self):
        self.config = self.load_config()
        
    def render(self):
        st.title("AI System Control Panel")
        
        with st.sidebar:
            st.header("System Controls")
            st.toggle("Enable Cloud Processing", value=True)
            st.toggle("Enable Edge Processing", value=True)
            
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Risk Assessment")
            risk_score = st.slider("Risk Threshold", 0, 100, 75)
            
        with col2:
            st.subheader("Approval Status")
            st.metric("Current Approval Rate", "95%")
            
    def load_config(self):
        with open('config.json', 'r') as f:
            return json.load(f)
