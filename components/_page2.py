import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from visualization import plot_material_frequency, plot_advanced_analytics

def show_page():
    st.markdown('<div class="section-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.preprocessed_data is not None:
        df = st.session_state.preprocessed_data
        
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("JOBS Machine Records", len(df[df['MachineType'] == 'JOBS']))
        with col2:
            st.metric("CRAFT Machine Records", len(df[df['MachineType'] == 'CRAFT']))
        with col3:
            st.metric("Average Price", f"${df['Price'].mean():.2f}")
        
        st.subheader("Material Analysis Dashboard")
        fig = plot_material_frequency(df)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Most Frequently Used Materials")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**JOBS Machine Materials**")
            jobs_materials = df[df['MachineType'] == 'JOBS']['MatName'].value_counts().head(10)
            st.bar_chart(jobs_materials)
        
        with col2:
            st.write("**CRAFT Machine Materials**")
            craft_materials = df[df['MachineType'] == 'CRAFT']['MatName'].value_counts().head(10)
            st.bar_chart(craft_materials)
        
        st.subheader("Price Analysis")
        
        price_data = df[df['Price'] > 0]
        fig_price = px.histogram(price_data, x='Price', nbins=30, title='Price Distribution (Price > 0)')
        st.plotly_chart(fig_price, use_container_width=True)
        
        fig_price_machine = px.box(price_data, x='MachineType', y='Price', 
                                 title='Price Distribution by Machine Type')
        st.plotly_chart(fig_price_machine, use_container_width=True)
        
        st.subheader("Quantity Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_qty = px.histogram(df, x='QtyOut', nbins=20, title='Quantity Distribution')
            st.plotly_chart(fig_qty, use_container_width=True)
        
        with col2:
            qty_by_machine = df.groupby('MachineType')['QtyOut'].mean().reset_index()
            fig_qty_machine = px.bar(qty_by_machine, x='MachineType', y='QtyOut', 
                                   title='Average Quantity by Machine Type')
            st.plotly_chart(fig_qty_machine, use_container_width=True)
        
        st.subheader("Time Series Analysis")
        
        if 'TransOutDate' in df.columns and df['TransOutDate'].notna().any():
            df['YearMonth'] = df['TransOutDate'].dt.to_period('M')
            monthly_trend = df.groupby('YearMonth').size().reset_index(name='count')
            monthly_trend['YearMonth'] = monthly_trend['YearMonth'].astype(str)
            
            fig_trend = px.line(monthly_trend, x='YearMonth', y='count', 
                              title='Monthly Material Usage Trend')
            fig_trend.update_xaxes(tickangle=45)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        st.subheader("Problem Description Analysis")
        
        all_problems = ' '.join(df['ProblemDesc_Clean'].fillna('').astype(str))
        words = all_problems.split()
        word_freq = pd.Series(words).value_counts().head(20)
        
        fig_words = px.bar(x=word_freq.index, y=word_freq.values, 
                         title='Top 20 Words in Problem Descriptions')
        fig_words.update_xaxes(tickangle=45)
        st.plotly_chart(fig_words, use_container_width=True)
        
        st.subheader("Advanced Analytics")
        fig_advanced = plot_advanced_analytics(df)
        st.plotly_chart(fig_advanced, use_container_width=True)
        
    else:
        st.warning("Please upload and preprocess data first!")