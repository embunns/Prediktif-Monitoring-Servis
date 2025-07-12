import streamlit as st
import pandas as pd
import io
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from preprocessing import get_data_summary

def show_page():
    st.markdown('<div class="section-header">📋 Results & Export</div>', unsafe_allow_html=True)
    
    if st.session_state.preprocessed_data is not None:
        df = st.session_state.preprocessed_data
        
        st.subheader("Analysis Summary Report")
        
        summary = get_data_summary(df)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records Processed", summary['total_records'])
            st.metric("Unique Work Orders", summary['unique_work_orders'])
            st.metric("Unique Materials", summary['unique_materials'])
        
        with col2:
            st.metric("JOBS Machine Usage", summary['machine_types'].get('JOBS', 0))
            st.metric("CRAFT Machine Usage", summary['machine_types'].get('CRAFT', 0))
            st.metric("Average Price", f"${summary['price_stats']['mean']:.2f}")
        
        with col3:
            st.metric("Max Price", f"${summary['price_stats']['max']:.2f}")
            st.metric("Average Quantity", f"{summary['qty_stats']['mean']:.2f}")
            st.metric("Max Quantity", f"{summary['qty_stats']['max']:.0f}")
        
        st.subheader("Key Insights")
        
        top_materials = df['MatName'].value_counts().head(5)
        st.write("**Top 5 Most Used Materials:**")
        for i, (material, count) in enumerate(top_materials.items(), 1):
            st.write(f"{i}. {material}: {count} times")
        
        expensive_materials = df[df['Price'] > 0].nlargest(5, 'Price')[['MatName', 'Price', 'MachineType']]
        st.write("**Top 5 Most Expensive Materials:**")
        st.dataframe(expensive_materials.reset_index(drop=True))
        
        machine_analysis = df.groupby('MachineType').agg({
            'QtyOut': 'mean',
            'Price': 'mean',
            'MatName': 'count'
        }).round(2)
        machine_analysis.columns = ['Avg Quantity', 'Avg Price', 'Total Usage']
        
        st.write("**Machine Type Analysis:**")
        st.dataframe(machine_analysis)
        
        if hasattr(st.session_state.model, 'classification_results') and st.session_state.model.classification_results:
            st.subheader("Model Performance Summary")
            
            classification_results = st.session_state.model.classification_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Classification Model:**")
                st.metric("Accuracy", f"{classification_results['accuracy']:.3f}")
                st.metric("CV Mean", f"{classification_results['cv_mean']:.3f}")
            
            if hasattr(st.session_state.model, 'regression_results') and st.session_state.model.regression_results:
                regression_results = st.session_state.model.regression_results
                
                with col2:
                    st.write("**Regression Model:**")
                    st.metric("RF R² Score", f"{regression_results['random_forest']['r2_score']:.3f}")
                    st.metric("LR R² Score", f"{regression_results['linear_regression']['r2_score']:.3f}")
        
        st.subheader("Export Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Preprocessed Data"):
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()
                
                st.download_button(
                    label="Download CSV",
                    data=csv_string,
                    file_name="preprocessed_material_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("Export Summary Report"):
                report_data = {
                    'metric': ['Total Records', 'Unique Work Orders', 'Unique Materials', 
                             'JOBS Usage', 'CRAFT Usage', 'Avg Price', 'Max Price'],
                    'value': [summary['total_records'], summary['unique_work_orders'], 
                            summary['unique_materials'], summary['machine_types'].get('JOBS', 0),
                            summary['machine_types'].get('CRAFT', 0), summary['price_stats']['mean'],
                            summary['price_stats']['max']]
                }
                
                report_df = pd.DataFrame(report_data)
                csv_buffer = io.StringIO()
                report_df.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()
                
                st.download_button(
                    label="Download Report",
                    data=csv_string,
                    file_name="material_analysis_report.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("Export Top Materials"):
                top_materials_df = df['MatName'].value_counts().reset_index()
                top_materials_df.columns = ['Material', 'Usage_Count']
                
                csv_buffer = io.StringIO()
                top_materials_df.to_csv(csv_buffer, index=False)
                csv_string = csv_buffer.getvalue()
                
                st.download_button(
                    label="Download Top Materials",
                    data=csv_string,
                    file_name="top_materials.csv",
                    mime="text/csv"
                )
        
        st.subheader("Advanced Data Filtering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            machine_filter = st.selectbox("Filter by Machine Type", 
                                        ['All'] + list(df['MachineType'].unique()))
            
            if machine_filter != 'All':
                filtered_df = df[df['MachineType'] == machine_filter]
            else:
                filtered_df = df
        
        with col2:
            min_price = st.number_input("Minimum Price", min_value=0.0, value=0.0)
            filtered_df = filtered_df[filtered_df['Price'] >= min_price]
        
        st.write(f"Filtered data: {len(filtered_df)} records")
        st.dataframe(filtered_df[['WorkOrderNo', 'ProblemDesc', 'MatName', 'QtyOut', 'Price', 'MachineType']].head(10))
        
    else:
        st.warning("Please upload and preprocess data first!")