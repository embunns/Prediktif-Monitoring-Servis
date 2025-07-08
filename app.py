import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import sys
import os

# Add scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

# Import custom modules
from preprocessing import preprocess_data, create_feature_columns, get_data_summary
from model import MaterialPredictionModel

# Configure page
st.set_page_config(
    page_title="Material Requirement Analysis",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    """
    Create sample data untuk testing
    """
    sample_data = {
        'WorkOrderNo': ['17RFM034931', '15RFM029301', '15RFM029584', 'MWO-1412-00954', 'MWO-1412-00954', '20RFM041056'],
        'ProblemDesc': ['AMPLITUD TO HIGH ENCODER', 'PINTU TROUBLE', 'MOTOR COOLANT KEBAKAR', '#N/A', '#N/A', 'C ENCODER ERROR'],
        'StockNo': ['A01M127', 'C07H050', 'A02M077', 'A02RM001', 'A02RM017', 'A07E064'],
        'MatName': ['RING, O', 'Valve', 'Valve', 'ANGLE BAR', 'ANGLE BAR', 'Timer'],
        'Specification': ['551.10.148', 'WM-781-946/MCH-3-1/8', 'ART-NR-280.1020.2', 'STANDAR SII COMM-STEEL/#5X50X50X6000MM', 'COMM-STEEL/#6X60X60X6000MM', 'H3Y-2/24VDC/5A'],
        'QtyOut': [2, 1, 1, 1, 3, 1],
        'UOM': ['EA', 'EA', 'EA', 'EA', 'EA', 'EA'],
        'Codification': ['AABG01', 'TFAK01', 'AABG01', 'TFAK01', 'TFAK01', 'AABG01'],
        'MaterialTicketNo': ['MMT-1704-00726', 'MMT-1507-01258', 'MMT-1512-02142', 'MMT-1501-00001', 'MMT-1501-00001', 'MMT-2003-00338'],
        'TransOutDate': ['21/04/2017', '27/07/2015', '23/12/2015', '05/01/2015', '05/01/2015', '19/03/2020'],
        'Currency': ['DEM', 'CHF', 'DEM', 'IDR', 'IDR', 'IDR'],
        'Price': [0.05, 152, 15, 250000, 325000, 155000],
        'MovingRate': ['FM', 'SM', 'SM', 'SM', 'FM', 'SM']
    }
    
    return pd.DataFrame(sample_data)

def plot_material_frequency(df):
    """
    Plot frekuensi penggunaan material berdasarkan jenis mesin
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Material Frequency by Machine Type', 'Top 10 Materials Overall', 
                       'Machine Type Distribution', 'Price Distribution'),
        specs=[[{"colspan": 2}, None],
               [{"type": "pie"}, {"type": "histogram"}]]
    )
    
    # 1. Material frequency by machine type
    material_by_machine = df.groupby(['MachineType', 'MatName']).size().reset_index(name='count')
    
    for machine_type in material_by_machine['MachineType'].unique():
        machine_data = material_by_machine[material_by_machine['MachineType'] == machine_type]
        top_materials = machine_data.nlargest(10, 'count')
        
        fig.add_trace(
            go.Bar(
                x=top_materials['MatName'],
                y=top_materials['count'],
                name=machine_type,
                text=top_materials['count'],
                textposition='auto'
            ),
            row=1, col=1
        )
    
    # 2. Machine type distribution (pie)
    machine_counts = df['MachineType'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=machine_counts.index,
            values=machine_counts.values,
            name="Machine Types"
        ),
        row=2, col=1
    )
    
    # 3. Price distribution (histogram)
    price_data = df[df['Price'] > 0]['Price']
    fig.add_trace(
        go.Histogram(
            x=price_data,
            nbinsx=30,
            name="Price Distribution"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Material Analysis Dashboard"
    )
    
    return fig

def main():
    st.markdown('<div class="main-header">🔧 Material Requirement Analysis System</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Upload & Preprocessing", "Exploratory Data Analysis", "Model Training & Prediction", "Results & Export"]
    )
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    if 'model' not in st.session_state:
        st.session_state.model = MaterialPredictionModel()
    
    # Page 1: Data Upload & Preprocessing
    if page == "Data Upload & Preprocessing":
        st.markdown('<div class="section-header">📊 Data Upload & Preprocessing</div>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        # Option to use sample data
        if st.button("Use Sample Data"):
            st.session_state.data = create_sample_data()
            st.success("Sample data loaded successfully!")
        
        if uploaded_file is not None:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        if st.session_state.data is not None:
            st.subheader("Raw Data Preview")
            st.dataframe(st.session_state.data.head())
            
            st.subheader("Data Info")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Records", len(st.session_state.data))
                st.metric("Total Columns", len(st.session_state.data.columns))
            
            with col2:
                st.metric("Missing Values", st.session_state.data.isnull().sum().sum())
                st.metric("Duplicate Records", st.session_state.data.duplicated().sum())
            
            # Preprocessing
            if st.button("Start Preprocessing"):
                with st.spinner("Processing data..."):
                    # Basic preprocessing
                    st.session_state.preprocessed_data = preprocess_data(st.session_state.data)
                    
                    # Create additional features
                    st.session_state.preprocessed_data = create_feature_columns(st.session_state.preprocessed_data)
                    
                st.success("Data preprocessing completed!")
                
                # Show preprocessing results
                st.subheader("Preprocessed Data Preview")
                st.dataframe(st.session_state.preprocessed_data.head())
                
                # Summary statistics
                summary = get_data_summary(st.session_state.preprocessed_data)
                
                st.subheader("Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Records", summary['total_records'])
                with col2:
                    st.metric("Unique Work Orders", summary['unique_work_orders'])
                with col3:
                    st.metric("Unique Materials", summary['unique_materials'])
                with col4:
                    st.metric("Date Range", f"{summary['date_range']['start'].strftime('%Y-%m-%d') if summary['date_range']['start'] else 'N/A'} to {summary['date_range']['end'].strftime('%Y-%m-%d') if summary['date_range']['end'] else 'N/A'}")
    
    # Page 2: Exploratory Data Analysis
    elif page == "Exploratory Data Analysis":
        st.markdown('<div class="section-header">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
        
        if st.session_state.preprocessed_data is not None:
            df = st.session_state.preprocessed_data
            
            # Basic statistics
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("JOBS Machine Records", len(df[df['MachineType'] == 'JOBS']))
            with col2:
                st.metric("CRAFT Machine Records", len(df[df['MachineType'] == 'CRAFT']))
            with col3:
                st.metric("Average Price", f"${df['Price'].mean():.2f}")
            
            # Visualizations
            st.subheader("Material Analysis Dashboard")
            fig = plot_material_frequency(df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Top materials per machine type
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
            
            # Price analysis
            st.subheader("Price Analysis")
            
            # Price histogram
            price_data = df[df['Price'] > 0]
            fig_price = px.histogram(price_data, x='Price', nbins=30, title='Price Distribution (Price > 0)')
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Price by machine type
            fig_price_machine = px.box(price_data, x='MachineType', y='Price', 
                                     title='Price Distribution by Machine Type')
            st.plotly_chart(fig_price_machine, use_container_width=True)
            
            # Quantity analysis
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
            
            # Time series analysis
            st.subheader("Time Series Analysis")
            
            if 'TransOutDate' in df.columns and df['TransOutDate'].notna().any():
                # Monthly trend
                df['YearMonth'] = df['TransOutDate'].dt.to_period('M')
                monthly_trend = df.groupby('YearMonth').size().reset_index(name='count')
                monthly_trend['YearMonth'] = monthly_trend['YearMonth'].astype(str)
                
                fig_trend = px.line(monthly_trend, x='YearMonth', y='count', 
                                  title='Monthly Material Usage Trend')
                fig_trend.update_xaxes(tickangle=45)
                st.plotly_chart(fig_trend, use_container_width=True)
            
            # Problem description analysis
            st.subheader("Problem Description Analysis")
            
            # Word cloud alternative - top words
            all_problems = ' '.join(df['ProblemDesc_Clean'].fillna('').astype(str))
            words = all_problems.split()
            word_freq = pd.Series(words).value_counts().head(20)
            
            fig_words = px.bar(x=word_freq.index, y=word_freq.values, 
                             title='Top 20 Words in Problem Descriptions')
            fig_words.update_xaxes(tickangle=45)
            st.plotly_chart(fig_words, use_container_width=True)
            
        else:
            st.warning("Please upload and preprocess data first!")
    
    # Page 3: Model Training & Prediction
    elif page == "Model Training & Prediction":
        st.markdown('<div class="section-header">🤖 Model Training & Prediction</div>', unsafe_allow_html=True)
        
        if st.session_state.preprocessed_data is not None:
            df = st.session_state.preprocessed_data
            
            # Model training section
            st.subheader("Model Training")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Train Material Classification Model"):
                    with st.spinner("Training classification model..."):
                        results = st.session_state.model.train_material_classifier(df)
                        
                        if 'error' not in results:
                            st.success("Classification model trained successfully!")
                            
                            # Show results
                            st.write("**Model Performance:**")
                            st.metric("Accuracy", f"{results['accuracy']:.3f}")
                            st.metric("Cross-validation Mean", f"{results['cv_mean']:.3f}")
                            st.metric("Cross-validation Std", f"{results['cv_std']:.3f}")
                            
                            # Classification report
                            st.text("Classification Report:")
                            st.code(results['classification_report'])
                        else:
                            st.error(results['error'])
            
            with col2:
                if st.button("Train Quantity Regression Model"):
                    with st.spinner("Training regression model..."):
                        results = st.session_state.model.train_quantity_regressor(df)
                        
                        if 'error' not in results:
                            st.success("Regression model trained successfully!")
                            
                            # Show results
                            st.write("**Random Forest Performance:**")
                            st.metric("R² Score", f"{results['random_forest']['r2_score']:.3f}")
                            st.metric("RMSE", f"{results['random_forest']['rmse']:.3f}")
                            
                            st.write("**Linear Regression Performance:**")
                            st.metric("R² Score", f"{results['linear_regression']['r2_score']:.3f}")
                            st.metric("RMSE", f"{results['linear_regression']['rmse']:.3f}")
                        else:
                            st.error(results['error'])
            
            # Model visualization
            if hasattr(st.session_state.model, 'regression_results') and st.session_state.model.regression_results:
                st.subheader("Model Performance Visualization")
                
                results = st.session_state.model.regression_results
                
                # Actual vs Predicted scatter plot
                fig_scatter = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Random Forest', 'Linear Regression')
                )
                
                # Random Forest
                fig_scatter.add_trace(
                    go.Scatter(
                        x=results['test_predictions']['actual'],
                        y=results['test_predictions']['rf_predicted'],
                        mode='markers',
                        name='RF Predictions',
                        marker=dict(color='blue', alpha=0.6)
                    ),
                    row=1, col=1
                )
                
                # Linear Regression
                fig_scatter.add_trace(
                    go.Scatter(
                        x=results['test_predictions']['actual'],
                        y=results['test_predictions']['lr_predicted'],
                        mode='markers',
                        name='LR Predictions',
                        marker=dict(color='red', alpha=0.6)
                    ),
                    row=1, col=2
                )
                
                # Add diagonal line for perfect predictions
                min_val = min(results['test_predictions']['actual'].min(), 
                            results['test_predictions']['rf_predicted'].min(),
                            results['test_predictions']['lr_predicted'].min())
                max_val = max(results['test_predictions']['actual'].max(), 
                            results['test_predictions']['rf_predicted'].max(),
                            results['test_predictions']['lr_predicted'].max())
                
                for i in range(1, 3):
                    fig_scatter.add_trace(
                        go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='black', dash='dash'),
                            showlegend=i==1
                        ),
                        row=1, col=i
                    )
                
                fig_scatter.update_layout(
                    height=500,
                    title_text="Actual vs Predicted Quantities"
                )
                
                fig_scatter.update_xaxes(title_text="Actual Quantity")
                fig_scatter.update_yaxes(title_text="Predicted Quantity")
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Prediction interface
            st.subheader("Make Predictions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Material Prediction**")
                problem_input = st.text_area("Enter problem description:", 
                                           placeholder="e.g., MOTOR COOLANT KEBAKAR")
                
                if st.button("Predict Material"):
                    if problem_input:
                        prediction = st.session_state.model.predict_material(problem_input)
                        
                        if isinstance(prediction, dict):
                            st.success(f"Predicted Material: **{prediction['material']}**")
                            st.info(f"Confidence: {prediction['confidence']:.3f}")
                        else:
                            st.error(prediction)
                    else:
                        st.warning("Please enter a problem description")
            
            with col2:
                st.write("**Quantity Prediction**")
                desc_length = st.number_input("Problem description length:", min_value=1, value=25)
                word_count = st.number_input("Word count:", min_value=1, value=5)
                
                if st.button("Predict Quantity"):
                    prediction = st.session_state.model.predict_quantity(desc_length, word_count)
                    
                    if isinstance(prediction, dict):
                        st.success(f"Random Forest Prediction: **{prediction['rf_prediction']}** units")
                        st.info(f"Linear Regression Prediction: **{prediction['lr_prediction']}** units")
                    else:
                        st.error(prediction)
            
            # Save models
            st.subheader("Model Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Save Models"):
                    try:
                        st.session_state.model.save_models()
                        st.success("Models saved successfully!")
                    except Exception as e:
                        st.error(f"Error saving models: {e}")
            
            with col2:
                if st.button("Load Models"):
                    try:
                        success = st.session_state.model.load_models()
                        if success:
                            st.success("Models loaded successfully!")
                        else:
                            st.error("Failed to load models")
                    except Exception as e:
                        st.error(f"Error loading models: {e}")
        
        else:
            st.warning("Please upload and preprocess data first!")
    
    # Page 4: Results & Export
    elif page == "Results & Export":
        st.markdown('<div class="section-header">📋 Results & Export</div>', unsafe_allow_html=True)
        
        if st.session_state.preprocessed_data is not None:
            df = st.session_state.preprocessed_data
            
            # Summary report
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
            
            # Top insights
            st.subheader("Key Insights")
            
            # Most used materials
            top_materials = df['MatName'].value_counts().head(5)
            st.write("**Top 5 Most Used Materials:**")
            for i, (material, count) in enumerate(top_materials.items(), 1):
                st.write(f"{i}. {material}: {count} times")
            
            # Most expensive materials
            expensive_materials = df[df['Price'] > 0].nlargest(5, 'Price')[['MatName', 'Price', 'MachineType']]
            st.write("**Top 5 Most Expensive Materials:**")
            st.dataframe(expensive_materials.reset_index(drop=True))
            
            # Machine type analysis
            machine_analysis = df.groupby('MachineType').agg({
                'QtyOut': 'mean',
                'Price': 'mean',
                'MatName': 'count'
            }).round(2)
            machine_analysis.columns = ['Avg Quantity', 'Avg Price', 'Total Usage']
            
            st.write("**Machine Type Analysis:**")
            st.dataframe(machine_analysis)
            
            # Model performance summary
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
            
            # Export options
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
            
            # Advanced filtering
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
    
    # Footer
    st.markdown("---")
    st.markdown("**Material Requirement Analysis System** - Built with Streamlit")

if __name__ == "__main__":
    main()