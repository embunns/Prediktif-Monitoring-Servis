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

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from preprocessing import preprocess_data, create_feature_columns, get_data_summary, validate_data_quality
from model import MaterialPredictionModel

st.set_page_config(
    page_title="Material Requirement Analysis",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_data():
    sample_data = {
        'WorkOrderNo': ['17RFM034931', '15RFM029301', '15RFM029584', 'MWO-1412-00954', 'MWO-1412-00954', '20RFM041056', '17RFM034931', '19RFM040317', '15RFM025622', '15RFM025622'],
        'ProblemDesc': ['AMPLITUD TO HIGH ENCODER', 'PINTU TROUBLE', 'MOTOR COOLANT KEBAKAR', 'SPARE PART', 'SPARE PART', 'C ENCODER ERROR', 'AMPLITUD TO HIGH ENCODER', 'KRAFT FURNACE TROUBLE', 'AXIS C TROUBLE MESIN OFF TERUS', 'AXIS C TROUBLE MESIN OFF TERUS'],
        'StockNo': ['A01M127', 'C07H050', 'A02M077', 'A02RM001', 'A02RM017', 'A07E064', 'A25M008', 'A30E016', 'A34M051', 'A36H001'],
        'MatName': ['RING O', 'VALVE', 'VALVE', 'ANGLE BAR', 'ANGLE BAR', 'TIMER', 'RING O', 'EMERGENCY SWITCH', 'HOSE HYDRAULIC', 'MANOMETERS'],
        'Specification': ['551.10.148', 'WM-781-946/MCH-3-1/8', 'ART-NR-280.1020.2', 'STANDAR SII COMM-STEEL', 'COMM-STEEL', 'H3Y-2/24VDC/5A', 'YC4-100-140', 'DIA 30MM 5A/250V', '3/4 X 65 cm', '0-2000 PSI'],
        'QtyOut': [2, 1, 1, 1, 3, 1, 1, 1, 1, 1],
        'UOM': ['EA', 'EA', 'EA', 'EA', 'EA', 'EA', 'EA', 'EA', 'EA', 'EA'],
        'Codification': ['AABG01', 'TFAK01', 'AABG01', 'TFAK01', 'TFAK01', 'AABG01', 'AABG01', 'TFAK01', 'AABG01', 'AABG01'],
        'MaterialTicketNo': ['MMT-1704-00726', 'MMT-1507-01258', 'MMT-1512-02142', 'MMT-1501-00001', 'MMT-1501-00001', 'MMT-2003-00338', 'MMT-1704-00726', 'MMT-1907-00900', 'MMT-1503-00530', 'MMT-1501-00097'],
        'TransOutDate': ['21/04/2017', '27/07/2015', '23/12/2015', '05/01/2015', '05/01/2015', '19/03/2020', '21/04/2017', '30/07/2019', '25/03/2015', '23/01/2015'],
        'Currency': ['DEM', 'CHF', 'DEM', 'IDR', 'IDR', 'IDR', 'IDR', 'IDR', 'IDR', 'IDR'],
        'Price': [0.05, 152, 15, 250000, 325000, 155000, 750, 25000, 170000, 1064000],
        'MovingRate': ['FM', 'SM', 'SM', 'SM', 'FM', 'SM', 'FM', 'SM', 'SM', 'FM']
    }
    
    return pd.DataFrame(sample_data)

def read_excel_file(uploaded_file):
    """
    Improved file reading function with better error handling and encoding support
    """
    try:
        # Check file extension
        if uploaded_file.name.endswith('.xlsx'):
            # For .xlsx files, use openpyxl engine
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.xls'):
            # For .xls files, use xlrd engine
            df = pd.read_excel(uploaded_file, engine='xlrd')
        elif uploaded_file.name.endswith('.csv'):
            # For CSV files, try multiple encodings
            try:
                # Try UTF-8 first
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                # If UTF-8 fails, try other common encodings
                uploaded_file.seek(0)  # Reset file pointer
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin1')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)  # Reset file pointer
                    try:
                        df = pd.read_csv(uploaded_file, encoding='cp1252')
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)  # Reset file pointer
                        df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
        else:
            st.error("Please upload a valid file (.xlsx, .xls, or .csv)")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def handle_file_upload():
    """
    Enhanced file upload handling with multiple format support
    """
    uploaded_file = st.file_uploader(
        "Upload your data file", 
        type=['xlsx', 'xls', 'csv'],
        help="Supported formats: Excel (.xlsx, .xls) and CSV (.csv)"
    )
    
    if uploaded_file is not None:
        # Display file information
        st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")
        
        # Read the file
        df = read_excel_file(uploaded_file)
        
        if df is not None:
            st.success("File uploaded successfully!")
            
            # Display basic info about the data
            st.subheader("File Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Show column names
            st.subheader("Column Names")
            st.write(list(df.columns))
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            return df
    
    return None

# Alternative function for CSV files with robust encoding detection
def read_csv_with_encoding_detection(uploaded_file):
    """
    Read CSV file with automatic encoding detection
    """
    import chardet
    
    try:
        # Read raw bytes to detect encoding
        raw_data = uploaded_file.read()
        
        # Detect encoding
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        
        st.info(f"Detected encoding: {encoding} (confidence: {confidence:.2%})")
        
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Read with detected encoding
        df = pd.read_csv(uploaded_file, encoding=encoding)
        
        return df
    
    except Exception as e:
        st.error(f"Error with encoding detection: {str(e)}")
        
        # Fallback to manual encoding attempts
        uploaded_file.seek(0)
        
        encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1', 'utf-16']
        
        for encoding in encodings_to_try:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                st.success(f"Successfully read file with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
        
        st.error("Unable to read file with any supported encoding")
        return None
    
def plot_material_frequency(df):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Material Frequency by Machine Type', 'Top 15 Materials Overall', 
                       'Machine Type Distribution', 'Price Distribution (> 0)'),
        specs=[[{"colspan": 2}, None],
               [{"type": "pie"}, {"type": "histogram"}]]
    )
    
    material_by_machine = df.groupby(['MachineType', 'MatName']).size().reset_index(name='count')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, machine_type in enumerate(material_by_machine['MachineType'].unique()):
        if machine_type == 'UNKNOWN':
            continue
        machine_data = material_by_machine[material_by_machine['MachineType'] == machine_type]
        top_materials = machine_data.nlargest(15, 'count')
        
        fig.add_trace(
            go.Bar(
                x=top_materials['MatName'],
                y=top_materials['count'],
                name=machine_type,
                text=top_materials['count'],
                textposition='auto',
                marker_color=colors[i % len(colors)]
            ),
            row=1, col=1
        )
    
    machine_counts = df['MachineType'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=machine_counts.index,
            values=machine_counts.values,
            name="Machine Types",
            hole=0.4
        ),
        row=2, col=1
    )
    
    price_data = df[df['Price'] > 0]['Price']
    if len(price_data) > 0:
        fig.add_trace(
            go.Histogram(
                x=price_data,
                nbinsx=30,
                name="Price Distribution",
                marker_color='lightblue'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Material Analysis Dashboard",
        title_x=0.5
    )
    
    fig.update_xaxes(tickangle=45, row=1, col=1)
    
    return fig

def plot_time_series_analysis(df):
    if 'TransOutDate' not in df.columns or df['TransOutDate'].isna().all():
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Monthly Usage Trend', 'Quarterly Distribution', 
                       'Usage by Machine Type Over Time', 'Seasonal Patterns'),
        specs=[[{"colspan": 2}, None],
               [{"secondary_y": True}, {"type": "pie"}]]
    )
    
    df_with_date = df.dropna(subset=['TransOutDate'])
    
    df_with_date['YearMonth'] = df_with_date['TransOutDate'].dt.to_period('M')
    monthly_trend = df_with_date.groupby('YearMonth').size().reset_index(name='count')
    monthly_trend['YearMonth_str'] = monthly_trend['YearMonth'].astype(str)
    
    fig.add_trace(
        go.Scatter(
            x=monthly_trend['YearMonth_str'],
            y=monthly_trend['count'],
            mode='lines+markers',
            name='Monthly Usage',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    quarterly_data = df_with_date.groupby(['Quarter', 'MachineType']).size().reset_index(name='count')
    
    for machine_type in quarterly_data['MachineType'].unique():
        if machine_type == 'UNKNOWN':
            continue
        machine_data = quarterly_data[quarterly_data['MachineType'] == machine_type]
        fig.add_trace(
            go.Bar(
                x=machine_data['Quarter'],
                y=machine_data['count'],
                name=f'{machine_type} Quarterly',
                text=machine_data['count'],
                textposition='auto'
            ),
            row=2, col=1
        )
    
    seasonal_data = df_with_date.groupby('Season').size().reset_index(name='count')
    fig.add_trace(
        go.Pie(
            labels=seasonal_data['Season'],
            values=seasonal_data['count'],
            name="Seasonal Usage"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Time Series Analysis",
        title_x=0.5
    )
    
    fig.update_xaxes(tickangle=45, row=1, col=1)
    
    return fig

def plot_advanced_analytics(df):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Price vs Quantity Relationship', 'Material Category Analysis', 
                       'Currency Distribution', 'Moving Rate Analysis'),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "bar"}]]
    )
    
    price_qty_data = df[(df['Price'] > 0) & (df['QtyOut'] > 0)]
    if len(price_qty_data) > 0:
        fig.add_trace(
            go.Scatter(
                x=price_qty_data['Price'],
                y=price_qty_data['QtyOut'],
                mode='markers',
                name='Price vs Quantity',
                marker=dict(
                    color=price_qty_data['Price'],
                    colorscale='Viridis',
                    showscale=True,
                    size=8,
                    opacity=0.7
                ),
                text=price_qty_data['MatName'],
                hovertemplate='<b>%{text}</b><br>Price: %{x}<br>Quantity: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
    
    material_categories = ['VALVE', 'RING', 'HOSE', 'SWITCH', 'ENCODER', 'FAN', 'TIMER']
    category_counts = []
    
    for category in material_categories:
        count = df[df['MatName'].str.contains(category, case=False, na=False)].shape[0]
        category_counts.append(count)
    
    fig.add_trace(
        go.Bar(
            x=material_categories,
            y=category_counts,
            name='Material Categories',
            marker_color='lightgreen',
            text=category_counts,
            textposition='auto'
        ),
        row=1, col=2
    )
    
    currency_dist = df['Currency'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=currency_dist.index,
            values=currency_dist.values,
            name="Currency Distribution"
        ),
        row=2, col=1
    )
    
    moving_rate_dist = df['MovingRate'].value_counts()
    fig.add_trace(
        go.Bar(
            x=moving_rate_dist.index,
            y=moving_rate_dist.values,
            name='Moving Rate',
            marker_color='orange',
            text=moving_rate_dist.values,
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Advanced Analytics Dashboard",
        title_x=0.5
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
    
    # Page 1: Data Upload & Preprocessingif page == "Data Upload & Preprocessing":
    st.markdown('<div class="section-header">📊 Data Upload & Preprocessing</div>', unsafe_allow_html=True)
    
    # File upload with improved error handling
    st.subheader("Upload Data File")
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['xlsx', 'xls', 'csv'],
        help="Supported formats: Excel (.xlsx, .xls) and CSV (.csv)"
    )
    
    # Option to use sample data
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Use Sample Data"):
            st.session_state.data = create_sample_data()
            st.success("Sample data loaded successfully!")
    
    with col2:
        if st.button("Clear Data"):
            st.session_state.data = None
            st.session_state.preprocessed_data = None
            st.success("Data cleared!")
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Display file information
            st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
            
            # Read file based on extension
            if uploaded_file.name.endswith('.xlsx'):
                st.session_state.data = pd.read_excel(uploaded_file, engine='openpyxl')
                st.success("✅ Excel file (.xlsx) loaded successfully!")
                
            elif uploaded_file.name.endswith('.xls'):
                st.session_state.data = pd.read_excel(uploaded_file, engine='xlrd')
                st.success("✅ Excel file (.xls) loaded successfully!")
                
            elif uploaded_file.name.endswith('.csv'):
                # Try multiple encodings for CSV
                encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
                
                for encoding in encodings_to_try:
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        st.session_state.data = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"✅ CSV file loaded successfully with {encoding} encoding!")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    # If all encodings fail, try with error handling
                    uploaded_file.seek(0)
                    st.session_state.data = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore')
                    st.warning("⚠️ CSV file loaded with some characters ignored due to encoding issues")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 **Troubleshooting tips:**")
            st.info("- Make sure the file is not corrupted")
            st.info("- Try saving the file in a different format")
            st.info("- Check if the file is currently open in another program")
            st.info("- For CSV files, try saving with UTF-8 encoding")
    
    # Display data information if loaded
    if st.session_state.data is not None:
        st.markdown("---")
        st.subheader("📋 Data Overview")
        
        # Basic metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Records", len(st.session_state.data))
        with col2:
            st.metric("📋 Total Columns", len(st.session_state.data.columns))
        with col3:
            st.metric("❓ Missing Values", st.session_state.data.isnull().sum().sum())
        with col4:
            st.metric("🔄 Duplicate Records", st.session_state.data.duplicated().sum())
        
        # Column information
        st.subheader("📑 Column Information")
        col_info = pd.DataFrame({
            'Column': st.session_state.data.columns,
            'Data Type': st.session_state.data.dtypes,
            'Missing Values': st.session_state.data.isnull().sum(),
            'Unique Values': st.session_state.data.nunique()
        })
        st.dataframe(col_info)
        
        # Data preview
        st.subheader("👀 Data Preview")
        preview_rows = st.slider("Number of rows to preview", 1, min(100, len(st.session_state.data)), 10)
        st.dataframe(st.session_state.data.head(preview_rows))
        
        # Data quality checks
        st.subheader("🔍 Data Quality Check")
        
        # Check for required columns
        required_columns = ['WorkOrderNo', 'ProblemDesc', 'StockNo', 'MatName', 'QtyOut']
        missing_columns = [col for col in required_columns if col not in st.session_state.data.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("💡 Please ensure your data contains the following columns:")
            st.info("- WorkOrderNo, ProblemDesc, StockNo, MatName, QtyOut")
        else:
            st.success("✅ All required columns are present!")
        
        # Check data types
        numeric_columns = ['QtyOut', 'Price'] if 'Price' in st.session_state.data.columns else ['QtyOut']
        for col in numeric_columns:
            if col in st.session_state.data.columns:
                non_numeric = st.session_state.data[col].apply(lambda x: not pd.api.types.is_numeric_dtype(type(x)) and pd.notna(x))
                if non_numeric.any():
                    st.warning(f"⚠️ Column '{col}' contains non-numeric values")
        
        # Preprocessing button
        st.markdown("---")
        if st.button("🔄 Start Preprocessing", type="primary"):
            if missing_columns:
                st.error("Cannot proceed with preprocessing due to missing required columns")
            else:
                with st.spinner("Processing data..."):
                    try:
                        # Basic preprocessing
                        st.session_state.preprocessed_data = preprocess_data(st.session_state.data)
                        
                        # Create additional features
                        st.session_state.preprocessed_data = create_feature_columns(st.session_state.preprocessed_data)
                        
                        st.success("✅ Data preprocessing completed successfully!")
                        
                        # Show preprocessing results
                        st.subheader("📊 Preprocessed Data Preview")
                        st.dataframe(st.session_state.preprocessed_data.head())
                        
                        # Summary statistics
                        summary = get_data_summary(st.session_state.preprocessed_data)
                        
                        st.subheader("📈 Processing Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("📊 Total Records", summary['total_records'])
                        with col2:
                            st.metric("🔧 Unique Work Orders", summary['unique_work_orders'])
                        with col3:
                            st.metric("📦 Unique Materials", summary['unique_materials'])
                        with col4:
                            date_range = "N/A"
                            if summary['date_range']['start'] and summary['date_range']['end']:
                                date_range = f"{summary['date_range']['start'].strftime('%Y-%m-%d')} to {summary['date_range']['end'].strftime('%Y-%m-%d')}"
                            st.metric("📅 Date Range", date_range)
                        
                    except Exception as e:
                        st.error(f"❌ Error during preprocessing: {str(e)}")
                        st.info("💡 Please check your data format and try again")
    
    # Instructions for users
    if st.session_state.data is None:
        st.info("📝 **Instructions:**")
        st.info("1. Upload your Excel (.xlsx, .xls) or CSV file using the file uploader above")
        st.info("2. Or click 'Use Sample Data' to try the application with sample data")
        st.info("3. Make sure your data contains the required columns: WorkOrderNo, ProblemDesc, StockNo, MatName, QtyOut")
        st.info("4. Once uploaded, click 'Start Preprocessing' to prepare your data for analysis")
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