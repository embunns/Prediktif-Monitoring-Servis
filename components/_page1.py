import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from preprocessing import preprocess_data, create_feature_columns, get_data_summary

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
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif uploaded_file.name.endswith('.xls'):
            df = pd.read_excel(uploaded_file, engine='xlrd')
        elif uploaded_file.name.endswith('.csv'):
            try:
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                try:
                    df = pd.read_csv(uploaded_file, encoding='latin1')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    try:
                        df = pd.read_csv(uploaded_file, encoding='cp1252')
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
        else:
            st.error("Please upload a valid file (.xlsx, .xls, or .csv)")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def show_page():
    st.markdown('<div class="section-header">📊 Data Upload & Preprocessing</div>', unsafe_allow_html=True)
    
    st.subheader("Upload Data File")
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=['xlsx', 'xls', 'csv'],
        help="Supported formats: Excel (.xlsx, .xls) and CSV (.csv)"
    )
    
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
    
    if uploaded_file is not None:
        try:
            st.info(f"📁 File: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
            
            if uploaded_file.name.endswith('.xlsx'):
                st.session_state.data = pd.read_excel(uploaded_file, engine='openpyxl')
                st.success("✅ Excel file (.xlsx) loaded successfully!")
                
            elif uploaded_file.name.endswith('.xls'):
                st.session_state.data = pd.read_excel(uploaded_file, engine='xlrd')
                st.success("✅ Excel file (.xls) loaded successfully!")
                
            elif uploaded_file.name.endswith('.csv'):
                encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
                
                for encoding in encodings_to_try:
                    try:
                        uploaded_file.seek(0)
                        st.session_state.data = pd.read_csv(uploaded_file, encoding=encoding)
                        st.success(f"✅ CSV file loaded successfully with {encoding} encoding!")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
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
    
    if st.session_state.data is not None:
        st.markdown("---")
        st.subheader("📋 Data Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Records", len(st.session_state.data))
        with col2:
            st.metric("📋 Total Columns", len(st.session_state.data.columns))
        with col3:
            st.metric("❓ Missing Values", st.session_state.data.isnull().sum().sum())
        with col4:
            st.metric("🔄 Duplicate Records", st.session_state.data.duplicated().sum())
        
        st.subheader("📑 Column Information")
        col_info = pd.DataFrame({
            'Column': st.session_state.data.columns,
            'Data Type': st.session_state.data.dtypes,
            'Missing Values': st.session_state.data.isnull().sum(),
            'Unique Values': st.session_state.data.nunique()
        })
        st.dataframe(col_info)
        
        st.subheader("👀 Data Preview")
        preview_rows = st.slider("Number of rows to preview", 1, min(100, len(st.session_state.data)), 10)
        st.dataframe(st.session_state.data.head(preview_rows))
        
        st.subheader("🔍 Data Quality Check")
        
        required_columns = ['WorkOrderNo', 'ProblemDesc', 'StockNo', 'MatName', 'QtyOut']
        missing_columns = [col for col in required_columns if col not in st.session_state.data.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.info("💡 Please ensure your data contains the following columns:")
            st.info("- WorkOrderNo, ProblemDesc, StockNo, MatName, QtyOut")
        else:
            st.success("✅ All required columns are present!")
        
        numeric_columns = ['QtyOut', 'Price'] if 'Price' in st.session_state.data.columns else ['QtyOut']
        for col in numeric_columns:
            if col in st.session_state.data.columns:
                non_numeric = st.session_state.data[col].apply(lambda x: not pd.api.types.is_numeric_dtype(type(x)) and pd.notna(x))
                if non_numeric.any():
                    st.warning(f"⚠️ Column '{col}' contains non-numeric values")
        
        st.markdown("---")
        if st.button("🔄 Start Preprocessing", type="primary"):
            if missing_columns:
                st.error("Cannot proceed with preprocessing due to missing required columns")
            else:
                with st.spinner("Processing data..."):
                    try:
                        st.session_state.preprocessed_data = preprocess_data(st.session_state.data)
                        st.session_state.preprocessed_data = create_feature_columns(st.session_state.preprocessed_data)
                        
                        st.success("✅ Data preprocessing completed successfully!")
                        
                        st.subheader("📊 Preprocessed Data Preview")
                        st.dataframe(st.session_state.preprocessed_data.head())
                        
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
    
    if st.session_state.data is None:
        st.info("📝 **Instructions:**")
        st.info("1. Upload your Excel (.xlsx, .xls) or CSV file using the file uploader above")
        st.info("2. Or click 'Use Sample Data' to try the application with sample data")
        st.info("3. Make sure your data contains the required columns: WorkOrderNo, ProblemDesc, StockNo, MatName, QtyOut")
        st.info("4. Once uploaded, click 'Start Preprocessing' to prepare your data for analysis")