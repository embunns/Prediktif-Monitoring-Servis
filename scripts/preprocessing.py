import pandas as pd
import numpy as np
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def clean_text(text):
    """
    Clean and standardize text data
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Convert to uppercase
    text = text.upper()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters except letters, numbers, and basic punctuation
    text = re.sub(r'[^\w\s\-\.]', '', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text

def parse_date(date_str):
    """
    Parse various date formats into datetime
    """
    if pd.isna(date_str):
        return None
    
    # Convert to string if not already
    date_str = str(date_str).strip()
    
    # If it's already a datetime object, return it
    if isinstance(date_str, datetime):
        return date_str
    
    # Common date formats to try
    date_formats = [
        '%d/%m/%Y',     # 21/04/2017
        '%m/%d/%Y',     # 04/21/2017
        '%Y-%m-%d',     # 2017-04-21
        '%d-%m-%Y',     # 21-04-2017
        '%Y/%m/%d',     # 2017/04/21
        '%d.%m.%Y',     # 21.04.2017
        '%Y.%m.%d',     # 2017.04.21
        '%d %m %Y',     # 21 04 2017
        '%Y %m %d',     # 2017 04 21
    ]
    
    # Try each format
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # If no format works, try pandas to_datetime with automatic parsing
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except:
        try:
            return pd.to_datetime(date_str)
        except:
            return None

def determine_machine_type(work_order, problem_desc):
    """
    Determine machine type based on work order and problem description
    """
    if pd.isna(work_order) and pd.isna(problem_desc):
        return 'UNKNOWN'
    
    # Convert to string and uppercase
    work_order = str(work_order).upper()
    problem_desc = str(problem_desc).upper()
    
    # Keywords for different machine types
    jobs_keywords = ['RFM', 'JOBS', 'RING', 'VALVE', 'MOTOR', 'ENCODER', 'AXIS']
    craft_keywords = ['MWO', 'CRAFT', 'FURNACE', 'KRAFT', 'SPARE', 'PART']
    
    # Check work order patterns
    if 'RFM' in work_order:
        return 'JOBS'
    elif 'MWO' in work_order:
        return 'CRAFT'
    
    # Check problem description for keywords
    jobs_score = sum(1 for keyword in jobs_keywords if keyword in problem_desc)
    craft_score = sum(1 for keyword in craft_keywords if keyword in problem_desc)
    
    if jobs_score > craft_score:
        return 'JOBS'
    elif craft_score > jobs_score:
        return 'CRAFT'
    else:
        return 'UNKNOWN'

def clean_price(price_value):
    """
    Clean and standardize price values
    """
    if pd.isna(price_value):
        return 0.0
    
    # Convert to string first
    price_str = str(price_value).strip()
    
    # Remove currency symbols and commas
    price_str = re.sub(r'[^\d\.]', '', price_str)
    
    # Try to convert to float
    try:
        return float(price_str)
    except:
        return 0.0

def clean_quantity(qty_value):
    """
    Clean and standardize quantity values
    """
    if pd.isna(qty_value):
        return 1  # Default quantity
    
    # Convert to string first
    qty_str = str(qty_value).strip()
    
    # Remove non-numeric characters except decimal point
    qty_str = re.sub(r'[^\d\.]', '', qty_str)
    
    # Try to convert to float, then to int
    try:
        qty_float = float(qty_str)
        return max(1, int(qty_float))  # Ensure minimum quantity is 1
    except:
        return 1

def preprocess_data(df):
    """
    Main preprocessing function
    """
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # 1. Clean text columns
    text_columns = ['ProblemDesc', 'MatName', 'Specification', 'Codification']
    for col in text_columns:
        if col in df_processed.columns:
            df_processed[f'{col}_Clean'] = df_processed[col].apply(clean_text)
    
    # 2. Parse and convert TransOutDate
    if 'TransOutDate' in df_processed.columns:
        print("Processing TransOutDate column...")
        df_processed['TransOutDate'] = df_processed['TransOutDate'].apply(parse_date)
        
        # Remove rows with invalid dates
        invalid_dates = df_processed['TransOutDate'].isna()
        if invalid_dates.sum() > 0:
            print(f"Warning: {invalid_dates.sum()} rows have invalid dates and will be excluded from time-based analysis")
    
    # 3. Clean numeric columns
    if 'Price' in df_processed.columns:
        df_processed['Price'] = df_processed['Price'].apply(clean_price)
    
    if 'QtyOut' in df_processed.columns:
        df_processed['QtyOut'] = df_processed['QtyOut'].apply(clean_quantity)
    
    # 4. Determine machine type
    if 'WorkOrderNo' in df_processed.columns and 'ProblemDesc' in df_processed.columns:
        df_processed['MachineType'] = df_processed.apply(
            lambda row: determine_machine_type(row['WorkOrderNo'], row['ProblemDesc']), 
            axis=1
        )
    
    # 5. Clean and standardize other columns
    if 'Currency' in df_processed.columns:
        df_processed['Currency'] = df_processed['Currency'].fillna('IDR').str.upper()
    
    if 'UOM' in df_processed.columns:
        df_processed['UOM'] = df_processed['UOM'].fillna('EA').str.upper()
    
    if 'MovingRate' in df_processed.columns:
        df_processed['MovingRate'] = df_processed['MovingRate'].fillna('SM').str.upper()
    
    # 6. Handle missing values
    df_processed['StockNo'] = df_processed['StockNo'].fillna('UNKNOWN')
    df_processed['MatName'] = df_processed['MatName'].fillna('UNKNOWN MATERIAL')
    df_processed['Specification'] = df_processed['Specification'].fillna('NO SPEC')
    
    # 7. Create standardized ID columns
    df_processed['WorkOrderNo'] = df_processed['WorkOrderNo'].fillna('UNKNOWN').astype(str)
    df_processed['StockNo'] = df_processed['StockNo'].astype(str)
    
    print(f"Preprocessing completed. Records processed: {len(df_processed)}")
    
    return df_processed

def create_feature_columns(df):
    """
    Create additional feature columns for analysis
    """
    df_features = df.copy()
    
    # Only create date-based features if TransOutDate is valid
    if 'TransOutDate' in df_features.columns:
        valid_dates = df_features['TransOutDate'].notna()
        
        if valid_dates.sum() > 0:
            # Create date-based features only for valid dates
            df_features.loc[valid_dates, 'Year'] = df_features.loc[valid_dates, 'TransOutDate'].dt.year
            df_features.loc[valid_dates, 'Month'] = df_features.loc[valid_dates, 'TransOutDate'].dt.month
            df_features.loc[valid_dates, 'Quarter'] = df_features.loc[valid_dates, 'TransOutDate'].dt.quarter
            df_features.loc[valid_dates, 'DayOfWeek'] = df_features.loc[valid_dates, 'TransOutDate'].dt.dayofweek
            df_features.loc[valid_dates, 'WeekOfYear'] = df_features.loc[valid_dates, 'TransOutDate'].dt.isocalendar().week
            
            # Create season feature
            def get_season(month):
                if pd.isna(month):
                    return 'Unknown'
                if month in [12, 1, 2]:
                    return 'Winter'
                elif month in [3, 4, 5]:
                    return 'Spring'
                elif month in [6, 7, 8]:
                    return 'Summer'
                else:
                    return 'Fall'
            
            df_features['Season'] = df_features['Month'].apply(get_season)
            
            # Fill NaN values for date-based features
            df_features['Year'] = df_features['Year'].fillna(0).astype(int)
            df_features['Month'] = df_features['Month'].fillna(0).astype(int)
            df_features['Quarter'] = df_features['Quarter'].fillna(0).astype(int)
            df_features['DayOfWeek'] = df_features['DayOfWeek'].fillna(0).astype(int)
            df_features['WeekOfYear'] = df_features['WeekOfYear'].fillna(0).astype(int)
        else:
            # If no valid dates, create dummy columns
            df_features['Year'] = 0
            df_features['Month'] = 0
            df_features['Quarter'] = 0
            df_features['DayOfWeek'] = 0
            df_features['WeekOfYear'] = 0
            df_features['Season'] = 'Unknown'
    
    # Text-based features
    if 'ProblemDesc_Clean' in df_features.columns:
        df_features['ProblemDesc_Length'] = df_features['ProblemDesc_Clean'].str.len()
        df_features['ProblemDesc_WordCount'] = df_features['ProblemDesc_Clean'].str.split().str.len()
    else:
        df_features['ProblemDesc_Length'] = 0
        df_features['ProblemDesc_WordCount'] = 0
    
    # Fill NaN values
    df_features['ProblemDesc_Length'] = df_features['ProblemDesc_Length'].fillna(0)
    df_features['ProblemDesc_WordCount'] = df_features['ProblemDesc_WordCount'].fillna(0)
    
    # Material category features
    if 'MatName' in df_features.columns:
        def categorize_material(mat_name):
            if pd.isna(mat_name):
                return 'OTHER'
            
            mat_name = str(mat_name).upper()
            
            if any(keyword in mat_name for keyword in ['VALVE', 'KATUP']):
                return 'VALVE'
            elif any(keyword in mat_name for keyword in ['RING', 'GASKET', 'SEAL']):
                return 'SEALING'
            elif any(keyword in mat_name for keyword in ['HOSE', 'TUBE', 'PIPE']):
                return 'PIPING'
            elif any(keyword in mat_name for keyword in ['SWITCH', 'BUTTON', 'RELAY']):
                return 'ELECTRICAL'
            elif any(keyword in mat_name for keyword in ['ENCODER', 'SENSOR']):
                return 'SENSOR'
            elif any(keyword in mat_name for keyword in ['MOTOR', 'PUMP']):
                return 'MECHANICAL'
            elif any(keyword in mat_name for keyword in ['TIMER', 'CONTROL']):
                return 'CONTROL'
            else:
                return 'OTHER'
        
        df_features['MaterialCategory'] = df_features['MatName'].apply(categorize_material)
    
    # Price category features
    if 'Price' in df_features.columns:
        def categorize_price(price):
            if pd.isna(price) or price <= 0:
                return 'FREE'
            elif price < 1000:
                return 'LOW'
            elif price < 100000:
                return 'MEDIUM'
            elif price < 1000000:
                return 'HIGH'
            else:
                return 'VERY_HIGH'
        
        df_features['PriceCategory'] = df_features['Price'].apply(categorize_price)
    
    # Quantity category features
    if 'QtyOut' in df_features.columns:
        def categorize_quantity(qty):
            if pd.isna(qty) or qty <= 0:
                return 'NONE'
            elif qty == 1:
                return 'SINGLE'
            elif qty <= 5:
                return 'SMALL'
            elif qty <= 20:
                return 'MEDIUM'
            else:
                return 'LARGE'
        
        df_features['QtyCategory'] = df_features['QtyOut'].apply(categorize_quantity)
    
    print(f"Feature creation completed. Total features: {len(df_features.columns)}")
    
    return df_features

def get_data_summary(df):
    """
    Generate summary statistics for the processed data
    """
    summary = {}
    
    # Basic statistics
    summary['total_records'] = len(df)
    summary['unique_work_orders'] = df['WorkOrderNo'].nunique() if 'WorkOrderNo' in df.columns else 0
    summary['unique_materials'] = df['MatName'].nunique() if 'MatName' in df.columns else 0
    
    # Machine type distribution
    if 'MachineType' in df.columns:
        summary['machine_types'] = df['MachineType'].value_counts().to_dict()
    else:
        summary['machine_types'] = {}
    
    # Price statistics
    if 'Price' in df.columns:
        price_data = df[df['Price'] > 0]['Price']
        summary['price_stats'] = {
            'mean': price_data.mean() if len(price_data) > 0 else 0,
            'median': price_data.median() if len(price_data) > 0 else 0,
            'std': price_data.std() if len(price_data) > 0 else 0,
            'min': price_data.min() if len(price_data) > 0 else 0,
            'max': price_data.max() if len(price_data) > 0 else 0
        }
    else:
        summary['price_stats'] = {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
    
    # Quantity statistics
    if 'QtyOut' in df.columns:
        summary['qty_stats'] = {
            'mean': df['QtyOut'].mean(),
            'median': df['QtyOut'].median(),
            'std': df['QtyOut'].std(),
            'min': df['QtyOut'].min(),
            'max': df['QtyOut'].max()
        }
    else:
        summary['qty_stats'] = {'mean': 0, 'median': 0, 'std': 0, 'min': 0, 'max': 0}
    
    # Date range
    if 'TransOutDate' in df.columns:
        valid_dates = df['TransOutDate'].dropna()
        summary['date_range'] = {
            'start': valid_dates.min() if len(valid_dates) > 0 else None,
            'end': valid_dates.max() if len(valid_dates) > 0 else None,
            'total_days': (valid_dates.max() - valid_dates.min()).days if len(valid_dates) > 0 else 0
        }
    else:
        summary['date_range'] = {'start': None, 'end': None, 'total_days': 0}
    
    # Missing data analysis
    summary['missing_data'] = {}
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            summary['missing_data'][col] = {
                'count': missing_count,
                'percentage': (missing_count / len(df)) * 100
            }
    
    return summary

def validate_data_quality(df):
    """
    Validate data quality and return issues
    """
    issues = []
    
    # Check for required columns
    required_columns = ['WorkOrderNo', 'ProblemDesc', 'StockNo', 'MatName', 'QtyOut']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
    
    # Check for empty dataframe
    if len(df) == 0:
        issues.append("DataFrame is empty")
        return issues
    
    # Check for duplicate work orders with different materials
    if 'WorkOrderNo' in df.columns and 'MatName' in df.columns:
        duplicates = df.groupby('WorkOrderNo')['MatName'].nunique()
        multi_material_orders = duplicates[duplicates > 1]
        if len(multi_material_orders) > 0:
            issues.append(f"Found {len(multi_material_orders)} work orders with multiple materials")
    
    # Check for invalid quantities
    if 'QtyOut' in df.columns:
        invalid_qty = df[(df['QtyOut'] <= 0) | (df['QtyOut'].isna())].shape[0]
        if invalid_qty > 0:
            issues.append(f"Found {invalid_qty} records with invalid quantities")
    
    # Check for invalid prices
    if 'Price' in df.columns:
        negative_prices = df[df['Price'] < 0].shape[0]
        if negative_prices > 0:
            issues.append(f"Found {negative_prices} records with negative prices")
    
    # Check for invalid dates
    if 'TransOutDate' in df.columns:
        invalid_dates = df['TransOutDate'].isna().sum()
        if invalid_dates > 0:
            issues.append(f"Found {invalid_dates} records with invalid dates")
    
    # Check for missing critical data
    if 'MatName' in df.columns:
        missing_materials = df['MatName'].isna().sum()
        if missing_materials > 0:
            issues.append(f"Found {missing_materials} records with missing material names")
    
    return issues

def export_preprocessing_report(df, original_df, filename='preprocessing_report.txt'):
    """
    Export a detailed preprocessing report
    """
    with open(filename, 'w') as f:
        f.write("MATERIAL REQUIREMENT PREPROCESSING REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Basic statistics
        f.write("BASIC STATISTICS:\n")
        f.write(f"Original records: {len(original_df)}\n")
        f.write(f"Processed records: {len(df)}\n")
        f.write(f"Records removed: {len(original_df) - len(df)}\n\n")
        
        # Column information
        f.write("COLUMN INFORMATION:\n")
        f.write(f"Original columns: {len(original_df.columns)}\n")
        f.write(f"Processed columns: {len(df.columns)}\n")
        f.write(f"New columns added: {len(df.columns) - len(original_df.columns)}\n\n")
        
        # Data quality
        issues = validate_data_quality(df)
        f.write("DATA QUALITY ISSUES:\n")
        if issues:
            for issue in issues:
                f.write(f"- {issue}\n")
        else:
            f.write("No major data quality issues found.\n")
        
        f.write("\nPROCESSING COMPLETED SUCCESSFULLY!\n")
    
    print(f"Preprocessing report exported to {filename}")

# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    sample_data = {
        'WorkOrderNo': ['17RFM034931', '15RFM029301', 'MWO-1412-00954'],
        'ProblemDesc': ['AMPLITUD TO HIGH ENCODER', 'PINTU TROUBLE', 'SPARE PART'],
        'StockNo': ['A01M127', 'C07H050', 'A02RM001'],
        'MatName': ['RING O', 'VALVE', 'ANGLE BAR'],
        'QtyOut': [2, 1, 1],
        'Price': [0.05, 152, 250000],
        'TransOutDate': ['21/04/2017', '27/07/2015', '05/01/2015']
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Testing preprocessing functions...")
    print("Original data:")
    print(df)
    
    # Test preprocessing
    processed_df = preprocess_data(df)
    print("\nProcessed data:")
    print(processed_df)
    
    # Test feature creation
    feature_df = create_feature_columns(processed_df)
    print("\nFeature columns:")
    print(feature_df.columns.tolist())
    
    # Test summary
    summary = get_data_summary(feature_df)
    print("\nData summary:")
    print(summary)
    
    print("\nTesting completed successfully!")