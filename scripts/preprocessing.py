import pandas as pd
import numpy as np
from datetime import datetime
import re

def parse_date(date_str):
    """Parse date string dalam format DD/MM/YYYY"""
    try:
        if pd.isna(date_str) or date_str == '':
            return None
        return datetime.strptime(date_str, '%d/%m/%Y')
    except:
        return None

def clean_currency_price(price_str):
    """Clean dan convert price string ke float"""
    if pd.isna(price_str) or price_str == '':
        return 0.0
    
    # Remove currency symbols dan whitespace
    price_str = str(price_str).replace(',', '.').strip()
    
    # Extract numeric value
    numeric_match = re.search(r'[\d.,]+', price_str)
    if numeric_match:
        try:
            return float(numeric_match.group())
        except:
            return 0.0
    return 0.0

def determine_machine_type(codification):
    """Tentukan jenis mesin berdasarkan codification"""
    if pd.isna(codification):
        return 'UNKNOWN'
    
    codification = str(codification).upper()
    if 'AABG' in codification:
        return 'JOBS'
    elif 'TFAK' in codification:
        return 'CRAFT'
    else:
        return 'UNKNOWN'

def preprocess_data(df):
    """
    Fungsi utama untuk preprocessing data material
    
    Args:
        df: DataFrame raw data
    
    Returns:
        DataFrame yang sudah dibersihkan
    """
    # Copy dataframe untuk menghindari modifikasi asli
    data = df.copy()
    
    # 1. Handle missing values
    data['ProblemDesc'] = data['ProblemDesc'].fillna('No Description')
    data['MatName'] = data['MatName'].fillna('Unknown Material')
    data['Specification'] = data['Specification'].fillna('No Specification')
    data['UOM'] = data['UOM'].fillna('EA')
    data['Currency'] = data['Currency'].fillna('IDR')
    
    # 2. Convert dan clean kolom tanggal
    data['TransOutDate'] = data['TransOutDate'].apply(parse_date)
    
    # 3. Clean dan convert kolom Price
    data['Price'] = data['Price'].apply(clean_currency_price)
    
    # 4. Convert QtyOut ke numeric
    data['QtyOut'] = pd.to_numeric(data['QtyOut'], errors='coerce').fillna(0)
    
    # 5. Buat kolom MachineType berdasarkan Codification
    data['MachineType'] = data['Codification'].apply(determine_machine_type)
    
    # 6. Extract year dari TransOutDate
    data['Year'] = data['TransOutDate'].dt.year
    data['Month'] = data['TransOutDate'].dt.month
    
    # 7. Buat kolom untuk analisis lebih lanjut
    data['ProblemDesc_Length'] = data['ProblemDesc'].str.len()
    data['MatName_Length'] = data['MatName'].str.len()
    
    # 8. Normalisasi text columns
    data['ProblemDesc_Clean'] = data['ProblemDesc'].str.upper().str.strip()
    data['MatName_Clean'] = data['MatName'].str.upper().str.strip()
    
    # 9. Remove duplicate rows berdasarkan key columns
    data = data.drop_duplicates(subset=['WorkOrderNo', 'StockNo', 'MaterialTicketNo'])
    
    # 10. Sort berdasarkan tanggal
    data = data.sort_values('TransOutDate', ascending=False)
    
    # 11. Reset index
    data = data.reset_index(drop=True)
    
    return data

def create_feature_columns(df):
    """
    Buat kolom fitur tambahan untuk modeling
    
    Args:
        df: DataFrame hasil preprocessing
    
    Returns:
        DataFrame dengan kolom fitur tambahan
    """
    data = df.copy()
    
    # 1. Fitur berbasis teks
    data['ProblemDesc_WordCount'] = data['ProblemDesc'].str.split().str.len()
    data['HasNumberInDesc'] = data['ProblemDesc'].str.contains(r'\d+', na=False)
    data['HasSpecialCharInDesc'] = data['ProblemDesc'].str.contains(r'[^a-zA-Z0-9\s]', na=False)
    
    # 2. Fitur berbasis material
    data['IsValve'] = data['MatName'].str.contains('VALVE', case=False, na=False)
    data['IsRing'] = data['MatName'].str.contains('RING', case=False, na=False)
    data['IsHose'] = data['MatName'].str.contains('HOSE', case=False, na=False)
    
    # 3. Fitur berbasis harga
    data['PriceCategory'] = pd.cut(data['Price'], 
                                   bins=[0, 1000, 10000, 100000, np.inf], 
                                   labels=['Low', 'Medium', 'High', 'Very High'])
    
    # 4. Fitur berbasis quantity
    data['QtyCategory'] = pd.cut(data['QtyOut'], 
                                 bins=[0, 1, 5, 10, np.inf], 
                                 labels=['Single', 'Few', 'Many', 'Bulk'])
    
    return data

def get_data_summary(df):
    """
    Generate summary statistik dari data
    
    Args:
        df: DataFrame yang sudah dipreprocess
    
    Returns:
        Dictionary berisi summary statistik
    """
    summary = {
        'total_records': len(df),
        'unique_work_orders': df['WorkOrderNo'].nunique(),
        'unique_materials': df['MatName'].nunique(),
        'machine_types': df['MachineType'].value_counts().to_dict(),
        'date_range': {
            'start': df['TransOutDate'].min(),
            'end': df['TransOutDate'].max()
        },
        'price_stats': {
            'min': df['Price'].min(),
            'max': df['Price'].max(),
            'mean': df['Price'].mean(),
            'median': df['Price'].median()
        },
        'qty_stats': {
            'min': df['QtyOut'].min(),
            'max': df['QtyOut'].max(),
            'mean': df['QtyOut'].mean(),
            'median': df['QtyOut'].median()
        }
    }
    
    return summary