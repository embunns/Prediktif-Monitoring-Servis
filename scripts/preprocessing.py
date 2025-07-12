import pandas as pd
import numpy as np
import re
from datetime import datetime

def preprocess_data(df):
    df_cleaned = df.copy()
    
    if 'TransOutDate' in df_cleaned.columns:
        df_cleaned['TransOutDate'] = pd.to_datetime(df_cleaned['TransOutDate'], errors='coerce', dayfirst=True)
    
    if 'Price' in df_cleaned.columns:
        df_cleaned['Price'] = pd.to_numeric(df_cleaned['Price'], errors='coerce').fillna(0)
    
    if 'QtyOut' in df_cleaned.columns:
        df_cleaned['QtyOut'] = pd.to_numeric(df_cleaned['QtyOut'], errors='coerce').fillna(1)
    
    df_cleaned = df_cleaned.fillna('')
    
    return df_cleaned

def clean_text(text):
    if pd.isna(text):
        return ''
    text = str(text).upper()
    text = re.sub(r'[^A-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def classify_machine_type(work_order, problem_desc):
    work_order_str = str(work_order).upper()
    problem_desc_str = str(problem_desc).upper()
    
    jobs_keywords = ['RFM', 'JOBS', 'ENCODER', 'AXIS', 'MOTOR']
    craft_keywords = ['MWO', 'KRAFT', 'FURNACE', 'COOLER', 'BOILER']
    
    for keyword in jobs_keywords:
        if keyword in work_order_str or keyword in problem_desc_str:
            return 'JOBS'
    
    for keyword in craft_keywords:
        if keyword in work_order_str or keyword in problem_desc_str:
            return 'CRAFT'
    
    return 'UNKNOWN'

def create_feature_columns(df):
    df_features = df.copy()
    
    df_features['ProblemDesc_Clean'] = df_features['ProblemDesc'].apply(clean_text)
    df_features['ProblemDesc_Length'] = df_features['ProblemDesc_Clean'].apply(len)
    df_features['ProblemDesc_WordCount'] = df_features['ProblemDesc_Clean'].apply(lambda x: len(x.split()) if x else 0)
    
    df_features['MachineType'] = df_features.apply(
        lambda row: classify_machine_type(row['WorkOrderNo'], row['ProblemDesc']), 
        axis=1
    )
    
    if 'TransOutDate' in df_features.columns:
        df_features['Year'] = df_features['TransOutDate'].dt.year
        df_features['Month'] = df_features['TransOutDate'].dt.month
        df_features['Quarter'] = df_features['TransOutDate'].dt.quarter
        df_features['DayOfWeek'] = df_features['TransOutDate'].dt.dayofweek
        df_features['Season'] = df_features['Month'].apply(get_season)
    
    if 'Price' in df_features.columns:
        df_features['PriceCategory'] = pd.cut(df_features['Price'], 
                                            bins=[0, 100, 1000, 10000, float('inf')],
                                            labels=['Low', 'Medium', 'High', 'Very High'])
    
    return df_features

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def get_data_summary(df):
    summary = {
        'total_records': len(df),
        'unique_work_orders': df['WorkOrderNo'].nunique() if 'WorkOrderNo' in df.columns else 0,
        'unique_materials': df['MatName'].nunique() if 'MatName' in df.columns else 0,
        'machine_types': df['MachineType'].value_counts().to_dict() if 'MachineType' in df.columns else {},
        'date_range': {
            'start': df['TransOutDate'].min() if 'TransOutDate' in df.columns else None,
            'end': df['TransOutDate'].max() if 'TransOutDate' in df.columns else None
        },
        'price_stats': {
            'mean': df['Price'].mean() if 'Price' in df.columns else 0,
            'max': df['Price'].max() if 'Price' in df.columns else 0,
            'min': df['Price'].min() if 'Price' in df.columns else 0
        },
        'qty_stats': {
            'mean': df['QtyOut'].mean() if 'QtyOut' in df.columns else 0,
            'max': df['QtyOut'].max() if 'QtyOut' in df.columns else 0,
            'min': df['QtyOut'].min() if 'QtyOut' in df.columns else 0
        }
    }
    return summary

def validate_data_quality(df):
    quality_issues = []
    
    required_columns = ['WorkOrderNo', 'ProblemDesc', 'MatName', 'QtyOut']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        quality_issues.append(f"Missing required columns: {missing_columns}")
    
    if 'QtyOut' in df.columns:
        invalid_qty = df[df['QtyOut'] <= 0].shape[0]
        if invalid_qty > 0:
            quality_issues.append(f"Found {invalid_qty} records with invalid quantities")
    
    if 'Price' in df.columns:
        negative_prices = df[df['Price'] < 0].shape[0]
        if negative_prices > 0:
            quality_issues.append(f"Found {negative_prices} records with negative prices")
    
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        quality_issues.append(f"Found {duplicate_count} duplicate records")
    
    return quality_issues