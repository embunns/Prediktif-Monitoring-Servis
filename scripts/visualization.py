
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

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
