import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import sys
import os
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from model import MaterialPredictionModel

def show_page():
    st.markdown('<div class="section-header">🤖 Enhanced Material Prediction & Recommendation</div>', unsafe_allow_html=True)
    
    if st.session_state.preprocessed_data is not None:
        df = st.session_state.preprocessed_data
        
        # Initialize enhanced model if not exists
        if 'enhanced_model' not in st.session_state:
            st.session_state.enhanced_model = MaterialPredictionModel()
        
        # Model Training Section
        st.subheader("🎯 Model Training")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Train Material Classification Model", key="train_classification"):
                with st.spinner("Training classification model..."):
                    results = st.session_state.enhanced_model.train_material_classifier(df)
                    
                    if 'error' not in results:
                        st.success("Classification model trained successfully!")
                        
                        st.write("**Model Performance:**")
                        st.metric("Accuracy", f"{results['accuracy']:.3f}")
                        st.metric("Cross-validation Mean", f"{results['cv_mean']:.3f}")
                        st.metric("Cross-validation Std", f"{results['cv_std']:.3f}")
                        st.metric("Number of Classes", results['n_classes'])
                        
                        with st.expander("View Classification Report"):
                            st.text(results['classification_report'])
                    else:
                        st.error(f"Error: {results['error']}")
        
        with col2:
            if st.button("Train Quantity Regression Model", key="train_regression"):
                with st.spinner("Training regression model..."):
                    results = st.session_state.enhanced_model.train_quantity_regressor(df)
                    
                    if 'error' not in results:
                        st.success("Regression model trained successfully!")
                        
                        st.write("**Random Forest Performance:**")
                        st.metric("R² Score", f"{results['random_forest']['r2_score']:.3f}")
                        st.metric("RMSE", f"{results['random_forest']['rmse']:.3f}")
                        
                        st.write("**Linear Regression Performance:**")
                        st.metric("R² Score", f"{results['linear_regression']['r2_score']:.3f}")
                        st.metric("RMSE", f"{results['linear_regression']['rmse']:.3f}")
                    else:
                        st.error(f"Error: {results['error']}")
        
        with col3:
            if st.button("Train Repair Time Predictor", key="train_repair_time"):
                with st.spinner("Training repair time predictor..."):
                    results = st.session_state.enhanced_model.train_repair_time_predictor(df)
                    
                    if 'error' not in results:
                        st.success("Repair time predictor trained successfully!")
                        
                        st.write("**Repair Time Model Performance:**")
                        st.metric("R² Score", f"{results['r2_score']:.3f}")
                        st.metric("RMSE (days)", f"{results['rmse']:.1f}")
                        st.metric("Mean Repair Cycle", f"{results['mean_repair_time']:.1f} days")
                        st.metric("Median Repair Cycle", f"{results['median_repair_time']:.1f} days")
                    else:
                        st.error(f"Error: {results['error']}")
        
        # Model Performance Visualization
        if hasattr(st.session_state.enhanced_model, 'regression_results') and st.session_state.enhanced_model.regression_results:
            st.subheader("📊 Model Performance Visualization")
            
            results = st.session_state.enhanced_model.regression_results
            
            fig_scatter = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Random Forest Regression', 'Linear Regression')
            )
            
            fig_scatter.add_trace(
                go.Scatter(
                    x=results['test_predictions']['actual'],
                    y=results['test_predictions']['rf_predicted'],
                    mode='markers',
                    name='RF Predictions',
                    marker=dict(color='blue', opacity=0.6)
                ),
                row=1, col=1
            )
            
            fig_scatter.add_trace(
                go.Scatter(
                    x=results['test_predictions']['actual'],
                    y=results['test_predictions']['lr_predicted'],
                    mode='markers',
                    name='LR Predictions',
                    marker=dict(color='red', opacity=0.6)
                ),
                row=1, col=2
            )
            
            # Add perfect prediction line
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
            
            fig_scatter.update_layout(height=500, title_text="Actual vs Predicted Quantities")
            fig_scatter.update_xaxes(title_text="Actual Quantity")
            fig_scatter.update_yaxes(title_text="Predicted Quantity")
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("---")
        
        # 2.1 Predictive Material Needs
        st.subheader("🔮 2.1 Predictive Material Needs")
        st.write("Select a problem category to get material predictions based on historical patterns.")
        
        problem_categories = st.session_state.enhanced_model.get_problem_categories()
        
        if problem_categories:
            category_options = [f"{cat_id}: {cat_data['name']}" for cat_id, cat_data in problem_categories.items()]
            
            selected_category = st.selectbox(
                "Select Problem Category:",
                options=category_options,
                key="problem_category_select"
            )
            
            if st.button("Get Material Predictions", key="predict_material_needs"):
                if selected_category:
                    category_id = int(selected_category.split(':')[0])
                    
                    with st.spinner("Predicting materials..."):
                        predictions = st.session_state.enhanced_model.predict_material_by_category(category_id)
                        
                        if isinstance(predictions, list):
                            st.success("Material predictions generated!")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Top Material Recommendations:**")
                                for i, pred in enumerate(predictions, 1):
                                    confidence_pct = pred['confidence'] * 100
                                    st.metric(
                                        f"#{i} Material", 
                                        pred['material'],
                                        f"Confidence: {confidence_pct:.1f}%"
                                    )
                            
                            with col2:
                                st.write("**Sample Problems in This Category:**")
                                sample_problems = problem_categories[category_id]['problems'][:3]
                                for i, problem in enumerate(sample_problems, 1):
                                    st.write(f"{i}. {problem[:100]}...")
                            
                            # Visualize confidence levels
                            materials = [pred['material'] for pred in predictions]
                            confidences = [pred['confidence'] * 100 for pred in predictions]
                            
                            fig_conf = px.bar(
                                x=materials,
                                y=confidences,
                                title="Material Recommendation Confidence",
                                labels={'x': 'Material', 'y': 'Confidence (%)'}
                            )
                            fig_conf.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig_conf, use_container_width=True)
                        else:
                            st.error(predictions)
        else:
            st.info("Please train the classification model first to see problem categories.")
        
        st.markdown("---")
        
        # 2.2 Automatic Material Recommendation System
        st.subheader("🎯 2.2 Automatic Material Recommendation System")
        st.write("Get material recommendations based on specific problem descriptions.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Select from Common Problems:**")
            
            # Get common problem descriptions
            common_problems = df['ProblemDesc'].value_counts().head(10).index.tolist()
            
            selected_problem = st.selectbox(
                "Choose a problem:",
                options=["Select a problem..."] + common_problems,
                key="common_problem_select"
            )
            
            if st.button("Get Recommendations", key="get_recommendations"):
                if selected_problem != "Select a problem...":
                    with st.spinner("Analyzing problem and generating recommendations..."):
                        # Get materials historically used for this problem
                        problem_materials = df[df['ProblemDesc'] == selected_problem]['MatName'].value_counts()
                        
                        if len(problem_materials) > 0:
                            st.success("Historical material usage found!")
                            
                            # Display top materials used
                            st.write("**Most Frequently Used Materials:**")
                            for i, (material, count) in enumerate(problem_materials.head(5).items(), 1):
                                frequency_pct = (count / len(df[df['ProblemDesc'] == selected_problem])) * 100
                                st.metric(
                                    f"#{i} {material}", 
                                    f"{count} times used",
                                    f"Frequency: {frequency_pct:.1f}%"
                                )
                            
                            # Show efficiency metrics
                            st.write("**Efficiency Analysis:**")
                            problem_data = df[df['ProblemDesc'] == selected_problem]
                            
                            efficiency_metrics = []
                            for material in problem_materials.head(3).index:
                                material_data = problem_data[problem_data['MatName'] == material]
                                avg_qty = material_data['QtyOut'].mean()
                                avg_price = material_data['Price'].mean()
                                
                                efficiency_metrics.append({
                                    'Material': material,
                                    'Avg Quantity': avg_qty,
                                    'Avg Price': avg_price,
                                    'Cost per Unit': avg_price / avg_qty if avg_qty > 0 else 0
                                })
                            
                            efficiency_df = pd.DataFrame(efficiency_metrics)
                            st.dataframe(efficiency_df)
                        else:
                            st.warning("No historical data found for this problem.")
                else:
                    st.warning("Please select a problem.")
        
        with col2:
            st.write("**Custom Problem Description:**")
            
            custom_problem = st.text_area(
                "Enter your problem description:",
                placeholder="e.g., Motor coolant overheating, valve malfunction, etc.",
                key="custom_problem_input"
            )
            
            if st.button("Analyze Custom Problem", key="analyze_custom"):
                if custom_problem:
                    with st.spinner("Analyzing custom problem..."):
                        # Use the trained model to predict material
                        if hasattr(st.session_state.enhanced_model, 'material_classifier') and st.session_state.enhanced_model.material_classifier:
                            prediction = st.session_state.enhanced_model.predict_material(custom_problem)
                            
                            if isinstance(prediction, dict):
                                st.success("Material prediction generated!")
                                
                                confidence_pct = prediction['confidence'] * 100
                                st.metric(
                                    "Recommended Material", 
                                    prediction['material'],
                                    f"Confidence: {confidence_pct:.1f}%"
                                )
                                
                                # Show similar problems
                                st.write("**Similar Problems Found:**")
                                problem_words = custom_problem.lower().split()
                                similar_problems = []
                                
                                for _, row in df.iterrows():
                                    desc = str(row['ProblemDesc']).lower()
                                    if any(word in desc for word in problem_words):
                                        similar_problems.append({
                                            'Problem': row['ProblemDesc'],
                                            'Material': row['MatName'],
                                            'Quantity': row['QtyOut']
                                        })
                                
                                if similar_problems:
                                    similar_df = pd.DataFrame(similar_problems[:5])
                                    st.dataframe(similar_df)
                            else:
                                st.error(prediction)
                        else:
                            st.warning("Please train the classification model first.")
                else:
                    st.warning("Please enter a problem description.")
        
        st.markdown("---")
        
        # 2.3 Repair Time Cycle Prediction
        st.subheader("⏰ 2.3 Repair Time Cycle Prediction & Schedule")
        st.write("Predict repair cycles and view upcoming maintenance schedule.")
        
        # Generate repair schedule
        if st.button("Generate Repair Schedule", key="generate_schedule"):
            with st.spinner("Generating repair schedule..."):
                if hasattr(st.session_state.enhanced_model, 'repair_time_predictor') and st.session_state.enhanced_model.repair_time_predictor:
                    
                    # Generate schedule
                    schedule_df = st.session_state.enhanced_model.generate_repair_schedule(df, days_ahead=90)
                    
                    if not schedule_df.empty:
                        st.success("Repair schedule generated!")
                        
                        # Display metrics
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            high_priority = len(schedule_df[schedule_df['Priority'] == 'High'])
                            st.metric("🔴 High Priority", high_priority, "≤ 7 days")
                        
                        with col2:
                            medium_priority = len(schedule_df[schedule_df['Priority'] == 'Medium'])
                            st.metric("🟡 Medium Priority", medium_priority, "8-30 days")
                        
                        with col3:
                            low_priority = len(schedule_df[schedule_df['Priority'] == 'Low'])
                            st.metric("🟢 Low Priority", low_priority, "> 30 days")
                        
                        with col4:
                            avg_cycle = schedule_df['DaysFromNow'].mean()
                            st.metric("📊 Avg Cycle", f"{avg_cycle:.1f} days")
                        
                        # Display schedule table with color coding
                        st.write("**Upcoming Repair Schedule:**")
                        
                        # Create styled dataframe
                        def color_priority(val):
                            if val == 'High':
                                return 'background-color: #ffebee'
                            elif val == 'Medium':
                                return 'background-color: #fff3e0'
                            else:
                                return 'background-color: #e8f5e8'
                        
                        styled_df = schedule_df[['WorkOrderNo', 'ProblemType', 'LastRepairDate', 
                                               'PredictedNextRepair', 'DaysFromNow', 'Priority']].style.applymap(
                            color_priority, subset=['Priority']
                        )
                        
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Visualize repair timeline
                        fig_timeline = px.scatter(
                            schedule_df,
                            x='DaysFromNow',
                            y='WorkOrderNo',
                            color='Priority',
                            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'},
                            title="Repair Timeline - Days from Today",
                            hover_data=['ProblemType', 'PredictedNextRepair']
                        )
                        
                        fig_timeline.update_layout(
                            xaxis_title="Days from Today",
                            yaxis_title="Work Order",
                            height=500
                        )
                        
                        st.plotly_chart(fig_timeline, use_container_width=True)
                        
                        # Priority distribution
                        priority_counts = schedule_df['Priority'].value_counts()
                        
                        fig_priority = px.pie(
                            values=priority_counts.values,
                            names=priority_counts.index,
                            title="Repair Priority Distribution",
                            color_discrete_map={'High': 'red', 'Medium': 'orange', 'Low': 'green'}
                        )
                        
                        st.plotly_chart(fig_priority, use_container_width=True)
                        
                    else:
                        st.warning("No repair schedule data available.")
                else:
                    st.warning("Please train the repair time predictor first.")
        
        # Individual repair time prediction
        st.write("**Individual Repair Time Prediction:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            work_orders = df['WorkOrderNo'].unique()
            selected_wo = st.selectbox(
                "Select Work Order:",
                options=work_orders,
                key="work_order_select"
            )
        
        with col2:
            last_repair_date = st.date_input(
                "Last Repair Date:",
                value=datetime.now() - timedelta(days=30),
                key="last_repair_date"
            )
        
        if st.button("Predict Next Repair", key="predict_individual_repair"):
            if selected_wo:
                with st.spinner("Predicting repair time..."):
                    prediction = st.session_state.enhanced_model.predict_repair_time(
                        df, selected_wo, last_repair_date
                    )
                    
                    if isinstance(prediction, dict):
                        st.success("Repair time predicted!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Predicted Cycle", f"{prediction['predicted_days']} days")
                        
                        with col2:
                            st.metric("Next Repair Date", prediction['next_repair_date'])
                        
                        with col3:
                            days_from_now = prediction['days_from_now']
                            if days_from_now <= 7:
                                st.metric("Days from Now", days_from_now, "🔴 High Priority")
                            elif days_from_now <= 30:
                                st.metric("Days from Now", days_from_now, "🟡 Medium Priority")
                            else:
                                st.metric("Days from Now", days_from_now, "🟢 Low Priority")
                        
                        # Show historical data for this work order
                        wo_data = df[df['WorkOrderNo'] == selected_wo]
                        if len(wo_data) > 1:
                            st.write("**Historical Repair Data:**")
                            
                            wo_history = wo_data[['TransOutDate', 'ProblemDesc', 'MatName', 'QtyOut']].sort_values('TransOutDate')
                            st.dataframe(wo_history)
                    else:
                        st.error(prediction)
            else:
                st.warning("Please select a work order.")
        
        st.markdown("---")
        
        # Model Management
        st.subheader("💾 Model Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Save All Models", key="save_models"):
                try:
                    st.session_state.enhanced_model.save_models()
                    st.success("All models saved successfully!")
                except Exception as e:
                    st.error(f"Error saving models: {e}")
        
        with col2:
            if st.button("Load Saved Models", key="load_models"):
                try:
                    success = st.session_state.enhanced_model.load_models()
                    if success:
                        st.success("Models loaded successfully!")
                    else:
                        st.error("Failed to load models")
                except Exception as e:
                    st.error(f"Error loading models: {e}")
        
        with col3:
            if st.button("Check Model Status", key="check_status"):
                status = st.session_state.enhanced_model.get_model_status()
                
                st.write("**Model Status:**")
                for model_name, is_trained in status.items():
                    status_icon = "✅" if is_trained else "❌"
                    st.write(f"{status_icon} {model_name.replace('_', ' ').title()}")
        
        # Export predictions
        st.subheader("📤 Export Predictions")
        
        if st.button("Export Repair Schedule", key="export_schedule"):
            if hasattr(st.session_state.enhanced_model, 'repair_time_predictor') and st.session_state.enhanced_model.repair_time_predictor:
                schedule_df = st.session_state.enhanced_model.generate_repair_schedule(df, days_ahead=365)
                
                if not schedule_df.empty:
                    csv = schedule_df.to_csv(index=False)
                    st.download_button(
                        label="Download Repair Schedule CSV",
                        data=csv,
                        file_name=f"repair_schedule_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No schedule data to export.")
            else:
                st.warning("Please train the repair time predictor first.")
    
    else:
        st.warning("Please upload and preprocess data first!")
        
        # Show sample data format
        st.subheader("📋 Expected Data Format")
        st.write("Your data should contain the following columns:")
        
        sample_data = pd.DataFrame({
            'WorkOrderNo': ['17RFM034931', '15RFM029301', '15RFM029584'],
            'ProblemDesc': ['AMPLITUD TO HIGH ENCODER', 'PINTU TROUBLE', 'MOTOR COOLANT KEBAKAR'],
            'MatName': ['RING, O', 'Valve', 'Valve'],
            'QtyOut': [2, 1, 1],
            'Price': [0.05, 152, 15],
            'TransOutDate': ['2017-04-21', '2015-07-27', '2015-12-23']
        })
        
        st.dataframe(sample_data)