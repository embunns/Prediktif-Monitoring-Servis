import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_squared_error
from sklearn.cluster import KMeans
import joblib
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MaterialPredictionModel:
    def __init__(self):
        self.material_classifier = None
        self.quantity_regressor_rf = None
        self.quantity_regressor_lr = None
        self.repair_time_predictor = None
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.classification_results = None
        self.regression_results = None
        self.repair_time_results = None
        self.repair_time_features = None  # Store feature names used in training
        self.models_dir = "models/saved_models"
        self.problem_categories = {}
        self.material_recommendations = {}
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def prepare_features(self, df):
        """Prepare features for model training"""
        features = []
        df_encoded = df.copy()
        
        # Text-based features
        if 'ProblemDesc_Length' in df.columns:
            features.append('ProblemDesc_Length')
        if 'ProblemDesc_WordCount' in df.columns:
            features.append('ProblemDesc_WordCount')
        
        # Machine type encoding
        if 'MachineType' in df.columns:
            df_encoded['MachineType_Encoded'] = pd.Categorical(df_encoded['MachineType']).codes
            features.append('MachineType_Encoded')
        
        # Time-based features
        if 'Year' in df.columns:
            features.append('Year')
        if 'Month' in df.columns:
            features.append('Month')
        if 'Quarter' in df.columns:
            features.append('Quarter')
        
        # Price and quantity features
        if 'Price' in df.columns:
            df_encoded['Price_Log'] = np.log1p(df_encoded['Price'])
            features.append('Price_Log')
        
        return df_encoded[features] if features else pd.DataFrame()
    
    def create_problem_categories(self, df):
        """Create problem categories from descriptions"""
        try:
            # Extract common problem patterns
            problem_desc = df['ProblemDesc_Clean'].fillna('').astype(str)
            
            # Use TF-IDF to find similar problems
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(problem_desc)
            
            # Use KMeans to cluster similar problems
            n_clusters = min(20, max(2, len(df)//10))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(tfidf_matrix)
            
            # Create problem categories
            df['ProblemCategory'] = clusters
            
            # Create category mapping
            category_problems = {}
            for i, category in enumerate(clusters):
                if category not in category_problems:
                    category_problems[category] = []
                category_problems[category].append(problem_desc.iloc[i])
            
            # Create readable category names
            self.problem_categories = {}
            for category, problems in category_problems.items():
                # Get most common words in this category
                all_words = ' '.join(problems).split()
                word_freq = pd.Series(all_words).value_counts()
                top_words = word_freq.head(3).index.tolist()
                category_name = f"Category_{category}: {' '.join(top_words)}"
                self.problem_categories[category] = {
                    'name': category_name,
                    'problems': problems[:5]  # Store sample problems
                }
            
            return df
            
        except Exception as e:
            print(f"Error creating problem categories: {str(e)}")
            return df
    
    def train_material_classifier(self, df):
        """Train material classification model with improved handling"""
        try:
            if 'ProblemDesc_Clean' not in df.columns or 'MatName' not in df.columns:
                return {'error': 'Required columns not found'}
            
            # Create problem categories first
            df = self.create_problem_categories(df)
            
            X_text = df['ProblemDesc_Clean'].fillna('')
            y = df['MatName']
            
            # Remove empty or invalid entries
            valid_mask = (X_text != '') & (y != '') & (~y.isna())
            X_text = X_text[valid_mask]
            y = y[valid_mask]
            
            if len(X_text) == 0:
                return {'error': 'No valid training data found'}
            
            # Filter out classes with very few samples
            class_counts = y.value_counts()
            valid_classes = class_counts[class_counts >= 2].index
            
            # Filter dataset to only include valid classes
            valid_mask = y.isin(valid_classes)
            X_text = X_text[valid_mask]
            y = y[valid_mask]
            
            if len(X_text) == 0:
                return {'error': 'No valid classes with sufficient samples'}
            
            # TF-IDF Vectorization
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000, 
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            X_tfidf = self.tfidf_vectorizer.fit_transform(X_text)
            
            # Label encoding
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_tfidf, y_encoded, test_size=0.2, random_state=42, 
                stratify=y_encoded if len(np.unique(y_encoded)) > 1 else None
            )
            
            # Train model
            self.material_classifier = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                min_samples_split=5,
                min_samples_leaf=2
            )
            self.material_classifier.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.material_classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(self.material_classifier, X_tfidf, y_encoded, cv=min(5, len(np.unique(y_encoded))))
            
            # Classification report with proper labels
            unique_labels = np.unique(y_test)
            target_names = [self.label_encoder.classes_[i] for i in unique_labels]
            
            classification_rep = classification_report(
                y_test, y_pred, 
                labels=unique_labels,
                target_names=target_names,
                output_dict=False,
                zero_division=0
            )
            
            self.classification_results = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_rep,
                'n_classes': len(self.label_encoder.classes_)
            }
            
            return self.classification_results
            
        except Exception as e:
            return {'error': str(e)}
    
    def train_quantity_regressor(self, df):
        """Train quantity regression model"""
        try:
            features_df = self.prepare_features(df)
            
            if features_df.empty:
                return {'error': 'No suitable features found for regression'}
            
            X = features_df
            y = df['QtyOut']
            
            # Remove NaN values
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) == 0:
                return {'error': 'No valid data for regression'}
            
            # Remove outliers
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (y >= lower_bound) & (y <= upper_bound)
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                return {'error': 'No data remaining after outlier removal'}
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Random Forest Regressor
            self.quantity_regressor_rf = RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                min_samples_split=5,
                min_samples_leaf=2
            )
            self.quantity_regressor_rf.fit(X_train, y_train)
            
            # Linear Regression
            self.quantity_regressor_lr = LinearRegression()
            self.quantity_regressor_lr.fit(X_train, y_train)
            
            # Predictions
            y_pred_rf = self.quantity_regressor_rf.predict(X_test)
            y_pred_lr = self.quantity_regressor_lr.predict(X_test)
            
            # Metrics
            rf_r2 = r2_score(y_test, y_pred_rf)
            rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
            
            lr_r2 = r2_score(y_test, y_pred_lr)
            lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
            
            self.regression_results = {
                'random_forest': {
                    'r2_score': rf_r2,
                    'rmse': rf_rmse
                },
                'linear_regression': {
                    'r2_score': lr_r2,
                    'rmse': lr_rmse
                },
                'test_predictions': {
                    'actual': y_test,
                    'rf_predicted': y_pred_rf,
                    'lr_predicted': y_pred_lr
                }
            }
            
            return self.regression_results
            
        except Exception as e:
            return {'error': str(e)}
    
    def train_repair_time_predictor(self, df):
        """Train repair time prediction model"""
        try:
            if 'TransOutDate' not in df.columns:
                return {'error': 'TransOutDate column not found'}
            
            # Create repair time features
            df_repair = df.copy()
            df_repair['TransOutDate'] = pd.to_datetime(df_repair['TransOutDate'], errors='coerce')
            
            # Remove rows with invalid dates
            df_repair = df_repair.dropna(subset=['TransOutDate'])
            
            if len(df_repair) == 0:
                return {'error': 'No valid dates found'}
            
            # Group by WorkOrderNo and calculate repair intervals
            repair_intervals = []
            
            for work_order in df_repair['WorkOrderNo'].unique():
                wo_data = df_repair[df_repair['WorkOrderNo'] == work_order].sort_values('TransOutDate')
                
                if len(wo_data) > 1:
                    # Calculate intervals between consecutive repairs
                    for i in range(1, len(wo_data)):
                        days_between = (wo_data.iloc[i]['TransOutDate'] - wo_data.iloc[i-1]['TransOutDate']).days
                        
                        if days_between > 0 and days_between <= 365:  # Valid interval
                            repair_intervals.append({
                                'WorkOrderNo': work_order,
                                'Days_Since_Last_Repair': days_between,
                                'ProblemDesc_Length': len(str(wo_data.iloc[i]['ProblemDesc'])),
                                'ProblemDesc_WordCount': len(str(wo_data.iloc[i]['ProblemDesc']).split()),
                                'MachineType': wo_data.iloc[i]['MachineType'] if 'MachineType' in wo_data.columns else 'UNKNOWN',
                                'Price': wo_data.iloc[i]['Price'] if 'Price' in wo_data.columns else 0,
                                'QtyOut': wo_data.iloc[i]['QtyOut'] if 'QtyOut' in wo_data.columns else 1
                            })
                else:
                    # Single repair record - use average interval
                    repair_intervals.append({
                        'WorkOrderNo': work_order,
                        'Days_Since_Last_Repair': 60,  # Default 60 days
                        'ProblemDesc_Length': len(str(wo_data.iloc[0]['ProblemDesc'])),
                        'ProblemDesc_WordCount': len(str(wo_data.iloc[0]['ProblemDesc']).split()),
                        'MachineType': wo_data.iloc[0]['MachineType'] if 'MachineType' in wo_data.columns else 'UNKNOWN',
                        'Price': wo_data.iloc[0]['Price'] if 'Price' in wo_data.columns else 0,
                        'QtyOut': wo_data.iloc[0]['QtyOut'] if 'QtyOut' in wo_data.columns else 1
                    })
            
            if not repair_intervals:
                return {'error': 'No repair intervals could be calculated'}
            
            # Convert to DataFrame
            repair_df = pd.DataFrame(repair_intervals)
            
            # Prepare features
            features = ['ProblemDesc_Length', 'ProblemDesc_WordCount']
            
            # Add machine type encoding
            if 'MachineType' in repair_df.columns:
                repair_df['MachineType_Encoded'] = pd.Categorical(repair_df['MachineType']).codes
                features.append('MachineType_Encoded')
            
            # Add price log
            if 'Price' in repair_df.columns:
                repair_df['Price_Log'] = np.log1p(repair_df['Price'])
                features.append('Price_Log')
            
            # Add quantity
            if 'QtyOut' in repair_df.columns:
                features.append('QtyOut')
            
            # Store the feature names used in training
            self.repair_time_features = features
            
            X = repair_df[features]
            y = repair_df['Days_Since_Last_Repair']
            
            # Remove NaN values
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) == 0:
                return {'error': 'No valid data for repair time prediction'}
            
            # Remove outliers
            Q1 = y.quantile(0.25)
            Q3 = y.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = max(1, Q1 - 1.5 * IQR)  # Minimum 1 day
            upper_bound = min(365, Q3 + 1.5 * IQR)  # Maximum 365 days
            
            mask = (y >= lower_bound) & (y <= upper_bound)
            X = X[mask]
            y = y[mask]
            
            if len(X) == 0:
                return {'error': 'No data remaining after outlier removal'}
            
            # Train-test split
            if len(X) < 10:
                # Use all data for training if too few samples
                X_train, X_test, y_train, y_test = X, X, y, y
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            
            # Train repair time predictor
            self.repair_time_predictor = RandomForestRegressor(
                n_estimators=100, 
                random_state=42,
                min_samples_split=max(2, len(X_train)//10),
                min_samples_leaf=1
            )
            self.repair_time_predictor.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.repair_time_predictor.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            self.repair_time_results = {
                'r2_score': r2,
                'rmse': rmse,
                'mean_repair_time': y.mean(),
                'median_repair_time': y.median(),
                'feature_names': features
            }
            
            return self.repair_time_results
                
        except Exception as e:
            return {'error': str(e)}
    
    def get_problem_categories(self):
        """Get available problem categories"""
        return self.problem_categories
    
    def predict_material_by_category(self, problem_category):
        """Predict material based on problem category"""
        try:
            if self.material_classifier is None or self.tfidf_vectorizer is None:
                return "Model not trained. Please train the classification model first."
            
            if problem_category not in self.problem_categories:
                return "Invalid problem category"
            
            # Use a sample problem from the category
            sample_problem = self.problem_categories[problem_category]['problems'][0]
            
            X_tfidf = self.tfidf_vectorizer.transform([sample_problem])
            
            prediction_encoded = self.material_classifier.predict(X_tfidf)[0]
            probabilities = self.material_classifier.predict_proba(X_tfidf)[0]
            
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[-3:][::-1]
            top_materials = []
            
            for idx in top_indices:
                material_name = self.label_encoder.inverse_transform([idx])[0]
                confidence = probabilities[idx]
                top_materials.append({
                    'material': material_name,
                    'confidence': confidence
                })
            
            return top_materials
            
        except Exception as e:
            return f"Error making prediction: {str(e)}"
    
    def predict_material(self, problem_description):
        """Predict material based on problem description"""
        try:
            if self.material_classifier is None or self.tfidf_vectorizer is None:
                return "Model not trained. Please train the classification model first."
            
            if self.label_encoder is None:
                return "Label encoder not available. Please train the classification model first."
            
            # Clean and vectorize the problem description
            problem_clean = str(problem_description).upper()
            X_tfidf = self.tfidf_vectorizer.transform([problem_clean])
            
            # Make prediction
            prediction_encoded = self.material_classifier.predict(X_tfidf)[0]
            probabilities = self.material_classifier.predict_proba(X_tfidf)[0]
            
            # Get material name and confidence
            material_name = self.label_encoder.inverse_transform([prediction_encoded])[0]
            confidence = probabilities[prediction_encoded]
            
            return {
                'material': material_name,
                'confidence': confidence
            }
            
        except Exception as e:
            return f"Error making prediction: {str(e)}"
    
    def predict_repair_time(self, df, work_order_no, last_repair_date=None):
        """Predict next repair time using actual work order data"""
        try:
            if self.repair_time_predictor is None:
                return "Repair time model not trained."
            
            # Use the stored feature names from training
            if self.repair_time_features is None:
                return "Feature information not available. Please retrain the model."
            
            # Get actual data for the specific work order
            wo_data = df[df['WorkOrderNo'] == work_order_no]
            
            if wo_data.empty:
                return f"Work order {work_order_no} not found in data."
            
            # Get the most recent record for this work order
            latest_record = wo_data.iloc[-1]
            
            # Extract actual features from the work order data
            problem_desc = str(latest_record.get('ProblemDesc', ''))
            
            feature_data = {
                'ProblemDesc_Length': len(problem_desc),
                'ProblemDesc_WordCount': len(problem_desc.split()) if problem_desc else 0,
                'MachineType_Encoded': latest_record.get('MachineType_Encoded', 0),
                'Price_Log': np.log1p(latest_record.get('Price', 50)),
                'QtyOut': latest_record.get('QtyOut', 1)
            }
            
            # Create DataFrame with only the features used in training
            features = pd.DataFrame({
                feature: [feature_data.get(feature, 0)] 
                for feature in self.repair_time_features
            })
            
            # Make prediction
            predicted_days = self.repair_time_predictor.predict(features)[0]
            
            # Ensure prediction is within reasonable bounds
            predicted_days = max(7, min(365, predicted_days))
            
            # Calculate next repair date
            if last_repair_date:
                try:
                    if isinstance(last_repair_date, str):
                        last_date = pd.to_datetime(last_repair_date)
                    else:
                        last_date = pd.to_datetime(last_repair_date)
                    next_repair_date = last_date + timedelta(days=int(predicted_days))
                except:
                    next_repair_date = datetime.now() + timedelta(days=int(predicted_days))
            else:
                next_repair_date = datetime.now() + timedelta(days=int(predicted_days))
            
            days_from_now = (next_repair_date - datetime.now()).days
            
            return {
                'predicted_days': int(predicted_days),
                'next_repair_date': next_repair_date.strftime('%Y-%m-%d'),
                'days_from_now': days_from_now
            }
            
        except Exception as e:
            return f"Error predicting repair time: {str(e)}"

    def generate_repair_schedule(self, df, days_ahead=90):
        """Generate repair schedule table"""
        try:
            if self.repair_time_predictor is None:
                return pd.DataFrame()
            
            # Get unique work orders
            work_orders = df['WorkOrderNo'].unique()
            
            schedule_data = []
            current_date = datetime.now()
            
            for wo in work_orders:
                wo_data = df[df['WorkOrderNo'] == wo]
                if len(wo_data) > 0:
                    # Get last repair date
                    try:
                        wo_dates = pd.to_datetime(wo_data['TransOutDate'], errors='coerce')
                        valid_dates = wo_dates.dropna()
                        
                        if len(valid_dates) > 0:
                            last_repair = valid_dates.max()
                        else:
                            continue
                    except:
                        continue
                    
                    # Predict next repair using actual data
                    prediction = self.predict_repair_time(df, wo, last_repair)
                    
                    if isinstance(prediction, dict):
                        days_from_now = prediction['days_from_now']
                        
                        if days_from_now <= days_ahead:
                            # Determine priority
                            if days_from_now <= 7:
                                priority = 'High'
                                color = 'red'
                            elif days_from_now <= 30:
                                priority = 'Medium'
                                color = 'orange'
                            else:
                                priority = 'Low'
                                color = 'green'
                            
                            # Get problem description
                            problem_desc = str(wo_data['ProblemDesc'].iloc[0])
                            problem_type = problem_desc[:50] + '...' if len(problem_desc) > 50 else problem_desc
                            
                            schedule_data.append({
                                'WorkOrderNo': wo,
                                'LastRepairDate': last_repair.strftime('%Y-%m-%d'),
                                'PredictedNextRepair': prediction['next_repair_date'],
                                'DaysFromNow': days_from_now,
                                'Priority': priority,
                                'Color': color,
                                'ProblemType': problem_type
                            })
            
            # Sort by days from now
            schedule_df = pd.DataFrame(schedule_data)
            if not schedule_df.empty:
                schedule_df = schedule_df.sort_values('DaysFromNow')
            
            return schedule_df
            
        except Exception as e:
            print(f"Error generating repair schedule: {str(e)}")
            return pd.DataFrame()
    
    def save_models(self):
        """Save all trained models"""
        try:
            models_to_save = {
                'material_classifier': self.material_classifier,
                'quantity_regressor_rf': self.quantity_regressor_rf,
                'quantity_regressor_lr': self.quantity_regressor_lr,
                'repair_time_predictor': self.repair_time_predictor,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'label_encoder': self.label_encoder
            }
            
            for model_name, model in models_to_save.items():
                if model is not None:
                    joblib.dump(model, os.path.join(self.models_dir, f'{model_name}.pkl'))
            
            # Save repair time features
            if self.repair_time_features is not None:
                joblib.dump(self.repair_time_features, 
                           os.path.join(self.models_dir, 'repair_time_features.pkl'))
            
            # Save problem categories
            if self.problem_categories:
                joblib.dump(self.problem_categories, 
                           os.path.join(self.models_dir, 'problem_categories.pkl'))
            
            # Save results
            results_to_save = {
                'classification_results': self.classification_results,
                'regression_results': self.regression_results,
                'repair_time_results': self.repair_time_results
            }
            
            for result_name, result in results_to_save.items():
                if result is not None:
                    joblib.dump(result, os.path.join(self.models_dir, f'{result_name}.pkl'))
            
            return True
            
        except Exception as e:
            raise Exception(f"Error saving models: {str(e)}")
    
    def load_models(self):
        """Load all saved models"""
        try:
            model_files = {
                'material_classifier': 'material_classifier.pkl',
                'quantity_regressor_rf': 'quantity_regressor_rf.pkl',
                'quantity_regressor_lr': 'quantity_regressor_lr.pkl',
                'repair_time_predictor': 'repair_time_predictor.pkl',
                'tfidf_vectorizer': 'tfidf_vectorizer.pkl',
                'label_encoder': 'label_encoder.pkl'
            }
            
            for attr_name, filename in model_files.items():
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    setattr(self, attr_name, joblib.load(filepath))
            
            # Load repair time features
            features_path = os.path.join(self.models_dir, 'repair_time_features.pkl')
            if os.path.exists(features_path):
                self.repair_time_features = joblib.load(features_path)
            
            # Load problem categories
            categories_path = os.path.join(self.models_dir, 'problem_categories.pkl')
            if os.path.exists(categories_path):
                self.problem_categories = joblib.load(categories_path)
            
            # Load results
            result_files = {
                'classification_results': 'classification_results.pkl',
                'regression_results': 'regression_results.pkl',
                'repair_time_results': 'repair_time_results.pkl'
            }
            
            for attr_name, filename in result_files.items():
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    setattr(self, attr_name, joblib.load(filepath))
            
            return True
            
        except Exception as e:
            raise Exception(f"Error loading models: {str(e)}")
    
    def get_model_status(self):
        """Get status of all models"""
        status = {
            'material_classifier': self.material_classifier is not None,
            'quantity_regressor_rf': self.quantity_regressor_rf is not None,
            'quantity_regressor_lr': self.quantity_regressor_lr is not None,
            'repair_time_predictor': self.repair_time_predictor is not None,
            'tfidf_vectorizer': self.tfidf_vectorizer is not None,
            'label_encoder': self.label_encoder is not None,
            'problem_categories': len(self.problem_categories) > 0
        }
        return status