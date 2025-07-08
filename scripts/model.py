import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
import joblib
import os

class MaterialPredictionModel:
    """
    Class untuk model prediksi kebutuhan material
    """
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.material_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.quantity_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.simple_regressor = LinearRegression()
        
        # Storage untuk hasil training
        self.classification_results = {}
        self.regression_results = {}
        
    def prepare_classification_data(self, df):
        """
        Prepare data untuk klasifikasi material
        
        Args:
            df: DataFrame yang sudah dipreprocess
        
        Returns:
            X, y untuk training
        """
        # Filter data yang valid
        valid_data = df[
            (df['ProblemDesc'].notna()) & 
            (df['ProblemDesc'] != 'No Description') &
            (df['MatName'].notna()) &
            (df['MatName'] != 'Unknown Material')
        ].copy()
        
        # Ambil material yang paling sering muncul (top 20)
        top_materials = valid_data['MatName'].value_counts().head(20).index.tolist()
        valid_data = valid_data[valid_data['MatName'].isin(top_materials)]
        
        X = valid_data['ProblemDesc_Clean']
        y = valid_data['MatName']
        
        return X, y
    
    def prepare_regression_data(self, df):
        """
        Prepare data untuk regresi quantity
        
        Args:
            df: DataFrame yang sudah dipreprocess
        
        Returns:
            X, y untuk training
        """
        # Filter data yang valid
        valid_data = df[
            (df['ProblemDesc'].notna()) & 
            (df['QtyOut'] > 0) &
            (df['ProblemDesc_Length'] > 0)
        ].copy()
        
        # Feature engineering untuk regresi
        X = valid_data[['ProblemDesc_Length', 'ProblemDesc_WordCount']].values
        y = valid_data['QtyOut'].values
        
        return X, y
    
    def train_material_classifier(self, df):
        """
        Train model klasifikasi untuk memprediksi material
        
        Args:
            df: DataFrame training data
        
        Returns:
            Dictionary berisi hasil training
        """
        print("Training material classification model...")
        
        # Prepare data
        X, y = self.prepare_classification_data(df)
        
        if len(X) == 0:
            return {"error": "No valid data for classification"}
        
        # Transform text dengan TF-IDF
        X_tfidf = self.tfidf_vectorizer.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Training
        self.material_classifier.fit(X_train, y_train)
        
        # Prediction
        y_pred = self.material_classifier.predict(X_test)
        
        # Evaluation
        accuracy = self.material_classifier.score(X_test, y_test)
        
        # Cross validation
        cv_scores = cross_val_score(self.material_classifier, X_train, y_train, cv=5)
        
        results = {
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': dict(zip(
                self.tfidf_vectorizer.get_feature_names_out(),
                self.material_classifier.feature_importances_
            ))
        }
        
        self.classification_results = results
        return results
    
    def train_quantity_regressor(self, df):
        """
        Train model regresi untuk memprediksi quantity
        
        Args:
            df: DataFrame training data
        
        Returns:
            Dictionary berisi hasil training
        """
        print("Training quantity regression model...")
        
        # Prepare data
        X, y = self.prepare_regression_data(df)
        
        if len(X) == 0:
            return {"error": "No valid data for regression"}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Training Random Forest
        self.quantity_regressor.fit(X_train, y_train)
        y_pred_rf = self.quantity_regressor.predict(X_test)
        
        # Training Linear Regression
        self.simple_regressor.fit(X_train, y_train)
        y_pred_lr = self.simple_regressor.predict(X_test)
        
        # Evaluation
        rf_r2 = r2_score(y_test, y_pred_rf)
        rf_mse = mean_squared_error(y_test, y_pred_rf)
        rf_rmse = np.sqrt(rf_mse)
        
        lr_r2 = r2_score(y_test, y_pred_lr)
        lr_mse = mean_squared_error(y_test, y_pred_lr)
        lr_rmse = np.sqrt(lr_mse)
        
        results = {
            'random_forest': {
                'r2_score': rf_r2,
                'mse': rf_mse,
                'rmse': rf_rmse,
                'feature_importance': self.quantity_regressor.feature_importances_
            },
            'linear_regression': {
                'r2_score': lr_r2,
                'mse': lr_mse,
                'rmse': lr_rmse
            },
            'test_predictions': {
                'actual': y_test,
                'rf_predicted': y_pred_rf,
                'lr_predicted': y_pred_lr
            }
        }
        
        self.regression_results = results
        return results
    
    def predict_material(self, problem_description):
        """
        Prediksi material berdasarkan deskripsi masalah
        
        Args:
            problem_description: String deskripsi masalah
        
        Returns:
            String nama material yang diprediksi
        """
        if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            return "Model not trained yet"
        
        # Transform text
        text_tfidf = self.tfidf_vectorizer.transform([problem_description])
        
        # Predict
        prediction = self.material_classifier.predict(text_tfidf)[0]
        probability = self.material_classifier.predict_proba(text_tfidf)[0].max()
        
        return {
            'material': prediction,
            'confidence': probability
        }
    
    def predict_quantity(self, problem_desc_length, word_count):
        """
        Prediksi quantity berdasarkan karakteristik deskripsi masalah
        
        Args:
            problem_desc_length: Panjang deskripsi masalah
            word_count: Jumlah kata dalam deskripsi
        
        Returns:
            Float quantity yang diprediksi
        """
        if not hasattr(self.quantity_regressor, 'feature_importances_'):
            return "Model not trained yet"
        
        # Create feature array
        features = np.array([[problem_desc_length, word_count]])
        
        # Predict dengan Random Forest
        rf_prediction = self.quantity_regressor.predict(features)[0]
        
        # Predict dengan Linear Regression
        lr_prediction = self.simple_regressor.predict(features)[0]
        
        return {
            'rf_prediction': max(1, int(rf_prediction)),  # Minimal 1
            'lr_prediction': max(1, int(lr_prediction))   # Minimal 1
        }
    
    def save_models(self, model_dir='models'):
        """
        Simpan trained models
        
        Args:
            model_dir: Directory untuk menyimpan models
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Save models
        joblib.dump(self.tfidf_vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
        joblib.dump(self.material_classifier, os.path.join(model_dir, 'material_classifier.pkl'))
        joblib.dump(self.quantity_regressor, os.path.join(model_dir, 'quantity_regressor.pkl'))
        joblib.dump(self.simple_regressor, os.path.join(model_dir, 'simple_regressor.pkl'))
        
        print(f"Models saved to {model_dir}")
    
    def load_models(self, model_dir='models'):
        """
        Load trained models
        
        Args:
            model_dir: Directory models tersimpan
        """
        try:
            self.tfidf_vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
            self.material_classifier = joblib.load(os.path.join(model_dir, 'material_classifier.pkl'))
            self.quantity_regressor = joblib.load(os.path.join(model_dir, 'quantity_regressor.pkl'))
            self.simple_regressor = joblib.load(os.path.join(model_dir, 'simple_regressor.pkl'))
            print("Models loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False