import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import streamlit as st

@st.cache_resource
def train_models(X, y):
    """Train multiple ML models for student performance prediction"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
        'Linear Regression': LinearRegression()
    }
    
    trained_models = {}
    model_metrics = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        trained_models[name] = model
        model_metrics[name] = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cv_score': cross_val_score(model, X_train, y_train, cv=5, scoring='r2').mean()
        }
    
    return trained_models, model_metrics, X_test, y_test

def predict_performance(model, features):
    """Predict student performance given features"""
    prediction = model.predict([features])[0]
    return max(0, min(100, prediction))

def get_feature_importance(model, feature_names):
    """Get feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        return importance_df
    return None

def classify_risk_level(score):
    """Classify student risk level based on predicted score"""
    if score >= 80:
        return "Low Risk", "🟢", "Excellent performance, no intervention needed"
    elif score >= 65:
        return "Medium-Low Risk", "🟡", "Good performance, minor support recommended"
    elif score >= 50:
        return "Medium-High Risk", "🟠", "At-risk student, support strongly recommended"
    else:
        return "High Risk", "🔴", "Critical intervention needed"

def generate_recommendations(features_dict):
    """Generate personalized recommendations based on student features"""
    recommendations = []
    
    if features_dict['attendance_rate'] < 75:
        recommendations.append("📅 **Attendance Alert**: Attendance below 75%. Encourage regular class attendance.")
    
    if features_dict['study_hours_per_week'] < 10:
        recommendations.append("📚 **Study Time**: Increase study hours. Recommend at least 10-15 hours per week.")
    
    if features_dict['previous_grade'] < 60:
        recommendations.append("📖 **Academic Support**: Previous grades indicate need for tutoring or extra help.")
    
    if features_dict['internet_usage_hours'] > 4:
        recommendations.append("💻 **Screen Time**: High internet usage detected. Balance online time with studies.")
    
    if features_dict['extracurricular_activities'] == 0:
        recommendations.append("⚽ **Engagement**: No extracurricular activities. Consider clubs or sports for balanced development.")
    
    if features_dict['parental_involvement'] < 3:
        recommendations.append("👨‍👩‍👧 **Parent Engagement**: Increase parental involvement in academic activities.")
    
    if not recommendations:
        recommendations.append("✅ **Great Work**: Student showing positive indicators across all metrics!")
    
    return recommendations
