import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from student_model import (
    train_models, predict_performance, get_feature_importance,
    classify_risk_level, generate_recommendations
)
from data_generator import (
    generate_student_data, get_sample_student,
    get_feature_descriptions, get_feature_ranges
)
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.markdown("""Exam Prediction and Risk Assessment""")

if 'dataset' not in st.session_state:
    st.session_state.dataset = generate_student_data(500)
    
if 'models_trained' not in st.session_state:
    df = st.session_state.dataset
    feature_cols = ['attendance_rate', 'study_hours_per_week', 'previous_grade', 
                    'internet_usage_hours', 'sleep_hours', 'extracurricular_activities',
                    'parental_involvement', 'tutoring_sessions', 'assignment_completion_rate']
    X = df[feature_cols]
    y = df['exam_score']
    
    models, metrics, X_test, y_test = train_models(X, y)
    st.session_state.models = models
    st.session_state.metrics = metrics
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test
    st.session_state.feature_cols = feature_cols
    st.session_state.models_trained = True

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Single Student Prediction",
    "📊 Batch Analysis",
    "📁 Upload Data",
    "📈 Model Insights",
    "🔍 Dataset Explorer"
])

with tab1:
    st.subheader("Predict Individual Student Performance")
    st.markdown("Enter student information to predict their exam score and get personalized recommendations.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Student Information")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            attendance = st.slider("📅 Attendance Rate (%)", 0, 100, 85, 
                                  help="Percentage of classes attended",
                                  key="attendance_rate")
            study_hours = st.slider("📚 Study Hours/Week", 0, 40, 12,
                                   help="Hours spent studying per week",
                                   key="study_hours_per_week")
            previous_grade = st.slider("📝 Previous Grade", 0, 100, 75,
                                      help="Score from previous exam",
                                      key="previous_grade")
            internet_hours = st.slider("💻 Internet Usage (hrs/day)", 0, 12, 3,
                                      help="Daily internet usage hours",
                                      key="internet_usage_hours")
            sleep_hours = st.slider("😴 Sleep Hours/Night", 4, 11, 7,
                                   help="Average sleep hours per night",
                                   key="sleep_hours")
        
        with col_b:
            extra_activities = st.number_input("⚽ Extracurricular Activities", 0, 5, 2,
                                              help="Number of activities",
                                              key="extracurricular_activities")
            parental_involvement = st.slider("👨‍👩‍👧 Parental Involvement", 1, 5, 4,
                                            help="Parent engagement level (1=low, 5=high)",
                                            key="parental_involvement")
            tutoring = st.number_input("📖 Tutoring Sessions/Month", 0, 20, 2,
                                      help="Extra tutoring sessions per month",
                                      key="tutoring_sessions")
            assignment_rate = st.slider("✅ Assignment Completion (%)", 0, 100, 85,
                                       help="Percentage of assignments completed",
                                       key="assignment_completion_rate")
    
    with col2:
        st.markdown("### Model Selection")
        model_choice = st.selectbox(
            "Choose ML Model:",
            ["Random Forest", "Gradient Boosting", "Linear Regression"]
        )
        
        st.markdown("### Quick Actions")
        if st.button("📋 Load Sample Student", use_container_width=True):
            sample = get_sample_student()
            st.session_state.update(sample)
            st.rerun()
        
        if st.button("🔄 Reset All", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['dataset', 'models', 'metrics', 'X_test', 'y_test', 'feature_cols', 'models_trained']:
                    del st.session_state[key]
            st.rerun()
    
    features = [
        attendance, study_hours, previous_grade, internet_hours,
        sleep_hours, extra_activities, parental_involvement,
        tutoring, assignment_rate
    ]
    
    features_dict = {
        'attendance_rate': attendance,
        'study_hours_per_week': study_hours,
        'previous_grade': previous_grade,
        'internet_usage_hours': internet_hours,
        'sleep_hours': sleep_hours,
        'extracurricular_activities': extra_activities,
        'parental_involvement': parental_involvement,
        'tutoring_sessions': tutoring,
        'assignment_completion_rate': assignment_rate
    }
    
    st.markdown("---")
    
    if st.button("🎯 Predict Performance", type="primary", use_container_width=True):
        model = st.session_state.models[model_choice]
        predicted_score = predict_performance(model, features)
        risk_level, emoji, description = classify_risk_level(predicted_score)
        
        st.markdown("### 📊 Prediction Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Predicted Score", f"{predicted_score:.1f}/100")
        with col2:
            st.metric("Risk Level", risk_level)
        with col3:
            st.metric("Status", emoji)
        with col4:
            grade_letter = 'A' if predicted_score >= 90 else 'B' if predicted_score >= 80 else 'C' if predicted_score >= 70 else 'D' if predicted_score >= 60 else 'F'
            st.metric("Letter Grade", grade_letter)
        
        st.markdown(f"**Assessment:** {description}")
        
        fig, ax = plt.subplots(figsize=(10, 2))
        colors = ['#ff4444', '#ff8844', '#ffcc44', '#44ff44']
        zones = [50, 65, 80, 100]
        zone_labels = ['High Risk', 'Med-High', 'Med-Low', 'Low Risk']
        
        for i, (zone, color, label) in enumerate(zip(zones, colors, zone_labels)):
            start = 0 if i == 0 else zones[i-1]
            ax.barh(0, zone - start, left=start, color=color, alpha=0.3, height=0.5)
            ax.text((start + zone) / 2, 0, label, ha='center', va='center', fontsize=9, weight='bold')
        
        ax.plot([predicted_score, predicted_score], [-0.3, 0.3], 'k-', linewidth=3)
        ax.plot(predicted_score, 0, 'ko', markersize=15)
        ax.text(predicted_score, 0.5, f'{predicted_score:.1f}', ha='center', fontsize=12, weight='bold')
        
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.8)
        ax.axis('off')
        ax.set_title('Score Range Visualization', fontsize=12, weight='bold', pad=20)
        
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        st.markdown("### 💡 Personalized Recommendations")
        
        recommendations = generate_recommendations(features_dict)
        for rec in recommendations:
            st.markdown(f"- {rec}")

with tab2:
    st.subheader("📊 Batch Student Analysis")
    st.markdown("Analyze multiple students at once by entering their data manually or uploading a CSV file.")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        batch_model = st.selectbox(
            "Model:",
            ["Random Forest", "Gradient Boosting", "Linear Regression"],
            key="batch_model"
        )
        
        num_students = st.number_input("Number of students:", 1, 20, 5, key="num_students")
        
        if st.button("Generate Sample Data", use_container_width=True):
            st.session_state.batch_df = generate_student_data(num_students)
    
    with col1:
        st.markdown("### Batch Input")
        
        if 'batch_df' not in st.session_state:
            st.session_state.batch_df = pd.DataFrame(columns=['student_id'] + st.session_state.feature_cols)
        
        edited_df = st.data_editor(
            st.session_state.batch_df[['student_id'] + st.session_state.feature_cols],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True
        )
    
    if st.button("🎯 Analyze Batch", type="primary"):
        if len(edited_df) > 0:
            model = st.session_state.models[batch_model]
            predictions = []
            
            for idx, row in edited_df.iterrows():
                features = row[st.session_state.feature_cols].values
                pred_score = predict_performance(model, features)
                risk, emoji, _ = classify_risk_level(pred_score)
                
                predictions.append({
                    'Student ID': row['student_id'],
                    'Predicted Score': f"{pred_score:.1f}",
                    'Risk Level': risk,
                    'Status': emoji
                })
            
            results_df = pd.DataFrame(predictions)
            
            st.success(f"✅ Analyzed {len(results_df)} students!")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Students", len(results_df))
            with col2:
                high_risk = results_df['Risk Level'].str.contains('High Risk').sum()
                st.metric("High Risk", high_risk)
            with col3:
                low_risk = results_df['Risk Level'].str.contains('Low Risk').sum()
                st.metric("Low Risk", low_risk)
            with col4:
                avg_score = pd.to_numeric(results_df['Predicted Score']).mean()
                st.metric("Avg Score", f"{avg_score:.1f}")
            
            st.markdown("---")
            st.markdown("### Results Table")
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                csv,
                file_name="student_predictions.csv",
                mime="text/csv"
            )
            
            st.session_state.batch_results = results_df
        else:
            st.warning("Please add student data to analyze")

with tab3:
    st.subheader("📁 Upload Student Data")
    st.markdown("Upload a CSV file containing student data for bulk predictions.")
    
    st.markdown("""
    **Required CSV columns:**
    - `student_id` (optional)
    - `attendance_rate`
    - `study_hours_per_week`
    - `previous_grade`
    - `internet_usage_hours`
    - `sleep_hours`
    - `extracurricular_activities`
    - `parental_involvement`
    - `tutoring_sessions`
    - `assignment_completion_rate`
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    with col2:
        upload_model = st.selectbox(
            "Model:",
            ["Random Forest", "Gradient Boosting", "Linear Regression"],
            key="upload_model"
        )
    
    if uploaded_file:
        try:
            upload_df = pd.read_csv(uploaded_file)
            
            st.info(f"📄 Loaded {len(upload_df)} students from file")
            
            st.markdown("### Preview")
            st.dataframe(upload_df.head(10), use_container_width=True)
            
            if st.button("🎯 Predict All", type="primary"):
                model = st.session_state.models[upload_model]
                
                if not all(col in upload_df.columns for col in st.session_state.feature_cols):
                    st.error("❌ Missing required columns in CSV file")
                else:
                    predictions = []
                    
                    for idx, row in upload_df.iterrows():
                        student_id = row.get('student_id', f'Student_{idx+1}')
                        features = row[st.session_state.feature_cols].values
                        pred_score = predict_performance(model, features)
                        risk, emoji, desc = classify_risk_level(pred_score)
                        
                        predictions.append({
                            'Student ID': student_id,
                            'Predicted Score': pred_score,
                            'Risk Level': risk,
                            'Status': emoji,
                            'Recommendation': desc
                        })
                    
                    pred_df = pd.DataFrame(predictions)
                    
                    st.success(f"✅ Predictions complete for {len(pred_df)} students!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total", len(pred_df))
                    with col2:
                        high_risk = (pred_df['Predicted Score'] < 50).sum()
                        st.metric("High Risk", high_risk)
                    with col3:
                        avg_score = pred_df['Predicted Score'].mean()
                        st.metric("Avg Score", f"{avg_score:.1f}")
                    with col4:
                        at_risk = (pred_df['Predicted Score'] < 65).sum()
                        st.metric("Needs Support", at_risk)
                    
                    st.markdown("---")
                    st.dataframe(pred_df, use_container_width=True, hide_index=True)
                    
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Predictions (CSV)",
                        csv,
                        file_name="predictions_output.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

with tab4:
    st.subheader("📈 Model Performance & Insights")
    
    st.markdown("### Model Comparison")
    
    metrics_df = pd.DataFrame(st.session_state.metrics).T
    metrics_df.columns = ['RMSE', 'MAE', 'R² Score', 'CV Score']
    metrics_df = metrics_df.round(3)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Performance Metrics")
        st.dataframe(metrics_df, use_container_width=True)
        
        best_model = metrics_df['R² Score'].idxmax()
        st.success(f"🏆 Best Model: **{best_model}** (R² = {metrics_df.loc[best_model, 'R² Score']:.3f})")
    
    with col2:
        st.markdown("#### Metrics Visualization")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        x = np.arange(len(metrics_df))
        width = 0.25
        
        ax.bar(x - width, metrics_df['R² Score'], width, label='R² Score', color='skyblue')
        ax.bar(x, metrics_df['CV Score'], width, label='CV Score', color='lightcoral')
        ax.bar(x + width, 1 - (metrics_df['RMSE']/100), width, label='1 - RMSE/100', color='lightgreen')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df.index, rotation=15)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    st.markdown("---")
    st.markdown("### Feature Importance Analysis")
    
    importance_model = st.selectbox(
        "Select model for feature importance:",
        ["Random Forest", "Gradient Boosting"]
    )
    
    model = st.session_state.models[importance_model]
    importance_df = get_feature_importance(model, st.session_state.feature_cols)
    
    if importance_df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = sns.color_palette("viridis", len(importance_df))
            bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
            ax.set_xlabel('Importance Score')
            ax.set_title(f'{importance_model} - Feature Importance')
            ax.grid(alpha=0.3, axis='x')
            
            for bar, imp in zip(bars, importance_df['Importance']):
                width = bar.get_width()
                ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                       f'{imp:.3f}', ha='left', va='center', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Feature Importance Table")
            display_imp = importance_df.copy()
            display_imp['Importance'] = display_imp['Importance'].apply(lambda x: f"{x:.4f}")
            st.dataframe(display_imp, use_container_width=True, hide_index=True)
            
            st.markdown("#### Key Insights")
            top_feature = importance_df.iloc[0]['Feature']
            top_importance = importance_df.iloc[0]['Importance']
            
            st.info(f"""
            **Most Important Feature:** {top_feature}
            
            **Impact:** {top_importance:.1%} of prediction power
            
            This indicates that {top_feature.replace('_', ' ')} is the strongest predictor of student performance.
            """)
    
    st.markdown("---")
    st.markdown("### Prediction Distribution")
    
    y_test = st.session_state.y_test
    model_for_dist = st.session_state.models["Random Forest"]
    y_pred = model_for_dist.predict(st.session_state.X_test)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
    ax1.plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
    ax1.set_xlabel('Actual Scores')
    ax1.set_ylabel('Predicted Scores')
    ax1.set_title('Actual vs Predicted Scores')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    residuals = y_test - y_pred
    ax2.hist(residuals, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Residuals (Actual - Predicted)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Prediction Error Distribution')
    ax2.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab5:
    st.subheader("🔍 Dataset Explorer")
    st.markdown("Explore the training dataset and understand patterns in student performance.")
    
    df = st.session_state.dataset
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        avg_score = df['exam_score'].mean()
        st.metric("Avg Exam Score", f"{avg_score:.1f}")
    with col3:
        high_performers = (df['exam_score'] >= 80).sum()
        st.metric("High Performers (≥80)", high_performers)
    with col4:
        at_risk = (df['exam_score'] < 50).sum()
        st.metric("At Risk (<50)", at_risk)
    
    st.markdown("---")
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Score Distribution")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.hist(df['exam_score'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(df['exam_score'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['exam_score'].mean():.1f}")
    ax1.axvline(df['exam_score'].median(), color='green', linestyle='--', linewidth=2, label=f"Median: {df['exam_score'].median():.1f}")
    ax1.set_xlabel('Exam Score')
    ax1.set_ylabel('Number of Students')
    ax1.set_title('Exam Score Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    
    score_ranges = pd.cut(df['exam_score'], bins=[0, 50, 65, 80, 100], labels=['<50 (At Risk)', '50-65 (Medium)', '65-80 (Good)', '80+ (Excellent)'])
    range_counts = score_ranges.value_counts().sort_index()
    
    colors_pie = ['#ff4444', '#ff8844', '#ffcc44', '#44ff44']
    ax2.pie(range_counts.values, labels=range_counts.index, autopct='%1.1f%%', startangle=90, colors=colors_pie)
    ax2.set_title('Student Performance Categories')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    st.markdown("### Feature Correlations")
    
    corr_features = st.session_state.feature_cols + ['exam_score']
    corr_matrix = df[corr_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=14, weight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.markdown("---")
    st.markdown("### Statistical Summary")
    st.dataframe(df[st.session_state.feature_cols + ['exam_score']].describe().T, use_container_width=True)

st.markdown("---")
st.markdown("")