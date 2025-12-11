import numpy as np
import pandas as pd

def generate_student_data(n_students=500, random_state=42):
    """Generate realistic synthetic student performance data"""
    np.random.seed(random_state)
    
    data = {
        'student_id': [f'STU{str(i).zfill(4)}' for i in range(1, n_students + 1)],
        'attendance_rate': np.random.beta(8, 2, n_students) * 100,
        'study_hours_per_week': np.random.gamma(3, 3, n_students),
        'previous_grade': np.random.normal(70, 15, n_students),
        'internet_usage_hours': np.random.gamma(2, 2, n_students),
        'sleep_hours': np.random.normal(7, 1.5, n_students),
        'extracurricular_activities': np.random.poisson(1.5, n_students),
        'parental_involvement': np.random.randint(1, 6, n_students),
        'tutoring_sessions': np.random.poisson(2, n_students),
        'assignment_completion_rate': np.random.beta(7, 2, n_students) * 100,
    }
    
    df = pd.DataFrame(data)
    
    df['attendance_rate'] = df['attendance_rate'].clip(40, 100)
    df['study_hours_per_week'] = df['study_hours_per_week'].clip(0, 40)
    df['previous_grade'] = df['previous_grade'].clip(30, 100)
    df['internet_usage_hours'] = df['internet_usage_hours'].clip(0, 12)
    df['sleep_hours'] = df['sleep_hours'].clip(4, 11)
    df['extracurricular_activities'] = df['extracurricular_activities'].clip(0, 5)
    
    base_score = (
        df['attendance_rate'] * 0.25 +
        df['previous_grade'] * 0.30 +
        df['study_hours_per_week'] * 1.2 +
        df['assignment_completion_rate'] * 0.15 +
        df['parental_involvement'] * 2.5 +
        df['tutoring_sessions'] * 1.5
    )
    
    penalties = (
        df['internet_usage_hours'] * 0.8 +
        (10 - df['sleep_hours']).clip(0, None) * 1.5
    )
    
    df['exam_score'] = (base_score - penalties).clip(0, 100)
    
    noise = np.random.normal(0, 3, n_students)
    df['exam_score'] = (df['exam_score'] + noise).clip(0, 100)
    
    df = df.round(2)
    df['exam_score'] = df['exam_score'].round(1)
    
    return df

def get_sample_student():
    """Get a sample student with average characteristics"""
    return {
        'attendance_rate': 85.0,
        'study_hours_per_week': 12.0,
        'previous_grade': 75.0,
        'internet_usage_hours': 3.0,
        'sleep_hours': 7.0,
        'extracurricular_activities': 2,
        'parental_involvement': 4,
        'tutoring_sessions': 2,
        'assignment_completion_rate': 85.0
    }

def get_feature_descriptions():
    """Get descriptions for all features"""
    return {
        'attendance_rate': 'Percentage of classes attended (0-100%)',
        'study_hours_per_week': 'Hours spent studying per week (0-40)',
        'previous_grade': 'Grade from previous exam (0-100)',
        'internet_usage_hours': 'Daily internet usage hours (0-12)',
        'sleep_hours': 'Average sleep hours per night (4-11)',
        'extracurricular_activities': 'Number of activities (0-5)',
        'parental_involvement': 'Parent engagement level (1-5 scale)',
        'tutoring_sessions': 'Extra tutoring sessions per month (0+)',
        'assignment_completion_rate': 'Percentage of assignments completed (0-100%)'
    }

def get_feature_ranges():
    """Get valid ranges for each feature"""
    return {
        'attendance_rate': (0, 100),
        'study_hours_per_week': (0, 40),
        'previous_grade': (0, 100),
        'internet_usage_hours': (0, 12),
        'sleep_hours': (4, 11),
        'extracurricular_activities': (0, 5),
        'parental_involvement': (1, 5),
        'tutoring_sessions': (0, 20),
        'assignment_completion_rate': (0, 100)
    }
