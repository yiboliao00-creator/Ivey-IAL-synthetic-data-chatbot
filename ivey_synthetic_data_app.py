import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import io
from scipy import stats
import os
from typing import List, Dict, Tuple
import json
import requests

# Page configuration with Ivey Business School theme
st.set_page_config(
    page_title="Ivey Synthetic Data Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Ivey Business School green theme
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #00693e;
        border-radius: 10px;
        padding: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        color: white;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        color: #00693e !important;
        border-radius: 5px;
    }
    h1, h2, h3 {
        color: #00693e;
    }
    .stButton > button {
        background-color: #00693e;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #004d2c;
        transform: translateY(-2px);
    }
    .info-box {
        background-color: #e8f5f0;
        border-left: 4px solid #00693e;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .risk-low {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .risk-medium {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
    .risk-high {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .ai-insight-box {
        background: linear-gradient(135deg, #00693e10 0%, #00693e20 100%);
        border: 2px solid #00693e;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": """You are an expert AI assistant specializing in synthetic data generation, 
        data science, and privacy-preserving technologies. You work for Ivey Business School and help students, 
        researchers, and professionals understand synthetic data concepts, applications, and best practices. 
        You provide detailed, educational responses about:
        - Synthetic data generation techniques (GANs, VAEs, statistical methods)
        - Privacy preservation (differential privacy, k-anonymity, l-diversity)
        - Use cases in healthcare, finance, and business
        - Statistical validation methods
        - Implementation strategies
        - Regulatory compliance (GDPR, HIPAA)
        - Machine learning applications
        Always be helpful, thorough, and educational in your responses."""}
    ]
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'original_stats' not in st.session_state:
    st.session_state.original_stats = None
if 'generation_method' not in st.session_state:
    st.session_state.generation_method = None
if 'privacy_level' not in st.session_state:
    st.session_state.privacy_level = None
if 'pending_question' not in st.session_state:
    st.session_state.pending_question = None
if 'data_analysis_report' not in st.session_state:
    st.session_state.data_analysis_report = None


# Enhanced Analysis Functions
def generate_ai_data_analysis(df: pd.DataFrame) -> Dict:
    """Generate comprehensive AI-powered data analysis and recommendations"""
    
    analysis = {
        'overview': {},
        'quality_assessment': {},
        'privacy_concerns': [],
        'recommended_methods': [],
        'anonymization_needed': False,
        'key_insights': [],
        'warnings': []
    }
    
    # Data Overview
    analysis['overview'] = {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object']).columns),
        'missing_data_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    }
    
    # Quality Assessment
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 0:
        # Check for outliers
        outlier_cols = []
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(df) * 0.05:  # More than 5% outliers
                outlier_cols.append(col)
        
        # Check for skewness
        skewed_cols = []
        for col in numeric_cols:
            skew = df[col].skew()
            if abs(skew) > 1:
                skewed_cols.append((col, skew))
        
        # Check correlations
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.8:
                        high_corr_pairs.append(
                            (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                        )
            
            analysis['quality_assessment']['high_correlations'] = high_corr_pairs
        
        analysis['quality_assessment']['outlier_columns'] = outlier_cols
        analysis['quality_assessment']['skewed_columns'] = skewed_cols
    
    # Privacy Concerns Analysis
    for col in df.columns:
        col_lower = col.lower()
        
        # Check for potential PII
        pii_keywords = ['id', 'name', 'email', 'phone', 'ssn', 'social', 'address', 
                       'birth', 'dob', 'license', 'passport', 'account']
        
        if any(keyword in col_lower for keyword in pii_keywords):
            analysis['privacy_concerns'].append({
                'column': col,
                'type': 'Potential PII',
                'risk_level': 'High'
            })
            analysis['anonymization_needed'] = True
        
        # Check for quasi-identifiers
        quasi_keywords = ['age', 'gender', 'zip', 'postal', 'city', 'state', 'country']
        if any(keyword in col_lower for keyword in quasi_keywords):
            analysis['privacy_concerns'].append({
                'column': col,
                'type': 'Quasi-identifier',
                'risk_level': 'Medium'
            })
    
    # Check uniqueness
    for col in df.columns:
        if df[col].nunique() / len(df) > 0.95:  # More than 95% unique values
            analysis['privacy_concerns'].append({
                'column': col,
                'type': 'High uniqueness',
                'risk_level': 'Medium'
            })
    
    # Method Recommendations
    if analysis['overview']['missing_data_pct'] > 10:
        analysis['recommended_methods'].append({
            'method': 'Statistical Sampling',
            'reason': 'High missing data percentage - statistical methods handle this well'
        })
    
    if len(outlier_cols) > len(numeric_cols) * 0.3:
        analysis['recommended_methods'].append({
            'method': 'Gaussian Copula',
            'reason': 'Multiple columns with outliers - copula preserves marginal distributions'
        })
    
    if len(high_corr_pairs) > 0:
        analysis['recommended_methods'].append({
            'method': 'Gaussian Copula',
            'reason': 'High correlations detected - copula methods preserve correlation structure'
        })
    
    if not analysis['recommended_methods']:
        analysis['recommended_methods'].append({
            'method': 'Statistical Sampling',
            'reason': 'Standard data characteristics - statistical sampling is efficient and effective'
        })
    
    # Generate Key Insights
    if analysis['overview']['missing_data_pct'] > 5:
        analysis['key_insights'].append(
            f"📊 Your data has {analysis['overview']['missing_data_pct']:.1f}% missing values. "
            "Consider imputation before generation."
        )
    
    if outlier_cols:
        analysis['key_insights'].append(
            f"🔍 Found outliers in {len(outlier_cols)} columns. "
            "These may represent important edge cases or data quality issues."
        )
    
    if analysis['anonymization_needed']:
        analysis['key_insights'].append(
            "🔒 Detected potential PII or sensitive data. "
            "Strong anonymization techniques are recommended."
        )
    
    if high_corr_pairs:
        analysis['key_insights'].append(
            f"🔗 Found {len(high_corr_pairs)} highly correlated column pairs. "
            "Consider feature selection or ensure correlation preservation."
        )
    
    # Warnings
    if analysis['overview']['rows'] < 100:
        analysis['warnings'].append(
            "⚠️ Small dataset (<100 rows) may limit synthetic data quality"
        )
    
    if analysis['overview']['missing_data_pct'] > 30:
        analysis['warnings'].append(
            "⚠️ High missing data percentage may affect generation quality"
        )
    
    return analysis


def generate_healthcare_simulation_insights(data: pd.DataFrame, scenario: str) -> Dict:
    """Generate AI-powered insights for healthcare simulation data"""
    
    insights = {
        'operational_metrics': {},
        'bottlenecks': [],
        'optimization_opportunities': [],
        'staff_recommendations': [],
        'cost_analysis': {},
        'predictive_insights': []
    }
    
    # Calculate operational metrics
    if 'WaitTime' in data.columns:
        insights['operational_metrics']['avg_wait_time'] = data['WaitTime'].mean()
        insights['operational_metrics']['p95_wait_time'] = data['WaitTime'].quantile(0.95)
        
        if insights['operational_metrics']['avg_wait_time'] > 60:
            insights['bottlenecks'].append({
                'area': 'Emergency Department',
                'issue': 'High average wait times',
                'impact': 'Patient satisfaction and health outcomes',
                'severity': 'High'
            })
    
    if 'LengthOfStay' in data.columns:
        insights['operational_metrics']['avg_los'] = data['LengthOfStay'].mean()
        
        # Identify departments with longest LOS
        if 'Department' in data.columns:
            dept_los = data.groupby('Department')['LengthOfStay'].mean().sort_values(ascending=False)
            if len(dept_los) > 0:
                worst_dept = dept_los.index[0]
                insights['bottlenecks'].append({
                    'area': f'{worst_dept} Department',
                    'issue': f'Highest average LOS ({dept_los.iloc[0]:.1f} days)',
                    'impact': 'Bed availability and costs',
                    'severity': 'Medium'
                })
    
    # Staff recommendations based on patient flow
    if 'AdmissionDate' in data.columns:
        # Analyze hourly patterns
        data['Hour'] = pd.to_datetime(data['AdmissionDate']).dt.hour
        peak_hours = data.groupby('Hour').size().sort_values(ascending=False).head(3)
        
        for hour in peak_hours.index:
            insights['staff_recommendations'].append({
                'time': f'{hour:02d}:00-{(hour+1)%24:02d}:00',
                'action': 'Increase staff by 20%',
                'reason': f'Peak admission time ({peak_hours[hour]} admissions)'
            })
    
    # Cost analysis
    if 'TotalCost' in data.columns:
        insights['cost_analysis']['total_cost'] = data['TotalCost'].sum()
        insights['cost_analysis']['avg_cost_per_patient'] = data['TotalCost'].mean()
        
        # Cost reduction opportunities
        if 'LengthOfStay' in data.columns:
            potential_savings = data['TotalCost'].sum() * 0.15  # 15% reduction potential
            insights['optimization_opportunities'].append({
                'opportunity': 'Reduce Length of Stay by 10%',
                'potential_savings': f'${potential_savings:,.2f}',
                'implementation': 'Improve discharge planning and care coordination'
            })
    
    # Predictive insights based on patterns
    if 'Readmission30Day' in data.columns:
        readmit_rate = data['Readmission30Day'].mean() * 100
        if readmit_rate > 15:
            insights['predictive_insights'].append({
                'metric': 'High Readmission Risk',
                'value': f'{readmit_rate:.1f}%',
                'recommendation': 'Implement post-discharge follow-up program'
            })
    
    # Scenario-specific insights
    if scenario == "Emergency Department Flow":
        insights['optimization_opportunities'].append({
            'opportunity': 'Implement Fast-Track System',
            'potential_impact': 'Reduce low-acuity wait times by 40%',
            'implementation': 'Dedicate staff and space for Triage 4-5 patients'
        })
    elif scenario == "Bed Occupancy Optimization":
        insights['optimization_opportunities'].append({
            'opportunity': 'Dynamic Bed Management System',
            'potential_impact': 'Increase bed utilization by 10-15%',
            'implementation': 'Real-time tracking and predictive bed assignment'
        })
    
    return insights


# Privacy Risk Assessment Functions
def assess_privacy_risk(df, synthetic_df):
    """Assess privacy risks in synthetic data"""
    
    risks = {
        'overall_risk': 'Low',
        'membership_inference_risk': 0.0,
        'attribute_inference_risk': 0.0,
        'linkability_risk': 0.0,
        'uniqueness_risk': 0.0,
        'recommendations': []
    }
    
    # Simple membership inference risk (based on exact matches)
    exact_matches = 0
    for idx, row in synthetic_df.iterrows():
        if any((df == row).all(axis=1)):
            exact_matches += 1
    
    membership_risk = (exact_matches / len(synthetic_df)) * 100
    risks['membership_inference_risk'] = membership_risk
    
    if membership_risk > 5:
        risks['overall_risk'] = 'High'
        risks['recommendations'].append("🚨 High exact match rate - reduce noise or change generation method")
    elif membership_risk > 1:
        risks['overall_risk'] = 'Medium'
        risks['recommendations'].append("⚠️ Consider adding more noise to synthetic data")
    
    # Attribute inference risk (correlation preservation)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        orig_corr = df[numeric_cols].corr()
        synth_corr = synthetic_df[numeric_cols].corr()
        
        correlation_similarity = np.corrcoef(orig_corr.values.flatten(),
                                             synth_corr.values.flatten())[0, 1]
        
        attribute_risk = max(0, correlation_similarity * 100)  # Scale to percentage
        risks['attribute_inference_risk'] = attribute_risk
        
        if attribute_risk > 85:
            risks['recommendations'].append("🔍 Very high correlation preservation may enable attribute inference")
            if risks['overall_risk'] == 'Low':
                risks['overall_risk'] = 'Medium'
    
    # Uniqueness risk
    unique_combinations = len(df.drop_duplicates()) / len(df) * 100
    risks['uniqueness_risk'] = unique_combinations
    
    if unique_combinations > 90:
        risks['linkability_risk'] = 75
        risks['recommendations'].append("🎯 High uniqueness in original data increases linkability risk")
        if risks['overall_risk'] != 'High':
            risks['overall_risk'] = 'Medium'
    elif unique_combinations > 70:
        risks['linkability_risk'] = 40
        risks['recommendations'].append("📊 Moderate uniqueness detected - monitor for linking attacks")
    else:
        risks['linkability_risk'] = 20
    
    # Overall risk assessment
    avg_risk = (membership_risk + attribute_risk + risks['linkability_risk']) / 3
    if avg_risk > 60 and risks['overall_risk'] != 'High':
        risks['overall_risk'] = 'High'
    elif avg_risk > 30 and risks['overall_risk'] == 'Low':
        risks['overall_risk'] = 'Medium'
    
    return risks


def generate_data_quality_report(original_df, synthetic_df, generation_method, privacy_level):
    """Generate comprehensive data quality report"""
    
    report = {
        'metadata': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_shape': original_df.shape,
            'synthetic_shape': synthetic_df.shape,
            'generation_method': generation_method,
            'privacy_level': privacy_level,
        },
        'statistical_comparison': {},
        'distribution_tests': {},
        'privacy_assessment': {},
        'recommendations': []
    }
    
    # Statistical comparisons
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(original_df[col].dropna(), synthetic_df[col].dropna())
            
            mean_diff = abs(original_df[col].mean() - synthetic_df[col].mean())
            mean_diff_pct = (mean_diff / original_df[col].mean() * 100) if original_df[col].mean() != 0 else 0
            
            report['statistical_comparison'][col] = {
                'original_mean': float(original_df[col].mean()),
                'synthetic_mean': float(synthetic_df[col].mean()),
                'original_std': float(original_df[col].std()),
                'synthetic_std': float(synthetic_df[col].std()),
                'mean_difference_pct': float(mean_diff_pct),
                'std_difference_pct': float(
                    abs(original_df[col].std() - synthetic_df[col].std()) / original_df[col].std() * 100) if
                original_df[col].std() != 0 else 0
            }
            
            report['distribution_tests'][col] = {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pvalue),
                'distribution_similarity': 'Excellent' if ks_pvalue > 0.05 else 'Needs Review',
                'similarity_score': float(1 - ks_stat)  # Higher is better
            }
            
        except Exception as e:
            report['distribution_tests'][col] = {'error': str(e)}
    
    # Privacy assessment
    privacy_risks = assess_privacy_risk(original_df, synthetic_df)
    report['privacy_assessment'] = privacy_risks
    
    # Generate recommendations
    if len([col for col in report['distribution_tests'] if
            report['distribution_tests'][col].get('ks_pvalue', 0) > 0.05]) < len(numeric_cols) * 0.7:
        report['recommendations'].append(
            "Consider using a different generation method for better distribution matching")
    
    if privacy_risks['overall_risk'] == 'High':
        report['recommendations'].append("Increase privacy level or add more noise to reduce privacy risks")
    
    avg_similarity = np.mean([report['distribution_tests'][col].get('similarity_score', 0) for col in numeric_cols])
    if avg_similarity < 0.7:
        report['recommendations'].append("Low similarity detected - consider tuning generation parameters")
    
    report['overall_quality_score'] = float(avg_similarity * 100)
    
    return report


def apply_anonymization_techniques(df, technique, quasi_identifiers, params=None):
    """Apply various anonymization techniques"""
    
    anonymized_df = df.copy()
    
    if technique == "k-Anonymity":
        k_value = params.get('k_value', 5)
        
        for col in quasi_identifiers:
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    # Generalize numeric values into ranges
                    try:
                        bins = pd.qcut(df[col], q=min(k_value, len(df[col].unique())), duplicates='drop')
                        anonymized_df[col] = bins.astype(str)
                    except:
                        # Fallback to equal-width bins if qcut fails
                        bins = pd.cut(df[col], bins=min(k_value, len(df[col].unique())))
                        anonymized_df[col] = bins.astype(str)
                else:
                    # Generalize categorical values
                    value_counts = df[col].value_counts()
                    rare_values = value_counts[value_counts < k_value].index
                    anonymized_df[col] = df[col].replace(rare_values, 'Other')
    
    elif technique == "l-Diversity":
        l_value = params.get('l_value', 3)
        sensitive_attr = params.get('sensitive_attr')
        
        if sensitive_attr and sensitive_attr in df.columns:
            # Group by quasi-identifiers and ensure diversity in sensitive attribute
            for col in quasi_identifiers:
                if col in df.columns and col != sensitive_attr:
                    groups = anonymized_df.groupby(col)[sensitive_attr].nunique()
                    insufficient_groups = groups[groups < l_value].index
                    
                    # Suppress records with insufficient diversity
                    mask = anonymized_df[col].isin(insufficient_groups)
                    anonymized_df.loc[mask, col] = 'Suppressed'
    
    elif technique == "Data Masking":
        for col in quasi_identifiers:
            if col in df.columns:
                if 'name' in col.lower():
                    # Mask names
                    anonymized_df[col] = anonymized_df[col].astype(str).apply(
                        lambda x: x[:2] + '*' * (len(str(x)) - 2) if len(str(x)) > 2 else '***'
                    )
                elif 'id' in col.lower():
                    # Mask IDs
                    anonymized_df[col] = anonymized_df[col].astype(str).apply(
                        lambda x: x[:3] + '*' * max(0, len(str(x)) - 3)
                    )
                elif 'email' in col.lower():
                    # Mask emails
                    anonymized_df[col] = anonymized_df[col].astype(str).str.replace(
                        r'@.*', '@***.com', regex=True
                    )
                elif 'phone' in col.lower():
                    # Mask phone numbers
                    anonymized_df[col] = anonymized_df[col].astype(str).apply(
                        lambda x: x[:3] + '-***-****' if len(str(x)) >= 3 else '***-***-****'
                    )
    
    elif technique == "Noise Addition":
        noise_level = params.get('noise_level', 0.1)
        
        for col in quasi_identifiers:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                std_dev = df[col].std()
                noise = np.random.normal(0, std_dev * noise_level, len(df))
                anonymized_df[col] = df[col] + noise
    
    return anonymized_df


# Header with Ivey branding
st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #00693e; font-size: 2.5rem; margin-bottom: 0.5rem;'>
            🎓 Ivey Business School
        </h1>
        <h2 style='color: #00693e; font-size: 1.8rem; font-weight: 400;'>
            Advanced Synthetic Data Generation Platform
        </h2>
        <p style='color: #666; font-size: 1rem;'>
            Educational Tool for Data Science & Privacy-Preserving Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["📚 AI Education Assistant", "🔧 Advanced Data Generator", "🥼 Healthcare Simulator"])



            
           
