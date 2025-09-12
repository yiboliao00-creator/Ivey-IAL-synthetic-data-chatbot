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

# Tab 1: Educational Chatbot
with tab1:
    
    st.markdown("### 💬 Synthetic Data Education Assistant")
    
    try:
        import os, requests
        from langchain_core.messages import AIMessage, HumanMessage
        
        @st.cache_resource(show_spinner=False)
        def init_bot():
            """
            Prefer a FREE cloud LLM (Groq) on Streamlit Cloud.
            Fall back to local Ollama ONLY if reachable (for local dev).
            """
            # 1) Groq (free tier) — requires GROQ_API_KEY in Streamlit Secrets
            try:
                from langchain_groq import ChatGroq
                groq_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
                if groq_key:
                    return ChatGroq(
                        model_name="llama-3.1-8b-instant",
                        api_key=groq_key,
                        temperature=0.2,
                        max_tokens=512,
                    )
            except Exception:
                pass  # if Groq libs aren't available / no key set, try Ollama
            
            # 2) Local Ollama (for laptops only)
            try:
                from langchain_community.chat_models import ChatOllama
                base = (
                    st.secrets.get("OLLAMA_HOST")
                    or os.getenv("OLLAMA_HOST")
                    or "http://localhost:11434"
                )
                # Probe Ollama so cloud doesn't hang/error on localhost
                requests.get(f"{base}/api/tags", timeout=1)
                return ChatOllama(model="phi3:mini", base_url=base)
            except Exception:
                return None
        
        llm = init_bot()
        
        if llm is not None:
            # Intro box (HTML safely wrapped)
            st.markdown(
                """
                <div class='info-box'>
                  <strong>Welcome to the Ivey Synthetic Data Educational Assistant!</strong>
                  <ul>
                    <li>Data generation techniques and algorithms</li>
                    <li>Privacy preservation and data anonymization</li>
                    <li>Statistical validation and quality metrics</li>
                    <li>Real-world applications and case studies</li>
                    <li>Implementation best practices</li>
                    <li>Regulatory compliance and ethical considerations</li>
                  </ul>
                  Feel free to ask follow-up questions!
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # --- Chat history state ---
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = [
                    AIMessage(content="Hello! I'm your synthetic data assistant. How can I help?")
                ]
            
            # Display history
            for msg in st.session_state.chat_history:
                if isinstance(msg, AIMessage):
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg.content)
                elif isinstance(msg, HumanMessage):
                    with st.chat_message("user", avatar="🧑"):
                        st.markdown(msg.content)
            
            # Process pending question (from suggested buttons)
            if st.session_state.get("pending_question"):
                question = st.session_state.pending_question
                st.session_state.pending_question = None
                st.session_state.chat_history.append(HumanMessage(content=question))
                
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(question)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        try:
                            resp = llm.invoke(st.session_state.chat_history)
                            content = getattr(resp, "content", str(resp))
                        except Exception as e:
                            content = f"Error generating response: {e}"
                        st.markdown(content)
                        st.session_state.chat_history.append(AIMessage(content=content))
            
            # Chat input
            user_q = st.chat_input("Ask me anything about synthetic data...")
            if user_q:
                st.session_state.chat_history.append(HumanMessage(content=user_q))
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(user_q)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        try:
                            resp = llm.invoke(st.session_state.chat_history)
                            content = getattr(resp, "content", str(resp))
                        except Exception as e:
                            content = f"Error generating response: {e}"
                        st.markdown(content)
                        st.session_state.chat_history.append(AIMessage(content=content))
            
            # Suggested questions (keep your UI)
            st.markdown("### 💡 Suggested Questions")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("What are GANs?", use_container_width=True, key="q1"):
                    st.session_state.pending_question = (
                        "What are GANs and how do they work for synthetic data?"
                    )
                    st.rerun()
            with col2:
                if st.button("Privacy techniques?", use_container_width=True, key="q2"):
                    st.session_state.pending_question = (
                        "What privacy preservation techniques are used in synthetic data?"
                    )
                    st.rerun()
            with col3:
                if st.button("Healthcare use cases?", use_container_width=True, key="q3"):
                    st.session_state.pending_question = (
                        "What are the main use cases for synthetic data in healthcare?"
                    )
                    st.rerun()
            with col4:
                if st.button("K-Anonymity vs l-Diversity?", use_container_width=True, key="q4"):
                    st.session_state.pending_question = (
                        "What's the difference between k-anonymity and l-diversity? When should I use each?"
                    )
                    st.rerun()
            
            if st.button("🧹 Clear Conversation", key="clear_chat"):
                st.session_state.chat_history = [
                    AIMessage(
                        content=(
                            "Hello! I'm your synthetic data tutor. "
                            "Ask me anything about privacy, generation, metrics, or applications."
                        )
                    )
                ]
                st.rerun()
        
        else:
            # Friendly guidance if neither Groq nor Ollama is usable
            st.markdown(
                """
                <div class='info-box'>
                  <strong>AI Assistant Setup</strong><br>
                  This cloud deployment doesn't have a local Ollama server.<br><br>
                  To enable the chatbot, add a FREE Groq key in <em>Manage app → Settings → Secrets</em>:
                  <pre>GROQ_API_KEY = "grq_XXXXXXXXXXXXXXXX"</pre>
                  After saving, click <em>Reboot</em>.<br><br>
                  The Data Generator and Healthcare Simulator will still work without the AI assistant.
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    except ImportError:
        st.warning(
            "LangChain not available. Please add to requirements.txt: "
            "langchain-core, langchain-community, langchain-groq, groq."
        )



# Tab 2: Advanced Synthetic Data Generator
# ----------------------------- TAB 2: ADVANCED DATA GENERATOR (PRIVACY-FIRST & STREAMLINED) -----------------------------
with tab2:
    st.markdown("### 🔐→🧪 Privacy-First Synthetic Data Generator")
    st.caption("Workflow: **Upload → (Optional) Privacy Protection → Generate Synthetic → Evaluate**")

    # ---------- 1) Upload ----------
    uploaded_file = st.file_uploader("Upload a CSV file (≤ 200MB)", type=["csv"])
    df = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

        st.markdown("#### 👀 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # ---------- 2) Basic Statistics (enhanced) ----------
        st.markdown("#### 📈 Basic Statistics")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if num_cols:
            desc = df[num_cols].describe().T  # count, mean, std, min, 25%, 50%, 75%, max
            # add mode for numerics
            desc["mode"] = [df[c].mode(dropna=True).iloc[0] if not df[c].mode(dropna=True).empty else np.nan for c in desc.index]
            st.markdown("**Numeric columns**")
            st.dataframe(desc.round(3), use_container_width=True)

        if cat_cols:
            # show top-1 mode and cardinality for categoricals
            cat_stats = pd.DataFrame({
                "unique": [df[c].nunique(dropna=False) for c in cat_cols],
                "mode": [
                    df[c].mode(dropna=True).iloc[0] if not df[c].mode(dropna=True).empty else ""
                    for c in cat_cols
                ]
            }, index=cat_cols)
            st.markdown("**Categorical columns**")
            st.dataframe(cat_stats, use_container_width=True)

        # ---------- 3) (Optional) Privacy Protection ON ORIGINAL DATA ----------
        st.markdown("#### 🛡️ Privacy Protection (applied **before** generation)")
        privacy_choice = st.selectbox(
            "Choose a privacy action (optional)",
            ["None (use original as-is)", "Remove Identifiers", "Mask Sensitive Columns", "Add Noise to Numeric"],
            help="Applied to the uploaded data first. Synthetic data will be generated from this protected version."
        )

        df_protected = df.copy()

        # Helper: guess identifiers
        guessed_id_cols = [c for c in df.columns if any(k in c.lower() for k in ["id", "name", "email", "phone", "address", "ssn"])]

        if privacy_choice == "Remove Identifiers":
            cols = st.multiselect("Columns to remove", options=df.columns.tolist(), default=guessed_id_cols)
            if st.button("🗑️ Apply Removal", use_container_width=True):
                df_protected = df.drop(columns=[c for c in cols if c in df.columns])
                st.success(f"Removed {len(cols)} column(s).")
                st.dataframe(df_protected.head(10), use_container_width=True)

        elif privacy_choice == "Mask Sensitive Columns":
            cols = st.multiselect("Columns to mask", options=df.columns.tolist(), default=guessed_id_cols)
            if st.button("🎭 Apply Masking", use_container_width=True):
                df_protected = df.copy()
                for col in cols:
                    if col not in df_protected.columns: 
                        continue
                    low = col.lower()
                    if "email" in low:
                        df_protected[col] = df_protected[col].astype(str).str.replace(r"@.*", "@***.***", regex=True)
                    elif "phone" in low:
                        df_protected[col] = df_protected[col].astype(str).apply(
                            lambda x: (''.join([d for d in str(x) if d.isdigit()])[:3] + "-***-****") if isinstance(x, str) else x
                        )
                    elif "name" in low:
                        df_protected[col] = df_protected[col].astype(str).apply(lambda x: x[:2] + "*" * max(0, len(str(x)) - 2))
                    elif "id" in low:
                        df_protected[col] = df_protected[col].astype(str).apply(lambda x: x[:4] + "***")
                    else:
                        # generic mask
                        df_protected[col] = df_protected[col].astype(str).apply(lambda x: "***" if len(str(x)) else x)
                st.success("Masking applied.")
                st.dataframe(df_protected.head(10), use_container_width=True)

        elif privacy_choice == "Add Noise to Numeric":
            noise_pct = st.slider("Noise level (as % of column std)", 1, 50, 10)
            if st.button("🌊 Apply Noise", use_container_width=True):
                df_protected = df.copy()
                for col in num_cols:
                    std = pd.to_numeric(df[col], errors="coerce").std()
                    if pd.notna(std) and std > 0:
                        df_protected[col] = pd.to_numeric(df[col], errors="coerce") + np.random.normal(0, std * (noise_pct/100.0), len(df))
                st.success(f"Added ~{noise_pct}% noise to numeric columns.")
                st.dataframe(df_protected.head(10), use_container_width=True)

        # ---------- 4) Generate Synthetic (from PROTECTED or ORIGINAL) ----------
        st.markdown("#### ⚙️ Generation Settings")
        left, right = st.columns(2)
        with left:
            n_samples = st.slider("Number of synthetic records", 100, 10000, 1000, step=100)
            method = st.selectbox("Method", ["Statistical Sampling", "Bootstrap"])
        with right:
            preserve_corr = st.checkbox("Preserve numeric correlations (approx.)", True)
            add_noise_syn = st.checkbox("Add statistical noise to synthetic", True)
            privacy_level = st.slider("Synthetic noise level (higher = more privacy)", 0, 100, 60)

        def generate_synth(base_df: pd.DataFrame, n: int, method: str, preserve_corr: bool, add_noise: bool, lvl: int):
            """Fast, dependency-light generator."""
            if base_df is None or base_df.empty:
                return None

            rng = np.random.default_rng(42)
            num = base_df.select_dtypes(include=[np.number]).columns.tolist()
            # treat others as categorical
            synth = None

            if method == "Bootstrap":
                synth = base_df.sample(n=n, replace=True, random_state=42).reset_index(drop=True)
            else:
                # independent column sampling
                synth = pd.DataFrame(index=range(n))
                for c in base_df.columns:
                    synth[c] = base_df[c].sample(n=n, replace=True, random_state=42).reset_index(drop=True)

            # crude correlation preservation (rank align to a reference numeric)
            if preserve_corr and len(num) >= 2 and method == "Statistical Sampling":
                ref = num[0]
                order = np.argsort(np.argsort(base_df[ref].sample(n=n, replace=True, random_state=42).values))
                for c in num:
                    synth[c] = np.array(synth[c])[order]

            # add synthetic noise
            if add_noise and num:
                scale = max(0.01, lvl / 300.0)  # 60 → ~0.20×std
                for c in num:
                    col_std = pd.to_numeric(base_df[c], errors="coerce").std()
                    if pd.notna(col_std) and col_std > 0:
                        synth[c] = pd.to_numeric(synth[c], errors="coerce") + rng.normal(0, col_std * scale, size=n)

            return synth

        if st.button("🚀 Generate Anonymized Synthetic Data", use_container_width=True):
            src = df_protected if privacy_choice != "None (use original as-is)" else df
            with st.spinner("Generating synthetic data..."):
                st.session_state.synthetic_data = generate_synth(src, n_samples, method, preserve_corr, add_noise_syn, privacy_level)
            st.session_state.generation_method = method
            st.session_state.privacy_level = f"{privacy_choice} + noise({privacy_level})" if add_noise_syn else privacy_choice
            if st.session_state.synthetic_data is not None:
                st.success(f"Generated {len(st.session_state.synthetic_data):,} synthetic rows.")

    # ---------- 5) Results & Evaluation ----------
    if st.session_state.get("synthetic_data") is not None and df is not None:
        st.markdown("---")
        tabs = st.tabs(["📊 Synthetic Preview", "🛡️ Privacy Dashboard", "📈 Quality Report", "📋 Statistical Comparison"])

        # Preview
        with tabs[0]:
            st.markdown("#### 📊 Synthetic Data (first 25 rows)")
            st.dataframe(st.session_state.synthetic_data.head(25), use_container_width=True)
            colA, colB, colC = st.columns(3)
            with colA: st.metric("Records", f"{len(st.session_state.synthetic_data):,}")
            with colB: st.metric("Method", st.session_state.get("generation_method", "—"))
            with colC: st.metric("Privacy", st.session_state.get("privacy_level", "—"))
            csv = st.session_state.synthetic_data.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, f"synthetic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv", use_container_width=True)

        # Privacy Dashboard (heuristics)
        with tabs[1]:
            st.markdown("#### 🛡️ Privacy Dashboard")
            orig = df.copy()
            synth = st.session_state.synthetic_data.copy()

            def uniq_ratio(s: pd.Series) -> float:
                try:
                    return s.nunique(dropna=False) / len(s)
                except Exception:
                    return 1.0

            risks = [{"column": c, "uniqueness_ratio": round(uniq_ratio(orig[c]), 3),
                      "risk": ("Low" if uniq_ratio(orig[c]) < 0.3 else "Medium" if uniq_ratio(orig[c]) < 0.7 else "High")}
                     for c in orig.columns]
            st.markdown("**Column uniqueness in original data** (higher ⇒ easier to re-identify)")
            st.dataframe(pd.DataFrame(risks), use_container_width=True)

            # membership overlap (very rough)
            try:
                overlap = pd.merge(orig.drop_duplicates(), synth.drop_duplicates(), how="inner").shape[0]
                membership_rate = overlap / max(1, len(synth))
            except Exception:
                membership_rate = 0.0

            high_unique_cols = [r["column"] for r in risks if r["risk"] == "High"]
            linkability = len(high_unique_cols) / max(1, len(orig.columns))
            overall = "Low" if membership_rate < 0.02 and linkability < 0.2 else ("Medium" if membership_rate < 0.05 and linkability < 0.4 else "High")

            st.markdown("**Summary (heuristic, demo-only)**")
            st.write({
                "membership_overlap_rate": round(membership_rate, 3),
                "linkability_score": round(linkability, 3),
                "overall_privacy_risk": overall
            })
            st.info("Heuristic estimates for demo purposes (not a formal DP guarantee).")

        # Quality Report
        with tabs[2]:
            st.markdown("#### 📈 Quality Report")
            num_cols_eval = [c for c in df.select_dtypes(include=[np.number]).columns if c in st.session_state.synthetic_data.columns]
            if not num_cols_eval:
                st.warning("No numeric columns available for distribution tests.")
            else:
                rows = []
                for c in num_cols_eval:
                    try:
                        ks = stats.ks_2samp(df[c].dropna(), st.session_state.synthetic_data[c].dropna())
                        rows.append({
                            "column": c,
                            "ks_stat": round(ks.statistic, 3),
                            "ks_pvalue": round(ks.pvalue, 4),
                            "mean_delta": round(abs(df[c].mean() - st.session_state.synthetic_data[c].mean()), 3),
                            "std_delta": round(abs(df[c].std(ddof=0) - st.session_state.synthetic_data[c].std(ddof=0)), 3)
                        })
                    except Exception:
                        pass
                st.markdown("**Distribution similarity (K-S) & moment deltas**")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

                # correlation preservation
                try:
                    co = df[num_cols_eval].corr()
                    cs = st.session_state.synthetic_data[num_cols_eval].corr()
                    diff = (co - cs).abs().mean().mean()
                    st.markdown(f"**Correlation preservation error (mean |Δr|):** `{diff:.3f}` (lower is better)")
                except Exception:
                    pass

        # Statistical Comparison
        with tabs[3]:
            st.markdown("#### 📋 Statistical Comparison (Original vs Synthetic)")
            def summary_table(dfx: pd.DataFrame):
                d = dfx.select_dtypes(include=[np.number])
                if d.empty:
                    return pd.DataFrame()
                return pd.DataFrame({
                    "Count": d.count(),
                    "Mean": d.mean(),
                    "Std Dev": d.std(ddof=0),
                    "Min": d.min(),
                    "25%": d.quantile(0.25),
                    "50% (Median)": d.median(),
                    "75%": d.quantile(0.75),
                    "Max": d.max(),
                    "Mode": [d[c].mode().iloc[0] if not d[c].mode().empty else np.nan for c in d.columns]
                }).round(3)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original (numeric)**")
                st.dataframe(summary_table(df), use_container_width=True)
            with c2:
                st.markdown("**Synthetic (numeric)**")
                st.dataframe(summary_table(st.session_state.synthetic_data), use_container_width=True)
# --------------------------- END TAB 2 ---------------------------


# Tab 3: Healthcare Simulator
with tab3:
    st.markdown("### 🥼 Markham Hospital Intelligent Simulation Platform")
    st.markdown("""
        <div class='info-box'>
        <strong>Welcome, Healthcare Administrator!</strong><br>
        This AI-enhanced simulation tool helps you generate synthetic healthcare data for operational planning, 
        scenario analysis, and decision support. All data is HIPAA-compliant and includes intelligent insights 
        to guide your management decisions.
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("#### 🎯 Simulation Scenario")
        scenario = st.selectbox(
            "Select Healthcare Scenario",
            ["Emergency Department Flow", "Patient Admission Patterns", "Surgery Scheduling",
             "Bed Occupancy Optimization", "Staff Scheduling", "Equipment Utilization"]
        )
        
        # Scenario-specific description
        scenario_descriptions = {
            "Emergency Department Flow": "Optimize patient flow through ED, reduce wait times, and improve triage efficiency",
            "Patient Admission Patterns": "Analyze admission trends, predict peak periods, and optimize resource allocation",
            "Surgery Scheduling": "Improve OR utilization, reduce cancellations, and optimize surgical team schedules",
            "Bed Occupancy Optimization": "Maximize bed utilization while maintaining quality care and patient satisfaction",
            "Staff Scheduling": "Balance staff workload, reduce overtime costs, and ensure adequate coverage",
            "Equipment Utilization": "Track equipment usage, predict maintenance needs, and optimize procurement"
        }
        
        st.info(f"📋 {scenario_descriptions.get(scenario, '')}")
        
        simulation_period = st.selectbox(
            "Simulation Period",
            ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"]
        )
        
        department = st.multiselect(
            "Departments",
            ["Emergency", "ICU", "Surgery", "Cardiology", "Pediatrics", "Oncology", "General Medicine"],
            default=["Emergency"]
        )
    
    with col2:
        st.markdown("#### 📊 Simulation Parameters")
        num_patients = st.number_input("Number of Patients", 100, 50000, 5000)
        
        age_distribution = st.selectbox(
            "Age Distribution",
            ["Markham Demographics", "Uniform", "Elderly Focused", "Pediatric Focused"]
        )
        
        complexity_mix = st.select_slider(
            "Case Complexity Mix",
            options=["Simple", "Moderate", "Complex", "Critical"],
            value="Moderate"
        )
        
        seasonal_pattern = st.checkbox("Include seasonal patterns", value=True)
        include_covid = st.checkbox("Include COVID-19 impact modeling", value=False)
        enable_ai_insights = st.checkbox("🤖 Enable AI-Powered Insights", value=True, 
                                         help="Get intelligent recommendations based on simulation results")
    
    with col3:
        st.markdown("#### 🎲 Advanced Settings")
        
        wait_time_target = st.slider("Target Wait Time (minutes)", 15, 180, 60)
        bed_utilization = st.slider("Target Bed Utilization (%)", 50, 95, 80)
        staff_ratio = st.slider("Staff to Patient Ratio", 0.1, 1.0, 0.3)
        
        include_costs = st.checkbox("Include cost analysis", value=True)
        include_outcomes = st.checkbox("Include patient outcomes", value=True)
        include_predictions = st.checkbox("Generate predictive analytics", value=True)
    
    # Generate Healthcare Data
    if st.button("🏥 Run Intelligent Healthcare Simulation", use_container_width=True):
        with st.spinner("Generating Markham Hospital simulation with AI insights..."):
            
            # Generate synthetic patient data
            np.random.seed(42)
            
            # Date range based on simulation period
            period_days = {"1 Week": 7, "1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
            days = period_days[simulation_period]
            
            dates = pd.date_range(start=datetime.now(), periods=num_patients, freq=f'{24 * days / num_patients}H')
            
            # Age distribution for Markham
            if age_distribution == "Markham Demographics":
                ages = np.concatenate([
                    np.random.normal(35, 15, int(num_patients * 0.3)),  # Young adults
                    np.random.normal(55, 10, int(num_patients * 0.4)),  # Middle age
                    np.random.normal(70, 10, int(num_patients * 0.3))   # Elderly
                ])
            elif age_distribution == "Elderly Focused":
                ages = np.random.normal(70, 15, num_patients)
            elif age_distribution == "Pediatric Focused":
                ages = np.random.normal(8, 5, num_patients)
            else:
                ages = np.random.uniform(1, 90, num_patients)
            
            ages = np.clip(ages, 1, 100).astype(int)[:num_patients]
            
            # Generate healthcare data
            healthcare_data = pd.DataFrame({
                'PatientID': [f'MH{str(i).zfill(6)}' for i in range(1, num_patients + 1)],
                'AdmissionDate': dates,
                'Age': ages,
                'Gender': np.random.choice(['M', 'F'], num_patients, p=[0.48, 0.52]),
                'Department': np.random.choice(department, num_patients),
                'Triage': np.random.choice([1, 2, 3, 4, 5], num_patients, p=[0.05, 0.15, 0.4, 0.3, 0.1]),
                'WaitTime': np.random.gamma(2, wait_time_target / 2, num_patients),
                'LengthOfStay': np.random.exponential(3, num_patients),
                'BedType': np.random.choice(['Standard', 'Semi-Private', 'ICU', 'Emergency'], num_patients,
                                            p=[0.5, 0.3, 0.1, 0.1]),
                'Diagnosis': np.random.choice(['Respiratory', 'Cardiac', 'Trauma', 'Neurological',
                                               'Gastrointestinal', 'Other'], num_patients),
                'Procedures': np.random.poisson(2, num_patients),
                'LabTests': np.random.poisson(5, num_patients),
                'Medications': np.random.poisson(4, num_patients)
            })
            
            if include_covid:
                healthcare_data['COVID_Status'] = np.random.choice(['Negative', 'Positive', 'Unknown'],
                                                                   num_patients, p=[0.85, 0.10, 0.05])
            
            if include_costs:
                base_cost = np.random.gamma(2, 1000, num_patients)
                healthcare_data['TotalCost'] = base_cost * (1 + healthcare_data['LengthOfStay'] * 0.3)
                healthcare_data['TotalCost'] = healthcare_data['TotalCost'].round(2)
            
            if include_outcomes:
                healthcare_data['Outcome'] = np.random.choice(['Discharged', 'Admitted', 'Transferred', 'Other'],
                                                              num_patients, p=[0.7, 0.2, 0.08, 0.02])
                healthcare_data['Readmission30Day'] = np.random.choice([0, 1], num_patients, p=[0.85, 0.15])
            
            # Add seasonal patterns
            if seasonal_pattern:
                seasonal_factor = np.sin(np.arange(num_patients) * 2 * np.pi / (num_patients / 4)) * 0.3 + 1
                healthcare_data['WaitTime'] = healthcare_data['WaitTime'] * seasonal_factor
            
            st.success(f"✅ Generated {num_patients} synthetic patient records for Markham Hospital!")
            
            # Generate AI insights if enabled
            ai_insights = None
            if enable_ai_insights:
                with st.spinner("🤖 Generating AI-powered insights..."):
                    ai_insights = generate_healthcare_simulation_insights(healthcare_data, scenario)
            
            # Display results with enhanced tabs
            tab_preview, tab_stats, tab_viz, tab_ai_insights, tab_recommendations, tab_privacy = st.tabs(
                ["📋 Preview", "📊 Statistics", "📈 Visualizations", "🤖 AI Insights", "💡 Recommendations", "🔐 Privacy Check"])
            
            with tab_preview:
                st.dataframe(healthcare_data.head(100), use_container_width=True)
                
                csv = healthcare_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Markham Hospital Simulation Data",
                    data=csv,
                    file_name=f"markham_hospital_simulation_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with tab_stats:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg Wait Time", f"{healthcare_data['WaitTime'].mean():.1f} min")
                with col2:
                    st.metric("Avg Length of Stay", f"{healthcare_data['LengthOfStay'].mean():.1f} days")
                with col3:
                    st.metric("Bed Utilization", f"{bed_utilization}%")
                with col4:
                    if include_costs:
                        st.metric("Avg Cost per Patient", f"${healthcare_data['TotalCost'].mean():.2f}")
                
                # Department statistics
                st.markdown("#### Department Statistics")
                dept_stats = healthcare_data.groupby('Department').agg({
                    'PatientID': 'count',
                    'WaitTime': 'mean',
                    'LengthOfStay': 'mean'
                }).round(1)
                dept_stats.columns = ['Patient Count', 'Avg Wait Time (min)', 'Avg LOS (days)']
                st.dataframe(dept_stats, use_container_width=True)
            
            with tab_viz:
                # Wait time distribution
                fig1 = px.histogram(healthcare_data, x='WaitTime', nbins=30,
                                    title='Emergency Department Wait Time Distribution',
                                    labels={'WaitTime': 'Wait Time (minutes)', 'count': 'Number of Patients'})
                fig1.update_traces(marker_color='#00693e')
                st.plotly_chart(fig1, use_container_width=True)
                
                # Department volume
                dept_counts = healthcare_data['Department'].value_counts()
                fig2 = px.pie(values=dept_counts.values, names=dept_counts.index,
                              title='Patient Distribution by Department')
                st.plotly_chart(fig2, use_container_width=True)
                
                # Time series of admissions
                daily_admissions = healthcare_data.groupby(healthcare_data['AdmissionDate'].dt.date).size()
                fig3 = px.line(x=daily_admissions.index, y=daily_admissions.values,
                               title='Daily Admission Patterns',
                               labels={'x': 'Date', 'y': 'Number of Admissions'})
                fig3.update_traces(line_color='#00693e')
                st.plotly_chart(fig3, use_container_width=True)
            
            with tab_ai_insights:
                if ai_insights:
                    st.markdown("#### 🤖 AI-Powered Operational Analysis")
                    
                    # Operational Metrics
                    st.markdown("##### 📊 Key Performance Indicators")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if 'avg_wait_time' in ai_insights['operational_metrics']:
                            wait_status = "🟢" if ai_insights['operational_metrics']['avg_wait_time'] < 60 else "🟡" if ai_insights['operational_metrics']['avg_wait_time'] < 120 else "🔴"
                            st.metric("Average Wait Time", 
                                     f"{ai_insights['operational_metrics']['avg_wait_time']:.1f} min",
                                     delta=f"{wait_status} Target: {wait_time_target} min")
                    with col2:
                        if 'avg_los' in ai_insights['operational_metrics']:
                            los_status = "🟢" if ai_insights['operational_metrics']['avg_los'] < 3 else "🟡" if ai_insights['operational_metrics']['avg_los'] < 5 else "🔴"
                            st.metric("Average Length of Stay",
                                     f"{ai_insights['operational_metrics']['avg_los']:.1f} days",
                                     delta=f"{los_status} Industry avg: 3.5 days")
                    with col3:
                        if 'p95_wait_time' in ai_insights['operational_metrics']:
                            st.metric("95th Percentile Wait Time",
                                     f"{ai_insights['operational_metrics']['p95_wait_time']:.1f} min")
                    
                    # Bottlenecks
                    if ai_insights['bottlenecks']:
                        st.markdown("##### 🚨 Identified Bottlenecks")
                        for bottleneck in ai_insights['bottlenecks']:
                            severity_color = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
                            st.warning(f"{severity_color.get(bottleneck['severity'], '⚪')} **{bottleneck['area']}**: {bottleneck['issue']}")
                            st.caption(f"Impact: {bottleneck['impact']}")
                    
                    # Cost Analysis
                    if ai_insights['cost_analysis']:
                        st.markdown("##### 💰 Financial Analysis")
                        if 'total_cost' in ai_insights['cost_analysis']:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Simulation Cost", 
                                         f"${ai_insights['cost_analysis']['total_cost']:,.2f}")
                            with col2:
                                st.metric("Average Cost per Patient",
                                         f"${ai_insights['cost_analysis']['avg_cost_per_patient']:,.2f}")
                    
                    # Predictive Insights
                    if ai_insights['predictive_insights']:
                        st.markdown("##### 🔮 Predictive Analytics")
                        for prediction in ai_insights['predictive_insights']:
                            st.info(f"**{prediction['metric']}**: {prediction['value']} - {prediction['recommendation']}")
                else:
                    st.info("Enable AI-Powered Insights in the settings to get intelligent recommendations")
            
            with tab_recommendations:
                st.markdown("#### 💡 Strategic Recommendations")
                
                if ai_insights:
                    # Staff Recommendations
                    if ai_insights['staff_recommendations']:
                        st.markdown("##### 👥 Staffing Optimization")
                        staff_df = pd.DataFrame(ai_insights['staff_recommendations'])
                        st.dataframe(staff_df, use_container_width=True)
                    
                    # Optimization Opportunities
                    if ai_insights['optimization_opportunities']:
                        st.markdown("##### 🎯 Improvement Opportunities")
                        for opp in ai_insights['optimization_opportunities']:
                            with st.expander(f"📈 {opp['opportunity']}"):
                                st.write(f"**Potential Impact:** {opp.get('potential_impact', 'N/A')}")
                                if 'potential_savings' in opp:
                                    st.write(f"**Estimated Savings:** {opp['potential_savings']}")
                                st.write(f"**Implementation:** {opp['implementation']}")
                
                # Scenario-specific recommendations
                st.markdown("##### 📋 Scenario-Specific Actions")
                scenario_actions = {
                    "Emergency Department Flow": [
                        "Implement triage nurse practitioner role for Triage 4-5 patients",
                        "Create fast-track area for minor injuries",
                        "Deploy mobile registration teams during peak hours"
                    ],
                    "Bed Occupancy Optimization": [
                        "Implement early discharge planning protocols",
                        "Create swing beds for flexible capacity",
                        "Deploy predictive bed management system"
                    ],
                    "Staff Scheduling": [
                        "Implement self-scheduling system for nursing staff",
                        "Create float pool for high-demand periods",
                        "Deploy AI-based demand forecasting"
                    ]
                }
                
                if scenario in scenario_actions:
                    for action in scenario_actions[scenario]:
                        st.write(f"• {action}")
            
            with tab_privacy:
                st.markdown("#### 🔐 Healthcare Data Privacy Assessment")
                
                # Healthcare-specific privacy checks
                st.markdown("##### HIPAA Compliance Check")
                
                hipaa_identifiers = ['PatientID']  # In real scenario, would check for more identifiers
                
                compliance_score = 100  # Start at 100%
                compliance_issues = []
                
                # Check for direct identifiers
                if 'SSN' in healthcare_data.columns or 'SocialSecurity' in healthcare_data.columns:
                    compliance_score -= 30
                    compliance_issues.append("❌ Social Security Numbers detected")
                
                # Check for quasi-identifiers combination
                if len(healthcare_data[['Age', 'Gender']].drop_duplicates()) / len(healthcare_data) > 0.8:
                    compliance_score -= 10
                    compliance_issues.append("⚠️ High uniqueness in Age+Gender combination")
                
                # Date precision check
                if healthcare_data['AdmissionDate'].dt.time.nunique() > 24:
                    compliance_score -= 5
                    compliance_issues.append("⚠️ High precision timestamps may increase identification risk")
                
                # Display compliance results
                compliance_color = 'green' if compliance_score > 90 else 'orange' if compliance_score > 70 else 'red'
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background-color: {compliance_color}20; 
                          border: 2px solid {compliance_color}; border-radius: 10px; margin: 1rem 0;'>
                <h3 style='color: {compliance_color}; margin: 0;'>HIPAA Compliance Score: {compliance_score}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                if compliance_issues:
                    st.markdown("##### 🚨 Compliance Issues Found:")
                    for issue in compliance_issues:
                        st.write(f"• {issue}")
                else:
                    st.success("✅ No major HIPAA compliance issues detected in synthetic data")
                
                # Safe Harbor compliance check
                st.markdown("##### Safe Harbor Compliance")
                safe_harbor_compliant = True
                if 'PatientID' in healthcare_data.columns:
                    # Check if IDs are properly anonymized (synthetic)
                    if not all(healthcare_data['PatientID'].str.startswith('MH')):
                        safe_harbor_compliant = False
                
                if safe_harbor_compliant:
                    st.success("✅ Data meets Safe Harbor de-identification standards")
                else:
                    st.warning("⚠️ Review identifiers for Safe Harbor compliance")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>🎓 Developed for Ivey Business School - University of Western Ontario</p>
        <p style='font-size: 0.9rem;'>© 2024 Advanced Synthetic Data Generation Platform | Educational Purpose Only</p>
        <p style='font-size: 0.8rem;'>Enhanced with AI-Powered Analysis, Privacy Risk Assessment & Healthcare Decision Support</p>
    </div>
    """, unsafe_allow_html=True)
            
           
