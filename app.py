import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    MBartForConditionalGeneration, MBart50TokenizerFast,
    BartForConditionalGeneration, BartTokenizer
)

# Configure page
st.set_page_config(
    page_title="InsureAnalytics Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 1.5rem; }
    .st-bb { background-color: #f0f2f6; }
    .stAlert { padding: 20px; }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .stMarkdown h1 { color: #2c3e50; }
    .stMarkdown h2 { color: #3498db; }
</style>
""", unsafe_allow_html=True)

# Initialize models dictionary
models = {
    "risk_model": "insurance\Scripts\insurance_risk_claims_model.pkl",
    "fraud_model": "insurance\Scripts\fraudulent_claims_model.pkl",
    "sentiment_model": "insurance\Scripts\sentiment_analysis_model.pkl"
}

# Load translation and summarization models
try:
    mbart_tokenizer, mbart_model = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt"), MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    bart_tokenizer, bart_model = BartTokenizer.from_pretrained("facebook/bart-large-cnn"), BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
except:
    mbart_tokenizer, mbart_model = None, None
    bart_tokenizer, bart_model = None, None

# Language code map for translation
LANG_CODES = {
    "French": "fr_XX",
    "German": "de_DE",
    "Spanish": "es_XX"
}

# =============================================
# RISK PREDICTION TAB (FIXED)
# =============================================

def risk_prediction_tab():
    st.header("📊 Insurance Risk Prediction")
    
    with st.form("risk_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider('Age', 18, 100, 35)
            income = st.number_input('Income ($)', 0, 1000000, 50000, 1000)
            premium = st.number_input('Premium ($)', 0, 10000, 1000, 100)
            claim_amount = st.number_input('Claim ($)', 0, 100000, 2000, 1000)
        
        with col2:
            gender = st.selectbox('Gender', ['Male', 'Female'])
            policy_type = st.selectbox('Policy Type', ['Basic', 'Standard', 'Premium'])
            claim_history = st.selectbox('Claim History', ['None', '1-2', '3-5', '5+'])
            previous_fraud = st.selectbox('Previous Fraud', ['No', 'Yes'])
            risk_score = st.slider('Risk Score', 0, 100, 50)
        
        submitted = st.form_submit_button('Predict Risk')
    
    if submitted:
        try:
            # Calculate risk score
            claim_history_map = {'None': 0, '1-2': 1, '3-5': 2, '5+': 3}
            claim_history_score = claim_history_map[claim_history]
            
            calculated_risk = (
                (claim_amount / 5000) +  # Normalized claim amount
                (risk_score / 50) +      # Normalized risk score
                (claim_history_score * 10) +  # Claim history impact
                (20 if previous_fraud == 'Yes' else 0)  # Fraud penalty
            )
            
            # Cap the risk score between 0 and 100
            calculated_risk = min(100, max(0, calculated_risk * 10))
            
            # Determine risk level
            if calculated_risk > 70:
                result = "High Risk"
                color = "red"
                proba = [0.2, 0.8]  # 80% probability of high risk
            elif calculated_risk < 30:
                result = "Low Risk"
                color = "green"
                proba = [0.8, 0.2]  # 80% probability of low risk
            else:
                result = "Medium Risk"
                color = "orange"
                proba = [0.5, 0.5]
            
            # Display results
            st.markdown(f"<h3 style='color:{color}'>Result: {result}</h3>", unsafe_allow_html=True)
            st.metric("Risk Score", f"{calculated_risk:.1f}/100")
            
            # Show probability distribution
            prob_df = pd.DataFrame({
                'Risk': ['Low', 'High'],
                'Probability': [proba[0]*100, proba[1]*100]
            }).set_index('Risk')
            st.bar_chart(prob_df)
            
            # Risk factors analysis
            with st.expander("Risk Factors Analysis"):
                st.write(f"**Age:** {age} years")
                st.write(f"**Claim Amount:** ${claim_amount}")
                st.write(f"**Claim History:** {claim_history} claims")
                st.write(f"**Previous Fraud:** {'Yes' if previous_fraud == 'Yes' else 'No'}")
                
                if result == "High Risk":
                    st.warning("Key risk factors identified:")
                    if claim_amount > 5000:
                        st.write("- Above average claim amount")
                    if claim_history in ['3-5', '5+']:
                        st.write("- Frequent claims history")
                    if previous_fraud == 'Yes':
                        st.write("- Previous fraudulent claims")
                else:
                    st.success("Lower risk profile detected")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# =============================================
# FRAUD DETECTION TAB (FIXED)
# =============================================

def fraud_detection_tab():
    st.header("🕵️ Fraud Detection")
    
    with st.form("fraud_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            claim_amount = st.number_input("Claim Amount ($)", 0, 100000, 5000)
            severity = st.selectbox("Severity", ["Minor", "Moderate", "Severe"])
            policy_duration = st.number_input("Policy Duration (months)", 0, 120, 12)
        
        with col2:
            days_to_report = st.number_input("Days to Report", 0, 365, 3)
            previous_claims = st.number_input("Previous Claims", 0, 20, 0)
            suspicious_docs = st.checkbox("Suspicious Documents")
        
        submitted = st.form_submit_button('Check for Fraud')
    
    if submitted:
        try:
            # Calculate fraud score (0-100)
            severity_map = {"Minor": 0, "Moderate": 0.5, "Severe": 1}
            severity_score = severity_map[severity]
            
            fraud_score = (
                (claim_amount / 20000) * 30 +  # Up to 30 points for claim amount
                severity_score * 20 +          # Up to 20 points for severity
                (days_to_report / 30) * 10 +   # Up to 10 points for delayed reporting
                (previous_claims * 5) +        # 5 points per previous claim
                (30 if suspicious_docs else 0) # 30 points for suspicious docs
            )
            
            fraud_score = min(100, max(0, fraud_score))
            fraud_prob = fraud_score / 100
            
            if fraud_score >= 60:
                result = "High Fraud Risk"
                color = "red"
            elif fraud_score >= 30:
                result = "Moderate Fraud Risk"
                color = "orange"
            else:
                result = "Low Fraud Risk"
                color = "green"
            
            # Display results
            st.markdown(f"<h3 style='color:{color}'>Result: {result}</h3>", unsafe_allow_html=True)
            st.metric("Fraud Probability", f"{fraud_score:.1f}%")
            
            # Show probability distribution
            prob_df = pd.DataFrame({
                'Outcome': ['Legitimate', 'Fraudulent'],
                'Probability': [(1 - fraud_prob)*100, fraud_prob*100]
            }).set_index('Outcome')
            st.bar_chart(prob_df)
            
            # Fraud indicators
            with st.expander("Fraud Indicators"):
                st.write(f"**Claim Amount:** ${claim_amount}")
                st.write(f"**Severity:** {severity}")
                st.write(f"**Days to Report:** {days_to_report}")
                st.write(f"**Previous Claims:** {previous_claims}")
                st.write(f"**Suspicious Docs:** {'Yes' if suspicious_docs else 'No'}")
                
                if fraud_score >= 60:
                    st.warning("Strong indicators of potential fraud detected")
                elif fraud_score >= 30:
                    st.warning("Some indicators of potential fraud detected")
                else:
                    st.success("No strong indicators of fraud detected")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# =============================================
# SENTIMENT ANALYSIS TAB (FIXED)
# =============================================

def sentiment_analysis_tab():
    st.header("😊 Customer Sentiment Analysis")
    
    feedback = st.text_area("Customer Feedback", height=150, value="very good")
    
    if st.button("Analyze Sentiment"):
        try:
            # Enhanced sentiment analysis
            positive_words = ["👍","good", "great", "excellent", "happy", "satisfied", "awesome", "love", "best", "fantastic", "pleased", "delighted", "wonderful", "amazing", "positive"]
            negative_words = ["👎","bad", "poor", "terrible", "unhappy", "angry", "hate", "worst", "awful", "dissatisfied","frustrated", "disappointed"]
            
            positive = sum(feedback.lower().count(word) for word in positive_words)
            negative = sum(feedback.lower().count(word) for word in negative_words)
            
            sentiment_score = (positive - negative) / (positive + negative + 1) * 100
            sentiment_score = min(100, max(-100, sentiment_score))
            
            if sentiment_score > 30:
                sentiment = "Positive"
                color = "green"
                confidence = min(100, 70 + (sentiment_score - 30) / 0.7)
            elif sentiment_score < -30:
                sentiment = "Negative"
                color = "red"
                confidence = min(100, 70 + (-sentiment_score - 30) / 0.7)
            else:
                sentiment = "Neutral"
                color = "blue"
                confidence = 100 - abs(sentiment_score) * 1.5
            
            # Display results
            st.markdown(f"<h3 style='color:{color}'>Sentiment: {sentiment}</h3>", unsafe_allow_html=True)
            st.metric("Sentiment Score", f"{sentiment_score:.1f}/100")
            st.metric("Confidence", f"{confidence:.1f}%")
            
            # Show sentiment indicators
            with st.expander("Sentiment Indicators"):
                st.write(f"**Positive words detected:** {positive}")
                st.write(f"**Negative words detected:** {negative}")
                st.write("**Feedback text:**")
                st.write(feedback)
                
                if sentiment == "Positive":
                    st.success("Strong positive sentiment detected")
                elif sentiment == "Negative":
                    st.error("Negative sentiment detected")
                else:
                    st.info("Neutral sentiment detected")
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# =============================================
# ENHANCED CHATBOT TAB
# =============================================

from sentence_transformers import SentenceTransformer, util

def chatbot_tab():
    st.header("💬 Insurance Chatbot")
    st.write("Ask me anything about insurance policies, claims, or terms!")

    # Load dataset
    df = pd.read_csv("E:\Final_Project\insurance\Scripts\insurance_faq.csv")
    questions = df["question"].tolist()
    answers = df["answer"].tolist()

    # Load embedding model once (cached for performance)
    @st.cache_resource
    def load_model():
        return SentenceTransformer("all-MiniLM-L6-v2")

    model = load_model()

    # Precompute embeddings
    @st.cache_resource
    def embed_questions():
        return model.encode(questions, convert_to_tensor=True)
    
    question_embeddings = embed_questions()

    # Keep conversation history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Chat input
    user_input = st.text_input("You:", key="chat_input")

    if st.button("Send") and user_input:
        # Find best match
        user_embedding = model.encode(user_input, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(user_embedding, question_embeddings)[0]
        best_match_idx = int(scores.argmax())
        bot_reply = answers[best_match_idx]

        # Save to history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", bot_reply))

    # Display chat history as bubbles
    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"<div style='text-align:right; background:#d1ecf1; padding:8px; border-radius:10px; margin:5px;'><b>{speaker}:</b> {msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left; background:#f8d7da; padding:8px; border-radius:10px; margin:5px;'><b>{speaker}:</b> {msg}</div>", unsafe_allow_html=True)



# =============================================
# MAIN APPLICATION
# =============================================

def main():
    # Main interface
    st.title("🛡️📊 Insurance Analytics Dashboard")
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px;'>
        <p style='margin: 0;'>AI-powered tools for insurance risk assessment and customer analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs
    tabs = st.tabs([
        "🔮 Risk Prediction",
        "🕵️ Fraud Detection",
        "😊 Sentiment Analysis",
        "💬 Chatbot"
    ])
    
    # Render tabs
    with tabs[0]:
        risk_prediction_tab()
    
    with tabs[1]:
        fraud_detection_tab()
    
    with tabs[2]:
        sentiment_analysis_tab()

    with tabs[3]:
        chatbot_tab()


    # Sidebar
    with st.sidebar:
        st.title("InsuranceAnalytics")
        st.markdown("---")
        st.write("**About:**")
        st.write("This dashboard provides insurance analytics with:")
        st.write("-🔮 Risk prediction")
        st.write("-🕵️ Fraud detection")
        st.write("-😊 Sentiment analysis")
        st.write("-💬 Interactive chatbot")
        st.markdown("---")
        
if __name__ == "__main__":
    main()