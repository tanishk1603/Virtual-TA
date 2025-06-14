#!/usr/bin/env python3
"""
TDS Virtual TA - Streamlit Web Interface
A web-based interface for the TDS Virtual Teaching Assistant
"""

import streamlit as st
import json
import pickle
import os
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Import the main Virtual TA class
# Note: Make sure tds_virtual_ta.py is in the same directory
try:
    from tds_virtual_ta import TDSVirtualTA
except ImportError:
    st.error("Please ensure tds_virtual_ta.py is in the same directory as this file.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="TDS Virtual TA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .question-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1e3c72;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .answer-box {
        background-color: #e8f4fd;
        border-left: 4px solid #2a5298;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_virtual_ta():
    """Load and cache the Virtual TA instance"""
    ta = TDSVirtualTA(data_dir=".", min_similarity=0.3)
    
    # Try to load pre-trained model first
    model_path = "tds_ta_model.pkl"
    if os.path.exists(model_path):
        try:
            ta.load_model(model_path)
            return ta
        except Exception as e:
            st.warning(f"Could not load pre-trained model: {e}")
    
    # Build knowledge base if no model exists
    with st.spinner("Building knowledge base... This may take a few minutes."):
        ta.build_knowledge_base()
        # Save the model for future use
        try:
            ta.save_model(model_path)
        except Exception as e:
            st.warning(f"Could not save model: {e}")
    
    return ta

def get_confidence_class(confidence):
    """Get CSS class based on confidence level"""
    if confidence >= 0.7:
        return "confidence-high"
    elif confidence >= 0.4:
        return "confidence-medium"
    else:
        return "confidence-low"

def display_sources(sources):
    """Display source information in a nice format"""
    if not sources:
        return
    
    st.subheader("📚 Sources")
    
    for i, source in enumerate(sources[:3], 1):  # Show top 3 sources
        with st.expander(f"Source {i} - {source['type'].title()} (Similarity: {source['similarity']:.2f})"):
            if source['type'] == 'qa_pair':
                st.write(f"**Question:** {source.get('question', 'N/A')}")
                st.write(f"**Author:** {source.get('username', 'Unknown')}")
                if source.get('post_id'):
                    st.write(f"**Post ID:** {source['post_id']}")
            elif source['type'] == 'website_page':
                st.write(f"**Page Title:** {source.get('title', 'N/A')}")
                if source.get('file_path'):
                    st.write(f"**File:** {Path(source['file_path']).name}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 TDS Virtual Teaching Assistant</h1>
        <p>Ask questions about Tools in Data Science course content</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Confidence threshold
        min_similarity = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Answers below this confidence level won't be shown"
        )
        
        st.header("📊 Quick Stats")
        
        # Load Virtual TA
        try:
            ta = load_virtual_ta()
            ta.min_similarity = min_similarity
            
            stats = ta.get_statistics()
            if 'error' not in stats:
                st.metric("Total Documents", stats['total_documents'])
                st.metric("Vocabulary Size", stats['vocabulary_size'])
                st.metric("Avg Document Length", f"{stats['avg_document_length']:.0f} chars")
                
                # Document type distribution
                if 'document_types' in stats:
                    st.subheader("Document Types")
                    for doc_type, count in stats['document_types'].items():
                        st.write(f"• {doc_type.replace('_', ' ').title()}: {count}")
            
        except Exception as e:
            st.error(f"Error loading Virtual TA: {e}")
            st.stop()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Question")
        
        # Question input
        user_question = st.text_area(
            "What would you like to know about TDS?",
            height=100,
            placeholder="e.g., What is data preprocessing? How do I use pandas for data analysis?",
            help="Ask any question related to the Tools in Data Science course content"
        )
        
        # Submit button
        if st.button("🔍 Get Answer", type="primary", use_container_width=True):
            if user_question.strip():
                with st.spinner("Searching for the best answer..."):
                    result = ta.generate_answer(user_question)
                
                # Display question
                st.markdown(f"""
                <div class="question-box">
                    <strong>❓ Your Question:</strong><br>
                    {user_question}
                </div>
                """, unsafe_allow_html=True)
                
                # Display answer
                confidence_class = get_confidence_class(result['confidence'])
                confidence_text = f"{result['confidence']:.2f}"
                
                if result['confidence'] >= min_similarity:
                    st.markdown(f"""
                    <div class="answer-box">
                        <strong>🤖 Answer:</strong>
                        <span class="{confidence_class}" style="float: right;">
                            Confidence: {confidence_text}
                        </span><br><br>
                        {result['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display sources
                    if result['sources']:
                        display_sources(result['sources'])
                        
                else:
                    st.warning(f"""
                    I couldn't find a confident answer to your question (confidence: {confidence_text}).
                    Try rephrasing your question or being more specific about the topic.
                    """)
                    
                # Store in session state for history
                if 'qa_history' not in st.session_state:
                    st.session_state.qa_history = []
                
                st.session_state.qa_history.append({
                    'question': user_question,
                    'answer': result['answer'],
                    'confidence': result['confidence'],
                    'timestamp': datetime.now(),
                    'sources_count': len(result['sources'])
                })
                
            else:
                st.warning("Please enter a question!")
    
    with col2:
        st.header("💡 Sample Questions")
        
        sample_questions = [
            "What is data science?",
            "How do I use pandas for data analysis?",
            "What are the course requirements?",
            "How to handle missing data?",
            "What is machine learning?",
            "How to create visualizations?",
            "What is statistical analysis?",
            "How to clean messy data?",
            "What are the assignment deadlines?",
            "How to use Jupyter notebooks?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{hash(question)}", use_container_width=True):
                st.session_state.sample_question = question
                st.rerun()
        
        # Handle sample question selection
        if hasattr(st.session_state, 'sample_question'):
            user_question = st.session_state.sample_question
            del st.session_state.sample_question
            
            with st.spinner("Searching for the best answer..."):
                result = ta.generate_answer(user_question)
            
            st.markdown(f"""
            <div class="question-box">
                <strong>❓ Sample Question:</strong><br>
                {user_question}
            </div>
            """, unsafe_allow_html=True)
            
            confidence_class = get_confidence_class(result['confidence'])
            confidence_text = f"{result['confidence']:.2f}"
            
            if result['confidence'] >= min_similarity:
                st.markdown(f"""
                <div class="answer-box">
                    <strong>🤖 Answer:</strong>
                    <span class="{confidence_class}" style="float: right;">
                        Confidence: {confidence_text}
                    </span><br><br>
                    {result['answer'][:300]}...
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"Low confidence answer ({confidence_text})")
    
    # Question History Section
    if 'qa_history' in st.session_state and st.session_state.qa_history:
        st.header("📝 Recent Questions")
        
        # Show history in reverse chronological order
        for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):  # Last 5 questions
            with st.expander(f"{qa['timestamp'].strftime('%H:%M')} - {qa['question'][:50]}..."):
                st.write(f"**Question:** {qa['question']}")
                st.write(f"**Answer:** {qa['answer'][:200]}...")
                st.write(f"**Confidence:** {qa['confidence']:.2f}")
                st.write(f"**Sources:** {qa['sources_count']}")
        
        if st.button("Clear History"):
            st.session_state.qa_history = []
            st.rerun()
    
    # Analytics Section
    st.header("📈 Analytics")
    
    if 'qa_history' in st.session_state and len(st.session_state.qa_history) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence distribution
            confidences = [qa['confidence'] for qa in st.session_state.qa_history]
            fig = px.histogram(
                x=confidences,
                nbins=10,
                title="Answer Confidence Distribution",
                labels={'x': 'Confidence Score', 'y': 'Number of Questions'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Questions over time
            timestamps = [qa['timestamp'] for qa in st.session_state.qa_history]
            df_time = pd.DataFrame({'timestamp': timestamps})
            df_time['hour'] = df_time['timestamp'].dt.hour
            
            hourly_counts = df_time['hour'].value_counts().sort_index()
            
            fig = px.bar(
                x=hourly_counts.index,
                y=hourly_counts.values,
                title="Questions by Hour",
                labels={'x': 'Hour of Day', 'y': 'Number of Questions'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Ask some questions to see analytics!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>TDS Virtual TA - Built for Tools in Data Science Course</p>
        <p>💡 Tip: Try asking specific questions for better results</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()