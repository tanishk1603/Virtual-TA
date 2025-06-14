# TDS Virtual TA - Demo and Analysis
# This notebook demonstrates the TDS Virtual Teaching Assistant functionality
# and provides analysis of the knowledge base and performance

# Import required libraries
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("📚 TDS Virtual TA - Demo and Analysis Notebook")
print("=" * 50)

# Import the Virtual TA
from tds_virtual_ta import TDSVirtualTA

# Initialize the Virtual TA
print("🤖 Initializing TDS Virtual Teaching Assistant...")
ta = TDSVirtualTA(data_dir=".", min_similarity=0.3)

# Build knowledge base
print("🏗️ Building knowledge base...")
ta.build_knowledge_base()

# Get basic statistics
stats = ta.get_statistics()
print("\n📊 Knowledge Base Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Demo Questions
demo_questions = [
    "What is data science?",
    "How do I use pandas?",
    "What are the course requirements?",
    "How to handle missing data?",
    "What is machine learning?",
    "How to create visualizations?",
    "What is statistical analysis?",
    "How to clean messy data?",
    "What are Python libraries for data science?",
    "How to perform exploratory data analysis?"
]

print(f"\n🎯 Testing with {len(demo_questions)} sample questions...")

# Test questions and collect results
results = []
for i, question in enumerate(demo_questions, 1):
    print(f"\n--- Question {i} ---")
    print(f"Q: {question}")
    
    result = ta.generate_answer(question)
    results.append(result)
    
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Answer length: {len(result['answer'])} characters")
    print(f"Sources: {len(result['sources'])}")
    
    if result['confidence'] >= 0.3:
        # Show first 200 characters of answer
        answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
        print(f"A: {answer_preview}")
    else:
        print("A: [Low confidence - answer not shown]")

# Analyze results
confidences = [r['confidence'] for r in results]
answer_lengths = [len(r['answer']) for r in results]
source_counts = [len(r['sources']) for r in results]

print(f"\n📈 Performance Analysis:")
print(f"  Average Confidence: {np.mean(confidences):.3f}")
print(f"  Questions with high confidence (>0.7): {sum(1 for c in confidences if c > 0.7)}")
print(f"  Questions with medium confidence (0.4-0.7): {sum(1 for c in confidences if 0.4 <= c <= 0.7)}")
print(f"  Questions with low confidence (<0.4): {sum(1 for c in confidences if c < 0.4)}")
print(f"  Average answer length: {np.mean(answer_lengths):.0f} characters")
print(f"  Average number of sources: {np.mean(source_counts):.1f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('TDS Virtual TA - Performance Analysis', fontsize=16, fontweight='bold')

# Confidence distribution
axes[0, 0].hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Confidence Score Distribution')
axes[0, 0].set_xlabel('Confidence Score')
axes[0, 0].set_ylabel('Number of Questions')
axes[0, 0].axvline(np.mean(confidences), color='red', linestyle='--', label=f'Mean: {np.mean(confidences):.3f}')
axes[0, 0].legend()

# Answer length vs confidence
axes[0, 1].scatter(confidences, answer_lengths, alpha=0.7, color='lightcoral')
axes[0, 1].set_title('Answer Length vs Confidence')
axes[0, 1].set_xlabel('Confidence Score')
axes[0, 1].set_ylabel('Answer Length (characters)')

# Source count distribution
axes[1, 0].bar(range(len(source_counts)), source_counts, alpha=0.7, color='lightgreen')
axes[1, 0].set_title('Number of Sources per Question')
axes[1, 0].set_xlabel('Question Index')
axes[1, 0].set_ylabel('Number of Sources')
axes[1, 0].set_xticks(range(len(demo_questions)))
axes[1, 0].set_xticklabels([f'Q{i+1}' for i in range(len(demo_questions))], rotation=45)

# Confidence categories
conf_categories = ['Low (<0.4)', 'Medium (0.4-0.7)', 'High (>0.7)']
conf_counts = [
    sum(1 for c in confidences if c < 0.4),
    sum(1 for c in confidences if 0.4 <= c <= 0.7),
    sum(1 for c in confidences if c > 0.7)
]
axes[1, 1].pie(conf_counts, labels=conf_categories, autopct='%1.1f%%', startangle=90)
axes[1, 1].set_title('Confidence Categories')

plt.tight_layout()
plt.show()

# Detailed results table
print(f"\n📋 Detailed Results Table:")
results_df = pd.DataFrame({
    'Question': [q[:50] + "..." if len(q) > 50 else q for q in demo_questions],
    'Confidence': [f"{c:.3f}" for c in confidences],
    'Answer_Length': answer_lengths,
    'Sources': source_counts,
    'Status': ['High' if c > 0.7 else 'Medium' if c >= 0.4 else 'Low' for c in confidences]
})

print(results_df.to_string(index=False))

# Knowledge base analysis
print(f"\n🔍 Knowledge Base Deep Dive:")

# Document type analysis
doc_types = stats.get('document_types', {})
if doc_types:
    print("\nDocument Type Distribution:")
    for doc_type, count in doc_types.items():
        percentage = (count / stats['total_documents']) * 100
        print(f"  {doc_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

# Vocabulary analysis
if ta.vectorizer:
    print(f"\nVocabulary Analysis:")
    print(f"  Total vocabulary size: {len(ta.vectorizer.vocabulary_)}")
    print(f"  Feature names sample: {list(ta.vectorizer.get_feature_names_out())[:10]}")
    
    # Most important terms (highest average TF-IDF scores)
    tfidf_scores = ta.tfidf_matrix.mean(axis=0).A1
    feature_names = ta.vectorizer.get_feature_names_out()
    
    # Get top terms
    top_indices = tfidf_scores.argsort()[-20:][::-1]
    top_terms = [(feature_names[i], tfidf_scores[i]) for i in top_indices]
    
    print(f"\nTop 10 Most Important Terms:")
    for term, score in top_terms[:10]:
        print(f"  {term}: {score:.4f}")

# Sample document analysis
print(f"\n📄 Sample Document Analysis:")
if ta.documents:
    sample_doc = ta.documents[0]
    sample_metadata = ta.document_metadata[0]
    
    print(f"Sample Document Type: {sample_metadata['type']}")
    print(f"Sample Document Length: {len(sample_doc)} characters")
    print(f"Sample Content Preview: {sample_doc[:200]}...")

# Performance recommendations
print(f"\n💡 Recommendations:")
avg_confidence = np.mean(confidences)
if avg_confidence < 0.5:
    print("  - Consider lowering the similarity threshold for more answers")
    print("  - Add more diverse content to the knowledge base")
    print("  - Improve text preprocessing for better matching")
elif avg_confidence > 0.8:
    print("  - System is performing very well!")
    print("  - Consider raising similarity threshold for higher precision")
else:
    print("  - System performance is balanced")
    print("  - Current settings appear optimal")

low_conf_count = sum(1 for c in confidences if c < 0.4)
if low_conf_count > len(confidences) * 0.3:
    print("  - Many questions have low confidence - consider expanding knowledge base")

# Interactive demo
print(f"\n🎮 Interactive Demo:")
print("You can now ask questions interactively!")
print("Type 'quit' to exit")

while True:
    try:
        user_question = input("\n🤔 Your question: ").strip()
        
        if user_question.lower() == 'quit':
            break
        
        if not user_question:
            continue
        
        result = ta.generate_answer(user_question)
        
        print(f"\n🤖 Answer (Confidence: {result['confidence']:.3f}):")
        if result['confidence'] >= 0.3:
            print(result['answer'])
            
            if result['sources']:
                print(f"\n📚 Top Sources:")
                for i, source in enumerate(result['sources'][:2], 1):
                    print(f"  {i}. {source['type']} (similarity: {source['similarity']:.3f})")
        else:
            print("I couldn't find a confident answer. Try rephrasing your question.")
    
    except KeyboardInterrupt:
        print(f"\n👋 Thanks for trying the TDS Virtual TA!")
        break
    except Exception as e:
        print(f"Error: {e}")

print(f"\n✅ Demo completed successfully!")
print("📝 Summary:")
print(f"  - Processed {stats['total_documents']} documents")
print(f"  - Built vocabulary of {stats['vocabulary_size']} terms")
print(f"  - Tested {len(demo_questions)} sample questions")
print(f"  - Average confidence: {np.mean(confidences):.3f}")
print(f"  - Ready for production use!")
