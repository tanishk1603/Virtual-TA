#!/usr/bin/env python3
"""
TDS Virtual TA - A comprehensive implementation for the TDS Project 1
This script creates a virtual teaching assistant using the preprocessed data
from discourse posts and website pages.
"""

import json
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
from collections import defaultdict, Counter
import argparse
import logging
from datetime import datetime

# Text processing libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import PorterStemmer
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except ImportError:
    print("Please install required packages: pip install scikit-learn nltk pandas numpy")
    exit(1)

class TDSVirtualTA:
    """
    TDS Virtual Teaching Assistant
    
    This class processes discourse posts and website content to create
    a question-answering system for the TDS course.
    """
    
    def __init__(self, data_dir: str = ".", min_similarity: float = 0.3):
        """
        Initialize the Virtual TA
        
        Args:
            data_dir: Directory containing the data files
            min_similarity: Minimum similarity threshold for answers
        """
        self.data_dir = Path(data_dir)
        self.min_similarity = min_similarity
        self.documents = []
        self.document_metadata = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_discourse_data(self) -> List[Dict]:
        """Load discourse posts from JSON file"""
        discourse_file = self.data_dir / "discourse_posts.json"
        
        if not discourse_file.exists():
            self.logger.warning(f"Discourse file not found: {discourse_file}")
            return []
            
        try:
            with open(discourse_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            posts = []
            if isinstance(data, list):
                posts = data
            elif isinstance(data, dict):
                # Handle different JSON structures
                if 'posts' in data:
                    posts = data['posts']
                elif 'topic_posts' in data:
                    posts = data['topic_posts']
                else:
                    # Flatten nested structure
                    for key, value in data.items():
                        if isinstance(value, list):
                            posts.extend(value)
            
            self.logger.info(f"Loaded {len(posts)} discourse posts")
            return posts
            
        except Exception as e:
            self.logger.error(f"Error loading discourse data: {e}")
            return []
    
    def load_website_data(self) -> List[Dict]:
        """Load website markdown files"""
        md_dir = self.data_dir / "tds_pages_md"
        
        if not md_dir.exists():
            self.logger.warning(f"Markdown directory not found: {md_dir}")
            return []
        
        pages = []
        for md_file in md_dir.glob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                pages.append({
                    'title': md_file.stem,
                    'content': content,
                    'source': 'website',
                    'file_path': str(md_file)
                })
            except Exception as e:
                self.logger.error(f"Error loading {md_file}: {e}")
        
        self.logger.info(f"Loaded {len(pages)} website pages")
        return pages
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better matching
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove markdown formatting
        text = re.sub(r'[#*`_~\[\]()]', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def extract_qa_pairs(self, posts: List[Dict]) -> List[Dict]:
        """
        Extract question-answer pairs from discourse posts
        
        Args:
            posts: List of discourse posts
            
        Returns:
            List of QA pairs with metadata
        """
        qa_pairs = []
        
        for post in posts:
            try:
                # Extract post content
                content = ""
                if isinstance(post, dict):
                    content = post.get('content', post.get('raw', post.get('cooked', '')))
                    title = post.get('title', post.get('topic_title', ''))
                    post_id = post.get('id', post.get('post_number', ''))
                    username = post.get('username', post.get('display_username', 'Unknown'))
                    
                    # Combine title and content
                    full_content = f"{title} {content}".strip()
                    
                    if full_content:
                        qa_pairs.append({
                            'question': title if title else full_content[:100] + "...",
                            'answer': full_content,
                            'source': 'discourse',
                            'post_id': post_id,
                            'username': username,
                            'content_length': len(full_content)
                        })
                        
            except Exception as e:
                self.logger.error(f"Error processing post: {e}")
                continue
        
        return qa_pairs
    
    def build_knowledge_base(self):
        """Build the knowledge base from all available data"""
        self.logger.info("Building knowledge base...")
        
        # Load data
        discourse_posts = self.load_discourse_data()
        website_pages = self.load_website_data()
        
        # Process discourse posts
        qa_pairs = self.extract_qa_pairs(discourse_posts)
        
        # Combine all documents
        all_docs = []
        all_metadata = []
        
        # Add QA pairs
        for qa in qa_pairs:
            processed_text = self.preprocess_text(qa['answer'])
            if processed_text and len(processed_text) > 20:  # Filter very short content
                all_docs.append(processed_text)
                all_metadata.append({
                    'type': 'qa_pair',
                    'question': qa['question'],
                    'source': qa['source'],
                    'post_id': qa.get('post_id', ''),
                    'username': qa.get('username', ''),
                    'original_text': qa['answer']
                })
        
        # Add website pages
        for page in website_pages:
            processed_text = self.preprocess_text(page['content'])
            if processed_text and len(processed_text) > 50:
                all_docs.append(processed_text)
                all_metadata.append({
                    'type': 'website_page',
                    'title': page['title'],
                    'source': page['source'],
                    'file_path': page.get('file_path', ''),
                    'original_text': page['content']
                })
        
        self.documents = all_docs
        self.document_metadata = all_metadata
        
        self.logger.info(f"Knowledge base built with {len(self.documents)} documents")
        
        # Build TF-IDF vectors
        self.build_tfidf_index()
    
    def build_tfidf_index(self):
        """Build TF-IDF index for document retrieval"""
        if not self.documents:
            self.logger.error("No documents to index")
            return
        
        self.logger.info("Building TF-IDF index...")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Fit and transform documents
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        
        self.logger.info(f"TF-IDF index built with {self.tfidf_matrix.shape[1]} features")
    
    def find_similar_documents(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """
        Find documents similar to the query
        
        Args:
            query: User query
            top_k: Number of top similar documents to return
            
        Returns:
            List of (doc_index, similarity_score, metadata) tuples
        """
        if not self.vectorizer or self.tfidf_matrix is None:
            return []
        
        # Preprocess query
        processed_query = self.preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top-k similar documents
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] >= self.min_similarity:
                results.append((idx, similarities[idx], self.document_metadata[idx]))
        
        return results
    
    def generate_answer(self, query: str) -> Dict:
        """
        Generate answer for a given query
        
        Args:
            query: User question
            
        Returns:
            Dictionary containing answer and metadata
        """
        similar_docs = self.find_similar_documents(query, top_k=3)
        
        if not similar_docs:
            return {
                'answer': "I couldn't find a relevant answer to your question. Could you please rephrase or provide more specific details?",
                'confidence': 0.0,
                'sources': []
            }
        
        # Use the most similar document as the primary answer
        best_doc_idx, best_similarity, best_metadata = similar_docs[0]
        
        # Extract relevant portions
        original_text = best_metadata['original_text']
        answer_text = self.extract_relevant_portion(original_text, query)
        
        # Format answer
        if best_metadata['type'] == 'qa_pair':
            formatted_answer = f"Based on a similar discussion:\n\n{answer_text}"
        else:
            formatted_answer = f"From the course materials:\n\n{answer_text}"
        
        # Prepare sources
        sources = []
        for doc_idx, similarity, metadata in similar_docs:
            source_info = {
                'type': metadata['type'],
                'similarity': float(similarity),
            }
            
            if metadata['type'] == 'qa_pair':
                source_info.update({
                    'question': metadata.get('question', ''),
                    'username': metadata.get('username', ''),
                    'post_id': metadata.get('post_id', '')
                })
            else:
                source_info.update({
                    'title': metadata.get('title', ''),
                    'file_path': metadata.get('file_path', '')
                })
            
            sources.append(source_info)
        
        return {
            'answer': formatted_answer,
            'confidence': float(best_similarity),
            'sources': sources,
            'query': query
        }
    
    def extract_relevant_portion(self, text: str, query: str, max_length: int = 500) -> str:
        """
        Extract the most relevant portion of text for the query
        
        Args:
            text: Source text
            query: User query
            max_length: Maximum length of extracted text
            
        Returns:
            Relevant portion of text
        """
        sentences = sent_tokenize(text)
        query_words = set(word_tokenize(query.lower()))
        
        # Score sentences based on query word overlap
        sentence_scores = []
        for sentence in sentences:
            sentence_words = set(word_tokenize(sentence.lower()))
            overlap = len(query_words.intersection(sentence_words))
            sentence_scores.append((sentence, overlap))
        
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        result_text = ""
        for sentence, score in sentence_scores:
            if len(result_text) + len(sentence) <= max_length:
                result_text += sentence + " "
            else:
                break
        
        return result_text.strip() if result_text.strip() else text[:max_length]
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'documents': self.documents,
            'document_metadata': self.document_metadata,
            'vectorizer': self.vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'min_similarity': self.min_similarity
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.documents = model_data['documents']
        self.document_metadata = model_data['document_metadata']
        self.vectorizer = model_data['vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.min_similarity = model_data['min_similarity']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def evaluate_performance(self, test_queries: List[str]) -> Dict:
        """
        Evaluate the performance of the Virtual TA
        
        Args:
            test_queries: List of test questions
            
        Returns:
            Performance metrics
        """
        total_queries = len(test_queries)
        answered_queries = 0
        total_confidence = 0.0
        
        for query in test_queries:
            result = self.generate_answer(query)
            if result['confidence'] >= self.min_similarity:
                answered_queries += 1
                total_confidence += result['confidence']
        
        coverage = answered_queries / total_queries if total_queries > 0 else 0
        avg_confidence = total_confidence / answered_queries if answered_queries > 0 else 0
        
        return {
            'total_queries': total_queries,
            'answered_queries': answered_queries,
            'coverage': coverage,
            'average_confidence': avg_confidence
        }
    
    def get_statistics(self) -> Dict:
        """Get statistics about the knowledge base"""
        if not self.documents:
            return {'error': 'No knowledge base loaded'}
        
        # Document type distribution
        type_counts = defaultdict(int)
        for metadata in self.document_metadata:
            type_counts[metadata['type']] += 1
        
        # Content length statistics
        lengths = [len(doc) for doc in self.documents]
        
        return {
            'total_documents': len(self.documents),
            'document_types': dict(type_counts),
            'avg_document_length': np.mean(lengths),
            'median_document_length': np.median(lengths),
            'vocabulary_size': len(self.vectorizer.vocabulary_) if self.vectorizer else 0
        }

def interactive_mode(ta: TDSVirtualTA):
    """Run the Virtual TA in interactive mode"""
    print("TDS Virtual TA - Interactive Mode")
    print("Type 'quit' to exit, 'stats' for statistics, 'help' for help")
    print("-" * 50)
    
    while True:
        try:
            query = input("\nYour question: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'stats':
                stats = ta.get_statistics()
                print("\nKnowledge Base Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            elif query.lower() == 'help':
                print("\nAvailable commands:")
                print("  - Ask any question about TDS course content")
                print("  - 'stats' - Show knowledge base statistics")
                print("  - 'quit' - Exit the program")
                continue
            
            if not query:
                continue
            
            result = ta.generate_answer(query)
            
            print(f"\nAnswer (Confidence: {result['confidence']:.2f}):")
            print(result['answer'])
            
            if result['sources']:
                print(f"\nSources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'][:2], 1):
                    print(f"  {i}. {source['type']} (similarity: {source['similarity']:.2f})")
                    if source['type'] == 'qa_pair' and 'question' in source:
                        print(f"     Question: {source['question'][:100]}...")
                    elif source['type'] == 'website_page' and 'title' in source:
                        print(f"     Page: {source['title']}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='TDS Virtual Teaching Assistant')
    parser.add_argument('--data-dir', default='.', help='Directory containing data files')
    parser.add_argument('--build', action='store_true', help='Build knowledge base')
    parser.add_argument('--save-model', help='Save model to file')
    parser.add_argument('--load-model', help='Load model from file')
    parser.add_argument('--query', help='Single query mode')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--min-similarity', type=float, default=0.3, 
                       help='Minimum similarity threshold')
    
    args = parser.parse_args()
    
    # Initialize Virtual TA
    ta = TDSVirtualTA(data_dir=args.data_dir, min_similarity=args.min_similarity)
    
    # Load or build knowledge base
    if args.load_model:
        ta.load_model(args.load_model)
    else:
        ta.build_knowledge_base()
        
        if args.save_model:
            ta.save_model(args.save_model)
    
    # Handle different modes
    if args.query:
        result = ta.generate_answer(args.query)
        print(f"Query: {args.query}")
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']:.2f}")
        
    elif args.interactive:
        interactive_mode(ta)
        
    else:
        # Default: show statistics and sample queries
        stats = ta.get_statistics()
        print("TDS Virtual TA - Knowledge Base Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\nSample queries to try:")
        sample_queries = [
            "What is data science?",
            "How do I use pandas?",
            "What are the course requirements?",
            "How to handle missing data?",
            "What is machine learning?"
        ]
        
        for query in sample_queries:
            result = ta.generate_answer(query)
            if result['confidence'] > 0.3:
                print(f"\nQ: {query}")
                print(f"A: {result['answer'][:200]}...")
                print(f"Confidence: {result['confidence']:.2f}")

if __name__ == "__main__":
    main()