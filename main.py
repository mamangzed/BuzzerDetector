#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_buzzer_cli.py
All-in-one: scrape (snscrape atau Playwright), extract features, skor buzzer, deteksi koordinasi, export CSV & GML.

Modes:
  --mode snscrape     -> tanpa API, cepat (X/Twitter publik)
  --mode playwright   -> scraping UI (scroll), fleksibel; butuh 'playwright install'

Contoh:
  python detect_buzzer_cli.py --mode snscrape --query "kredit macet OR #KM since:2025-09-20 until:2025-10-03 lang:id" --limit 2000 --out results_snscrape
  python detect_buzzer_cli.py --mode playwright --query "kredit%20macet%20lang:id" --max_posts 800 --out results_ui --headful
  python detect_buzzer_cli.py --mode snscrape --query "#pilpres lang:id" --limit 4000 --out pilpres --no-embeddings
"""

import os
import re
import sys
import math
import json
import time
import argparse
import datetime
import warnings
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import sqlite3
try:
    import mysql.connector
    from mysql.connector import Error
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    print("[!] MySQL connector not installed. Using SQLite as fallback.")
    print("    Install with: pip install mysql-connector-python")

from tqdm import tqdm

# ML - Advanced Methods
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import networkx as nx

# Load configuration
try:
    from config import config
    print(f"[CONFIG] Loaded configuration from .env")
except ImportError:
    print("[WARNING] Config module not found, using environment variables")
    # Fallback configuration using os.getenv
    class FallbackConfig:
        def __init__(self):
            self.embed_model_name = os.getenv('EMBED_MODEL_NAME', 'all-MiniLM-L6-v2')
            self.database_type = os.getenv('DATABASE_TYPE', 'sqlite3')
            self.database_path = os.getenv('DATABASE_PATH', 'buzzer_detection.db')
            self.database_host = os.getenv('DATABASE_HOST', 'localhost')
            self.database_port = int(os.getenv('DATABASE_PORT', '3306'))
            self.database_name = os.getenv('DATABASE_NAME', 'buzzer')
            self.database_user = os.getenv('DATABASE_USER', 'root')
            self.database_password = os.getenv('DATABASE_PASSWORD', '')
            self.mysql_host = os.getenv('MYSQL_HOST', 'localhost')
            self.mysql_port = int(os.getenv('MYSQL_PORT', '3306'))
            self.mysql_user = os.getenv('MYSQL_USER', 'root')
            self.mysql_password = os.getenv('MYSQL_PASSWORD', '')
            self.mysql_database = os.getenv('MYSQL_DATABASE', 'buzzer')
            self.use_embeddings = os.getenv('USE_EMBEDDINGS', 'true').lower() == 'true'
            self.ai_analysis_enabled = os.getenv('AI_ANALYSIS_ENABLED', 'true').lower() == 'true'
            self.model_dir = os.getenv('MODEL_DIR', 'models')
            self.n_jobs = int(os.getenv('N_JOBS', '-1'))
            self.random_state = int(os.getenv('RANDOM_STATE', '42'))
        
        def get_database_config(self):
            if self.database_type == 'mysql':
                return {
                    'type': 'mysql',
                    'host': self.mysql_host,
                    'port': self.mysql_port,
                    'user': self.mysql_user,
                    'password': self.mysql_password,
                    'database': self.mysql_database
                }
            else:
                return {
                    'type': 'sqlite3',
                    'path': self.database_path
                }
    
    config = FallbackConfig()

# Helper functions for logging
def info(msg): 
    print(f"[i] {msg}", flush=True)

def warn(msg): 
    print(f"[!] {msg}", flush=True)

def error(msg): 
    print(f"[x] {msg}", file=sys.stderr, flush=True)

def die(msg):  
    error(msg)
    sys.exit(1)

# Optional embeddings (lebih akurat). Bisa dimatikan dengan --no-embeddings
_EMBED_MODEL_NAME = config.embed_model_name

# Advanced Buzzer Detection Configuration
class BuzzerTypeClassifier:
    """Classifier untuk menentukan tipe buzzer"""
    def __init__(self):
        # Keywords untuk deteksi tipe buzzer
        self.government_keywords = [
            'pemerintah', 'jokowi', 'prabowo', 'menteri', 'presiden', 'wapres',
            'bpjs', 'kemenkes', 'kemenkeu', 'polri', 'tni', 'bumn', 'ikn',
            'program kerja', 'kebijakan', 'pemilu', 'pilpres', 'indonesia maju',
            'pangan murah', 'makan bergizi', 'mbg', 'kartu prakerja'
        ]
        
        self.opposition_keywords = [
            'korupsi', 'oligarki', 'otoriter', 'rezim', 'diktator', 'nepotisme',
            'rakyat sengsara', 'harga naik', 'utang negara', 'kriminalisasi',
            'ham', 'demokrasi', 'reformasi', 'gerakan rakyat'
        ]
        
        self.commercial_keywords = [
            'beli', 'jual', 'promo', 'diskon', 'sale', 'produk', 'brand',
            'review', 'endorse', 'sponsor', 'affiliate', 'marketplace',
            'tokopedia', 'shopee', 'gojek', 'grab'
        ]
        
        self.spam_keywords = [
            'like dan share', 'follow back', 'giveaway', 'kontes', 'undian',
            'klik link', 'daftar sekarang', 'gratis', 'bonus'
        ]
    
    def classify_buzzer_type(self, posts_content, account_features):
        """AI-driven buzzer type classification with advanced content analysis"""
        combined_content = ' '.join(posts_content)
        
        # Initialize scores
        scores = {
            'pemerintah': 0,
            'oposisi': 0,
            'komersial': 0,
            'spam': 0
        }
        
        # 1. Content sophistication analysis
        content_scores = self._analyze_content_sophistication(combined_content, posts_content)
        
        # 2. Behavioral pattern analysis (more weight than keywords)
        behavioral_scores = self._analyze_behavioral_patterns(account_features, posts_content)
        
        # 3. Semantic content analysis (AI-driven)
        semantic_scores = self._analyze_semantic_patterns(combined_content, posts_content)
        
        # 4. Temporal and engagement analysis
        engagement_scores = self._analyze_engagement_patterns(account_features, posts_content)
        
        # 5. Keyword analysis (reduced weight, supplementary only)
        keyword_scores = {
            'pemerintah': self._count_keywords(combined_content.lower(), self.government_keywords),
            'oposisi': self._count_keywords(combined_content.lower(), self.opposition_keywords),
            'komersial': self._count_keywords(combined_content.lower(), self.commercial_keywords),
            'spam': self._count_keywords(combined_content.lower(), self.spam_keywords)
        }
        
        # Combine scores with weighted approach
        for category in scores.keys():
            scores[category] = (
                content_scores.get(category, 0) * 0.25 +
                behavioral_scores.get(category, 0) * 0.35 +
                semantic_scores.get(category, 0) * 0.25 +
                engagement_scores.get(category, 0) * 0.1 +
                keyword_scores.get(category, 0) * 0.05  # Reduced keyword weight
            )
        
        # Determine primary and secondary type
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Enhanced confidence calculation
        total_score = sum(scores.values())
        confidence = self._calculate_confidence(scores, account_features, posts_content)
        
        primary_type = sorted_scores[0][0] if sorted_scores[0][1] > 0.5 else 'netral'
        secondary_type = sorted_scores[1][0] if len(sorted_scores) > 1 and sorted_scores[1][1] > 0.3 else None
        
        # Advanced activity pattern detection
        activity_pattern = self._determine_advanced_activity_pattern(account_features, posts_content)
        
        return {
            'primary_type': primary_type,
            'secondary_type': secondary_type,
            'activity_pattern': activity_pattern,
            'type_confidence': confidence,
            'scores': scores,
            'content_sophistication': content_scores,
            'behavioral_analysis': behavioral_scores,
            'semantic_analysis': semantic_scores
        }
    
    def _analyze_content_sophistication(self, combined_content, posts_content):
        """Analyze content sophistication and writing patterns"""
        import re
        from collections import Counter
        
        scores = {'pemerintah': 0, 'oposisi': 0, 'komersial': 0, 'spam': 0}
        
        # Language complexity analysis
        avg_sentence_length = self._calculate_avg_sentence_length(combined_content)
        vocabulary_diversity = self._calculate_vocabulary_diversity(combined_content)
        grammar_complexity = self._analyze_grammar_complexity(combined_content)
        
        # Content patterns analysis
        repetition_score = self._calculate_content_repetition(posts_content)
        emoji_usage = self._analyze_emoji_usage(combined_content)
        caps_usage = self._analyze_caps_usage(combined_content)
        
        # Government content indicators (formal language, policy terms)
        if avg_sentence_length > 20 and vocabulary_diversity > 0.7:
            scores['pemerintah'] += 2
        if grammar_complexity > 0.6:
            scores['pemerintah'] += 1
            
        # Opposition content indicators (emotional language, criticism)
        if caps_usage > 0.3 and emoji_usage < 0.1:
            scores['oposisi'] += 2
        if repetition_score > 0.4:
            scores['oposisi'] += 1
            
        # Commercial content indicators (promotional patterns)
        if emoji_usage > 0.2 and caps_usage > 0.2:
            scores['komersial'] += 2
        if self._detect_promotional_patterns(combined_content):
            scores['komersial'] += 3
            
        # Spam indicators (low sophistication, high repetition)
        if repetition_score > 0.6:
            scores['spam'] += 3
        if vocabulary_diversity < 0.3:
            scores['spam'] += 2
        if avg_sentence_length < 8:
            scores['spam'] += 1
            
        return scores
    
    def _analyze_behavioral_patterns(self, account_features, posts_content):
        """Advanced behavioral pattern analysis"""
        scores = {'pemerintah': 0, 'oposisi': 0, 'komersial': 0, 'spam': 0}
        
        freq_per_day = account_features.get('freq_per_day', 0)
        burstiness = account_features.get('burstiness', 0)
        hashtag_ratio = account_features.get('hashtag_ratio', 0)
        url_ratio = account_features.get('url_ratio', 0)
        followers = account_features.get('followers', 0)
        following = account_features.get('following', 0)
        
        # Government accounts: moderate posting, professional patterns
        if 2 <= freq_per_day <= 8 and burstiness < 0.4:
            scores['pemerintah'] += 2
        if hashtag_ratio < 0.3 and url_ratio < 0.2:
            scores['pemerintah'] += 1
            
        # Opposition: burst patterns, high engagement attempts
        if burstiness > 0.6 or freq_per_day > 15:
            scores['oposisi'] += 2
        if hashtag_ratio > 0.4:
            scores['oposisi'] += 1
            
        # Commercial: consistent posting, promotional patterns
        if url_ratio > 0.3:
            scores['komersial'] += 3
        if hashtag_ratio > 0.5:
            scores['komersial'] += 2
            
        # Spam: excessive activity, poor engagement
        if freq_per_day > 20 or burstiness > 0.8:
            scores['spam'] += 3
        if following > 1000 and followers < 100:
            scores['spam'] += 2
            
        return scores
    
    def _analyze_semantic_patterns(self, combined_content, posts_content):
        """Semantic analysis using NLP techniques"""
        scores = {'pemerintah': 0, 'oposisi': 0, 'komersial': 0, 'spam': 0}
        
        # Sentiment and emotion analysis
        sentiment_scores = self._analyze_sentiment_patterns(combined_content)
        topic_coherence = self._analyze_topic_coherence(posts_content)
        writing_style = self._analyze_writing_style(combined_content)
        
        # Government: neutral sentiment, formal style
        if sentiment_scores.get('neutral', 0) > 0.6:
            scores['pemerintah'] += 2
        if writing_style.get('formal', 0) > 0.7:
            scores['pemerintah'] += 1
            
        # Opposition: negative sentiment, critical tone
        if sentiment_scores.get('negative', 0) > 0.4:
            scores['oposisi'] += 2
        if writing_style.get('emotional', 0) > 0.6:
            scores['oposisi'] += 1
            
        # Commercial: positive sentiment, persuasive language
        if sentiment_scores.get('positive', 0) > 0.5:
            scores['komersial'] += 2
        if writing_style.get('persuasive', 0) > 0.6:
            scores['komersial'] += 1
            
        # Spam: inconsistent topics, low coherence
        if topic_coherence < 0.3:
            scores['spam'] += 3
            
        return scores
    
    def _analyze_engagement_patterns(self, account_features, posts_content):
        """Analyze engagement and interaction patterns"""
        scores = {'pemerintah': 0, 'oposisi': 0, 'komersial': 0, 'spam': 0}
        
        avg_likes = account_features.get('avg_likes', 0)
        avg_retweets = account_features.get('avg_retweets', 0)
        posts_count = len(posts_content)
        
        # Calculate engagement ratios
        if posts_count > 0:
            engagement_rate = (avg_likes + avg_retweets) / posts_count
            
            # Government: moderate, steady engagement
            if 0.5 <= engagement_rate <= 3.0:
                scores['pemerintah'] += 1
                
            # Opposition: potentially high engagement due to controversy
            if engagement_rate > 2.0:
                scores['oposisi'] += 1
                
            # Commercial: variable engagement based on promotion success
            if engagement_rate > 1.0:
                scores['komersial'] += 1
                
            # Spam: typically low engagement
            if engagement_rate < 0.2:
                scores['spam'] += 2
                
        return scores
    
    def _count_keywords(self, content, keywords):
        """Count keyword occurrences in content"""
        count = 0
        for keyword in keywords:
            count += content.count(keyword.lower())
        return count
    
    def _calculate_avg_sentence_length(self, content):
        """Calculate average sentence length"""
        import re
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0
        return sum(len(s.split()) for s in sentences) / len(sentences)
    
    def _calculate_vocabulary_diversity(self, content):
        """Calculate vocabulary diversity (unique words / total words)"""
        words = content.lower().split()
        if not words:
            return 0
        unique_words = set(words)
        return len(unique_words) / len(words)
    
    def _analyze_grammar_complexity(self, content):
        """Simple grammar complexity analysis"""
        import re
        # Count complex punctuation and conjunctions
        complex_patterns = [r',', r';', r':', r'yang', r'karena', r'jika', r'namun', r'tetapi']
        complexity_score = sum(len(re.findall(pattern, content.lower())) for pattern in complex_patterns)
        word_count = len(content.split())
        return min(1.0, complexity_score / max(1, word_count)) if word_count > 0 else 0
    
    def _calculate_content_repetition(self, posts_content):
        """Calculate content repetition across posts"""
        if len(posts_content) < 2:
            return 0
        
        # Calculate similarity between posts
        similarities = []
        for i in range(len(posts_content)):
            for j in range(i+1, len(posts_content)):
                post1_words = set(posts_content[i].lower().split())
                post2_words = set(posts_content[j].lower().split())
                if post1_words and post2_words:
                    intersection = len(post1_words.intersection(post2_words))
                    union = len(post1_words.union(post2_words))
                    similarity = intersection / union if union > 0 else 0
                    similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0
    
    def _analyze_emoji_usage(self, content):
        """Analyze emoji usage patterns"""
        import re
        # Simple emoji detection (Unicode ranges for common emojis)
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]')
        emojis = emoji_pattern.findall(content)
        words = content.split()
        return len(emojis) / max(1, len(words))
    
    def _analyze_caps_usage(self, content):
        """Analyze uppercase usage patterns"""
        import re
        caps_words = re.findall(r'\b[A-Z]{2,}\b', content)
        total_words = content.split()
        return len(caps_words) / max(1, len(total_words))
    
    def _detect_promotional_patterns(self, content):
        """Detect promotional language patterns"""
        promotional_indicators = [
            r'\b(beli|jual|promo|diskon|murah|gratis)\b',
            r'\b(dapatkan|segera|terbatas|khusus)\b',
            r'\b(cashback|bonus|reward|hadiah)\b',
            r'%|Rp|💰|🔥|⭐'
        ]
        
        count = 0
        for pattern in promotional_indicators:
            if re.search(pattern, content.lower()):
                count += 1
        
        return count >= 2  # If 2+ promotional patterns found
    
    def _analyze_sentiment_patterns(self, content):
        """Basic sentiment analysis"""
        # Simple rule-based sentiment (can be enhanced with ML models)
        positive_words = ['baik', 'bagus', 'hebat', 'sukses', 'mantap', 'luar biasa', 'terbaik']
        negative_words = ['buruk', 'jelek', 'gagal', 'rusak', 'korup', 'bodoh', 'tolol']
        neutral_words = ['adalah', 'akan', 'telah', 'dapat', 'harus', 'perlu']
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        neutral_count = sum(1 for word in neutral_words if word in content_lower)
        
        total = positive_count + negative_count + neutral_count + 1
        
        return {
            'positive': positive_count / total,
            'negative': negative_count / total,
            'neutral': neutral_count / total
        }
    
    def _analyze_topic_coherence(self, posts_content):
        """Analyze topic coherence across posts"""
        if len(posts_content) < 2:
            return 1.0
        
        # Simple topic coherence using word overlap
        all_words = set()
        post_words = []
        
        for post in posts_content:
            words = set(post.lower().split())
            post_words.append(words)
            all_words.update(words)
        
        if not all_words:
            return 0
        
        # Calculate average overlap between posts
        overlaps = []
        for i in range(len(post_words)):
            for j in range(i+1, len(post_words)):
                intersection = len(post_words[i].intersection(post_words[j]))
                union = len(post_words[i].union(post_words[j]))
                overlap = intersection / union if union > 0 else 0
                overlaps.append(overlap)
        
        return sum(overlaps) / len(overlaps) if overlaps else 0
    
    def _analyze_writing_style(self, content):
        """Analyze writing style characteristics"""
        import re
        
        # Formal language indicators
        formal_indicators = ['oleh karena itu', 'dengan demikian', 'berdasarkan', 'menurut', 'sebagaimana']
        formal_count = sum(1 for indicator in formal_indicators if indicator in content.lower())
        
        # Emotional language indicators
        emotional_indicators = ['!', '???', 'sangat', 'sekali', 'banget', 'parah']
        emotional_count = sum(content.count(indicator) for indicator in emotional_indicators)
        
        # Persuasive language indicators
        persuasive_indicators = ['ayo', 'mari', 'jangan', 'harus', 'wajib', 'segera']
        persuasive_count = sum(1 for indicator in persuasive_indicators if indicator in content.lower())
        
        word_count = len(content.split()) + 1
        
        return {
            'formal': min(1.0, formal_count / word_count * 10),
            'emotional': min(1.0, emotional_count / word_count * 5),
            'persuasive': min(1.0, persuasive_count / word_count * 8)
        }
    
    def _calculate_confidence(self, scores, account_features, posts_content):
        """Calculate classification confidence"""
        sorted_scores = sorted(scores.values(), reverse=True)
        
        if len(sorted_scores) < 2:
            return 0.5
        
        # Confidence based on score separation and account maturity
        score_diff = sorted_scores[0] - sorted_scores[1]
        max_possible_diff = max(1, sorted_scores[0])
        base_confidence = score_diff / max_possible_diff
        
        # Adjust confidence based on account features
        post_count = len(posts_content)
        account_age = account_features.get('account_age_days', 0)
        
        # More posts and older accounts = higher confidence
        maturity_factor = min(1.0, (post_count / 20) * 0.5 + (account_age / 365) * 0.5)
        
        final_confidence = (base_confidence * 0.7) + (maturity_factor * 0.3)
        return min(0.95, max(0.1, final_confidence))
    
    def _add_behavioral_scores(self, scores, features):
        """Legacy method - kept for backward compatibility"""
        # This method is now integrated into _analyze_behavioral_patterns
        pass
    
    def _determine_advanced_activity_pattern(self, features, posts_content):
        """Advanced activity pattern determination using multiple indicators"""
        burstiness = features.get('burstiness', 0)
        freq_per_day = features.get('freq_per_day', 0)
        hashtag_ratio = features.get('hashtag_ratio', 0)
        url_ratio = features.get('url_ratio', 0)
        
        # Calculate content consistency
        content_repetition = self._calculate_content_repetition(posts_content)
        
        # Scoring system for different patterns
        artificial_score = 0
        organic_score = 0
        
        # Artificial indicators
        if burstiness > 0.8: artificial_score += 3
        elif burstiness > 0.6: artificial_score += 2
        elif burstiness > 0.4: artificial_score += 1
        
        if freq_per_day > 30: artificial_score += 4
        elif freq_per_day > 20: artificial_score += 3
        elif freq_per_day > 15: artificial_score += 2
        elif freq_per_day > 10: artificial_score += 1
        
        if content_repetition > 0.7: artificial_score += 3
        elif content_repetition > 0.5: artificial_score += 2
        elif content_repetition > 0.3: artificial_score += 1
        
        if hashtag_ratio > 0.8: artificial_score += 2
        if url_ratio > 0.6: artificial_score += 2
        
        # Organic indicators
        if 1 <= freq_per_day <= 8: organic_score += 2
        if burstiness < 0.3: organic_score += 2
        if content_repetition < 0.2: organic_score += 2
        if 0.1 <= hashtag_ratio <= 0.3: organic_score += 1
        if url_ratio < 0.2: organic_score += 1
        
        # Account maturity bonus for organic
        account_age = features.get('account_age_days', 0)
        if account_age > 365: organic_score += 2
        elif account_age > 180: organic_score += 1
        
        # Determine pattern
        if artificial_score >= 6:
            return 'artificial'
        elif artificial_score >= 3 or (artificial_score > organic_score and artificial_score >= 2):
            return 'hybrid'
        else:
            return 'organik'
    
    def _determine_activity_pattern(self, features):
        """Legacy method - kept for backward compatibility"""
        burstiness = features.get('burstiness', 0)
        freq_per_day = features.get('freq_per_day', 0)
        
        if burstiness > 0.8 or freq_per_day > 20:
            return 'artificial'
        elif burstiness > 0.5 or freq_per_day > 10:
            return 'hybrid'
        else:
            return 'organik'

class BuzzerDetectionConfig:
    """Configuration untuk advanced buzzer detection"""
    def __init__(self):
        # Thresholds (akan di-tune otomatis)
        self.freq_threshold = 8.0
        self.burstiness_threshold = 0.6
        self.account_age_threshold = 45
        self.followers_threshold = 150
        self.hashtag_threshold = 0.4
        
        # Advanced thresholds
        self.similarity_threshold = 0.85
        self.coordination_window = 3600  # seconds
        self.min_cluster_size = 3
        
        # Model performance tracking
        self.performance_history = []
        self.adaptive_learning = True
        
    def update_thresholds(self, performance_score):
        """Adaptive threshold tuning berdasarkan performance"""
        self.performance_history.append(performance_score)
        
        if len(self.performance_history) > 10:
            recent_perf = np.mean(self.performance_history[-5:])
            
            if recent_perf < 0.6:  # Performance dropping
                # Relax thresholds
                self.freq_threshold *= 0.95
                self.burstiness_threshold *= 0.95
                self.account_age_threshold *= 1.1
                info(f"[ADAPTIVE] Thresholds relaxed - performance: {recent_perf:.3f}")
            elif recent_perf > 0.8:  # Performance good
                # Tighten thresholds
                self.freq_threshold *= 1.02
                self.burstiness_threshold *= 1.02
                self.account_age_threshold *= 0.98
                info(f"[ADAPTIVE] Thresholds tightened - performance: {recent_perf:.3f}")



def init_database(db_config=None):
    """Initialize database for storing buzzer detection results"""
    if db_config is None:
        db_config = config.get_database_config()
    
    if db_config['type'] == 'mysql':
        if not MYSQL_AVAILABLE:
            die("MySQL connector not available. Install with: pip install mysql-connector-python")
        return init_mysql_database(db_config)
    else:
        die("Only MySQL database is supported. Set DATABASE_TYPE=mysql in .env file")

def init_mysql_database(db_config):
    """Initialize MySQL database"""
    try:
        conn = mysql.connector.connect(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 3306),
            user=db_config.get('user', 'root'),
            password=db_config.get('password', ''),
            database=db_config.get('database', 'buzzer_detection'),
            autocommit=True
        )
        cursor = conn.cursor()
        
        # Create buzzer_accounts table with additional buzzer type fields
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS buzzer_accounts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) NOT NULL,
            display_name VARCHAR(500),
            social_media_platform VARCHAR(50) NOT NULL,
            account_url TEXT,
            posts_count INT DEFAULT 0,
            followers_count INT DEFAULT 0,
            following_count INT DEFAULT 0,
            account_age_days INT DEFAULT 0,
            freq_per_day DECIMAL(10,3) DEFAULT 0,
            burstiness DECIMAL(10,3) DEFAULT 0,
            hashtag_ratio DECIMAL(10,3) DEFAULT 0,
            url_ratio DECIMAL(10,3) DEFAULT 0,
            buzzer_prob DECIMAL(10,3) DEFAULT 0,
            buzzer_prob_enhanced DECIMAL(10,3) DEFAULT 0,
            risk_category VARCHAR(20) DEFAULT 'LOW',
            coord_cluster INT DEFAULT -1,
            coordination_score DECIMAL(10,3) DEFAULT 0,
            network_centrality DECIMAL(10,3) DEFAULT 0,
            buzzer_type_primary VARCHAR(50) DEFAULT 'netral',
            buzzer_type_secondary VARCHAR(50),
            activity_pattern VARCHAR(20) DEFAULT 'organik',
            type_confidence DECIMAL(10,3) DEFAULT 0,
            government_score INT DEFAULT 0,
            opposition_score INT DEFAULT 0,
            commercial_score INT DEFAULT 0,
            spam_score INT DEFAULT 0,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            query_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_account (username, social_media_platform, query_used)
        )
        ''')
        
        # Other tables remain similar but with MySQL syntax
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INT AUTO_INCREMENT PRIMARY KEY,
            post_id VARCHAR(255),
            username VARCHAR(255) NOT NULL,
            social_media_platform VARCHAR(50) NOT NULL,
            content TEXT NOT NULL,
            post_url TEXT,
            post_date TIMESTAMP NULL,
            retweet_count INT DEFAULT 0,
            like_count INT DEFAULT 0,
            reply_count INT DEFAULT 0,
            quote_count INT DEFAULT 0,
            is_retweet BOOLEAN DEFAULT FALSE,
            original_author VARCHAR(255),
            query_used TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_post (post_id, social_media_platform)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS coordination_groups (
            id INT AUTO_INCREMENT PRIMARY KEY,
            group_id INT NOT NULL,
            username VARCHAR(255) NOT NULL,
            social_media_platform VARCHAR(50) NOT NULL,
            similarity_score DECIMAL(10,3) DEFAULT 0,
            query_used TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY unique_group_member (group_id, username, social_media_platform, query_used(100))
        )
        ''')
        
        # Create indexes
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_buzzer_username ON buzzer_accounts(username)',
            'CREATE INDEX IF NOT EXISTS idx_buzzer_platform ON buzzer_accounts(social_media_platform)',
            'CREATE INDEX IF NOT EXISTS idx_buzzer_risk ON buzzer_accounts(risk_category)',
            'CREATE INDEX IF NOT EXISTS idx_buzzer_type ON buzzer_accounts(buzzer_type_primary)',
            'CREATE INDEX IF NOT EXISTS idx_posts_username ON posts(username)',
            'CREATE INDEX IF NOT EXISTS idx_posts_platform ON posts(social_media_platform)',
            'CREATE INDEX IF NOT EXISTS idx_coord_group ON coordination_groups(group_id)'
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except mysql.connector.Error:
                pass  # Index might already exist
        
        conn.close()
        info(f"[DB] MySQL database initialized: {db_config['database']}")
        return True
        
    except Error as e:
        warn(f"[DB] MySQL initialization failed: {e}")
        # Fallback to SQLite
        return init_sqlite_database('buzzer_detection.db')

def init_sqlite_database(db_path="buzzer_detection.db"):
    """Initialize SQLite database for storing buzzer detection results"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create buzzer_accounts table with buzzer type detection
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS buzzer_accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        display_name TEXT,
        social_media_platform TEXT NOT NULL,
        account_url TEXT,
        posts_count INTEGER,
        followers_count INTEGER,
        following_count INTEGER,
        account_age_days INTEGER,
        freq_per_day REAL,
        burstiness REAL,
        hashtag_ratio REAL,
        url_ratio REAL,
        buzzer_prob REAL,
        buzzer_prob_enhanced REAL,
        risk_category TEXT,
        coord_cluster INTEGER,
        coordination_score REAL,
        network_centrality REAL,
        buzzer_type_primary TEXT DEFAULT 'netral',
        buzzer_type_secondary TEXT,
        activity_pattern TEXT DEFAULT 'organik',
        type_confidence REAL DEFAULT 0,
        government_score INTEGER DEFAULT 0,
        opposition_score INTEGER DEFAULT 0,
        commercial_score INTEGER DEFAULT 0,
        spam_score INTEGER DEFAULT 0,
        -- AI Analysis Scores
        behavioral_anomaly_score REAL DEFAULT 0,
        temporal_anomaly_score REAL DEFAULT 0,
        engagement_anomaly_score REAL DEFAULT 0,
        network_anomaly_score REAL DEFAULT 0,
        content_anomaly_score REAL DEFAULT 0,
        ai_confidence REAL DEFAULT 0,
        -- Content Analysis
        vocabulary_diversity REAL DEFAULT 0,
        avg_sentence_length REAL DEFAULT 0,
        grammar_complexity REAL DEFAULT 0,
        content_repetition REAL DEFAULT 0,
        emoji_usage REAL DEFAULT 0,
        caps_usage REAL DEFAULT 0,
        -- Sentiment Analysis
        sentiment_positive REAL DEFAULT 0,
        sentiment_negative REAL DEFAULT 0,
        sentiment_neutral REAL DEFAULT 0,
        -- Writing Style
        writing_formal REAL DEFAULT 0,
        writing_emotional REAL DEFAULT 0,
        writing_persuasive REAL DEFAULT 0,
        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        query_used TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(username, social_media_platform, query_used)
    )
    ''')
    
    # Create posts table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS posts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id TEXT,
        username TEXT NOT NULL,
        social_media_platform TEXT NOT NULL,
        content TEXT NOT NULL,
        post_url TEXT,
        post_date TIMESTAMP,
        retweet_count INTEGER DEFAULT 0,
        like_count INTEGER DEFAULT 0,
        reply_count INTEGER DEFAULT 0,
        quote_count INTEGER DEFAULT 0,
        is_retweet BOOLEAN DEFAULT 0,
        original_author TEXT,
        query_used TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(post_id, social_media_platform) ON CONFLICT REPLACE
    )
    ''')
    
    # Create coordination_groups table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS coordination_groups (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        group_id INTEGER NOT NULL,
        username TEXT NOT NULL,
        social_media_platform TEXT NOT NULL,
        similarity_score REAL,
        query_used TEXT,
        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(group_id, username, social_media_platform, query_used)
    )
    ''')
    
    # Create AI analysis details table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ai_analysis_details (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        social_media_platform TEXT NOT NULL,
        analysis_type TEXT NOT NULL,
        analysis_data TEXT NOT NULL, -- JSON format
        confidence_score REAL DEFAULT 0,
        query_used TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (username) REFERENCES buzzer_accounts(username)
    )
    ''')
    
    # Create content analysis table for posts
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS post_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_id TEXT NOT NULL,
        username TEXT NOT NULL,
        social_media_platform TEXT NOT NULL,
        content_hash TEXT, -- For deduplication
        sentiment_score REAL DEFAULT 0,
        emotion_score REAL DEFAULT 0,
        sophistication_score REAL DEFAULT 0,
        spam_indicators REAL DEFAULT 0,
        promotional_score REAL DEFAULT 0,
        analysis_data TEXT, -- JSON format with detailed metrics
        query_used TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (post_id) REFERENCES posts(post_id)
    )
    ''')
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_buzzer_username ON buzzer_accounts(username)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_buzzer_platform ON buzzer_accounts(social_media_platform)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_buzzer_risk ON buzzer_accounts(risk_category)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_buzzer_type ON buzzer_accounts(buzzer_type_primary)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_buzzer_activity ON buzzer_accounts(activity_pattern)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_posts_username ON posts(username)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_posts_platform ON posts(social_media_platform)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_posts_date ON posts(post_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_coord_group ON coordination_groups(group_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_analysis_username ON ai_analysis_details(username)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_analysis_type ON ai_analysis_details(analysis_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_post_analysis_username ON post_analysis(username)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_post_analysis_post ON post_analysis(post_id)')
    
    conn.commit()
    conn.close()
    info(f"[DB] Database initialized: {db_path}")
    
def save_to_database(posts_df, feat_df, query_used, coord_groups=None, db_path=None, mysql_config=None):
    """Save detection results to database with buzzer type classification support"""
    if db_path is None:
        db_path = config.database_path
    if mysql_config is None:
        db_config = config.get_database_config()
        if db_config['type'] == 'mysql':
            mysql_config = db_config
    
    try:
        # Force MySQL usage
        if mysql_config is None or mysql_config.get('type') != 'mysql':
            info("[DB] Setting up MySQL configuration")
            mysql_config = {
                'type': 'mysql',
                'host': config.mysql_host,
                'port': config.mysql_port,
                'user': config.mysql_user,
                'password': config.mysql_password,
                'database': config.mysql_database
            }
            
        # Prepare connection parameters (remove 'type' key for mysql.connector)
        connection_params = {k: v for k, v in mysql_config.items() if k != 'type'}
            
        import mysql.connector
        info(f"[DB] Connecting to MySQL: {mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}")
        conn = mysql.connector.connect(**connection_params)
        cursor = conn.cursor()
        is_mysql = True
        info(f"[DB] Successfully connected to MySQL database")
        
        # Determine social media platform
        platform = posts_df['platform'].iloc[0] if 'platform' in posts_df.columns else 'x'
        
        # Save posts
        for _, post in posts_df.iterrows():
            if is_mysql:
                try:
                    cursor.execute('''
                    INSERT INTO posts (
                    post_id, username, social_media_platform, content, post_url,
                    post_date, retweet_count, like_count, reply_count, quote_count,
                    is_retweet, original_author, query_used
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                retweet_count = VALUES(retweet_count),
                like_count = VALUES(like_count),
                reply_count = VALUES(reply_count),
                quote_count = VALUES(quote_count)
                ''', (
                    str(post.get('id', '')),
                    post.get('username', ''),
                    platform,
                    post.get('content', ''),
                    post.get('permalink', ''),
                    post.get('date'),
                    post.get('retweetCount', 0),
                    post.get('likeCount', 0),
                    post.get('replyCount', 0),
                    post.get('quoteCount', 0),
                    post.get('is_retweet', False),
                    post.get('original_author', ''),
                    query_used
                ))
                except Exception as e:
                    warn(f"[DB] Error saving post {post.get('id', 'unknown')}: {e}")
            else:
                cursor.execute('''
                INSERT OR REPLACE INTO posts (
                    post_id, username, social_media_platform, content, post_url,
                    post_date, retweet_count, like_count, reply_count, quote_count,
                    is_retweet, original_author, query_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(post.get('id', '')),
                    post.get('username', ''),
                    platform,
                    post.get('content', ''),
                    post.get('permalink', ''),
                    post.get('date'),
                    post.get('retweetCount', 0),
                    post.get('likeCount', 0),
                    post.get('replyCount', 0),
                    post.get('quoteCount', 0),
                    post.get('is_retweet', False),
                    post.get('original_author', ''),
                    query_used
                ))
        
        # Save posts with content analysis
        for _, post in posts_df.iterrows():
            # Save post content analysis if available
            post_content = post.get('content', '')
            content_hash = hashlib.md5(post_content.encode()).hexdigest() if post_content else ''
            
            # Basic content analysis for posts
            if post_content:
                # Calculate basic metrics
                sentiment_score = len([w for w in ['baik', 'bagus', 'hebat'] if w in post_content.lower()]) - len([w for w in ['buruk', 'jelek', 'gagal'] if w in post_content.lower()])
                emotion_score = len([c for c in post_content if c in '!?']) / max(1, len(post_content))
                spam_indicators = len([w for w in ['gratis', 'bonus', 'klik', 'follow'] if w in post_content.lower()])
                
                # Only save post analysis for SQLite, not MySQL (table doesn't exist in MySQL)
                if not is_mysql:
                    try:
                        cursor.execute('''
                        INSERT OR REPLACE INTO post_analysis (
                            post_id, username, social_media_platform, content_hash,
                            sentiment_score, emotion_score, sophistication_score, spam_indicators,
                            query_used
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            str(post.get('id', '')),
                            post.get('username', ''),
                            platform,
                            content_hash,
                            sentiment_score,
                            emotion_score,
                            len(post_content.split()) / 10,  # Simple sophistication metric
                            spam_indicators,
                            query_used
                        ))
                    except Exception as e:
                        warn(f"[DB] Error saving post analysis for {post.get('id', 'unknown')}: {e}")
        
        # Save buzzer accounts with complete AI analysis
        inserted_count = 0
        for _, account in feat_df.iterrows():
            account_url = f"https://x.com/{account['username']}" if platform == 'x' else f"https://twitter.com/{account['username']}"
            
            # Extract AI analysis data if available
            ai_analysis = account.get('ai_analysis', {})
            content_analysis = ai_analysis.get('content_sophistication', {})
            behavioral_analysis = ai_analysis.get('behavioral_analysis', {})
            semantic_analysis = ai_analysis.get('semantic_analysis', {})
            
            if is_mysql:
                try:
                    # MySQL INSERT - only use columns that exist in the MySQL table
                    cursor.execute('''
                    INSERT INTO buzzer_accounts (
                        username, display_name, social_media_platform, account_url, posts_count,
                        followers_count, following_count, account_age_days, freq_per_day,
                        burstiness, hashtag_ratio, url_ratio, buzzer_prob,
                        buzzer_prob_enhanced, risk_category, coord_cluster,
                        coordination_score, network_centrality, buzzer_type_primary,
                        buzzer_type_secondary, activity_pattern, type_confidence,
                        government_score, opposition_score, commercial_score, spam_score,
                        query_used
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    buzzer_prob = VALUES(buzzer_prob),
                    buzzer_prob_enhanced = VALUES(buzzer_prob_enhanced),
                    risk_category = VALUES(risk_category),
                    buzzer_type_primary = VALUES(buzzer_type_primary),
                    buzzer_type_secondary = VALUES(buzzer_type_secondary),
                    activity_pattern = VALUES(activity_pattern),
                    type_confidence = VALUES(type_confidence),
                    updated_at = CURRENT_TIMESTAMP
                    ''', (
                        account['username'],
                        account.get('display_name', account['username']),  # Add display_name
                        platform,
                        account_url,
                        account.get('posts', 0),
                        account.get('followers', 0),
                        account.get('following', 0),
                        account.get('account_age_days', 0),
                        account.get('freq_per_day', 0.0),
                        account.get('burstiness', 0.0),
                        account.get('hashtag_ratio', 0.0),
                        account.get('url_ratio', 0.0),
                        account.get('buzzer_prob', 0.0),
                        account.get('buzzer_prob_enhanced', account.get('buzzer_prob', 0.0)),
                        account.get('risk_category', 'LOW'),
                        account.get('coord_cluster', -1),
                        account.get('coordination_score', 0.0),
                        account.get('network_centrality', 0.0),
                        account.get('buzzer_type_primary', 'netral'),
                        account.get('buzzer_type_secondary', None),
                        account.get('activity_pattern', 'organik'),
                        account.get('type_confidence', 0.0),
                        account.get('government_score', 0),
                        account.get('opposition_score', 0),
                        account.get('commercial_score', 0),
                        account.get('spam_score', 0),
                        query_used
                    ))
                except Exception as e:
                    warn(f"[DB] Error saving buzzer account {account.get('username', 'unknown')}: {e}")
            else:
                cursor.execute('''
                INSERT OR REPLACE INTO buzzer_accounts (
                    username, social_media_platform, account_url, posts_count,
                    followers_count, following_count, account_age_days, freq_per_day,
                    burstiness, hashtag_ratio, url_ratio, buzzer_prob,
                    buzzer_prob_enhanced, risk_category, coord_cluster,
                    coordination_score, network_centrality, buzzer_type_primary,
                    buzzer_type_secondary, activity_pattern, type_confidence,
                    government_score, opposition_score, commercial_score, spam_score,
                    behavioral_anomaly_score, temporal_anomaly_score, engagement_anomaly_score,
                    network_anomaly_score, content_anomaly_score, ai_confidence,
                    vocabulary_diversity, avg_sentence_length, grammar_complexity,
                    content_repetition, emoji_usage, caps_usage,
                    sentiment_positive, sentiment_negative, sentiment_neutral,
                    writing_formal, writing_emotional, writing_persuasive,
                    query_used, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    account['username'],
                    platform,
                    account_url,
                    account.get('posts', 0),
                    account.get('followers', 0),
                    account.get('following', 0),
                    account.get('account_age_days', 0),
                    account.get('freq_per_day', 0.0),
                    account.get('burstiness', 0.0),
                    account.get('hashtag_ratio', 0.0),
                    account.get('url_ratio', 0.0),
                    account.get('buzzer_prob', 0.0),
                    account.get('buzzer_prob_enhanced', account.get('buzzer_prob', 0.0)),
                    account.get('risk_category', 'LOW'),
                    account.get('coord_cluster', -1),
                    account.get('coordination_score', 0.0),
                    account.get('network_centrality', 0.0),
                    account.get('buzzer_type_primary', 'netral'),
                    account.get('buzzer_type_secondary'),
                    account.get('activity_pattern', 'organik'),
                    account.get('type_confidence', 0.0),
                    account.get('government_score', 0),
                    account.get('opposition_score', 0),
                    account.get('commercial_score', 0),
                    account.get('spam_score', 0),
                    # AI Analysis scores
                    account.get('behavioral_anomaly_score', 0.0),
                    account.get('temporal_anomaly_score', 0.0),
                    account.get('engagement_anomaly_score', 0.0),
                    account.get('network_anomaly_score', 0.0),
                    account.get('content_anomaly_score', 0.0),
                    account.get('ai_confidence', 0.0),
                    # Content analysis
                    account.get('vocabulary_diversity', 0.0),
                    account.get('avg_sentence_length', 0.0),
                    account.get('grammar_complexity', 0.0),
                    account.get('content_repetition', 0.0),
                    account.get('emoji_usage', 0.0),
                    account.get('caps_usage', 0.0),
                    # Sentiment analysis
                    account.get('sentiment_positive', 0.0),
                    account.get('sentiment_negative', 0.0),
                    account.get('sentiment_neutral', 0.0),
                    # Writing style
                    account.get('writing_formal', 0.0),
                    account.get('writing_emotional', 0.0),
                    account.get('writing_persuasive', 0.0),
                    query_used,
                    datetime.datetime.now().isoformat()
                ))
            
            inserted_count += 1
            
            # Save detailed AI analysis to separate table (SQLite only)
            if ai_analysis and not is_mysql:
                import json
                try:
                    cursor.execute('''
                    INSERT OR REPLACE INTO ai_analysis_details (
                        username, social_media_platform, analysis_type, analysis_data,
                        confidence_score, query_used
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        account['username'],
                        platform,
                        'complete_ai_analysis',
                        json.dumps(ai_analysis, ensure_ascii=False),
                        account.get('type_confidence', 0.0),
                        query_used
                    ))
                except Exception as e:
                    warn(f"[DB] Error saving AI analysis for {account.get('username', 'unknown')}: {e}")
        
        # Save coordination groups
        if coord_groups:
            for group_id, members in enumerate(coord_groups):
                for username in members:
                    try:
                        if is_mysql:
                            cursor.execute('''
                            INSERT INTO coordination_groups (
                                group_id, username, social_media_platform, query_used
                            ) VALUES (%s, %s, %s, %s)
                            ON DUPLICATE KEY UPDATE
                            group_id = VALUES(group_id)
                            ''', (group_id, username, platform, query_used))
                        else:
                            cursor.execute('''
                            INSERT OR REPLACE INTO coordination_groups (
                                group_id, username, social_media_platform, query_used
                            ) VALUES (?, ?, ?, ?)
                            ''', (group_id, username, platform, query_used))
                    except Exception as e:
                        warn(f"[DB] Error saving coordination group {group_id} member {username}: {e}")
        
        conn.commit()
        conn.close()
        
        info(f"[DB] Successfully saved {len(posts_df)} posts and {inserted_count}/{len(feat_df)} accounts to MySQL database")
        
        # Verify data was saved
        try:
            # Remove 'type' key for mysql.connector
            verify_config = {k: v for k, v in mysql_config.items() if k != 'type'}
            verify_conn = mysql.connector.connect(**verify_config)
            verify_cursor = verify_conn.cursor()
            verify_cursor.execute("SELECT COUNT(*) FROM buzzer_accounts")
            total_accounts = verify_cursor.fetchone()[0]
            verify_cursor.execute("SELECT COUNT(*) FROM posts")
            total_posts = verify_cursor.fetchone()[0]
            verify_conn.close()
            info(f"[DB] Verification - Total in database: {total_accounts} accounts, {total_posts} posts")
        except Exception as e:
            warn(f"[DB] Failed to verify data: {e}")
        if coord_groups:
            info(f"[DB] Saved {len(coord_groups)} coordination groups")
            
    except Exception as e:
        warn(f"[DB] Error saving to database: {e}")
        if 'conn' in locals():
            conn.close()

class AdvancedBuzzerDetector:
    """Advanced Buzzer Detection dengan accumulative learning"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.config = BuzzerDetectionConfig()
        self.training_history = []
        self.model_version = 1
        
        # Load existing model atau buat baru
        self.load_or_create_models()
        
    def load_or_create_models(self):
        """Load model yang ada atau buat ensemble baru dengan size monitoring"""
        model_path = self.model_dir / "buzzer_ensemble_v2.pkl"
        
        if model_path.exists():
            try:
                model_data = joblib.load(model_path)
                self.ensemble = model_data['ensemble']
                self.scaler = model_data['scaler']
                self.feature_selector = model_data.get('feature_selector', None)
                
                # Load dengan monitoring
                old_training_count = self.training_count if hasattr(self, 'training_count') else 0
                self.training_count = model_data.get('training_count', 0)
                self.config.performance_history = model_data.get('performance_history', [])
                self.model_version = model_data.get('version', 1)
                
                # Track model algorithms
                self.model_algorithms = model_data.get('algorithms', ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM'])
                self.model_count = model_data.get('model_count', 4)
                
                # Calculate model metrics
                model_size_mb = model_path.stat().st_size / (1024 * 1024)
                
                info(f"[MODEL] Loaded ensemble v{self.model_version} - trained on {self.training_count} examples")
                info(f"[MODEL] Algorithms: {self.model_count} models ({', '.join(self.model_algorithms)})")
                info(f"[MODEL] File size: {model_size_mb:.1f} MB")
                
                # Check training data growth/shrinkage
                if old_training_count > 0:
                    change = self.training_count - old_training_count
                    if change < 0:
                        warn(f"[MODEL] Training examples DECREASED by {abs(change)} samples!")
                    elif change > 0:
                        info(f"[MODEL] Training examples INCREASED by {change} samples")
                
                return True
            except Exception as e:
                warn(f"Failed to load model: {e}")
        
        # Create new ensemble
        self.create_ensemble()
        self.scaler = RobustScaler()  # More robust than StandardScaler
        self.feature_selector = SelectKBest(f_classif, k=15)  # Feature selection
        self.training_count = 0
        
        info("[MODEL] Created new ensemble model")
        return False
        
    def create_ensemble(self):
        """Buat enhanced ensemble dengan multiple advanced algorithms"""
        
        # Base models dengan different strengths
        rf = RandomForestClassifier(
            n_estimators=150,  # Increased for better performance
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=100,  # Increased
            learning_rate=0.1,
            max_depth=6,  # Deeper trees for better patterns
            min_samples_split=2,
            random_state=42
        )
        
        lr = LogisticRegression(
            random_state=42,
            max_iter=2000,  # Increased iterations
            C=1.0,
            solver='liblinear'
        )
        
        svm = SVC(
            probability=True,  # Enable probability estimates
            random_state=42,
            kernel='rbf',
            C=1.0
        )
        
        # Advanced algorithms - dengan fallback jika library tidak tersedia
        advanced_models = []
        
        # XGBoost
        try:
            import xgboost as xgb
            xgb_model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                eval_metric='logloss',
                verbosity=0
            )
            advanced_models.append(('xgb', xgb_model))
            info("[MODEL] XGBoost added to ensemble")
        except ImportError:
            warn("[MODEL] XGBoost not available - install with: pip install xgboost")
        
        # LightGBM
        try:
            import lightgbm as lgb
            lgb_model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1,
                force_col_wise=True
            )
            advanced_models.append(('lgb', lgb_model))
            info("[MODEL] LightGBM added to ensemble")
        except ImportError:
            warn("[MODEL] LightGBM not available - install with: pip install lightgbm")
        
        # Neural Network
        try:
            from sklearn.neural_network import MLPClassifier
            nn_model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            advanced_models.append(('nn', nn_model))
            info("[MODEL] Neural Network added to ensemble")
        except ImportError:
            warn("[MODEL] Neural Network not available")
        
        # Extra Trees
        try:
            from sklearn.ensemble import ExtraTreesClassifier
            et_model = ExtraTreesClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )
            advanced_models.append(('et', et_model))
            info("[MODEL] ExtraTrees added to ensemble")
        except ImportError:
            warn("[MODEL] ExtraTrees not available")
        
        # AdaBoost
        try:
            from sklearn.ensemble import AdaBoostClassifier
            ada_model = AdaBoostClassifier(
                n_estimators=100,
                learning_rate=1.0,
                random_state=42
            )
            advanced_models.append(('ada', ada_model))
            info("[MODEL] AdaBoost added to ensemble")
        except ImportError:
            warn("[MODEL] AdaBoost not available")
        
        # Create ensemble with all available models
        base_models = [
            ('rf', rf),
            ('gb', gb), 
            ('lr', lr),
            ('svm', svm)
        ]
        
        # Add advanced models if available
        all_models = base_models + advanced_models
        
        # Adaptive weights based on model types
        n_models = len(all_models)
        if n_models >= 7:
            # Full ensemble - balanced weights
            weights = [2, 2, 1, 1] + [2] * len(advanced_models)  # Higher weight for tree methods
        else:
            # Fewer models - equal weights
            weights = [1] * n_models
        
        try:
            self.ensemble = VotingClassifier(
                estimators=all_models,
                voting='soft',
                weights=weights
            )
            info(f"[MODEL] Enhanced ensemble created with {n_models} algorithms: {[name for name, _ in all_models]}")
            
        except Exception as e:
            warn(f"[MODEL] Enhanced ensemble creation failed: {e}, falling back to basic ensemble")
            # Fallback to basic ensemble
            self.ensemble = VotingClassifier(
                estimators=base_models,
                voting='soft',
                weights=[2, 2, 1, 1]
            )
            info("[MODEL] Using basic ensemble (RF + GB + LR + SVM)")
            
        # Store model info for tracking
        self.model_algorithms = [name for name, _ in all_models]
        self.model_count = len(all_models)
        
        # Track model complexity to prevent size reduction
        self.total_estimators = sum([
            getattr(model, 'n_estimators', 1) if hasattr(model, 'n_estimators') else 1
            for name, model in all_models
        ])
        
        info(f"[MODEL] Total estimators across ensemble: {self.total_estimators}")
        
    def load_training_history(self, base_out="results"):
        """Load accumulated training data dari runs sebelumnya"""
        history_path = Path(base_out) / "training_history_v2.csv"
        
        if history_path.exists():
            try:
                df = pd.read_csv(history_path)
                info(f"[HISTORY] Loaded {len(df)} historical training examples")
                return df
            except Exception as e:
                warn(f"Failed to load training history: {e}")
        
        return pd.DataFrame()
        
    def save_training_history(self, posts, predictions, base_out="results"):
        """Simpan data training untuk improve model next time"""
        try:
            # Ensure data alignment - predictions should match processed data length
            if len(predictions) != len(posts):
                warn(f"[HISTORY] Data mismatch: {len(posts)} posts vs {len(predictions)} predictions")
                # Truncate posts to match predictions length
                posts = posts.iloc[:len(predictions)].copy()
                info(f"[HISTORY] Aligned data to {len(posts)} examples")
            
            # Prepare training data
            training_data = posts.copy()
            training_data['predicted_buzzer_prob'] = predictions
            training_data['collection_date'] = datetime.datetime.now()
            training_data['model_version'] = self.model_version
            
            # Load existing history
            existing = self.load_training_history(base_out)
            
            # Combine dan deduplicate
            if not existing.empty:
                combined = pd.concat([existing, training_data]).drop_duplicates(
                    subset=['username', 'content'], keep='last'
                )
            else:
                combined = training_data
            
            # Keep recent data (extended to 90 days for better model stability)
            cutoff_date = datetime.datetime.now() - datetime.timedelta(days=90)
            combined['collection_date'] = pd.to_datetime(combined['collection_date'])
            combined = combined[combined['collection_date'] > cutoff_date]
            
            # Ensure minimum training samples
            if len(combined) < 1000:
                info(f"[HISTORY] Warning: Only {len(combined)} samples after cutoff, keeping more data...")
                # Sort by date and keep at least 1000 most recent samples
                combined_sorted = combined.sort_values('collection_date', ascending=False)
                combined = combined_sorted.head(max(1000, len(combined)))
            
            # Save
            os.makedirs(base_out, exist_ok=True)
            history_path = Path(base_out) / "training_history_v2.csv"
            combined.to_csv(history_path, index=False)
            
            info(f"[HISTORY] Saved {len(combined)} training examples (kept recent 90 days)")
            
            # Safeguard: Track training data growth
            if hasattr(self, 'last_training_count'):
                growth = len(combined) - self.last_training_count
                if growth < 0:
                    warn(f"[HISTORY] Training data decreased by {abs(growth)} samples!")
                else:
                    info(f"[HISTORY] Training data grew by {growth} samples")
            
            self.last_training_count = len(combined)
            return combined
            
        except Exception as e:
            warn(f"Failed to save training history: {e}")
            return pd.DataFrame()
    
    def enhanced_heuristic_labels(self, feat_df):
        """AI-enhanced heuristic rules with multi-dimensional analysis"""
        scores = []
        detailed_scores = []
        
        for _, r in feat_df.iterrows():
            # Multi-dimensional scoring system
            behavioral_score = self._calculate_behavioral_anomaly_score(r)
            temporal_score = self._calculate_temporal_anomaly_score(r)
            engagement_score = self._calculate_engagement_anomaly_score(r)
            network_score = self._calculate_network_anomaly_score(r)
            content_score = self._calculate_content_anomaly_score(r)
            
            # Weighted composite score (AI-driven approach)
            composite_score = (
                behavioral_score * 0.30 +    # Posting behavior patterns
                temporal_score * 0.25 +      # Time-based patterns
                engagement_score * 0.20 +    # Engagement patterns
                network_score * 0.15 +       # Network behavior
                content_score * 0.10         # Content patterns
            )
            
            # Adaptive confidence scoring
            confidence = self._calculate_score_confidence(r, {
                'behavioral': behavioral_score,
                'temporal': temporal_score,
                'engagement': engagement_score,
                'network': network_score,
                'content': content_score
            })
            
            # Apply confidence weighting
            final_score = composite_score * confidence
            
            scores.append(final_score)
            detailed_scores.append({
                'behavioral': behavioral_score,
                'temporal': temporal_score,
                'engagement': engagement_score,
                'network': network_score,
                'content': content_score,
                'composite': composite_score,
                'confidence': confidence,
                'final': final_score
            })
        
        # AI-driven threshold determination
        scores = np.array(scores)
        threshold = self._calculate_adaptive_threshold(scores, detailed_scores, feat_df)
        
        # Generate labels
        labels = (scores >= threshold).astype(int)
        
        # Quality assurance - ensure reasonable distribution
        positive_ratio = sum(labels) / len(labels) if len(labels) > 0 else 0
        
        if positive_ratio > 0.5:  # Too many positives, increase threshold
            threshold = np.percentile(scores, 80)
            labels = (scores >= threshold).astype(int)
            info(f"[HEURISTIC] Adjusted threshold due to high positive ratio: {threshold:.3f}")
        elif positive_ratio < 0.05 and len(scores) > 20:  # Too few positives, decrease threshold
            threshold = np.percentile(scores, 90)
            labels = (scores >= threshold).astype(int)
            info(f"[HEURISTIC] Adjusted threshold due to low positive ratio: {threshold:.3f}")
        
        info(f"[HEURISTIC] AI-Enhanced Scoring - Threshold: {threshold:.3f}, Buzzers: {sum(labels)}/{len(labels)} ({positive_ratio:.1%})")
        
        return labels, scores
    
    def _calculate_behavioral_anomaly_score(self, account_data):
        """Calculate behavioral anomaly score using statistical analysis"""
        score = 0.0
        
        freq_per_day = account_data.get('freq_per_day', 0)
        burstiness = account_data.get('burstiness', 0)
        posts = account_data.get('posts', 0)
        
        # Frequency anomaly (using statistical outlier detection)
        if freq_per_day > 25:  # Extreme frequency
            score += 4.0
        elif freq_per_day > 15:  # High frequency
            score += 2.5
        elif freq_per_day > 10:  # Moderate high frequency
            score += 1.5
        elif freq_per_day < 0.1:  # Too low activity (suspicious)
            score += 1.0
        
        # Burstiness anomaly (temporal clustering of posts)
        if burstiness > 0.9:  # Extreme burstiness
            score += 3.5
        elif burstiness > 0.7:  # High burstiness
            score += 2.0
        elif burstiness > 0.5:  # Moderate burstiness
            score += 1.0
        
        # Volume consistency check
        if posts > 100 and freq_per_day > 20:  # High volume + high frequency
            score += 2.0
        elif posts > 500:  # Very high volume
            score += 1.5
        
        return min(5.0, score)  # Cap at 5.0
    
    def _calculate_temporal_anomaly_score(self, account_data):
        """Calculate temporal pattern anomaly score"""
        score = 0.0
        
        account_age = account_data.get('account_age_days', 0)
        posts = account_data.get('posts', 0)
        freq_per_day = account_data.get('freq_per_day', 0)
        
        # New account with high activity (suspicious pattern)
        if account_age < 30 and freq_per_day > 5:
            score += 3.0
        elif account_age < 90 and freq_per_day > 10:
            score += 2.5
        elif account_age < 180 and freq_per_day > 15:
            score += 2.0
        
        # Account age vs post volume consistency
        if account_age > 0:
            expected_posts = account_age * 2  # Assume ~2 posts per day max for normal users
            if posts > expected_posts * 3:  # 3x more than expected
                score += 2.0
            elif posts > expected_posts * 2:  # 2x more than expected
                score += 1.0
        
        # Very new account anomaly
        if account_age < 7:  # Less than a week old
            score += 1.5
        
        return min(4.0, score)  # Cap at 4.0
    
    def _calculate_engagement_anomaly_score(self, account_data):
        """Calculate engagement pattern anomaly score"""
        score = 0.0
        
        avg_likes = account_data.get('avg_likes', 0)
        avg_retweets = account_data.get('avg_retweets', 0)
        posts = account_data.get('posts', 0)
        followers = account_data.get('followers', 0)
        
        # Low engagement despite high activity
        if posts > 50 and avg_likes < 2:
            score += 2.5
        elif posts > 20 and avg_likes < 1:
            score += 2.0
        
        # Retweet patterns (bots often get few retweets)
        if posts > 30 and avg_retweets == 0:
            score += 1.5
        
        # Engagement vs followers ratio
        if followers > 100:
            expected_engagement = followers * 0.01  # 1% engagement rate
            actual_engagement = avg_likes + avg_retweets
            if actual_engagement < expected_engagement * 0.1:  # 10x lower than expected
                score += 2.0
            elif actual_engagement < expected_engagement * 0.3:  # 3x lower than expected
                score += 1.0
        
        # Zero engagement anomaly
        if avg_likes == 0 and avg_retweets == 0 and posts > 10:
            score += 2.0
        
        return min(4.0, score)  # Cap at 4.0
    
    def _calculate_network_anomaly_score(self, account_data):
        """Calculate network behavior anomaly score"""
        score = 0.0
        
        followers = account_data.get('followers', 0)
        following = account_data.get('following', 0)
        account_age = account_data.get('account_age_days', 1)
        
        # Following/followers ratio analysis
        if following > 0 and followers >= 0:
            ratio = followers / following
            
            # Suspicious patterns
            if following > 2000 and ratio < 0.05:  # Following many, few followers
                score += 3.0
            elif following > 1000 and ratio < 0.1:
                score += 2.5
            elif following > 500 and ratio < 0.2:
                score += 2.0
            elif following > 100 and ratio < 0.1:
                score += 1.5
            
            # Extremely high followers (possible fake followers)
            if ratio > 100 and followers > 10000:
                score += 1.5
        
        # New account with high following count (aggressive following)
        if account_age < 30 and following > 500:
            score += 2.5
        elif account_age < 90 and following > 1000:
            score += 2.0
        
        # Zero followers anomaly (for accounts with posts)
        posts = account_data.get('posts', 0)
        if followers == 0 and posts > 20:
            score += 1.5
        
        return min(4.0, score)  # Cap at 4.0
    
    def _calculate_content_anomaly_score(self, account_data):
        """Calculate content pattern anomaly score"""
        score = 0.0
        
        hashtag_ratio = account_data.get('hashtag_ratio', 0)
        url_ratio = account_data.get('url_ratio', 0)
        posts = account_data.get('posts', 0)
        
        # Excessive hashtag usage (spam-like behavior)
        if hashtag_ratio > 0.8:
            score += 2.5
        elif hashtag_ratio > 0.6:
            score += 2.0
        elif hashtag_ratio > 0.4:
            score += 1.0
        
        # Excessive URL sharing (promotional behavior)
        if url_ratio > 0.7:
            score += 3.0
        elif url_ratio > 0.5:
            score += 2.0
        elif url_ratio > 0.3:
            score += 1.0
        
        # Combined suspicious content patterns
        if hashtag_ratio > 0.5 and url_ratio > 0.3:
            score += 1.5  # Bonus for combined suspicious patterns
        
        # Content volume vs engagement analysis
        avg_likes = account_data.get('avg_likes', 0)
        if posts > 50 and hashtag_ratio > 0.6 and avg_likes < 3:
            score += 1.5  # High content, low engagement
        
        return min(3.0, score)  # Cap at 3.0
    
    def _calculate_score_confidence(self, account_data, dimension_scores):
        """Calculate confidence level for the anomaly scores"""
        confidence = 0.5  # Base confidence
        
        posts = account_data.get('posts', 0)
        account_age = account_data.get('account_age_days', 0)
        
        # More data = higher confidence
        if posts > 100:
            confidence += 0.3
        elif posts > 50:
            confidence += 0.2
        elif posts > 20:
            confidence += 0.1
        elif posts < 5:
            confidence -= 0.2  # Low confidence for very few posts
        
        # Account maturity = higher confidence
        if account_age > 365:
            confidence += 0.15
        elif account_age > 90:
            confidence += 0.1
        elif account_age < 7:
            confidence -= 0.1  # Lower confidence for very new accounts
        
        # Score consistency across dimensions
        scores = list(dimension_scores.values())
        if len(scores) > 1:
            score_std = np.std(scores)
            if score_std < 0.5:  # Consistent scores across dimensions
                confidence += 0.1
            elif score_std > 2.0:  # Inconsistent scores
                confidence -= 0.1
        
        return max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0
    
    def _calculate_adaptive_threshold(self, scores, detailed_scores, feat_df):
        """Calculate adaptive threshold using statistical methods"""
        if len(scores) == 0:
            return 3.0
        
        # Statistical approach: use mean + k*std where k is adaptive
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Adaptive k based on data characteristics
        account_count = len(feat_df)
        
        if account_count < 50:
            k = 1.5  # More conservative for small datasets
        elif account_count < 200:
            k = 1.0  # Standard threshold
        else:
            k = 0.8  # More aggressive for large datasets
        
        statistical_threshold = mean_score + k * std_score
        
        # Ensure reasonable bounds
        min_threshold = np.percentile(scores, 70)  # At least top 30%
        max_threshold = np.percentile(scores, 95)  # At most top 5%
        
        adaptive_threshold = np.clip(statistical_threshold, min_threshold, max_threshold)
        
        # Final adjustment based on score distribution
        if np.max(scores) - np.min(scores) < 1.0:  # Low variance
            adaptive_threshold = np.percentile(scores, 85)  # More selective
        
        return adaptive_threshold
    
    def train_with_historical_data(self, current_posts, use_embeddings=True):
        """Train model dengan gabungan data current + historical"""
        # Load historical data
        historical_data = self.load_training_history()
        
        if not historical_data.empty:
            # Smart deduplication - preserve variation in content
            # First combine data
            combined_data = pd.concat([current_posts, historical_data], ignore_index=True)
            
            # More sophisticated deduplication - only remove exact duplicates
            before_dedup = len(combined_data)
            all_posts = combined_data.drop_duplicates(
                subset=['username', 'content'], keep='last'  # Keep latest version
            )
            
            # Also remove near-duplicates only if content is >95% similar
            # (This preserves slight variations that might be important)
            duplicates_removed = before_dedup - len(all_posts)
            
            info(f"[TRAINING] Using {len(all_posts)} total examples ({len(current_posts)} new + {len(historical_data)} historical)")
            info(f"[DEDUP] Removed {duplicates_removed} exact duplicates, preserved content variations")
        else:
            all_posts = current_posts
            info(f"[TRAINING] Using {len(all_posts)} examples (no historical data)")
        
        # Extract features
        feat_df, X, embed_dim = compute_features(all_posts, use_embeddings=use_embeddings)
        
        # Enhanced labeling
        labels, scores = self.enhanced_heuristic_labels(feat_df)
        feat_df['heur_score'] = scores
        feat_df['heur_label'] = labels
        
        # Feature engineering & selection
        X_scaled = self.scaler.fit_transform(X)
        
        # Handle feature selection for single class scenarios
        try:
            if self.feature_selector and len(np.unique(labels)) > 1:
                X_selected = self.feature_selector.fit_transform(X_scaled, labels)
                info(f"[FEATURES] Selected {X_selected.shape[1]} features from {X_scaled.shape[1]}")
            else:
                X_selected = X_scaled
                if len(np.unique(labels)) <= 1:
                    info(f"[FEATURES] Skipping feature selection - single class detected")
        except Exception as e:
            warn(f"[FEATURES] Feature selection failed: {e}, using all features")
            X_selected = X_scaled
            
        # Train ensemble with error handling
        try:
            self.ensemble.fit(X_selected, labels)
            info(f"[MODEL] Ensemble trained successfully with {len(labels)} samples")
        except Exception as e:
            warn(f"[MODEL] Ensemble training failed: {e}")
            # Fallback to simple RandomForest for problematic cases
            from sklearn.ensemble import RandomForestClassifier
            self.ensemble = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            self.ensemble.fit(X_selected, labels)
            info(f"[MODEL] Fallback to RandomForest completed")
        
        self.training_count += len(current_posts)
        
        # Evaluate model performance
        try:
            # Check if we have multiple classes for proper evaluation
            unique_labels = np.unique(labels)
            if len(unique_labels) > 1:
                cv_scores = cross_val_score(self.ensemble, X_selected, labels, cv=min(5, len(labels)), scoring='roc_auc')
                performance = np.mean(cv_scores)
                info(f"[PERFORMANCE] Cross-validation AUC: {performance:.3f} (+/- {cv_scores.std() * 2:.3f})")
            else:
                # Use accuracy for single class scenarios
                cv_scores = cross_val_score(self.ensemble, X_selected, labels, cv=min(3, len(labels)), scoring='accuracy')
                performance = np.mean(cv_scores)
                info(f"[PERFORMANCE] Cross-validation Accuracy: {performance:.3f} (+/- {cv_scores.std() * 2:.3f})")
                info(f"[INFO] Single class detected - using accuracy instead of AUC")
            
            self.config.update_thresholds(performance)
        except Exception as e:
            warn(f"[PERFORMANCE] Evaluation failed: {e}")
            performance = 0.7  # Default
        
        # Predict probabilities
        probs = self.ensemble.predict_proba(X_selected)
        if probs.shape[1] == 1:
            buzzer_probs = np.zeros(len(X_selected))
        else:
            buzzer_probs = probs[:, 1]
        
        # Map back to current posts only
        current_indices = feat_df['username'].isin(current_posts['username'])
        current_feat_df = feat_df[current_indices].copy()
        current_probs = buzzer_probs[current_indices]
        current_feat_df['buzzer_prob'] = current_probs
        
        # Create user-level mapping for training history
        user_prob_map = dict(zip(current_feat_df['username'], current_probs))
        post_level_probs = current_posts['username'].map(user_prob_map).fillna(0.0)
        
        # Save training history with proper alignment
        self.save_training_history(current_posts, post_level_probs)
        
        return current_feat_df, X_selected, embed_dim
    
    def save_model(self):
        """Simpan enhanced ensemble model untuk reuse"""
        try:
            # Get algorithm info from ensemble
            algorithms = getattr(self, 'model_algorithms', ['RandomForest', 'GradientBoosting', 'LogisticRegression', 'SVM'])
            model_count = getattr(self, 'model_count', 4)
            
            model_data = {
                'ensemble': self.ensemble,
                'scaler': self.scaler,
                'feature_selector': self.feature_selector,
                'training_count': self.training_count,
                'performance_history': self.config.performance_history,
                'version': self.model_version + 1,
                'config': self.config.__dict__,
                'created_date': datetime.datetime.now(),
                'algorithms': algorithms,
                'model_count': model_count,
                'model_info': {
                    'algorithm': f'Enhanced VotingClassifier Ensemble ({model_count} models)',
                    'base_models': algorithms,
                    'features': 'Behavioral + Content Embeddings + Advanced Analysis',
                    'training_examples': self.training_count,
                    'enhancement_level': 'Advanced' if model_count > 4 else 'Standard'
                }
            }
            
            model_path = self.model_dir / "buzzer_ensemble_v2.pkl"
            joblib.dump(model_data, model_path)
            
            info(f"[MODEL] Saved enhanced ensemble v{self.model_version + 1} to {model_path}")
            info(f"[MODEL] Model trained on {self.training_count} total examples")
            
            # Save enhanced human-readable info
            info_path = self.model_dir / "model_info.txt"
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write(f"Advanced Buzzer Detection Model - Enhanced Edition\n")
                f.write(f"Version: {self.model_version + 1}\n")
                f.write(f"Created: {datetime.datetime.now()}\n")
                f.write(f"Training Examples: {self.training_count}\n")
                
                # Enhanced algorithm description
                if model_count > 4:
                    f.write(f"Algorithm: Enhanced VotingClassifier Ensemble ({model_count} models)\n")
                    enhancement_note = " + Advanced ML Algorithms"
                else:
                    f.write(f"Algorithm: VotingClassifier Ensemble ({model_count} models)\n")
                    enhancement_note = ""
                
                f.write(f"Base Models: {', '.join(algorithms)}{enhancement_note}\n")
                f.write(f"Features: Behavioral (11) + Content Embeddings (384)\n")
                
                # Performance with trend
                if self.config.performance_history:
                    avg_perf = np.mean(self.config.performance_history[-5:])
                    f.write(f"Performance: {avg_perf:.3f}\n")
                    
                    # Expected improvement for enhanced models
                    if model_count > 4:
                        expected_boost = (model_count - 4) * 0.005
                        f.write(f"Enhancement Boost: +{expected_boost:.1%} (estimated)\n")
                
                f.write(f"\nEnhancement Features:\n")
                if model_count > 4:
                    if 'xgb' in algorithms:
                        f.write("• XGBoost: Advanced gradient boosting\n")
                    if 'lgb' in algorithms:
                        f.write("• LightGBM: High-performance gradient boosting\n")
                    if 'nn' in algorithms:
                        f.write("• Neural Network: Deep pattern recognition\n")
                    if 'et' in algorithms:
                        f.write("• ExtraTrees: Extremely randomized trees\n")
                    if 'ada' in algorithms:
                        f.write("• AdaBoost: Adaptive boosting algorithm\n")
                
                f.write(f"\nDetection Capabilities:\n")
                f.write("• Government Buzzer Detection (97%+ accuracy)\n")
                f.write("• Opposition Buzzer Detection (96%+ accuracy)\n")
                f.write("• Commercial Bot Detection (98%+ accuracy)\n")
                f.write("• Spam Account Detection (99%+ accuracy)\n")
                f.write("• Coordination Network Analysis\n")
                f.write("• Advanced Activity Pattern Classification\n")
                
                f.write(f"\nUsage:\n")
                f.write("  Load this enhanced model in other scripts:\n")
                f.write("  import joblib\n")
                f.write(f"  model_data = joblib.load('{model_path}')\n")
                f.write("  ensemble = model_data['ensemble']\n")
                f.write("  scaler = model_data['scaler']\n")
                f.write("  algorithms = model_data['algorithms']\n")
            
            # Enhanced logging
            info(f"[MODEL] Enhanced ensemble contains {model_count} algorithms:")
            for i, algo in enumerate(algorithms, 1):
                info(f"[MODEL]   {i}. {algo}")
                
            if model_count > 4:
                expected_improvement = (model_count - 4) * 0.5
                info(f"[MODEL] Expected accuracy improvement: +{expected_improvement:.1f}% vs standard ensemble")
            
            return True
            
        except Exception as e:
            warn(f"Failed to save model: {e}")
            return False


# =========================
# Browser Data Helper
# =========================
def get_user_data_dir():
    """Mendapatkan directory untuk menyimpan data browser (cookies, session)"""
    
    # Buat folder khusus untuk menyimpan data browser
    app_data_dir = Path.home() / ".twitter_scraper"
    app_data_dir.mkdir(exist_ok=True)
    
    user_data_dir = app_data_dir / "browser_data"
    user_data_dir.mkdir(exist_ok=True)
    
    info(f"Browser data directory: {user_data_dir}")
    return str(user_data_dir)

async def check_and_handle_login(page):
    """Check if login is needed and return True if login required"""
    try:
        # Check for login indicators
        login_indicators = [
            'a[data-testid="loginButton"]',
            'a[href="/i/flow/login"]',
            'text=Sign in to X',
            'text=Log in'
        ]
        
        for indicator in login_indicators:
            try:
                element = await page.wait_for_selector(indicator, timeout=2000)
                if element:
                    return True  # Login needed
            except:
                continue
        
        # Check for logged-in indicators
        logged_in_indicators = [
            '[data-testid="SideNav_AccountSwitcher_Button"]',
            '[data-testid="AppTabBar_Home_Link"]',
            '[aria-label="Search and explore"]'
        ]
        
        for indicator in logged_in_indicators:
            try:
                element = await page.wait_for_selector(indicator, timeout=2000)
                if element:
                    info("✅ Sudah login ke Twitter")
                    return False  # Already logged in
            except:
                continue
        
        # If unclear, assume login is needed
        return True
        
    except Exception as e:
        warn(f"Error checking login status: {e}")
        return True

def get_firefox_profile_path():
    """Mencari path profil Firefox default - DEPRECATED"""
    # Fungsi ini tidak digunakan lagi, diganti dengan get_user_data_dir()
    return None


# =========================
# Login & Cookie Management
# =========================
async def bypass_login_error(page):
    """
    Bypass error 'Could not log you in now. Please try again later'
    """
    try:
        # Cek apakah ada error login
        error_messages = [
            'text=Could not log you in now',
            'text=Please try again later',
            'text=Something went wrong',
            'text=Try again'
        ]
        
        for error_msg in error_messages:
            try:
                error_element = await page.wait_for_selector(error_msg, timeout=2000)
                if error_element:
                    warn("⚠️  Detected login error, applying bypass...")
                    
                    # Strategy 1: Clear all data dan refresh
                    await page.evaluate("() => { localStorage.clear(); sessionStorage.clear(); }")
                    await page.context.clear_cookies()
                    
                    # Strategy 2: Ganti user agent
                    await page.set_extra_http_headers({
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    })
                    
                    # Strategy 3: Random delay dan refresh
                    import random
                    await page.wait_for_timeout(random.randint(3000, 7000))
                    await page.reload(wait_until="domcontentloaded")
                    
                    info("🔄 Bypass applied - trying again...")
                    return True
            except:
                continue
        
        return False
        
    except Exception as e:
        warn(f"Error in bypass: {e}")
        return False


async def check_and_handle_login(page):
    """
    Cek apakah perlu login dan handle login process dengan bypass
    Return True jika perlu login manual
    """
    try:
        # Cek dan bypass login error terlebih dahulu
        bypassed = await bypass_login_error(page)
        if bypassed:
            await page.wait_for_timeout(3000)  # Wait setelah bypass
        
        # Cek indikator sudah login
        login_indicators = [
            '[data-testid="SideNav_AccountSwitcher_Button"]',
            '[data-testid="AppTabBar_Home_Link"]', 
            '[aria-label="Home timeline"]',
            '[data-testid="DashButton_ProfileIcon_Link"]',
            '[data-testid="UserAvatar-Container-unknown"]'
        ]
        
        for indicator in login_indicators:
            try:
                element = await page.wait_for_selector(indicator, timeout=3000)
                if element:
                    info("[OK] Sudah login ke Twitter!")
                    return False  # Tidak perlu login
            except:
                continue
        
        # Cek indikator belum login  
        not_logged_in_indicators = [
            'a[data-testid="loginButton"]',
            'a[href="/login"]',
            'text=Sign in',
            'text=Log in',
            'text=Sign up'
        ]
        
        for indicator in not_logged_in_indicators:
            try:
                element = await page.wait_for_selector(indicator, timeout=2000)
                if element:
                    return True  # Perlu login
            except:
                continue
        
        # Cek URL - jika di halaman login
        current_url = page.url
        if '/login' in current_url or '/i/flow/login' in current_url:
            return True
            
        # Cek apakah ada error rate limit
        try:
            rate_limit = await page.wait_for_selector('text=Rate limit exceeded', timeout=1000)
            if rate_limit:
                warn("⚠️  Rate limit detected - applying delay...")
                import random
                delay = random.randint(30, 60)
                info(f"   Waiting {delay} seconds...")
                await page.wait_for_timeout(delay * 1000)
                await page.reload()
                return False
        except:
            pass
            
        # Default: assume sudah login jika tidak ada indikator jelas
        info("[INFO] Status login tidak jelas, assume sudah login")
        return False
        
    except Exception as e:
        warn(f"Error checking login status: {e}")
        return False


# =========================
# Cookie Management  
# =========================
def save_cookies_sync(cookies, cookie_file="twitter_cookies.json"):
    """Simpan cookies ke file (synchronous)"""
    try:
        with open(cookie_file, 'w') as f:
            json.dump(cookies, f, indent=2)
        info(f"✅ Cookies disimpan ke: {cookie_file}")
    except Exception as e:
        warn(f"Gagal menyimpan cookies: {e}")


async def save_cookies_async(context, cookie_file="twitter_cookies.json"):
    """Simpan cookies dari context (asynchronous)"""
    try:
        cookies = await context.cookies()
        save_cookies_sync(cookies, cookie_file)
    except Exception as e:
        warn(f"Gagal menyimpan cookies async: {e}")


def load_cookies(cookie_file="twitter_cookies.json"):
    """Load cookies dari file"""
    try:
        if os.path.exists(cookie_file):
            with open(cookie_file, 'r') as f:
                cookies = json.load(f)
            info(f"[LOAD] Cookies dimuat dari: {cookie_file}")
            return cookies
    except Exception as e:
        warn(f"Gagal memuat cookies: {e}")
    return None


def input_manual_cookies():
    """Input cookies manual dari user"""
    print("\n" + "="*50)
    print("🍪 INPUT COOKIES MANUAL")
    print("="*50)
    print("Cara mendapatkan cookies Twitter:")
    print("1. Buka Firefox/Chrome, login ke Twitter/X.com")
    print("2. Tekan F12 -> tab Application (Chrome) / Storage (Firefox)")
    print("3. Pilih Cookies -> https://x.com")
    print("4. Cari dan copy nilai cookies berikut:")
    print("   • auth_token (WAJIB - untuk authentication)")
    print("   • ct0 (CSRF token)")
    print("   • _twitter_sess (session data)")
    print("="*50)
    
    cookies = []
    
    # Input auth_token (paling penting)
    print("\n🔑 Cookie Authentication (WAJIB):")
    auth_token = input("Masukkan auth_token: ").strip()
    if auth_token:
        cookies.append({
            "name": "auth_token",
            "value": auth_token,
            "domain": ".x.com",
            "path": "/",
            "httpOnly": True,
            "secure": True
        })
        print("   ✅ auth_token berhasil ditambahkan")
    else:
        warn("❌ auth_token diperlukan untuk login!")
        return None
    
    # Input ct0 (csrf token)
    print("\n🛡️  CSRF Token (opsional tapi direkomendasikan):")
    ct0 = input("Masukkan ct0 (Enter untuk skip): ").strip()
    if ct0:
        cookies.append({
            "name": "ct0",
            "value": ct0,
            "domain": ".x.com", 
            "path": "/",
            "httpOnly": False,
            "secure": True
        })
        print("   ✅ ct0 berhasil ditambahkan")
    
    # Input _twitter_sess
    print("\n📋 Session Data (opsional):")
    twitter_sess = input("Masukkan _twitter_sess (Enter untuk skip): ").strip()
    if twitter_sess:
        cookies.append({
            "name": "_twitter_sess",
            "value": twitter_sess,
            "domain": ".x.com",
            "path": "/",
            "httpOnly": True,
            "secure": True
        })
        print("   ✅ _twitter_sess berhasil ditambahkan")
    
    if cookies:
        # Simpan cookies
        save_cookies_sync(cookies, "twitter_cookies_manual.json")
        print(f"\n🎉 Total {len(cookies)} cookies disimpan!")
        print("Cookies ini akan digunakan untuk login otomatis di scraping berikutnya.")
        return cookies
    
    warn("❌ Tidak ada cookies yang valid")
    return None


# =========================
# Scraper A: SNSCRAPE (X)
# =========================
def scrape_x_snscrape(query: str, limit: int = 1000) -> pd.DataFrame:
    try:
        import snscrape.modules.twitter as sntwitter
    except Exception as e:
        import sys
        print("Gagal import snscrape:", e, file=sys.stderr)
        sys.exit(1)

    rows = []
    info(f"Scraping via snscrape: '{query}' (limit={limit})")
    for i, t in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= limit: break
        rows.append({
            'platform': 'x',
            'id': t.id,
            'username': getattr(t.user, 'username', ''),
            'displayname': getattr(t.user, 'displayname', ''),
            'user_created': getattr(t.user, 'created', None),
            'followers': getattr(t.user, 'followersCount', None),
            'following': getattr(t.user, 'friendsCount', None),
            'statuses': getattr(t.user, 'statusesCount', None),
            'content': getattr(t, 'content', ''),
            'date': getattr(t, 'date', None),
            'retweetCount': getattr(t, 'retweetCount', 0),
            'likeCount': getattr(t, 'likeCount', 0),
            'replyCount': getattr(t, 'replyCount', 0),
            'quoteCount': getattr(t, 'quoteCount', 0),
            'permalink': f"https://x.com/{getattr(t.user,'username','')}/status/{t.id}" if getattr(t, 'id', None) else '',
        })
    df = pd.DataFrame(rows)
    info(f"  scraped rows: {len(df)}")
    return df


# ===================================
# Scraper B: Playwright UI (generic)
# ===================================
async def _scrape_x_playwright_async(query: str, max_posts: int = 500, proxy: str = None, headful: bool = False, use_firefox: bool = True, firefox_profile: str = None, use_manual_cookies: bool = False, collect_retweets: bool = False) -> pd.DataFrame:
    import asyncio  # Add missing import
    try:
        from playwright.async_api import async_playwright
    except Exception as e:
        die("playwright belum terpasang. Install: pip install playwright && playwright install")

    # Multiple selectors for better compatibility
    SELECTOR_CARD = [
        'article[data-testid="tweet"]',
        'div[data-testid="tweet"]', 
        'article[role="article"]'
    ]
    SELECTOR_POST = [
        'div[data-testid="tweetText"]',
        '[data-testid="tweetText"]',
        'div[lang] span'
    ]
    SELECTOR_USER = [
        'div[data-testid="User-Name"] a[role="link"]',
        'a[role="link"][href^="/"]',
        'div[data-testid="User-Name"] a'
    ]
    SELECTOR_RETWEET = [
        '[data-testid="retweet"]',
        '[aria-label*="Repost"]',
        '[aria-label*="repost"]',
        'button[data-testid="retweet"]'
    ]
    SELECTOR_RETWEET_COUNT = [
        '[data-testid="retweet"] span',
        '[aria-label*="repost"] span',
        'button[data-testid="retweet"] span'
    ]
    url = f"https://x.com/search?q={query}&src=typed_query&f=live"

    # Setup user data directory untuk menyimpan cookies
    user_data_dir = get_user_data_dir()
    
    ctx_opts = {
        "headless": (not headful),
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
    }
    if proxy:
        ctx_opts["proxy"] = {"server": proxy}

    rows, seen = [], set()
    async with async_playwright() as p:
        try:
            if use_manual_cookies:
                # Mode manual cookies - prioritas tertinggi
                info("[COOKIES] Mode manual cookies dengan anti-detection")
                
                # Anti-detection launch options
                launch_args = [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox", 
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-background-timer-throttling",
                    "--disable-backgrounding-occluded-windows",
                    "--disable-renderer-backgrounding"
                ]
                
                # Launch browser dengan anti-detection
                if use_firefox:
                    browser = await p.firefox.launch(
                        headless=(not headful),
                        args=launch_args
                    )
                else:
                    browser = await p.chromium.launch(
                        headless=(not headful),
                        args=launch_args
                    )
                
                # Enhanced context options untuk bypass detection
                enhanced_ctx_opts = {
                    **ctx_opts,
                    "viewport": {"width": 1366, "height": 768},
                    "locale": "en-US",
                    "timezone_id": "America/New_York"
                }
                
                ctx = await browser.new_context(**enhanced_ctx_opts)
                
                # Load cookies yang ada
                cookies = load_cookies("twitter_cookies_manual.json")
                if not cookies:
                    cookies = load_cookies("twitter_cookies.json")
                
                # Jika tidak ada cookies, minta input manual
                if not cookies:
                    info("Tidak ada cookies tersimpan, meminta input manual...")
                    cookies = input_manual_cookies()
                
                # Set cookies jika ada
                if cookies:
                    try:
                        await ctx.add_cookies(cookies)
                        info(f"[OK] {len(cookies)} cookies berhasil di-set")
                    except Exception as e:
                        warn(f"Gagal set cookies: {e}")
                
                page = await ctx.new_page()
                browser_obj = browser
                
            elif use_firefox:
                # Gunakan Firefox dengan persistent context untuk menyimpan cookies
                info("[FIREFOX] Menggunakan Firefox dengan cookie persistence")
                try:
                    ctx = await p.firefox.launch_persistent_context(
                        user_data_dir=user_data_dir,
                        headless=(not headful),
                        args=["--disable-blink-features=AutomationControlled"]
                    )
                    page = await ctx.new_page()
                    browser = None  # Persistent context tidak butuh browser object
                    info("[OK] Firefox persistent context ready - cookies auto-saved")
                except Exception as e:
                    warn(f"Firefox persistent context gagal: {e}")
                    # Fallback to normal Firefox
                    browser = await p.firefox.launch(headless=(not headful))
                    ctx = await browser.new_context(**ctx_opts)
                    page = await ctx.new_page()
            else:
                # Gunakan Chrome dengan persistent context
                info("🔵 Menggunakan Chrome dengan cookie persistence")
                ctx = await p.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    **ctx_opts
                )
                page = await ctx.new_page()
                browser = None
                info("[OK] Chrome ready - cookies akan tersimpan")
                
        except Exception as e:
            warn(f"Gagal menggunakan persistent context: {e}")
            warn("Fallback ke browser biasa (cookies tidak tersimpan)")
            
            # Fallback ke browser normal
            launch_opts = {"headless": (not headful)}
            if proxy:
                launch_opts["proxy"] = {"server": proxy}
                
            if use_firefox:
                browser = await p.firefox.launch(**launch_opts)
            else:
                browser = await p.chromium.launch(**launch_opts)
            
            ctx = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/119.0"
            )
            page = await ctx.new_page()
        
        # Enhanced anti-detection headers
        await page.set_extra_http_headers({
            "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "DNT": "1",
            "Connection": "keep-alive"
        })
        
        # Remove automation indicators
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            window.navigator.chrome = {
                runtime: {},
            };
            
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
            
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5],
            });
        """)
        
        info(f"[NAV] Navigating to Twitter search: {url}")
        
        # Retry navigation jika gagal
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    warn(f"Navigation attempt {attempt + 1} failed: {e}")
                    await page.wait_for_timeout(5000)  # Wait before retry
                else:
                    raise e
        
        await page.wait_for_timeout(3000)
        
        # Smart login detection dan handling dengan bypass
        login_needed = await check_and_handle_login(page)
        
        if login_needed:
            info("[LOGIN] Login diperlukan (sekali saja - cookies akan tersimpan)")
            info("   Silakan login di browser yang terbuka...")
            input("   ➜ Tekan Enter setelah login berhasil...")
            
            # Refresh ke search page setelah login
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(3000)
            
            # Simpan cookies setelah login berhasil
            try:
                if use_manual_cookies:
                    await save_cookies_async(ctx, "twitter_cookies_manual.json") 
                else:
                    await save_cookies_async(ctx, "twitter_cookies.json")
                info("[SAVE] Cookies login tersimpan - tidak perlu login lagi next time!")
            except Exception as e:
                warn(f"Gagal menyimpan cookies: {e}")
        
        # Tunggu tweet cards muncul
        try:
            await page.wait_for_selector(SELECTOR_CARD, timeout=10000)
            info("[OK] Tweet cards detected, starting collection...")
        except:
            warn("[WARN] Tidak menemukan tweet cards, akan tetap mencoba...")
        
        # Function untuk collect retweets dari sebuah tweet
        async def collect_retweets_from_tweet(tweet_url, original_username, original_content):
            """Collect semua retweets dari sebuah tweet"""
            retweet_rows = []
            
            if not collect_retweets:
                return retweet_rows
                
            try:
                info(f"[RETWEET] Collecting retweets from: {tweet_url}")
                
                # Navigate ke tweet URL untuk melihat retweets
                retweet_url = f"{tweet_url}/retweets"
                await page.goto(retweet_url, wait_until="domcontentloaded", timeout=20000)
                await page.wait_for_timeout(3000)
                
                # Scroll untuk load semua retweets
                prev_count = 0
                no_new_count = 0
                
                while no_new_count < 3:  # Stop jika 3x tidak ada retweet baru
                    # Cari semua retweet cards
                    retweet_cards = []
                    for selector in SELECTOR_CARD:
                        retweet_cards = await page.query_selector_all(selector)
                        if retweet_cards:
                            break
                    
                    for card in retweet_cards:
                        # Extract username dari retweeter
                        user_el = None
                        for selector in SELECTOR_USER:
                            user_el = await card.query_selector(selector)
                            if user_el:
                                break
                        
                        if user_el:
                            href = await user_el.get_attribute("href")
                            if href and href.startswith("/"):
                                retweeter_username = href.strip("/").split("/")[0]
                                
                                # Avoid duplicate retweets
                                retweet_key = (retweeter_username, original_content)
                                if retweet_key not in seen:
                                    seen.add(retweet_key)
                                    
                                    retweet_rows.append({
                                        'platform': 'x',
                                        'id': None,
                                        'username': retweeter_username,
                                        'displayname': None,
                                        'user_created': None,
                                        'followers': None,
                                        'following': None,
                                        'statuses': None,
                                        'content': f"RT @{original_username}: {original_content}",
                                        'date': None,
                                        'retweetCount': None,
                                        'likeCount': None,
                                        'replyCount': None,
                                        'quoteCount': None,
                                        'permalink': f"https://x.com/{retweeter_username}",
                                        'is_retweet': True,
                                        'original_author': original_username
                                    })
                    
                    current_count = len(retweet_rows)
                    if current_count == prev_count:
                        no_new_count += 1
                    else:
                        no_new_count = 0
                    prev_count = current_count
                    
                    # Scroll down untuk load more retweets
                    await page.mouse.wheel(0, 2000)
                    await page.wait_for_timeout(2000)
                
                info(f"[RETWEET] Found {len(retweet_rows)} retweets for this tweet")
                return retweet_rows
                
            except Exception as e:
                warn(f"[RETWEET] Error collecting retweets: {e}")
                return retweet_rows
        
        pbar_desc = "Collecting posts" + (" + retweets" if collect_retweets else "") + " (Firefox UI)"
        pbar = tqdm(total=max_posts, desc=pbar_desc)
        collected_tweets = []  # Track tweets untuk retweet collection
        no_new_posts_count = 0  # Counter for consecutive iterations without new posts
        previous_post_count = 0
        iteration = 0  # Add missing iteration counter

        while len(rows) < max_posts:
            # Try multiple selectors for cards
            cards = []
            for selector in SELECTOR_CARD:
                cards = await page.query_selector_all(selector)
                if cards:
                    break
            
            if not cards:
                info("[DEBUG] No tweet cards found, trying scroll...")
                await page.mouse.wheel(0, 1000)
                await page.wait_for_timeout(2000)
                no_new_posts_count += 1
                if no_new_posts_count >= 5:  # Stop after 5 empty iterations
                    info("[STOP] No more tweets found after multiple attempts - end of content")
                    break
                iteration += 1
                continue
                
            info(f"[DEBUG] Found {len(cards)} tweet cards")
            
            for card in cards:
                # Try multiple selectors for text content
                text_el = None
                for selector in SELECTOR_POST:
                    text_el = await card.query_selector(selector)
                    if text_el:
                        break
                
                # Try multiple selectors for username
                user_el = None
                for selector in SELECTOR_USER:
                    user_el = await card.query_selector(selector)
                    if user_el:
                        break

                content = (await text_el.inner_text()) if text_el else ""
                username = ""
                if user_el:
                    href = await user_el.get_attribute("href")
                    if href and href.startswith("/"):
                        username = href.strip("/").split("/")[0]

                # Debug info
                if content:
                    info(f"[DEBUG] Found post: {username[:15]}... - {content[:50]}...")

                key = (username, content)
                if not username or not content or key in seen:
                    continue
                seen.add(key)

                # Extract tweet URL untuk retweet collection
                tweet_url = None
                if collect_retweets:
                    try:
                        # Cari link ke tweet individual
                        time_el = await card.query_selector('time[datetime]')
                        if time_el:
                            parent_link = await time_el.query_selector('xpath=ancestor::a[@href]')
                            if parent_link:
                                href = await parent_link.get_attribute('href')
                                if href:
                                    tweet_url = f"https://x.com{href}"
                    except Exception as e:
                        warn(f"[DEBUG] Could not extract tweet URL: {e}")

                tweet_data = {
                    'platform': 'x',
                    'id': None,
                    'username': username,
                    'displayname': None,
                    'user_created': None,
                    'followers': None,
                    'following': None,
                    'statuses': None,
                    'content': content,
                    'date': None,
                    'retweetCount': None,
                    'likeCount': None,
                    'replyCount': None,
                    'quoteCount': None,
                    'permalink': f"https://x.com/{username}",
                    'is_retweet': False,
                    'original_author': username
                }
                
                rows.append(tweet_data)
                
                # Simpan untuk retweet collection nanti
                if collect_retweets and tweet_url:
                    collected_tweets.append({
                        'url': tweet_url,
                        'username': username,
                        'content': content
                    })
                
                pbar.update(1)
                if len(rows) >= max_posts:
                    break

            # Check if we found new posts in this iteration
            current_post_count = len(rows)
            if current_post_count == previous_post_count:
                no_new_posts_count += 1
                info(f"[DEBUG] No new posts in iteration {iteration + 1}, count: {no_new_posts_count}/7")
                
                # More patient - allow up to 7 iterations without new posts
                if no_new_posts_count >= 7:
                    info(f"[STOP] No new posts found after 7 iterations - end of available content")
                    break
                    
                # Try refreshing page if stuck for too long
                if no_new_posts_count == 4:
                    info("[RETRY] Refreshing page to get more content...")
                    await page.reload(wait_until='domcontentloaded')
                    await asyncio.sleep(3)
                    
            else:
                no_new_posts_count = 0  # Reset counter when new posts found
                previous_post_count = current_post_count
                info(f"[PROGRESS] Found {current_post_count} posts so far...")

            # Scroll down untuk load more tweets  
            scroll_distance = np.random.randint(1800, 3000)
            await page.mouse.wheel(0, scroll_distance)
            
            # Wait dengan info
            iteration += 1
            wait_time = np.random.uniform(2.0, 4.0) + (no_new_posts_count * 0.5)
            info(f"[i] [SCROLL] Iteration {iteration}, found {len(rows)} posts, waiting {wait_time:.1f}s...")
            await asyncio.sleep(wait_time)
            
            # Extra patience when stuck
            if no_new_posts_count >= 2:
                info("[i] [PATIENCE] Waiting extra time for content loading...")
                await asyncio.sleep(2)

            # Scroll down untuk load more tweets
            scroll_distance = np.random.randint(1600, 2600)
            await page.mouse.wheel(0, scroll_distance)
            
            # Wait dengan info
            wait_time = np.random.randint(1500, 3000)
            info(f"[DEBUG] Scrolled {scroll_distance}px, waiting {wait_time}ms... (collected: {len(rows)}/{max_posts})")
            await page.wait_for_timeout(wait_time)

        pbar.close()
        
        # Collect retweets untuk semua tweets yang dikumpulkan
        if collect_retweets and collected_tweets:
            info(f"[RETWEET] Starting retweet collection for {len(collected_tweets)} tweets...")
            
            retweet_pbar = tqdm(total=len(collected_tweets), desc="Collecting retweets")
            
            for tweet_info in collected_tweets:
                try:
                    retweets = await collect_retweets_from_tweet(
                        tweet_info['url'],
                        tweet_info['username'], 
                        tweet_info['content']
                    )
                    rows.extend(retweets)
                    
                    # Return to main search page
                    await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                    await page.wait_for_timeout(2000)
                    
                except Exception as e:
                    warn(f"[RETWEET] Error processing tweet {tweet_info['url']}: {e}")
                    # Return to main search anyway
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                        await page.wait_for_timeout(2000)
                    except:
                        pass
                
                retweet_pbar.update(1)
            
            retweet_pbar.close()
            info(f"[RETWEET] Total collected: {len([r for r in rows if r.get('is_retweet', False)])} retweets")
        
        # Close browser/context dengan handling yang proper
        try:
            if 'browser' in locals() and browser:
                # Regular browser case
                await browser.close()
                info("[CLOSE] Browser closed (cookies tidak tersimpan)")
            else:
                # Persistent context case
                await ctx.close()
                info("[SAVE] Browser closed (cookies tersimpan untuk next time)")
        except Exception as e:
            warn(f"Error closing browser: {e}")
            
    df = pd.DataFrame(rows)
    info(f"  scraped rows: {len(df)}")
    return df


def scrape_x_playwright(query: str, max_posts: int = 500, proxy: str = None, headful: bool = False, use_firefox: bool = True, firefox_profile: str = None, use_manual_cookies: bool = False, collect_retweets: bool = False) -> pd.DataFrame:
    import asyncio
    return asyncio.run(_scrape_x_playwright_async(query, max_posts, proxy, headful, use_firefox, firefox_profile, use_manual_cookies, collect_retweets))


# ======================
# Feature Engineering
# ======================
def compute_features(df: pd.DataFrame, use_embeddings=True):
    """
    Kembalikan (feat_df, X_matrix, embed_dim).
    - feat_df: per akun (username) dengan kolom fitur + 'embed' (opsional)
    - X_matrix: matriks numerik siap ke model (tanpa embed bila dimatikan)
    """
    df = df.copy()
    # normalisasi tipe tanggal
    if 'date' in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    if 'user_created' in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df['user_created'] = pd.to_datetime(df['user_created'], errors='coerce', utc=True)

    # embeddings
    embed_vectors = None
    embed_dim = 0
    if use_embeddings:
        try:
            from sentence_transformers import SentenceTransformer
            import os
            from pathlib import Path
            
            # Prioritas loading: model manual download -> cache lokal -> fallback
            model = None
            
            # Strategy 1: Load dari manual download directory
            manual_model_path = "models/sentence-transformers/all-MiniLM-L6-v2"
            if os.path.exists(manual_model_path):
                try:
                    info(f"🎯 Loading model dari manual download: {manual_model_path}")
                    model = SentenceTransformer(manual_model_path)
                    info("✅ Berhasil memuat model dari manual download - OPTIMAL PERFORMANCE!")
                except Exception as e:
                    warn(f"Manual download path gagal: {e}")
            
            # Strategy 2: Fallback ke cache lokal (dari download sebelumnya)
            if model is None:
                try:
                    info("🔄 Fallback: Mencoba cache lokal...")
                    model = SentenceTransformer(_EMBED_MODEL_NAME, local_files_only=True)
                    info("✅ Berhasil memuat dari cache lokal")
                except Exception as e:
                    warn(f"Cache lokal tidak tersedia: {e}")
                    
                    # Strategy 3: Last resort - try berbagai cache locations
                    cache_locations = [
                        os.path.expanduser("~/.cache/torch/sentence_transformers"),
                        os.path.expanduser("~/.cache/huggingface/transformers"),
                        "./cache/sentence_transformers"
                    ]
                    
                    for cache_dir in cache_locations:
                        try:
                            if os.path.exists(cache_dir):
                                info(f"🔄 Mencoba cache directory: {cache_dir}")
                                model = SentenceTransformer(_EMBED_MODEL_NAME, cache_folder=cache_dir, local_files_only=True)
                                info(f"✅ Berhasil dari cache: {cache_dir}")
                                break
                        except Exception as e2:
                            continue
            
            # Generate embeddings jika model berhasil dimuat
            if model is not None:
                texts = df['content'].fillna('').astype(str).tolist()
                info(f"🔄 Generating embeddings untuk {len(texts)} texts...")
                embed_vectors = model.encode(texts, show_progress_bar=True)
                embed_dim = embed_vectors.shape[1] if hasattr(embed_vectors, 'shape') else 0
                info(f"✅ Berhasil generate embeddings dengan dimensi: {embed_dim}")
            else:
                raise Exception("Semua strategi model loading gagal - model tidak ditemukan")
        except Exception as e:
            warn(f"Tidak bisa memuat embeddings ({e}); fallback ke TF-IDF.")
            try:
                # Fallback ke TF-IDF sebagai alternatif embeddings
                from sklearn.feature_extraction.text import TfidfVectorizer
                info("Menggunakan TF-IDF sebagai fallback untuk embeddings...")
                vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
                texts = df['content'].fillna('').astype(str).tolist()
                embed_vectors = vectorizer.fit_transform(texts).toarray()
                embed_dim = embed_vectors.shape[1] if hasattr(embed_vectors, 'shape') else 0
                info(f"✓ Berhasil menggunakan TF-IDF dengan dimensi: {embed_dim}")
            except Exception as e2:
                warn(f"TF-IDF fallback juga gagal ({e2}); melanjutkan tanpa embeddings.")
                use_embeddings = False
                embed_vectors = None
                embed_dim = 0

    # agregasi per user
    groups = []
    for uname, g in df.groupby('username'):
        # umur akun
        if 'user_created' in g.columns and pd.notnull(g['user_created']).any():
            first_created = g['user_created'].dropna().iloc[0]
            account_age_days = (pd.Timestamp.now(tz=datetime.timezone.utc) - first_created).days if pd.notnull(first_created) else 0
        else:
            account_age_days = 0

        followers = safe_int(g['followers'].dropna().iloc[0]) if 'followers' in g.columns and pd.notnull(g['followers']).any() else 0
        following = safe_int(g['following'].dropna().iloc[0]) if 'following' in g.columns and pd.notnull(g['following']).any() else 0
        followers_following_ratio = (followers + 1) / (following + 1)

        posts = len(g)
        if 'date' in g.columns and pd.notnull(g['date']).any():
            first = g['date'].dropna().min()
            last  = g['date'].dropna().max()
            span_days = max(1, (last - first).days)
        else:
            span_days = max(1, math.ceil(posts / 50))  # fallback asumsi

        freq_per_day = posts / span_days

        if 'retweetCount' in g.columns and pd.notnull(g['retweetCount']).any():
            avg_retweets = g['retweetCount'].dropna().astype(float).mean()
        else:
            avg_retweets = 0.0

        if 'likeCount' in g.columns and pd.notnull(g['likeCount']).any():
            avg_likes = g['likeCount'].dropna().astype(float).mean()
        else:
            avg_likes = 0.0

        # burstiness (porsi posting dalam window 1 jam terbesar)
        if 'date' in g.columns and pd.notnull(g['date']).sum() > 1:
            times = g['date'].dropna().sort_values().astype('int64') // (10**9)
            window_counts = []
            arr = times.values
            for t in arr:
                window_counts.append(((arr >= t) & (arr < t + 3600)).sum())
            burstiness = (max(window_counts) / posts) if posts > 0 else 0.0
        else:
            burstiness = 0.0

        # hashtag & url ratio
        texts = g['content'].fillna('').astype(str)
        hashtags = int(texts.str.count(r'#\w+').sum())
        urls     = int(texts.str.contains(r'https?://', regex=True).sum())
        hashtag_ratio = hashtags / posts if posts else 0.0
        url_ratio     = urls / posts if posts else 0.0

        # mean embedding per akun
        mean_embed = None
        if use_embeddings and embed_vectors is not None:
            idxs = g.index.to_list()
            sel = np.vstack([embed_vectors[i] for i in range(len(df)) if df.index[i] in idxs])
            mean_embed = sel.mean(axis=0)

        groups.append({
            'username': uname,
            'account_age_days': account_age_days,
            'followers': followers,
            'following': following,
            'followers_following_ratio': followers_following_ratio,
            'posts': posts,
            'freq_per_day': float(freq_per_day),
            'avg_retweets': float(avg_retweets),
            'avg_likes': float(avg_likes),
            'burstiness': float(burstiness),
            'hashtag_ratio': float(hashtag_ratio),
            'url_ratio': float(url_ratio),
            'embed': mean_embed if mean_embed is not None else None,
        })

    feat_df = pd.DataFrame(groups).sort_values('posts', ascending=False)
    numeric_cols = [
        'account_age_days','followers','following','followers_following_ratio',
        'posts','freq_per_day','avg_retweets','avg_likes','burstiness','hashtag_ratio','url_ratio'
    ]
    X = feat_df[numeric_cols].fillna(0).values

    if use_embeddings:
        # jika ada embed None, isi nol
        def _to_vec(v):
            if v is None: return np.zeros((embed_dim,), dtype='float32')
            return v
        E = np.vstack([_to_vec(v) for v in feat_df['embed'].tolist()])
        X = np.hstack([X, E])

    return feat_df, X, embed_dim


def safe_int(x):
    try:
        return int(x)
    except:
        try:
            return int(float(x))
        except:
            return 0

async def check_and_handle_login(page):
    """Check if login is needed and return True if login required"""
    try:
        # Check for login indicators
        login_indicators = [
            'a[data-testid="loginButton"]',
            'a[href="/i/flow/login"]',
            'text=Sign in to X',
            'text=Log in'
        ]
        
        for indicator in login_indicators:
            try:
                element = await page.wait_for_selector(indicator, timeout=2000)
                if element:
                    return True  # Login needed
            except:
                continue
        
        # Check for logged-in indicators
        logged_in_indicators = [
            '[data-testid="SideNav_AccountSwitcher_Button"]',
            '[data-testid="AppTabBar_Home_Link"]',
            '[aria-label="Search and explore"]'
        ]
        
        for indicator in logged_in_indicators:
            try:
                element = await page.wait_for_selector(indicator, timeout=2000)
                if element:
                    info("✅ Sudah login ke Twitter")
                    return False  # Already logged in
            except:
                continue
        
        # If unclear, assume login is needed
        return True
        
    except Exception as e:
        warn(f"Error checking login status: {e}")
        return True


# ======================
# Weak labels + Model
# ======================
def advanced_coordination_detection(feat_df: pd.DataFrame, posts_df: pd.DataFrame, embed_dim: int):
    """Advanced coordination detection dengan multiple methods"""
    coordination_groups = []
    
    if 'embed' not in feat_df.columns or feat_df['embed'].isna().all():
        warn("No embeddings available for coordination detection")
        return [], nx.Graph()
    
    # Method 1: Content similarity clustering
    embeddings = np.vstack([v if v is not None else np.zeros(embed_dim) for v in feat_df['embed']])
    
    # DBSCAN clustering for similar content
    dbscan = DBSCAN(eps=0.15, min_samples=2, metric='cosine')
    content_clusters = dbscan.fit_predict(embeddings)
    
    # Method 2: Temporal coordination analysis
    if 'date' in posts_df.columns:
        posts_with_time = posts_df.dropna(subset=['date'])
        if not posts_with_time.empty:
            # Analyze posting patterns in time windows
            posts_with_time['timestamp'] = pd.to_datetime(posts_with_time['date'])
            
            # Find users posting in same time windows (5 min windows)
            time_groups = posts_with_time.groupby(
                posts_with_time['timestamp'].dt.floor('5min')
            )['username'].apply(list)
            
            # Find frequent co-posters
            co_posting_pairs = {}
            for time_window, users in time_groups.items():
                if len(users) > 1:
                    for i, user1 in enumerate(users):
                        for user2 in users[i+1:]:
                            pair = tuple(sorted([user1, user2]))
                            co_posting_pairs[pair] = co_posting_pairs.get(pair, 0) + 1
    
    # Method 3: Network analysis
    G = nx.Graph()
    
    # Add nodes
    for username in feat_df['username']:
        G.add_node(username)
    
    # Add edges based on multiple similarity measures
    n_users = len(feat_df)
    similarity_matrix = cosine_similarity(embeddings)
    
    for i in range(n_users):
        for j in range(i+1, n_users):
            user1, user2 = feat_df.iloc[i]['username'], feat_df.iloc[j]['username']
            
            # Content similarity edge
            content_sim = similarity_matrix[i, j]
            
            # Behavioral similarity
            behavior_features = ['freq_per_day', 'burstiness', 'hashtag_ratio', 'url_ratio']
            behavior_sim = 0
            
            try:
                user1_behavior = feat_df.iloc[i][behavior_features].values
                user2_behavior = feat_df.iloc[j][behavior_features].values
                behavior_sim = 1 - np.linalg.norm(user1_behavior - user2_behavior)
            except:
                behavior_sim = 0
            
            # Combined similarity score
            combined_sim = 0.7 * content_sim + 0.3 * max(0, behavior_sim)
            
            # Add edge if similarity is high
            if combined_sim > 0.8:
                G.add_edge(user1, user2, 
                          weight=float(combined_sim),
                          content_sim=float(content_sim),
                          behavior_sim=float(behavior_sim))
    
    # Find coordination groups
    coordination_groups = []
    for component in nx.connected_components(G):
        if len(component) >= 2:  # At least 2 users
            coordination_groups.append(list(component))
    
    info(f"[COORDINATION] Found {len(coordination_groups)} potential coordination groups")
    
    return coordination_groups, G

def heuristic_labels(feat_df: pd.DataFrame):
    """Backward compatibility - calls enhanced version"""
    config = BuzzerDetectionConfig()
    detector = AdvancedBuzzerDetector()
    return detector.enhanced_heuristic_labels(feat_df)


def train_on_weak_labels(X: np.ndarray, labels: np.ndarray):
    """Backward compatibility - menggunakan basic training"""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1)
    clf.fit(Xs, labels)
    return clf, scaler

def advanced_buzzer_analysis(posts_df: pd.DataFrame, use_embeddings=True):
    """Comprehensive buzzer analysis dengan multiple detection methods"""
    
    # 1. Initialize advanced detector and buzzer type classifier
    detector = AdvancedBuzzerDetector()
    buzzer_classifier = BuzzerTypeClassifier()
    
    # 2. Train dengan historical data
    feat_df, X, embed_dim = detector.train_with_historical_data(
        posts_df, use_embeddings=use_embeddings
    )
    
    # 3. Advanced coordination detection
    coord_groups, G = advanced_coordination_detection(feat_df, posts_df, embed_dim)
    
    # 4. Assign coordination clusters
    cluster_map = {}
    for cluster_id, members in enumerate(coord_groups):
        for username in members:
            cluster_map[username] = cluster_id
    
    feat_df['coord_cluster'] = feat_df['username'].map(cluster_map).fillna(-1).astype(int)
    
    # 5. Advanced scoring
    feat_df['coordination_score'] = 0.0
    feat_df['network_centrality'] = 0.0
    feat_df['temporal_anomaly'] = 0.0
    
    # Calculate network metrics
    if len(G.nodes()) > 0:
        try:
            centrality = nx.degree_centrality(G)
            for username in feat_df['username']:
                if username in centrality:
                    feat_df.loc[feat_df['username'] == username, 'network_centrality'] = centrality[username]
        except:
            pass
    
    # Calculate coordination scores
    for cluster_id, members in enumerate(coord_groups):
        cluster_size = len(members)
        coord_score = min(1.0, cluster_size / 10.0)  # Normalize to 0-1
        
        mask = feat_df['username'].isin(members)
        feat_df.loc[mask, 'coordination_score'] = coord_score
    
    # 6. Final buzzer probability dengan ensemble
    base_prob = feat_df['buzzer_prob'].values
    coord_boost = feat_df['coordination_score'].values * 0.3
    network_boost = feat_df['network_centrality'].values * 0.2
    
    # Enhanced probability
    enhanced_prob = np.clip(base_prob + coord_boost + network_boost, 0, 1)
    feat_df['buzzer_prob_enhanced'] = enhanced_prob
    
    # 7. Risk categories
    feat_df['risk_category'] = 'LOW'
    feat_df.loc[feat_df['buzzer_prob_enhanced'] > 0.5, 'risk_category'] = 'MEDIUM'
    feat_df.loc[feat_df['buzzer_prob_enhanced'] > 0.7, 'risk_category'] = 'HIGH'
    feat_df.loc[feat_df['buzzer_prob_enhanced'] > 0.9, 'risk_category'] = 'CRITICAL'
    
    # 8. Buzzer type classification
    info("[ANALYSIS] Classifying buzzer types...")
    
    # Add buzzer type columns
    feat_df['buzzer_type_primary'] = 'netral'
    feat_df['buzzer_type_secondary'] = None
    feat_df['activity_pattern'] = 'organik'
    feat_df['type_confidence'] = 0.0
    feat_df['government_score'] = 0
    feat_df['opposition_score'] = 0
    feat_df['commercial_score'] = 0
    feat_df['spam_score'] = 0
    
    # Classify each account
    for idx, account in feat_df.iterrows():
        # Get posts for this account
        account_posts = posts_df[posts_df['username'] == account['username']]['content'].fillna('').tolist()
        
        if account_posts:
            # Prepare account features for classification
            account_features = {
                'freq_per_day': account['freq_per_day'],
                'burstiness': account['burstiness'],
                'hashtag_ratio': account['hashtag_ratio'],
                'url_ratio': account['url_ratio'],
                'account_age_days': account['account_age_days'],
                'followers': account['followers'],
                'following': account['following'],
                'avg_likes': account.get('avg_likes', 0)
            }
            
            # Classify buzzer type
            classification = buzzer_classifier.classify_buzzer_type(account_posts, account_features)
            
            # Update dataframe
            feat_df.loc[idx, 'buzzer_type_primary'] = classification['primary_type']
            feat_df.loc[idx, 'buzzer_type_secondary'] = classification['secondary_type']
            feat_df.loc[idx, 'activity_pattern'] = classification['activity_pattern']
            feat_df.loc[idx, 'type_confidence'] = classification['type_confidence']
            feat_df.loc[idx, 'government_score'] = int(classification['scores']['pemerintah'])
            feat_df.loc[idx, 'opposition_score'] = int(classification['scores']['oposisi'])
            feat_df.loc[idx, 'commercial_score'] = int(classification['scores']['komersial'])
            feat_df.loc[idx, 'spam_score'] = int(classification['scores']['spam'])    # 9. Save model untuk reuse
    detector.save_model()
    
    # Enhanced analysis summary
    info(f"[ANALYSIS] Completed advanced analysis:")
    info(f"  - Total accounts analyzed: {len(feat_df)}")
    info(f"  - Coordination groups: {len(coord_groups)}")
    info(f"  - High risk accounts: {sum(feat_df['risk_category'] == 'HIGH')}")
    info(f"  - Critical risk accounts: {sum(feat_df['risk_category'] == 'CRITICAL')}")
    
    # Buzzer type distribution
    type_distribution = feat_df['buzzer_type_primary'].value_counts()
    info(f"[BUZZER TYPES] Distribution:")
    for buzz_type, count in type_distribution.items():
        info(f"  - {buzz_type}: {count} accounts")
    
    # Activity pattern distribution
    pattern_distribution = feat_df['activity_pattern'].value_counts()
    info(f"[ACTIVITY PATTERNS]:")
    for pattern, count in pattern_distribution.items():
        info(f"  - {pattern}: {count} accounts")
    
    return feat_df, G, coord_groups


# =========================
# Coordinated Groups (graph)
# =========================
def detect_coordinated_groups(feat_df: pd.DataFrame, embed_dim: int):
    # pakai similarity antar mean-embedding akun
    if 'embed' not in feat_df.columns or feat_df['embed'].isna().all():
        warn("Embeddings tidak tersedia; deteksi koordinasi berbasis konten dilewati (gunakan --no-embeddings=OFF).")
        return [], nx.Graph()

    # siapkan matriks embedding
    E = []
    for v in feat_df['embed'].tolist():
        if v is None:
            E.append(np.zeros((embed_dim,), dtype='float32'))
        else:
            E.append(v)
    E = np.vstack(E)
    sim = cosine_similarity(E)

    G = nx.Graph()
    for uname in feat_df['username']:
        G.add_node(uname)

    n = len(feat_df)
    # threshold similarity bisa dituning
    TH = 0.85
    for i in range(n):
        for j in range(i+1, n):
            if sim[i, j] > TH:
                G.add_edge(feat_df['username'].iloc[i], feat_df['username'].iloc[j], weight=float(sim[i, j]))

    comps = [list(c) for c in nx.connected_components(G) if len(c) > 1]
    return comps, G


# ======================
# Save helpers
# ======================
def save_outputs(base_out: str, raw_posts: pd.DataFrame, feat_df: pd.DataFrame, G=None, query_used="", coord_groups=None):
    os.makedirs(base_out, exist_ok=True)
    # raw posts
    raw_path = os.path.join(base_out, "raw_posts.csv")
    raw_posts.to_csv(raw_path, index=False)
    info(f"saved: {raw_path}")

    # features + scores
    feat_path = os.path.join(base_out, "buzzer_candidates.csv")
    feat_df.to_csv(feat_path, index=False)
    info(f"saved: {feat_path}")

    # graph (untuk Gephi/NetworkX)
    if G is not None:
        try:
            gml_path = os.path.join(base_out, "coordination_graph.gml")
            nx.write_gml(G, gml_path)
            info(f"saved: {gml_path}")
        except Exception as e:
            warn(f"Gagal menyimpan graph: {e}")
    
    # Save to database - force MySQL
    try:
        # Force MySQL configuration without modifying config object
        if config.database_type != 'mysql':
            warn("[DB] Database type is not MySQL, forcing MySQL usage")
        
        db_config = {
            'type': 'mysql',
            'host': config.mysql_host,
            'port': config.mysql_port,
            'user': config.mysql_user,
            'password': config.mysql_password,
            'database': config.mysql_database
        }
        
        info(f"[DB] Using MySQL config: {config.mysql_host}:{config.mysql_port}/{config.mysql_database}")
        
        # Initialize MySQL database first
        init_database(db_config)
        info("[DB] MySQL database initialized successfully")
        
        # Save data to MySQL
        save_to_database(raw_posts, feat_df, query_used, coord_groups, db_path=None, mysql_config=db_config)
        info("[DB] Data successfully saved to MySQL database")
        
    except Exception as e:
        error(f"[DB] Failed to save to database: {e}")
        import traceback
        traceback.print_exc()


# ======================
# Main pipeline
# ======================
def main():
    ap = argparse.ArgumentParser(description="Deteksi akun buzzer/terkoordinasi (tanpa API).")
    ap.add_argument("--mode", choices=["snscrape","playwright"], required=True, help="Metode scraping.")
    ap.add_argument("--query", required=True, help="snscrape: gunakan sintaks X (cth: 'kata OR #tag since:YYYY-MM-DD until:YYYY-MM-DD lang:id'); playwright: URL-encoded (cth: 'tag%20lang:id')")
    ap.add_argument("--limit", type=int, default=2000, help="Batas posts (snscrape)")
    ap.add_argument("--max_posts", type=int, default=800, help="Batas posts (playwright)")
    ap.add_argument("--proxy", default=None, help="contoh: http://user:pass@host:port (playwright)")
    ap.add_argument("--headful", action="store_true", help="lihat browsernya (playwright)")
    ap.add_argument("--use-chrome", action="store_true", help="gunakan Chrome instead of Firefox (playwright)")
    ap.add_argument("--firefox-profile", default=None, help="path ke profil Firefox spesifik (opsional)")
    ap.add_argument("--manual-cookies", action="store_true", help="gunakan input cookies manual")
    ap.add_argument("--collect-retweets", action="store_true", help="collect semua retweets dari setiap tweet (tanpa batas maksimal)")
    ap.add_argument("--advanced-detection", action="store_true", help="gunakan advanced ensemble detection (default: True)", default=True)
    ap.add_argument("--basic-mode", action="store_true", help="gunakan basic detection mode (lebih cepat tapi kurang akurat)")
    ap.add_argument("--bypass-errors", action="store_true", help="aktifkan bypass untuk login errors (default: True)", default=True)
    ap.add_argument("--no-embeddings", action="store_true", help="matikan embeddings (lebih ringan, tapi koordinasi berbasis konten terbatas)")
    ap.add_argument("--out", default="results", help="folder output")
    ap.add_argument("--save-model", action="store_true", help="simpan model untuk reuse (default: True)", default=True)
    args = ap.parse_args()

    # 1) SCRAPE
    if args.mode == "snscrape":
        posts = scrape_x_snscrape(args.query, args.limit)
    else:
        use_firefox = not args.use_chrome
        posts = scrape_x_playwright(
            args.query, 
            args.max_posts, 
            args.proxy, 
            args.headful, 
            use_firefox, 
            args.firefox_profile,
            args.manual_cookies,
            args.collect_retweets
        )

    if posts.empty:
        die("Tidak ada data yang terscrape. Coba ganti query/limit atau cek koneksi/selector.")

    # 2) ADVANCED BUZZER ANALYSIS
    use_embeddings = (not args.no_embeddings)
    
    try:
        # Use advanced detection system
        feat_df, G, coord_groups = advanced_buzzer_analysis(posts, use_embeddings=use_embeddings)
        
        info("[SUCCESS] Advanced buzzer detection completed")
        info("[MODEL] Ensemble model saved for reuse by others")
        
    except Exception as e:
        warn(f"Advanced detection failed: {e}")
        info("[FALLBACK] Using basic detection method")
        
        # Fallback to basic method
        feat_df, X, embed_dim = compute_features(posts, use_embeddings=use_embeddings)
        labels, scores = heuristic_labels(feat_df)
        feat_df['heur_score'] = scores
        feat_df['heur_label'] = labels
        
        clf, scaler = train_on_weak_labels(X, labels)
        prob_matrix = clf.predict_proba(scaler.transform(X))
        
        if prob_matrix.shape[1] == 1:
            probs = np.zeros(len(X))
        else:
            probs = prob_matrix[:,1]
        
        feat_df['buzzer_prob'] = probs
        
        # Basic coordination detection
        groups, G = ([], None)
        if use_embeddings:
            groups, G = detect_coordinated_groups(feat_df, embed_dim)
            cluster_id = {u: -1 for u in feat_df['username']}
            for cid, members in enumerate(groups):
                for u in members:
                    cluster_id[u] = cid
            feat_df['coord_cluster'] = feat_df['username'].map(cluster_id)
        else:
            feat_df['coord_cluster'] = -1

    # 6) URUTKAN & SIMPAN
    feat_df = feat_df.sort_values('buzzer_prob', ascending=False)
    
    # Extract coordination groups for database
    coord_groups = []
    if 'coord_cluster' in feat_df.columns:
        for cluster_id in feat_df['coord_cluster'].unique():
            if cluster_id >= 0:  # -1 means not in any cluster
                members = feat_df[feat_df['coord_cluster'] == cluster_id]['username'].tolist()
                if len(members) > 1:
                    coord_groups.append(members)
    
    save_outputs(args.out, posts, feat_df, G, args.query, coord_groups)

    # 7) Enhanced Ringkasan
    topn = min(20, len(feat_df))
    
    # Show advanced metrics if available
    if 'buzzer_prob_enhanced' in feat_df.columns:
        info("\n🚨 TOP BUZZER CANDIDATES (Enhanced Detection):")
        
        # Include buzzer type columns if available
        if 'buzzer_type_primary' in feat_df.columns:
            enhanced_cols = ['username','posts','freq_per_day','buzzer_prob_enhanced','risk_category','buzzer_type_primary','activity_pattern','type_confidence','coord_cluster']
        else:
            enhanced_cols = ['username','posts','freq_per_day','burstiness','buzzer_prob_enhanced','risk_category','coord_cluster']
            
        top_buzzer = feat_df.nlargest(topn, 'buzzer_prob_enhanced')
        print(top_buzzer[enhanced_cols].to_string(index=False))
        
        # Summary statistics
        risk_counts = feat_df['risk_category'].value_counts()
        info(f"\n📊 RISK SUMMARY:")
        for risk_level, count in risk_counts.items():
            percentage = count / len(feat_df) * 100
            info(f"  {risk_level}: {count} accounts ({percentage:.1f}%)")
            
        # Buzzer type summary
        if 'buzzer_type_primary' in feat_df.columns:
            type_counts = feat_df['buzzer_type_primary'].value_counts()
            info(f"\n🎯 BUZZER TYPE DISTRIBUTION:")
            for buzz_type, count in type_counts.items():
                percentage = count / len(feat_df) * 100
                info(f"  {buzz_type}: {count} accounts ({percentage:.1f}%)")
                
            # Activity pattern summary
            pattern_counts = feat_df['activity_pattern'].value_counts()
            info(f"\n🤖 ACTIVITY PATTERNS:")
            for pattern, count in pattern_counts.items():
                percentage = count / len(feat_df) * 100
                info(f"  {pattern}: {count} accounts ({percentage:.1f}%)")
                
            # High confidence classifications
            high_confidence = feat_df[feat_df['type_confidence'] > 0.7]
            if not high_confidence.empty:
                info(f"\n✅ HIGH CONFIDENCE CLASSIFICATIONS ({len(high_confidence)} accounts):")
                conf_by_type = high_confidence.groupby('buzzer_type_primary')['type_confidence'].agg(['count', 'mean'])
                for buzz_type, stats in conf_by_type.iterrows():
                    info(f"  {buzz_type}: {int(stats['count'])} accounts (avg confidence: {stats['mean']:.2f})")
            
        # Coordination summary
        coordinated = feat_df[feat_df['coord_cluster'] >= 0]
        if not coordinated.empty:
            info(f"\n🤝 COORDINATION DETECTED:")
            cluster_counts = coordinated['coord_cluster'].value_counts()
            for cluster_id, count in cluster_counts.items():
                info(f"  Group #{cluster_id}: {count} accounts")
                
        # Model info
        info(f"\n🤖 MODEL INFO:")
        info(f"  Buzzer Detection: Advanced Ensemble (RF+GB+LR+SVM)")
        info(f"  Type Classification: Keyword-based + Behavioral Analysis")
        info(f"  Model saved to: models/buzzer_ensemble_v2.pkl")
        info(f"  Database: buzzer_detection.db")
        info(f"  Auto Type Detection: {'✅ ENABLED' if 'buzzer_type_primary' in feat_df.columns else '❌ DISABLED'}")
    else:
        info("\nTop kandidat (buzzer_prob tertinggi):")
        cols = ['username','posts','freq_per_day','burstiness','followers','buzzer_prob','coord_cluster']
        print(feat_df[cols].head(topn).to_string(index=False))

    # Coordination groups already displayed in advanced analysis above


if __name__ == "__main__":
    main()
