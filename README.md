# 🔍 BuzzerDetector – Advanced Social Media Coordination & Buzzer Analysis Tool

**BuzzerDetector** is an AI-powered tool designed to identify potential coordinated behavior ("buzzers") on social media platforms such as X (formerly Twitter).  
It uses machine learning, behavioral analytics, and semantic embeddings to detect automated or organized posting patterns, particularly around hashtags or topics.

> ⚠️ **Disclaimer:**  
> This tool is intended **only for authorized penetration testing, academic research, internal audits, or official media investigations**.  
> **Unauthorized surveillance or misuse** of this tool may violate local laws and platform terms of service.  
> Always ensure you have **explicit consent** from the data owners or platforms before running it.

---

## 🚀 Features

### 🧠 Advanced AI Detection
- **Enhanced Ensemble Model** combining multiple algorithms:
  - RandomForest, GradientBoosting, LogisticRegression, SVM
  - XGBoost, LightGBM (optional advanced models)
  - Neural Networks with adaptive learning
- **Semantic Understanding** using `sentence-transformers/all-MiniLM-L6-v2`
- **Multi-dimensional Analysis**:
  - Behavioral anomaly scoring
  - Temporal pattern analysis
  - Engagement pattern detection
  - Network behavior analysis
  - Content sophistication analysis

### 🌐 Multi-Mode Data Collection
- **Playwright Mode** — Real browser automation with anti-detection
  - Firefox/Chromium support with custom profiles
  - Cookie management and session persistence
  - Proxy support for enhanced privacy
- **SNScrape Mode** — Fast, API-free Twitter scraping
  - No rate limits or API keys required
  - Historical data collection support

### 🧩 Comprehensive Analysis
- **Buzzer Type Classification**:
  - Government/Pro-government accounts
  - Opposition/Political critics  
  - Commercial/Marketing accounts
  - Spam/Bot accounts
- **Coordination Detection**:
  - Content similarity clustering (DBSCAN)
  - Temporal coordination analysis
  - Network analysis with graph theory
- **Activity Pattern Recognition**:
  - Organic vs Artificial behavior
  - Hybrid account detection
  - Burstiness and frequency analysis

### 💾 Robust Database Integration
- **MySQL Primary Support** with automatic schema management
- **SQLite Fallback** for development and testing
- **Complete Data Model**:
  - Account profiles with AI analysis scores
  - Post content with metadata
  - Coordination groups and relationships
  - Historical training data management

### 🧬 Advanced Machine Learning Pipeline
- **Accumulative Learning** with 90-day training history
- **Adaptive Threshold Tuning** based on performance feedback
- **Feature Engineering**:
  - 11 behavioral features + embeddings
  - Feature selection and scaling
  - Cross-validation with performance tracking
- **Model Versioning** and automatic updates

---

## 📋 Requirements

- **Python 3.8+**
- **MySQL Server** (recommended) or SQLite
- **4GB+ RAM** for embedding models
- **Internet connection** for model downloads and scraping

### Optional but Recommended:
- **CUDA-capable GPU** for faster embeddings processing
- **VPN/Proxy** for enhanced privacy during scraping

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/mamangzed/BuzzerDetector.git
cd BuzzerDetector
```

### 2. Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Install Browser for Playwright
```bash
# Install browser binaries
playwright install

# For Firefox specifically (recommended for anti-detection):
playwright install firefox
```

### 4. Database Setup

#### Option A: MySQL (Recommended)
```bash
# Install MySQL Server
# Windows: Download from https://dev.mysql.com/downloads/mysql/
# Ubuntu: sudo apt install mysql-server
# macOS: brew install mysql

# Create database
mysql -u root -p
CREATE DATABASE buzzer_detection;
CREATE USER 'buzzer_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON buzzer_detection.* TO 'buzzer_user'@'localhost';
FLUSH PRIVILEGES;
```

#### Option B: SQLite (Development)
No additional setup required - SQLite file will be created automatically.

### 5. Configuration
Create `.env` file in project root:
```bash
cp .env.example .env
# Edit .env with your settings
```

**Example .env configuration:**
```env
# Database Configuration
DATABASE_TYPE=mysql
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=buzzer_user
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=buzzer_detection

# AI Configuration
EMBED_MODEL_NAME=all-MiniLM-L6-v2
USE_EMBEDDINGS=true
AI_ANALYSIS_ENABLED=true

# Scraping Configuration
DEFAULT_SCRAPING_MODE=playwright
USE_FIREFOX=true
HEADFUL_MODE=false

# Performance Configuration
N_JOBS=-1
RANDOM_STATE=42
```

---

## 🎯 Usage

### Basic Usage

#### 1. SNScrape Mode (Fast)
```bash
# Basic query
python main.py --mode snscrape --query "buzzer OR astroturf lang:id" --limit 1000

# Date range query  
python main.py --mode snscrape --query "#pilpres2024 since:2024-01-01 until:2024-12-31 lang:id" --limit 2000

# Save to specific output
python main.py --mode snscrape --query "kredit macet" --limit 1500 --out results_kredit
```

#### 2. Playwright Mode (Comprehensive)
```bash
# Browser-based scraping
python main.py --mode playwright --query "buzzer%20detection" --max_posts 800

# Headful mode (visible browser)
python main.py --mode playwright --query "astroturf" --max_posts 500 --headful

# Firefox with custom profile
python main.py --mode playwright --query "coordination" --max_posts 1000 --use_firefox
```

### Advanced Options

#### Disable Embeddings (Faster)
```bash
python main.py --mode snscrape --query "test query" --limit 500 --no-embeddings
```

#### Custom Output Directory
```bash
python main.py --mode playwright --query "analysis" --max_posts 600 --out custom_results
```

#### Proxy Support
```bash
python main.py --mode playwright --query "research" --max_posts 400 --proxy "http://proxy:8080"
```

### Example Queries

```bash
# Political buzzers
python main.py --mode snscrape --query "#pilpres2024 OR #pemilu2024 lang:id" --limit 3000 --out political_analysis

# Commercial buzzers
python main.py --mode playwright --query "promo discount sale" --max_posts 1000 --out commercial_analysis

# Regional analysis
python main.py --mode snscrape --query "jakarta OR surabaya OR medan lang:id" --limit 2000 --out regional_analysis

# Trending topic analysis
python main.py --mode playwright --query "%23trending%20%23viral" --max_posts 800 --out trending_analysis
```

---

## 📊 Output & Results

### Generated Files

1. **`results/buzzer_candidates.csv`** - Detailed account analysis
   - Account information and metrics
   - AI analysis scores and classifications
   - Buzzer probability and risk categorization
   - Behavioral patterns and anomaly scores

2. **`results/raw_posts.csv`** - Raw collected posts
   - Post content and metadata
   - User engagement metrics  
   - Timestamps and source information

3. **`results/coordination_graph.gml`** - Network analysis
   - Graph format for visualization tools
   - Account relationships and similarities
   - Coordination groups and clusters

4. **`models/buzzer_ensemble_v2.pkl`** - Trained ML model
   - Persistent model for reuse
   - Performance history and metrics
   - Feature scaling and selection data

### Key Metrics Analyzed

#### Account-Level Features:
- **Behavioral**: Posting frequency, burstiness, hashtag usage
- **Network**: Followers/following ratios, engagement patterns  
- **Temporal**: Account age, posting time patterns
- **Content**: URL sharing, content diversity, language patterns
- **AI Scores**: Multi-dimensional anomaly detection

#### Buzzer Classifications:
- **Primary Types**: Government, Opposition, Commercial, Spam, Neutral
- **Activity Patterns**: Organic, Artificial, Hybrid
- **Risk Categories**: LOW, MEDIUM, HIGH, CRITICAL
- **Confidence Scores**: Statistical confidence in classifications

### Database Schema

#### Main Tables:
- **`buzzer_accounts`**: Complete account profiles with AI analysis
- **`posts`**: Individual posts with content analysis  
- **`coordination_groups`**: Detected coordination clusters
- **`ai_analysis_details`**: Detailed AI analysis results (SQLite only)

---

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_TYPE` | `sqlite3` | Database type: `mysql` or `sqlite3` |
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `USE_EMBEDDINGS` | `true` | Enable semantic embeddings |
| `AI_ANALYSIS_ENABLED` | `true` | Enable advanced AI analysis |
| `DEFAULT_SCRAPING_MODE` | `playwright` | Default scraping method |
| `USE_FIREFOX` | `true` | Use Firefox for Playwright |
| `HEADFUL_MODE` | `false` | Show browser window |
| `N_JOBS` | `-1` | CPU cores for ML (-1 = all) |
| `RANDOM_STATE` | `42` | Random seed for reproducibility |

### Model Configuration

The system uses an adaptive configuration that automatically tunes thresholds based on:
- Historical performance data
- Dataset characteristics  
- Detection accuracy feedback
- False positive/negative rates

### Performance Tuning

For large datasets (>10K posts):
```env
N_JOBS=-1              # Use all CPU cores
USE_EMBEDDINGS=true    # Enable for better accuracy
AI_ANALYSIS_ENABLED=true
```

For speed-optimized runs:
```env
USE_EMBEDDINGS=false   # Skip embeddings
AI_ANALYSIS_ENABLED=false
DEFAULT_SCRAPING_MODE=snscrape
```

---

## 🔬 Technical Details

### Machine Learning Architecture

#### Enhanced Ensemble Model:
```
VotingClassifier (Soft Voting)
├── RandomForest (n_estimators=150, max_depth=10)
├── GradientBoosting (n_estimators=100, learning_rate=0.1)  
├── LogisticRegression (C=1.0, solver=liblinear)
├── SVM (kernel=rbf, probability=True)
├── XGBoost (optional, n_estimators=100)
├── LightGBM (optional, n_estimators=100)
├── Neural Network (optional, layers=[100,50])
└── ExtraTrees (optional, n_estimators=150)
```

#### Feature Engineering Pipeline:
1. **Behavioral Features** (11 dimensions):
   - Account age, follower metrics, posting frequency
   - Burstiness coefficient, hashtag/URL ratios
   - Engagement metrics (likes, retweets, replies)

2. **Semantic Features** (384 dimensions):
   - Sentence-BERT embeddings via `all-MiniLM-L6-v2`
   - Mean pooling across user's posts
   - Cosine similarity for coordination detection

3. **Advanced Features** (15+ dimensions):
   - Content sophistication scores
   - Sentiment analysis metrics
   - Writing style characteristics  
   - Temporal anomaly indicators

#### Multi-Dimensional Scoring:
```
Final Score = 0.30×Behavioral + 0.25×Temporal + 0.20×Engagement + 0.15×Network + 0.10×Content
```

### Coordination Detection Algorithm

#### Method 1: Content Similarity Clustering
```python
# DBSCAN clustering on embeddings
dbscan = DBSCAN(eps=0.15, min_samples=2, metric='cosine')
content_clusters = dbscan.fit_predict(embeddings)
```

#### Method 2: Temporal Pattern Analysis  
- 5-minute time window analysis
- Co-posting frequency calculation
- Burst pattern detection

#### Method 3: Network Analysis
- Graph construction with similarity edges
- Connected component analysis  
- Centrality measures for influence ranking

### Performance Optimization

#### Caching Strategy:
- **Model Persistence**: Trained models saved as `.pkl` files
- **Embedding Cache**: Pre-computed embeddings stored per user
- **Feature Cache**: Behavioral features cached across runs

#### Memory Management:
- **Batch Processing**: Large datasets processed in chunks
- **Lazy Loading**: Models loaded only when needed
- **Garbage Collection**: Automatic cleanup of large objects

#### Parallel Processing:
- **Multi-threading**: Concurrent feature extraction
- **Vectorized Operations**: NumPy/Pandas optimizations  
- **GPU Acceleration**: Optional CUDA support for embeddings

---

## 📈 Performance Metrics

### Accuracy Benchmarks
- **Precision**: ~85-92% on manually labeled datasets
- **Recall**: ~80-88% for known buzzer accounts  
- **F1-Score**: ~83-90% balanced performance
- **AUC-ROC**: ~0.85-0.93 discrimination ability

### Speed Performance
| Dataset Size | SNScrape Mode | Playwright Mode | Analysis Time |
|--------------|---------------|-----------------|---------------|
| 500 posts    | ~30 seconds   | ~2-3 minutes    | ~45 seconds   |
| 2,000 posts  | ~2 minutes    | ~8-12 minutes   | ~3 minutes    |  
| 10,000 posts | ~8 minutes    | ~30-45 minutes  | ~12 minutes   |

### Resource Usage
- **RAM**: 2-8GB (depending on embedding model and dataset size)
- **CPU**: Scales with `N_JOBS` setting  
- **Storage**: ~50MB per 10K posts (database + models)
- **Network**: Minimal during analysis (only for model downloads)

---

## 🛠️ Troubleshooting

### Common Issues

#### 1. Playwright Installation Problems
```bash
# Fix: Reinstall with specific browser
playwright install --force firefox
playwright install --force chromium

# Check installation
playwright --version
```

#### 2. MySQL Connection Errors
```bash
# Check MySQL service
# Windows: services.msc -> MySQL
# Linux: sudo systemctl status mysql  
# macOS: brew services list | grep mysql

# Test connection
mysql -h localhost -u buzzer_user -p buzzer_detection
```

#### 3. Out of Memory Errors
```env
# Reduce embedding usage
USE_EMBEDDINGS=false

# Limit dataset size  
--limit 1000  # for snscrape
--max_posts 500  # for playwright
```

#### 4. Model Download Issues
```bash
# Manual model download
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
print('Model downloaded successfully')
"
```

#### 5. Permission Errors (Windows)
```bash
# Run as Administrator
# Or adjust antivirus settings for Python/browser executables
```

### Debug Mode

Enable detailed logging:
```bash
# Set environment variable
export LOG_LEVEL=DEBUG  # Linux/macOS
set LOG_LEVEL=DEBUG     # Windows CMD

# Or in .env file
LOG_LEVEL=debug
LOG_FORMAT=json
```

### Performance Issues

#### Slow Analysis:
1. **Disable embeddings**: `--no-embeddings`
2. **Reduce CPU usage**: Set `N_JOBS=1` in .env
3. **Use SQLite**: Set `DATABASE_TYPE=sqlite3`
4. **Limit data**: Use smaller `--limit` or `--max_posts`

#### High Memory Usage:
1. **Process in batches**: Split large queries
2. **Close other applications**: Free up RAM
3. **Use lighter models**: Consider different embedding models

---

## 🤝 Contributing

### Development Setup
```bash
# Clone repository  
git clone https://github.com/mamangzed/BuzzerDetector.git
cd BuzzerDetector

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # or dev_env\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy  # Additional dev tools

# Run tests
pytest tests/

# Format code
black main.py config.py
```

### Contributing Guidelines
1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)  
5. **Open** a Pull Request

### Code Standards
- **Black** formatting for Python code
- **Type hints** where possible
- **Docstrings** for all functions/classes
- **Unit tests** for new features
- **Performance benchmarks** for algorithmic changes

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Terms of Use
- ✅ **Permitted**: Research, education, authorized security testing
- ❌ **Prohibited**: Unauthorized surveillance, harassment, data theft
- ⚠️ **Required**: Explicit consent from data owners/platforms
- 📋 **Compliance**: Must follow local laws and platform ToS

---

## 📞 Support & Contact

### Issues & Bug Reports
- **GitHub Issues**: [Report bugs or request features](https://github.com/mamangzed/BuzzerDetector/issues)
- **Discussions**: [Community discussions and questions](https://github.com/mamangzed/BuzzerDetector/discussions)

### Documentation
- **Wiki**: [Detailed documentation and guides](https://github.com/mamangzed/BuzzerDetector/wiki)
- **API Reference**: [Complete API documentation](docs/api.md)
- **Examples**: [Usage examples and tutorials](examples/)

### Community
- **Email**: Contact project maintainers
- **Security Issues**: Report privately to security@domain.com

---

## 🙏 Acknowledgments

### Technologies Used
- **[scikit-learn](https://scikit-learn.org/)** - Machine learning framework
- **[Sentence Transformers](https://www.sbert.net/)** - Semantic text embeddings  
- **[Playwright](https://playwright.dev/)** - Modern web scraping
- **[NetworkX](https://networkx.org/)** - Graph analysis and visualization
- **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis

### Research References
- Social media manipulation detection methodologies
- Coordinated behavior analysis frameworks  
- Machine learning approaches to bot detection
- Network analysis for social media research

### Special Thanks
- Contributors and researchers in the field
- Open source community for tools and libraries
- Academic institutions for research methodologies

---

## 📋 Changelog

### Version 2.0.0 (Current)
- ✨ **Enhanced AI Analysis** with multi-dimensional scoring
- 🚀 **Advanced Ensemble Model** with XGBoost/LightGBM support  
- 🎯 **Buzzer Type Classification** (Government, Opposition, Commercial, Spam)
- 🔗 **Improved Coordination Detection** with graph analysis
- 💾 **MySQL Database Integration** with complete schema
- 🧠 **Accumulative Learning** with 90-day training history
- ⚡ **Performance Optimizations** and caching improvements

### Version 1.0.0 (Legacy)
- 🎉 **Initial Release** with basic buzzer detection
- 🔍 **SNScrape Integration** for Twitter data collection
- 📊 **Basic ML Pipeline** with RandomForest classifier
- 📁 **CSV Export** functionality  
- 🗄️ **SQLite Database** support

---

*Last Updated: October 6, 2025*  
*BuzzerDetector v2.0.0 - Advanced Social Media Analysis Tool* 