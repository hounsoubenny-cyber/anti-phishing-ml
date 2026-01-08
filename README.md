# 🛡️ Anti-Phishing ML

Machine Learning system for detecting phishing URLs using ensemble methods and advanced feature engineering.

## 📊 Performance Metrics

- **Accuracy**: 99.30%
- **F1-Score**: 0.99 (both classes)
- **Hamming Loss**: 0.007
- **Jaccard Score**: 0.986

### Confusion Matrix
```
[[28002   206]
 [  187 27831]]
```

## 🚀 Features

- **Ensemble Learning**: Stacking classifier combining XGBoost, HistGradientBoosting, and Random Forest
- **Smart Whitelist**: Pre-approved trusted domains for instant validation
- **IP Detection**: Automatic flagging of URLs with IP addresses
- **Bayesian Optimization**: Hyperparameter tuning for optimal performance
- **Disk Cache**: Fast predictions with intelligent caching
- **CLI Interface**: Interactive command-line tool

## 📋 Feature Engineering

The model analyzes 19 URL characteristics:

- URL length and domain entropy
- IP address presence
- Special characters (@, -, etc.)
- HTTPS usage
- Suspicious keywords (login, verify, secure, etc.)
- TLD analysis
- Subdomain count
- Query parameters
- Punycode detection

## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/anti-phishing-ml.git
cd anti-phishing-ml

# Install dependencies
pip install -r requirements.txt
```

## 📦 Requirements

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
imbalanced-learn>=0.9.0
scikit-optimize>=0.9.0
joblib>=1.1.0
diskcache>=5.4.0
tldextract>=3.1.0
```

## 💻 Usage

### CLI Mode

```bash
python main.py
```

**Available commands:**
- Enter any URL to analyze: `google.com` or `https://example.com`
- `help` - Display help information
- `change_seuil 0.7` - Modify detection threshold (0-1)
- `ls` or `history` - View command history
- `clear` - Clear cache
- `printia` - Display model information
- `train <path>` - Train new model with custom dataset
- `quit` or `q` - Exit

### Python API

```python
from train import AntiPhishingIA
from features import features_extractor

# Initialize model
ia = AntiPhishingIA(model_file='model.pkl', dataset_file='dataset.pkl')

# Predict single URL
url_features = features_extractor('https://example.com')
prediction = ia.predict([url_features])

print(prediction)
# Output:
# {
#   "predict": {0: "safe"},
#   "predict_proba": {0: {"phishing": 0.01, "safe": 0.99}},
#   "true_label": {}
# }
```

### Direct Prediction Function

```python
from main import predict

result = predict(ia, 'https://suspicious-site.tk', seuil=0.6)
print(result)
# Output: {"phishing": True, "prob": 0.95}
```

## 🎯 Model Architecture

```
StackingClassifier
├── Base Estimators:
│   ├── XGBoost (n_estimators=2944, max_depth=6, lr=0.023)
│   ├── HistGradientBoosting (max_iter=2587, max_depth=12, lr=0.048)
│   └── RandomForest (n_estimators=355, max_depth=None)
└── Meta Estimator:
    └── LogisticRegression with PolynomialFeatures (degree=2)
```

## 📊 Training Your Own Model

```bash
# Prepare dataset in format:
# [{"url": "https://example.com", "label": "safe"}, ...]

python main.py
>>> train path/to/your/dataset.pkl
```

**Dataset format requirements:**
- File: `.pkl` or `.joblib`
- Structure: List of dictionaries
- Keys: `url` (string), `label` ("safe" or "phishing")

## 🧪 Example Output

```bash
>>> google.com
Analyse pour : https://google.com
{
  "phishing": false,
  "prob": 0.01
}

>>> 192.0.0.1
Analyse pour : https://192.0.0.1
{
  "phishing": true,
  "prob": 0.99
}

>>> google@attaquant.com
Analyse pour : https://google@attaquant.com
{
  "phishing": true,
  "prob": 0.996
}
```

## 📁 Project Structure

```
anti-phishing-ml/
├── config.py           # Configuration and whitelist
├── features.py         # Feature extraction logic
├── model.py           # Model architecture
├── train.py           # Training pipeline
├── main.py            # CLI interface
├── data/              # Model and dataset storage
├── cache/             # Prediction cache
└── README.md          # Documentation
```

## 🔬 Model Training Process

1. **Data Preprocessing**: URL cleaning and validation
2. **Feature Extraction**: 19 engineered features per URL
3. **SMOTE Balancing**: Handle class imbalance
4. **Bayesian Optimization**: 10 iterations per base model
5. **Stacking**: Meta-learner combines predictions
6. **Evaluation**: Cross-validation with stratified K-fold

## ⚙️ Configuration

Edit `config.py` to customize:

```python
SEUIL = 0.6  # Detection threshold (0-1)
WHITELIST = ['google.com', 'youtube.com', ...]  # Trusted domains
MODEL_NAME = 'model.pkl'
DATA_NAME = 'dataset.pkl'
```

## 🛠️ Advanced Features

### Caching System
- 24-hour cache expiration
- 1MB cache size limit
- Instant responses for repeated URLs

### Whitelist System
- 80+ pre-approved domains
- Bypasses ML prediction for known-safe sites
- Instant validation

### IP Detection
- Detects numeric IPs
- Identifies hexadecimal IPs (0x format)
- Automatic phishing flag with 0.99 probability

## 📈 Performance Optimization

The model uses:
- **Early stopping** to prevent overfitting
- **Histogram-based boosting** for speed
- **Parallel processing** (n_jobs=-1)
- **Disk caching** for frequent predictions

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

MIT License - see LICENSE file for details

## 👤 Author

Samuel Hounsou
- LinkedIn: www.linkedin.com/in/benny-hounsou-00a267374
- GitHub: @hounsoubenny-cyber(https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset sources: Multiple phishing URL datasets
- Libraries: scikit-learn, XGBoost, imbalanced-learn
- Inspiration: Cybersecurity research community

## 📧 Contact

For questions or collaboration: hounsoubenny@gmail.com

---

⭐ Star this repo if you find it useful!
