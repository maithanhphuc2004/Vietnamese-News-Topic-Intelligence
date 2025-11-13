# 📰 Vietnamese-News-Topic-Intelligence
![TF-IDF](https://img.shields.io/badge/Embedding-TF--IDF-blue?style=flat-square)
![PhoBERT-base](https://img.shields.io/badge/Embedding-PhoBERT--base-green?style=flat-square)
![PhoBERT-large](https://img.shields.io/badge/Embedding-PhoBERT--large-purple?style=flat-square)

![LogisticRegression](https://img.shields.io/badge/Model-Logistic%20Regression-orange?style=flat-square)
![RandomForest](https://img.shields.io/badge/Model-Random%20Forest-darkgreen?style=flat-square)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-red?style=flat-square)
![NaiveBayes](https://img.shields.io/badge/Model-Naive%20Bayes-yellow?style=flat-square)
![MLP](https://img.shields.io/badge/Model-MLP%20Neural%20Network-pink?style=flat-square)


![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Status](https://img.shields.io/badge/Status-Completed-green?style=flat-square)
![Model](https://img.shields.io/badge/Models-LR%20%7C%20RF%20%7C%20XGBoost%20%7C%20NB%20%7C%20MLP-purple?style=flat-square)

---

## 📌 Overview *(Tổng quan)*

This project builds an automatic topic classification system for Vietnamese online news using two main text-representation approaches:

- **TF-IDF (traditional statistical approach)**  
- **PhoBERT (deep contextual embedding for Vietnamese)**  

*(Dự án xây dựng hệ thống phân loại chủ đề bài báo tiếng Việt bằng hai phương pháp: TF-IDF truyền thống và PhoBERT hiện đại.)*

Five machine learning models were trained and evaluated:

- Logistic Regression  
- Random Forest  
- XGBoost  
- Naive Bayes  
- MLP Neural Network  

The study compares TF-IDF vs PhoBERT and evaluates model performance using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

---

## 📚 Features *(Tính năng)*

- 🧹 Automatic data preprocessing  
- 📝 Vietnamese text normalization & tokenization  
- 🔤 Word embedding via **TF-IDF** and **PhoBERT**
- 🤖 ML models training & evaluation  
- 📊 Visualization: Distribution plots, Heatmaps, Confusion Matrices  
- 📈 Performance comparison across all models  

---

## 🏗️ Dataset *(Tập dữ liệu)*

- **7,678** Vietnamese news articles from various categories (18 classes).  
- Cleaned, normalized, and balanced using **Random Oversampling**.  
- Fields used:
  - `category` — news topic *(chủ đề)*  
  - `content` — full article text *(nội dung bài)*  

---

## 🔧 Preprocessing Steps *(Tiền xử lý dữ liệu)*

- Convert text to lowercase *(chuyển chữ thường)*  
- Remove URLs, numbers, emojis, punctuation  
- Vietnamese word segmentation using **Underthesea**  
- Stopword removal using custom Vietnamese stopword list  
- Remove duplicates + short texts  
- Dataset balancing with **RandomOverSampler**

---

## 🔡 Embedding Methods *(Biểu diễn đặc trưng)*

### **1️⃣ TF-IDF (3000 features)**
- N-grams: 1–2  
- Sublinear TF  
- Suitable for linear models  
- Fast & efficient for short-news content  

### **2️⃣ PhoBERT (768-dim contextual embedding)**
- Pretrained on 20GB Vietnamese text  
- Captures semantic & contextual meaning  
- Better for deep models (MLP)  

---

## 🧠 Machine Learning Models *(Các mô hình học máy)*

Five ML models were applied:

| Model | Description *(Mô tả)* |
|-------|------------------------|
| **Logistic Regression** | Linear classifier, strong baseline |
| **Random Forest** | Ensemble of decision trees |
| **XGBoost** | Gradient boosting, powerful for tabular features |
| **Naive Bayes** | Probabilistic baseline |
| **MLP Neural Network** | Deep learning, nonlinear representation |

---

## 📊 Algorithm Comparison Table *(Bảng so sánh thuật toán)*

### **📌 Performance Comparison (F1-Score)**

| Model | TF-IDF | PhoBERT |
|-------|--------|---------|
| **Logistic Regression** | ⭐ **0.8631** | 0.7941 |
| **Random Forest** | ⭐ **0.8270** | 0.7363 |
| **XGBoost** | ⭐ **0.8587** | 0.7683 |
| **Naive Bayes** | 0.7062 | 0.7206 |
| **MLP Neural Network (Test)** | ⭐ **0.8702** | 0.8205 |

➡️ **MLP + TF-IDF is the best overall performer.**  
*(MLP + TF-IDF đạt hiệu suất cao nhất.)*

---

## 📈 Visualization (Trực quan hóa)

Distribution plots

Topic frequency bar chart

Text length distribution

Confusion matrices for all models

Model comparison charts

## 👨‍💻 Authors (Tác giả)

Mai Thanh Phúc
Hoàng Thị Yến Nhi
Trần Trọng Thành
Supervisor: Lê Nhật Tùng (GVHD)

