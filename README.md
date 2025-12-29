📩 Spam SMS Detection using Machine Learning
📌 Project Overview

Spam messages are a common problem in mobile communication, often used for scams, advertisements, and phishing attacks. This project focuses on building an AI-based text classification system that automatically identifies whether an SMS message is spam or legitimate (ham) using Natural Language Processing (NLP) and machine learning techniques.

🎯 Objectives

Classify SMS messages as spam or legitimate

Apply text preprocessing and feature extraction techniques

Compare multiple machine learning models for text classification

Improve accuracy while minimizing false positives

📂 Dataset Description

The dataset contains labeled SMS messages with two classes:

ham → Legitimate message

spam → Unwanted or malicious message

Each record consists of raw SMS text and its corresponding label.

🧠 Techniques & Models Used
🔹 Text Preprocessing

Lowercasing

Removing punctuation and special characters

Stopword removal

Tokenization

🔹 Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency)

(Optional) Word embeddings for enhanced semantic representation

🔹 Machine Learning Models

Naive Bayes – Probabilistic baseline model

Logistic Regression – Linear classifier for text data

Support Vector Machine (SVM) – High-performance classifier for sparse text features

⚙️ Methodology

Data cleaning and preprocessing

Text vectorization using TF-IDF

Train-test data split

Model training and optimization

Model evaluation using standard classification metrics

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-score

🚀 Results

Among the tested models, Support Vector Machine and Logistic Regression demonstrated strong performance with high precision and recall, making them suitable for real-world spam detection applications.

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

NLTK / SpaCy (for text preprocessing)

📌 Conclusion

This project demonstrates the effectiveness of NLP and machine learning techniques in detecting spam SMS messages. The system can be further enhanced using deep learning models and deployed as a real-time spam filtering service.# Spam-SmS-Detection
