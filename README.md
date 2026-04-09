# 📧 Spam Email Classification using Deep Learning

## 🚀 Overview

This project presents a **Spam Email Classification System** that uses **Deep Learning (Bi-directional LSTM)** to classify emails as **Spam** or **Ham (Legitimate)**.

The system is trained on a **combined dataset of TREC 2007 Spam Corpus and Enron Spam Dataset**, and includes a **modern interactive GUI built using Streamlit** with explainable predictions.

---

## 🎯 Objectives

* Detect spam emails with high accuracy
* Build an end-to-end ML pipeline (EDA → preprocessing → model → evaluation → deployment)
* Develop an interactive GUI for real-time classification
* Provide **explainability** for predictions

---

## 📂 Dataset

The dataset consists of **83,000+ email samples** labeled as:

* **Spam (1)** → Unwanted / malicious emails
* **Ham (0)** → Legitimate emails

### 📊 Sources:

* TREC 2007 Public Spam Corpus
* Enron Spam Dataset

---

## ⚙️ Workflow

### 1️⃣ Data Preprocessing

* Lowercasing text
* Removing special characters
* Cleaning email content

> ⚠️ Stopword removal was avoided to preserve important spam keywords.

---

### 2️⃣ Exploratory Data Analysis (EDA)

* Spam vs Ham distribution
* Word frequency analysis
* Word clouds
* N-gram (bigram) analysis

---

### 3️⃣ Feature Engineering

* Tokenization using Keras Tokenizer
* Padding sequences for uniform input length
* Vocabulary size control

---

### 4️⃣ Model Architecture

* **Embedding Layer**
* **Bidirectional LSTM**
* Dense layers with ReLU
* Dropout for regularization
* Sigmoid output for binary classification

---

## 🧠 Model Summary

| Layer     | Purpose                              |
| --------- | ------------------------------------ |
| Embedding | Convert words to vectors             |
| Bi-LSTM   | Capture context (forward + backward) |
| Dense     | Learn complex patterns               |
| Dropout   | Prevent overfitting                  |
| Output    | Spam probability                     |

---

## 📈 Performance Metrics

* **Accuracy:** ~98%
* **Precision:** High (low false positives)
* **Recall:** High (captures most spam)
* **F1 Score:** Balanced performance

---

## 🧪 Evaluation Tools

* Confusion Matrix
* Precision, Recall, F1 Score
* Accuracy

---

## 💻 GUI Application (Streamlit)

The project includes a **modern web interface** where users can:

### ✨ Features:

* Paste email content
* Classify as Spam or Ham
* View **confidence score**
* See **highlighted keywords**
* Get **reason-based explanations**

---

## 🧠 Explainability (Key Feature)

The system provides insights into predictions by:

* Detecting **spam keywords** (e.g., *urgent, free, verify*)
* Detecting **ham indicators** (e.g., *meeting, project*)
* Highlighting important words in the email

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras**
* **NLTK**
* **Scikit-learn**
* **Pandas / NumPy**
* **Matplotlib / Seaborn**
* **Streamlit**

---

## 📦 Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                  # Streamlit GUI
├── model.keras            # Trained LSTM model
├── tokenizer.pkl          # Saved tokenizer
├── notebook.ipynb         # Training notebook
├── dataset/               # Dataset files
├── README.md              # Documentation
```

---

## ⚠️ Challenges Faced

* Overfitting in LSTM model
* Dataset noise and inconsistency
* Model calibration (confidence ~0.6 issue)
* Synchronizing preprocessing between notebook and GUI

---

## ✅ Improvements Made

* Fixed preprocessing mismatch
* Removed harmful stopword removal
* Standardized tokenizer usage
* Adjusted prediction threshold
* Added explainability layer
* Improved UI/UX

---

## 🚀 Future Enhancements

* Add TF-IDF + Naive Bayes hybrid model
* Attention-based LSTM for better interpretability
* Real-time email API integration
* Deployment on cloud (Streamlit Cloud / AWS)
* Spam word importance visualization

---

## 🎓 Conclusion

This project demonstrates how **Deep Learning can effectively solve real-world problems like spam detection**, combining:

* Strong ML pipeline
* High model performance
* User-friendly interface
* Explainable AI

---

## 👨‍💻 Author

**Shreshth Panda **

---

## ⭐ Acknowledgements

* TREC Spam Corpus
* Enron Dataset
* TensorFlow & Streamlit community

---
