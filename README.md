# 🎬 Sentiment Analysis — IMDB Movie Reviews

> **Internship Project** | [Pinnacle Labs](https://pinnaclelabs.tech/internship/)  
> Built as part of a Data Science Internship task to apply NLP and Deep Learning on real-world text data.
---
## 🖥️ Live Demo

Run locally:
```bash
python -m streamlit run app.py
```
| Has Positive Reviews | Has Negative Reviews |
|---|---|
|<img width="1919" height="1119" alt="Screenshot 2026-06-03 144135" src="https://github.com/user-attachments/assets/6b8d4ec2-91d1-4aae-b1a7-1845514d9634" /> |
 <img width="1919" height="1109" alt="Screenshot 2026-06-03 144115" src="https://github.com/user-attachments/assets/7b4e6b5b-b63e-4afc-a778-3405b6ee615b" />
 | 

---
## 📁 Project Structure

```
Sentiment_Review/
│
├── Sentiment_Analysis.ipynb      # Main notebook — EDA, preprocessing, model, evaluation
├── app.py                        # Streamlit web application for live predictions
├── cnn_sentiment_model.h5        # Saved trained CNN model
├── tokenizer.pkl                 # Saved tokenizer (fitted on training data)
├── IMDB Dataset.csv              # Dataset — 50,000 labeled movie reviews
└── README.md                     # Project documentation
```

---

## 🧠 What Does It Do?

This project builds a **CNN (Convolutional Neural Network) based NLP model** that reads a movie review written in plain English and classifies it as either **Positive** or **Negative**.

### Complete Pipeline:

| Step | What Happens |
|---|---|
| **1. Load Data** | 50,000 IMDB reviews loaded from CSV |
| **2. EDA** | Sentiment distribution via bar chart & pie chart |
| **3. Text Preprocessing** | HTML removal, lowercasing, stopword removal, Porter Stemming |
| **4. Encoding** | Label encoding — `positive → 1`, `negative → 0` |
| **5. Train-Test Split** | 80% train / 20% test |
| **6. Tokenization** | Keras Tokenizer → sequences → padded arrays |
| **7. CNN Model** | Embedding → Conv1D → GlobalMaxPooling → Dense → Sigmoid |
| **8. Train** | 5 epochs, batch size 64, Adam optimizer |
| **9. Evaluate** | Accuracy, Classification Report, Confusion Matrix |
| **10. Predict** | Custom review prediction function |
| **11. Save** | Model saved as `.h5`, tokenizer as `.pkl` |
| **12. Web App** | Streamlit app for live predictions |

---

## 🏗️ CNN Model Architecture

```
Input Review Text
        ↓
   Tokenization & Padding
        ↓
   Embedding Layer (vocab_size × 100)
        ↓
   Conv1D (128 filters, kernel=5, relu)
        ↓
   GlobalMaxPooling1D
        ↓
   Dense (10, relu)
        ↓
   Dense (1, sigmoid)  →  0 = Negative | 1 = Positive
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `TensorFlow / Keras` | CNN model building & training |
| `NLTK` | Stopword removal & Porter Stemming |
| `Scikit-learn` | Label encoding, train-test split, evaluation metrics |
| `Pandas / NumPy` | Data loading and manipulation |
| `Matplotlib / Seaborn` | EDA charts and confusion matrix |
| `Streamlit` | Web application for live predictions |
| `Pickle` | Saving and loading the tokenizer |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow nltk scikit-learn pandas numpy matplotlib seaborn streamlit
```

### Run the Notebook
```bash
jupyter notebook Sentiment_Analysis.ipynb
```

### Run the Web App
```bash
cd Sentiment_Review
streamlit run app.py
```
Open browser at `http://localhost:8501`

---

## 📊 Results

| Metric | Value |
|---|---|
| Training Accuracy | ~95%+ |
| Validation Accuracy | ~88–90% |
| Loss | Converges smoothly within 5 epochs |

The model accurately distinguishes positive reviews like *"This movie was absolutely brilliant!"* from negative ones like *"Worst film I have ever watched."*

---

## 🌐 Web App Features

- 📝 Text area to type any movie review
- 🔍 One-click **Analyze Sentiment** button
- ✅ Displays **Positive 😊 / Negative 😞** result
- 📊 Confidence score with a progress bar
- 💡 Clickable example reviews to test instantly

---

## 👥 Who Uses This?

| User | Use Case |
|---|---|
| **OTT Platforms** | Netflix, Amazon Prime analyzing user reviews automatically |
| **Movie Studios** | Gauging public reaction to trailers and releases |
| **Review Aggregators** | IMDb, Rotten Tomatoes automating review classification |
| **Data Science Learners** | Hands-on practice with NLP + Deep Learning |
| **Market Researchers** | Understanding audience sentiment at scale |

---

## 🌍 Real-World Impact

- **🎬 Entertainment Industry** — Studios use sentiment analysis to predict box office performance based on early audience reactions and social media buzz
- **📢 Brand Monitoring** — Companies track whether public conversations about their products are positive or negative in real time
- **🛍️ E-Commerce** — Amazon, Flipkart use similar models to auto-tag product reviews, helping buyers make faster decisions
- **📰 News & Media** — Outlets analyze reader sentiment toward articles and topics to improve content strategy
- **🤖 Chatbots & Virtual Assistants** — Sentiment detection helps AI systems respond more empathetically based on the user's emotional tone
- **📈 Stock Market** — Financial firms analyze news and social sentiment to predict stock movements

---

## 🐛 Problems Faced & How I Solved Them

### 1. HTML Tags in Reviews
**Problem:** Reviews contained raw HTML like `<br />`, `&amp;`, which polluted the text.

**Fix:** Used `html.unescape()` and `re.sub(r'<.*?>', '', text)` to strip all HTML:
```python
review = html.unescape(review)
review = re.sub(r'<.*?>', '', review)
```

---

### 2. `ModuleNotFoundError: No module named 'tensorflow'`
**Problem:** TensorFlow was not installed on the local machine.

**Fix:**
```bash
pip install tensorflow
# or for Windows
pip install tensorflow-cpu
```

---

### 3. `FileNotFoundError` — Wrong File Path
**Problem:** Running `streamlit run app.py` from the parent folder instead of the project folder.

**Fix:** Navigate to the correct folder first:
```bash
cd Sentiment_Review
streamlit run app.py
```

---

### 4. Tokenizer Shape Mismatch in Web App
**Problem:** Passing raw text `X_test` to `model.predict()` instead of padded sequences.

**Fix:** Always use the padded version:
```python
# ❌ Wrong
y_pred = model.predict(X_test)

# ✅ Correct
y_pred = model.predict(x1_test)   # x1_test = padded sequences
```

---

### 5. `maxlen` Too Large — Memory Crash
**Problem:** Using `maxlen = max(len(x) for x in x_train)` gave an extremely large value, crashing memory.

**Fix:** Set a fixed, reasonable maxlen:
```python
maxlen = 200   # covers most reviews without wasting memory
```

---

### 6. Black Screen on Streamlit App
**Problem:** App opened at `localhost:8501` but showed a blank black page.

**Cause:** TensorFlow was missing — the app crashed silently on import.

**Fix:** Installed TensorFlow and restarted the app.

---

## 📌 Future Improvements

- [ ] Upgrade from CNN to **Bidirectional LSTM** for better context understanding
- [ ] Fine-tune a **BERT** model for state-of-the-art accuracy (~95%+)
- [ ] Add **word cloud visualization** of most common positive/negative words
- [ ] Deploy on **Streamlit Cloud** for public access
- [ ] Extend to **multi-class sentiment** (very positive / neutral / very negative)

---

## 🏢 About This Internship

This project was built as a task during my **Data Science Internship at [Pinnacle Labs](https://pinnaclelabs.tech/internship/)**.

Pinnacle Labs provides hands-on internship tasks designed to upskill students in real-world Data Science, Machine Learning, and AI applications. Each task is a complete end-to-end project covering data preprocessing, model building, evaluation, and deployment.

---

## 📄 License

This project is open-source and free to use for educational purposes.

---

## 🙋 Author

**ojaspaul123**  
Data Science Intern @ [Pinnacle Labs](https://pinnaclelabs.tech/internship/)  
Part of the **DL-journey** repository — a collection of deep learning projects.  
[![GitHub](https://img.shields.io/badge/GitHub-ojaspaul123-black?logo=github)](https://github.com/ojaspaul123)
