# 💬 AI Echo: Your Smartest Conversational Partner

## 📌 Project Overview

**AI Echo** is an **end-to-end NLP & Sentiment Analysis system** designed to understand and analyze **ChatGPT user reviews**.
It processes text data, classifies user sentiments as **Positive, Neutral, or Negative**, and visualizes meaningful patterns through an interactive **Streamlit dashboard**.

The goal is to help product teams and analysts gain insights into **customer satisfaction**, identify **pain points**, and enhance **user experience** based on data-driven feedback.

This project integrates **Python (Pandas, NLTK, Scikit-learn)** for text analysis and **Streamlit** for building a powerful, interactive dashboard.

---

## 🛠️ What I Did in This Project

### 1. 🧹 Data Preprocessing (Python + NLP)

* Cleaned and normalized review text:

  * Removed punctuation, special characters, and URLs.
  * Performed **tokenization, stopword removal, and lemmatization** using **NLTK**.
* Handled missing values and standardized casing.
* Derived additional features:

  * `review_length`, `verified_purchase`, and `sentiment` based on rating.

---

### 2. 📊 Exploratory Data Analysis (EDA)

* Analyzed the dataset (`chatgpt_style_reviews_dataset.xlsx`) containing:

  * **Date, Review, Rating, Platform, Version, Location, Verified Purchase**
* Explored patterns using:

  * **Sentiment distribution** (Positive, Neutral, Negative)
  * **Ratings vs Sentiment**
  * **Sentiment trends over time**
  * **Platform-based analysis (Web vs Mobile)**
  * **Location-based insights**
  * **Review length comparison**
* Visualized using:

  * Matplotlib, Seaborn, and WordCloud

---

### 3. 🤖 Sentiment Classification Logic

* Mapped ratings to sentiment:

  * ⭐ 1–2 → Negative
  * ⭐ 3 → Neutral
  * ⭐ 4–5 → Positive
* Preprocessed reviews for token-based word cloud visualization.
* Prepared the dataset for ML model training (TF-IDF / Word Embeddings – optional extension).

---

### 4. 📈 Streamlit Dashboard

Built a fully interactive **Streamlit app (`sentiment_dashboard.py`)** with the following sections:

1️⃣ **Overall Sentiment Distribution** – Pie chart showing proportion of each sentiment.

2️⃣ **Sentiment by Rating** – Bar chart comparing sentiment across ratings.

3️⃣ **Word Clouds** – Separate clouds for Positive, Neutral, and Negative words.

4️⃣ **Sentiment Trends Over Time** – Line graph tracking sentiment evolution by month.

5️⃣ **Verified Purchase Analysis** – Compare sentiments between verified & non-verified users.

6️⃣ **Review Length by Sentiment** – Understand emotional depth vs. word count.

7️⃣ **Location-Based Sentiment** – Identify top countries with most positive or negative reviews.

8️⃣ **Platform Analysis** – Contrast feedback from Web and Mobile users.

9️⃣ **ChatGPT Version Analysis** – Track satisfaction across product versions.

🔟 **Common Negative Words** – Identify frequent complaint terms for improvement focus.

---

## 🎯 Motive of the Project

* To understand **user sentiment trends** for ChatGPT reviews.
* To identify **key areas of dissatisfaction** and **positive engagement drivers**.
* To help product and marketing teams make **data-informed improvements**.
* To demonstrate **NLP-based business analytics** capabilities.

---

## 🌍 Real-Life Use Cases

* **Customer Experience Teams** → Identify trends in satisfaction and complaints.
* **Product Managers** → Detect features that need enhancement.
* **Marketing Analysts** → Measure brand perception and sentiment shifts.
* **Data Scientists** → Extend model with deep learning (LSTM/BERT) for richer predictions.

---

## 📊 Evaluation Metrics

* **Accuracy** – Overall correctness of model predictions.
* **Precision & Recall** – Reliability for positive/negative detection.
* **F1-Score** – Balance between precision and recall.
* **Confusion Matrix** – Understand misclassification behavior.
* **AUC-ROC Curve** – Model’s ability to distinguish sentiments.

---

## ⚙️ Tech Stack

| Category          | Tools & Libraries                                                 |
| ----------------- | ----------------------------------------------------------------- |
| **Programming**   | Python                                                            |
| **Libraries**     | Pandas, NumPy, Matplotlib, Seaborn, NLTK, WordCloud, Scikit-learn |
| **Visualization** | Streamlit                                                         |
| **Deployment**    | Streamlit Cloud / AWS (optional)                                  |

---

## 📁 Dataset Used

**File:** `chatgpt_style_reviews_dataset.xlsx`
**Columns:** date, title, review, rating, username, helpful_votes, review_length, platform, language, location, version, verified_purchase


## ✅ Conclusion

**AI Echo: Your Smartest Conversational Partner** demonstrates how **NLP + Data Visualization** can transform unstructured feedback into actionable business insights.

Through this project, you can:
✔ Detect overall sentiment trends.
✔ Identify regions, versions, and platforms driving satisfaction or complaints.
✔ Empower teams to take data-backed decisions for product enhancement.

It is a step toward **AI-driven customer experience analytics**.


## 🚀 How to Run the Project

 **Run the Streamlit App**

```bash
streamlit run sentiment_dashboard.py



