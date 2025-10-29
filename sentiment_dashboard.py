# ==========================================================
# sentiment_insights_app.py (Streamlit version)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------
# NLTK setup
# ----------------------------------------------------------
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'\b\w+\b')

# ----------------------------------------------------------
# Streamlit UI Setup
# ----------------------------------------------------------
st.set_page_config(page_title="Sentiment Insights Dashboard", layout="centered")
st.title("💬 ChatGPT Style Reviews – Sentiment Insights Dashboard")

# ----------------------------------------------------------
# Preprocessing function
# ----------------------------------------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    tokens = tokenizer.tokenize(text)
    tokens = [t for t in tokens if t not in stop_words]
    lemmatized = []
    for token, tag in nltk.pos_tag(tokens):
        pos = tag[0].lower()
        if pos not in ['a','r','n','v']:
            pos = 'n'
        lemmatized.append(lemmatizer.lemmatize(token, pos))
    return ' '.join(lemmatized)

# ----------------------------------------------------------
# Load dataset
# ----------------------------------------------------------
df = pd.read_excel("chatgpt_style_reviews_dataset.xlsx")
st.success("✅ Dataset loaded successfully!")

# ----------------------------------------------------------
# Sentiment Mapping
# ----------------------------------------------------------
def map_sentiment(rating):
    if rating <= 2: return 'Negative'
    elif rating == 3: return 'Neutral'
    else: return 'Positive'

df['sentiment'] = df['rating'].apply(map_sentiment)

# ----------------------------------------------------------
# Preprocess reviews
# ----------------------------------------------------------
with st.spinner("🧹 Cleaning and lemmatizing reviews..."):
    df['cleaned_review'] = df['review'].apply(preprocess_text)
st.success("✅ Text preprocessing complete!")

# ----------------------------------------------------------
# 🔹 SENTIMENT PREDICTION USING TRAINED MODEL
# ----------------------------------------------------------
st.header("🧠 Sentiment Prediction (Your Own Review)")
st.write("Enter a review below and let the trained model predict its sentiment.")

# ----------------------------------------------------------
# 🔹 Load Trained Model & TF-IDF Vectorizer
# ----------------------------------------------------------
try:
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    st.success("✅ Trained model and vectorizer loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model or vectorizer: {e}")

# User input
user_input = st.text_area("✍️ Write your review here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review text before predicting.")
    else:
        cleaned_input = preprocess_text(user_input)
        input_vector = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_vector)[0]

        # Display result with emoji and color
        if prediction == "Positive":
            st.success("😊 **Positive Sentiment Detected!** 🎉")
        elif prediction == "Neutral":
            st.info("😐 **Neutral Sentiment Detected.**")
        else:
            st.error("😞 **Negative Sentiment Detected.**")

# ==========================================================
# VISUALIZATIONS
# ==========================================================

# 1️⃣ Overall Sentiment Distribution
st.header("1️⃣ Overall Sentiment Distribution")
sentiment_counts = df['sentiment'].value_counts(normalize=True) * 100
fig1, ax1 = plt.subplots(figsize=(5, 5))
ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%',
        colors=['#ff6b6b', '#ffd93d', '#6bcf63'])
ax1.set_title("Overall Sentiment Distribution")
st.pyplot(fig1)
st.dataframe(sentiment_counts.round(2))

# 2️⃣ Sentiment vs Rating
st.header("2️⃣ Sentiment by Star Rating")
fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x='rating', hue='sentiment',
              palette=['#ff6b6b', '#ffd93d', '#6bcf63'], ax=ax2)
ax2.set_title("Sentiment by Star Rating")
st.pyplot(fig2)

# 3️⃣ Word Clouds by Sentiment
st.header("3️⃣ Word Clouds by Sentiment")
sentiments = ['Negative', 'Neutral', 'Positive']
colors = {'Negative': '#ff6b6b', 'Neutral': '#ffd93d', 'Positive': '#6bcf63'}
cols = st.columns(3)
for i, sentiment in enumerate(sentiments):
    text = " ".join(df[df['sentiment'] == sentiment]['cleaned_review'].dropna())
    wc = WordCloud(width=800, height=400, background_color='white',
                   color_func=lambda *args, **kwargs: colors[sentiment],
                   max_words=100).generate(text)
    with cols[i]:
        st.subheader(sentiment)
        st.image(wc.to_array(), use_container_width=True)

# 4️⃣ Sentiment Trends Over Time
st.header("4️⃣ Sentiment Trends Over Time")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['month'] = df['date'].dt.to_period('M')
trend = df.groupby(['month', 'sentiment']).size().reset_index(name='count')
trend_pivot = trend.pivot(index='month', columns='sentiment', values='count').fillna(0)
fig3, ax3 = plt.subplots(figsize=(10, 5))
trend_pivot.plot(kind='line', marker='o', ax=ax3)
ax3.set_title("Sentiment Trend Over Time")
st.pyplot(fig3)

# 5️⃣ Sentiment by Verified Purchase
st.header("5️⃣ Sentiment by Verified Purchase")
df['verified_purchase'] = df['verified_purchase'].astype(str).str.lower().replace({'yes': 'Yes', 'no': 'No'})
fig4, ax4 = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x='verified_purchase', hue='sentiment',
              palette=['#ff6b6b', '#ffd93d', '#6bcf63'], ax=ax4)
ax4.set_title("Sentiment by Verified Purchase")
st.pyplot(fig4)

verified_sentiment_pct = df.groupby('verified_purchase')['sentiment'].value_counts(normalize=True).unstack() * 100
st.dataframe(verified_sentiment_pct.round(2))

# 6️⃣ Review Length vs Sentiment
st.header("6️⃣ Average Review Length by Sentiment")
df['review_length'] = df['cleaned_review'].apply(lambda x: len(str(x).split()))
avg_length = df.groupby('sentiment')['review_length'].mean()
fig5, ax5 = plt.subplots(figsize=(6, 4))
ax5.bar(avg_length.index, avg_length.values, color=['#ff6b6b', '#ffd93d', '#6bcf63'])
ax5.set_title("Average Review Length by Sentiment")
st.pyplot(fig5)
st.dataframe(avg_length.round(2))

# 7️⃣ Sentiment by Location
st.header("7️⃣ Sentiment by Location")
loc_counts = df.groupby(['location','sentiment']).size().unstack(fill_value=0)

top_pos = loc_counts.sort_values('Positive', ascending=False).head(70)['Positive']
fig7a, ax7a = plt.subplots(figsize=(6, 15))
ax7a.barh(top_pos.index, top_pos.values, color='#6bcf63')
ax7a.set_xlabel("Number of Positive Reviews")
ax7a.set_ylabel("Location")
ax7a.set_title("Top Locations with Most Positive Reviews")
ax7a.invert_yaxis()
ax7a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
st.pyplot(fig7a)

top_neg = loc_counts.sort_values('Negative', ascending=False).head(30)['Negative']
fig7b, ax7b = plt.subplots(figsize=(6, 5))
ax7b.barh(top_neg.index, top_neg.values, color='#ff6b6b')
ax7b.set_xlabel("Number of Negative Reviews")
ax7b.set_ylabel("Location")
ax7b.set_title("Top Locations with Most Negative Reviews")
ax7b.invert_yaxis()
ax7b.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
st.pyplot(fig7b)

# 8️⃣ Sentiment by Platform
st.header("8️⃣ Sentiment by Platform")
platform_sentiment = df.groupby(['platform', 'sentiment']).size().reset_index(name='count')
fig8, ax8 = plt.subplots(figsize=(6, 4))
sns.barplot(data=platform_sentiment, x='platform', y='count', hue='sentiment',
            palette=['#ff6b6b', '#ffd93d', '#6bcf63'], ax=ax8)
ax8.set_title("Sentiment Distribution by Platform")
st.pyplot(fig8)

platform_total = df.groupby('platform').size()
platform_sentiment['percentage'] = platform_sentiment.apply(
    lambda x: (x['count'] / platform_total[x['platform']]) * 100, axis=1)
platform_pct = platform_sentiment.pivot(index='platform', columns='sentiment', values='percentage')
st.dataframe(platform_pct.round(2))

# 9️⃣ Sentiment by ChatGPT Version
st.header("9️⃣ Sentiment by ChatGPT Version")
version_counts = df.groupby(['version', 'sentiment']).size().unstack(fill_value=0)
version_pct = version_counts.div(version_counts.sum(axis=1), axis=0) * 100
top_versions = version_counts.sum(axis=1).sort_values(ascending=False).head(30).index
version_pct_top = version_pct.loc[top_versions]

fig9, ax9 = plt.subplots(figsize=(6, 5))
version_pct_top[['Positive', 'Neutral', 'Negative']].plot(
    kind='barh', stacked=True, color=['#6bcf63', '#ffd93d', '#ff6b6b'], ax=ax9)
ax9.set_xlabel("Percentage of Reviews (%)")
ax9.set_ylabel("ChatGPT Version")
ax9.set_title("Sentiment Distribution by ChatGPT Version (Top 30)")
ax9.invert_yaxis()
st.pyplot(fig9)

# 1️⃣0️⃣ Common Negative Feedback
st.header("1️⃣0️⃣ Common Words in Negative Reviews")
neg_reviews = df[df['sentiment'] == 'Negative']['cleaned_review'].dropna()
text = " ".join(neg_reviews)
wc = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(text)
st.image(wc.to_array(), caption="Frequent Words in Negative Reviews", use_container_width=False)

tokens = [w for w in word_tokenize(text) if w not in stop_words]
freq = Counter(tokens)
common_words = freq.most_common(51)
st.dataframe(pd.DataFrame(common_words, columns=["Word", "Frequency"]))


# python -m streamlit run sentiment_dashboard.py