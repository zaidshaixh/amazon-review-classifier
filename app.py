import os
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Amazon Sentiment Analysis", page_icon="🛒", layout="wide")

# Apply custom Amazon-style background
st.markdown(
    """
    <style>
    body {
        background-color: #FF9900;
        color: black;
        font-family: Arial, sans-serif;
    }
    .stButton>button {
        background-color: #146EB4;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 10px;
    }
    .stSidebar {
        background-color: #232F3E !important;
    }
    .stSidebar h2 {
        color: white;
        font-weight: bold;
    }
    .stSidebar label {
        color: white;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load Amazon logo
png_path = "amazon_logo.png"
if os.path.exists(png_path):
    amazon_logo = Image.open(png_path)
    st.image(amazon_logo, width=200)
else:
    st.warning("Amazon logo not found! Add 'amazon_logo.png' to the project folder.")

# Title
st.title("📊 Customer Sentiment Analysis for Amazon Reviews")
st.write("Classifying reviews as **Positive, Negative, or Neutral**.")

# Generate sample dataset
data = {
    "review": [
        "This product is amazing! Loved it!",
        "Worst purchase ever, very disappointed.",
        "It's okay, not the best but works fine.",
        "Fantastic quality! Highly recommend!",
        "Not worth the money, really bad experience.",
        "Decent product, could be better.",
        "Excellent! Will buy again.",
        "Terrible, never buying again.",
        "Good value for money.",
        "Mediocre at best, not impressed."
    ]
}
df = pd.DataFrame(data)

# Sample sentiment classification (you can replace this with NLP model)
def classify_sentiment(text):
    text = str(text).lower()
    if "good" in text or "excellent" in text or "amazing" in text or "fantastic" in text:
        return "Positive"
    elif "bad" in text or "worst" in text or "terrible" in text or "disappointed" in text:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["review"].apply(classify_sentiment)

# Filter options
st.sidebar.header("🔍 **Filter Reviews**")
sentiment_filter = st.sidebar.multiselect("**Select Sentiment Type**", ["Positive", "Negative", "Neutral"], default=["Positive", "Negative", "Neutral"])
filtered_df = df[df["Sentiment"].isin(sentiment_filter)]

# Show Data
st.subheader("📜 Sample Reviews with Sentiment Classification")
st.dataframe(filtered_df)

# Sentiment Distribution
st.subheader("📊 Sentiment Distribution")
sentiment_counts = filtered_df["Sentiment"].value_counts()
fig = px.pie(names=sentiment_counts.index, values=sentiment_counts.values, color=sentiment_counts.index,
             color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gray"})
st.plotly_chart(fig)

# Word Cloud for Positive and Negative Reviews
st.subheader("☁️ Word Clouds")
col1, col2 = st.columns(2)

with col1:
    st.write("### Positive Reviews")
    positive_text = " ".join(filtered_df[filtered_df["Sentiment"] == "Positive"]["review"].astype(str))
    if positive_text:
        wordcloud = WordCloud(width=400, height=200, background_color="white").generate(positive_text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.write("No positive reviews available.")

with col2:
    st.write("### Negative Reviews")
    negative_text = " ".join(filtered_df[filtered_df["Sentiment"] == "Negative"]["review"].astype(str))
    if negative_text:
        wordcloud = WordCloud(width=400, height=200, background_color="white").generate(negative_text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)
    else:
        st.write("No negative reviews available.")
