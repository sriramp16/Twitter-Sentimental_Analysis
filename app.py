import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

nltk.download('punkt')
nltk.download('stopwords')

# Set stop words
stop_words = set(stopwords.words("english"))
tokenizer = RegexpTokenizer(r'\w+')

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https\S+|www\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = tokenizer.tokenize(text)
    filtered_text = [w for w in tokens if w not in stop_words]
    return " ".join(filtered_text)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

with open("logreg_model.pkl", "rb") as logreg_file:
    logreg_model = pickle.load(logreg_file)

with open("svc_model.pkl", "rb") as svc_file:
    svc_model = pickle.load(svc_file)

st.title("📊 Twitter Sentiment Analyzer")

tabs = st.tabs(["🔍 Predict Sentiment", "📁 Analyze CSV Dataset"])

with tabs[0]:
    st.subheader("Enter a Tweet or Text")

    model_choice = st.selectbox("Choose Model", ["LinearSVC", "Logistic Regression"])

    user_input = st.text_area("Write your tweet here...")

    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a tweet or sentence.")
        else:
            cleaned = clean_text(user_input)
            vect_input = vectorizer.transform([cleaned])

            if model_choice == "Logistic Regression":
                model = logreg_model
            else:
                model = svc_model

            # Safety check
            if vect_input.shape[1] != model.n_features_in_:
                st.error(
                    f"❌ Feature size mismatch!\n"
                    f"Vectorizer has {vect_input.shape[1]} features, but model expects {model.n_features_in_}.\n"
                    f"🛠 Please retrain your model with the same vectorizer."
                )
            else:
                prediction = model.predict(vect_input)[0]
                sentiment_emoji = {
                    "positive": "😊",
                    "negative": "😞",
                    "neutral": "😐"
                }
                st.success(f"**Predicted Sentiment:** {prediction} {sentiment_emoji.get(prediction.lower(), '')}")

with tabs[1]:
    st.subheader("Upload Twitter Dataset (CSV with 'Tweet' column)")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file).fillna('')

        if "Tweet" not in df.columns:
            st.warning("CSV must contain a 'Tweet' column.")
        else:
            df["cleaned_text"] = df["Tweet"].apply(clean_text)

            model_choice_2 = st.selectbox("Choose Model for Dataset", ["LinearSVC", "Logistic Regression"], key="model2")
            model = svc_model if model_choice_2 == "LinearSVC" else logreg_model

            def predict_sentiment(text):
                vect_text = vectorizer.transform([text])
                return model.predict(vect_text)[0]

            df["Sentiment"] = df["cleaned_text"].apply(predict_sentiment)

            st.markdown("### 📈 Basic Statistics")
            st.write(f"Total Tweets: {len(df)}")
            st.write(f"Sentiment Counts:")
            st.dataframe(df["Sentiment"].value_counts().reset_index().rename(columns={'index': 'Sentiment', 'Sentiment': 'Count'}))

            st.markdown("### 📊 Sentiment Distribution (Bar Chart)")
            fig_bar, ax_bar = plt.subplots()
            sns.countplot(x="Sentiment", data=df, palette="Set2", ax=ax_bar)
            st.pyplot(fig_bar)

            st.markdown("### 🥧 Sentiment Distribution (Pie Chart)")
            sentiment_counts = df["Sentiment"].value_counts()
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
            st.pyplot(fig_pie)

            st.markdown("### ☁️ Word Cloud of Tweets")
            text = " ".join(df["cleaned_text"])
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            fig_wc, ax_wc = plt.subplots()
            ax_wc.imshow(wc, interpolation="bilinear")
            ax_wc.axis("off")
            st.pyplot(fig_wc)

            st.markdown("### 📥 Download Analyzed Dataset")
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "analyzed_tweets.csv", "text/csv")
