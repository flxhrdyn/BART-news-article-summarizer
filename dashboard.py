import streamlit as st
import bs4 as bs
import urllib.request
import string
import nltk
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer
import textwrap

# Download stopwords
nltk.download("stopwords")

# Streamlit page setup
st.set_page_config(page_title="News Article Summarizer", layout="wide")
st.title("📰 News Article Summarizer")
st.markdown("Enter the URL of a news article page, and this app will summarize it using the BART model.")

# User input for URL
url = st.text_input("Enter the URL of the article", placeholder="https://www.bbc.co.uk/rd/projects/natural-language-processing")

# Function to clean text
def clean_text(text):
    if not text:
        return ""
    text = ''.join([char for char in text if char not in string.punctuation])
    words = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    return ' '.join(words)

# Function to wrap text nicely
def wrap_text(text, width=80):
    return "\n".join(textwrap.wrap(text, width=width))

# On button click
if st.button("🔍 Fetch & Summarize!") and url:
    try:
        # Fetch and parse webpage content
        web_scraping = urllib.request.urlopen(url)
        content = web_scraping.read()
        parsing = bs.BeautifulSoup(content, 'lxml')

        # Extract main text from the page
        content_div = parsing.find('div', {'class': 'mw-parser-output'})
        paragraphs = content_div.find_all(['p', 'li']) if content_div else parsing.find_all('p')
        article_text = " ".join([p.text for p in paragraphs])

        # Clean the text
        cleaned_text = clean_text(article_text)

        # Validate cleaned text
        if not cleaned_text:
            st.warning("The text is empty after cleaning. Cannot summarize.")
        elif len(cleaned_text.split()) < 50:
            st.warning("The article is too short to summarize. Please try a longer one.")
        else:
            # Limit input to 1024 words
            max_words = 1024
            word_list = cleaned_text.split()
            if len(word_list) > max_words:
                cleaned_text = " ".join(word_list[:max_words])

            # Display original text
            st.markdown("#### 📄 Original Article Text:")
            st.write(wrap_text(article_text[:2000]), width=90)

            # Load summarization model
            with st.spinner("Summarizing article..."):
                model_name = "facebook/bart-large-cnn"
                summarizer = pipeline("summarization", model=model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)

                # Generate summary
                summary = summarizer(cleaned_text, max_length=150, min_length=50, do_sample=False)
                result = summary[0]['summary_text']

            # Display summary
            st.markdown("### ✨ Article Summary:")
            st.write(wrap_text(result, width=90))

    except Exception as e:
        st.error(f"An error occurred while processing the URL: {e}")
