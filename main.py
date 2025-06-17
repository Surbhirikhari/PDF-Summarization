import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import os

# Set up Hugging Face pipelines
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
en_fr_translator = pipeline("translation_en_to_fr")

st.title("📄 PDF Summarizer & Translator")
st.write("Upload a PDF file, and we'll summarize it and translate the summary to French.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

def chunk_text(text, max_words=500):
    """Split text into chunks of max_words words for summarization."""
    words = text.split()
    chunks = [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
    return chunks

if uploaded_file is not None:
    # Save uploaded file
    file_path = os.path.join("saved_uploads", uploaded_file.name)
    os.makedirs("saved_uploads", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ File '{uploaded_file.name}' uploaded and saved successfully.")

    # Extract text from PDF
    with st.spinner("Extracting text from PDF..."):
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"

    st.subheader("📚 Extracted Text")
    st.text_area("Raw PDF Content", text, height=200)

    # Summarize the text in chunks
    with st.spinner("Summarizing text..."):
        chunks = chunk_text(text, max_words=500)
        summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=300, min_length=30, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        summary_text = ' '.join(summaries)

    st.subheader("📝 Summary")
    st.write(summary_text)

    # Translate the summary
    with st.spinner("Translating summary to French..."):
        translated = en_fr_translator(summary_text)
        translated_text = translated[0]['translation_text']

    st.subheader("🇫🇷 Translated Summary (French)")
    st.write(translated_text)

    # Option to download summary
    st.download_button("⬇️ Download Summary (English)", summary_text, file_name="summary.txt")
    st.download_button("⬇️ Download Translated Summary (French)", translated_text, file_name="summary_fr.txt")
