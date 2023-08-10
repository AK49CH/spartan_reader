import re
import nltk
nltk.download("punkt")
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse.linalg import svds
import streamlit as st
from PIL import Image

def text_summarizer(user_input, num_sentences=10):
    # Tokenize the user input into sentences
    sentences = nltk.sent_tokenize(user_input)

    # Preprocess the document
    def normalize_document(doc):
        doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I | re.A)
        doc = doc.lower()
        doc = doc.strip()
        return doc

    # Normalize the sentences
    norm_sentences = [normalize_document(sentence) for sentence in sentences]

    # TF-IDF vectorization
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    dt_matrix = tv.fit_transform(norm_sentences)
    dt_matrix = dt_matrix.toarray()

    vocab = tv.get_feature_names_out()
    td_matrix = dt_matrix.T

    # Singular Value Decomposition (SVD)
    num_topics = 1

    def low_rank_svd(matrix):
        num_sentences = matrix.shape[0]
        singular_count = min(matrix.shape) - 1  # Set to one less than the minimum dimension
        u, s, vt = svds(matrix, k=singular_count)
        return u, s, vt

    u, s, vt = low_rank_svd(td_matrix)
    term_topic_mat, singular_values, topic_document_mat = u, s, vt

    # Extract the top sentences based on salience scores
    sv_threshold = 0.5
    min_sigma_value = max(singular_values) * sv_threshold
    singular_values[singular_values < min_sigma_value] = 0

    salience_scores = np.sqrt(np.dot(np.square(singular_values), np.square(topic_document_mat)))

    top_sentence_indices = (-salience_scores).argsort()[:num_sentences]
    top_sentence_indices.sort()

    # Return the top sentences
    return '\n'.join(np.array(sentences)[top_sentence_indices])

# Set the Streamlit theme using the TOML configuration file
st.set_page_config("text_summarization/st_config.toml")

# Your Streamlit layout
st.title("Spartan Reader")
st.header("SOCCENT Text Summarization")

st.markdown("#")

st.header("Instructions")
st.markdown("-Paste your text into the text box")
st.markdown("-Select the number of sentences you would like the summary to be")
st.markdown("-Hit Run")

st.markdown("##")


st.write("*It is recomended that a minimum of five sentences be used per page!*")

st.markdown("##")

# Slider to control the number of sentences for the summary
num_sentences = st.slider("Use The Slider To Select How Many Sentences Your Summary Should Be:", 1, 20, 5)
st.write("Sentence Length", num_sentences)

# Example usage of the text_summarizer function inside the Streamlit app
user_input = st.text_area("Paste Text Here...", height = 400)


if st.button("Run"):
    # Call the text_summarizer function with user input and num_sentences
    summary = text_summarizer(user_input, num_sentences)

    # Display the summary
    st.subheader("Summary:")
    st.write(summary)

# Custom CSS to change the background color of the sidebar to black
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: black;
    }
</style>
""", unsafe_allow_html=True)

# from PIL import Image
# image = Image.open("/home/vboxuser/Documents/text_summarization/SOCCENT.png")
# st.sidebar.image(image)


