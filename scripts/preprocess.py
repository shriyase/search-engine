import re
import nltk

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")


def preprocess_text(data):
    """
    Preprocess the text data: lowercase conversion, punctuation removal, etc.
    """
    data["text"] = data["text"].str.lower()  # Convert to lowercase
    data["text"] = data["text"].apply(lambda x: re.sub(r"\d+", "", x))  # Remove numbers
    data["text"] = data["text"].apply(
        lambda x: re.sub(r"[^a-z\s]", " ", x).strip()
    )  # Remove punctuation
    return data


def tokenize_and_normalize(data):
    """
    Tokenize and normalize the text: stopwords removal, stemming, and lemmatization
    """
    stop_words = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def process_text(text):
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [stemmer.stem(token) for token in tokens]
        return [lemmatizer.lemmatize(token) for token in tokens]

    data["text"] = data["text"].apply(process_text)
    return data
