import pandas as pd
from scripts.preprocess import preprocess_text, tokenize_and_normalize
from scripts.index import create_inverted_index
from scripts.classifier import train_classifier
from scripts.rank import rank_documents
from scripts.utils import setup_logging, log_info


def main(file_path, query):
    setup_logging()  # Setup logging
    try:
        # Load and preprocess data
        df = pd.read_csv(file_path)
        df = preprocess_text(df)
        df = tokenize_and_normalize(df)

        # Create Inverted Index and calculate TF-IDF
        inv_index, tf_idf_docs, terms = create_inverted_index(df)

        # Rank documents based on query similarity
        ranked_docs = rank_documents(query, inv_index, tf_idf_docs, terms, df)
        log_info("Top ranked documents based on query:")
        for idx, (doc, score) in enumerate(ranked_docs[:5]):
            log_info(f"{idx + 1}. {doc} (Score: {score:.4f})")

    except Exception as e:
        log_error(f"Error: {e}")
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Example usage:
    file_path = "data/tfidf_dataset.csv"  # Path to your dataset
    query = input("Enter query: ")
    main(file_path, query)
