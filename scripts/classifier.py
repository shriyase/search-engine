from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def train_classifier(X_train, y_train):
    """
    Train a classifier using a pipeline: CountVectorizer -> TfidfTransformer -> LinearSVC
    """
    clf_pipeline = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", LinearSVC()),
        ]
    )
    clf_pipeline.fit(X_train, y_train)
    return clf_pipeline
