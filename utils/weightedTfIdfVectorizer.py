from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

class TfIdfEmbeddingVectorizer(object):
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = dim
        self.word2vec['_UNK'] = np.zeros((self.dim,))

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda x: max_idf,
            [(key, tfidf.idf_[val]) for key, val in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                    np.mean([self.word2vec[word] * self.word2weight[word] for word in words if word in self.word2vec] or
                            [np.zeros(self.dim)], axis=0)
                    for words in X
                ])

