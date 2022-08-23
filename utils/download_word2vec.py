
import gensim.downloader

def _download_word2vec_newspaper():
	word2vec_vectors = gensim.downloader.load('word2vec-google-news-300')
	return word2vec_vectors