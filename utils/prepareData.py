import pandas as pd
import os
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

BASE_DIR = '../data'

def _return_stopword_removed_dataframe():
	stop = stopwords.words('english')

	df = pd.read_csv(os.path.join(BASE_DIR, 'train.csv'), index_col=False)
	df['question1_without_stopword'] = df['question1_without_stopword'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word not in (stop)]))
	df['question2_without_stopword'] = df['question2_without_stopword'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word not in (stop)]))

	return df


def _train_test_split(df):
	train, test = train_test_split(df, test_size=0.2)
	return train, test


