
from torch.utils.data import Dataset
from utils.weightedTfIdfVectorizer import TfIdfEmbeddingVectorizer


class SiameseDataset(Dataset):
    def __init__(self, df, vectorizer):
        self.df = df
        self.vectorizer = vectorizer
    
    def __getitem__(self, idx):
        return self.vectorizer(self.df.iloc[idx]['question1']),
        		self.vectorizer(self.df.iloc[idx]['question2']),
        		self.df.iloc[idx]['is_duplicate']
    
    def __len__(self):
        return self.df.shape[0]