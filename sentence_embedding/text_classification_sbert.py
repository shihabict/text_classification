import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import string
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words

DIR_REPORTS = '../REPORT'


class TextClassificationSBERT:
    def __init__(self):
        self.model = SentenceTransformer('/home/shihab/learning_projects/bangla-sentence-transformer/bangla_snt')
        self.LE = LabelEncoder()

    def data_preparation(self,path):
        data = pd.read_json(path)
        # data = data[['cleanText','category']][:500]
        data = data[['cleanText','category']]

        data['embeddings'] = data['cleanText'].apply(self.model.encode)

        data['label'] = self.LE.fit_transform(data['category'])

        X = data['embeddings'].to_list()
        y = data['label'].to_list()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        return X_train, X_test, y_train, y_test

    def train_model(self,alogrithm,X_train, X_val, y_train, y_val):
        if alogrithm == 'rf':
            print(f"Training model: RandomForestClassifier")
            rf = RandomForestClassifier(n_estimators=300, criterion='gini', random_state=0)
            rf.fit(X_train, y_train)
            print(f"Classification report: {rf.score(X_val, y_val)}")
            with open(f'{DIR_REPORTS}/Sbert_classification_report.txt', 'a') as f:
                f.write(f"Training model: RandomForestClassifier\n")
                f.write(f"Classification Accuracy: {rf.score(X_val, y_val)}\n")
        elif alogrithm == 'lr':
            print(f"Training model: LogisticRegression")
            lr = LogisticRegression(random_state=123, max_iter=300)
            lr.fit(X_train, y_train)
            print(f"Classification report: {lr.score(X_val, y_val)}")
            with open(f'{DIR_REPORTS}/Sbert_classification_report.txt', 'a') as f:
                f.write(f"Training model: LogisticRegression\n")
                f.write(f"Classification Accuracy: {lr.score(X_val, y_val)}\n")

        elif alogrithm == 'svm_linear':
            print(f"Training model: LinearSVC")
            svm = SVC(kernel='linear', C=0.2, probability=True, random_state=0)
            svm.fit(X_train, y_train)
            print(f"Classification report: {svm.score(X_val, y_val)}")
            with open(f'{DIR_REPORTS}/Sbert_classification_report.txt', 'a') as f:
                f.write(f"Training model: LinearSVC\n")
                f.write(f"Classification Accuracy: {svm.score(X_val, y_val)}\n")

        elif alogrithm == 'svm_rbf':
            print(f"Training model: SVC RBF")
            svm = SVC(kernel='rbf', C=0.2, probability=True, random_state=0)
            svm.fit(X_train, y_train)
            print(f"Classification report: {svm.score(X_val, y_val)}")
            with open(f'{DIR_REPORTS}/Sbert_classification_report.txt', 'a') as f:
                f.write(f"Training model: SVC RBF\n")
                f.write(f"Classification Accuracy: {svm.score(X_val, y_val)}\n")

        elif alogrithm == 'knn':
            print(f"Training model: KNeighborsClassifier")
            knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
            knn.fit(X_train, y_train)
            print(f"Classification report: {knn.score(X_val, y_val)}")
            with open(f'{DIR_REPORTS}/Sbert_classification_report.txt', 'a') as f:
                f.write(f"Training model: KNeighborsClassifier\n")
                f.write(f"Classification Accuracy: {knn.score(X_val, y_val)}\n")

        elif alogrithm == 'dt':
            print(f"Training model: DecisionTreeClassifier")
            dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
            dt.fit(X_train, y_train)
            print(f"Classification report: {dt.score(X_val, y_val)}")
            with open(f'{DIR_REPORTS}/Sbert_classification_report.txt', 'a') as f:
                f.write(f"Training model: DecisionTreeClassifier\n")
                f.write(f"Classification Accuracy: {dt.score(X_val, y_val)}\n")

    def main(self,data_path):
        x_train, y_train, x_val, y_val = self.data_preparation(data_path)
        for rf in ['rf', 'lr', 'svm_linear', 'svm_rbf', 'knn', 'dt', 'nb']:
            self.train_model(rf, x_train, y_train, x_val, y_val)


if __name__ == "__main__":
    sbert = TextClassificationSBERT()
    sbert.main('../data/racism_v4.json')
