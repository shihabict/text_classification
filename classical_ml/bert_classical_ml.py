import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
DIR_REPORTS = '../REPORT'

class BERTClassicalMLClassifier:
    def __init__(self):
        self.model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.LE = LabelEncoder()

    def process_data(self):
        df = pd.read_json("../data/racism_v4.json").sample(frac=1)
        df['label'] = self.LE.fit_transform(df['category'])

        df_train = df[0:2200]
        df_val = df[2200:]

        # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # model = AutoModel.from_pretrained("distilbert-base-uncased").to(device)

        tokenized_train = self.tokenizer(df_train["cleanText"].values.tolist(), padding=True, truncation=True,
                                    return_tensors="pt")
        tokenized_val = self.tokenizer(df_val["cleanText"].values.tolist(), padding=True, truncation=True,
                                  return_tensors="pt")

        # move on device (GPU)
        tokenized_train = {k: torch.tensor(v).to(device) for k, v in tokenized_train.items()}
        tokenized_val = {k: torch.tensor(v).to(device) for k, v in tokenized_val.items()}

        with torch.no_grad():
            hidden_train = self.model(**tokenized_train)  # dim : [batch_size(nr_sentences), tokens, emb_dim]
            hidden_val = self.model(**tokenized_val)

        # get only the [CLS] hidden states
        cls_train = hidden_train.last_hidden_state[:, 0, :]
        cls_val = hidden_val.last_hidden_state[:, 0, :]

        x_train = cls_train.to("cpu")
        y_train = df_train["label"]

        x_val = cls_val.to("cpu")
        y_val = df_val["label"]

        return x_train, y_train, x_val, y_val

    def train_model(self, alogorithm, x_train, y_train, x_val, y_val):

        if alogorithm == 'rf':
            print(f"Training model: RandomForestClassifier")
            rf = RandomForestClassifier(n_estimators=300, criterion='gini', random_state=0)
            rf.fit(x_train, y_train)
            print(f"Classification report: {rf.score(x_val, y_val)}")
            with open(f'{DIR_REPORTS}/bert_ml_classification_report.txt', 'a') as f:
                f.write(f"Training model: RandomForestClassifier\n")
                f.write(f"Classification Accuracy: {rf.score(x_val, y_val)}\n")
        elif alogorithm == 'lr':
            print(f"Training model: LogisticRegression")
            lr = LogisticRegression(random_state=123, max_iter=300)
            lr.fit(x_train, y_train)
            print(f"Classification report: {lr.score(x_val, y_val)}")
            with open(f'{DIR_REPORTS}/bert_ml_classification_report.txt', 'a') as f:
                f.write(f"Training model: LogisticRegression\n")
                f.write(f"Classification Accuracy: {lr.score(x_val, y_val)}\n")

        elif alogorithm == 'svm_linear':
            print(f"Training model: LinearSVC")
            svm = SVC(kernel='linear', C=0.2, probability=True, random_state=0)
            svm.fit(x_train, y_train)
            print(f"Classification report: {svm.score(x_val, y_val)}")
            with open(f'{DIR_REPORTS}/bert_ml_classification_report.txt', 'a') as f:
                f.write(f"Training model: LinearSVC\n")
                f.write(f"Classification Accuracy: {svm.score(x_val, y_val)}\n")

        elif alogorithm == 'svm_rbf':
            print(f"Training model: SVC RBF")
            svm = SVC(kernel='rbf', C=0.2, probability=True, random_state=0)
            svm.fit(x_train, y_train)
            print(f"Classification report: {svm.score(x_val, y_val)}")
            with open(f'{DIR_REPORTS}/bert_ml_classification_report.txt', 'a') as f:
                f.write(f"Training model: SVC RBF\n")
                f.write(f"Classification Accuracy: {svm.score(x_val, y_val)}\n")

        elif alogorithm == 'knn':
            print(f"Training model: KNeighborsClassifier")
            knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski')
            knn.fit(x_train, y_train)
            print(f"Classification report: {knn.score(x_val, y_val)}")
            with open(f'{DIR_REPORTS}/bert_ml_classification_report.txt', 'a') as f:
                f.write(f"Training model: KNeighborsClassifier\n")
                f.write(f"Classification Accuracy: {knn.score(x_val, y_val)}\n")

        elif alogorithm == 'dt':
            print(f"Training model: DecisionTreeClassifier")
            dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
            dt.fit(x_train, y_train)
            print(f"Classification report: {dt.score(x_val, y_val)}")
            with open(f'{DIR_REPORTS}/bert_ml_classification_report.txt', 'a') as f:
                f.write(f"Training model: DecisionTreeClassifier\n")
                f.write(f"Classification Accuracy: {dt.score(x_val, y_val)}\n")

        # elif alogorithm == 'nb':
        #     print(f"Training model: MultinomialNB")
        #     nb = MultinomialNB()
        #     nb.fit(x_train, y_train)
        #     print(f"Classification report: {nb.score(x_val, y_val)}")

    def main(self):
        x_train, y_train, x_val, y_val = self.process_data()
        for rf in ['rf', 'lr', 'svm_linear', 'svm_rbf', 'knn', 'dt', 'nb']:
            self.train_model(rf, x_train, y_train, x_val, y_val)


if __name__ == "__main__":
    classifier = BERTClassicalMLClassifier()
    classifier.main()
