from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Dropout,Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from fasttext import load_model

from common_functions import plot_accuracy_and_loss, plot_confusion_matrix, classification_report_with_accuracy_score
from preprocessing.preprocess import BanglaTextPreprocessor
from settings import DIR_RESOURCES


class DLClassifier:
    def __init__(self, architecture):
        self.preprocessor = BanglaTextPreprocessor()
        self.epochs = 2
        self.architecture = architecture
        self.batch_size = 32
        self.feature_col = 'text'
        self.target_col = 'sentiment'
        self.tokenizer = Tokenizer()
        self.dropout_rate = 0.2
        self.embedding_units = 128

    def load_data(self, data_path):
        # check data type csv,json
        if data_path.split('.')[-1] == 'csv':
            data = pd.read_csv(data_path)
        elif data_path.split('.')[-1] == 'json':
            data = pd.read_json(data_path)
        else:
            pass
        return data

    def preprocess_data(self, data):
        cleaned_data = self.preprocessor.preprocess(data,self.feature_col,self.target_col)
        return cleaned_data

    def calc_tfidf(self, data, col_name):

        tfidf = TfidfVectorizer(use_idf=True, tokenizer=lambda x: x.split())
        X = tfidf.fit_transform(data[col_name])
        coo_matrix = X.tocoo()
        tuples = zip(coo_matrix.col, coo_matrix.data)
        feature_names = tfidf.get_feature_names_out()

        sorted_items = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

        return tfidf, X

    def label_encoding(self, data, sentiment):

        le = LabelEncoder()
        le.fit(data[sentiment])
        encoded_labels = le.transform(data[sentiment])
        labels = np.array(encoded_labels)  # Converting into numpy array
        class_names = le.classes_  ## Define the class names again

        return labels

    def dataset_split(self, X, Y):

        X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.9,
                                                            test_size=0.1, random_state=0)

        return X_train, X_test, y_train, y_test

    def create_model(self, architecture, vocab_size, max_len, units, dropout_rate, num_classes):
        model = Sequential()
        model.add(Embedding(vocab_size, 128, input_length=max_len))
        if architecture == 'LSTM':
            model.add(LSTM(units))
        elif architecture == 'GRU':
            model.add(GRU(units))
        elif architecture == 'BiLSTM':
            model.add(Bidirectional(LSTM(units)))
        else:
            model.add(SimpleRNN(units))
        model.add(Dropout(dropout_rate))
        if num_classes > 2:
            model.add(Dense(num_classes, activation='softmax'))
        else:
            model.add(Dense(num_classes, activation='sigmoid'))
        return model

    def train_model(self, data_path):

        # load data
        data = self.load_data(data_path)
        cleaned_data = self.preprocess_data(data)
        # preprocess data
        data = cleaned_data

        texts = data[self.feature_col].tolist()
        labels = data[self.target_col].tolist()

        # Convert labels to numerical values
        label_set = sorted(set(labels))
        label_mapping = {label: index for index, label in enumerate(label_set)}
        classes = label_mapping.keys()
        labels = [label_mapping[label] for label in labels]
        num_classes = len(label_set)

        # Split the data into training and testing sets
        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2,
                                                                              random_state=42)

        train_labels = to_categorical(train_labels, num_classes=num_classes)
        test_labels = to_categorical(test_labels, num_classes=num_classes)
        self.tokenizer.fit_on_texts(train_texts)
        train_sequences = self.tokenizer.texts_to_sequences(train_texts)
        test_sequences = self.tokenizer.texts_to_sequences(test_texts)
        max_sequence_length = max(max(len(seq) for seq in train_sequences), max(len(seq) for seq in test_sequences))
        vocab_size = len(self.tokenizer.word_index) + 1
        train_data = pad_sequences(train_sequences, maxlen=max_sequence_length)
        test_data = pad_sequences(test_sequences, maxlen=max_sequence_length)

        # Load FastText embeddings
        embedding_model = load_model("resources/cc.bn.300.bin")
        embedding_dim = embedding_model.get_dimension()

        # Prepare embedding matrix
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for word, i in self.tokenizer.word_index.items():
            if word in embedding_model:
                embedding_matrix[i] = embedding_model[word]

        # Initialize the model
        model = self.create_model(self.architecture, vocab_size=vocab_size, max_len=max_sequence_length,
                                  units=self.embedding_units, dropout_rate=self.dropout_rate, num_classes=num_classes)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(train_data, np.array(train_labels), batch_size=self.batch_size, epochs=self.epochs,
                            verbose=1)

        plot_accuracy_and_loss(history,self.architecture)

        # Evaluate the model
        test_loss, test_acc = model.evaluate(test_data, np.array(test_labels), verbose=0)
        print('Test loss:', test_loss)
        print('Test accuracy:', test_acc)

        y_pred = model.predict(test_data)

        predicted_labels = np.argmax(y_pred, axis=1)
        test_labels = np.argmax(test_labels, axis=1)
        # predicted_classes = [label_mapping[label] for label in predicted_labels]
        # print(f"Accuracy score validation data : {accuracy_score(y_test, y_pred)}")
        # lstm_acc = f1_score(test_labels, predicted_labels, average='macro')

        # add_into_existing_json({'lstm_word_embedding': lstm_acc}, f'{DIR_IMAGES_EDA}mul_accuracy_score.json')
        plot_confusion_matrix(test_labels, predicted_labels, classes, self.architecture)
        classification_report_with_accuracy_score(test_labels, predicted_labels, classes,self.architecture)
