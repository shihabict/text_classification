# Train Deep Learning Text Classifier
import os
import warnings
from deep_learning_approach.train import DLClassifier
from transformer_based_approach.train import TransformerClassifier

architecture = 'Transformer'
data_dir = 'data/racism_v4.json'
feature_col = 'cleanText'
target_col = 'category'

warnings.filterwarnings("ignore")

if architecture == 'LSTM':

    dl_classifier = DLClassifier(architecture)
    dl_classifier.train_model(data_dir,feature_col,target_col)
    print(f"Training Completed")
elif architecture == 'GRU':
    dl_classifier = DLClassifier(architecture)
    dl_classifier.train_model(data_dir,feature_col,target_col)
    print(f"Training Completed")
elif architecture == 'BiLSTM':
    dl_classifier = DLClassifier(architecture)
    dl_classifier.train_model(data_dir,feature_col,target_col)
    print(f"Training Completed")
elif architecture == 'SimpleRNN':
    dl_classifier = DLClassifier(architecture)
    dl_classifier.train_model(data_dir,feature_col,target_col)
    print(f"Training Completed")
elif architecture == 'Transformer':
    transformer_classifier = TransformerClassifier()
    transformer_classifier = TransformerClassifier()
    transformer_classifier.train_model(data_dir,feature_col,target_col)
    print(f"Training Completed")

