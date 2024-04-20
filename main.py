# Train Deep Learning Text Classifier
import os

from deep_learning_approach.train import DLClassifier
architecture = 'BiLSTM'
dl_classifier = DLClassifier(architecture)
dl_classifier.train_model('data/bn_sentiment.csv')
print(f"Training Completed")

