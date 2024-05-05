import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import AutoModelForPreTraining, AutoTokenizer
from normalizer import normalize # pip install git+https://github.com/csebuetnlp/normalizer

from preprocessing.preprocess import BanglaTextPreprocessor
from settings import DIR_REPORTS

# datapath = f'/content/drive/MyDrive/Interview/Silicon Orchard/DATA/cleaned_data.csv'
# df = pd.read_csv(datapath)
# df.dropna(inplace=True)
# df.head()

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
labels = {'Neutral': 0,
          'Political': 1,
          'Threat': 2,
          'sexual': 3,
          'troll': 4
          }


class Dataset(torch.utils.data.Dataset):

    def __init__(self, df, feature_col, target_col, labels, tokenizer):
        self.labels = [labels[label] for label in df[target_col]]
        self.texts = [tokenizer(text,
                                padding='max_length', max_length=512, truncation=True,
                                return_tensors="pt") for text in df[feature_col]]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y


class BERTMODEL(nn.Module):

    def __init__(self, dropout=0.5):
        super(BERTMODEL, self).__init__()
        # self.bert = AutoModelForPreTraining.from_pretrained("csebuetnlp/banglabert")
        self.bert = BertModel.from_pretrained('csebuetnlp/banglabert')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)

        return final_layer


class TransformerClassifier:
    def __init__(self):
        self.preprocessor = BanglaTextPreprocessor()
        self.epochs = 10
        # self.architecture = architecture
        self.train_batch_size = 8
        self.val_batch_size = 8
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BERTMODEL()
        self.tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
        self.report_path = f"{DIR_REPORTS}"
        # self.dropout_rate = 0.2
        # self.embedding_units = 128

    def load_data(self, data_path):
        # check data type csv,json
        if data_path.split('.')[-1] == 'csv':
            data = pd.read_csv(data_path)
        elif data_path.split('.')[-1] == 'json':
            data = pd.read_json(data_path)
        else:
            pass
        return data

    def preprocess_data(self, data, feature_col, target_col):
        cleaned_data = self.preprocessor.preprocess(data, feature_col, target_col)
        return cleaned_data

    def draw_figure(self, train_loss, val_loss, type):
        # plot the train loss and test loss per iteration
        plt.plot(train_loss, label=f'train_{type}')
        plt.plot(val_loss, label=f'test_{type}')
        plt.legend()
        # plt.show()
        plt.savefig(f'{self.report_path}/transformer_{type}.png')
        plt.close()

    # def train_model(self,model, train_data, val_data, learning_rate, epochs):
    def train_model(self, data_path, feature_col, target_col):
        # load data
        data = self.load_data(data_path)
        cleaned_data = self.preprocess_data(data, feature_col, target_col)
        # preprocess data
        data = cleaned_data
        # get n rows from each class
        # data = data.groupby(target_col).head(100)
        data[target_col] = data[target_col].astype('int')
        np.random.seed(112)
        df_train, df_val, df_test = np.split(data.sample(frac=1, random_state=42),
                                             [int(.8 * len(data)), int(.9 * len(data))])

        # texts = data[feature_col].tolist()
        # labels = data[target_col].tolist()
        #
        # # Convert labels to numerical values
        # label_set = sorted(set(labels))
        # label_mapping = {label: index for index, label in enumerate(label_set)}
        # classes = label_mapping.keys()
        # labels = [label_mapping[label] for label in labels]
        # num_classes = len(label_set)

        labels = data[target_col].tolist()
        # Convert labels to numerical values
        label_set = sorted(set(labels))
        label_mapping = {label: index for index, label in enumerate(label_set)}
        labels = [label_mapping[label] for label in labels]
        train, val = Dataset(df_train, feature_col, target_col, labels, self.tokenizer), Dataset(df_val, feature_col,
                                                                                                 target_col, labels,
                                                                                                 self.tokenizer)

        train_dataloader = torch.utils.data.DataLoader(train, batch_size=self.train_batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val, batch_size=self.val_batch_size)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        criterion = nn.CrossEntropyLoss()
        model = BERTMODEL()

        optimizer = Adam(model.parameters(), lr=1e-6)

        if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()

        train_losses = np.zeros(self.epochs)
        val_losses = np.zeros(self.epochs)
        train_accuracies = np.zeros(self.epochs)
        val_accuracies = np.zeros(self.epochs)

        for epoch_num in range(self.epochs):

            total_acc_train = 0
            total_loss_train = 0
            train_loss = []
            val_loss = []
            train_acc = []
            val_acc = []
            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                mask = train_input['attention_mask'].to(device)
                input_id = train_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()
                train_loss.append(batch_loss.item())

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc
                train_acc.append(acc)

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()
            train_loss = np.mean(train_loss)
            train_acc = np.mean(train_acc)

            total_acc_val = 0
            total_loss_val = 0

            with torch.no_grad():

                for val_input, val_label in val_dataloader:
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)
                    input_id = val_input['input_ids'].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()
                    val_loss.append(batch_loss.item())

                    acc = (output.argmax(dim=1) == val_label).sum().item()
                    total_acc_val += acc
                    val_acc.append(acc)
            val_loss = np.mean(val_loss)
            val_acc = np.mean(val_acc)

            # save accuracy and losses
            train_losses[epoch_num] = total_loss_train
            train_accuracies[epoch_num] = total_acc_train
            val_losses[epoch_num] = total_loss_val
            val_accuracies[epoch_num] = total_acc_val

            print(
                f'Epochs: {epoch_num} | Train Loss: {total_loss_train / len(df_train): .3f} | Train Accuracy: {total_acc_train / len(df_train): .3f} | Val Loss: {total_loss_val / len(df_val): .3f} | Val Accuracy: {total_acc_val / len(df_val): .3f}')

        self.draw_figure(train_losses, val_losses, 'loss')
        self.draw_figure(train_accuracies, val_accuracies, 'accuracy')

    def evaluate(self, model, test_data):
        test = Dataset(test_data)

        test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")

        if use_cuda:
            model = model.cuda()

        total_acc_test = 0
        with torch.no_grad():

            for test_input, test_label in test_dataloader:
                test_label = test_label.to(device)
                mask = test_input['attention_mask'].to(device)
                input_id = test_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                acc = (output.argmax(dim=1) == test_label).sum().item()
                total_acc_test += acc
        bert_json = {'transformerbasedEmbedding': total_acc_test / len(test_data)}
        add_into_existing_json(bert_json, f'{DIR_IMAGES_EDA}mul_accuracy_score.json')
        print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

# np.random.seed(112)
# df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
#                                      [int(.8 * len(df)), int(.9 * len(df))])
#
# print(len(df_train), len(df_val), len(df_test))
#
# EPOCHS = 5
# model = BERTMODEL()
# LR = 1e-6
#
# train(model, df_train, df_val, LR, EPOCHS)
