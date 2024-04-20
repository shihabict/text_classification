import re
import pandas as pd
import string
import unicodedata
from settings import DIR_RESOURCES

class BanglaTextPreprocessor():

    def __init__(self):
        self.number_pattern = re.compile('[0-9]+')
        self.punct = string.punctuation
        self.punct = self.punct + '\n' + '’' + '‘' + ',' + '।'

    def remove_garbage(self, text):
        # remove non bangla text
        text = "".join(i for i in text if i in [".", "।"] or 2432 <= ord(i) <= 2559 or ord(i) == 32)
        # remove newline
        text = text.replace('\n', ' ')
        # remove unnecessary punctuation
        text = re.sub('[^\u0980-\u09FF]', ' ', str(text))
        # remove stopwords
        with open(DIR_RESOURCES + 'bangla_stopwords.txt', 'r', encoding='utf-8') as file:
            stp = file.read().split()
        result = text.split()
        text = [word.strip() for word in result if word not in stp]
        text = " ".join(text)
        return text

    def clean_text(self, text):

        if type(text) is not str:
            return None
        # print(f'Before Cleaning: {text}')
        cleaned_text = self.remove_garbage(text)
        cleaned_text = re.sub(self.number_pattern, ' ', cleaned_text)
        cleaned_text = cleaned_text.strip(self.punct + '\n')
        cleaned_text = cleaned_text.translate(str.maketrans('', '', self.punct))
        cleaned_text = ' '.join(cleaned_text.split())
        cleaned_text = unicodedata.normalize('NFKC', cleaned_text)
        # print(f'After cleaning: {cleaned_text}')
        return cleaned_text

    def preprocess(self,data,feature_col,target_col):
      cleaned_df = pd.DataFrame()
      cleaned_df[feature_col] = data[feature_col].apply(self.clean_text)
      cleaned_df[target_col] = data[target_col]
      cleaned_df.dropna(inplace=True)
      return cleaned_df
