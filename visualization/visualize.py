import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DataVisualizer():

    # def __init__(self, data_path):
    #     data = pd.read_csv(data_path)
    #     data['text'].astype(str)

    def show_class_distribution(self, data, target_col):
        sns.set(font_scale=1.4)
        data[target_col].value_counts().plot(kind='barh', figsize=(8, 6))
        plt.xlabel("Number of Articles", labelpad=12)
        plt.ylabel(f"{target_col}", labelpad=12)
        plt.yticks(rotation=45)
        plt.title(f"Dataset Distribution of {target_col}", y=1.02)
        plt.show()
        plt.close()

    def show_document_length_distribution(self, data, col_name):
        data['Length'] = data[col_name].apply(lambda x: len(str(x).split()))
        matplotlib.rc_file_defaults()
        frequency = dict()
        for i in data.Length:
            frequency[i] = frequency.get(i, 0) + 1

        plt.figure(figsize=(4, 2))
        plt.bar(frequency.keys(), frequency.values(), color=(0.2, 0.4, 0.6, 0.6))
        # plt.xlim(12, 100)

        plt.xlabel('Length of the Documents')
        plt.ylabel('Frequency')
        plt.title('Length-Frequency Distribution')
        file_path = DIR_IMAGES_EDA + f'document_length_distribution.png'
        plt.savefig(file_path)
        plt.show()
        plt.close()
        print(f'document length distribution image saved to - {file_path}')

        print(f"Min : {data.Length.min()}\n")
        print(f"Max : {data.Length.max()}\n")
        print(f"Maximum Length of a Document: {max(data.Length)}\n")
        print(f"Minimum Length of a Document: {min(data.Length)}\n")
        print(f"Average Length of a Document: {round(np.mean(data.Length), 0)}\n")

        report_file_path = DIR_IMAGES_EDA + f'document_length.txt'
        with open(report_file_path, 'w') as file:
            file.write(f"Min : {data.Length.min()}\n")
            file.write(f"Max : {data.Length.max()}\n")
            file.write(f"Maximum Length of a Document: {max(data.Length)}\n")
            file.write(f"Minimum Length of a Document: {min(data.Length)}\n")
            file.write(f"Average Length of a Document: {round(np.mean(data.Length), 0)}\n")

    def show_data_summary(self):

        report_file_path = DIR_IMAGES_EDA + f'data_summary.txt'
        with open(report_file_path, 'w') as file:
            file.write('\n______________________showing data summary ________________________________\n')
            documents = []
            words = []
            u_words = []
            # find class names
            class_label = [k for k, v in data['category'].value_counts().to_dict().items()]
            print(class_label)
            for label in class_label:
                word_list = [word.strip().lower() for t in list(data[data['category'] == label]['text']) for word in
                             str(t).strip().split()]
                counts = dict()
                for word in word_list:
                    counts[word] = counts.get(word, 0) + 1
                # sort the dictionary of word list
                ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
                # Documents per class
                documents.append(len(list(data[data['category'] == label]['text'])))
                # Total Word per class
                words.append(len(word_list))
                # Unique words per class
                u_words.append(len(np.unique(word_list)))

                file.write(f"\nClass Name : {label}")
                file.write(f"\nNumber of Documents: {len(list(data[data['category'] == label]['text']))}")
                file.write(f"\nNumber of Words:{len(word_list)}")
                file.write(f"\nNumber of Unique Words:{len(np.unique(word_list))}")
                file.write(f"\nMost Frequent Words:\n")
                for k, v in ordered[:10]:
                    file.write(f"\n{k}\t{v}")

            data_matrix = pd.DataFrame({'Total Documents': documents,
                                        'Total Words': words,
                                        'Unique Words': u_words,
                                        'Class Names': class_label})
            file.write('\n______________________summary________________________________\n')
            file.write(f'{data_matrix}\n')

        df = pd.melt(data_matrix, id_vars="Class Names", var_name="Category", value_name="Values")
        # plt.figure(figsize=(8, 6))
        ax = plt.subplot()

        sns.barplot(data=df, x='Class Names', y='Values', hue='Category')
        ax.set_xlabel('Class Names')
        ax.set_title('Data Statistics')
        class_names = class_label
        ax.xaxis.set_ticklabels(class_names, rotation=45)
        file_path = DIR_IMAGES_EDA + f'data_summary.png'
        plt.savefig(file_path)
        plt.show()
        plt.close()

    def main(self):
        self.show_class_distribution()
        self.show_document_length_distribution()
        self.show_data_summary()
