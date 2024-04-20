from matplotlib import pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def plot_accuracy_and_loss(history,architecture_name):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.savefig(f'{DIR_IMAGES_HISTORY}/{name}_accuracy.png')
    plt.show()
    plt.savefig(f'{architecture_name}_accuracy.png')
    plt.close()
    # Plot training & validation loss values
    if history.history['loss']:
        plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        # plt.savefig(f'{DIR_IMAGES_HISTORY}/{name}_loss.png')
        plt.show()
        plt.savefig(f'{architecture_name}_loss.png')
        plt.close()
    else:
        pass


def plot_confusion_matrix(test_labels, predicted_labels, classes,architecture_name, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues,):
    cm = confusion_matrix(test_labels, predicted_labels)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    plt.savefig(f'{architecture_name}_confusion_matrix.png')
    plt.close()

def classification_report_with_accuracy_score(y_true, y_pred,classes,architecture):
    cls_report = classification_report(y_true, y_pred, target_names=classes) # print classification report
    with open(f'{architecture}_Classification_Report.txt', 'w') as f:
        print(cls_report, file=f)
