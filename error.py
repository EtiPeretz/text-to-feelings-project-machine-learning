from sklearn.metrics import confusion_matrix
#%matplotlib inline
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from classifiers import func_classifiers


def error_analysis(X_train,y_train,X_test,y_test):
    update_grid = func_classifiers(X_train=X_train,
                                     y_train=y_train,
                                     X_test=X_test,
                                     y_test=y_test)
    matrix = confusion_matrix(y_test, update_grid.predict(X_test))
    print(matrix)

    list_of_feelings = ["joy", "sadness", "anger", "fear", "love", "surprise"]
    list_of_feelings.sort()
    df_cm = pd.DataFrame(matrix, index=list_of_feelings, columns=list_of_feelings)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt="d")
    plt.show()
