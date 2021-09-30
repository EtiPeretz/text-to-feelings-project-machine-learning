from tabulate import tabulate
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from features import feature

def train_test(classifier, X_train, X_test, y_train, y_test):
    classifier.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, classifier.predict(X_train))
    test_acc = accuracy_score(y_test, classifier.predict(X_test))
    return train_acc, test_acc


def func_classifiers(X_train, X_test, y_train, y_test):
    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Classifiers
    svc = SVC()
    lsvc = LinearSVC(dual=False, max_iter=10000)
    adb = AdaBoostClassifier(n_estimators=100, random_state=0)
    rforest = RandomForestClassifier(n_estimators=10)
    dtree = DecisionTreeClassifier()

    classifiers = [svc, lsvc, adb, rforest, dtree]

    # train and test them
    #print("| {:25} | {} | {} |".format("Classifier", "Training Accuracy", "Test Accuracy"))
    #print("| {} | {} | {} |".format("-"*25, "-"*17, "-"*13))
    # collect the results
    table = []
    for classifier in classifiers:
        train_acc, test_acc = train_test(classifier, X_train, X_test, y_train, y_test)
        result_of_classifier = []
        #adding the name of the classifer to the first column of the classified name
        clf_name = classifier.__class__.__name__
        result_of_classifier.append(clf_name)
        result_of_classifier.append(train_acc)
        result_of_classifier.append(test_acc)
        #add all the result of per classifier to the table
        table.append(result_of_classifier)
        #print("| {:25} | {:17.7f} | {:13.7f} |".format(clf_name, train_acc, test_acc))

    headers = ["Classifier", "Training Accuracy", "Test Accuracy"]
    print(tabulate(table, headers, tablefmt="fancy_grid"))

    parameters = {'C':[3, 5, 10],
                  'tol': [0.1, 0.01, 0.001]}
    lsvc = LinearSVC(dual=False,random_state=123, max_iter=10000)
    grid_obj = GridSearchCV(lsvc, param_grid=parameters, cv=5)
    grid_obj.fit(X_train, y_train)
    print("Validation acc: {}".format(grid_obj.best_score_))
    print("Training acc: {}".format(accuracy_score(y_train, grid_obj.predict(X_train))))
    print("Test acc    : {}".format(accuracy_score(y_test, grid_obj.predict(X_test))))
    print("Best parameter: {}".format(grid_obj.best_params_))

    matrix = confusion_matrix(y_test, grid_obj.predict(X_test))
    print(matrix)

    list_of_feelings = ["joy", "sadness", "anger", "fear", "love", "surprise"]
    list_of_feelings.sort()
    # df_cm = pd.DataFrame(matrix, index=list_of_feelings, columns=list_of_feelings)
    # plt.figure(figsize=(10, 6))
    # sn.heatmap(df_cm, annot=True, fmt="d")
    # plt.show()

    list_of_emoji = {"joy": "😂", "sadness": "😢", "anger": "😠", "fear": "😱", "love": "😍", "surprise": "😵"}

    # example sentences and the emojis the classifier associates them with
    sentence1 = "I am so surprised"
    sentence2 = "I had such an energetic time on the hike!"
    sentence3 = "I feel very sad"
    sentence4 = "I am so mad"
    sentence5 = "How can you do this to me?"
    texts = [sentence1, sentence2, sentence3, sentence4, sentence5]

    for text in texts:
        print(text)
        feat = feature(text, _range=(1, 4))
        feat = vectorizer.transform(feat)
        prediction = grid_obj.predict(feat)[0]
        if prediction == '':
            print("Couldnt Classify")
        else:
            print("{} {}".format(list_of_emoji[prediction], text))

