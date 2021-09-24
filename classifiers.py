from tabulate import tabulate
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def train_test(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))

    #     print("Training acc: {}".format(train_acc))
    #     print("Test acc    : {}".format(test_acc))

    return train_acc, test_acc


def classifiers(X_train, X_test, y_train, y_test):
    vectorizer = DictVectorizer(sparse=True)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Classifiers
    svc = SVC()
    lsvc = LinearSVC(dual=False, random_state=123)
    gnb = GaussianNB()
    rforest = RandomForestClassifier(random_state=123)
    dtree = DecisionTreeClassifier()

    classifiers = [svc,
                   lsvc,
                   gnb,
                   rforest,
                   dtree]

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


    parameters = {'C':[1, 2, 3, 5, 10, 15, 20, 30, 50, 70, 100],
                 'tol':[0.1, 0.01, 0.001, 0.0001, 0.00001]}

    lsvc = LinearSVC(random_state=123)
    grid_obj = GridSearchCV(lsvc, param_grid = parameters, cv=5)
    grid_obj.fit(X_train, y_train)

    print(f"Validation Accuracy: {grid_obj.best_score_}")
    print(f"Training Accuracy  : {accuracy_score(y_train, grid_obj.predict(X_train))}")
    print(f"Test Accuracy      : {accuracy_score(y_test, grid_obj.predict(X_test))}")
    print(f"Best parameter     : {grid_obj.best_params_}")