# from sklearn.svm import LinearSVC
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV
#
# #tune parameters
#
#
# def tune_parameters(X_train, y_train, X_test, y_test):
#     parameters = {'C': [3,5,10], 'tol': [0.1, 0.01, 0.001]}
#
#     lsvc = LinearSVC(dual=False, random_state=123, max_iter=10000)
#     print("eti1")
#     grid_obj = GridSearchCV(lsvc, param_grid=parameters, cv=5)
#     print("eti2")
#     grid_obj.fit(X_train, y_train)
#     print("eti3")
#
#     print(f"Validation Accuracy: {grid_obj.best_score_}")
#     print(f"Training Accuracy  : {accuracy_score(y_train, grid_obj.predict(X_train))}")
#     print(f"Test Accuracy      : {accuracy_score(y_test, grid_obj.predict(X_test))}")
#     print(f"Best parameter     : {grid_obj.best_params_}")
