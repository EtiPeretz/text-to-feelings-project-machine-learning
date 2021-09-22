from utils import read_and_transform_data_to_vec, create_features_and_labels, show_info
from classifiers import classifiers

TRAIN = "train"
TEST = "test"


def main():
    show_info()
    X_train, y_train = create_features_and_labels(f_name=TRAIN)
    X_test, y_test = create_features_and_labels(f_name=TEST)
    choose_classifiers = classifiers(X_train=X_train,
                                     y_train=y_train,
                                     X_test=X_test,
                                     y_test=y_test)

if __name__ == '__main__':
    main()
