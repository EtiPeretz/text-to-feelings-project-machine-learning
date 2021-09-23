import pandas as pd
from labels import find_label_from_vector
from features import features


def show_info():
    train_data = pd.read_csv("data/train.txt", sep=";", header=None, names=["content", "feeling"])
    print("*** TRAIN: DATA INFO ***")
    print(train_data["feeling"].value_counts())
    test_data = pd.read_csv("data/test.txt", sep=";", header=None, names=["content", "feeling"])
    print("")
    print("*** TEST: DATA INFO ***")
    print(test_data["feeling"].value_counts())
    print("")

def read_and_transform_data_to_vec(f_name, transform=False):
    files_dict = {
        "train": "data/train.txt",
        "test": "data/test.txt",
        "val": "data/val.txt"
    }
    path = files_dict[f_name]
    data = __read_and_transform(path)
    if transform:
        data = __transform_to_vec(data)
    return data


def __read_and_transform(f_name: str):
    data = []
    with open(f_name, 'r') as file:
        for l in file:
            index = l.find(";")
            content = l[:index].strip()
            feeling = l[index+1:].strip()
            data.append([feeling, content])
    return data


def __transform_to_vec(data: list):
    """joy:     [1. 0. 0. 0. 0. 0.]
       sadness: [0. 1. 0. 0. 0. 0.]
       anger:   [0. 0. 1. 0. 0. 0.]
       fear:    [0. 0. 0. 1. 0. 0.]
       love:    [0. 0. 0. 0. 1. 0.]
       surprise:[0. 0. 0. 0. 0. 1.]"""
    feelings_dict = {
        "joy": "1. 0. 0. 0. 0. 0.",
        "sadness": "0. 1. 0. 0. 0. 0.",
        "anger": "0. 0. 1. 0. 0. 0.",
        "fear": "0. 0. 0. 1. 0. 0.",
        "love": "0. 0. 0. 0. 1. 0.",
        "surprise": "0. 0. 0. 0. 0. 1."
        }
    transformed = []
    for d in data:
        transformed_feeling = feelings_dict[d[0]]
        transformed.append([transformed_feeling, d[1]])
    return transformed


def create_features_and_labels(f_name: str, transform=False):
    data = read_and_transform_data_to_vec(f_name)
    feelings = ["joy", "sadness", "anger", "fear", "love", "surprise"]
    X, y = [], []
    for emotion, content in data:
        feeling = find_label_from_vector(emotion, feelings) if transform else emotion
        X.append(features(content=content, _range=(1, 4)))
        y.append(feeling)
    return X, y
