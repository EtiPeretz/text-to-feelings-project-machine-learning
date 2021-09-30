import re
from collections import Counter as C


def feature(content, _range):
    _range = _range if _range else (1, 1)
    content = content.lower()
    features_ = []

    def make_ngram(token, n):
        output = []
        for i in range(n - 1, len(token)):
            ngram = ' '.join(token[i - n + 1:i + 1])
            output.append(ngram)
        return output

    # find punctuation and use them
    punctuation = re.sub('[a-z0-9]', ' ', content)
    punc_list = punctuation.split()
    features_ += make_ngram(punc_list, 1)  # add to features

    only_alphanumeric = re.sub('[^a-z0-9]', ' ', content)  # if it's not alpahnumeric -> replace with space
    for n in range(_range[0], _range[1] + 1):
        alphanumeric_list = only_alphanumeric.split()
        features_ += make_ngram(alphanumeric_list, n)

    return C(features_)