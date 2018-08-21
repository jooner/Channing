import re
import string

import numpy as np
import torch
import torch.nn as nn

from stop_words import get_stop_words
from contraction_list import conlst


emb_dim = 50
window = 2


with open('../data/glove.6B.50d.txt', 'r') as f:
    glove = f.readlines()

w2v = dict()
for line in glove:
    line = line.split()
    w2v[line[0]] = np.array(list(map(float, line[1:])))

sample = """ 
Now, Mr. Wehrum is about to deliver one
of the biggest victories yet for his industry
clients — this time from inside the Trump
administration as the government's top air
pollution official. On Tuesday, President Trump
is expected to propose a vast rollback of
regulations on emissions from coal plants,
including many owned by members of a coal-burning trade
association that had retained Mr. Wehrum and his firm as
recently as last year to push for the changes.
"""


class PreProcessor(object):
    def __init__(self):
        self.stop_words = get_stop_words('english')

    def clean(self, text):
        def remove_punc_stop(tokens):
            cleaned = []
            for tok in tokens:
                t = []
                for c in tok:
                    if not c.isdigit() and c not in string.punctuation:
                        t.append(c)
                processed_token = ''.join(t)
                if processed_token != '':
                    if processed_token not in self.stop_words:
                        cleaned.append(processed_token)
            return cleaned
        text = text.lower()
        pattern = re.compile(r'\b(' + '|'.join(conlst.keys()) + r')\b')
        text = pattern.sub(lambda x: conlst[x.group()], text)
        tokens = remove_punc_stop(text.split())
        return tokens


def encode(tokens):
    ret = []
    for t in tokens:
        if t in w2v.keys():
            ret.append(w2v[t])
    return np.stack(ret)


def check(tokens):
    ret = []
    for t in tokens:
        if t in w2v.keys():
            ret.append(t)
    return np.stack(ret)


def _prep_train(vectorized_txt):
    x, y = [], []
    for i in range(window, vectorized_txt.shape[0] - (window + 1), 1):
        snippet = []
        for j in range(-window, window + 1):
            if j != 0:
                snippet.append(vectorized_txt[i + j])
        x.append(np.stack(snippet))
        y.append(vectorized_txt[i])
    return np.stack(x), np.stack(y)


def prep_train(text):
    x, y = [], []
    for i in range(window, len(text) - (window + 1), 1):
        snippet = []
        for j in range(-window, window + 1):
            if j != 0:
                snippet.append(text[i + j])
        x.append(np.stack(snippet))
        y.append(text[i])
    return np.stack(x), np.stack(y)


def w2v_dict_to_torch_emb(w2v_dict):
    w2x = dict()
    pretrained_weight = list()
    embs = nn.Embedding(len(w2v_dict), emb_dim)
    for i, (w, e) in enumerate(w2v_dict.items()):
        w2x[w] = i
        pretrained_weight.append(e)
    embs.weight.data.copy_(torch.from_numpy(np.stack(pretrained_weight)))
    return embs, w2x


# embs, w2x = w2v_dict_to_torch_emb(w2v)
# embs(torch.LongTensor([w2x['hillary']]))
