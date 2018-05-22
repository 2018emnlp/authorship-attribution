import json
import re

import gensim
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.utils import shuffle
from sklearn.externals import joblib
np.set_printoptions(threshold=np.nan)
from data_helper import *


# load data
texts, authors = get_train_data('./pan11-corpus-train/LargeTrain.xml')
print(len(texts), len(authors))

author_dict = get_json("./dict_data/author_dict.json")
print(len(author_dict))
lda_model = joblib.load('./lda_model/LDA_model_LargeTrain_S_150_ac40.6.m')
word_dict = get_json("./dict_data/word_dic_Large.json")
# word_num = word_dict["word_num"]
# word_dict = word_dict["w2it"]
# print(word_num, word_dict)
# print(word_dict.keys())
# print(type(word_dict['5215']))
gen_topic = gen_topic_batch(texts, authors, author_dict, word_dict, lda_model, 2)
for i in range(10):
    x_topic, y_topic = next(gen_topic)
    print(x_topic[0:10][0:10])
