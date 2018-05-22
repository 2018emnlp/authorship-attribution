import json
import re

import numpy as np
from nltk.tokenize import word_tokenize
from sklearn.externals import joblib

np.set_printoptions(threshold=np.nan)


# get data
# ================================================================================
def get_train_data(path):
    # print('blog path', path)
    with open(path, 'r') as f:
        content = f.read()
        text_list = re.findall(r'<body>([\s\S]*?)</body>', content)
        author_list = re.findall(r'<author id="([\s\S]*?)"/>', content)
        # shuffle data在这shuffle是搞笑呢吗
        # text_list, author_list = shuffle(text_list, author_list, random_state=0)
        return text_list, author_list


def get_dev_data(path1, path2):
    # get texts
    with open(path1, 'r') as f:
        content = f.read()
        text_list = re.findall(r'<body>([\s\S]*?)</body>', content)
    # get authors
    # ./pan11-corpus-train/SmallValid.xml -> ./pan11-corpus-train/GroundTruthSmallValid.xml
    with open(path2, 'r') as f:
        content = f.read()
        author_list = re.findall(r'<author id="([\s\S]*?)"/>', content)
    return text_list, author_list


# get json
# ================================================================================
def get_json(path):
    with open(path, 'r') as f:
        return json.load(f)


# get dict
# ================================================================================
def get_n_grams_dict(texts, n=2):
    n_grams_dict = dict()
    for text in texts:
        char_text = tokenize(text, mode="char")
        for i in range(len(char_text) - n + 1):
            if char_text[i:i + n:1] not in n_grams_dict:
                n_grams_dict[char_text[i:i + n:1]] = len(n_grams_dict) + 1
    return n_grams_dict


def get_author_dict(author_list):
    author_dict = {}
    for author in author_list:
        if author not in author_dict.keys():
            author_dict[author] = 1
        else:
            author_dict[author] += 1
    return author_dict


# zip dict func
# ================================================================================
def zip_dict_0(my_dict):
    values = range(1, len(my_dict) + 1)
    keys = my_dict.keys()
    new_dict = dict(zip(keys, values))
    return new_dict


def zip_dict(my_dict):
    values = range(len(my_dict))
    keys = my_dict.keys()
    new_dict = dict(zip(keys, values))
    return new_dict


# tokenize
# ================================================================================
def tokenize(text, mode="word"):
    # address some problems in text
    text = text.replace(r"<NAME/>", 'NAME')
    text = text.replace(r"<email/>", "EMAIL")
    text = re.sub(r"<([\s\S]*?)>", repl='TAG', string=text)
    # stop words
    # english_stopwords = stopwords.words('english')
    # english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    # english_stopwords = english_stopwords + english_punctuations
    # texts_tokenized = [word.lower() for word in word_tokenize(text) if word.lower() not in english_stopwords]
    if mode == "word":
        text = [word.lower() for word in word_tokenize(text)]
    return text


# generate batch
# ================================================================================
def gen_char_batch(texts, authors, author_dict, n_grams_dict, batch_size, max_len_char, n=2):
    text_list = []
    author_list = []

    for text, author in zip(texts, authors):
        char_list = []
        char_text = tokenize(text, mode="char")
        for i in range(len(char_text) - n + 1):
            if max_len_char == len(char_list):
                break
            else:
                char_list.append(n_grams_dict.get(char_text[i:i + n:1], 0))
        if max_len_char > len(char_list):
            for j in range(max_len_char - len(char_list)):
                char_list.append(0)
        text_list.append(char_list)

        if author not in author_dict:
            raise Exception("There is an author not in my dict!")
        else:
            author_list.append(author_dict[author])

        assert isinstance(batch_size, int)
        if len(text_list) == batch_size:
            yield np.asarray(text_list), np.asarray(author_list)
            text_list = []
            author_list = []


def gen_char_batch_back(texts, authors, char_dict, author_dict, batch_size, max_len_char, max_len_word):
    """

    :param texts:
    :param authors:
    :param char_dict:
    :param author_dict:
    :param batch_size:
    :param max_len_char:
    :param max_len_word:
    :return: B, max_len_word, max_len_char, len(char_dict)+1
    """
    text_list = []
    author_list = []
    for text, author in zip(texts, authors):
        word_list = []
        for step, word in enumerate(tokenize(text, mode="word")):
            if max_len_word == len(word_list):
                break
            else:
                char_list = []
                for char in word:
                    if max_len_char == len(char_list):
                        break
                    else:
                        if char in char_dict.keys():
                            one_hot = np.zeros(len(char_dict) + 1)
                            one_hot[char_dict[char]] = 1
                            char_list.append(one_hot)
                        else:
                            one_hot = np.zeros(len(char_dict) + 1)
                            one_hot[0] = 1
                            char_list.append(one_hot)
                if max_len_char > len(char_list):
                    for i in range(max_len_char - len(char_list)):
                        one_hot = np.zeros(len(char_dict) + 1)
                        char_list.append(one_hot)
                word_list.append(char_list)
        if max_len_word > len(word_list):
            for j in range(max_len_word - len(word_list)):
                word_list.append(np.zeros([max_len_char, len(char_dict) + 1]))
        text_list.append(word_list)

        if author not in author_dict:
            author_list.append(0)
        else:
            author_list.append(author_dict[author])

        assert isinstance(batch_size, int)
        if len(text_list) == batch_size:
            yield np.asarray(text_list, dtype=np.int32), np.asarray(author_list, dtype=np.int32)
            text_list = []
            author_list = []


# use nlp tokenizer
def gen_word_batch(texts, authors, word_vectors, author_dict, batch_size, max_len_word):
    text_list = []
    author_list = []
    for text, author in zip(texts, authors):
        word_list = []
        for step, word in enumerate(tokenize(text, mode="word")):
            if step == max_len_word:
                break
            word_list.append(word_vectors.get(word, np.zeros(300)))
        if max_len_word > len(word_list):
            for i in range(max_len_word - len(word_list)):
                word_list.append(np.zeros(300))
        text_list.append(word_list)

        if author not in author_dict:
            raise Exception("There is an author not in my dict!")
        else:
            author_list.append(author_dict[author])

        assert isinstance(batch_size, int)
        if len(text_list) == batch_size:
            yield np.asarray(text_list), np.asarray(author_list)
            text_list = []
            author_list = []


def gen_topic_batch_back(texts, authors, author_dict, lda_model, batch_size):
    text_list = []
    author_list = []
    id2word = lda_model.id2word
    for text, author in zip(texts, authors):
        # x
        bow_vector = id2word.doc2bow(tokenize(text, mode="word"))
        # topic_distribution
        # print(lda_model[bow_vector])
        topic_distribution = lda_model.get_document_topics(bow_vector, minimum_probability=0)
        topic_p = [topic_probability for topic_id, topic_probability in topic_distribution]
        text_list.append(topic_p)

        if author not in author_dict:
            raise Exception("There is an author not in my dict!")
        else:
            author_list.append(author_dict[author])

        assert isinstance(batch_size, int)
        if len(text_list) == batch_size:
            yield np.asarray(text_list), np.asarray(author_list)
            text_list = []
            author_list = []


def gen_topic_batch(texts, authors, author_dict, word_dict, lda_model, batch_size):
    text_list = []
    author_list = []
    word_num = word_dict["word_num"]
    word_dict = word_dict["w2it"]
    for text, author in zip(texts, authors):
        word_bag = []
        word_count = [0] * word_num
        for step, word in enumerate(tokenize(text, mode="word")):
            if word in word_dict.keys():
                # print(word_dict[word])
                word_count[word_dict[word]] += 1
            else:
                pass
        # print(word_count)
        word_bag.append(word_count)
        word_bag = np.asarray(word_bag)
        topics = np.asarray(lda_model.transform(word_bag))
        # topics = (topics - np.mean(topics)) / np.std(topics)
        # print(np.std(topics))
        # print(np.var(topics))
        # topics = topics / np.var(topics)
        text_list.append(np.squeeze(topics))

        if author not in author_dict:
            raise Exception("There is an author not in my dict!")
        else:
            author_list.append(author_dict[author])

        assert isinstance(batch_size, int)
        if len(text_list) == batch_size:
            yield np.asarray(text_list), np.asarray(author_list)
            text_list = []
            author_list = []


if __name__ == "__main__":
    # load data
    texts, authors = get_train_data('./pan11-corpus-train/LargeTrain.xml')
    print(len(texts), len(authors))

    author_dict = get_json("./dict_data/author_dict.json")
    print(len(author_dict))
    # =========================================================================
    # ===============   gen char  =======================================
    # =========================================================================
    grams_dict = get_json("./dict_data/char_dic.json")
    print(len(grams_dict))
    gen_char = gen_char_batch_back(texts, authors, author_dict=author_dict, char_dict=grams_dict, batch_size=2,
                                   max_len_char=67,
                                   max_len_word=150)
    x_char, y_char = next(gen_char)
    # print(x_char, y_char)
    # =========================================================================
    # ===============   gen word  =======================================
    # =========================================================================
    word2vec = get_json("./dict_data/word_embedding_dic.json")
    gen_word = gen_word_batch(texts, authors, word2vec, author_dict, batch_size=2, max_len_word=150)
    x_word, y_word = next(gen_word)
    # print(x_word, y_word)
    # =========================================================================
    # ===============   gen topic  =======================================
    # =========================================================================
    # lda_model = gensim.models.LdaModel.load("./lda_model/model" + str(150), mmap="r")
    lda_model = joblib.load('./lda_model/LDA_model_LargeTrain_S_150_ac40.6.m')
    word_dict = get_json("./dict_data/word_dic_Large.json")
    # texts, authors, author_dict, word_dict, lda_model, batch_size
    gen_topic = gen_topic_batch(texts, authors, author_dict, word_dict, lda_model, 2)
    for i in range(10):
        x_topic, y_topic = next(gen_topic)
        print(x_topic[0:10][0:10])
