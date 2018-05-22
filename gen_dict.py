from data_helper import *
import json


def save_dict(path, data):
    with open(path, 'w') as f:
        f.write(json.dumps(data))


# author_dict
# ==========================================================================
train_texts, train_authors = get_train_data('./pan11-corpus-train/LargeTrain.xml')
author_dict = zip_dict(get_author_dict(train_authors))
save_dict('./dict_data/author_dict.json', author_dict)

# char_dict
# ==========================================================================
# char_dict = zip_dict_0(get_char_dict(train_texts))
# save_dict('./dict_data/char_dict.json', char_dict)
# char_dict = get_json('./dict_data/char_dict.json')
# print(char_dict)
# print(len(char_dict))

# 1-gram == char_dict
n_grams_dict1 = get_n_grams_dict(train_texts, n=1)
save_dict('./dict_data/n_grams_dict1.json', n_grams_dict1)

# 2-grams
n_grams_dict2 = get_n_grams_dict(train_texts, n=2)
save_dict('./dict_data/n_grams_dict2.json', n_grams_dict2)

if __name__ == "__main__":
    # test
    # ==========================================================================
    author_dict = get_json('./dict_data/author_dict.json')
    assert type(author_dict) == dict, 'error'
    print(author_dict)
    print(len(author_dict))

    n_grams_dict = get_json("./dict_data/n_grams_dict1.json")
    print(n_grams_dict)
    print(len(n_grams_dict))

    n_grams_dict1 = get_json("./dict_data/n_grams_dict2.json")
    print(n_grams_dict1)
    print(len(n_grams_dict1))
    # word2vec  we use Jiang's dict namely word_embedding_dic.json just for now
    # ==========================================================================
