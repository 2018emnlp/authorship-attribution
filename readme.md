# Multi-task learning for Authorship Attribution with Hierarchical Features

This repository is An implementation of our paper [Multi-task learning for Authorship Attribution with Hierarchical Features]() in Tensorflow.

The [dict_data](dict_data) are a collection of dictionary data that be used in our model.

The [lda_model](lda_model) are a collection of pre-trained LDA models implemented in gensim by gen_lda.py 

The others contain train/dev/test data.

## Requirements

- Python 3
- Tensorflow > 1.1
- Numpy
- nltk == 3.2.4
- gensim == 2.3.0
- lda == 1.0.5

## Training
Print parameters:

```bash
./train.py --help
```

```
optional arguments:
  --[no]allow_soft_placement: Allow device soft device placement
    (default: 'true')
  --author_dict: Data source for author dict
    (default: './dict_data/author_dict.json')
  --batch_size: Batch Size (default: 128)
    (default: '128')
    (an integer)
  --char_dropout_keep: Char-level dropout keep probability (default: 1.0)
    (default: '1.0')
    (a number)
  --char_embedding_dim: Dimensionality of char embedding (default: 300)
    (default: '300')
    (an integer)
  --char_filter_sizes: Comma-separated filter sizes (default: 3,4,5)
    (default: '3,4,5')
  --char_num_filters: Number of filters per filter size (default: 256)
    (default: '256')
    (an integer)
  --char_size: Number of characters in docs (default: 4750)
    (default: '4750')
    (an integer)
  --checkpoint_every: Save model after this many steps (default: 100)
    (default: '100')
    (an integer)
  --dev_data_x: Data source for the X of dev data
    (default: './pan11-corpus-test/LargeValid.xml')
  --dev_data_y: Data source for the Y of dev data
    (default: './pan11-corpus-test/GroundTruthLargeValid.xml')
  --difficulty: Submodule. 1: char level, 2: word level, 3: topic level(default:
    1,2,3)
    (default: '1,2,3')
  --dropout_keep_prob: FC layer dropout keep probability (default: 1.0)
    (default: '1.0')
    (a number)
  --evaluate_every: Evaluate model on dev set after this many steps (default:
    100)
    (default: '100')
    (an integer)
  --fc_units: Number of final penultimate FC layer's units (default: 256)
    (default: '360')
    (an integer)
  --gpu_fraction: gpu_memory_fraction (default: 0.0, gpu_options.allow_growth =
    True
    (default: '0.0')
    (a number)
  --[no]is_verbose: Print loss (default: True)
    (default: 'false')
  --l2_reg_lambda: L2 regularization lambda (default: 0.0)
    (default: '0.0')
    (a number)
  --lda_path: LDA model file path
    (default: './lda_model/model')
  --learning_rate: Learning rate (default: 0.003)
    (default: '0.003')
    (a number)
  --[no]log_device_placement: Log placement of ops on devices
    (default: 'false')
  --max_len_char: Number of characters in a sequence (default: 5 >> 140)
    (default: '5')
    (an integer)
  --max_len_word: Number of words in a sequence (default: 150)
    (default: '150')
    (an integer)
  --[no]multi_task: Train profile data (default: False)
    (default: 'false')
  --n: n-grams, 1-gram equals char_dict (default: 1)
    (default: '1')
    (an integer)
  --n_grams_dict: Data path for n-grams dict
    (default: './dict_data/n_grams_dict')
  --num_checkpoints: Number of checkpoints to store (default: 5)
    (default: '5')
    (an integer)
  --num_classes: Number of authors(default: 72
    (default: '72')
    (an integer)
  --num_epochs: Number of training epochs (default: 200)
    (default: '200')
    (an integer)
  --num_topics: Number of LDA topics (default: 150)
    (default: '150')
    (an integer)
  --profile_data: The path to profile data
    (default: './pan14-profile-train/train.json')
  --run_mode: A type of running model. Possible options are: tiny, random,
    standard
    (default: 'standard')
  --tl_batch_size: TL batch Size (default: 128)
    (default: '128')
    (an integer)
  --tl_learning_rate: TL learning rate (default: 0.001)
    (default: '0.0003')
    (a number)
  --tl_num_epochs: Number of TL training epochs (default: 10)
    (default: '10')
    (an integer)
  --topic_dropout_keep: Topic-level dropout keep probability (default: 1.0)
    (default: '1.0')
    (a number)
  --train_data: Data source for the train data
    (default: './pan11-corpus-train/LargeTrain.xml')
  --word2vec: Data source for prepared word2vec dict
    (default: './dict_data/word_embedding_dic.json')
  --word2vec_dim: Dimensionality of word embedding (default: 300)
    (default: '300')
    (an integer)
  --word_dropout_keep: Word-level dropout keep probability (default: 1.0)
    (default: '1.0')
    (a number)
  --word_filter_sizes: Comma-separated filter sizes (default: 3)
    (default: '3')
  --word_num_filters: Number of filters per filter size (default: 200)
    (default: '200')
    (an integer)
```

Train:

```bash
python train.py --run_mode=standard --difficulty=1,2,3 --multi_task=True --dropout_keep_pro=0.2
--char_dropout_keep=0.6 --word_dropout_keep =0.6 --topic_dropout_keep=0.9 --l2_reg_lambda=0.001
```

## Evaluating

```bash
./eval.py --checkpoint_dir="./123m/1525940731/checkpoints/"
```

Replace the checkpoint dir with the output from the training. To use your own data, change the `eval.py` script to load your data.

## Accuracies

PAN11

| multi-task        | Accuracy   |
| --------   | -----:  |
| multi-task     | 0.6144 |
| w/o multi-task        |   0.601   |
| w/o word-level features        |    0.556    |
| w/o chr-level features        |    0.598    |
| w/o topic-level features        |    0.503    |
