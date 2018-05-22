# coding=utf-8
import tensorflow as tf
import numpy as np
import math


class MTANet(object):
    def __init__(self,
                 n,
                 max_len_char, char_embedding_dim, char_filter_sizes, char_num_filters, char_size,
                 max_len_word, word2vec_dim, word_filter_sizes, word_num_filters,
                 num_topics,
                 fc_units, num_classes,
                 difficulty,
                 l2_reg_lambda=0.0):

        self.char_back = tf.placeholder(tf.float32, shape=[None, max_len_word, max_len_char, 68],
                                        name="char_back")  # max char 10    max word 200
        self.x_char = tf.placeholder(tf.int32, shape=[None, max_len_char], name='x_char')
        self.x_word = tf.placeholder(tf.float32, shape=[None, max_len_word, word2vec_dim], name="x_word")
        self.x_topic = tf.placeholder(tf.float32, shape=[None, num_topics], name="x_topic")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        self.y_profile = tf.placeholder(tf.int32, shape=[None, 2], name="y_profile")

        self.char_dropout_keep = tf.placeholder(tf.float32, name="char_dropout_keep")
        self.word_dropout_keep = tf.placeholder(tf.float32, name="word_dropout_keep")
        self.topic_dropout_keep = tf.placeholder(tf.float32, name="topic_dropout_keep")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # char-level parameters
        self.n = n
        self.char_size = char_size
        self.char_embedding_dim = char_embedding_dim
        self.char_filter_sizes = char_filter_sizes
        self.char_num_filters = char_num_filters
        self.max_len_char = max_len_char

        # word-lever parameters
        self.word2vec_dim = word2vec_dim
        self.max_len_word = max_len_word
        self.word_filter_sizes = word_filter_sizes
        self.word_num_filters = word_num_filters

        # topic-lever parameters
        self.num_topics = num_topics

        # FC and soft-max
        self.fc_units = fc_units
        self.num_classes = num_classes

        # ======================================================================================
        self.difficulty = difficulty
        self.l2_reg_lambda = l2_reg_lambda
        self.l2_loss = 0
        self.pos_info = False

        self.aa_loss = 0
        self.aa_acc = 0
        self.pp_loss = 0
        self.pp_acc = 0

        self._input_info()
        self.build_arch()

    def _input_info(self):
        print("=" * 100)
        print('INFO:')
        print(self.char_back)
        print(self.x_char)
        print(self.x_word)
        print(self.x_topic)
        print(self.y)
        print(self.y_profile)
        print("=" * 100)

    def build_arch(self):
        emb_char = self.char_model()
        emb_word = self._word_model()
        emb_topic = self._topic_model()
        emb = self._feature_fusion(emb_char, emb_word, emb_topic)
        self.aa_loss, self.aa_acc = self.authorship_attribution(emb)
        self.pp_loss, self.pp_acc = self.personality_prediction(emb)

    def _char_model(self):
        # B,T -> B,T,D
        with tf.device("/cpu:0"), tf.name_scope("char_embedding"):
            # vocab size * hidden size
            embedding_var = tf.get_variable(
                name='char_embedding',
                shape=[self.char_size, self.char_embedding_dim],
                trainable=True, )
            embedded_char = tf.nn.embedding_lookup(embedding_var, self.x_char)  # B,T,D

        if self.n == 2:
            if self.pos_info:
                embedded_char = self._positional_encoding(embedded_char)
            # CNN N-GRAMS   效果贼好那个。
            # ===================================================================================
            # B,T,D -> B,T,D,1
            conv_input = tf.expand_dims(embedded_char, -1)
            pooled_outputs = []
            for i, filter_size in enumerate(self.char_filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, self.char_embedding_dim, 1, self.char_num_filters]
                    b = tf.Variable(tf.constant(0.1, shape=[self.char_num_filters]), name="b")
                    conv_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_filter")
                    conv_output = tf.nn.conv2d(
                        input=conv_input,
                        filter=conv_filter,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply non-linearity
                    h = tf.nn.relu(tf.nn.bias_add(conv_output, b), name="relu")
                    # Max pooling over the outputs
                    pooled = tf.nn.max_pool(
                        value=h,
                        ksize=[1, self.max_len_char - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = self.char_num_filters * len(self.char_filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            print("CHAR LEVEL OUTPUT", h_pool_flat)
            print("CHAR Model Done.")
            return h_pool_flat

        if self.n == 1:
            #  embedding_lookup budui
            # LSTM + Max Pooling + MLP
            # ===================================================================================
            with tf.name_scope("bi-LSTM"):
                bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(tf.nn.rnn_cell.BasicLSTMCell(300),
                                                                tf.nn.rnn_cell.BasicLSTMCell(300),
                                                                inputs=embedded_char, dtype=tf.float32)
            h = tf.concat(bi_outputs, 2)  # B,T,H*2
            h = tf.expand_dims(h, -1)
            q_pooling = tf.nn.max_pool(value=h, ksize=[1, self.max_len_char, 1, 1],
                                       padding='VALID', strides=[1, 1, 1, 1], name='biRNN_pooling')
            q_squeezed = tf.squeeze(input=q_pooling, squeeze_dims=[1, 3])
            print("CHAR LEVEL OUTPUT", q_squeezed)  # B,2H
            print("CHAR-LSTM Model Done.")
            return q_squeezed

    # 两层 LSTM
    def char_model(self):
        char_cell = tf.contrib.rnn.LSTMCell(
            256,
            state_is_tuple=True
        )
        word_cell = tf.contrib.rnn.LSTMCell(
            256,
            state_is_tuple=True
        )
        out = []
        # self.batch_size,5,150,300
        with tf.variable_scope("CharLstm"):
            for i in range(self.max_len_word):
                # print("\t\t\tword_step in", i)
                state = char_cell.zero_state(128, dtype=tf.float32)
                for j in range(self.max_len_char):
                    # print("\t\t\t\tchar_step in", j)
                    if i != 0 or j != 0:
                        tf.get_variable_scope().reuse_variables()
                    _, (c_state, h_state) = char_cell(self.char_back[:, i, j, :], state)
                out.append(tf.reshape(h_state, shape=[-1, 1, 256]))
        out = tf.nn.dropout(tf.concat(out, axis=1), 1)
        with tf.variable_scope("WordLstm"):
            for i in range(self.max_len_word):
                state = word_cell.zero_state(128, dtype=tf.float32)
                if i != 0:
                    tf.get_variable_scope().reuse_variables()
                out_, state = word_cell(out[:, i, :], state)
            _, state = state
            out = tf.reshape(state, shape=[-1, 256])
        print("CHAR LEVEL OUTPUT", out)
        print("CHAR Model Done.")
        return out

    def _word_model(self):
        if self.pos_info:
            self.x_word = self._positional_encoding(self.x_word)

        conv_input = tf.expand_dims(self.x_word, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(self.word_filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.word2vec_dim, 1, self.word_num_filters]
                b = tf.Variable(tf.constant(0.1, shape=[self.word_num_filters]), name="b")
                conv_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_filter")
                conv_output = tf.nn.conv2d(
                    input=conv_input,
                    filter=conv_filter,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                # h = tf.nn.relu(tf.nn.bias_add(conv_output, b), name="relu")
                h = tf.nn.dropout(conv_output, self.word_dropout_keep)
                # Max pooling over the outputs
                pooled = tf.nn.max_pool(
                    value=h,
                    ksize=[1, self.max_len_word - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.word_num_filters * len(self.word_filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        print("WORD LEVEL OUTPUT", h_pool_flat)
        print("WORD Model Done.")
        return h_pool_flat

    def _topic_model(self):
        print("TOPIC LEVEL OUTPUT", self.x_topic)
        print("TOPIC Model Done.")
        return self.x_topic

    def _feature_fusion(self, emb_char, emb_word, emb_topic):

        with tf.name_scope('fc_char'):
            char_dropout = tf.nn.dropout(emb_char, self.char_dropout_keep)
            W_char = tf.get_variable(
                "W_char",
                shape=[256, self.fc_units])
            b_char = tf.Variable(tf.constant(1 / self.fc_units, shape=[self.fc_units]), name="b_char")
            # self.l2_loss += tf.nn.l2_loss(W_char)
            # self.l2_loss += tf.nn.l2_loss(b_char)
            char_output = tf.nn.tanh(tf.nn.xw_plus_b(char_dropout, W_char, b_char, name="char_output"))

        with tf.name_scope('fc_word'):
            word_dropout = tf.nn.dropout(emb_word, self.word_dropout_keep)
            W_word = tf.get_variable(
                "W_word",
                shape=[200, self.fc_units])
            b_word = tf.Variable(tf.constant(1 / self.fc_units, shape=[self.fc_units]), name="b_word")
            # self.l2_loss += tf.nn.l2_loss(W_word)
            # self.l2_loss += tf.nn.l2_loss(b_word)
            word_output = tf.nn.tanh(tf.nn.xw_plus_b(word_dropout, W_word, b_word, name="word_output"))

        with tf.name_scope('fc_topic'):
            topic_dropout = tf.nn.dropout(emb_topic, self.topic_dropout_keep)
            W_topic = tf.get_variable(
                "W_topic",
                shape=[self.num_topics, self.fc_units])
            b_topic = tf.Variable(tf.constant(1 / self.fc_units, shape=[self.fc_units]), name="b_topic")
            # self.l2_loss += tf.nn.l2_loss(W_topic)
            # self.l2_loss += tf.nn.l2_loss(b_topic)
            topic_output = tf.nn.tanh(tf.nn.xw_plus_b(topic_dropout, W_topic, b_topic, name="topic_output"))

        combine_list = []
        tensor_list = [char_output, word_output, topic_output]
        for level in self.difficulty:
            combine_list.append(tensor_list[level - 1])
        emb = tf.concat(combine_list, axis=1)
        print("FEATURE FUSION OUTPUT", emb)
        print("FEATURE FUSION Done.")
        return emb

    def authorship_attribution(self, emb):
        with tf.name_scope('fc'):
            fc_dropout = tf.nn.dropout(emb, self.dropout_keep_prob)
            W_fc = tf.get_variable(
                "W_fc",
                shape=[768, self.num_classes])
            b_fc = tf.Variable(tf.constant(1 / self.num_classes, shape=[self.num_classes]), name="b_fc")
            self.l2_loss += tf.nn.l2_loss(W_fc)
            self.l2_loss += tf.nn.l2_loss(b_fc)
            output = tf.nn.tanh(tf.nn.xw_plus_b(fc_dropout, W_fc, b_fc, name="output"))
            # output = tf.layers.dense(inputs=fc_dropout, units=self.num_classes, activation=tf.nn.relu, use_bias=True)

        labels = tf.one_hot(indices=self.y, depth=self.num_classes)
        logits = output

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels),
                                  name="loss")
            loss = loss + self.l2_reg_lambda * self.l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(input=logits, axis=1), tf.argmax(input=labels, axis=1)), tf.float32),
                name="accuracy")
        print("AA DONE")
        return loss, accuracy

    def personality_prediction(self, emb):
        fc_dropout = tf.nn.dropout(emb, self.dropout_keep_prob)

        with tf.name_scope('PP_gender'):
            logits_gender = tf.layers.dense(inputs=fc_dropout, units=2)
            print(self.y_profile[:, 0])
            gender_labels = tf.one_hot(indices=self.y_profile[:, 0], depth=2)
            print(gender_labels)
            gender_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits_gender, labels=gender_labels), name="gender_loss")
            gender_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(input=logits_gender, axis=1), tf.argmax(input=gender_labels, axis=1)),
                        tf.float32),
                name="gender_acc")

        with tf.name_scope('PP_age'):
            logits_age = tf.layers.dense(inputs=fc_dropout, units=5)
            age_labels = tf.one_hot(indices=self.y_profile[:, 1], depth=5)
            age_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_age, labels=age_labels),
                                      name="age_loss")
            age_accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.argmax(input=logits_age, axis=1), tf.argmax(input=age_labels, axis=1)), tf.float32),
                name="age_acc")

        print("PP DONE")
        return age_loss + gender_loss, (age_accuracy, gender_accuracy)

    @staticmethod
    def _pos_func(pos, i, d_model, func="sin"):
        """
        generate tht positional encoding in the sequence used in ATTENTION IS ALL YOU NEED"
        :param pos: the position of the sequence
        :param i: the dimension of the encoding
        :param d_model: the dimension of embedding
        :param func:
        :return:
        """
        # print("we use {} function as default".format(func))
        return np.sin(pos / math.pow(10000, 2 * i / d_model))

    def _positional_encoding(self, emb, mode="simply add"):
        """
        B,T,D -> B,T,D  deal with D,
        :param emb: np type
        :param mode:
        :return: emb + positional_encoding
        """
        # print("we use {} between emb and pos_emb".format(mode))
        _, length, dimension = emb.get_shape().as_list()
        pos_encoding = np.zeros(shape=[length, dimension])
        # pos_encoding = np.zeros()
        # print(length, dimension)
        for pos in range(length):
            for i in range(dimension):
                # tf.assign(pos_encoding[:, pos, i], self._pos_func(pos, i, dimension))
                pos_encoding[pos, i] = self._pos_func(pos, i, dimension)
                print(pos_encoding[pos, i])
                print(emb[:, pos, i])
                emb = emb[:, pos, i] + pos_encoding[pos, i]
        return emb
