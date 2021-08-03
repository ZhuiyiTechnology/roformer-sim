#! -*- coding: utf-8 -*-
# SimBERT v2 监督训练代码
# 训练环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.10.6

import json, glob
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_weight_decay
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.snippets import text_segmentate, truncate_sequences
from bert4keras.snippets import AutoRegressiveDecoder, open

# 基本信息
maxlen = 64
batch_size = 192
steps_per_epoch = 1000
epochs = 10000
labels = ['contradiction', 'entailment', 'neutral']

# bert配置
config_path = '/root/kg/bert/chinese_roformer-sim-char_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roformer-sim-char_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roformer-sim-char_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def split(text):
    """分割句子
    """
    seps, strips = u'\n。！？!?；;，, ', u'；;，, '
    return text_segmentate(text, maxlen * 1.2, seps, strips)


def load_data_1(filename, threshold=0.5):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                l[0], l[1] = split(l[0])[0], split(l[1])[0]
                D.append((l[0], l[1], int(float(l[2]) > threshold)))
    return D


# 加载数据集
data_path = '/root/senteval_cn/'
datasets_1 = []
for task_name in ['ATEC', 'BQ', 'LCQMC', 'PAWSX', 'STS-B', 'SOHU21-SSB']:
    for f in ['train', 'valid']:
        threshold = 2.5 if task_name == 'STS-B' else 0.5
        filename = '%s%s/%s.%s.data' % (data_path, task_name, task_name, f)
        datasets_1 += load_data_1(filename, threshold)


def load_data_2(filename):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            if l['gold_label'] not in labels:
                continue
            text1 = split(l['sentence1'])[0]
            text2 = split(l['sentence2'])[0]
            label = labels.index(l['gold_label']) + 2
            D.append((text1, text2, label))
    return D


# 加载数据集
datasets_2 = []
for f in glob.glob('/root/cnsd/cnsd-*/*.jsonl'):
    datasets_2 += load_data_2(f)


def corpus():
    """合并语料，1:1采样
    """
    def generator(dataset):
        while True:
            idxs = np.random.permutation(len(dataset))
            for i in idxs:
                yield dataset[i]

    corpus_1 = generator(datasets_1)
    corpus_2 = generator(datasets_2)

    while True:
        yield next(corpus_1)
        yield next(corpus_2)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            for text in [text1, text2]:
                token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = np.array(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


def merge(inputs):
    """向量合并：a、b、|a-b|拼接
    """
    a, b = inputs[::2], inputs[1::2]
    o = K.concatenate([a, b, K.abs(a - b)], axis=1)
    return K.repeat_elements(o, 2, 0)


def special_crossentropy(y_true, y_pred):
    """特殊的交叉熵
    """
    task = K.cast(y_true < 1.5, K.floatx())
    mask = K.constant([[0, 0, 1, 1, 1]])
    y_pred_1 = y_pred - mask * 1e12
    y_pred_2 = y_pred - (1 - mask) * 1e12
    y_pred = task * y_pred_1 + (1 - task) * y_pred_2
    y_true = K.cast(y_true, 'int32')
    loss = K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    return K.mean(loss)


# 建立加载模型
encoder = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roformer',
    with_pool='linear',
    dropout_rate=0.2,
    ignore_invalid_weights=True
)
output = keras.layers.Lambda(merge)(encoder.output)
output = keras.layers.Dense(5, use_bias=False)(output)

model = keras.models.Model(encoder.inputs, output)
AdamW = extend_with_weight_decay(Adam, 'AdamW')
optimizer = AdamW(learning_rate=2e-5, weight_decay_rate=0.01)
model.compile(loss=special_crossentropy, optimizer=optimizer)
model.summary()


class Evaluator(keras.callbacks.Callback):
    """保存模型
    """
    def on_epoch_end(self, epoch, logs=None):
        encoder.save_weights('./latest_model.weights')
        if (epoch + 1) % 5 == 0:
            encoder.save_weights('roformer-sim.%s.weights' % (epoch + 1))


if __name__ == '__main__':

    train_generator = data_generator(corpus(), batch_size)
    evaluator = Evaluator()

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )

else:

    encoder.load_weights('./latest_model.weights')
