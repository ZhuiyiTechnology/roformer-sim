#! -*- coding: utf-8 -*-
# 测试有监督版RoFormer-Sim-FT的相似度效果

import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from keras.models import Model

maxlen = 64

# bert配置
config_path = '/root/kg/bert/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roformer-sim-char-ft_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roformer-sim-char-ft_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roformer',
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])


def similarity(text1, text2):
    """"计算text1与text2的相似度
    """
    texts = [text1, text2]
    X, S = [], []
    for t in texts:
        x, s = tokenizer.encode(t, maxlen=maxlen)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    return (Z[0] * Z[1]).sum()


similarity(u'今天天气不错', u'今天天气很好')
similarity(u'今天天气不错', u'今天天气不好')
similarity(u'我喜欢北京', u'我很喜欢北京')
similarity(u'我喜欢北京', u'我不喜欢北京')
similarity(u'电影不错', u'电影很好')
similarity(u'电影不错', u'电影不好')
similarity(u'红色的苹果', u'绿色的苹果')
similarity(u'给我推荐一款红色的车', u'给我推荐一款黑色的车')
similarity(u'给我推荐一款红色的车', u'推荐一辆红车')
similarity(u'给我推荐一款红色的车', u'麻烦来一辆红车')
