#! -*- coding: utf-8 -*-
# RoFormer-Sim base 基本例子
# 测试环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.10.6

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder
from bert4keras.snippets import uniout

maxlen = 64

# 模型配置
config_path = '/root/kg/bert/chinese_roformer-sim-char_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/root/kg/bert/chinese_roformer-sim-char_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/root/kg/bert/chinese_roformer-sim-char_L-12_H-768_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
roformer = build_transformer_model(
    config_path,
    checkpoint_path,
    model='roformer',
    application='unilm',
    with_pool='linear'
)

encoder = keras.models.Model(roformer.inputs, roformer.outputs[0])
seq2seq = keras.models.Model(roformer.inputs, roformer.outputs[1])


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(seq2seq).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95, mask_idxs=[]):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        for i in mask_idxs:
            token_ids[i] = tokenizer._token_mask_id
        output_ids = self.random_sample([token_ids, segment_ids], n,
                                        topp=topp)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)


def gen_synonyms(text, n=100, k=20, mask_idxs=[]):
    ''''含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    '''
    r = synonyms_generator.generate(text, n, mask_idxs=mask_idxs)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]


"""
gen_synonyms(u'广州和深圳哪个好？')
[
    '深圳和广州哪个好？',
    '广州和深圳哪个好',
    '广州和深圳哪个更好？',
    '深圳和广州哪个更好？',
    '深圳和广州，那个更好？',
    '深圳和广州哪个好一些呢？',
    '深圳好还是广州好？',
    '广州和深圳哪个地方好点？',
    '广州好还是深圳好？',
    '广州和深圳哪个好一点',
    '广州和深圳哪个发展好？',
    '深圳好还是广州好',
    '深圳和广州哪个城市更好些',
    '深圳比广州好吗？',
    '到底深圳和广州哪个好？为什么呢？',
    '深圳究竟好还是广州好',
    '一般是深圳好还是广州好',
    '广州和深圳那个发展好点',
    '好一点的深圳和广州那边好？',
    '深圳比广州好在哪里？'
]

gen_synonyms(u'科学技术是第一生产力。')
[
    '科学技术是第一生产力！',
    '科学技术是第一生产力',
    '一、科学技术是第一生产力。',
    '一是科学技术是第一生产力。',
    '第一，科学技术是第一生产力。',
    '第一生产力是科学技术。',
    '因为科学技术是第一生产力。',
    '科学技术是第一生产力知。',
    '也即科学技术是第一生产力。',
    '科学技术是第一生产力吗',
    '科技是第一生产力。',
    '因此，科学技术是第一生产力。',
    '其次，科学技术是第一生产力。',
    '科学技术才是第一生产力。',
    '科学技术是第一生产力吗？',
    '第二，科学技术是第一生产力。',
    '所以说科学技术是第一生产力。',
    '科学技术确实是第一生产力。',
    '科学技术还是第一生产力',
    '科学技术是第一生产力对吗？'
]
"""
