# -*- coding: utf-8 -*-

"""
@Datetime: 2022/12/26
@Author: Shen Yajun
"""

import random
from time import time
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import LdaModel, TfidfModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import pandas as pd
from matplotlib import pyplot as plt

import jieba
jieba.setLogLevel(jieba.logging.INFO)

from lda_topic import get_lda_input
from basic import split_by_comment, MyComments



#计算coherence主题一致性
def coherence(num_topics, comments):
    bow_corpus = list(comments)
    # 所有词的词典
    dct = Dictionary(bow_corpus)
    # 词袋模型 bag-of-words (BoW) format = list of (token_id, token_count).
    bow_corpus = [dct.doc2bow(_) for _ in bow_corpus]
    # tfidf词频逆文档矩阵
    tfidf = TfidfModel(bow_corpus)
    # corpus_tfidf = tfidf[bow_corpus]
    bow_corpus = tfidf[bow_corpus]


    ldamodel = LdaModel(corpus=bow_corpus, num_topics=num_topics, id2word = dct, passes=30,random_state = 1)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
    ldacm = CoherenceModel(model=ldamodel, corpus=bow_corpus , dictionary=dct, coherence='u_mass')
    print(ldacm.get_coherence())

    return ldacm.get_coherence()




def main():
    comment_list = split_by_comment("data/data.xlsx")
    comments = MyComments(comment_list)

    x = range(3, 21)
    # z = [perplexity(i) for i in x]  #如果想用困惑度就选这个
    y = [coherence(i, comments) for i in x]
    plt.plot(x, y)
    plt.xlabel('主题数目')
    plt.ylabel('coherence大小')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('主题-coherence变化情况')
    plt.savefig('lda_topic_coherence.png', bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    main()