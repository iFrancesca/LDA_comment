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
from gensim.corpora import Dictionary
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from lda_topic import get_lda_input
from basic import split_by_comment, MyComments


def topic_analyze(comments):
    # screen_names, texts = self.get_all_texts()  ####获取所有文本
    bow_corpus = [" ".join(word_list) for word_list in comments]
    # bow_corpus = []
    # for trace in texts:
    #     bow_corpus.append(trace)
    train_size = int(round(len(bow_corpus) * 0.8))  ###分解训练集和测试集
    train_index = sorted(random.sample(range(len(bow_corpus)), train_size))  ###随机选取下标
    test_index = sorted(set(range(len(bow_corpus))) - set(train_index))
    train_corpus = [bow_corpus[i] for i in train_index]  #训练集
    test_corpus = [bow_corpus[j] for j in test_index]    #测试集

    n_features = 2000
    n_top_words = 1000

    print("Extracting tf features for lda...")
    tf_vectorizer = CountVectorizer()
    tf = tf_vectorizer.fit_transform(train_corpus)  ###使用向量生成器转化测试集
    # Use tf (raw term count) features for lda.
    print("Extracting tf features for lda...")
    tf_test = tf_vectorizer.transform(test_corpus)

    grid = dict()
    t0 = time()
    for i in range(1, 100, 5):  ###100个主题，以5为间隔
        grid[i] = list()
        n_topics = i

        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5, learning_method='online',
                                        learning_offset=50., random_state=0)  ###定义lda模型
        lda.fit(tf)  ###训练参数
        train_gamma = lda.transform(tf)  # 得到topic-document 分布
        test_perplexity = lda.perplexity(tf_test)  # s计算测试集困惑度
        print('sklearn preplexity: test=%.3f' % (test_perplexity))

        grid[i].append(test_perplexity)

    print("done in %0.3fs." % (time() - t0))

    df = pd.DataFrame(grid)
    df.to_csv('sklearn_perplexity.csv')
    print(df)
    plt.figure(figsize=(14, 8), dpi=120)
    # plt.subplot(221)
    plt.plot(df.columns.values, df.iloc[0].values, '#007A99')
    plt.xticks(df.columns.values)
    plt.ylabel('train Perplexity')
    plt.show()
    plt.savefig('lda_topic_perplexity_ana.png', bbox_inches='tight', pad_inches=0.1)


def gensim_lda_perplexity(comments):
    bow_corpus = list(comments)
    # 所有词的词典
    dct = Dictionary(bow_corpus)
    # 词袋模型 bag-of-words (BoW) format = list of (token_id, token_count).
    bow_corpus = [dct.doc2bow(_) for _ in bow_corpus]
    # tfidf词频逆文档矩阵
    tfidf = TfidfModel(bow_corpus)
    # corpus_tfidf = tfidf[bow_corpus]
    bow_corpus = tfidf[bow_corpus]

    train_size = int(round(len(bow_corpus) * 0.8))  ###分解训练集和测试集
    train_index = sorted(random.sample(range(len(bow_corpus)), train_size))  ###随机选取下标
    test_index = sorted(set(range(len(bow_corpus))) - set(train_index))
    train_corpus = [bow_corpus[i] for i in train_index]
    test_corpus = [bow_corpus[j] for j in test_index]
    grid = dict()
    t0 = time()
    for i in range(3, 100, 3):  ###100个主题，以5为间隔
        grid[i] = list()
        ldamodel = LdaModel(corpus=train_corpus, num_topics=i, id2word=dct)
        ldamodel.print_topics(num_topics=i)
        perword_perplexity = ldamodel.log_perplexity(test_corpus)
        print(perword_perplexity)
        grid[i].append(perword_perplexity)

    print("done in %0.3fs." % (time() - t0))

    df = pd.DataFrame(grid)
    df.to_csv('gensim_perplexity.csv')
    plt.figure(figsize=(14, 8), dpi=120)
    plt.subplot(221)
    plt.plot(df.columns.values, df.iloc[0].values, '#007A99')
    plt.xticks(df.columns.values)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('主题数目')
    plt.ylabel('train Perplexity大小')
    plt.title('主题-困惑度变化情况')
    plt.show()
    plt.savefig('lda_topic_perplexity_gensium.png', bbox_inches='tight', pad_inches=0.1)


def main():
    comment_list = split_by_comment("data/data.xlsx")
    comments = MyComments(comment_list)
    #topic_analyze(comments)
    gensim_lda_perplexity(comments)  #两种计算混淆度的方法


if __name__ == '__main__':
    main()
