# -*- coding:utf-8 -*-

import numpy as np
from basic import *
from sklearn.feature_extraction.text import CountVectorizer
import lda
import matplotlib.pyplot as plt
import seaborn as sns


def get_lda_input(comments):
    """
    获取ldas输入数据
    :param chapters:
    :return: 词频矩阵（one-hot编码）, vectorizer(lda建模时用的词汇表)
    """
    corpus = [" ".join(word_list) for word_list in comments]
    vectorizer = CountVectorizer()         #词袋
    X = vectorizer.fit_transform(corpus)   #文档---->词语向量
    return X.toarray(), vectorizer


def get_lda_topic_num(weight, vectorizer):
    likehoods = []
    for n in range(3, 21, 2):
        model = lda.LDA(n_topics=n, n_iter=1000, random_state=1)
        model.fit(weight)
        likehood = model.loglikelihood()
        likehoods.append(likehood)
    index = list(range(3, 21, 2))
    plt.figure(figsize=(14, 8), dpi=120)
    plt.plot(index, likehoods, '#007A99')
    plt.xticks(index)
    plt.ylabel('train likehood')
    plt.show()
    plt.savefig('lda_topic_likehood.png', bbox_inches='tight', pad_inches=0.1)


def lda_train(weight, vectorizer):
    """
    训练模型
    :param weight:
    :param vectorizer:
    :return:
    """
    model = lda.LDA(n_topics=15, n_iter=1000, random_state=1)
    model.fit(weight)
    model.loglikelihood()
    doc_num = len(weight)
    topic_word = model.topic_word_
    vocab = vectorizer.get_feature_names()
    titles = ["第{}个评论".format(i) for i in range(1, doc_num + 1)]
    # 获取每个主题最相关的20个单词
    n_top_words = 20
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    # 获取每个文档最相关的3个主题
    doc_topic = model.doc_topic_
    print(doc_topic, type(doc_topic))
    topic_word = model.topic_word_
    print(topic_word, type(doc_topic))
    plot_topic(doc_topic[:20,:])   #绘制前10个评论的主题热力图
    for i in range(doc_num):
        print("{} (top topic: {})".format(titles[i], np.argsort(doc_topic[i])[:-4:-1]))


def plot_topic(doc_topic):
    """ 热力图 """
    f, ax = plt.subplots(figsize=(10, 4))
    cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(doc_topic, cmap=cmap, linewidths=0.05, ax=ax)
    ax.set_title('proportion per topic in every comment')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ax.set_xlabel('主题')
    ax.set_ylabel('每条评论')
    plt.show()
    f.savefig('output/topic_heatmap_comm20.jpg', bbox_inches='tight')


def main():
    comment_list = split_by_comment("data/data.xlsx")
    comments = MyComments(comment_list)
    weight, vectorizer = get_lda_input(comments)

    #print(weight)
    lda_train(weight, vectorizer)

    #get_lda_topic_num(weight=weight, vectorizer=vectorizer)


if __name__ == '__main__':
    main()
