# -*- coding: utf-8 -*-

"""
@Datetime: 2019/4/9
@Author: Zhang Yafei
"""
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import LdaModel, TfidfModel
from gensim.corpora import Dictionary
from basic import split_by_comment, MyComments


def gensim_lda(comments):
    bow_corpus = list(comments)
    # 所有词的词典
    dct = Dictionary(bow_corpus)
    # 词袋模型 bag-of-words (BoW) format = list of (token_id, token_count).
    bow_corpus = [dct.doc2bow(_) for _ in bow_corpus]
    # tfidf词频逆文档矩阵
    tfidf = TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    ldamodel = LdaModel(corpus=corpus_tfidf, num_topics=18, id2word=dct)
    for corpus in corpus_tfidf:
        for index, score in sorted(ldamodel[corpus], key=lambda tup: -1 * tup[1]):
            print("Score: {}\t Topic {}: {}".format(score, index, ldamodel.print_topic(index, 20)))
    for corpus in corpus_tfidf:
        print(ldamodel.get_document_topics(corpus))
        # for index, score in sorted(ldamodel[corpus], key=lambda tup: -1 * tup[1]):
        #     print("Score: {}\t Topic {}: {}".format(score, index, ldamodel.get_document_topics(index)))

    print("\n" + "end" + "\n" + 100 * "-")


def main():
    comment_list = split_by_comment("data/data.xlsx")
    comments = MyComments(comment_list)
    gensim_lda(comments)


if __name__ == '__main__':
    main()