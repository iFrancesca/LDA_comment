# -*- coding:utf-8 -*-

import re

import pandas as pd

import jieba.posseg

jieba.load_userdict("data/person.txt")

STOP_WORDS = set([w.strip() for w in open("data/stopwords.txt", encoding='utf-8').readlines()])


class MyComments(object):
    def __init__(self, comment_list):
        self.comment_list = comment_list

    def __iter__(self):
        for comment in self.comment_list:
            yield cut_words_with_pos(comment)


def split_by_comment(filepath):
    """
    将文档按章节切割
    comment_list： [[comment1],[comment2],[..],[..],...]
    """
    df = pd.read_excel(filepath, header=None)
    df.rename(columns={0: 'reviews'}, inplace=True)
    #去除重复评论
    reviews = df.copy()
    reviews = reviews.drop_duplicates()
    print(f'去重前：{df.shape[0]}条； 去重后:{reviews.shape[0]}条')
    print(f'删除重复评论{df.shape[0] - reviews.shape[0]}条')

    comment_list = reviews['reviews'].to_list()

    return comment_list


def cut_words_with_pos(text):
    """
    按照词性对每一评论进行切词
    :param text:
    :return:
    """
    seg = jieba.posseg.cut(text)
    res = []
    for i in seg:
        if i.flag in ["a", "v", "x", "n", "an", "vn", "nz", "nt", "nr"] and is_fine_word(i.word):
            res.append(i.word)

    return res


def is_fine_word(word, min_length=2):
    """
    过滤词长，过滤停用词，只保留中文
    :param word:
    :param min_length:
    :return:
    """
    rule = re.compile(r"^[\u4e00-\u9fa5]+$")  #只保留汉字
    if len(word) >= min_length and word not in STOP_WORDS and re.search(rule, word):
        return True
    else:
        return False
