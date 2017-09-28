import sys
from natto import MeCab
import numpy as np


class MorphologicalAnalysis:
    @staticmethod
    def split(doc):
        words = []
        with MeCab() as nm:
            for n in nm.parse(doc, as_nodes=True):
                if not n.is_eos():
                    words.append(n.surface)

        return [word for word in words]

    def get_word_list(self, doc):
        words = [s.lower() for s in self.split(doc)]
        return tuple(w for w in words)


class NaiveBayes:
    def __init__(self):
        self.vocabularies = set()
        self.word_count = {}
        self.category_count = {}

    def train(self, document, category):
        ma = MorphologicalAnalysis()
        words = ma.get_word_list(document)
        for word in words:
            self.__word_count_up(word, category)

        self.__category_count_up(category)

    def __word_count_up(self, word, category):
        self.word_count.setdefault(category, {})
        self.word_count[category].setdefault(word, 0)
        self.word_count[category][word] += 1
        self.vocabularies.add(word)

    def __category_count_up(self, category):
        self.category_count.setdefault(category, 0)
        self.category_count[category] += 1

    def classifier(self, document):
        best_category = None
        max_prob = -sys.maxsize
        ma = MorphologicalAnalysis()
        word_list = ma.get_word_list(document)

        for category in self.category_count.keys():
            prob = self.__score(word_list, category)
            if prob > max_prob:
                max_prob = prob
                best_category = category

        return best_category

    def __score(self, word_list, category):
        score = np.log(self.__prior_prob(category))
        for word in word_list:
            score += np.log(self.__word_prob(word, category))
        return score

    def __prior_prob(self, category):
        cat_count = self.category_count[category]
        cat_values = sum(self.category_count.values())
        return np.float(cat_count / cat_values)

    def __word_prob(self, word, category):
        cat_num = self.__in_category(word, category) + 1.0
        cat_sum = sum(
            self.word_count[category].values()) + len(self.vocabularies) * 1.0
        return cat_num / cat_sum

    def __in_category(self, word, category):
        if word in self.word_count[category]:
            return np.float(self.word_count[category][word])
        return 0.0


if __name__ == '__main__':

    text = 'ラケットスポーツ'

    nb = NaiveBayes()

    nb.train('ボール スポーツ ワールドカップ ボール', 'サッカー')
    nb.train('ボール スポーツ グローブ バット', '野球')
    nb.train('ボール ラケット コート スポーツ', 'テニス')

    result = '%s => 推定カテゴリ: %s' % (text, nb.classifier(text))

    print(result)
