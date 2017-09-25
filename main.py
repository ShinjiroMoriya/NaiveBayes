import math
import sys
from janome.tokenizer import Tokenizer


class MorphologicalAnalysis:
    @staticmethod
    def split(doc):
        t = Tokenizer()
        tokens = t.tokenize(doc)
        word_list = []
        for token in tokens:
            word_list.append(token.surface)
        return [word for word in word_list]

    def get_word_list(self, doc):
        words = [s.lower() for s in self.split(doc)]
        return tuple(w for w in words)


class NaiveBayes:

    # コンストラクタ
    def __init__(self):

        # 学習データの全単語の集合(加算スムージング用)
        self.vocabularies = set()

        # 学習データのカテゴリー毎の単語セット用
        self.word_count = {}

        # 学習データのカテゴリー毎の文書数セット用
        self.category_count = {}

    # 学習
    def train(self, document, category):

        # 学習文書を形態素解析
        ma = MorphologicalAnalysis()
        word_list = ma.get_word_list(document)

        for word in word_list:

            # カテゴリー内の単語出現回数をUP
            self.__word_count_up(word, category)

        # カテゴリーの文書数をUP
        self.__category_count_up(category)

    # 学習データのカテゴリー内の単語出現回数をUP
    def __word_count_up(self, word, category):

        # 新カテゴリーなら追加
        self.word_count.setdefault(category, {})

        # カテゴリー内で新単語なら追加
        self.word_count[category].setdefault(word, 0)

        # カテゴリー内の単語出現回数をUP
        self.word_count[category][word] += 1

        # 学習データの全単語集合に加える(重複排除)
        self.vocabularies.add(word)

    # 学習データのカテゴリーの文書数をUP
    def __category_count_up(self, category):

        # 新カテゴリーなら追加
        self.category_count.setdefault(category, 0)

        # カテゴリーの文書数をUP
        self.category_count[category] += 1

    # 分類
    def classifier(self, document):

        # もっとも近いカテゴリ
        best_category = None

        # 最小整数値を設定
        max_prob = -sys.maxsize

        # 対象文書を形態素解析
        ma = MorphologicalAnalysis()
        word_list = ma.get_word_list(document)

        # カテゴリ毎に文書内のカテゴリー出現率P(C|D)を求める
        for category in self.category_count.keys():

            # 文書内のカテゴリー出現率P(C|D)を求める
            prob = self.__score(word_list, category)

            if prob > max_prob:
                max_prob = prob
                best_category = category

        return best_category

    # 文書内のカテゴリー出現率P(C|D)を計算
    def __score(self, word_list, category):

        # カテゴリー出現率P(C)を取得 (アンダーフロー対策で対数をとり、加算)
        score = math.log(self.__prior_prob(category))

        # カテゴリー内の単語出現率を文書内のすべての単語で求める
        for word in word_list:

            # カテゴリー内の単語出現率P(Wn|C)を計算 (アンダーフロー対策で対数をとり、加算)
            score += math.log(self.__word_prob(word, category))

        return score

    # カテゴリー出現率P(C)を計算　
    def __prior_prob(self, category):

        # 学習データの対象カテゴリーの文書数　/ 学習データの文書数合計
        return float(self.category_count[category] / sum(self.category_count.values()))

    # カテゴリー内の単語出現率P(Wn|C)を計算
    def __word_prob(self, word, category):

        # 単語のカテゴリー内出現回数 + 1 / カテゴリー内単語数 + 学習データの全単語数 (加算スムージング)
        prob = (self.__in_category(word, category) + 1.0) / (sum(self.word_count[category].values())
                                                             + len(self.vocabularies) * 1.0)
        return prob

    # 単語のカテゴリー内出現回数を返す
    def __in_category(self, word, category):

        if word in self.word_count[category]:
            # 単語のカテゴリー内出現回数
            return float(self.word_count[category][word])
        return 0.0


if __name__ == '__main__':

    text = 'ラケットスポーツ'

    nb = NaiveBayes()

    # 学習データセット
    nb.train('ボール スポーツ ワールドカップ ボール', 'サッカー')
    nb.train('ボール スポーツ グローブ バット', '野球')
    nb.train('ボール ラケット コート スポーツ', 'テニス')

    # 分類した結果を取得
    res = nb.classifier(text)

    result = {'category': res}

    print(result)
