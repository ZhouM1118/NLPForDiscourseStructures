import nltk
from nltk.corpus import treebank
nltk.data.path.append('/Users/ming.zhou/NLP/Tools/nltk/nltk_data')

# 分割
sentence = "At eight o'clock on Thursday morning Arthur didn't feel very good."
tokens = nltk.word_tokenize(sentence)
print(tokens)

# 词性标注
tagged = nltk.pos_tag(tokens)
print(tagged)

entities = nltk.chunk.ne_chunk(tagged)
print(entities)
print(entities.height())
#entities.draw()

t = treebank.parsed_sents('wsj_0001.mrg')[0]
print(t)
print(type(t))
# t.draw()

from nltk.grammar import PCFG, induce_pcfg, toy_pcfg1, toy_pcfg2
from nltk.parse import pchart

grammar = toy_pcfg2
parser = pchart.InsideChartParser(grammar)
tokens = "Jack saw Bob with my cookie".split()
t1 = parser.parse(tokens)
# print(next(t1).height())
# next(t1).draw()
# print(type(t1))
for t in parser.parse(tokens):
    print(t.height())
    print(type(t))
    print(t.draw())