import nltk
import config

configs = config.configs

nltk.data.path.append('/Users/ming.zhou/NLP/Tools/nltk/nltk_data')

# 分割
sentence = "My father had borrowed some money to one of his friend Wang two years ago, who was going to buy a new room that time."
tokens1 = nltk.grammar.nonterminals(sentence)
# print("len is ", len(tokens1))
print(tokens1)

tokens = nltk.word_tokenize(sentence)
print(tokens)

# 词性标注
tagged = nltk.pos_tag(tokens)
print(tagged)
i_1 = 0
i_2 = 0
# 判断时态
for tag in tagged:
    if tag[1] not in configs['tense']:
        continue
    elif configs['tense'][tag[1]] == 1:
        print('1', tag[0], tag[1])
        i_1 += 1
    elif configs['tense'][tag[1]] == 2:
        print('2', tag[0], tag[1])
        i_2 += 1
print(1) if i_1 >= i_2 else print(2)

# 构建分析树
# entities = nltk.chunk.ne_chunk(tagged)
# print(entities)
# print(entities.height())

#分句
# sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# paragraph = """In most cases, borrowing money from others is an injury to our relationship. However, is it still right under the condition that the man who ask you for money is your friend? This question triggers an interesting debate. As far as I am concerned, borrowing money from freinds will not damage our friendship. My reasons and examples are given below."""
# paragraphTag = """<BG>In most cases, borrowing money from others is an injury to our relationship.</BG><BG> However, is it still right under the condition that the man who ask you for money is your friend?</BG><BG> This question triggers an interesting debate.</BG><PT> As far as I am concerned, borrowing money from freinds will not damage our friendship.</PT><TST>My reasons and examples are given below.</TST>"""
# sentences = sent_tokenizer.tokenize(paragraph)
# import re
# sentencesTag = re.split(r'(<[A-Z]{2,4}>)', paragraphTag)
# result = []
# print(sentencesTag)
# for index in range(len(sentencesTag)):
#     # if s != "":
#     print("index[", index, "]: ", sentencesTag[index])
#     if sentencesTag[index] == "" or index % 2 == 0:
#         continue
#     else:
#         result.append(sentencesTag[index])
#
# print(result)