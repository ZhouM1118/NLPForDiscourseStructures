import os
from nltk.parse import stanford
import time

start = time.clock()
startt = time.time()
os.environ["STANFORD_PARSER"] = "/Users/ming.zhou/NLP/Tools/stanford-parser/stanford-parser-full-2017-06-09/stanford-parser.jar"
os.environ["STANFORD_MODELS"] = "/Users/ming.zhou/NLP/Tools/stanford-parser/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar"

parser = stanford.StanfordParser(model_path=u"edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
sentences = parser.raw_parse("That is because we are good friends and greatly know about each other, which means there is no need for us to worry about the situation that we can not get our money back.")
print(sentences)
tree = next(sentences)
print(type(tree))
if tree is None:
    print(0)
print(tree.height())
# tree.draw()
elapsed = (time.clock() - start)
elapsedt = (time.time() - startt)
print("Time clock used : ", elapsed)
print("Time time used : ", elapsedt)
# print(next(sentences).height())
# for s in sentences:
#     print(s)
#     print(s.height())
#     s.draw()

# sentences2 = parser.raw_parse("she was a student")
# print(sentences2)
# for s in sentences2:
#     print(s)