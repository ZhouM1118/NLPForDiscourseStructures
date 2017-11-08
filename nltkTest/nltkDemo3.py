import os
from nltk.parse import stanford

os.environ["STANFORD_PARSER"] = "/Users/ming.zhou/NLP/Tools/stanford-parser/stanford-parser-full-2017-06-09/stanford-parser.jar"
os.environ["STANFORD_MODELS"] = "/Users/ming.zhou/NLP/Tools/stanford-parser/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar"

parser = stanford.StanfordParser(model_path=u"edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
sentences = parser.raw_parse("At eight o'clock on Thursday morning Arthur didn't feel very good.")
print(sentences)
for s in sentences:
    print(s)
    # s.draw()

sentences2 = parser.raw_parse("she was a student")
print(sentences2)
for s in sentences2:
    print(s)