import openpyxl
import nltk
import re
import config
configs = config.configs
import os
from nltk.parse import stanford
import time

os.environ["STANFORD_PARSER"] = configs['stanfordParserPath']
os.environ["STANFORD_MODELS"] = configs['stanfordParserModelsPath']
parser = stanford.StanfordParser(model_path=u"edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

nltk.data.path.append(configs['nltkDataPath'])

# 记录程序开始执行时间
start = time.time()

readBook = openpyxl.load_workbook(configs['dataSetPath'])
readSheet = readBook.active

writeBook = openpyxl.load_workbook(configs['extractFeaturesPath'])
writeSheet = writeBook.active

def segregateSentence(paraContent):
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_tokenizer.tokenize(paraContent)
    return sentences

def segregateSentenceTag(paraContentTag):
    sentencesTag = re.split(r'(<[A-Z]{2,4}>)', paraContentTag)
    result = []
    for index in range(len(sentencesTag)):
        if sentencesTag[index] == "" or index%2 == 0:
            continue
        else:
            result.append(configs['Tags'][sentencesTag[index]])
    return result

def getWordCount(sentence):
    words = re.split(r"\s+", sentence)
    return  len(words), words[-1][-1]
    #return len(nltk.word_tokenize(sentence))

def getParseTreeDepth(sentence):
    trees = parser.raw_parse(sentence)
    tree = next(trees)
    if tree is None:
        return 0
    return tree.height()

rows = []
i = 2
while i <= 6:
    paraContentTag = readSheet.cell(row = i, column = 13).value
    tags = segregateSentenceTag(paraContentTag)

    paraContent = readSheet.cell(row= i, column = 7).value
    sentences = segregateSentence(paraContent)

    for index in range(len(sentences)):

        rows.append((readSheet.cell(row=i, column=1).value,
                     readSheet.cell(row=i, column=2).value,
                     readSheet.cell(row=i, column=3).value,
                     readSheet.cell(row=i, column=4).value,
                     readSheet.cell(row=i, column=8).value,
                     sentences[index], tags[index],
                     getWordCount(sentences[index])[0],
                     getWordCount(sentences[index])[1],
                     index, getParseTreeDepth(sentences[index])))
    i+=1

for row in rows:
    writeSheet.append(row)

writeBook.save(configs['extractFeaturesPath'])

# 计算程序运行总时间
elapsed = (time.time() - start)
print('Time used : ', elapsed)