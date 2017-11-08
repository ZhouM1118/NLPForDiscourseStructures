import openpyxl
import nltk
import re

tagDict = {
    '<BG>' : 1,
    '<PT>' : 2,
    '<TST>' : 3,
    '<RS>' : 4,
    '<REXP>': 5,
    '<EG>': 6,
    '<EEXP>': 7,
    '<GRL>': 8,
    '<ADM>': 9,
    '<RTT>': 10,
    '<SRS>': 11,
    '<RAFM>': 12,
    '<IRL>': 13
}

nltk.data.path.append('/Users/ming.zhou/NLP/Tools/nltk/nltk_data')

readBook = openpyxl.load_workbook('/Users/ming.zhou/NLP/datasets/作文评分Sample_LSM_20171102.xlsx')
readSheet = readBook.active

writeBook = openpyxl.load_workbook('/Users/ming.zhou/NLP/datasets/result.xlsx')
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
            result.append(tagDict[sentencesTag[index]])
    return result

def getWordCount(sentence):
    words = re.split(r"\s+", sentence)
    return  len(words), words[-1][-1]
    #return len(nltk.word_tokenize(sentence))

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
                     index))
    i+=1

for row in rows:
    writeSheet.append(row)

writeBook.save('/Users/ming.zhou/NLP/datasets/result.xlsx')

