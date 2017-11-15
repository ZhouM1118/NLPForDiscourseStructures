import openpyxl
import nltk
import re
import config
import os
from nltk.parse import stanford
from datetime import datetime
# from nltk.tokenize import sent_tokenize

configs = config.configs
os.environ["STANFORD_PARSER"] = configs['stanfordParserPath']
os.environ["STANFORD_MODELS"] = configs['stanfordParserModelsPath']
nltk.data.path.append(configs['nltkDataPath'])

dataSetPath = configs['dataSetPath']
featuresPath = configs['featuresPath']
condensedFeaturesPath = configs['condensedFeaturesPath']
testFeaturesPath = configs['testFeaturesPath']
condensedTestFeaturesPath = configs['condensedTestFeaturesPath']
allFeaturesPath = configs['allFeaturesPath']
parser = stanford.StanfordParser(model_path=u"edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

def time(func):
    def wrapper(*args):
        # 记录程序开始执行时间
        start = datetime.now()
        result = func(*args)
        elapsed = (datetime.now() - start).seconds
        print('"', func.__name__, '"function :', 'Time used : ', elapsed, 's')
        return result
    return wrapper


class DataProcessing:
    """数据预处理类，提供所有的预处理操作"""

    # 分割不带标签的句子 注：无法识别标点符号后不带空格的句子，故放弃这种方式
    @staticmethod
    def segregateSentence(paraContent):
        sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_tokenizer.tokenize(paraContent)
        return sentences

    # 分割带标签的段落
    @staticmethod
    def segregateSentenceByTag(paraContentTag):
        sentencesTag = re.split(r'(<[A-Z]{2,4}>)', paraContentTag)
        tags = []
        sentences = []

        for index in range(len(sentencesTag)):
            if sentencesTag[index] == "":
                continue
            elif index % 2 == 0:
                sentence = re.split(r'(</[A-Z]{2,4}>)', sentencesTag[index])
                sentences.append(sentence[0].strip())
                continue
            else:
                tags.append(configs['Tags'][sentencesTag[index]])

        return sentences, tags

    # 获取一个句子中的单词个数以及标点符号
    @staticmethod
    def getWordCountAndPunctuation(sentence):
        words = re.split(r"\s+", sentence)
        punctuation = words[-1][-1]
        tag = 0
        if punctuation in configs['punctuation']:
            tag = configs['punctuation'][punctuation]
        return len(words), tag
        # return len(nltk.word_tokenize(sentence))

    # 获取一个句子的分析树的深度
    @staticmethod
    def getParseTreeDepth(sentence):
        trees = parser.raw_parse(sentence)
        tree = next(trees)
        if tree is None:
            return 0
        print(tree)
        return tree.height()

    # 获取句子时态特征
    @staticmethod
    def getSentenceTenseAndNNPFlag(sentence):
        # 分词
        tokens = nltk.word_tokenize(sentence)
        # 词性标注
        tags = nltk.pos_tag(tokens)
        NNPFlag = 0
        i_1 = 0
        i_2 = 0
        for tag in tags:
            if tag[1] == 'NNP':
                NNPFlag = 1
            if tag[1] not in configs['tense']:
                continue
            elif configs['tense'][tag[1]] == 1:
                i_1 += 1
            elif configs['tense'][tag[1]] == 2:
                i_2 += 1
        tenseFlag = 1 if i_1 >= i_2 else 2
        return tenseFlag, NNPFlag

    # 提取训练集中的篇章结构特征并持久化
    @staticmethod
    @time
    def extractTrainStructuralFeature():

        readBook = openpyxl.load_workbook(dataSetPath)
        readSheet = readBook.active
        writeBook = openpyxl.load_workbook(featuresPath)
        writeSheet = writeBook.active

        rows = []

        for i in range(readSheet.max_row - 51):
            paraContentTag = readSheet.cell(row=i + 2, column=8).value
            sentencesAndTag = DataProcessing.segregateSentenceByTag(paraContentTag)

            sentences = sentencesAndTag[0]
            tags = sentencesAndTag[1]

            for index in range(len(sentences)):
                print((i + 2), sentences[index])
                wordCountAndPunctuation = DataProcessing.getWordCountAndPunctuation(sentences[index])
                tenseAndNNPFlag = DataProcessing.getSentenceTenseAndNNPFlag(sentences[index])

                row = [readSheet['A' + str(i + 2)].value,  # ID
                       readSheet['B' + str(i + 2)].value,  # ParaType
                       readSheet['E' + str(i + 2)].value,  # Structure
                       sentences[index],  # SentenceContent
                       tags[index],  # SentenceTag
                       configs['paraType'][readSheet['B' + str(i + 2)].value],  # ParaTag
                       0, 0,  # BeforeSentenceTag,AfterSentenceTag
                       wordCountAndPunctuation[0],  # WordCount
                       wordCountAndPunctuation[1],  # Punctuation
                       index,  # position
                       DataProcessing.getParseTreeDepth(sentences[index]),  # parseTreeDepth
                       tenseAndNNPFlag[0],# tense
                       tenseAndNNPFlag[1], # NNPFlag
                       0, 0, 0, 0, 0, 0, 0, 0, 0
                       ]

                for indicator in configs['indicators']:
                    row.append(1) if indicator in sentences[index] \
                                     or indicator.capitalize() in sentences[index] else row.append(0)

                rows.append(row)

        for row in rows:
            writeSheet.append(row)

        writeBook.save(featuresPath)

    # 提取测试集中的篇章结构特征并持久化
    @staticmethod
    @time
    def extractTestStructuralFeature():

        readBook = openpyxl.load_workbook(dataSetPath)
        readSheet = readBook.active
        writeTestBook = openpyxl.load_workbook(testFeaturesPath)
        writeTestSheet = writeTestBook.active

        rows = []

        for i in range(51):
            k = i + readSheet.max_row - 50
            paraContentTag = readSheet['H' + str(k)].value
            sentencesAndTag = DataProcessing.segregateSentenceByTag(paraContentTag.strip())

            sentences = sentencesAndTag[0]
            tags = sentencesAndTag[1]

            for index in range(len(sentences)):
                print(str(k), sentences[index])
                wordCountAndPunctuation = DataProcessing.getWordCountAndPunctuation(sentences[index])
                tenseAndNNPFlag = DataProcessing.getSentenceTenseAndNNPFlag(sentences[index])

                row = [readSheet['A' + str(k)].value,  # ID
                       readSheet['B' + str(k)].value,  # ParaType
                       readSheet['E' + str(k)].value,  # Structure
                       sentences[index],  # SentenceContent
                       tags[index],  # SentenceTag
                       configs['paraType'][readSheet['B' + str(k)].value],  # ParaTag
                       0, 0,  # BeforeSentenceTag,AfterSentenceTag
                       wordCountAndPunctuation[0],  # WordCount
                       wordCountAndPunctuation[1],  # Punctuation
                       index,  # position
                       DataProcessing.getParseTreeDepth(sentences[index]),  # parseTreeDepth
                       tenseAndNNPFlag[0],  # tense
                       tenseAndNNPFlag[1],  # NNPFlag
                       0, 0, 0, 0, 0, 0, 0, 0, 0
                       ]

                for indicator in configs['indicators']:
                    row.append(1) if indicator in sentences[index] \
                                     or indicator.capitalize() in sentences[index] else row.append(0)

                rows.append(row)

        for row in rows:
            writeTestSheet.append(row)

        writeTestBook.save(testFeaturesPath)

    @staticmethod
    # 从featuresPath中分别提取不同标签的句子到指定路径下的对应文件中
    def extractTagSentenceFile():
        readBook = openpyxl.load_workbook(allFeaturesPath)
        readSheet = readBook.active

        BGFile = open('/Users/ming.zhou/NLP/datasets/ngram/allBGFile.txt', 'a')
        PTFile = open('/Users/ming.zhou/NLP/datasets/ngram/allPTFile.txt', 'a')
        TSTFile = open('/Users/ming.zhou/NLP/datasets/ngram/allTSTFile.txt', 'a')
        RSFile = open('/Users/ming.zhou/NLP/datasets/ngram/allRSFile.txt', 'a')
        REXPFile = open('/Users/ming.zhou/NLP/datasets/ngram/allREXPFile.txt', 'a')
        EGFile = open('/Users/ming.zhou/NLP/datasets/ngram/allEGFile.txt', 'a')
        EEXPFile = open('/Users/ming.zhou/NLP/datasets/ngram/allEEXPFile.txt', 'a')
        GRLFile = open('/Users/ming.zhou/NLP/datasets/ngram/allGRLFile.txt', 'a')
        ADMFile = open('/Users/ming.zhou/NLP/datasets/ngram/allADMFile.txt', 'a')
        RTTFile = open('/Users/ming.zhou/NLP/datasets/ngram/allRTTFile.txt', 'a')
        SRSFile = open('/Users/ming.zhou/NLP/datasets/ngram/allSRSFile.txt', 'a')
        RAFMFile = open('/Users/ming.zhou/NLP/datasets/ngram/allRAFMFile.txt', 'a')
        IRLFile = open('/Users/ming.zhou/NLP/datasets/ngram/allIRLFile.txt', 'a')

        BGContent = []
        PTContent = []
        TSTContent = []
        RSContent = []
        REXPContent = []
        EGContent = []
        EEXPContent = []
        GRLContent = []
        ADMContent = []
        RTTContent = []
        SRSContent = []
        RAFMContent = []
        IRLContent = []

        putSentenceByTag = {
            1: lambda x: BGContent.append(x),
            2: lambda x: PTContent.append(x),
            3: lambda x: TSTContent.append(x),
            4: lambda x: RSContent.append(x),
            5: lambda x: REXPContent.append(x),
            6: lambda x: EGContent.append(x),
            7: lambda x: EEXPContent.append(x),
            8: lambda x: GRLContent.append(x),
            9: lambda x: ADMContent.append(x),
            10: lambda x: RTTContent.append(x),
            11: lambda x: SRSContent.append(x),
            12: lambda x: RAFMContent.append(x),
            13: lambda x: IRLContent.append(x)
        }

        for i in range(readSheet.max_row - 1):
            sentenceTag = readSheet['E' + str(i + 2)].value
            sentence = readSheet['D' + str(i + 2)].value
            putSentenceByTag[sentenceTag](sentence + '\n')

        BGFile.writelines(BGContent)
        BGFile.close()
        PTFile.writelines(PTContent)
        PTFile.close()
        TSTFile.writelines(TSTContent)
        TSTFile.close()
        RSFile.writelines(RSContent)
        RSFile.close()
        REXPFile.writelines(REXPContent)
        REXPFile.close()
        EGFile.writelines(EGContent)
        EGFile.close()
        EEXPFile.writelines(EEXPContent)
        EEXPFile.close()
        GRLFile.writelines(GRLContent)
        GRLFile.close()
        ADMFile.writelines(ADMContent)
        ADMFile.close()
        RTTFile.writelines(RTTContent)
        RTTFile.close()
        SRSFile.writelines(SRSContent)
        SRSFile.close()
        RAFMFile.writelines(RAFMContent)
        RAFMFile.close()
        IRLFile.writelines(IRLContent)
        IRLFile.close()

    @staticmethod
    # 从指定路径（fromFilePath）下读取文件内容并将其中所有的句子提取到指定路径下（toFilePath）
    def extractAllSentenceFile(fromFilePath, toFilePath):
        readBook = openpyxl.load_workbook(fromFilePath)
        readSheet = readBook.active

        allSentenceFile = open(toFilePath, 'a')
        sentences = []

        for i in range(readSheet.max_row - 1):
            sentence = readSheet['D' + str(i + 2)].value
            sentences.append(sentence + '\n')

        allSentenceFile.writelines(sentences)
        allSentenceFile.close()

    @staticmethod
    # 设置特征文件中句子的上一句和下一句的句子标签特征以及段落标签特征
    def integrateSentenceTagContextAndParaTagFeature():

        writeBook = openpyxl.load_workbook(condensedTestFeaturesPath)
        writeSheet = writeBook.active

        nowEssayId = 0
        beforeEssayId = 0
        for i in range(writeSheet.max_row - 1):
            # 设置特征文件中句子的段落标签特征
            paraType = writeSheet['B' + str(i + 2)].value
            writeSheet['F' + str(i + 2)] = configs['paraType'][paraType]

            nowEssayId = writeSheet['A' + str(i + 2)].value.strip().split('-')[0]
            isNewEssay = nowEssayId != beforeEssayId

            if isNewEssay:
                writeSheet['G' + str(i + 2)] = 0  # 设置当前句的BeforeSentenceTag为0
                if i > 0:
                    writeSheet['H' + str(i + 1)] = 0  # 设置上一句的AfterSentenceTag为0
                writeSheet['H' + str(i + 2)] = writeSheet['E' + str(i + 3)].value * 10  # 设置当前句的AfterSentenceTag
            else:
                writeSheet['G' + str(i + 2)] = writeSheet['E' + str(i + 1)].value * 10  # 设置当前句的BeforeSentenceTag
                if (i + 2) == writeSheet.max_row:
                    writeSheet['H' + str(i + 2)] = 0
                else:
                    writeSheet['H' + str(i + 2)] = writeSheet['E' + str(i + 3)].value * 10  # 设置当前句的AfterSentenceTag

            beforeEssayId = writeSheet['A' + str(i + 2)].value.strip().split('-')[0]
        writeBook.save(condensedTestFeaturesPath)

    @staticmethod
    # 设置特征文件中句子的NGram特征
    def integrateNGramFeature(ngramResultFilePath, featureFileColumnIndex):
        writeTestBook = openpyxl.load_workbook(allFeaturesPath)
        writeTestSheet = writeTestBook.active

        resultFile = open(ngramResultFilePath)

        i = 2
        for line in resultFile:
            result = re.findall('ppl=.*? ppl', line)
            if result:
                writeTestSheet[featureFileColumnIndex + str(i)] = re.split('=', result[0])[1].strip().split(' ')[0]
                i += 1

        writeTestBook.save(allFeaturesPath)

    @staticmethod
    def addNewIndicators(filePath, indicators, indicatorIndex):
        writeBook = openpyxl.load_workbook(filePath)
        writeSheet = writeBook.active

        for index in range(writeSheet.max_row - 1):
            sentence = writeSheet['D' + str(index + 2)].value
            row = []
            for i in range(len(indicators)):
                if indicators[i] in sentence or indicators[i].capitalize() in sentence:
                    row.append(1)
                else:
                    row.append(0)

            for k in range(len(indicatorIndex)):
                writeSheet[indicatorIndex[k] + str(index + 2)] = row[k]

        writeBook.save(filePath)

indicators = ['above', 'conclusion', 'agree', 'admittedly']
indicatorIndex = ['CF', 'CG', 'CH', 'CI']
DataProcessing.addNewIndicators(condensedTestFeaturesPath, indicators, indicatorIndex)