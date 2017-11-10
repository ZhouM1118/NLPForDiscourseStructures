import openpyxl
import config
configs = config.configs

readBook = openpyxl.load_workbook(configs['extractFeaturesPath'])
# 获取当前正在显示的sheet
readSheet = readBook.active

testFile = open('/Users/ming.zhou/NLP/datasets/ngram/test/testFile.txt', 'a')
testContent = []

for i in range(readSheet.max_row - 1):
    sentence = readSheet.cell(row = i + 2, column = 4).value
    testContent.append(sentence  + '\n')

testFile.writelines(testContent)
testFile.close()
