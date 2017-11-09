import openpyxl
import config
configs = config.configs

readBook = openpyxl.load_workbook(configs['extractFeaturesPath'])
# 获取当前正在显示的sheet
readSheet = readBook.active

BGFile = open('/Users/ming.zhou/NLP/datasets/BGFile.txt', 'a')

BGContent = []
i = 2
print(readSheet.max_row)
while i <= readSheet.max_row-1:
    setenTag = readSheet.cell(row=i, column=7).value
    if setenTag == 1:
        BGContent.append(readSheet.cell(row=i, column=6).value)
        BGContent.append('\n')
    i += 1
BGFile.writelines(BGContent)
BGFile.close()