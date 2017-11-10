import openpyxl
import config
configs = config.configs

# BGResultFile = open('/Users/ming.zhou/NLP/datasets/ngram/modal/BGResult')
# PTResultFile = open('/Users/ming.zhou/NLP/datasets/ngram/modal/PTResult')
# REXPResultFile = open('/Users/ming.zhou/NLP/datasets/ngram/modal/REXPResult')
# EGResultFile = open('/Users/ming.zhou/NLP/datasets/ngram/modal/EGResult')
# EEXPResultFile = open('/Users/ming.zhou/NLP/datasets/ngram/modal/EEXPResult')
# ADMResultFile = open('/Users/ming.zhou/NLP/datasets/ngram/modal/ADMResult')
# RTTResultFile = open('/Users/ming.zhou/NLP/datasets/ngram/modal/RTTResult')
# SRSResultFile = open('/Users/ming.zhou/NLP/datasets/ngram/modal/SRSResult')
RAFMResultFile = open('/Users/ming.zhou/NLP/datasets/ngram/modal/RAFMResult')

writeBook = openpyxl.load_workbook(configs['extractFeaturesPath'])
writeSheet = writeBook.active

import re
i = 2
for line in RAFMResultFile:
    result = re.findall('ppl=.*? ppl', line)
    if result:
        writeSheet['V' + str(i)] = re.split('=', result[0])[1].strip().split(' ')[0]
        i += 1
    # if 'zeroprobs' in line and 'logprob=' in line and 'ppl=' in line and 'ppl1=' in line:
    #     print(line)
writeBook.save(configs['extractFeaturesPath'])

