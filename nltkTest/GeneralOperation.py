s = 'so that'
ses = 'In most cases, borrowing money so that from others is an injury to our relationship.'

print(1) if 'so that' in ses else print(2)

import openpyxl
import config
configs = config.configs

writeBook = openpyxl.load_workbook(configs['extractFeaturesPath'])
writeSheet = writeBook.active

row = ['CompID','UserID', 'WriteScore', 'SysCompID', 'ParaID', 'SetenContent', 'SetenTag',
       'WordCount', 'Punctuation', 'position', 'parseTreeDepth', 'tense']
for indicator in configs['indicators']:
    row.append(indicator)

writeSheet.append(row)
writeBook.save(configs['extractFeaturesPath'])