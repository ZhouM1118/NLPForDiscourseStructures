import openpyxl
import config
configs = config.configs

writeBook = openpyxl.load_workbook(configs['extractFeaturesPath'])
writeSheet = writeBook.active

# V1.0.0
# row = ['CompID','UserID', 'WriteScore', 'SysCompID', 'ParaID', 'SetenContent', 'SetenTag',
#        'WordCount', 'Punctuation', 'position', 'parseTreeDepth', 'tense']
row = ['ID', 'ParaType', 'Structure', 'SentenceContent', 'SentenceTag',
       'WordCount', 'Punctuation', 'Position', 'ParseTreeDepth', 'Tense', 'NGram']
for indicator in configs['indicators']:
    row.append(indicator)

writeSheet.append(row)
writeBook.save(configs['extractFeaturesPath'])