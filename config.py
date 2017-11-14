configs = {
    'featuresPath' : '/Users/ming.zhou/NLP/datasets/Features.xlsx',
    'testFeaturesPath' : '/Users/ming.zhou/NLP/datasets/TestFeatures.xlsx',
    'dataSetPath' : '/Users/ming.zhou/NLP/datasets/AW_MLParaData.xlsx',
    'Tags' : {
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
    },
    'stanfordParserPath' : '/Users/ming.zhou/NLP/Tools/stanford-parser/stanford-parser-full-2017-06-09/stanford-parser.jar',
    'stanfordParserModelsPath' : '/Users/ming.zhou/NLP/Tools/stanford-parser/stanford-parser-full-2017-06-09/stanford-parser-3.8.0-models.jar',
    'nltkDataPath' : '/Users/ming.zhou/NLP/Tools/nltk/nltk_data',
    'tense' : {
        'VB' : 1,
        'VBZ' : 1,
        'VBP' : 1,
        'VBG' : 1,
        'VBN' : 2,
        'VBD' : 2
    },
    'indicators' : ('accordingly', 'additionally', 'after', 'afterward', 'also', 'alternatively',
                    'alternative', 'although', 'as a result', 'as an alternative', 'as if', 'as long as',
                    'as soon as', 'as though', 'as well', 'because', 'before', 'before and after',
                    'besides', 'by comparison',  'but', 'by contrast', 'by then', 'consequently',
                    'conversely', 'earlier', 'either', 'except', 'finally', 'for example', 'for instance',
                    'further', 'furthermore','hence', 'however', 'if', 'if and when', 'in addition',
                    'in contrast', 'in fact', 'in other words',  'in particular', 'in short', 'in sum',
                    'in the end', 'in turn', 'indeed', 'insofar as', 'instead', 'later', 'lest',
                    'likewise', 'meantime', 'meanwhile', 'moreover', 'much as', 'neither',
                    'nevertheless', 'now that', 'next', 'nonetheless', 'nor', 'on the contrary', 'on the one hand',
                    'on the other hand', 'once', 'otherwise', 'overall', 'plus', 'previously', 'rather', 'regardless',
                    'separately', 'similarly', 'simultaneously', 'since', 'so that', 'specifically',
                    'still', 'then', 'thereafter', 'thereby', 'therefore', 'though', 'thus',
                    'till', 'ultimately', 'unless', 'until', 'when', 'when and if', 'whereas', 'while',
                    'yet'),
    'punctuation' : {
        '.' : 1,
        '?' : 2,
        '!' : 3
    },
    'paraType' : {
        'Introduction' : 50,
        'Reason' : 100,
        'Concession' : 150,
        'Conclusion' : 200
    }
}