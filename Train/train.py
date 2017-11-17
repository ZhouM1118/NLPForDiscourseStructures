import openpyxl
import config
import math
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import cross_validation
from sklearn import metrics
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
# from sklearn.cross_validation import cross_val_score
from sklearn.metrics import hamming_loss
import pylab as pl
from datetime import datetime

configs = config.configs

# 记录程序开始执行时间
start = datetime.now()

# readBook = openpyxl.load_workbook(configs['condensedFeaturesPath'])
# readSheet = readBook.active

read_book = openpyxl.load_workbook('/Users/ming.zhou/NLP/datasets/eliminateParaTag.xlsx')
read_sheet = read_book.active

# readTestBook = openpyxl.load_workbook(configs['condensedTestFeaturesPath'])
# readTestSheet = readTestBook.active

read_test_book = openpyxl.load_workbook('/Users/ming.zhou/NLP/datasets/eliminateParaTagTest.xlsx')
read_test_sheet = read_test_book.active

read_all_book = openpyxl.load_workbook(configs['allFeaturesPath'])
read_all_sheet = read_all_book.active


def get_feature_vector_and_tag():
    """
    获取训练集的句子特征向量以及句子标签
    :return:
        feature_vector_list：句子特征向量列表
        sentence_tag：句子标签
    """

    feature_vector_list = []
    sentence_tag = []
    for i in range(read_sheet.max_row - 1):

        feature_vector = []
        sentence_tag.append(read_sheet['E' + str(i + 2)].value)
        # 先添加段落特征
        feature_vector.append(float(read_sheet['A' + str(i + 2)].value.strip().split('-')[1]))
        for j in range(read_sheet.max_column - 5):
            if isinstance(read_sheet.cell(row=i + 2, column=j + 6).value, str):
                feature_vector.append(float(read_sheet.cell(row=i + 2, column=j + 6).value))
            else:
                feature_vector.append(read_sheet.cell(row=i + 2, column=j + 6).value)
        feature_vector_list.append(feature_vector)
    return feature_vector_list, sentence_tag


def get_test_feature_vector_and_tag():
    """
    获取测试集的句子特征向量以及句子标签
    :return:
        feature_vector_list：句子特征向量列表
        sentence_tag：句子标签
    """
    feature_vector_list = []
    sentence_tag = []
    for i in range(read_test_sheet.max_row - 1):

        feature_vector = []
        sentence_tag.append(read_test_sheet['E' + str(i + 2)].value)
        # 先添加段落特征
        feature_vector.append(float(read_test_sheet['A' + str(i + 2)].value.strip().split('-')[1]))
        for j in range(read_test_sheet.max_column - 5):
            if isinstance(read_test_sheet.cell(row=i + 2, column=j + 6).value, str):
                feature_vector.append(float(read_test_sheet.cell(row=i + 2, column=j + 6).value))
            else:
                feature_vector.append(read_test_sheet.cell(row=i + 2, column=j + 6).value)
        feature_vector_list.append(feature_vector)
    return feature_vector_list, sentence_tag


def adjust_parameter(x, y, flag='ExtraTrees'):
    """
    为模型调参
    :param x: 样本集的特征向量
    :param y: 样本集的标签列表
    :param flag: 模型指示词
    :return:
    """

    clf = ExtraTreesClassifier(min_samples_split=100, min_samples_leaf=20,
                               max_depth=8, max_features='sqrt', random_state=10)
    if flag == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=10)
    elif flag == 'DecisionTree':
        clf = DecisionTreeClassifier()
    elif flag == 'SVM':
        clf = svm.SVC()
    param_grid = {
        'n_estimators': range(5, 20, 2)
    }
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc', cv=5)
    grid_search.fit(x, y)
    print(grid_search.grid_scores_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)


def do_train(x, y, flag='RandomForest'):
    """
    训练并测试训练集的准确度
    :param x: 样本集的特征向量
    :param y: 样本集的标签列表
    :param flag: 模型指示词
    :return:
    """

    clf = RandomForestClassifier(n_estimators=10)
    if flag == 'ExtraTrees':
        clf = ExtraTreesClassifier(n_estimators=10)
    elif flag == 'DecisionTree':
        clf = DecisionTreeClassifier()
    elif flag == 'SVM':
        clf = svm.SVC()

    clf.fit(x, y)
    # clf = clf.fit(X, Y)

    y_pred = clf.predict(x)
    print(accuracy_score(y, y_pred))
    print(y)
    print(y_pred)
    print('Y len is %s，y_pred len is %s \n' % (len(y), len(y_pred)))
    # scores = cross_val_score(clf, X, Y)
    # print(scores.mean())

    compare_result = {}
    j = 0
    for i in range(len(y)):
        if y[i] == 5:
            j += 1
        y_and_ypred = str(y[i]) + '-' + str(y_pred[i])
        # print(y_and_ypred)
        if y[i] != y_pred[i]:
            if y_and_ypred not in compare_result:
                compare_result[y_and_ypred] = 1
            else:
                compare_result[y_and_ypred] = compare_result[y_and_ypred] + 1

    print(sorted(compare_result.items(), key=lambda d: -d[1]))


def do_train_by_test_set(train_x, train_y, test_x, test_y, flag='RandomForest'):
    """
    测试测试集的准确度
    :param train_x: 样本集中的特征向量列表
    :param train_y: 样本集中的标签列表
    :param test_x: 测试集中的特征向量列表
    :param test_y: 测试集中的标签列表
    :param flag: 模型指示词
    :return:
    """

    clf = RandomForestClassifier(n_estimators=10)
    file_path = '/Users/ming.zhou/NLP/DiscourseStructures/result/' + flag + 'Result20171115.text'

    if flag == 'ExtraTrees':
        clf = ExtraTreesClassifier(n_estimators=10)
    elif flag == 'DecisionTree':
        clf = DecisionTreeClassifier()
    elif flag == 'SVM':
        clf = svm.SVC()

    clf.fit(train_x, train_y)
    y_pred = clf.predict(test_x)

    result = open(file_path, 'a')
    result_content = list()
    result_content.append(str(clf.score(test_x, test_y)) + '\n')
    result_content.append('test_Y:' + '\n')
    result_content.append(str(test_y) + '\n')
    result_content.append('y_pred:' + '\n')
    result_content.append(str(y_pred) + '\n')
    result_content.append('test_Y len is ' + str(len(test_y)) + 'y_pred len is ' + str(len(y_pred)) + '\n')

    print(clf.score(test_x, test_y))
    print(accuracy_score(test_y, y_pred))
    print(test_y)
    print(y_pred)
    print('test_Y len is %s，y_pred len is %s \n' % (len(test_y), len(y_pred)))

    compare_result = {}
    for i in range(len(test_y)):
        y_and_ypred = str(test_y[i]) + '-' + str(y_pred[i])
        # print(y_and_ypred)
        if test_y[i] != y_pred[i]:
            if y_and_ypred not in compare_result:
                compare_result[y_and_ypred] = 1
            else:
                compare_result[y_and_ypred] = compare_result[y_and_ypred] + 1

    sorted_result = sorted(compare_result.items(), key=lambda d: -d[1])

    result_content.append(str(sorted_result) + '\n')
    print(sorted_result)
    result.writelines(result_content)
    result.close()


def do_train_by_cv(x, y, times_num, flag='ExtraTrees'):
    """
    交叉验证模型在训练集上的准确度
    :param x: 样本集中的特征向量列表
    :param y: 样本集中的标签列表
    :param times_num: 模型的n_estimators值
    :param flag: 模型指示词
    :return: 返回模型10次准确度的平均值
    """

    clf = ExtraTreesClassifier(n_estimators=times_num)
    file_path = '/Users/ming.zhou/NLP/DiscourseStructures/result/' + flag + 'CrossValidationResult20171117.text'

    result = open(file_path, 'a')
    result_content = list()
    result_content.append('**********************step=' + str(times_num) + '**********************' + '\n')
    if flag == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=10)
    elif flag == 'DecisionTree':
        clf = DecisionTreeClassifier()
    elif flag == 'SVM':
        clf = svm.SVC()

    # 随机划分训练集与测试集，是交叉验证中常用的函数
    sum_num = 0
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=i)
        clf.fit(x_train, y_train)
        # Y_pred = clf.predict(x_test)
        score = clf.score(x_test, y_test)
        result_content.append('index[' + str(i) + ']' + str(score) + '\n')
        print(score)

        sum_num += score
    result_content.append('平均准确度：' + str(sum_num/10) + '\n')
    result.writelines(result_content)
    result.close()
    print('平均准确度：', str(sum_num/10))
    return sum_num/10


def do_train_by_cv_and_norm(x, y, times_num, flag='ExtraTrees'):
    """
    使用交叉验证以及规范化技术训练模型在训练集上的准确度
    :param x: 样本集中的特征向量列表
    :param y: 样本集中的标签列表
    :param times_num: 模型的n_estimators值
    :param flag: 模型指示词
    :return:
    """
    clf = ExtraTreesClassifier(n_estimators=10)
    file_path = '/Users/ming.zhou/NLP/DiscourseStructures/result/' + flag + 'CrossValidationResult20171117.text'

    result = open(file_path, 'a')
    result_content = list()
    result_content.append('**********************step=' + str(times_num) + '**********************' + '\n')
    if flag == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=10)
    elif flag == 'DecisionTree':
        clf = DecisionTreeClassifier()
    elif flag == 'SVM':
        clf = svm.SVC()

    # 随机划分训练集与测试集，是交叉验证中常用的函数
    sum_num = 0
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=i)
        x_train_norm = preprocessing.normalize(x_train, norm='l2')
        x_test_norm = preprocessing.normalize(x_test, norm='l2')
        clf.fit(x_train_norm, y_train)
        # Y_pred = clf.predict(x_test)
        score = clf.score(x_test_norm, y_test)
        result_content.append('index[' + str(i) + ']' + str(score) + '\n')
        print(score)

        sum_num += score
    result_content.append('平均准确度：' + str(sum_num/10) + '\n')
    result.writelines(result_content)
    result.close()
    print('平均准确度：', str(sum_num/10))


def do_predict(x_train, y_train, x_test, y_test, times_num, flag='ExtraTrees'):
    """
    预测模型在测试集上的准确度
    :param x_train: 样本集中的特征向量列表
    :param y_train: 样本集中的标签列表
    :param x_test: 测试集中的特征向量列表
    :param y_test: 测试集中的标签列表
    :param times_num: 模型的n_estimators值
    :param flag: 模型指示词
    :return: 返回预测的准确度
    """
    clf = ExtraTreesClassifier(n_estimators=times_num)
    file_path = '/Users/ming.zhou/NLP/DiscourseStructures/result/' + flag + 'PredictResult20171117.text'

    if flag == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=10)
    elif flag == 'DecisionTree':
        clf = DecisionTreeClassifier()
    elif flag == 'SVM':
        clf = svm.SVC()

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    score = clf.score(x_test, y_test)

    # print(score)
    # print(Y_test)
    # print(y_pred)
    # print('test_Y len is %s，y_pred len is %s \n' %(len(Y_test), len(y_pred)))

    result = open(file_path, 'a')
    result_content = list()
    result_content.append('**********************step=' + str(times_num) + '**********************' + '\n')
    result_content.append(str(score) + '\n')
    result_content.append('test_Y:' + '\n')
    result_content.append(str(y_test) + '\n')
    result_content.append('y_pred:' + '\n')
    result_content.append(str(y_pred) + '\n')
    result_content.append('test_Y len is ' + str(len(y_test)) + '，and y_pred len is ' + str(len(y_pred)) + '\n')

    for i in range(len(y_test)):
        y_and_ypred = str(y_test[i]) + '-' + str(y_pred[i])
        # print(y_and_ypred)
        if y_test[i] != y_pred[i]:
            if y_and_ypred not in compare:
                compare[y_and_ypred] = 1
            else:
                compare[y_and_ypred] = compare[y_and_ypred] + 1

    # sortedResult = sorted(compare.items(), key=lambda d: -d[1])
    # print(sortedResult)
    # result_content.append(str(sortedResult) + '\n')
    result.writelines(result_content)
    result.close()
    return score


def do_predict_by_cv_and_norm(x_train, y_train, x_test, y_test, times_num, flag='ExtraTrees'):
    """
    使用规范化技术预测模型在测试集上的准确度
    :param x_train: 样本集中的特征向量列表
    :param y_train: 样本集中的标签列表
    :param x_test: 测试集中的特征向量列表
    :param y_test: 测试集中的标签列表
    :param times_num: 模型的n_estimators值
    :param flag: 模型指示词
    :return: 返回预测的准确度
    """
    clf = ExtraTreesClassifier(n_estimators=10)
    file_path = '/Users/ming.zhou/NLP/DiscourseStructures/result/' + flag + 'PredictResult20171117.text'

    if flag == 'RandomForest':
        clf = RandomForestClassifier(n_estimators=10)
    elif flag == 'DecisionTree':
        clf = DecisionTreeClassifier()
    elif flag == 'SVM':
        clf = svm.SVC()

    normalizer = preprocessing.Normalizer().fit(x_train)
    x_train_norm = normalizer.transform(x_train)
    x_test_norm = normalizer.transform(x_test)
    clf.fit(x_train_norm, y_train)
    y_pred = clf.predict(x_test_norm)
    score = clf.score(x_test_norm, y_test)

    print(str(score) + '\n')
    print(y_test + '\n')
    print(y_pred + '\n')
    print('test_Y len is %s，y_pred len is %s \n' % (len(y_test), len(y_pred)))

    result = open(file_path, 'a')
    result_content = list()
    result_content.append('**********************step=' + str(times_num) + '**********************' + '\n')
    result_content.append(str(score) + '\n')
    result_content.append('test_Y:' + '\n')
    result_content.append(str(y_test) + '\n')
    result_content.append('y_pred:' + '\n')
    result_content.append(str(y_pred) + '\n')
    result_content.append('test_Y len is ' + str(len(y_test)) + '，and y_pred len is ' + str(len(y_pred)) + '\n')

    for i in range(len(y_test)):
        y_and_ypred = str(y_test[i]) + '-' + str(y_pred[i])

        if y_test[i] != y_pred[i]:
            if y_and_ypred not in compare:
                compare[y_and_ypred] = 1
            else:
                compare[y_and_ypred] = compare[y_and_ypred] + 1

    # sortedResult = sorted(compare.items(), key=lambda d: -d[1])
    # print(sortedResult)
    # result_content.append(str(sortedResult) + '\n')
    result.writelines(result_content)
    result.close()
    return score


params = get_feature_vector_and_tag()
# testParams = getTestFeatureVectorAndTag()
# doTrain(params[0], params[1], 'ExtraTrees')
# doTrainByTestSet(params[0], params[1], testParams[0], testParams[1], 'ExtraTrees')
# doTrainByCrossValidation(params[0], params[1], 10, 'ExtraTrees')
# adjustParameter(params[0], params[1])
# doTrainByCVAndNorm(params[0], params[1], 2, 'ExtraTrees')
# print('*********************begin test*********************')
compare = {}

xs = []
ys = []
for index in range(10, 101, 5):
    xs.append(index)
    scores = 0
    # for j in range(3):
    #     score = doPredict(params[0], params[1], testParams[0], testParams[1], i, 'ExtraTrees')
    #     scores += score
    s = do_train_by_cv(params[0], params[1], index, 'ExtraTrees')
    ys.append(s)
    print('index[' + str(index) + ']:' + str(s))

# sortedResult = sorted(compare.items(), key=lambda d: -d[1])
# print(sortedResult)
# print(scores / 30)

pl.plot(xs, ys)
pl.title('ExtraTrees adjust n_estimators')
pl.xlabel('n_estimators')
pl.ylabel('score')
pl.show()

# 计算程序运行总时间(秒)
elapsed = (datetime.now() - start).seconds
print('Time used : ', elapsed)
