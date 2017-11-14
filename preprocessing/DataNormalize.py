import math


class DataNormalize:

    @staticmethod
    # 归一化处理，使用sigmoid函数
    # sigmoid函数做归一主要是把两边的一些噪声数据拉回来，不要让噪声数据影响模型效果，
    # 而我们是自己提取的特征，已经经过了预处理，没有很多噪声数据
    # 这就是在这种情况下使用sigmoid函数准确率低的原因
    def normalizeFeatureBySigmoid(featureVector):
        normalizeFeatureVector = []
        for feature in featureVector:
            normalizeFeatureVector.append(1.0 / (1 + math.exp(-float(feature))))
        return normalizeFeatureVector

    @staticmethod
    # 归一化处理，使用(0,1)标准化
    def normalizeFeatureByMaxMin(featureVector):
        normalizeFeatureVector = []
        maxNum = max(featureVector)
        minNum = min(featureVector)
        for feature in featureVector:
            normalizeFeatureVector.append((feature - minNum) / (maxNum - minNum))
        return normalizeFeatureVector