import math


class DataNormalize(object):

    @staticmethod
    def normalize_feature_by_sigmoid(feature_vector):
        """
        归一化处理，使用sigmoid函数
        sigmoid函数做归一主要是把两边的一些噪声数据拉回来，不要让噪声数据影响模型效果，
        而我们是自己提取的特征，已经经过了预处理，没有很多噪声数据
        这就是在这种情况下使用sigmoid函数准确率低的原因
        :param feature_vector: 特征向量
        :return: 归一化后的特征向量
        """
        normalize_feature_vector = []
        for feature in feature_vector:
            normalize_feature_vector.append(1.0 / (1 + math.exp(-float(feature))))
        return normalize_feature_vector

    @staticmethod
    def normalize_feature_by_maxmin(feature_vector):
        """
        归一化处理，使用(0,1)标准化
        :param feature_vector: 特征向量
        :return: 归一化后的特征向量
        """
        normalize_feature_vector = []
        max_num = max(feature_vector)
        min_num = min(feature_vector)
        for feature in feature_vector:
            normalize_feature_vector.append((feature - min_num) / (max_num - min_num))
        return normalize_feature_vector
