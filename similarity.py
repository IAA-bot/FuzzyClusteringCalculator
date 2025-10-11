import numpy as np
from scipy.spatial.distance import pdist, squareform


class SimilarityMatrixBuilder:
    """模糊相似矩阵构建类"""

    def __init__(self):
        self.methods = {
            '1': {'name': '夹角余弦法', 'function': self.cosine_similarity},
            '2': {'name': '相关系数法', 'function': self.correlation_similarity},
            '3': {'name': '距离法', 'function': self.distance_similarity},
            '4': {'name': '最大最小法', 'function': self.max_min_similarity},
            '5': {'name': '算术平均最小法', 'function': self.arithmetic_mean_min_similarity},
            '6': {'name': '几何平均最小法', 'function': self.geometric_mean_min_similarity}
        }

    @staticmethod
    def cosine_similarity(data):
        """夹角余弦法"""
        EPSILON = 1e-8
        norm = np.linalg.norm(data, axis=1)  # 计算每行的范数
        norm = np.where(norm == 0, EPSILON, norm)  # 防止除零
        R = data @ data.T / np.outer(norm, norm)
        np.fill_diagonal(R, 1.0)
        return np.clip(R, 0, 1)  # 确保值在[0,1]范围内

    @staticmethod
    def correlation_similarity(data):
        """相关系数法"""
        corr_matrix = np.corrcoef(data)
        R = (corr_matrix + 1) / 2  # 映射到[0,1]区间
        R = np.nan_to_num(R, nan=0.0)  # 替换nan为0
        np.fill_diagonal(R, 1.0)  # 对角线设为1
        return R

    @staticmethod
    def distance_similarity(data, metric='euclidean'):
        """距离法"""
        distances = pdist(data, metric=metric)
        distance_matrix = squareform(distances)
        max_distance = np.max(distance_matrix)
        if max_distance == 0:
            return np.ones_like(distance_matrix)
        R = 1 - distance_matrix / max_distance  # 使用最大值标准化（未选用1-c(distance)^{\alpha}的方式，如有需要可后续实现）
        return R

    @staticmethod
    def max_min_similarity(data):
        """最大最小法"""
        data1 = data[:, None, :]  # (n,1,m)
        data2 = data[None, :, :]  # (1,n,m)
        min_sum = np.sum(np.minimum(data1, data2), axis=2)  # (n,n)
        max_sum = np.sum(np.maximum(data1, data2), axis=2)  # (n,n)
        R = np.divide(min_sum, max_sum, out=np.ones_like(min_sum), where=max_sum != 0)
        np.fill_diagonal(R, 1.0)
        return R

    @staticmethod
    def arithmetic_mean_min_similarity(data):
        """算术平均最小法"""
        data1 = data[:, None, :]
        data2 = data[None, :, :]
        min_sum = np.sum(np.minimum(data1, data2), axis=2)
        mean_sum = np.sum((data1 + data2) / 2, axis=2)
        R = np.divide(min_sum, mean_sum, out=np.ones_like(min_sum), where=mean_sum != 0)
        np.fill_diagonal(R, 1.0)
        return R

    @staticmethod
    def geometric_mean_min_similarity(data):
        """几何平均最小法"""
        data_non_neg = np.maximum(data, 0)
        data1 = data_non_neg[:, None, :]
        data2 = data_non_neg[None, :, :]
        min_sum = np.sum(np.minimum(data1, data2), axis=2)
        geo_mean_sum = np.sum(np.sqrt(data1 * data2), axis=2)
        R = np.divide(min_sum, geo_mean_sum, out=np.ones_like(min_sum), where=geo_mean_sum != 0)
        np.fill_diagonal(R, 1.0)
        return R

    def build_similarity_matrix(self, data, method_key):
        """构建相似矩阵"""
        if method_key not in self.methods:
            raise ValueError(f"未知的相似矩阵建立方法: {method_key}")

        method_info = self.methods[method_key]
        similarity_matrix = method_info['function'](data)
        return similarity_matrix, method_info['name']

    def get_available_methods(self):
        """获取可用的相似矩阵建立方法"""
        return self.methods