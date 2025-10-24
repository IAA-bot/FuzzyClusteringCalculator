import numpy as np


class DataNormalizer:
    """数据规格化处理类"""

    def __init__(self):
        self.methods = {
            '1': {'name': '标准差标准化', 'function': self.std_normalization},
            '2': {'name': '极差正规化', 'function': self.range_normalization},
            '3': {'name': '极差标准化', 'function': self.range_standardization},
            '4': {'name': '最大值规格化', 'function': self.max_normalization}
        }

    @staticmethod
    def std_normalization(data: np.ndarray) -> np.ndarray:
        """标准差标准化"""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0, ddof=1)
        std[std == 0] = 1  # 避免除零
        return (data - mean) / std

    @staticmethod
    def range_normalization(data: np.ndarray) -> np.ndarray:
        """极差正规化"""
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1  # 避免除零
        return (data - min_vals) / ranges

    @staticmethod
    def range_standardization(data: np.ndarray) -> np.ndarray:
        """极差标准化"""
        mean = np.mean(data, axis=0)
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1  # 避免除零
        return (data - mean) / ranges

    @staticmethod
    def max_normalization(data: np.ndarray) -> np.ndarray:
        """最大值规格化"""
        max_vals = np.max(data, axis=0)
        max_vals[max_vals == 0] = 1  # 避免除零
        return data / max_vals

    def normalize(self, data: np.ndarray, method_key: str) -> tuple[np.ndarray, str]:
        """执行规格化"""
        if method_key not in self.methods:
            raise ValueError(f"未知的规格化方法: {method_key}")

        method_info = self.methods[method_key]
        normalized_data = method_info['function'](data)
        return normalized_data, method_info['name']

    def get_available_methods(self):
        """获取可用的规格化方法"""
        return self.methods