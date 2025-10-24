import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


class ClusterVisualizer:
    """聚类可视化类"""

    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')

    @staticmethod
    def display_matrix(matrix: np.ndarray, title : str) -> None:
        """显示矩阵"""
        print(f"\n{title}:")
        print("=" * 60)
        np.set_printoptions(precision=4, suppress=True)
        print(matrix)
        print("=" * 60)

    @staticmethod
    def fuzzy_transitive_closure(R: np.ndarray, tol=1e-6) -> np.ndarray:
        """模糊传递闭包算法"""
        T = R.copy()
        while True:
            T_new = np.maximum(T, np.minimum(T[:, None, :], T[None, :, :]).min(axis=2))
            if np.allclose(T, T_new, atol=tol):
                break
            T = T_new
        return T

    def plot_dendrogram(self, R: np.ndarray, labels=None, title='Fuzzy Clustering Dendrogram'):
        """绘制动态聚类图"""
        if labels is None:
            labels = [f'Sample{i + 1}' for i in range(len(R))]
        T = self.fuzzy_transitive_closure(R)
        # λ取闭包矩阵的唯一值（降序）
        lambdas = np.unique(T)[::-1]
        n = T.shape[0]
        # 用字典跟踪当前有效簇
        active_clusters = {i: {i} for i in range(n)}
        next_cluster_id = n
        merge_history = []
        for lam in lambdas:
            C = (T >= lam).astype(int)
            n_components, component_labels = connected_components(C, directed=False)
            new_clusters = [set(np.where(component_labels == i)[0]) for i in range(n_components)]
            # 检查是否有合并
            if len(new_clusters) < len(active_clusters):
                # 找到需要合并的簇
                used = set()
                for nc in new_clusters:
                    involved = [cid for cid, members in active_clusters.items() if len(members & nc) > 0]
                    if len(involved) > 1:
                        merge_history.append([involved[0], involved[1], 1 - lam, len(nc)])
                        # 合并后移除旧簇ID，添加新簇ID
                        for cid in involved:
                            used.add(cid)
                        active_clusters[next_cluster_id] = set().union(*[active_clusters[cid] for cid in involved])
                        next_cluster_id += 1
                # 移除已合并的簇ID
                for cid in used:
                    active_clusters.pop(cid)
        # 转为linkage矩阵
        Z = np.array(merge_history)
        if len(Z) != n - 1 or Z.shape[1] != 4:
            print("✗ 相似性过低或聚类合并次数不足，无法绘制完整树状图。")
            self._plot_simple_cluster(R, labels, title='simple cluster')
            return
        try:
            plt.figure(figsize=(10, 6))
            dendrogram(Z, labels=labels)
            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Samples')
            plt.ylabel('Distance (1 - lambda)')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"绘制树状图时出错: {e}")
            self._plot_simple_cluster(R, labels, title='simple cluster')

    def _plot_simple_cluster(self, similarity_matrix, labels, title):
        """绘制动态聚类图(备用方法)"""
        if labels is None:
            labels = [f'样本{i + 1}' for i in range(len(similarity_matrix))]

        # 将相似度矩阵转换为距离矩阵
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)

        # 使用层次聚类
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='average')

        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix,
                labels=labels,
                orientation='top',
                distance_sort='descending',
                show_leaf_counts=True,
                leaf_rotation=45)

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Sample', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.tight_layout()
        plt.show()

    # def plot_comparison(self, original_data, normalized_data, similarity_matrix):
    #     """绘制数据对比图"""
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    #
    #     # 原始数据热图
    #     im1 = axes[0].imshow(original_data, cmap='viridis', aspect='auto')
    #     axes[0].set_title('原始数据', fontsize=14)
    #     axes[0].set_xlabel('特征')
    #     axes[0].set_ylabel('样本')
    #     plt.colorbar(im1, ax=axes[0])
    #
    #     # 规格化数据热图
    #     im2 = axes[1].imshow(normalized_data, cmap='viridis', aspect='auto')
    #     axes[1].set_title('规格化数据', fontsize=14)
    #     axes[1].set_xlabel('特征')
    #     axes[1].set_ylabel('样本')
    #     plt.colorbar(im2, ax=axes[1])
    #
    #     # 相似矩阵热图
    #     im3 = axes[2].imshow(similarity_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
    #     axes[2].set_title('相似矩阵', fontsize=14)
    #     axes[2].set_xlabel('样本')
    #     axes[2].set_ylabel('样本')
    #     plt.colorbar(im3, ax=axes[2])
    #
    #     plt.tight_layout()
    #     plt.show()