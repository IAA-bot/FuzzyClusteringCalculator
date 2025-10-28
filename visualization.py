import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


class ClusterVisualizer:
    """聚类可视化类（改进：避免 lambda 标签被遮挡）"""

    def __init__(self):
        plt.style.use('seaborn-v0_8-darkgrid')

    @staticmethod
    def display_matrix(matrix: np.ndarray, title: str) -> None:
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

    def _place_label(self, ax, x, y, text, color='blue', fontsize=8):
        """在轴上放置不重叠的文本标签：添加偏移、bbox并检测冲突"""
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        x_range = xmax - xmin if xmax != xmin else 1.0
        y_range = abs(ymax - ymin) if ymax != ymin else 1.0

        # 初始向上偏移：占 y_range 的 2%
        y_offset_unit = 0.02 * y_range
        y_pos = y + y_offset_unit

        # 简单碰撞检测：保持类级别短期缓存（存储在 ax 属性，避免全局变量）
        placed = getattr(ax, "_placed_labels", [])
        x_thresh = 0.03 * x_range
        y_thresh = 0.03 * y_range

        # 若与已放置标签接近，则逐次上移直到不冲突
        max_iter = 20
        it = 0
        while any(abs(x - px) < x_thresh and abs(y_pos - py) < y_thresh for px, py in placed) and it < max_iter:
            y_pos += y_offset_unit
            it += 1

        # 绘制文本，使用白色半透明背景，置于上层，不被剪裁
        txt = ax.text(x, y_pos, text,
                      va='bottom', ha='center',
                      fontsize=fontsize, color=color,
                      bbox=dict(facecolor='white', edgecolor='none', alpha=0.75),
                      zorder=10,
                      clip_on=False)
        placed.append((x, y_pos))
        setattr(ax, "_placed_labels", placed)
        return txt

    def plot_dendrogram(self, R: np.ndarray, labels=None, title='Fuzzy Clustering Dendrogram'):
        """绘制动态聚类图，并在每次合并处标注 lambda 值（λ = 1 - distance）"""
        if labels is None:
            labels = [f'Sample{i + 1}' for i in range(len(R))]
        T = self.fuzzy_transitive_closure(R)
        lambdas = np.unique(T)[::-1]
        n = T.shape[0]
        active_clusters = {i: {i} for i in range(n)}
        next_cluster_id = n
        merge_history = []
        for lam in lambdas:
            C = (T >= lam).astype(int)
            n_components, component_labels = connected_components(C, directed=False)
            new_clusters = [set(np.where(component_labels == i)[0]) for i in range(n_components)]
            if len(new_clusters) < len(active_clusters):
                used = set()
                for nc in new_clusters:
                    involved = [cid for cid, members in active_clusters.items() if len(members & nc) > 0]
                    if len(involved) > 1:
                        merge_history.append([involved[0], involved[1], 1 - lam, len(nc)])
                        for cid in involved:
                            used.add(cid)
                        active_clusters[next_cluster_id] = set().union(*[active_clusters[cid] for cid in involved])
                        next_cluster_id += 1
                for cid in used:
                    active_clusters.pop(cid)
        Z = np.array(merge_history)
        if len(Z) != n - 1 or Z.shape[1] != 4:
            print("✗ 相似性过低或聚类合并次数不足，无法绘制完整树状图。")
            self._plot_simple_cluster(R, labels, title='simple cluster')
            return
        try:
            plt.figure(figsize=(10, 6))
            dendro_res = dendrogram(Z, labels=labels)
            ax = plt.gca()
            # 清空任何已记录的标签位置信息
            if hasattr(ax, "_placed_labels"):
                delattr(ax, "_placed_labels")

            try:
                for i, dist in enumerate(Z[:, 2]):
                    # 从 dendrogram 提取合并点坐标
                    icoord = dendro_res.get('icoord', [])[i]
                    dcoord = dendro_res.get('dcoord', [])[i]
                    x = (icoord[1] + icoord[2]) / 2.0
                    y = dcoord[1]
                    lam_val = 1.0 - float(dist)
                    # 使用改进的标签放置函数
                    self._place_label(ax, x, y, f'λ={lam_val:.4f}', color='red', fontsize=8)
            except Exception:
                pass

            plt.title(title, fontsize=16, fontweight='bold')
            plt.xlabel('Samples')
            plt.ylabel('Distance (1 - lambda)')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"绘制树状图时出错: {e}")
            self._plot_simple_cluster(R, labels, title='simple cluster')

    def _plot_simple_cluster(self, similarity_matrix, labels, title):
        """绘制动态聚类图(备用方法)，并标注每次合并的 lambda 值"""
        if labels is None:
            labels = [f'样本{i + 1}' for i in range(len(similarity_matrix))]

        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0)

        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method='average')

        plt.figure(figsize=(12, 8))
        dendro_res = dendrogram(linkage_matrix,
                                labels=labels,
                                orientation='top',
                                distance_sort='descending',
                                show_leaf_counts=True,
                                leaf_rotation=45)
        ax = plt.gca()
        if hasattr(ax, "_placed_labels"):
            delattr(ax, "_placed_labels")

        try:
            for i, dist in enumerate(linkage_matrix[:, 2]):
                icoord = dendro_res.get('icoord', [])[i]
                dcoord = dendro_res.get('dcoord', [])[i]
                x = (icoord[1] + icoord[2]) / 2.0
                y = dcoord[1]
                lam_val = 1.0 - float(dist)
                self._place_label(ax, x, y, f'λ={lam_val:.4f}', color='blue', fontsize=8)
        except Exception:
            pass

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