from normalization import DataNormalizer
from similarity import SimilarityMatrixBuilder
from visualization import ClusterVisualizer
from data_input import DataInputHandler


class FuzzyClusteringCalculator:
    """模糊聚类分析计算器主类"""

    def __init__(self):
        self.normalizer = DataNormalizer()
        self.similarity_builder = SimilarityMatrixBuilder()
        self.visualizer = ClusterVisualizer()
        self.data_handler = DataInputHandler()

    def display_welcome(self):
        """显示欢迎信息"""
        print("=" * 70)
        print("              模糊聚类分析计算器")
        print("=" * 70)
        print("功能：")
        print("  • 多种数据规格化方法")
        print("  • 多种模糊相似矩阵建立方法")
        print("  • 动态聚类图可视化")
        print("=" * 70)

    def select_method(self, available_methods, method_type):
        """选择方法"""
        print(f"\n请选择{method_type}：")
        for key, method_info in available_methods.items():
            print(f"  {key}. {method_info['name']}")

        while True:
            choice = input(f"请输入选择 ({min(available_methods.keys())}-{max(available_methods.keys())}): ").strip()
            if choice in available_methods:
                return choice
            else:
                print("无效选择，请重新输入")

    def run_analysis(self, data_matrix):
        """运行聚类分析"""
        # 显示原始数据
        self.visualizer.display_matrix(data_matrix, "原始数据矩阵")

        # 选择并执行规格化
        norm_methods = self.normalizer.get_available_methods()
        norm_choice = self.select_method(norm_methods, "数据规格化方法")
        normalized_data, norm_method_name = self.normalizer.normalize(data_matrix, norm_choice)
        self.visualizer.display_matrix(normalized_data, f"规格化后数据矩阵 ({norm_method_name})")

        # 选择并构建相似矩阵
        sim_methods = self.similarity_builder.get_available_methods()
        sim_choice = self.select_method(sim_methods, "模糊相似矩阵建立方法")

        metric = None
        if sim_choice == '3':
            print("\n请选择距离类型：")
            metrics = {
                '1': ('euclidean', '欧氏距离'),
                '2': ('hamming', '汉明距离'),
                '3': ('chebyshev', '切比雪夫距离')
            }
            for k, v in metrics.items():
                print(f"  {k}. {v[1]}")
            while True:
                metric_choice = input("请输入选择 (1-3): ").strip()
                if metric_choice in metrics:
                    metric = metrics[metric_choice][0]
                    break
                else:
                    print("无效选择，请重新输入")

        similarity_matrix, sim_method_name = self.similarity_builder.build_similarity_matrix(normalized_data,
                                                                                             sim_choice, metric=metric)
        self.visualizer.display_matrix(similarity_matrix, f"模糊相似矩阵 ({sim_method_name})")

        # 可视化结果
        print("\n正在生成可视化结果...")
        labels = [f'Sample{i + 1}' for i in range(len(data_matrix))]

        # 绘制动态聚类图
        self.visualizer.plot_dendrogram(similarity_matrix, labels)

        # 绘制对比图
        # self.visualizer.plot_comparison(data_matrix, normalized_data, similarity_matrix)

        print("\n✓ 分析完成！")

    def run(self):
        """运行主程序"""
        self.display_welcome()

        try:
            # 获取数据
            data_matrix = self.data_handler.get_input_method()

            # 运行分析
            self.run_analysis(data_matrix)

        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
        except Exception as e:
            print(f"\n✗ 程序运行出错: {e}")


def main():
    """主函数"""
    calculator = FuzzyClusteringCalculator()
    calculator.run()


if __name__ == "__main__":
    main()