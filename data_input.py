import numpy as np


class DataInputHandler:
    """数据输入处理类"""

    def __init__(self):
        self.example_datasets = {
            '1': {
                'name': '简单示例数据',
                'data': np.array([
                    [1, 2, 3, 4],
                    [2, 3, 4, 5],
                    [5, 6, 7, 8],
                    [1, 1, 2, 2],
                    [8, 7, 6, 5]
                ])
            },
            '2': {
                'name': '分类明显的数据',
                'data': np.array([
                    [1, 1, 1],
                    [1, 2, 1],
                    [8, 9, 8],
                    [9, 8, 9],
                    [2, 1, 2]
                ])
            },
            '3': {
                'name': '高维数据',
                'data': np.array([
                    [1, 3, 5, 2, 4],
                    [2, 4, 6, 3, 5],
                    [5, 7, 9, 6, 8],
                    [3, 1, 2, 4, 3],
                    [7, 5, 6, 8, 7]
                ])
            }
        }

    def input_matrix(self):
        """手动输入矩阵"""
        print("\n请输入数据矩阵：")
        print("格式示例：")
        print("  [[1,2,3], [4,5,6], [7,8,9]]")
        print("  或")
        print("  1,2,3;4,5,6;7,8,9")

        while True:
            try:
                matrix_input = input("请输入矩阵: ").strip()

                # 处理不同的输入格式
                if ';' in matrix_input:
                    # 处理分号格式
                    rows = matrix_input.split(';')
                    data = [list(map(float, row.split(','))) for row in rows]
                else:
                    # 处理列表格式
                    matrix_input = matrix_input.replace(' ', '').replace('],[', '];[')
                    data = eval(matrix_input)

                data_matrix = np.array(data)

                if len(data_matrix.shape) != 2:
                    raise ValueError("请输入二维矩阵")

                if data_matrix.shape[0] < 2 or data_matrix.shape[1] < 1:
                    raise ValueError("矩阵至少需要2行1列")

                print(f"✓ 成功输入矩阵，形状: {data_matrix.shape}")
                return data_matrix

            except Exception as e:
                print(f"✗ 输入格式错误: {e}，请重新输入")

    def select_example_data(self):
        """选择示例数据"""
        print("\n请选择示例数据集：")
        for key, dataset in self.example_datasets.items():
            print(f"  {key}. {dataset['name']} (形状: {dataset['data'].shape})")

        while True:
            choice = input("请输入选择: ").strip()
            if choice in self.example_datasets:
                dataset = self.example_datasets[choice]
                print(f"✓ 选择数据集: {dataset['name']}")
                return dataset['data']
            else:
                print("无效选择，请重新输入")

    def get_input_method(self):
        """获取输入方式"""
        print("\n请选择数据输入方式：")
        print("  1. 手动输入矩阵")
        print("  2. 使用示例数据")

        while True:
            choice = input("请输入选择 (1-2): ").strip()
            if choice == '1':
                return self.input_matrix()
            elif choice == '2':
                return self.select_example_data()
            else:
                print("无效选择，请重新输入")