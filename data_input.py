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
        """手动输入矩阵（改进版：更健壮的解析与错误提示）"""
        import ast
        print("\n请输入数据矩阵：")
        print("格式示例：")
        print("  [[1,2,3], [4,5,6], [7,8,9]]")
        print("  或")
        print("  1,2,3;4,5,6;7,8,9")

        while True:
            try:
                matrix_input = input("请输入矩阵: ").strip()
                if not matrix_input:
                    raise ValueError("输入为空")

                # 分号分行格式
                if ';' in matrix_input:
                    rows = [r.strip() for r in matrix_input.split(';') if r.strip()]
                    if not rows:
                        raise ValueError("未检测到有效行")
                    data = []
                    expected_len = None
                    for i, row in enumerate(rows, start=1):
                        parts = [p.strip() for p in row.split(',') if p.strip()]
                        if not parts:
                            raise ValueError(f"第 {i} 行为空或格式不正确")
                        try:
                            nums = [float(p) for p in parts]
                        except ValueError as ve:
                            raise ValueError(f"第 {i} 行包含非法数字: {parts}") from ve
                        if expected_len is None:
                            expected_len = len(nums)
                        elif len(nums) != expected_len:
                            raise ValueError(f"第 {i} 行与第一行列数不一致（期望 {expected_len}，实际 {len(nums)}）")
                        data.append(nums)
                else:
                    # 列表格式：使用 ast.literal_eval 替代 eval（更安全）
                    try:
                        parsed = ast.literal_eval(matrix_input)
                    except Exception as e:
                        raise ValueError("列表格式解析失败，请使用类似 [[1,2],[3,4]] 的格式") from e
                    if not isinstance(parsed, (list, tuple)) or not parsed:
                        raise ValueError("解析结果不是有效的二维列表")
                    # 验证并转换为浮点
                    data = []
                    expected_len = None
                    for i, row in enumerate(parsed, start=1):
                        if not isinstance(row, (list, tuple)):
                            raise ValueError(f"第 {i} 行不是列表")
                        try:
                            nums = [float(x) for x in row]
                        except Exception as ve:
                            raise ValueError(f"第 {i} 行包含非法数字: {row}") from ve
                        if expected_len is None:
                            expected_len = len(nums)
                        elif len(nums) != expected_len:
                            raise ValueError(f"第 {i} 行与第一行列数不一致（期望 {expected_len}，实际 {len(nums)}）")
                        data.append(nums)

                data_matrix = np.array(data)
                if data_matrix.ndim != 2:
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