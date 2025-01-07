import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# 函数：遍历文件夹，获取所有符合条件的CSV文件
def get_csv_files(directory):
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


# 函数：读取CSV文件并二值化第二列的标签（0 或 1），然后统计为0的记录数
def count_values_in_column(file_path):
    df = pd.read_csv(file_path, header=0, delimiter=",")
    if df.shape[1] < 2:  # 检查是否至少有两列
        return 0  # 如果文件列数少于2列，返回0
    labels = df.iloc[:, 1]  # 获取第二列标签
    # 二值化标签：大于0.5的标签为1，<=0.5的标签为0
    binarized_labels = labels.apply(lambda x: 1 if x > 0.5 else 0)
    # 统计标签为0的数量
    count_zeros = (binarized_labels == 0).sum()  # 统计标签为0的数量
    return count_zeros


# 函数：统计所有文件的平均值
def calculate_average(counts):
    total_zeros = sum(counts)
    num_files = len(counts)
    if num_files == 0:
        return 0  # 如果没有文件，返回0
    average_zeros = total_zeros / num_files
    return average_zeros


# 函数：进行bootstrap采样并计算95%置信区间
def bootstrap_confidence_interval(data, num_samples=5000, alpha=0.05):
    # 进行bootstrap采样
    bootstrap_means = []
    for _ in range(num_samples):
        sampled_data = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sampled_data))

    # 计算95%置信区间
    lower_bound = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper_bound = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

    return lower_bound, upper_bound


# 函数：计算两组数据的p-value
def calculate_p_value(group1_rates, group2_rates):
    # 使用Welch's t-test（适用于方差不相等的情况）
    t_stat, p_value = ttest_ind(group1_rates, group2_rates, equal_var=False)
    return p_value


# 主函数：遍历文件夹并进行统计
def main(directory_suicide, directory_non_suicide):
    # 获取自杀组和非自杀组的CSV文件
    suicide_files = get_csv_files(directory_suicide)
    non_suicide_files = get_csv_files(directory_non_suicide)

    suicide_counts = []  # 用于存储自杀组零的数量
    non_suicide_counts = []  # 用于存储非自杀组零的数量

    # 统计自杀组的零的数量
    for file in suicide_files:
        print(f'Processing suicide group file: {file}')
        count_zeros = count_values_in_column(file)
        suicide_counts.append(count_zeros)

    # 统计非自杀组的零的数量
    for file in non_suicide_files:
        print(f'Processing non-suicide group file: {file}')
        count_zeros = count_values_in_column(file)
        non_suicide_counts.append(count_zeros)

    # 计算每个组的零的数量平均值和置信区间
    average_zeros_suicide = calculate_average(suicide_counts)
    average_zeros_non_suicide = calculate_average(non_suicide_counts)

    # Bootstrap置信区间
    lower_bound_suicide, upper_bound_suicide = bootstrap_confidence_interval(suicide_counts)
    lower_bound_non_suicide, upper_bound_non_suicide = bootstrap_confidence_interval(non_suicide_counts)

    print(f'\nSuicide Group:')
    print(f'Average number of zeros: {average_zeros_suicide:.2f}')
    print(f'Bootstrap 95% Confidence Interval: ({lower_bound_suicide:.2f}, {upper_bound_suicide:.2f})')

    print(f'\nNon-Suicide Group:')
    print(f'Average number of zeros: {average_zeros_non_suicide:.2f}')
    print(f'Bootstrap 95% Confidence Interval: ({lower_bound_non_suicide:.2f}, {upper_bound_non_suicide:.2f})')

    # 计算p-value
    p_value = calculate_p_value(suicide_counts, non_suicide_counts)
    print(f'\nP-value between suicide and non-suicide groups: {p_value:.4f}')


# 示例：设置目标文件夹路径并运行
if __name__ == "__main__":
    folder_path_suicide = 'suicide'  # 自杀组文件夹路径
    folder_path_non_suicide = 'none_suicide'  # 非自杀组文件夹路径
    main(folder_path_suicide, folder_path_non_suicide)
