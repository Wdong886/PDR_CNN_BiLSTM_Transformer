import pandas as pd
import os

# ========= 原有代码：提取Excel中指定行和列的数据 =========

# 读取Excel文件
file_path = '../data/1/dan_bag1_data_with_errors.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')  # 若sheet名不同需调整

# 每隔200行提取数据（从第200行开始，即索引199）
step = 200
rows_to_extract = range(199, len(df), step)  # Python索引从0开始，第201行对应索引200

# 提取指定列
columns_needed = ['pos_x', 'pos_y', 'Estimated X', 'Estimated Y']
result = df.iloc[rows_to_extract][columns_needed]

# 重置索引并保存结果
result.reset_index(drop=True, inplace=True)
result.to_excel('extracted_data.xlsx', index=False)

print("提取完成，结果已保存至 extracted_data.xlsx")
print("提取的数据如下：")
print(result)

# ========= 新增代码：将两个CSV文件内容复制到Excel结果的后面 =========

# 读取之前保存的Excel数据（也可以直接使用 result 变量）
df_excel = result.copy()

# 读取两个CSV文件（请将文件名替换为实际文件路径）
csv_file1 = 'predictRes/distance_predict.csv'
csv_file2 = 'predictRes/direction_predicted.csv'
csv1 = pd.read_csv(csv_file1)
csv2 = pd.read_csv(csv_file2)

# 可选：如果不需要 CSV 文件中的全部列，可以只选取需要的列，如：
# csv1 = csv1[['desired_col1', 'desired_col2']]
# csv2 = csv2[['desired_col3', 'desired_col4']]

# 检查行数并扩展至最大行数，未填充位置以NaN填充
max_rows = max(len(df_excel), len(csv1), len(csv2))
df_excel = df_excel.reindex(range(max_rows))
csv1 = csv1.reindex(range(max_rows))
csv2 = csv2.reindex(range(max_rows))

# 拼接DataFrame：按列合并，CSV数据将追加在原有Excel数据的右侧
df_merged = pd.concat([df_excel.reset_index(drop=True),
                       csv1.reset_index(drop=True),
                       csv2.reset_index(drop=True)], axis=1)

# 保存合并后的结果到CSV文件
merged_output_path = 'predictRes/positionCorrection_preprocess.csv'
df_merged.to_csv(merged_output_path, index=False)

print("合并完成，结果已保存为", merged_output_path)
