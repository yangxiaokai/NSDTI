# import pandas as pd
#
#
# def fasta_to_csv(fasta_path, csv_path):
#     targets = []
#     sequences = []
#     with open(fasta_path, 'r') as file:
#         current_target = None
#         current_sequence = []
#         for line in file:
#             line = line.strip()
#             if line.startswith('>'):
#                 if current_target:
#                     targets.append(current_target)
#                     sequences.append(''.join(current_sequence))
#                 # 去掉 >，然后按冒号分割，取第一个部分
#                 current_target = line[1:].strip().split(':')[0]
#                 current_sequence = []
#             else:
#                 current_sequence.append(line)
#         if current_target:
#             targets.append(current_target)
#             sequences.append(''.join(current_sequence))
#
#     df = pd.DataFrame({'drug': targets, 'smile': sequences})
#     df.to_csv(csv_path, index=False)
#
#
# # 调用示例
# fasta_to_csv('../datasets/TTD/P3-06-Biologic-drug-sequence.txt', '../datasets/TTD/drug_smile.csv')

#
# import pandas as pd
# # 读取三个 CSV 文件
# protein_df = pd.read_csv('../datasets/TTD/target_sequence.csv')  # 包含 target 和 sequence
# drug_df = pd.read_csv('../datasets/TTD/drug_smile.csv')        # 包含 drug 和 smile
# link_df = pd.read_csv('../datasets/TTD/P1-07-Drug-TargetMapping.csv')        # 包含 TargetID, DrugID, Highest_status, MOA
# # 重命名列以便匹配
# protein_df = protein_df.rename(columns={"target": "TargetID"})
# drug_df = drug_df.rename(columns={"drug": "DrugID"})
# # 合并 sequence
# merged_df = pd.merge(link_df, protein_df, on="TargetID", how="left")
# # 合并 smile
# merged_df = pd.merge(merged_df, drug_df, on="DrugID", how="left")
# # 保存最终结果
# merged_df.to_csv("../datasets/TTD/merged.csv", index=False)
# # 读取 CSV 文件
# df = pd.read_csv("../datasets/TTD/merged.csv")  # 替换为你的文件名
# df_clean = df.dropna()
# # 保存为新的文件
# df_clean.to_csv("../datasets/TTD/merged.csv", index=False)
# print(f"清洗后的文件共有 {len(df_clean)} 行")


# import pandas as pd
#
# # 打开 txt 文件
# with open("../datasets/TTD/P3-07-Approved_smi_inchi.txt", "r") as f:
#     lines = f.read().splitlines()
#
# # 初始化列表
# drugs = []
# current = {}
#
# # 逐行解析
# for line in lines:
#     if not line.strip():  # 空行，表示一组结束
#         if current:
#             drugs.append(current)
#             current = {}
#     else:
#         key, value = line.split("\t")
#         if key in ["DRUG__ID", "DRUGSMIL"]:
#             current[key] = value
#
# # 添加最后一组
# if current:
#     drugs.append(current)
#
# # 转为 DataFrame 并保存
# df = pd.DataFrame(drugs)
# df = df.rename(columns={"DRUG__ID": "drug", "DRUGSMIL": "smile"})
# df.to_csv("../datasets/TTD/drug_smile.csv", index=False)

# import pandas as pd
#
# # 1. 读取原始合并后 CSV（替换为你的文件名）
# df = pd.read_csv("../datasets/TTD/merged.csv")
#
# # 2. 只提取需要的两列，并调换顺序
# df2 = df[['smile', 'sequence']].copy()
#
# # 3. 重命名列
# df2.columns = ['SMILES', 'Protein']
#
# # 4. 新增一列 Y，全部设为 1
# df2['Y'] = 1
#
# # 5. 保存结果
# df2.to_csv("../datasets/TTD/merged.csv", index=False)
# print("处理完成，输出文件：output.csv")

# import pandas as pd
#
# # 1. 读取已有的正样本文件（所有 Y=1）
# pos_df = pd.read_csv("../datasets/TTD/ttd.csv")   # 包含 SMILES, Protein, Y=1
#
# # 2. 读取另一个文件，提取负样本
# #    假设它也至少包含两列：'smile' 和 'sequence'，或已重命名为 'SMILES'、'Protein'
# neg_source = pd.read_csv("../datasets/EC50/random1/train.csv")
#
# # 如果源文件列名还是小写，先选列并重命名：
# if set(['smile','sequence']).issubset(neg_source.columns):
#     neg_df = neg_source[['smile','sequence']].rename(columns={
#         'smile':'SMILES',
#         'sequence':'Protein'
#     })
# else:
#     # 如果已经是大写列名，直接拷贝
#     neg_df = neg_source[['SMILES','Protein']].copy()
#
# # 3. 随机抽取 1800 条作为负样本
# neg_df = neg_df.sample(n=1800, random_state=42).reset_index(drop=True)
#
# # 4. 标记 Y=0
# neg_df['Y'] = 0
#
# # 5. 合并正、负样本
# full_df = pd.concat([pos_df, neg_df], ignore_index=True)
#
# # 6. （可选）去重，防止重复对
# full_df = full_df.drop_duplicates(subset=['SMILES','Protein'], keep='first')
#
# # 7. （可选）打乱顺序
# full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)
#
# # 8. 保存最终文件
# full_df.to_csv("../datasets/TTD/ttd.csv", index=False)
# print(f"拼接完成，共 {len(full_df)} 条样本，保存在 final_dataset.csv")

#
# import pandas as pd
# from rdkit import Chem
#
# # 1. 读取你的数据
# df = pd.read_csv("../datasets/TTD/ttd.csv")  # 这是之前生成的总数据集
#
# # 2. 定义一个函数，判断SMILES是否有效
# def is_valid_smiles(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     return mol is not None
#
# # 3. 应用这个函数，筛选出有效的行
# valid_df = df[df['SMILES'].apply(is_valid_smiles)].reset_index(drop=True)
#
# # 4. 保存干净的数据
# valid_df.to_csv("../datasets/TTD/ttd.csv", index=False)
#
# print(f"清理完成！原始有 {len(df)} 条，清理后剩 {len(valid_df)} 条。")

import pandas as pd

# 设置 txt 文件路径和目标 csv 路径
txt_file = "../datasets/Celegans/Celegans_test.txt"
csv_file = "../datasets/Celegans/test.csv"

# 读取并解析数据
data = []
with open(txt_file, 'r') as f:
    for line in f:
        parts = line.strip().split(' ', 2)  # 最多分割两次，避免蛋白质序列中空格出错
        if len(parts) == 3:
            smiles, protein_seq, label = parts
            data.append([smiles, protein_seq, label])

# 转为 DataFrame 并保存为 CSV
df = pd.DataFrame(data, columns=['SMILES', 'Protein', 'Y'])
df.to_csv(csv_file, index=False)

print(f"文件已保存为 {csv_file}")
