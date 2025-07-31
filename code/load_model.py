import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizer
import torch
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import seaborn as sns
from sklearn.preprocessing import minmax_scale
import os
import math
import py3Dmol
import re
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def format_legend(text: str, max_line_length=40):
    """
    将 legend 文本自动分成最多两行，每行最多 max_line_length 字符，并居中。
    """
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) <= max_line_length:
            current_line = (current_line + " " + word).strip()
        else:
            lines.append(current_line)
            current_line = word
        if len(lines) == 2:  # 最多两行
            break
    if current_line and len(lines) < 2:
        lines.append(current_line)

    # 居中对齐（RDKit 不是真正支持居中，但我们尽量模拟）
    centered_lines = [line.center(max_line_length) for line in lines]
    return "\n".join(centered_lines)

class DrugTargetVisualizer:
    def __init__(self, drug_smiles, protein_sequence,
                 drug_model_path="seyonec/ChemBERTa-zinc-base-v1",
                 protein_model_path="Rostlab/prot_bert"):
        """
        初始化药物-靶标配对可视化工具

        参数:
        drug_smiles (str): 药物的SMILES字符串
        protein_sequence (str): 蛋白质序列
        drug_model_path (str): 药物模型的路径或名称
        protein_model_path (str): 蛋白质模型的路径或名称
        """
        self.drug_smiles = drug_smiles
        self.protein_sequence = protein_sequence

        # 准备药物分子
        self.drug_mol = self._prepare_molecule(drug_smiles)

        # 加载预训练模型
        logger.info(f"Loading drug model from: {drug_model_path}")
        self.drug_tokenizer = AutoTokenizer.from_pretrained(drug_model_path)
        self.drug_model = AutoModel.from_pretrained(drug_model_path, output_attentions=True)

        logger.info(f"Loading protein model from: {protein_model_path}")
        self.protein_tokenizer = BertTokenizer.from_pretrained(protein_model_path)
        self.protein_model = BertModel.from_pretrained(protein_model_path, output_attentions=True)

        # 计算注意力
        self.cross_attention = self._calculate_cross_attention()

        # 获取token信息
        self.drug_atoms, self.protein_residues = self._get_token_info()

        # 识别关键相互作用
        self.key_interactions = self._identify_key_interactions()

        # 准备颜色方案
        self.colors = [
            (1, 0, 0),  # 红色
            (0, 0, 1),  # 蓝色
            (0, 1, 0),  # 绿色
            (1, 1, 0),  # 黄色
            (1, 0, 1),  # 紫色
            (0, 1, 1),  # 青色
            (1, 0.5, 0),  # 橙色
            (0.5, 0, 0.5)  # 紫罗兰
        ]

    def _prepare_molecule(self, smiles):
        """准备RDKit分子对象"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.error(f"Invalid SMILES string: {smiles}")
            raise ValueError(f"Invalid SMILES string: {smiles}")

        mol = Chem.AddHs(mol)  # 添加氢原子以获得完整结构
        AllChem.EmbedMolecule(mol)  # 生成3D坐标
        return mol

    def _calculate_cross_attention(self):
        """计算药物-蛋白质的跨模态注意力"""
        # 处理药物输入
        drug_input = self.drug_tokenizer(
            self.drug_smiles,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding='max_length'
        )
        drug_output = self.drug_model(**drug_input, output_attentions=True)

        # 处理蛋白质输入
        protein_input = self.protein_tokenizer(
            self.protein_sequence,
            return_tensors='pt',
            truncation=True,
            max_length=128,
            padding='max_length'
        )
        protein_output = self.protein_model(**protein_input, output_attentions=True)

        # 计算跨模态注意力 (使用最后一层所有注意力头的平均值)
        layer = -1  # 最后一层
        all_heads_attention = torch.stack(drug_output.attentions)[layer].mean(dim=1)[0]  # 平均所有注意力头

        # 提取药物和蛋白质部分的注意力
        drug_tokens = self.drug_tokenizer.convert_ids_to_tokens(drug_input['input_ids'][0])
        protein_tokens = self.protein_tokenizer.convert_ids_to_tokens(protein_input['input_ids'][0])

        # 去除特殊token
        special_tokens = self.drug_tokenizer.all_special_tokens
        drug_atoms = [token for token in drug_tokens if token not in special_tokens and token != '<pad>']
        # 清理token：移除特殊字符
        drug_atoms = [re.sub(r'[^\w]', '', token) for token in drug_atoms]

        special_tokens = self.protein_tokenizer.all_special_tokens
        protein_residues = [token for token in protein_tokens if token not in special_tokens and token != '<pad>']
        # 清理token：移除特殊字符
        protein_residues = [re.sub(r'[^\w]', '', token) for token in protein_residues]

        drug_len = len(drug_atoms)
        protein_len = len(protein_residues)

        # 提取相关子矩阵
        cross_attention = all_heads_attention[:drug_len, :protein_len].detach().numpy()

        return cross_attention

    def _get_token_info(self):
        """获取药物原子和蛋白质残基信息"""
        # 获取药物token对应的原子
        drug_tokens = self.drug_tokenizer.tokenize(self.drug_smiles)
        special_tokens = self.drug_tokenizer.all_special_tokens
        drug_atoms = [token for token in drug_tokens if token not in special_tokens and token != '<pad>']
        # 清理token：移除特殊字符
        drug_atoms = [re.sub(r'[^\w]', '', token) for token in drug_atoms]

        # 获取蛋白质token对应的残基
        protein_tokens = self.protein_tokenizer.tokenize(self.protein_sequence)
        special_tokens = self.protein_tokenizer.all_special_tokens
        protein_residues = [token for token in protein_tokens if token not in special_tokens and token != '<pad>']
        # 清理token：移除特殊字符
        protein_residues = [re.sub(r'[^\w]', '', token) for token in protein_residues]

        return drug_atoms, protein_residues

    def _identify_key_interactions(self):
        """识别关键相互作用"""
        key_interactions = []

        # 找到每个药物原子最关注的蛋白质残基
        for i, atom in enumerate(self.drug_atoms):
            if i >= self.cross_attention.shape[0]:
                continue

            max_idx = np.argmax(self.cross_attention[i])
            max_val = self.cross_attention[i, max_idx]

            # 只关注显著的相互作用
            if max_val > np.median(self.cross_attention) * 1.5:
                # 确保索引在范围内
                if max_idx < len(self.protein_residues):
                    key_interactions.append({
                        'drug_atom': atom,
                        'protein_residue': self.protein_residues[max_idx],
                        'attention': max_val,
                        'atom_index': i,
                        'residue_index': max_idx
                    })

        # 按注意力权重排序
        key_interactions.sort(key=lambda x: x['attention'], reverse=True)

        return key_interactions

    def visualize_attention(self, save_path=None, top_k=80):
        """可视化注意力热力图"""
        if len(self.drug_atoms) == 0 or len(self.protein_residues) == 0:
            logger.error("No valid drug atoms or protein residues to visualize")
            return

        # 归一化
        norm_att = minmax_scale(self.cross_attention.flatten()).reshape(self.cross_attention.shape)

        # 选出 top_k 个最重