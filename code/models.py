import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from ACmix import ACmix
from Intention import BiIntention


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


# class KDTree(nn.Module):
#     def __init__(self, feature_dim, attention_dim=128):
#         super(KDTree, self).__init__()
#         self.feature_dim = feature_dim
#
#         # Attention mechanism to learn similarity between nodes
#         self.attention_network = nn.Sequential(
#             nn.Linear(feature_dim, attention_dim),
#             nn.ReLU(),
#             nn.Linear(attention_dim, 1)  # Output the similarity score
#         )
#
#     def learn_similarity(self, query, candidates):
#         """
#         Learn the similarity between a query and a list of candidates.
#         :param query: Query node, shape (batch_size, 1, feature_dim)
#         :param candidates: Candidate nodes, shape (batch_size, num_candidates, feature_dim)
#         :return: Similarity scores, shape (batch_size, num_candidates)
#         """
#         combined_features = torch.cat([query, candidates], dim=1)  # Combine query and candidates
#         # Pass through attention network to compute similarity scores
#         similarity_scores = self.attention_network(combined_features)  # Shape (batch_size, num_candidates, 1)
#         similarity_scores = similarity_scores.squeeze(-1)  # Shape (batch_size, num_candidates)
#         return similarity_scores
#
#     def query(self, query_point, candidates, k=1):
#         """
#         Query the nearest neighbors from the candidates.
#         :param query_point: Query point, shape (batch_size, 1, feature_dim)
#         :param candidates: Candidate points, shape (batch_size, num_candidates, feature_dim)
#         :param k: Number of nearest neighbors to return
#         :return: Indices of the nearest neighbors, shape (batch_size, k)
#         """
#         # Calculate the similarity scores between the query and the candidates
#         similarity_scores = self.learn_similarity(query_point, candidates)
#
#         # Get the top-k most similar neighbors
#         _, nearest_neighbors = torch.topk(similarity_scores, k=k, dim=1, largest=True, sorted=False)
#
#         return nearest_neighbors
#
#
# class DisjointSetWithAttention:
#     def __init__(self, n, feature_dim, attention_dim=128):
#         """
#         Initialize the disjoint set (ADT).
#         :param n: Number of nodes
#         :param feature_dim: Feature dimension
#         :param attention_dim: Attention dimension for merging
#         """
#         self.parent = torch.arange(n)  # Initially, each node is its own parent
#         self.rank = torch.ones(n, dtype=torch.long)  # Initialize rank to 1
#         self.feature_dim = feature_dim
#
#         # Define an attention network for learning merging strategies
#         self.attention_network = nn.Sequential(
#             # nn.Linear(feature_dim, feature_dim * 2),
#             nn.Linear(feature_dim * 2, attention_dim),
#             nn.ReLU(),
#             nn.Linear(attention_dim, 1)  # Output the merging attention score
#         )
#
#     def find(self, x):
#         """
#         Find the representative element of the set to which x belongs.
#         :param x: Node index
#         :return: The representative element of the set containing x
#         """
#         if self.parent[x] != x:
#             self.parent[x] = self.find(self.parent[x])  # Path compression
#         return self.parent[x]
#
#     def union(self, x, y, x_features, y_features):
#         """
#         Union operation: Merge the sets containing x and y, using attention to decide the merge.
#         :param x: Node x index
#         :param y: Node y index
#         :param x_features: Features of node x
#         :param y_features: Features of node y
#         """
#         rootX = self.find(x)
#         rootY = self.find(y)
#         if rootX != rootY:
#             # Use attention network to compute the merge score
#             combined_features = torch.cat([x_features, y_features], dim=-1)  # Concatenate features
#             self.attention_network = self.attention_network.to(combined_features.device)
#             attention_score = self.attention_network(combined_features)  # Compute merge attention score
#             # If attention score is greater than a threshold, merge the sets
#             if torch.max(attention_score) > 0.5:
#                 if self.rank[rootX] > self.rank[rootY]:
#                     self.parent[rootY] = rootX
#                 elif self.rank[rootX] < self.rank[rootY]:
#                     self.parent[rootX] = rootY
#                 else:
#                     self.parent[rootY] = rootX
#                     self.rank[rootX] += 1
#             return attention_score
#         else:
#             return torch.zeros(64, 1).to(x_features.device)

# class KDTree(nn.Module):
#     class Node:
#         def __init__(self, point, left=None, right=None, axis=None):
#             self.point = point  # [feature_dim]
#             self.left = left  # 左子树
#             self.right = right  # 右子树
#             self.axis = axis  # 划分轴
#
#     def __init__(self, feature_dim, max_depth=10, k=5):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.max_depth = max_depth
#         self.root = None
#         self.k = k
#
#     def _build(self, points, depth=0):
#         if len(points) == 0 or depth >= self.max_depth:
#             return None
#
#         axis = depth % self.feature_dim  # 循环选择划分轴
#         sorted_points = sorted(points, key=lambda x: x[axis])
#         median = len(sorted_points) // 2
#
#         node = self.Node(
#             point=sorted_points[median],
#             axis=axis,
#             left=self._build(sorted_points[:median], depth + 1),
#             right=self._build(sorted_points[median + 1:], depth + 1)
#         )
#         return node
#
#     def build(self, features):
#         """构建K-d树
#         :param features: [batch_size, num_nodes, feature_dim]
#         """
#         self.trees = []
#         for batch in features:
#             self.trees.append(self._build(batch.unbind(0)))
#         return self.trees
#
#     def _search(self, node, point, k=5):
#         if node is None:
#             return []
#
#         axis = node.axis
#         if point[axis] < node.point[axis]:
#             next_node = node.left
#             opposite_node = node.right
#         else:
#             next_node = node.right
#             opposite_node = node.left
#
#         best = self._search(next_node, point, k)
#         best.append((node.point, torch.norm(node.point - point)))
#
#         if len(best) < k or (point[axis] - node.point[axis]) ** 2 < best[-1][1] ** 2:
#             best += self._search(opposite_node, point, k)
#
#         best.sort(key=lambda x: x[1])
#         return best[:k]
#
#     def query(self, queries):
#         """批量查询最近邻
#         :param queries: [batch_size, feature_dim]
#         :return: 最近邻特征集合 [batch_size, k, feature_dim]
#         """
#         results = []
#         for q, tree in zip(queries, self.trees):
#             neighbors = self._search(tree, q)[:self.k]
#             results.append(torch.stack([n[0] for n in neighbors]))
#         return torch.stack(results)

class KDTree(nn.Module):
    class Node:
        def __init__(self, point, left=None, right=None, axis=None):
            self.point = point  # [feature_dim]
            self.left = left  # 左子树
            self.right = right  # 右子树
            self.axis = axis  # 划分轴

    def __init__(self, feature_dim, max_depth=2, k=2):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_depth = max_depth
        self.root = None
        self.k = k

    def _build(self, points, depth=0):
        if len(points) == 0 or depth >= self.max_depth:
            return None

        axis = depth % self.feature_dim  # 循环选择划分轴

        # 确保 points 是列表，可以进行排序
        points = list(points)

        # 使用 sorted() 而非 sort()，因为 points 可能是元组
        sorted_points = sorted(points, key=lambda x: x[axis])
        median = len(sorted_points) // 2

        node = self.Node(
            point=sorted_points[median],
            axis=axis,
            left=self._build(sorted_points[:median], depth + 1),
            right=self._build(sorted_points[median + 1:], depth + 1)
        )
        return node

    def build(self, features):
        """构建K-d树
        :param features: [batch_size, num_nodes, feature_dim]
        """
        self.trees = []
        for batch in features:
            self.trees.append(self._build(batch.unbind(0)))
        return self.trees

    def _search(self, node, point, k=3):
        if node is None:
            return []

        axis = node.axis
        # 使用划分轴比较点
        if point[axis] < node.point[axis]:
            next_node = node.left
            opposite_node = node.right
        else:
            next_node = node.right
            opposite_node = node.left

        # 递归搜索
        best = self._search(next_node, point, k)
        best.append((node.point, torch.norm(node.point - point)))

        # 进行剪枝优化
        if len(best) < k or (point[axis] - node.point[axis]) ** 2 < best[-1][1] ** 2:
            best += self._search(opposite_node, point, k)

        best.sort(key=lambda x: x[1])
        return best[:k]

    def query(self, queries):
        """批量查询最近邻
        :param queries: [batch_size, feature_dim]
        :return: 最近邻特征集合 [batch_size, k, feature_dim]
        """
        results = []
        for q, tree in zip(queries, self.trees):
            # 减少不必要的计算，直接构建张量
            neighbors = self._search(tree, q)[:self.k]
            results.append(torch.stack([n[0] for n in neighbors]))
        return torch.stack(results)

class DisjointSet(nn.Module):
    def __init__(self, num_nodes, feature_dim, device):
        super().__init__()
        self.parent = torch.arange(num_nodes, device=device)
        self.rank = torch.ones(num_nodes, device=device)
        self.sim_threshold = 0.7  # 相似度阈值

        # 跨域相似度计算
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

        # 注意力权重网络
        self.attn = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def find(self, x):
        # 路径压缩
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y, x_feat, y_feat):
        # 跨域相似度计算
        sim = self.cosine_sim(x_feat, y_feat)
        if sim < self.sim_threshold:
            return 0.0

        # 计算注意力权重
        attn_weight = self.attn(torch.cat([x_feat, y_feat], -1))

        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return attn_weight

        # 按秩合并
        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.rank[root_x] += self.rank[root_y] * attn_weight
        else:
            self.parent[root_x] = root_y
            self.rank[root_y] += self.rank[root_x] * attn_weight

        return attn_weight


class FastKDTree(nn.Module):
    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def build(self, features):
        """直接存储特征矩阵"""
        self.features = features  # [B, N, D]

    def query(self, queries):
        """批量矩阵运算替代树搜索"""
        # queries: [B, D]
        distances = torch.cdist(queries.unsqueeze(1), self.features).squeeze(1)  # [B, N]
        _, indices = torch.topk(distances, k=self.k, dim=1, largest=False)
        return torch.stack([self.features[i, idx] for i, idx in enumerate(indices)])  # [B, k, D]


class FastDisjointSet(nn.Module):
    def __init__(self, num_nodes, device):
        super().__init__()
        self.parent = torch.arange(num_nodes, device=device)
        self.rank = torch.ones(num_nodes, device=device)

    def find(self, x):
        """路径压缩优化"""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y, sim):
        """简化合并条件"""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y or sim < 0.6:
            return

        if self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
            self.rank[root_x] += self.rank[root_y]
        else:
            self.parent[root_x] = root_y
            self.rank[root_y] += self.rank[root_x]

class BINDTI(nn.Module):
    def __init__(self, device='cuda', **config):
        super(BINDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
        cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
        cross_layer = config['CROSSINTENTION']['LAYER']
        batch_size = config["SOLVER"]["BATCH_SIZE"]

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                           padding=drug_padding,
                                           hidden_feats=drug_hidden_feats)
        self.protein_extractor = ProteinACmix(protein_emb_dim, num_filters, protein_num_head, protein_padding)

        # K-d树和不相交集合（ADT）
        # 替换为简化模块
        self.kd_tree = FastKDTree(k=5)
        self.disjoint_sets = nn.ModuleList()

        # 增加批量相似度计算
        self.sim_layer = nn.CosineSimilarity(dim=-1)

        # 增加跨域特征投影
        self.drug_proj = nn.Linear(drug_hidden_feats[-1], 128)
        self.prot_proj = nn.Linear(num_filters[-1], 128)
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2)
        )

        self.cross_intention = BiIntention(embed_dim=cross_emb_dim, num_head=cross_num_head, layer=cross_layer,
                                           device=device)
        self.s_cross_intention = CrossAttentionLayer(1)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, mode="train"):

        v_d = self.drug_extractor(bg_d)  # v_d.shape(64, 290, 128)
        v_p = self.protein_extractor(v_p)  # v_p.shape:(64, 1200, 128)

        nv_d = self.drug_proj(v_d)
        nv_p = self.prot_proj(v_p)
        # 批量构建和查询
        combined_feats = torch.cat([nv_d, nv_p], 1)  # [B, N, D]
        self.kd_tree.build(combined_feats)
        neighbors = self.kd_tree.query(nv_d.mean(1))  # 使用药物特征均值查询

        # 批量相似度计算
        prot_feats = combined_feats[:, nv_d.size(1):]  # [B, N_p, D]
        sim_matrix = self.sim_layer(
            nv_d.unsqueeze(2),  # [B, N_d, 1, D]
            prot_feats.unsqueeze(1)  # [B, 1, N_p, D]
        )  # [B, N_d, N_p]
        # 批量合并操作
        batch_size, num_nodes = combined_feats.shape[:2]

        # 初始化批量DisjointSet
        self.disjoint_sets = nn.ModuleList(
            [FastDisjointSet(num_nodes, v_d.device) for _ in range(batch_size)]
        )

        # 遍历每个样本进行合并操作
        for b in range(batch_size):
            # 获取当前样本的相似度矩阵
            sim_matrix_b = sim_matrix[b]  # [N_d, N_p]

            # 找到相似度大于阈值的索引
            mask = sim_matrix_b > 0.6
            rows, cols = torch.where(mask)  # 获取满足条件的药物-靶点对

            # 遍历满足条件的药物-靶点对
            for i, j in zip(rows, cols):
                # 执行合并操作
                self.disjoint_sets[b].union(i, j, sim_matrix_b[i, j])

        # 优化特征聚合
        fused_feats = []
        for b in range(batch_size):
            # 获取当前样本的特征和DisjointSet
            cfeat = combined_feats[b]  # [N, D]
            ds = self.disjoint_sets[b]  # DisjointSet实例

            # 获取所有根节点
            root_nodes = torch.unique(ds.parent)

            # 对每个根节点聚合特征
            group_feats = []
            for root in root_nodes:
                # 找到属于当前根节点的所有节点
                members = (ds.parent == root)
                # 计算这些节点的特征均值
                group_feats.append(cfeat[members].mean(0))

            # 对所有组特征取均值
            fused_feats.append(torch.stack(group_feats).mean(0))

        # 将结果堆叠为批量张量
        fused = torch.stack(fused_feats)  # [B, D]
        fused = self.fusion(fused)
        fused = torch.squeeze(fused, 1)
        final_score = self.mlp_classifier(fused)
        f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p)  # f:[64, 256]
        # score = self.mlp_classifier(f)
        if mode == "train":
            return v_d, v_p, f, final_score
        elif mode == "eval":
            return v_d, v_p, final_score, att

# class BINDTI(nn.Module):
#     def __init__(self, device='cuda', **config):
#         super(BINDTI, self).__init__()
#         drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
#         drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
#         drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
#         protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
#         num_filters = config["PROTEIN"]["NUM_FILTERS"]
#         mlp_in_dim = config["DECODER"]["IN_DIM"]
#         mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
#         mlp_out_dim = config["DECODER"]["OUT_DIM"]
#         drug_padding = config["DRUG"]["PADDING"]
#         protein_padding = config["PROTEIN"]["PADDING"]
#         out_binary = config["DECODER"]["BINARY"]
#         protein_num_head = config['PROTEIN']['NUM_HEAD']
#         cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
#         cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
#         cross_layer = config['CROSSINTENTION']['LAYER']
#         batch_size = config["SOLVER"]["BATCH_SIZE"]
#
#         self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
#                                            padding=drug_padding,
#                                            hidden_feats=drug_hidden_feats)
#         self.protein_extractor = ProteinACmix(protein_emb_dim, num_filters, protein_num_head, protein_padding)
#
#         # K-d树和不相交集合（ADT）
#         # 修改后的模块
#         self.kd_tree = KDTree(feature_dim=128, max_depth=2)
#         self.disjoint_sets = nn.ModuleList()  # 每个样本独立DS
#
#         # 增加跨域特征投影
#         self.drug_proj = nn.Linear(drug_hidden_feats[-1], 128)
#         self.prot_proj = nn.Linear(num_filters[-1], 128)
#         # 特征融合层
#         self.fusion = nn.Sequential(
#             nn.Linear(128, 256),
#             nn.LayerNorm(256),
#             nn.GELU(),
#             nn.Dropout(0.2)
#         )
#
#         self.cross_intention = BiIntention(embed_dim=cross_emb_dim, num_head=cross_num_head, layer=cross_layer,
#                                            device=device)
#         self.s_cross_intention = CrossAttentionLayer(1)
#         self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
#
#     def forward(self, bg_d, v_p, mode="train"):
#
#         v_d = self.drug_extractor(bg_d)  # v_d.shape(64, 290, 128)
#         v_p = self.protein_extractor(v_p)  # v_p.shape:(64, 1200, 128)
#
#         nv_d = self.drug_proj(v_d)
#         nv_p = self.prot_proj(v_p)
#         # 构建K-d树
#         combined_feats = torch.cat([nv_d, nv_p], dim=1)  # [B, N_d+N_p, 128]
#         self.kd_tree.build(combined_feats)
#
#         # 动态维护DisjointSet
#         batch_size, num_nodes = combined_feats.shape[:2]
#         self.disjoint_sets.extend([  # 使用 extend 方法
#             DisjointSet(num_nodes, 128, v_d.device) for _ in range(batch_size)
#         ])
#         # 跨域特征交互
#         cross_attns = []
#         for b in range(batch_size):
#             # 查询每个药物节点的k近邻
#             drug_feats = v_d[b]  # [N_d, 128]
#             neighbors = self.kd_tree.query(drug_feats)  # [N_d, k, 128]
#
#             # 与靶点特征交互
#             for i, (drug_node, prot_group) in enumerate(zip(drug_feats, neighbors)):
#                 for j, prot_feat in enumerate(prot_group):
#                     # 计算跨域合并
#                     sim = self.disjoint_sets[b].cosine_sim(drug_node, prot_feat)
#                     if sim > 0.6:  # 相似度阈值
#                         attn = self.disjoint_sets[b].union(
#                             i, num_nodes - prot_feat.shape[0] + j,  # 调整索引
#                             drug_node,
#                             prot_feat
#                         )
#                         cross_attns.append(attn)
#
#         # 特征聚合
#         fused_feats = []
#         for b in range(batch_size):
#             group_feats = []
#             root_nodes = set(self.disjoint_sets[b].parent.tolist())
#             for root in root_nodes:
#                 members = (self.disjoint_sets[b].parent == root).nonzero()
#                 group_feats.append(combined_feats[b, members].mean(0))
#             fused_feats.append(torch.stack(group_feats).mean(0))  # [128]
#         # 最终预测
#         fused = self.fusion(torch.stack(fused_feats))  # [B, 256]
#         fused = torch.squeeze(fused, 1)
#         final_score = self.mlp_classifier(fused)
#         f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p)  # f:[64, 256]
#         # score = self.mlp_classifier(f)
#         if mode == "train":
#             return v_d, v_p, f, final_score
#         elif mode == "eval":
#             return v_d, v_p, final_score, att


# class BINDTI(nn.Module):
#     def __init__(self, device='cuda', **config):
#         super(BINDTI, self).__init__()
#         drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
#         drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
#         drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
#         protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
#         num_filters = config["PROTEIN"]["NUM_FILTERS"]
#         mlp_in_dim = config["DECODER"]["IN_DIM"]
#         mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
#         mlp_out_dim = config["DECODER"]["OUT_DIM"]
#         drug_padding = config["DRUG"]["PADDING"]
#         protein_padding = config["PROTEIN"]["PADDING"]
#         out_binary = config["DECODER"]["BINARY"]
#         protein_num_head = config['PROTEIN']['NUM_HEAD']
#         cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
#         cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
#         cross_layer = config['CROSSINTENTION']['LAYER']
#
#         self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
#                                            padding=drug_padding,
#                                            hidden_feats=drug_hidden_feats)
#         self.protein_extractor = ProteinACmix(protein_emb_dim, num_filters, protein_num_head, protein_padding)
#
#         # K-d树和不相交集合（ADT）
#         self.kd_tree = KDTree(feature_dim=128, attention_dim=128)
#         self.disjoint_set = None  # 不相交集合（ADT）
#
#         self.cross_intention = BiIntention(embed_dim=cross_emb_dim, num_head=cross_num_head, layer=cross_layer,
#                                            device=device)
#         self.s_cross_intention = CrossAttentionLayer(1)
#         self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)
#
#     def forward(self, bg_d, v_p, mode="train"):
#
#         v_d = self.drug_extractor(bg_d)  # v_d.shape(64, 290, 128)
#         v_p = self.protein_extractor(v_p)  # v_p.shape:(64, 1200, 128)
#
#         # 使用K-d树进行特征查询
#         combined_features = torch.cat((v_d, v_p), dim=1)  # 拼接药物和靶点特征
#         nearest_neighbors = self.kd_tree.query(v_d, combined_features, k=5)
#         # 使用不相交集合（ADT）进行节点合并
#         num_nodes = v_d.size(1) + v_p.size(1)
#         self.disjoint_set = DisjointSetWithAttention(num_nodes, 128, 128)
#         tensor_list = []
#         # 药物图和靶点图的索引是分开的，索引范围不重叠,v_d 和 v_p 分别对应不同的节点集合
#         for i in range(nearest_neighbors.size(1)):  # i 遍历邻居数量（k）
#             for j in range(nearest_neighbors.size(0)):  # j 遍历每个batch中的样本
#                 idx1 = nearest_neighbors[j, i]  # 获取第j个样本的第i个邻居的索引
#                 if idx1 >= num_nodes:
#                     continue  # 跳过当前操作
#                 # 假设药物和靶点的节点索引是分开的，检查是否在 v_d 中
#                 if idx1 < v_d.size(1):
#                     v_d_feat = v_d[:, idx1, :]
#                 else:
#                     v_d_feat = torch.zeros_like(v_d[:, 0, :])  # 处理越界情况
#
#                 # 获取第二个邻居的索引
#                 idx2 = nearest_neighbors[j, i + 1] if i + 1 < nearest_neighbors.size(1) else idx1
#                 if idx2 >= num_nodes:
#                     continue  # 跳过当前操作
#                 # 同样检查是否在 v_p 中
#                 if idx2 < v_p.size(1):
#                     v_p_feat = v_p[:, idx2, :]
#                 else:
#                     v_p_feat = torch.zeros_like(v_p[:, 0, :])  # 处理越界情况
#
#                 # 调用 union 操作
#                 temp = self.disjoint_set.union(idx1.item(), idx2.item(), v_d_feat, v_p_feat)
#                 tensor_list.append(temp)
#         stacked_tensors = torch.stack(tensor_list, dim=0)
#         adt_score = torch.sum(stacked_tensors, dim=0)
#         f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p)  # f:[64, 256]
#         score = self.mlp_classifier(f)
#         final_score = self.s_cross_intention(score, adt_score)
#         if mode == "train":
#             return v_d, v_p, f, final_score
#         elif mode == "eval":
#             return v_d, v_p, final_score, att


class CrossAttentionLayer(torch.nn.Module):
    def __init__(self, input_dim):
        super(CrossAttentionLayer, self).__init__()
        # 注意力机制中的查询（Q），键（K）和值（V）权重
        self.query_weight = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.key_weight = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.value_weight = torch.nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, x1, x2):
        # 输入张量 x1 和 x2 的形状是 [32, 1]

        # 计算 Q, K, V
        Q = self.query_weight(x1)  # [32, 1]
        K = self.key_weight(x2)  # [32, 1]
        V = self.value_weight(x2)  # [32, 1]

        # 计算 Q 和 K 的点积，得到注意力分数
        attention_scores = torch.matmul(Q, K.transpose(0, 1))  # [32, 32]

        # 使用 softmax 归一化注意力分数
        attention_weights = F.softmax(attention_scores, dim=-1)  # [32, 32]

        # 根据注意力权重加权 V
        attention_output = torch.matmul(attention_weights, V)  # [32, 1]

        return attention_output


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN, self).__init__()
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)
        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
        self.output_feats = hidden_feats[-1]

    def forward(self, batch_graph):
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)
        node_feats = self.gnn(batch_graph, node_feats)
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        return node_feats


class ProteinACmix(nn.Module):
    def __init__(self, embedding_dim, num_filters, num_head, padding=True):
        super(ProteinACmix, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]

        self.acmix1 = ACmix(in_planes=in_ch[0], out_planes=in_ch[1], head=num_head)
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.acmix2 = ACmix(in_planes=in_ch[1], out_planes=in_ch[2], head=num_head)
        self.bn2 = nn.BatchNorm1d(in_ch[2])

        self.acmix3 = ACmix(in_planes=in_ch[2], out_planes=in_ch[3], head=num_head)
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)  # 64*128*1200

        v = self.bn1(F.relu(self.acmix1(v.unsqueeze(-2))).squeeze(-2))
        v = self.bn2(F.relu(self.acmix2(v.unsqueeze(-2))).squeeze(-2))

        v = self.bn3(F.relu(self.acmix3(v.unsqueeze(-2))).squeeze(-2))

        v = v.view(v.size(0), v.size(2), -1)
        return v


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):  # x.shpae[64, 256]
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x