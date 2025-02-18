import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class ActorCritic(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        """
        in_channels: 노드 feature 차원
        hidden_channels: 은닉 차원
        """
        super(ActorCritic, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.policy_head = nn.Linear(
            hidden_channels, 1
        )  # policy head: 각 노드에 대해 스코어를 산출 (후에 mask 적용)
        self.value_head = nn.Linear(
            hidden_channels, 1
        )  # value head: 전체 그래프(상태)의 가치를 global pooling 후 산출

    def forward(self, data):
        """
        data: torch_geometric.data.Data 형태의 객체
        - data.x: [num_nodes, in_channels]
        - data.edge_index: [2, num_edges]
        - data.batch: 각 노드가 속한 배치 인덱스 (여기서는 단일 에피소드이므로 모두 0)
        - data.mask: [num_nodes] (available 작업 mask)
        """
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # policy: 각 노드별 스코어 산출
        logits = self.policy_head(x).squeeze(-1)  # [num_nodes]
        # available mask 처리: mask가 False면 아주 낮은 값(-1e9) 부여
        logits = torch.where(data.mask, logits, torch.full_like(logits, -1e9))
        # value: global pooling 후 산출 (여기서는 mean pooling)
        pooled = global_mean_pool(x, data.batch)
        state_value = self.value_head(pooled).squeeze(-1)
        return logits, state_value
