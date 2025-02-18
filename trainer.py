import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from torch_geometric.data import Data


def ppo_update(
    model, optimizer, trajectories, clip_epsilon=0.2, epochs=4, batch_size=8, gamma=1.0
):
    """
    trajectories: list of dicts with
    keys:
    'obs': observation (Data 객체)
    'action': int (action index)
    'reward': float
    'log_prob': tensor
    'value': tensor 에피소드가 짧으므로 Monte-Carlo 리턴을 사용합니다.
    """
    # 에피소드 전체 return (reward는 sparse하므로 마지막 reward만 nonzero)
    returns = []
    G = 0
    for t in reversed(trajectories):
        G = t["reward"] + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float)

    obs_list = [t["obs"] for t in trajectories]
    actions = torch.tensor([t["action"] for t in trajectories], dtype=torch.long)
    old_log_probs = torch.stack([t["log_prob"] for t in trajectories]).detach()
    values = torch.stack([t["value"] for t in trajectories]).squeeze()

    advantages = returns - values.detach()

    # 여러 epoch 동안 미니배치 업데이트
    for _ in range(epochs):
        for i in range(0, len(trajectories), batch_size):
            # 여기서는 배치 사이즈 단위로 순회 (에피소드 길이 작으므로 단순하게)
            batch_obs = obs_list[i : i + batch_size]
            # 배치 내 각 Data 객체의 batch 벡터 설정 (모두 0)
            for data in batch_obs:
                data.batch = torch.zeros(data.x.shape[0], dtype=torch.long)
            batch_data = Data.from_dict(
                {
                    "x": torch.cat([d.x for d in batch_obs], dim=0),
                    "edge_index": torch.cat([d.edge_index for d in batch_obs], dim=1),
                    "mask": torch.cat([d.mask for d in batch_obs], dim=0),
                    "batch": torch.cat(
                        [
                            torch.zeros(d.x.shape[0], dtype=torch.long)
                            for d in batch_obs
                        ],
                        dim=0,
                    ),
                }
            )

            batch_logits, batch_values = model(batch_data)
            new_log_probs = []
            pointer = 0
            for d in batch_obs:
                num_nodes = d.x.shape[0]
                logits = batch_logits[pointer : pointer + num_nodes]
                dist = Categorical(logits=logits)
                new_log_probs.append(dist.log_prob(actions))
                pointer += num_nodes
            new_log_probs = torch.stack(new_log_probs)
            ratio = torch.exp(
                new_log_probs
                - old_log_probs[i : i + batch_size]
                .unsqueeze(1)
                .repeat(1, new_log_probs.size(1))
            )
            advantages_ = (
                advantages[i : i + batch_size]
                .unsqueeze(1)
                .repeat(1, new_log_probs.size(1))
            )
            surr1 = ratio * advantages_
            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(
                batch_values.squeeze(), returns[i : i + batch_size].mean()
            )
            loss = policy_loss + 0.5 * value_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
