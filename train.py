import torch
import torch.optim as optim
from torch.distributions import Categorical

from torch_geometric.data import Data

from parse import parse_rcpsp_instance
from network import ActorCritic
from env import RCPSPEnv
from trainer import ppo_update


def train():
    # 파일로부터 인스턴스 데이터를 불러옵니다.
    with open("dataset/raw_data/j305_7.sm", "r", encoding="utf-8") as f:
        data_str = f.read()
        instance = parse_rcpsp_instance(data_str)
        env = RCPSPEnv(instance)

    # 모델 초기화
    in_channels = 7  # 노드 feature 차원
    hidden_channels = 128
    model = ActorCritic(in_channels, hidden_channels)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    max_episodes = 10000
    gamma = 0.99
    print_interval = 50

    for ep in range(max_episodes):
        obs_dict = env.reset()
        # torch_geometric의 Data 객체로 변환: batch는 단일 에피소드이므로 모두 0
        obs_data = Data(
            x=obs_dict["x"],
            edge_index=obs_dict["edge_index"],
            mask=obs_dict["mask"],
            batch=torch.zeros(obs_dict["x"].shape[0], dtype=torch.long),
        )
        trajectories = []
        done = False
        step_count = 0

        while not done:
            # 모델 forward
            available_idx = torch.nonzero(obs_data["mask"]).squeeze().tolist()
            logits, state_value = model(obs_data)
            masked_logits = torch.where(
                obs_data.mask, logits, torch.full_like(logits, -1e9)
            )
            dist = Categorical(logits=masked_logits)
            action = (
                dist.sample().item()
            )  # action은 available list 내 인덱스 (여기서는 단순화)
            log_prob = dist.log_prob(torch.tensor(action))
            # 환경 step
            next_obs, reward, done, _ = env.step(action)
            # 다음 observation도 Data 객체로 변환
            obs_data_next = Data(
                x=next_obs["x"],
                edge_index=next_obs["edge_index"],
                mask=next_obs["mask"],
                batch=torch.zeros(next_obs["x"].shape[0], dtype=torch.long),
            )
            trajectories.append(
                {
                    "obs": obs_data,
                    "action": action,
                    "reward": reward,
                    "log_prob": log_prob,
                    "value": state_value,
                }
            )
            obs_data = obs_data_next
            step_count += 1

        # 에피소드 종료 후 PPO 업데이트 (에피소드당 trajectory가 짧으므로 단일 업데이트)
        ppo_update(model, optimizer, trajectories, gamma=gamma)
        if (ep + 1) % print_interval == 0:
            # 마지막 에피소드의 makespan(음수 보상이므로 음수값의 절댓값)이 목표치
            makespan = -trajectories[-1]["reward"]
            print(f"Episode {ep+1}, Steps: {step_count}, Makespan: {makespan}")
    env.render()


if __name__ == "__main__":
    train()
