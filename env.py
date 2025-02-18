import numpy as np
import gym
from gym import spaces
import torch


class RCPSPEnv(gym.Env):
    """
    각 단계에서 사용 가능한 작업(모든 선행 작업이 완료된 작업) 중 하나를 선택하면,
    선택한 작업을 가능한 가장 빠른 시점에 스케줄합니다.
    에피소드는 모든 작업이 스케줄될 때 종료되며, 최종 보상은 -makespan입니다.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, instance):
        super(RCPSPEnv, self).__init__()
        self.instance = instance
        self.jobs = instance["jobs"]
        self.resources_capacity = instance["resources"]
        self.horizon = instance["horizon"]
        self.num_resources = len(self.resources_capacity)
        self.num_jobs = len(self.jobs)
        self.job_ids = sorted(list(self.jobs.keys()))

        # 타임라인 자원 사용량 (각 자원별 horizon 길이의 배열)
        self.res_usage = np.zeros((self.num_resources, self.horizon), dtype=int)

        # 각 작업에 대해 스케줄 정보 저장: { job_id: (start_time, finish_time) }
        self.schedule = {}

        # 각 작업의 상태: scheduled 여부
        self.scheduled = {j: False for j in self.job_ids}

        # Gym의 action_space는 “현재 가능한 작업들 중 하나 선택”이므로
        # action은 정수(index)이며, observation에서 mask로 선택가능 작업을 알 수 있음.
        # (action_space는 변동적이지만 여기서는 최대 작업 수를 범위로 설정)
        self.action_space = spaces.Discrete(self.num_jobs)
        # observation은 graph 데이터(노드 feature, edge_index, mask)를 dict로 구성
        # (딕셔너리로 구성하여 추후 GNN 입력에 활용)
        num_edges = sum(len(s) for s in instance["precedence"].values())
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.num_jobs, 7), dtype=np.float32
                ),
                "edge_index": spaces.Box(
                    low=0, high=self.num_jobs, shape=(2, num_edges), dtype=np.int64
                ),
                "mask": spaces.Box(
                    low=0, high=1, shape=(self.num_jobs,), dtype=np.int8
                ),
            }
        )

        # 에피소드 시작 시 dummy start 작업(예: duration 0인 작업) 자동 스케줄 여부: 파일 상에서는 1번이 duration 0
        self._auto_schedule_start()

    def _auto_schedule_start(self):
        # 만약 duration==0인 작업이 있다면, 먼저 스케줄합니다.
        for j in self.job_ids:
            if self.jobs[j]["duration"] == 0 and not self.scheduled[j]:
                # 시작시간 0에 스케줄
                self.schedule[j] = (0, 0)
                self.scheduled[j] = True

    def reset(self):
        # 환경 상태 초기화
        self.res_usage = np.zeros((self.num_resources, self.horizon), dtype=int)
        self.schedule = {}
        self.scheduled = {j: False for j in self.job_ids}
        self._auto_schedule_start()
        return self._get_obs()

    def _get_obs(self):
        """
        각 작업을 노드로 하는 그래프를 구성합니다.
        노드 feature: [duration, req_R1, req_R2, req_R3, req_R4, scheduled_flag, available_flag]
        available_flag: 해당 작업의 모든 선행 작업이 스케줄되었으면 1.
        edge_index: 선행→후행 에지 목록 (양방향 메시지 전달도 가능)
        mask: 에이전트가 선택할 수 있는(즉, 아직 스케줄되지 않고 available한) 작업 표시 (bool)
        """
        num_nodes = self.num_jobs
        x = np.zeros((num_nodes, 7), dtype=np.float32)
        mask = np.zeros(num_nodes, dtype=bool)

        for i, j in enumerate(self.job_ids):
            duration = self.jobs[j]["duration"]
            req = self.jobs[j]["resources"]
            scheduled_flag = 1.0 if self.scheduled[j] else 0.0
            avail = 0.0
            if (not self.scheduled[j]) and self._is_available(j):
                avail = 1.0
                mask[i] = True
            x[i, :] = np.array(
                [duration] + req + [scheduled_flag, avail]
            )  # 마지막 0.0 자리(플레이스홀더)

        # edge_index: job i -> successor (인덱스는 job_ids 내 위치)
        edge_list = []
        for i, j in enumerate(self.job_ids):
            # 현재 작업의 후행 작업 목록 (만약 precedence 정보가 없으면 빈 리스트)
            succs = self.instance["precedence"].get(j, [])
            for s in succs:
                if s in self.job_ids:
                    j_idx = self.job_ids.index(s)
                    edge_list.append([i, j_idx])
        if len(edge_list) > 0:
            edge_index = np.array(edge_list).T  # shape [2, num_edges]
        else:
            edge_index = np.empty((2, 0), dtype=np.int64)

        # dict 반환 (추후 torch_geometric.data.Data로 변환하여 사용)
        obs = {
            "x": torch.tensor(x, dtype=torch.float),
            "edge_index": torch.tensor(edge_index, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }
        return obs

    def _is_available(self, job_id):
        """모든 선행 작업이 스케줄되었으면 available"""
        preds = self.jobs[job_id].get("predecessors", [])
        for p in preds:
            if not self.scheduled.get(p, False):
                return False
        return True

    def _earliest_start_time(self, job_id):
        """job_id의 선행 작업들의 finish 시간 중 최댓값"""
        preds = self.jobs[job_id].get("predecessors", [])
        if not preds:
            return 0
        t = 0
        for p in preds:
            if p in self.schedule:
                t = max(t, self.schedule[p][1])
            else:
                t = max(t, 0)
        return t

    def _find_feasible_start(self, job_id):
        """
        job_id를 스케줄하기 위한 가능한 가장 빠른 시작 시간을 찾습니다.
        자원 제약을 만족하는 시점을 선형검색합니다.
        """
        duration = self.jobs[job_id]["duration"]
        req = self.jobs[job_id]["resources"]
        t0 = self._earliest_start_time(job_id)
        t = t0
        while t + duration <= self.horizon:
            feasible = True
            for r in range(self.num_resources):
                # t부터 t+duration 구간의 자원 사용량 확인
                if np.any(
                    self.res_usage[r, t : t + duration] + req[r]
                    > self.resources_capacity[r]
                ):
                    feasible = False
                    break
            if feasible:
                return t
            t += 1
        # horizon 내에 feasible하지 않으면 horizon 반환(비현실적이지만)
        return self.horizon

    def _update_resource_usage(self, job_id, start, finish):
        """스케줄된 job_id의 자원 사용량 업데이트"""
        req = self.jobs[job_id]["resources"]
        for r in range(self.num_resources):
            self.res_usage[r, start:finish] += req[r]

    def step(self, action):
        """
        action: observation의 mask에 해당하는 available 작업들 중 하나를 선택한 인덱스.
        내부적으로 실제 job id를 매핑하여 스케줄합니다.
        """
        obs = self._get_obs()
        available_idx = torch.nonzero(obs["mask"]).squeeze().tolist()
        if isinstance(available_idx, int):
            available_idx = [available_idx]
        if len(available_idx) == 0:
            # 더 이상 선택할 작업이 없으면 에피소드 종료
            done = True
            # makespan = 스케줄된 작업 중 최대 finish 시간
            makespan = (
                max([finish for (_, finish) in self.schedule.values()])
                if self.schedule
                else 0
            )
            return obs, -makespan, done, {}
        if action not in available_idx:
            raise ValueError("Invalid action index")
        # 선택한 작업의 실제 job id
        job_id = self.job_ids[action]
        # 작업을 스케줄
        start_time = self._find_feasible_start(job_id)
        finish_time = start_time + self.jobs[job_id]["duration"]
        self._update_resource_usage(job_id, start_time, finish_time)
        self.schedule[job_id] = (start_time, finish_time)
        self.scheduled[job_id] = True

        # 보상은 에피소드 종료 시에만 -makespan (sparse reward)
        done = all(self.scheduled.values())
        if done:
            makespan = max([finish for (_, finish) in self.schedule.values()])
            reward = -makespan
        else:
            reward = 0.0
        next_obs = self._get_obs()
        return next_obs, reward, done, {}

    def render(self, mode="human"):
        # 간단히 현재 스케줄 상태와 자원 사용량 출력
        print("Scheduled jobs:")
        for j in self.job_ids:
            if self.scheduled[j]:
                print(f"  Job {j}: {self.schedule[j]}")
        print("Resource usage (첫 50 타임스텝):")
        print(self.res_usage[:, :50])
