import re


def parse_rcpsp_instance(data_str):
    """
    주어진 텍스트 데이터를 파싱하여 RCPSP 인스턴스를 구성합니다.
    반환 dict에는
      - jobs: { job_id: { 'duration': int, 'resources': [R1,R2,R3,R4] } }
      - precedence: { job_id: [successor_job_ids] }
      - resources: [R1_capacity, R2_capacity, R3_capacity, R4_capacity]
      - horizon: int (최대시간)
    """
    jobs = {}
    precedence = {}
    resources = None
    horizon = None

    # 1) RESOURCEAVAILABILITIES 파싱
    match = re.search(r"RESOURCEAVAILABILITIES:\s*([\s\S]+?)\n\s*[*]+", data_str)
    if match:
        lines = match.group(1).strip().splitlines()
        # 예: "  R 1  R 2  R 3  R 4" 다음 줄: "   12   13    4   12"
        if len(lines) >= 2:
            cap_line = lines[1].strip()
            parts = cap_line.split()
            resources = list(map(int, parts))
    else:
        raise ValueError("RESOURCEAVAILABILITIES 섹션을 찾지 못했습니다.")

    # 2) PRECEDENCE RELATIONS 파싱
    match = re.search(r"PRECEDENCE RELATIONS:\s*([\s\S]+?)\n\s*[*]+", data_str)
    if match:
        lines = match.group(1).strip().splitlines()[1:]  # 첫 줄은 헤더
        for line in lines:
            # line 예: "   1        1          3           2   3   4"
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            job_id = int(parts[0])
            num_successors = int(parts[2])
            succ = list(map(int, parts[3 : 3 + num_successors]))
            precedence[job_id] = succ
    else:
        raise ValueError("PRECEDENCE RELATIONS 섹션을 찾지 못했습니다.")

    # 3) REQUESTS/DURATIONS 파싱 (각 작업의 duration과 자원소요량)
    match = re.search(r"REQUESTS/DURATIONS:\s*([\s\S]+?)\n\s*[*]+", data_str)
    if match:
        lines = match.group(1).strip().splitlines()[1:]  # 첫 줄은 헤더
        for line in lines:
            # line 예: "  2      1     8       4    0    0    0"
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            job_id = int(parts[0])
            mode = int(parts[1])
            duration = int(parts[2])
            reqs = list(map(int, parts[3:7]))
            jobs[job_id] = {"duration": duration, "resources": reqs, "mode": mode}
    else:
        raise ValueError("REQUESTS/DURATIONS 섹션을 찾지 못했습니다.")

    # 4) horizon 파싱 (최대 시간; 파일 상단의 "horizon" 항목)
    match = re.search(r"horizon\s*:\s*(\d+)", data_str)
    if match:
        horizon = int(match.group(1))
    else:
        horizon = 200  # 기본값

    # 5) 각 작업의 선행(predecessor) 정보 생성 (후행(successor) 정보로부터 역산)
    # jobs 번호는 1부터 32라고 가정
    for j in jobs.keys():
        jobs[j]["predecessors"] = []
    for pred, succ_list in precedence.items():
        for s in succ_list:
            if s in jobs:
                jobs[s].setdefault("predecessors", []).append(pred)
            else:
                # dummy 작업 (예: 마지막 작업)이 없는 경우
                jobs[s] = {
                    "duration": 0,
                    "resources": [0] * len(resources),
                    "predecessors": [pred],
                    "mode": 1,
                }
    return {
        "jobs": jobs,
        "precedence": precedence,
        "resources": resources,
        "horizon": horizon,
    }
