import torch
import os
import sys
import pandas as pd
from tqdm import tqdm
from ortools.sat.python import cp_model

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.config_loader import load_configuration_from_file
from common.data_classes import Battery, Load, LDO, BuckConverter
from or_tools_solver.core import create_solver_model
from transformer_solver.definitions import FEATURE_INDEX, NODE_TYPE_LOAD, NODE_TYPE_IC

def tensor_to_ortools_instance(batch_idx, td, config_data):
    """
    TensorDict의 특정 배치(batch_idx) 데이터를 OR-Tools용 객체들로 변환합니다.
    """
    # 1. 원본 Config에서 정적 정보(이름, 시퀀스 등) 가져오기
    base_battery, base_ics, base_loads, constraints = config_data
    
    # 2. 텐서 데이터 추출
    nodes = td["nodes"][batch_idx] # (N_max, F)
    max_sleep_current = td["scalar_prompt_features"][batch_idx, 1].item()
    
    # 제약조건 업데이트 (텐서의 값이 우선)
    constraints = constraints.copy()
    constraints['max_sleep_current'] = max_sleep_current
    
    # 3. 객체 복원
    
    # [Battery] - 항상 0번 노드라고 가정
    # [수정] Battery 생성 시 'capacity_mah' 인자 추가 (base_battery에서 가져옴)
    battery = Battery(
        name=base_battery.name,
        voltage_min=base_battery.voltage_min,
        voltage_max=base_battery.voltage_max,
        capacity_mah=base_battery.capacity_mah  # 👈 추가된 부분
    )
    
    # [Loads]
    loads = []
    load_start_idx = 1
    node_types = nodes[:, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
    
    for i, base_load in enumerate(base_loads):
        idx = load_start_idx + i
        if node_types[idx] != NODE_TYPE_LOAD:
            continue
            
        feat = nodes[idx]
        load_voltage = feat[FEATURE_INDEX["vin_min"]].item()
        
        # 객체 생성
        new_load = Load(
            name=base_load.name,
            voltage_typical=load_voltage, # 👈 수정됨
            current_active=feat[FEATURE_INDEX["current_active"]].item(),
            current_sleep=feat[FEATURE_INDEX["current_sleep"]].item(),
            voltage_req_min=load_voltage * 0.95, # 오차 범위 (필요시 조정)
            voltage_req_max=load_voltage * 1.05,
            independent_rail_type=base_load.independent_rail_type,
            always_on_in_sleep=base_load.always_on_in_sleep
        )
        loads.append(new_load)

    # [Candidate ICs]
    candidate_ics = []
    ic_indices = torch.where(node_types == NODE_TYPE_IC)[0]
    
    for idx in ic_indices:
        feat = nodes[idx]
        
        # 공급망 이슈(품절) 체크: is_template이 0.5 미만이면 품절로 간주하고 제외
        if feat[FEATURE_INDEX["is_template"]].item() < 0.5:
            continue
            
        # 텐서에서 스펙 읽기
        specs = {
            "name": f"IC_Node_{idx.item()}",
            "vin_min": feat[FEATURE_INDEX["vin_min"]].item(),
            "vin_max": feat[FEATURE_INDEX["vin_max"]].item(),
            "vout_min": feat[FEATURE_INDEX["vout_min"]].item(),
            "vout_max": feat[FEATURE_INDEX["vout_max"]].item(),
            "original_i_limit": feat[FEATURE_INDEX["i_limit"]].item(),
            "i_limit": feat[FEATURE_INDEX["i_limit"]].item(),
            "cost": feat[FEATURE_INDEX["cost"]].item(),
            "operating_current": feat[FEATURE_INDEX["op_current"]].item(),
            "quiescent_current": feat[FEATURE_INDEX["quiescent_current"]].item(),
            "shutdown_current": feat[FEATURE_INDEX["shutdown_current"]].item(),
            "theta_ja": feat[FEATURE_INDEX["theta_ja"]].item(),
            "t_junction_max": int(feat[FEATURE_INDEX["t_junction_max"]].item()),
            "efficiency_active": feat[FEATURE_INDEX["efficiency_active"]].item(),
            "efficiency_sleep": feat[FEATURE_INDEX["efficiency_sleep"]].item(),
            
            # ✅ [핵심 수정] vin 필드 초기화 (0.0 방지)
            # OR-Tools Solver는 Topology 결정 전 전류 계산 시 self.vin을 참조함.
            # 0.0이면 Buck Converter 계산 시 Overflow 발생하므로 vin_min으로 설정.
            "vin": feat[FEATURE_INDEX["vin_min"]].item() 
        }
        ic_type_idx = feat[FEATURE_INDEX["ic_type_idx"]].item()
        
        # Buck/LDO 구분 생성
        if abs(ic_type_idx - 2.0) < 0.1: # Buck
            ic_obj = BuckConverter(**specs)
        else: # LDO
            specs["v_dropout"] = 0.0 # LDO 필수 인자 추가 (데이터셋에 없으면 0.0 가정)
            ic_obj = LDO(**specs)
            
        candidate_ics.append(ic_obj)
        
    return battery, loads, candidate_ics, constraints, {}

def evaluate_ortools(pt_file_path, config_file_path, output_csv, time_limit=10.0):
    """
    .pt 데이터셋의 모든 문제에 대해 OR-Tools 솔버를 실행하고 평가합니다.
    """
    print(f"🚀 Evaluating OR-Tools on {pt_file_path}")
    print(f"   - Config Base: {config_file_path}")
    print(f"   - Time Limit per Instance: {time_limit}s")
    
    # 1. 데이터 로드 [수정: weights_only=False 추가]
    td_dataset = torch.load(pt_file_path, weights_only=False)
    num_instances = td_dataset.shape[0]
    
    # 2. Config 로드 (템플릿용)
    config_data = load_configuration_from_file(config_file_path)
    
    results = []
    
    # tqdm으로 진행 상황 표시
    for i in tqdm(range(num_instances), desc="Solving"):
        # 3. 문제 복원
        battery, loads, candidate_ics, constraints, ic_groups = \
            tensor_to_ortools_instance(i, td_dataset, config_data)
            
        # 4. 모델 생성
        model, edges, ic_is_used = create_solver_model(
            candidate_ics, loads, battery, constraints, ic_groups
        )
        
        # 5. 솔버 설정 및 실행
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.log_search_progress = False # 로그 끄기
        
        status = solver.Solve(model)
        
        # 6. 결과 기록
        is_feasible = (status in [cp_model.OPTIMAL, cp_model.FEASIBLE])
        # OR-Tools 모델은 비용을 정수(scaling factor 10000 등)로 다룰 수 있으므로 확인 필요.
        # 여기서는 core.py 구현에 따라 ObjectiveValue가 scale된 값이라고 가정하고 10000.0으로 나눔.
        # (만약 core.py에서 scaling을 안 했다면 그대로 사용)
        cost = solver.ObjectiveValue() / 10000.0 if is_feasible else float('inf')
        
        results.append({
            "instance_idx": i,
            "feasible": is_feasible,
            "cost": cost,
            "status": solver.StatusName(status),
            "wall_time": solver.WallTime()
        })
        
    # 7. CSV 저장 및 요약
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    feas_rate = df["feasible"].mean() * 100
    avg_cost = df[df["feasible"]]["cost"].mean()
    
    print(f"\n📊 Evaluation Result ({pt_file_path})")
    print(f"   - Feasibility Rate: {feas_rate:.2f}%")
    print(f"   - Avg Cost (Valid): ${avg_cost:.4f}")
    print(f"   - Saved to: {output_csv}")


if __name__ == "__main__":
    # 설정
    CONFIG_PATH = "configs/config_TII.json"
    
    # 1. Clean 데이터셋 평가
    if os.path.exists("validation_data/val_set_TII_100_clean.pt"):
        evaluate_ortools(
            "validation_data/val_set_TII_100_clean.pt",
            CONFIG_PATH,
            "validation_data/ortools_result_clean.csv",
            time_limit=10.0 # 문제당 제한시간
        )
        
    # 2. Crisis 데이터셋 평가
    if os.path.exists("validation_data/val_set_TII_100_crisis.pt"):
        evaluate_ortools(
            "validation_data/val_set_TII_100_crisis.pt",
            CONFIG_PATH,
            "validation_data/ortools_result_crisis.csv",
            time_limit=10.0
        )