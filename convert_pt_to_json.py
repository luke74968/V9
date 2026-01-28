# Copyright (c) 2025 Minuk Lee. All rights reserved.
# 
# This source code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# 
# For licensing terms, see the LICENSE file.
# Contact: minuklee@snu.ac.kr
# 
import torch
import os
import json
import sys
from transformer_solver.definitions import FEATURE_INDEX 

def R_val(value): return round(value, 4) if isinstance(value, float) else value
def R_current(value):
    if isinstance(value, float):
        if value == 0.0: return 0.0
        if abs(value) < 1e-4: return float(f"{value:.6e}")
        return round(value, 6)
    return value

def convert_pt_to_json(pt_file_path, config_file_path, output_dir):
    """
    PyTorch 텐서 파일(.pt)을 읽어 OR-Tools용 JSON 문제 파일들로 변환합니다.
    - Cost: 소수점 3자리 반올림
    - i_limit: 이중 감가 방지를 위해 원본 스펙 값으로 복원
    - Rail Type: 올바른 매핑으로 수정
    """
    print(f"📂 Extracting JSONs from {pt_file_path} -> {output_dir}...")
    
    try: 
        pack = torch.load(pt_file_path, weights_only=False) 
    except Exception as e:
        print(f"❌ 로드 실패: {e}")
        return

    td_dataset = pack["tensor_data"] 
    metadata_list = pack["metadata"] 
    
    with open(config_file_path, 'r', encoding='utf-8') as f:
        raw_base_config = json.load(f)
    
    # IC 이름으로 정적 스펙 조회를 위한 딕셔너리 (i_limit 복원용)
    ic_static_specs = {ic['name']: ic for ic in raw_base_config['available_ics']}
        
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(td_dataset.shape[0]):
        nodes = td_dataset["nodes"][i] 
        meta = metadata_list[i] 
        
        # Battery & Constraints
        battery_data = raw_base_config['battery'].copy() 
        constraints_data = raw_base_config['constraints'].copy() 
        constraints_data['max_sleep_current'] = R_val(td_dataset["scalar_prompt_features"][i, 1].item()) 

        # Loads
        new_loads = []
        for load_info in meta["loads"]: 
            idx = load_info["idx"]
            feat = nodes[idx]
            voltage = feat[FEATURE_INDEX["vin_min"]].item()
            
            # [수정] Rail Type 복원 로직 개선
            # 2.0 -> exclusive_path, 1.0 -> exclusive_supplier
            rail_val = feat[FEATURE_INDEX["independent_rail_type"]].item()
            if rail_val > 1.5:
                rail_type = "exclusive_path"
            elif rail_val > 0.5:
                rail_type = "exclusive_supplier"
            else:
                rail_type = None

            new_loads.append({
                "name": load_info["name"],
                "voltage_req_min": R_val(voltage * 0.95),  
                "voltage_req_max": R_val(voltage * 1.05),  
                "voltage_typical": R_val(voltage),
                "current_active": R_val(feat[FEATURE_INDEX["current_active"]].item()),
                "current_sleep": R_current(feat[FEATURE_INDEX["current_sleep"]].item()),
                "independent_rail_type": rail_type,
                "always_on_in_sleep": feat[FEATURE_INDEX["always_on_in_sleep"]].item() > 0.5
            })

        # ICs
        new_ics = []
        unique_check = set()
        
        for ic_info in meta["ics"]: 
            idx = ic_info["idx"]
            feat = nodes[idx]
            
            # PT 파일의 이름(display_name)이 'IC@Vin_Vout' 형태
            display_name = ic_info["name"] 
            
            if display_name in unique_check: continue
            unique_check.add(display_name)
            
            # Config 조회를 위해 @ 뒷부분 제거 (예: LT8638SEV@... -> LT8638SEV)
            base_name = display_name.split('@')[0]
            
            # Config 파일에서 정적 스펙 조회
            static_spec = ic_static_specs.get(base_name, {})
            
            # [수정] i_limit 복원 로직
            # 텐서의 i_limit은 이미 Derating된 값이므로, 이를 JSON에 쓰면 OR-Tools가 또 Derating을 수행함.
            # 따라서 원본 스펙의 i_limit (예: 4.0)을 가져와서 기록해야 함.
            original_i_limit = static_spec.get("i_limit")
            if original_i_limit is not None:
                final_i_limit = R_val(original_i_limit)
            else:
                # 스펙을 못 찾은 경우에만 텐서 값 사용 (대신 ThetaJA를 0으로 만들어 이중 감가 방지 필요)
                final_i_limit = R_val(feat[FEATURE_INDEX["i_limit"]].item())
            
            # [수정] ThetaJA 설정
            # 원본 스펙을 찾았다면 원본 ThetaJA 사용 (OR-Tools가 정상적으로 Derating 수행)
            # 못 찾아서 텐서값(Derated)을 썼다면 ThetaJA=0으로 설정하여 Derating 방지
            if original_i_limit is not None:
                 final_theta_ja = R_val(feat[FEATURE_INDEX["theta_ja"]].item())
            else:
                 final_theta_ja = 0.0

            # 공통 필드 구성
            ic_dict = {
                "type": ic_info["type"],
                "name": display_name,
                
                # Config에서 가져온 값 적용
                "is_fixed": static_spec.get("is_fixed", False),
                "min_fb_res": static_spec.get("min_fb_res", 0.0),
                
                "vin_min": R_val(feat[FEATURE_INDEX["vin_min"]].item()),
                "vin_max": R_val(feat[FEATURE_INDEX["vin_max"]].item()),
                "vout_min": R_val(feat[FEATURE_INDEX["vout_min"]].item()),
                "vout_max": R_val(feat[FEATURE_INDEX["vout_max"]].item()),
                
                # [적용] 복원된 i_limit 및 ThetaJA
                "i_limit": R_val(feat[FEATURE_INDEX["i_limit"]].item()), 
                "theta_ja": R_val(feat[FEATURE_INDEX["theta_ja"]].item()),
                
                # [수정] Cost 소수점 3자리 반올림
                "cost": round(feat[FEATURE_INDEX["cost"]].item(), 3),
                
                "shut_current": R_current(feat[FEATURE_INDEX["shutdown_current"]].item()),
                "t_junction_max": int(feat[FEATURE_INDEX["t_junction_max"]].item())
            }

            # 타입별 분기 처리
            if ic_dict["type"] == "LDO":
                ic_dict["v_dropout"] = static_spec.get("v_dropout", 0.0)
                ic_dict["op_current"] = R_current(feat[FEATURE_INDEX["op_current"]].item())
                ic_dict["q_current"] = R_current(feat[FEATURE_INDEX["quiescent_current"]].item())
            
            else: # Buck
                ic_dict["eff_op"] = R_val(feat[FEATURE_INDEX["efficiency_active"]].item())
                ic_dict["eff_sleep"] = R_val(feat[FEATURE_INDEX["efficiency_sleep"]].item())
                ic_dict["not_switching_current"] = R_current(feat[FEATURE_INDEX["quiescent_current"]].item())

            new_ics.append(ic_dict)

        with open(os.path.join(output_dir, f"problem_{i:03d}.json"), 'w') as f:
            json.dump({
                "battery": battery_data, "loads": new_loads, 
                "available_ics": new_ics, "constraints": constraints_data
            }, f, indent=2)

    print(f"✅ Extracted {td_dataset.shape[0]} JSON files with corrected static specs.")

if __name__ == "__main__":
    CONFIG_FILE = "configs/config_TII.json" 
    if os.path.exists("validation_data/val_set_TII_100_clean.pt"):
        convert_pt_to_json("validation_data/val_set_TII_100_clean.pt", CONFIG_FILE, "validation_data/json_clean")
    if os.path.exists("validation_data/val_set_TII_100_crisis.pt"):
        convert_pt_to_json("validation_data/val_set_TII_100_crisis.pt", CONFIG_FILE, "validation_data/json_crisis")