import torch
import sys
import os

# 프로젝트 경로 설정 (실행 위치에 따라 조정 필요할 수 있음)
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformer_solver.solver_env import PocatEnv
from transformer_solver.definitions import FEATURE_INDEX
from transformer_solver.debug_env import run_interactive_debugger, get_node_name

# --- Monkey Patching: 내부 함수 가로채기 ---
original_method = PocatEnv._get_thermal_current_mask

def patched_get_thermal_current_mask(self, td, b_idx_node, child_nodes, base_valid_parents):
    # 1. 원래 로직 실행
    mask = original_method(self, td, b_idx_node, child_nodes, base_valid_parents)
    
    # 2. 만약 후보가 있었는데(base > 0) 결과가 전멸(mask == 0)이라면 원인 분석
    if mask.sum() == 0 and base_valid_parents.sum() > 0:
        print("\n" + "="*60)
        print("🚨 [PATCH DEBUG] 모든 후보가 'Current/Thermal' 체크에서 탈락했습니다!")
        print(f"   - 1차 통과(전압 등) 후보 수: {base_valid_parents.sum().item()}개")
        print("   - 모든 후보에 대한 상세 분석을 시작합니다:\n")
        
        # 후보 인덱스 추출 (배치는 1개라고 가정)
        b_idx_in_batch = 0 
        candidate_indices = torch.where(base_valid_parents[b_idx_in_batch])[0]
        
        child_idx = child_nodes[b_idx_in_batch].item()
        child_node_feat = td["nodes"][b_idx_in_batch, child_idx]
        
        # Load 정보는 공통이므로 한 번만 추출
        load_current = child_node_feat[FEATURE_INDEX["current_active"]].item()
        
        # 설정값 로드
        margin_I = float(self.generator.config.constraints.get("current_margin", 0.0))
        margin_T = float(self.generator.config.constraints.get("thermal_margin_percent", 0.0))
        ambient = self.generator.config.constraints.get("ambient_temperature", 25.0)

        # 🔄 모든 후보 순회
        for i, target_p_idx_tensor in enumerate(candidate_indices):
            target_p_idx = target_p_idx_tensor.item()
            
            # 텐서에서 값 추출
            parent_node = td["nodes"][b_idx_in_batch, target_p_idx]
            
            # 이름 조회
            p_name = f"Node_idx_{target_p_idx}"
            if hasattr(self.generator, 'config') and target_p_idx < len(self.generator.config.node_names):
                 p_name = self.generator.config.node_names[target_p_idx]
            
            print(f"--- [후보 {i+1}/{len(candidate_indices)}] {p_name} ---")

            # --- A. 전류 제한 체크 ---
            i_limit_raw = parent_node[FEATURE_INDEX["i_limit"]].item()
            i_limit_derated = i_limit_raw * (1.0 - margin_I)
            
            current_status = "✅ PASS"
            if load_current > i_limit_derated:
                current_status = f"❌ FAIL (Load {load_current:.3f}A > Limit {i_limit_derated:.3f}A)"
            
            # --- B. 발열(Thermal) 체크 ---
            t_max_raw = parent_node[FEATURE_INDEX["t_junction_max"]].item()
            t_max_derated = t_max_raw * (1.0 - margin_T)
            theta = parent_node[FEATURE_INDEX["theta_ja"]].item()
            
            # 예상 손실 계산 (약식)
            vin = parent_node[FEATURE_INDEX["vin_min"]].item()
            vout = parent_node[FEATURE_INDEX["vout_min"]].item()
            ic_type = parent_node[FEATURE_INDEX["ic_type_idx"]].item()
            
            est_p_loss = 0.0
            type_str = "Unknown"
            
            if ic_type == 1.0: # LDO
                type_str = "LDO"
                # LDO 단순 계산: (Vin - Vout) * I
                est_p_loss = (vin - vout) * load_current
            elif ic_type == 2.0: # Buck
                type_str = "Buck"
                # Buck 단순 계산: P_out * (1/Eff - 1)
                eff = parent_node[FEATURE_INDEX["efficiency_active"]].item()
                if eff <= 0: eff = 0.9
                p_out = vout * load_current
                est_p_loss = p_out * (1/eff - 1)
            
            est_temp = ambient + est_p_loss * theta
            
            thermal_status = "✅ PASS"
            if est_temp > t_max_derated:
                thermal_status = f"❌ FAIL (Temp {est_temp:.1f}C > Max {t_max_derated:.1f}C)"

            # 결과 출력
            print(f"  • Type: {type_str} | Theta: {theta:.1f} | Vin: {vin:.1f}V")
            print(f"  • Current Check: {current_status}")
            print(f"  • Thermal Check: {thermal_status}")
            print("-" * 40)
            
        print("="*60 + "\n")
            
    return mask

# Monkey Patch 적용
PocatEnv._get_thermal_current_mask = patched_get_thermal_current_mask

if __name__ == "__main__":
    # 설정 파일 경로와 N_max는 필요에 따라 수정하세요
    config_file = "configs/config_TII.json"
    n_max = 600
    
    print(f"🔧 Debug Patch Loaded (Full Scan Mode). Running Debugger on {config_file}...")
    run_interactive_debugger(config_file, n_max)