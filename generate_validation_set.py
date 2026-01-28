import torch
import os
import random
import numpy as np
import sys

# 프로젝트 루트 경로 추가 (필요시)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformer_solver.env_generator import PocatGenerator
from transformer_solver.definitions import FEATURE_INDEX, NODE_TYPE_LOAD, NODE_TYPE_IC

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_validation_set(
    config_path: str,
    output_path: str,
    num_instances: int = 100,
    n_max: int = 500,
    seed: int = 42,
    supply_chain_prob: float = 0.0,
    desc: str = "Validation"
):
    print(f"\n⚙️ Generating {desc} set with Seed {seed}...")
    
    set_seed(seed)
    generator = PocatGenerator(config_file_path=config_path, N_max=n_max)

    try:
        val_dataset = generator.generate_random_batch(
            batch_size=num_instances, 
            device="cpu",
            supply_chain_prob=supply_chain_prob
        )
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return

    # -------------------------------------------------------------------------
    # 2. 메타데이터 생성 (config 내부의 딕셔너리 리스트 참조)
    # -------------------------------------------------------------------------
    print("📝 Attaching Original Metadata...")
    
    # [핵심 수정] Generator는 템플릿을 self.config.available_ics (List[dict])에 저장함
    if hasattr(generator, 'config') and hasattr(generator.config, 'available_ics'):
        original_candidates_dicts = generator.config.available_ics
        print(f"   -> Found template dicts in 'generator.config.available_ics' (Size: {len(original_candidates_dicts)})")
    else:
        print("❌ FATAL: Cannot find 'available_ics' in generator.config!")
        return

    metadata_list = []
    nodes_batch = val_dataset["nodes"]
    
    for i in range(num_instances):
        nodes = nodes_batch[i]
        node_types = nodes[:, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        
        # --- (A) Load ---
        load_meta = []
        load_indices = torch.where(node_types == NODE_TYPE_LOAD)[0]
        for k, idx in enumerate(load_indices):
            feat = nodes[idx]
            v = feat[FEATURE_INDEX["vin_min"]].item()
            i_act = feat[FEATURE_INDEX["current_active"]].item()
            load_meta.append({
                "idx": idx.item(),
                "name": f"Load_{k:02d}_{v:.1f}V_{i_act:.2f}A",
                "type": "Load"
            })

        # --- (B) IC (원본 이름 복원) ---
        ic_meta = []
        ic_indices = torch.where(node_types == NODE_TYPE_IC)[0]
        
        for k, idx in enumerate(ic_indices):
            feat = nodes[idx]
            # 품절(Padding) 체크
            if feat[FEATURE_INDEX["is_template"]].item() < 0.5: continue 
            
            # 인덱스로 원본 정보(딕셔너리) 찾기
            if k < len(original_candidates_dicts):
                ic_dict = original_candidates_dicts[k]
                
                # 딕셔너리 키로 접근!
                real_name = ic_dict.get('name', f"Unknown_IC_{k}")
                
                # 타입 정보 확인
                # (보통 'type' 키에 'Buck' 또는 'LDO' 문자열이 저장됨)
                raw_type = ic_dict.get('type', 'Buck') 
                if 'Buck' in str(raw_type):
                    real_type = "Buck"
                else:
                    real_type = "LDO"
            else:
                real_name = f"Unknown_IC_{k}"
                real_type = "Buck"

            ic_meta.append({
                "idx": idx.item(),
                "name": real_name, # 예: TPS62130
                "type": real_type
            })
            
        metadata_list.append({"loads": load_meta, "ics": ic_meta})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "tensor_data": val_dataset,
        "metadata": metadata_list
    }, output_path)
    
    print(f"✅ Saved to '{output_path}' (Metadata Count: {len(metadata_list)})")

if __name__ == "__main__":
    CONFIG_FILE = "configs/config_TII.json"
    BASE_OUTPUT_PATH = "validation_data/val_set_TII_100.pt"
    
    # 1. Clean
    generate_validation_set(
        CONFIG_FILE, BASE_OUTPUT_PATH.replace(".pt", "_clean.pt"), 
        100, 500, 42, 0.0, "[Clean Val]"
    )
    # 2. Crisis
    generate_validation_set(
        CONFIG_FILE, BASE_OUTPUT_PATH.replace(".pt", "_crisis.pt"), 
        100, 500, 42, 0.05, "[Crisis Val]"
    )