import torch
import os
from transformer_solver.model import PocatModel
from transformer_solver.solver_env import PocatEnv
from evaluation import PocatEvaluator # 위에서 작성한 클래스

def run_eval():
    # --- 설정 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    N_MAX = 400 # 학습된 모델의 N_MAX와 일치해야 함
    BATCH_SIZE = 16
    
    # 경로 설정
    CONFIG_PATH = "configs/config_TII.json"
    MODEL_PATH = "checkpoints/best_model.pt" # 학습된 모델 경로
    VAL_DATA_PATH = "validation_data/val_set_TII_100.pt"

    print(f"🖥️ Using Device: {DEVICE}")

    # 1. 데이터셋 로드
    if not os.path.exists(VAL_DATA_PATH):
        print(f"⚠️ Validation data not found at {VAL_DATA_PATH}. Please run generate_test_data.py first.")
        return
    
    val_dataset = torch.load(VAL_DATA_PATH)
    print(f"📂 Loaded validation set: {len(val_dataset)} instances")

    # 2. 환경 및 모델 초기화
    # (모델 파라미터는 학습 시점과 동일하게 맞춰야 함)
    model_params = {
        "N_MAX": N_MAX,
        "embedding_dim": 128,
        "encoder_layer_num": 3,
        "qkv_dim": 16,
        "head_num": 8,
        "decoder_layer_num": 1, # [중요] model.py에 추가된 파라미터 확인
        "logit_clipping": 10
    }
    
    # 모델 생성
    model = PocatModel(**model_params)
    
    # 체크포인트 로드
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        # 만약 체크포인트가 state_dict만 가지고 있다면:
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"✅ Loaded model weights from {MODEL_PATH}")
    else:
        print(f"⚠️ Warning: Checkpoint not found at {MODEL_PATH}. Using random weights.")

    # 환경 생성 (PocatGenerator 내부 호출)
    generator_params = {
        "config_file_path": CONFIG_PATH,
        # N_max는 PocatEnv __init__에서 전달됨
    }
    env = PocatEnv(generator_params=generator_params, device=DEVICE, N_max=N_MAX)

    # 3. 평가 실행
    evaluator = PocatEvaluator(env, model, DEVICE)
    
    # Greedy Decoding (POMO 적용)
    print("\n[Test 1] Greedy Decoding with POMO")
    evaluator.evaluate(val_dataset, batch_size=BATCH_SIZE, decode_type="greedy", pomo_sampling=True)

    # (옵션) Sampling Decoding
    # print("\n[Test 2] Sampling Decoding (Temperature=1.0) with POMO")
    # evaluator.evaluate(val_dataset, batch_size=BATCH_SIZE, decode_type="sampling", pomo_sampling=True)

if __name__ == "__main__":
    run_eval()