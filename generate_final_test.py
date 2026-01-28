from generate_validation_set import generate_validation_set

# === 최종 테스트 설정 ===
CONFIG_FILE = "configs/config_TII.json"
BASE_OUTPUT_PATH = "test_data/test_set_final_1000.pt" # 폴더/파일명 구분
NUM_SAMPLES = 1000  # 100개 -> 1000개로 증량
TEST_SEED = 9999    # 검증용(42)과 다른 시드 사용!

if __name__ == "__main__":
    print("🔒 [FINAL TEST] 데이터셋 생성을 시작합니다...")
    
    # 1. Clean Test Set
    generate_validation_set(
        config_path=CONFIG_FILE,
        output_path=BASE_OUTPUT_PATH.replace(".pt", "_clean.pt"),
        num_instances=NUM_SAMPLES,
        n_max=1000,
        seed=TEST_SEED,
        supply_chain_prob=0.0,
        desc="[FINAL TEST - Clean]"
    )

    # 2. Crisis Test Set
    generate_validation_set(
        config_path=CONFIG_FILE,
        output_path=BASE_OUTPUT_PATH.replace(".pt", "_crisis.pt"),
        num_instances=NUM_SAMPLES,
        n_max=1000,
        seed=TEST_SEED,
        supply_chain_prob=0.05,
        desc="[FINAL TEST - Crisis]"
    )