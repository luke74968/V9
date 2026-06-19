# Copyright (c) 2025 Minuk Lee. All rights reserved.
# 
# This source code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# 
# For licensing terms, see the LICENSE file.
# Contact: minuklee@snu.ac.kr
# 
import os
import sys
import time
import yaml
import json
import random
import torch
import logging
import argparse
import torch.distributed as dist

# (PyTorch 2.0+ TensorFloat32 최적화)
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
    torch.set_float32_matmul_precision('high')

# --- 핵심 모듈 임포트 ---
# (이 파일들은 우리가 방금/앞으로 만들 파일들입니다)
from common.config_loader import load_configuration_from_file
from common.ic_preprocessor import expand_ic_templates, prune_dominated_ics
from .model import PocatModel
from .solver_env import PocatEnv
from .trainer import PocatTrainer
from .expert_dataset import ExpertReplayDataset # (Trainer가 사용할 수 있으므로 임포트)


def setup_logger(result_dir, rank=0):
    """
    로그 파일을 설정하고, 0번 GPU(메인)에서만 콘솔 출력을 하도록 설정합니다.
    """
    log_file = os.path.join(result_dir, 'log.txt')
    logging.basicConfig(
        filename=log_file, 
        format='%(asctime)-15s %(message)s', 
        level=logging.INFO
    )
    logger = logging.getLogger()
    
    # 0번 프로세스(메인)에서만 콘솔에 로그를 출력
    if rank <= 0:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)-15s %(message)s')
        console.setFormatter(formatter)
        logger.addHandler(console)
    return logger

def main(args):
    """
    메인 실행 함수 (DDP 설정, 환경/모델/트레이너 초기화)
    """
    
    # --- DDP (Distributed Data Parallel) 설정 ---
    args.local_rank = int(os.environ.get('LOCAL_RANK', -1))
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.ddp = args.world_size > 1
    if args.test_only and args.test_json:
        if not os.path.exists(args.config_file):
            print(f"ℹ️ Config file '{args.config_file}' not found. Auto-switching to '{args.test_json}'")
            args.config_file = args.test_json

    if args.ddp:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.local_rank}")
        if args.local_rank <= 0:
            args.log(f"🚀 DDP 모드 ({args.world_size} GPUs) 실행. 디바이스: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.log(f"🚀 단일 디바이스 모드 실행. 디바이스: {device}")
    # --- DDP 설정 완료 ---

    # --- 1. config.yaml에서 N_MAX 값 추출 ---
    try:
        n_max = int(args.model_params['N_MAX'])
        args.log(f"Config loaded: N_MAX set to {n_max}")
    except (AttributeError, KeyError, TypeError, ValueError):
        args.log("❌ CRITICAL: 'model_params: N_MAX:'를 config.yaml에서 찾을 수 없거나 유효하지 않습니다.")
        return

    # --- 2. PocatEnv (환경) 생성 ---
    # (N_max 값을 Generator와 Env 양쪽에 주입)
    env = PocatEnv(
        generator_params={
            "config_file_path": args.config_file,
        },
        device=device,
        N_max=n_max # Env가 Spec을 생성할 때 사용
    )

    # --- 3. PocatModel (모델) 생성 준비 ---
    # (args.model_params['N_MAX'] = n_max 는 이미 __main__에서 로드됨)
    
    # --- 4. PocatTrainer (트레이너) 생성 ---
    trainer = PocatTrainer(args, env, device)

    # --- 6. 메인 훈련 또는 테스트 실행 ---
    # [추가] 중요 파라미터에 대한 안전한 기본값 처리 (YAML에도 없고 CLI도 없으면 기본값 적용)
    if args.batch_size is None: args.batch_size = 64
    if args.num_pomo_samples is None: args.num_pomo_samples = 8
    
    # config.yaml의 'pomo_size'를 'args.pomo_size'로 로드하므로, 이를 Trainer가 쓸 수 있게 매핑
    if not hasattr(args, 'pomo_size') and args.num_pomo_samples is not None:
        args.pomo_size = args.num_pomo_samples

    if args.test_only:
        if args.local_rank <= 0: # 테스트는 0번 프로세스에서만
            if getattr(args, 'test_json', None):
                args.log(f"🔍 Loading custom test JSON: {args.test_json}")
                
                # 1. 원본 JSON 데이터 로드 (문제 정의)
                with open(args.test_json, 'r', encoding='utf-8') as f:
                    problem_data = json.load(f)
                
                if isinstance(problem_data, dict):
                    problem_data = [problem_data]

                # [수정] 불필요한 재계산 제거. Generator 내부의 템플릿 사용 (ic_list=None)
                custom_test_data = trainer.env.generator.preprocess_to_tensordict(
                    problem_data, 
                    ic_list=None, 
                    device=device
                )
                
                trainer.test(test_data_override=custom_test_data)
            else:
                trainer.test()
    else:
        trainer.run() # 훈련 (DDP/단일 모드 모두 실행)
    
    if args.ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # --- 실행 경로 및 설정 ---
    parser.add_argument("--config_file", type=str, default="configs/config.json", help="Path to POCAT config file")
    parser.add_argument("--config_yaml", type=str, default="configs/config.yaml", help="Path to model/training config YAML")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")

    # [수정] default=None으로 변경하여 config.yaml 값이 우선 적용되도록 함
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch_size (per GPU)")
    parser.add_argument("--num_pomo_samples", type=int, default=None, 
                        help="Number of POMO samples. (Matches 'pomo_size' in config)")

    # --- 추론(Test) / 모델 로드 ---
    parser.add_argument('--test_only', action='store_true', help="Only run test/inference")
    parser.add_argument('--load_path', type=str, default=None, help="Path to a saved model checkpoint (.pth)")
    parser.add_argument('--test_json', type=str, default=None, help="Path to a specific JSON problem file for testing")
    parser.add_argument("--test_num_pomo_samples", type=int, default=None, 
                        help="Number of POMO samples for testing. (Defaults to num_pomo_samples)")
    parser.add_argument('--decode_type', type=str, default='greedy', choices=['greedy', 'sampling'],
                        help="Decoding strategy for test mode: 'greedy' or 'sampling'.")

    # --- 로그 관련 ---
    parser.add_argument('--log_idx', type=int, default=0, help='Instance index to log (for POMO)')
    parser.add_argument('--log_mode', type=str, default='progress', choices=['progress', 'detail'],
                        help="Logging mode: 'progress' (pbar) or 'detail' (step-by-step).")

    args = parser.parse_args()

    if args.test_num_pomo_samples is None and args.num_pomo_samples is not None:
       args.test_num_pomo_samples = args.num_pomo_samples

    # --- 결과 저장 디렉토리 및 로거 설정 ---
    args.start_time = time.strftime("%Y-%m%d-%H%M%S", time.localtime())
    args.result_dir = os.path.join('result_transformer', args.start_time)
    
    # (DDP 랭크 0만 디렉토리 생성 시도)
    local_rank_init = int(os.environ.get('LOCAL_RANK', 0))
    #if local_rank_init <= 0:
    os.makedirs(args.result_dir, exist_ok=True)
        
    logger = setup_logger(args.result_dir, rank=local_rank_init)
    args.log = logger.info
    
    # --- YAML 설정 파일 로드 ---
    # (YAML 파일의 모든 설정을 args 객체에 병합)
    try:
        with open(args.config_yaml, "r", encoding="utf-8") as f:
            cfg_yaml = yaml.safe_load(f)
        for key, value in cfg_yaml.items():
            if not hasattr(args, key):
                setattr(args, key, value)
            # (명령줄 인자가 None/False일 때 YAML 값으로 덮어쓰기)
            elif getattr(args, key) is None or isinstance(getattr(args, key), bool):
                 setattr(args, key, value)
    except FileNotFoundError:
        logger.error(f"❌ CRITICAL: config.yaml 파일을 찾을 수 없습니다: {args.config_yaml}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ CRITICAL: config.yaml 파일 파싱 오류: {e}")
        sys.exit(1)


    # --- DDP 속성 및 시드 설정 (main 함수 호출 전) ---
    # (main 함수 내의 DDP 설정 로직을 여기서 미리 실행)
    args.world_size = int(os.environ.get('WORLD_SIZE', 1))
    args.ddp = args.world_size > 1
    
    seed = args.seed
    if args.ddp:
        seed += local_rank_init # DDP 랭크별 오프셋 추가
        
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    # np.random.seed(seed) # (numpy 사용하는 경우)

    # --- 실행 인자 로그 기록 ---
    if local_rank_init <= 0:
        args_dict_for_log = {k: v for k, v in vars(args).items() if k != 'log'}
        args.log("--- 🚀 실행 인자 (Args) ---")
        args.log(json.dumps(args_dict_for_log, indent=4, default=str))
        args.log("---------------------------")
        
    main(args)