import torch
import numpy as np
import time
from tqdm import tqdm
from typing import Dict, Any, Tuple
from tensordict import TensorDict

# 사용자 프로젝트 모듈 임포트
from transformer_solver.solver_env import PocatEnv, FAILURE_PENALTY

class PocatEvaluator:
    def __init__(self, env: PocatEnv, model: torch.nn.Module, device: str):
        self.env = env
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(device)

    def evaluate(self, 
                 dataset: TensorDict, 
                 batch_size: int = 16, 
                 decode_type: str = "greedy",
                 pomo_sampling: bool = True) -> Dict[str, Any]:
        """
        데이터셋에 대해 평가를 수행하고 통계를 반환합니다.
        
        Args:
            dataset: 평가할 문제들이 담긴 TensorDict (전체 데이터)
            batch_size: 한 번에 처리할 배치 크기
            decode_type: 'greedy' or 'sampling'
            pomo_sampling: True일 경우 POMO(Multi-start) 적용, False면 단일 실행
        """
        
        # 통계 저장을 위한 리스트
        stats = {
            "total_instances": 0,
            "feasible_instances": 0,  # 구조적/물리적으로 성공한 케이스
            "optimal_rewards": [],    # 각 배치의 Best Reward
            "bom_costs": [],          # 성공한 케이스의 BOM Cost
            "sleep_costs": [],        # 성공한 케이스의 Sleep Penalty Cost
            "inference_times": [],
            "avg_starts": 0,          # 평균 POMO 시도 횟수
        }

        # 데이터셋 분할 (Manual Batching using TensorDict)
        total_items = dataset.batch_size[0]
        num_batches = (total_items + batch_size - 1) // batch_size
        
        print(f"🚀 Starting Evaluation: {total_items} instances ({num_batches} batches)")
        
        for i in tqdm(range(num_batches), desc="Evaluating"):
            # 1. 배치 슬라이싱
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_items)
            # clone()하여 원본 보존 및 디바이스 이동
            batch_td = dataset[start_idx:end_idx].clone().to(self.device)
            current_batch_size = end_idx - start_idx
            
            # 2. POMO 설정 확인 (시작 노드 개수 확인)
            # 환경에서 가능한 시작점(Load) 개수를 가져옴
            num_starts, _ = self.env.select_start_nodes(batch_td)
            
            if not pomo_sampling:
                num_starts = 1 # 강제 단일 실행 (필요 시 로직 수정 필요)

            start_time = time.time()
            
            with torch.no_grad():
                # 3. 모델 추론 (PocatModel.forward 내부에서 POMO 확장 및 루프 처리)
                # model.forward()는 {reward, actions, bom_cost, sleep_cost...} 반환
                # 반환된 텐서의 크기: (Batch_Size * Num_Starts, ...)
                result = self.model(
                    batch_td, 
                    self.env, 
                    decode_type=decode_type,
                    return_final_td=False # 메모리 절약을 위해 False
                )
                
            elapsed = time.time() - start_time
            stats["inference_times"].append(elapsed)

            # 4. 결과 분석 및 Best-of-N 선정 (핵심)
            
            # (B * N_starts, 1) -> (B, N_starts)
            flat_rewards = result["reward"].view(current_batch_size, num_starts)
            flat_bom = result["bom_cost"].view(current_batch_size, num_starts)
            flat_sleep = result["sleep_cost"].view(current_batch_size, num_starts)

            # 각 문제(Instance)별로 가장 높은 보상을 받은 Trajectory 선택
            best_rewards, best_indices = flat_rewards.max(dim=1) # (B,)
            
            # Best 인덱스에 해당하는 Cost 추출
            # gather를 위해 차원 맞춤: (B, 1)
            best_indices = best_indices.unsqueeze(1)
            best_bom = flat_bom.gather(1, best_indices).squeeze(1)
            best_sleep = flat_sleep.gather(1, best_indices).squeeze(1)

            # 5. 통계 집계
            for b in range(current_batch_size):
                r = best_rewards[b].item()
                bom = best_bom[b].item()
                sleep = best_sleep[b].item()
                
                stats["total_instances"] += 1
                stats["optimal_rewards"].append(r)
                
                # 성공 기준: 실패 페널티보다 보상이 커야 함
                # (환경 설정의 FAILURE_PENALTY = -20000.0)
                is_feasible = (r > FAILURE_PENALTY * 0.5) # 여유 있게 절반 이상이면 성공 간주
                
                if is_feasible:
                    stats["feasible_instances"] += 1
                    stats["bom_costs"].append(bom)
                    stats["sleep_costs"].append(sleep)
            
            stats["avg_starts"] += num_starts

        # 6. 최종 요약
        stats["avg_starts"] /= num_batches
        feasibility_rate = (stats["feasible_instances"] / stats["total_instances"]) * 100
        avg_reward = np.mean(stats["optimal_rewards"])
        avg_bom = np.mean(stats["bom_costs"]) if stats["bom_costs"] else 0.0
        avg_sleep = np.mean(stats["sleep_costs"]) if stats["sleep_costs"] else 0.0
        avg_time = np.mean(stats["inference_times"])

        print("\n" + "="*50)
        print(f"📊 Evaluation Summary (N={total_items})")
        print("="*50)
        print(f"✅ Feasibility Rate : {feasibility_rate:.2f}% ({stats['feasible_instances']}/{stats['total_instances']})")
        print(f"🏆 Average Reward   : {avg_reward:.4f}")
        print(f"💰 Avg BOM Cost     : ${avg_bom:.4f} (Valid Only)")
        print(f"⚡ Avg Sleep Penalty: {avg_sleep:.4f} (Valid Only)")
        print(f"⏱️ Avg Inference    : {avg_time:.4f} sec/batch")
        print(f"🔄 POMO Starts      : {stats['avg_starts']:.1f}")
        print("="*50)
        
        return stats