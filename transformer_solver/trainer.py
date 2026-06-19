# Copyright (c) 2025 Minuk Lee. All rights reserved.
# 
# This source code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# 
# For licensing terms, see the LICENSE file.
# Contact: minuklee@snu.ac.kr
# 
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader
from tensordict import TensorDict
# AMP 모듈을 torch.amp로 변경한다.
from torch.amp import GradScaler, autocast

from tqdm import tqdm
import os
import time
from datetime import datetime
import logging
from collections import defaultdict, Counter
import json

# --- 핵심 모듈 임포트 ---
from .model import PocatModel, PrecomputedCache, reshape_by_heads
from .solver_env import PocatEnv, BATTERY_NODE_IDX
from .expert_dataset import ExpertReplayDataset, expert_collate_fn
from .utils.common import TimeEstimator, clip_grad_norms, unbatchify, batchify

# --- 시각화 모듈 임포트 ---
from graphviz import Digraph
from common.data_classes import LDO, BuckConverter # (common)
from .definitions import FEATURE_INDEX, NODE_TYPE_LOAD, NODE_TYPE_IC, NODE_TYPE_BATTERY, NODE_TYPE_EMPTY

def update_progress(pbar, metrics, step):
    """ tqdm 진행률 표시줄을 업데이트합니다. """
    if pbar is None:
        return
    
    metrics_str = (
        f"Loss: {metrics['Loss']:.4f} "
        f"($Avg: {metrics['Avg Cost']:.2f}, $AvgMin: {metrics['Avg Min Batch']:.2f}, $Min: {metrics['Min Cost']:.2f})| " 
        f"Ent: {metrics['Entropy']:.4f} | "
        f"BOM ${metrics['Avg BOM']:.2f} + Sleep {metrics['Avg Sleep']:.1f}"
    )
    pbar.set_postfix_str(metrics_str, refresh=False)
    pbar.update(1)


def cal_model_size(model, log_func):
    """ 모델의 파라미터 및 버퍼 크기를 계산하여 로그에 기록합니다. """
    param_count = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    buffer_count = sum(b.nelement() for b in model.buffers())
    log_func(f'모델 파라미터 수: {param_count:,}')
    log_func(f'모델 버퍼 수: {buffer_count:,}')

class PocatTrainer:
    """
    PocatModel과 PocatEnv를 사용하여 훈련, 검증, 테스트를
    수행하는 메인 트레이너 클래스입니다. (A2C 기반)
    """
    
    def __init__(self, args, env: PocatEnv, device: str):
        self.args = args
        self.env = env
        self.is_ddp = args.ddp
        self.local_rank = args.local_rank
        self.device = device

        self.result_dir = args.result_dir
        self.log = args.log

        # --- 1. 모델 초기화 및 DDP 래핑 ---
        self.model = PocatModel(**args.model_params).to(self.device)
        
        if self.is_ddp:
            self.model = DDP(
                self.model, 
                device_ids=[self.local_rank], 
                find_unused_parameters=True # (모델은 모든 파라미터 사용)
            )
        
        if self.local_rank <= 0:
            cal_model_size(self.model, self.log)
        
        # --- 2. 옵티마이저 및 스케줄러 ---
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(args.optimizer_params['optimizer']['lr']),
            weight_decay=float(args.optimizer_params['optimizer'].get('weight_decay', 0)),
        )
        
        if args.optimizer_params['scheduler']['name'] == 'MultiStepLR':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=args.optimizer_params['scheduler']['milestones'],
                gamma=args.optimizer_params['scheduler']['gamma']
            )
        else:
            raise NotImplementedError
            
        self.start_epoch = 1


        # [AMP] Mixed Precision Scaler 초기화
        # config에 use_amp가 없으면 기본값 False
        self.use_amp = getattr(args, 'use_amp', False)
        self.scaler = GradScaler(enabled=self.use_amp)
        
        if self.local_rank <= 0 and self.use_amp:
            self.log("⚡ Mixed Precision (AMP) Training Enabled.")

        # --- 3. 모델 로드 (Checkpoint) ---
        if args.load_path is not None:
            self.log(f"모델 체크포인트 로드 중: {args.load_path}")
            try:
                checkpoint = torch.load(args.load_path, map_location=device, weights_only=False)
                
                # DDP/일반 모델 상태 호환 로드
                model_to_load = self.model.module if self.is_ddp else self.model
                model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=False)                

                if not args.test_only: # 훈련 재개 시
                    # [주의] 모델 구조가 바뀌었으므로(파라미터 수 감소), 
                    # 기존 옵티마이저 상태(Momentum 등)는 호환되지 않습니다.
                    # 따라서 옵티마이저는 로드하지 않고 새로 초기화하는 것이 안전합니다.
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except ValueError:
                        self.log("⚠️ 경고: 모델 구조 변경으로 인해 옵티마이저 상태 로드를 건너뜁니다. (새로운 옵티마이저로 시작)")
                    self.start_epoch = checkpoint['epoch'] + 1
                self.log("모델 로드 완료.")
            except Exception as e:
                self.log(f"❌ 모델 로드 실패: {e}. 무작위 초기화로 시작합니다.")

        self.time_estimator = TimeEstimator(log_fn=self.log)

        # --- 4. 검증(Evaluate)용 데이터셋 ---
        # [수정] Clean / Crisis 검증 데이터셋 로드
        self.val_datasets = {}
        self.best_eval_bom = float("inf") # Clean Set 기준 Best

        if self.local_rank <= 0:  # 0번 GPU에서만 로드
            val_base_path = "validation_data"
            clean_path = os.path.join(val_base_path, "val_set_TII_100_clean.pt")
            crisis_path = os.path.join(val_base_path, "val_set_TII_100_crisis.pt")
            
            def load_safe(path):
                if not os.path.exists(path): return None
                try:
                    # 1. 파일 로드
                    loaded = torch.load(path, weights_only=False)
                    
                    # 2. [핵심] 딕셔너리 포장("tensor_data")이 되어 있으면 내용물만 꺼냄
                    if isinstance(loaded, dict) and "tensor_data" in loaded:
                        data = loaded["tensor_data"]
                    else:
                        data = loaded # 구버전 파일 호환

                    # 3. CPU로 이동 (TensorDict인 경우)
                    if hasattr(data, "to"):
                        return data.to("cpu")
                    return data
                    
                except Exception as e:
                    print(f"⚠️ Validation Data Load Error ({path}): {e}")
                    return None

            # [Clean Set]
            if os.path.exists(clean_path):
                self.log(f"📂 Validation Clean Set 로드 중: {clean_path}")
                self.val_datasets["clean"] = load_safe(clean_path)  # CPU 보관
            else:
                self.log(f"⚠️ Clean Validation 파일을 찾을 수 없음: {clean_path}")

            # [Crisis Set]
            if os.path.exists(crisis_path):
                self.log(f"📂 Validation Crisis Set 로드 중: {crisis_path}")
                self.val_datasets["crisis"] = load_safe(crisis_path)
            else:
                self.log(f"⚠️ Crisis Validation 파일을 찾을 수 없음: {crisis_path}")

    def run(self):
        """ 메인 훈련 루프 (A2C) """
        args = self.args
        self.time_estimator.reset(self.start_epoch)
        
        if args.test_only:
            self.test()
            return

        for epoch in range(self.start_epoch, args.trainer_params['epochs'] + 1):
            if self.local_rank <= 0:
                self.log('=' * 60)
            
            self.model.train()
            
            # (DDP) DDP Sampler가 에폭마다 시드를 변경하도록 설정
            #if self.is_ddp and hasattr(self.env_dataset, 'sampler'):
            #    self.env_dataset.sampler.set_epoch(epoch)
            
            #  그래디언트 누적 스텝 설정 (기본값 1: 누적 안 함)
            accumulation_steps = args.trainer_params.get('gradient_accumulation_steps')
            if accumulation_steps is None:
                accumulation_steps = 1

            # 실제 루프 횟수는 (목표 업데이트 횟수 * 누적 스텝)으로 늘어남
            total_steps = args.trainer_params['train_step'] * accumulation_steps
            
            # (DDP) 0번 GPU에서만 tqdm 진행률 표시
            train_pbar = None
            if self.local_rank <= 0:
                train_pbar = tqdm(
                    total=total_steps,
                    desc=f"Epoch {epoch}",
                    dynamic_ncols=True,
                )
            
            total_loss = 0.0
            total_cost = 0.0
            total_min_batch_cost = 0.0 
            total_policy_loss = 0.0
            total_critic_loss = 0.0
            min_epoch_cost = float('inf')

            # [추가] 엔트로피 가중치 스케줄링 (Exponential Decay)
            # Epoch 1: 0.01 -> Epoch 20: ~0.019 -> Epoch 50: ~0.004
            current_entropy_weight = max(0.01, 0.1 * (0.99 ** (epoch - 1)))

            self.optimizer.zero_grad() # [이동] 루프 시작 전 최초 1회 초기화

            for step in range(1, total_steps + 1):

                # -----------------------------------------------------------
                # 데이터 생성 파이프라인 변경 (Random Batch + POMO)
                # -----------------------------------------------------------
                pomo_size = getattr(args, 'pomo_size', 16)  # Config에서 로드 (기본값 16)
                
                # env.reset 대신 Generator 직접 호출
                # td shape: [Batch, POMO, N, D]
                raw_td = self.env.generator.generate_random_batch(
                    batch_size=args.batch_size, 
                    device=self.device
                )
                
                # 2. 환경 초기화 (Environment Reset)
                # 생성된 문제(raw_td)를 reset에 전달하여 동적 상태('done' 등)를 초기화함
                td = self.env.reset(init_td=raw_td, current_epoch=epoch)

                # ------------------------------------------------------------------
                # 랜덤 생성된 실제 Layout 정보 로그 (에폭의 첫 스텝만 출력)
                # ------------------------------------------------------------------
                if self.local_rank <= 0 and step == 1:
                    node_types = td["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
                    n_b = (node_types == NODE_TYPE_BATTERY).sum().item()
                    n_l = (node_types == NODE_TYPE_LOAD).sum().item()
                    n_ic = (node_types == NODE_TYPE_IC).sum().item()
                    n_e = (node_types == NODE_TYPE_EMPTY).sum().item()
                    self.log(f"🎲 [Epoch {epoch} Sample] Layout: [{n_b} B] + [{n_l} L] + [{n_ic} T] + [{n_e} E] (Total: {self.env.N_max})")


                # 3. 모델 포워드 (솔루션 생성)
                # AMP 적용을 위해 autocast 컨텍스트에서 수행
                with autocast(device_type='cuda', enabled=self.use_amp):
                    out = self.model(
                        td, self.env, decode_type='sampling', pbar=train_pbar,
                        status_msg=f"Epoch {epoch}", log_fn=self.log,
                        log_idx=args.log_idx, log_mode=args.log_mode,
                        return_final_td=True
                    )
               
                # out["reward"] shape: (Batch * Actual_POMO, 1)
                # -> (Batch, Actual_POMO)
                reward = out["reward"].view(args.batch_size, -1)
                log_likelihood = out["log_likelihood"].view(args.batch_size, -1)
                
                bom_cost = out["bom_cost"].view(args.batch_size, -1)
                sleep_cost = out["sleep_cost"].view(args.batch_size, -1)    
                
                # 1. POMO Baseline (현재 배치의 평균)
                pomo_baseline = reward.mean(dim=1, keepdim=True)
                
                advantage = reward - pomo_baseline
                if advantage.numel() > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                policy_loss = -(advantage * log_likelihood).mean()
                # 엔트로피 손실 추가 (Maximize Entropy)
                entropy_loss = - current_entropy_weight * out["entropy"].mean()
                loss = policy_loss + entropy_loss

                # 5. 역전파 (AMP: loss를 스케일링 후 backprop)
                self.scaler.scale(loss / accumulation_steps).backward()

                # 지정된 누적 스텝마다 가중치 업데이트 수행
                if step % accumulation_steps == 0:
                    max_norm = float(self.args.optimizer_params.get('max_grad_norm', 0))
                    if max_norm > 0:
                        # 스케일된 그래디언트 클리핑을 위해 unscale 실행
                        self.scaler.unscale_(self.optimizer)
                        clip_grad_norms(self.optimizer.param_groups, max_norm=max_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                # (DDP) 모든 GPU의 통계를 집계
                if self.is_ddp:
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)
                    dist.all_reduce(policy_loss, op=dist.ReduceOp.AVG)
                    #dist.all_reduce(critic_loss, op=dist.ReduceOp.AVG)
                    # (min_cost는 all_reduce(op=dist.ReduceOp.MIN) 필요)
                
                # (DDP) 0번 GPU에서만 로그 기록
                if self.local_rank <= 0:
                    avg_cost = -reward.mean().item()
                    avg_bom = bom_cost.mean().item()
                    avg_sleep = sleep_cost.mean().item()
                    min_batch_cost = -reward.max().item()
                    total_min_batch_cost += min_batch_cost
                    min_epoch_cost = min(min_epoch_cost, min_batch_cost)

                    total_loss += loss.item()
                    total_cost += avg_cost
                    total_policy_loss += policy_loss.item()
                    #total_critic_loss += critic_loss.item()

                    # [수정] 변수 선언 위치 확인 (Rank 0 블록 내부)
                    avg_entropy_val = out["entropy"].mean().item()

                    update_progress(
                        train_pbar,
                        {
                            "Loss": loss.item(),
                            "Avg Cost": total_cost / step,
                            "Avg Min Batch": total_min_batch_cost / step, # [추가] 배치별 최소값의 평균
                            "Min Cost": min_epoch_cost,
                            "Entropy": avg_entropy_val, # [추가]
                            "Avg BOM": avg_bom,    # [추가]
                            "Avg Sleep": avg_sleep # [추가]
                        },
                        step
                    )

            if train_pbar:
                train_pbar.close()

            # 에폭이 끝날 때 한 번 더 캐시 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # (DDP) 0번 GPU에서만 에폭 요약, 검증, 저장
            if self.local_rank <= 0:
                epoch_summary = (
                    f"Epoch {epoch}/{args.trainer_params['epochs']} | "
                    f"Avg Loss {total_loss / total_steps:.4f} | "
                    f"P_Loss {total_policy_loss / total_steps:.4f} | "
                    #f"V_Loss {total_critic_loss / total_steps:.4f} | "
                    f"Min Cost ${min_epoch_cost:.2f}"
                )
                tqdm.write(epoch_summary)
                self.log(epoch_summary)
                
                # --- 검증 (Evaluate) ---
                val_metrics = self.evaluate(epoch)
                
                # 로그 출력
                log_msg = f"[Eval Summary] Epoch {epoch}"
                if "clean" in val_metrics:
                    c = val_metrics["clean"]
                    log_msg += f" | Clean: ${c['avg_bom']:.2f} (Feas: {c['feasibility']*100:.0f}%)"
                if "crisis" in val_metrics:
                    c = val_metrics["crisis"]
                    log_msg += f" | Crisis: ${c['avg_bom']:.2f} (Feas: {c['feasibility']*100:.0f}%)"
                self.log(log_msg)

                # --- 체크포인트 저장 ---
                if (epoch % args.trainer_params['model_save_interval'] == 0) or \
                   (epoch == args.trainer_params['epochs']):
                       
                    save_path = os.path.join(args.result_dir, f'epoch-{epoch}.pth')
                    self.log(f"모델 저장 중... (Epoch {epoch} -> {save_path})")
                    self._run_test_visualization(epoch, is_best=False) # 시각화
                    
                    model_state_dict = self.model.module.state_dict() if self.is_ddp else self.model.state_dict()
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_state_dict,
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, save_path)

            self.scheduler.step()

            if self.local_rank <= 0:
                self.time_estimator.print_est_time(epoch, args.trainer_params['epochs'])
            
            if self.is_ddp:
                dist.barrier() # 에폭 종료 시 모든 GPU 동기화

        if self.local_rank <= 0:
            self.log(" *** 훈련 완료 *** ")

    @torch.no_grad()
    def validate_on_dataset(self, dataset_td: TensorDict, desc: str):
        """ 특정 데이터셋에 대해 Greedy 평가를 수행하고 메트릭 반환 """
        self.model.eval()
        
        total_instances = dataset_td.shape[0]
        batch_size = 16 
        
        total_bom_cost = 0.0
        total_sleep_current = 0.0 # 이름 변경: Penalty -> Current (실제 값이므로)
        total_feasible_count = 0
        
        num_batches = (total_instances + batch_size - 1) // batch_size
        
        iterator = range(num_batches)
        if desc:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc=desc, dynamic_ncols=True)
            except ImportError:
                pass
        
        for i in iterator:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_instances)
            current_batch_size = end_idx - start_idx
            
            # [1] 배치 로드 및 환경 초기화
            raw_batch = dataset_td[start_idx:end_idx]
            batch_td = self.env.reset(init_td=raw_batch).to(self.device)
            
            # [2] 모델 실행
            with autocast(device_type='cuda', enabled=self.use_amp):
                out = self.model(
                    batch_td, 
                    self.env, 
                    decode_type='greedy',
                    return_final_td=False
                )
            
            # [3] 결과 분석
            # (Batch * POMO, 1) -> (Batch, POMO) 변환
            total_reward = out["reward"].view(current_batch_size, -1)
            
            # POMO 중 가장 높은 보상을 받은 궤적(Best Trajectory) 선택
            best_values, best_indices = total_reward.max(dim=1)
            
            # --- [수정] 모델의 실제 출력 키(bom_cost, sleep_cost) 사용 ---
            # V9 모델은 실제 값(Cost)을 반환하므로 바로 가져옵니다.
            cost_bom = out["bom_cost"].view(current_batch_size, -1)
            cost_sleep = out["sleep_cost"].view(current_batch_size, -1)
            
            # Best 인덱스에 해당하는 Cost 추출 (gather)
            best_indices_unsqueezed = best_indices.unsqueeze(-1)
            best_bom = cost_bom.gather(1, best_indices_unsqueezed).squeeze(-1)
            best_sleep = cost_sleep.gather(1, best_indices_unsqueezed).squeeze(-1)
            
            # 통계 합산 (Cost는 양수이므로 그대로 더함)
            total_bom_cost += best_bom.sum().item()
            total_sleep_current += best_sleep.sum().item()
            
            # 성공 여부 판별 
            # (실패 시 페널티가 -20000.0 이하로 떨어지므로, -10000.0 보다 크면 Feasible로 간주)
            is_feasible = (best_values > -10000.0)
            total_feasible_count += is_feasible.sum().item()
            
        # 평균 계산
        avg_bom = total_bom_cost / total_instances
        avg_sleep = total_sleep_current / total_instances
        feasibility_rate = total_feasible_count / total_instances
        
        # 로그 출력용 딕셔너리 반환
        return {
            "avg_bom": avg_bom,
            "avg_sleep": avg_sleep,
            "feasibility": feasibility_rate,
            # 편의상 합산 비용을 total_cost로 표기 (단위가 다를 수 있음 주의)
            "avg_total_cost": avg_bom + avg_sleep 
        }

    @torch.no_grad()
    def evaluate(self, epoch: int):
        """ 로드된 Clean / Crisis 데이터셋에 대해 평가 수행 """
        metrics = {}
        
        # ... (평가 수행 코드는 기존과 동일) ...
            
        # 로그 포맷 수정
        log_msg = f"[Eval Summary] Epoch {epoch}"
        
        if "clean" in metrics:
            c = metrics["clean"]
            # Total / BOM / Sleep / Feasibility 출력
            log_msg += (f"\n   👉 Clean : Total {c['avg_total_cost']:.1f} "
                        f"(BOM {c['avg_bom']:.1f} + Sleep {c['avg_sleep']:.1f}) "
                        f"| Feas: {c['feasibility']*100:.1f}%")
                        
        if "crisis" in metrics:
            c = metrics["crisis"]
            log_msg += (f"\n   👉 Crisis: Total {c['avg_total_cost']:.1f} "
                        f"(BOM {c['avg_bom']:.1f} + Sleep {c['avg_sleep']:.1f}) "
                        f"| Feas: {c['feasibility']*100:.1f}%")
                        
        self.log(log_msg)
        
        # CSV 로깅 (컬럼 추가)
        csv_path = os.path.join(self.result_dir, "val_log.csv")
        header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if header: 
                f.write("epoch,clean_total,clean_bom,clean_sleep,clean_feas,crisis_total,crisis_bom,crisis_sleep,crisis_feas\n")
            
            c = metrics.get("clean", {})
            cr = metrics.get("crisis", {})
            
            f.write(f"{epoch},"
                    f"{c.get('avg_total_cost', -1):.2f},{c.get('avg_bom', -1):.2f},{c.get('avg_sleep', -1):.2f},{c.get('feasibility', -1):.2f},"
                    f"{cr.get('avg_total_cost', -1):.2f},{cr.get('avg_bom', -1):.2f},{cr.get('avg_sleep', -1):.2f},{cr.get('feasibility', -1):.2f}\n")
            
        return metrics

    def test(self, test_data_override=None):
        """ 
        [최종 실험] Clean / Crisis 테스트 데이터셋(1000개)에 대한 정량 평가 수행 
        (또는 사용자가 지정한 JSON 파일 테스트)
        """
        self.model.eval()
        self.log("=" * 60)
        self.log("🔬 최종 테스트 (Final Test) 시작...")

        # [케이스 1] 사용자가 지정한 커스텀 JSON 데이터 테스트
        if test_data_override is not None:
            self.log("🚀 Testing with custom provided data (from JSON)...")
            
            # 1. 전체 데이터셋에 대한 정량 평가 (Metrics 계산)
            res = self.validate_on_dataset(test_data_override, desc="Custom-JSON-Test")
            self.log(f"   📊 [Summary] Avg BOM ${res['avg_bom']:.4f} | Feas: {res['feasibility']*100:.1f}%")
            
            # 2. [추가] 첫 번째 문제에 대한 상세 로그 및 시각화 수행
            self.log("-" * 60)
            self.log("🔎 Running Detailed Visualization for the first custom case...")
            
            # (1) 첫 번째 샘플 추출 및 환경 리셋
            # 배치 크기가 1인 TensorDict로 만들어서 전달
            single_case_td = test_data_override[:1].clone().to(self.device)
            td = self.env.reset(init_td=single_case_td)
            
            # (2) 모델 실행 (log_mode='detail' 활성화)
            with autocast(device_type='cuda', enabled=self.use_amp):
                out = self.model(
                    td, self.env,
                    decode_type=self.args.decode_type, # 실행 인자(greedy/sampling) 따름
                    log_fn=self.log,
                    log_idx=0,
                    log_mode='detail',   # 👈 핵심: 상세 로그 출력 활성화
                    return_final_td=True # 👈 핵심: 시각화용 최종 상태 반환
                )
            
            # (3) 시각화를 위한 정보 추출
            # POMO 결과 중 가장 Reward가 높은(Cost가 낮은) 궤적 선택
            reward = out['reward']
            best_idx = reward.argmax()
            final_cost = -reward[best_idx].item()
            
            # 최종 상태 TensorDict 추출
            final_td_all = out["final_td"]
            final_td_instance = final_td_all[best_idx] # (N_max, D)
            
            # 시작 노드 이름 찾기
            num_starts, start_nodes_idx = self.env.select_start_nodes(td)
            if num_starts > 0:
                best_start_node_local_idx = best_idx % num_starts
                best_start_node_idx = start_nodes_idx[best_start_node_local_idx].item()
                best_start_node_name = self.env.generator.config.node_names[best_start_node_idx]
            else:
                best_start_node_name = "Unknown"

            # (4) 시각화 파일명 생성 (JSON 파일명 활용)
            json_name = "custom_problem"
            if getattr(self.args, 'test_json', None):
                base_name = os.path.basename(self.args.test_json)
                json_name = os.path.splitext(base_name)[0]
            
            filename_prefix = f"vis_{json_name}"
            
            # (5) 시각화 함수 호출 (PNG 저장)
            self.visualize_result(
                final_td_instance, 
                final_cost, 
                best_start_node_name, 
                filename_prefix=filename_prefix
            )
            
            self.log("=" * 60)
            return
        
        # [케이스 2] 기존 Clean/Crisis 데이터셋 테스트 (기존 로직 유지)
        # 1. 테스트 데이터셋 경로 설정
        test_base_path = "test_data"
        test_files = {
            "clean": "test_set_final_1000_clean.pt",
            "crisis": "test_set_final_1000_crisis.pt"
        }
        
        test_datasets = {}
        
        # 2. 데이터셋 로드
        for key, filename in test_files.items():
            path = os.path.join(test_base_path, filename)
            if os.path.exists(path):
                self.log(f"📂 Loading Test Set [{key.upper()}]: {path}")
                try:
                    test_datasets[key] = torch.load(path, weights_only=False).to("cpu")
                except Exception as e:
                    self.log(f"❌ 데이터 로드 실패: {e}")
            else:
                self.log(f"⚠️ 파일 없음: {path} (generate_final_test.py를 먼저 실행하세요)")

        if not test_datasets:
            self.log("❌ 수행할 테스트 데이터가 없습니다. 종료합니다.")
            return

        # 3. 평가 수행
        results = {}
        self.log("-" * 60)
        
        for name, ds in test_datasets.items():
            self.log(f"🚀 Evaluating {name.upper()} Set ({len(ds)} samples)...")
            res = self.validate_on_dataset(ds, desc=f"Test-{name}")
            results[name] = res
            self.log(f"   👉 {name.upper()} Result: Avg BOM ${res['avg_bom']:.4f}")

        # 4. 최종 리포트
        self.log("=" * 60)
        self.log("📊 [FINAL REPORT] 논문 실험 결과 요약")
        if "clean" in results:
            r = results["clean"]
            self.log(f"✅ Normal Condition (Clean) : Cost ${r['avg_bom']:.4f} | Feasibility {r['feasibility']*100:.1f}%")
        if "crisis" in results:
            r = results["crisis"]
            self.log(f"⚠️ Supply Crisis (Crisis) : Cost ${r['avg_bom']:.4f} | Feasibility {r['feasibility']*100:.1f}%")
        self.log("=" * 60)
        
        # (옵션) 마지막 시각화
        self._run_test_visualization(epoch=9999, is_best=False)

    @torch.no_grad()
    def _run_test_visualization(self, epoch: int, is_best: bool = False):
        """
        단일 인스턴스에 대해 추론을 실행하고,
        최종 텐서(TensorDict) 상태를 기반으로 파워트리 시각화(PNG)를 저장합니다.
        """
        self.model.eval()
        args = self.args
        
        # --- [파일 이름 접두사 설정] ---
        if is_best:
            log_prefix = f"[Test Viz @ Epoch {epoch} (BEST)]"
            filename_prefix = f"epoch_{epoch}_best"
        elif epoch > 0:
            log_prefix = f"[Test Viz @ Epoch {epoch}]"
            filename_prefix = f"epoch_{epoch}"
        else:
            log_prefix = "[Test Viz (Standalone)]"
            filename_prefix = "test_solution"

        self.log(f"{log_prefix} 추론 및 시각화 시작...")

        # 1. 단일 배치(B=1)로 환경 리셋
        #td = self.env.reset(batch_size=1)
        # 1. 검증 데이터셋(Clean)이 있는지 확인하고, 있으면 0번 문제를 가져옵니다.
        if "clean" in self.val_datasets and self.val_datasets["clean"] is not None and len(self.val_datasets["clean"]) > 0:
            # 0번 인덱스만 잘라서 가져옴 (항상 같은 문제)
            sample_td = self.val_datasets["clean"][:1].clone().to(self.device)
            td = self.env.reset(init_td=sample_td)
            self.log(f"   👉 [Fixed] 검증 데이터셋(Clean)의 첫 번째 샘플을 시각화합니다.")
        else:
            # 데이터셋이 없으면 어쩔 수 없이 랜덤 생성
            td = self.env.reset(batch_size=1)
            self.log(f"   👉 [Random] 검증 데이터셋이 없어 랜덤 샘플을 시각화합니다.")
        
        # 2. POMO 확장
        test_samples, start_nodes_idx = self.env.select_start_nodes(td)

        pbar_desc = f"Solving (Mode: {args.decode_type}, Samples: {test_samples})"
        pbar = tqdm(total=1, desc=pbar_desc, dynamic_ncols=True)
        
        # 3. 모델 추론 (AMP 사용)
        with autocast(device_type='cuda', enabled=self.use_amp):
            out = self.model(
                td, self.env,
                decode_type=args.decode_type,
                pbar=pbar,
                log_fn=self.log,
                log_idx=args.log_idx,
                log_mode='detail',
                return_final_td=True,
            )
        pbar.close()

        # 4. 최고 성능 솔루션 선택
        reward = out['reward'] # (B_total,)
        best_idx = reward.argmax()
        final_cost = -reward[best_idx].item()
        
        # 5. 모델이 돌리고 온 최종 TensorDict에서 해당 sample만 추출
        final_td_all = out["final_td"]        # (B_total, N_max, ...)
        final_td_instance = final_td_all[best_idx].clone()

        # 6. POMO 시작 노드 이름 찾기
        best_start_node_local_idx = best_idx % test_samples
        best_start_node_idx = start_nodes_idx[best_start_node_local_idx].item()
        best_start_node_name = self.env.generator.config.node_names[best_start_node_idx]
        
        # 1. BOM Cost 계산 (final_td_instance에서 직접 합산)
        #    (TensorDict에서 Cost 피처 인덱스는 5번입니다 - definitions.py 기준)
        active_nodes_mask = final_td_instance["is_active_mask"].bool()
        all_nodes = final_td_instance["nodes"]
        
        # Active 노드 중 IC 타입인 것들의 Cost 합산
        # (노드 타입 인덱스: 0~3, IC는 3번)
        node_types = all_nodes[..., FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        ic_mask = (node_types == NODE_TYPE_IC)
        
        # 최종적으로 Active 상태인 IC들의 가격 합계
        total_bom_cost = all_nodes[active_nodes_mask & ic_mask, FEATURE_INDEX["cost"]].sum().item()

        # [수정] V9은 Hard Constraint이므로 별도의 Sleep Penalty가 없음.
        # (final_cost는 WEIGHT_BOM=100 배율이 적용된 점수이므로 시각화 텍스트에서 제외하거나 보정)
        
        self.log(f"추론 완료 (Score: {final_cost:.1f} | "
                 f"Real BOM Cost: ${total_bom_cost:.4f}), "
                 f"Start: '{best_start_node_name}'")

        # 7. 시각화 실행 (최종 TD와 계산된 값을 사용)
        self.visualize_result(
            final_td_instance, 
            final_cost, 
            best_start_node_name, 
            filename_prefix
        )

        # ── 메모리 정리 ────────────────────────────────────────────────
        # out 안에는 reward, log_likelihood 등 GPU 텐서가 포함되어 있다.
        # 함수가 끝나면 어차피 파이썬 참조는 사라지지만,
        # CUDA 캐시를 조금이라도 되돌리고 싶다면 여기서 정리해 줄 수 있다.
        try:
            del out
            del final_td_all
            del final_td_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except NameError:
            # 혹시라도 위 변수 이름이 바뀌어도 크래시는 방지
            pass


        self.log(f"{log_prefix} 시각화 다이어그램 저장 완료.")

    def visualize_result(self, 
                         final_td: TensorDict, 
                         final_cost: float, 
                         best_start_node_name: str, 
                         filename_prefix: str = "solution"):
        """
        최종 TensorDict 상태를 기반으로 파워트리 시각화(PNG)를 저장합니다.
        """

        if self.result_dir is None: return
        os.makedirs(self.result_dir, exist_ok=True)

        # 1. 정보 추출 및 맵 생성
        node_names = self.env.generator.config.node_names
        loads_map = {load['name']: load for load in self.env.generator.config.loads}
        candidate_ics_map = {ic['name']: ic for ic in self.env.generator.config.available_ics}
        battery_conf = self.env.generator.config.battery
        constraints = self.env.generator.config.constraints

        all_nodes_features = final_td["nodes"].squeeze(0)
        is_active_mask = final_td["is_active_mask"].squeeze(0)

        # --- Spawn된 슬롯의 이름을 템플릿 기반으로 생성 ---
        dynamic_node_names = list(node_names)
        if len(dynamic_node_names) < self.env.N_max:
            dynamic_node_names.extend([None] * (self.env.N_max - len(dynamic_node_names)))

        spawn_name_counter: Counter = Counter()
        for idx in range(len(node_names), self.env.N_max):
            if idx >= len(is_active_mask) or not is_active_mask[idx]:
                continue

            node_feat = all_nodes_features[idx]
            node_id_val = node_feat[FEATURE_INDEX["node_id"]].item()
            template_idx = int(round(node_id_val * self.env.N_max))

            if 0 <= template_idx < len(node_names):
                base_name = node_names[template_idx]
            else:
                base_name = f"Spawned_Template_{template_idx}"

            spawn_name_counter[base_name] += 1
            dynamic_node_names[idx] = f"{base_name}#{spawn_name_counter[base_name]}"

        # --- Safe Name Lookup Helper ---
        def get_node_name_safe(idx: int) -> str:
            if 0 <= idx < len(dynamic_node_names):
                name = dynamic_node_names[idx]
                if name:
                    return name
            if idx == -1:
                return "N/A"
            return f"Spawned_IC_{idx}"
        # --- Safe Name Lookup Helper ---

        # 2. 엣지 재구성 (adj_matrix를 사용)
        adj_matrix = final_td["adj_matrix"].squeeze(0) # (N_max, N_max)
        
        used_ic_names = set()
        child_to_parent = {}
        parent_to_children = defaultdict(list)
        
        parent_indices, child_indices = adj_matrix.nonzero(as_tuple=True)
        for p_idx, c_idx in zip(parent_indices, child_indices):
            p_name = get_node_name_safe(p_idx.item())
            c_name = get_node_name_safe(c_idx.item())
            
            child_to_parent[c_name] = p_name
            parent_to_children[p_name].append(c_name)
            
            if p_name in candidate_ics_map:
                used_ic_names.add(p_name)
        
        # 3. Always-On, Independent Rail 경로 추적
        always_on_nodes = {
            name for name, conf in loads_map.items() if conf.get("always_on_in_sleep", False)
        }
        always_on_nodes.add(battery_conf['name'])
        nodes_to_process = list(always_on_nodes)

        while nodes_to_process:
            node = nodes_to_process.pop(0)
            if node in child_to_parent:
                parent = child_to_parent[node]
                if parent not in always_on_nodes:
                    always_on_nodes.add(parent)
                    nodes_to_process.append(parent)

        supplier_nodes = set()
        path_nodes = set()
        for name, conf in loads_map.items():
            rail_type = conf.get("independent_rail_type")
            if rail_type == 'exclusive_supplier':
                supplier_nodes.add(name)
                if name in child_to_parent:
                    supplier_nodes.add(child_to_parent.get(name))
            elif rail_type == 'exclusive_path':
                current_node = name
                while current_node in child_to_parent:
                    path_nodes.add(current_node)
                    parent = child_to_parent[current_node]
                    path_nodes.add(parent)
                    if parent == battery_conf['name']: break
                    current_node = parent

        # 4. 액티브/슬립 전류 및 전력 계산 (Bottom-up 방식) 
        junction_temps, actual_i_ins_active, actual_i_outs_active = {}, {}, {}
        actual_i_ins_sleep, actual_i_outs_sleep, ic_self_consumption_sleep = {}, {}, {}
        
        active_current_draw = {name: conf["current_active"] for name, conf in loads_map.items()}
        sleep_current_draw = {name: conf["current_sleep"] for name, conf in loads_map.items()}

        node_types = all_nodes_features[..., FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        is_active = final_td["is_active_mask"].squeeze(0)
        active_indices = torch.where(is_active)[0]

        active_ics_indices = [
            idx.item() for idx in active_indices
            if node_types[idx] == NODE_TYPE_IC
        ]
        
        processed_ics = set()
        
        while len(processed_ics) < len(active_ics_indices):
            progress_made = False
            
            for ic_idx in active_ics_indices:
                ic_name = get_node_name_safe(ic_idx)
                if ic_name in processed_ics: continue

                if ic_name not in candidate_ics_map:
                    node_feat = all_nodes_features[ic_idx]
                    ic_type_idx = node_feat[FEATURE_INDEX["ic_type_idx"]].item()
                    ic_type = 'LDO' if ic_type_idx == 1.0 else 'Buck'
                    
                    ic_data_for_obj = {
                        'type': ic_type,
                        'name': ic_name,
                        'vin': node_feat[FEATURE_INDEX["vin_min"]].item(),
                        'vout': node_feat[FEATURE_INDEX["vout_min"]].item(),
                        # --- FIX: Missing required positional arguments ---
                        'vin_min': node_feat[FEATURE_INDEX["vin_min"]].item(),
                        'vin_max': node_feat[FEATURE_INDEX["vin_max"]].item(),
                        'vout_min': node_feat[FEATURE_INDEX["vout_min"]].item(),
                        'vout_max': node_feat[FEATURE_INDEX["vout_max"]].item(),
                        # ------------------------------------------------
                        'original_i_limit': node_feat[FEATURE_INDEX["i_limit"]].item() / (1.0 - constraints.get('current_margin', 0.1)),
                        'i_limit': node_feat[FEATURE_INDEX["i_limit"]].item(),
                        'operating_current': node_feat[FEATURE_INDEX["op_current"]].item(),
                        'quiescent_current': node_feat[FEATURE_INDEX["quiescent_current"]].item(),
                        'shutdown_current': node_feat[FEATURE_INDEX["shutdown_current"]].item(),
                        'cost': node_feat[FEATURE_INDEX["cost"]].item(),
                        'theta_ja': node_feat[FEATURE_INDEX["theta_ja"]].item(),
                        't_junction_max': node_feat[FEATURE_INDEX["t_junction_max"]].item(),
                    }
                    if ic_type == 'LDO': ic_data_for_obj['v_dropout'] = 0.0
                    
                else: 
                    ic_data_for_obj = candidate_ics_map[ic_name].copy()
                    ic_type = ic_data_for_obj['type']
                
                ic_obj = LDO(**ic_data_for_obj) if ic_type == 'LDO' else BuckConverter(**ic_data_for_obj)
                
                children_names = parent_to_children.get(ic_name, [])

                if all(c in loads_map or c in processed_ics for c in children_names):
                    
                    # --- Active 전류/발열 계산 ---
                    total_i_out_active = sum(active_current_draw.get(c, 0) for c in children_names)
                    actual_i_outs_active[ic_name] = total_i_out_active
                    
                    i_in_active = ic_obj.calculate_active_input_current(vin=ic_obj.vin, i_out=total_i_out_active)
                    power_loss = ic_obj.calculate_power_loss(vin=ic_obj.vin, i_out=total_i_out_active)
                    
                    active_current_draw[ic_name] = i_in_active
                    actual_i_ins_active[ic_name] = i_in_active
                    ambient_temp = constraints.get('ambient_temperature', 25)
                    junction_temps[ic_name] = ambient_temp + (power_loss * ic_obj.theta_ja)

                    # --- Sleep 전류 계산 ---
                    parent_name = child_to_parent.get(ic_name)
                    is_ao = ic_name in always_on_nodes
                    parent_is_ao = parent_name in always_on_nodes or parent_name == battery_conf['name']
                    
                    total_i_out_sleep = sum(sleep_current_draw.get(c, 0) for c in children_names)

                    ic_self_sleep = ic_obj.get_self_sleep_consumption(is_ao, parent_is_ao)
                    i_in_for_children = ic_obj.calculate_sleep_input_for_children(vin=ic_obj.vin, i_out_sleep=total_i_out_sleep)
                    
                    i_in_sleep = ic_self_sleep + i_in_for_children

                    actual_i_ins_sleep[ic_name] = i_in_sleep
                    actual_i_outs_sleep[ic_name] = total_i_out_sleep
                    ic_self_consumption_sleep[ic_name] = ic_self_sleep
                    sleep_current_draw[ic_name] = i_in_sleep

                    processed_ics.add(ic_name)
                    progress_made = True
            
            if not progress_made and len(active_ics_indices) > 0 and len(processed_ics) < len(active_ics_indices): 
                self.log(f"⚠️ 경고: Power Tree 계산 순환 참조 발생 또는 미처리 IC 잔존.")
                break

        # 5. 최종 시스템 전체 값 계산
        primary_nodes = parent_to_children.get(battery_conf['name'], [])
        total_active_current = sum(active_current_draw.get(name, 0) for name in primary_nodes)
        total_sleep_current = sum(sleep_current_draw.get(name, 0) for name in primary_nodes)
        battery_avg_voltage = (battery_conf['voltage_min'] + battery_conf['voltage_max']) / 2
        total_active_power = battery_avg_voltage * total_active_current

        # 6. Graphviz 다이어그램 생성
        # --- 👇 [신규] BOM 비용과 암전류 페널티 분리 계산 ---
        #total_bom_cost = sum(candidate_ics_map[name]['cost'] for name in used_ic_names)

        # --- [수정 후] 노드 피처 텐서에서 직접 Cost 합산 ---
        total_bom_cost = 0.0
        # active_ics_indices는 함수 상단(4번 섹션)에서 이미 구해져 있습니다.
        for ic_idx in active_ics_indices:
            # FEATURE_INDEX["cost"] = 5 (definitions.py 기준)
            node_cost = all_nodes_features[ic_idx, FEATURE_INDEX["cost"]].item()
            total_bom_cost += node_cost
       
        label_str = (f"Transformer Solution (Start: {best_start_node_name})\\n"
                     f"Total BOM Cost: ${total_bom_cost:.4f}")

        dot = Digraph(comment=f"Power Tree - Cost ${final_cost:.4f}")
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial')
        
        margin_info = f"Current Margin: {constraints.get('current_margin', 0)*100:.0f}%"
        temp_info = f"Ambient Temp: {constraints.get('ambient_temperature', 25)}°C"
        dot.attr(rankdir='LR', label=label_str, labelloc='t')

        max_sleep_current_target = constraints.get('max_sleep_current', 0.0)
        battery_label = (f"🔋 {battery_conf['name']}\n\n"
            f"Total Active Power: {total_active_power:.2f} W\n"
            f"Total Active Current: {total_active_current * 1000:.1f} mA\n"
            f"Target Sleep Current: <= {max_sleep_current_target * 1000000:,.1f} µA\n"
            f"Total Sleep Current: {total_sleep_current * 1000000:,.1f} µA")
        dot.node(battery_conf['name'], battery_label, shape='box', color='darkgreen', fillcolor='white')

        sequenced_loads = set()
        if 'power_sequences' in constraints:
            for seq in constraints['power_sequences']:
                sequenced_loads.add(seq['j']); sequenced_loads.add(seq['k'])
        
        for ic_idx in active_ics_indices:
            ic_name = get_node_name_safe(ic_idx)
            
            if ic_name not in candidate_ics_map:
                node_feat = all_nodes_features[ic_idx]
                ic_data_for_label = {
                    'name': ic_name,
                    'vin': node_feat[FEATURE_INDEX["vin_min"]].item(),
                    'vout': node_feat[FEATURE_INDEX["vout_min"]].item(),
                    'operating_current': node_feat[FEATURE_INDEX["op_current"]].item(),
                    't_junction_max': node_feat[FEATURE_INDEX["t_junction_max"]].item(),
                    'cost': node_feat[FEATURE_INDEX["cost"]].item(),
                }
            else:
                ic_data_for_label = candidate_ics_map[ic_name]
            
            
            i_in_active_val = actual_i_ins_active.get(ic_name, 0)
            i_out_active_val = actual_i_outs_active.get(ic_name, 0)
            i_in_sleep_val = actual_i_ins_sleep.get(ic_name, 0)
            i_out_sleep_val = actual_i_outs_sleep.get(ic_name, 0)
            i_self_sleep_val = ic_self_consumption_sleep.get(ic_name, 0)
            calculated_tj = junction_temps.get(ic_name, 0) 
            
            thermal_margin = ic_data_for_label['t_junction_max'] - calculated_tj
            node_color = 'blue'
            if thermal_margin < 10: node_color = 'red'
            elif thermal_margin < 25: node_color = 'orange'
            
            node_style = 'rounded,filled'
            if ic_name not in always_on_nodes:
                node_style += ',dashed'

            fill_color = 'white'
            if ic_name in path_nodes:
                fill_color = 'lightblue'
            elif ic_name in supplier_nodes:
                fill_color = 'lightyellow'
            
            label = (f"📦 {ic_name.split('@')[0]}\n\n"
                     f"Vin: {ic_data_for_label['vin']:.2f}V, Vout: {ic_data_for_label['vout']:.2f}V\n"
                     f"Iin: {i_in_active_val*1000:.1f}mA (Act) | {i_in_sleep_val*1000000:,.1f}µA (Slp)\n"
                     f"Iout: {i_out_active_val*1000:.1f}mA (Act) | {i_out_sleep_val*1000000:,.1f}µA (Slp)\n"
                     f"I_self: {ic_data_for_label['operating_current']*1000:.1f}mA (Act) | {i_self_sleep_val*1000000:,.1f}µA (Slp)\n"
                     f"Tj: {calculated_tj:.1f}°C (Max: {ic_data_for_label['t_junction_max']}°C)\n"
                     f"Cost: ${ic_data_for_label['cost']:.2f}")
            dot.node(ic_name, label, color=node_color, fillcolor=fill_color, style=node_style, penwidth='3')

        for name, conf in loads_map.items():
            node_style = 'rounded,filled'
            if name not in always_on_nodes: node_style += ',dashed'
            fill_color = 'white'
            if name in path_nodes: fill_color = 'lightblue'
            elif name in supplier_nodes: fill_color = 'lightyellow'
            
            label = f"💡 {name}\nActive: {conf['voltage_typical']}V | {conf['current_active']*1000:.1f}mA\n"
            if conf['current_sleep'] > 0: label += f"Sleep: {conf['current_sleep'] * 1000000:,.1f}µA\n"
            conditions = []
            if conf.get("independent_rail_type"): conditions.append(f"🔒 {conf['independent_rail_type']}")
            if name in sequenced_loads: conditions.append("⛓️ Sequence")
            if conditions: label += " ".join(conditions)
            
            penwidth = '3' if conf.get("always_on_in_sleep", False) else '1'
            dot.node(name, label, color='dimgray', fillcolor=fill_color, style=node_style, penwidth=penwidth)

        for p_name, children in parent_to_children.items():
            for c_name in children:
                dot.edge(p_name, c_name)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_cost_{final_cost:.4f}_{timestamp}"
        output_path = os.path.join(self.result_dir, filename)
        
        try:
            dot.render(output_path, view=False, format='png', cleanup=True)
            self.log(f"✅ 상세 시각화 다이어그램을 {output_path}.png 파일로 저장했습니다.")
        except Exception as e:
            self.log(f"❌ 시각화 렌더링 실패. (Graphviz 설치 확인 필요): {e}")