# Copyright (c) 2025 Minuk Lee. All rights reserved.
# 
# This source code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# 
# For licensing terms, see the LICENSE file.
# Contact: minuklee@snu.ac.kr
# 
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Unbounded, UnboundedDiscrete, Composite
from typing import Optional, Dict, Union, Tuple, List

# --- 현재 패키지(transformer_solver) 모듈 임포트 ---
from .definitions import (
    FEATURE_DIM, FEATURE_INDEX, SCALAR_PROMPT_FEATURE_DIM,
    NODE_TYPE_PADDING, NODE_TYPE_BATTERY, NODE_TYPE_LOAD, 
    NODE_TYPE_IC, NODE_TYPE_EMPTY
)
from .env_generator import PocatGenerator

# --- 환경 상수 ---
BATTERY_NODE_IDX = 0 # 배터리 노드는 항상 0번 인덱스
REWARD_WEIGHT_ACTION = 0.0  # (A2C) 액션(IC 스폰) 즉시 비용에 대한 가중치
REWARD_WEIGHT_PATH = 1.0    # (A2C) 경로(Load->BATT) 완성 시 누적 비용 가중치
STEP_PENALTY = 0.0          # (A2C) 스텝당 페널티
FAILURE_PENALTY = -20000.0    # (A2C) 실패(막다른 길) 페널티

# 1. 가격(BOM) 민감도 가중치 (Cost Sensitivity)
#    기본값 1.0 -> 100.0으로 상향 (0.01달러 절약 = +1.0점 보상)
#    이제 모델은 1센트(0.01) 차이도 1점 차이로 크게 느낍니다.
WEIGHT_BOM = 100.0
# [수정] V9: 가격 민감도는 유지하되, 암전류 페널티 관련 상수 제거

class PocatEnv(EnvBase):
    """
    Pocat 문제를 풀기 위한 강화학습 환경(Environment)입니다.
    
    TensorDict를 상태(State)로 사용하며, Parameterized Action(Connect/Spawn)을
    처리하여 상태를 전이시키고 보상(Reward)을 계산합니다.
    """
    """
    Pocat V9 환경 (Hard Constraint Version)
    
    특징:
    1. Phase 1 (Always-On Backbone): Always-On 부하들을 우선적으로 연결하여 암전류 구조를 결정합니다.
    2. Hard Sleep Constraint: 보상 페널티 대신, Action Masking을 통해 예산을 초과하는 행동을 원천 차단합니다.
    """
    
    name = "pocat_env"

    def __init__(self, generator_params: dict, device: str = "cpu", N_max: int = 500, **kwargs):
        """
        PocatEnv를 초기화합니다.
        
        Args:
            generator_params (dict): PocatGenerator에 전달될 파라미터
            device (str): 텐서 연산을 수행할 디바이스
            N_max (int): 모델이 처리할 고정된 최대 노드 크기
        """
        super().__init__(device=device)
        
        # 1. N_max 값을 저장합니다.
        self.N_max = N_max
        
        # 2. 제너레이터 초기화 (N_max 전달)
        self.generator = PocatGenerator(**generator_params, N_max=N_max)
        
        # 3. 마스킹 및 계산에 사용할 버퍼 등록
        self.register_buffer("arange_nodes", None, persistent=False)
        self.register_buffer("node_type_tensor", None, persistent=False)
        self.register_buffer("load_idx_tensor", None, persistent=False)
        self.register_buffer("rail_types", None, persistent=False)

        # 4. Observation, Action 스펙 정의
        self._make_spec()

        # 5. 제약조건(시퀀싱, 독립) 정보 로드
        self._load_constraint_info()

    def _make_spec(self):
        """환경의 Observation, Action, Reward 스펙을 정의합니다."""

        num_nodes = self.N_max
        
        # 1. Observation 스펙 정의
        self.observation_spec = Composite({
            # --- 정적 텐서 (Generator 제공) ---
            "nodes": Unbounded(shape=(num_nodes, FEATURE_DIM)),
            "scalar_prompt_features": Unbounded(shape=(SCALAR_PROMPT_FEATURE_DIM,)),
            "matrix_prompt_features": Unbounded(shape=(num_nodes, num_nodes)),
            "connectivity_matrix": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.bool),
            "attention_mask": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.bool),
            
            # --- 동적 텐서 (Env 관리) ---
            "adj_matrix": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.bool),
            "adj_matrix_T": Unbounded(shape=(num_nodes, num_nodes), dtype=torch.bool),
            "unconnected_loads_mask": Unbounded(shape=(num_nodes,), dtype=torch.bool),
            "trajectory_head": UnboundedDiscrete(shape=(1,)),
            "step_count": UnboundedDiscrete(shape=(1,)),
            "current_cost": Unbounded(shape=(1,)),
            "staging_cost": Unbounded(shape=(1,)), # 현재 경로의 누적 비용
            "sleep_cost": Unbounded(shape=(1,)), # (V9: 로깅용으로 유지하거나 0 처리)
    
            "log_reward_bom": Unbounded(shape=(1,)),
            # "log_reward_sleep": Unbounded(shape=(1,)), # 제거
            "log_reward_fail": Unbounded(shape=(1,)),   
            
            "is_used_ic_mask": Unbounded(shape=(num_nodes,), dtype=torch.bool),
            "current_target_load": UnboundedDiscrete(shape=(1,)),
            "is_exclusive_mask": Unbounded(shape=(num_nodes,), dtype=torch.long),
            "next_empty_slot_idx": UnboundedDiscrete(shape=(1,)),
        })
        
        # 2. Action 스펙 정의 (Parameterized Action)
        self.action_spec = Composite({
            # (0: Connect, 1: Spawn)
            "action_type": UnboundedDiscrete(shape=(1,)),
            # (0 ~ N_max-1): Connect 대상
            "connect_target": UnboundedDiscrete(shape=(1,)),
            # (0 ~ N_max-1): Spawn할 템플릿
            "spawn_template": UnboundedDiscrete(shape=(1,)),
        })
        
        # 3. Reward 스펙 정의
        self.reward_spec = Unbounded(shape=(1,))

    def _load_constraint_info(self):
        """
        config 파일에서 제약조건 정보를 로드하고
        마스킹에 사용하기 쉽도록 텐서로 변환하여 저장합니다.
        """
        # (B,L,IC) 순서가 텐서 순서와 일치
        self.node_name_to_idx = {name: i for i, name in enumerate(self.generator.config.node_names)}
        
        # 1. Independent Rail (독립 레일) 정보
        rail_type_map = {"exclusive_supplier": 1, "exclusive_path": 2}
        rail_types_list = []

        # (Load 노드는 1번 인덱스부터 시작)
        load_start_idx = self.generator.num_battery
        for i, load_cfg in enumerate(self.generator.config.loads):
            load_idx = load_start_idx + i
            rail_type = rail_type_map.get(load_cfg.get("independent_rail_type"), 0)
            rail_types_list.append((load_idx, rail_type))
        
        # (N_max,) 크기의 텐서로 변환
        self.rail_types_tensor = torch.zeros(self.N_max, dtype=torch.long, device=self.device)
        if rail_types_list:
            indices = torch.tensor([idx for idx, _ in rail_types_list], dtype=torch.long, device=self.device)
            values = torch.tensor([val for _, val in rail_types_list], dtype=torch.long, device=self.device)
            self.rail_types_tensor.scatter_(0, indices, values)

        # 2. Power Sequence (전원 시퀀싱) 정보
        self.power_sequences = []
        for seq in self.generator.config.constraints.get("power_sequences", []):
            f_flag = seq.get("f", 1)
            j_idx = self.node_name_to_idx.get(seq['j'])
            k_idx = self.node_name_to_idx.get(seq['k'])
            if j_idx is not None and k_idx is not None:
                self.power_sequences.append((j_idx, k_idx, f_flag))

    def _ensure_buffers(self, td: TensorDict):
        """
        Observation 텐서가 변경될 때마다(주로 _reset 시)
        마스킹 계산에 필요한 헬퍼 텐서들을 미리 계산합니다.
        """
        num_nodes = td["nodes"].shape[1] # (N_max)

        if self.arange_nodes is None or self.arange_nodes.numel() != num_nodes:
            self.arange_nodes = torch.arange(num_nodes, device=self.device)
        
        if self.node_type_tensor is None:
            # (N_max,)
            node_types = td["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
            self.node_type_tensor = node_types
            
            # (N_loads,)
            self.load_idx_tensor = torch.where(node_types == NODE_TYPE_LOAD)[0]

    def select_start_nodes(self, td: TensorDict) -> Tuple[int, torch.Tensor]:
        #[수정] POMO 시작 노드 선택 시, Phase 1 전략에 따라 Always-On Load를 우선 반환합니다. 
        
        self._ensure_buffers(td) # load_idx_tensor 최신화

        # Always-On 속성 확인 (B, N) -> (N,) 가정 (모든 샘플이 동일 Config라고 가정)
        # 하지만 배치마다 다를 수 있으므로 첫 번째 샘플 기준으로 추출
        ao_feature = td["nodes"][0, :, FEATURE_INDEX["always_on_in_sleep"]]
        is_ao_load = (ao_feature == 1.0) & (self.node_type_tensor == NODE_TYPE_LOAD)
        
        ao_load_indices = torch.where(is_ao_load)[0]
        
        if len(ao_load_indices) > 0:
            return len(ao_load_indices), ao_load_indices
        else:
            # AO 부하가 없으면 모든 부하 반환
            return len(self.load_idx_tensor), self.load_idx_tensor

    def _set_seed(self, seed: Optional[int] = None):
        if seed is not None:
            torch.manual_seed(seed)

    def _reset(self, td: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        """ 환경을 초기 상태(State)로 리셋합니다. """

        # [추가] 'init_td' 키워드 인자가 있으면 그것을 입력 데이터로 사용
        # (EnvBase.reset의 엄격한 shape check를 우회하기 위함)
        if td is None and "init_td" in kwargs:
            td = kwargs["init_td"]

        batch_size = kwargs.get("batch_size", self.batch_size)
        if td is None:
            if isinstance(batch_size, tuple): batch_size = batch_size[0]
            # batch_size가 torch.Size 객체일 경우 int로 변환
            if isinstance(batch_size, torch.Size):
                 batch_size = batch_size[0] if len(batch_size) > 0 else 1

            td_initial = self.generator(batch_size=batch_size).to(self.device)
        else:
            td_initial = td
            batch_size = td_initial.batch_size[0]

        num_nodes = self.N_max

        # --- 1. 동적 상태 텐서 초기화 ---
        
        # adj_matrix: (B, N_max, N_max) - 실제 연결된 엣지 (모두 0)
        adj_matrix = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=self.device)
        adj_matrix_T = torch.zeros(batch_size, num_nodes, num_nodes, dtype=torch.bool, device=self.device)

        # is_active_mask: (B, N_max) - 현재 활성화된 노드 (정적 피처에서 복사)
        is_active_mask = td_initial["nodes"][..., FEATURE_INDEX["is_active"]].bool()
        # (정적 피처에서 동적 마스크로 복사)
        is_template_mask = td_initial["nodes"][..., FEATURE_INDEX["is_template"]].bool()
        can_spawn_into_mask = td_initial["nodes"][..., FEATURE_INDEX["can_spawn_into"]].bool()

        # unconnected_loads_mask: (B, N_max) - 아직 연결 안 된 로드
        node_types = td_initial["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        unconnected_loads_mask = (node_types == NODE_TYPE_LOAD).unsqueeze(0).expand(batch_size, -1)
        
        # next_empty_slot_idx: (B, 1) - 다음 스폰 위치 (BATT+LOADS+TEMPLATES 개수)
        next_empty_slot_idx = torch.full((batch_size, 1), self.generator.num_components, dtype=torch.long, device=self.device)

        # 2. TensorDict 생성
        reset_td = TensorDict({
            # 정적 텐서 (Generator로부터 복사)
            "nodes": td_initial["nodes"].clone(),
            "scalar_prompt_features": td_initial["scalar_prompt_features"],
            "matrix_prompt_features": td_initial["matrix_prompt_features"],
            "connectivity_matrix": td_initial["connectivity_matrix"],
            "attention_mask": td_initial["attention_mask"],
            
            # 동적 텐서 (초기화)
            "adj_matrix": adj_matrix,
            "adj_matrix_T": adj_matrix_T,
            "unconnected_loads_mask": unconnected_loads_mask,
            "is_active_mask": is_active_mask,
            "is_template_mask": is_template_mask,
            "can_spawn_into_mask": can_spawn_into_mask,
            "next_empty_slot_idx": next_empty_slot_idx,
            "trajectory_head": torch.full((batch_size, 1), BATTERY_NODE_IDX, dtype=torch.long, device=self.device),
            "step_count": torch.zeros(batch_size, 1, dtype=torch.long, device=self.device),
            "current_cost": torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device),
            "staging_cost": torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device),
            "sleep_cost": torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device),
            "is_used_ic_mask": torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device),
            "current_target_load": torch.full((batch_size, 1), -1, dtype=torch.long, device=self.device),
            "is_exclusive_mask": torch.zeros(batch_size, num_nodes, dtype=torch.long, device=self.device),
            "done": torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device),
        }, batch_size=[batch_size], device=self.device)
       
        self._ensure_buffers(reset_td)
        return reset_td

    @torch.no_grad()
    def step(self, tensordict: TensorDict) -> TensorDict:
        """ _step을 호출 (torchrl EnvBase 호환용) """
        return self._step(tensordict)

    def _step(self, td: TensorDict) -> TensorDict:
        """
        모델이 결정한 Parameterized Action을 실행하여
        환경의 상태(State)를 다음 스텝으로 전이시킵니다.
        
        - (액션 타입 0: Connect) -> 기존 활성 노드에 연결
        - (액션 타입 1: Spawn)   -> 템플릿을 Empty 슬롯에 복사(Spawn) 후 연결
        """
        batch_size, num_nodes = td["nodes"].shape[0], self.N_max
        action_dict = td["action"]
        
        # (B,)
        action_type = action_dict["action_type"].squeeze(-1)
        connect_target = action_dict["connect_target"].squeeze(-1)
        spawn_template = action_dict["spawn_template"].squeeze(-1)

        current_head = td["trajectory_head"].squeeze(-1)  # (B,)


        # --- 0. 이미 'done'인 배치는 무시 ---
        is_already_done = td["done"].squeeze(-1)
        if is_already_done.all():
            return TensorDict({
                "next": td, 
                "reward": torch.zeros(batch_size, 1, device=self.device), 
                "done": td["done"]}, batch_size=td.batch_size)

        # --- 1. 상태 텐서 준비 (필요한 것만 선택 복사) ---
        # 전체 td.clone() 제거 → 메모리 사용량 절감
        next_obs = TensorDict({}, batch_size=td.batch_size, device=self.device)

        # 1-1. 에피소드 동안 변하지 않는 정적 텐서들은 참조만 공유
        #      (clone 불필요, 큰 행렬 재할당 방지)
        if "scalar_prompt_features" in td.keys():
            next_obs["scalar_prompt_features"] = td["scalar_prompt_features"]
        if "matrix_prompt_features" in td.keys():
            next_obs["matrix_prompt_features"] = td["matrix_prompt_features"]
        if "attention_mask" in td.keys():
            next_obs["attention_mask"] = td["attention_mask"]

        # ⚠ connectivity_matrix는 Spawn 시 in-place로 갱신되므로 현재 구조에서는
        #   동적 텐서 취급이 필요하다. (정적으로 만들려면 추가 리팩터링 필요)

        # 1-2. 실제로 스텝마다 변하는 동적 텐서들만 clone()
        next_obs["nodes"] = td["nodes"].clone()  # (가장 중요)
        next_obs["adj_matrix"] = td["adj_matrix"].clone()
        next_obs["adj_matrix_T"] = td["adj_matrix_T"].clone()
        if "connectivity_matrix" in td.keys():
            next_obs["connectivity_matrix"] = td["connectivity_matrix"].clone()

        next_obs["unconnected_loads_mask"] = td["unconnected_loads_mask"].clone()
        next_obs["is_active_mask"] = td["is_active_mask"].clone()
        next_obs["is_template_mask"] = td["is_template_mask"].clone()
        next_obs["can_spawn_into_mask"] = td["can_spawn_into_mask"].clone()
        next_obs["next_empty_slot_idx"] = td["next_empty_slot_idx"].clone()

        next_obs["trajectory_head"] = td["trajectory_head"].clone()

        next_obs["step_count"] = td["step_count"].clone()
        next_obs["current_cost"] = td["current_cost"].clone()
        next_obs["staging_cost"] = td["staging_cost"].clone()
        next_obs["sleep_cost"] = td.get("sleep_cost", torch.zeros_like(td["current_cost"])).clone()
        next_obs["is_used_ic_mask"] = td["is_used_ic_mask"].clone()
        next_obs["current_target_load"] = td["current_target_load"].clone()
        if "is_exclusive_mask" in td.keys():
            next_obs["is_exclusive_mask"] = td["is_exclusive_mask"].clone()
        # done은 아래에서 새로 계산하지만, '이미 done인 배치 롤백'에서 사용되므로 먼저 복사
        if "done" in td.keys():
            next_obs["done"] = td["done"].clone()
        else:
            next_obs["done"] = torch.zeros_like(td["step_count"], dtype=torch.bool)

        step_reward = torch.full((batch_size,), STEP_PENALTY, dtype=torch.float32, device=self.device)
        batch_indices = torch.arange(batch_size, device=self.device)
        bom_reward_batch = torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device)

        # --- 2. 액션 타입 분기 ---
        
        # --- 2a. [Select New Load] ---
        # (현재 헤드가 배터리일 때)
        head_is_battery = (current_head == BATTERY_NODE_IDX)
        if head_is_battery.any():
            # 'Select New Load' 액션은 'Connect' 액션으로 전달됨
            b_idx_batt = batch_indices[head_is_battery]
            selected_load = connect_target[head_is_battery]

            is_load_selection = (selected_load != BATTERY_NODE_IDX)
            if is_load_selection.any():
                load_rows = b_idx_batt[is_load_selection]
                load_node_idx = selected_load[is_load_selection]
                
                # Head를 선택된 Load로 이동
                next_obs["trajectory_head"][load_rows, 0] = load_node_idx
                # '연결 안 됨' 마스크에서 제거
                next_obs["unconnected_loads_mask"][load_rows, load_node_idx] = False
                # 현재 경로의 최종 타겟으로 설정
                next_obs["current_target_load"][load_rows, 0] = load_node_idx
                # 경로 비용 초기화
                next_obs["staging_cost"][load_rows] = 0.0
                
                # '독립 레일' 상태 전파 시작
                # (B_load,)
                rail_status = self.rail_types_tensor[load_node_idx]
                next_obs["is_exclusive_mask"][load_rows, load_node_idx] = rail_status

        # --- 2b. [Find Parent / Spawn] ---
        # (현재 헤드가 Load 또는 IC일 때)
        head_is_node = ~head_is_battery
        if head_is_node.any():
            b_idx_node = batch_indices[head_is_node]
            child_node = current_head[head_is_node] # (B_node,)
            
            # (B_node,)
            is_connect = (action_type[head_is_node] == 0)
            is_spawn = ~is_connect
            
            # (B_node,) - 최종 부모가 될 노드의 인덱스
            parent_node = torch.zeros_like(child_node)
            
            # --- Connect 액션 처리 ---
            if is_connect.any():
                b_idx_connect = b_idx_node[is_connect]
                # 'Connect' 헤드에서 부모 인덱스 가져오기
                parent_connect = connect_target[b_idx_connect]
                parent_node[is_connect] = parent_connect
            
            # --- Spawn 액션 처리 ---
            if is_spawn.any():
                b_idx_spawn = b_idx_node[is_spawn]
                child_spawn = child_node[is_spawn]

                # 'Spawn' 헤드에서 템플릿 인덱스 가져오기
                template_idx = spawn_template[b_idx_spawn] # (B_spawn,)
                # 스폰될 빈 슬롯 인덱스 가져오기
                # NOTE:
                #   next_empty_slot_idx 텐서는 이후 스폰 로직에서 in-place로 +1 갱신된다.
                #   clone() 없이 그대로 사용하면, 아래에서 next_empty_slot_idx를 수정할 때
                #   autograd가 저장해 둔 slot_idx 인덱스가 변경되어 "modified by an inplace"
                #   오류가 발생한다. 안전하게 별도 복사본을 사용한다.
                slot_idx = next_obs["next_empty_slot_idx"][b_idx_spawn].clone().squeeze(-1) # (B_spawn,)

                # 1. Spawn: 템플릿 피처 -> 빈 슬롯으로 복사
                template_features = next_obs["nodes"][b_idx_spawn, template_idx]
                next_obs["nodes"][b_idx_spawn, slot_idx] = template_features.detach()

                # Spawn된 슬롯은 템플릿과 동일한 전압 호환성을 가져야 하므로
                # connectivity_matrix의 행/열을 템플릿에서 복사한다.
                connectivity_matrix = next_obs["connectivity_matrix"]
                connectivity_matrix[b_idx_spawn, :, slot_idx] = connectivity_matrix[b_idx_spawn, :, template_idx]
                connectivity_matrix[b_idx_spawn, slot_idx, :] = connectivity_matrix[b_idx_spawn, template_idx, :]

                # 2. 상태 변경: (Template -> Active)
                next_obs["nodes"][b_idx_spawn, slot_idx, FEATURE_INDEX["is_active"]] = 1.0
                next_obs["nodes"][b_idx_spawn, slot_idx, FEATURE_INDEX["is_template"]] = 0.0
                next_obs["nodes"][b_idx_spawn, slot_idx, FEATURE_INDEX["can_spawn_into"]] = 0.0

                # 🚨 수정 2: 노드 타입 One-Hot 필드를 IC 타입으로 확정합니다.
                node_type_idx_start = FEATURE_INDEX["node_type"][0]
                node_type_idx_end = FEATURE_INDEX["node_type"][1]

                # 기존 원-핫 인코딩 필드를 0으로 초기화
                next_obs["nodes"][b_idx_spawn, slot_idx, node_type_idx_start:node_type_idx_end] = 0.0
                # IC 타입(3)을 1.0으로 설정
                next_obs["nodes"][b_idx_spawn, slot_idx, node_type_idx_start + NODE_TYPE_IC] = 1.0

                # 3. 환경 동적 마스크 업데이트
                next_obs["is_active_mask"][b_idx_spawn, slot_idx] = True
                next_obs["is_template_mask"][b_idx_spawn, slot_idx] = False
                next_obs["can_spawn_into_mask"][b_idx_spawn, slot_idx] = False

                # 4. 다음 빈 슬롯 인덱스 +1 (In-place 방지: += 대신 = ... + 1 사용)
                # (Autograd 오류 수정: modified by an inplace operation)
                next_obs["next_empty_slot_idx"][b_idx_spawn] = \
                    next_obs["next_empty_slot_idx"][b_idx_spawn] + 1
                
                # 5. 스폰 비용(cost) 즉시 반영
                template_cost = next_obs["nodes"][b_idx_spawn, template_idx, FEATURE_INDEX["cost"]]
                staging_cost_increase = template_cost.unsqueeze(-1) # (B_spawn, 1)
                
                # staging_cost 및 current_cost에 스폰 비용 추가
                next_obs["staging_cost"][b_idx_spawn] += staging_cost_increase
                next_obs["current_cost"][b_idx_spawn] += staging_cost_increase
                
                # R_action 보상 (스폰 즉시)
                step_reward[b_idx_spawn] += REWARD_WEIGHT_ACTION * (-staging_cost_increase.squeeze(-1))
                
                # 'is_used_ic_mask'에 템플릿 인덱스 대신 *스폰된 슬롯 인덱스*를 기록
                next_obs["is_used_ic_mask"][b_idx_spawn, slot_idx] = True
                
                # 6. 최종 부모를 '스폰된 슬롯'으로 설정
                parent_node[is_spawn] = slot_idx

            # --- 3. 공통 연결 로직 (Connect/Spawn 공통) ---
            
            # 3a. 엣지 추가: (parent_node) -> (child_node)
            next_obs["adj_matrix"][b_idx_node, parent_node, child_node] = True
            # (T에도 엣지 추가: (child_node) -> (parent_node))
            next_obs["adj_matrix_T"][b_idx_node, child_node, parent_node] = True

            # 3b. '독립 레일' 상태 전파 (자식 -> 부모)
            child_status = next_obs["is_exclusive_mask"][b_idx_node, child_node] # (B_node,)
            if (child_status > 0).any():
                parent_status = next_obs["is_exclusive_mask"][b_idx_node, parent_node]

                # 'Path'(2)는 IC를 타고 계속 전파됨
                status_to_propagate = torch.where(
                    child_status == 2, 
                    child_status, 
                    torch.tensor(0, device=self.device, dtype=torch.long)
                )
                
                # 'Supplier'(1)는 Load에서 시작할 때만 전파됨
                is_child_load = (self.node_type_tensor[child_node] == NODE_TYPE_LOAD)
                status_from_supplier = torch.where(
                    (child_status == 1) & is_child_load,
                    child_status,
                    torch.tensor(0, device=self.device, dtype=torch.long)
                )

                status_from_child = torch.max(status_to_propagate, status_from_supplier)
                new_parent_status = torch.max(parent_status, status_from_child)
                next_obs["is_exclusive_mask"][b_idx_node, parent_node] = new_parent_status
                next_obs["nodes"][b_idx_node, parent_node, FEATURE_INDEX["independent_rail_type"]] = new_parent_status.float()

            # --- [추가] 3c. 'Always-On' 상태 전파 (자식 -> 부모) ---
            # 자식이 잘 때도 켜져야 한다면(AO), 부모도 무조건 켜져 있어야 함(AO).
            # 모델에게 "이 부모는 이미 깨어있음(Penalty Pre-paid)"을 알려주어 그룹화를 유도함.
            child_is_ao = next_obs["nodes"][b_idx_node, child_node, FEATURE_INDEX["always_on_in_sleep"]]

            # (이미 1.0이면 유지, 아니면 자식 상태에 따라 1.0으로 변경)
            current_parent_ao = next_obs["nodes"][b_idx_node, parent_node, FEATURE_INDEX["always_on_in_sleep"]]
            new_parent_ao = torch.max(current_parent_ao, child_is_ao)
            next_obs["nodes"][b_idx_node, parent_node, FEATURE_INDEX["always_on_in_sleep"]] = new_parent_ao

            # 3d. 다음 Head 설정
            parent_is_battery = (parent_node == BATTERY_NODE_IDX)

            # 헤드(parent_node)가 이미 부모를 가졌는지 확인
            # adj_matrix_T[b, node, :]가 1이라도 있으면, node는 이미 부모가 있음
            parent_already_has_parent = next_obs["adj_matrix_T"][b_idx_node, parent_node].any(dim=-1)

            # 배터리에 도달하거나, 이미 연결된 노드에 도달하면 경로 완성
            path_is_finished = parent_is_battery | parent_already_has_parent

            next_obs["trajectory_head"][b_idx_node, 0] = torch.where(
                path_is_finished,  # 💡 조건 변경
                BATTERY_NODE_IDX,  # 경로가 끝났으면 배터리로 복귀
                parent_node        # 아니면 경로 추적 계속
            )
              
            # 3e. 경로 완성 (R_path 보상) [수정]
            # parent_is_battery 대신 path_is_finished 조건을 사용하여
            # 기존 트리에 연결된 경우(parent_already_has_parent)에도 비용을 정산하도록 변경


            if path_is_finished.any():
                finished_rows = b_idx_node[path_is_finished]
                
                # 경로 완성 시, 누적된 staging_cost(BOM)를 R_path 보상으로 추가
                sub_trajectory_total_cost = next_obs["staging_cost"][finished_rows]
                
                # [수정] 가격에 가중치(WEIGHT_BOM)를 곱해 민감도 증가
                # Cost가 높을수록(-) Reward는 낮아짐
                weighted_cost = WEIGHT_BOM * sub_trajectory_total_cost.squeeze(-1)
                
                r_bom = - REWARD_WEIGHT_PATH * weighted_cost
                step_reward[finished_rows] += r_bom
                bom_reward_batch[finished_rows] = r_bom.unsqueeze(-1) # 기록     

                # staging_cost 리셋
                next_obs["staging_cost"][finished_rows] = 0.0
                next_obs["current_target_load"][finished_rows, 0] = -1

        # --- 4. 전력/발열 재계산 (연산 비용 높음) ---
        # (모든 배치가 최소 1스텝 이상 진행했을 때만 계산)
        if td["step_count"].min() > 0 or head_is_node.any():
            final_i_out, power_loss, new_temp = self._calculate_tree_loads(
                next_obs["nodes"], 
                next_obs["adj_matrix"],
                next_obs["adj_matrix_T"] # 💡 adj_matrix_T 전달
            )
            next_obs["nodes"][..., FEATURE_INDEX["current_out"]] = final_i_out
            next_obs["nodes"][..., FEATURE_INDEX["junction_temp"]] = new_temp
        
        next_obs.set("step_count", td["step_count"] + 1)

        # --- 5. 종료 조건 확인 ---
        # (get_action_mask가 3종 마스크를 모두 반환한다고 가정)
        next_masks = self.get_action_mask(next_obs)
        # Connect/Spawn 둘 다 불가능한 경우
        is_stuck = ~(next_masks["mask_type"].any(dim=-1))
        
        all_loads_connected = (next_obs["unconnected_loads_mask"].sum(dim=1) == 0)
        trajectory_finished = (next_obs["trajectory_head"].squeeze(-1) == BATTERY_NODE_IDX)
        
        done_successfully = all_loads_connected & trajectory_finished
        max_steps = 2 * self.N_max # 최대 스텝 제한
        timed_out = (next_obs["step_count"] > max_steps).squeeze(-1)
        
        is_done = done_successfully | timed_out | is_stuck
        next_obs["done"] = is_done.unsqueeze(-1)

        # --- 6. 최종 보상 계산---
        final_reward, fail_penalty_val = self.get_reward(
            next_obs, step_reward, done_successfully, timed_out, is_stuck
        )
        next_obs["log_reward_bom"] = bom_reward_batch
        next_obs["log_reward_fail"] = fail_penalty_val.unsqueeze(-1)

        # 이미 'done'이었던 샘플은 보상 0, 상태 롤백
        if is_already_done.any():
            final_reward[is_already_done] = 0.0
            next_obs[is_already_done] = td[is_already_done]

        return TensorDict({
            "next": next_obs,
            "reward": final_reward.unsqueeze(-1),
            "done": next_obs["done"],
        }, batch_size=batch_size)
        
    def get_reward(self,
                   td: TensorDict,
                   step_reward: torch.Tensor,
                   done_successfully: torch.Tensor,
                   timed_out: torch.Tensor,
                   is_stuck: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]: # [수정] 리턴 타입 변경

        """
        [수정] 암전류 제약 위반 페널티 제거 (하드 제약으로 변경됨)
        """
        reward = step_reward.clone()
        fail_penalty_record = torch.zeros_like(reward)

        # 실패한 경우만 고정 페널티 (성공 시 암전류 위반은 마스킹으로 방지됨)
        failed = (timed_out | is_stuck) & ~done_successfully
        if failed.any():
            reward[failed] = FAILURE_PENALTY
            fail_penalty_record[failed] = FAILURE_PENALTY
            
        return reward, fail_penalty_record

    # ---
    # 섹션 5: 액션 마스킹 (연산 집약적)
    # ---
    @torch.no_grad()
    def get_action_mask(self, td: TensorDict, debug: bool = False) -> Dict[str, torch.Tensor]:
        self._ensure_buffers(td)

        batch_size = td.batch_size[0]
        num_nodes = self.N_max
        current_head = td["trajectory_head"].squeeze(-1) 

        is_active = td["is_active_mask"]
        is_template = td["is_template_mask"]
        
        mask_type = torch.zeros(batch_size, 2, dtype=torch.bool, device=self.device)
        mask_connect = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)
        mask_spawn = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)
        
        reasons = {} # [Debug] 디버그 정보 저장용 딕셔너리

        # --- 2. [Select New Load] (Phase Logic) ---
        head_is_battery = (current_head == BATTERY_NODE_IDX)

        if head_is_battery.any():
            b_idx_batt = torch.where(head_is_battery)[0]
            
            mask_type[b_idx_batt, 0] = True 
            
            # 1. 일단 연결 안 된 모든 부하 후보
            candidate_loads = td["unconnected_loads_mask"][b_idx_batt] # (B, N)
            
            # 2. [Phase 1 강제] Always-On 부하 우선 정책
            ao_feature = td["nodes"][b_idx_batt, :, FEATURE_INDEX["always_on_in_sleep"]]
            ao_unconnected = candidate_loads & (ao_feature == 1.0)
            has_ao_remaining = ao_unconnected.any(dim=-1) # (B,)
            
            # 3. AO 부하가 남아있는 배치는 AO 부하만 선택 가능하도록 필터링
            final_candidates = candidate_loads.clone()
            
            if has_ao_remaining.any():
                idx_with_ao = torch.where(has_ao_remaining)[0] 
                not_ao_mask = (ao_feature[idx_with_ao] != 1.0)
                final_candidates[idx_with_ao] &= ~not_ao_mask

            mask_connect[b_idx_batt] = final_candidates
            
            # 종료 조건 (모든 로드 연결 완료 시 배터리 선택)
            all_connected = (candidate_loads.sum(dim=-1) == 0)
            if all_connected.any():
                b_idx_finish = b_idx_batt[all_connected]
                mask_connect[b_idx_finish, BATTERY_NODE_IDX] = True
        

        # --- 3. [Find Parent / Spawn] (Hard Constraints) ---
        head_is_node = ~head_is_battery
        if head_is_node.any():
            b_idx_node = torch.where(head_is_node)[0]
            child_nodes = current_head[b_idx_node]
            B_act = len(b_idx_node)
            
            # 3a. 기본 제약 (전압, 사이클, 독점, 시퀀스)
            connectivity = td["connectivity_matrix"][b_idx_node]
            volt_ok = connectivity[torch.arange(B_act), :, child_nodes]
            path_mask = self._trace_path_batch(child_nodes, td["adj_matrix_T"][b_idx_node])
            cycle_ok = ~path_mask            
            exclusive_ok = self._get_exclusive_mask(td, b_idx_node, child_nodes)
            power_seq_ok = self._get_power_sequence_mask(td["adj_matrix"][b_idx_node], child_nodes, td, b_idx_node)
            
            base_valid_parents = volt_ok & cycle_ok & exclusive_ok & power_seq_ok
            
            # 3b. Thermal Simulation Mask
            thermal_current_ok = self._get_thermal_current_mask(
                td, b_idx_node, child_nodes, base_valid_parents
            )

            # 3c. [Sleep Logic] Sleep Current Simulation Mask (Hard Constraint)
            # Thermal까지 통과한 후보들에 대해 암전류 예산 검사
            # (debug를 위해 변수를 분리했습니다)
            sleep_current_ok = self._get_sleep_current_mask(
                td, b_idx_node, child_nodes, base_valid_parents & thermal_current_ok
            )
            
            # 최종 유효 부모 (모든 하드 제약 통과)
            # (sleep_current_ok 함수가 이미 base & thermal 조건을 포함하여 계산하지만 명시적으로 AND 연산)
            final_valid_parents = sleep_current_ok 
            
            # 3d. 마스크 적용
            mask_connect[b_idx_node] = final_valid_parents & is_active[b_idx_node]
            mask_spawn[b_idx_node] = final_valid_parents & is_template[b_idx_node]
            
            can_connect = mask_connect[b_idx_node].any(dim=-1)
            has_empty_slots = (td["next_empty_slot_idx"][b_idx_node] < self.N_max)
            can_spawn = mask_spawn[b_idx_node].any(dim=-1) & has_empty_slots.squeeze(-1)
            
            mask_type[b_idx_node, 0] = can_connect
            mask_type[b_idx_node, 1] = can_spawn

            # --- [Debug Info Collection] ---
            if debug:
                # 0번 배치 샘플이 현재 노드 선택 단계에 있다면 정보를 수집
                if 0 in b_idx_node:
                    # b_idx_node 내에서 0번 배치의 로컬 인덱스 찾기
                    local_idx = (b_idx_node == 0).nonzero(as_tuple=True)[0].item()
                    
                    reasons["volt_ok"] = volt_ok[local_idx]
                    reasons["cycle_ok"] = cycle_ok[local_idx]
                    reasons["exclusive_ok"] = exclusive_ok[local_idx]
                    reasons["power_seq_ok"] = power_seq_ok[local_idx]
                    reasons["base_valid_parents"] = base_valid_parents[local_idx]
                    reasons["thermal_current_ok"] = thermal_current_ok[local_idx]
                    # [New] 암전류 마스크 정보 추가
                    reasons["sleep_current_ok"] = sleep_current_ok[local_idx]
                    reasons["final_valid_parents"] = final_valid_parents[local_idx]

        # ✅ [Return] debug 여부에 따라 분기
        if debug:
            return {
                "mask_type": mask_type,
                "mask_connect": mask_connect,
                "mask_spawn": mask_spawn,
                "reasons": reasons # 디버그 정보 포함
            }
        else:
            return {
                "mask_type": mask_type,
                "mask_connect": mask_connect,
                "mask_spawn": mask_spawn,
            }
    # ---
    # 섹션 6: 마스킹 헬퍼 함수 
    # ---

    def _trace_path_batch(self, start_nodes: torch.Tensor, adj_matrix_T: torch.Tensor) -> torch.Tensor:
        """ start_nodes의 모든 조상(ancestors)을 찾아 마스크로 반환 (사이클 방지용) """
        batch_size, num_nodes, _ = adj_matrix_T.shape
        path_mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=self.device)

        if start_nodes.numel() > 0:
            path_mask.scatter_(1, start_nodes.unsqueeze(-1), True)

        adj_matrix_T_float = adj_matrix_T.float()

        for _ in range(num_nodes):
            # (B,N,N) @ (B,N,1) -> (B,N)
            parents_mask = (adj_matrix_T_float @ path_mask.float().unsqueeze(-1)).squeeze(-1).bool()
            if (parents_mask & ~path_mask).sum() == 0: break
            path_mask |= parents_mask
            
        return path_mask

    def _get_exclusive_mask(self, 
                            td: TensorDict,
                            b_idx_node: torch.Tensor, 
                            child_nodes: torch.Tensor
                            ) -> torch.Tensor:
        """ 독립 레일(Exclusive Rail) 제약조건 마스크 생성 """
        # (td와 b_idx_node에서 필요한 텐서를 가져옴)
        is_exclusive_mask_batch = td["is_exclusive_mask"][b_idx_node]
        adj_matrix_batch = td["adj_matrix"][b_idx_node]
        B_act, N_nodes = is_exclusive_mask_batch.shape
        
        # 1. Head(Child)의 상태adj_matrix_T
        head_status = is_exclusive_mask_batch[torch.arange(B_act), child_nodes] # (B_act,)
        node_type_indices_full = td["nodes"][b_idx_node, child_nodes, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        head_is_load = (node_type_indices_full == NODE_TYPE_LOAD) # (B_act,)

        # 2. Parent(후보)의 상태
        parent_status = is_exclusive_mask_batch # (B_act, N_nodes)
        parent_is_exclusive = (parent_status > 0)

        # (B_act, N_nodes) - 부모가 이미 자식을 가졌는지?
        parent_has_any_child = adj_matrix_batch.any(dim=-1)

        
        # 3. 위반(Violation) 규칙 (True = 위반 = 금지)
        
        # 규칙 1: Head가 'Path'(2) -> 부모는 자식이 없어야 함
        # (B_act, 1) & (B_act, N_nodes) -> (B_act, N_nodes)
        violation_Rule1 = (head_status.unsqueeze(-1) == 2) & parent_has_any_child

        # 규칙 2: Head가 'Supplier Load'(1) -> 부모는 자식이 없어야 함
        violation_Rule2 = ((head_status == 1) & head_is_load).unsqueeze(-1) & parent_has_any_child

        # 규칙 3: Head가 'Normal'(0) 또는 'Supplier IC'(1) -> 부모는 Exclusive이면 안 됨
        violation_Rule3 = ((head_status == 0) | ((head_status == 1) & ~head_is_load)).unsqueeze(-1) & parent_is_exclusive
        
        violations = violation_Rule1 | violation_Rule2 | violation_Rule3
        
        # 4. 배터리는 항상 허용
        is_battery_mask = (self.arange_nodes.unsqueeze(0) == BATTERY_NODE_IDX)
        exclusive_ok = torch.logical_not(violations) | is_battery_mask

        return exclusive_ok

    def _get_power_sequence_mask(self, 
                                 adj_matrix_batch: torch.Tensor, 
                                 child_nodes: torch.Tensor, 
                                 td: TensorDict, 
                                 b_idx_node: torch.Tensor
                                 ) -> torch.Tensor:
        """ 전원 시퀀싱(Power Sequence) 제약조건 마스크 생성 """
        B_act, N_nodes, _ = adj_matrix_batch.shape
        adj_matrix_T_batch = td["adj_matrix_T"][b_idx_node]
        candidate_mask = torch.ones(B_act, N_nodes, dtype=torch.bool, device=self.device)

        for j_idx, k_idx, f_flag in self.power_sequences:
            # Case 1: 현재 child가 'k' (j의 부모를 찾는 중)
            is_k_mask = (child_nodes == k_idx)
            if is_k_mask.any():
                b_idx_check = torch.where(is_k_mask)[0] # (B_k,)
                # 'j'의 부모가 이미 존재하는가?
                parent_of_j_exists = adj_matrix_batch[b_idx_check, :, j_idx].any(dim=-1) # (B_k,)

                if parent_of_j_exists.any():
                    b_constr = b_idx_check[parent_of_j_exists] # (B_constr,)
                    # 'j'의 부모 인덱스 
                    parent_of_j_idx = adj_matrix_batch[b_constr, :, j_idx].long().argmax(-1) # (B_constr,)
                    # 'j'의 부모(parent_of_j)의 모든 조상(ancestors)을 찾음
                    anc_mask = self._trace_path_batch(parent_of_j_idx, adj_matrix_T_batch[b_constr])
                    anc_mask[:, BATTERY_NODE_IDX] = False # 배터리 제외
                    # 'k'의 부모는 'j'의 조상이 될 수 없음
                    candidate_mask[b_constr] &= ~anc_mask
                    # (f=1) 'k'의 부모는 'j'의 부모와 같을 수 없음
                    if f_flag == 1:
                        same_parent_mask = (self.arange_nodes == parent_of_j_idx.unsqueeze(1))
                        candidate_mask[b_constr] &= ~same_parent_mask

            # Case 2: 현재 child가 'j' (k의 부모를 찾는 중)
            is_j_mask = (child_nodes == j_idx)
            if is_j_mask.any():
                b_idx_check = torch.where(is_j_mask)[0]
                parent_of_k_exists = adj_matrix_batch[b_idx_check, :, k_idx].any(dim=-1)

                if parent_of_k_exists.any():
                    b_constr = b_idx_check[parent_of_k_exists]
                    parent_of_k_idx = adj_matrix_batch[b_constr, :, k_idx].long().argmax(-1)

                    anc_mask = self._trace_path_batch(parent_of_k_idx, adj_matrix_T_batch[b_constr])
                    anc_mask[:, BATTERY_NODE_IDX] = False
                    # 'j'의 부모는 'k'의 조상이 될 수 없음
                    candidate_mask[b_constr] &= ~anc_mask
                    if f_flag == 1:
                        same_parent_mask = (self.arange_nodes == parent_of_k_idx.unsqueeze(1))
                        candidate_mask[b_constr] &= ~same_parent_mask
        return candidate_mask
    
    @torch.no_grad()
    def _get_thermal_current_mask(self,
                                  td: TensorDict,
                                  b_idx_node: torch.Tensor,
                                  child_nodes: torch.Tensor,
                                  base_valid_parents: torch.Tensor) -> torch.Tensor:
        """
        전류/발열 한계를 만족하는지 시뮬레이션하여 마스크 생성.
        (연산 비용이 가장 높은 함수)
        """
        
        # (B_act, N_max)
        thermal_current_ok = base_valid_parents.clone()
        # 시뮬레이션 청크 크기 (메모리/속도 트레이드오프)
        SIM_CHUNK_SIZE = 8

        B_act, N_nodes = base_valid_parents.shape

        base_nodes = td["nodes"][b_idx_node]
        base_adj_matrix = td["adj_matrix"][b_idx_node]
        base_adj_matrix_T = td["adj_matrix_T"][b_idx_node]
        
        # 마진(Margin) 값 미리 로드
        margin_I = float(self.generator.config.constraints.get("current_margin", 0.0))
        THERMAL_MARGIN_DEG = float(self.generator.config.constraints.get("thermal_margin_deg", 5.0))        

        # (N_max) 크기의 청크로 나누어 시뮬레이션
        for chunk_start in range(0, N_nodes, SIM_CHUNK_SIZE):
            chunk_end = min(chunk_start + SIM_CHUNK_SIZE, N_nodes)
            parent_indices_in_chunk = torch.arange(chunk_start, chunk_end, device=self.device)

            # (B_act, N_chunk) - 이번 청크에서 시뮬레이션할 (배치, 부모) 후보
            candidates_in_chunk_mask = base_valid_parents[:, chunk_start:chunk_end]
            
            # (N_sim,) - (B_act 기준 인덱스, 로컬 부모 인덱스)
            b_idx_sim_chunk, p_idx_sim_chunk_local = candidates_in_chunk_mask.nonzero(as_tuple=True)

            if b_idx_sim_chunk.numel() == 0:
                continue # 시뮬레이션할 후보 없음
            
            N_sim = b_idx_sim_chunk.numel()

            # 1. 시뮬레이션 데이터 준비 (N_sim,)
            sim_nodes = base_nodes[b_idx_sim_chunk]
            sim_adj_matrix = base_adj_matrix[b_idx_sim_chunk].clone()
            sim_adj_matrix_T = base_adj_matrix_T[b_idx_sim_chunk].clone()
            sim_child_nodes = child_nodes[b_idx_sim_chunk]

            # (N_sim,) - 실제 부모 노드 인덱스
            sim_parent_indices_global = parent_indices_in_chunk[p_idx_sim_chunk_local]
            
            # 2. (가상) 엣지 추가: (parent) -> (child)
            sim_rows = torch.arange(N_sim, device=self.device)
            sim_adj_matrix[sim_rows, sim_parent_indices_global, sim_child_nodes] = True
            sim_adj_matrix_T[sim_rows, sim_child_nodes, sim_parent_indices_global] = True

            # 3. 🚀 트리 전체 부하 시뮬레이션
            (final_i_out, power_loss, junction_temp) = self._calculate_tree_loads(
                sim_nodes,
                sim_adj_matrix, 
                sim_adj_matrix_T
            )

            # 4. 시뮬레이션 결과 검증
            i_limit = sim_nodes[..., FEATURE_INDEX["i_limit"]] * (1.0 - margin_I)
            t_max_raw = sim_nodes[..., FEATURE_INDEX["t_junction_max"]]
            t_max = t_max_raw - THERMAL_MARGIN_DEG            
            
            # (N_sim, N_max)
            current_check_ok = (final_i_out <= i_limit + 1e-6)
            temp_check_ok = (junction_temp <= t_max + 1e-6)
            
            # (N_sim, N_max)
            all_checks_ok = current_check_ok & temp_check_ok
            
            # (N_sim, N_max) - IC가 아닌 노드는 항상 OK
            ic_type_indices = sim_nodes[..., FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
            ic_mask_sim = (ic_type_indices == NODE_TYPE_IC)   

            # [수정] 검증 대상 필터링: "현재 Active 노드" + "이번 시뮬레이션의 후보 부모"만 검사
            # (사용하지 않는 다른 템플릿들이 고온에서 FAIL 뜨는 것을 무시하기 위함)
            relevant_mask = td["is_active_mask"][b_idx_sim_chunk].clone()
            relevant_mask[sim_rows, sim_parent_indices_global] = True
            is_valid_simulation = (all_checks_ok | ~ic_mask_sim | ~relevant_mask).all(dim=-1)

            failed_sim_mask = ~is_valid_simulation
            if failed_sim_mask.any():
                b_idx_failed = b_idx_sim_chunk[failed_sim_mask]
                p_idx_failed_global = sim_parent_indices_global[failed_sim_mask]
                thermal_current_ok[b_idx_failed, p_idx_failed_global] = False
                
        return thermal_current_ok

    # ---
    # 섹션 7: 계산 헬퍼 함수 
    # ---
    @torch.no_grad()
    def _get_sleep_current_mask(self, td: TensorDict, b_idx_node: torch.Tensor, child_nodes: torch.Tensor, base_valid_parents: torch.Tensor) -> torch.Tensor:
        """
        [추가] 암전류 제약 검증용 마스킹 함수
        가상으로 연결 후 Total Sleep Current가 예산(Budget)을 초과하면 마스킹합니다.
        """
        sleep_ok = base_valid_parents.clone()
        SIM_CHUNK_SIZE = 8 # 메모리 문제 시 조절
        
        B_act, N_nodes = base_valid_parents.shape
        base_nodes = td["nodes"][b_idx_node]
        base_adj_matrix = td["adj_matrix"][b_idx_node]
        base_adj_matrix_T = td["adj_matrix_T"][b_idx_node]
        
        # Budget 가져오기
        # scalar_prompt_features[1] = max_sleep_current
        max_sleep_current = td["scalar_prompt_features"][b_idx_node, 1]

        for chunk_start in range(0, N_nodes, SIM_CHUNK_SIZE):
            chunk_end = min(chunk_start + SIM_CHUNK_SIZE, N_nodes)
            parent_indices_in_chunk = torch.arange(chunk_start, chunk_end, device=self.device)
            candidates_in_chunk_mask = base_valid_parents[:, chunk_start:chunk_end]
            
            b_idx_sim_chunk, p_idx_sim_chunk_local = candidates_in_chunk_mask.nonzero(as_tuple=True)
            if b_idx_sim_chunk.numel() == 0: continue
            
            N_sim = b_idx_sim_chunk.numel()
            sim_adj_matrix = base_adj_matrix[b_idx_sim_chunk].clone()
            sim_adj_matrix_T = base_adj_matrix_T[b_idx_sim_chunk].clone()
            sim_child_nodes = child_nodes[b_idx_sim_chunk]
            sim_parent_indices_global = parent_indices_in_chunk[p_idx_sim_chunk_local]
            
            # 가상 연결
            sim_rows = torch.arange(N_sim, device=self.device)
            sim_adj_matrix[sim_rows, sim_parent_indices_global, sim_child_nodes] = True
            sim_adj_matrix_T[sim_rows, sim_child_nodes, sim_parent_indices_global] = True
            
            # 노드 정보 (Spawn시 마스크 업데이트 등은 _calculate_total_sleep_current 내부에서
            # is_used_ic_mask 등을 참조하므로, 여기서는 노드 Feature만으로 충분한지 확인 필요)
            # _calculate_total_sleep_current는 is_used_ic_mask를 사용하여 Self-consumption 계산.
            # Spawn의 경우 새 노드가 'Used'가 되어야 전류가 계산됨.
            
            # Spawn 후보인 경우(is_active가 False인 경우), 가상으로 Used 마스크를 켜야 함
            # 여기서는 편의상 시뮬레이션용 Used Mask를 생성
            sim_is_used_mask = td["is_used_ic_mask"][b_idx_node][b_idx_sim_chunk].clone()
            sim_is_used_mask[sim_rows, sim_parent_indices_global] = True # 후보 부모(템플릿 포함)를 Used로 처리
            
            sim_nodes = base_nodes[b_idx_sim_chunk] # Feature는 그대로 사용 (Spawn 템플릿도 Feature 보유)

            # 암전류 계산
            total_sleep = self._calculate_total_sleep_current(
                nodes=sim_nodes, 
                adj_matrix=sim_adj_matrix, 
                adj_matrix_T=sim_adj_matrix_T,
                is_used_ic_mask=sim_is_used_mask 
            )
            
            # Budget 체크
            budget = max_sleep_current[b_idx_sim_chunk]
            # 아주 약간의 오차 허용 (1e-9)
            is_valid = (total_sleep <= budget + 1e-9)
            
            failed_sim_mask = ~is_valid
            if failed_sim_mask.any():
                b_idx_failed = b_idx_sim_chunk[failed_sim_mask]
                p_idx_failed_global = sim_parent_indices_global[failed_sim_mask]
                sleep_ok[b_idx_failed, p_idx_failed_global] = False
                
        return sleep_ok

    @torch.no_grad()
    def _calculate_tree_loads(self, 
                              nodes_tensor: torch.Tensor, 
                              adj_matrix: torch.Tensor,
                              adj_matrix_T: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Adjacency Matrix를 기반으로 트리 전체의 전류/전력손실/온도를 계산합니다. """
        
        batch_size, num_nodes, _ = nodes_tensor.shape
        
        # 1. 초기 수요 = Load의 활성 전류
        current_demands = nodes_tensor[..., FEATURE_INDEX["current_active"]].clone()
        load_mask_1d = (self.node_type_tensor == NODE_TYPE_LOAD)
        current_demands[:, ~load_mask_1d] = 0.0
        
        adj_matrix_float = adj_matrix.float()
        
        ic_type = nodes_tensor[..., FEATURE_INDEX["ic_type_idx"]]
        ldo_mask_b = torch.isclose(ic_type, torch.tensor(1.0, device=ic_type.device))
        buck_mask_b = torch.isclose(ic_type, torch.tensor(2.0, device=ic_type.device))
        
        op_current = nodes_tensor[..., FEATURE_INDEX["op_current"]]
        vout = nodes_tensor[..., FEATURE_INDEX["vout_min"]]
        vin = nodes_tensor[..., FEATURE_INDEX["vin_min"]]
        safe_vin = torch.where(vin > 0, vin, 1e-6)
        eff_active = nodes_tensor[..., FEATURE_INDEX["efficiency_active"]]
        safe_eff_active = torch.where(eff_active > 0, eff_active, 0.9)
        min_fb_res = nodes_tensor[..., FEATURE_INDEX["min_fb_res"]]
        fb_current = torch.zeros_like(vout)
        valid_res_mask = (min_fb_res > 1e-6)
        fb_current[valid_res_mask] = vout[valid_res_mask] / min_fb_res[valid_res_mask]

        i_in_total = current_demands.clone() 

        for _ in range(num_nodes):
            i_out = (adj_matrix_float @ i_in_total.unsqueeze(-1)).squeeze(-1)            
            i_in_ldo = i_out + fb_current + op_current
            p_out_buck = vout * (i_out + fb_current)
            i_in_buck = (p_out_buck / safe_eff_active) / safe_vin
            new_ic_demands = torch.zeros_like(i_in_total)
            new_ic_demands[ldo_mask_b] = i_in_ldo[ldo_mask_b]
            new_ic_demands[buck_mask_b] = i_in_buck[buck_mask_b]
            new_i_in_total = current_demands + new_ic_demands
            if torch.allclose(i_in_total, new_i_in_total, atol=1e-8): break
            i_in_total = new_i_in_total

        i_out = (adj_matrix_float @ i_in_total.unsqueeze(-1)).squeeze(-1)
        power_loss = self._calculate_power_loss(nodes_tensor, i_out, ldo_mask_b, buck_mask_b)
        theta_ja = nodes_tensor[..., FEATURE_INDEX["theta_ja"]]
        ambient_temp = self.generator.config.constraints.get("ambient_temperature", 25.0)
        junction_temp = ambient_temp + power_loss * theta_ja
        
        return i_out, power_loss, junction_temp

    def _calculate_power_loss(self, 
                              ic_node_features: torch.Tensor, 
                              i_out: torch.Tensor,
                              ldo_mask: torch.Tensor,
                              buck_mask: torch.Tensor
                              ) -> torch.Tensor:
        """ I_out을 기반으로 IC의 전력 손실(P_loss)을 계산합니다. """
        vin = ic_node_features[..., FEATURE_INDEX["vin_min"]]
        vout = ic_node_features[..., FEATURE_INDEX["vout_min"]]
        op_current = ic_node_features[..., FEATURE_INDEX["op_current"]]
        min_fb_res = ic_node_features[..., FEATURE_INDEX["min_fb_res"]]
        fb_current = torch.zeros_like(vout)
        valid_res_mask = (min_fb_res > 1e-6)
        fb_current[valid_res_mask] = vout[valid_res_mask] / min_fb_res[valid_res_mask]

        power_loss = torch.zeros_like(i_out, dtype=ic_node_features.dtype)
        if ldo_mask.any():
            power_loss[ldo_mask] = (vin[ldo_mask] - vout[ldo_mask]) * (i_out[ldo_mask] + fb_current[ldo_mask]) + \
                                   vin[ldo_mask] * op_current[ldo_mask]
        if buck_mask.any():
            eff_active = ic_node_features[buck_mask, FEATURE_INDEX["efficiency_active"]]
            safe_eff = torch.where(eff_active > 0, eff_active, 0.9)
            p_out_buck = vout[buck_mask] * (i_out[buck_mask] + fb_current[buck_mask])
            conversion_loss = (p_out_buck / safe_eff) - p_out_buck
            power_loss[buck_mask] = conversion_loss 
        return power_loss
    
    @torch.no_grad()
    def _calculate_total_sleep_current(self, nodes=None, adj_matrix=None, adj_matrix_T=None, is_used_ic_mask=None, td=None) -> torch.Tensor:
        """ 
        [수정] 인자를 직접 받을 수 있도록 변경하여 시뮬레이션 지원 
        """
        if td is not None:
            nodes = td["nodes"]
            adj_matrix = td["adj_matrix"]
            adj_matrix_T = td["adj_matrix_T"]
            is_used_ic_mask = td["is_used_ic_mask"]

        batch_size, num_nodes, _ = nodes.shape
        adj_matrix = adj_matrix.float()
        adj_matrix_T = adj_matrix_T.float()

        # 1. "Always-On" 상태 전파
        always_on_loads = (nodes[..., FEATURE_INDEX["always_on_in_sleep"]] == 1.0)
        always_on_nodes = always_on_loads.clone()
        always_on_nodes[:, BATTERY_NODE_IDX] = True
        
        for _ in range(num_nodes):
            parents_mask = (adj_matrix @ always_on_nodes.float().unsqueeze(-1)).squeeze(-1).bool()
            if (parents_mask & ~always_on_nodes).sum() == 0: break
            always_on_nodes |= parents_mask
        
        # 2. IC 자체 암전류
        is_ao = always_on_nodes
        is_used = is_used_ic_mask
        parent_is_ao = (adj_matrix_T @ is_ao.float().unsqueeze(-1)).squeeze(-1).bool()

        quiescent_current = nodes[..., FEATURE_INDEX["quiescent_current"]]
        shutdown_current = nodes[..., FEATURE_INDEX["shutdown_current"]]
        
        use_ishut_current = torch.where(shutdown_current > 1e-9, shutdown_current, quiescent_current)
        ic_self_sleep = torch.zeros(batch_size, num_nodes, device=self.device)
        
        # AO 상태: Iq
        ic_self_sleep[is_ao & is_used] = quiescent_current[is_ao & is_used]
        # 비-AO지만 부모 On: Shutdown
        ic_self_sleep[~is_ao & is_used & parent_is_ao] = use_ishut_current[~is_ao & is_used & parent_is_ao]

        # 피드백 전류
        vout = nodes[..., FEATURE_INDEX["vout_min"]]
        min_fb_res = nodes[..., FEATURE_INDEX["min_fb_res"]]
        fb_current_draw = torch.zeros_like(vout)
        valid_res_mask = (min_fb_res > 1e-6)
        fb_current_draw[valid_res_mask] = vout[valid_res_mask] / min_fb_res[valid_res_mask]
        fb_current_draw = fb_current_draw * always_on_nodes.float()
        
        shut_mask = ~is_ao & is_used & parent_is_ao
        ic_self_sleep[shut_mask] = shutdown_current[shut_mask]

        # 3. Load 암전류
        load_sleep_draw_base = nodes[..., FEATURE_INDEX["current_sleep"]].clone()
        load_sleep_draw = load_sleep_draw_base * always_on_nodes.float()
        load_sleep_draw[~always_on_nodes] = 0.0

        # 4. 전류 수요 전파
        current_demands_sleep = load_sleep_draw + ic_self_sleep
        
        ic_type = nodes[..., FEATURE_INDEX["ic_type_idx"]]
        ldo_mask_b = torch.isclose(ic_type, torch.tensor(1.0, device=ic_type.device))
        buck_mask_b = torch.isclose(ic_type, torch.tensor(2.0, device=ic_type.device))
        
        vin = nodes[..., FEATURE_INDEX["vin_min"]]
        safe_vin = torch.where(vin > 0, vin, 1e-6)
        eff_sleep_tensor = nodes[..., FEATURE_INDEX["efficiency_sleep"]]
        safe_eff_sleep = torch.where(eff_sleep_tensor > 0, eff_sleep_tensor, 0.35)
        
        for _ in range(num_nodes):
            i_out_sleep = (adj_matrix_T.transpose(-1, -2) @ current_demands_sleep.unsqueeze(-1)).squeeze(-1)
            new_demands_sleep = load_sleep_draw + ic_self_sleep
            
            # LDO
            ldo_demand = i_out_sleep[ldo_mask_b] + fb_current_draw[ldo_mask_b]
            new_demands_sleep[ldo_mask_b] += ldo_demand            

            # Buck
            i_out_buck = i_out_sleep[buck_mask_b]
            if_fb_buck = fb_current_draw[buck_mask_b]
            p_out_total_buck = vout[buck_mask_b] * (i_out_buck + if_fb_buck)
            eff = safe_eff_sleep[buck_mask_b]
            i_in_switching_buck = (p_out_total_buck / eff) / safe_vin[buck_mask_b]
            new_demands_sleep[buck_mask_b] += i_in_switching_buck
            
            if torch.allclose(current_demands_sleep, new_demands_sleep, atol=1e-8): break
            current_demands_sleep = new_demands_sleep

        # 5. 배터리 총 암전류
        battery_children_mask = adj_matrix[:, BATTERY_NODE_IDX, :]
        total_sleep_current = (current_demands_sleep * battery_children_mask).sum(dim=1)
        
        return total_sleep_current