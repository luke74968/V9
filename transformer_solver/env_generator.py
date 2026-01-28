# Copyright (c) 2025 Minuk Lee. All rights reserved.
# 
# This source code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# 
# For licensing terms, see the LICENSE file.
# Contact: minuklee@snu.ac.kr
# 
import json
import random
import numpy as np  
import torch
from tensordict import TensorDict
import math
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# --- 공용(common) 모듈 임포트 ---
from common.config_loader import load_configuration_from_file
from common.ic_preprocessor import expand_ic_templates, prune_dominated_ics
from common.data_classes import Battery, LDO, BuckConverter, Load

# --- 현재 패키지(transformer_solver) 모듈 임포트 ---
from .definitions import (
    PocatConfig, FEATURE_DIM, FEATURE_INDEX, SCALAR_PROMPT_FEATURE_DIM,
    NODE_TYPE_PADDING, NODE_TYPE_BATTERY, NODE_TYPE_LOAD, 
    NODE_TYPE_IC, NODE_TYPE_EMPTY
)

class PocatGenerator:
    """
    설정 파일(.json)을 읽고, 모델이 학습/추론에 사용할
    '문제 텐서(Problem Tensor)'를 생성하는 클래스입니다.
    
    주요 역할:
    1. 설정 파일을 로드하고 'IC 템플릿' 목록을 전처리합니다.
    2. (N_max, FEATURE_DIM) 크기의 텐서를 생성하고,
       [BATT][LOADS][IC_TEMPLATES][EMPTY] 레이아웃에 맞게 데이터를 채웁니다.
    3. 어텐션 마스크, 연결성 행렬 등 모델에 필요한 보조 텐서를 생성합니다.
    """
    
    def __init__(self, config_file_path: str, N_max: int, seed: int = None):
        """
        PocatGenerator를 초기화합니다.
        
        Args:
            config_file_path (str): 로드할 .json 설정 파일 경로
            N_max (int): 모델이 처리할 고정된 최대 노드 크기 (N_MAX)
            seed (int): 재현성을 위한 시드값 (None이면 랜덤)
        """
        
        self.N_max = N_max # (예: 500)
        
        # 1. 설정 파일 로드
        battery_obj, original_ics_obj, loads_obj, constraints_obj = \
            load_configuration_from_file(config_file_path)

        if not battery_obj or not loads_obj:
            raise ValueError(f"설정 파일 로드 실패: {config_file_path}")

        # ----------------------------------------------------------------
        # [신규] 랜덤 학습 범용성을 위한 "표준 전압 더미 로드" 주입
        # (설정 파일에 없는 전압이라도, 학습 때 나오려면 IC 템플릿이 미리 생성되어야 함)
        # ----------------------------------------------------------------
        standard_voltages = [1.2, 1.8, 2.5, 3.3, 5.0]
        existing_vouts = {l.voltage_typical for l in loads_obj}
        
        for v in standard_voltages:
            if v not in existing_vouts:
                # 더미 로드 객체 생성 및 추가 (common.data_classes.Load)
                dummy = Load(
                    name=f"DUMMY_{v}V", # 이름으로 구분
                    voltage_req_min=v*0.95, voltage_req_max=v*1.05, voltage_typical=v,
                    current_active=0.1, current_sleep=0.0
                )
                loads_obj.append(dummy)
        # ----------------------------------------------------------------

        # 2. "IC 템플릿" 생성 (더미 로드 덕분에 모든 표준 전압용 IC가 생성됨)
        template_ic_objs = expand_ic_templates(
            original_ics_obj, loads_obj, battery_obj, constraints_obj
        )
        
        # 3. "IC 템플릿" 목록에서 지배적인(dominated) 후보 제거
        pruned_template_objs = prune_dominated_ics(template_ic_objs)
        
        # 4. 내부용 PocatConfig 생성 (노드 이름/타입 목록 생성용)
        #    (definitions.py의 [B, L, IC] 순서를 따름)
        config_data = {
            "battery": battery_obj.__dict__,
            "available_ics": [ic.__dict__ for ic in pruned_template_objs],
            "loads": [load.__dict__ for load in loads_obj],
            "constraints": constraints_obj
        }
        self.config = PocatConfig(**config_data)

        # ----------------------------------------------------------------
        # [수정] 전압별 최대 공급 가능 전류(Thermal Derating 반영됨) 계산
        # Load 생성 시에는 '실제 공급 가능한' Derated Limit을 사용해야 함
        # ----------------------------------------------------------------
        self.max_current_per_vout = defaultdict(float)
        for ic in self.config.available_ics:
            v = round(ic['vout'], 2)
            # i_limit은 ic_preprocessor에서 열 제약(derating)이 반영된 값임
            limit = ic.get('i_limit', 0.0)
            self.max_current_per_vout[v] = max(self.max_current_per_vout[v], limit)
        
        print(f"✅ Max Current per Voltage (Derated for Load Gen): {dict(self.max_current_per_vout)}")

        # 5. 텐서 레이아웃 계산
        self.num_battery = 1
        self.num_loads = len(self.config.loads)
        self.num_templates = len(self.config.available_ics)
        
        # 실제 부품(BATT, LOADS, TEMPLATES)의 총 개수
        self.num_components = self.num_battery + self.num_loads + self.num_templates
        
        if self.num_components > self.N_max:
            raise ValueError(
                f"설정 파일의 부품 개수({self.num_components})가 "
                f"N_MAX ({self.N_max})를 초과합니다."
            )
            
        # IC가 스폰(Spawn)될 수 있는 빈 슬롯의 개수
        self.num_empty_slots = self.N_max - self.num_components
        
        print(f"PocatGenerator: N_max={self.N_max} | "
              f"Layout: [1 B] + [{self.num_loads} L] + "
              f"[{self.num_templates} T] + [{self.num_empty_slots} E]")

        # 6. 재사용할 기본 텐서들을 미리 계산하고 캐시합니다.
        self._base_tensors = {}
        self._tensor_cache_by_device = {}
        self._initialize_base_tensors()

        # [신규] 템플릿에서 지원하는 출력 전압(Vout) 목록 추출 (Stuck 방지용)
        valid_vouts = set()
        for ic in self.config.available_ics:
            valid_vouts.add(ic["vout"])
        self.valid_vouts = sorted(list(valid_vouts))
        print(f"✅ Valid Random Load Voltages (from Templates): {self.valid_vouts}")

        # -------------------s--------------------------------------------
        # [개선 1] 전류/전압 분포 설정 파라미터화 (실험용)
        # ----------------------------------------------------------------
        # 전압 가중치 (Probability Weights) - 합이 100이 아니어도 됨 (상대 비율)
        self.voltage_weights = {
            1.8: 35, 3.3: 35,  # Main (70%)
            1.2: 10, 2.5: 10, 5.0: 10  # Others (30%)
        }
        
        # [개선] 전압별 전류 분포 설정 (확률, Min_A, Max_A)
        # 요지: 5V는 Heavy 희소, 1.2V/1.8V는 고전류 꼬리, 3.3V는 중간 전류 중심
        self.voltage_current_dist = {
            5.0: [ # 수A 거의 없음
                (0.70, 0.001, 0.2),   # 1-200mA
                (0.25, 0.2, 0.8),     # 200-800mA
                (0.05, 0.8, 2.0)      # 0.8-2.0A (드물게)
            ],
            3.3: [ # 가장 흔함, 중저전류 위주
                (0.60, 0.0005, 0.12), # 0.5-120mA
                (0.30, 0.12, 0.6),    # 120-600mA
                (0.10, 0.6, 1.5)      # 0.6-1.5A
            ],
            1.8: [ # 디지털 레일 (평균 전류 상승)
                (0.50, 0.001, 0.15),  # 1-150mA
                (0.35, 0.15, 0.8),    # 150-800mA
                (0.15, 0.8, 2.5)      # 0.8-2.5A
            ],
            1.2: [ # 코어 레일 (고전류 꼬리)
                (0.35, 0.005, 0.2),   # 5-200mA
                (0.40, 0.2, 1.2),     # 200mA-1.2A
                (0.25, 1.2, 4.0)      # 1.2-4.0A
            ],
            2.5: [ # 아날로그 (대부분 저전류)
                (0.75, 0.0005, 0.08), # 0.5-80mA
                (0.20, 0.08, 0.3),    # 80-300mA
                (0.05, 0.3, 1.0)      # 0.3-1.0A
            ]
        }

        # [개선 2] 시드 설정
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int):
        """실험 재현성을 위한 시드 설정"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _initialize_base_tensors(self) -> None:
        """모델에 필요한 모든 기본 텐서(노드 피처, 마스크 등)를 미리 계산합니다."""
        
        # (N_max, D)
        node_features = self._create_feature_tensor()
        # (N_max, N_max)
        connectivity_matrix = self._create_connectivity_matrix(node_features)
        # (4,), (N_max, N_max)
        scalar_prompt, matrix_prompt = self._create_prompt_tensors()
        # (N_max, N_max)
        attention_mask = self._create_attention_mask(node_features)

        # 계산된 텐서들을 (detach된 상태로) 기본 캐시에 저장
        self._base_tensors = {
            "nodes": node_features.detach(),
            "connectivity_matrix": connectivity_matrix.detach(),
            "scalar_prompt_features": scalar_prompt.detach(),
            "matrix_prompt_features": matrix_prompt.detach(),
            "attention_mask": attention_mask.detach(),
        }
        
        base_device = node_features.device
        self._tensor_cache_by_device[base_device] = self._base_tensors

    def _create_attention_mask(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        모델의 Self-Attention을 위한 마스크를 생성합니다.
        PADDING 타입을 제외한 모든 노드(BATT, LOAD, IC, EMPTY)는
        서로 상호작용(Attend)할 수 있습니다.
        """
        # (N_max,)
        node_types = node_features[..., FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        
        # PADDING(0)이 아닌 모든 노드는 True
        alive_mask_1d = (node_types != NODE_TYPE_PADDING)
        
        # (N_max, N_max)
        attention_mask = alive_mask_1d.unsqueeze(1) & alive_mask_1d.unsqueeze(0)
        return attention_mask

    def _create_feature_tensor(self) -> torch.Tensor:
        """
        [BATT][LOADS][IC_TEMPLATES][EMPTY] 레이아웃으로
        (N_max, FEATURE_DIM) 크기의 노드 피처 텐서를 생성합니다.
        """
        
        features = torch.zeros(self.N_max, FEATURE_DIM)
        ambient_temp = self.config.constraints.get("ambient_temperature", 25.0)
        
        # 모든 노드의 기본 정션 온도는 주변 온도로 설정
        features[:, FEATURE_INDEX["junction_temp"]] = ambient_temp
        
        current_idx = 0
        
        # --- 1. Battery (Active) (Slot 0) ---
        b_conf = self.config.battery
        features[current_idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_BATTERY] = 1.0
        features[current_idx, FEATURE_INDEX["is_active"]] = 1.0 # (상태 피처)
        features[current_idx, FEATURE_INDEX["vout_min"]] = b_conf["voltage_min"]
        features[current_idx, FEATURE_INDEX["vout_max"]] = b_conf["voltage_max"]
        current_idx += 1
        
        # --- 2. Loads (Active) (Slots 1 ~ N_loads) ---
        load_start_idx = current_idx
        for i, l_conf in enumerate(self.config.loads):
            idx = load_start_idx + i
            features[idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_LOAD] = 1.0
            features[idx, FEATURE_INDEX["is_active"]] = 1.0 # (상태 피처)
            
            # Load 요구사항 피처 설정
            features[idx, FEATURE_INDEX["vin_min"]] = l_conf["voltage_req_min"]
            features[idx, FEATURE_INDEX["vin_max"]] = l_conf["voltage_req_max"]
            features[idx, FEATURE_INDEX["current_active"]] = l_conf["current_active"]
            features[idx, FEATURE_INDEX["current_sleep"]] = l_conf["current_sleep"]
            
            rail_type = l_conf.get("independent_rail_type")
            if rail_type == "exclusive_supplier":
                features[idx, FEATURE_INDEX["independent_rail_type"]] = 1.0
            elif rail_type == "exclusive_path":
                features[idx, FEATURE_INDEX["independent_rail_type"]] = 2.0
            
            if l_conf.get("always_on_in_sleep", False):
                features[idx, FEATURE_INDEX["always_on_in_sleep"]] = 1.0
        
        current_idx += self.num_loads

        # --- 3. IC Templates (Template) (Slots N_loads+1 ~ N_components-1) ---
        template_start_idx = current_idx
        for i, ic_conf in enumerate(self.config.available_ics):
            idx = template_start_idx + i
            features[idx, FEATURE_INDEX["node_type"][0] + NODE_TYPE_IC] = 1.0
            features[idx, FEATURE_INDEX["is_template"]] = 1.0 # (상태 피처)
            
            # IC 템플릿 스펙 피처 설정
            features[idx, FEATURE_INDEX["cost"]] = ic_conf.get("cost", 0.0)
            features[idx, FEATURE_INDEX["vin_min"]] = ic_conf.get("vin", 0.0)
            features[idx, FEATURE_INDEX["vin_max"]] = ic_conf.get("vin", 0.0)
            features[idx, FEATURE_INDEX["vout_min"]] = ic_conf.get("vout", 0.0)
            features[idx, FEATURE_INDEX["vout_max"]] = ic_conf.get("vout", 0.0)
            
            # [수정] 텐서에는 무조건 '원본 스펙 i_limit'을 기록합니다.
            # (ic_conf['i_limit']은 derated 값이지만, original_i_limit 필드가 보존되어 있음)
            features[idx, FEATURE_INDEX["i_limit"]] = ic_conf.get("original_i_limit", ic_conf.get("i_limit", 0.0))
            
            features[idx, FEATURE_INDEX["theta_ja"]] = ic_conf.get("theta_ja", 999.0)
            features[idx, FEATURE_INDEX["t_junction_max"]] = ic_conf.get("t_junction_max", 125.0)
            features[idx, FEATURE_INDEX["quiescent_current"]] = ic_conf.get("quiescent_current", 0.0)
            features[idx, FEATURE_INDEX["shutdown_current"]] = ic_conf.get("shutdown_current", 0.0)
            features[idx, FEATURE_INDEX["op_current"]] = ic_conf.get("operating_current", 0.0)
            val_fb = ic_conf.get("min_fb_res")
            features[idx, FEATURE_INDEX["min_fb_res"]] = val_fb if val_fb is not None else 0.0
            val_eff_active = ic_conf.get("efficiency_active")
            features[idx, FEATURE_INDEX["efficiency_active"]] = val_eff_active if val_eff_active is not None else 0.9
            val_eff_sleep = ic_conf.get("efficiency_sleep")
            features[idx, FEATURE_INDEX["efficiency_sleep"]] = val_eff_sleep if val_eff_sleep is not None else 0.35            

            ic_type = ic_conf.get("type")
            if ic_type == 'LDO':
                features[idx, FEATURE_INDEX["ic_type_idx"]] = 1.0
            elif ic_type == 'Buck':
                features[idx, FEATURE_INDEX["ic_type_idx"]] = 2.0
                
        current_idx += self.num_templates
        
        # --- 4. Empty Slots (Spawnable) (Slots N_components ~ N_max-1) ---
        empty_start_idx = self.num_components
        if self.num_empty_slots > 0:
            features[empty_start_idx:, FEATURE_INDEX["node_type"][0] + NODE_TYPE_EMPTY] = 1.0
            features[empty_start_idx:, FEATURE_INDEX["can_spawn_into"]] = 1.0 # (상태 피처)
        
        # --- 5. Node ID (0 ~ N_max-1) 정규화 ---
        for idx in range(self.N_max):
             features[idx, FEATURE_INDEX["node_id"]] = float(idx) / self.N_max
             
        return features

    def _create_connectivity_matrix(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        노드 피처(전압)를 기반으로 (N_max, N_max) 크기의
        '잠재적 연결성' 인접 행렬을 생성합니다.
        
        (상태와 관계없이, 전압만 맞으면 True)
        """
        num_nodes = self.N_max
        
        # (N_max,)
        node_types = node_features[..., FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
        
        # 부모가 될 수 있는 타입: Battery(1), IC(3)
        is_parent = (node_types == NODE_TYPE_IC) | (node_types == NODE_TYPE_BATTERY)
        # 자식이 될 수 있는 타입: Load(2), IC(3)
        is_child = (node_types == NODE_TYPE_IC) | (node_types == NODE_TYPE_LOAD)
        
        parent_mask = is_parent.unsqueeze(1).expand(-1, num_nodes)
        child_mask = is_child.unsqueeze(0).expand(num_nodes, -1)
        
        # 전압 호환성 계산
        parent_vout_min = node_features[:, FEATURE_INDEX["vout_min"]].unsqueeze(1)
        parent_vout_max = node_features[:, FEATURE_INDEX["vout_max"]].unsqueeze(1)
        child_vin_min = node_features[:, FEATURE_INDEX["vin_min"]].unsqueeze(0)
        child_vin_max = node_features[:, FEATURE_INDEX["vin_max"]].unsqueeze(0)
        
        voltage_compatible = (parent_vout_min <= child_vin_max) & (parent_vout_max >= child_vin_min)
        
        # adj_matrix[i, j] = True => i가 j의 부모가 될 수 있음 (i -> j)
        adj_matrix = parent_mask & child_mask & voltage_compatible
        adj_matrix.diagonal().fill_(False) # 자기 자신에게 연결 방지
        return adj_matrix.to(torch.bool)

    def _create_prompt_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        스칼라 및 행렬 프롬프트 텐서를 생성합니다.
        (행렬 프롬프트는 파워 시퀀싱 제약조건을 인코딩합니다.)
        """
        
        constraints = self.config.constraints
        
        # 1. 스칼라 프롬프트 (제약조건 값)
        scalar_prompt_list = [
            constraints.get("ambient_temperature", 25.0),
            constraints.get("max_sleep_current", 0.0),
            constraints.get("current_margin", 0.0),
            constraints.get("thermal_margin_deg", 0.0)
        ]
        scalar_prompt_features = torch.tensor(scalar_prompt_list, dtype=torch.float32)

        # 2. 행렬 프롬프트 (파워 시퀀싱)
        matrix_prompt_features = torch.zeros(self.N_max, self.N_max, dtype=torch.float32)
        
        # (definitions.py의 PocatConfig가 [B,L,IC] 순서를 보장하므로
        #  node_names 리스트의 인덱스는 텐서 레이아웃과 일치합니다)
        node_name_to_idx = {name: i for i, name in enumerate(self.config.node_names)}

        for seq in constraints.get("power_sequences", []):
            j_name, k_name = seq.get('j'), seq.get('k')
            j_idx = node_name_to_idx.get(j_name)
            k_idx = node_name_to_idx.get(k_name)
            
            # (j_idx, k_idx가 실제 부품 슬롯 내에 있는지 확인)
            if j_idx is not None and k_idx is not None and \
               j_idx < self.num_components and k_idx < self.num_components:
                
                # (예: matrix_prompt[LOAD_A, LOAD_B] = 1.0)
                matrix_prompt_features[j_idx, k_idx] = 1.0

        return scalar_prompt_features, matrix_prompt_features

    def _get_device_base_tensors(self, device: Any = None) -> Dict[str, torch.Tensor]:
        """미리 계산된 텐서를 요청된 디바이스로 이동시키고 캐시합니다."""
        if device is None:
            # 기본 디바이스의 텐서 사용
            return next(iter(self._tensor_cache_by_device.values()))

        device = torch.device(device)
        if device in self._tensor_cache_by_device:
            # 캐시된 텐서 반환
            return self._tensor_cache_by_device[device]

        base_device, base_tensors = next(iter(self._tensor_cache_by_device.items()))
        if device == base_device:
            return base_tensors

        # 새 디바이스로 텐서 이동 및 캐시
        device_tensors = {
            name: tensor.to(device, non_blocking=True)
            for name, tensor in base_tensors.items()
        }
        self._tensor_cache_by_device[device] = device_tensors
        return device_tensors


    def __call__(self, batch_size: int, **kwargs) -> TensorDict:
        """
        Generator를 호출하여, 미리 계산된 텐서들을
        요청된 배치 크기(batch_size)만큼 복제하여 TensorDict로 반환합니다.
        """
        device = kwargs.get("device", None)
        base_tensors = self._get_device_base_tensors(device)

        # (B, N_max, D)
        nodes = base_tensors["nodes"].detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        # (B, 4)
        scalar_prompt = base_tensors["scalar_prompt_features"].detach().unsqueeze(0).expand(batch_size, -1).clone()
        # (B, N_max, N_max)
        matrix_prompt = base_tensors["matrix_prompt_features"].detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        # (B, N_max, N_max)
        connectivity = base_tensors["connectivity_matrix"].detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        # (B, N_max, N_max)
        attention_mask = base_tensors["attention_mask"].detach().unsqueeze(0).expand(batch_size, -1, -1).clone()
        
        return TensorDict({
            "nodes": nodes,
            "scalar_prompt_features": scalar_prompt,
            "matrix_prompt_features": matrix_prompt,
            "connectivity_matrix": connectivity,
            "attention_mask": attention_mask, # 모델 어텐션용 마스크
        }, batch_size=[batch_size])

    # ---------------------------------------------------------------------------------------
    # [신규 추가] 범용 솔버 학습을 위한 랜덤 생성 및 POMO 지원 메서드
    # ---------------------------------------------------------------------------------------
    def _generate_load_profiles(self, count: int) -> List[Dict[str, float]]:
        """
        [개선된 부하 생성기]
        요청된 개수(count)만큼의 Load Profile을 즉석에서 생성합니다.
        
        특징:
        1. 가중치 기반 전압 선택 (random.choices)
        2. Rejection Sampling으로 Limit 준수 (분포 왜곡 방지)
        """
        pool = []
        
        # 전압 목록과 가중치 추출
        voltages = list(self.voltage_weights.keys())
        weights = list(self.voltage_weights.values())
        
        while len(pool) < count:
            # 1. 전압 선택 (Weighted)
            v = random.choices(voltages, weights=weights, k=1)[0]
            
            # 해당 전압을 지원하는 IC가 없거나 한계가 너무 낮으면 스킵
            max_limit = self.max_current_per_vout.get(v, 0.0)
            if max_limit <= 0.001: 
                continue

            # [수정] 전압별 정의된 분포 가져오기 (없으면 3.3V 기준 사용)
            # 딕셔너리 키 조회 시 반올림 이슈 방지를 위해, 가장 가까운 키를 찾거나 기본값을 사용
            # 여기서는 standard_voltages에 있는 값이면 정확히 일치한다고 가정합니다.
            dist_specs = self.voltage_current_dist.get(v, self.voltage_current_dist[3.3])
            dist_probs = [spec[0] for spec in dist_specs]

            # 2. 전류 생성 (Rejection Sampling)
            valid_current = None
            for _ in range(10): # Max Retries
                # 해당 전압의 구간 분포에서 선택
                spec = random.choices(dist_specs, weights=dist_probs, k=1)[0]
                _, i_min, i_max = spec
                
                if i_min > max_limit: continue
                
                # [수정] 로그 스케일 샘플링 (Log-Uniform)
                # val = exp( uniform( log(min), log(max) ) )
                log_min = math.log(i_min)
                log_max = math.log(i_max)
                val = math.exp(random.uniform(log_min, log_max))

                if val <= max_limit:
                    valid_current = val
                    break
            
            # 3. Fallback (10번 시도 실패 시)
            if valid_current is None:
                fallback_max = min(max_limit, 0.3) 
                valid_current = random.uniform(0.001, fallback_max)
                
            pool.append({'v': v, 'i': valid_current})

            
        return pool

    def generate_random_batch(self, batch_size: int, device='cpu', supply_chain_prob: float = 0.0) -> TensorDict:
        """ 
        배치 x POMO 형태의 랜덤 문제 생성 (IC 셔플링 적용됨)
        - supply_chain_prob: 각 IC가 품절될 확률 (0.0 ~ 1.0)
        """
        batch_nodes_list = []
        batch_scalar_prompt = []
        batch_matrix_prompt = []
        batch_adj_list = []
        batch_mask_list = []

        # 통계 집계용 리스트
        stats_num_loads = []
        stats_num_masked_ics = []

        # IC 템플릿 가져오기 (초기 설정된 base_tensor에서 추출)
        # Layout: [Battery(1)] + [Config_Loads] + [Templates] + [Empty]
        template_start_idx = 1 + self.num_loads
        template_end_idx = template_start_idx + self.num_templates
        # (CPU에 있는 기본 텐서 사용)
        master_ic_features = self._base_tensors["nodes"][template_start_idx:template_end_idx].clone()

        for _ in range(batch_size):
            # -------------------------------------------------------
            # [수정] 0. IC 순서 셔플 (가장 먼저 수행!)
            # -------------------------------------------------------
            current_ics_feat = master_ic_features.clone()
            
            # IC 템플릿의 행 순서를 무작위로 섞습니다.
            # (Feature들도 함께 이동하므로 데이터 무결성은 유지됩니다)
            perm = torch.randperm(current_ics_feat.size(0))
            current_ics_feat = current_ics_feat[perm]

            # -------------------------------------------------------
            # 1. IC 가격 변동 (Price Fluctuation)
            # -------------------------------------------------------
            # 가격 노이즈 (±20%)
            price_noise = torch.rand(self.num_templates) * 0.4 + 0.8
            current_ics_feat[:, FEATURE_INDEX["cost"]] *= price_noise
            
            # -------------------------------------------------------
            # [신규] 1.5 공급망 이슈 (Supply Chain Issues)
            # -------------------------------------------------------
            active_ic_indices = list(range(self.num_templates)) # 기본: 모든 IC 사용 가능
            num_masked = 0
            if supply_chain_prob > 0.0:
                # 각 템플릿별로 supply_chain_prob 확률로 품절(False) 처리
                # (True: 재고 있음, False: 품절)
                is_in_stock = torch.rand(self.num_templates) > supply_chain_prob
                
                # 품절 통계
                num_masked = (~is_in_stock).sum().item()
                
                # 품절된 IC는 'is_template' 상태를 0으로 만들어 선택 불가하게 함
                current_ics_feat[:, FEATURE_INDEX["is_template"]] *= is_in_stock.float()

                # [신규] 살아남은 IC들의 인덱스 추적
                active_ic_indices = torch.where(is_in_stock)[0].tolist()
            
            stats_num_masked_ics.append(num_masked)

            # [신규] 현재 살아있는 IC로 지원 가능한 (Vout -> Max_Current) 맵 생성
            # 주의: current_ics_feat가 이미 셔플되었으므로, 해당 인덱스의 스펙을 그대로 읽으면 됩니다.
            current_max_i_per_v = defaultdict(float)
            for idx in active_ic_indices:
                # master_ic_features 텐서에서 스펙 조회 (Vout, I_limit)
                ic_vout_raw = current_ics_feat[idx, FEATURE_INDEX["vout_max"]].item()

                ic_vout = round(ic_vout_raw, 2)
                # 텐서에 저장된 값은 원본 스펙(original_i_limit)이므로 이를 사용
                ic_limit = current_ics_feat[idx, FEATURE_INDEX["i_limit"]].item()
                current_max_i_per_v[ic_vout] = max(current_max_i_per_v[ic_vout], ic_limit)

            # -------------------------------------------------------
            # 2. Load Sampling (Stratified)
            # -------------------------------------------------------
            # N_max 공간 확인
            max_possible_loads = self.N_max - 1 - self.num_templates
            limit = min(30, max_possible_loads)
            #low_bound = max(5, int(limit * 0.5))
            low_bound = 25
            try:
                num_loads = int(random.triangular(low_bound, limit, limit))
            except:
                # 만약 공간 부족으로 limit가 10보다 작아지면 예외 처리
                num_loads = limit

            stats_num_loads.append(num_loads)

            # [개선] 1. 동적 로드 생성 (시스템 전체 Max Limit 기준)
            raw_profiles = self._generate_load_profiles(num_loads)
            
            # [개선] 2. 공급망 이슈 필터링 (현재 살아있는 IC로 지원 가능한지 확인)
            selected_profiles = []
            if active_ic_indices:
                for p in raw_profiles:
                    v_key = round(p['v'], 2)
                    max_i_avail = current_max_i_per_v.get(v_key, 0.0)
                    
                    if max_i_avail >= p['i'] - 1e-6:
                        selected_profiles.append(p)
            
            # 실제 생성된 개수로 업데이트 (필터링으로 인해 num_loads보다 적을 수 있음)
            actual_num_loads = len(selected_profiles)

            # -------------------------------------------------------
            # 3. 텐서 조립 (full_nodes 생성)
            # -------------------------------------------------------
            loads_feat = torch.zeros(actual_num_loads, FEATURE_DIM)
            load_indices = []

            for k, profile in enumerate(selected_profiles):
                loads_feat[k, FEATURE_INDEX["node_type"][0] + NODE_TYPE_LOAD] = 1.0
                loads_feat[k, FEATURE_INDEX["is_active"]] = 1.0
                loads_feat[k, FEATURE_INDEX["vin_min"]] = profile['v']
                loads_feat[k, FEATURE_INDEX["vin_max"]] = profile['v']
                loads_feat[k, FEATURE_INDEX["vout_min"]] = profile['v']
                loads_feat[k, FEATURE_INDEX["current_active"]] = profile['i']

                # [수정] 절전모드(AO) 제한 (10%) & Sleep 전류 처리
                is_ao = (random.random() < 0.10)
                if is_ao:
                    # 사용자 요청: Active 전류 비례가 아닌, 랜덤하게 10uA ~ 200uA 사이 값으로 설정
                    # 10uA = 0.00001, 200uA = 0.0002
                    random_sleep = random.uniform(0.00001, 0.0002)
                    
                    loads_feat[k, FEATURE_INDEX["current_sleep"]] = random_sleep
                    loads_feat[k, FEATURE_INDEX["always_on_in_sleep"]] = 1.0
                else:
                    loads_feat[k, FEATURE_INDEX["current_sleep"]] = 0.0
                    loads_feat[k, FEATURE_INDEX["always_on_in_sleep"]] = 0.0
                
                # [수정] Independent Rail 제약 확률 상향 (5% -> 10%)
                if random.random() < 0.20:
                    loads_feat[k, FEATURE_INDEX["independent_rail_type"]] = random.choice([1.0, 2.0])
                
                load_indices.append(k)

            # Battery (기본 설정 사용)
            batt_feat = torch.zeros(1, FEATURE_DIM)
            batt_feat[0, FEATURE_INDEX["node_type"][0] + NODE_TYPE_BATTERY] = 1.0
            batt_feat[0, FEATURE_INDEX["is_active"]] = 1.0
            batt_feat[0, FEATURE_INDEX["vout_min"]] = self.config.battery["voltage_min"]
            batt_feat[0, FEATURE_INDEX["vout_max"]] = self.config.battery["voltage_max"]

            # Empty Slots
            curr_cnt = 1 + actual_num_loads + self.num_templates
            num_empty = self.N_max - curr_cnt
            empty_feat = torch.zeros(num_empty, FEATURE_DIM)
            if num_empty > 0:
                empty_feat[:, FEATURE_INDEX["node_type"][0] + NODE_TYPE_EMPTY] = 1.0
                empty_feat[:, FEATURE_INDEX["can_spawn_into"]] = 1.0

            # [Concat] -> 이것이 full_nodes 입니다!
            full_nodes = torch.cat([batt_feat, loads_feat, current_ics_feat, empty_feat], dim=0)
            
            # Node ID 정규화 (여기서 순서를 다시 0~N으로 매기므로, 셔플된 IC도 올바른 ID를 가짐)
            full_nodes[:, FEATURE_INDEX["node_id"]] = torch.arange(self.N_max).float() / self.N_max
            
            # -------------------------------------------------------
            # 4. Prompt 생성
            # -------------------------------------------------------
            scalar_p = torch.tensor([random.choice([70.0, 85.0]), 0.001, 0.1, 5.0])
            
            # Power Sequence (20%)
            mat_p = torch.zeros(self.N_max, self.N_max)
            
            # [수정] Power Sequence 제약조건 증가 (10% of Loads)
            if len(load_indices) >= 2:
                # 전체 로드의 10% 개수만큼 제약 생성 (최소 1개)
                num_constraints = max(1, int(len(load_indices) * 0.10))
                
                for _ in range(num_constraints):
                    src, dst = random.sample(load_indices, 2)
                    mat_p[1+src, 1+dst] = 1.0

            # Connectivity & Mask (Helper 메서드 활용)
            adj = self._create_connectivity_matrix(full_nodes)
            mask = self._create_attention_mask(full_nodes)

            # -------------------------------------------------------
            # 5. POMO 확장 (Instance 복제)
            # -------------------------------------------------------
            # full_nodes: [N, D] -> [POMO, N, D]
            batch_nodes_list.append(full_nodes)
            batch_scalar_prompt.append(scalar_p)
            batch_matrix_prompt.append(mat_p)
            batch_adj_list.append(adj)
            batch_mask_list.append(mask)

        # -------------------------------------------------------
        # 📊 [통계 출력] 실제 생성된 Load 및 IC 현황
        # -------------------------------------------------------
        avg_loads = sum(stats_num_loads) / len(stats_num_loads)
        avg_masked = sum(stats_num_masked_ics) / len(stats_num_masked_ics)
        print(f"   📊 [Generated Stats] Load Count: Min {min(stats_num_loads)} / Max {max(stats_num_loads)} / Avg {avg_loads:.1f}")
        if supply_chain_prob > 0.0:
            print(f"   📊 [Supply Chain] Avg Masked ICs: {avg_masked:.1f} / {self.num_templates} ({supply_chain_prob*100}%)")
        
        # 최종 반환 형태: [Batch, N, D]
        return TensorDict({
            "nodes": torch.stack(batch_nodes_list).to(device),
            "scalar_prompt_features": torch.stack(batch_scalar_prompt).to(device),
            "matrix_prompt_features": torch.stack(batch_matrix_prompt).to(device),
            "connectivity_matrix": torch.stack(batch_adj_list).to(device),
            "attention_mask": torch.stack(batch_mask_list).to(device),
        }, batch_size=[batch_size])