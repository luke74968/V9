# Copyright (c) 2025 Minuk Lee. All rights reserved.
# 
# This source code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# 
# For licensing terms, see the LICENSE file.
# Contact: minuklee@snu.ac.kr
# 
import torch
import torch.nn as nn
import torch.nn.functional as F
import random # [추가] Epsilon Mixing용
from torch.distributions import Categorical
from tensordict import TensorDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --- 현재 패키지(transformer_solver) 모듈 임포트 ---
from .definitions import (
    FEATURE_DIM, FEATURE_INDEX, SCALAR_PROMPT_FEATURE_DIM,
    NODE_TYPE_PADDING, NODE_TYPE_BATTERY, NODE_TYPE_LOAD, 
    NODE_TYPE_IC, NODE_TYPE_EMPTY
)
from .utils.common import batchify
from .solver_env import PocatEnv, BATTERY_NODE_IDX 


# ---
# 섹션 1: 표준 트랜스포머 빌딩 블록 (효율성)
# ---

class RMSNorm(nn.Module):
    """ Root Mean Square Layer Normalization """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class Normalization(nn.Module):
    """ 정규화 레이어 래퍼 (RMSNorm 또는 LayerNorm) """
    def __init__(self, embedding_dim, norm_type='rms', **kwargs):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == 'rms':
            self.norm = RMSNorm(embedding_dim)
        elif self.norm_type == 'layer':
            self.norm = nn.LayerNorm(embedding_dim)
        else:
            raise NotImplementedError(f"Unknown norm_type: {norm_type}")

    def forward(self, x):
        return self.norm(x)

class ParallelGatedMLP(nn.Module):
    """ SwiGLU FFN (FeedForward) 구현체 """
    def __init__(self, hidden_size: int, **kwargs):
        super().__init__()
        # LLAMA 아키텍처에서 사용하는 FFN 차원 계산
        inner_size = int(2 * hidden_size * 4 / 3)
        multiple_of = 256
        inner_size = multiple_of * ((inner_size + multiple_of - 1) // multiple_of)
        
        self.l1 = nn.Linear(hidden_size, inner_size, bias=False)
        self.l2 = nn.Linear(hidden_size, inner_size, bias=False)
        self.l3 = nn.Linear(inner_size, hidden_size, bias=False)
        self.act = F.silu

    def forward(self, z):
        z1 = self.l1(z)
        z2 = self.l2(z)
        return self.l3(self.act(z1) * z2)

def reshape_by_heads(qkv: torch.Tensor, head_num: int) -> torch.Tensor:
    """ (B, N, H*D) -> (B, H, N, D) """
    batch_s, n = qkv.size(0), qkv.size(1)
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    return q_reshaped.transpose(1, 2)

def multi_head_attention(q, k, v, attention_mask=None):
    """ 
    PyTorch 2.0+ Scaled Dot Product Attention (SDPA) 적용
    (메모리 효율성 및 속도 최적화 - FlashAttention 자동 사용)
    """
    batch_s, head_num, n, key_dim = q.shape
    
    # SDPA를 위한 마스크 처리
    # (PyTorch SDPA는 Boolean 마스크 지원이 버전마다 상이하므로, 
    #  확실하게 -inf를 더하는 방식의 Float 마스크로 변환하여 전달)
    attn_mask = None
    if attention_mask is not None:
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1) # (B, N, N) -> (B, 1, N, N)
        
        # True(유효) -> 0.0, False(마스킹) -> -inf
        attn_mask = torch.zeros_like(attention_mask, dtype=q.dtype)
        attn_mask.masked_fill_(~attention_mask, -float('inf'))

    # PyTorch 2.0+ 최적화 함수 사용
    # (내부적으로 FlashAttention 등을 사용하여 메모리 사용량을 획기적으로 줄임)
    out = F.scaled_dot_product_attention(
        q, k, v, 
        attn_mask=attn_mask
    )
    
    # 4. (B, H, N, D) -> (B, N, H*D)s
    out_transposed = out.transpose(1, 2)
    return out_transposed.contiguous().view(batch_s, n, head_num * key_dim)

class EncoderLayer(nn.Module):
    """ 
    표준 트랜스포머 인코더 레이어 (Post-Normalization)
    """
    def __init__(self, embedding_dim, head_num, qkv_dim, ffd='siglu', **model_params):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        self.normalization1 = Normalization(embedding_dim, **model_params)
        
        if ffd == 'siglu':
            self.feed_forward = ParallelGatedMLP(hidden_size=embedding_dim, **model_params)
        else:
            raise NotImplementedError
            
        self.normalization2 = Normalization(embedding_dim, **model_params)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        # 1. MHA (Post-Normalization)
        q = reshape_by_heads(self.Wq(x), self.head_num)
        k = reshape_by_heads(self.Wk(x), self.head_num)
        v = reshape_by_heads(self.Wv(x), self.head_num)
        
        mha_out = self.multi_head_combine(multi_head_attention(q, k, v, attention_mask=attention_mask))
        h = self.normalization1(x + mha_out) # Residual + Norm
        
        # 2. FFN (Post-Normalization)
        ffn_out = self.feed_forward(h)
        out = self.normalization2(h + ffn_out) # Residual + Norm
        return out

class PocatDecoderLayer(nn.Module):
    """
    Cross-Attention과 FFN으로 구성된 디코더 레이어
    (Query가 1개이므로 Self-Attention은 생략하고 Cross-Attention에 집중)
    """
    def __init__(self, embedding_dim, head_num, qkv_dim, **model_params):
        super().__init__()
        
        # 1. Cross-Attention (Query는 이전 레이어 출력, Key/Val은 인코더 출력)
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        # (Wk, Wv는 인코더 쪽에서 미리 계산된 캐시를 재사용하거나, 여기서 별도 정의 가능)
        # 효율성을 위해 여기서는 인코더의 K, V를 공유(Sharing)하거나 
        # 별도로 투영(Projection)할 수 있습니다. 여기서는 별도 투영을 가정합니다.
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)
        
        self.norm1 = Normalization(embedding_dim, **model_params)
        self.norm2 = Normalization(embedding_dim, **model_params)
        
        # 2. Feed Forward Network
        self.feed_forward = ParallelGatedMLP(hidden_size=embedding_dim, **model_params)
        
        self.head_num = head_num
        self.qkv_dim = qkv_dim

    def forward(self, x, cross_k, cross_v):
        """
        x: (B, 1, D) - 현재 디코더의 Query 상태
        cross_k, cross_v: (B, H, N, D/H) - 미리 계산된 Key, Value
        """
        # --- Cross Attention ---
        # Query: 현재 레이어의 입력 x
        q = reshape_by_heads(self.Wq(x), self.head_num)
        
        # Key, Value: 인자로 받은 캐시 사용 (재계산 X)
        # k = reshape_by_heads(self.Wk(encoder_out), self.head_num)
        # v = reshape_by_heads(self.Wv(encoder_out), self.head_num)
        
        mha_out = multi_head_attention(q, cross_k, cross_v)
        mha_out = self.multi_head_combine(mha_out)
        
        h = self.norm1(x + mha_out) # Residual + Norm
        
        # --- FFN ---
        ffn_out = self.feed_forward(h)
        out = self.norm2(h + ffn_out) # Residual + Norm
        
        return out
# ---
# 섹션 2: 디코딩 효율을 위한 캐시
# ---

@dataclass
class PrecomputedCache:
    """
    디코딩 루프에서 반복 계산을 피하기 위해
    인코더의 Key, Value 값을 저장하는 캐시 객체입니다.
    """
    node_embeddings: torch.Tensor
    #glimpse_key: torch.Tensor
    #glimpse_val: torch.Tensor
    logit_key_connect: torch.Tensor # 'Connect' 포인터용 Key
    logit_key_spawn: torch.Tensor   # 'Spawn' 포인터용 Key
    # [추가] 디코더 레이어별 Cross-Attention Key/Value 캐시
    decoder_layer_kvs: List[Tuple[torch.Tensor, torch.Tensor]] = None

    def batchify(self, num_starts: int):
        """ POMO 샘플링을 위해 캐시를 N_starts 배수만큼 복제합니다. """
        # kv 리스트 확장
        new_kvs = []
        if self.decoder_layer_kvs:
            for k, v in self.decoder_layer_kvs:
                # [수정] repeat_interleave 사용
                new_kvs.append((k.repeat_interleave(num_starts, dim=0), v.repeat_interleave(num_starts, dim=0)))

        return PrecomputedCache(
            self.node_embeddings.repeat_interleave(num_starts, dim=0),
            #batchify(self.glimpse_key, num_starts),
            #batchify(self.glimpse_val, num_starts),
            self.logit_key_connect.repeat_interleave(num_starts, dim=0),
            self.logit_key_spawn.repeat_interleave(num_starts, dim=0),
            new_kvs # [추가]
        )

# ---
# 섹션 3: POCAT 모델 아키텍처
# ---

class PocatPromptNet(nn.Module):
    """
    스칼라/행렬 제약조건을 임베딩하는 프롬프트 네트워크 (N_MAX 대응 수정판)
    """
    def __init__(self, embedding_dim: int, N_MAX: int, **kwargs):
        super().__init__()
        self.N_max = N_MAX
        
        # 1. 스칼라 피처 처리 (기존 유지)
        self.scalar_net = nn.Sequential(
            nn.Linear(SCALAR_PROMPT_FEATURE_DIM, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )

        # 2. 매트릭스 피처 처리 (구조 변경)
        # N*N을 펼치는 대신, 한 줄(Row, N)씩 처리하여 차원을 줄입니다.
        self.matrix_proj = nn.Linear(N_MAX, embedding_dim // 2) 

        # 최종 결합부
        self.final_net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, scalar_features, matrix_features):
        # scalar_features: (B, SCALAR_DIM)
        # matrix_features: (B, N, N)

        # 1. 스칼라 임베딩
        scalar_emb = self.scalar_net(scalar_features) # (B, D/2)

        # 2. 매트릭스 임베딩 (Row-wise Projection & Mean Pooling)
        # (B, N, N) -> float 변환 -> Linear -> (B, N, D/2)
        mat_feat = self.matrix_proj(matrix_features.float()) 
        
        # (B, N, D/2) -> 평균 -> (B, D/2)
        matrix_emb = mat_feat.mean(dim=1) 

        # 3. 결합
        combined = torch.cat([scalar_emb, matrix_emb], dim=-1) # (B, D)
        
        # [수정] (B, D) -> (B, 1, D)로 차원을 맞춰줍니다.
        return self.final_net(combined).unsqueeze(1)


class PocatEncoder(nn.Module):
    """
    Pocat 인코더 (듀얼 어텐션 및 다중 임베딩 주입).
    
    1. 노드 타입(5종)별로 기본 임베딩 적용
    2. 노드 속성/상태(4종)별로 추가 임베딩 주입
    3. 듀얼 어텐션(Sparse/Global) 통과

    [수정 사항] 
    - On-the-fly Log Normalization 적용
    - FEATURE_INDEX 참조 시 정수/튜플 타입 체크하여 에러 수정
    """
    def __init__(self, embedding_dim: int, encoder_layer_num: int, **model_params):
        super().__init__()
        
        # 1. 노드 "타입" (5종) 임베딩
        self.embedding_padding = nn.Linear(FEATURE_DIM, embedding_dim)
        self.embedding_battery = nn.Linear(FEATURE_DIM, embedding_dim)
        self.embedding_load = nn.Linear(FEATURE_DIM, embedding_dim)
        self.embedding_ic = nn.Linear(FEATURE_DIM, embedding_dim)
        self.embedding_empty = nn.Linear(FEATURE_DIM, embedding_dim)
        
        # 2. 노드 "속성/상태" (4종) 임베딩 (0 또는 1 값을 인덱스로 사용)
        self.embedding_is_active = nn.Embedding(2, embedding_dim)
        self.embedding_is_template = nn.Embedding(2, embedding_dim)
        self.embedding_can_spawn_into = nn.Embedding(2, embedding_dim)
        self.embedding_rail_type = nn.Embedding(3, embedding_dim) # 0:N/A, 1:Supp, 2:Path

        # 3. 듀얼 어텐션(CaDA) 레이어
        self.sparse_layers = nn.ModuleList([
            EncoderLayer(embedding_dim=embedding_dim, **model_params) 
            for _ in range(encoder_layer_num)
        ])
        self.global_layers = nn.ModuleList([
            EncoderLayer(embedding_dim=embedding_dim, **model_params) 
            for _ in range(encoder_layer_num)
        ])
        self.sparse_fusion = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) 
            for _ in range(encoder_layer_num)
        ])
        self.global_fusion = nn.ModuleList([
            nn.Linear(embedding_dim, embedding_dim) 
            for _ in range(encoder_layer_num - 1)
        ])

    def forward(self, td: TensorDict, prompt_embedding: torch.Tensor) -> torch.Tensor:
        # 1. 원본 노드 피처 가져오기 (마스킹 등 물리 계산용 원본 보존)
        raw_node_features = td['nodes'] # (B, N_MAX, FEATURE_DIM)
        
        # 2. [핵심] 모델 입력용으로 복제 후 On-the-fly Log Normalization 적용
        # clone()을 사용하여 원본 td['nodes']가 수정되지 않도록 함
        nodes_input = raw_node_features.clone()
        
        # ---------------------------------------------------------------------
        # 전략 A: 값이 너무 큰 변수 (Large Scale) -> Log1p 적용
        # (저항: 수십만~수조, 열저항: 수백, 비용: 수십)
        # ---------------------------------------------------------------------
        large_val_keys = ["min_fb_res", "theta_ja", "cost"]
        
        for key in large_val_keys:
            if key in FEATURE_INDEX:
                # [수정] definitions.py에서 스칼라 피처는 정수형이므로 바로 사용
                idx = FEATURE_INDEX[key]
                if isinstance(idx, tuple): idx = idx[0] # 혹시 모를 튜플일 경우 대비
                
                nodes_input[..., idx] = torch.log1p(torch.relu(nodes_input[..., idx]))

        # ---------------------------------------------------------------------
        # 전략 B: 값이 너무 작은 변수 (Tiny Scale) -> uA 변환(x 1e6) 후 Log1p 적용
        # (전류: 수 mA ~ 수 uA 수준 -> 모델이 0으로 인식하는 문제 방지)
        # ---------------------------------------------------------------------
        small_val_keys = [
            "current_active",      # Load Active Current
            "current_sleep",       # Load Sleep Current
            "op_current",          # Operating Current
            "quiescent_current",   # Quiescent Current
            "shutdown_current",    # Shutdown Current
            "not_switching_current"# Buck Non-switching Current
        ]
        
        for key in small_val_keys:
            if key in FEATURE_INDEX:
                idx = FEATURE_INDEX[key]
                if isinstance(idx, tuple): idx = idx[0]

                # x 1,000,000 (uA 단위로 변환)
                # 예: 1uA (1e-6) -> 1.0 -> ln(2) = 0.69
                val = torch.relu(nodes_input[..., idx]) * 1e6 
                nodes_input[..., idx] = torch.log1p(val)

        # ---------------------------------------------------------------------
        # 전략 C: 일반적인 변수 (Normal Scale) -> 그대로 통과
        # (Vin, Vout, I_limit, Efficiency 등은 0~50 범위 내라 변환 불필요)
        # ---------------------------------------------------------------------
        
        batch_size, num_nodes, _ = nodes_input.shape # num_nodes = N_MAX
        embedding_dim = self.embedding_battery.out_features
        
        # AMP 실행 시 임베딩 출력이 fp16/bf16으로 다운캐스팅되므로,
        # node_embeddings의 dtype을 현재 autocast dtype에 맞춰 생성한다.
        if torch.is_autocast_enabled():
            try:
                embedding_dtype = torch.get_autocast_gpu_dtype()
            except Exception:
                # PyTorch < 2.0에는 get_autocast_gpu_dtype가 없으므로 fp16으로 고정
                embedding_dtype = torch.float16
        else:
            embedding_dtype = nodes_input.dtype
        node_embeddings = torch.zeros(
            batch_size, num_nodes, embedding_dim,
            device=nodes_input.device, dtype=embedding_dtype
        )

        # --- 1. 타입별 기본 임베딩 적용 (변환된 nodes_input 사용) ---
        # node_type은 One-hot 인코딩된 범위를 가지므로 튜플 인덱싱 유지
        node_type_indices = nodes_input[..., FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(dim=-1)
        
        masks = {
            NODE_TYPE_PADDING: (node_type_indices == NODE_TYPE_PADDING),
            NODE_TYPE_BATTERY: (node_type_indices == NODE_TYPE_BATTERY),
            NODE_TYPE_LOAD: (node_type_indices == NODE_TYPE_LOAD),
            NODE_TYPE_IC: (node_type_indices == NODE_TYPE_IC),
            NODE_TYPE_EMPTY: (node_type_indices == NODE_TYPE_EMPTY),
        }
        
        # 각 타입별 임베딩을 node_embeddings dtype으로 변환하여 할당한다.
        if masks[NODE_TYPE_PADDING].any():
            emb = self.embedding_padding(nodes_input[masks[NODE_TYPE_PADDING]])
            node_embeddings[masks[NODE_TYPE_PADDING]] = emb.to(node_embeddings.dtype)
        if masks[NODE_TYPE_BATTERY].any():
            emb = self.embedding_battery(nodes_input[masks[NODE_TYPE_BATTERY]])
            node_embeddings[masks[NODE_TYPE_BATTERY]] = emb.to(node_embeddings.dtype)
        if masks[NODE_TYPE_LOAD].any():
            emb = self.embedding_load(nodes_input[masks[NODE_TYPE_LOAD]])
            node_embeddings[masks[NODE_TYPE_LOAD]] = emb.to(node_embeddings.dtype)
        if masks[NODE_TYPE_IC].any():
            emb = self.embedding_ic(nodes_input[masks[NODE_TYPE_IC]])
            node_embeddings[masks[NODE_TYPE_IC]] = emb.to(node_embeddings.dtype)
        if masks[NODE_TYPE_EMPTY].any():
            emb = self.embedding_empty(nodes_input[masks[NODE_TYPE_EMPTY]])
            node_embeddings[masks[NODE_TYPE_EMPTY]] = emb.to(node_embeddings.dtype)

        # --- 2. 속성/상태 임베딩 주입 (Injection) ---
        # (인덱스 값들은 정수형이므로 로그 변환의 영향을 받지 않지만, 일관성을 위해 nodes_input에서 가져옴)
        active_ids = nodes_input[..., FEATURE_INDEX["is_active"]].long()
        template_ids = nodes_input[..., FEATURE_INDEX["is_template"]].long()
        spawn_ids = nodes_input[..., FEATURE_INDEX["can_spawn_into"]].long()
        rail_ids = nodes_input[..., FEATURE_INDEX["independent_rail_type"]].round().long().clamp(0, 2)
        
        # 속성/상태 임베딩도 dtype을 맞춰 더한다.
        node_embeddings.add_(self.embedding_is_active(active_ids).to(node_embeddings.dtype))
        node_embeddings.add_(self.embedding_is_template(template_ids).to(node_embeddings.dtype))
        node_embeddings.add_(self.embedding_can_spawn_into(spawn_ids).to(node_embeddings.dtype))
        node_embeddings.add_(self.embedding_rail_type(rail_ids).to(node_embeddings.dtype))

        
        # --- 3. 듀얼 어텐션 (CaDA) 실행 ---
        connectivity_mask = td['connectivity_matrix'] # (B, N_MAX, N_MAX)
        attention_mask = td['attention_mask'] # (B, N_MAX, N_MAX)

        global_input = torch.cat((node_embeddings, prompt_embedding), dim=1)
        
        global_attention_mask = torch.zeros(
            batch_size, num_nodes + 1, num_nodes + 1, 
            dtype=torch.bool, device=node_embeddings.device
        )
        global_attention_mask[:, :num_nodes, :num_nodes] = attention_mask
        
        alive_mask_1d = (node_type_indices != NODE_TYPE_PADDING)
        global_attention_mask[:, num_nodes, :num_nodes] = alive_mask_1d
        global_attention_mask[:, :num_nodes, num_nodes] = alive_mask_1d
        global_attention_mask[:, num_nodes, num_nodes] = True
        
        sparse_out, global_out = node_embeddings, global_input
        for i in range(len(self.sparse_layers)):
            sparse_out = self.sparse_layers[i](sparse_out, attention_mask=connectivity_mask)
            global_out = self.global_layers[i](global_out, attention_mask=global_attention_mask)
            
            sparse_out = sparse_out + self.sparse_fusion[i](global_out[:, :num_nodes])
            if i < len(self.global_layers) - 1:
                global_nodes = global_out[:, :num_nodes] + self.global_fusion[i](sparse_out)
                global_out = torch.cat((global_nodes, global_out[:, num_nodes:]), dim=1)  
                
        return global_out[:, :num_nodes] # 프롬프트 임베딩 제외 (B, N_MAX, D)

class PocatDecoder(nn.Module):
    def __init__(self, embedding_dim, head_num, qkv_dim, N_MAX, **model_params):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.head_num = head_num
        self.qkv_dim = qkv_dim
        self.N_MAX = N_MAX
        
        # config.yaml에서 decoder_layer_num을 가져옵니다 (기본값 1)
        self.layer_num = model_params.get('decoder_layer_num', 1)

        # 1. 초기 컨텍스트 쿼리 생성용 (입력 차원 변환)
        # (embedding_dim + 3 features) -> embedding_dim
        self.input_projector = nn.Linear(embedding_dim + 3, embedding_dim)

        # 2. 디코더 레이어 스택 (ModuleList)
        self.layers = nn.ModuleList([
            PocatDecoderLayer(embedding_dim, head_num, qkv_dim, **model_params)
            for _ in range(self.layer_num)
        ])
        
        # 3. 포인터 네트워크용 Key 생성 (인코더 임베딩을 변환)
        self.Wk_connect_logit = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk_spawn_logit = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # --- 4. 4-Heads (q_vec을 입력으로 받음) ---
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1)
        )
        self.type_head = nn.Linear(embedding_dim, 2)
        self.connect_head = nn.Linear(embedding_dim, embedding_dim)
        self.spawn_head = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, td: TensorDict, cache: PrecomputedCache) -> Tuple[torch.Tensor, ...]:
        
        # 1. 초기 쿼리 입력 준비
        avg_current = td["nodes"][..., FEATURE_INDEX["current_out"]].clone().mean(dim=1, keepdim=True)
        unconnected_ratio = td["unconnected_loads_mask"].clone().float().mean(dim=1, keepdim=True)
        step_ratio = td["step_count"].clone().float() / (2 * self.N_MAX)
        state_features = torch.cat([avg_current, unconnected_ratio, step_ratio], dim=1)

        head_idx = td["trajectory_head"].detach().squeeze(-1).clone()
        batch_indices = torch.arange(td.batch_size[0], device=head_idx.device)
        head_emb = cache.node_embeddings[batch_indices, head_idx]
        
        # (B, D+3) -> (B, 1, D)
        query_input = torch.cat([head_emb, state_features], dim=1).unsqueeze(1)
        
        # 초기 q_vec (Projection)
        q_vec = self.input_projector(query_input)

        # 2. 디코더 레이어 순차 통과 (Stacking)
        # q_vec이 각 레이어를 거치며 점점 더 정교한 Context Vector가 됩니다.
        # encoder_out = cache.node_embeddings # (B, N, D) <-- 삭제 (캐시 사용)
        
        for i, layer in enumerate(self.layers):
            k_cache, v_cache = cache.decoder_layer_kvs[i]
            q_vec = layer(q_vec, k_cache, v_cache)

        # --- 3. 최종 결정 (Heads) ---
        value = self.value_head(q_vec).squeeze(-1)
        logits_action_type = self.type_head(q_vec).squeeze(1)
        
        query_connect = self.connect_head(q_vec) 
        logits_connect_target = torch.matmul(
            query_connect, cache.logit_key_connect
        ).squeeze(1) / (self.embedding_dim ** 0.5)
        
        query_spawn = self.spawn_head(q_vec) 
        logits_spawn_template = torch.matmul(
            query_spawn, cache.logit_key_spawn
        ).squeeze(1) / (self.embedding_dim ** 0.5)

        return logits_action_type, logits_connect_target, logits_spawn_template, value

class PocatModel(nn.Module):
    """
    Pocat V7 (Padding + Lazy Spawn) 메인 모델
    """
    
    def __init__(self, **model_params):
        super().__init__()
        self.logit_clipping = model_params.get('logit_clipping', 10)
        
        # config.yaml에서 N_MAX 주입
        self.N_MAX = model_params['N_MAX']
        # model_params에서 N_MAX를 pop하여 중복 전달 방지
        # (PocatPromptNet과 PocatDecoder는 N_MAX를 명시적 인자로 받음)s
        n_max_value = model_params.pop('N_MAX')
        self.prompt_net = PocatPromptNet(N_MAX=n_max_value, **model_params)
        self.encoder = PocatEncoder(**model_params)
        self.decoder = PocatDecoder(N_MAX=n_max_value, **model_params)

    def _get_masked_probs(self, logits, mask):
        """ 로짓과 마스크를 받아 정규화된 확률 분포를 반환합니다. """
        scores = self.logit_clipping * torch.tanh(logits)
        scores.masked_fill_(~mask, -float('inf'))
        
        # [수정] NaN 및 In-place 오류 방지 (Out-of-place)
        # 전부 -inf인 경우(막다른 길) -> softmax가 NaN 반환 방지
        if (scores == -float('inf')).all(dim=-1).any():
             # 마스킹된 곳은 그대로 두고, 전부 마스킹된 행만 0으로 대체하여 softmax 계산
             safe_scores = scores.clone()
             safe_scores[safe_scores == -float('inf')] = -1e9 # 매우 작은 수로 대체
             probs = F.softmax(safe_scores, dim=-1)
        else:
             probs = F.softmax(scores, dim=-1)

        return probs  

    def _sample_action(self, logits, mask, decode_type, temperature=1.0): # [추가] temperature
        """ 
        로짓과 마스크를 받아 액션(idx)과 로그 확률(log_prob)을 반환합니다.
        (막다른 길 방지 로직 포함)
        """
        scores = self.logit_clipping * torch.tanh(logits)
        scores.masked_fill_(~mask, -float('inf'))

        # [추가] Temperature Scaling (확률 분포를 평평하게 만듦)
        # 값이 클수록(>1.0) 무작위성이 강해짐
        scores = scores / temperature

        # 모든 액션이 마스킹된 '막다른 길' 상태 방지
        # (모든 값이 -inf이면 max도 -inf)
        max_scores, _ = scores.max(dim=-1)
        is_stuck = (max_scores == -float('inf'))
        
        # [수정] In-place 연산 방지: scores[is_stuck, 0] = 0.0 대신 torch.where 사용
        if is_stuck.any():
            # 막다른 길인 경우 0번 인덱스에 0.0을 할당하고 나머지는 유지
            # (In-place 할당 대신 마스킹으로 처리)
            fallback_scores = torch.zeros_like(scores)
            fallback_scores.fill_(-float('inf'))
            fallback_scores[:, 0] = 0.0
            
            scores = torch.where(is_stuck.unsqueeze(-1), fallback_scores, scores)
        
        log_prob = F.log_softmax(scores, dim=-1)
        probs = log_prob.exp()
        # [수정] NaN 안전장치 (In-place 연산 완전 배제)
        if torch.isnan(probs).any():
            # 1. NaN을 0으로 대체 (Out-of-place)
            clean_probs = torch.where(torch.isnan(probs), torch.zeros_like(probs), probs)
            
            # 2. 합이 0인 행(전부 NaN/Masked) 처리
            sum_probs = clean_probs.sum(dim=-1, keepdim=True)
            # 합이 0이면 균등 분포(또는 0번 몰빵)로 대체
            fallback_probs = torch.zeros_like(clean_probs)
            fallback_probs[:, 0] = 1.0
            
            final_probs = torch.where(sum_probs == 0, fallback_probs, clean_probs)
            probs = final_probs # 교체
            
            # 3. Log Prob 재계산 (그래프 단절 없이 안전하게)
            log_prob = torch.log(probs + 1e-10)
      
        # --- [추가] 엔트로피 계산 ---
        dist = Categorical(probs=probs)
        entropy = dist.entropy()
        # ---------------------------

        if decode_type == 'greedy':
            action = probs.argmax(dim=-1)
        else: # 'sampling'
            # -------------------------------------------------------
            # 🛡️ 전략 2: Epsilon Mixing (Policy + Uniform Sampling)
            #    - 학습 중일 때 10% 확률로 Policy 무시하고 무작위 탐색
            # -------------------------------------------------------
            if self.training and random.random() < 0.1:
                # [수정] 배치 처리를 지원하는 Uniform Sampling
                # (랜덤 노이즈를 더해서 argmax를 하면 유효한 액션 중 랜덤 선택과 동일함)
                random_noise = torch.rand_like(logits)
                random_noise.masked_fill_(~mask, -float('inf')) # 무효 액션 제외
                action = random_noise.argmax(dim=-1) # (B,) 크기의 랜덤 액션 반환
            else:
                # 기존 Policy Sampling (90%)
                action = Categorical(probs=probs).sample()
            # -------------------------------------------------------
            
        # 선택된 액션의 로그 확률 반환
        return action, log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1), entropy

    def _combine_log_probs(self, 
                           log_prob_type, action_type, 
                           log_prob_connect, log_prob_spawn):
        """
        Parameterized Action의 로그 확률을 결합합니다.
        logπ(a|s) = logπ(type|s) + logπ(arg|type,s)
        """
        # 'Connect' (0)를 선택한 경우의 로그 확률
        log_prob_if_connect = log_prob_type + log_prob_connect
        # 'Spawn' (1)을 선택한 경우의 로그 확률
        log_prob_if_spawn = log_prob_type + log_prob_spawn
        
        # (B,)
        final_log_prob = torch.where(
            action_type == 0,       # 'Connect'를 선택했으면
            log_prob_if_connect,    # 이 확률을 사용
            log_prob_if_spawn       # 아니면 (Spawn) 이 확률을 사용
        )
        return final_log_prob

    def forward(self, 
                td: TensorDict, 
                env: PocatEnv, # (solver_env.py의 환경 객체)
                decode_type: str = 'greedy', 
                pbar: object = None,
                status_msg: str = "", 
                log_fn=None, log_idx: int = 0, 
                log_mode: str = 'progress',
                return_final_td: bool = False,   # 👈 이 줄 추가
                ) -> Dict[str, torch.Tensor]:
        
        base_desc = pbar.desc.split(' | ')[0] if pbar else ""
        if pbar: pbar.set_description(f"{base_desc} | {status_msg} | ▶ Encoding")
        
        # --- 1. 인코딩 및 캐시 생성 ---
        prompt_embedding = self.prompt_net(td["scalar_prompt_features"], td["matrix_prompt_features"])
        encoded_nodes = self.encoder(td, prompt_embedding) # (B, N_MAX, D)
        
        # 디코더가 사용할 Key/Value 사전 계산
        #glimpse_key = reshape_by_heads(self.decoder.Wk_glimpse(encoded_nodes), self.decoder.head_num)
        #glimpse_val = reshape_by_heads(self.decoder.Wv_glimpse(encoded_nodes), self.decoder.head_num)
        
        # 포인터 헤드별 Key 생성
        logit_key_connect = self.decoder.Wk_connect_logit(encoded_nodes).transpose(1, 2)
        logit_key_spawn = self.decoder.Wk_spawn_logit(encoded_nodes).transpose(1, 2)

        # [추가] 디코더 레이어용 K, V 미리 계산 (Pre-computation)
        # 루프 밖에서 한 번만 계산하므로 메모리와 연산량이 획기적으로 줄어듭니다.
        decoder_layer_kvs = []
        for layer in self.decoder.layers:
            # (B, N, D) -> (B, H, N, D/H)
            k = reshape_by_heads(layer.Wk(encoded_nodes), layer.head_num)
            v = reshape_by_heads(layer.Wv(encoded_nodes), layer.head_num)
            decoder_layer_kvs.append((k, v))

        cache = PrecomputedCache(
            node_embeddings=encoded_nodes,
            #glimpse_key=glimpse_key,
            #glimpse_val=glimpse_val,
            logit_key_connect=logit_key_connect,
            logit_key_spawn=logit_key_spawn,
            decoder_layer_kvs=decoder_layer_kvs # [추가]
        )
        
        # --- 2. POMO (Multi-Start) 준비 ---
        num_starts, start_nodes_idx = env.select_start_nodes(td)
        if num_starts == 0:
             # (B, 1) 형태의 0점 리워드 반환
            zero_reward = torch.zeros(td.batch_size[0], 1, device=td.device)
            return {"reward": zero_reward} # (POMO 시작 불가)

        batch_size = td.batch_size[0]
        
        # (B) -> (B * num_starts)
        # [수정] utils.batchify 대신 tensordict 내장 메서드 사용
        # (POMO: 각 샘플을 num_starts만큼 복제)
        td = td.repeat_interleave(num_starts, dim=0)

        cache = cache.batchify(num_starts) # 캐시도 확장

        # POMO 시작: 첫 액션(Load 선택)을 환경에 강제 적용
        first_action_tensor = start_nodes_idx.repeat(batch_size).unsqueeze(-1)
        
        # (POMO의 첫 스텝은 env._reset에서 처리되도록 solver_env.py에서 구현 필요)
        # (여기서는 td가 이미 첫 Load가 Head로 설정된 상태라고 가정합니다.)
        
        # --- 3. 디코딩 루프 ---
        log_probs: List[torch.Tensor] = []
        actions: List[Dict[str, torch.Tensor]] = []
        rewards: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = [] # [추가] 엔트로피 저장용
        first_value: torch.Tensor = None
        
        decoding_step = 0
        while not td["done"].all():
            decoding_step += 1
            if pbar and log_mode == 'progress':
                # [수정] 0번 샘플의 실제 Load 개수 계산 (랜덤 배치 대응)
                node_types = td["nodes"][0, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
                sample_num_loads = (node_types == NODE_TYPE_LOAD).sum().item()
                
                # 현재 남은 연결 안 된 Load 수
                unconnected = td['unconnected_loads_mask'][0].sum().item() 
                connected = sample_num_loads - unconnected
                
                pbar.set_description(f"{base_desc} | {status_msg} | Loads {connected}/{sample_num_loads}")

            # 1. 디코더 호출 (4개 텐서 반환)
            logits_type, logits_connect, logits_spawn, value = self.decoder(td, cache)
            
            # A2C를 위해 첫 스텝의 Value(가치) 저장
            if decoding_step == 1:
                first_value = value.squeeze(-1) # (B * N_loads, 1) -> (B * N_loads)
            
            # 2. 환경에서 3종 마스크 가져오기
            # (solver_env.py가 반환할 마스크 딕셔너리)
            with torch.no_grad():
                masks: Dict[str, torch.Tensor] = env.get_action_mask(td)
            
            # [추가] Temperature 스케줄링 (학습 모드일 때만 적용)
            # 학습 초반에는 5.0 등으로 높게 설정하여 강제 탐색 유도 필요
            temp = 1.0 
            if self.training: # model.train() 상태일 때
                 # 예: 로그 등을 통해 외부에서 제어하거나, 일단 상수로 테스트
                 temp = 2.0


            # 3. 3개 헤드에서 각각 샘플링
            action_type, log_prob_type, ent_type = self._sample_action(
                logits_type, masks["mask_type"], decode_type, temperature=temp
            )
            action_connect, log_prob_connect, ent_connect = self._sample_action(
                logits_connect, masks["mask_connect"], decode_type, temperature=temp
            )
            action_spawn, log_prob_spawn, ent_spawn = self._sample_action(
                logits_spawn, masks["mask_spawn"], decode_type, temperature=temp
            )

            # [추가] 스텝별 총 엔트로피 합 산 (Action Type + Argument)
            # Connect를 골랐으면 Connect 엔트로피, Spawn이면 Spawn 엔트로피 사용
            step_entropy = ent_type + torch.where(action_type == 0, ent_connect, ent_spawn)
            entropies.append(step_entropy)

            # 4. Parameterized Action Log Prob 결합
            final_log_prob = self._combine_log_probs(
                log_prob_type, action_type, 
                log_prob_connect, log_prob_spawn
            )
            
            # 5. 환경에 전달할 액션 딕셔너리 생성
            action_dict = {
                "action_type": action_type.unsqueeze(-1),
                "connect_target": action_connect.unsqueeze(-1),
                "spawn_template": action_spawn.unsqueeze(-1),
            }
            
            # [START]: 'detail' 모드 액션 로깅 (수정됨)
            if log_fn and log_mode == 'detail':
                # (첫 번째 샘플(B=0) 기준으로 로그 출력)
                sample_idx = 0
                if sample_idx < td.batch_size[0]:
                    current_head = td["trajectory_head"][sample_idx].item()
                    
                    # --- 1. 확률 분포 계산 ---
                    # (위에서 정의한 _get_masked_probs 사용)
                    probs_type = self._get_masked_probs(logits_type[sample_idx], masks["mask_type"][sample_idx])
                    probs_connect = self._get_masked_probs(logits_connect[sample_idx], masks["mask_connect"][sample_idx])
                    probs_spawn = self._get_masked_probs(logits_spawn[sample_idx], masks["mask_spawn"][sample_idx])

                    # [추가] 원본 점수(Score) 계산 (Softmax 전 단계의 값)
                    # Score = Tanh(Logit) * Clipping_Value (예: -10 ~ +10 사이)
                    scores_type = self.logit_clipping * torch.tanh(logits_type[sample_idx]) # [추가] Type 점수
                    scores_connect = self.logit_clipping * torch.tanh(logits_connect[sample_idx])
                    scores_spawn = self.logit_clipping * torch.tanh(logits_spawn[sample_idx])

                    # [추가] 클리핑 전 원본 로짓(Raw Logit) 추출
                    raw_type = logits_type[sample_idx]
                    raw_connect = logits_connect[sample_idx]
                    raw_spawn = logits_spawn[sample_idx]

                    # --- 2. 이름 매핑 준비 ---
                    # (환경 설정에서 정적 이름 목록 가져오기)
                    node_names = env.generator.config.node_names
                    # [수정] 원본 템플릿을 추적하여 이름을 반환하는 함수
                    def get_name_with_origin(idx):

                        # 1. 텐서에서 실제 노드 타입 및 정보 확인
                        node_feat = td["nodes"][sample_idx, idx]
                        node_type = node_feat[FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax().item()

                        # 2. Battery
                        if node_type == NODE_TYPE_BATTERY:
                            return "BATTERY"
                        
                        # 3. Load (랜덤 생성된 스펙 표시)
                        if node_type == NODE_TYPE_LOAD:
                            v = node_feat[FEATURE_INDEX["vin_min"]].item()
                            i = node_feat[FEATURE_INDEX["current_active"]].item()
                            return f"RandomLoad_{idx} ({v:.1f}V, {i:.2f}A)"
                        
                        # 4. IC (Template) - 정적 리스트 매핑
                        if node_type == NODE_TYPE_IC:
                            # 현재 배치의 실제 Load 개수 계산
                            node_types_all = td["nodes"][sample_idx, :, FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax(-1)
                            num_current_loads = (node_types_all == NODE_TYPE_LOAD).sum().item()

                            # 현재 노드가 템플릿 섹션(Battery + Loads 이후)에 있는지 확인
                            ic_start_idx = 1 + num_current_loads
                            
                            # 템플릿이라면 정적 리스트에서 이름 찾기
                            if idx >= ic_start_idx:
                                # 정적 리스트 구조: [Batt(1)] + [FixedLoads] + [Templates]
                                # 동적 텐서 구조: [Batt(1)] + [RandomLoads] + [Templates]
                                # 따라서 Templates 내에서의 상대적 위치(offset)는 동일함
                                offset = idx - ic_start_idx
                                
                                static_ic_start = 1 + env.generator.num_loads # (초기 JSON 로드 개수)
                                target_static_idx = static_ic_start + offset
                                
                                if 0 <= target_static_idx < len(node_names):
                                    return node_names[target_static_idx]

                        # 5. 그 외 (Empty, Spawned 등)
                        return f"Node_{idx}"

                    head_name = get_name_with_origin(current_head) # [수정]

                    # ---------------------------------------------------------
                    # [추가] 첫 스텝에서 생성된 문제(Load & Constraints) 명세서 출력
                    # ---------------------------------------------------------
                    if decoding_step == 1:
                        log_fn("="*80)
                        log_fn(f"🎲 Generated Problem Specification (Sample 0)")
                        log_fn("-" * 80)
                        
                        # 1. Scalar Constraints (Prompt)
                        # scalar_p: [Temp, MaxSleep, I_Margin, Therm_Margin] (env_generator 참조)
                        scalar_feats = td["scalar_prompt_features"][sample_idx]
                        temp = scalar_feats[0].item()
                        max_sleep = scalar_feats[1].item()
                        log_fn(f"🌡️ Conditions: Ambient Temp={temp:.1f}°C, Max Sleep Current={max_sleep*1000:.1f}mA")
                        
                        # 2. Sequence Constraints (Matrix Prompt)
                        mat_feats = td["matrix_prompt_features"][sample_idx]
                        seq_srcs, seq_dsts = torch.where(mat_feats > 0.5)
                        if len(seq_srcs) > 0:
                            log_fn(f"⛓️ Power Sequences ({len(seq_srcs)} Constraints):")
                            for s, d in zip(seq_srcs, seq_dsts):
                                s_name = get_name_with_origin(s.item())
                                d_name = get_name_with_origin(d.item())
                                log_fn(f"   ► {s_name}  -->  {d_name}")
                        else:
                            log_fn(f"⛓️ Power Sequences: None")

                        # 3. Load List
                        log_fn("-" * 80)
                        log_fn(f"📦 Generated Loads List:")
                        
                        load_count = 0
                        for i in range(self.N_MAX):
                            # 노드 타입 확인
                            nf = td["nodes"][sample_idx, i]
                            nt = nf[FEATURE_INDEX["node_type"][0]:FEATURE_INDEX["node_type"][1]].argmax().item()
                            
                            if nt == NODE_TYPE_LOAD:
                                load_count += 1
                                v = nf[FEATURE_INDEX["vin_min"]].item()
                                i_act = nf[FEATURE_INDEX["current_active"]].item()
                                i_slp = nf[FEATURE_INDEX["current_sleep"]].item()
                                
                                rail_val = nf[FEATURE_INDEX["independent_rail_type"]].item()
                                ao_val = nf[FEATURE_INDEX["always_on_in_sleep"]].item()
                                
                                # 태그 정보 (Supplier, Path, AO 등)
                                tags = []
                                if rail_val == 1.0: tags.append("Supplier")
                                if rail_val == 2.0: tags.append("Path")
                                if ao_val == 1.0: tags.append("AO")
                                tag_str = f"[{', '.join(tags)}]" if tags else ""
                                
                                log_fn(f"   - [Idx {i:03d}] {v:.2f}V / {i_act:.2f}A (Sleep: {i_slp*1000:.2f}mA) {tag_str}")
                        
                        log_fn(f"   (Total {load_count} Loads)")
                        log_fn("="*80)
                    # ---------------------------------------------------------

                    
                    # =========================================================
                    # ✨ [수정] Rail Type + AO 상태 정보 추출 및 로그 포맷 ✨                    # =========================================================
                    # 1. Rail Type (독립 여부)
                    rail_val = td["nodes"][sample_idx, current_head, FEATURE_INDEX["independent_rail_type"]].item()
                    # 가독성을 위한 문자열 매핑 (0:Normal, 1:Sup, 2:Path)
                    if rail_val == 1.0: rail_str = "Type: Supplier(1)"
                    elif rail_val == 2.0: rail_str = "Type: Path(2)"
                    else: rail_str = "Type: Normal(0)"
                    
                    # 2. AO State (암전류 상태)
                    ao_val = td["nodes"][sample_idx, current_head, FEATURE_INDEX["always_on_in_sleep"]].item()
                    ao_str = "AO: Yes" if ao_val == 1.0 else "AO: No"
                    
                    # (idx와 type을 함께 출력)
                    log_fn(f"\n[Step {decoding_step:02d}] Current Head: {head_name} (idx: {current_head} | {rail_str} | {ao_str})")
                    # =========================================================

                    # --- 3. Action Type 확률 출력 ---
                    p_conn = probs_type[0].item()
                    p_spwn = probs_type[1].item()
                    
                    s_conn = scores_type[0].item()
                    s_spwn = scores_type[1].item()

                    r_conn = raw_type[0].item()
                    r_spwn = raw_type[1].item()

                    chosen_type = action_type[sample_idx].item()
                    type_str = "Connect" if chosen_type == 0 else "Spawn"
                    
                    is_connect_valid = masks["mask_type"][sample_idx, 0].item()
                    is_spawn_valid = masks["mask_type"][sample_idx, 1].item()

                    tag_conn = "" if is_connect_valid else " 🚫 [Masked]"
                    tag_spwn = "" if is_spawn_valid else " 🚫 [Masked]"
                    
                    log_fn(f"  📊 Action Type Probabilities:")
                    log_fn(f"     - Connect: {p_conn*100:.2f}% (Sc: {s_conn:6.3f} | Raw: {r_conn:6.3f}){tag_conn} {'👈 Selected' if chosen_type==0 else ''}")
                    log_fn(f"     - Spawn  : {p_spwn*100:.2f}% (Sc: {s_spwn:6.3f} | Raw: {r_spwn:6.3f}){tag_spwn} {'👈 Selected' if chosen_type==1 else ''}")
                    # --- 4. 상세 후보 확률 출력 ---
                    
                    # (A) Connect 후보들
                    if masks["mask_type"][sample_idx, 0]: # Connect가 가능한 경우만
                        log_fn(f"  🔗 Connect Candidates (P(Target | Connect)):")
                        valid_connect_indices = torch.where(masks["mask_connect"][sample_idx])[0]
                        
                        # 확률순 정렬
                        cand_probs = []
                        for idx in valid_connect_indices:
                            i = idx.item()
                            prob = probs_connect[i].item()
                            score = scores_connect[i].item()
                            raw = raw_connect[i].item()
                            
                            # Connect 마스크 확인 (디버깅용)
                            is_valid = masks["mask_connect"][sample_idx, i].item()
                            tag = "" if is_valid else " 🚫 [Masked] (Error?)"
                            
                            cand_probs.append((prob, score, raw, i, tag))                            
                        cand_probs.sort(key=lambda x: x[0], reverse=True)

                        # [수정] 5개 항목 언패킹 (tag 포함)
                        for prob, score, raw, idx, tag in cand_probs:
                            name = get_name_with_origin(idx)
                            is_picked = (chosen_type == 0 and action_connect[sample_idx].item() == idx)
                            # 이름 공간을 25 -> 60으로 늘림 (긴 이름 표시용)
                            log_fn(f"     - {name:<60} : {prob*100:.2f}% (Sc: {score:6.3f} | Raw: {raw:6.3f}){tag} {'✅' if is_picked else ''}")

                    if masks["mask_type"][sample_idx, 1]: # Spawn이 가능한 경우만
                        log_fn(f"  📦 Spawn Candidates (P(Template | Spawn)):")
                        valid_spawn_indices = torch.where(masks["mask_spawn"][sample_idx])[0]
                        
                        cand_probs = []
                        for idx in valid_spawn_indices:
                            i = idx.item()
                            prob = probs_spawn[i].item()
                            score = scores_spawn[i].item()
                            raw = raw_spawn[i].item()

                            is_valid = masks["mask_spawn"][sample_idx, i].item()
                            tag = "" if is_valid else " 🚫 [Masked] (Error?)"

                            cand_probs.append((prob, score, raw, i, tag))

                        cand_probs.sort(key=lambda x: x[0], reverse=True)

                        for prob, score, raw, idx, tag in cand_probs:
                            name = get_name_with_origin(idx)
                            is_picked = (chosen_type == 1 and action_spawn[sample_idx].item() == idx)
                            log_fn(f"     - {name:<60} : {prob*100:.2f}% (Sc: {score:6.3f} | Raw: {raw:6.3f}){tag} {'✅' if is_picked else ''}")

                    log_fn("-" * 60)
            # [END]: 'detail' 모드 액션 로깅

            # 6. 환경 스텝 실행
            with torch.no_grad():
                td.set("action", action_dict)
                output_td = env.step(td)
            
            reward = output_td["reward"]
            td = output_td["next"]
            
            # 7. A2C 학습을 위한 데이터 수집
            log_probs.append(final_log_prob)
            actions.append(action_dict)
            rewards.append(reward)

        # 8. 최종 결과 취합
        if not rewards:
            # (디코딩 루프가 1번도 돌지 않은 경우 - 예: 이미 완료된 상태)
            B_total = td.batch_size[0]
            dummy_reward = torch.zeros(B_total, 1, device=td.device)
            dummy_log_prob = torch.zeros(B_total, device=td.device)
            dummy_value = torch.zeros(B_total, 1, device=td.device)
            return {
                "reward": dummy_reward,
                "log_likelihood": dummy_log_prob,
                "actions": [],
                "value": dummy_value,
            }

        # (B_total, T) -> (B_total, 1)
        total_reward = torch.stack(rewards, 1).sum(1)
        # (B_total, T) -> (B_total)
        total_log_likelihood = torch.stack(log_probs, 1).sum(1)

        # [추가] 평균 엔트로피 계산
        if entropies:
            avg_entropy = torch.stack(entropies, 1).mean(1) # (B,) 에피소드 평균
        else:
            avg_entropy = torch.zeros_like(total_log_likelihood)

        # [추가] 최종 상태에서 비용 정보 추출
        final_bom_cost = td["current_cost"].squeeze(-1)
        final_sleep_cost = td["sleep_cost"].squeeze(-1)


        result = {
            "reward": total_reward,
            "log_likelihood": total_log_likelihood,
            "entropy": avg_entropy, # [추가]
            "actions": actions,  # (디버깅용)
            "value": first_value,
            "bom_cost": final_bom_cost, # [추가]
            "sleep_cost": final_sleep_cost, # [추가]
        }

        if return_final_td:
            # 시각화/디버깅용 최종 상태는 GPU 전체 TensorDict를 통째로
            # clone() 하는 대신,
            #  - 그래디언트 연결을 끊고(detach)
            #  - 필요한 키만 골라서
            #  - CPU 메모리로만 저장한다.
            #
            # visualize_result()에서 사용하는 키:
            #   - "nodes"
            #   - "adj_matrix"
            #   - "is_active_mask"
            final_td_cpu = TensorDict(
                {
                    "nodes": td["nodes"].detach().cpu(),
                    "adj_matrix": td["adj_matrix"].detach().cpu(),
                    "is_active_mask": td["is_active_mask"].detach().cpu(),
                },
                batch_size=td.batch_size,
            )
            result["final_td"] = final_td_cpu


        return result