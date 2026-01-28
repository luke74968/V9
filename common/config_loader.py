# Copyright (c) 2025 Minuk Lee. All rights reserved.
# 
# This source code is proprietary and confidential.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# 
# For licensing terms, see the LICENSE file.
# Contact: minuklee@snu.ac.kr
# 
"""
JSON 설정 파일 로더 (common/config_loader.py)

이 파일은 JSON 설정 파일 (예: config.json)을 읽어들여,
`common/data_classes.py`에 정의된 
Battery, Load, PowerIC (LDO, Buck) 객체 리스트로 변환하는 역할을 합니다.

주요 기능:
- `load_configuration_from_file`: 파일 경로를 받아 JSON 문자열을 읽어옵니다.
- `load_configuration_from_json`: JSON 문자열을 파싱하여 
                                Battery, Load, LDO, BuckConverter 객체를 생성합니다.
"""

import json
from typing import List, Dict, Tuple, Any

# data_classes에서 클래스들을 임포트합니다.
from .data_classes import Battery, Load, PowerIC, LDO, BuckConverter

def load_configuration_from_json(config_string: str) -> Tuple[Battery, List[PowerIC], List[Load], Dict[str, Any]]:
    """
    JSON 설정 문자열을 파싱하여 데이터 객체들과 제약조건을 반환합니다.
    
    Args:
        config_string (str): JSON 파일의 내용을 담고 있는 문자열
        
    Returns:
        Tuple: (battery 객체, IC 객체 리스트, Load 객체 리스트, constraints 딕셔너리)
    """
    config = json.loads(config_string)
    
    # 1. 배터리 로드
    battery = Battery(**config['battery'])
    
    # 2. Power IC 로드
    available_ics = []
    # [추가] JSON 키 -> 파이썬 클래스 필드 매핑 테이블
    key_mapping = {
        # 공통 및 LDO
        "op_current": "operating_current",
        "q_current": "quiescent_current",
        "shut_current": "shutdown_current",
        "t_j_max": "t_junction_max",
        
        # Buck 전용 (효율 정보 포함)
        "eff_op": "efficiency_active",
        "eff_sleep": "efficiency_sleep",
        "not_switching_current": "quiescent_current"
    }

    for ic_data in config['available_ics']:
        
        # 1) 변수명 매핑 적용 (JSON Key -> Class Field)
        for json_key, class_key in key_mapping.items():
            if json_key in ic_data:
                ic_data[class_key] = ic_data.pop(json_key)

        # 2) i_limit 처리 (기존 로직 유지)
        if 'i_limit' in ic_data:
            ic_data['original_i_limit'] = ic_data.pop('i_limit')
        
        # 3) Buck의 'operating_current' 누락 처리 (중요!)
        # Buck은 효율(Efficiency)로 손실을 계산하므로, 별도의 operating_current는 0.0으로 설정하여
        # 중복 계산을 방지하고 __init__ 에러를 해결합니다.
        if 'operating_current' not in ic_data:
            ic_data['operating_current'] = 0.0

        # 4) 객체 생성
        ic_type = ic_data.pop('type')
        if ic_type == 'LDO':
            available_ics.append(LDO(**ic_data))
        elif ic_type == 'Buck':
            available_ics.append(BuckConverter(**ic_data))
        
        # 다른 모듈 참조용 type 복구
        ic_data['type'] = ic_type

    # 3. Load 로드
    loads = [Load(**load_data) for load_data in config['loads']]
    
    # 4. 제약조건 로드
    constraints = config['constraints']
    
    return battery, available_ics, loads, constraints

def load_configuration_from_file(filepath: str) -> Tuple[Battery, List[PowerIC], List[Load], Dict[str, Any]]:
    """
    JSON 파일 경로를 입력받아 설정 객체들을 로드합니다.
    
    Args:
        filepath (str): 로드할 config.json 파일의 경로
        
    Returns:
        Tuple: (battery 객체, IC 객체 리스트, Load 객체 리스트, constraints 딕셔너리)
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            json_config_string = f.read()
        print(f"✅ 설정 파일 로드 성공: '{filepath}'")
        return load_configuration_from_json(json_config_string)
    except FileNotFoundError:
        print(f"❌ 설정 파일 로드 실패: '{filepath}'을(를) 찾을 수 없습니다.")
        # 빈 리스트와 딕셔너리를 반환하여 프로그램이 즉시 중단되는 것을 방지
        return Battery(name="Error", voltage_min=0, voltage_max=0, capacity_mah=0), [], [], {}
    except Exception as e:
        print(f"❌ 설정 파일 처리 중 오류 발생: {e}")
        return Battery(name="Error", voltage_min=0, voltage_max=0, capacity_mah=0), [], [], {}