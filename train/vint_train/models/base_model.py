"""
로봇이 이미지 기반으로 목표까지 가는 경로를 예측하는 뼈대 역할
"""

import torch
import torch.nn as nn

from typing import List, Dict, Optional, Tuple


class BaseModel(nn.Module):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
    ) -> None:
        """
        Base Model의 기본 클래스

        Args:
            context_size (int): 로봇이 과거 몇 개의 관측 정보를 참고할지 지정 (문맥 크기)
            len_traj_pred (int): 미래에 예측할 경로 포인트의 개수
            learn_angle (bool): 로봇의 회전 각도(yaw)를 학습할지 여부
        """
        super(BaseModel, self).__init__()
        self.context_size = context_size  # 문맥 크기 저장
        self.learn_angle = learn_angle  # 각도 학습 여부 저장
        self.len_trajectory_pred = len_traj_pred  # 예측할 경로 길이 저장

        # 학습할 액션 파라미터 개수 설정
        # 각도를 학습하는 경우 (cos, sin 포함하여) 4개, 아니면 2개만 사용
        if self.learn_angle:
            self.num_action_params = 4  # x, y, cos(각도), sin(각도)
        else:
            self.num_action_params = 2  # x, y만 사용

    def flatten(self, z: torch.Tensor) -> torch.Tensor:
        """
        입력 텐서를 평균 풀링으로 (1, 1) 크기로 만들고, 평탄화(flatten)하는 함수

        Args:
            z (torch.Tensor): 입력 특성 맵 (Feature Map)

        Returns:
            torch.Tensor: 평탄화된 텐서 (Batch, Feature) 형태
        """
        z = nn.functional.adaptive_avg_pool2d(z, (1, 1))  # (1, 1) 크기로 평균 풀링
        z = torch.flatten(z, 1)  # 배치 차원 유지한 채 평탄화
        return z

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        모델의 순전파 (Forward Pass) 함수
        > 여기에 모델 구현해야 함

        Args:
            obs_img (torch.Tensor): 관측 이미지 (현재 로봇이 보는 이미지들)
            goal_img (torch.Tensor): 목표 이미지 (도착지점 혹은 목표 위치 이미지)

        Returns:
            dist_pred (torch.Tensor): 목표까지의 거리 예측값
            action_pred (torch.Tensor): 로봇의 다음 동작 (이동 및 회전) 예측값
        """
        raise NotImplementedError  # BaseModel에서는 구현되지 않음 (상속받아 구현해야 함)
