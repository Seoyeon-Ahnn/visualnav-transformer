import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Dict, Optional, Tuple
from vint_train.models.gnm.modified_mobilenetv2 import MobileNetEncoder
from vint_train.models.base_model import BaseModel


# GNM 모델 클래스 정의 (BaseModel을 상속)
class GNM(BaseModel):
    def __init__(
        self,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        goal_encoding_size: Optional[int] = 1024,
    ) -> None:
        """
        GNM 메인 클래스
        Args:
            context_size (int): 이전 관측 정보를 몇 개 사용할지 결정
            len_traj_pred (int): 미래 경로로 예측할 중간 목표 지점(waypoint)의 수
            learn_angle (bool): 로봇의 회전각(yaw)을 예측할지 여부
            obs_encoding_size (int): 관측 이미지 인코딩 크기
            goal_encoding_size (int): 목표 이미지 인코딩 크기
        """
        super(GNM, self).__init__(context_size, len_traj_pred, learn_angle)

        # MobileNetEncoder를 사용하여 관측 이미지 인코딩
        mobilenet = MobileNetEncoder(num_images=1 + self.context_size)    # 현재 관측 이미지 1장 추가
        self.obs_mobilenet = mobilenet.features
        self.obs_encoding_size = obs_encoding_size

        # 관측 이미지의 특징 벡터 차원 압축 (선형 변환 후 ReLU 적용)
        self.compress_observation = nn.Sequential(
            nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
        )
        
        # 목표(goal) 이미지 1장과 현재 관측 이미지 1장을 스택하여 인코딩 (총 2 + context_size 이미지)
        stacked_mobilenet = MobileNetEncoder(
            num_images=2 + self.context_size
        )
        self.goal_mobilenet = stacked_mobilenet.features
        self.goal_encoding_size = goal_encoding_size

        # 목표 이미지의 특징 벡터 차원 압축 및 변환 (여러 선형 레이어와 ReLU 적용)
        self.compress_goal = nn.Sequential(
            nn.Linear(stacked_mobilenet.last_channel, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.goal_encoding_size),
            nn.ReLU(),
        )

        # 관측 인코딩과 목표 인코딩을 합쳐서 중간 특성 벡터로 변환하는 선형 레이어들
        self.linear_layers = nn.Sequential(
            nn.Linear(self.goal_encoding_size + self.obs_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )

        # 목표와 관련된 거리 예측을 위한 선형 레이어
        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )

        # 행동(경로) 예측을 위한 선형 레이어
        # 예측할 행동의 파라미터 수는 len_traj_pred * num_action_params (num_action_params는 BaseModel에서 정의)
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    # 모델의 순전파(forward) 함수 정의
    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. 관측 이미지 처리 (로봇의 위치나 주변 상황을 이해하는 데 중점을 둠)
        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self.flatten(obs_encoding)
        obs_encoding = self.compress_observation(obs_encoding)

        # 2. 목표 이미지 처리 (관측 이미지와 목표 이미지를 채널 차원(dim=1)에서 이어붙여 입력 생성, 관측 정보와 목표 정보 간의 상호작용과 관계를 파악)
        obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
        goal_encoding = self.goal_mobilenet(obs_goal_input)
        goal_encoding = self.flatten(goal_encoding)
        goal_encoding = self.compress_goal(goal_encoding)

        # 3. 관측 인코딩과 목표 인코딩을 결합하여 중간 특징 벡터 생성
        z = torch.cat([obs_encoding, goal_encoding], dim=1)
        z = self.linear_layers(z)

        # 4. 거리 예측: 중간 벡터를 사용해 목표와 관련된 거리를 예측
        dist_pred = self.dist_predictor(z)

        # 5. 행동(경로) 예측: 중간 벡터를 사용해 행동 파라미터를 예측
        action_pred = self.action_predictor(z)

        # 6. 출력 형상 조정: 예측된 행동 파라미터를 (batch_size, len_traj_pred, num_action_params) 형태로 reshape
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        # 위치 정보(첫 2개 파라미터): 누적 합(cumsum)을 통해 델타값을 실제 waypoint로 변환
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )
        # 7. 회전각(yaw) 예측: learn_angle이 True이면, 나머지 파라미터(각도)를 정규화 (벡터의 길이를 1로)
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )
        # 거리 예측과 행동 예측을 튜플로 반환
        return dist_pred, action_pred
