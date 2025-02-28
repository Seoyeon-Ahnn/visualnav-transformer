import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from efficientnet_pytorch import EfficientNet
from vint_train.models.base_model import BaseModel
from vint_train.models.vint.self_attention import MultiLayerDecoder

class ViNT(BaseModel):
    def __init__(
        self,
        context_size: int = 5,  # 과거 몇 장의 이미지를 참고할지 (과거 프레임 수)
        len_traj_pred: Optional[int] = 5,  # 앞으로 몇 개의 waypoint를 예측할지
        learn_angle: Optional[bool] = True,  # 각도(yaw) 예측할지 여부
        obs_encoder: Optional[str] = "efficientnet-b0",  # 관찰 이미지 인코더 종류
        obs_encoding_size: Optional[int] = 512,  # 관찰 이미지 인코딩 크기
        late_fusion: Optional[bool] = False,  # 관찰과 목표 이미지를 나중에 합칠지 여부
        mha_num_attention_heads: Optional[int] = 2,  # Transformer Multi-head Attention 헤드 수 (각각 다른 패턴을 찾음)
        mha_num_attention_layers: Optional[int] = 2,  # Transformer 레이어 수
        mha_ff_dim_factor: Optional[int] = 4,  # Transformer 피드포워드 레이어 확장 비율 (확장을 통해 더 다양한 패턴 학습)
    ) -> None:
        # BaseModel 초기화 (기본 설정)
        super(ViNT, self).__init__(context_size, len_traj_pred, learn_angle)

        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size  # 목표 이미지 인코딩 크기도 동일하게 설정

        self.late_fusion = late_fusion  # late fusion 여부 저장 (과거 이미지와 목표 이미지를 따로 분석 후 마지막에 한꺼번에 합쳐서 미래 경로를 예측하는 방식)

        # EfficientNet 설정 (관찰 이미지 인코더)
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)  # RGB 3채널 입력
            self.num_obs_features = self.obs_encoder._fc.in_features  # EfficientNet 최종 출력 차원

            if self.late_fusion:
                # 목표 이미지도 별도로 EfficientNet 처리
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=3)
            else:
                # 관찰+목표 이미지를 합쳐서 처리하는 경우 (6채널 입력)
                self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)    # early fusion (합쳐서 처리하는 방식)

            self.num_goal_features = self.goal_encoder._fc.in_features  # 목표 이미지 인코딩 크기
        else:
            raise NotImplementedError  # EfficientNet 외 다른 모델은 아직 구현 안 됨

        # EfficientNet 출력 크기와 원하는 크기 맞추기 (안 맞으면 Linear로 변환)
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)    ## nnLinear(입력 크기, 출력 크기)
        else:
            self.compress_obs_enc = nn.Identity()

        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        # Transformer 디코더 구성 (과거+목표 특징을 순서대로 처리하는 역할)
        self.decoder = MultiLayerDecoder(
            embed_dim=self.obs_encoding_size,
            seq_len=self.context_size+2,  # 과거 이미지 수 + 현재 + 목표 이미지
            output_layers=[256, 128, 64, 32],  # 출력 크기 점점 줄이기
            nhead=mha_num_attention_heads,
            num_layers=mha_num_attention_layers,
            ff_dim_factor=mha_ff_dim_factor,
        )

        # 최종 예측 레이어 (거리 예측, 경로 예측)
        self.dist_predictor = nn.Sequential(nn.Linear(32, 1))  # 거리 예측 (scalar 값 1개 출력)
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),  # 각 waypoint 위치+각도 예측
        )

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        실제 이미지와 목표를 입력받아 미래 경로와 거리 예측하는 함수
        """

        # 목표 이미지 인코딩 (late_fusion일 때 별도 처리)
        if self.late_fusion:
            goal_encoding = self.goal_encoder.extract_features(goal_img)
        else:
            # 현재 이미지와 목표 이미지 합쳐서 6채널로 만든 후 인코딩
            obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
            goal_encoding = self.goal_encoder.extract_features(obsgoal_img)

        # EfficientNet 최종 특징 처리 (평균 풀링 → Flatten → 드롭아웃)
        goal_encoding = self.goal_encoder._avg_pooling(goal_encoding)
        if self.goal_encoder._global_params.include_top:
            goal_encoding = goal_encoding.flatten(start_dim=1)
            goal_encoding = self.goal_encoder._dropout(goal_encoding)

        # 목표 이미지 인코딩 크기 조절
        goal_encoding = self.compress_goal_enc(goal_encoding)

        # 목표 인코딩 차원 맞추기 (batch_size, 1, encoding_size)
        if len(goal_encoding.shape) == 2:
            goal_encoding = goal_encoding.unsqueeze(1)

        # 관찰 이미지를 context_size만큼 분할 (과거 프레임 나누기)
        obs_img = torch.split(obs_img, 3, dim=1)

        # (batch_size, 3, H, W) 형태로 펼치기
        obs_img = torch.concat(obs_img, dim=0)

        # 관찰 이미지 인코딩 (EfficientNet 특징 추출)
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)

        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)

        # 관찰 이미지 인코딩 크기 조절
        obs_encoding = self.compress_obs_enc(obs_encoding)

        # (context_size+1, batch, encoding_size)로 변환
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)

        # 관찰 인코딩 + 목표 인코딩 합치기 (시퀀스 형태로 연결)
        tokens = torch.cat((obs_encoding, goal_encoding), dim=1)

        # Transformer 디코더로 최종 특징 생성
        final_repr = self.decoder(tokens)

        # 거리 예측 (scalar 값 1개)
        dist_pred = self.dist_predictor(final_repr)

        # 경로 예측 (waypoint들의 x, y, 각도 변화량 예측)
        action_pred = self.action_predictor(final_repr)

        # 경로 예측 결과 (batch_size, len_traj_pred, num_action_params) 형태로 변환
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )

        # x, y 변화량을 누적해서 실제 좌표로 변환 (cumsum 사용)
        action_pred[:, :, :2] = torch.cumsum(action_pred[:, :, :2], dim=1)

        # 각도(yaw)도 필요하면 정규화 (길이가 1인 단위 벡터로 만들기)
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )

        return dist_pred, action_pred  # 거리 예측, 경로 예측 반환
