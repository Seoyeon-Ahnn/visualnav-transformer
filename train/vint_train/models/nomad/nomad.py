import os
import argparse
import time
import pdb

import torch
import torch.nn as nn

# NoMaD 전체 모델 클래스 정의
class NoMaD(nn.Module):

    def __init__(self, vision_encoder, 
                       noise_pred_net,
                       dist_pred_net):
        super(NoMaD, self).__init__()

        # 시각 인코더(vision_encoder), 노이즈 예측 네트워크(noise_pred_net), 거리 예측 네트워크(dist_pred_net) 설정
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net    # diffusion 쓰임
        self.dist_pred_net = dist_pred_net
    
    def forward(self, func_name, **kwargs):
        # 입력으로 받은 func_name에 따라 서로 다른 네트워크 실행
        if func_name == "vision_encoder" :
            # 시각 인코더 실행 (obs_img, goal_img, input_goal_mask 필요)
            output = self.vision_encoder(kwargs["obs_img"], kwargs["goal_img"], input_goal_mask=kwargs["input_goal_mask"])
        elif func_name == "noise_pred_net":
            # 노이즈 예측 네트워크 실행 (sample, timestep, global_cond 필요)
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            # 거리 예측 네트워크 실행 (obsgoal_cond 필요)
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            # 정의되지 않은 func_name일 경우 에러 발생
            raise NotImplementedError
        return output  # 선택한 네트워크의 결과 반환


# 거리 예측 네트워크 (Dense Network) 클래스 정의
class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim):
        super(DenseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim  # 입력 차원 저장
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),  # 1/4로 차원 축소
            nn.ReLU(),  # 활성화 함수
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),  # 다시 1/16로 축소
            nn.ReLU(),  # 활성화 함수
            nn.Linear(self.embedding_dim//16, 1)  # 최종 출력 차원: 1 (거리 예측 값)
        )
    
    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))  # 입력을 (batch_size, embedding_dim) 형태로 변환
        output = self.network(x)  # 네트워크 통과
        return output  # 최종 거리 값 반환
