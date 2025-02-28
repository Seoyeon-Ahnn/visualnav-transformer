import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 위치 인코딩 추가
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # max_seq_len 길이의 위치 인코딩 행렬 초기화 (각 위치마다 d_model 차원의 벡터)
        pos_enc = torch.zeros(max_seq_len, d_model)

        # 각 시점의 위치 인덱스 (0, 1, 2, ..., max_seq_len-1), shape: [max_seq_len, 1]
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # 주파수 조절 파트 (위치별로 주파수를 다르게 구성해, 위치차이를 부드럽게 표현)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 짝수 차원은 sin 값으로 위치 표현
        pos_enc[:, 0::2] = torch.sin(pos * div_term)

        # 홀수 차원은 cos 값으로 위치 표현
        pos_enc[:, 1::2] = torch.cos(pos * div_term)

        # (1, max_seq_len, d_model)로 만들어서 배치처리용으로 만듦
        pos_enc = pos_enc.unsqueeze(0)

        # 학습 파라미터가 아니라, 고정된 값이므로 버퍼로 등록 (저장 시 제외됨)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # 입력에 위치 인코딩 더하기 (각 시점별 순서 정보 부여)
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

# Self Attention 기반 Transformer 디코더
# - ViNT에서 과거+현재+목표 특징 시퀀스를 입력받아 Self Attention으로 각 시점 특징 간 관계를 학습하고, 최종 예측을 위한 하나의 벡터로 변환하는 역할
class MultiLayerDecoder(nn.Module):
    def __init__(self, 
                 embed_dim=512,  # 각 시점 특징 벡터 크기
                 seq_len=6,      # 시퀀스 길이 (과거 5장 + 현재 1장)
                 output_layers=[256, 128, 64],  # 최종 출력 단계별 크기
                 nhead=8,        # Self Attention 헤드 수
                 num_layers=8,   # Transformer 레이어 수
                 ff_dim_factor=4):  # 피드포워드 레이어 확장 비율
        super(MultiLayerDecoder, self).__init__()

        # 위치 인코딩 모듈 (시퀀스 순서 정보 부여용)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)

        # Self Attention 기반 Transformer 레이어 정의
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,               # 입력 특징 벡터 크기
            nhead=nhead,                     # 각 레이어마다 몇 개의 시선으로 Self Attention 수행할지
            dim_feedforward=ff_dim_factor * embed_dim,  # 피드포워드 네트워크 크기
            activation="gelu",                # 비선형 활성화 함수
            batch_first=True,                  # 입력 형태: [batch_size, seq_len, embed_dim]
            norm_first=True                    # LayerNorm을 먼저 수행
        )

        # Self Attention 레이어 여러 개 쌓아서 Transformer 구성
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)

        # 최종 출력 레이어들 정의: 시퀀스 전체를 하나로 합쳐서 (flatten) 예측 벡터로 변환
        self.output_layers = nn.ModuleList([nn.Linear(seq_len * embed_dim, embed_dim)])

        # 점점 크기 줄이면서 최종 벡터로 압축하는 과정
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers)-1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i+1]))

    def forward(self, x):
        ## 1. 입력받은 시퀀스 특징에 위치 인코딩 추가: Transformer는 시퀀스 순서를 모르기 때문에 각 시점의 순서를 인코딩
        if self.positional_encoding:
            x = self.positional_encoding(x)

        ## 2. Multi-head Self Attention으로 시점 간 관계 학습
        # Self Attention 기반 Transformer 디코더 통과 (각 시점 간 관계 학습)
        x = self.sa_decoder(x)

        ## 3. 최종 예측을 위해 시퀀스(모든 시점의 특징)를 하나로 합쳐 Flatten
        x = x.reshape(x.shape[0], -1)

        ## 4. Fully Connected Layers로 최종 특징 압축
        # 여러 단계의 Linear 레이어 + ReLU 활성화 함수 거치며 최종 벡터 생성
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)

        ## 5. 최종 예측에 필요한 특징 벡터 반환
        return x
