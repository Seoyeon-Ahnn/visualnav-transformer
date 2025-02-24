"""
여러 장의 이미지를 입력으로 받아 이미지의 특징을 추출하는 CNN 기반 인코더
"""

## 1. 필요한 라이브러리 및 모듈 가져오기
# PyTorch torchvision 라이브러리를 수정하여 사용
from typing import Callable, Any, Optional, List  # 타입 힌트를 위한 모듈 import

import torch
from torch import Tensor
from torch import nn

# ConvNormActivation: 컨볼루션, 정규화, 활성화 함수 적용을 한 번에 수행하는 모듈
from torchvision.ops.misc import ConvNormActivation
# 채널 수를 지정된 배수(round_nearest)의 배수로 맞추는 함수
from torchvision.models._utils import _make_divisible
# MobileNetV2에서 사용되는 InvertedResidual 블록 (모바일 네트워크의 핵심 구성 요소, 채널 수를 확장한 뒤 다시 축소)
from torchvision.models.mobilenetv2 import InvertedResidual

## 2. MobileNetEncoder 클래스 정의 및 초기화 (nn.Module 상속)
class MobileNetEncoder(nn.Module):
    def __init__(
        self,
        num_images: int = 1,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        """
        MobileNet V2 주요 클래스
        Args:
            num_images (int): 입력 텐서에 스택된 이미지의 수 (예: RGB 이미지가 3채널이므로 num_images*3 채널이 됨)
            num_classes (int): 분류할 클래스의 수
            width_mult (float): 각 레이어의 채널 수를 조정하는 배수
            inverted_residual_setting: inverted residual 블록 구성 설정 리스트, 네트워크 구조 설정, 각 요소는 [t, c, n, s] (확장비율, 출력채널, 반복횟수, 다운샘플링을 위한 stride)
            round_nearest (int): 채널 수를 이 값의 배수로 반올림 (1이면 반올림 없이 그대로 사용, 모델 최적화를 위해)
            block: MobileNet의 inverted residual 블록을 정의하는 모듈 (기본값 InvertedResidual)
            norm_layer: 사용할 정규화 층 (기본값 BatchNorm2d)
            dropout (float): classifier 부분에서 사용할 dropout 확률
        """
        super().__init__()  # nn.Module의 생성자 호출

        # block이 지정되지 않으면 기본적으로 InvertedResidual 사용
        if block is None:
            block = InvertedResidual

        # norm_layer가 지정되지 않으면 기본적으로 BatchNorm2d 사용
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # 초기 입력 채널 수와 마지막 레이어 채널 수 설정
        input_channel = 32
        last_channel = 1280

        # inverted residual 설정이 없으면 기본 설정 사용
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # 각 리스트 항목: [확장비율(t), 출력채널 수(c), 반복 횟수(n), stride(s)]
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # inverted_residual_setting의 첫 번째 요소가 4개의 값을 가지는지 확인
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # 첫 번째 컨볼루션 레이어를 위한 채널 수 조정
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        # 마지막 레이어의 채널 수 조정 (width_mult이 1.0 미만인 경우에도 최소값 보장)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )
        # 첫 번째 레이어: ConvNormActivation 모듈을 사용해 이미지의 기본 특징 추출
        features: List[nn.Module] = [
            ConvNormActivation(
                num_images * 3,  # 입력 채널 수: num_images * 3 (각 이미지가 RGB라 3채널)
                input_channel,   # 출력 채널 수: 조정된 input_channel
                stride=2,        # 다운샘플링을 위해 stride=2 사용
                norm_layer=norm_layer,  # 지정된 정규화 층 적용
                activation_layer=nn.ReLU6,  # ReLU6 활성화 함수 사용
            )
        ]
        # inverted residual 블록들을 구성하여 features 리스트에 추가
        for t, c, n, s in inverted_residual_setting:
            # 출력 채널 수를 width_mult와 round_nearest에 맞춰 조정
            output_channel = _make_divisible(c * width_mult, round_nearest)
            # n번 반복하여 동일한 블록을 쌓음
            for i in range(n):
                # 첫 번째 블록은 지정된 stride, 이후 블록은 stride=1
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,    # 이전 레이어의 출력 채널 수
                        output_channel,   # 현재 블록의 출력 채널 수
                        stride,           # stride 값
                        expand_ratio=t,   # 확장 비율 (t 값)
                        norm_layer=norm_layer,  # 정규화 층
                    )
                )
                # 다음 블록의 입력 채널은 현재 블록의 출력 채널로 업데이트
                input_channel = output_channel
        # 마지막 특징 추출 레이어 추가: 1x1 컨볼루션으로 채널 수를 last_channel로 변경
        features.append(
            ConvNormActivation(
                input_channel,
                self.last_channel,
                kernel_size=1,   # 1x1 컨볼루션 사용
                norm_layer=norm_layer,
                activation_layer=nn.ReLU6,
            )
        )
        # features 리스트에 있는 모든 레이어를 nn.Sequential로 묶어서 self.features에 저장
        self.features = nn.Sequential(*features)

        # classifier 구성: dropout 후 선형 레이어로 분류 수행
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),  # dropout 적용
            nn.Linear(self.last_channel, num_classes),  # 마지막 채널에서 num_classes로 매핑
        )

        # 전체 네트워크의 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Conv2d 레이어는 Kaiming Normal 초기화를 사용 (fan_out 모드)
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 정규화 레이어는 가중치는 1, 편향은 0으로 초기화
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 선형 레이어는 평균 0, 표준편차 0.01의 정규분포로 초기화
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    ## 3. Forward Pass: _forward_impl()와 forward() 함수
    # _forward_impl 함수: 실제 forward 동작을 구현 (TorchScript 호환성을 위해 별도의 이름 사용)
    def _forward_impl(self, x: Tensor) -> Tensor:
        # x를 self.features를 통해 순차적으로 통과시켜 특징 추출 수행
        x = self.features(x)
        # adaptive 평균 풀링을 통해 공간 차원을 (1,1)로 축소 (글로벌 특징 벡터 생성)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # 배치 차원을 제외하고 평탄화 (flatten)하여 1차원 벡터로 변환
        x = torch.flatten(x, 1)
        # classifier를 통해 최종 분류 결과 계산 (경로 계획에 필요한 환경의 시각적 특성과 목표 사이의 관계를 파악하기 위한 특징 추출)
        x = self.classifier(x)
        return x

    # forward 함수: _forward_impl을 호출하여 순전파 실행
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
