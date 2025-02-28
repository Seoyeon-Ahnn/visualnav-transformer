import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from vint_train.models.vint.self_attention import PositionalEncoding  # 포지셔널 인코딩 모듈

class NoMaD_ViNT(nn.Module):
    def __init__(
        self,
        context_size: int = 5,  # 사용할 과거 프레임 수
        obs_encoder: Optional[str] = "efficientnet-b0",  # 관찰 이미지 인코더 모델
        obs_encoding_size: Optional[int] = 512,  # 관찰 이미지 인코딩 벡터 크기
        mha_num_attention_heads: Optional[int] = 2,  # 멀티헤드 어텐션 헤드 수
        mha_num_attention_layers: Optional[int] = 2,  # 트랜스포머 인코더 레이어 수
        mha_ff_dim_factor: Optional[int] = 4,  # FF 레이어 차원 확장 비율
    ) -> None:
        """
        NoMaD ViNT Encoder class
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size
        self.context_size = context_size

        # EfficientNet을 관찰 이미지 인코더로 사용
        if obs_encoder.split("-")[0] == "efficientnet":
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)  # 관찰 이미지는 3채널 (RGB)
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)  # BatchNorm을 GroupNorm으로 변경
            self.num_obs_features = self.obs_encoder._fc.in_features  # EfficientNet 출력 차원 저장
            self.obs_encoder_type = "efficientnet"
        else:
            raise NotImplementedError
        
        # 목표 이미지 인코더 설정 (관찰+목표 이미지 합친 6채널 입력)
        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)
        self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
        self.num_goal_features = self.goal_encoder._fc.in_features

        # 관찰 인코더 출력과 타겟 인코딩 크기가 다를 경우 선형 레이어로 매핑
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        # 포지셔널 인코딩 초기화 (최대 시퀀스 길이 = context_size + 2)
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2)

        # 트랜스포머 인코더 레이어 정의
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size, 
            nhead=mha_num_attention_heads, 
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

        # 목표 이미지 마스크 정의 (goal 위치 마스킹)
        self.goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        self.goal_mask[:, -1] = True  # 마지막 토큰이 목표 이미지
        self.no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)  # 마스크 없음 상태
        
        # 마스크들을 하나로 합침 (mask 없음과 goal mask 포함)
        self.all_masks = torch.cat([self.no_mask, self.goal_mask], dim=0)

        # 평균 풀링 시 마스크 고려 (goal 포함/제외시 가중치 조정용)
        self.avg_pool_mask = torch.cat([1 - self.no_mask.float(), (1 - self.goal_mask.float()) * ((self.context_size + 2)/(self.context_size + 1))], dim=0)


    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        device = obs_img.device

        # goal_encoding 초기화
        goal_encoding = torch.zeros((obs_img.size()[0], 1, self.goal_encoding_size)).to(device)
        
        # 입력 마스크가 있으면 적용
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)

        # 관찰 이미지의 마지막 프레임과 목표 이미지 결합 후 인코딩
        obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
        obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        obsgoal_encoding = self.goal_encoder._avg_pooling(obsgoal_encoding)
        
        if self.goal_encoder._global_params.include_top:
            obsgoal_encoding = obsgoal_encoding.flatten(start_dim=1)
            obsgoal_encoding = self.goal_encoder._dropout(obsgoal_encoding)
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)

        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)
        assert obsgoal_encoding.shape[2] == self.goal_encoding_size
        goal_encoding = obsgoal_encoding
        
        # 과거 프레임 이미지들을 시간 순서대로 분리하고 다시 연결
        obs_img = torch.split(obs_img, 3, dim=1)
        obs_img = torch.concat(obs_img, dim=0)

        # 관찰 이미지 인코딩 수행
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1)
        
        # goal mask 적용 (필요시)
        if goal_mask is not None:
            no_goal_mask = goal_mask.long()
            src_key_padding_mask = torch.index_select(self.all_masks.to(device), 0, no_goal_mask)
        else:
            src_key_padding_mask = None
        
        # 포지셔널 인코딩 적용
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)

        # Transformer 인코더 통과
        obs_encoding_tokens = self.sa_encoder(obs_encoding, src_key_padding_mask=src_key_padding_mask)
        if src_key_padding_mask is not None:
            avg_mask = torch.index_select(self.avg_pool_mask.to(device), 0, no_goal_mask).unsqueeze(-1)
            obs_encoding_tokens = obs_encoding_tokens * avg_mask
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)

        return obs_encoding_tokens



# BatchNorm을 GroupNorm으로 교체하는 함수
def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module


# 재귀적으로 서브모듈 순회하며 BatchNorm을 GroupNorm으로 교체
def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    assert len(bn_list) == 0
    return root_module
