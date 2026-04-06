# av_nav/rl/models/fusion_dmrm.py
# 双向残差融合（DMRN-min，适配 AVN/AudioGoal）
# 变更：
#   1) beta_a/beta_v 初始值从 0.5 -> 0.2（弱耦合起步，便于学习）
#   2) 在融合输出端增加 LayerNorm（stabilize GRU input）
# 其余接口与行为保持一致，可直接替换原文件。

import torch
import torch.nn as nn

class TwoBranchResidualFusion(nn.Module):
    """
    两路对称残差融合：
      a,v: (B,H) 或 (B,T,H)  ->  fusion: 同形
    结构：
      U_a/U_v: LN -> Linear -> Tanh -> Linear  （保形 H->H）
      残差更新：x <- act( LN(x) + beta_x * merged )
      融合输出：fusion = 0.5 * (a + v)  ->  (可选) LayerNorm
    参数：
      H            : 通道维
      nb_blocks    : 堆叠的 DMRN 层数（默认 1）
      beta_init    : 残差缩放初值（建议 0.2）
      use_out_ln   : 是否在 fusion 输出端做 LayerNorm（建议 True）
      act_name     : 'tanh' 或 'relu'（默认 'tanh'）
    """
    def __init__(
        self,
        H: int,
        nb_blocks: int = 1,
        beta_init: float = 1,
        use_out_ln: bool = True,
        act_name: str = "tanh",
    ):
        super().__init__()
        self.nb_blocks = nb_blocks
        self.use_out_ln = use_out_ln

        # 子网 U_a / U_v：对称、保形
        self.U_a = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.Tanh(),
            nn.Linear(H, H),
        )
        self.U_v = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H),
            nn.Tanh(),
            nn.Linear(H, H),
        )

        # 残差路径上的归一化
        self.ln_a = nn.LayerNorm(H)
        self.ln_v = nn.LayerNorm(H)

        # 可学习残差缩放（弱耦合起步，更易学到合适强度）
        self.beta_a = nn.Parameter(torch.tensor(float(beta_init)))
        self.beta_v = nn.Parameter(torch.tensor(float(beta_init)))

        # 激活函数（默认 tanh，更稳；如需对照可设 act_name='relu'）
        if act_name.lower() == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Tanh()

        # 融合输出端的 LayerNorm：稳定送入 GRU 的尺度
        self.ln_out = nn.LayerNorm(H) if use_out_ln else nn.Identity()

    def forward(self, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        a, v: (B,H) 或 (B,T,H)
        return: fusion with same shape
        """
        added_T = False
        if a.dim() == 2:
            a = a.unsqueeze(1)  # (B,1,H)
            v = v.unsqueeze(1)
            added_T = True

        for _ in range(self.nb_blocks):
            ua = self.U_a(a)               # (B,T,H)
            uv = self.U_v(v)               # (B,T,H)
            merged = 0.5 * (ua + uv)       # (B,T,H)

            # 交叉残差更新（对称）
            v = self.act(self.ln_v(v) + self.beta_v * merged)
            a = self.act(self.ln_a(a) + self.beta_a * merged)

        fusion = 0.5 * (a + v)             # (B,T,H)
        fusion = self.ln_out(fusion)       # 输出端 LN，稳住尺度

        return fusion.squeeze(1) if added_T else fusion
