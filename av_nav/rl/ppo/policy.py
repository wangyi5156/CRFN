#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary  # 模型结构可视化（原文件已有）

from av_nav.common.utils import CategoricalNet
from av_nav.rl.models.rnn_state_encoder import RNNStateEncoder
from av_nav.rl.models.visual_cnn import VisualCNN
from av_nav.rl.models.audio_cnn import AudioCNN
from av_nav.rl.models.fusion_dmrm import TwoBranchResidualFusion  # ✅ 新增：双向残差融合（最小改动版）

logger = logging.getLogger(__name__)

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(self.net.output_size, self.dim_actions)
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action = distribution.mode() if deterministic else distribution.sample()
        action_log_probs = distribution.log_probs(action)
        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(observations, rnn_hidden_states, prev_actions, masks)
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()
        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class PointNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size=512,
        extra_rgb=False,
    ):
        super().__init__(
            PointNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                extra_rgb=extra_rgb,
            ),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class PointNavBaselineNet(Net):
    r"""视觉/音频经各自编码器得到嵌入；（本修改）使用 DMRN 在 GRU 之前做可学习融合；
    GRU 输出再接 Actor/Critic。
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, extra_rgb=False):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size

        # 标记任务类型：AudioGoal / PointGoal（可能为双传感器）
        self._audiogoal = False
        self._pointgoal = False
        self._n_pointgoal = 0

        if DUAL_GOAL_DELIMITER in self.goal_sensor_uuid:
            goal1_uuid, goal2_uuid = self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)
            self._audiogoal = True
            self._pointgoal = True
            self._n_pointgoal = observation_space.spaces[goal1_uuid].shape[0]
        else:
            if self.goal_sensor_uuid == "pointgoal_with_gps_compass":
                self._pointgoal = True
                self._n_pointgoal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
            else:
                self._audiogoal = True

        # 编码器
        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb)

        audiogoal_sensor = None
        if self._audiogoal:
            if "audiogoal" in self.goal_sensor_uuid:
                audiogoal_sensor = "audiogoal"
            elif "spectrogram" in self.goal_sensor_uuid:
                audiogoal_sensor = "spectrogram"
            self.audio_encoder = AudioCNN(observation_space, hidden_size, audiogoal_sensor)

        # 原始的（concat）GRU 输入维
        rnn_input_size = (
            (0 if self.is_blind else self._hidden_size)
            + (self._n_pointgoal if self._pointgoal else 0)
            + (self._hidden_size if self._audiogoal else 0)
        )

        # ✅ DMRN 启用条件：仅 AudioGoal 且具备视觉（不含 PointGoal）
        self.use_dmrm = self._audiogoal and (not self._pointgoal) and (not self.is_blind)
        if self.use_dmrm:
            self.fusion = TwoBranchResidualFusion(H=self._hidden_size, nb_blocks=1)
            prev_in = rnn_input_size
            rnn_input_size = self._hidden_size  # 由 1024(A+V) → 512(fusion)
            logger.info(
                "[DMRN] enabled: H=%d, nb_blocks=%d, rnn_input_size %d -> %d",
                self._hidden_size, 1, prev_in, rnn_input_size
            )
            # 低开销 DEBUG 计数器（每 _dbg_every 次 forward 打一条）
            self._dbg_counter = 0
            self._dbg_every = 100

        # 策略 RNN
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        # （保留原 summary 行为，便于启动时快速 sanity check）
        try:
            if "rgb" in observation_space.spaces and not extra_rgb:
                rgb_shape = observation_space.spaces["rgb"].shape
                summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device="cpu")
            if "depth" in observation_space.spaces:
                depth_shape = observation_space.spaces["depth"].shape
                summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device="cpu")
            if self._audiogoal and audiogoal_sensor is not None:
                audio_shape = observation_space.spaces[audiogoal_sensor].shape
                summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device="cpu")
        except Exception as e:
            logger.debug("summary skipped: %s", str(e))

        # 供 TensorBoard 使用的 DMRN 统计缓存（policy 不直接写 TB）
        self._last_dmrm_stats = {}

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        """
        observations: dict-like
        rnn_hidden_states: (N, hidden)
        prev_actions, masks: 按现有 PPO 接口传递
        """
        x_cat_list = []
        a_emb = None
        v_emb = None

        # PointGoal（如有）
        if self._pointgoal:
            x_cat_list.append(observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]])

        # AudioGoal（如有）
        if self._audiogoal:
            a_emb = self.audio_encoder(observations)   # (N, 512)
            x_cat_list.append(a_emb)

        # 视觉（如有）
        if not self.is_blind:
            v_emb = self.visual_encoder(observations)  # (N, 512)
            x_cat_list.append(v_emb)

        # ✅ 使用 DMRN 融合（仅 AG + 有视觉），否则回退到 concat
        if self.use_dmrm and (a_emb is not None) and (v_emb is not None):
            x1 = self.fusion(a_emb, v_emb)            # (N, 512)

            # -------- DEBUG（低频） ----------
            if logger.isEnabledFor(logging.DEBUG) and (self._dbg_counter % self._dbg_every == 0):
                with torch.no_grad():
                    def _m_s(t):  # 批内均值/方差
                        return t.mean().item(), t.std().item()
                    ma, sa = _m_s(a_emb)
                    mv, sv = _m_s(v_emb)
                    mf, sf = _m_s(x1)
                    ba = float(self.fusion.beta_a.detach().cpu())
                    bv = float(self.fusion.beta_v.detach().cpu())
                    logger.debug(
                        "[DMRN] used=1 "
                        "a_emb[mean=%.4f std=%.4f] v_emb[mean=%.4f std=%.4f] "
                        "fusion[mean=%.4f std=%.4f] beta_a=%.3f beta_v=%.3f",
                        ma, sa, mv, sv, mf, sf, ba, bv
                    )
            self._dbg_counter += 1

            # -------- 统计缓存（供 ppo_trainer 写入 TensorBoard） ----------
            with torch.no_grad():
                def _mean_l2(t):  # 每样本 L2，再做 batch 平均
                    return t.norm(dim=1).mean().item()
                stats = {
                    "beta_a": float(self.fusion.beta_a.detach().cpu()),
                    "beta_v": float(self.fusion.beta_v.detach().cpu()),
                    "a_norm": _mean_l2(a_emb),
                    "v_norm": _mean_l2(v_emb),
                    "f_norm": _mean_l2(x1),
                    "cos_a_v": F.cosine_similarity(a_emb, v_emb, dim=1).mean().item(),
                    "cos_a_f": F.cosine_similarity(a_emb, x1, dim=1).mean().item(),
                    "cos_v_f": F.cosine_similarity(v_emb, x1, dim=1).mean().item(),
                }
                stats["gru_in_norm"] = x1.norm(dim=1).mean().item()
                self._last_dmrm_stats = stats
        else:
            # 回退：仍用 concat（兼容 PointGoal、视觉缺失等情形）
            x1 = torch.cat(x_cat_list, dim=1)

        # 进入策略 RNN
        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        # 数值健壮性检查
        assert not torch.isnan(x2).any().item(), "NaN detected in GRU output"

        return x2, rnn_hidden_states1
