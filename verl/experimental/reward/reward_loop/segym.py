# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from verl import DataProto
from verl.experimental.reward.reward_loop import register as register_loop
from verl.experimental.reward.reward_loop.base import RewardLoopManagerBase
from verl.workers.reward_manager.segym import SEGymRewardManager


def _to_plain_dict(cfg: Any) -> dict[str, Any]:
    """Convert DictConfig/listconfig to a vanilla dict recursively."""
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if cfg is None:
        return {}
    return dict(cfg)


@register_loop("segym")
class SEGymRewardLoopManager(RewardLoopManagerBase):
    """Adapter that exposes the SEGym reward manager to the experimental agent loop."""

    def __init__(
        self,
        config,
        tokenizer,
        compute_score=None,
        reward_router_address=None,
        reward_model_tokenizer=None,
    ) -> None:
        super().__init__(config, tokenizer)
        reward_kwargs_cfg = config.reward_model.get("reward_kwargs") or {}
        reward_kwargs = _to_plain_dict(reward_kwargs_cfg)

        # The legacy reward managers expose a `num_examine` knob to print samples.
        num_examine = int(reward_kwargs.pop("num_examine", 0) or 0)

        self._segym_manager = SEGymRewardManager(
            tokenizer=tokenizer,
            num_examine=num_examine,
            compute_score=compute_score,
            reward_fn_key=config.data.reward_fn_key,
            **reward_kwargs,
        )

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        assert len(data) == 1, "SEGym reward loop only supports a single sample per call."

        reward_result = await self.loop.run_in_executor(
            None,
            lambda: self._segym_manager(data, return_dict=True),
        )

        reward_tensor: torch.Tensor = reward_result["reward_tensor"]
        reward_extra_info = dict(reward_result.get("reward_extra_info", {}))
        reward_score = self._extract_reward_score(data, reward_tensor)
        return {"reward_score": reward_score, "reward_extra_info": reward_extra_info}

    def _extract_reward_score(self, data: DataProto, reward_tensor: torch.Tensor) -> float:
        """SEGym stores the scalar reward at the last valid token of each response."""
        prompts = data.batch["prompts"]
        attention_mask = data.batch["attention_mask"]

        prompt_len = prompts.shape[-1]
        response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        if response_lengths.numel() == 0:
            return 0.0

        resp_len = int(response_lengths[0].item())
        if resp_len <= 0:
            return 0.0

        resp_len = min(resp_len, reward_tensor.size(1))
        return float(reward_tensor[0, resp_len - 1].item())
