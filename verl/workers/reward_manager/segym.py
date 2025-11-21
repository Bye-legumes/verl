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

import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, DefaultDict

import torch

from verl import DataProto
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

logger = logging.getLogger(__name__)


def _maybe_add_repo_to_path(repo_root: str | None) -> None:
    """Add the SEGym repo to sys.path if it is provided."""
    if not repo_root:
        return
    abs_root = os.path.abspath(repo_root)
    if abs_root not in sys.path:
        sys.path.append(abs_root)


def _resolve_numpy_object(value: Any) -> Any:
    """RL datasets return numpy.object_ entries. Convert them back to native Python types."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive
            return value
    return value


@dataclass
class _SegymDatasetInfo:
    dataset: str
    index: int
    language: str
    timeout: int | None


@register("segym")
class SEGymRewardManager(AbstractRewardManager):
    """Reward manager that queries SEGym for pass/fail signals on generated programs."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score: Callable[..., Any] | None = None,  # Unused but kept for API compatibility
        reward_fn_key: str = "data_source",
        segym_repo_root: str | None = None,
        bootstrap_servers: str | None = None,
        service: str | list[str] = "rllm_sandbox",
        post_topic: str = "evaluation-post",
        wait_timeout_s: float = 300.0,
        getmany_timeout_ms: float = 100.0,
        dataset_metadata_path: str | None = None,
        metadata_lookup_keys: list[str] | tuple[str, ...] | None = None,
        default_language: str = "python",
        default_timeout_s: int | None = 30,
        client_id: str | None = None,
        verbose_client: bool = False,
        language_field: str = "language",
        timeout_field: str = "timeout",
    ) -> None:
        if not bootstrap_servers:
            raise ValueError("SEGymRewardManager requires `bootstrap_servers` to be set.")

        _maybe_add_repo_to_path(segym_repo_root)
        try:
            from segym.client import SEGymClient  # type: ignore
        except ImportError as exc:  # pragma: no cover - runtime import
            raise ImportError(
                "Failed to import `segym.client.SEGymClient`. Please install pgcodellm-rl-segym "
                "or specify `segym_repo_root` that points to the repo."
            ) from exc

        try:
            from segym.util.sandbox_utils import extract_code_from_model  # type: ignore
        except ImportError:  # pragma: no cover - optional helper
            extract_code_from_model = None

        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key

        self._segym_client_cls = SEGymClient
        self._extract_code_fn = extract_code_from_model

        if isinstance(service, str):
            services = [service]
        else:
            services = list(service)

        if not services:
            raise ValueError("At least one SEGym service must be provided.")

        self._services = services
        self._primary_service = services[0]
        self._bootstrap_servers = bootstrap_servers
        self._post_topic = post_topic
        self._wait_timeout_s = wait_timeout_s
        self._getmany_timeout_ms = getmany_timeout_ms
        self._default_language = default_language
        self._default_timeout_s = default_timeout_s
        self._client_id = client_id
        self._verbose_client = verbose_client
        self._language_field = language_field
        self._timeout_field = timeout_field

        self._metadata_lookup_keys = tuple(metadata_lookup_keys or ("prompt_md5hash", "dataset_problem_md5hash"))
        self._metadata_by_prompt: dict[str, dict[str, Any]] = {}
        self._metadata_by_problem: dict[str, dict[str, Any]] = {}
        if dataset_metadata_path:
            self._load_dataset_metadata(dataset_metadata_path)

    def _load_dataset_metadata(self, metadata_path: str) -> None:
        """Load the original dataset for mapping prompt hashes back to dataset/index."""
        abs_path = os.path.abspath(metadata_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"SEGymRewardManager cannot find dataset metadata: {metadata_path}")

        logger.info("Loading SEGym dataset metadata from %s", abs_path)
        with open(abs_path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:  # pragma: no cover - depends on dataset
                    raise ValueError(f"Malformed JSON line at {line_no} in {metadata_path}: {exc}") from exc

                prompt_hash = record.get("prompt_md5hash")
                if prompt_hash:
                    self._metadata_by_prompt[str(prompt_hash)] = record

                prob_hash = record.get("dataset_problem_md5hash")
                if prob_hash:
                    self._metadata_by_problem[str(prob_hash)] = record

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self._compute_rewards_async(data, return_dict))
        finally:
            loop.close()

    async def _compute_rewards_async(self, data: DataProto, return_dict: bool) -> torch.Tensor | dict[str, Any]:
        # Support RM score shortcut
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        prompts = data.batch["prompts"]
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        device = responses.device
        prompt_len = prompts.shape[-1]

        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        sample_rewards = torch.zeros(len(data), dtype=torch.float32, device=device)
        reward_extra_info = defaultdict(list)

        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        extra_infos = data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))])

        request_payloads: list[dict[str, Any]] = []
        request_context: list[dict[str, Any]] = []

        for i in range(len(data)):
            extra_info = _resolve_numpy_object(extra_infos[i])
            if extra_info is None:
                extra_info = {}

            response_len = int(valid_response_lengths[i].item())
            if response_len <= 0:
                reward_extra_info["segym_detail"].append("empty_response")
                continue

            response_token_ids = responses[i][:response_len]
            prompt_token_ids = prompts[i]

            prompt_str = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)

            dataset_info = self._resolve_dataset_info(extra_info, data_sources[i])
            if dataset_info is None:
                logger.warning("Missing SEGym dataset/index metadata for sample %s", i)
                reward_extra_info["segym_detail"].append("missing_dataset_info")
                continue

            code_payload = self._extract_code(response_str)
            if not code_payload.strip():
                code_payload = response_str

            payload: dict[str, Any] = {
                "code": code_payload,
                "dataset": dataset_info.dataset,
                "index": dataset_info.index,
            }
            if dataset_info.language:
                payload["language"] = dataset_info.language
            if dataset_info.timeout is not None:
                payload["timeout"] = dataset_info.timeout

            request_payloads.append(payload)
            request_context.append(
                {
                    "sample_idx": i,
                    "response_length": response_len,
                    "prompt": prompt_str,
                    "response": response_str,
                    "dataset": dataset_info.dataset,
                    "dataset_index": dataset_info.index,
                }
            )

        if not request_payloads:
            logger.warning("No valid SEGym requests were constructed for this batch.")
            data.batch["acc"] = sample_rewards
            if return_dict:
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            return reward_tensor

        client = self._segym_client_cls(
            self._bootstrap_servers,
            self._services,
            client_id=self._client_id,
            post_topic=self._post_topic,
            verbose=self._verbose_client,
        )
        await client.init()
        try:
            segym_results = await client.send_and_wait_for_replies(
                msgs=request_payloads,
                wait_timeout=self._wait_timeout_s,
                getmany_timeout=self._getmany_timeout_ms,
            )
        finally:
            await client.close()

        service_payloads = segym_results.get(self._primary_service, [])
        if len(service_payloads) != len(request_context):
            raise RuntimeError(
                f"SEGym returned {len(service_payloads)} replies for {len(request_context)} requests "
                f"on service {self._primary_service}"
            )

        already_printed: DefaultDict[str, int] = defaultdict(int)

        for ctx, reply in zip(request_context, service_payloads, strict=False):
            payload = reply.get("payload", {})
            reward_value = float(payload.get("reward", 0.0))
            detail = payload.get("detail", "")
            elapsed = payload.get("time", 0.0)

            idx = ctx["sample_idx"]
            resp_len = ctx["response_length"]
            reward_tensor[idx, resp_len - 1] = reward_value
            sample_rewards[idx] = reward_value

            reward_extra_info["segym_detail"].append(detail)
            reward_extra_info["segym_time"].append(elapsed)
            reward_extra_info["segym_dataset"].append(ctx["dataset"])
            reward_extra_info["segym_index"].append(ctx["dataset_index"])

            dataset_key = ctx["dataset"]
            if already_printed[dataset_key] < self.num_examine:
                already_printed[dataset_key] += 1
                print("[SEGym prompt]", ctx["prompt"])
                print("[SEGym response]", ctx["response"])
                print("[SEGym dataset]", dataset_key, "index:", ctx["dataset_index"])
                print("[SEGym detail]", detail)
                print("[SEGym reward]", reward_value)

        data.batch["acc"] = sample_rewards
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor

    def _extract_code(self, response: str) -> str:
        if self._extract_code_fn is None:
            return response
        try:
            return self._extract_code_fn(response).strip()
        except Exception:  # pragma: no cover - depends on sandbox parsing
            logger.exception("Failed to extract code from model response; falling back to raw text.")
            return response

    def _resolve_dataset_info(self, extra_info: dict[str, Any], fallback_dataset: str) -> _SegymDatasetInfo | None:
        extra_info = extra_info or {}

        dataset = extra_info.get("dataset") or extra_info.get("segym_dataset") or fallback_dataset
        dataset_index = extra_info.get("dataset_index") or extra_info.get("index")
        language = extra_info.get(self._language_field) or extra_info.get("code_language") or self._default_language
        timeout = extra_info.get(self._timeout_field) or extra_info.get("segym_timeout") or self._default_timeout_s

        metadata = None
        if (dataset is None or dataset_index is None) and (self._metadata_by_prompt or self._metadata_by_problem):
            metadata = self._lookup_metadata(extra_info)
            if metadata:
                dataset = dataset or metadata.get("dataset")
                dataset_index = dataset_index or metadata.get("dataset_index")
                language = language or metadata.get("language")

        if dataset_index is None or dataset is None:
            return None

        try:
            dataset_idx_int = int(dataset_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid dataset index value: {dataset_index}") from exc

        timeout_int = None
        if timeout is not None:
            try:
                timeout_int = int(timeout)
            except (TypeError, ValueError):
                logger.warning("Invalid timeout value %s, falling back to default.", timeout)
                timeout_int = self._default_timeout_s

        return _SegymDatasetInfo(
            dataset=str(dataset),
            index=dataset_idx_int,
            language=str(language) if language else self._default_language,
            timeout=timeout_int,
        )

    def _lookup_metadata(self, extra_info: dict[str, Any]) -> dict[str, Any] | None:
        for key in self._metadata_lookup_keys:
            value = extra_info.get(key)
            if not value:
                continue
            record = self._metadata_by_prompt.get(str(value))
            if record:
                return record
            record = self._metadata_by_problem.get(str(value))
            if record:
                return record
        return None
