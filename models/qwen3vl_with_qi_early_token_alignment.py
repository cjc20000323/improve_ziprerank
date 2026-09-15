"""Diagnostic Qwen3-VL/QI-Early variant that records text-token argmax indices.

This class is deliberately separate from :mod:`qwen3vl_with_qi_early` so the
normal training and evaluation model is untouched.  It produces identical
pruning scores and selections; the only extra work is retaining the argmax
text-token index when diagnostic collection is enabled.
"""

from __future__ import annotations

from typing import List, Optional, cast

import torch
from torch.nn.functional import normalize

from .qi_early_pruner import QIEarlyInteractionPruner
from .qwen3vl_with_qi_early import Qwen3VLWithQIEarly, TextExtractionMode


class QIEarlyTokenAlignmentPruner(QIEarlyInteractionPruner):
    """QI-Early pruner that retains ``argmax(text_token)`` for each patch."""

    def __init__(self, temperature: float = 0.1, keep_ratio: float = 0.25) -> None:
        super().__init__(temperature=temperature, keep_ratio=keep_ratio)

        # 这三项共同描述本次 forward 真正参与 QI-Early 相似度计算的文本序列：
        # token ID 用于恢复可读文本，input position 用于定位完整多模态输入中的位置。
        # 它们属于 query 级元数据，因此同一次 forward 中的所有候选图像共用一份。
        self.last_query_token_ids: Optional[torch.Tensor] = None
        self.last_query_input_positions: Optional[torch.Tensor] = None
        self.last_text_extraction_mode: Optional[str] = None

        # 基类先调用 compute_t2i_similarity，再通过 _record_similarity_stats
        # 建立对应图像的统计字典。这里用 FIFO 暂存 argmax，维持两次回调之间的逐图对应。
        self._pending_max_text_token_indices: List[torch.Tensor] = []

        # 数据集查询问题使用独立的坐标系和相似度队列。这样实际剪枝仍可沿用检查点
        # 配置的 all_before_image/query_only，诊断 argmax 却不会混入任何固定提示词。
        self.last_dataset_query_token_ids: Optional[torch.Tensor] = None
        self.last_dataset_query_input_positions: Optional[torch.Tensor] = None
        self.last_alignment_text_scope: Optional[str] = None
        self._dataset_query_normalized_embeds: Optional[torch.Tensor] = None
        self._pending_dataset_query_max_similarities: List[torch.Tensor] = []
        self._pending_dataset_query_max_text_token_indices: List[torch.Tensor] = []

    def set_collect_similarity_stats(self, enabled: bool) -> None:
        super().set_collect_similarity_stats(enabled)
        if not enabled:
            # 关闭诊断时同时释放 query 元数据和尚未写入统计记录的 argmax，防止后续
            # forward 误读上一条 query 的映射，也避免 tensor 长时间占用设备内存。
            self.last_query_token_ids = None
            self.last_query_input_positions = None
            self.last_text_extraction_mode = None
            self._pending_max_text_token_indices = []
            self.clear_dataset_query_alignment_metadata()

    def set_dataset_query_alignment_metadata(
        self,
        token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        query_embeds: torch.Tensor,
    ) -> None:
        """Attach the dataset-query-only token axis used by the diagnostic."""
        if not self.collect_similarity_stats:
            return
        if token_ids.ndim != 1 or input_positions.ndim != 1 or query_embeds.ndim != 2:
            raise ValueError("Dataset-query alignment metadata has invalid dimensions")
        if not (
            token_ids.numel()
            == input_positions.numel()
            == query_embeds.shape[0]
        ):
            raise ValueError(
                "Dataset-query token IDs, positions, and embeddings must have "
                "the same length"
            )
        if token_ids.numel() == 0:
            raise ValueError("Dataset-query alignment cannot use an empty token sequence")

        # Only the normalized query embeddings remain on the model device. Token IDs
        # and positions are detached alongside them and copied to CPU by the consumer
        # immediately after the forward pass.
        self.last_dataset_query_token_ids = token_ids.detach()
        self.last_dataset_query_input_positions = input_positions.detach()
        self.last_alignment_text_scope = "dataset_query_only"
        self._dataset_query_normalized_embeds = normalize(
            query_embeds.detach(), dim=-1
        )

    def clear_dataset_query_alignment_metadata(self) -> None:
        """Release dataset-query tensors after one diagnostic forward."""
        self.last_dataset_query_token_ids = None
        self.last_dataset_query_input_positions = None
        self.last_alignment_text_scope = None
        self._dataset_query_normalized_embeds = None
        self._pending_dataset_query_max_similarities = []
        self._pending_dataset_query_max_text_token_indices = []

    def set_query_token_metadata(
        self,
        token_ids: torch.Tensor,
        input_positions: torch.Tensor,
        text_extraction_mode: str,
    ) -> None:
        """Attach the exact input-token mapping used to build ``text_embeds``."""
        if not self.collect_similarity_stats:
            return

        # 只切断自动求导关系，不改变顺序：第 i 个 token ID 和 input position
        # 必须始终对应 text_embeds 的第 i 行，后续 argmax 才能安全地回查原始输入。
        self.last_query_token_ids = token_ids.detach()
        self.last_query_input_positions = input_positions.detach()
        self.last_text_extraction_mode = text_extraction_mode

    def _begin_similarity_collection(self) -> None:
        super()._begin_similarity_collection()

        # 基类已开始一轮新的逐图统计；同步清空辅助队列，使其生命周期也严格限制
        # 在当前 forward 内，即使上一次诊断中途失败也不会串用残留索引。
        self._pending_max_text_token_indices = []
        self._pending_dataset_query_max_similarities = []
        self._pending_dataset_query_max_text_token_indices = []

    def compute_t2i_similarity(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the same cosine scores as the base pruner and retain argmax."""
        text_norm = normalize(text_embeds, dim=-1)
        image_norm = normalize(image_embeds, dim=-1)

        # similarity 的行对应 QI 文本 token，列对应剪枝前视觉 token。
        # 沿 dim=0 取最大值后，每一列同时得到剪枝分数和命中的文本序列下标；
        # 若多个文本 token 分数并列，PyTorch 会返回最先出现的那个下标。
        similarity = torch.mm(text_norm, image_norm.t())
        max_similarity, max_text_token_indices = similarity.max(dim=0)

        if self.last_similarity_stats is not None:
            # argmax 与 max_similarity 等长，并保持原始视觉 token 顺序。先暂存而不
            # 单独创建统计字典，确保分数、保留掩码和 argmax 最终落在同一条记录中。
            self._pending_max_text_token_indices.append(
                max_text_token_indices.detach()
            )

            # 另行计算严格查询范围内的最大值和 argmax，只用于分析输出，不返回给
            # top-k 剪枝。因此无论查询范围如何收紧，模型保留的视觉 token 都不变。
            if self._dataset_query_normalized_embeds is None:
                raise RuntimeError(
                    "Dataset-query embeddings were not registered before QI-Early pruning"
                )
            dataset_query_similarity = torch.mm(
                self._dataset_query_normalized_embeds,
                image_norm.t(),
            )
            (
                dataset_query_max_similarity,
                dataset_query_max_text_token_indices,
            ) = dataset_query_similarity.max(dim=0)
            self._pending_dataset_query_max_similarities.append(
                dataset_query_max_similarity.detach()
            )
            self._pending_dataset_query_max_text_token_indices.append(
                dataset_query_max_text_token_indices.detach()
            )
        return max_similarity

    def _record_similarity_stats(
        self,
        t2i_scores: torch.Tensor,
        selected_idx: torch.Tensor,
        n_patches: int,
        k: int,
    ) -> None:
        collecting = self.last_similarity_stats is not None
        super()._record_similarity_stats(t2i_scores, selected_idx, n_patches, k)
        if not collecting:
            return
        if self.last_similarity_stats is None:
            raise RuntimeError("QI-Early statistics disappeared while being recorded")
        if not self._pending_max_text_token_indices:
            raise RuntimeError("Missing max-text-token indices for QI-Early statistics")
        if not self._pending_dataset_query_max_similarities:
            raise RuntimeError("Missing dataset-query similarities for QI-Early statistics")
        if not self._pending_dataset_query_max_text_token_indices:
            raise RuntimeError("Missing dataset-query argmax indices for QI-Early statistics")

        # super() 刚把当前图像的基础统计追加到列表末尾；取出队首的 argmax 后
        # 写入该字典，由此形成 visual_token_index -> query_sequence_index 的映射。
        self.last_similarity_stats[-1]["max_text_token_indices"] = (
            self._pending_max_text_token_indices.pop(0)
        )
        self.last_similarity_stats[-1]["dataset_query_max_similarities"] = (
            self._pending_dataset_query_max_similarities.pop(0)
        )
        self.last_similarity_stats[-1]["dataset_query_max_text_token_indices"] = (
            self._pending_dataset_query_max_text_token_indices.pop(0)
        )


class Qwen3VLWithQIEarlyTokenAlignment(Qwen3VLWithQIEarly):
    """Diagnostic model that maps QI text embeddings back to input token IDs."""

    def __init__(self, config) -> None:
        super().__init__(config)
        # The pruner contains no learned parameters, so replacing it does not
        # alter checkpoint loading or model weights.
        self.qi_early_pruner = QIEarlyTokenAlignmentPruner(
            temperature=self.qi_early_temperature,
            keep_ratio=self.qi_early_keep_ratio,
        )
        self._pending_dataset_query_input_positions: Optional[torch.Tensor] = None

    def set_token_alignment_query_input_positions(
        self,
        input_positions: torch.Tensor,
    ) -> None:
        """Register exact dataset-query positions for the next model forward."""
        positions = torch.as_tensor(input_positions).detach().to(
            device="cpu", dtype=torch.long
        ).contiguous()
        if positions.ndim != 1 or positions.numel() == 0:
            raise ValueError("Dataset-query input positions must be a non-empty 1-D tensor")
        if positions.numel() > 1 and not torch.all(positions[1:] > positions[:-1]):
            raise ValueError("Dataset-query input positions must be strictly increasing")
        self._pending_dataset_query_input_positions = positions

    def _query_text_input_positions(
        self,
        input_ids: torch.Tensor,
        mode: TextExtractionMode,
    ) -> torch.Tensor:
        """Reconstruct the exact mask used by the base embedding extractor."""
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"QI-Early token alignment requires batch_size=1, got {input_ids.shape[0]}"
            )

        ids = input_ids[0]
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        vision_end_token_id = self.config.vision_end_token_id

        # 第一枚图像占位 token 是文本区间的右边界。当前诊断仅支持单 batch，
        # 因而返回的一维位置可以直接索引 input_ids[0] 和 inputs_embeds[0]。
        image_mask = (ids == image_token_id) | (ids == video_token_id)
        image_positions = torch.where(image_mask)[0]
        if image_positions.numel() == 0:
            raise ValueError("No image tokens found while mapping QI text tokens")
        first_image_pos = int(image_positions[0])

        exclude_mask = (
            (ids == image_token_id)
            | (ids == video_token_id)
            | (ids == vision_start_token_id)
            | (ids == vision_end_token_id)
        )
        if self.query_emb_token_id is not None:
            exclude_mask = exclude_mask | (ids == self.query_emb_token_id)

        # 这里必须逐项复刻生产模型的文本抽取规则。query_only 仅保留查询正文，
        # all_before_image 则保留首图之前的全部文本；视觉边界和查询适配器占位符
        # 在两种模式下都不能进入相似度矩阵。
        sequence_positions = torch.arange(ids.numel(), device=ids.device)
        if mode == "query_only":
            query_start = self._find_query_start_position(ids)
            if query_start is None or query_start >= first_image_pos:
                raise ValueError("Could not map query_only embeddings to input token IDs")
            position_mask = (
                (sequence_positions >= query_start)
                & (sequence_positions < first_image_pos)
            )
        elif mode == "all_before_image":
            position_mask = sequence_positions < first_image_pos
        else:
            raise ValueError(f"Unsupported QI-Early text mode: {mode}")

        return torch.where(position_mask & ~exclude_mask)[0]

    def _extract_query_text_embeds(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        mode: Optional[TextExtractionMode] = None,
    ) -> torch.Tensor:
        # Let the production implementation compute the embeddings.  We only
        # reconstruct and validate their source token positions afterward.
        query_embeds = super()._extract_query_text_embeds(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            mode=mode,
        )
        effective_mode = cast(TextExtractionMode, mode or self.qi_early_text_mode)
        input_positions = self._query_text_input_positions(input_ids, effective_mode)

        # 数量相等是整个对齐导出的核心不变量：若映射长度与 embedding 行数不同，
        # 即使相似度仍可计算，argmax 下标也会指向错误的 token，因此直接终止。
        if input_positions.numel() != query_embeds.shape[0]:
            raise RuntimeError(
                "QI text embedding/token mapping length mismatch: "
                f"embeddings={query_embeds.shape[0]}, positions={input_positions.numel()}"
            )

        # 将过滤后的 token ID 与其在完整输入中的位置交给 pruner；这样 collector
        # 既能输出紧凑的 query_sequence_index，也能输出可追溯的 input position。
        self.qi_early_pruner.set_query_token_metadata(
            token_ids=input_ids[0, input_positions],
            input_positions=input_positions,
            text_extraction_mode=effective_mode,
        )

        # The evaluation entry point derives these positions from tokenizer offsets
        # around the literal dataset query. They define a second, diagnostic-only
        # similarity matrix whose rows cannot include RankGPT instructions or labels.
        if self.qi_early_pruner.collect_similarity_stats:
            dataset_query_positions = self._pending_dataset_query_input_positions
            self._pending_dataset_query_input_positions = None
            if dataset_query_positions is None:
                raise RuntimeError(
                    "Dataset-query positions were not registered for token alignment"
                )
            dataset_query_positions = dataset_query_positions.to(input_ids.device)
            if input_ids.shape[0] != 1:
                raise ValueError(
                    "Dataset-query token alignment requires batch_size=1"
                )
            if (
                int(dataset_query_positions.min()) < 0
                or int(dataset_query_positions.max()) >= input_ids.shape[1]
            ):
                raise ValueError("A dataset-query position falls outside input_ids")

            forbidden_token_ids = {
                self.config.image_token_id,
                self.config.video_token_id,
                self.config.vision_start_token_id,
                self.config.vision_end_token_id,
            }
            dataset_query_token_ids = input_ids[0, dataset_query_positions]
            if any(
                int(token_id) in forbidden_token_ids
                for token_id in dataset_query_token_ids
            ):
                raise ValueError("Dataset-query positions contain a visual special token")

            self.qi_early_pruner.set_dataset_query_alignment_metadata(
                token_ids=dataset_query_token_ids,
                input_positions=dataset_query_positions,
                query_embeds=inputs_embeds[0, dataset_query_positions],
            )
        return query_embeds
