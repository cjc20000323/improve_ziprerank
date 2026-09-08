"""
Query-Image Early Interaction Visual Token Pruner for Qwen3-VL.

This pruner uses Text-to-Image (T2I) similarity for efficient visual token selection,
reducing the number of visual tokens processed by the LLM while preserving relevance
to the query.
"""

from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class QIEarlyInteractionPruner(nn.Module):
    """
    Query-Image Early Interaction visual token pruner.
    
    Token selection is based on Text-to-Image similarity:
    - Compute cosine similarity between each visual patch and text tokens
    - Take max similarity per patch (how relevant is this patch to the query)
    - Keep top-k patches by T2I score
    
    Args:
        temperature: Temperature for T2I similarity softmax scaling (default: 0.1)
        keep_ratio: Fraction of tokens to keep per image (default: 0.25)
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        keep_ratio: float = 0.25,
    ):
        super().__init__()
        self.temperature = temperature
        self.keep_ratio = keep_ratio

        # 【相似度分析开关】仅用于评估阶段的可视化诊断。
        # 默认关闭，因此正常训练/评估时不会额外持有 score tensor，也不会增加显存占用。
        # 开启后，last_similarity_stats 按“输入图片顺序”保存本次 forward 中每张图的统计。
        self.collect_similarity_stats = False
        self.last_similarity_stats: Optional[List[Dict[str, object]]] = None

    def set_collect_similarity_stats(self, enabled: bool) -> None:
        """控制是否暂存最近一次 forward 的逐图 token 相似度。"""
        self.collect_similarity_stats = enabled
        if not enabled:
            self.last_similarity_stats = None

    def _begin_similarity_collection(self) -> None:
        """每次剪枝前清空旧结果，防止评估脚本误读上一个 query 的分数。"""
        self.last_similarity_stats = [] if self.collect_similarity_stats else None

    def _record_similarity_stats(
        self,
        t2i_scores: torch.Tensor,
        selected_idx: torch.Tensor,
        n_patches: int,
        k: int,
    ) -> None:
        """暂存一张图中真正用于 top-k 剪枝的分数，不重新计算也不改变剪枝。"""
        if self.last_similarity_stats is None:
            return

        # t2i_scores[j] 是第 j 个视觉 token 与所有文本 token 的最大余弦相似度。
        # kept_mask 与原始视觉 token 顺序对齐，True 表示该 token 被保留。
        kept_mask = torch.zeros(n_patches, dtype=torch.bool, device=t2i_scores.device)
        kept_mask[selected_idx] = True

        # 每张图片独立做 top-k，因此每张图都有自己的 50% 保留阈值。
        # 阈值等于所有被保留 token 中的最小相似度。
        threshold = t2i_scores[selected_idx].min()

        # detach 只切断梯度关系，不改变数值。这里暂时留在当前设备，等完整模型 forward
        # 结束后由 evaluate.py 一次性复制到 CPU，随后立即清空 GPU 引用。
        self.last_similarity_stats.append({
            "scores": t2i_scores.detach(),
            "kept_mask": kept_mask.detach(),
            "selected_indices": selected_idx.detach(),
            "threshold": threshold.detach(),
            "num_original_tokens": n_patches,
            "num_kept_tokens": k,
        })
    
    def compute_t2i_similarity(
        self,
        text_embeds: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Text-to-Image similarity scores for each visual token.
        
        For each visual patch, compute the maximum cosine similarity with any
        text token. This identifies patches semantically aligned with the query.
        
        Args:
            text_embeds: Text embeddings, shape (N_text, D)
            image_embeds: Visual patch embeddings, shape (N_patches, D)
            
        Returns:
            T2I similarity scores, shape (N_patches,)
        """
        # Normalize embeddings for cosine similarity
        text_norm = F.normalize(text_embeds, dim=-1)  # (N_text, D)
        image_norm = F.normalize(image_embeds, dim=-1)  # (N_patches, D)
        
        # Compute cosine similarity matrix: (N_text, N_patches)
        similarity = torch.mm(text_norm, image_norm.t())
        
        # Max over text tokens for each patch
        max_sim, _ = similarity.max(dim=0)  # (N_patches,)
        
        return max_sim
    
    @torch.no_grad()
    def forward(
        self,
        text_embeds: torch.Tensor,
        image_embeds_list: List[torch.Tensor],
        keep_ratio: Optional[float] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Prune visual tokens using T2I similarity.
        
        Args:
            text_embeds: Text embeddings for T2I similarity, shape (N_text, D)
            image_embeds_list: List of visual embeddings per image, each (N_i, D)
            keep_ratio: Override default keep_ratio if provided
            
        Returns:
            pruned_embeds_list: List of pruned embeddings per image, each (K_i, D)
            selected_indices_list: List of selected indices per image, each (K_i,)
        """
        if keep_ratio is None:
            keep_ratio = self.keep_ratio

        self._begin_similarity_collection()
        
        pruned_embeds_list = []
        selected_indices_list = []
        
        for image_embeds in image_embeds_list:
            n_patches = image_embeds.shape[0]
            k = max(1, int(round(keep_ratio * n_patches)))  # At least 1 token
            
            # Compute T2I similarity scores
            t2i_scores = self.compute_t2i_similarity(text_embeds, image_embeds)
            
            # Simply take top-k by T2I score
            _, top_indices = torch.topk(t2i_scores, k, dim=0)
            
            # Sort indices to maintain spatial order
            selected_idx = torch.sort(top_indices)[0]

            # 记录的是 top-k 已经使用的原始分数；该调用不参与模型输出计算。
            self._record_similarity_stats(t2i_scores, selected_idx, n_patches, k)
            
            # Select tokens
            pruned_embeds = image_embeds[selected_idx]
            
            pruned_embeds_list.append(pruned_embeds)
            selected_indices_list.append(selected_idx)
        
        return pruned_embeds_list, selected_indices_list
    
    def forward_with_grad(
        self,
        text_embeds: torch.Tensor,
        image_embeds_list: List[torch.Tensor],
        keep_ratio: Optional[float] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Prune visual tokens with gradient flow through kept tokens.
        
        Same as forward() but allows gradients to flow through selected tokens.
        """
        if keep_ratio is None:
            keep_ratio = self.keep_ratio

        self._begin_similarity_collection()
        
        pruned_embeds_list = []
        selected_indices_list = []
        
        for image_embeds in image_embeds_list:
            n_patches = image_embeds.shape[0]
            k = max(1, int(round(keep_ratio * n_patches)))
            
            # Compute scores WITHOUT gradients - just for selection
            with torch.no_grad():
                t2i_scores = self.compute_t2i_similarity(text_embeds, image_embeds)
                _, top_indices = torch.topk(t2i_scores, k, dim=0)
                selected_idx = torch.sort(top_indices)[0]

            # training mode 也保持相同的诊断格式，便于后续复用。
            self._record_similarity_stats(t2i_scores, selected_idx, n_patches, k)
            
            # Select tokens WITH gradients
            pruned_embeds = image_embeds[selected_idx]
            
            pruned_embeds_list.append(pruned_embeds)
            selected_indices_list.append(selected_idx)
        
        return pruned_embeds_list, selected_indices_list
    
    def prune_deepstack_features(
        self,
        deepstack_embeds_list: List[torch.Tensor],
        selected_indices_list: List[torch.Tensor],
        tokens_per_image: List[int],
    ) -> List[torch.Tensor]:
        """
        Apply the same token selection to deepstack features.
        """
        pruned_deepstack_list = []
        
        for deepstack_embeds in deepstack_embeds_list:
            pruned_per_image = []
            offset = 0
            
            for n_i, selected_idx in zip(tokens_per_image, selected_indices_list):
                img_embeds = deepstack_embeds[offset:offset + n_i]
                pruned_img = img_embeds[selected_idx]
                pruned_per_image.append(pruned_img)
                offset += n_i
            
            pruned_deepstack = torch.cat(pruned_per_image, dim=0)
            pruned_deepstack_list.append(pruned_deepstack)
        
        return pruned_deepstack_list


def create_qi_early_pruner(
    temperature: float = 0.1,
    keep_ratio: float = 0.25,
) -> QIEarlyInteractionPruner:
    """
    Factory function to create a QI-Early pruner.
    
    Args:
        temperature: Temperature for T2I similarity (lower = sharper)
        keep_ratio: Fraction of tokens to keep (0.25 = 75% reduction)
        
    Returns:
        Configured QIEarlyInteractionPruner instance
    """
    return QIEarlyInteractionPruner(
        temperature=temperature,
        keep_ratio=keep_ratio,
    )

