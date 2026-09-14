"""Diagnostic Qwen3-VL variant for visual-token Logit Lens analysis.

The production QI-Early model deliberately returns only the final language-model
state.  This subclass leaves its pruning and ranking path unchanged and installs
forward hooks on a configurable number of late decoder blocks.  Each hook keeps
only the hidden states at visual positions that survived QI-Early pruning.

The vocabulary projection is intentionally performed later, after reranking has
identified the four candidates of interest.  Capturing hidden states first avoids
materializing a ``num_visual_tokens x vocab_size`` tensor for every candidate in
the Top-K window.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from transformers.processing_utils import Unpack
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
)
from transformers.utils import TransformersKwargs

from .qwen3vl_with_qi_early import Qwen3VLWithQIEarly


def select_spaced_late_layer_indices(
    num_layers: int,
    num_selected_layers: int = 4,
    start_ratio: float = 0.6,
    min_layer_index: int = 0,
) -> Tuple[int, ...]:
    """Choose evenly spaced decoder blocks from the late-depth region.

    ``start_ratio`` describes the earliest one-based model depth that may be
    selected. The final decoder block is always included, and the intervening
    selections are distributed as evenly as integer layer indices permit.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")
    if num_selected_layers <= 0:
        raise ValueError(
            "num_selected_layers must be positive, got "
            f"{num_selected_layers}"
        )
    if not 0.5 <= start_ratio < 1.0:
        raise ValueError(
            f"start_ratio must be in [0.5, 1.0), got {start_ratio}"
        )
    if not 0 <= min_layer_index < num_layers:
        raise ValueError(
            f"min_layer_index must be in [0, {num_layers}), got {min_layer_index}"
        )

    first_ratio_layer = math.ceil(num_layers * start_ratio) - 1
    first_layer = max(first_ratio_layer, min_layer_index)
    available_layers = num_layers - first_layer
    if num_selected_layers > available_layers:
        raise ValueError(
            f"Cannot select {num_selected_layers} distinct layers from late range "
            f"[{first_layer}, {num_layers - 1}]. Lower start_ratio or request "
            "fewer layers."
        )
    if num_selected_layers == 1:
        return (num_layers - 1,)

    # Rounding equally spaced points keeps the first and final blocks fixed while
    # distributing any indivisible remainder across the internal layer gaps.
    span = num_layers - 1 - first_layer
    layer_indices = tuple(
        first_layer + round(position * span / (num_selected_layers - 1))
        for position in range(num_selected_layers)
    )
    if len(set(layer_indices)) != num_selected_layers:
        raise RuntimeError(
            f"Spaced layer selection unexpectedly produced duplicates: {layer_indices}"
        )
    return layer_indices


class Qwen3VLWithQIEarlyLogitLens(Qwen3VLWithQIEarly):
    """QI-Early model that captures selected visual states from late layers."""

    def __init__(self, config) -> None:
        super().__init__(config)

        # Hooks are registered only after checkpoint loading, when the caller has
        # selected how many late layers to inspect.  Keeping this list also makes
        # repeated configuration safe in notebooks and tests.
        self._logit_lens_hook_handles: List[Any] = []
        self._logit_lens_layer_indices: Tuple[int, ...] = ()

        # ``last_logit_lens_capture`` belongs to the most recent multimodal
        # prefill.  Generation may subsequently call text-only decode forwards;
        # those calls must not overwrite the visual states saved here.
        self.last_logit_lens_capture: Optional[Dict[str, Any]] = None
        self._logit_lens_capture_active = False
        self._active_visual_sequence_positions: Optional[torch.Tensor] = None

    @property
    def logit_lens_layer_indices(self) -> Tuple[int, ...]:
        """Zero-based decoder-block indices currently observed by the lens."""
        return self._logit_lens_layer_indices

    def configure_logit_lens(self, last_n_layers: int = 4) -> Tuple[int, ...]:
        """Capture the outputs of the last ``last_n_layers`` decoder blocks.

        Qwen3-VL adds DeepStack image features immediately after its first few
        decoder blocks.  A module hook on one of those blocks would run before
        that addition and therefore would not represent the completed layer
        state.  This diagnostic is explicitly for late layers and rejects a
        range that overlaps the DeepStack injection region.
        """
        if last_n_layers <= 0:
            raise ValueError(
                f"last_n_layers must be positive, got {last_n_layers}"
            )

        layers = self.model.language_model.layers
        num_layers = len(layers)
        if last_n_layers > num_layers:
            raise ValueError(
                f"Requested {last_n_layers} layers, but the model has {num_layers}"
            )

        first_layer = num_layers - last_n_layers
        num_deepstack_layers = len(
            getattr(self.model.visual, "deepstack_visual_indexes", [])
        )
        if first_layer < num_deepstack_layers:
            raise ValueError(
                "Late-layer Logit Lens hooks may not overlap Qwen3-VL's "
                f"DeepStack injection layers [0, {num_deepstack_layers - 1}]"
            )

        self.remove_logit_lens_hooks()
        self._logit_lens_layer_indices = tuple(range(first_layer, num_layers))

        # Each hook receives the residual stream immediately after its decoder
        # block.  The final RMSNorm is not part of a block, so it is deliberately
        # applied later by the vocabulary projector, as required by Logit Lens.
        for layer_index in self._logit_lens_layer_indices:
            handle = layers[layer_index].register_forward_hook(
                self._make_logit_lens_hook(layer_index)
            )
            self._logit_lens_hook_handles.append(handle)

        return self._logit_lens_layer_indices

    def configure_spaced_late_logit_lens(
        self,
        num_selected_layers: int = 4,
        start_ratio: float = 0.6,
    ) -> Tuple[int, ...]:
        """Capture spaced decoder blocks concentrated in the later model depth."""
        layers = self.model.language_model.layers
        num_layers = len(layers)
        num_deepstack_layers = len(
            getattr(self.model.visual, "deepstack_visual_indexes", [])
        )
        layer_indices = select_spaced_late_layer_indices(
            num_layers=num_layers,
            num_selected_layers=num_selected_layers,
            start_ratio=start_ratio,
            min_layer_index=num_deepstack_layers,
        )

        self.remove_logit_lens_hooks()
        self._logit_lens_layer_indices = layer_indices

        # Hooks observe the residual stream after each selected decoder block.
        # The shared final RMSNorm remains deferred to the vocabulary projector,
        # so all four layer states use the same Logit Lens readout operation.
        for layer_index in self._logit_lens_layer_indices:
            handle = layers[layer_index].register_forward_hook(
                self._make_logit_lens_hook(layer_index)
            )
            self._logit_lens_hook_handles.append(handle)

        return self._logit_lens_layer_indices

    def remove_logit_lens_hooks(self) -> None:
        """Remove previously registered hooks without changing captured data."""
        for handle in self._logit_lens_hook_handles:
            handle.remove()
        self._logit_lens_hook_handles = []

    def _make_logit_lens_hook(self, layer_index: int):
        def capture_visual_hidden_states(_module, _inputs, output) -> None:
            if not self._logit_lens_capture_active:
                return
            if self.last_logit_lens_capture is None:
                raise RuntimeError("Logit Lens capture storage was not initialized")
            if self._active_visual_sequence_positions is None:
                raise RuntimeError("Pruned visual sequence positions were not recorded")

            hidden_states = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
                raise TypeError(
                    "A Qwen3-VL decoder hook must receive [batch, sequence, hidden] "
                    "tensor output"
                )
            if hidden_states.shape[0] != 1:
                raise ValueError(
                    "Visual Logit Lens analysis requires decoder batch_size=1"
                )

            # Only retained visual positions are copied.  The tensor stays in
            # visual sequence order, which is also image order followed by each
            # image's spatial token order.  Moving it to CPU here prevents four
            # late-layer captures from occupying GPU memory during generation.
            positions = self._active_visual_sequence_positions.to(
                hidden_states.device
            )
            visual_hidden_states = hidden_states[0].index_select(0, positions)
            self.last_logit_lens_capture["layer_hidden_states"][layer_index] = (
                visual_hidden_states.detach().to(device="cpu").contiguous()
            )

        return capture_visual_hidden_states

    def _build_pruned_sequence_with_positions(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        input_ids: torch.Tensor,
        original_position_ids: torch.Tensor,
        image_mask: torch.Tensor,
        pruned_image_embeds: torch.Tensor,
        tokens_per_image: List[int],
        pruned_tokens_per_image: List[int],
        selected_indices_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        result = super()._build_pruned_sequence_with_positions(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            input_ids=input_ids,
            original_position_ids=original_position_ids,
            image_mask=image_mask,
            pruned_image_embeds=pruned_image_embeds,
            tokens_per_image=tokens_per_image,
            pruned_tokens_per_image=pruned_tokens_per_image,
            selected_indices_list=selected_indices_list,
        )

        if not self._logit_lens_capture_active:
            return result
        if self.last_logit_lens_capture is None:
            raise RuntimeError("Logit Lens capture storage was not initialized")

        visual_pos_mask = result[3]
        visual_positions = torch.where(visual_pos_mask[0])[0]
        expected_visual_tokens = sum(pruned_tokens_per_image)
        if visual_positions.numel() != expected_visual_tokens:
            raise RuntimeError(
                "Pruned visual-position count mismatch: "
                f"mask={visual_positions.numel()}, expected={expected_visual_tokens}"
            )

        # Split the new sequence positions by image using the same counts that
        # built the pruned sequence.  Together with selected_indices_list, this
        # provides both coordinate systems needed later: position in the LLM
        # sequence and position in the original unpruned visual grid.
        sequence_positions_by_image = []
        offset = 0
        for count in pruned_tokens_per_image:
            sequence_positions_by_image.append(
                visual_positions[offset:offset + count].detach().cpu().contiguous()
            )
            offset += count

        self._active_visual_sequence_positions = visual_positions.detach()
        self.last_logit_lens_capture.update({
            "original_tokens_per_image": [int(value) for value in tokens_per_image],
            "pruned_tokens_per_image": [
                int(value) for value in pruned_tokens_per_image
            ],
            "selected_indices_by_image": [
                indices.detach().to(device="cpu", dtype=torch.long).contiguous()
                for indices in selected_indices_list
            ],
            "visual_sequence_positions_by_image": sequence_positions_by_image,
        })
        return result

    def _validate_completed_capture(self) -> None:
        if self.last_logit_lens_capture is None:
            raise RuntimeError("Logit Lens capture disappeared during model forward")

        missing_layers = set(self._logit_lens_layer_indices) - set(
            self.last_logit_lens_capture["layer_hidden_states"]
        )
        if missing_layers:
            raise RuntimeError(
                f"No visual hidden states were captured for layers {sorted(missing_layers)}"
            )

        raw_counts = self.last_logit_lens_capture.get("pruned_tokens_per_image")
        if not isinstance(raw_counts, (list, tuple)):
            raise RuntimeError("Pruned token metadata was not captured")
        counts = [int(count) for count in raw_counts]
        expected_total = sum(counts)
        for layer_index, hidden_states in self.last_logit_lens_capture[
            "layer_hidden_states"
        ].items():
            if hidden_states.shape[0] != expected_total:
                raise RuntimeError(
                    f"Layer {layer_index} captured {hidden_states.shape[0]} visual "
                    f"states, expected {expected_total}"
                )

        self.last_logit_lens_capture["capture_complete"] = True

    def pop_last_logit_lens_capture(self) -> Optional[Dict[str, Any]]:
        """Return and clear the most recent completed multimodal capture."""
        capture = self.last_logit_lens_capture
        self.last_logit_lens_capture = None
        return capture

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Qwen3VLCausalLMOutputWithPast:
        # Only the multimodal prefill enters the QI-Early branch.  Autoregressive
        # decode calls have no pixel_values and must leave the saved visual
        # capture untouched until the evaluation consumer retrieves it.
        capture_this_forward = (
            self.qi_early_enabled
            and pixel_values is not None
            and bool(self._logit_lens_layer_indices)
        )

        if capture_this_forward:
            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required for Logit Lens capture")
            self.last_logit_lens_capture = {
                "layer_indices": list(self._logit_lens_layer_indices),
                "image_grid_thw": image_grid_thw.detach()
                .to(device="cpu", dtype=torch.long)
                .contiguous(),
                "layer_hidden_states": {},
                "capture_complete": False,
            }
            self._active_visual_sequence_positions = None
            self._logit_lens_capture_active = True

        try:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
            if capture_this_forward:
                self._validate_completed_capture()
            return outputs
        except Exception:
            if capture_this_forward:
                self.last_logit_lens_capture = None
            raise
        finally:
            if capture_this_forward:
                self._logit_lens_capture_active = False
                self._active_visual_sequence_positions = None
