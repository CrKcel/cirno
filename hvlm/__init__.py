"""Hybrid (Gated DeltaNet + Gated Attention) vision-language model, ~130M parameters.

Importing the package does not import torch/transformers, so the numpy-only helpers
(`hvlm.positions`, `hvlm.data_core`, `hvlm.monitor`) can be used and tested anywhere.  Model classes are
loaded lazily.
"""

__all__ = ["HVLMConfig", "HVLMVisionConfig", "HVLMForConditionalGeneration", "register"]


def __getattr__(name):
    if name in ("HVLMConfig", "HVLMVisionConfig"):
        from . import configuration_hvlm as m

        return getattr(m, name)
    if name == "HVLMForConditionalGeneration":
        from . import modeling_hvlm as m

        return m.HVLMForConditionalGeneration
    raise AttributeError(name)


def register():
    """Register the model with the transformers Auto classes (AutoConfig / AutoModelForCausalLM)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from .configuration_hvlm import HVLMConfig
    from .modeling_hvlm import HVLMForConditionalGeneration

    AutoConfig.register("hvlm", HVLMConfig, exist_ok=True)
    AutoModelForCausalLM.register(HVLMConfig, HVLMForConditionalGeneration, exist_ok=True)
