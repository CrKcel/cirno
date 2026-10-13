"""Configuration classes for the hybrid vision-language model (HVLM)."""
try:  # transformers >= 5 renamed the base class (the old name is kept as an alias in most releases)
    from transformers import PreTrainedConfig as _BaseConfig
except ImportError:  # pragma: no cover
    from transformers import PretrainedConfig as _BaseConfig


class HVLMVisionConfig(_BaseConfig):
    """A small ViT (fixed square input) followed by a 2x2 spatial-merge MLP projector."""

    model_type = "hvlm_vision"

    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        hidden_size=384,
        intermediate_size=1536,
        depth=6,
        num_heads=6,
        spatial_merge_size=2,
        layer_norm_eps=1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.depth = depth
        self.num_heads = num_heads
        self.spatial_merge_size = spatial_merge_size
        self.layer_norm_eps = layer_norm_eps
        if image_size % patch_size != 0 or (image_size // patch_size) % spatial_merge_size != 0:
            raise ValueError("image_size / patch_size must be divisible by spatial_merge_size")

    @property
    def grid_size(self):  # patches per side
        return self.image_size // self.patch_size

    @property
    def merged_grid_size(self):  # LLM tokens per side after the 2x2 merge
        return self.grid_size // self.spatial_merge_size

    @property
    def num_image_tokens(self):  # LLM tokens per image
        return self.merged_grid_size ** 2


class HVLMConfig(_BaseConfig):
    """Text backbone = Qwen3.5-style hybrid stack: (N-1) Gated DeltaNet layers then 1 Gated Attention layer,
    every layer followed by a dense SwiGLU MLP.  `mtp_num_layers` extra multi-token-prediction blocks."""

    model_type = "hvlm"
    sub_configs = {"vision_config": HVLMVisionConfig}

    def __init__(
        self,
        vocab_size=32768,
        hidden_size=640,
        intermediate_size=1792,
        num_hidden_layers=20,
        full_attention_interval=4,
        layer_types=None,
        num_attention_heads=10,
        num_key_value_heads=2,
        head_dim=64,
        partial_rotary_factor=0.5,
        rope_theta=1_000_000.0,
        mrope_section=(8, 4, 4),
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        linear_attn_impl="torch",
        linear_chunk_size=64,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        initializer_range=0.02,
        tie_word_embeddings=False,  # the LM head IS embed_tokens.weight by construction; see modeling_hvlm.py
        mtp_num_layers=1,
        mtp_loss_weight=0.3,
        ce_chunk_size=2048,
        image_token_id=5,
        vision_start_token_id=3,
        vision_end_token_id=4,
        vision_config=None,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            vision_config = HVLMVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = HVLMVisionConfig()
        self.vision_config = vision_config

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.full_attention_interval = full_attention_interval
        if layer_types is None:
            layer_types = [
                "full_attention" if (i + 1) % full_attention_interval == 0 else "linear_attention"
                for i in range(num_hidden_layers)
            ]
        if len(layer_types) != num_hidden_layers:
            raise ValueError("len(layer_types) must equal num_hidden_layers")
        self.layer_types = list(layer_types)
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.partial_rotary_factor = partial_rotary_factor
        self.rope_theta = rope_theta
        self.mrope_section = list(mrope_section)
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_attn_impl = linear_attn_impl  # "torch" (pure PyTorch) or "fla" (flash-linear-attention kernels)
        self.linear_chunk_size = linear_chunk_size
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.mtp_num_layers = mtp_num_layers
        self.mtp_loss_weight = mtp_loss_weight
        self.ce_chunk_size = ce_chunk_size
        self.image_token_id = image_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id

        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if linear_num_value_heads % linear_num_key_heads != 0:
            raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
        if self.rotary_dim % 2 != 0 or sum(self.mrope_section) != self.rotary_dim // 2:
            raise ValueError(
                f"sum(mrope_section)={sum(self.mrope_section)} must equal rotary_dim/2={self.rotary_dim // 2}"
            )
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    @property
    def rotary_dim(self):
        return int(self.head_dim * self.partial_rotary_factor)
