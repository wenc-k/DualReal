# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
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

import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import colorama
from typing import Any, Dict, Optional, Tuple, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.attention_processor import AttentionProcessor, CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

class Adapter(nn.Module):
    def __init__(self, in_dim, hidden_dim, condition_dim=None, use_ff_adapter=False):
        super().__init__()
        self.down_linear = nn.Linear(in_dim, hidden_dim)
        self.up_linear = nn.Linear(hidden_dim, in_dim)
        self.condition_dim = condition_dim
        self.use_ff_adapter = use_ff_adapter
        self.single_frame_num = 30*45
        if condition_dim is not None:
            self.condition_linear = nn.Linear(condition_dim, in_dim)
            self._need_condition = True
        else:
            self._need_condition = False

        init.zeros_(self.up_linear.weight)
        init.zeros_(self.up_linear.bias)

    def forward(self, hidden_states, encoder_hidden_states=None, condition=None, condition_lam=1, text_seq_length=226):
        if self.use_ff_adapter: 
            encoder_hidden_states = hidden_states[:, :text_seq_length]
            hidden_states = hidden_states[:, text_seq_length:]
        # cat video and text
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        hidden_states_in = hidden_states
        if self._need_condition and condition is not None:
            hidden_states = hidden_states + condition_lam * self.condition_linear(condition)
        hidden_states = self.down_linear(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.up_linear(hidden_states)
        hidden_states += hidden_states_in

        # split video and text
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        if self.use_ff_adapter:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            return hidden_states
        return hidden_states, encoder_hidden_states

@maybe_allow_in_graph
class CogVideoXBlockWithAdapter(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        # flag init
        use_IdAdapter: bool = False,
        use_MotionAdapter: bool = False,
        # id adapter
        spatial_adapter_alpha: float = 0.1,
        spatial_adapter_hidden_dim: int = 64,
        # motion adapter
        temporal_adapter_alpha: float = 0.1,
        temporal_adapter_hidden_dim: int = 64,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        self.time_embed_dim = time_embed_dim
        self.dim = dim
        # adapter
        self.use_id_adapter = use_IdAdapter
        self.use_motion_adapter = use_MotionAdapter
        self.id_adapter_alpha = spatial_adapter_alpha
        self.id_hidden_dim = dim//2 if not spatial_adapter_hidden_dim else spatial_adapter_hidden_dim
        self.motion_adapter_alpha = temporal_adapter_alpha
        self.motion_hidden_dim = dim//2 if not temporal_adapter_hidden_dim else temporal_adapter_hidden_dim

    def set_id_adapter(self, adapter_condition_dim=None, use_hydra=False):
        self.id_attn_adapter = Adapter(self.dim, self.id_hidden_dim, condition_dim=adapter_condition_dim)
        self.id_ff_adapter = Adapter(self.dim, self.id_hidden_dim, condition_dim=adapter_condition_dim, use_ff_adapter=True)

    def set_motion_adapter(self, adapter_condition_dim=None):
        self.motion_attn_adapter = Adapter(self.dim, self.motion_hidden_dim, condition_dim=adapter_condition_dim)
        self.motion_ff_adapter = Adapter(self.dim, self.motion_hidden_dim, condition_dim=adapter_condition_dim, use_ff_adapter=True)

    def weights_compute(self, weights, dtype, device):
        if isinstance(weights, torch.Tensor):
            weights = weights.to(device=device, dtype=dtype)
            weights = weights.view(-1, 1, 1)
        else:
            weights = torch.tensor(weights, dtype=dtype).view(-1, 1, 1)
            weights = weights.to(device)
        return weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        dynamic_alpha_beta: list[float] = None,
        adapter_condition: Optional[torch.Tensor] = None,
        adapter_condition_lam: float = 1.0,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        id_flag = False
        motion_flag = False
        # attn adapter
        if self.use_id_adapter:
            id_flag = True
            id_adapter_hidden_states, id_adapter_encoder_hidden_states = self.id_attn_adapter(
                hidden_states=norm_hidden_states, 
                encoder_hidden_states=norm_encoder_hidden_states,
                condition=adapter_condition, 
                condition_lam=adapter_condition_lam,
            )
            if dynamic_alpha_beta: # from controller
                id_weight = self.weights_compute(dynamic_alpha_beta[0], dtype=attn_hidden_states.dtype, device=attn_hidden_states.device)
            else:
                id_weight = self.id_adapter_alpha 
        else:
            id_weight, id_adapter_hidden_states, id_adapter_encoder_hidden_states = 0.0, 0.0, 0.0

        if self.use_motion_adapter:
            motion_flag = True
            motion_adapter_hidden_states, motion_adapter_encoder_hidden_states = self.motion_attn_adapter(
                hidden_states=norm_hidden_states, 
                encoder_hidden_states=norm_encoder_hidden_states, 
                condition=adapter_condition, 
                condition_lam=adapter_condition_lam,
            )
            if dynamic_alpha_beta:
                motion_weight = self.weights_compute(dynamic_alpha_beta[1], dtype=attn_hidden_states.dtype, device=attn_hidden_states.device)
            else:
                motion_weight = self.motion_adapter_alpha
        else:
            motion_weight, motion_adapter_hidden_states, motion_adapter_encoder_hidden_states = 0.0, 0.0, 0.0

        # add hidden_states separately 
        if id_flag:
            id_adapter_hidden_states = id_weight * id_adapter_hidden_states
            id_adapter_encoder_hidden_states = id_weight * id_adapter_encoder_hidden_states
        
        if motion_flag:
            motion_adapter_hidden_states = motion_weight * motion_adapter_hidden_states
            motion_adapter_encoder_hidden_states = motion_weight * motion_adapter_encoder_hidden_states

        # add hidden_states separately
        attn_hidden_states = attn_hidden_states + id_adapter_hidden_states + motion_adapter_hidden_states
        attn_encoder_hidden_states = attn_encoder_hidden_states + id_adapter_encoder_hidden_states + motion_adapter_encoder_hidden_states

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        id_flag = False
        motion_flag = False
        # ff adapter
        if self.use_id_adapter:
            id_flag = True
            id_adapter_output = self.id_ff_adapter(
                hidden_states=norm_hidden_states,
                condition=adapter_condition,
                condition_lam=adapter_condition_lam,
            )
        else: 
            id_adapter_output = 0.0

        if self.use_motion_adapter:
            motion_flag = True
            motion_adapter_output = self.motion_ff_adapter(
                hidden_states=norm_hidden_states,
                condition=adapter_condition,
                condition_lam=adapter_condition_lam,
            )
        else:
            motion_adapter_output = 0.0

        # add hidden_states separately
        if id_flag:
            id_adapter_output = id_weight * id_adapter_output
                
        if motion_flag:
            motion_adapter_output = motion_weight * motion_adapter_output
        
        # add hidden_states separately
        ff_output = ff_output + id_adapter_output + motion_adapter_output

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states

class Controller(nn.Module): 
    def __init__(
        self, 
        timestep_dim=512, 
        channel_dim=3072, 
        hidden_dim=128, 
        output_dim=1,
        elementwise_affine=True,
        eps=1e-5,
        approximate="tanh",
        bias=True,
    ):
        super().__init__()
        self.approximate = approximate
        # temb
        self.timestep_encoder = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_dim, 3 * timestep_dim, bias=bias),
        )
        # hidden_states
        self.channel_encoder = nn.Sequential(
            nn.Linear(channel_dim, timestep_dim, bias=bias),
        )
        self.norm = nn.LayerNorm(timestep_dim, eps=eps, elementwise_affine=elementwise_affine)

        self.fuse_proj = nn.Linear(timestep_dim, timestep_dim*4, bias=bias)
        self.fuse_proj_down = nn.Linear(timestep_dim*4, timestep_dim, bias=bias)

        self.fusion_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(timestep_dim, hidden_dim, bias=bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim, bias=bias),
            nn.Sigmoid()
        )
        self.scale = 0.2

        init.zeros_(self.fusion_net[3].weight)
        init.zeros_(self.fusion_net[3].bias)

    def gelu(self, gate):
        if gate.device.type != "mps":
            return F.gelu(gate, approximate=self.approximate)
        return F.gelu(gate.to(dtype=torch.float32), approximate=self.approximate).to(dtype=gate.dtype)

    def forward(self, timestep, hidden_states):
        shift, scale, gate = self.timestep_encoder(timestep).chunk(3, dim=1)

        pooled_hidden = hidden_states.max(dim=1)[0]  # [B, 3072] 
        c_embed = self.channel_encoder(pooled_hidden)  # [B, 512]

        # fuse
        fuse_feature = self.norm(c_embed) * (1 + scale) + shift # [B, 512]
        fuse_feature = self.fuse_proj(fuse_feature)  # [B, 2048]
        fuse_feature = self.gelu(fuse_feature)
        fuse_feature = self.fuse_proj_down(fuse_feature) + gate * c_embed # [B, 512]

        weight = self.fusion_net(fuse_feature) * self.scale
        return weight
       
class CogVideoXTransformer3DModelWithAdapter(ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).

    Parameters:
        num_attention_heads (`int`, defaults to `30`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `16`):
            The number of channels in the output.
        flip_sin_to_cos (`bool`, defaults to `True`):
            Whether to flip the sin to cos in the time embedding.
        time_embed_dim (`int`, defaults to `512`):
            Output dimension of timestep embeddings.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        num_layers (`int`, defaults to `30`):
            The number of layers of Transformer blocks to use.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        attention_bias (`bool`, defaults to `True`):
            Whether or not to use bias in the attention projection layers.
        sample_width (`int`, defaults to `90`):
            The width of the input latents.
        sample_height (`int`, defaults to `60`):
            The height of the input latents.
        sample_frames (`int`, defaults to `49`):
            The number of frames in the input latents. Note that this parameter was incorrectly initialized to 49
            instead of 13 because CogVideoX processed 13 latent frames at once in its default and recommended settings,
            but cannot be changed to the correct value to ensure backwards compatibility. To create a transformer with
            K latent frames, the correct value to pass here would be: ((K - 1) * temporal_compression_ratio + 1).
        patch_size (`int`, defaults to `2`):
            The size of the patches to use in the patch embedding layer.
        temporal_compression_ratio (`int`, defaults to `4`):
            The compression ratio across the temporal dimension. See documentation for `sample_frames`.
        max_text_seq_length (`int`, defaults to `226`):
            The maximum sequence length of the input text embeddings.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to use in feed-forward.
        timestep_activation_fn (`str`, defaults to `"silu"`):
            Activation function to use when generating the timestep embeddings.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether or not to use elementwise affine in normalization layers.
        norm_eps (`float`, defaults to `1e-5`):
            The epsilon value to use in normalization layers.
        spatial_interpolation_scale (`float`, defaults to `1.875`):
            Scaling factor to apply in 3D positional embeddings across spatial dimensions.
        temporal_interpolation_scale (`float`, defaults to `1.0`):
            Scaling factor to apply in 3D positional embeddings across temporal dimensions.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 48,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 42,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = False,
        # adapter
        use_IdAdapter: bool = False,
        use_MotionAdapter: bool = False,
        spatial_adapter_alpha: float = 0.1,
        spatial_adapter_hidden_dim: int = 64,
        temporal_adapter_alpha: float = 0.1,
        temporal_adapter_hidden_dim: int = 64,
    ):
        super().__init__()
        colorama.init(autoreset=True)
        highlight_color = colorama.Fore.CYAN
        self._controller = False

        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=True,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn) # [B, 512]
        
        # 3. Define spatio-temporal transformers blocks
        adapter_params = None
        if use_IdAdapter and use_MotionAdapter:
            print(highlight_color+"Using both IdAdapter and MotionAdapter....")
            adapter_params = {
                "spatial_adapter_alpha": spatial_adapter_alpha,
                "spatial_adapter_hidden_dim": spatial_adapter_hidden_dim,
                "temporal_adapter_alpha": temporal_adapter_alpha,
                "temporal_adapter_hidden_dim": temporal_adapter_hidden_dim,
            }
        elif use_IdAdapter and not use_MotionAdapter:
            print(highlight_color+"Only using IdAdapter....")
            adapter_params = {
                "spatial_adapter_alpha": spatial_adapter_alpha,
                "spatial_adapter_hidden_dim": spatial_adapter_hidden_dim,
            }
        elif not use_IdAdapter and use_MotionAdapter:
            print(highlight_color+"Only using MotionAdapter....")
            adapter_params = {
                "temporal_adapter_alpha": temporal_adapter_alpha,
                "temporal_adapter_hidden_dim": temporal_adapter_hidden_dim,
            }
        if adapter_params:
            for name, value in adapter_params.items():
                print(f"{highlight_color}{name}:{value}")

        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlockWithAdapter(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    # flag init
                    use_IdAdapter=use_IdAdapter,
                    use_MotionAdapter=use_MotionAdapter,
                    # id adapter
                    spatial_adapter_alpha=spatial_adapter_alpha,
                    spatial_adapter_hidden_dim=spatial_adapter_hidden_dim,
                    # motion adapter
                    temporal_adapter_alpha=temporal_adapter_alpha,
                    temporal_adapter_hidden_dim=temporal_adapter_hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )
        self.proj_out = nn.Linear(inner_dim, patch_size * patch_size * out_channels)

        self.gradient_checkpointing = False

        self.inner_dim=inner_dim
        self.time_embed_dim=time_embed_dim

    def set_id_adapter(self, conditioning_dim=None, hydra_part_num=None):
        for block in self.transformer_blocks:
            block.set_id_adapter(adapter_condition_dim=conditioning_dim)
    
    def set_motion_adapter(self, conditioning_dim=None):
        for block in self.transformer_blocks:
            block.set_motion_adapter(adapter_condition_dim=conditioning_dim)

    def set_controller(self, output_dim=1):
        self.controller = Controller(timestep_dim=self.time_embed_dim, channel_dim=self.inner_dim, output_dim=output_dim) 
        self._controller = True

    def _set_gradient_checkpointing(self, gradient_checkpointing_func, enable=False):
        self.gradient_checkpointing = enable
    
    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedCogVideoXAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def get_dynamic_weight(self, alpha, beta, block_num, length_model):
        n = alpha.shape[1]
        assert length_model % n == 0, "n must be divisible by length_model"
        interval = length_model // n
        m = block_num // interval
        assert m < n, "block_num out of range"
        alpha_m = alpha[:, m:m+1]
        beta_m = beta[:, m:m+1]
        return [alpha_m, beta_m]

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Union[int, float, torch.LongTensor],
        timestep_cond: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        y_image: Optional[torch.Tensor] = None,
    ):  
        
        highlight_color = colorama.Fore.RED
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        dynamic_alpha_beta = []

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        context_image = y_image
        if context_image is not None and context_image.shape[1] == 1:
            context_image = context_image.repeat_interleave(repeats=hidden_states.size(1)+encoder_hidden_states.size(1),dim=1)

        if self._controller:
            cat_video_txt_features = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            alpha = self.controller(timestep=emb, hidden_states=cat_video_txt_features)
            beta = self.controller.scale - alpha #[B, n]

        # 3. Transformer blocks
        for i, block in enumerate(self.transformer_blocks):
            if self._controller:
                dynamic_alpha_beta = self.get_dynamic_weight(alpha=alpha, beta=beta, block_num=i, length_model=len(self.transformer_blocks))

            if self.training and self.gradient_checkpointing:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint( 
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    emb,
                    image_rotary_emb,
                    dynamic_alpha_beta,
                    context_image,
                    **ckpt_kwargs,
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                    dynamic_alpha_beta=dynamic_alpha_beta,
                    adapter_condition=context_image,  
                )

        if not self.config.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        # Note: we use `-1` instead of `channels`:
        #   - It is okay to `channels` use for CogVideoX-2b and CogVideoX-5b (number of input channels is equal to output channels)
        #   - However, for CogVideoX-5b-I2V also takes concatenated input image latents (number of input channels is twice the output channels)
        p = self.config.patch_size
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)