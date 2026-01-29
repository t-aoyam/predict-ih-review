import os
os.environ['HF_HOME'] = '../data/hf_models'


from collections.abc import Callable
from typing import Optional, Union

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig




from transformers import GPTNeoXModel, GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer, GPTNeoXAttention, apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast)

from transformers.utils import logging
import torch
from torch import nn, cuda
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union, List
import einops

device = 'cuda:0' if cuda.is_available() else 'cpu'

logger = logging.get_logger(__name__)

class HookedGPTNeoXAttention(GPTNeoXAttention):
    def __init__(self, config, layer_idx: int,
                 ablation_head_idx: dict = None,
                 hook: bool = True):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.ablation_head_idx = ablation_head_idx  # {layer: [head, head, ...], layer: [head, head, ...]
        self.hook = hook
        self.config = config

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        layer_past: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, 3 * self.head_size)

        qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states, value_states = qkv.chunk(3, dim=-1)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Cache QKV values
        if layer_past is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_ndims,
                "cache_position": cache_position,
            }
            key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # Compute attention
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs,
        )

        if self.hook:
            # attn_output is (batch_size, num_heads, seq_len, head_dim)
            # W^O.shape is (hidden_size, hidden_size)
            W_O = self.dense.weight.t()  # (hidden, hidden) = (out, in) so transpose -> (in, out)
            num_heads = self.config.num_attention_heads
            head_dim = W_O.shape[1] // num_heads
            hidden_size = W_O.shape[1]

            # reshape W_O to per-head slices
            W_O_heads = W_O.view(num_heads, head_dim, hidden_size)  # (num_heads, head_dim, hidden_size)
            # print(f"W_O_heads shape: {W_O_heads.shape}")
            # print(f"attn_output shape: {attn_output.shape}")

            # batch matmul
            # attn_output: (batch, seq_len, num_heads, head_dim)
            # W_O_heads:   (num_heads, head_dim, hidden_size)
            # broadcast across batch and seq_len
            # print(attn_output.shape)
            # print(W_O_heads.shape)

            # per_head_output = torch.matmul(
            #     attn_output,  # (batch, num_heads, seq_len, head_dim)
            #     W_O_heads  # (num_heads, head_dim, hidden_size)
            # )

            # per_head_output = torch.matmul(
            #     attn_output.permute(0, 2, 1, 3),  # (batch, num_heads, seq_len, head_dim)
            #     W_O_heads  # (num_heads, head_dim, hidden_size)
            # )

            per_head_output = einops.einsum(
                attn_output, W_O_heads,
                'batch seq_len num_heads head_dim, num_heads head_dim hidden_size -> batch num_heads seq_len hidden_size'
            )
            # Output: (batch, num_heads, seq_len, hidden_size)
            self.per_head_output = per_head_output

        # Reshape outputs and final projection
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.dense(attn_output)

        return attn_output, attn_weights


class HookedGPTNeoXLayer(GPTNeoXLayer):
    def __init__(self,config, layer_idx = None,
                 custom_attention_class = HookedGPTNeoXAttention,
                 ablation_head_idx: dict = None,
                 hook: bool = True):
        super().__init__(config, layer_idx)
        self.attention = custom_attention_class(config, layer_idx,
                                                ablation_head_idx=ablation_head_idx,
                                                hook=True)
        self.hook = hook
        self.per_head_output = None

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        attn_output, attn_weights = self.attention(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            layer_past=layer_past,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        attn_output = self.post_attention_dropout(attn_output)

        if self.hook:
            self.per_head_output = self.attention.per_head_output

        if self.use_parallel_residual:
            # pseudocode:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_output = self.mlp(self.post_attention_layernorm(hidden_states))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output + hidden_states
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))
            attn_output = attn_output + hidden_states
            mlp_output = self.mlp(self.post_attention_layernorm(attn_output))
            mlp_output = self.post_mlp_dropout(mlp_output)
            hidden_states = mlp_output + attn_output

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class HookedGPTNeoXModel(GPTNeoXModel):
    def __init__(self, config,
                 ablation_head_idx: dict = None,
                 hook: bool=False):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                HookedGPTNeoXLayer(config,
                                   layer_idx=i,
                                   ablation_head_idx=ablation_head_idx
                                   )\
                                    for i in range(config.num_hidden_layers)
            ]
        )
        self.hook = hook


class HookedGPTNeoXForCausalLM(GPTNeoXForCausalLM):

    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config,
                 ablation_head_idx: dict = None,
                 hook: bool = True):
        super().__init__(config)
        self.hook = hook
        if self.hook:
            self.set_attn_implementation('eager')
        self.gpt_neox = HookedGPTNeoXModel(config, ablation_head_idx=ablation_head_idx, hook=hook)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forced_attention: Optional[torch.Tensor] = None,
        do_ablation: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            forced_attention=forced_attention,
            do_ablation=do_ablation,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


    def forward_plus(self, input_ids, ablation_mode=None, **kwargs):
        if not ablation_mode:
            outputs = self.forward(input_ids, forced_attention=None,
                                   do_ablation=False,
                                   **kwargs)
        elif ablation_mode == 'full':  # full ablation
            outputs = self.forward(input_ids, forced_attention=None,
                                   do_ablation=True,
                                   **kwargs)
        elif ablation_mode == 'pp':  # pattern-preserving ablation
            first_pass = self.forward(input_ids, forced_attention=None,
                                      do_ablation=False,
                                      output_attentions=True,
                                      **kwargs)
            outputs = self.forward(input_ids, forced_attention=first_pass.attentions,
                                   do_ablation=True,
                                   **kwargs)
        else:
            raise IOError('ablation_mode has to be chosen from [None, "full", "pp"]')
        return outputs

    def get_logit_attribution(self, input_ids, mean_normalize=True, relu=True, ratio=True):
        """
        src is the position whose logit contributions we are measuring (the second A in AB...AB)
        dst is the token ID whose logit increase we are measuring (the B token in AB...AB)
        
        :param mean_normalize: subtract the mean logit contribution across vocab *in the sample* from each logit contribution
        :param relu: pass the logit contributions through ReLU to zero-out negative contributions
        :param ratio: take the ratio of target logit contribution to total logit contribution
        """
        # input_ids = self.transformer.input_ids # (bs, seq_len)
        bs, seq_len = input_ids.shape
        mid = seq_len // 2
        
        # check if the input is a proper synthetic induction head input (i.e., repetition of the same sequence)
        if not torch.all(input_ids[:, :mid] == input_ids[:, mid:]):
            print("Warning: Some batches are not repetitions.")

        # 1. Get the tokens for the denominator (all tokens in the sample)
        # this assumes that the input sequence is a unique and random sequence repeated twice
        sample_ids = input_ids[:, :mid] # (bs, all_unique_ids = seqlen)
        sample_unembed = self.embed_out.weight.t()[:, sample_ids] # (d, bs, dst_ids) -> these are token IDs and not positions

        # 2. Get the target IDs (what we WANT to predict at each step)
        # For ABCDABCD, at second half positions [4,5,6,7],
        # targets are [1,2,3,4] (Indices of B,C,D,A)
        target_ids = input_ids[:, 1:mid+1] # This handles the wrap-around if mid+1 is used

        logit_attribution_scores = []

        for i in range(self.config.num_hidden_layers):
            # Head output at the destination (the second half)
            # Shape: (batch, head, src_positions, dim)
            head_out = self.gpt_neox.layers[i].per_head_output[:, :, mid:, :]

            # --- DENOMINATOR MATH ---
            # Boost to every token in the sample (bs, head, mid, mid)
            logits_all_in_sample = einops.einsum(
                head_out, sample_unembed,
                'batch head src_positions dim, dim batch dst_ids -> batch head src_positions dst_ids'
            )

            # Center and ReLU
            if mean_normalize:
                sample_mean = logits_all_in_sample.mean(dim=-1, keepdim=True)
            if relu:
                logits_relu_all = torch.relu(logits_all_in_sample - sample_mean)
                denominator = logits_relu_all.sum(dim=-1)

            # --- NUMERATOR MATH (The Target B) ---
            # Unembed the specific targets for each position
            target_unembed = self.embed_out.weight.t()[:, target_ids] # (dim, bs, src_positions) (last dimension is dst_id for each of the src_positions)

            # (bs, head, mid) - Each pos gets its own specific target boost
            target_logits = einops.einsum(
                head_out, target_unembed,
                'batch head src_positions dim, dim batch src_positions -> batch head src_positions'
            )

            # Center using the same mean and ReLU
            if mean_normalize:
                target_boost = target_logits - sample_mean.squeeze(-1)
            if relu:
                target_boost = torch.relu(target_boost)

            # --- FINAL RATIO ---
            ratio = target_boost / denominator
            logit_attribution_scores.append(ratio)

        logit_attribution_scores = torch.stack(logit_attribution_scores, dim=1) # (bs, layer, head, src_positions)
        return logit_attribution_scores