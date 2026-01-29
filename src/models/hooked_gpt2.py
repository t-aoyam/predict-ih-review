from transformers import AutoModelForCausalLM, GPTNeoXForCausalLM, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2Block, GPT2Model, GPT2LMHeadModel
from transformers.modeling_outputs import (BaseModelOutputWithPastAndCrossAttentions,
                                           CausalLMOutputWithCrossAttentions)
# from transformers.modeling_attn_mask_utils import (_prepare_4d_attention_mask_for_sdpa,
#                                                    _prepare_4d_causal_attention_mask_for_sdpa)
from transformers.utils import logging
import torch
from torch import nn
from torch import cuda
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union, List
import einops

device = 'cuda:0' if cuda.is_available() else 'cpu'

logger = logging.get_logger(__name__)

class HookedGPT2Attention(GPT2Attention):
    def __init__(self,
                 config,
                 layer_idx: int,
                 ablation_head_idx: dict = None,
                 hook: bool = False):
        super().__init__(config)
        self.config = config
        self.layer_idx = layer_idx
        self.ablation_head_idx = ablation_head_idx  # {layer: [head, head, ...], layer: [head, head, ...]
        self.hook = hook
        self.per_head_output = None  # if hook, QKV @ W_O result per head will be stored here

    def _attn(self, query, key, value, attention_mask=None, head_mask=None,
              forced_attention=None, do_ablation=False):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / torch.full(
                [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
            )

        # Layer-wise attention scaling
        if self.scale_attn_by_inverse_layer_idx:
            attn_weights = attn_weights / float(self.layer_idx + 1)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
            # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
            mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = torch.where(causal_mask.bool(), attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if forced_attention:  # for pattern-preserving ablation
            attn_weights = forced_attention[self.layer_idx]

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)

        if do_ablation and self.layer_idx in self.ablation_head_idx:
            for head_idx in self.ablation_head_idx[self.layer_idx]:
                attn_output[:, head_idx, :, :] = 0  # zero-out all attention output of induction heads

        return attn_output, attn_weights

    # not supported in newer transformers versions so we redefine it here
    def _split_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = tensor.size()[:-1] + (num_attention_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        if len(tensor.shape) == 5:
            return tensor.permute(0, 1, 3, 2, 4)  # (batch, blocks, head, block_length, head_features)
        elif len(tensor.shape) == 4:
            return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)


    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        forced_attention: Optional[torch.Tensor] = None,
        do_ablation: Optional[bool] = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )

            query = self.q_attn(hidden_states)
            key, value = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        if self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(query, key, value, attention_mask, head_mask)
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask,
                                                   forced_attention, do_ablation)

        if self.hook:
            # attn_output is (batch_size, num_heads, seq_len, head_dim)
            # W^O.shape is (hidden_size, hidden_size)
            W_O = self.c_proj.weight
            num_heads = self.config.n_head
            head_dim = W_O.shape[1] // num_heads
            hidden_size = W_O.shape[1]

            # reshape W_O to per-head slices
            W_O_heads = W_O.view(num_heads, head_dim, hidden_size)  # (num_heads, head_dim, hidden_size)

            # batch matmul
            # attn_output: (batch, num_heads, seq_len, head_dim)
            # W_O_heads:   (num_heads, head_dim, hidden_size)
            # broadcast across batch and seq_len
            # print(attn_output.shape)
            # print(W_O_heads.shape)
            per_head_output = torch.matmul(
                attn_output,  # (batch, num_heads, seq_len, head_dim)
                W_O_heads  # (num_heads, head_dim, hidden_size)
            )
            # Output: (batch, num_heads, seq_len, hidden_size)
            self.per_head_output = per_head_output

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

class HookedGPT2Block(GPT2Block):
    def __init__(self,
                 config,
                 layer_idx=None,
                 custom_attention_class=HookedGPT2Attention,
                 ablation_head_idx: dict=None,
                 hook=False):
        super().__init__(config, layer_idx)
        self.attn = custom_attention_class(
            config=config,
            layer_idx=layer_idx,
            ablation_head_idx=ablation_head_idx,
            hook=True
        )
        self.layer_idx = layer_idx
        self.ablation_head_idx = ablation_head_idx
        if config.add_cross_attention:
            self.crossattention = custom_attention_class(
                config=config,
                is_cross_attention=True,
                layer_idx=layer_idx,
                hook=hook
            )
        self.hook = hook
        self.per_head_output = None

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        forced_attention: Optional[torch.Tensor] = None,
        do_ablation: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            forced_attention=forced_attention,
            do_ablation=do_ablation,
        )
        if self.hook:
            self.per_head_output = self.attn.per_head_output
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)

class HookedGPT2Model(GPT2Model):
    def __init__(
            self,
            config,
            ablation_head_idx: dict = None,
            hook=False
    ):
        super().__init__(config)
        self._attn_implementation = None  # we are using custom attention
        self.h = nn.ModuleList([
            HookedGPT2Block(
                config,
                layer_idx=i,
                ablation_head_idx=ablation_head_idx,
                hook=hook
            ) for i in range(config.num_hidden_layers)
        ])

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forced_attention: Optional[torch.Tensor] = None,
        do_ablation: bool = False,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            # self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        # Attention mask.
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        attention_mask = attention_mask.view(batch_size, -1) if attention_mask is not None else None
        if self._attn_implementation == "flash_attention_2":
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        # elif _use_sdpa:
        #     attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
        #         attention_mask=attention_mask,
        #         input_shape=(batch_size, input_shape[-1]),
        #         inputs_embeds=inputs_embeds,
        #         past_key_values_length=past_length,
        #     )
        else:
            if attention_mask is not None:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            # if _use_sdpa:
            #     encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
            #         mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
            #     )
            if not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    forced_attention=forced_attention,
                    do_ablation=do_ablation,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class HookedGPT2LMHeadModel(GPT2LMHeadModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self,
                 config,
                 ablation_head_idx: dict = None,
                 hook=True):
        super().__init__(config)
        self.transformer = HookedGPT2Model(
            config,
            ablation_head_idx=ablation_head_idx,
            hook=hook
        )

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
        sample_unembed = self.transformer.wte.weight.t()[:, sample_ids] # (d, bs, dst_ids) -> these are token IDs and not positions

        # 2. Get the target IDs (what we WANT to predict at each step)
        # For ABCDABCD, at second half positions [4,5,6,7],
        # targets are [1,2,3,4] (Indices of B,C,D,A)
        target_ids = input_ids[:, 1:mid+1] # This handles the wrap-around if mid+1 is used

        logit_attribution_scores = []

        for i in range(self.config.n_layer):
            # Head output at the destination (the second half)
            # Shape: (batch, head, src_positions, dim)
            head_out = self.transformer.h[i].per_head_output[:, :, mid:, :]

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
            target_unembed = self.transformer.wte.weight.t()[:, target_ids] # (dim, bs, src_positions) (last dimension is dst_id for each of the src_positions)

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


    def _get_logit_attribution(self, mean_normalize=True, relu=True, ratio=False):
        # create warning if input sequence is not a repetition of the same sequence
        # just test the first (self.input_ids should be (batch, seq_len))
        input_ids = self.transformer.input_ids
        bs, seq_len = input_ids.shape
        mid = seq_len // 2
        if not torch.all(input_ids[:, :mid] == input_ids[:, mid:]):
            print("Warning: Some batches are not repetitions.")        
        # given ABCDABCD, then at positions in the second half (ABCD),
        # induction heads should promote BCDA, respectively.
        # so the target ids sohuld be input_ids[1:mid+1]
        target_ids = input_ids[:, 1:mid+1] # (bs, seqlen/2)
        unembed = self.transformer.wte.weight.t()[:, target_ids]  # (hidden_size, b, seqlen/2)
        logit_attribution = []
        for i in range(self.config.n_layer):
            per_layer_logit_attribution = einops.einsum(
                self.transformer.h[i].per_head_output[:, :, mid:, :],  # (batch, head, seqlen/2, hidden_size
                unembed,  # hidden_size, b, seqhalf,
                'b h s hidden , hidden b s -> b h s'
            )
            del self.transformer.h[i].per_head_output  # free up memory
            logit_attribution.append(per_layer_logit_attribution)
        logit_attribution = torch.stack(logit_attribution, dim=1)  # (batch, layer, head, seq_len/2)

        """Old Implementation: Keep for reference

            per_layer_logit_attribution = torch.matmul(
                self.transformer.h[i].per_head_output[:, :, mid:, :],  # (batch, head, seqhalf, hidden_size
                unembed  # hidden_size, b, seqhalf
                )

        logit_attribution = torch.stack(
            [self.transformer.h[i].per_head_output for i in range(self.config.n_layer)],
            dim=1
        )

        # Shape: (batch, len(intermediate_outputs), num_heads, seq_len, hidden_size)
        logit_attribution = torch.matmul(
            logit_attribution,  # (batch, layer, head, seq_len, hidden_size
            self.lm_head.weight.t  # (hidden_size, vocab_size)
        )
        """

        if mean_normalize:
            logit_attribution -= torch.mean(logit_attribution, dim=4, keepdim=True)
        if relu:
            logit_attribution = torch.relu(logit_attribution)
        if ratio:
            logit_attribution /= torch.sum(logit_attribution, dim=4, keepdim=True)

        return logit_attribution  # (batch, layer, head, seq_len, vocab_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forced_attention: Optional[torch.Tensor] = None,
        do_ablation: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            forced_attention=forced_attention,
            do_ablation=do_ablation
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
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


#
# def test():
#     num_batches = 10
#     seq_len = 50
#     rep = 2
#     model = HookedGPT2LMHeadModel.from_pretrained('gpt2', ablation_head_idx=None, hook=True)
#     # GPT2Config.from_pretrained('gpt2').model_parallel
#
#     size = (num_batches, seq_len)
#     input_tensor = torch.randint(0, 50000, size)
#     # input_tensor.shape
#     # random_tokens = input_tensor.to(model.cfg.device)
#     repeated_tokens = einops.repeat(input_tensor, f"batch seq_len -> batch ({rep} seq_len)")
#     model.eval()
#     with torch.no_grad():
#         # outputs = model.forward_plus(repeated_tokens, ablation_mode=None)
#         _ = model(repeated_tokens)
#         attrs = model.get_logit_attribution()
#     return attrs
#
# attrs = test()