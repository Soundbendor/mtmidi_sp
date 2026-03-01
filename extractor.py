import sys,os,time,argparse,copy,types
from torch import nn
import util.util_main as UMN
import util.util_constants as UC
import util.util_hf as UHF
from dataclasses import dataclass
import librosa as lr
from librosa import feature as lrf
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from typing import TYPE_CHECKING, Any, Optional, Union
import numpy as np
import torch
import random
from distutils.util import strtobool
from transformers.cache_utils import Cache, DynamicCache,EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutput, Seq2SeqLMOutput, ModelOutput

dur = 4.0
# need to override forward method of MusicgenDecoderLayer
# https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/musicgen/modeling_musicgen.py#L304
# change outputs so need to override forward of MusicgenDecoder
# https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/musicgen/modeling_musicgen.py#L437
# which then overrides forward of MusicgenModel
# https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/musicgen/modeling_musicgen.py#L693

# override at instance level
# https://stackoverflow.com/questions/394770/override-a-method-at-instance-level

# forward method of MusicgenDecoderLayer
# https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/musicgen/modeling_musicgen.py#L338

#MusicgenDecoderLayer
# https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/musicgen/modeling_musicgen.py#L304

# MusicgenDecoder can be passed a config argt in its constructor of class MusicgenDecoderConfig defined:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/musicgen/configuration_musicgen.py#L25

# MusicgenDecoderConfig inherits from PreTrainedConfig:
# https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L362


# define a new class based off BaseModelOutputWithPastAndCrossAttentions 
# https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L238
@dataclass
class BaseModelOutputWithPostActivations(ModelOutput):
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.

            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    post_activations: tuple[torch.FloatTensor, ...] | None = None


# define a new class based off Seq2SeqLMOutput
# https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_outputs.py#L799

@dataclass
class Seq2SeqLMOutputWithPostActivations(ModelOutput):
    """
    Base class for sequence-to-sequence language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.EncoderDecoderCache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: EncoderDecoderCache | None = None
    decoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    decoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    encoder_last_hidden_state: torch.FloatTensor | None = None
    encoder_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    encoder_attentions: tuple[torch.FloatTensor, ...] | None = None
    decoder_post_activations: tuple[torch.FloatTensor, ...] | None = None

# new output for MusicgenforCausalLM
# based off of CausalLMOutputWithCrossAttentions
#https://github.com/huggingface/transformers/blob/393b4b3d28e29b4b05b19b4b7f3242a7fc893637/src/transformers/modeling_outputs.py#L693
@dataclass
class CausalLMOutputWithPostActivations(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Cross attentions weights after the attention softmax, used to compute the weighted average in the
            cross-attention heads.
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
    """

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    cross_attentions: tuple[torch.FloatTensor, ...] | None = None
    post_activations: tuple[torch.FloatTensor, ...] | None = None
    


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/musicgen/modeling_musicgen.py
def forward_musicgendecoderlayer(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            past_key_values (`Cache`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        # Cross-Attention Block
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # ===== MY CHANGE: taking representations post activation ======
        post_activation = hidden_states.clone().detach()
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # ====== MY CHANGE: append post_activation representations ===== 
        outputs += (post_activation,)

        return outputs

# MusicgenDecoder
# https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/musicgen/modeling_musicgen.py#L472
# ===== MY CHANGE: changed return to include post_activations ====== 
def forward_musicgendecoder(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPostActivations]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size * num_codebooks, sequence_length)`):
            Indices of input sequence tokens in the vocabulary, corresponding to the sequence of audio codes.

            Indices can be obtained by encoding an audio prompt with an audio encoder model to predict audio codes,
            such as with the [`EncodecModel`]. See [`EncodecModel.encode`] for details.

            [What are input IDs?](../glossary#input-ids)

            <Tip warning={true}>

            The `input_ids` will automatically be converted from shape `(batch_size * num_codebooks,
            target_sequence_length)` to `(batch_size, num_codebooks, target_sequence_length)` in the forward pass. If
            you obtain audio codes from an audio encoding model, such as [`EncodecModel`], ensure that the number of
            frames is equal to 1, and that you reshape the audio codes from `(frames, batch_size, num_codebooks,
            target_sequence_length)` to `(batch_size * num_codebooks, target_sequence_length)` prior to passing them as
            `input_ids`.

            </Tip>
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of
            the decoder.
        encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
            Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
            selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        """

        # MY NOTES: output_attentions is a method that returns a bool defined in PreTrainedConfig
        # https://github.com/huggingface/transformers/blob/main/src/transformers/configuration_utils.py#L362
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            # (bsz * codebooks, seq_len) -> (bsz, codebooks, seq_len)
            input = input_ids.reshape(-1, self.num_codebooks, input_ids.shape[-1])
            bsz, num_codebooks, seq_len = input.shape
            input_shape = (bsz, seq_len)
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1:]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing`. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(config=self.config), DynamicCache(config=self.config))

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = sum(self.embed_tokens[codebook](input[:, codebook]) for codebook in range(num_codebooks))

        attention_mask = self._update_causal_mask(
            attention_mask,
            input_shape,
            inputs_embeds,
            past_key_values_length,
        )
        encoder_attention_mask = self._update_cross_attn_mask(
            encoder_hidden_states,
            encoder_attention_mask,
            input_shape,
            inputs_embeds,
        )
        
        # MY NOTES: adding the initial embeddings here
        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

        # ===== MY ADDITION: adding a place to accumulate post_activations =====
        all_post_activations = ()

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://huggingface.co/papers/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]

            # ====== MY ADDITION: here is where I accumulate post activations ===== 
            cur_post_activations = layer_outputs[-1]
            all_post_activations += (cur_post_activations,)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )

        # ===== MY CHANGE: Change to new custom class ======= 
        return BaseModelOutputWithPostActivations(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
            post_activations=all_post_activations,
        )

# MusicgenModel
# https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/musicgen/modeling_musicgen.py#L707
# decoder is musicgendecoder
# ===== MY CHANGE: changed return to include post_activations ====== 
def forward_musicgenmodel(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPostActivations]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size * num_codebooks, sequence_length)`):
            Indices of input sequence tokens in the vocabulary, corresponding to the sequence of audio codes.

            Indices can be obtained by encoding an audio prompt with an audio encoder model to predict audio codes,
            such as with the [`EncodecModel`]. See [`EncodecModel.encode`] for details.

            [What are input IDs?](../glossary#input-ids)

            <Tip warning={true}>

            The `input_ids` will automatically be converted from shape `(batch_size * num_codebooks,
            target_sequence_length)` to `(batch_size, num_codebooks, target_sequence_length)` in the forward pass. If
            you obtain audio codes from an audio encoding model, such as [`EncodecModel`], ensure that the number of
            frames is equal to 1, and that you reshape the audio codes from `(frames, batch_size, num_codebooks,
            target_sequence_length)` to `(batch_size * num_codebooks, target_sequence_length)` prior to passing them as
            `input_ids`.

            </Tip>
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of
            the decoder.
        encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
            Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
            selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, past_key_values, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs

        # ===== MY CHANGE: Change to new custom class ======= 
        return BaseModelOutputWithPostActivations(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            post_activations=decoder_outputs.post_activations,
        )

# MusicgenForCausalLM forward override https://github.com/huggingface/transformers/blob/393b4b3d28e29b4b05b19b4b7f3242a7fc893637/src/transformers/models/musicgen/modeling_musicgen.py#L825
def forward_mgcausal(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.FloatTensor | None = None,
        encoder_attention_mask: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple | CausalLMOutputWithPostActivations:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size * num_codebooks, sequence_length)`):
            Indices of input sequence tokens in the vocabulary, corresponding to the sequence of audio codes.

            Indices can be obtained by encoding an audio prompt with an audio encoder model to predict audio codes,
            such as with the [`EncodecModel`]. See [`EncodecModel.encode`] for details.

            [What are input IDs?](../glossary#input-ids)

            <Tip warning={true}>

            The `input_ids` will automatically be converted from shape `(batch_size * num_codebooks,
            target_sequence_length)` to `(batch_size, num_codebooks, target_sequence_length)` in the forward pass. If
            you obtain audio codes from an audio encoding model, such as [`EncodecModel`], ensure that the number of
            frames is equal to 1, and that you reshape the audio codes from `(frames, batch_size, num_codebooks,
            target_sequence_length)` to `(batch_size * num_codebooks, target_sequence_length)` prior to passing them as
            `input_ids`.

            </Tip>
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of
            the decoder.
        encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
            Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
            selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (labels is not None) and (input_ids is None and inputs_embeds is None):
            input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.bos_token_id)

    
        # MY NOTES: this is musicgenmodel
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        lm_logits = torch.stack([head(hidden_states) for head in self.lm_heads], dim=1)

        loss = None
        if labels is not None:
            # since encoder hidden states have been concatenated to the decoder hidden states,
            # we take the last timestamps corresponding to labels
            logits = lm_logits[:, :, -labels.shape[1] :]

            loss_fct = CrossEntropyLoss()
            loss = torch.zeros([], device=self.device)

            # per codebook cross-entropy
            # -100 labels are ignored
            labels = labels.masked_fill(labels == self.config.pad_token_id, -100)

            # per codebook cross-entropy
            # ref: https://github.com/facebookresearch/audiocraft/blob/69fea8b290ad1b4b40d28f92d1dfc0ab01dbab85/audiocraft/solvers/musicgen.py#L242-L243
            for codebook in range(self.config.num_codebooks):
                codebook_logits = logits[:, codebook].contiguous().view(-1, logits.shape[-1])
                codebook_labels = labels[..., codebook].contiguous().view(-1)
                loss += loss_fct(codebook_logits, codebook_labels)

            loss = loss / self.config.num_codebooks

        # (bsz, num_codebooks, seq_len, vocab_size) -> (bsz * num_codebooks, seq_len, vocab_size)
        lm_logits = lm_logits.reshape(-1, *lm_logits.shape[2:])

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # MY CHANGE: change dataclass class
        return CausalLMOutputWithPostActivations(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
            post_activations=outputs.post_activations,
        )

#MusicforConditionalGeneration forward
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/musicgen/modeling_musicgen.py#L1606C1-L1781C1
# decoder is MusicgenforCausalLM
# ===== MY CHANGE: changed return to include post_activations ====== 
def forward_mgcond(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.BoolTensor | None = None,
    input_values: torch.FloatTensor | None = None,
    padding_mask: torch.BoolTensor | None = None,
    decoder_input_ids: torch.LongTensor | None = None,
    decoder_attention_mask: torch.BoolTensor | None = None,
    encoder_outputs: tuple[torch.FloatTensor] | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    decoder_inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    return_dict: bool | None = None,
    **kwargs,
) -> tuple | Seq2SeqLMOutputWithPostActivations:
    r"""
    padding_mask (`torch.BoolTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.

        [What are attention masks?](../glossary#attention-mask)
    decoder_input_ids (`torch.LongTensor` of shape `(batch_size * num_codebooks, target_sequence_length)`, *optional*):
        Indices of decoder input sequence tokens in the vocabulary, corresponding to the sequence of audio codes.

        Indices can be obtained by encoding an audio prompt with an audio encoder model to predict audio codes,
        such as with the [`EncodecModel`]. See [`EncodecModel.encode`] for details.

        [What are decoder input IDs?](../glossary#decoder-input-ids)

        <Tip warning={true}>

        The `decoder_input_ids` will automatically be converted from shape `(batch_size * num_codebooks,
        target_sequence_length)` to `(batch_size, num_codebooks, target_sequence_length)` in the forward pass. If
        you obtain audio codes from an audio encoding model, such as [`EncodecModel`], ensure that the number of
        frames is equal to 1, and that you reshape the audio codes from `(frames, batch_size, num_codebooks,
        target_sequence_length)` to `(batch_size * num_codebooks, target_sequence_length)` prior to passing them as
        `decoder_input_ids`.

        </Tip>
    decoder_attention_mask (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
        Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also
        be used by default.
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length, num_codebooks)`, *optional*):
        Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
        `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
        are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

    Examples:
    ```python
    >>> from transformers import AutoProcessor, MusicgenForConditionalGeneration
    >>> import torch

    >>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

    >>> inputs = processor(
    ...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    ...     padding=True,
    ...     return_tensors="pt",
    ... )

    >>> pad_token_id = model.generation_config.pad_token_id
    >>> decoder_input_ids = (
    ...     torch.ones((inputs.input_ids.shape[0] * model.decoder.num_codebooks, 1), dtype=torch.long)
    ...     * pad_token_id
    ... )

    >>> logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits
    >>> logits.shape  # (bsz * num_codebooks, tgt_len, vocab_size)
    torch.Size([8, 1, 2048])
    ```"""
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    kwargs_text_encoder = {
        argument[len("text_encoder_")]: value
        for argument, value in kwargs.items()
        if argument.startswith("text_encoder_")
    }

    kwargs_audio_encoder = {
        argument[len("audio_encoder_")]: value
        for argument, value in kwargs.items()
        if argument.startswith("audio_encoder_")
    }

    kwargs_decoder = {
        argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
    }

    if encoder_outputs is None:
        encoder_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_text_encoder,
        )
    elif isinstance(encoder_outputs, tuple):
        encoder_outputs = BaseModelOutput(*encoder_outputs)

    encoder_hidden_states = encoder_outputs[0]

    # optionally project encoder_hidden_states
    if (
        self.text_encoder.config.hidden_size != self.decoder.config.hidden_size
        and self.decoder.config.cross_attention_hidden_size is None
    ):
        encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

    if attention_mask is not None:
        encoder_hidden_states = encoder_hidden_states * attention_mask[..., None]

    if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
        decoder_input_ids = shift_tokens_right(
            labels, self.config.decoder.pad_token_id, self.config.decoder.decoder_start_token_id
        )

    elif decoder_input_ids is None and decoder_inputs_embeds is None:
        audio_encoder_outputs = self.audio_encoder(
            input_values=input_values,
            padding_mask=padding_mask,
            **kwargs_audio_encoder,
        )
        audio_codes = audio_encoder_outputs.audio_codes
        frames, bsz, codebooks, seq_len = audio_codes.shape
        if frames != 1:
            raise ValueError(
                f"Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is "
                "disabled by setting `chunk_length=None` in the audio encoder."
            )

        if self.config.decoder.audio_channels == 2 and audio_codes.shape[2] == self.decoder.num_codebooks // 2:
            # mono input through encodec that we convert to stereo
            audio_codes = audio_codes.repeat_interleave(2, dim=2)

        decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        use_cache=use_cache,
        past_key_values=past_key_values,
        return_dict=return_dict,
        labels=labels,
        **kwargs_decoder,
    )

    if not return_dict:
        return decoder_outputs + encoder_outputs

    return Seq2SeqLMOutputWithPostActivations(
        loss=decoder_outputs.loss,
        logits=decoder_outputs.logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
        decoder_post_activations=decoder_outputs.post_activations,
    )



def override_mcg_forwards(mgc_instance):
    mgc_decoder = mgc_instance.decoder
    mgc_model = mgc_instance.decoder.model
    mgc_dm = mgc_model.decoder
    mgc_layers = mgc_dm.layers
    for l in mgc_layers:
        l.forward = types.MethodType(forward_musicgendecoderlayer, l)
    mgc_decoder.forward = types.MethodType(forward_mgcausal, mgc_decoder)
    mgc_dm.forward = types.MethodType(forward_musicgendecoder, mgc_dm)
    mgc_model.forward = types.MethodType(forward_musicgenmodel, mgc_model)
    mgc_instance.forward = types.MethodType(forward_mgcond, mgc_instance)

### porting old code from mtmidi
def get_print_name(dataset, model_size, is_csv = False, normalize = True, timestamp = 0):
    base_fname = f'{dataset}_musicgen-{model_size}-{timestamp}'
    if normalize == True:
        base_fname = f'{dataset}_musicgen-{model_size}_norm-{timestamp}'
    ret = None
    if is_csv == False:
        ret = f'{base_fname}.log'
    else:
        ret = f'{base_fname}.csv'
    return ret

def path_handler(in_filepath, using_hf=False, model_sr = 44100, dur = 4., normalize = True, out_ext = 'dat', logfile_handle=None):
    out_fname = None
    audio = None
    out_fname = None
    fbasename = None
    fold_num = -1 
    if using_hf == False:
        print(f'loading {in_filepath}', file=logfile_handle)
        fbasename = UMN.get_basename(in_filepath)
        fold_num = UMN.get_fold_num_from_filepath(in_filepath)
        out_fname = f'{fbasename}.{out_ext}'
        # don't need to load audio if jukebox
        audio = UMN.load_wav(in_filepath, dur = dur, normalize = normalize, sr = model_sr)
    else:
        hf_path = in_filepath['audio']['path']
        print(f"loading {hf_path}", file=lf)
        out_fname = UMN.ext_replace(hf_path, new_ext=out_ext)
        fbasename = UMN.ext_replace(hf_path, new_ext='')
        audio = UHF.get_from_entry_syntheory_audio(in_filepath, mono=True, normalize =normalize, dur = dur, sr=model_sr)
    return {'in_fpath': in_filepath, 'out_fname': out_fname, 'audio': audio, 'fname': fbasename, 'fold_num': fold_num}

# same as get_musicgen_lm_hidden_states but swap out outputs.decoder_hidden_states with decoder_post_activations
def get_musicgen_lm_postacts(model, proc, audio, text="", meanpool = True, model_sr = 32000, device = 'cpu'):
    procd = proc(audio = audio, text = text, sampling_rate = model_sr, padding=True, return_tensors = 'pt')
    procd.to(device)
    outputs = model(**procd, output_attentions=False, output_hidden_states=True)
    dhs = None
    
    #dat = None

    # hidden
    # outputs is a tuple of tensors with  shape (batch_size, seqlen, dimension) with 1 per layer
    # torch stack makes it so we have (num_layers, batch_size, seqlen, dimension)
    # then we average over seqlen in the meanpool case
    # then squeeze to get rid of the 1 dim (if batch_size == 1)
    # final shape is (num_layers, batch_size, dim)  (or (num_layers, dim) if bs=1)
    
    # attentions
    # outputs is a tuple of tensors with  shape (batch_size, num_heads, seqlen, seqlen) with 1 per layer
    # torch stack makes it so we have (num_layers, batch_size, num_heads, seqlen, sequlen)
    # then we average over seqlens in the meanpool case
    # then squeeze to get rid of the 1 dim (if batch_size == 1)
    # final shape is (num_layers, batch_size, num_heads) (or (num_layers, num_heads) if bs = 1)

    if meanpool == True:
        dhs = torch.stack(outputs.decoder_post_activations).mean(axis=2).squeeze()
        #dat = torch.stack(outputs.decoder_attentions).mean(axis=(3,4)).squeeze()
    else:
        dhs = torch.stack(outputs.decoder_post_activations).squeeze()
        #dat = torch.stack(outputs.decoder_attentions).squeeze()
    return dhs.detach().cpu().numpy()


def get_postacts(model_size, cur_dataset, normalize = True, dur = 4., use_64bit = True, logfile_handle=None, recfile_handle = None, memmap = True, pickup = False, fold_num = -1, from_dir = "", to_dir = ""):
    
    using_hf = cur_dataset in UC.SYNTHEORY_DATASETS
    # musicgen stuff
    device = 'cpu'
    num_layers = None
    proc = None
    model = None
    text = ""
    wav_path = os.path.join(UMN.by_projpath('wav'), cur_dataset)
    if len(from_dir) > 0:
        wav_path = os.path.join(from_dir, cur_dataset)
    cur_pathlist = None
    out_ext = 'dat'
    if memmap == False:
        out_ext = 'npy'
    if using_hf == True:
        fold_num = -1 # don't care about fold folders
        cur_pathlist = UHF.load_syntheory_train_dataset(cur_dataset)
    else:
        cur_pathlist = UMN.filepath_list(wav_path, fold_num=fold_num, ignore_exts = set(['.csv']))

    device = 'cpu'
    if torch.cuda.is_available() == True:
        device = 'cuda'
        torch.cuda.empty_cache()
        torch.set_default_device(device)
    
    model_str = UMN.get_hf_model_str(model_size) 
    proc = AutoProcessor.from_pretrained(model_str)
    model = MusicgenForConditionalGeneration.from_pretrained(model_str, device_map=device)
    model_sr = model.config.audio_encoder.sampling_rate

    override_mcg_forwards(model)

    # existing files removing latest (since it may be partially written) and removing extension for each of checking
    existing_name_set = None
    if pickup == True:
        # pass -1 for fold_num to omit fold_num folder since remove_latest_file takes care of it
        _file_dir = UMN.get_model_postacts_path(model_size, dataset=cur_dataset, return_relative = False, make_dir = False, other_projdir = to_dir, fold_num=-1)
        existing_files = UMN.remove_latest_file(_file_dir, is_relative = False, fold_num = fold_num)
        existing_name_set = set([UMN.get_basename(_f, with_ext = False) for _f in existing_files])
    for fidx,fpath in enumerate(cur_pathlist):
        if pickup == True:
            cur_name = UMN.get_basename(fpath, with_ext = False)
            if cur_name in existing_name_set:
                continue
        fdict = path_handler(fpath, model_sr = model_sr, normalize = normalize, dur = dur,using_hf = using_hf, logfile_handle=logfile_handle, out_ext = out_ext)
        #outpath = os.path.join(out_dir, outname)
        out_fname = fdict['out_fname']
        in_fpath = fdict['in_fpath']
        audio_ipt = fdict['audio']
        fold_num = fdict['fold_num']
        # store by model_size (and fold_num if not using_hf)
        emb_file = None
        np_arr = None
        if memmap == True:
            emb_file = UMN.get_postacts_file(model_size, dataset=cur_dataset, fname=out_fname, use_64bit = use_64bit, write=True, use_shape = None, other_projdir = to_dir, fold_num = fold_num)
        print(f'--- extracting musicgen_lm for {fpath} ---', file=logfile_handle)
        rep_arr =  get_musicgen_lm_postacts(model, proc, audio_ipt, text="", meanpool = True, model_sr = model_sr, device=device)
        if memmap == True:
            emb_file[:,:] = rep_arr
            emb_file.flush()
        else:
            UMN.save_npy(rep_arr, out_fname, model_size, dataset=cur_dataset, other_projdir = to_dir)
        fname = fdict['fname']
        print(f'{fname},1', file=recfile_handle)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ub", "--use_64bit", type=strtobool, default=False, help="use 64-bit")
    parser.add_argument("-ds", "--dataset", type=str, default="polyrhythms", help="dataset")
    parser.add_argument("-ms", "--model_size", type=str, default="small", help="small, medium, or large")
    parser.add_argument("-l", "--layer_num", type=int, default=-1, help="1-indexed layer num (all if < 0, for jukebox)")
    parser.add_argument("-n", "--normalize", type=strtobool, default=True, help="normalize audio")
    parser.add_argument("-m", "--memmap", type=strtobool, default=True, help="save as memmap, else save as npy")
    parser.add_argument("-db", "--debug", type=strtobool, default=False, help="debug mode")
    parser.add_argument("-p", "--pickup", type=strtobool, default=False, help="pickup where script left off")
    parser.add_argument("-tsh", "--to_share", type=strtobool, default=False, help="save on share partition")
    parser.add_argument("-fsh", "--from_share", type=strtobool, default=False, help="load on share partition")
    parser.add_argument("-fn", "--fold_num", type=int, default=0, help="fold number to extract (-1 for no folds, 0 for all folds, else specific fold)")

    
    args = parser.parse_args()
    use_64bit = args.use_64bit
    lnum = args.layer_num
    memmap = args.memmap
    normalize = args.normalize
    model_size = args.model_size
    dataset = args.dataset
    debug = args.debug
    pickup = args.pickup
    to_share = args.to_share
    from_share = args.from_share
    fold_num = args.fold_num
    # exit if not a "real" dataset
    logdir = UMN.by_projpath(subpath='log', make_dir = True)
    timestamp = int(time.time() * 1000)

    from_dir = ""
    if args.from_share == True:
        from_dir = os.path.join(UC.SHARE_PATH, 'syntheory_plus')
    if args.to_share == True:
        to_dir = os.path.join(UC.SHARE_PATH, 'mtmidi_sp')
    # miscellaneous logs
    log_fname = get_print_name(dataset, model_size, is_csv = False, normalize = normalize, timestamp = timestamp)
    rec_fname = get_print_name(dataset, model_size, is_csv = True, normalize = normalize, timestamp = timestamp)
    log_fpath = os.path.join(logdir, log_fname)
    rec_fpath = os.path.join(logdir, rec_fname)
    if debug == True:
        exit()
    if (dataset in UC.ALL_DATASETS) == False:
        sys.exit('not a dataset')
    else:
        lf = open(log_fpath, 'a')
        rf = open(rec_fpath, 'w')
        print(f'=== running extraction for {dataset} with {model_size} at {timestamp} ===', file=lf)
        get_postacts(model_size, dataset, normalize = normalize, dur = dur, use_64bit = use_64bit, logfile_handle=lf, recfile_handle=rf, memmap = memmap, pickup = pickup, fold_num = fold_num, from_dir = from_dir, to_dir = to_dir)
        lf.close()
        rf.close()
