import sys,os,time,argparse,copy,types
import librosa as lr
from librosa import feature as lrf
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from typing import TYPE_CHECKING, Any, Optional, Union
import numpy as np
import torch
from distutils.util import strtobool
from transformers.cache_utils import Cache, DynamicCache,EncoderDecoderCache
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, BaseModelOutput, Seq2SeqLMOutput

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
        # MY CHANGE: taking representations post activation
        post_activation = hidden_states.clone().detach()
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        # MY CHANGE: append post_activation representations
        outputs += (post_activation,)

        return outputs

# MusicgenDecoder
# https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/musicgen/modeling_musicgen.py#L472
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
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
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

        # embed positions
        positions = self.embed_positions(input, past_key_values_length)
        hidden_states = inputs_embeds + positions.to(inputs_embeds.device)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None

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
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

# MusicgenModel
# https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/musicgen/modeling_musicgen.py#L707
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
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
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

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )


#MusicforConditionalGeneration forward
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/musicgen/modeling_musicgen.py#L1606C1-L1781C1
def mcg_forward(
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
) -> tuple | Seq2SeqLMOutput:
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

    return Seq2SeqLMOutput(
        loss=decoder_outputs.loss,
        logits=decoder_outputs.logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )



def override_mcg_forwards(mcg_instance):
    mcg_model = mcg_instance.decoder.model
    mcg_dm = mcg_model.decoder
    mcg_layers = mcg_model.decoder.layers
    for l in mcg_layers:
        l.forward = types.MethodType(forward_musicgendecoderlayer, l)
    mcg_dm.forward = types.MethodType(forward_musicgendecoder, mcg_dm)
    mcg_model.forward = types.MethodType(forward_musicgenmodel, mcg_model)
"""
def get_hf_audio(f, model_sr = 44100, normalize=True):
    audio, aud_sr = uhf.get_from_entry_syntheory_audio(f, mono=True, normalize =normalize, dur = dur)
    if aud_sr != model_sr:
        audio = lr.resample(audio, orig_sr=aud_sr, target_sr=model_sr)
    return audio



def path_handler(f, using_hf=False, model_sr = 44100, wav_path = None, model_type = 'jukebox', dur = 4., normalize = True, out_ext = 'dat', logfile_handle=None):
    out_fname = None
    audio = None
    in_dir = um.by_projpath(wav_path)
    in_fpath = None
    out_fname = None
    fname = None
    if using_hf == False:
        print(f'loading {f}', file=logfile_handle)
        in_fpath = os.path.join(in_dir, f)
        out_fname = um.ext_replace(f, new_ext=out_ext)
        fname = um.ext_replace(f, new_ext='')
        # don't need to load audio if jukebox
        if model_type != 'jukebox':
            audio = um.load_wav(f, dur = dur, normalize = normalize, sr = model_sr,  load_dir = in_dir)
    else:
        hf_path = f['audio']['path']
        print(f"loading {hf_path}", file=lf)
        out_fname = um.ext_replace(hf_path, new_ext=out_ext)
        fname = um.ext_replace(hf_path, new_ext='')
    aud_sr = None
    if using_hf == True:
        audio, aud_sr = uhf.get_from_entry_syntheory_audio(f, mono=True, normalize =normalize, dur = dur)
        if aud_sr != model_sr:
            audio = lr.resample(audio, orig_sr=aud_sr, target_sr=model_sr)
    return {'in_fpath': in_fpath, 'out_fname': out_fname, 'audio': audio, 'fname': fname}


def get_musicgen_encoder_embeddings(model, proc, audio, meanpool = True, model_sr = 32000, device='cpu'):
    procd = proc(audio = audio, sampling_rate = model_sr, padding=True, return_tensors = 'pt')
    procd.to(device)
    enc = model.get_audio_encoder()
    out = procd['input_values']
    
    # iterating through layers as in original syntheory codebase
    # https://github.com/brown-palm/syntheory/blob/main/embeddings/models.py
    for layer in enc.encoder.layers:
        out = layer(out)

    # output shape, (1, 128, 200), where 200 are the timesteps
    # so average across timesteps for max pooling


    if meanpool == True:
        # gives shape (128)
        out = torch.mean(out,axis=2).squeeze()
    else:
        # still need to squeeze
        # gives shape (128, 200)
        out = out.squeeze()
    return out.detach().cpu().numpy()

def get_musicgen_lm_hidden_states(model, proc, audio, text="", meanpool = True, model_sr = 32000, device = 'cpu'):
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
        dhs = torch.stack(outputs.decoder_hidden_states).mean(axis=2).squeeze()
        #dat = torch.stack(outputs.decoder_attentions).mean(axis=(3,4)).squeeze()
    else:
        dhs = torch.stack(outputs.decoder_hidden_states).squeeze()
        #dat = torch.stack(outputs.decoder_attentions).squeeze()
    return dhs.detach().cpu().numpy()


def get_embeddings(cur_act_type, cur_dataset, layers_per = 4, layer_num = -1, normalize = True, dur = 4., use_64bit = True, logfile_handle=None, recfile_handle = None, memmap = True, pickup = False, other_projdir = ''):
    cur_model_type = um.get_model_type(cur_act_type)
    model_sr = um.model_sr[cur_model_type]
    model_longhand = um.model_longhand[cur_act_type]

    using_hf = cur_dataset in um.hf_datasets
    # musicgen stuff
    device = 'cpu'
    num_layers = None
    proc = None
    model = None
    text = ""
    wav_path = os.path.join(um.by_projpath('wav'), cur_dataset)
    cur_pathlist = None
    out_ext = 'dat'
    if memmap == False:
        out_ext = 'npy'
    if using_hf == True:
        cur_pathlist = uhf.load_syntheory_train_dataset(cur_dataset)
    else:
        cur_pathlist = um.path_list(wav_path)


    if torch.cuda.is_available() == True:
        device = 'cuda'
        torch.cuda.empty_cache()
        torch.set_default_device(device)


    model_str = f"facebook/{cur_model_type}" 

    proc = AutoProcessor.from_pretrained(model_str)
    model = MusicgenForConditionalGeneration.from_pretrained(model_str, device_map=device)
    model_sr = model.config.audio_encoder.sampling_rate

    #print('file,is_extracted', file=rf)

    # existing files removing latest (since it may be partially written) and removing extension for each of checking
    existing_name_set = None
    if pickup == True:
        _file_dir = um.get_model_act_path(cur_act_type, acts_folder = acts_folder, dataset=cur_dataset, return_relative = False, make_dir = False, other_projdir = other_projdir)
        existing_files = um.remove_latest_file(_file_dir, is_relative = False)
        existing_name_set = set([um.get_basename(_f, with_ext = False) for _f in existing_files])
    for fidx,f in enumerate(cur_pathlist):
        if pickup == True:
            cur_name = um.get_basename(f, with_ext = False)
            if cur_name in existing_name_set:
                continue
        fdict = path_handler(f, model_sr = model_sr, wav_path = wav_path, normalize = normalize, dur = dur,model_type = cur_model_type, using_hf = using_hf, logfile_handle=logfile_handle, out_ext = out_ext)
        #outpath = os.path.join(out_dir, outname)
        out_fname = fdict['out_fname']
        fpath = fdict['in_fpath']
        audio = fdict['audio']
        # store by cur_act_type (model shorthand)
        emb_file = None
        np_arr = None
        if memmap == True:
            emb_file = um.get_embedding_file(cur_act_type, acts_folder=acts_folder, dataset=cur_dataset, fname=out_fname, use_64bit = use_64bit, write=True, use_shape = None, other_projdir = other_projdir)
        if cur_model_type == 'jukebox':
            print(f'--- extracting jukebox for {f} with {layers_per} layers at a time ---', file=logfile_handle)
            # note that layers are 1-indexed in jukebox
            # so let's 0-idx and then add 1 when feeding into jukebox fn
            layer_gen = (list(range(l, min(um.model_num_layers['jukebox'], l + layers_per))) for l in range(0,um.model_num_layers['jukebox'], layers_per))
            has_last_layer = False
            if memmap == False:
                np_shape = um.get_embedding_shape(cur_act_type)
                np_arr = np.zeros(np_shape)
            if layer_num > 0:
                # 0-idx from 1-idxed argt
                layer_gen = ([l-1] for l in [layer_num])
            for layer_arr in layer_gen:
                # 1-idx for passing into fn
                j_idx = [l+1 for l in layer_arr]
                has_last_layer = um.model_num_layers['jukebox'] in j_idx
                print(f'extracting layers {j_idx}', file=logfile_handle)
                rep_arr = get_jukebox_layer_embeddings(fpath=fpath, audio = audio, layers=j_idx)
                if memmap == True:
                    emb_file[layer_arr,:] = rep_arr
                    emb_file.flush()
                else:
                    np_arr[layer_arr,:] = rep_arr
                    # should be the last layer to save
                    if has_last_layer == True:
                        um.save_npy(np_arr, out_fname, cur_act_type, acts_folder = acts_folder, dataset=cur_dataset, other_projdir = other_projdir)
        else:
            audio_ipt = fdict['audio']
            if model_longhand == "musicgen-encoder":
                print(f'--- extracting musicgen-encoder for {f} ---', file=logfile_handle)

                rep_arr = get_musicgen_encoder_embeddings(model, proc, audio_ipt, meanpool = True, model_sr = model_sr, device=device)
                if memmap == True:
                    emb_file[:,:] = rep_arr
                    emb_file.flush()
                else:
                    um.save_npy(rep_arr, out_fname, cur_act_type, acts_folder = acts_folder, dataset=cur_dataset, other_projdir = other_projdir)
            else:

                print(f'--- extracting musicgen_lm for {f} ---', file=logfile_handle)
                rep_arr =  get_musicgen_lm_hidden_states(model, proc, audio_ipt, text="", meanpool = True, model_sr = model_sr, device=device)
                if memmap == True:
                    emb_file[:,:] = rep_arr
                    emb_file.flush()
                else:
                    um.save_npy(rep_arr, out_fname, cur_act_type, acts_folder = acts_folder, dataset=cur_dataset, other_projdir = other_projdir)
        fname = fdict['fname']
        print(f'{fname},1', file=recfile_handle)
"""
