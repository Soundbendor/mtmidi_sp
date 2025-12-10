# need to override forward method of MusicgenDecoderLayer: https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/musicgen/modeling_musicgen.py#L304
# change outputs so need to override forward of MusicgenDecoder: https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/musicgen/modeling_musicgen.py#L437
# which then overrides forward of MusicgenModel: https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/musicgen/modeling_musicgen.py#L693

# override at instance level: https://stackoverflow.com/questions/394770/override-a-method-at-instance-level

# forward method of MusicgenDecoderLayer
# https://github.com/huggingface/transformers/blob/ff13eb668aa03f151ded71636d723f2e490ad967/src/transformers/models/musicgen/modeling_musicgen.py#L338


