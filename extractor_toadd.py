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
