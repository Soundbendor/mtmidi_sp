import util_main as UM
import ..data_helpers.polyrhythms as PL
import ..data_helpers.dynamics as DYN
import ..data_helpers.seventh_chords as CH7
import ..data_helpers.mode_mixture as MM
import ..data_helpers.secondary_dominants as SD

import ..data_helpers.time_signatures as TSG
import ..data_helpers.chords as CHD
import ..data_helpers.notes as NTS
import ..data_helpers.scales as SCL
import ..data_helpers.intervals as IVL
import ..data_helpers.simple_progressions as SPG

def get_df(dataset):
    fname = f'{dataset}-metadata.csv'
    csvpath = os.path.join(UM.by_projpath('csv', make_dir = False), fname)
    cur_data = pl.read_csv(csvpath)
    return cur_data

def load_data_dict(dataset):
    num_classes = None
    classdict = None
    label = None
    is_balanced = True
    is_classification = dataset != 'tempos'

    cur_df = get_df(dataset)
    if dataset == 'polyrhythms':
        num_classes = PL.num_poly
        classdict = PL.polystr_to_idx
        label = 'poly'
    elif dataset == 'dynamics':
        is_balanced = False
        num_classes = DYN.num_categories
        classdict = DYN.dyn_category_to_idx
        label = 'dyn_category'
    elif dataset == "seventh_chords":
        num_classes =  CH7.num_chords
        classdict = CH7.quality_to_idx
        label = 'quality'
    elif dataset == 'mode_mixture':
        num_classes = MM.num_is_modemix
        classdict = MM.is_modemix_to_idx
        label = 'is_modemix'
    elif dataset == 'secondary_dominants':
        num_classes = SD.num_subtypes
        classdict = SD.sub_type_to_idx
        label = 'sub_type'
    elif dataset == 'tempos':
        num_classes = float('inf') # regression
        classdict = {} # regression, no classes
        label = 'bpm'
    elif dataset == 'time_signatures':
        num_classes = TSG.num_timesig
        classdict = TSG.timesig_to_idx
        label = 'time_signature'
    elif dataset == 'chords':
        num_classes = CHD.num_chords
        classdict = CHD.quality_to_idx
        label = 'chord_type'
    elif dataset == 'notes':
        num_classes = NTS.num_pc
        classdict = NTS.pc_to_idx 
        label = 'root_note_pitch_class'
    elif dataset == 'scales':
        num_classes = SCL.num_modes
        classdict = SCL.mode_to_idx
        label = 'mode'
    elif dataset == 'intervals':
        num_classes = IVL.num_intervals
        classdict = IVL.interval_to_idx
        label = 'interval'
    elif dataset == 'simple_progressions':
        num_classes = SPG.num_progs
        classdict = SPG.prog_to_idx
        label = 'orig_prog'

    label_arr = cur_df.select([label]).to_numpy().flatten()

    ret = {
            'num_classes': num_classes,
            'df': cur_df,
            'classdict': classdict,
            'is_classification': is_classification,
            'label': label,
            'is_balanced': is_balanced
            }
    return ret





