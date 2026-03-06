import os
import polars as pl
import util as UM
from util import util_main as UMN
from util import util_constants as UC
import sys

mv_test = True
datdir = '/nfs/hpc/share/kwand/mtmidi_sp/postacts'
dataset = 'notes'
if len(sys.argv) > 1:
    dataset = sys.argv[1]

dsdatdir = os.path.join(datdir, dataset)
csvdir = UMN.by_projpath(subpath='csv',make_dir = False)
csvfile = os.path.join(csvdir, f'{dataset}-metadata.csv')
df = pl.read_csv(csvfile)

#os.mkdir(dstestdir)

for i in range(len(df)):
    cur_name = df[i]['name'][0]
    cur_dat = f'{cur_name}.dat'
    cur_dat = os.path.join(dsdatdir, cur_dat)
    cur_fold = df[i]['fold'][0]
    fold_folder = f'fold_{cur_fold}'
    from_dir = None
    cur_file = None
    from_dir = dsdatdir

    fold_fp = os.path.join(from_dir, fold_folder)
    to_fp = os.path.join(fold_fp, cur_dat)
    if os.path.isdir(fold_fp) == False:
        os.mkdir(fold_fp)
    os.rename(from_fp, to_fp)



