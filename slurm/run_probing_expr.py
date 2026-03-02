import argparse
from ..util import util_main as UMN
from ..util import util_constants as UC

from distutils.util import strtobool
import os, time, subprocess

if __name__ == "__main__":
    #### arg parsing
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-ds", "--datasets", nargs="+" type=str, default=["polyrhythms"], help="datasets")
    parser.add_argument("-nd", "--number_of_days", type=int, default=1, help="number of days")
    parser.add_argument("-pt", "--partition", type=str, default="preempt", help="partition to run on")
    parser.add_argument("-ms", "--model_sizes", nargs="+", type=str, default=["small","medium","large"], help="small/medium/large")
    parser.add_argument("-et", "--expr_type", type=str, default="linearnn_full", help="experiment type")
    parser.add_argument("-wdb", "--use_wandb", type=strtobool, default=True, help="sync to wandb")
    parser.add_argument("-cd", "--use_cuda", type=strtobool, default=True, help="use cuda")
    parser.add_argument("-ev", "--eval", type=strtobool, default=False, help="eval")
    parser.add_argument("-rs", "--restart_study", type=strtobool, default=False, help="force restart of optuna study")
    parser.add_argument("-sh", "--from_share", type=strtobool, default=False, help="load from share partition")
    parser.add_argument("-sf", "--suffix", type=int, default=1, help="suffix")
    parser.add_argument("-tsd", "--torch_seed", type=int, default=UC.SEED, help="torch random seed")
    parser.add_argument("-ssd", "--split_seed", type=int, default=UC.SEED, help="seed for splitting")
    parser.add_argument("-ram", "--ram_mem", type=int, default=40, help="ram in gigs")
    parser.add_argument("-gpu", "--gpus", type=int, default=1, help="num of gpus to use")
    parser.add_argument("-sj", "--slurm_job", type=int, default=0, help="slurm job")
    


    args = parser.parse_args()
   
    project_root = Path(__file__).resolve().parent.parent
    cur_dir = Path__file__).resolve().parent
    sh_dir = os.path.join(cur_dir, 'sh')
    if os.path.exists(sh_dir) == False:
        os.makedirs(cur_path)

    py_path = os.path.join(project_root, 'probing.py')


    script_idx = 0
    start_time = str(int(time.time() * 1000))

    expr_short = UC.EXPR_SHORT[args.expr_type] 
    for dataset in args.datasets:
        ds_short = UC.DATASET_SHORT[dataset]
        for model_size in args.model_sizes:
            size_short = UC.SIZES_SHORT[model_size]         
            job_str = f'{expr_short}-{ds_short}-{size_short}'
            slurm_strarr1 = ["#!/bin/bash"]
            slurm_strarr2 = [f"#SBATCH -p {args.partition}"]
            if args.partition != 'preempt':
                if args.partition != 'soundbendor':
                    slurm_strarr2 = ['#SBATCH -A eecs', f"#SBATCH -p {args.partition}"]
                else:
                    slurm_strarr2 = ['#SBATCH -A soundbendor', f"#SBATCH -p {args.partition}"]
            slurm_strarr3 = [f"#SBATCH --mem={args.ram_mem}G", f"#SBATCH --gres=gpu:{args.gpus}", "#SBATCH -t {args.number_of_days}-00:00:00", f"#SBATCH --job-name={job_str}", "#SBATCH --export=ALL", f"#SBATCH --output=/nfs/guille/eecs_research/soundbendor/kwand/out_mtmidi_sp/{job_str}-%j.out", ""]
            slurm_strarr = slurm_strarr1 + slurm_strarr2 + slurm_strarr3
            p_str = f"python {py_path} -ev {args.eval} -ds {dataset} -et {args.expr_type} -ms {model_size} -sh {args.from_share} -wdb {args.use_wandb} -cd {args.use_cuda} -tsd {args.torch_seed} -ssd {args.split_seed}" 
            slurm_strarr.append(p_str)
            script_fname = f"{start_time}_{job_str}.sh"
            script_idx += 1
            script_path = os.path.join(sh_dir, script_fname)
            script_str = "\n".join(slurm_strarr)
            print(f"===== {args.expr_type} | {dataset} | {model_size} =====")
            print(f"Creating {script_fname}")
            with open(script_path, 'w') as f:
                f.write(script_str)
            subprocess.run(f"chmod u+x {script_path}", shell=True)
            if args.debug == False:
                print(f"Running {script_fname}")
                subprocess.run(f"sbatch -W {script_path}", shell=True)




