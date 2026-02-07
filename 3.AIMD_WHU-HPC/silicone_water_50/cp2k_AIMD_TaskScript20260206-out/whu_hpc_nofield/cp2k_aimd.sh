#!/bin/bash
#SBATCH --job-name=aimd_nofield               
#SBATCH --partition=9a14a               
#SBATCH --account=lijinjun             
#SBATCH --nodes=1                        
#SBATCH --ntasks-per-node=192            
#SBATCH --cpus-per-task=1                
#SBATCH --error=slurm-%j.error           
#SBATCH --output=slurm-%j.out            

#activate cp2k env
source /home/baihaiyan/project/cp2k/env_cp2k

#relocate to submit dir
cd $SLURM_SUBMIT_DIR

#run cp2k job
mpirun -np $SLURM_NTASKS cp2k.popt -i aimd.inp -o aimd.out

