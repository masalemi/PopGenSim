#!/bin/bash -x

# sbatch -N $nodes --ntasks-per-node=$tasks --partition=dcs --gres=gpu:1,nvme -t 5 ./slurmSpectrum.sh cudagensim.exe

if [ "x$SLURM_NPROCS" = "x" ]
then
    if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
    then
        SLURM_NTASKS_PER_NODE=1
    fi
    SLURM_NPROCS=`expr $SLURM_JOB_NUM_NODES \* $SLURM_NTASKS_PER_NODE`
    else
        if [ "x$SLURM_NTASKS_PER_NODE" = "x" ]
        then
            SLURM_NTASKS_PER_NODE=`expr $SLURM_NPROCS / $SLURM_JOB_NUM_NODES`
        fi
fi


srun hostname -s | sort -u > /tmp/hosts.$SLURM_JOB_ID
awk "{ print \$0 \"-ib slots=$SLURM_NTASKS_PER_NODE\"; }" /tmp/hosts.$SLURM_JOB_ID > /tmp/tmp.$SLURM_JOB_ID
mv /tmp/tmp.$SLURM_JOB_ID /tmp/hosts.$SLURM_JOB_ID

# module load gcc/7.4.0/1
# module load spectrum-mpi
# module load cuda
module load xl_r spectrum-mpi cuda/10.2

cpu_list="0"
end_iter="$(expr $SLURM_NPROCS - 1)"
for i in $(seq 1 $end_iter)
do
    index="$(expr 4 \* $i)"
    cpu_list="$cpu_list,$index"
done

taskset --cpu-list "$cpu_list" mpirun -hostfile /tmp/hosts.$SLURM_JOB_ID -np $SLURM_NPROCS $1
rm /tmp/hosts.$SLURM_JOB_ID
