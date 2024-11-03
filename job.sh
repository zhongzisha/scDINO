source /data/zhongz2/anaconda3/bin/activate th24
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12
module load gcc/11.3.0


source $FRCE_DATA_ROOT/anaconda3/bin/activate th23
module load cuda/11.8
module load cudnn/8.8.3-cuda11



snakemake -s full_pipeline_snakefile1_biowulf all \
--configfile="configs/scDINO_full_pipeline1_biofulf.yaml" \
--keep-incomplete \
--drop-metadata \
--cores 8 \
--jobs 40 \
-k \
--cluster "sbatch --time=24:00:00 --partition=gpu --gres=gpu:v100x:2 --ntasks=8 --mem=64G --cpus-per-task=16 --output=slurm_output.txt --error=slurm_error.txt" \
--latency-wait 45


snakemake -s full_pipeline_snakefile1 all \
--configfile="configs/scDINO_full_pipeline1.yaml" \
--keep-incomplete \
--drop-metadata \
--cores 32 \
--jobs 1 \
-k \
--latency-wait 45

snakemake -s full_pipeline_snakefile1_4gpus all \
--configfile="configs/scDINO_full_pipeline1_4gpus.yaml" \
--keep-incomplete \
--drop-metadata \
--cores 1 \
--jobs 1 \
-k \
--latency-wait 45

snakemake -s only_downstream_snakefile1 all \
--configfile="configs/only_downstream_analyses1.yaml" \
--keep-incomplete \
--drop-metadata \
--keep-going \
--cores 8 \
--jobs 40 \
-k \
--cluster "sbatch --time=01:00:00 \
--gpus=1 \
-n 8 \
--mem-per-cpu=9000 \
--output=slurm_output_evaluate.txt \
--error=slurm_error_evaluate.txt" \
--latency-wait 45 \