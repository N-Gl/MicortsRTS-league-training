#!/usr/bin/env bash
#SBATCH --job-name=league3x_25envs2
#SBATCH --output=slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
# Optional (bei Bedarf auf deinem Cluster aktivieren):
##SBATCH --partition=gpu
##SBATCH --account=your_account

set -euo pipefail

REPO_DIR="/home/nglaser/BA/MicortsRTS-league-training"
CONFIG_NAME="league_sp_conf_old_ppo_args_14_1_no_Bots_2__25envs_2"
RUN_NAMES=(
  "league_sp_conf_old_ppo_args_14_1_no_Bots_2__25envs_1"
  "league_sp_conf_old_ppo_args_14_1_no_Bots_2__25envs_2"
  "league_sp_conf_old_ppo_args_14_1_no_Bots_2__25envs_3"
)

cd "${REPO_DIR}"

# Conda-Umgebung laden
source "${HOME}/miniforge/etc/profile.d/conda.sh"
conda activate py39

echo "Using config: ${CONFIG_NAME}"
echo "Running ${#RUN_NAMES[@]} times sequentially on one allocated GPU."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

for i in "${!RUN_NAMES[@]}"; do
  RUN_INDEX=$((i + 1))
  EXP_NAME="${RUN_NAMES[$i]}"
  echo "[$(date -Is)] Start run ${RUN_INDEX}/${#RUN_NAMES[@]}: ${EXP_NAME}"

  python3 main.py \
    --config-name "${CONFIG_NAME}" \
    ExperimentConfig.exp_name="${EXP_NAME}"

  echo "[$(date -Is)] Finished run ${RUN_INDEX}/${#RUN_NAMES[@]}: ${EXP_NAME}"
done

echo "All ${#RUN_NAMES[@]} runs completed."
