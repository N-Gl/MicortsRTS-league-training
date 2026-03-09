#!/usr/bin/env bash
#SBATCH --job-name=league3x_25envs2
#SBATCH --output=slurm-%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
# Optional (bei Bedarf auf deinem Cluster aktivieren):
##SBATCH --partition=gpu
##SBATCH --account=your_account

set -euo pipefail

REPO_DIR="/home/nglaser/BA/MicortsRTS-league-training"
CONFIG_NAME="league_sp_conf_old_ppo_args_14_1_no_Bots_2__25envs_7_test_5"
RUN_NAMES=(
  "league_sp_conf_old_ppo_args_14_1_no_Bots_2__25envs_7_test_5_1"
  "league_sp_conf_old_ppo_args_14_1_no_Bots_2__25envs_7_test_5_2"
  "league_sp_conf_old_ppo_args_14_1_no_Bots_2__25envs_7_test_5_3"
)

cd "${REPO_DIR}"

# Conda-Umgebung laden
source "${HOME}/miniforge/etc/profile.d/conda.sh"
conda activate py39

echo "Using config: ${CONFIG_NAME}"
echo "Running ${#RUN_NAMES[@]} runs in parallel on one allocated GPU."
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

LOG_DIR="${REPO_DIR}/parallel_run_logs"
mkdir -p "${LOG_DIR}"

RUN_PIDS=()
RUN_EXPERIMENTS=()

for i in "${!RUN_NAMES[@]}"; do
  RUN_INDEX=$((i + 1))
  EXP_NAME="${RUN_NAMES[$i]}"
  RUN_SEED="$(od -An -N4 -tu4 /dev/urandom | tr -d '[:space:]')"
  LOG_FILE="${LOG_DIR}/run_${RUN_INDEX}_${EXP_NAME}.log"
  echo "[$(date -Is)] Launching run ${RUN_INDEX}/${#RUN_NAMES[@]}: ${EXP_NAME} (seed: ${RUN_SEED}, log: ${LOG_FILE})"

  (
    echo "[$(date -Is)] Start ${EXP_NAME} (seed: ${RUN_SEED})"
    python3 main.py \
      --config-name "${CONFIG_NAME}" \
      ExperimentConfig.exp_name="${EXP_NAME}" \
      ExperimentConfig.seed="${RUN_SEED}"
    echo "[$(date -Is)] Finished ${EXP_NAME}"
  ) > "${LOG_FILE}" 2>&1 &

  RUN_PIDS+=("$!")
  RUN_EXPERIMENTS+=("${EXP_NAME}")
done

echo "All ${#RUN_NAMES[@]} runs started. Waiting for completion..."

FAILED_RUNS=()
for i in "${!RUN_PIDS[@]}"; do
  PID="${RUN_PIDS[$i]}"
  EXP_NAME="${RUN_EXPERIMENTS[$i]}"
  if wait "${PID}"; then
    echo "[$(date -Is)] Completed: ${EXP_NAME} (pid ${PID})"
  else
    echo "[$(date -Is)] FAILED: ${EXP_NAME} (pid ${PID})"
    FAILED_RUNS+=("${EXP_NAME}")
  fi
done

if [[ "${#FAILED_RUNS[@]}" -gt 0 ]]; then
  echo "Completed with failed runs (${#FAILED_RUNS[@]}/${#RUN_NAMES[@]}): ${FAILED_RUNS[*]}"
  echo "Slurm job is kept successful intentionally."
else
  echo "All ${#RUN_NAMES[@]} parallel runs completed successfully."
fi
