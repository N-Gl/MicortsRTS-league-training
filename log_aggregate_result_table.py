
# used in main.py
import time
from typing import List, Tuple

import numpy as np

class Logger:
    @staticmethod
    def log_local_results(
        opponent_name: str,
        local_stats: dict,
        local_episode_rewards: List[float],
        aggregate_stats: dict,
        aggregate_episode_rewards: List[float],
        global_step: int,
        start_time: int
    ):
        total_games = sum(local_stats.values())
        avg_reward = float(np.mean(local_episode_rewards)) if local_episode_rewards else 0.0
        win_rate = local_stats["win"] / total_games if total_games else 0.0
        draw_rate = local_stats["draw"] / total_games if total_games else 0.0
        loss_rate = local_stats["loss"] / total_games if total_games else 0.0
    
        aggregate_episode_rewards.extend(local_episode_rewards)
        for key in aggregate_stats:
            aggregate_stats[key] += local_stats[key]
    
        print(
            f"Evaluation vs {opponent_name} over {total_games} games | "
            f"win: {local_stats['win']} ({win_rate:.2%}), "
            f"draw: {local_stats['draw']} ({draw_rate:.2%}), "
            f"loss: {local_stats['loss']} ({loss_rate:.2%}), "
            f"avg reward: {avg_reward:.3f}"
        )
        sps = int(global_step / (time.time() - start_time))
        print("SPS:", sps)
        
    @staticmethod
    def build_table_row(opponent_name: str, local_stats: dict, local_episode_rewards: List[float]):
        total_games = sum(local_stats.values())
        avg_reward = float(np.mean(local_episode_rewards)) if local_episode_rewards else 0.0
        win_rate = local_stats["win"] / total_games if total_games else 0.0
        draw_rate = local_stats["draw"] / total_games if total_games else 0.0
        loss_rate = local_stats["loss"] / total_games if total_games else 0.0
        return (
            opponent_name,
            total_games,
            local_stats["win"],
            local_stats["draw"],
            local_stats["loss"],
            win_rate,
            draw_rate,
            loss_rate,
            avg_reward,
        )

    @staticmethod
    def log_aggregate_results(aggregate_stats: dict, aggregate_episode_rewards: List[float], writer) -> None:
        total_games = sum(aggregate_stats.values())
        avg_reward = float(np.mean(aggregate_episode_rewards)) if aggregate_episode_rewards else 0.0
        win_rate = aggregate_stats["win"] / total_games if total_games else 0.0
        draw_rate = aggregate_stats["draw"] / total_games if total_games else 0.0
        loss_rate = aggregate_stats["loss"] / total_games if total_games else 0.0

        print(
            f"Aggregate evaluation over {total_games} games | "
            f"win: {aggregate_stats['win']} ({win_rate:.2%}), "
            f"draw: {aggregate_stats['draw']} ({draw_rate:.2%}), "
            f"loss: {aggregate_stats['loss']} ({loss_rate:.2%}), "
            f"avg reward: {avg_reward:.2f}"
        )

        if writer is not None:
            writer.add_scalar("eval/win_rate", win_rate, total_games)
            writer.add_scalar("eval/draw_rate", draw_rate, total_games)
            writer.add_scalar("eval/loss_rate", loss_rate, total_games)
            writer.add_scalar("eval/avg_episode_reward", avg_reward, total_games)

    @staticmethod
    def log_wandb_summary(
        args,
        opponent_table_rows: List[Tuple],
        aggregate_stats: dict = None,
        no_reward: bool = False,
        step: int = None,
        table_name: str = "eval/opponent_summary",
        with_name: str = None,
        table = None
    ) -> None:
        if not (args.prod_mode and opponent_table_rows):
            return
        import wandb

        if step is None:
            step = sum(aggregate_stats.values())

        # Keep a stable schema to avoid UI panel errors if the same table key
        # previously had avg_reward. For no_reward, we still include the column
        # but fill it with NaN.
        columns = ["agent", "opponent", "games", "wins", "draws", "losses", "win_rate", "draw_rate", "loss_rate", "avg_reward"]

        opponent_table = table if table is not None else wandb.Table(columns=columns)
        for row in opponent_table_rows:
            # Prepend agent name once; then ensure the last column is NaN when no_reward is set.
            base_row = (with_name if with_name else "nan",) + tuple(row)
            if no_reward:
                if len(base_row) == len(columns):
                    row_with_reward = base_row
                elif len(base_row) == len(columns) - 1:
                    row_with_reward = (*base_row, float("nan"))
                else:
                    raise ValueError(f"Unexpected row length {len(base_row)} for columns {len(columns)}")
            else:
                row_with_reward = base_row

            opponent_table.add_data(*row_with_reward)
        # if with_name:
        #     table_name = f"{table_name}_{with_name}"
        wandb.log({table_name: opponent_table}, step=step)

        # try:
        #     wandb.log({table_name: opponent_table}, step=step)
        # except Exception as exc:
        #     # if WANDB cannot reach the network we skip the upload.
        #     try:
        #         import requests
        #         network_errors = (requests.exceptions.RequestException,)
        #     except Exception:
        #         network_errors = tuple()
# 
        #     try:
        #         from wandb.errors import CommError
        #         comm_errors = (CommError,)
        #     except Exception:
        #         comm_errors = tuple()
# 
        #     if isinstance(exc, tuple(set(network_errors + comm_errors))):
        #         print(f"[wandb] Skipping table upload '{table_name}' at step {step} due to WANDB connectivity issue: {exc}")
        #         return opponent_table
        #     raise
        return opponent_table # TODO: benutze diesen return auch in League training

        
