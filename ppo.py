import os
import time
import math
from collections import deque
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from jpype.types import JArray, JInt

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

try:
    import tkinter as tk
except Exception:
    tk = None


class PPOTrainer:
    def __init__(
        self,
        agent: nn.Module,
        supervised_agent: Optional[nn.Module],
        envs,
        args,
        writer,
        device: torch.device,
        experiment_name: str,
        get_scalar_features: Callable,
        checkpoint_frequency: int = 10,
    ):
        self.agent = agent
        self.supervised_agent = supervised_agent
        self.envs = envs
        self.args = args
        self.writer = writer
        self.device = device
        self.experiment_name = experiment_name
        self.get_scalar_features = get_scalar_features
        self.checkpoint_frequency = checkpoint_frequency



    def train(self):
        optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.args.PPO_learning_rate,
            eps=1e-5,
            weight_decay=self.args.PPO_weight_decay,
        )
        if self.args.anneal_lr:
            lr_fn = lambda frac: frac * self.args.PPO_learning_rate  # noqa: E731
        else:
            lr_fn = None

        mapsize = 16 * 16

        action_space_shape = (mapsize, self.envs.action_plane_space.shape[0])
        invalid_action_shape = (mapsize, self.envs.action_plane_space.nvec.sum() + 1)

        obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_observation_space.shape).to(
            self.device
        )
        actions = torch.zeros((self.args.num_steps, self.args.num_envs) + action_space_shape).to(self.device)
        logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        rewards_attack = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        rewards_winloss = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        rewards_score = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        invalid_action_masks = torch.zeros((self.args.num_steps, self.args.num_envs) + invalid_action_shape).to(
            self.device
        )

        global_step = 0
        start_time = time.time()
        next_obs_np, _, res = self.envs.reset()

        next_obs = torch.Tensor(next_obs_np).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)
        num_updates = self.args.total_timesteps // self.args.batch_size
        scalar_features = torch.zeros((self.args.num_steps, self.args.num_envs, 11)).to(self.device)
        z_features = torch.zeros((self.args.num_steps, self.args.num_envs, 8), dtype=torch.long).to(self.device)

        starting_update = 1

        print("PPO training started")

        for update in range(starting_update, num_updates + 1):
            if lr_fn is not None:
                frac = 1.0 - (update - 1.0) / num_updates
                lr_now = lr_fn(frac)
                optimizer.param_groups[0]["lr"] = lr_now

            for step in range(0, self.args.num_steps):
                for env_index in range(self.args.num_envs):
                    with torch.no_grad():
                        z_features[step][env_index] = self.agent.z_encoder(obs[step][env_index].view(-1))

                if self.args.render:
                    if getattr(self.args, "render_all", False):
                        Rendering.render_all_envs(self.envs)
                    else:
                        self.envs.render("human")

                global_step += self.args.num_envs

                obs[step] = next_obs

                scalar_features[step] = self.get_scalar_features(next_obs, res, self.args.num_envs)
                dones[step] = next_done

                with torch.no_grad():
                    values[step] = self.agent.get_value(obs[step], scalar_features[step], z_features[step]).flatten()
                    action, logprob, _, invalid_action_masks[step] = self.agent.get_action(
                        obs[step], scalar_features[step], z_features[step], envs=self.envs
                    )

                actions[step] = action
                logprobs[step] = logprob

                real_action = torch.cat(
                    [
                        torch.stack(
                            [torch.arange(0, mapsize).to(self.device) for _ in range(self.envs.num_envs)]
                        ).unsqueeze(2),
                        action,
                    ],
                    2,
                )

                real_action = real_action.cpu().numpy()

                valid_actions = real_action[invalid_action_masks[step][:, :, 0].bool().cpu().numpy()]
                valid_actions_counts = invalid_action_masks[step][:, :, 0].sum(1).long().cpu().numpy()

                java_valid_actions = []
                valid_action_index = 0
                for valid_action_count in valid_actions_counts:
                    java_valid_action = []
                    for _ in range(valid_action_count):
                        java_valid_action += [JArray(JInt)(valid_actions[valid_action_index])]
                        valid_action_index += 1
                    java_valid_actions += [JArray(JArray(JInt))(java_valid_action)]
                java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)

                try:
                    next_obs_np, dense_reward, attack_reward, winloss_reward, score_reward, done_flags, infos, res = (
                        self.envs.step(java_valid_actions)
                    )
                    next_obs_np = self.envs._from_microrts_obs(next_obs_np)
                    next_obs = torch.Tensor(next_obs_np).to(self.device)
                except Exception as exc:
                    exc.printStackTrace()
                    raise

                densereward = 0
                winloss_weight = 10
                score_weight = 0.2
                attack_weight = self.args.attack_reward_weight

                rewards_attack[step] = torch.Tensor(attack_reward * attack_weight).to(self.device)
                rewards_winloss[step] = torch.Tensor(winloss_reward * winloss_weight + 1).to(self.device)  # TODO: würde hier +1 gegen viele Draws helfen?
                rewards_score[step] = torch.Tensor(score_reward * score_weight).to(self.device)
                next_done = torch.Tensor(done_flags).to(self.device)

                for info in infos:
                    if "episode" not in info:
                        continue

                    if not hasattr(self.writer, "recent_winloss"):
                        self.writer.recent_winloss = deque(maxlen=150)
                    self.writer.recent_winloss.append(info["microrts_stats"]["RAIWinLossRewardFunction"])
                    winrate_over_150_games = np.mean(np.clip(self.writer.recent_winloss, 0, 1))
                    winloss_values = np.array(self.writer.recent_winloss)
                    winrate = np.mean(np.add(self.writer.recent_winloss, 1)) / 2
                    mask = winloss_values != 0
                    if mask.any():
                        winrate_no_draw = np.mean(np.clip(winloss_values[mask], 0, 1))
                    else:
                        winrate_no_draw = 0.0
                    self.writer.add_scalar("charts/winrate_over_150_games", winrate_over_150_games, global_step)
                    self.writer.add_scalar(
                        "charts/winrate_over_150_games_no_draw", winrate_no_draw, global_step
                    )
                    self.writer.add_scalar("winrate", winrate, global_step)

                    game_length = info["episode"]["l"]
                    winloss_weight = winloss_weight * (-0.00013 * game_length + 1.16)
                    episode_reward = (
                        info["microrts_stats"]["RAIWinLossRewardFunction"] * winloss_weight
                        + info["microrts_stats"]["AttackRewardFunction"] * attack_weight
                    )
                    print(
                        f"global_step={global_step}, "
                        f"episode_reward={episode_reward}, "
                        f"winrate_150={winrate_over_150_games}, "
                        f"winrate_150_no_draw={winrate_no_draw}, "
                        f"winrate={winrate}"
                    )
                    self.writer.add_scalar("charts/old_episode_reward", info["episode"]["r"], global_step)
                    self.writer.add_scalar("charts/Game_length", game_length, global_step)
                    self.writer.add_scalar("charts/Episode_reward", episode_reward, global_step)
                    self.writer.add_scalar(
                        "charts/AttackReward", info["microrts_stats"]["AttackRewardFunction"] * attack_weight, global_step
                    )
                    self.writer.add_scalar(
                        "charts/WinLossRewardFunction",
                        info["microrts_stats"]["RAIWinLossRewardFunction"] * winloss_weight,
                        global_step,
                    )
                    break

            with torch.no_grad():
                last_value = self.agent.get_value(next_obs.to(self.device), scalar_features[step], z_features[step]).reshape(1, -1)
                advantages = torch.zeros_like(rewards_winloss).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = last_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = (
                        rewards_winloss[t]
                        + rewards_attack[t]
                        + self.args.gamma * nextvalues * nextnonterminal
                        - values[t]
                    )
                    advantages[t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            flat_z = z_features.reshape(-1, 8)
            flat_scalar = scalar_features.reshape(-1, 11)
            flat_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            flat_actions = actions.reshape((-1,) + action_space_shape)
            flat_logprobs = logprobs.reshape(-1)
            flat_advantages = advantages.reshape(-1)
            flat_returns = returns.reshape(-1)
            flat_values = values.reshape(-1)
            flat_invalid_masks = invalid_action_masks.reshape((-1,) + invalid_action_shape)

            indices = np.arange(self.args.batch_size)
            for epoch_pi in range(self.args.update_epochs):
                np.random.shuffle(indices)

                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    minibatch_ind = indices[start:end]
                    mb_advantages = flat_advantages[minibatch_ind]

                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    _, new_logproba, entropy, _ = self.agent.get_action(
                        flat_obs[minibatch_ind],
                        flat_scalar[minibatch_ind],
                        flat_z[minibatch_ind],
                        flat_actions.long()[minibatch_ind],
                        flat_invalid_masks[minibatch_ind],
                        self.envs,
                    )
                    ratio = (new_logproba - flat_logprobs[minibatch_ind]).exp()

                    approx_kl = (flat_logprobs[minibatch_ind] - new_logproba).mean()

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef, 1 + self.args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    entropy_loss = entropy.mean()

                    new_values = self.agent.get_value(
                        flat_obs[minibatch_ind], flat_scalar[minibatch_ind], flat_z[minibatch_ind]
                    ).view(-1)
                    if self.args.clip_vloss:
                        v_loss_unclipped = (new_values - flat_returns[minibatch_ind]) ** 2
                        v_clipped = flat_values[minibatch_ind] + torch.clamp(
                            new_values - flat_values[minibatch_ind], -self.args.clip_coef, self.args.clip_coef
                        )
                        v_loss_clipped = (v_clipped - flat_returns[minibatch_ind]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((new_values - flat_returns[minibatch_ind]) ** 2)

                    if self.supervised_agent is not None:
                        with torch.no_grad():
                            _, sl_logprobs, _, _ = self.supervised_agent.get_action(
                                flat_obs[minibatch_ind],
                                flat_scalar[minibatch_ind],
                                flat_z[minibatch_ind],
                                flat_actions.long()[minibatch_ind],
                                flat_invalid_masks[minibatch_ind],
                                self.envs,
                            )
                        kl_div = F.kl_div(new_logproba, sl_logprobs, log_target=True, reduction="batchmean")
                    else:
                        kl_div = torch.tensor(0.0, device=self.device)
                    kl_loss = self.args.kl_coeff * kl_div

                    loss = pg_loss - self.args.ent_coef * entropy_loss + self.args.vf_coef * v_loss + kl_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)
                    optimizer.step()

            if self.args.prod_mode and update % self.checkpoint_frequency == 0:
                print("checkpoint")
                os.makedirs(f"models/{self.experiment_name}/ppo", exist_ok=True)
                torch.save(self.agent.state_dict(), f"models/{self.experiment_name}/ppo/agent.pt")

                if update < 1500:
                    if update % 100 == 0:
                        self._save_progress(update)
                else:
                    self._save_progress(update)

            self.writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            self.writer.add_scalar("charts/update", update, global_step)
            self.writer.add_scalar("losses/value_loss", self.args.vf_coef * v_loss.item(), global_step)
            self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            self.writer.add_scalar("losses/kl_loss", kl_loss.item(), global_step)
            self.writer.add_scalar("losses/total_loss", loss.item(), global_step)
            self.writer.add_scalar("losses/entropy_loss", self.args.ent_coef * entropy_loss.item(), global_step)
            self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            if self.args.kle_stop or self.args.kle_rollback:
                self.writer.add_scalar("debug/pg_stop_iter", epoch_pi, global_step)
            sps = int(global_step / (time.time() - start_time))
            self.writer.add_scalar("charts/sps", sps, global_step)
            print("SPS:", sps)

        self.envs.close()
        self.writer.close()

    def _save_progress(self, update: int):
        os.makedirs(f"models/PPO_finished_training/new/{self.experiment_name}_update_{update}", exist_ok=True)
        torch.save(
            self.agent.state_dict(),
            f"models/PPO_finished_training/new/{self.experiment_name}_update_{update}/agent.pt",
        )


class Rendering:
    _screen_size = None
    _vec_client_cache = {}
    _clients_cache = {}
    _frame_size = None
    _tk_root = None
    _tk_canvas = None
    _tk_canvas_img = None
    _viewer_size = None
    _viewer_disabled = False
    _render_interval = None
    _last_render_ts = 0.0

    @staticmethod
    def render_all_envs(env_transform):
        vec_client = Rendering._vec_client_cache.get(id(env_transform))
        if vec_client is None:
            vec_client = Rendering._locate_vec_client(env_transform)
            Rendering._vec_client_cache[id(env_transform)] = vec_client

        if vec_client is not None:
            clients = Rendering._get_clients(vec_client)
            if clients and not Rendering._viewer_disabled:
                if not Rendering._should_render_now():
                    return
                if Rendering._render_single_window(clients):
                    return

        try:
            env_transform.render()
        except Exception:
            pass

    @staticmethod
    def _locate_vec_client(env_transform):
        visited = set()
        stack = [env_transform]
        while stack:
            current = stack.pop()
            if current is None:
                continue
            current_id = id(current)
            if current_id in visited:
                continue
            visited.add(current_id)

            try:
                interface = getattr(current, "interface", None)
                vec_client = getattr(interface, "vec_client", None) if interface is not None else None
                if vec_client is not None:
                    return vec_client
            except Exception:
                pass

            try:
                vec_client = getattr(current, "vec_client", None)
                if vec_client is not None:
                    return vec_client
            except Exception:
                pass

            for attr in ("venv", "env", "envs", "wrapped_env"):
                try:
                    next_obj = getattr(current, attr)
                    if next_obj is not None:
                        stack.append(next_obj)
                except Exception:
                    continue
        return None

    @staticmethod
    def _collect_clients(vec_client):
        clients = []
        for attr in ("selfPlayClients", "clients"):
            try:
                collection = getattr(vec_client, attr)
            except Exception:
                collection = None
            if collection:
                clients.extend([client for client in collection if client is not None])
        return clients

    @staticmethod
    def _get_clients(vec_client):
        cache_key = id(vec_client)
        cached_entry = Rendering._clients_cache.get(cache_key)
        if cached_entry:
            cached_clients, cached_len = cached_entry
            if cached_len == Rendering._count_vec_clients(vec_client):
                return cached_clients
        clients = Rendering._collect_clients(vec_client)
        if clients:
            Rendering._clients_cache[cache_key] = (clients, len(clients))
        else:
            Rendering._clients_cache.pop(cache_key, None)
        return clients

    @staticmethod
    def _count_vec_clients(vec_client):
        total = 0
        for attr in ("selfPlayClients", "clients"):
            try:
                collection = getattr(vec_client, attr)
            except Exception:
                collection = None
            if collection:
                total += len(collection)
        return total

    @staticmethod
    def _render_single_window(clients):
        if Rendering._viewer_disabled:
            return False
        if not clients or tk is None or Image is None or ImageTk is None:
            return False

        frames = []
        for client in clients:
            try:
                frame_bytes = client.render(True)
            except Exception:
                continue
            if frame_bytes is None:
                continue
            image = Rendering._bytes_to_image(frame_bytes)
            if image is not None:
                frames.append(image)

        if not frames:
            return False

        mosaic = Rendering._tile_images(frames)
        if mosaic is None:
            return False

        return Rendering._show_single_window(mosaic)

    @staticmethod
    def _bytes_to_image(frame_bytes):
        if Image is None:
            return None
        Rendering._ensure_frame_size(len(frame_bytes))
        if Rendering._frame_size is None:
            return None
        width, height = Rendering._frame_size
        try:
            buffer = frame_bytes.tobytes() if hasattr(frame_bytes, "tobytes") else bytes(frame_bytes)
            return Image.frombytes("RGB", (width, height), buffer, "raw", "BGR")
        except Exception:
            return None

    @staticmethod
    def _ensure_frame_size(byte_length=None):
        if Rendering._frame_size:
            return

        width = int(os.environ.get("MICRORTS_FRAME_WIDTH", 640))
        height = int(os.environ.get("MICRORTS_FRAME_HEIGHT", 640))

        if byte_length:
            pixels = byte_length // 3
            side = int(math.isqrt(pixels))
            if side * side == pixels and side > 0:
                width = height = side

        Rendering._frame_size = (max(1, width), max(1, height))

    @staticmethod
    def _tile_images(images):
        if not images:
            return None
        Rendering._ensure_frame_size()
        target_w, target_h = Rendering._frame_size or images[0].size
        desired_cols = Rendering._desired_columns()
        cols = max(1, min(len(images), desired_cols))
        rows = math.ceil(len(images) / cols)
        gap = Rendering._get_tile_gap()
        mosaic_w = cols * target_w + gap * max(cols - 1, 0)
        mosaic_h = rows * target_h + gap * max(rows - 1, 0)
        mosaic = Image.new("RGB", (mosaic_w, mosaic_h))
        for idx, img in enumerate(images):
            row = idx // cols
            col = idx % cols
            x = col * (target_w + gap)
            y = row * (target_h + gap)
            if img.size != (target_w, target_h):
                img = img.resize((target_w, target_h))
            mosaic.paste(img, (x, y))
        return mosaic

    @staticmethod
    def _show_single_window(image):
        try:
            root, canvas = Rendering._get_tk_window()
        except Exception:
            return False
        if root is None or canvas is None or ImageTk is None:
            return False
        try:
            if not root.winfo_exists():
                Rendering._destroy_tk_window()
                return False
        except Exception:
            Rendering._destroy_tk_window()
            return False
        try:
            photo = ImageTk.PhotoImage(image=image)
        except Exception:
            return False
        Rendering._tk_canvas = canvas
        canvas.photo = photo
        if Rendering._tk_canvas_img is None:
            Rendering._tk_canvas_img = canvas.create_image(0, 0, anchor="nw", image=photo)
        else:
            canvas.itemconfigure(Rendering._tk_canvas_img, image=photo)
        canvas.configure(scrollregion=(0, 0, image.width, image.height))
        try:
            root.update_idletasks()
            root.update()
        except tk.TclError:
            Rendering._destroy_tk_window()
            return False
        except Exception:
            return False
        return True

    @staticmethod
    def _get_tk_window():
        if tk is None or Rendering._viewer_disabled:
            return None, None
        if Rendering._tk_root is not None:
            try:
                if not Rendering._tk_root.winfo_exists():
                    Rendering._destroy_tk_window()
            except Exception:
                Rendering._destroy_tk_window()
        if Rendering._tk_root is None or Rendering._tk_canvas is None:
            try:
                Rendering._tk_root = tk.Tk()
                Rendering._tk_root.title("MicroRTS Environments")
                viewer_w, viewer_h = Rendering._get_viewer_size()
                Rendering._tk_root.geometry(f"{viewer_w}x{viewer_h}")
                Rendering._tk_root.minsize(400, 300)
                Rendering._tk_root.protocol("WM_DELETE_WINDOW", lambda: Rendering._destroy_tk_window(permanent=True))

                container = tk.Frame(Rendering._tk_root)
                container.grid(row=0, column=0, sticky="nsew")
                Rendering._tk_root.grid_rowconfigure(0, weight=1)
                Rendering._tk_root.grid_columnconfigure(0, weight=1)
                container.grid_rowconfigure(0, weight=1)
                container.grid_columnconfigure(0, weight=1)

                canvas = tk.Canvas(container, highlightthickness=0)
                scroll_y = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
                scroll_x = tk.Scrollbar(container, orient="horizontal", command=canvas.xview)
                canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
                canvas.grid(row=0, column=0, sticky="nsew")
                scroll_y.grid(row=0, column=1, sticky="ns")
                scroll_x.grid(row=1, column=0, sticky="ew")

                canvas_width = max(200, viewer_w - 4)
                canvas_height = max(200, viewer_h - 4)
                canvas.config(width=canvas_width, height=canvas_height)
                canvas.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))

                Rendering._tk_canvas = canvas
            except Exception:
                Rendering._destroy_tk_window()
        return Rendering._tk_root, Rendering._tk_canvas

    @staticmethod
    def _destroy_tk_window(permanent=False):
        if Rendering._tk_root is not None:
            try:
                if Rendering._tk_root.winfo_exists():
                    Rendering._tk_root.destroy()
            except Exception:
                pass
        Rendering._tk_root = None
        Rendering._tk_canvas = None
        Rendering._tk_canvas_img = None
        if permanent:
            Rendering._viewer_disabled = True

    @staticmethod
    def _get_screen_size():
        if Rendering._screen_size:
            return Rendering._screen_size

        width = int(os.environ.get("MICRORTS_SCREEN_WIDTH", 1920))
        height = int(os.environ.get("MICRORTS_SCREEN_HEIGHT", 1080))
        if width <= 0 or height <= 0:
            width, height = 1920, 1080

        if tk is not None:
            root = None
            try:
                root = tk.Tk()
                root.withdraw()
                width = root.winfo_screenwidth()
                height = root.winfo_screenheight()
            except Exception:
                pass
            finally:
                if root is not None:
                    try:
                        root.destroy()
                    except Exception:
                        pass

        Rendering._screen_size = (int(width), int(height))
        return Rendering._screen_size

    @staticmethod
    def _get_tile_gap():
        return -50

    @staticmethod
    def _get_viewer_size():
        if Rendering._viewer_size:
            return Rendering._viewer_size
        screen_w, screen_h = Rendering._get_screen_size()
        width = int(os.environ.get("MICRORTS_VIEWER_WIDTH", 1280))
        height = int(os.environ.get("MICRORTS_VIEWER_HEIGHT", 720))
        width = max(400, min(screen_w, width))
        height = max(300, min(screen_h, height))
        Rendering._viewer_size = (width, height)
        return Rendering._viewer_size

    @staticmethod
    def _desired_columns():
        try:
            return max(1, int(os.environ.get("MICRORTS_VIEWER_COLS", 6)))
        except Exception:
            return 6

    @staticmethod
    def _should_render_now():
        interval = Rendering._get_render_interval()
        now = time.time()
        if interval <= 0:
            Rendering._last_render_ts = now
            return True
        if now - Rendering._last_render_ts < interval:
            return False
        Rendering._last_render_ts = now
        return True

    @staticmethod
    def _get_render_interval():
        if Rendering._render_interval is not None:
            return Rendering._render_interval
        Rendering._render_interval = Rendering._compute_render_interval()
        return Rendering._render_interval

    @staticmethod
    def _compute_render_interval():
        try:
            fps = float(os.environ.get("MICRORTS_VIEWER_FPS", 6.0))
        except Exception:
            fps = 6.0
        if fps <= 0:
            return 0.0
        fps = min(60.0, fps)
        return 1.0 / fps
