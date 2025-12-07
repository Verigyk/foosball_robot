#!/usr/bin/env python3
"""
Version OPTIMISÉE pour entraînement rapide
- Physique simplifiée
- LSTM optionnel (deux phases)
- Configuration GPU
- Batch size optimisé
"""

import sys
sys.path.insert(0, '.')

from claude_code_rllib import SelfPlayFoosballEnv, FoosballMARLEnv
import gymnasium as gym
import numpy as np
import pybullet as p


class FastFoosballMARLEnv(FoosballMARLEnv):
    """Version optimisée de l'environnement avec physique simplifiée"""
    
    def __init__(self, render_mode=None, fast_physics=True):
        # Appeler le constructeur parent AVANT de modifier la physique
        super().__init__(render_mode=render_mode)
        
        # Optimisations physique si demandé
        if fast_physics:
            # Physique simplifiée pour entraînement rapide
            p.setPhysicsEngineParameter(
                fixedTimeStep=1./120.,      # 120Hz au lieu de 240Hz (2x plus rapide)
                numSubSteps=5,              # 5 au lieu de 10 (2x plus rapide)
                numSolverIterations=50,     # 50 au lieu de 100 (2x plus rapide)
            )
            
            # Episodes plus courts
            self.max_steps = 500  # Au lieu de 1000
            
            print("⚡ Mode Fast Physics activé (2-3x plus rapide)")


class FastSelfPlayFoosballEnv(SelfPlayFoosballEnv):
    """Version optimisée avec physique rapide"""
    
    def __init__(self, config=None):
        # Configurer l'environnement de base
        render_mode = "human" if (config and config.get("render", False)) else None
        fast_physics = config.get("fast_physics", True) if config else True
        
        # Créer l'environnement optimisé
        self.base_env = FastFoosballMARLEnv(render_mode=render_mode, fast_physics=fast_physics)
        
        # Copier les attributs nécessaires
        self._agent_ids = set([f"agent_{i}" for i in range(8)])
        self.possible_agents = [f"agent_{i}" for i in range(8)]
        self.agents = self.possible_agents.copy()
        
        # Espaces
        single_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        single_action_space = gym.spaces.Box(
            low=np.array([-0.18, -np.pi], dtype=np.float32),
            high=np.array([0.18, np.pi], dtype=np.float32),
            dtype=np.float32
        )
        
        self.observation_space = gym.spaces.Dict({
            agent_id: single_obs_space for agent_id in self._agent_ids
        })
        self.action_space = gym.spaces.Dict({
            agent_id: single_action_space for agent_id in self._agent_ids
        })
        
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
    
    def reset(self, *, seed=None, options=None):
        obs_team1, info = self.base_env.reset(seed=seed, options=options)
        obs = {f"agent_{i}": obs_team1[f"agent_{i}"] for i in range(4)}
        for i in range(4):
            obs[f"agent_{i+4}"] = self.base_env._get_opponent_observation(i)
        return obs, info
    
    def step(self, action_dict):
        # Appliquer actions (simplifié - moins de steps de simulation)
        for agent_id in range(4):
            action = action_dict[f"agent_{agent_id}"]
            config = self.base_env.agent_configs[agent_id]
            
            target_slide = np.clip(float(action[0]), config["slide_min"], config["slide_max"])
            target_rotation = np.clip(float(action[1]), -np.pi, np.pi)
            
            p.setJointMotorControl2(
                bodyIndex=self.base_env.table_id, jointIndex=config["slide_idx"],
                controlMode=p.POSITION_CONTROL, targetPosition=target_slide,
                force=500.0, maxVelocity=2.0
            )
            p.setJointMotorControl2(
                bodyIndex=self.base_env.table_id, jointIndex=config["rotate_idx"],
                controlMode=p.POSITION_CONTROL, targetPosition=target_rotation,
                force=200.0, maxVelocity=30.0
            )
        
        for opponent_id in range(4):
            action = action_dict[f"agent_{opponent_id+4}"]
            opp_config = self.base_env.opponent_joints[opponent_id]
            
            target_slide = np.clip(float(action[0]), opp_config["slide_min"], opp_config["slide_max"])
            target_rotation = np.clip(float(action[1]), -np.pi, np.pi)
            
            p.setJointMotorControl2(
                bodyIndex=self.base_env.table_id, jointIndex=opp_config["slide_idx"],
                controlMode=p.POSITION_CONTROL, targetPosition=target_slide,
                force=500.0, maxVelocity=2.0
            )
            p.setJointMotorControl2(
                bodyIndex=self.base_env.table_id, jointIndex=opp_config["rotate_idx"],
                controlMode=p.POSITION_CONTROL, targetPosition=target_rotation,
                force=200.0, maxVelocity=30.0
            )
        
        # Simulation (OPTIMISÉ: moins de steps)
        self.base_env.current_step += 1
        for _ in range(2):  # 2 au lieu de 4
            p.stepSimulation()
        
        # Observer
        global_obs = self.base_env._get_observation()
        ball_x = global_obs[0]
        
        obs = {f"agent_{i}": global_obs for i in range(4)}
        for i in range(4):
            obs[f"agent_{i+4}"] = self.base_env._get_opponent_observation(i)
        
        # Récompenses
        base_reward = -0.1
        progress_reward = (ball_x - self.base_env.previous_ball_x) * 10
        distance_to_goal = self.base_env.goal_line_right - ball_x
        distance_reward = -distance_to_goal * 0.5
        
        team1_reward = base_reward + progress_reward + distance_reward
        
        rewards = {}
        for i in range(4):
            rewards[f"agent_{i}"] = team1_reward
            rewards[f"agent_{i+4}"] = -team1_reward
        
        # Terminaison
        goal_scored = False
        ball_out = False
        
        if self.base_env._is_ball_out_of_bounds():
            ball_out = True
            for i in range(8):
                rewards[f"agent_{i}"] = -5.0
        elif ball_x >= self.base_env.goal_line_right:
            self.base_env.team1_goals += 1
            goal_scored = True
            for i in range(4):
                rewards[f"agent_{i}"] += 10.0
                rewards[f"agent_{i+4}"] -= 10.0
        elif ball_x <= self.base_env.goal_line_left:
            self.base_env.team2_goals += 1
            goal_scored = True
            for i in range(4):
                rewards[f"agent_{i}"] -= 10.0
                rewards[f"agent_{i+4}"] += 10.0
        
        truncated = self.base_env.current_step >= self.base_env.max_steps
        episode_done = goal_scored or ball_out or truncated
        
        dones = {f"agent_{i}": episode_done for i in range(8)}
        dones["__all__"] = episode_done
        
        self.base_env.previous_ball_x = ball_x
        info = {}
        
        return obs, rewards, dones, dones, info
    
    def close(self):
        self.base_env.close()


def train_fast(
    num_iterations: int = 100,
    num_workers: int = 4,
    use_lstm: bool = False,
    lstm_size: int = 128,
    use_gpu: bool = False,
    fast_physics: bool = True,
    render: bool = False
):
    """
    Entraînement OPTIMISÉ
    
    Args:
        num_iterations: Nombre d'itérations
        num_workers: Nombre de workers parallèles
        use_lstm: Utiliser LSTM ou feedforward
        lstm_size: Taille LSTM (128 ou 256)
        use_gpu: Utiliser GPU si disponible
        fast_physics: Physique simplifiée (plus rapide)
        render: Afficher la simulation (très lent)
    """
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec
    import os
    import warnings
    import logging
    
    print("=" * 70)
    print("⚡ ENTRAÎNEMENT OPTIMISÉ - BABY-FOOT RL")
    print("=" * 70)
    print(f"Itérations: {num_iterations}")
    print(f"Workers: {num_workers}")
    print(f"LSTM: {'OUI' if use_lstm else 'NON'} ({lstm_size if use_lstm else 'N/A'})")
    print(f"GPU: {'OUI' if use_gpu else 'NON'}")
    print(f"Fast Physics: {'OUI' if fast_physics else 'NON'}")
    print(f"Render: {'OUI (lent)' if render else 'NON'}")
    print("=" * 70)
    
    # Suppress warnings
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning"
    warnings.filterwarnings("ignore")
    logging.getLogger("ray").setLevel(logging.ERROR)
    
    # Détection GPU
    num_gpus = 0
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                num_gpus = 1
                print(f"✅ GPU détecté: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️  GPU demandé mais pas disponible, utilisation CPU")
        except:
            print("⚠️  PyTorch GPU non disponible")
    
    ray.init(
        ignore_reinit_error=True,
        num_cpus=num_workers+1,
        num_gpus=num_gpus,
        logging_level=logging.ERROR,
        _metrics_export_port=None,
        _system_config={"metrics_report_interval_ms": 0}
    )
    
    # Configuration modèle
    if use_lstm:
        model_config = {
            "fcnet_hiddens": [lstm_size, lstm_size],
            "fcnet_activation": "relu",
            "use_lstm": True,
            "lstm_cell_size": lstm_size,
            "max_seq_len": 10 if lstm_size == 128 else 20,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": True,
        }
        print(f"🧠 Réseau: FC[{lstm_size},{lstm_size}] → LSTM[{lstm_size}]")
    else:
        model_config = {
            "fcnet_hiddens": [128, 128],
            "fcnet_activation": "relu",
            "use_lstm": False,
        }
        print(f"🧠 Réseau: FC[128,128] (Feedforward)")
    
    # Policies
    policies = {
        "shared_policy": PolicySpec(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=np.array([-0.18, -np.pi], dtype=np.float32),
                high=np.array([0.18, np.pi], dtype=np.float32),
                dtype=np.float32
            ),
            config={"model": model_config}
        ),
    }
    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"
    
    # Configuration OPTIMISÉE
    env_config = {
        "render": render,
        "fast_physics": fast_physics
    }
    
    config = (
        PPOConfig()
        .environment(FastSelfPlayFoosballEnv, env_config=env_config)
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(
            lr=5e-4 if not use_lstm else 3e-4,  # LR plus élevé sans LSTM
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            train_batch_size=8000,  # OPTIMISÉ: 8000 au lieu de 4000
            num_sgd_iter=5,         # OPTIMISÉ: 5 au lieu de 10
            model=model_config,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .env_runners(
            num_env_runners=1 if render else num_workers,
            num_envs_per_env_runner=1,
        )
        .resources(
            num_gpus=num_gpus,
            num_cpus_per_worker=1,
        )
        .debugging(log_level="ERROR")
    )
    
    print(f"\n⚙️  Configuration:")
    print(f"   Batch size: 8000 (optimisé)")
    print(f"   SGD iter: 5 (optimisé)")
    print(f"   Workers: {num_workers}")
    print(f"   GPU: {num_gpus}")
    
    # Construire
    print("\n[Construction PPO...]")
    algo = config.build()
    print("  ✓ Algorithme prêt\n")
    
    # Entraînement
    print("=" * 70)
    print(f"DÉMARRAGE ({num_iterations} itérations)")
    print("=" * 70)
    
    import time
    start_time = time.time()
    
    try:
        for i in range(num_iterations):
            iter_start = time.time()
            result = algo.train()
            iter_time = time.time() - iter_start
            
            # Barre de progression
            progress = (i + 1) / num_iterations
            bar_length = 50
            filled = int(bar_length * progress)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            # Temps estimé restant
            elapsed = time.time() - start_time
            if i > 0:
                avg_time_per_iter = elapsed / (i + 1)
                remaining_iters = num_iterations - (i + 1)
                eta_seconds = avg_time_per_iter * remaining_iters
                eta_minutes = int(eta_seconds / 60)
                eta_hours = eta_minutes // 60
                eta_mins = eta_minutes % 60
                
                if eta_hours > 0:
                    eta_str = f"{eta_hours}h{eta_mins:02d}m"
                else:
                    eta_str = f"{eta_mins}m"
            else:
                eta_str = "calculant..."
            
            # Affichage compact avec barre
            reward = result['env_runners']['episode_reward_mean']
            ep_len = result['env_runners']['episode_len_mean']
            
            print(f"\r[{bar}] {i+1}/{num_iterations} ({progress*100:.1f}%) | "
                  f"Reward: {reward:>7.2f} | Len: {ep_len:>6.1f} | "
                  f"ETA: {eta_str:<8} | {iter_time:.1f}s/iter", end="", flush=True)
            
            # Affichage détaillé tous les 10 iterations
            if (i + 1) % 10 == 0 or i == 0:
                print()  # Nouvelle ligne
            
            # Checkpoint tous les 20
            if (i + 1) % 20 == 0:
                checkpoint_dir = algo.save()
                print(f"\n  💾 Checkpoint sauvegardé: ...{checkpoint_dir[-40:]}")
        
        print("\n" + "=" * 70)
        print("✓ ENTRAÎNEMENT TERMINÉ!")
        print("=" * 70)
        
        final_checkpoint = algo.save()
        print(f"✓ Modèle final: {final_checkpoint}")
        
        return algo, final_checkpoint
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrompu")
        checkpoint = algo.save()
        print(f"✓ Checkpoint: {checkpoint}")
        return algo, checkpoint
    
    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    import sys
    import argparse
    
    # Détecter GPU
    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except:
        pass
    
    # Détecter nombre de cores
    import os
    num_cores = os.cpu_count() or 4
    num_workers_default = max(2, min(num_cores - 4, 12))  # Entre 2 et 12
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        # Parser les arguments
        render = False
        num_workers = num_workers_default
        
        for i, arg in enumerate(sys.argv):
            if arg == "--render" or arg == "-r":
                render = True
            elif arg in ["--workers", "-w", "--threads", "-t"]:
                if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
                    num_workers = int(sys.argv[i + 1])
                    print(f"✓ Nombre de threads/workers défini: {num_workers}")
        
        if command == "fast":
            # Mode rapide: feedforward + fast physics
            num_iter = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 50
            print("🚀 Mode RAPIDE: Feedforward + Fast Physics")
            train_fast(
                num_iterations=num_iter,
                num_workers=num_workers if not render else 1,
                use_lstm=False,
                use_gpu=use_gpu,
                fast_physics=True,
                render=render
            )
        
        elif command == "lstm":
            # Mode LSTM: LSTM petit + fast physics
            num_iter = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 100
            print("🧠 Mode LSTM: LSTM 128 + Fast Physics")
            train_fast(
                num_iterations=num_iter,
                num_workers=num_workers if not render else 1,
                use_lstm=True,
                lstm_size=128,
                use_gpu=use_gpu,
                fast_physics=True,
                render=render
            )
        
        elif command == "quality":
            # Mode qualité: LSTM grand + physique normale
            num_iter = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 100
            print("✨ Mode QUALITÉ: LSTM 256 + Physique réaliste")
            train_fast(
                num_iterations=num_iter,
                num_workers=max(2, num_workers // 2) if not render else 1,
                use_lstm=True,
                lstm_size=256,
                use_gpu=use_gpu,
                fast_physics=False,
                render=render
            )
        
        else:
            print("Usage:")
            print("  python claude_code_rllib_fast.py fast [iterations] [options]    # Mode rapide")
            print("  python claude_code_rllib_fast.py lstm [iterations] [options]    # Mode LSTM")
            print("  python claude_code_rllib_fast.py quality [iterations] [options] # Mode qualité")
            print("")
            print("Options:")
            print("  --render, -r              Afficher la simulation (TRÈS lent)")
            print("  --threads N, -t N         Nombre de threads/workers parallèles")
            print("  --workers N, -w N         Alias pour --threads")
            print("")
            print("Exemples:")
            print("  python claude_code_rllib_fast.py fast 50")
            print("  python claude_code_rllib_fast.py lstm 100 --threads 8")
            print("  python claude_code_rllib_fast.py quality 100 -t 4")
            print("  python claude_code_rllib_fast.py fast 10 --render  # Avec visualisation")
    else:
        print("="*70)
        print("⚡ ENTRAÎNEMENT OPTIMISÉ - BABY-FOOT RL")
        print("="*70)
        print("")
        print("Modes disponibles:")
        print("  fast    - Feedforward + Fast Physics (5-10x plus rapide)")
        print("  lstm    - LSTM 128 + Fast Physics (3-5x plus rapide)")
        print("  quality - LSTM 256 + Physique réaliste (qualité max)")
        print("")
        print("Usage:")
        print("  python claude_code_rllib_fast.py <mode> [iterations] [options]")
        print("")
        print("Options:")
        print("  --render, -r              Afficher la simulation PyBullet (TRÈS lent)")
        print("                            Utilise 1 worker au lieu de plusieurs")
        print("  --threads N, -t N         Nombre de threads/workers parallèles")
        print("  --workers N, -w N         Alias pour --threads")
        print("")
        print("Exemples:")
        print("  python claude_code_rllib_fast.py fast 50          # Rapide")
        print("  python claude_code_rllib_fast.py lstm 100         # Recommandé")
        print("  python claude_code_rllib_fast.py quality 100      # Qualité")
        print("  python claude_code_rllib_fast.py fast 10 --render # Avec GUI")
        print("  python claude_code_rllib_fast.py lstm 100 -t 8    # 8 threads")
        print("")
        print(f"Configuration détectée:")
        print(f"  CPU cores: {num_cores}")
        print(f"  Workers par défaut: {num_workers_default}")
        print(f"  GPU disponible: {'OUI' if use_gpu else 'NON'}")
        print("")
        print("💡 Notes:")
        print("  - Par défaut utilise (cores - 4) threads, max 12")
        print("  - --render force 1 thread (affichage temps réel)")
        print("  - Plus de threads = plus rapide (jusqu'à saturation CPU)")
        print("  - Recommandé: 4-8 threads pour la plupart des machines")
