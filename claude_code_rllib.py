import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
from typing import Dict, Tuple, List
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces

class FoosballMARLEnv(gym.Env):
    """
    Environnement de baby-foot multi-agents (MARL) avec Self-Play
    
    4 agents contrôlent Team1 (rods 1-4, côté gauche):
        - Agent 0: Rod 1 Goalie
        - Agent 1: Rod 2 Defense
        - Agent 2: Rod 3 Forward
        - Agent 3: Rod 4 Midfield
    
    Team2 (rods 5-8, côté droit) contrôlée par les mêmes agents (self-play)
    """

    def __init__(self, render_mode=None):
        super(FoosballMARLEnv, self).__init__()
        
        self.render_mode = render_mode
        self.max_steps = 1000
        self.current_step = 0
        
        # Scores
        self.team1_goals = 0
        self.team2_goals = 0
        
        # Configuration des barres Team1 (rods 1-4, joints 2-17)
        self.agent_configs = {
            0: {"name": "Team1_Rod1_Goalie", "slide_idx": 2, "rotate_idx": 3, "x_pos": -0.625},
            1: {"name": "Team1_Rod2_Defense", "slide_idx": 7, "rotate_idx": 8, "x_pos": -0.45},
            2: {"name": "Team1_Rod3_Forward", "slide_idx": 11, "rotate_idx": 12, "x_pos": -0.275},
            3: {"name": "Team1_Rod4_Midfield", "slide_idx": 16, "rotate_idx": 17, "x_pos": -0.10},
        }
        
        # Joints Team2 (rods 5-8, joints 23-40)
        self.opponent_joints = {
            0: {"name": "Team2_Rod8_Goalie", "slide_idx": 39, "rotate_idx": 40, "x_pos": 0.625},
            1: {"name": "Team2_Rod7_Defense", "slide_idx": 35, "rotate_idx": 36, "x_pos": 0.45},
            2: {"name": "Team2_Rod6_Forward", "slide_idx": 30, "rotate_idx": 31, "x_pos": 0.275},
            3: {"name": "Team2_Rod5_Midfield", "slide_idx": 23, "rotate_idx": 24, "x_pos": 0.10},
        }
        
        # Limites
        self.slide_min = -0.1
        self.slide_max = 0.1
        self.x_max = 1.0
        self.y_max = 0.5
        
        # PyBullet
        if render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        # Observation space
        self.observation_space = gym.spaces.Box(
            low=np.array([
                -self.x_max, -self.y_max, -50, -50,  # Balle
                self.slide_min, -np.pi,  # Agent 0
                self.slide_min, -np.pi,  # Agent 1
                self.slide_min, -np.pi,  # Agent 2
                self.slide_min, -np.pi,  # Agent 3
                self.slide_min, -np.pi,  # Rod 5
                self.slide_min, -np.pi,  # Rod 6
                self.slide_min, -np.pi,  # Rod 7
                self.slide_min, -np.pi,  # Rod 8
            ], dtype=np.float32),
            high=np.array([
                self.x_max, self.y_max, 50, 50,
                self.slide_max, np.pi,
                self.slide_max, np.pi,
                self.slide_max, np.pi,
                self.slide_max, np.pi,
                self.slide_max, np.pi,
                self.slide_max, np.pi,
                self.slide_max, np.pi,
                self.slide_max, np.pi,
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space
        self.action_space = gym.spaces.Dict({
            f"agent_{i}": gym.spaces.Box(
                low=np.array([self.slide_min, -np.pi], dtype=np.float32),
                high=np.array([self.slide_max, np.pi], dtype=np.float32),
                shape=(2,),
                dtype=np.float32
            ) for i in range(4)
        })
        
        # Initialisation table
        self.urdf_path = os.path.join(os.path.dirname(__file__), 'foosball.urdf')
        self.table_id = p.loadURDF(self.urdf_path, basePosition=[0, 0, 0.5], useFixedBase=True)
        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.1, 0.4, 0.1, 1])
        
        # Balle
        ball_radius = 0.025
        self.ball_mass = 0.025
        self.visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 1, 1, 1])
        self.collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        self.ball_id = None
        
        # Lignes de but
        self.goal_line_left = -0.75
        self.goal_line_right = 0.75
        
        # Physique
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1./240.,
            numSubSteps=10,
            numSolverIterations=100
        )
        
        if render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5, 
                cameraYaw=0, 
                cameraPitch=-45, 
                cameraTargetPosition=[0, 0, 0.5]
            )

    def _get_observation(self) -> np.ndarray:
        """Récupère l'observation globale"""
        try:
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            ball_vel, _ = p.getBaseVelocity(self.ball_id)
            
            obs_list = [
                np.clip(ball_pos[0], -self.x_max, self.x_max),
                np.clip(ball_pos[1], -self.y_max, self.y_max),
                np.clip(ball_vel[0], -50, 50),
                np.clip(ball_vel[1], -50, 50),
            ]
            
            # États Team1
            for agent_id in range(4):
                config = self.agent_configs[agent_id]
                slide_state = p.getJointState(self.table_id, config["slide_idx"])
                rotate_state = p.getJointState(self.table_id, config["rotate_idx"])
                obs_list.extend([slide_state[0], rotate_state[0]])
            
            # États Team2
            for opponent_id in range(4):
                opp_cfg = self.opponent_joints[opponent_id]
                slide = p.getJointState(self.table_id, opp_cfg["slide_idx"])[0]
                angle = p.getJointState(self.table_id, opp_cfg["rotate_idx"])[0]
                obs_list.extend([slide, angle])
            
            return np.array(obs_list, dtype=np.float32)
        
        except Exception as e:
            print(f"❌ Erreur dans _get_observation: {e}")
            return np.zeros(20, dtype=np.float32)

    def _get_opponent_observation(self, opponent_id: int) -> np.ndarray:
        """
        Observation miroir pour Team2
        
        La table est inversée par rapport à l'axe X (x=0)
        Team2 voit le terrain comme si elle défendait le but de gauche
        et attaquait le but de droite (comme Team1)
        """
        try:
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            ball_vel, _ = p.getBaseVelocity(self.ball_id)
            
            # Miroir complet de la balle
            # Team2 voit la balle comme si le terrain était retourné
            obs_list = [
                np.clip(-ball_pos[0], -self.x_max, self.x_max),  # Inverser position X
                np.clip(ball_pos[1], -self.y_max, self.y_max),    # Y reste identique
                np.clip(-ball_vel[0], -50, 50),                   # Inverser vélocité X
                np.clip(ball_vel[1], -50, 50),                    # Y reste identique
            ]
            
            # Team2 voit ses propres rods en premier (comme "son équipe")
            # Dans l'ordre : Goalie (8) -> Defense (7) -> Forward (6) -> Midfield (5)
            # Mais réorganisé comme Team1 : Goalie -> Defense -> Forward -> Midfield
            # opponent_joints[0] = Goalie (rod8), [1] = Defense (rod7), [2] = Forward (rod6), [3] = Midfield (rod5)
            for opp_id in range(4):
                opp_cfg = self.opponent_joints[opp_id]
                slide_state = p.getJointState(self.table_id, opp_cfg["slide_idx"])
                rotate_state = p.getJointState(self.table_id, opp_cfg["rotate_idx"])
                
                # Slide reste identique (translation locale)
                # Rotation reste identique (rotation locale)
                obs_list.extend([slide_state[0], rotate_state[0]])
            
            # Team2 voit Team1 comme "adversaires" (dans l'ordre inverse spatial)
            # Team1 : Goalie (rod1, x=-0.625) -> Defense (rod2, x=-0.45) -> Forward (rod3, x=-0.275) -> Midfield (rod4, x=-0.10)
            # Pour Team2, après miroir, ça devient dans l'ordre spatial :
            # Midfield (x=0.10) -> Forward (x=0.275) -> Defense (x=0.45) -> Goalie (x=0.625)
            # Mais on garde l'ordre logique : Goalie -> Defense -> Forward -> Midfield
            for agent_id in range(4):
                cfg = self.agent_configs[agent_id]
                slide = p.getJointState(self.table_id, cfg["slide_idx"])[0]
                angle = p.getJointState(self.table_id, cfg["rotate_idx"])[0]
                obs_list.extend([slide, angle])
            
            return np.array(obs_list, dtype=np.float32)
        
        except Exception as e:
            print(f"❌ Erreur dans _get_opponent_observation: {e}")
            return np.zeros(20, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.ball_id is not None:
            try:
                p.removeBody(self.ball_id)
            except:
                pass
        
        p.setGravity(0, 0, -9.81)
        
        ball_start_x = np.random.uniform(-0.2, 0.2)
        ball_start_y = np.random.uniform(-0.09, 0.09)
        
        self.ball_id = p.createMultiBody(
            baseMass=self.ball_mass,
            baseCollisionShapeIndex=self.collision_shape,
            baseVisualShapeIndex=self.visual_shape,
            basePosition=[ball_start_x, ball_start_y, 0.55]
        )
        p.changeDynamics(
            self.ball_id, -1,
            restitution=0.8,
            rollingFriction=0.001,
            spinningFriction=0.001,
            lateralFriction=0.01
        )
        
        # Configuration des joints
        for config in self.agent_configs.values():
            p.changeDynamics(bodyUniqueId=self.table_id, linkIndex=config["slide_idx"],
                           jointDamping=2.0, linearDamping=0.5)
            p.changeDynamics(bodyUniqueId=self.table_id, linkIndex=config["rotate_idx"],
                           jointDamping=0.5)
        
        for opp_config in self.opponent_joints.values():
            p.changeDynamics(bodyUniqueId=self.table_id, linkIndex=opp_config["slide_idx"],
                           jointDamping=2.0, linearDamping=0.5)
            p.changeDynamics(bodyUniqueId=self.table_id, linkIndex=opp_config["rotate_idx"],
                           jointDamping=0.5)
        
        for _ in range(100):
            p.stepSimulation()
        
        initial_velocity_x = np.random.uniform(-5, 5)
        initial_velocity_y = np.random.uniform(-2, 2)
        p.resetBaseVelocity(
            objectUniqueId=self.ball_id,
            linearVelocity=[initial_velocity_x, initial_velocity_y, 0],
            angularVelocity=[0, 0, 0]
        )
        
        self.current_step = 0
        self.previous_ball_x = ball_start_x
        self.team1_goals = 0
        self.team2_goals = 0
        
        global_obs = self._get_observation()
        observations = {f"agent_{i}": global_obs for i in range(4)}
        
        info = {"team1_goals": self.team1_goals, "team2_goals": self.team2_goals}
        return observations, info

    def close(self):
        if p.isConnected(self.client):
            p.disconnect(self.client)


class SelfPlayFoosballEnv(MultiAgentEnv):
    """Environnement RLlib pour self-play avec 8 agents"""
    
    def __init__(self, config=None):
        super().__init__()
        # Si config contient "render", activer le GUI
        render_mode = "human" if (config and config.get("render", False)) else None
        self.base_env = FoosballMARLEnv(render_mode=render_mode)
        
        # 8 agents : agent_0-3 = Team1, agent_4-7 = Team2
        self._agent_ids = set([f"agent_{i}" for i in range(8)])
        
        # Attributes required by RLlib
        self.possible_agents = [f"agent_{i}" for i in range(8)]
        self.agents = self.possible_agents.copy()
        
        # Définir les espaces pour chaque agent individuellement
        single_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )
        single_action_space = gym.spaces.Box(
            low=np.array([-0.1, -np.pi], dtype=np.float32),
            high=np.array([0.1, np.pi], dtype=np.float32),
            dtype=np.float32
        )
        
        # RLlib nécessite observation_space et action_space comme dictionnaires
        self.observation_space = gym.spaces.Dict({
            agent_id: single_obs_space for agent_id in self._agent_ids
        })
        self.action_space = gym.spaces.Dict({
            agent_id: single_action_space for agent_id in self._agent_ids
        })
        
        # Aussi définir get_observation_space et get_action_space pour compatibilité
        self._obs_space_in_preferred_format = True
        self._action_space_in_preferred_format = True
    
    def reset(self, *, seed=None, options=None):
        obs_team1, info = self.base_env.reset(seed=seed, options=options)
        
        # Team1 : agent_0 à agent_3
        obs = {f"agent_{i}": obs_team1[f"agent_{i}"] for i in range(4)}
        
        # Team2 : agent_4 à agent_7 (observations miroir)
        for i in range(4):
            obs[f"agent_{i+4}"] = self.base_env._get_opponent_observation(i)
        
        return obs, info
    
    def step(self, action_dict):
        # Appliquer actions Team1
        for agent_id in range(4):
            action = action_dict[f"agent_{agent_id}"]
            config = self.base_env.agent_configs[agent_id]
            
            target_slide = np.clip(float(action[0]), self.base_env.slide_min, self.base_env.slide_max)
            target_rotation = np.clip(float(action[1]), -np.pi, np.pi)
            
            p.setJointMotorControl2(
                bodyIndex=self.base_env.table_id,
                jointIndex=config["slide_idx"],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_slide,
                force=500.0,
                maxVelocity=2.0
            )
            
            p.setJointMotorControl2(
                bodyIndex=self.base_env.table_id,
                jointIndex=config["rotate_idx"],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_rotation,
                force=200.0,
                maxVelocity=30.0
            )
        
        # Appliquer actions Team2
        for opponent_id in range(4):
            action = action_dict[f"agent_{opponent_id+4}"]
            opp_config = self.base_env.opponent_joints[opponent_id]
            
            target_slide = np.clip(float(action[0]), self.base_env.slide_min, self.base_env.slide_max)
            target_rotation = np.clip(float(action[1]), -np.pi, np.pi)
            
            p.setJointMotorControl2(
                bodyIndex=self.base_env.table_id,
                jointIndex=opp_config["slide_idx"],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_slide,
                force=500.0,
                maxVelocity=2.0
            )
            
            p.setJointMotorControl2(
                bodyIndex=self.base_env.table_id,
                jointIndex=opp_config["rotate_idx"],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_rotation,
                force=200.0,
                maxVelocity=30.0
            )
        
        # Simulation
        self.base_env.current_step += 1
        for _ in range(4):
            p.stepSimulation()
            # Si render activé, ralentir pour visualisation
            if self.base_env.render_mode == "human":
                time.sleep(1./240.)
        
        # Observer
        global_obs = self.base_env._get_observation()
        ball_x = global_obs[0]
        ball_y = global_obs[1]
        
        # Observations
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
            rewards[f"agent_{i+4}"] = -team1_reward  # Inverse pour Team2
        
        # Terminaison
        goal_scored = False
        if ball_x >= self.base_env.goal_line_right:
            self.base_env.team1_goals += 1
            goal_scored = True
        elif ball_x <= self.base_env.goal_line_left:
            self.base_env.team2_goals += 1
            goal_scored = True
        
        truncated = self.base_env.current_step >= self.base_env.max_steps
        
        dones = {f"agent_{i}": goal_scored or truncated for i in range(8)}
        dones["__all__"] = goal_scored or truncated
        
        self.base_env.previous_ball_x = ball_x
        
        # Info doit être vide ou contenir des clés par agent
        # On utilise un dict vide car RLlib n'accepte pas les clés globales
        info = {}
        
        return obs, rewards, dones, dones, info
    
    def close(self):
        self.base_env.close()
    
    def get_agent_ids(self):
        """Retourne les IDs de tous les agents"""
        return self._agent_ids


# =============================================================================
# ENTRAÎNEMENT AVEC RLLIB - SELF-PLAY
# =============================================================================

def train_selfplay_rllib(num_iterations: int = 100,
                        checkpoint_freq: int = 10,
                        num_workers: int = 4,
                        render: bool = False):
    """
    Entraînement avec RLlib en self-play
    
    8 agents (4 Team1 + 4 Team2) partagent la MÊME politique
    """
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec
    
    # Check for lz4
    try:
        import lz4
    except ImportError:
        print("⚠️  lz4 not installed. Install it for better performance: pip install lz4")
        print("")
    
    print("=" * 70)
    print("ENTRAÎNEMENT SELF-PLAY RLLIB - 8 AGENTS (4v4)")
    print("=" * 70)
    print(f"Itérations: {num_iterations}")
    print(f"Workers: {num_workers}")
    print(f"Render: {'OUI (lent)' if render else 'NON (rapide)'}")
    print(f"NOTE: Les 8 agents utilisent la MÊME politique (self-play)")
    print("=" * 70)
    
    # Initialiser Ray
    import os
    import warnings
    import logging
    
    # Suppress all warnings
    os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"
    os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning,ignore::FutureWarning"
    os.environ["RAY_DEDUP_LOGS"] = "0"
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Reduce Ray logging
    logging.getLogger("ray").setLevel(logging.ERROR)
    logging.getLogger("ray.rllib").setLevel(logging.ERROR)
    logging.getLogger("ray.tune").setLevel(logging.ERROR)
    
    ray.init(ignore_reinit_error=True, num_cpus=num_workers+1, logging_level=logging.ERROR)
    
    # UNE SEULE politique pour les 8 agents
    policies = {
        "shared_policy": PolicySpec(
            observation_space=gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
            ),
            action_space=gym.spaces.Box(
                low=np.array([-0.1, -np.pi], dtype=np.float32),
                high=np.array([0.1, np.pi], dtype=np.float32),
                dtype=np.float32
            ),
        ),
    }
    
    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "shared_policy"
    
    # Configuration PPO
    env_config = {"render": render} if render else {}
    
    config = (
        PPOConfig()
        .environment(SelfPlayFoosballEnv, env_config=env_config)
        .framework("torch")
        # Explicitly use new API stack (suppresses warnings)
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .experimental(_disable_preprocessor_api=True)
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            train_batch_size=4000,
            minibatch_size=128,
            num_epochs=10,  # Renamed from num_sgd_iter
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .env_runners(
            num_env_runners=1 if render else num_workers,  # 1 worker si render
            num_envs_per_env_runner=1,
        )
        .resources(
            num_gpus=0,
        )
        .debugging(
            log_level="ERROR"  # Reduce noise
        )
    )
    
    print("\n  ✓ 1 politique partagée pour les 8 agents")
    if render:
        print(f"  ⚠️  Mode VISUALISATION : 1 worker (lent)")
    else:
        print(f"  ✓ {num_workers} workers parallèles")
    
    # Construire
    print("\n[Construction PPO...]")
    algo = config.build_algo()
    print("  ✓ Algorithme prêt\n")
    
    # Entraînement
    print("=" * 70)
    print(f"DÉMARRAGE ({num_iterations} itérations)")
    print("=" * 70)
    
    try:
        for i in range(num_iterations):
            result = algo.train()
            
            if i % 10 == 0:
                print(f"\n📊 Itération {i+1}/{num_iterations}")
                print(f"  Reward mean: {result['env_runners']['episode_reward_mean']:.2f}")
                print(f"  Episode length: {result['env_runners']['episode_len_mean']:.1f}")
            
            if (i + 1) % checkpoint_freq == 0:
                checkpoint_dir = algo.save()
                print(f"  💾 Checkpoint: {checkpoint_dir}")
        
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


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train":
            num_iterations = int(sys.argv[2]) if len(sys.argv) > 2 else 100
            num_workers = 4
            render = False
            
            for i, arg in enumerate(sys.argv):
                if arg == "--workers" and i + 1 < len(sys.argv):
                    num_workers = int(sys.argv[i + 1])
                if arg == "--render" or arg == "-r":
                    render = True
            
            train_selfplay_rllib(
                num_iterations=num_iterations,
                num_workers=num_workers,
                render=render
            )
        
        else:
            print("Usage:")
            print("  python claude_code_rllib.py train [iterations] [--workers N] [--render]")
            print("")
            print("Options:")
            print("  --workers N    Nombre de workers parallèles (défaut: 4)")
            print("  --render       Afficher la simulation (lent, 1 worker)")
            print("")
            print("Exemples:")
            print("  python claude_code_rllib.py train 100")
            print("  python claude_code_rllib.py train 200 --workers 8")
            print("  python claude_code_rllib.py train 50 --render")
    else:
        print("Usage:")
        print("  python claude_code_rllib.py train [iterations] [--workers N] [--render]")
        print("")
        print("Exemples:")
        print("  python claude_code_rllib.py train 100 --workers 4")
        print("  python claude_code_rllib.py train 50 --render")
