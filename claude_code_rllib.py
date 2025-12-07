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
        # Limites de glissement selon URDF
        self.agent_configs = {
            0: {"name": "Team1_Rod1_Goalie", "slide_idx": 2, "rotate_idx": 3, "x_pos": -0.625, "slide_min": -0.12, "slide_max": 0.12},
            1: {"name": "Team1_Rod2_Defense", "slide_idx": 7, "rotate_idx": 8, "x_pos": -0.45, "slide_min": -0.18, "slide_max": 0.18},
            2: {"name": "Team1_Rod3_Forward", "slide_idx": 11, "rotate_idx": 12, "x_pos": -0.275, "slide_min": -0.12, "slide_max": 0.12},
            3: {"name": "Team1_Rod4_Midfield", "slide_idx": 16, "rotate_idx": 17, "x_pos": -0.10, "slide_min": -0.08, "slide_max": 0.08},
        }
        
        # Joints Team2 (rods 5-8, joints 23-40)
        self.opponent_joints = {
            0: {"name": "Team2_Rod8_Goalie", "slide_idx": 39, "rotate_idx": 40, "x_pos": 0.625, "slide_min": -0.12, "slide_max": 0.12},
            1: {"name": "Team2_Rod7_Defense", "slide_idx": 35, "rotate_idx": 36, "x_pos": 0.45, "slide_min": -0.18, "slide_max": 0.18},
            2: {"name": "Team2_Rod6_Forward", "slide_idx": 30, "rotate_idx": 31, "x_pos": 0.275, "slide_min": -0.12, "slide_max": 0.12},
            3: {"name": "Team2_Rod5_Midfield", "slide_idx": 23, "rotate_idx": 24, "x_pos": 0.10, "slide_min": -0.08, "slide_max": 0.08},
        }
        
        # Limites globales pour observation space (valeurs maximales)
        self.slide_min_global = -0.18  # Max range pour observation space
        self.slide_max_global = 0.18
        self.x_max = 1.0
        self.y_max = 0.5
        
        # PyBullet
        if render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        # Observation space (utilise les limites globales maximales)
        self.observation_space = gym.spaces.Box(
            low=np.array([
                -self.x_max, -self.y_max, -50, -50,  # Balle
                self.slide_min_global, -np.pi,  # Agent 0
                self.slide_min_global, -np.pi,  # Agent 1
                self.slide_min_global, -np.pi,  # Agent 2
                self.slide_min_global, -np.pi,  # Agent 3
                self.slide_min_global, -np.pi,  # Rod 5
                self.slide_min_global, -np.pi,  # Rod 6
                self.slide_min_global, -np.pi,  # Rod 7
                self.slide_min_global, -np.pi,  # Rod 8
            ], dtype=np.float32),
            high=np.array([
                self.x_max, self.y_max, 50, 50,
                self.slide_max_global, np.pi,
                self.slide_max_global, np.pi,
                self.slide_max_global, np.pi,
                self.slide_max_global, np.pi,
                self.slide_max_global, np.pi,
                self.slide_max_global, np.pi,
                self.slide_max_global, np.pi,
                self.slide_max_global, np.pi,
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space (utilise les limites globales maximales pour uniformité)
        self.action_space = gym.spaces.Dict({
            f"agent_{i}": gym.spaces.Box(
                low=np.array([self.slide_min_global, -np.pi], dtype=np.float32),
                high=np.array([self.slide_max_global, np.pi], dtype=np.float32),
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
        
        # Limites du plateau (pour détecter sortie de balle)
        # Basé sur URDF: table 1.5m × 0.68m, murs à ±0.365
        self.table_x_min = -0.80  # Un peu avant la ligne de but
        self.table_x_max = 0.80
        self.table_y_min = -0.40  # Limites Y du terrain (±0.365 + marge)
        self.table_y_max = 0.40
        self.table_z_min = 0.48   # Hauteur minimale (table à 0.5, balle ne doit pas être en dessous)
        self.table_z_max = 0.80   # Hauteur maximale (balle peut rebondir haut)
        
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

    def _is_ball_out_of_bounds(self) -> bool:
        """
        Vérifie si la balle est sortie du plateau
        
        Returns:
            True si la balle est hors limites, False sinon
        """
        try:
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            ball_x, ball_y, ball_z = ball_pos
            
            # Vérifier si la balle est hors des limites X, Y ou Z
            out_of_bounds = (
                ball_x < self.table_x_min or ball_x > self.table_x_max or
                ball_y < self.table_y_min or ball_y > self.table_y_max or
                ball_z < self.table_z_min or ball_z > self.table_z_max
            )
            
            return out_of_bounds
        
        except Exception as e:
            print(f"❌ Erreur dans _is_ball_out_of_bounds: {e}")
            return False
    
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
        # Action space utilise les limites maximales (±0.18 pour compatibilité avec toutes les barres)
        single_action_space = gym.spaces.Box(
            low=np.array([-0.18, -np.pi], dtype=np.float32),
            high=np.array([0.18, np.pi], dtype=np.float32),
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
            
            # Utiliser les limites spécifiques à chaque barre
            target_slide = np.clip(float(action[0]), config["slide_min"], config["slide_max"])
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
            
            # Utiliser les limites spécifiques à chaque barre
            target_slide = np.clip(float(action[0]), opp_config["slide_min"], opp_config["slide_max"])
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
        ball_out = False
        
        # Vérifier si la balle est sortie du plateau
        if self.base_env._is_ball_out_of_bounds():
            ball_out = True
            # Pénalité pour sortie de balle (aucune équipe ne marque)
            for i in range(8):
                rewards[f"agent_{i}"] = -5.0  # Pénalité pour tous
        # Vérifier les buts
        elif ball_x >= self.base_env.goal_line_right:
            self.base_env.team1_goals += 1
            goal_scored = True
            # Bonus pour but marqué
            for i in range(4):
                rewards[f"agent_{i}"] += 10.0  # Team1 marque
                rewards[f"agent_{i+4}"] -= 10.0  # Team2 encaisse
        elif ball_x <= self.base_env.goal_line_left:
            self.base_env.team2_goals += 1
            goal_scored = True
            # Bonus pour but marqué
            for i in range(4):
                rewards[f"agent_{i}"] -= 10.0  # Team1 encaisse
                rewards[f"agent_{i+4}"] += 10.0  # Team2 marque
        
        truncated = self.base_env.current_step >= self.base_env.max_steps
        
        # Episode terminé si: but marqué OU balle sortie OU max steps
        episode_done = goal_scored or ball_out or truncated
        
        dones = {f"agent_{i}": episode_done for i in range(8)}
        dones["__all__"] = episode_done
        
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
    
    # Check for lz4
    try:
        import lz4
    except ImportError:
        print("⚠️  lz4 not installed. Install it for better performance: pip install lz4")
        print("")
    
    print("=" * 70)
    print("ENTRAÎNEMENT SELF-PLAY RLLIB - 8 AGENTS (4v4) + LSTM")
    print("=" * 70)
    print(f"Itérations: {num_iterations}")
    print(f"Workers: {num_workers}")
    print(f"Render: {'OUI (lent)' if render else 'NON (rapide)'}")
    print(f"Architecture: FC[256,256] → LSTM[256] → Policy/Value")
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
    
    # Nettoyer les variables d'environnement Ray qui pourraient pointer vers un cluster distant
    for env_var in ["RAY_ADDRESS", "RAY_HEAD_SERVICE_IP", "RAY_HEAD_SERVICE_PORT"]:
        if env_var in os.environ:
            del os.environ[env_var]
            print(f"  ✓ Variable d'environnement {env_var} supprimée")
    
    # Tenter de se connecter à un cluster existant ou en créer un nouveau
    try:
        ray.init(
            address="local",  # Force l'utilisation d'un cluster local
            ignore_reinit_error=True, 
            num_cpus=num_workers+1, 
            logging_level=logging.ERROR,
            _metrics_export_port=None,  # Disable metrics exporter
            _system_config={
                "metrics_report_interval_ms": 0,  # Disable metrics reporting
            },
            # Configurations supplémentaires pour éviter les problèmes de connexion
            _temp_dir=None,  # Utiliser le répertoire temporaire par défaut
            include_dashboard=False,  # Désactiver le dashboard pour réduire les ressources
        )
        print("  ✓ Ray initialisé avec succès")
    except Exception as e:
        print(f"  ⚠️  Erreur lors de l'initialisation de Ray: {e}")
        print("  → Tentative d'arrêt des processus Ray existants...")
        os.system("ray stop --force")
        time.sleep(2)
        print("  → Nouvelle tentative d'initialisation...")
        ray.init(
            address="local",
            ignore_reinit_error=True, 
            num_cpus=num_workers+1, 
            logging_level=logging.ERROR,
            _metrics_export_port=None,
            _system_config={
                "metrics_report_interval_ms": 0,
            },
            include_dashboard=False,
        )
        print("  ✓ Ray initialisé après nettoyage")
    
    # Avec la nouvelle API stack, on définit juste le policy mapping
    # Les politiques sont créées automatiquement par RLlib
    def policy_mapping_fn(agent_id, episode, **kwargs):
        # Tous les agents partagent la même politique
        return "default_policy"
    
    # Configuration PPO
    env_config = {"render": render} if render else {}
    
    # Configuration du modèle LSTM pour la nouvelle API stack
    model_config = {
        "fcnet_hiddens": [256, 256],  # Couches FC avant LSTM
        "fcnet_activation": "relu",
        "use_lstm": True,  # Activer LSTM
        "lstm_cell_size": 256,  # Taille des cellules LSTM
        "max_seq_len": 20,  # Longueur maximale de séquence
        "lstm_use_prev_action": True,  # Utiliser l'action précédente
        "lstm_use_prev_reward": True,  # Utiliser la récompense précédente
    }
    
    # Avec l'ancienne API pour compatibilité multi-agent + LSTM
    from ray.rllib.policy.policy import PolicySpec
    
    # Définir la politique avec LSTM
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
            config={
                "model": model_config,
            }
        ),
    }
    
    def policy_mapping_fn_old(agent_id, episode, worker, **kwargs):
        return "shared_policy"
    
    config = (
        PPOConfig()
        .environment(SelfPlayFoosballEnv, env_config=env_config)
        .framework("torch")
        # Utiliser ancienne API pour multi-agent + LSTM
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            train_batch_size=4000,
            num_sgd_iter=10,
            model=model_config,  # Configuration LSTM
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn_old,
        )
        .env_runners(
            num_env_runners=1 if render else num_workers,
            num_envs_per_env_runner=1,
        )
        .resources(
            num_gpus=0,
        )
        .debugging(
            log_level="ERROR"
        )
    )
    
    print("\n  ✓ Politique partagée pour les 8 agents (self-play)")
    print("  ✓ Configuration: Nouvelle API stack avec LSTM")
    if render:
        print(f"  ⚠️  Mode VISUALISATION : 1 worker (lent)")
    else:
        print(f"  ✓ {num_workers} workers parallèles")
    
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
            
            # Checkpoint
            if (i + 1) % checkpoint_freq == 0:
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
