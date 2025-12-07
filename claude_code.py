import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
from typing import Dict, Tuple, List

class FoosballMARLEnv(gym.Env):
    """
    Environnement de baby-foot multi-agents (MARL) avec Self-Play
    
    4 agents contrôlent Team1 (rods 1-4, côté gauche):
        - Agent 0: Rod 1 Goalie
        - Agent 1: Rod 2 Defense
        - Agent 2: Rod 3 Forward
        - Agent 3: Rod 4 Midfield
    
    Team2 (rods 5-8, côté droit) est contrôlée par:
        - Une copie des agents (self-play)
        - Ou des agents fixes (pour entraînement)
    """

    def __init__(self, render_mode=None, opponent_models=None):
        super(FoosballMARLEnv, self).__init__()
        
        self.render_mode = render_mode
        self.max_steps = 1000
        self.current_step = 0
        
        # Scores
        self.team1_goals = 0
        self.team2_goals = 0
        
        # Modèles adverses pour self-play (optionnel)
        # Si None, Team2 reste statique
        # Si fourni, Team2 joue avec ces modèles
        self.opponent_models = opponent_models
        
        # Configuration des barres Team1 (rods 1-4, joints 2-17)
        # Positions X extraites du URDF
        self.agent_configs = {
            0: {"name": "Team1_Rod1_Goalie", "slide_idx": 2, "rotate_idx": 3, "x_pos": -0.625},      # Rod 1: Goalie
            1: {"name": "Team1_Rod2_Defense", "slide_idx": 7, "rotate_idx": 8, "x_pos": -0.45},      # Rod 2: Defenders
            2: {"name": "Team1_Rod3_Forward", "slide_idx": 11, "rotate_idx": 12, "x_pos": -0.275},   # Rod 3: Forwards (Blue in URDF)
            3: {"name": "Team1_Rod4_Midfield", "slide_idx": 16, "rotate_idx": 17, "x_pos": -0.10},   # Rod 4: Midfield
        }
        
        # Joints de l'équipe adverse Team2 (rods 5-8, joints 23-40)
        # Positions X extraites du URDF
        # Mapping: rod5 -> opponent_agent_3 (Midfield), rod6 -> opponent_agent_2 (Forward)
        #          rod7 -> opponent_agent_1 (Defense), rod8 -> opponent_agent_0 (Goalie)
        self.opponent_joints = {
            0: {"name": "Team2_Rod8_Goalie", "slide_idx": 39, "rotate_idx": 40, "x_pos": 0.625},   # Goalie (miroir de agent 0)
            1: {"name": "Team2_Rod7_Defense", "slide_idx": 35, "rotate_idx": 36, "x_pos": 0.45},   # Defense (miroir de agent 1)
            2: {"name": "Team2_Rod6_Forward", "slide_idx": 30, "rotate_idx": 31, "x_pos": 0.275},  # Forward (miroir de agent 2)
            3: {"name": "Team2_Rod5_Midfield", "slide_idx": 23, "rotate_idx": 24, "x_pos": 0.10},  # Midfield (miroir de agent 3)
        }
        
        # Limites des actions
        self.slide_min = -0.1
        self.slide_max = 0.1
        self.x_max = 1.0
        self.y_max = 0.5
        
        # Connexion PyBullet
        if render_mode == "human":
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        # Action space: chaque agent contrôle [translation, rotation]
        single_action_space = gym.spaces.Box(
            low=np.array([self.slide_min, -np.pi], dtype=np.float32),
            high=np.array([self.slide_max, np.pi], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space: [ball_x, ball_y, ball_vx, ball_vy, 
        #                     agent0_slide, agent0_angle,
        #                     agent1_slide, agent1_angle,
        #                     agent2_slide, agent2_angle,
        #                     agent3_slide, agent3_angle,
        #                     rod5_slide, rod5_angle, rod6_slide, rod6_angle,
        #                     rod7_slide, rod7_angle, rod8_slide, rod8_angle]
        # Total: 4 (ball) + 8 (4 agents Team1) + 8 (4 rods Team2) = 20 dimensions
        self.observation_space = gym.spaces.Box(
            low=np.array([
                -self.x_max, -self.y_max, -50, -50,  # Balle
                self.slide_min, -np.pi,  # Agent 0
                self.slide_min, -np.pi,  # Agent 1
                self.slide_min, -np.pi,  # Agent 2
                self.slide_min, -np.pi,  # Agent 3
                # Team2 Rods (4 rods)
                self.slide_min, -np.pi,  # Rod 5 (Midfield)
                self.slide_min, -np.pi,  # Rod 6 (Forward)
                self.slide_min, -np.pi,  # Rod 7 (Defense)
                self.slide_min, -np.pi,  # Rod 8 (Goalie)
            ], dtype=np.float32),
            high=np.array([
                self.x_max, self.y_max, 50, 50,
                self.slide_max, np.pi,
                self.slide_max, np.pi,
                self.slide_max, np.pi,
                self.slide_max, np.pi,
                # Team2 Rods (4 rods)
                self.slide_max, np.pi,   # Rod 5 (Midfield)
                self.slide_max, np.pi,   # Rod 6 (Forward)
                self.slide_max, np.pi,   # Rod 7 (Defense)
                self.slide_max, np.pi,   # Rod 8 (Goalie)
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space pour tous les agents
        self.action_space = gym.spaces.Dict({
            f"agent_{i}": single_action_space for i in range(4)
        })
        
        # Initialisation de la table
        self.urdf_path = os.path.join(os.path.dirname(__file__), 'foosball.urdf')
        self.table_id = p.loadURDF(self.urdf_path, basePosition=[0, 0, 0.5], useFixedBase=True)
        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.1, 0.4, 0.1, 1])
        
        # Création de la balle
        ball_radius = 0.025
        self.ball_mass = 0.025
        self.visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 1, 1, 1])
        self.collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)
        self.ball_id = None
        
        # Lignes de but
        self.goal_line_left = -0.75  # But équipe 1 (gauche)
        self.goal_line_right = 0.75   # But équipe 2 (droite)
        
        # Configuration physique
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
        """Récupère l'observation globale pour tous les agents, incluant l'état de toutes les rods"""
        try:
            # Position et vélocité de la balle
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            ball_vel, _ = p.getBaseVelocity(self.ball_id)
            
            obs_list = [
                np.clip(ball_pos[0], -self.x_max, self.x_max),
                np.clip(ball_pos[1], -self.y_max, self.y_max),
                np.clip(ball_vel[0], -50, 50),
                np.clip(ball_vel[1], -50, 50),
            ]
            
            # États de chaque agent Team1
            for agent_id in range(4):
                config = self.agent_configs[agent_id]
                
                slide_state = p.getJointState(self.table_id, config["slide_idx"])
                rotate_state = p.getJointState(self.table_id, config["rotate_idx"])
                
                obs_list.extend([
                    slide_state[0],  # Position translation
                    rotate_state[0]  # Angle rotation
                ])
            
            # Ajouter l'état de toutes les rods Team2 (dans l'ordre: 0, 1, 2, 3)
            for opponent_id in range(4):
                opp_cfg = self.opponent_joints[opponent_id]
                slide = p.getJointState(self.table_id, opp_cfg["slide_idx"])[0]
                angle = p.getJointState(self.table_id, opp_cfg["rotate_idx"])[0]
                obs_list.extend([slide, angle])
            
            return np.array(obs_list, dtype=np.float32)
        
        except Exception as e:
            print(f"❌ Erreur dans _get_observation: {e}")
            return np.zeros(12, dtype=np.float32)

    def _get_agent_observation(self, agent_id: int, global_obs: np.ndarray) -> np.ndarray:
        """
        Observation locale pour un agent spécifique
        Inclut: position balle, vitesse balle, état de sa barre, états adversaires
        """
        # On peut donner l'observation complète ou filtrer selon l'agent
        # Pour simplicité, on donne l'observation complète
        return global_obs

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict]:
        super().reset(seed=seed)
        
        # Supprimer l'ancienne balle
        if self.ball_id is not None:
            try:
                p.removeBody(self.ball_id)
            except:
                pass
        
        p.setGravity(0, 0, -9.81)
        
        # Position aléatoire de la balle au centre
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
        
        # Configuration des joints pour Team1
        for config in self.agent_configs.values():
            p.changeDynamics(
                bodyUniqueId=self.table_id,
                linkIndex=config["slide_idx"],
                jointDamping=2.0,
                linearDamping=0.5
            )
            p.changeDynamics(
                bodyUniqueId=self.table_id,
                linkIndex=config["rotate_idx"],
                jointDamping=0.5
            )
        
        # Configuration des joints pour Team2
        for opp_config in self.opponent_joints.values():
            p.changeDynamics(
                bodyUniqueId=self.table_id,
                linkIndex=opp_config["slide_idx"],
                jointDamping=2.0,
                linearDamping=0.5
            )
            p.changeDynamics(
                bodyUniqueId=self.table_id,
                linkIndex=opp_config["rotate_idx"],
                jointDamping=0.5
            )
        
        # Stabiliser la simulation
        for _ in range(100):
            p.stepSimulation()
        
        # Vitesse initiale aléatoire
        initial_velocity_x = np.random.uniform(-5, 5)
        initial_velocity_y = np.random.uniform(-2, 2)
        p.resetBaseVelocity(
            objectUniqueId=self.ball_id,
            linearVelocity=[initial_velocity_x, initial_velocity_y, 0],
            angularVelocity=[0, 0, 0]
        )
        
        self.current_step = 0
        self.previous_ball_x = ball_start_x
        
        # Observation globale
        global_obs = self._get_observation()
        
        # Observations par agent
        observations = {
            f"agent_{i}": self._get_agent_observation(i, global_obs)
            for i in range(4)
        }
        
        info = {"team1_goals": self.team1_goals, "team2_goals": self.team2_goals}
        
        return observations, info

    def _get_opponent_observation(self, opponent_id: int) -> np.ndarray:
        """
        Crée une observation pour l'adversaire en inversant la perspective
        L'observation est miroir par rapport à x=0
        """
        try:
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            ball_vel, _ = p.getBaseVelocity(self.ball_id)
            
            obs_list = [
                np.clip(-ball_pos[0], -self.x_max, self.x_max),  # Inverser x
                np.clip(ball_pos[1], -self.y_max, self.y_max),   # Y reste pareil
                np.clip(-ball_vel[0], -50, 50),                  # Inverser vx
                np.clip(ball_vel[1], -50, 50),                   # vy reste pareil
            ]
            
            # États des agents Team2 (vue comme "notre équipe" pour l'adversaire)
            for opponent_id in range(4):
                opp_cfg = self.opponent_joints[opponent_id]
                slide_state = p.getJointState(self.table_id, opp_cfg["slide_idx"])
                rotate_state = p.getJointState(self.table_id, opp_cfg["rotate_idx"])
                obs_list.extend([slide_state[0], rotate_state[0]])
            
            # États des agents Team1 (vue comme "adversaires" pour l'adversaire)
            for agent_id in range(4):
                cfg = self.agent_configs[agent_id]
                slide = p.getJointState(self.table_id, cfg["slide_idx"])[0]
                angle = p.getJointState(self.table_id, cfg["rotate_idx"])[0]
                obs_list.extend([slide, angle])
            
            return np.array(obs_list, dtype=np.float32)
        
        except Exception as e:
            print(f"❌ Erreur dans _get_opponent_observation: {e}")
            return np.zeros(20, dtype=np.float32)
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Exécute les actions de tous les agents (Team1) et des adversaires (Team2)
        
        Args:
            actions: Dict avec clés "agent_0", "agent_1", "agent_2", "agent_3"
                     chaque valeur est [translation, rotation]
        
        Returns:
            observations, rewards, terminated, truncated, info (tous des dicts)
        """
        self.current_step += 1
        
        # 1. Appliquer les actions de chaque agent Team1
        for agent_id in range(4):
            action = actions[f"agent_{agent_id}"]
            config = self.agent_configs[agent_id]
            
            target_slide = np.clip(float(action[0]), self.slide_min, self.slide_max)
            target_rotation = np.clip(float(action[1]), -np.pi, np.pi)
            
            # Contrôle de translation
            p.setJointMotorControl2(
                bodyIndex=self.table_id,
                jointIndex=config["slide_idx"],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_slide,
                force=500.0,
                maxVelocity=2.0
            )
            
            # Contrôle de rotation
            p.setJointMotorControl2(
                bodyIndex=self.table_id,
                jointIndex=config["rotate_idx"],
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_rotation,
                force=200.0,
                maxVelocity=30.0
            )
        
        # 2. Appliquer les actions de Team2 (adversaires)
        if self.opponent_models is not None:
            # Self-play: utiliser les modèles adverses
            for opponent_id in range(4):
                opponent_obs = self._get_opponent_observation(opponent_id)
                opponent_action, _ = self.opponent_models[opponent_id].predict(opponent_obs, deterministic=False)
                
                opp_config = self.opponent_joints[opponent_id]
                target_slide = np.clip(float(opponent_action[0]), self.slide_min, self.slide_max)
                target_rotation = np.clip(float(opponent_action[1]), -np.pi, np.pi)
                
                p.setJointMotorControl2(
                    bodyIndex=self.table_id,
                    jointIndex=opp_config["slide_idx"],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_slide,
                    force=500.0,
                    maxVelocity=2.0
                )
                
                p.setJointMotorControl2(
                    bodyIndex=self.table_id,
                    jointIndex=opp_config["rotate_idx"],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_rotation,
                    force=200.0,
                    maxVelocity=30.0
                )
        else:
            # Pas de self-play: Team2 reste en position neutre
            for opponent_id in range(4):
                opp_config = self.opponent_joints[opponent_id]
                
                p.setJointMotorControl2(
                    bodyIndex=self.table_id,
                    jointIndex=opp_config["slide_idx"],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=0.0,
                    force=500.0,
                    maxVelocity=2.0
                )
                
                p.setJointMotorControl2(
                    bodyIndex=self.table_id,
                    jointIndex=opp_config["rotate_idx"],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=0.0,
                    force=200.0,
                    maxVelocity=30.0
                )
        
        # Simulation
        for _ in range(4):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1./240.)
        
        # Observer le nouvel état
        global_obs = self._get_observation()
        ball_x = global_obs[0]
        ball_y = global_obs[1]
        
        observations = {
            f"agent_{i}": self._get_agent_observation(i, global_obs)
            for i in range(4)
        }
        
        # Calcul des récompenses
        rewards = self._compute_rewards(ball_x, ball_y, global_obs)
        
        # Vérifier les buts
        goal_scored = self._check_goals(ball_x, ball_y)
        
        terminated = {f"agent_{i}": goal_scored for i in range(4)}
        terminated["__all__"] = goal_scored
        
        truncated = {f"agent_{i}": self.current_step >= self.max_steps for i in range(4)}
        truncated["__all__"] = self.current_step >= self.max_steps
        
        self.previous_ball_x = ball_x
        
        info = {
            "team1_goals": self.team1_goals,
            "team2_goals": self.team2_goals,
            "ball_x": ball_x,
            "ball_y": ball_y
        }
        
        return observations, rewards, terminated, truncated, info

    def _compute_rewards(self, ball_x: float, ball_y: float, obs: np.ndarray) -> Dict[str, float]:
        """
        Calcule les récompenses pour chaque agent de Team1
        
        Récompenses coopératives:
        - Team1 (agents 0,1,2,3): gagne si but marqué à droite (x > 0.75)
        - Team1 défend le but de gauche (x < -0.75)
        """
        rewards = {}
        
        # Récompense de base (petite pénalité temporelle)
        base_reward = -0.1
        
        # Bonus si la balle progresse vers le but adverse (droite, +x)
        progress_reward = (ball_x - self.previous_ball_x) * 10
        
        # Distance de la balle au but adverse
        distance_to_goal = self.goal_line_right - ball_x
        distance_reward = -distance_to_goal * 0.5
        
        # Tous les agents de Team1 partagent la même récompense (coopération)
        for agent_id in range(4):
            rewards[f"agent_{agent_id}"] = base_reward + progress_reward + distance_reward
        
        return rewards

    def _check_goals(self, ball_x: float, ball_y: float) -> bool:
        """Vérifie si un but est marqué"""
        # But pour Team1 (balle passe la ligne de droite)
        if ball_x >= self.goal_line_right:
            self.team1_goals += 1
            return True
        
        # But pour Team2 (balle passe la ligne de gauche - Team1 encaisse)
        if ball_x <= self.goal_line_left:
            self.team2_goals += 1
            return True
        
        return False

    def close(self):
        """Ferme la connexion PyBullet"""
        if p.isConnected(self.client):
            p.disconnect(self.client)
    
    def set_opponent_models(self, models):
        """Met à jour les modèles adverses pour le self-play"""
        self.opponent_models = models


# =============================================================================
# ENTRAÎNEMENT AVEC RLLIB - VRAI SELF-PLAY
# =============================================================================
def train_selfplay_rllib(num_iterations: int = 100,
                        checkpoint_freq: int = 10,
                        num_workers: int = 4):
    """
    Entraînement avec RLlib en VRAI self-play multi-agents
    
    8 agents au total (4 Team1 + 4 Team2) partagent la MÊME politique.
    Team1 et Team2 apprennent ensemble en jouant l'un contre l'autre.
    
    Args:
        num_iterations: Nombre d'itérations d'entraînement
        checkpoint_freq: Fréquence de sauvegarde
        num_workers: Nombre de workers parallèles
    """
    import ray
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec
    
    print("=" * 70)
    print("ENTRAÎNEMENT SELF-PLAY RLLIB - 8 AGENTS (4v4)")
    print("=" * 70)
    print(f"Itérations: {num_iterations}")
    print(f"Workers: {num_workers}")
    print(f"NOTE: Les 8 agents utilisent la MÊME politique (self-play)")
    print("=" * 70)
    
    # Initialiser Ray
    ray.init(ignore_reinit_error=True, num_cpus=num_workers+1)
    
    # Créer un environnement modifié qui gère les 8 agents
    class SelfPlayFoosballEnv(MultiAgentEnv):
        """Environnement pour self-play avec 8 agents utilisant la même politique"""
        
        def __init__(self, config=None):
            super().__init__()
            self.base_env = FoosballMARLEnv(render_mode=None, opponent_models=None)
            
            # Les 8 agents : agent_0 à agent_7
            # agent_0-3 = Team1, agent_4-7 = Team2
            self._agent_ids = set([f"agent_{i}" for i in range(8)])
            
            # Tous partagent le même observation et action space
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=np.array([-0.1, -np.pi], dtype=np.float32),
                high=np.array([0.1, np.pi], dtype=np.float32),
                dtype=np.float32
            )
        
        def reset(self, *, seed=None, options=None):
            obs_team1, info = self.base_env.reset(seed=seed, options=options)
            
            # Créer observations pour les 8 agents
            # Team1 : agent_0 à agent_3
            obs = {f"agent_{i}": obs_team1[f"agent_{i}"] for i in range(4)}
            
            # Team2 : agent_4 à agent_7 (observations miroir)
            for i in range(4):
                obs[f"agent_{i+4}"] = self.base_env._get_opponent_observation(i)
            
            return obs, info
        
        def step(self, action_dict):
            # Extraire actions pour Team1 (agent_0 à agent_3)
            team1_actions = {f"agent_{i}": action_dict[f"agent_{i}"] for i in range(4)}
            
            # Extraire actions pour Team2 (agent_4 à agent_7)
            # et les mapper aux joints opposés
            team2_actions = {i: action_dict[f"agent_{i+4}"] for i in range(4)}
            
            # Appliquer Team1
            for agent_id in range(4):
                action = team1_actions[f"agent_{agent_id}"]
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
            
            # Appliquer Team2
            for opponent_id in range(4):
                action = team2_actions[opponent_id]
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
            
            # Observer
            global_obs = self.base_env._get_observation()
            ball_x = global_obs[0]
            ball_y = global_obs[1]
            
            # Observations pour Team1
            obs = {f"agent_{i}": self.base_env._get_agent_observation(i, global_obs) for i in range(4)}
            
            # Observations pour Team2 (miroir)
            for i in range(4):
                obs[f"agent_{i+4}"] = self.base_env._get_opponent_observation(i)
            
            # Récompenses
            rewards_team1 = self.base_env._compute_rewards(ball_x, ball_y, global_obs)
            
            # Team2 a les récompenses inversées
            rewards = {}
            for i in range(4):
                rewards[f"agent_{i}"] = rewards_team1[f"agent_{i}"]
                rewards[f"agent_{i+4}"] = -rewards_team1[f"agent_{i}"]  # Inverse pour Team2
            
            # Terminaison
            goal_scored = self.base_env._check_goals(ball_x, ball_y)
            truncated = self.base_env.current_step >= self.base_env.max_steps
            
            terminated = {f"agent_{i}": goal_scored for i in range(8)}
            terminated["__all__"] = goal_scored
            
            truncated_dict = {f"agent_{i}": truncated for i in range(8)}
            truncated_dict["__all__"] = truncated
            
            dones = {agent: terminated[agent] or truncated_dict[agent] for agent in self._agent_ids}
            dones["__all__"] = terminated["__all__"] or truncated_dict["__all__"]
            
            self.base_env.previous_ball_x = ball_x
            
            info = {
                "team1_goals": self.base_env.team1_goals,
                "team2_goals": self.base_env.team2_goals
            }
            
            return obs, rewards, dones, dones, info
        
        def close(self):
            self.base_env.close()
    
    # Configuration RLlib avec une seule politique partagée
    print("\n[1/3] Configuration de la politique partagée...")
    
    # UNE SEULE politique pour les 8 agents (self-play)
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
    
    # Tous les agents utilisent la même politique
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"
    
    # Configuration PPO
    config = (
        PPOConfig()
        .environment(SelfPlayFoosballEnv, env_config={})
        .framework("torch")
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .rollouts(
            num_rollout_workers=num_workers,
            num_envs_per_worker=1,
        )
        .resources(
            num_gpus=0,
        )
        .debugging(
            log_level="WARN"
        )
    )
    
    print("  ✓ 1 politique partagée pour les 8 agents")
    print(f"  ✓ {num_workers} workers parallèles")
    
    # Construire l'algorithme
    print("\n[2/3] Construction de l'algorithme PPO...")
    algo = config.build()
    print("  ✓ Algorithme prêt")
    
    # Entraînement
    print(f"\n[3/3] Démarrage de l'entraînement ({num_iterations} itérations)...")
    print("=" * 70)
    
    try:
        for i in range(num_iterations):
            result = algo.train()
            
            if i % 10 == 0:
                print(f"\n📊 Itération {i+1}/{num_iterations}")
                print(f"  Reward mean: {result['env_runners']['episode_reward_mean']:.2f}")
                print(f"  Episode length: {result['env_runners']['episode_len_mean']:.1f}")
                print(f"  Team1 goals: {result.get('custom_metrics', {}).get('team1_goals_mean', 'N/A')}")
                print(f"  Team2 goals: {result.get('custom_metrics', {}).get('team2_goals_mean', 'N/A')}")
            
            # Sauvegarder périodiquement
            if (i + 1) % checkpoint_freq == 0:
                checkpoint_dir = algo.save()
                print(f"  💾 Checkpoint: {checkpoint_dir}")
        
        print("\n" + "=" * 70)
        print("✓ ENTRAÎNEMENT TERMINÉ!")
        print("=" * 70)
        
        # Sauvegarder le modèle final
        final_checkpoint = algo.save()
        print(f"✓ Modèle final sauvegardé: {final_checkpoint}")
        
        return algo, final_checkpoint
    
    except KeyboardInterrupt:
        print("\n⚠️  Entraînement interrompu par l'utilisateur")
        checkpoint = algo.save()
        print(f"✓ Checkpoint sauvegardé: {checkpoint}")
        return algo, checkpoint
    
    finally:
        algo.stop()
        ray.shutdown()


def test_selfplay_rllib(checkpoint_path: str, n_episodes: int = 10):
    """
    Teste un modèle RLlib entraîné en self-play
    
    Args:
        checkpoint_path: Chemin vers le checkpoint
        n_episodes: Nombre d'épisodes de test
    """
    import ray
    from ray.rllib.algorithms.ppo import PPO
    
    print("=" * 70)
    print("TEST MODÈLE RLLIB SELF-PLAY")
    print("=" * 70)
    
    ray.init(ignore_reinit_error=True)
    
    # Charger l'algorithme
    print(f"Chargement du checkpoint: {checkpoint_path}")
    algo = PPO.from_checkpoint(checkpoint_path)
    
    # Créer environnement avec visualisation
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    
    class SelfPlayFoosballEnv(MultiAgentEnv):
        """Environnement self-play où Team2 = Team1 (même modèles)"""
        def __init__(self, models_ref):
            super().__init__()
            self.base_env = FoosballMARLEnv(
                render_mode="human" if render else None,
                opponent_models=models_ref  # Team2 utilise les mêmes modèles
            )
            # L'observation space et action space sont pour un seul agent
            # mais on doit gérer les 4 agents
            self.observation_space = self.base_env.observation_space
            self.action_space = self.base_env.action_space
            self.models_ref = models_ref
            
        def reset(self, **kwargs):
            obs_dict, info = self.base_env.reset(**kwargs)
            # Retourner les observations de tous les agents
            return obs_dict, info
        
        def step(self, actions_dict):
            # Actions_dict contient les actions des 4 agents Team1
            # Team2 joue automatiquement via opponent_models dans base_env
            return self.base_env.step(actions_dict)
        
        def render(self):
            return self.base_env.render()
        
        def close(self):
            self.base_env.close()
    
    # Pas besoin d'environnements vectorisés complexes ici
    # On crée un environnement simple qui gère le self-play
    def make_env():
        return SelfPlayEnv(models)
    
    if render:
        env = DummyVecEnv([make_env])
        print("\n🎨 Mode VISUALISATION activé")
    else:
        env = SubprocVecEnv([make_env for _ in range(num_envs)])
        print(f"\n⚡ Mode RAPIDE activé ({num_envs} environnements)")
    
    # 3. Créer les 4 modèles PPO indépendants
    total_steps_target = 16384
    n_steps_per_env = total_steps_target // (num_envs if not render else 1)
    
    print(f"\nConfiguration PPO:")
    print(f"  Steps par environnement: {n_steps_per_env}")
    print(f"  Total steps collectés: {n_steps_per_env * (num_envs if not render else 1)}")
    
    # Créer un environnement temporaire pour initialiser les modèles
    temp_obs_space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
    )
    temp_action_space = gym.spaces.Box(
        low=np.array([-0.1, -np.pi], dtype=np.float32),
        high=np.array([0.1, np.pi], dtype=np.float32),
        dtype=np.float32
    )
    
    for agent_id in range(4):
        if models[agent_id] is None:
            print(f"\n[Agent {agent_id}] Création d'un nouveau modèle...")
            # On crée un modèle "dummy" qui sera mis à jour après
            from stable_baselines3.common.env_checker import check_env
            
            class DummyEnv(gym.Env):
                def __init__(self):
                    self.observation_space = temp_obs_space
                    self.action_space = temp_action_space
                def reset(self, **kwargs):
                    return self.observation_space.sample(), {}
                def step(self, action):
                    return self.observation_space.sample(), 0.0, False, False, {}
            
            dummy_env = DummyVecEnv([DummyEnv])
            
            models[agent_id] = PPO(
                "MlpPolicy",
                dummy_env,
                learning_rate=3e-4,
                n_steps=n_steps_per_env,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                tensorboard_log=f"./foosball_tensorboard_selfplay/agent{agent_id}/"
            )
            dummy_env.close()
    
    print("\n" + "=" * 70)
    print("DÉMARRAGE DE L'ENTRAÎNEMENT MULTI-AGENTS SIMULTANÉ EN SELF-PLAY")
    print("=" * 70)
    
    # 4. Wrapper pour que chaque agent puisse s'entraîner individuellement
    # mais tous les 4 agents jouent ensemble dans l'environnement
    class MultiAgentTrainingWrapper(gym.Env):
        """Wrapper qui fait agir tous les 4 agents avec leurs modèles"""
        def __init__(self, base_env, agent_id, models_dict):
            super().__init__()
            self.base_env = base_env
            self.agent_id = agent_id  # L'agent qui s'entraîne
            self.models_dict = models_dict  # Tous les modèles
            self.observation_space = base_env.base_env.observation_space
            self.action_space = gym.spaces.Box(
                low=np.array([-0.1, -np.pi], dtype=np.float32),
                high=np.array([0.1, np.pi], dtype=np.float32),
                dtype=np.float32
            )
            self.last_obs_dict = None
            
        def reset(self, **kwargs):
            obs_dict, info = self.base_env.reset(**kwargs)
            self.last_obs_dict = obs_dict
            return obs_dict[f"agent_{self.agent_id}"], info
        
        def step(self, action):
            # Construire actions pour les 4 agents
            actions = {f"agent_{self.agent_id}": action}
            
            # Les 3 autres agents jouent avec leurs modèles
            for aid in range(4):
                if aid != self.agent_id:
                    obs = self.last_obs_dict[f"agent_{aid}"]
                    other_action, _ = self.models_dict[aid].predict(obs, deterministic=False)
                    actions[f"agent_{aid}"] = other_action
            
            # Exécuter le step (Team2 joue automatiquement via self-play)
            obs_dict, rewards_dict, terminated_dict, truncated_dict, info = self.base_env.step(actions)
            self.last_obs_dict = obs_dict
            
            # Retourner pour cet agent
            return (
                obs_dict[f"agent_{self.agent_id}"],
                rewards_dict[f"agent_{self.agent_id}"],
                terminated_dict["__all__"],
                truncated_dict["__all__"],
                info
            )
        
        def render(self):
            return self.base_env.render()
        
        def close(self):
            self.base_env.close()
    
    # Créer des environnements d'entraînement pour chaque agent
    print("\nCréation des environnements d'entraînement...")
    agent_envs = []
    for agent_id in range(4):
        def make_agent_env(aid=agent_id):
            # Créer l'env de base self-play
            base = SelfPlayEnv(models)
            # Wrapper pour cet agent spécifique
            wrapper = MultiAgentTrainingWrapper(base, aid, models)
            return wrapper
        
        if render:
            agent_env = DummyVecEnv([make_agent_env])
        else:
            agent_env = SubprocVecEnv([make_agent_env for _ in range(num_envs)])
        
        agent_envs.append(agent_env)
        
        # Configurer le modèle avec son environnement
        if models[agent_id] is not None:
            models[agent_id].set_env(agent_envs[agent_id])
            models[agent_id].n_steps = n_steps_per_env
        
        print(f"  ✓ Agent {agent_id} prêt")
    
    # 5. Boucle d'entraînement
    training_steps_per_iteration = 10000
    num_iterations = timesteps // (training_steps_per_iteration * 4)
    
    try:
        for iteration in range(num_iterations):
            print(f"\n{'=' * 70}")
            print(f"ITÉRATION {iteration + 1}/{num_iterations}")
            print(f"{'=' * 70}")
            
            # Entraîner chaque agent à tour de rôle
            for agent_id in range(4):
                print(f"  [Agent {agent_id}] Entraînement ({training_steps_per_iteration} steps)...")
                models[agent_id].learn(
                    total_timesteps=training_steps_per_iteration,
                    reset_num_timesteps=False if pretrained_models_path else True,
                    progress_bar=False
                )
                
                # Team2 utilise automatiquement les modèles mis à jour (self-play)
                # Pas besoin de copier explicitement
            
            # Sauvegarder périodiquement
            if (iteration + 1) % 5 == 0:
                os.makedirs("foosball_models", exist_ok=True)
                for agent_id in range(4):
                    models[agent_id].save(f"foosball_models/selfplay_agent{agent_id}_checkpoint")
                print(f"\n💾 Checkpoint sauvegardé")
        
        print("\n" + "=" * 70)
        print("✓ ENTRAÎNEMENT TERMINÉ!")
        print("=" * 70)
        
        # Sauvegarder les modèles finaux
        os.makedirs("foosball_models", exist_ok=True)
        print("\nSauvegarde des modèles finaux:")
        for agent_id in range(4):
            models[agent_id].save(f"foosball_models/selfplay_agent{agent_id}_final")
            print(f"  ✓ Agent {agent_id}: foosball_models/selfplay_agent{agent_id}_final.zip")
        print("=" * 70)
        
        return models
    
    except KeyboardInterrupt:
        print("\n⚠️  Entraînement interrompu par l'utilisateur")
        for agent_id in range(4):
            models[agent_id].save(f"foosball_models/selfplay_agent{agent_id}_interrupted")
        print("✓ Modèles sauvegardés avec suffixe '_interrupted'")
        return models
    
    finally:
        for agent_env in agent_envs:
            agent_env.close()
        env.close()


def test_selfplay_model(models_dir: str, n_episodes: int = 10):
    """
    Teste les modèles entraînés en self-play
    
    Args:
        models_dir: Répertoire contenant les 4 modèles (ou chemin vers un modèle unique)
        n_episodes: Nombre d'épisodes de test
    """
    from stable_baselines3 import PPO
    import os
    
    print("=" * 70)
    print("TEST MODÈLES SELF-PLAY")
    print("=" * 70)
    
    # Charger les 4 modèles
    models = {}
    if os.path.isdir(models_dir):
        # Charger depuis un répertoire
        print(f"Chargement des modèles depuis: {models_dir}")
        for agent_id in range(4):
            model_path = f"{models_dir}/selfplay_agent{agent_id}_final.zip"
            models[agent_id] = PPO.load(model_path)
            print(f"  ✓ Agent {agent_id} chargé")
    else:
        # Charger un seul modèle pour tous les agents
        print(f"Chargement du modèle unique: {models_dir}")
        base_model = PPO.load(models_dir)
        for agent_id in range(4):
            models[agent_id] = base_model
    
    # Créer adversaires (copie des modèles)
    opponent_models = {i: models[i] for i in range(4)}
    
    # Créer environnement avec visualisation
    env = FoosballMARLEnv(render_mode="human", opponent_models=opponent_models)
    
    team1_wins = 0
    team2_wins = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        print(f"\n🎮 Épisode {ep + 1}/{n_episodes}")
        
        while not done:
            # Actions pour Team1 (utilise les 4 modèles indépendants)
            actions = {}
            for agent_id in range(4):
                action, _ = models[agent_id].predict(obs[f"agent_{agent_id}"], deterministic=True)
                actions[f"agent_{agent_id}"] = action
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            episode_reward += sum(rewards.values())
            done = terminated["__all__"] or truncated["__all__"]
            step_count += 1
        
        # Résultats
        if info["team1_goals"] > info["team2_goals"]:
            team1_wins += 1
            result = "✓ TEAM1 GAGNE"
        elif info["team2_goals"] > info["team1_goals"]:
            team2_wins += 1
            result = "✗ TEAM2 GAGNE"
        else:
            result = "= MATCH NUL"
        
        print(f"  Score: Team1 {info['team1_goals']} - {info['team2_goals']} Team2")
        print(f"  Steps: {step_count}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  {result}")
    
    # Statistiques finales
    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"Victoires Team1: {team1_wins}/{n_episodes} ({team1_wins/n_episodes*100:.1f}%)")
    print(f"Victoires Team2: {team2_wins}/{n_episodes} ({team2_wins/n_episodes*100:.1f}%)")
    print(f"Matchs nuls: {n_episodes - team1_wins - team2_wins}")
    print("=" * 70)
    
    env.close()


# =============================================================================
# WRAPPER RLLIB (OPTIONNEL)
# =============================================================================

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces

class RLlibFoosballEnv(MultiAgentEnv):
    """Wrapper pour rendre l'environnement compatible avec RLlib"""
    
    def __init__(self, config=None):
        super().__init__()
        self.env = FoosballMARLEnv(render_mode=config.get("render_mode") if config else None)
        
        # Définir les agents
        self._agent_ids = {f"agent_{i}" for i in range(4)}
        
        # Observation et action spaces pour RLlib
        self.observation_space = self.env.observation_space
        self.action_space = spaces.Box(
            low=np.array([self.env.slide_min, -np.pi], dtype=np.float32),
            high=np.array([self.env.slide_max, np.pi], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
    
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
    
    def step(self, action_dict):
        obs, rewards, terminated, truncated, info = self.env.step(action_dict)
        
        # RLlib utilise "dones" au lieu de "terminated"
        dones = {agent: terminated[agent] or truncated[agent] for agent in self._agent_ids}
        dones["__all__"] = terminated["__all__"] or truncated["__all__"]
        
        return obs, rewards, dones, dones, info
    
    def close(self):
        self.env.close()


# =============================================================================
# ENTRAÎNEMENT AVEC RLLIB (Ray)
# =============================================================================

def train_with_rllib():
    """Entraînement multi-agents avec RLlib PPO"""
    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec
    
    print("=" * 60)
    print("ENTRAÎNEMENT MARL AVEC RLLIB")
    print("=" * 60)
    
    # Initialiser Ray
    ray.init(ignore_reinit_error=True)
    
    # Configuration des politiques (une par équipe)
    policies = {
        "team1_policy": PolicySpec(
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
            ),
            action_space=spaces.Box(
                low=np.array([-0.1, -np.pi]), 
                high=np.array([0.1, np.pi]),
                dtype=np.float32
            ),
        ),
        "team2_policy": PolicySpec(
            observation_space=spaces.Box(
                low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
            ),
            action_space=spaces.Box(
                low=np.array([-0.1, -np.pi]), 
                high=np.array([0.1, np.pi]),
                dtype=np.float32
            ),
        ),
    }
    
    # Mapping agents -> policies
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        # Agents 0,1 -> team1, Agents 2,3 -> team2
        agent_num = int(agent_id.split("_")[1])
        return "team1_policy" if agent_num < 2 else "team2_policy"
    
    # Configuration PPO
    config = (
        PPOConfig()
        .environment(RLlibFoosballEnv, env_config={})
        .framework("torch")
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            train_batch_size=4000,
            sgd_minibatch_size=128,
            num_sgd_iter=10,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .rollouts(
            num_rollout_workers=4,
            num_envs_per_worker=1,
        )
        .resources(
            num_gpus=0,  # Mettre 1 si GPU disponible
        )
        .debugging(
            log_level="INFO"
        )
    )
    
    print("\n[1/3] Configuration PPO créée")
    print(f"  - Workers: 4")
    print(f"  - Politiques: 2 (team1_policy, team2_policy)")
    print(f"  - Learning rate: 3e-4")
    
    # Construire l'algorithme
    print("\n[2/3] Construction de l'algorithme...")
    algo = config.build()
    
    # Entraînement
    print("\n[3/3] Démarrage de l'entraînement...")
    print("=" * 60)
    
    n_iterations = 100
    
    try:
        for i in range(n_iterations):
            result = algo.train()
            
            if i % 10 == 0:
                print(f"\n📊 Itération {i}/{n_iterations}")
                print(f"  Reward mean: {result['env_runners']['episode_reward_mean']:.2f}")
                print(f"  Episode length: {result['env_runners']['episode_len_mean']:.1f}")
                
                # Sauvegarder le checkpoint
                checkpoint_dir = algo.save()
                print(f"  💾 Checkpoint: {checkpoint_dir}")
        
        print("\n" + "=" * 60)
        print("✓ ENTRAÎNEMENT TERMINÉ!")
        print("=" * 60)
        
        # Sauvegarder le modèle final
        final_checkpoint = algo.save()
        print(f"✓ Modèle final sauvegardé: {final_checkpoint}")
        
        return algo, final_checkpoint
    
    except KeyboardInterrupt:
        print("\n⚠️  Entraînement interrompu par l'utilisateur")
        checkpoint = algo.save()
        print(f"✓ Checkpoint sauvegardé: {checkpoint}")
        return algo, checkpoint
    
    finally:
        algo.stop()
        ray.shutdown()


# =============================================================================
# ÉVALUATION DU MODÈLE
# =============================================================================

def evaluate_model(checkpoint_path):
    """Évalue le modèle entraîné"""
    import ray
    from ray.rllib.algorithms.ppo import PPO
    
    print("\n" + "=" * 60)
    print("ÉVALUATION DU MODÈLE")
    print("=" * 60)
    
    ray.init(ignore_reinit_error=True)
    
    # Charger l'algorithme
    algo = PPO.from_checkpoint(checkpoint_path)
    
    # Créer un environnement de test
    env = RLlibFoosballEnv(config={"render_mode": "human"})
    
    n_episodes = 10
    episode_rewards = []
    team1_wins = 0
    team2_wins = 0
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = {f"agent_{i}": 0 for i in range(4)}
        done = False
        step_count = 0
        
        print(f"\n🎮 Épisode {ep + 1}/{n_episodes}")
        
        while not done:
            actions = {}
            for agent_id in obs.keys():
                # Déterminer la politique
                agent_num = int(agent_id.split("_")[1])
                policy_id = "team1_policy" if agent_num < 2 else "team2_policy"
                
                # Prédire l'action
                action = algo.compute_single_action(
                    obs[agent_id],
                    policy_id=policy_id,
                    explore=False
                )
                actions[agent_id] = action
            
            obs, rewards, dones, _, info = env.step(actions)
            
            for agent_id, reward in rewards.items():
                episode_reward[agent_id] += reward
            
            done = dones.get("__all__", False)
            step_count += 1
        
        # Calculer le résultat
        team1_reward = episode_reward["agent_0"] + episode_reward["agent_1"]
        team2_reward = episode_reward["agent_2"] + episode_reward["agent_3"]
        
        episode_rewards.append({
            "team1": team1_reward,
            "team2": team2_reward
        })
        
        if info["team1_goals"] > info["team2_goals"]:
            team1_wins += 1
            winner = "Team 1"
        elif info["team2_goals"] > info["team1_goals"]:
            team2_wins += 1
            winner = "Team 2"
        else:
            winner = "Draw"
        
        print(f"  Steps: {step_count}")
        print(f"  Score: Team1={info['team1_goals']}, Team2={info['team2_goals']}")
        print(f"  Gagnant: {winner}")
        print(f"  Rewards: Team1={team1_reward:.1f}, Team2={team2_reward:.1f}")
    
    # Statistiques finales
    print("\n" + "=" * 60)
    print("RÉSULTATS FINAUX")
    print("=" * 60)
    print(f"Team 1 victoires: {team1_wins}/{n_episodes}")
    print(f"Team 2 victoires: {team2_wins}/{n_episodes}")
    print(f"Égalités: {n_episodes - team1_wins - team2_wins}")
    
    avg_team1 = np.mean([ep["team1"] for ep in episode_rewards])
    avg_team2 = np.mean([ep["team2"] for ep in episode_rewards])
    print(f"\nReward moyenne Team 1: {avg_team1:.2f}")
    print(f"Reward moyenne Team 2: {avg_team2:.2f}")
    print("=" * 60)
    
    env.close()
    ray.shutdown()


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "selfplay":
            # Entraînement self-play avec SB3
            timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 500000
            render = "--render" in sys.argv
            
            # Extraire num_envs
            num_envs = 4
            for i, arg in enumerate(sys.argv):
                if arg == "--num-envs" and i + 1 < len(sys.argv):
                    num_envs = int(sys.argv[i + 1])
                    break
            
            # Extraire pretrained path
            pretrained_path = None
            for i, arg in enumerate(sys.argv):
                if arg == "--pretrained" and i + 1 < len(sys.argv):
                    pretrained_path = sys.argv[i + 1]
                    break
            
            train_selfplay_sb3(
                timesteps=timesteps,
                pretrained_models_path=pretrained_path,
                render=render,
                num_envs=num_envs
            )
        
        elif command == "test":
            # Tester un modèle self-play
            if len(sys.argv) < 3:
                print("Usage: python claude_code.py test <model_path> [n_episodes]")
                sys.exit(1)
            
            model_path = sys.argv[2]
            n_episodes = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            test_selfplay_model(model_path, n_episodes)
        
        elif command == "train_rllib":
            # Mode entraînement RLlib
            algo, checkpoint = train_with_rllib()
            print(f"\n🎯 Pour évaluer: python claude_code.py eval_rllib {checkpoint}")
        
        elif command == "eval_rllib":
            # Mode évaluation RLlib
            if len(sys.argv) < 3:
                print("Usage: python claude_code.py eval_rllib <checkpoint_path>")
                sys.exit(1)
            checkpoint_path = sys.argv[2]
            evaluate_model(checkpoint_path)
        
        else:
            print("Commandes disponibles:")
            print("  selfplay [timesteps] [--pretrained PATH] [--render] [--num-envs N]")
            print("    Entraînement self-play avec Stable-Baselines3")
            print("")
            print("  test <model_path> [n_episodes]")
            print("    Tester un modèle self-play")
            print("")
            print("  train_rllib")
            print("    Entraînement avec RLlib")
            print("")
            print("  eval_rllib <checkpoint>")
            print("    Évaluer un modèle RLlib")
            print("")
            print("Exemples:")
            print("  python claude_code.py selfplay 500000 --num-envs 8")
            print("  python claude_code.py selfplay 500000 --pretrained pretraining_models --num-envs 4")
            print("  python claude_code.py test foosball_models 20")
            print("  python claude_code.py selfplay 100000 --render")
            sys.exit(1)
    
    else:
        # Test simple
        print("=" * 60)
        print("TEST ENVIRONNEMENT MARL - BABY-FOOT (4 AGENTS)")
        print("=" * 60)
        print("\nModes disponibles:")
        print("  python script.py train    - Entraîner le modèle")
        print("  python script.py eval <checkpoint> - Évaluer le modèle")
        print("\nLancement d'un test simple...\n")
        
        env = FoosballMARLEnv(render_mode="human")
        
        observations, info = env.reset()
        print(f"✓ Environnement initialisé")
        print(f"  Agents: {len(observations)}")
        print(f"  Observation shape: {observations['agent_0'].shape}")
        
        print("\n[Test] 100 steps avec actions aléatoires...")
        
        for step in range(100):
            actions = {
                f"agent_{i}": env.action_space[f"agent_{i}"].sample()
                for i in range(4)
            }
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            
            if step % 20 == 0:
                print(f"\nStep {step}:")
                print(f"  Position balle: ({obs['agent_0'][0]:.2f}, {obs['agent_0'][1]:.2f})")
                print(f"  Rewards: {[f'{r:.2f}' for r in rewards.values()]}")
                print(f"  Score: Team1={info['team1_goals']}, Team2={info['team2_goals']}")
            
            if terminated["__all__"]:
                print(f"\n🎯 BUT MARQUÉ au step {step}!")
                print(f"Score final: Team1={info['team1_goals']}, Team2={info['team2_goals']}")
                break
        
        env.close()
        print("\n✓ Test terminé!")