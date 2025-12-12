import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
from typing import Dict, Tuple, Literal
from enum import Enum

class TrainingPhase(Enum):
    """Phases de pré-entraînement"""
    DEFENSE = "defense"
    ATTACK = "attack"

class FoosballPreTrainingEnv(gym.Env):
    """
    Environnement de pré-entraînement pour un seul agent
    
    Phases:
    1. DEFENSE: Le ballon arrive rapidement vers les buts, l'agent doit défendre
    2. ATTACK: Le ballon est devant l'agent, il doit marquer un but
    
    Pendant l'entraînement, tous les autres joints sont en position inversée (pieds vers le haut)
    """

    def __init__(self, agent_id: int = 0, phase: TrainingPhase = TrainingPhase.DEFENSE, render_mode=None):
        super(FoosballPreTrainingEnv, self).__init__()
        
        self.agent_id = agent_id
        self.phase = phase
        self.render_mode = render_mode
        
        # Temps de réaction limité selon la phase
        if phase == TrainingPhase.DEFENSE:
            self.max_steps = 20  # ~0.6 secondes à 240Hz (réaction rapide requise)
        else:  # ATTACK
            self.max_steps = 30  # ~0.8 secondes (doit marquer rapidement)
        
        self.current_step = 0
        
        # Configuration des barres Team1 (rods 1-4, joints 2-17)
        # Positions X extraites du URDF
        self.agent_configs = {
            0: {"name": "Team1_Rod1_Goalie", "slide_idx": 2, "rotate_idx": 3, "x_pos": -0.625},      # Rod 1: Goalie
            1: {"name": "Team1_Rod2_Defense", "slide_idx": 7, "rotate_idx": 8, "x_pos": -0.45},      # Rod 2: Defenders
            2: {"name": "Team1_Rod3_Forward", "slide_idx": 11, "rotate_idx": 12, "x_pos": -0.275},   # Rod 3: Forwards (Blue in URDF)
            3: {"name": "Team1_Rod4_Midfield", "slide_idx": 16, "rotate_idx": 17, "x_pos": -0.10},   # Rod 4: Midfield
        }
        
        # Joints de l'équipe adverse Team2 (rods 5-8, joints 23-40) - à verrouiller
        # Positions X extraites du URDF
        self.opponent_joints = {
            "rod5": {"slide_idx": 23, "rotate_idx": 24, "x_pos": 0.10},    # Team B Midfield
            "rod6": {"slide_idx": 30, "rotate_idx": 31, "x_pos": 0.275},   # Team A Forwards (in URDF)
            "rod7": {"slide_idx": 35, "rotate_idx": 36, "x_pos": 0.45},    # Team B Defenders
            "rod8": {"slide_idx": 39, "rotate_idx": 40, "x_pos": 0.625},   # Team B Goalie
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
        
        # Action space: [translation, rotation]
        self.action_space = gym.spaces.Box(
            low=np.array([self.slide_min, -np.pi], dtype=np.float32),
            high=np.array([self.slide_max, np.pi], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space: [ball_x, ball_y, ball_vx, ball_vy, agent_slide, agent_angle,
        #                    rod1_slide, rod1_angle, rod2_slide, rod2_angle, rod3_slide, rod3_angle, rod4_slide, rod4_angle,
        #                    rod5_slide, rod5_angle, rod6_slide, rod6_angle, rod7_slide, rod7_angle, rod8_slide, rod8_angle]
        # Total: 6 (ball + agent) + 16 (8 rods × 2) = 22 dimensions
        self.observation_space = gym.spaces.Box(
            low=np.array([
                -self.x_max, -self.y_max, -50, -50,  # Balle
                self.slide_min, -np.pi,               # Agent actuel
                # Team1 Rods (4 rods)
                self.slide_min, -np.pi,  # Rod 1 (Goalie)
                self.slide_min, -np.pi,  # Rod 2 (Defense)
                self.slide_min, -np.pi,  # Rod 3 (Forward)
                self.slide_min, -np.pi,  # Rod 4 (Midfield)
                # Team2 Rods (4 rods)
                self.slide_min, -np.pi,  # Rod 5 (Midfield)
                self.slide_min, -np.pi,  # Rod 6 (Forward)
                self.slide_min, -np.pi,  # Rod 7 (Defense)
                self.slide_min, -np.pi,  # Rod 8 (Goalie)
            ], dtype=np.float32),
            high=np.array([
                self.x_max, self.y_max, 50, 50,      # Balle
                self.slide_max, np.pi,                # Agent actuel
                # Team1 Rods (4 rods)
                self.slide_max, np.pi,   # Rod 1 (Goalie)
                self.slide_max, np.pi,   # Rod 2 (Defense)
                self.slide_max, np.pi,   # Rod 3 (Forward)
                self.slide_max, np.pi,   # Rod 4 (Midfield)
                # Team2 Rods (4 rods)
                self.slide_max, np.pi,   # Rod 5 (Midfield)
                self.slide_max, np.pi,   # Rod 6 (Forward)
                self.slide_max, np.pi,   # Rod 7 (Defense)
                self.slide_max, np.pi,   # Rod 8 (Goalie)
            ], dtype=np.float32),
            dtype=np.float32
        )
        
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
        self.goal_line_left = -0.75
        self.goal_line_right = 0.75
        
        # Configuration physique
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)  # Mode pas-à-pas (important!)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1./240.,
            numSubSteps=10,
            numSolverIterations=100,
            enableConeFriction=1
        )
        
        if render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=0,
                cameraPitch=-45,
                cameraTargetPosition=[0, 0, 0.5]
            )
        
        # Statistiques
        self.successful_defenses = 0
        self.successful_attacks = 0
        self.total_episodes = 0

    def _lock_other_agents(self):
        """
        Met tous les autres agents de Team1 en position inversée (pieds vers le haut)
        Et verrouille TOUS les joints de Team2 en position inversée
        """
        # Verrouiller les autres agents de Team1
        for other_id, config in self.agent_configs.items():
            if other_id != self.agent_id:
                # Position inversée: rotation de π (pieds vers le haut)
                p.setJointMotorControl2(
                    bodyIndex=self.table_id,
                    jointIndex=config["rotate_idx"],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=np.pi,
                    force=5000.0,  # Force très élevée pour empêcher tout mouvement
                    maxVelocity=100  # Vitesse nulle
                )
                # Centrer la translation
                p.setJointMotorControl2(
                    bodyIndex=self.table_id,
                    jointIndex=config["slide_idx"],
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=0.0,
                    force=5000.0,
                    maxVelocity=100
                )
        
        # Verrouiller TOUS les joints de Team2 (adversaire)
        for rod_config in self.opponent_joints.values():
            p.setJointMotorControl2(
                bodyIndex=self.table_id,
                jointIndex=rod_config["rotate_idx"],
                controlMode=p.POSITION_CONTROL,
                targetPosition=np.pi,  # Pieds vers le haut
                force=5000.0,
                maxVelocity=100
            )
            p.setJointMotorControl2(
                bodyIndex=self.table_id,
                jointIndex=rod_config["slide_idx"],
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                force=5000.0,
                maxVelocity=100
            )

    def _get_observation(self) -> np.ndarray:
        """Récupère l'observation pour l'agent actuel, incluant l'état de toutes les rods"""
        try:
            # Position et vélocité de la balle
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            ball_vel, _ = p.getBaseVelocity(self.ball_id)
            
            # État de l'agent actuel
            config = self.agent_configs[self.agent_id]
            slide_state = p.getJointState(self.table_id, config["slide_idx"])
            rotate_state = p.getJointState(self.table_id, config["rotate_idx"])
            
            # Construire l'observation de base
            obs_list = [
                np.clip(ball_pos[0], -self.x_max, self.x_max),
                np.clip(ball_pos[1], -self.y_max, self.y_max),
                np.clip(ball_vel[0], -50, 50),
                np.clip(ball_vel[1], -50, 50),
                slide_state[0],
                rotate_state[0]
            ]
            
            # Ajouter l'état de toutes les rods Team1 (dans l'ordre des IDs: 0, 1, 2, 3)
            for agent_id in range(4):
                cfg = self.agent_configs[agent_id]
                slide = p.getJointState(self.table_id, cfg["slide_idx"])[0]
                angle = p.getJointState(self.table_id, cfg["rotate_idx"])[0]
                obs_list.extend([slide, angle])
            
            # Ajouter l'état de toutes les rods Team2 (dans l'ordre: rod5, rod6, rod7, rod8)
            for rod_name in ["rod5", "rod6", "rod7", "rod8"]:
                rod_cfg = self.opponent_joints[rod_name]
                slide = p.getJointState(self.table_id, rod_cfg["slide_idx"])[0]
                angle = p.getJointState(self.table_id, rod_cfg["rotate_idx"])[0]
                obs_list.extend([slide, angle])
            
            observation = np.array(obs_list, dtype=np.float32)
            
            return observation
        
        except Exception as e:
            print(f"❌ Erreur dans _get_observation: {e}")
            return np.zeros(22, dtype=np.float32)

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Réinitialiser la gravité (au cas où)
        p.setGravity(0, 0, -9.81)
        
        # Supprimer l'ancienne balle
        if self.ball_id is not None:
            try:
                p.removeBody(self.ball_id)
            except:
                pass
        
        # IMPORTANT: Reset des joints avant de créer la balle
        # Réinitialiser tous les joints à leur position par défaut
        for config in self.agent_configs.values():
            p.resetJointState(self.table_id, config["slide_idx"], 0.0, 0.0)
            p.resetJointState(self.table_id, config["rotate_idx"], 0.0, 0.0)
        
        for rod_config in self.opponent_joints.values():
            p.resetJointState(self.table_id, rod_config["slide_idx"], 0.0, 0.0)
            p.resetJointState(self.table_id, rod_config["rotate_idx"], np.pi, 0.0)  # Déjà inversés
        
        # Position et vitesse de la balle selon la phase
        config = self.agent_configs[self.agent_id]
        agent_x = config["x_pos"]
        
        if self.phase == TrainingPhase.DEFENSE:
            # DEFENSE: Ballon arrive rapidement vers les buts
            # 80% de chance que la balle se dirige vers les buts
            goes_to_goal = np.random.random() < 0.8
            
            # Déterminer de quel côté est l'agent
            if agent_x < 0:  # Agent à gauche, défend le but gauche
                ball_start_x = np.random.uniform(0.3, 0.6)
                ball_start_y = np.random.uniform(-0.15, 0.15)
                self.defending_goal = self.goal_line_left
                
                # Position du but à défendre
                goal_x = self.goal_line_left
                
                if goes_to_goal:
                    # La balle va vers le but (avec variation aléatoire sur Y)
                    goal_y = np.random.uniform(-0.15, 0.15)  # Point aléatoire dans le but
                    
                    # Vecteur de la balle vers ce point du but
                    direction_x = goal_x - ball_start_x  # Négatif (va vers la gauche)
                    direction_y = goal_y - ball_start_y
                    
                    # Normaliser et appliquer une vitesse élevée
                    norm = np.sqrt(direction_x**2 + direction_y**2)
                    speed = np.random.uniform(15, 25)  # Vitesse rapide
                    initial_velocity_x = (direction_x / norm) * speed
                    initial_velocity_y = (direction_y / norm) * speed
                else:
                    # Tir raté : la balle va dans la bonne direction X mais rate le but (trop haut ou bas en Y)
                    direction_x = goal_x - ball_start_x  # Négatif (va vers la gauche)
                    # Vise à côté du but (en dehors de la zone [-0.15, 0.15])
                    if np.random.random() < 0.5:
                        target_y = np.random.uniform(0.2, 0.4)  # Rate par le haut
                    else:
                        target_y = np.random.uniform(-0.4, -0.2)  # Rate par le bas
                    direction_y = target_y - ball_start_y
                    
                    norm = np.sqrt(direction_x**2 + direction_y**2)
                    speed = np.random.uniform(10, 20)  # Vitesse modérée
                    initial_velocity_x = (direction_x / norm) * speed
                    initial_velocity_y = (direction_y / norm) * speed
                    
            else:  # Agent à droite, défend le but droit
                ball_start_x = np.random.uniform(-0.6, -0.3)
                ball_start_y = np.random.uniform(-0.15, 0.15)
                self.defending_goal = self.goal_line_right
                
                # Position du but à défendre
                goal_x = self.goal_line_right
                
                if goes_to_goal:
                    # La balle va vers le but (avec variation aléatoire sur Y)
                    goal_y = np.random.uniform(-0.15, 0.15)  # Point aléatoire dans le but
                    
                    # Vecteur de la balle vers ce point du but
                    direction_x = goal_x - ball_start_x  # Positif (va vers la droite)
                    direction_y = goal_y - ball_start_y
                    
                    # Normaliser et appliquer une vitesse élevée
                    norm = np.sqrt(direction_x**2 + direction_y**2)
                    speed = np.random.uniform(15, 25)  # Vitesse rapide
                    initial_velocity_x = (direction_x / norm) * speed
                    initial_velocity_y = (direction_y / norm) * speed
                else:
                    # Tir raté : la balle va dans la bonne direction X mais rate le but (trop haut ou bas en Y)
                    direction_x = goal_x - ball_start_x  # Positif (va vers la droite)
                    # Vise à côté du but (en dehors de la zone [-0.15, 0.15])
                    if np.random.random() < 0.5:
                        target_y = np.random.uniform(0.2, 0.4)  # Rate par le haut
                    else:
                        target_y = np.random.uniform(-0.4, -0.2)  # Rate par le bas
                    direction_y = target_y - ball_start_y
                    
                    norm = np.sqrt(direction_x**2 + direction_y**2)
                    speed = np.random.uniform(10, 20)  # Vitesse modérée
                    initial_velocity_x = (direction_x / norm) * speed
                    initial_velocity_y = (direction_y / norm) * speed
        
        else:  # ATTACK
            # ATTACK: Ballon devant l'agent
            # 80% de chance que la balle se dirige déjà vers les buts
            goes_to_goal = np.random.random() < 0.8
            
            # Position légèrement devant l'agent
            limitation = np.random.uniform(-0.02, 0.10)
            ball_start_x = agent_x + (limitation if agent_x < 0 else -limitation)
            ball_start_y = np.random.uniform(-0.08, 0.08)
            
            # Déterminer le but à attaquer
            if agent_x < 0:  # Agent à gauche, attaque le but droit
                self.attacking_goal = self.goal_line_right
                
                # Position du but à attaquer
                goal_x = self.goal_line_right
                
                if goes_to_goal:
                    # La balle va vers le but (avec variation aléatoire sur Y)
                    goal_y = np.random.uniform(-0.15, 0.15)  # Point aléatoire dans le but
                    
                    # Vecteur de la balle vers ce point du but
                    direction_x = goal_x - ball_start_x  # Positif (va vers la droite)
                    direction_y = goal_y - ball_start_y
                    
                    # Normaliser et appliquer une vitesse modérée
                    norm = np.sqrt(direction_x**2 + direction_y**2)
                    speed = np.random.uniform(5, 15)  # Vitesse modérée pour l'attaque
                    initial_velocity_x = (direction_x / norm) * speed
                    initial_velocity_y = (direction_y / norm) * speed
                else:
                    # Tir raté : la balle va dans la bonne direction X mais rate le but (trop haut ou bas en Y)
                    direction_x = goal_x - ball_start_x  # Positif (va vers la droite)
                    # Vise à côté du but (en dehors de la zone [-0.15, 0.15])
                    if np.random.random() < 0.5:
                        target_y = np.random.uniform(0.2, 0.4)  # Rate par le haut
                    else:
                        target_y = np.random.uniform(-0.4, -0.2)  # Rate par le bas
                    direction_y = target_y - ball_start_y
                    
                    norm = np.sqrt(direction_x**2 + direction_y**2)
                    speed = np.random.uniform(3, 10)  # Vitesse faible à modérée
                    initial_velocity_x = (direction_x / norm) * speed
                    initial_velocity_y = (direction_y / norm) * speed
                    
            else:  # Agent à droite, attaque le but gauche
                self.attacking_goal = self.goal_line_left
                
                # Position du but à attaquer
                goal_x = self.goal_line_left
                
                if goes_to_goal:
                    # La balle va vers le but (avec variation aléatoire sur Y)
                    goal_y = np.random.uniform(-0.15, 0.15)  # Point aléatoire dans le but
                    
                    # Vecteur de la balle vers ce point du but
                    direction_x = goal_x - ball_start_x  # Négatif (va vers la gauche)
                    direction_y = goal_y - ball_start_y
                    
                    # Normaliser et appliquer une vitesse modérée
                    norm = np.sqrt(direction_x**2 + direction_y**2)
                    speed = np.random.uniform(5, 15)  # Vitesse modérée pour l'attaque
                    initial_velocity_x = (direction_x / norm) * speed
                    initial_velocity_y = (direction_y / norm) * speed
                else:
                    # Tir raté : la balle va dans la bonne direction X mais rate le but (trop haut ou bas en Y)
                    direction_x = goal_x - ball_start_x  # Négatif (va vers la gauche)
                    # Vise à côté du but (en dehors de la zone [-0.15, 0.15])
                    if np.random.random() < 0.5:
                        target_y = np.random.uniform(0.2, 0.4)  # Rate par le haut
                    else:
                        target_y = np.random.uniform(-0.4, -0.2)  # Rate par le bas
                    direction_y = target_y - ball_start_y
                    
                    norm = np.sqrt(direction_x**2 + direction_y**2)
                    speed = np.random.uniform(3, 10)  # Vitesse faible à modérée
                    initial_velocity_x = (direction_x / norm) * speed
                    initial_velocity_y = (direction_y / norm) * speed
        
        # Créer la balle
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
            lateralFriction=0.01,
            linearDamping=0.04,
            angularDamping=0.04
        )
        
        # Configuration des joints pour tous les agents Team1
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
        
        # Configuration des joints Team2 (adversaire)
        for rod_config in self.opponent_joints.values():
            p.changeDynamics(
                bodyUniqueId=self.table_id,
                linkIndex=rod_config["slide_idx"],
                jointDamping=2.0,
                linearDamping=0.5
            )
            p.changeDynamics(
                bodyUniqueId=self.table_id,
                linkIndex=rod_config["rotate_idx"],
                jointDamping=0.5
            )
        
        # Mettre les autres agents en position inversée
        self._lock_other_agents()
        
        # Stabiliser la simulation avec les joints verrouillés
        for _ in range(200):
            p.stepSimulation()
            # Re-verrouiller à chaque step pour être sûr
            self._lock_other_agents()
        
        # Appliquer la vitesse initiale
        p.resetBaseVelocity(
            objectUniqueId=self.ball_id,
            linearVelocity=[initial_velocity_x, initial_velocity_y, 0],
            angularVelocity=[0, 0, 0]
        )
        
        self.current_step = 0
        self.previous_ball_x = ball_start_x
        self.total_episodes += 1
        
        observation = self._get_observation()
        info = {
            "phase": self.phase.value,
            "agent_id": self.agent_id,
            "successful_defenses": self.successful_defenses,
            "successful_attacks": self.successful_attacks
        }
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Exécute une action"""
        self.current_step += 1
        
        # Actions: [translation, rotation]
        target_slide = np.clip(float(action[0]), self.slide_min, self.slide_max)
        target_rotation = np.clip(float(action[1]), -np.pi, np.pi)
        
        config = self.agent_configs[self.agent_id]
        
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
            maxVelocity=10.0
        )
        
        # Maintenir les autres agents en position inversée
        self._lock_other_agents()
        
        # Simulation
        for _ in range(4):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1./240.)
        
        # Observer le nouvel état
        observation = self._get_observation()
        ball_x = observation[0]
        ball_y = observation[1]
        ball_vx = observation[2]
        
        # Calcul de la récompense
        reward = 0.0
        terminated = False
        success = False
        
        if self.phase == TrainingPhase.DEFENSE:
            reward = self._compute_defense_reward(ball_x, ball_y, ball_vx)
            
            # Vérifier si l'agent a réussi à défendre
            if ball_x < self.goal_line_left or ball_x > self.goal_line_right:
                config = self.agent_configs[self.agent_id]
                if config["x_pos"] < 0:  # Agent gauche
                    if ball_x > self.goal_line_left:  # Ballon n'est pas passé dans son but
                        reward += 100.0
                        success = True
                        self.successful_defenses += 1
                    else:  # But encaissé
                        reward -= 100.0
                else:  # Agent droit
                    if ball_x < self.goal_line_right:  # Ballon n'est pas passé dans son but
                        reward += 100.0
                        success = True
                        self.successful_defenses += 1
                    else:  # But encaissé
                        reward -= 100.0
                terminated = True
        
        else:  # ATTACK
            reward = self._compute_attack_reward(ball_x, ball_y, ball_vx)
            
            # Vérifier si l'agent a marqué
            if ball_x < self.goal_line_left or ball_x > self.goal_line_right:
                config = self.agent_configs[self.agent_id]
                if config["x_pos"] < 0:  # Agent gauche attaque à droite
                    if ball_x > self.goal_line_right:  # But marqué
                        reward += 200.0
                        success = True
                        self.successful_attacks += 1
                    else:  # But dans son propre camp
                        reward -= 100.0
                else:  # Agent droit attaque à gauche
                    if ball_x < self.goal_line_left:  # But marqué
                        reward += 200.0
                        success = True
                        self.successful_attacks += 1
                    else:  # But dans son propre camp
                        reward -= 100.0
                terminated = True
        
        # Pénalité temporelle
        reward -= 0.1
        
        # Truncation
        truncated = self.current_step >= self.max_steps
        
        self.previous_ball_x = ball_x
        
        info = {
            "phase": self.phase.value,
            "agent_id": self.agent_id,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "success": success,
            "successful_defenses": self.successful_defenses,
            "successful_attacks": self.successful_attacks,
            "total_episodes": self.total_episodes
        }
        
        return observation, reward, terminated, truncated, info

    def _compute_defense_reward(self, ball_x: float, ball_y: float, ball_vx: float) -> float:
        """Calcule la récompense pour la phase de défense"""
        reward = 0.0
        
        config = self.agent_configs[self.agent_id]
        agent_x = config["x_pos"]
        
        # Distance entre la balle et l'agent
        distance_to_ball = abs(ball_x - agent_x)
        
        # Récompense pour se rapprocher de la balle
        reward -= distance_to_ball * 2.0
        
        # Bonus si la balle ralentit ou change de direction
        if agent_x < 0:  # Agent gauche
            if ball_vx > self.previous_ball_x - ball_x:  # Balle ralentit ou inverse
                reward += 5.0
        else:  # Agent droit
            if ball_vx < self.previous_ball_x - ball_x:
                reward += 5.0
        
        # Pénalité si la balle se rapproche du but
        if agent_x < 0:
            distance_to_goal = ball_x - self.goal_line_left
            if distance_to_goal < 0.3:
                reward -= (0.3 - distance_to_goal) * 20.0
        else:
            distance_to_goal = self.goal_line_right - ball_x
            if distance_to_goal < 0.3:
                reward -= (0.3 - distance_to_goal) * 20.0
        
        return reward

    def _compute_attack_reward(self, ball_x: float, ball_y: float, ball_vx: float) -> float:
        """Calcule la récompense pour la phase d'attaque"""
        reward = 0.0
        
        config = self.agent_configs[self.agent_id]
        agent_x = config["x_pos"]
        
        # Récompense pour faire progresser la balle vers le but adverse
        if agent_x < 0:  # Agent gauche attaque vers la droite
            progress = ball_x - self.previous_ball_x
            if progress > 0:
                reward += progress * 20.0
            
            # Bonus si la balle se rapproche du but
            distance_to_goal = self.goal_line_right - ball_x
            reward -= distance_to_goal * 2.0
            
            # Bonus si la balle va dans la bonne direction
            if ball_vx > 0:
                reward += ball_vx * 0.5
        
        else:  # Agent droit attaque vers la gauche
            progress = self.previous_ball_x - ball_x
            if progress > 0:
                reward += progress * 20.0
            
            distance_to_goal = ball_x - self.goal_line_left
            reward -= distance_to_goal * 2.0
            
            if ball_vx < 0:
                reward += abs(ball_vx) * 0.5
        
        return reward

    def close(self):
        """Ferme la connexion PyBullet"""
        if p.isConnected(self.client):
            p.disconnect(self.client)


# =============================================================================
# ENTRAÎNEMENT SÉQUENTIEL AVEC STABLE-BASELINES3
# =============================================================================

def pretrain_agent(agent_id: int, phase: TrainingPhase, timesteps: int = 100000, 
                   model_name: str = None, render: bool = False, num_envs: int = 4):
    """
    Pré-entraîne un agent sur une phase spécifique
    
    Args:
        agent_id: ID de l'agent (0-3)
        phase: Phase d'entraînement (DEFENSE ou ATTACK)
        timesteps: Nombre de timesteps d'entraînement
        model_name: Nom pour sauvegarder le modèle (optionnel)
        render: Afficher la simulation (GUI visible)
        num_envs: Nombre d'environnements parallèles (ignoré si render=True)
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder, AsyncVectorEnv
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    
    print("=" * 70)
    print(f"PRÉ-ENTRAÎNEMENT - Agent {agent_id} - Phase {phase.value.upper()}")
    print("=" * 70)
    
    # Créer l'environnement avec ou sans rendu
    def make_env():
        env = FoosballPreTrainingEnv(
            agent_id=agent_id,
            phase=phase,
            render_mode="human" if render else None
        )
        return env
    
    # Si render=True, utiliser seulement 1 environnement pour visualiser
    # Sinon, utiliser num_envs environnements parallèles pour accélérer
    if render:
        env = DummyVecEnv([make_env])
        print("🎨 Mode VISUALISATION activé (1 environnement)")
    else:
        env = AsyncVectorEnv([make_env for _ in range(num_envs)], shared_memory=True, daemon=True, context="fork")
        print(f"⚡ Mode RAPIDE activé ({num_envs} environnements parallèles)")
    
    # Callbacks
    if model_name is None:
        model_name = f"agent{agent_id}_{phase.value}"
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f'./pretraining_models/',
        name_prefix=model_name
    )
    
    # Créer le modèle
    print(f"\n[1/3] Création du modèle PPO pour l'agent {agent_id}...")

    agent_name = env.get_attr("agent_configs")[0][agent_id]['name']
    max_steps = env.get_attr("max_steps")[0]

    print(f"  Rôle: {agent_name}")
    print(f"  Phase: {phase.value}")
    print(f"  Max steps par épisode: {max_steps}")
    
    # Calculer n_steps pour que n_steps * num_envs = 16384
    total_steps_target = 16384
    n_steps_per_env = total_steps_target // num_envs
    actual_total_steps = n_steps_per_env * num_envs
    
    print(f"  Environnements parallèles: {num_envs}")
    print(f"  Steps par environnement: {n_steps_per_env}")
    print(f"  Total steps collectés: {actual_total_steps}")
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=n_steps_per_env,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=f"./pretraining_tensorboard/{model_name}/"
    )
    
    print(f"✓ Modèle créé!")
    
    # Entraînement
    print(f"\n[2/3] Entraînement sur {timesteps} timesteps...")
    print("=" * 70)
    
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        print("\n" + "=" * 70)
        print("✓ ENTRAÎNEMENT TERMINÉ!")
        print("=" * 70)
        
        # Sauvegarder le modèle final
        model.save(f"pretraining_models/{model_name}_final")
        print(f"✓ Modèle sauvegardé: pretraining_models/{model_name}_final.zip")
        
        # Statistiques
        test_env = make_env()
        obs, _ = test_env.reset()
        
        print(f"\n[3/3] Statistiques finales:")
        print(f"  Phase: {phase.value}")
        if phase == TrainingPhase.DEFENSE:
            success_rate = (test_env.successful_defenses / test_env.total_episodes * 100) if test_env.total_episodes > 0 else 0
            print(f"  Défenses réussies: {test_env.successful_defenses}/{test_env.total_episodes} ({success_rate:.1f}%)")
        else:
            success_rate = (test_env.successful_attacks / test_env.total_episodes * 100) if test_env.total_episodes > 0 else 0
            print(f"  Attaques réussies: {test_env.successful_attacks}/{test_env.total_episodes} ({success_rate:.1f}%)")
        
        test_env.close()
        
        return model
    
    except KeyboardInterrupt:
        print("\n⚠️  Entraînement interrompu par l'utilisateur")
        model.save(f"pretraining_models/{model_name}_interrupted")
        print(f"✓ Modèle sauvegardé: pretraining_models/{model_name}_interrupted.zip")
        return model
    
    finally:
        env.close()


def pretrain_all_agents(defense_timesteps: int = 100000, attack_timesteps: int = 100000, render: bool = False, num_envs: int = 4):
    """
    Pré-entraîne tous les 4 agents séquentiellement
    
    Pipeline:
    1. Agent 0 (Goalkeeper Team1) - Defense puis Attack (même modèle)
    2. Agent 1 (Defense Team1) - Defense puis Attack (même modèle)
    3. Agent 2 (Forward Team1) - Defense puis Attack (même modèle)
    4. Agent 3 (Midfield Team1) - Defense puis Attack (même modèle)
    
    Args:
        defense_timesteps: Nombre de timesteps pour chaque entraînement défense
        attack_timesteps: Nombre de timesteps pour chaque entraînement attaque
        render: Afficher la visualisation pendant l'entraînement
        num_envs: Nombre d'environnements parallèles (ignoré si render=True)
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    import shutil
    
    print("\n" + "=" * 70)
    print("PRÉ-ENTRAÎNEMENT COMPLET - 4 AGENTS - 2 PHASES")
    print("=" * 70)
    print(f"Defense timesteps: {defense_timesteps}")
    print(f"Attack timesteps: {attack_timesteps}")
    print(f"Visualisation: {'OUI (lent)' if render else 'NON (rapide)'}")
    print(f"Environnements parallèles: {num_envs if not render else 1}")
    print(f"NOTE: Le même modèle est utilisé pour défense et attaque")
    print("=" * 70)
    
    models = {}
    
    # Obtenir les noms des agents
    temp_env = FoosballPreTrainingEnv(agent_id=0, phase=TrainingPhase.DEFENSE)
    agent_configs = temp_env.agent_configs
    temp_env.close()
    
    for agent_id in range(4):
        print(f"\n{'='*70}")
        print(f"AGENT {agent_id}: {agent_configs[agent_id]['name']}")
        print(f"{'='*70}")
        
        # Phase 1: Defense
        print(f"\n>>> Phase 1/2: DEFENSE")
        defense_model = pretrain_agent(
            agent_id=agent_id,
            phase=TrainingPhase.DEFENSE,
            timesteps=defense_timesteps,
            render=render,
            num_envs=num_envs
        )
        models[f"agent{agent_id}_defense"] = defense_model
        
        # Sauvegarder une backup du modèle défense
        backup_path = f"pretraining_models/agent{agent_id}_defense_backup.zip"
        defense_model.save(backup_path)
        print(f"\n💾 Backup défense sauvegardée: {backup_path}")
        
        # Phase 2: Attack (continuer l'entraînement avec le même modèle)
        print(f"\n>>> Phase 2/2: ATTACK (continue avec le même modèle)")
        
        # Créer l'environnement d'attaque
        def make_attack_env():
            env = FoosballPreTrainingEnv(
                agent_id=agent_id,
                phase=TrainingPhase.ATTACK,
                render_mode="human" if render else None
            )
            return env
        
        # Environnements pour l'attaque
        if render:
            attack_env = DummyVecEnv([make_attack_env])
            print("🎨 Mode VISUALISATION activé (1 environnement)")
        else:
            attack_env = SubprocVecEnv([make_attack_env for _ in range(num_envs)])
            print(f"⚡ Mode RAPIDE activé ({num_envs} environnements parallèles)")
        
        # Changer l'environnement du modèle existant
        defense_model.set_env(attack_env)
        
        # Recalculer n_steps pour l'environnement d'attaque
        # PPO doit être reconfiguré avec le nouveau n_steps
        total_steps_target = 16384
        n_steps_per_env = total_steps_target // num_envs
        actual_total_steps = n_steps_per_env * num_envs
        
        print(f"Reconfiguration PPO pour l'attaque:")
        print(f"  Environnements parallèles: {num_envs}")
        print(f"  Steps par environnement: {n_steps_per_env}")
        print(f"  Total steps collectés: {actual_total_steps}")
        
        # Mettre à jour n_steps dans le modèle
        defense_model.n_steps = n_steps_per_env
        defense_model.rollout_buffer.buffer_size = n_steps_per_env
        defense_model.rollout_buffer.reset()
        
        # Callbacks pour l'attaque
        model_name = f"agent{agent_id}_attack"
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=f'./pretraining_models/',
            name_prefix=model_name
        )
        
        print(f"Entraînement sur {attack_timesteps} timesteps...")
        print("=" * 70)
        
        try:
            defense_model.learn(
                total_timesteps=attack_timesteps,
                callback=checkpoint_callback,
                progress_bar=True,
                reset_num_timesteps=False  # Continue l'entraînement
            )
            
            print("\n" + "=" * 70)
            print("✓ ENTRAÎNEMENT ATTACK TERMINÉ!")
            print("=" * 70)
            
            # Sauvegarder le modèle final (défense + attaque)
            final_path = f"pretraining_models/agent{agent_id}_combined_final.zip"
            defense_model.save(final_path)
            print(f"✓ Modèle combiné sauvegardé: {final_path}")
            
            models[f"agent{agent_id}_attack"] = defense_model
            models[f"agent{agent_id}_combined"] = defense_model
            
        except KeyboardInterrupt:
            print("\n⚠️  Entraînement interrompu par l'utilisateur")
            defense_model.save(f"pretraining_models/agent{agent_id}_attack_interrupted")
            print(f"✓ Modèle sauvegardé: pretraining_models/agent{agent_id}_attack_interrupted.zip")
        
        finally:
            attack_env.close()
        
        print(f"\n✓ Agent {agent_id} pré-entraîné avec succès!")
    
    print("\n" + "=" * 70)
    print("✓ PRÉ-ENTRAÎNEMENT COMPLET TERMINÉ!")
    print("=" * 70)
    print("\nModèles sauvegardés:")
    for i in range(4):
        print(f"  - Agent {i} Defense Backup: pretraining_models/agent{i}_defense_backup.zip")
        print(f"  - Agent {i} Defense+Attack Combined: pretraining_models/agent{i}_combined_final.zip")
    print("=" * 70)
    
    return models


# =============================================================================
# TEST ET ÉVALUATION
# =============================================================================

def test_pretrained_agent(agent_id: int, phase: TrainingPhase, model_path: str, n_episodes: int = 10):
    """Teste un agent pré-entraîné"""
    from stable_baselines3 import PPO
    
    print("=" * 70)
    print(f"TEST - Agent {agent_id} - Phase {phase.value.upper()}")
    print("=" * 70)
    
    # Charger le modèle
    model = PPO.load(model_path)
    
    # Créer l'environnement de test
    env = FoosballPreTrainingEnv(
        agent_id=agent_id,
        phase=phase,
        render_mode="human"
    )
    
    successes = 0
    total_rewards = []
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        print(f"\n🎮 Épisode {ep + 1}/{n_episodes}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            step_count += 1
        
        total_rewards.append(episode_reward)
        
        if info["success"]:
            successes += 1
            result = "✓ SUCCÈS"
        else:
            result = "✗ ÉCHEC"
        
        print(f"  Steps: {step_count}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Résultat: {result}")
    
    # Statistiques finales
    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"Succès: {successes}/{n_episodes} ({successes/n_episodes*100:.1f}%)")
    print(f"Reward moyen: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print("=" * 70)
    
    env.close()


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "train_all":
            # Entraîner tous les agents
            defense_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
            attack_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
            render = "--render" in sys.argv or "-r" in sys.argv
            
            # Extraire num_envs depuis les arguments
            num_envs = 4  # Valeur par défaut
            for i, arg in enumerate(sys.argv):
                if arg == "--num-envs" and i + 1 < len(sys.argv):
                    num_envs = int(sys.argv[i + 1])
                    break
            
            pretrain_all_agents(defense_steps, attack_steps, render=render, num_envs=num_envs)
        
        elif command == "train":
            # Entraîner un agent spécifique
            if len(sys.argv) < 4:
                print("Usage: python script.py train <agent_id> <defense|attack> [timesteps] [--render] [--num-envs N]")
                sys.exit(1)
            
            agent_id = int(sys.argv[2])
            phase_str = sys.argv[3]
            phase = TrainingPhase.DEFENSE if phase_str == "defense" else TrainingPhase.ATTACK
            timesteps = int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4].isdigit() else 100000
            render = "--render" in sys.argv or "-r" in sys.argv
            
            # Extraire num_envs depuis les arguments
            num_envs = 4  # Valeur par défaut
            for i, arg in enumerate(sys.argv):
                if arg == "--num-envs" and i + 1 < len(sys.argv):
                    num_envs = int(sys.argv[i + 1])
                    break
            
            pretrain_agent(agent_id, phase, timesteps, render=render, num_envs=num_envs)
        
        elif command == "test":
            # Tester un agent
            if len(sys.argv) < 5:
                print("Usage: python script.py test <agent_id> <defense|attack> <model_path> [n_episodes]")
                sys.exit(1)
            
            agent_id = int(sys.argv[2])
            phase_str = sys.argv[3]
            phase = TrainingPhase.DEFENSE if phase_str == "defense" else TrainingPhase.ATTACK
            model_path = sys.argv[4]
            n_episodes = int(sys.argv[5]) if len(sys.argv) > 5 else 10
            
            test_pretrained_agent(agent_id, phase, model_path, n_episodes)
        
        else:
            print("Commandes disponibles:")
            print("  train_all [defense_steps] [attack_steps] [--render] [--num-envs N]  - Entraîner tous les agents")
            print("  train <agent_id> <defense|attack> [steps] [--render] [--num-envs N] - Entraîner un agent spécifique")
            print("  test <agent_id> <defense|attack> <model>                             - Tester un agent")
            print("\nOptions:")
            print("  --render         Afficher la visualisation (lent, 1 environnement)")
            print("  --num-envs N     Nombre d'environnements parallèles (défaut: 4, ignoré si --render)")
            print("\nExemples:")
            print("  python script.py train 0 defense 50000 --render                # Voir l'entraînement en direct")
            print("  python script.py train 1 attack 100000 --num-envs 8            # Entraînement rapide avec 8 envs")
            print("  python script.py train_all 50000 50000 --num-envs 2            # Entraînement avec 2 envs")
            print("  python script.py train 0 defense 100000 --num-envs 16          # Maximum de parallélisme")
            sys.exit(1)
    
    else:
        # Test simple sans arguments
        print("=" * 70)
        print("TEST ENVIRONNEMENT PRÉ-ENTRAÎNEMENT")
        print("=" * 70)
        print("\nCommandes disponibles:")
        print("  python script.py train_all [defense_steps] [attack_steps] [--render]")
        print("  python script.py train <agent_id> <defense|attack> [timesteps] [--render]")
        print("  python script.py test <agent_id> <defense|attack> <model_path>")
        print("\nExemples:")
        print("  python script.py train 0 defense 50000 --render     # Visualiser l'entraînement")
        print("  python script.py train 1 attack 100000              # Rapide sans GUI")
        print("  python script.py train_all 50000 50000              # Tout entraîner (rapide)")
        print("  python script.py test 0 defense pretraining_models/agent0_defense_final")
        print("\n⏱️  Contraintes de temps:")
        print("  - DEFENSE: Maximum 150 steps (~0.6 secondes)")
        print("  - ATTACK: Maximum 200 steps (~0.8 secondes)")
        print("\nLancement d'un test simple...")
        
        # Test DEFENSE pour l'agent 0
        print("\n" + "=" * 70)
        print("TEST: Agent 0 (Goalie) - Phase DEFENSE (actions aléatoires)")
        print("=" * 70)
        
        env = FoosballPreTrainingEnv(
            agent_id=0,
            phase=TrainingPhase.DEFENSE,
            render_mode="human"
        )
        
        for ep in range(3):
            obs, info = env.reset()
            print(f"\n🎮 Épisode {ep + 1}/3 - Phase: {info['phase']} - Max steps: {env.max_steps}")
            
            done = False
            step_count = 0
            episode_reward = 0
            
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step_count += 1
            
            print(f"  Steps: {step_count}/{env.max_steps}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Succès: {'✓' if info['success'] else '✗'}")
            print(f"  Timeout: {'OUI' if truncated else 'NON'}")
        
        env.close()
        
        # Test ATTACK pour l'agent 3
        print("\n" + "=" * 70)
        print("TEST: Agent 3 (Midfield) - Phase ATTACK (actions aléatoires)")
        print("=" * 70)
        
        env = FoosballPreTrainingEnv(
            agent_id=3,
            phase=TrainingPhase.ATTACK,
            render_mode="human"
        )
        
        for ep in range(3):
            obs, info = env.reset()
            print(f"\n🎮 Épisode {ep + 1}/3 - Phase: {info['phase']} - Max steps: {env.max_steps}")
            
            done = False
            step_count = 0
            episode_reward = 0
            
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
                step_count += 1
            
            print(f"  Steps: {step_count}/{env.max_steps}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Succès: {'✓' if info['success'] else '✗'}")
            print(f"  Timeout: {'OUI' if truncated else 'NON'}")
        
        env.close()
        print("\n✓ Tests terminés!")