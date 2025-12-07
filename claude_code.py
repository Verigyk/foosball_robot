import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
from typing import Dict, Tuple, List

class FoosballMARLEnv(gym.Env):
    """
    Environnement de baby-foot multi-agents (MARL)
    4 agents contrôlent chacun une barre (2 par équipe)
    
    Équipe 1 (défense à gauche):
        - Agent 0: Gardien (gauche)
        - Agent 1: Attaquant (centre-gauche)
    
    Équipe 2 (défense à droite):
        - Agent 2: Attaquant (centre-droit)
        - Agent 3: Gardien (droite)
    """

    def __init__(self, render_mode=None):
        super(FoosballMARLEnv, self).__init__()
        
        self.render_mode = render_mode
        self.max_steps = 1000
        self.current_step = 0
        
        # Scores
        self.team1_goals = 0
        self.team2_goals = 0
        
        # Configuration des barres (indices dans le URDF)
        self.agent_configs = {
            0: {"name": "Team1_Goalkeeper", "slide_idx": 39, "rotate_idx": 40},
            1: {"name": "Team1_Attacker", "slide_idx": 41, "rotate_idx": 42},
            2: {"name": "Team2_Attacker", "slide_idx": 43, "rotate_idx": 44},
            3: {"name": "Team2_Goalkeeper", "slide_idx": 45, "rotate_idx": 46},
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
        #                     agent3_slide, agent3_angle]
        self.observation_space = gym.spaces.Box(
            low=np.array([
                -self.x_max, -self.y_max, -30, -30,  # Balle
                self.slide_min, -np.pi,  # Agent 0
                self.slide_min, -np.pi,  # Agent 1
                self.slide_min, -np.pi,  # Agent 2
                self.slide_min, -np.pi,  # Agent 3
            ], dtype=np.float32),
            high=np.array([
                self.x_max, self.y_max, 30, 30,
                self.slide_max, np.pi,
                self.slide_max, np.pi,
                self.slide_max, np.pi,
                self.slide_max, np.pi,
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
        self.goal_line_x_team1 = -0.75  # But équipe 1 (gauche)
        self.goal_line_x_team2 = 0.75   # But équipe 2 (droite)
        
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
        """Récupère l'observation globale pour tous les agents"""
        try:
            # Position et vélocité de la balle
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            ball_vel, _ = p.getBaseVelocity(self.ball_id)
            
            obs_list = [
                np.clip(ball_pos[0], -self.x_max, self.x_max),
                np.clip(ball_pos[1], -self.y_max, self.y_max),
                ball_vel[0],
                ball_vel[1],
            ]
            
            # États de chaque agent
            for agent_id in range(4):
                config = self.agent_configs[agent_id]
                
                slide_state = p.getJointState(self.table_id, config["slide_idx"])
                rotate_state = p.getJointState(self.table_id, config["rotate_idx"])
                
                obs_list.extend([
                    slide_state[0],  # Position translation
                    rotate_state[0]  # Angle rotation
                ])
            
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
        
        # Configuration des joints pour tous les agents
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

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Exécute les actions de tous les agents
        
        Args:
            actions: Dict avec clés "agent_0", "agent_1", "agent_2", "agent_3"
                     chaque valeur est [translation, rotation]
        
        Returns:
            observations, rewards, terminated, truncated, info (tous des dicts)
        """
        self.current_step += 1
        
        # Appliquer les actions de chaque agent
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
        Calcule les récompenses pour chaque agent
        
        Récompenses coopératives:
        - Équipe 1 (agents 0,1): gagne si but marqué à droite (x > 0.75)
        - Équipe 2 (agents 2,3): gagne si but marqué à gauche (x < -0.75)
        """
        rewards = {}
        
        # Récompense de base (petite pénalité temporelle)
        base_reward = -0.1
        
        # Bonus si la balle progresse vers le but adverse
        progress_reward_team1 = (ball_x - self.previous_ball_x) * 10  # Team 1 attaque vers +x
        progress_reward_team2 = -(ball_x - self.previous_ball_x) * 10  # Team 2 attaque vers -x
        
        # Équipe 1 (agents 0, 1)
        for agent_id in [0, 1]:
            rewards[f"agent_{agent_id}"] = base_reward + progress_reward_team1
        
        # Équipe 2 (agents 2, 3)
        for agent_id in [2, 3]:
            rewards[f"agent_{agent_id}"] = base_reward + progress_reward_team2
        
        return rewards

    def _check_goals(self, ball_x: float, ball_y: float) -> bool:
        """Vérifie si un but est marqué"""
        # But pour l'équipe 1 (balle passe la ligne de droite)
        if ball_x >= self.goal_line_x_team2:
            self.team1_goals += 1
            # Récompense massive pour l'équipe qui marque
            for agent_id in [0, 1]:
                # Sera ajouté dans le prochain step
                pass
            return True
        
        # But pour l'équipe 2 (balle passe la ligne de gauche)
        if ball_x <= self.goal_line_x_team1:
            self.team2_goals += 1
            return True
        
        return False

    def close(self):
        """Ferme la connexion PyBullet"""
        if p.isConnected(self.client):
            p.disconnect(self.client)


# =============================================================================
# WRAPPER RLLIB
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

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # Mode entraînement
        algo, checkpoint = train_with_rllib()
        print(f"\n🎯 Pour évaluer: python script.py eval {checkpoint}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "eval":
        # Mode évaluation
        if len(sys.argv) < 3:
            print("Usage: python script.py eval <checkpoint_path>")
            sys.exit(1)
        checkpoint_path = sys.argv[2]
        evaluate_model(checkpoint_path)
    
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