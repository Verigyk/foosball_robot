import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import os
import time

class FoosballEnv(gym.Env):

    def __init__(self):
        super(FoosballEnv, self).__init__()

        self.max_steps = 1000
        self.goals_scored = 0

        #Action space and observation space
        self.slide_min = -0.1  # Limite min de translation (mètres)
        self.slide_max = 0.1   # Limite max de translation (mètres)

        self.x_max = 1
        self.y_max = 0.5

        self.rod_slide_index = 39
        self.rod_rotate_index = 40

        #p.connect(p.GUI) 
        p.connect(p.DIRECT)

        # Action space: [position_translation, angle_rotation]
        self.action_space = gym.spaces.Box(
            low=np.array([self.slide_min, -np.pi], dtype=np.float32),
            high=np.array([self.slide_max, np.pi], dtype=np.float32),
            shape=(2,),
            dtype=np.float32
        )
        # Example: Box observation space (e.g., 4 float values)
        #x, y, vx, vy, translation, angle
        self.observation_space = gym.spaces.Box(
            low=np.array([-self.x_max, -self.y_max, -30, -30, self.slide_min, -np.pi], dtype=np.float32),
            high=np.array([self.x_max, self.y_max, 30, 30, self.slide_max, np.pi], dtype=np.float32),
            dtype=np.float32
        )

        #Initialisation board

        self.urdf_path = os.path.join(os.path.dirname(__file__), 'foosball.urdf')
        self.table_id = p.loadURDF(self.urdf_path, basePosition=[0, 0, 0.5], useFixedBase=True)

        p.changeVisualShape(self.table_id, -1, rgbaColor=[0.1, 0.4, 0.1, 1])

        #Ball

        ball_radius = 0.025
        self.ball_mass = 0.025
        self.visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=ball_radius, rgbaColor=[1, 1, 1, 1])
        self.collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=ball_radius)

        self.ball_id = p.createMultiBody(baseMass=self.ball_mass, baseCollisionShapeIndex=self.collision_shape, baseVisualShapeIndex=self.visual_shape, basePosition=[0, 0, 0.55])
        p.changeDynamics(self.ball_id, -1, restitution=0.8, rollingFriction=0.001, spinningFriction=0.001, lateralFriction=0.01)
        
        #Goal line
        
        self.goal_line_x_1 = -0.75
        self.goal_line_x_2 = 0.75

        #No cross line (For first training)
        self.line_x_2 = 0.70

        #Camera
        # Configuration de la simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1./240.,
            numSubSteps=10,
            numSolverIterations=100
        )

        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0, 0, 0.5])
        
        p.setJointMotorControl2(
            bodyIndex=self.table_id,
            jointIndex=39,
            # ----------------------------------------
            controlMode=p.POSITION_CONTROL,
            targetPosition=-0.1,
            maxVelocity = 5,
            )
        '''
        while p.isConnected():

            
            p.stepSimulation()


            time.sleep(1./240.) # Optional: control simulation speed

            #Goal detection

                    # Get the object's position (specifically the y-coordinate)
            pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            object_x_pos = pos[0]
            
            if object_x_pos >= self.line_x_2:
                print("Object crossed the line (X > 0.75)!")
                # You can add a flag, trigger an event, or stop the simulation here
        '''

    def _get_observation(self):
        try:
            """Récupère l'état actuel de l'environnement"""
            # Position et vélocité de la balle
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            ball_vel, ball_ang_vel = p.getBaseVelocity(self.ball_id)
            
            # État du joint de translation (slide)
            slide_state = p.getJointState(self.table_id, self.rod_slide_index)
            slide_position = slide_state[0]
            slide_velocity = slide_state[1]
            
            # État du joint de rotation (rotate)
            rotate_state = p.getJointState(self.table_id, self.rod_rotate_index)
            rod_angle = rotate_state[0]
            rod_velocity = rotate_state[1]
            
            observation = np.array([
                np.clip(ball_pos[0], -1, 1),       # x
                np.clip(ball_pos[1], -0.5, 0.5),       # y
                ball_vel[0],       # vx
                ball_vel[1],       # vy
                slide_position,    # position horizontale de la barre
                rod_angle,         # angle de rotation
            ], dtype=np.float32)
        
            return observation
        
        except Exception as e:
            print(f"❌ Erreur dans _get_observation: {e}")
            # Retourner une observation par défaut en cas d'erreur
            return np.zeros(6, dtype=np.float32)

    def reset(self, seed=None, options=None) -> tuple[any, any]:
        super().reset(seed=seed)

        if self.ball_id is not None:
            try:
                p.removeBody(self.ball_id)
            except:
                pass
        
        p.setGravity(0, 0, -9.81)
        
        # Créer la balle à une position aléatoire

        ball_start_x = 0.5375

        ball_start_y = np.random.uniform(-0.09, 0.09)
        
        self.ball_id = p.createMultiBody(baseMass=self.ball_mass, baseCollisionShapeIndex=self.collision_shape, baseVisualShapeIndex=self.visual_shape, basePosition=[ball_start_x, ball_start_y, 0.55])
        p.changeDynamics(self.ball_id, -1, restitution=0.8, rollingFriction=0.001, spinningFriction=0.001, lateralFriction=0.01)
        
        # Configuration des joints
        p.changeDynamics(
            bodyUniqueId=self.table_id,
            linkIndex=self.rod_slide_index,
            jointDamping=2.0,  # Plus d'amortissement pour la translation
            linearDamping=0.5
        )
        
        p.changeDynamics(
            bodyUniqueId=self.table_id,
            linkIndex=self.rod_rotate_index,
            jointDamping=0.5
        )
        
        # Stabiliser la simulation
        for _ in range(100):
            p.stepSimulation()

        initial_velocity_x = 15   # Lent
        initial_velocity_y = 0.0   # Tout droit
        angular_velocity = [0, 0, 0]  # Pas de spin

        linear_velocity = [initial_velocity_x, initial_velocity_y, 0]

        p.resetBaseVelocity(
        objectUniqueId=self.ball_id,
        linearVelocity=linear_velocity,
        angularVelocity=angular_velocity
        )
        
        # Réinitialiser les compteurs
        self.current_step = 0
        self.previous_ball_x = ball_start_x
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action):
        """Exécute une action et retourne (obs, reward, terminated, truncated, info)"""
        self.current_step += 1
        
        # Actions: [position_translation, angle_rotation]
        target_slide_position = float(action[0])
        target_rotation_angle = float(action[1])
        
        # Clipper les actions dans les limites valides
        target_slide_position = np.clip(target_slide_position, self.slide_min, self.slide_max)
        target_rotation_angle = np.clip(target_rotation_angle, -np.pi, np.pi)
        
        # Appliquer le contrôle de translation
        p.setJointMotorControl2(
            bodyIndex=self.table_id,
            jointIndex=self.rod_slide_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_slide_position,
            force=500.0,        # Force élevée pour translation rapide
            maxVelocity=2.0     # Vitesse max de translation (m/s)
        )
        
        # Appliquer le contrôle de rotation
        p.setJointMotorControl2(
            bodyIndex=self.table_id,
            jointIndex=self.rod_rotate_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_rotation_angle,
            force=200.0,
            maxVelocity=30.0    # Vitesse max de rotation (rad/s)
        )
        
        # Simuler plusieurs steps pour un meilleur contrôle
        for _ in range(4):  # 4 substeps = 60 Hz effectif
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1./240.)
        
        # Observer le nouvel état
        observation = self._get_observation()
        ball_x = observation[0]
        ball_y = observation[1]
        slide_pos = observation[4]
        
        # Calcul de la récompense
        reward = 0.0
        terminated = False
        
        # 4. GROS BONUS pour marquer un but !
        if self._check_goal(ball_x, ball_y):
            reward -= 1000.0
            terminated = True
            self.goals_scored += 1
        
        # 6. Petite pénalité par step (encourage l'efficacité)
        reward -= 0.1
        
        # Mise à jour de la position précédente
        self.previous_ball_x = ball_x
        
        # Truncation si trop de steps
        truncated = self.current_step >= self.max_steps
        
        info = {
            'ball_x': ball_x,
            'ball_y': ball_y,
            'slide_pos': slide_pos,
            'goals_scored': self.goals_scored
        }

        if (self.current_step > 10):
            terminated = True
        
        return observation, reward, terminated, truncated, info
    
    def _check_goal(self, ball_x, ball_y):
        """Vérifie si un but est marqué"""
        if ball_x >= self.goal_line_x_2:
            return True
        return False
    
    def close(self):
        """Ferme la connexion PyBullet"""
        p.disconnect(self.client)

if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    
    print("=" * 60)
    print("ENTRAÎNEMENT PPO - BABY-FOOT")
    print("=" * 60)
    
    # 1. Créer et vérifier l'environnement
    print("\n[1/5] Création de l'environnement...")
    env = FoosballEnv()
    
    print("[2/5] Vérification de l'environnement...")
    check_env(env, warn=True)
    print("✓ Environnement valide !")
    
    # 2. Créer des environnements parallèles pour accélérer l'entraînement
    print("\n[3/5] Création de 4 environnements parallèles...")
    def make_env():
        def _init():
            return FoosballEnv()
        return _init
    
    n_envs = 1
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    
    # 3. Callbacks pour sauvegarder le modèle
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./foosball_models/',
        name_prefix='ppo_foosball'
    )
    
    # 4. Créer le modèle PPO
    print("\n[4/5] Initialisation du modèle PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./foosball_tensorboard/"
    )
    
    print("✓ Modèle créé !")
    print(f"   - Politique: MLP (Multi-Layer Perceptron)")
    print(f"   - Learning rate: 3e-4")
    print(f"   - Environnements parallèles: {n_envs}")
    
    # 5. Entraînement
    print("\n[5/5] Démarrage de l'entraînement...")
    print("=" * 60)
    
    total_timesteps = 15000  # 500k steps (ajustez selon vos besoins)
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        print("\n" + "=" * 60)
        print("✓ ENTRAÎNEMENT TERMINÉ !")
        print("=" * 60)
        
        # Sauvegarder le modèle final
        model.save("ppo_foosball_final")
        print("✓ Modèle sauvegardé: ppo_foosball_final.zip")
        
    except KeyboardInterrupt:
        print("\n⚠️  Entraînement interrompu par l'utilisateur")
        model.save("ppo_foosball_interrupted")
        print("✓ Modèle sauvegardé: ppo_foosball_interrupted.zip")
    
    env.close()
    
    # =============================================================================
    # TEST DU MODÈLE ENTRAÎNÉ
    # =============================================================================
    
    print("\n" + "=" * 60)
    print("TEST DU MODÈLE")
    print("=" * 60)
    
    # Charger le modèle
    model = PPO.load("ppo_foosball_final")
    
    # Créer un environnement de test avec rendu visuel
    test_env = FoosballEnv()
    
    # Tester sur 10 épisodes
    n_test_episodes = 10
    total_rewards = []
    
    for episode in range(n_test_episodes):
        obs, info = test_env.reset()
        episode_reward = 0
        done = False
        
        print(f"\nÉpisode {episode + 1}/{n_test_episodes}")
        
        while not done:
            # Prédire l'action avec le modèle entraîné
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
        print(f"  Récompense: {episode_reward:.2f}")
    
    print("\n" + "=" * 60)
    print(f"Récompense moyenne: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Buts marqués: {test_env.goals_scored}/{n_test_episodes}")
    print("=" * 60)
    
    test_env.close()
