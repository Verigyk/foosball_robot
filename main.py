import os
import argparse
import sys
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from foosball_env import FoosballEnv

def main():
    FoosballEnv()

if __name__ == "__main__":
    main()
