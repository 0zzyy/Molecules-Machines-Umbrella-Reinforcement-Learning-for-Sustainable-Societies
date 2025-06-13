"""Umbrella RL demo environment for sustainable molecular design."""

import os
import numpy as np
import gym
from gym import spaces
from rdkit import Chem
from typing import Tuple, Dict, Any, Optional
import warnings
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import yaml

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def load_qsar_model(path: str) -> Optional[Any]:
    """
    Load a QSAR model from disk with proper error handling.
    
    Args:
        path: Path to the model file
        
    Returns:
        Loaded model or None if mock version
    """
    try:
        if os.path.exists(path):
            # In real implementation, load actual model
            return None
        return None
    except Exception as e:
        warnings.warn(f"Error loading QSAR model: {str(e)}")
        return None

def predict_toxicity(mol: Chem.Mol, qsar_model: Optional[Any]) -> float:
    """
    Predict molecular toxicity with better error handling.
    
    Args:
        mol: RDKit molecule
        qsar_model: Optional QSAR model (unused in demo)
        
    Returns:
        Toxicity score (0-1)
    """
    try:
        if mol is None:
            return 1.0
        
        # Improved toxicity estimation
        num_heavy_atoms = mol.GetNumHeavyAtoms()
        num_rings = Chem.GetSSSR(mol)
        num_halogens = sum(1 for atom in mol.GetAtoms() 
                          if atom.GetSymbol() in ["F", "Cl", "Br", "I"])
        
        # More sophisticated scoring
        base_score = min(num_heavy_atoms / 20.0, 1.0)
        ring_penalty = min(num_rings * 0.1, 0.3)
        halogen_penalty = min(num_halogens * 0.2, 0.4)
        
        return min(1.0, base_score + ring_penalty + halogen_penalty)
    except Exception as e:
        warnings.warn(f"Error in toxicity prediction: {str(e)}")
        return 1.0

def predict_biodegradability(mol: Chem.Mol) -> float:
    """
    Predict biodegradability with better heuristics.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        Biodegradability score (0-1)
    """
    try:
        if mol is None:
            return 0.0
            
        num_rings = Chem.GetSSSR(mol)
        num_atoms = mol.GetNumHeavyAtoms()
        
        # More nuanced scoring
        size_penalty = min(num_atoms / 30.0, 0.5)  # Larger molecules harder to break down
        ring_penalty = min(0.25 * num_rings, 0.5)  # Rings harder to break down
        
        return max(0.0, 1.0 - size_penalty - ring_penalty)
    except Exception as e:
        warnings.warn(f"Error in biodegradability prediction: {str(e)}")
        return 0.0

def predict_equity_feasibility(mol: Chem.Mol) -> float:
    """
    Predict social equity/feasibility score with better metrics.
    
    Args:
        mol: RDKit molecule
        
    Returns:
        Equity feasibility score (0-1)
    """
    try:
        if mol is None:
            return 0.0
            
        smi = Chem.MolToSmiles(mol)
        
        # More comprehensive scoring
        penalties = {
            "Cl": 0.3,  # Chlorine concerns
            "Br": 0.3,  # Bromine similar to chlorine
            "I": 0.3,   # Iodine similar concerns
            "F": 0.2,   # Fluorine slightly less concerning
            "P": 0.2,   # Phosphorus can be concerning
            "S": 0.1    # Sulfur minor concerns
        }
        
        total_penalty = sum(penalties[elem] for elem in penalties if elem in smi)
        return max(0.0, min(0.8 - total_penalty, 1.0))
    except Exception as e:
        warnings.warn(f"Error in equity prediction: {str(e)}")
        return 0.0


class UmbrellaMoleculeEnv(gym.Env):
    """
    Improved multi-objective environment for sustainable molecular design.
    """

    def __init__(self, 
                 starting_smiles: str = config['environment']['starting_smiles'],
                 max_steps: int = config['environment']['max_steps'],
                 qsar_model_path: str = "tox_model.joblib"):
        """
        Initialize environment with better validation.
        """
        super(UmbrellaMoleculeEnv, self).__init__()

        # Validate inputs
        if not isinstance(starting_smiles, str) or not starting_smiles:
            raise ValueError("Invalid starting_smiles")
            
        if not isinstance(max_steps, int) or max_steps < 1:
            raise ValueError("max_steps must be positive integer")

        # Initialize with validation
        self.starting_mol = Chem.MolFromSmiles(starting_smiles)
        if self.starting_mol is None:
            raise ValueError(f"Invalid SMILES: {starting_smiles}")
            
        self.starting_smiles = starting_smiles
        self.max_steps = max_steps
        self.qsar_model = load_qsar_model(qsar_model_path)

        # State tracking
        self.mol = None
        self.current_step = 0
        self.previous_states = []  # Track molecular history

        # Action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Reward weights with validation
        self._set_reward_weights(*config['environment']['reward_weights'])

    def _set_reward_weights(self, w1: float, w2: float, w3: float) -> None:
        """Set reward weights with validation."""
        if not (0 <= w1 <= 1 and 0 <= w2 <= 1 and 0 <= w3 <= 1):
            raise ValueError("Weights must be between 0 and 1")
        if not np.isclose(w1 + w2 + w3, 1.0):
            raise ValueError("Weights must sum to 1")
        self.w1, self.w2, self.w3 = w1, w2, w3

    def reset(self) -> np.ndarray:
        """Reset environment state."""
        self.mol = Chem.MolFromSmiles(self.starting_smiles)
        self.current_step = 0
        self.previous_states = [self.starting_smiles]
        return self._get_obs()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one environment step with better validation and tracking."""
        if not isinstance(action, (int, np.integer)):
            raise ValueError(f"Invalid action type: {type(action)}")
        if not 0 <= action < 4:
            raise ValueError(f"Invalid action value: {action}")

        # Store previous state
        old_smiles = Chem.MolToSmiles(self.mol) if self.mol else ""
        
        # Take action
        self._take_action(action)
        self.current_step += 1

        # Track new state if changed
        new_smiles = Chem.MolToSmiles(self.mol) if self.mol else ""
        if new_smiles and new_smiles != old_smiles:
            self.previous_states.append(new_smiles)

        # Calculate rewards
        obs = self._get_obs()
        tox, bio, eq = obs
        
        # Calculate partial rewards
        r1 = float(1.0 - tox)  # Toxicity reward
        r2 = float(bio)        # Biodegradability reward
        r3 = float(eq)         # Equity reward
        
        partial_rewards = np.array([r1, r2, r3], dtype=np.float32)
        umbrella_reward = float(self.w1 * r1 + self.w2 * r2 + self.w3 * r3)

        done = (self.current_step >= self.max_steps)

        info = {
            'smiles': new_smiles,
            'partial_rewards': partial_rewards,
            'umbrella_reward': umbrella_reward,
            'step': self.current_step,
            'valid_molecule': self.mol is not None
        }
        
        return obs, umbrella_reward, done, info

    def _get_obs(self) -> np.ndarray:
        """Get current observation with safety checks."""
        if self.mol is None:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
            
        tox = predict_toxicity(self.mol, self.qsar_model)
        bio = predict_biodegradability(self.mol)
        eq = predict_equity_feasibility(self.mol)
        
        return np.array([tox, bio, eq], dtype=np.float32)

    def _take_action(self, action: int) -> None:
        """Apply molecular transformation with better error handling."""
        if self.mol is None:
            return
            
        try:
            smi = Chem.MolToSmiles(self.mol)
            
            if action == 0:
                # Add ring with proper bond
                new_smi = f"{smi}c1ccccc1"
            elif action == 1:
                # Add chlorine with explicit bond
                new_smi = f"{smi}Cl"
            elif action == 2:
                # Add hydroxyl with explicit bond
                new_smi = f"{smi}O"
            else:
                # Revert to previous state or starting state
                if len(self.previous_states) > 1:
                    new_smi = self.previous_states[-2]
                    self.previous_states.pop()
                else:
                    new_smi = self.starting_smiles

            # Validate new molecule
            candidate = Chem.MolFromSmiles(new_smi)
            if candidate is not None:
                self.mol = candidate
                
        except Exception as e:
            warnings.warn(f"Error in molecular transformation: {str(e)}")


def train_umbrella_rl(timesteps: int = config['training']['timesteps'], seed: int = config['training']['seed']) -> Tuple[PPO, UmbrellaMoleculeEnv]:
    """
    Train the RL agent with better configuration and error handling.
    
    Args:
        timesteps: Number of training timesteps
        seed: Random seed for reproducibility
        
    Returns:
        Trained model and environment
    """
    try:
        # Create and validate environment
        env = UmbrellaMoleculeEnv()
        check_env(env, warn=True)

        # Configure PPO with better hyperparameters
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=config['training']['n_steps'],
            batch_size=config['training']['batch_size'],
            gamma=0.99,
            learning_rate=config['training']['learning_rate'],
            seed=seed,
            tensorboard_log=config['training']['tensorboard_log']
        )

        # Train with progress bar
        model.learn(total_timesteps=timesteps, progress_bar=True)
        return model, env
        
    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}")


def run_demo() -> None:
    """
    Run demonstration with better logging and error handling.
    """
    try:
        print("🧪 Training Umbrella RL agent...")
        model, env = train_umbrella_rl()

        print("\n🧪 Running demonstration episode...")
        obs = env.reset()
        total_reward = 0.0
        trajectory = []

        for step in range(env.max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            # Store step information
            trajectory.append({
                'step': step,
                'smiles': info['smiles'],
                'action': int(action),
                'partial_rewards': info['partial_rewards'].tolist(),
                'umbrella_reward': float(info['umbrella_reward']),
                'valid': info['valid_molecule']
            })

            # Print step information
            print(f"\nStep {step}:")
            print(f"Action: {action}")
            print(f"Molecule: {info['smiles']}")
            print(f"Partial Rewards: {info['partial_rewards']}")
            print(f"Umbrella Reward: {info['umbrella_reward']:.3f}")

            if done:
                break

        print(f"\n✨ Demo completed!")
        print(f"Total reward: {total_reward:.3f}")

        # Save trajectory for analysis
        np.save('umbrella_trajectory.npy', trajectory)
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_demo()
