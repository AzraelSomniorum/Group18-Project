import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import optuna
import json
import os
from tqdm import tqdm

#optuna.logging.set_verbosity(optuna.logging.WARNING)

TRAIN_EPISODES = 15000
EVAL_EPISODES = 500

def print_success_rate(rewards_per_episode):
    """Calculate and print the success rate of the agent."""
    total_episodes = len(rewards_per_episode)
    success_count = np.sum(rewards_per_episode)
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def run(episodes, is_training=True, render=False,
        learning_rate_a = 0.4901099676146236, epsilon_decay_rate = 0.00001, 
        min_exploration_rate = 0.00013244005314245704, discount_factor_g = 0.9, end_learning_rate = 0.9):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    #discount_factor_g = 0.95 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions

    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in tqdm(range(episodes), desc="Training" if is_training else "Evaluating", leave= False, unit="ep"):
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200

        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)
            
            if terminated and reward == 0:
                reward = -1
            
            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )

            state = new_state

        if is_training:
            epsilon = max(epsilon - epsilon_decay_rate, min_exploration_rate)

            if(epsilon==min_exploration_rate):
                learning_rate_a = end_learning_rate

        if reward == 1:
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    
    plt.figure()
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')
    plt.close()

    if is_training == False:
        success_rate = print_success_rate(rewards_per_episode)
        return success_rate

    if is_training:
        f = open("frozen_lake8x8.pkl","wb")
        pickle.dump(q, f)
        f.close()

def tune_hyperparameters():
    def objective(trial):
        learning_rate_a = trial.suggest_float("learning_rate_a", 0.1, 0.7)
        #learning_rate_a = 0.9
        epsilon_decay_rate = trial.suggest_float("epsilon_decay_rate", 0.00001, 0.001)
        min_exploration_rate = trial.suggest_float("min_exploration_rate", 0.00001, 0.00005)
        #epsilon_decay_rate = (1 - min_exploration_rate) / (TRAIN_EPISODES * 0.8)
        discount_factor_g = trial.suggest_float("discount_factor_g", 0.8, 0.999)
        end_learning_rate = trial.suggest_float("end_learning_rate", 0.5, 0.999)
        
        run(episodes=TRAIN_EPISODES, 
            is_training=True, render=False, 
            learning_rate_a=learning_rate_a, 
            epsilon_decay_rate=epsilon_decay_rate,            
            min_exploration_rate=min_exploration_rate, 
            discount_factor_g= discount_factor_g, end_learning_rate=end_learning_rate)

        return run(episodes=EVAL_EPISODES, is_training=False, render=False)

    study = optuna.create_study(direction='maximize', sampler =optuna.samplers.TPESampler())
    study.optimize(objective, n_trials = 3500)

    print("\nBest parameters found:")
    print(study.best_params)
    print("Best value")
    print(study.best_value)

    # save best params to JSON
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    
    print("best_params.json saved!")

def load_best_params():
    if os.path.exists("best_params.json"):
        with open("best_params.json", "r") as f:
            return json.load(f)
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument('--train', action='store_true', help="Train the Q-learning agent")
    group.add_argument('--tune', action='store_true', help="Tune hyperparameters")
    group.add_argument('--eval', action='store_true', help="Evaluate the trained agent")
    parser.add_argument('--render', action='store_true', help="Render during evaluation")

    args = parser.parse_args()

    if args.tune:
        print("Tuning hyperparameters...")
        tune_hyperparameters()
    elif args.train :
        print("Training...")

        params = load_best_params()
        if params:
            print("Using best_params.json:")
            print(params)
            run(episodes=TRAIN_EPISODES, is_training=True, **params)
        else:
            print("No best_params.json found. Using default parameters.")
            run(episodes=TRAIN_EPISODES, is_training=True)
    elif args.eval:
        print("Evaluating...")
        run(episodes=EVAL_EPISODES, is_training=False, render=args.render)