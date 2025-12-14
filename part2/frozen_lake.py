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
EVAL_EPISODES = 800

MAX_EPS = 1
DISCOUNT_FACTOR_G = 0.999

PHASE_1_RATIO = 0.85

def print_success_rate(rewards_per_episode):
    """Calculate and print the success rate of the agent."""
    total_episodes = len(rewards_per_episode)
    success_count = np.sum(rewards_per_episode)
    success_rate = (success_count / total_episodes) * 100
    print(f"✅ Success Rate: {success_rate:.2f}% ({int(success_count)} / {total_episodes} episodes)")
    return success_rate

def get_potential(state, map_size = 8):
    goal_row, goal_col = 7, 7
    row, col = state // map_size, state % map_size
    dist = abs(goal_row - row) + abs(goal_col - col)
    max_dist = 14

    return (max_dist - dist) / max_dist

def run(episodes, is_training=True, render=False, learning_rate_a = 0.03, min_eps = 0.001, min_lr = 0.001):

    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode='human' if render else None)

    if(is_training):
        q = np.random.uniform(low=0, high=0.001, size=(env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
        #q = np.ones((env.observation_space.n, env.action_space.n)) # init a 64 x 4 array
    else:
        f = open('frozen_lake8x8.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    epsilon = MAX_EPS if is_training else 0.0       # 1 = 100% random actions
    rng = np.random.default_rng()   # random number generator
    
    rewards_per_episode = np.zeros(episodes)

    best_success_rate = 0.0
    current_rate = 0.0

    epsilon_change = []
    lr_change = []
    current_lr = learning_rate_a
    
    phase_1_end = int(episodes * PHASE_1_RATIO)

    pbar = tqdm(range(episodes), desc="Training" if is_training else "Evaluating", unit = "ep")
    for i in pbar:
        state = env.reset()[0]  # states: 0 to 63, 0=top left corner,63=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 200
        
        if is_training:
            progress = i /  phase_1_end
            current_lr = max(min_lr, (learning_rate_a - (learning_rate_a - min_lr) * progress ** 0.7))

            epsilon =max(min_eps, (MAX_EPS - (MAX_EPS - min_eps) * progress ** 3))

            if i > phase_1_end :
                epsilon = 0
                current_lr = min_lr
            epsilon_change.append(epsilon)
            lr_change.append(current_lr)


        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
            else:
                action = np.argmax(q[state,:])

            new_state,reward,terminated,truncated,_ = env.step(action)
            original_reward = reward


            if is_training:
                potential_current = get_potential(state)
                if original_reward == 1:
                    reward = original_reward
                    
                elif not terminated:
                    potential_next = get_potential(new_state)
                    shaping = DISCOUNT_FACTOR_G * potential_next - potential_current
                    reward += shaping
                else:
                    reward = -potential_current * 1


            if is_training:
                if not terminated:
                    target = reward + DISCOUNT_FACTOR_G * np.max(q[new_state,:])
                else:
                    target = reward
                q[state,action] = q[state,action] + current_lr * (
                    target - q[state,action]
                )

            state = new_state

            
        if original_reward == 1:
            rewards_per_episode[i] = 1
        
        if is_training:
            window = 800
            start_idx = max(0, i - window)
            current_rate = np.mean(rewards_per_episode[start_idx:i+1])

            if i > window and current_rate > best_success_rate:
                best_success_rate = current_rate
                f = open("frozen_lake8x8.pkl","wb")
                pickle.dump(q, f)
                f.close()

            pbar.set_postfix({
                'episode':f'{i}',
                'epsilon':f'{epsilon:.5f}',
                'lr': f'{current_lr:.4f}',
                'success rate' : f'{100 * current_rate:.3f}%',
                'best success' : f'{100 * best_success_rate:.5f}%',
            })

        
    env.close()
    pbar.close()

    if is_training:
        sum_rewards = np.zeros(episodes)
        for t in range(episodes):
            sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-800):(t+1)])
    
        plt.figure()
        plt.plot(sum_rewards)
        plt.title('Training Progress')
        plt.xlabel('Episodes')
        plt.ylabel('Reward (Rolling Sum 800)')
        plt.savefig('frozen_lake8x8.png')
        print("Training graph saved to frozen_lake8x8.png")
        plt.close()

        plt.figure()
        plt.plot(epsilon_change)
        plt.title("Epsilon change")
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.savefig('epsilon_decay.png')
        plt.close()

        plt.figure()
        plt.plot(lr_change)
        plt.title("Learning rate change")
        plt.xlabel('Episodes')
        plt.ylabel('Learning')
        plt.savefig('Learning rate decay.png')
        plt.close()

    if is_training == False:
        success_rate = print_success_rate(rewards_per_episode)
        return success_rate
    
def tune_hyperparameters():
    def objective(trial):
        learning_rate_a = trial.suggest_float("learning_rate_a", 0.02, 0.04)
        min_lr = trial.suggest_float("min_lr", 0.00001, 0.00005)
        min_eps = trial.suggest_float("min_eps", 0.001, 0.003)

        run(episodes=TRAIN_EPISODES, 
            is_training=True, render=False, 
            learning_rate_a=learning_rate_a,
            min_lr=min_lr, min_eps=min_eps
        )

        return run(episodes=EVAL_EPISODES, is_training=False, render=False)

    study = optuna.create_study(direction='maximize', sampler = optuna.samplers.TPESampler())
    study.optimize(objective, n_trials = 50)

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