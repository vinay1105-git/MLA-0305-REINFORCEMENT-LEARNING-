import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_experiment_15():
    np.random.seed(42)
    random.seed(42)

    # --- 1. Environment & Dyna-Q Setup ---
    n_states = 16   # 4x4 Gridworld
    n_actions = 4  # 0: Up, 1: Right, 2: Down, 3: Left
    actions = ['Up', 'Right', 'Down', 'Left']

    # Q-Table and Model Table for Model-Based RL
    Q_table = np.zeros((n_states, n_actions))
    
    # Model stores environment transitions: Model[s][a] = (r, next_s)
    model = {}

    alpha = 0.1       # Learning rate
    gamma = 0.99      # Discount factor
    epsilon = 0.1     # Exploration rate
    planning_steps = 10  # Dyna-Q planning iterations per real step (n)
    episodes = 200

    def select_action(state_idx):
        if random.random() < epsilon:
            return random.randint(0, n_actions - 1)
        return np.argmax(Q_table[state_idx])

    # --- 2. Dyna-Q Training & Simulated Planning Loop ---
    episode_rewards = []
    dataset_rows = []

    for ep in range(1, episodes + 1):
        state_idx = 0
        goal_state = 15
        
        total_reward = 0
        done = False
        step = 0

        while not done and step < 50:
            step += 1
            action = select_action(state_idx)

            # Real Environment Step
            row, col = state_idx // 4, state_idx % 4
            if action == 0: row = max(0, row - 1)
            elif action == 1: col = min(3, col + 1)
            elif action == 2: row = min(3, row + 1)
            elif action == 3: col = max(0, col - 1)

            next_state_idx = row * 4 + col
            done = bool(next_state_idx == goal_state)
            reward = 10.0 if done else -0.1

            # --- Step A: Direct RL (Real Experience Update) ---
            best_next_q = np.max(Q_table[next_state_idx]) if not done else 0.0
            Q_table[state_idx][action] += alpha * (reward + gamma * best_next_q - Q_table[state_idx][action])

            # --- Step B: Model Learning ---
            if state_idx not in model:
                model[state_idx] = {}
            model[state_idx][action] = (reward, next_state_idx, done)

            # --- Step C: Planning (Simulated Experience Updates) ---
            for _ in range(planning_steps):
                # Sample previously visited state and action
                p_state = random.choice(list(model.keys()))
                p_action = random.choice(list(model[p_state].keys()))
                p_reward, p_next_state, p_done = model[p_state][p_action]

                # Update Q-table using simulated transition
                p_best_next_q = np.max(Q_table[p_next_state]) if not p_done else 0.0
                Q_table[p_state][p_action] += alpha * (p_reward + gamma * p_best_next_q - Q_table[p_state][p_action])

            total_reward += reward

            # Save early dataset samples
            if len(dataset_rows) < 10:
                dataset_rows.append({
                    "Step": len(dataset_rows) + 1,
                    "Episode": ep,
                    "State": state_idx,
                    "Action Chosen": actions[action],
                    "Reward": reward,
                    "Model States Learned": len(model),
                    "Planning Steps/Step": planning_steps
                })

            state_idx = next_state_idx

        episode_rewards.append(total_reward)

    df_dataset = pd.DataFrame(dataset_rows)

    # --- 3. Prepare Results DataFrame ---
    df_results = pd.DataFrame({
        "Metric": ["Total Episodes", "Planning Steps (n)", "Learning Rate (α)", "Max Episode Reward", "Final Avg Reward (Last 20 Ep)"],
        "Value": [episodes, planning_steps, alpha, round(np.max(episode_rewards), 3), round(np.mean(episode_rewards[-20:]), 3)]
    })

    # Save CSV Results
    df_results.to_csv("results_table.csv", index=False)
    print("Saved 'results_table.csv' successfully.")

    # --- 4. Save Summary Text File ---
    summary_text = (
        "=========================================================\n"
        "EXPERIMENT 15: MODEL-BASED RL (DYNA-Q IMPLEMENTATION)\n"
        "=========================================================\n\n"
        "1. AIM:\n"
        "   To implement Dyna-Q, combining direct Reinforcement Learning\n"
        "   with Model Learning and simulated background planning updates.\n\n"
        "2. PROCEDURE:\n"
        "   - Execute real environment step and update Q-Table directly.\n"
        "   - Update transition model M(s, a) -> (r, s') using real experience.\n"
        "   - Perform 'n' background planning steps by sampling previous state-action pairs from M.\n\n"
        "3. KEY TAKEAWAYS:\n"
        "   - Model-based planning significantly reduces required real-world environmental interactions.\n"
        "   - Accelerated learning speed enables fast policy convergence in sample-constrained scenarios.\n"
    )

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("Saved 'summary.txt' successfully.")

    # --- 5. Plot & Save Visualization ---
    plt.figure(figsize=(8, 4.5))
    plt.plot(episode_rewards, color='#17becf', alpha=0.35, label='Raw Episode Reward')
    
    smoothed = pd.Series(episode_rewards).rolling(15, min_periods=1).mean()
    plt.plot(smoothed, color='#17becf', linewidth=2.5, label='15-Episode Moving Average')
    
    plt.xlabel('Episodes', fontsize=11)
    plt.ylabel('Total Reward per Episode', fontsize=11)
    plt.title('Dyna-Q Model-Based Planning Convergence Curve', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualization.png", dpi=300)
    print("Saved 'visualization.png' successfully.")
    plt.show()

    return df_dataset, df_results

if __name__ == "__main__":
    df_dataset, df_results = run_experiment_15()
    print("\n--- DATASET (10 Dyna-Q Model Planning Step Samples) ---")
    print(df_dataset.to_string(index=False))