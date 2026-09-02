import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_experiment_16():
    np.random.seed(42)
    random.seed(42)

    # --- 1. Multi-Agent Gridworld Setup ---
    # 2 Cooperative Agents in a 4x4 Gridworld reaching separate targets
    n_states = 16   # 4x4 Gridworld state space per agent
    n_actions = 4  # 0: Up, 1: Right, 2: Down, 3: Left
    actions = ['Up', 'Right', 'Down', 'Left']

    # Independent Q-Tables for Agent 1 and Agent 2
    Q_table_A1 = np.zeros((n_states, n_actions))
    Q_table_A2 = np.zeros((n_states, n_actions))

    alpha = 0.1       # Learning rate
    gamma = 0.99      # Discount factor
    epsilon = 0.15    # Exploration rate
    episodes = 200

    def select_action(q_table, state_idx):
        if random.random() < epsilon:
            return random.randint(0, n_actions - 1)
        return np.argmax(q_table[state_idx])

    # --- 2. MARL Training Loop (Independent Q-Learning) ---
    episode_rewards = []
    dataset_rows = []

    for ep in range(1, episodes + 1):
        s1 = 0   # Agent 1 starts at Top-Left
        s2 = 12  # Agent 2 starts at Bottom-Left
        
        target1 = 15 # Agent 1 goal (Bottom-Right)
        target2 = 3  # Agent 2 goal (Top-Right)

        total_reward = 0
        done = False
        step = 0

        while not done and step < 50:
            step += 1
            a1 = select_action(Q_table_A1, s1)
            a2 = select_action(Q_table_A2, s2)

            # Move Agent 1
            r1_pos, c1_pos = s1 // 4, s1 % 4
            if a1 == 0: r1_pos = max(0, r1_pos - 1)
            elif a1 == 1: c1_pos = min(3, c1_pos + 1)
            elif a1 == 2: r1_pos = min(3, r1_pos + 1)
            elif a1 == 3: c1_pos = max(0, c1_pos - 1)
            ns1 = r1_pos * 4 + c1_pos

            # Move Agent 2
            r2_pos, c2_pos = s2 // 4, s2 % 4
            if a2 == 0: r2_pos = max(0, r2_pos - 1)
            elif a2 == 1: c2_pos = min(3, c2_pos + 1)
            elif a2 == 2: r2_pos = min(3, r2_pos + 1)
            elif a2 == 3: c2_pos = max(0, c2_pos - 1)
            ns2 = r2_pos * 4 + c2_pos

            # Rewards and collision penalty check
            done_a1 = (ns1 == target1)
            done_a2 = (ns2 == target2)

            step_r1 = 10.0 if done_a1 else -0.1
            step_r2 = 10.0 if done_a2 else -0.1

            # Shared collision penalty if agents occupy same cell
            if ns1 == ns2:
                step_r1 -= 2.0
                step_r2 -= 2.0

            # Independent Q-Value Updates
            best_q1 = np.max(Q_table_A1[ns1]) if not done_a1 else 0.0
            Q_table_A1[s1][a1] += alpha * (step_r1 + gamma * best_q1 - Q_table_A1[s1][a1])

            best_q2 = np.max(Q_table_A2[ns2]) if not done_a2 else 0.0
            Q_table_A2[s2][a2] += alpha * (step_r2 + gamma * best_q2 - Q_table_A2[s2][a2])

            total_reward += (step_r1 + step_r2)
            done = bool(done_a1 and done_a2)

            # Save early dataset samples
            if len(dataset_rows) < 10:
                dataset_rows.append({
                    "Step": len(dataset_rows) + 1,
                    "Episode": ep,
                    "Agent1 State": s1,
                    "Agent1 Action": actions[a1],
                    "Agent2 State": s2,
                    "Agent2 Action": actions[a2],
                    "Joint Reward": round(step_r1 + step_r2, 3)
                })

            s1 = ns1
            s2 = ns2

        episode_rewards.append(total_reward)

    df_dataset = pd.DataFrame(dataset_rows)

    # --- 3. Prepare Results DataFrame ---
    df_results = pd.DataFrame({
        "Metric": ["Total Episodes", "Number of Agents", "Learning Rate (α)", "Max Joint Reward", "Final Avg Reward (Last 20 Ep)"],
        "Value": [episodes, 2, alpha, round(np.max(episode_rewards), 3), round(np.mean(episode_rewards[-20:]), 3)]
    })

    # Save CSV Results
    df_results.to_csv("results_table.csv", index=False)
    print("Saved 'results_table.csv' successfully.")

    # --- 4. Save Summary Text File ---
    summary_text = (
        "=========================================================\n"
        "EXPERIMENT 16: MULTI-AGENT RL (INDEPENDENT Q-LEARNING)\n"
        "=========================================================\n\n"
        "1. AIM:\n"
        "   To implement Independent Q-Learning (IQL) for Multi-Agent Reinforcement Learning (MARL)\n"
        "   in a cooperative gridworld task with shared collision interactions.\n\n"
        "2. PROCEDURE:\n"
        "   - Instantiate separate independent Q-tables for Agent 1 and Agent 2.\n"
        "   - Execute simultaneous environment steps and evaluate joint transition rewards.\n"
        "   - Update each agent's individual policy independently assuming stationary environments.\n\n"
        "3. KEY TAKEAWAYS:\n"
        "   - IQL provides decentralization with zero inter-agent communication overhead.\n"
        "   - Handles non-stationarity in multi-agent settings to reach stable joint policies.\n"
    )

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("Saved 'summary.txt' successfully.")

    # --- 5. Plot & Save Visualization ---
    plt.figure(figsize=(8, 4.5))
    plt.plot(episode_rewards, color='#e377c2', alpha=0.35, label='Raw Joint Reward')
    
    smoothed = pd.Series(episode_rewards).rolling(15, min_periods=1).mean()
    plt.plot(smoothed, color='#e377c2', linewidth=2.5, label='15-Episode Moving Average')
    
    plt.xlabel('Episodes', fontsize=11)
    plt.ylabel('Total Joint Reward per Episode', fontsize=11)
    plt.title('Multi-Agent Independent Q-Learning Convergence Curve', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualization.png", dpi=300)
    print("Saved 'visualization.png' successfully.")
    plt.show()

    return df_dataset, df_results

if __name__ == "__main__":
    df_dataset, df_results = run_experiment_16()
    print("\n--- DATASET (10 MARL Independent Step Samples) ---")