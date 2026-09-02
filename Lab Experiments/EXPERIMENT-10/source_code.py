import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_experiment_10():
    np.random.seed(42)
    random.seed(42)

    # --- 1. Softmax Policy Parameterization ---
    n_states = 16  # 4x4 Gridworld environment
    n_actions = 4 # 0: Up, 1: Right, 2: Down, 3: Left
    actions = ['Up', 'Right', 'Down', 'Left']

    # Policy parameters theta initialized to zeros (equal probability initially)
    theta = np.zeros((n_states, n_actions))
    
    alpha = 0.02   # Learning rate
    gamma = 0.99   # Discount factor
    episodes = 200

    def get_policy_probs(state_idx):
        e_x = np.exp(theta[state_idx] - np.max(theta[state_idx]))
        return e_x / e_x.sum()

    def select_action(state_idx):
        probs = get_policy_probs(state_idx)
        return np.random.choice(n_actions, p=probs)

    # --- 2. Training Loop (REINFORCE Algorithm) ---
    episode_rewards = []
    dataset_rows = []

    for ep in range(1, episodes + 1):
        state_idx = 0  # Start state (top-left)
        goal_state = 15  # Goal state (bottom-right)
        
        states, actions_taken, rewards = [], [], []
        done = False
        step = 0

        # Step 1: Generate Episode Trajectory
        while not done and step < 50:
            step += 1
            action = select_action(state_idx)
            
            # Simple gridworld step dynamics
            row, col = state_idx // 4, state_idx % 4
            if action == 0: row = max(0, row - 1)
            elif action == 1: col = min(3, col + 1)
            elif action == 2: row = min(3, row + 1)
            elif action == 3: col = max(0, col - 1)
            
            next_state_idx = row * 4 + col
            reward = 10.0 if next_state_idx == goal_state else -0.1

            states.append(state_idx)
            actions_taken.append(action)
            rewards.append(reward)

            if next_state_idx == goal_state:
                done = True
            state_idx = next_state_idx

        # Step 2: Compute Returns (G_t) & Update Policy Parameters (Theta)
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)

        for t in range(len(states)):
            s_t = states[t]
            a_t = actions_taken[t]
            G_t = returns[t]

            probs = get_policy_probs(s_t)
            
            # Compute Gradient: nabla log pi(a_t | s_t)
            d_log = -probs
            d_log[a_t] += 1.0

            # REINFORCE Policy Parameter Update
            theta[s_t] += alpha * (gamma ** t) * G_t * d_log

            # Collect early dataset samples
            if len(dataset_rows) < 10:
                dataset_rows.append({
                    "Step": len(dataset_rows) + 1,
                    "Episode": ep,
                    "State": s_t,
                    "Action Chosen": actions[a_t],
                    "Step Reward": reward,
                    "Discounted Return (G_t)": round(G_t, 3),
                    "Action Prob": round(probs[a_t], 3)
                })

        episode_rewards.append(sum(rewards))

    df_dataset = pd.DataFrame(dataset_rows)

    # --- 3. Prepare Results DataFrame ---
    df_results = pd.DataFrame({
        "Metric": ["Total Episodes", "Learning Rate (α)", "Discount Factor (γ)", "Max Episode Reward", "Final Avg Reward (Last 20 Ep)"],
        "Value": [episodes, alpha, gamma, round(np.max(episode_rewards), 3), round(np.mean(episode_rewards[-20:]), 3)]
    })

    # Save CSV Results
    df_results.to_csv("results_table.csv", index=False)
    print("Saved 'results_table.csv' successfully.")

    # --- 4. Save Summary Text File ---
    summary_text = (
        "=========================================================\n"
        "EXPERIMENT 10: POLICY GRADIENT METHODS (REINFORCE)\n"
        "=========================================================\n\n"
        "1. AIM:\n"
        "   To implement Monte Carlo Policy Gradient (REINFORCE) algorithm\n"
        "   to directly optimize parameterized policy probabilities.\n\n"
        "2. PROCEDURE:\n"
        "   - Parameterize policy using Softmax over action preference weights (theta).\n"
        "   - Generate full episode trajectories and compute Monte Carlo returns G_t.\n"
        "   - Update theta using policy gradient: theta <- theta + alpha * gamma^t * G_t * nabla log pi(A_t|S_t).\n\n"
        "3. KEY TAKEAWAYS:\n"
        "   - Policy Gradients optimize stochastic policies directly without value function tables.\n"
        "   - Avoids value function oscillation and handles continuous action spaces efficiently.\n"
    )

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("Saved 'summary.txt' successfully.")

    # --- 5. Plot & Save Visualization ---
    plt.figure(figsize=(8, 4.5))
    plt.plot(episode_rewards, color='#8c564b', alpha=0.35, label='Raw Episode Reward')
    
    smoothed = pd.Series(episode_rewards).rolling(15, min_periods=1).mean()
    plt.plot(smoothed, color='#8c564b', linewidth=2.5, label='15-Episode Moving Average')
    
    plt.xlabel('Episodes', fontsize=11)
    plt.ylabel('Total Reward per Episode', fontsize=11)
    plt.title('REINFORCE Policy Gradient Convergence Profile', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualization.png", dpi=300)
    print("Saved 'visualization.png' successfully.")
    plt.show()

    return df_dataset, df_results

if __name__ == "__main__":
    df_dataset, df_results = run_experiment_10()
    print("\n--- DATASET (10 Policy Gradient Trajectory Samples) ---")
    print(df_dataset.to_string(index=False))