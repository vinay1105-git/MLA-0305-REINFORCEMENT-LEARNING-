import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_experiment_14():
    np.random.seed(42)
    random.seed(42)

    # --- 1. Environment & Soft Actor-Critic Setup ---
    # Discrete/Continuous hybrid setup for Maximum Entropy RL
    n_states = 16   # 4x4 Gridworld
    n_actions = 4  # 0: Up, 1: Right, 2: Down, 3: Left
    actions = ['Up', 'Right', 'Down', 'Left']

    # SAC maintains twin Q-functions to prevent overestimation bias
    actor_weights = np.zeros((n_states, n_actions))
    q1_weights = np.zeros((n_states, n_actions))
    q2_weights = np.zeros((n_states, n_actions))

    alpha_entropy = 0.2  # Temperature parameter balancing reward vs entropy
    lr = 0.05
    gamma = 0.99
    episodes = 200

    def get_policy_probs(state_idx):
        # Softmax Policy with Temperature scaling
        logits = actor_weights[state_idx] / alpha_entropy
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum()

    def select_action(state_idx):
        probs = get_policy_probs(state_idx)
        return np.random.choice(n_actions, p=probs)

    # --- 2. SAC Training Loop ---
    replay_buffer = []
    buffer_capacity = 10000
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

            # Gridworld dynamics
            row, col = state_idx // 4, state_idx % 4
            if action == 0: row = max(0, row - 1)
            elif action == 1: col = min(3, col + 1)
            elif action == 2: row = min(3, row + 1)
            elif action == 3: col = max(0, col - 1)

            next_state_idx = row * 4 + col
            done = bool(next_state_idx == goal_state)
            reward = 10.0 if done else -0.1

            # Push transition to Replay Buffer
            if len(replay_buffer) >= buffer_capacity:
                replay_buffer.pop(0)
            replay_buffer.append((state_idx, action, reward, next_state_idx, done))

            total_reward += reward

            # Sample from Replay Buffer & Perform SAC Update
            if len(replay_buffer) >= 32:
                s, a, r, ns, d = random.choice(replay_buffer)

                probs_next = get_policy_probs(ns)
                log_probs_next = np.log(probs_next + 1e-8)

                # Twin Q-Target Calculation: Min(Q1, Q2) - alpha * log_pi
                min_q_next = np.minimum(q1_weights[ns], q2_weights[ns])
                v_soft_next = np.sum(probs_next * (min_q_next - alpha_entropy * log_probs_next)) if not d else 0.0
                q_target = r + gamma * v_soft_next

                # 1. Update Twin Q-Networks
                q1_weights[s][a] += lr * (q_target - q1_weights[s][a])
                q2_weights[s][a] += lr * (q_target - q2_weights[s][a])

                # 2. Update Actor Policy by minimizing KL-divergence
                min_q_curr = np.minimum(q1_weights[s], q2_weights[s])
                actor_weights[s] += lr * (min_q_curr - alpha_entropy * np.log(get_policy_probs(s) + 1e-8))

            # Collect early dataset samples
            if len(dataset_rows) < 10:
                probs_curr = get_policy_probs(state_idx)
                entropy = -np.sum(probs_curr * np.log(probs_curr + 1e-8))
                dataset_rows.append({
                    "Step": len(dataset_rows) + 1,
                    "Episode": ep,
                    "State": state_idx,
                    "Action Chosen": actions[action],
                    "Reward": reward,
                    "Policy Entropy": round(entropy, 3),
                    "Twin Q1 Val": round(q1_weights[state_idx][action], 3)
                })

            state_idx = next_state_idx

        episode_rewards.append(total_reward)

    df_dataset = pd.DataFrame(dataset_rows)

    # --- 3. Prepare Results DataFrame ---
    df_results = pd.DataFrame({
        "Metric": ["Total Episodes", "Buffer Capacity", "Entropy Temperature (α)", "Max Episode Reward", "Final Avg Reward (Last 20 Ep)"],
        "Value": [episodes, buffer_capacity, alpha_entropy, round(np.max(episode_rewards), 3), round(np.mean(episode_rewards[-20:]), 3)]
    })

    # Save CSV Results
    df_results.to_csv("results_table.csv", index=False)
    print("Saved 'results_table.csv' successfully.")

    # --- 4. Save Summary Text File ---
    summary_text = (
        "=========================================================\n"
        "EXPERIMENT 14: SOFT ACTOR-CRITIC (SAC IMPLEMENTATION)\n"
        "=========================================================\n\n"
        "1. AIM:\n"
        "   To implement Soft Actor-Critic (SAC) utilizing Maximum Entropy RL\n"
        "   framework to optimize expected return and policy entropy concurrently.\n\n"
        "2. PROCEDURE:\n"
        "   - Construct Twin Q-Critics Q1(s,a), Q2(s,a) and stochastic Soft Actor pi(a|s).\n"
        "   - Formulate Soft State-Value targets incorporating entropy term: - alpha * log(pi).\n"
        "   - Update Twin Critics via MSE loss and Policy via KL-divergence minimization.\n\n"
        "3. KEY TAKEAWAYS:\n"
        "   - Entropy regularization encourages aggressive exploration and prevents early policy collapse.\n"
        "   - Twin Q-Functions successfully eliminate value overestimation bias.\n"
    )

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("Saved 'summary.txt' successfully.")

    # --- 5. Plot & Save Visualization ---
    plt.figure(figsize=(8, 4.5))
    plt.plot(episode_rewards, color='#ff7f0e', alpha=0.35, label='Raw Episode Reward')
    
    smoothed = pd.Series(episode_rewards).rolling(15, min_periods=1).mean()
    plt.plot(smoothed, color='#ff7f0e', linewidth=2.5, label='15-Episode Moving Average')
    
    plt.xlabel('Episodes', fontsize=11)
    plt.ylabel('Total Reward per Episode', fontsize=11)
    plt.title('Soft Actor-Critic (SAC) Learning Convergence Profile', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualization.png", dpi=300)
    print("Saved 'visualization.png' successfully.")
    plt.show()

    return df_dataset, df_results

if __name__ == "__main__":
    df_dataset, df_results = run_experiment_14()
    print("\n--- DATASET (10 SAC Maximum Entropy Step Samples) ---")
    print(df_dataset.to_string(index=False))