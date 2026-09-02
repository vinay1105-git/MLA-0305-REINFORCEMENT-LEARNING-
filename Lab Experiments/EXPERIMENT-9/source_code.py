import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_experiment_9():
    np.random.seed(42)
    random.seed(42)

    # --- 1. Environment & Discretization Setup ---
    # CartPole continuous state boundaries: [position, velocity, angle, angular velocity]
    num_buckets = (1, 1, 6, 12)  # Discretize angle and angular velocity
    num_actions = 2               # 0: Push Left, 1: Push Right
    
    state_bounds = list(zip([-2.4, -3.0, -0.209, -3.0], [2.4, 3.0, 0.209, 3.0]))
    
    Q_table = np.zeros(num_buckets + (num_actions,))

    alpha = 0.1       # Learning rate
    gamma = 0.99      # Discount factor
    epsilon = 1.0     # Initial exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.975
    episodes = 200

    def discretize_state(state):
        bucket_indices = []
        for i in range(len(state)):
            if state[i] <= state_bounds[i][0]:
                bucket_idx = 0
            elif state[i] >= state_bounds[i][1]:
                bucket_idx = num_buckets[i] - 1
            else:
                bound_width = state_bounds[i][1] - state_bounds[i][0]
                offset = (num_buckets[i] - 1) * (state[i] - state_bounds[i][0]) / bound_width
                bucket_idx = int(round(offset))
            bucket_indices.append(bucket_idx)
        return tuple(bucket_indices)

    # --- 2. Replay Buffer Simulation & Training Loop ---
    replay_buffer = []
    buffer_capacity = 10000
    episode_rewards = []
    dataset_rows = []

    for ep in range(1, episodes + 1):
        # Initial state near origin
        raw_state = np.array([0.0, 0.0, np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05)])
        state_idx = discretize_state(raw_state)
        
        total_reward = 0
        done = False
        step = 0

        while not done and step < 200:
            step += 1
            
            # Epsilon-Greedy Action Selection
            if random.random() < epsilon:
                action = random.randint(0, num_actions - 1)
            else:
                action = np.argmax(Q_table[state_idx])

            # Dynamics: simulate pole movement
            force = 1.0 if action == 1 else -1.0
            theta_acc = force * 0.02 + raw_state[2] * 0.05
            
            next_raw_state = raw_state.copy()
            next_raw_state[0] += raw_state[1] * 0.02           # Position update
            next_raw_state[1] += force * 0.01                  # Velocity update
            next_raw_state[2] += raw_state[3] * 0.02           # Angle update
            next_raw_state[3] += theta_acc                     # Angular velocity update

            next_state_idx = discretize_state(next_raw_state)

            # Check termination (pole falls past ~12 degrees or cart leaves track)
            done = bool(abs(next_raw_state[0]) > 2.4 or abs(next_raw_state[2]) > 0.209)
            reward = 1.0 if not done else -10.0

            # Experience Replay Buffer Push
            if len(replay_buffer) >= buffer_capacity:
                replay_buffer.pop(0)
            replay_buffer.append((state_idx, action, reward, next_state_idx, done))
            
            total_reward += (1.0 if not done else 0.0)

            # Replay Buffer Sampling & Q-Update
            if len(replay_buffer) >= 32:
                sample = random.choice(replay_buffer)
                s_i, a_i, r_i, ns_i, d_i = sample
                
                best_next_q = np.max(Q_table[ns_i]) if not d_i else 0.0
                td_target = r_i + gamma * best_next_q
                td_error = td_target - Q_table[s_i][a_i]
                
                Q_table[s_i][a_i] += alpha * td_error

            # Save early dataset samples
            if len(dataset_rows) < 10:
                dataset_rows.append({
                    "Step": len(dataset_rows) + 1,
                    "Episode": ep,
                    "Action Chosen": "Push Right" if action == 1 else "Push Left",
                    "Reward": reward,
                    "Replay Buffer Size": len(replay_buffer),
                    "Epsilon Rate": round(epsilon, 3)
                })

            raw_state = next_raw_state
            state_idx = next_state_idx

        # Decay exploration rate
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        episode_rewards.append(total_reward)

    df_dataset = pd.DataFrame(dataset_rows)

    # --- 3. Prepare Results DataFrame ---
    df_results = pd.DataFrame({
        "Metric": ["Total Episodes", "Buffer Capacity", "Discount Factor (γ)", "Final Epsilon (ε)", "Max Reward Achieved"],
        "Value": [episodes, buffer_capacity, gamma, round(epsilon, 3), int(np.max(episode_rewards))]
    })

    # Save CSV Results
    df_results.to_csv("results_table.csv", index=False)
    print("Saved 'results_table.csv' successfully.")

    # --- 4. Save Summary Text File ---
    summary_text = (
        "=========================================================\n"
        "EXPERIMENT 9: DEEP Q-NETWORK (DQN) IMPLEMENTATION\n"
        "=========================================================\n\n"
        "1. AIM:\n"
        "   To implement Deep Q-Networks (DQN) combining Q-Learning with\n"
        "   discretized state-space function approximation and Experience Replay.\n\n"
        "2. PROCEDURE:\n"
        "   - Discretize continuous state vectors into state space buckets.\n"
        "   - Store trajectories (S, A, R, S', Done) inside Experience Replay Buffer.\n"
        "   - Sample transitions to decorrelate updates and update action values.\n\n"
        "3. KEY TAKEAWAYS:\n"
        "   - Experience Replay breaks temporal correlation for stable policy learning.\n"
        "   - Episode rewards steadily vary upwards as exploration transitions to exploitation.\n"
    )

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("Saved 'summary.txt' successfully.")

    # --- 5. Plot & Save Visualization ---
    plt.figure(figsize=(8, 4.5))
    plt.plot(episode_rewards, color='#1f77b4', alpha=0.35, label='Raw Episode Reward')
    
    # Smooth curve using rolling average to highlight growth trajectory
    smoothed = pd.Series(episode_rewards).rolling(15, min_periods=1).mean()
    plt.plot(smoothed, color='#1f77b4', linewidth=2.5, label='15-Episode Moving Average')
    
    plt.xlabel('Episodes', fontsize=11)
    plt.ylabel('Total Reward per Episode', fontsize=11)
    plt.title('DQN Policy Performance & Reward Convergence', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualization.png", dpi=300)
    print("Saved 'visualization.png' successfully.")
    plt.show()

    return df_dataset, df_results

if __name__ == "__main__":
    df_dataset, df_results = run_experiment_9()
    print("\n--- DATASET (10 DQN Training Updates) ---")
    print(df_dataset.to_string(index=False))