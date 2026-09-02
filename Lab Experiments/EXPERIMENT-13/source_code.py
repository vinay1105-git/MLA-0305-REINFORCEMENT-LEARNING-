import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_experiment_13():
    np.random.seed(42)
    random.seed(42)

    # --- 1. Environment & DDPG Setup ---
    # Continuous State & Action Space (e.g., Continuous Pendulum Control)
    state_dim = 3
    action_dim = 1
    max_action = 2.0

    # Deterministic Policy Network (Actor) & Q-Value Network (Critic)
    # Parameterized via linear/polynomial continuous function approximators
    actor_weights = np.random.randn(state_dim, action_dim) * 0.1
    critic_weights = np.random.randn(state_dim + action_dim) * 0.1

    lr_actor = 0.01
    lr_critic = 0.02
    gamma = 0.99
    tau = 0.05        # Target network soft-update parameter
    episodes = 200

    # Target Networks initialized
    target_actor_w = actor_weights.copy()
    target_critic_w = critic_weights.copy()

    def get_action(state, noise_std=0.1):
        # Deterministic action + Exploration Noise (Ornstein-Uhlenbeck / Gaussian)
        raw_action = np.dot(state, actor_weights)
        action = np.clip(raw_action + np.random.normal(0, noise_std, size=action_dim), -max_action, max_action)
        return action[0]

    def get_critic_value(state, action, weights):
        sa_pair = np.append(state, action)
        return np.dot(sa_pair, weights)

    # --- 2. Replay Buffer & DDPG Training Loop ---
    replay_buffer = []
    buffer_capacity = 10000
    episode_rewards = []
    dataset_rows = []

    for ep in range(1, episodes + 1):
        state = np.random.uniform(-0.1, 0.1, size=state_dim)
        total_reward = 0
        done = False
        step = 0

        while not done and step < 100:
            step += 1
            action = get_action(state, noise_std=max(0.01, 0.2 * (1.0 - ep / episodes)))

            # Continuous environment dynamic step (Inverted Pendulum simulation)
            theta = state[0] + action * 0.05 + np.random.normal(0, 0.01)
            next_state = np.array([np.cos(theta), np.sin(theta), theta])
            
            # Continuous reward based on angle deviation
            reward = - (theta ** 2 + 0.1 * (action ** 2))
            done = bool(step >= 100)

            # Store in Replay Buffer
            if len(replay_buffer) >= buffer_capacity:
                replay_buffer.pop(0)
            replay_buffer.append((state, action, reward, next_state, done))

            total_reward += reward

            # Sample Mini-batch & DDPG Update
            if len(replay_buffer) >= 32:
                s, a, r, ns, d = random.choice(replay_buffer)

                # 1. Compute Target Q-Value using Target Actor and Target Critic
                target_a = np.clip(np.dot(ns, target_actor_w), -max_action, max_action)[0]
                target_q = r + (1 - float(d)) * gamma * get_critic_value(ns, target_a, target_critic_w)

                # 2. Update Critic by minimizing MSE Loss
                curr_sa = np.append(s, a)
                curr_q = get_critic_value(s, a, critic_weights)
                critic_error = target_q - curr_q
                critic_weights += lr_critic * critic_error * curr_sa

                # 3. Update Actor using Deterministic Policy Gradient
                pred_a = get_action(s, noise_std=0.0)
                actor_grad = s * (critic_weights[state_dim] if len(critic_weights) > state_dim else 1.0)
                actor_weights += lr_actor * np.outer(actor_grad, [1.0])

                # 4. Soft Update Target Networks
                target_actor_w = tau * actor_weights + (1 - tau) * target_actor_w
                target_critic_w = tau * critic_weights + (1 - tau) * target_critic_w

            # Collect early dataset steps
            if len(dataset_rows) < 10:
                dataset_rows.append({
                    "Step": len(dataset_rows) + 1,
                    "Episode": ep,
                    "Continuous Action": round(float(action), 3),
                    "Reward": round(reward, 3),
                    "Replay Buffer Size": len(replay_buffer),
                    "Target Q-Value": round(target_q if len(replay_buffer) >= 32 else 0.0, 3)
                })

            state = next_state

        episode_rewards.append(total_reward)

    df_dataset = pd.DataFrame(dataset_rows)

    # --- 3. Prepare Results DataFrame ---
    df_results = pd.DataFrame({
        "Metric": ["Total Episodes", "Buffer Capacity", "Target Soft-Update (τ)", "Max Action Limit", "Final Avg Reward (Last 20 Ep)"],
        "Value": [episodes, buffer_capacity, tau, max_action, round(np.mean(episode_rewards[-20:]), 3)]
    })

    # Save CSV Results
    df_results.to_csv("results_table.csv", index=False)
    print("Saved 'results_table.csv' successfully.")

    # --- 4. Save Summary Text File ---
    summary_text = (
        "=========================================================\n"
        "EXPERIMENT 13: DEEP DETERMINISTIC POLICY GRADIENT (DDPG)\n"
        "=========================================================\n\n"
        "1. AIM:\n"
        "   To implement Deep Deterministic Policy Gradient (DDPG) algorithm\n"
        "   for continuous action-space control problems.\n\n"
        "2. PROCEDURE:\n"
        "   - Construct Deterministic Actor pi(s|theta) and Action-Value Critic Q(s, a|phi).\n"
        "   - Implement Experience Replay Buffer and continuous action exploration noise.\n"
        "   - Update Critic via TD-error and Actor via Deterministic Policy Gradient Theorem.\n"
        "   - Softly update target networks (theta' <- tau*theta + (1-tau)*theta') for tracking stability.\n\n"
        "3. KEY TAKEAWAYS:\n"
        "   - DDPG extends DQN principles to continuous control domains efficiently.\n"
        "   - Soft target network updates prevent deadly divergence and optimization instability.\n"
    )

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("Saved 'summary.txt' successfully.")

    # --- 5. Plot & Save Visualization ---
    plt.figure(figsize=(8, 4.5))
    plt.plot(episode_rewards, color='#d62728', alpha=0.35, label='Raw Episode Reward')
    
    smoothed = pd.Series(episode_rewards).rolling(15, min_periods=1).mean()
    plt.plot(smoothed, color='#d62728', linewidth=2.5, label='15-Episode Moving Average')
    
    plt.xlabel('Episodes', fontsize=11)
    plt.ylabel('Total Reward per Episode', fontsize=11)
    plt.title('DDPG Continuous Control Learning Curve', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualization.png", dpi=300)
    print("Saved 'visualization.png' successfully.")
    plt.show()

    return df_dataset, df_results

if __name__ == "__main__":
    df_dataset, df_results = run_experiment_13()
    print("\n--- DATASET (10 DDPG Continuous Control Step Samples) ---")
    print(df_dataset.to_string(index=False))