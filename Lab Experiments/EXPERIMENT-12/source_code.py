import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_experiment_12():
    np.random.seed(42)
    random.seed(42)

    # --- 1. Environment & Policy Setup ---
    n_states = 16   # 4x4 Gridworld
    n_actions = 4  # 0: Up, 1: Right, 2: Down, 3: Left
    actions = ['Up', 'Right', 'Down', 'Left']

    actor_weights = np.zeros((n_states, n_actions))
    critic_weights = np.zeros(n_states)

    lr_actor = 0.03
    lr_critic = 0.08
    gamma = 0.99
    clip_eps = 0.2   # PPO Clipping hyperparameter (epsilon)
    ppo_epochs = 3   # Policy update iterations per trajectory
    episodes = 200

    def get_policy_probs(state_idx, weights=None):
        w = actor_weights[state_idx] if weights is None else weights[state_idx]
        e_x = np.exp(w - np.max(w))
        return e_x / e_x.sum()

    def select_action(state_idx):
        probs = get_policy_probs(state_idx)
        return np.random.choice(n_actions, p=probs)

    # --- 2. PPO Training Loop ---
    episode_rewards = []
    dataset_rows = []

    for ep in range(1, episodes + 1):
        state_idx = 0
        goal_state = 15
        
        trajectory = []
        done = False
        step = 0

        # Step 1: Collect Trajectory Batch using Old Policy
        while not done and step < 50:
            step += 1
            action = select_action(state_idx)
            old_prob = get_policy_probs(state_idx)[action]

            # Gridworld dynamics
            row, col = state_idx // 4, state_idx % 4
            if action == 0: row = max(0, row - 1)
            elif action == 1: col = min(3, col + 1)
            elif action == 2: row = min(3, row + 1)
            elif action == 3: col = max(0, col - 1)

            next_state_idx = row * 4 + col
            done = bool(next_state_idx == goal_state)
            reward = 10.0 if done else -0.1

            trajectory.append((state_idx, action, reward, next_state_idx, done, old_prob))
            state_idx = next_state_idx

        # Step 2: Compute Returns & Advantages
        G = 0
        ppo_updates = []
        for s, a, r, ns, d, old_p in reversed(trajectory):
            v_curr = critic_weights[s]
            v_next = critic_weights[ns] if not d else 0.0
            advantage = r + gamma * v_next - v_curr
            
            ppo_updates.insert(0, (s, a, advantage, old_p, r, v_curr))

        # Step 3: Clipped Surrogate Policy Optimization (PPO Epochs)
        for epoch in range(ppo_epochs):
            for s, a, advantage, old_p, r, v_curr in ppo_updates:
                # Update Critic
                critic_weights[s] += lr_critic * advantage

                # Current policy probability & Probability Ratio r_t(theta)
                new_probs = get_policy_probs(s)
                new_p = new_probs[a]
                ratio = new_p / (old_p + 1e-8)

                # Clipped Advantage Surrogate Objective
                surr1 = ratio * advantage
                surr2 = np.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
                clipped_advantage = min(surr1, surr2)

                # Policy Gradient update
                d_log = -new_probs
                d_log[a] += 1.0
                actor_weights[s] += lr_actor * clipped_advantage * d_log

                # Save early dataset samples
                if len(dataset_rows) < 10 and epoch == 0:
                    dataset_rows.append({
                        "Step": len(dataset_rows) + 1,
                        "Episode": ep,
                        "State": s,
                        "Action Chosen": actions[a],
                        "Prob Ratio r_t": round(ratio, 3),
                        "Clipped Advantage": round(clipped_advantage, 3),
                        "State Value V(s)": round(v_curr, 3)
                    })

        total_reward = sum([t[2] for t in trajectory])
        episode_rewards.append(total_reward)

    df_dataset = pd.DataFrame(dataset_rows)

    # --- 3. Prepare Results DataFrame ---
    df_results = pd.DataFrame({
        "Metric": ["Total Episodes", "Clip Epsilon (ε)", "PPO Epochs", "Actor LR (α)", "Max Episode Reward"],
        "Value": [episodes, clip_eps, ppo_epochs, lr_actor, round(np.max(episode_rewards), 3)]
    })

    # Save CSV Results
    df_results.to_csv("results_table.csv", index=False)
    print("Saved 'results_table.csv' successfully.")

    # --- 4. Save Summary Text File ---
    summary_text = (
        "=========================================================\n"
        "EXPERIMENT 12: PROXIMAL POLICY OPTIMIZATION (PPO)\n"
        "=========================================================\n\n"
        "1. AIM:\n"
        "   To implement Proximal Policy Optimization (PPO) using a Clipped\n"
        "   Surrogate Objective function to enforce stable policy updates.\n\n"
        "2. PROCEDURE:\n"
        "   - Collect state trajectories under the old policy parameters pi_old.\n"
        "   - Calculate probability ratios r_t(theta) = pi_theta(a|s) / pi_old(a|s).\n"
        "   - Compute clipped objective: min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t).\n"
        "   - Perform multiple minibatch optimization epochs per trajectory.\n\n"
        "3. KEY TAKEAWAYS:\n"
        "   - Clipping prevents destructively large policy updates, maintaining monotonicity.\n"
        "   - Enables sample-efficient multi-epoch training over single trajectory batches.\n"
    )

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("Saved 'summary.txt' successfully.")

    # --- 5. Plot & Save Visualization ---
    plt.figure(figsize=(8, 4.5))
    plt.plot(episode_rewards, color='#9467bd', alpha=0.35, label='Raw Episode Reward')
    
    smoothed = pd.Series(episode_rewards).rolling(15, min_periods=1).mean()
    plt.plot(smoothed, color='#9467bd', linewidth=2.5, label='15-Episode Moving Average')
    
    plt.xlabel('Episodes', fontsize=11)
    plt.ylabel('Total Reward per Episode', fontsize=11)
    plt.title('Proximal Policy Optimization (PPO) Convergence Curve', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualization.png", dpi=300)
    print("Saved 'visualization.png' successfully.")
    plt.show()

    return df_dataset, df_results

if __name__ == "__main__":
    df_dataset, df_results = run_experiment_12()
    print("\n--- DATASET (10 PPO Policy Update Step Samples) ---")
    print(df_dataset.to_string(index=False))