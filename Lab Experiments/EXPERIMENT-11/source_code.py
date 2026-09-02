import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_experiment_11():
    np.random.seed(42)
    random.seed(42)

    # --- 1. Environment & Network Parameters Setup ---
    n_states = 16   # 4x4 Gridworld
    n_actions = 4  # 0: Up, 1: Right, 2: Down, 3: Left
    actions = ['Up', 'Right', 'Down', 'Left']

    # Actor parameters (Policy probabilities) & Critic parameters (State values)
    actor_weights = np.zeros((n_states, n_actions))
    critic_weights = np.zeros(n_states)

    lr_actor = 0.05
    lr_critic = 0.1
    gamma = 0.99
    episodes = 200

    def get_policy_probs(state_idx):
        e_x = np.exp(actor_weights[state_idx] - np.max(actor_weights[state_idx]))
        return e_x / e_x.sum()

    def select_action(state_idx):
        probs = get_policy_probs(state_idx)
        return np.random.choice(n_actions, p=probs)

    # --- 2. Advantage Actor-Critic (A2C) Training Loop ---
    episode_rewards = []
    dataset_rows = []

    for ep in range(1, episodes + 1):
        state_idx = 0       # Start state (top-left)
        goal_state = 15     # Goal state (bottom-right)
        
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

            # Advantage Calculation: A(s, a) = r + gamma * V(s') - V(s)
            v_curr = critic_weights[state_idx]
            v_next = critic_weights[next_state_idx] if not done else 0.0
            td_target = reward + gamma * v_next
            advantage = td_target - v_curr

            # Update Critic (State Value Function)
            critic_weights[state_idx] += lr_critic * advantage

            # Update Actor (Policy Gradient weighted by Advantage)
            probs = get_policy_probs(state_idx)
            d_log = -probs
            d_log[action] += 1.0
            actor_weights[state_idx] += lr_actor * advantage * d_log

            total_reward += reward

            # Collect early dataset steps
            if len(dataset_rows) < 10:
                dataset_rows.append({
                    "Step": len(dataset_rows) + 1,
                    "Episode": ep,
                    "State": state_idx,
                    "Action Chosen": actions[action],
                    "Reward": reward,
                    "Advantage": round(advantage, 3),
                    "State Value V(s)": round(v_curr, 3)
                })

            state_idx = next_state_idx

        episode_rewards.append(total_reward)

    df_dataset = pd.DataFrame(dataset_rows)

    # --- 3. Prepare Results DataFrame ---
    df_results = pd.DataFrame({
        "Metric": ["Total Episodes", "Actor LR (α_actor)", "Critic LR (α_critic)", "Discount Factor (γ)", "Max Episode Reward"],
        "Value": [episodes, lr_actor, lr_critic, gamma, round(np.max(episode_rewards), 3)]
    })

    # Save CSV Results
    df_results.to_csv("results_table.csv", index=False)
    print("Saved 'results_table.csv' successfully.")

    # --- 4. Save Summary Text File ---
    summary_text = (
        "=========================================================\n"
        "EXPERIMENT 11: ACTOR-CRITIC METHODS (A2C IMPLEMENTATION)\n"
        "=========================================================\n\n"
        "1. AIM:\n"
        "   To implement Advantage Actor-Critic (A2C) architecture combining\n"
        "   value-based evaluation (Critic) with policy-based action selection (Actor).\n\n"
        "2. PROCEDURE:\n"
        "   - Maintain Actor parameters (policy) and Critic parameters (baseline state value).\n"
        "   - Calculate Temporal Difference Advantage: A(s, a) = R + gamma * V(s') - V(s).\n"
        "   - Update Critic using TD error and Actor using Advantage-weighted policy gradient.\n\n"
        "3. KEY TAKEAWAYS:\n"
        "   - Advantage subtraction reduces Monte Carlo variance, speeding up gradient convergence.\n"
        "   - Online boot-strapped updates learn efficiently without waiting for full episode completion.\n"
    )

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)
    print("Saved 'summary.txt' successfully.")

    # --- 5. Plot & Save Visualization ---
    plt.figure(figsize=(8, 4.5))
    plt.plot(episode_rewards, color='#2ca02c', alpha=0.35, label='Raw Episode Reward')
    
    smoothed = pd.Series(episode_rewards).rolling(15, min_periods=1).mean()
    plt.plot(smoothed, color='#2ca02c', linewidth=2.5, label='15-Episode Moving Average')
    
    plt.xlabel('Episodes', fontsize=11)
    plt.ylabel('Total Reward per Episode', fontsize=11)
    plt.title('Advantage Actor-Critic (A2C) Training Convergence', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("visualization.png", dpi=300)
    print("Saved 'visualization.png' successfully.")
    plt.show()

    return df_dataset, df_results

if __name__ == "__main__":
    df_dataset, df_results = run_experiment_11()
    print("\n--- DATASET (10 Actor-Critic Transition Step Samples) ---")
    print(df_dataset.to_string(index=False))