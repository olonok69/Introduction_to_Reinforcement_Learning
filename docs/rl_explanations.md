# RL Algorithm Explanations (Code-Referenced)

This document explains how the demo implementations work:

- Tabular Q-learning on FrozenLake
- REINFORCE (Policy Gradient) on CartPole
- PPO on CartPole (Stable-Baselines3)

All explanations point directly to the code.

Note: line numbers are approximate and can shift as scripts evolve; function names and code blocks are the primary reference.

---

## 1) Q-learning

### 1.1 What Q-learning is

Q-learning is a **model-free, value-based** reinforcement learning algorithm. Instead of learning a policy directly, it learns a function $Q(s,a)$ that estimates how good each action is in each state.

Core idea:

- Learn action values from interaction data.
- Derive behavior by choosing the action with maximum Q-value.

### 1.2 How Q-learning works (conceptually)

At each transition $(s,a,r,s')$, the algorithm updates the estimate for $Q(s,a)$ using a Bellman target:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
$$

High-level loop:

1. Select action (often with `epsilon`-greedy exploration).
2. Observe next state and reward.
3. Update current Q-value toward TD target.
4. Repeat for many episodes while reducing exploration over time.

### 1.3 How Q-learning works in this project (code mapping)

Main file: `demos/q_learning_frozenlake.py`

### 1.3.1 Problem setup

- Environment creation and dimensions are in the beginning of `train(...)`.
- The agent uses a **Q-table** initialized to zeros in the same setup block.

Interpretation:

- `q_table[s, a]` stores the current estimate of the expected discounted return for taking action `a` in state `s` and then following the greedy policy.

### 1.3.2 Action selection (`epsilon`-greedy)

Function: `epsilon_greedy_action(...)`

- With probability `epsilon`, choose a random action (exploration).
- Otherwise, choose `argmax_a Q(s,a)` (exploitation).

Why this matters:

- Early training needs exploration because Q-values are mostly unknown.
- Later training should exploit learned Q-values, which is achieved by epsilon decay.

### 1.3.3 Episode interaction loop

Core loop: episode/step loops inside `train(...)`

For each step:

1. Select action via `epsilon_greedy_action(...)`.
2. Step environment to get `(next_state, reward, terminated, truncated)`.
3. Mark terminal state (`done`) as `terminated or truncated`.

### 1.3.4 Bellman update (the learning rule)

Update block: TD target/error update lines in `train(...)`

The implementation corresponds to:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\right]
$$

Code mapping:

- `best_next = np.max(q_table[next_state])` is $\max_{a'}Q(s',a')$.
- `td_target = reward + gamma * best_next * (1-done)` is the target.
- `td_error = td_target - q_table[state, action]`.
- `q_table[state, action] += alpha * td_error`.

Terminal handling:

- Multiplying by `(1.0 - float(done))` removes bootstrapping on terminal transitions.

### 1.3.5 Exploration decay

Epsilon update line in `train(...)`:

$$
\epsilon \leftarrow \max(\epsilon_{min}, \epsilon \cdot \epsilon_{decay})
$$

This gradually shifts behavior from exploration to exploitation.

### 1.3.6 Monitoring and evaluation

- Training logs every `log_every` episodes using `avg100`.
- Final policy evaluation is in `evaluate_policy(...)`:
  - Uses pure greedy policy (`argmax`).
  - Reports success rate over many episodes.

---

## 2) REINFORCE

### 2.1 What REINFORCE is

REINFORCE is a **policy gradient** algorithm. It optimizes a parameterized policy $\pi_\theta(a|s)$ directly, rather than learning a Q-table.

Core idea:

- Sample trajectories from the current policy.
- Increase probability of actions that produced high return.
- Decrease probability of actions that produced low return.

### 2.2 How REINFORCE works (conceptually)

For an episode, it computes return $G_t$ at each step and updates policy parameters via gradient ascent on expected return, typically implemented as minimizing:

$$
\mathcal{L}(\theta) = -\sum_t \log \pi_\theta(a_t|s_t) G_t
$$

High-level loop:

1. Roll out one episode with the current stochastic policy.
2. Compute discounted returns.
3. Build policy-gradient loss from log-probabilities and returns.
4. Backpropagate and update policy parameters.

### 2.3 How REINFORCE works in this project (code mapping)

Main file: `demos/reinforce_cartpole.py`

### 2.3.1 Policy network

Class: `PolicyNetwork`

- Input: observation vector (`obs_size`).
- Output: action probabilities via `softmax`.

Interpretation:

- The model represents a stochastic policy $\pi_\theta(a|s)$.

### 2.3.2 Trajectory return computation

Function: `discounted_returns(...)`

For each time step `t`, the return is:

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}
$$

Implementation details:

- It computes returns backward.
- It normalizes returns to reduce gradient variance and stabilize training.

### 2.3.3 Data collection (on-policy)

Episode loop: interaction loop inside `train(...)`

At each step:

1. Convert observation to tensor.
2. Compute action probabilities from current policy.
3. Sample action from `Categorical(probs)`.
4. Store `log_prob(action)`.
5. Execute action and store reward.

Why sampling matters:

- REINFORCE requires sampling from the policy being optimized (on-policy Monte Carlo).

### 2.3.4 Policy loss and update

Loss block: policy-loss accumulation in `train(...)`

This code implements:

$$
\mathcal{L}(\theta) = -\sum_t \log \pi_\theta(a_t|s_t) G_t
$$

Interpretation:

- If $G_t$ is high, gradient ascent increases probability of sampled action.
- If $G_t$ is low, update tends to decrease that action probability.

Optimization step:

- `optimizer.zero_grad()`
- `loss.backward()`
- `optimizer.step()`

### 2.3.5 Convergence signal

- Episode reward and moving average (`avg100`) are tracked in `train(...)`.
- Solve condition uses `avg100 >= solve_score` after at least 100 episodes.

---

## 3) Q-learning vs REINFORCE (teaching summary)

### What is learned

- Q-learning: learns **state-action values** `Q(s,a)` and derives policy via argmax.
- REINFORCE: learns the **policy directly** `\pi_\theta(a|s)`.

### Update style

- Q-learning: temporal-difference bootstrapping (uses `max Q(s',a')`).
- REINFORCE: Monte Carlo policy gradient (uses full-episode returns).

### Typical fit

- Q-learning (tabular): small/discrete state spaces.
- REINFORCE (neural policy): continuous/high-dimensional observations.

---

## 4) Hyperparameters in these scripts

### Q-learning (`parse_args(...)`)

- `alpha`: learning rate.
- `gamma`: discount factor.
- `epsilon-start`, `epsilon-min`, `epsilon-decay`: exploration schedule.
- `episodes`, `max-steps`: training budget.

### REINFORCE (`parse_args(...)`)

- `lr`: optimizer learning rate.
- `gamma`: discount factor.
- `episodes`: maximum number of episodes.
- `solve-score`: threshold for early stop.

---

## 5) Practical demo tips

- If Q-learning appears flat early, increase episodes and/or reduce `epsilon-decay` speed.
- For classroom speed, start with deterministic FrozenLake (`--slippery` off).
- For REINFORCE stability, keep return normalization enabled and use enough episodes.

---

## 6) Post-training verification (latest workflow)

All three demos now include post-training verification modes:

### 6.1 Real-time render (`human`)

- Q-learning: `--render-eval --render-episodes N`
- REINFORCE: `--render-eval --render-episodes N`
- PPO: `--render-eval --render-episodes N`

This opens an interactive window so you can visually verify if the learned policy solves the task.

### 6.2 Video recording (`mp4` for slides)

- Q-learning: `--record-video --video-episodes N --video-dir <path>`
- REINFORCE: `--record-video --video-episodes N --video-dir <path>`
- PPO: `--record-video --video-episodes N --video-dir <path>`

This records evaluation episodes using Gymnasium `RecordVideo` into:

- `videos/q_learning_frozenlake` (default)
- `videos/reinforce_cartpole` (default)
- `videos/ppo_cartpole` (default)

### 6.3 Combined mode (single command)

- Q-learning: `--record-and-render`
- REINFORCE: `--record-and-render`
- PPO: `--record-and-render`

This enables both visualization modes in one run.

### 6.4 Dependency note

- Video recording requires `moviepy` (already added to `requirements.txt`).
- If missing, scripts print a friendly install hint: `uv pip install moviepy`.

---

## 7) PPO

### 7.1 What PPO is

PPO (Proximal Policy Optimization) is a modern **policy optimization** algorithm designed to improve stability and sample efficiency relative to vanilla policy gradient approaches.

Core idea:

- Optimize a surrogate objective that discourages overly large policy updates.
- Reuse trajectory data for several optimization epochs.
- Balance performance with training stability.

In this project, PPO is provided by `stable-baselines3`.

### 7.2 How PPO works (conceptually)

PPO collects rollouts, computes advantage estimates, and updates policy/value networks using a clipped objective so new policy probabilities do not move too far from old ones in one step.

High-level loop:

1. Collect rollout data with current policy.
2. Estimate advantages/returns.
3. Run several minibatch gradient updates with clipping.
4. Repeat until timesteps budget is consumed.

### 7.3 How PPO works in this project (code mapping)

Main file: `demos/gymnasium_ppo_cartpole.py`

### 7.3.1 Training setup

- `train_and_evaluate` creates `CartPole-v1` wrapped with `Monitor` and instantiates `PPO` (function start + constructor block).
- The script uses `MlpPolicy` and key hyperparameters from CLI args: `learning_rate`, `n_steps`, `batch_size`, `gamma`, and `seed`.

Practical interpretation:

- PPO is a policy optimization algorithm that performs stable updates by constraining policy changes (implemented internally by Stable-Baselines3).

### 7.3.2 Learn, save, and evaluate

- Training is triggered by `model.learn(total_timesteps=args.timesteps)`.
- Model persistence is handled with `model.save(args.model_path)` after ensuring parent directory exists.
- Deterministic evaluation uses `evaluate_policy(..., deterministic=True)` and reports mean/std reward.

This gives a clean “did it learn?” metric directly after training.

### 7.3.3 Visual verification and recording

- Human render path: `render_policy(...)` with `render_mode="human"`, deterministic actions via `model.predict(..., deterministic=True)`.
- Video recording path: `record_policy_video(...)` with `render_mode="rgb_array"` and Gymnasium `RecordVideo` wrapper.
- Combined convenience path: `--record-and-render` enables both render and recording in one run.

### 7.3.4 CLI parameters to highlight in class

- `--timesteps`: total training budget.
- `--learning-rate`, `--n-steps`, `--batch-size`, `--gamma`: core PPO controls.
- `--eval-episodes`: reliability of evaluation estimate.
- `--render-eval`, `--record-video`, `--record-and-render`: post-training policy verification modes.

