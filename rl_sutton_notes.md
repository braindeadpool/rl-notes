# NOTES
## Reinforcement Learning : An Introduction (Sutton)

### The RL Problem
Closed-loop problem of an agent learning what to do to maximize some reward signal in an environment. The agent relies on exploration of new actions vs exploiting known actions (and its consequences) to maximize the reward.

Four elements of the problem
* A policy: defines what the agent should do under a given condition.
* A reward signal: a single number to the agent by the environment at each time step. It is the immediate value of the current environment state (and thus, an agent's action).
* A value function: estimates the long-term reward a state can produce if an agent starts here. It accounts for likely future states and rewards to inform the agent the long-term desirability of a state.
* A model of the environment: something that allows us to infer how the environment will behave under different states and actions.


### k-armed bandit problem


### Finite Markov Decision Processes
* Environment characterized by its `state` $S_t \in S$
* Agent interacting with environment by `action` $A_t \in A(s)$
* At every discrete interaction (time step $t$), agent receives a `reward` $R_{t+1}$ and environment moves to a new state $S_{t+1}$.
* Markov property: $R_t$ and $S_t$ depends only on the preceding state and action.

Target is to maximize expected reward $G_t$ during agent-envinroment interaction. Two types of interaction (or tasks):
* Episodic - there's a terminal state and the environment resets for a new episode. Maximize $G_t = R_t + R_{t+1} + ... + R_T$
* Continuing - interaction goes on continuously indefinitely ($T = \infty$). Maximize discounted reward $G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2}...$ where $0 \leq\gamma \leq 1$ is discount factor.

Policy $\pi$ is a function that maps from states to probabilities of selecting each of the actions.
* $\pi (a|s) = Pr(A_t = a| S_t = s)$.

State-value function for a policy denotes the expected return starting at sa particular tate $s$ and following policy $\pi$.
* $\nu_{\pi}(s) = E_{\pi}[G_t | S_t = s] = E_{\pi}[\sum_{k=0}^{\infty}\gamma^k R_{t+k+1} | S_t = s]$

Action-value function for a policy denotes the expected return of taking a particular action at a particular state following the policy.
* $q_{\pi}(s, a) = E_{\pi}[G_t | S_t =s, A_t = a]$

Bellman equation (represents the recursive relationship)
* $\nu_{\pi}(s) = \sum_{a \in A(s)} \pi(a|s) \sum_{s', r} p(s', r | s, a)[r + \gamma \nu_{\pi}(s')]$

Optimality
* The optimal value functions assign to each state, or state–action pair, the largest expected return achievable by any policy. A policy whose value functions are optimal is an optimal policy. Whereas the optimal value functions for states and state–action pairs are unique for a given MDP, there can be many optimal policies. Any policy that is greedy with respect to the optimal value functions must be an optimal policy. The Bellman optimality equations are special consistency conditions that the optimal value functions must satisfy and that can, in principle, be solved for the optimal value functions, from which an optimal policy can be determined with relative ease.

### Dynamic Programming
How to use value functions to organize and structure the search for good policies.

#### Policy evaluation
How to compute the state-value function for an arbitrary policy $\pi$.

The Bellman equation gives the state-value function for each state under $\pi$. Even if complete environment dynamics are known, it is computationally impossible to solve the system of $|S|$ linear equations with $|S|$ variables. Thus, we resort to approximation methods.

##### Iterative policy evaluation
We can formulate the Bellman equation as an update rule to obtain iterative estimates of $\nu(s)$, the value function estimate that maps state-space $S$ to real numbers (reward space) $\mathcal{R}$. Let $\nu_0$ be the initial estimate. We iteratively update it to obtain a new value function estimate as:
* $\nu_{k+1}(s) = E_{\pi}[R_{t+1} + \gamma \nu_{k}(S_{t+1}) | S_t = s]$
This converges when $\nu_k = \nu_{\pi}$ (following Bellman's equation). But does it reach convergence? Yes, it does as $k \rightarrow \infty$ (proof not provided yet in the book).

##### Policy improvement
Essentially, we can construct a new greedy policy by selecting the best action for the current state under an existing policy (ie, after selecting the best action we follow the old policy) using the action value function.
As per policy improvement theorem, this new greedy policy should be better than or equal to the existing policy. Thus we have incrementally improved the policy - policy improvement! So if we are not able to improve this policy any further this way, then we have reached optimality (we reach the Bellman optimality condition).

##### Policy Iteration
Combining above two, we have a recipe for finding optimal policy.
* Initial policy -> evaluate policy -> improve policy with greedy update -> New policy -> REPEAT
* $\pi_0 \xrightarrow[]E \nu_{\pi_0} \xrightarrow[]I \pi_1 ... \xrightarrow[]I \pi_* \xrightarrow[]E \nu_*$
* Algorithm:
  1. Initialize $V(s) \in \mathcal{R}$, $\pi(s) \in A(s)$ arbitrarily $\forall s \in S$
  2. Policy evaluation step:
    * Iteratively evaluate $\nu(s)$ until it doesn't change more than a threshold (approximate convergence).
  3. Policy improvement step:
    * Iteratively improve the policy by updating the policy to select the best action for each state under the new state-value function (obtained in step 2.)
  4. If policy improved, repeat from step 2.


##### Value Iteration
Policy iteration consists of two nested iterative loops - for each iteration of policy, we evaluate it iteratively and then improve it iteratively. However, this can be simplified into a single step by combining the evaluation and improvement steps into a single update rule.
* $\nu_{k+1}(s) = \max_a E[R_{t+1} + \gamma \nu(S_{t+1}) | S_t = s, A_t = a]
= \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma \nu_k(s')]$
This is essentially the Bellman optimality condition used as an update rule.
* In each policy iteration, you update the value function once for each state - one sweep of update.

##### Asynchronous DP
When the state space is very very large, it is infeasible to perform one sweep update per policy iteration. In such cases, asynchronous updates can be used where each value function is updated in any order of states - some states are updated multiple times before some others are updated. However, it will converge as long as the algorithm continues to update all states.
