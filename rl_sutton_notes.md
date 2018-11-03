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

#### Policy improvement
Essentially, we can construct a new greedy policy by selecting the best action for the current state under an existing policy (ie, after selecting the best action we follow the old policy) using the action value function.
As per policy improvement theorem, this new greedy policy should be better than or equal to the existing policy. Thus we have incrementally improved the policy - policy improvement! So if we are not able to improve this policy any further this way, then we have reached optimality (we reach the Bellman optimality condition).

#### Policy Iteration
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


#### Value Iteration
Policy iteration consists of two nested iterative loops - for each iteration of policy, we evaluate it iteratively and then improve it iteratively. However, this can be simplified into a single step by combining the evaluation and improvement steps into a single update rule.
* $\nu_{k+1}(s) = \max_a E[R_{t+1} + \gamma \nu(S_{t+1}) | S_t = s, A_t = a]
= \max_a \sum_{s', r} p(s', r | s, a) [r + \gamma \nu_k(s')]$
This is essentially the Bellman optimality condition used as an update rule.
* In each policy iteration, you update the value function once for each state - one sweep of update.

#### Asynchronous DP
When the state space is very very large, it is infeasible to perform one sweep update per policy iteration. In such cases, asynchronous updates can be used where each value function is updated in any order of states - some states are updated multiple times before some others are updated. However, it will converge as long as the algorithm continues to update all states.

#### Generalized Policy iteration
General idea of policy evaluation and policy improvement interacting with each other to stabilize to the optimal policy and value function.

### Monte Carlo Methods
Monte Carlo methods allow us to learn from experience without prior knowledge of the environment. Essentially, MC methods allow us to sample from the desired probability distribution without obtaining the pdf in explict form (which is usually infeasible in many cases).

NOTE: For now, only episodic tasks are considered for MC learning.

#### Monte Carlo Prediction
Here, we learn the state-value function for a given policy by simulating multiple episodes of that policy and averaging the returns for each state - by law of large numbers, this should converge to the expected values.

#### Monte Carlo estimation of state values
* Goal: To estimate $\nu_{\pi}(s) \forall s \in S$ under a policy $\pi$.
* Simulate multiple episodes following $\pi$ and then average the returns for each state over episodes. NOTE: A state $s$ may occur more than once in an episode.
* First visit MC method: Estimates the average of the returns after the first occurence of $s$ in each episode.
  * Initialize $V(s) \forall s \in S$ randomly
  * $Returns(s) \leftarrow$ empty list $\forall s \in S$ : stores the returns for each state from all of the episodes.
  * For each generated episode following $\pi$: $S_0, A_0, R_1, S_1, A_1, R_1, ..., S_{T-1}, A_{T-1}, R_T$
    * $G \leftarrow 0$
    * For each $t = T-1, T-2, ..., 0$:
      * $G \leftarrow \gamma G + R_{t+1}$
      * If $S_t$ doesn't appear in $\{S_0, S_1, ..., S_{t-1}\}$, ie $S_t$ is the first occurence of that state in the current episode:
        * Append $G$ to $Returns(S_t)$
        * $V(S_t) \leftarrow Average(Returns(S_t))$
* Every visit MC method: Estimates the average of the returns following each occurence of $s$.
  * Same algorithm as first visit MC except there is no check for $S_t$'s previous occurence in the current episode.
* Both methds converge to $\nu_{\pi}(s)$ as number of (first) visits goes to $\infty$.

#### Monte Carlo estimation of action values
When you have a model, given a particular state, you know what is the next state for each action. And all you need is the state value function estimate - you look up the state value one step ahead at all next possible states  and choose the action that leads to the maximizal next state value and reward combination to determine the (best) policy.

However, when a model is not available, it becomes important to rather have the action value function so that the best action  can be chosen at each state. MC methods are heavily useful in the scenario of no model.
* The MC methods for estimating action value are very similar to ones for estimating state values.
* Instead of visiting and estimating value function for a state, we now do it instead for a state-action pair.
* But the state-action pair space is usually a much larger one than just the state space. And if we use a deterministic policy, many of the state-action pairs will never be visited and thus, we would have no estimates for them to improve the policy (ie by chosing the best action amongst all possible ones).
* This is the problem of maintaining exploration.

##### Exploring starts
* One solution to maintaining exploration in MC estimation is to start from a randomly chosen state-action pair in each episode with each possible state-action pair having non-zero likelihood of being chosen as the start.
* Con: In actual environment cases, this is not very reliable to estimates values just from the starting conditions (which would be the case for many state-action pairs).

##### Stochastic policies
* The other solution is to consider only those policies that are stochastic with respect to choosing an action for each state. Therefore at each state, there is a non-zero probability of choosing any of the actions.

#### Monte Carlo without exploring starts
##### On-policy vs off policy Methods
[Not exactly clear] In an on-policy method, you evaluate and iterate over a policy that is being used to make next step decisions (choose actions) and thus provide the next step reward data. In an off-policy method, the data is generated by a different policy than the one being evaluated / iterated upon by the method.

##### $\epsilon$-greedy policy
* $\pi(a|s) > 0 \forall s \in S, a \in A$ - the probability of choosing any action is non-zero at a state but eventually this is shifted closer and closer to a deterministic policy.
* Most of the time the action with the maximally estimated action value is chosen, but with probability $\epsilon$ a different action is chosen at random.

#### Off-policy prediction via Importance sampling
When exploring the action space to learn a policy, there's a dilemma: how do you learn about the optimal policy (ie. action values conditional on subsequent optimal behavior) but at the same time behave according to exploratory policy (non-optimal policy to search the action space and find the optimal ones).
Off-policy methods instead use two policies:
* `Target policy`: the policy being learnt about (that later becomes the optimal one).
* `Behaviour policy`: the policy used to generate behaviour and thereby data to learn from.
