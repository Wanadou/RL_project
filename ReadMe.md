# highway-fast-v0

## TO DO

The training does work and the agent goes forward (without crashing because no cars on its lane...)

- Run the saving video code (does not seem to work on my Windows laptop...)
- Try running the training for more N_episodes (current value is 300 but 1e5 seen in many examples...)

## Files

The highway folder contains three files: a configuration file (which contains the submitted configuration, but also other versions), a Python script containing the implementation of the DQN (adapted to our use-case, with modifications surrounded by "#######################" and made based on the DQN implementation from the TP4) and a notebook which loads the configured environement, trains an agent and shows results.

## Configuration

This environment (highway) and the agent (vehicule) that is learning inside it are defined by the configuration file. The envionment is mainly defined by the number of vehicules and lanes that it contains (as well as some display parameters).
However, there are 3 important apects of this configuration file that highly influence the agent's ability to learn what we want it to learn: progress as quickly on the road (towards the right) while avoiding collisions, as stated in the [rewards documentation](https://highway-env.farama.org/rewards/).
Indeed, an agent is defined by its:

- Observation (what the agent knows about its environment at each time step t, i.e. the state s): These observations can contain a wide array of features (position, velocity, angle etc. in the case of Kinematics and OccupancyGrid observation types) on the agent itself and on a certain number of nearby vehicules. An action is then chosen based on these observation. In our case (Kinematics observations), this info is a grid of vehicles_count \* n_features where each value corresponds to a feature of a particular vehicule (not including the ego-vehicule ?). This space is stored in a Box object defined by (lower bound, upper bound, (vehicles_count, n_features), numerical_type) (cf [documentation](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box)). Since this observation space is in 2D, we have to **flatten** this matrix before feeding it into the DQN. The same flattening logic applies to other kinds of 2D or 3D observation spaces such as the "OccupancyGrid"
- Actions (what the agent can do based on the observed state): The action space results from the available actions (defined by the booleans "longitudinal" for throttle and "lateral" for steering) and the quantization step (fixed by the parameter "actions_per_axis"). In the end there are actions_per_axis actions if only one of longiutdinal or lateral is True and actions_per_axis \*\* 2 if both are true.
- Rewards (how these actions are evaluated): Finally, we can assign rewards to specific events (high speed, lane change, collision...). These rewards are crucial to finetune if we hope to see our agent learn to drive efficiently in a congested highway. Different sets of rewards will encourage the agent to drive safely throughout the learning (highly negative collision reward) or, on the contrary, to be more "YOLO" (highly positive change lane and high speed rewards).

## Deep Q-learning

The learning paradigm used here is the Deep Q-Learning method, which uses a (shallow) neural network to approximate Q functions, i.e. the state-action value functions used to evaluate a policy. This DQN takes a state s as input and outputs a vector which contains the "value" of each action a in state s, according to the current parameters $\theta$, which are learnt throughout the episodes to get closer and closer to the "real"/"optimal" $Q$-function.

Here are some additional details on the Q-learning algorithm:

In addition to the network with parameters $\theta$, the algorithm keeps another network with the same architecture and parameters $\theta^-$, called **target network**.

The algorithm works as follows:

1.  At each time $t$, the agent is in state $s_t$ and has observed the transitions $(s_i, a_i, r_i, s_i')_{i=1}^{t-1}$, which are stored in a **replay buffer**.

2.  Choose action $a_t = \arg\max_a Q_\theta(s_t, a)$ with probability $1-\varepsilon_t$, and $a_t$=random action with probability $\varepsilon_t$.

3.  Take action $a_t$, observe reward $r_t$ and next state $s_t'$.

4.  Add transition $(s_t, a_t, r_t, s_t')$ to the **replay buffer**.

5.  Sample a minibatch $\mathcal{B}$ containing $B$ transitions from the replay buffer. Using this minibatch, we define the loss:

$$
L(\theta) = \sum_{(s_i, a_i, r_i, s_i') \in \mathcal{B}}
\left[
Q(s_i, a_i, \theta) -  y_i
\right]^2
$$

where the $y_i$ are the **targets** computed with the **target network** $\theta^-$:

$$
y_i = r_i + \gamma \max_{a'} Q(s_i', a', \theta^-).
$$

5. Update the parameters $\theta$ to minimize the loss, e.g., with gradient descent (**keeping $\theta^-$ fixed**):

$$
\theta \gets \theta - \eta \nabla_\theta L(\theta)
$$

where $\eta$ is the optimization learning rate.

6. Every $N$ transitions ($t\mod N$ = 0), update target parameters: $\theta^- \gets \theta$.

7. $t \gets t+1$. Stop if $t = T$, otherwise go to step 2.

# parking-v0

This second environment aims to teach an agent (vehicule) to park at a certain designated spot.

## TO DO

In this case, we can use the algorithm of our choice but we have to implement it (possibly using code from TP sessions):

- Either use the parking_model_based notebook as inpiration
- Or implement the Hindsight Experience Replay from scratch (which might be tedious but worth it since it seems to work very well on this example + we already have the knowledge of optimized hyperparameters for the training => in the example notebook mentioned below)
- Or simply reuse the dqn.py file and apply it for the parking environement (this last idea could be interesting as a first step since we are asked to compare our results to the highway environment with discrete actions !)

## Files

The parking folder contains the submitted configuration file, an implementation of the Hindsight Experience Replay ([HER](https://stable-baselines3.readthedocs.io/en/master/modules/her.html)) learning paradigm with the Soft Actor Critic ([SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html)) off-policy method as shown in this example from the [stable-baselines documentation](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html#hindsight-experience-replay-her) and based on this [example notebook](https://github.com/Farama-Foundation/HighwayEnv/blob/master/scripts/parking_her.ipynb), and another implementation which is model-based using Pytorch and which is exactly a copy of [this notebook](https://github.com/Farama-Foundation/HighwayEnv/blob/master/scripts/parking_model_based.ipynb) and hence requires additional investigation...

# racetrack-v0

## TO DO

- In the notebook, we need to understand how to pass a configured env (with our own configuration file) into the make_vec_env function (for now, the default environment is used in the notebook)
- Debugging : Loading a trained model does not seem to work ("Unpickling error"...)
- Once the previous steps are done, run the whole notebook and see what happens ! We don't need to implement anything from scratch here, so fiddling with the configuration file and training parameters (+ explaining how the PPO algorithm works) will be enough !

## Files

The racetrack folder contains the submitted configuration file, as well a copy of [this Python script](https://github.com/Farama-Foundation/HighwayEnv/blob/master/scripts/sb3_racetracks_ppo.py) which uses the Proximal Policy OPtimization ([PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)) algorithm from stable-baselines to train the agent.
