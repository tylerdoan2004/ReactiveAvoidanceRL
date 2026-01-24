---
layout: default
title: Proposal
---

## Summary of the Project

We will employ reinforcement learning to study reactive avoidance in a two-dimensional gridworld. In our problem setup, an agent must travel from some starting location to some goal location while one or more “seeker” agents pursue it. While traversing to the goal location, the agent, operating under limited visibility, must avoid static obstacles. Both the agent and the seekers traverse the gridworld at some constant velocity. Each seeker moves in a manner that maximally reduces its distance to the agent. A single episode terminates when the agent reaches the objective, when a seeker intercepts the agent, when the agent collides with a static obstacle, or when a time limit is reached. The training objective is to learn a policy that balances goal-oriented navigation with real-time evasive maneuvers. The overarching goal of our project is to develop a reinforcement learning framework that facilitates long-term autonomous navigation amidst short-term safety constraints.

Our program accepts as input several environment hyperparameters that determine a particular environment configuration. In particular, these hyperparameters include the gridworld dimensions, the static obstacle locations, the start and goal locations, the number of seekers, the initial positions of the seekers, the velocities of the agent and the seekers, the agent’s visibility radius, and the episode time limit. Given these inputs, the program instantiates a specific environment configuration. In a given environment configuration, the environment will operate in discrete time steps. At each discrete time step, the agent receives an observation of the environment state and a scalar reward. The agent’s observation of the environment state consists of the agent’s position, the relative position of the goal, the relative positions of visible obstacles and seekers, and the agent’s previous observations of the environment state. Given the agent’s observation of the environment state, the agent outputs one of nine possible actions: no movement or movement along one of eight directions (front, right, back, left, front-right, back-right, front-left, and back-left). After the agent performs its action, the environment transitions to its next state. Beyond supplying step-level feedback, the environment also provides episode-level feedback: after each episode, the environment outputs whether the agent reaches the objective, whether a seeker intercepts the agent, whether the agent collides with a static obstacle, or whether the episode timed out.

We believe our project has several real-world applications. The project relates to collision-avoidance and pursuit-evasion scenarios that appear in robotics and multi-agent systems. Specific applications include robotic exploration in cluttered environments, drone navigation in dynamic airspaces, and defensive maneuvers in hostile conditions.

We intend to implement the simulator in Python using Gymnasium and build on MiniGrid for efficient gridworld rendering. We will create a custom MiniGrid environment that supports specific environment configurations and incorporates moving seekers. We will use MiniGrid’s built-in rendering tools to visualize trajectories for debugging and evaluation purposes.

## Project Goals

### Minimum Goal
The minimum goal for this project is to implement a working two-dimensional gridworld environment with one scripted seeker and to train an RL agent operating under partial observability within this environment. To achieve our minimum goal, we must verify environment dynamics and reward specification through controlled experiments and qualitative rollouts.

### Realistic Goal
The realistic goal for this project is to expand upon the minimum goal by comparing policies learned by at least two RL algorithms and increasing environment complexity by adding multiple scripted seekers and static obstacles. To achieve our realistic goal, we must analyze and explain how performance and behavior change as a function of environmental complexity and algorithmic choice.

### Moonshot Goal
The moonshot goal for this project is to extend the realistic goal into a multi-agent setting: either A) we introduce multiple cooperative agents that may coordinate to reach their goal while avoiding one or more scripted seekers, or B) we introduce learning-based seekers that may adapt their pursuit strategy.

## AI/ML Algorithms
We plan to use model-free, on-policy reinforcement learning via Proximal Policy Optimization (PPO) to learn reactive avoidance policies.

## Evaluation Plan
We will quantitatively evaluate our approach by testing our agent across various environment configurations with varying degrees of difficulty. Specifically, we will vary grid dimensions, seeker quantity, obstacle density, and agent visibility radius. Our primary metrics include success rate (percentage of episodes where the agent reaches the objective), capture rate (percentage of episodes where a seeker intercepts the agent), collision rate (percentage of episodes where the agent collides with a static obstacle), time-to-goal (average time steps required for the agent to reach the objective), and cumulative episode return. We will compare learned policies against three baseline policies: a random policy, a shortest-path policy, and a simple heuristic-based policy (preferring the action that progresses the agent closer to the goal unless the action progresses the agent directly within reach of a seeker). We expect the learned policies to significantly outperform the naïve baseline policies in environments with a higher seeker quantity and a higher obstacle density. Specifically, in these environments, we roughly estimate that RL-based agents could improve success rates by up to 20-50% over baseline agents.

We will qualitatively validate our approach by visually inspecting trajectory overlays to ensure the learned policies exhibit interpretable behavior. We intend to employ MiniGrid’s built-in visualization tools to generate episode animations that facilitate observations of how the agent reacts to seekers in real time. We will test our learned policies on several sanity and toy cases: 1) we will confirm that learned policies roughly match the shortest-path policy when no seekers or no obstacles exist in the environment; 2) we will confirm that learned policies navigate around obstacles efficiently when no seekers exist in the environment; 3) we will confirm that learned policies route around stationary seekers; 4) we will confirm that learned policies fail consistently against extremely fast seekers; and 5) we will confirm that learned policies do not exhibit directional bias. Successful policies, in general, should exhibit smooth trajectories rather than oscillating movements, temporary retreats under threats followed by redirection towards the goal, and goal-prioritizing behavior when threats do not exist.

## AI Tool Usage
We used AI tools to explore possible RL algorithms and to perform grammar checking on this proposal document.
