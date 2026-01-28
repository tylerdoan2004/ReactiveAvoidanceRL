from environment import Environment



env = Environment()

obs, info = env.reset()
print("Initial conditions:", obs)

for i in range(10):
    action = env.action_space.sample()  # RANDOM action
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i}: Agent at {obs['agent']}, Reward: {reward}")
