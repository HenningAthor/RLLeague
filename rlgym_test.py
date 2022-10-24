import rlgym
import numpy as np

team_size = 1
env = rlgym.make(game_speed=10, spawn_opponents=True, team_size=team_size)

while True:
    obs = env.reset()
    done = False

    while not done:
        # Here we sample a random action. If you have an agent, you would get an action from it here.
        actions = []
        for i in range(team_size * 2):
            action_i = env.action_space.sample()
            actions.append(action_i)
        action1 = env.action_space.sample()
        action2 = env.action_space.sample()
        actions = [action1, action2]
        new_obs, reward, done, game_info = env.step(actions)
        print("Reward: {} | Reward Shape: {} | Observation Shape: {}".format(reward, np.shape(reward), np.shape(new_obs)))

        obs = new_obs
