import gym
import collections
import numpy as np
import random

env = gym.make('MountainCar-v0')

q = collections.defaultdict(lambda: np.zeros(3)) # zero is optimistic for MountainCar
gamma = 1.0 # discounting factor (1.0 -> no discounting)
alpha = 0.3 # learning rate
epsilon = 0.0 # exploration factor (0.0 -> no exploration)

def normalize(high, low, value):
    return int(45*(value-low)/(high-low))

def state_key(state):
    return "{},{}".format( \
        normalize(env.observation_space.high[0], env.observation_space.low[0], state[0]), \
        normalize(env.observation_space.high[1], env.observation_space.low[1], state[1]) \
    )

env.monitor.start('/tmp/MountainCar-v0/experiment-6')

sum = 0

# print(env.observation_space.high)
# print(env.observation_space.low)

for i_episode in range(15000):
    success = False
    state = env.reset()
    action = np.argmax(q[state_key(state)])

    for t in range(200):
        state_next, reward, done, _ = env.step(action)
        # if i_episode % 1000 == 99:
        #     env.render()
        # env.render()
        if epsilon > random.random():
            action_next = env.action_space.sample()
        else:
            action_next = np.argmax(q[state_key(state_next)])

        q[state_key(state)][action] += \
            alpha*( \
                reward + \
                gamma*q[state_key(state_next)][action_next] - \
                q[state_key(state)][action] \
            )

        state = state_next
        action = action_next

        if done:
            success = True
            sum += t+1
            # print("Episode {} completed in {} steps!".format(i_episode, t))
            break

    if not(success):
        sum += 200

    if i_episode % 100 == 99:
        print("{}: {}{}     -- {}".format(i_episode+1, '#'*int((sum/100)/10), '-'*(20 - int((sum/100)/10)), len(q)))
        sum = 0

    # print("Episode {} finished".format(i_episode))

env.monitor.close()
