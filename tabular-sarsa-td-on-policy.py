import gym
import collections
import numpy as np
import random

env = gym.make('MountainCar-v0')

q = collections.defaultdict(lambda: np.zeros(3)) # zero is optimistic for MountainCar
gamma = 0.995 # discounting factor (1.0 -> no discounting)
alpha = 0.1 # learning rate
epsilon = 0.1 # exploration factor (0.0 -> no exploration)

def state_key(state):
    return "{},{}".format(round(state[0],1), round(state[1],1))

# env.monitor.start('/tmp/MountainCar-v0/experiment-5')

sum = 0

for i_episode in range(10000):
    success = False
    state = env.reset()
    action = np.argmax(q[state_key(state)])

    for t in range(1000):
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
        sum += 1000

    if i_episode % 10 == 9:
        print("{}: {}".format(i_episode+1, '#'*int((sum/10)/10)))
        sum = 0

    # print("Episode {} finished".format(i_episode))

# env.monitor.close()
