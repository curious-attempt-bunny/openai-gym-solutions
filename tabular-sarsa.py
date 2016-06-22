import gym
import collections
import numpy as np

env = gym.make('MountainCar-v0')

q = collections.defaultdict(lambda: np.zeros(3)) # zero is optimistic for MountainCar
gamma = 1.0 # discounting factor (1.0 -> no discounting)
alpha = 0.1 # learning rate

def state_key(state):
    return "{},{}".format(round(state[0],1), round(state[1],1))

seen_success = False

env.monitor.start('/tmp/MountainCar-v0/experiment-2')

for i_episode in range(5000):
    state = env.reset()
    action = np.argmax(q[state_key(state)])

    for t in range(1000):
        state_next, reward, done, _ = env.step(action)
        if i_episode == 99 or i_episode == 0:
            env.render()
        # env.render()
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
            seen_success = True
            print("Episode {} completed!".format(i_episode))
            break

    print("Episode {} finished".format(i_episode))

env.monitor.close()