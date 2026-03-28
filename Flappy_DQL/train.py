from agent import Agent
from game import Game
import torch
import matplotlib.pyplot as plt

agent = Agent()

episodes = 600  # change to 600 or higher for a more smarter agent

scores = []

for episode in range(episodes):

    game = Game(render=False)
    state = game.get_state()

    done = False
    total_reward = 0

    while not done:

        action = agent.choose_action(state)

        next_state, reward, done = game.step(action)

        agent.store(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        agent.train(64)

    scores.append(total_reward)

    if agent.epsilon > 0.05:
        agent.epsilon *= 0.995

    print("Episode:", episode, "Score:", total_reward)



torch.save(agent.model.state_dict(), f"model_{episodes}.pth")


plt.plot(scores)
plt.xlabel("Episodes")
plt.ylabel("Score")
plt.title(f"Training Performance ({episodes} Episodes)")
plt.show()