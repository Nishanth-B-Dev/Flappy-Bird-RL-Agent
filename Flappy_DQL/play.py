from agent import Agent
from game import Game
import torch

#  Change This To Switch Between Models
#MODEL_PATH = "model_200.pth"
MODEL_PATH = "model_600.pth"

agent = Agent()

agent.model.load_state_dict(torch.load(MODEL_PATH))
agent.model.eval()

agent.epsilon = 0

game = Game(render=True)

state = game.get_state()

while True:

    action = agent.choose_action(state)

    state, _, done = game.step(action)

    if done:
        print("Restarting...")
        game = Game(render=True)
        state = game.get_state()