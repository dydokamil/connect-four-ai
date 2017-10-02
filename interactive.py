from Agent import Agent
from Brain import Brain
from ConnectFourEnvironment import ConnectFourEnvironment

if __name__ == '__main__':
    env = ConnectFourEnvironment(play_with_rng=False)
    model = './yellow.h5'

    brain = Brain(model)
    agent = Agent(brain=brain)
    while True:
        s = env.reset()
        env.render()
        while not env.is_finished():
            if env.yellows_turn():
                a = agent.act(s)
            else:
                a = input("Choose action (0-6): ")
                a = int(a)

            s_prime, r, done = env.step(a)
            env.render()
            s = s_prime
