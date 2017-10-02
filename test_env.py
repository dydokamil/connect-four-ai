from ConnectFourEnvironment import ConnectFourEnvironment

env = ConnectFourEnvironment()

env.reset()
env.render()
while not env.is_finished():
    action = int(input("Choose your action (0-6): "))
    env.step(action)
    env.render()
