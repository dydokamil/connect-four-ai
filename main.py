from ConnectFourEnvironment import ConnectFourEnvironment

env = ConnectFourEnvironment()
terminated = False

while True:
    s = env.reset()
    env.render()
    while not terminated:
        a = int(input("Choose the column (0-6): "))
        s_prime, reward_terminated = env.step(a)
        env.render()
