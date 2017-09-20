from ConnectFourEnvironment import ConnectFourEnvironment

env = ConnectFourEnvironment()
terminated = False

# while True:
terminated = False
s = env.reset()
env.render()
while not terminated:
    a = int(input("Choose the column (0-6): "))
    s_prime, reward, terminated = env.step(a)
    env.render()
