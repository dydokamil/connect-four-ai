import unittest
import numpy as np

from ConnectFourEnvironment import ConnectFourEnvironment


class TestGrid(unittest.TestCase):
    def setUp(self):
        self.env = ConnectFourEnvironment()
        self.env.reset()

    def test_state(self):
        s, _, _, _ = self.env.step(6)

        check = np.zeros(42)
        check[-1] = -1

        self.assertTrue(np.array_equal(check, s))

    def test_reward(self):
        _, r, _, _ = self.env.step(6)

        check = self.env.move_penalty
        self.assertEqual(check, r)

    def test_termination(self):
        _, _, done, _ = self.env.step(6)

        self.assertFalse(done)

    def test_prohibited(self):
        check_r = self.env.prohibited_penalty
        check_done = True
        check_info = 'prohibited'

        self.env.step(6)
        self.env.step(6)
        self.env.step(6)
        self.env.step(6)
        self.env.step(6)
        self.env.step(6)

        _, r, done, info = self.env.step(6)

        self.assertEqual(check_r, r)
        self.assertEqual(check_done, done)
        self.assertEqual(check_info, info)
