# Experimental

def custom_reward(self, source_reward):
    if self.info['x'] > self.pos_x:
        self.pos_x = self.info['x']

    delta_pos = self.info['x'] - self.pos_x
    self.reward = 0
    self.reward = source_reward
    delta_rings = self.info['rings'] - self.rings
    if delta_rings < 0:
        self.reward += 20 * delta_rings
    self.rings = self.info['rings']

    if self.info['lives'] < self.max_lives:
        self.max_lives = self.info['lives']
        self.reward = -450

    self.reward += self.researcher()