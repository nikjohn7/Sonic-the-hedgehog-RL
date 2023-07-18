import random
import numpy

class Sum_Tree:

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros(capacity, dtype=object)

        # Write index to overwrite old values - makes the tree a circular buffer.
        self.write = 0 

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            # Propagate the change recursively so the path from
            # the updated node to the root reflects the new priority.
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        # Determine the parent's left and right nodes
        left_child = 2 * idx + 1
        right_child = left_child + 1

        # Leaves do not have children, thus the indices of their
        # children will be invalid.
        if left_child >= len(self.tree):
            return idx

        if s <= self.tree[left_child]:
            return self._retrieve(left_child, s)
        else:
            # Remove the lower poriton of the priority associated 
            # with the left child.
            return self._retrieve(right_child, s-self.tree[right_child])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        # Leaf nodes start at index self.capacity - 1
        idx = self.write + self.capacity - 1

        # Write the new data and set the new priority.
        self.data[self.write] = data
        self.update(idx, p)
        
        # Circular buffer - replace oldest memories.
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        # The interior nodes must be updated to reflect the change
        # in prority, as these nodes values are the priorities
        # from various subsections of the tree.
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])



class PER_History:   

    def __init__(self, capacity):

        self.tree = Sum_Tree(capacity)
        self.e = 0.01 # Small constant to ensure all priorities > 0
        self.a = 0.6  # Constant to control the weight of error on priority

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, experience, error):
        p = self._getPriority(error)
        self.tree.add(p, experience) 

    def sample(self, n):
        mini_batch = []
        indicies = []

        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, _, experience) = self.tree.get(s)
            mini_batch.append(experience)
            indicies.append(idx)
        return mini_batch, indicies

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)