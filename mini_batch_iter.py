import numpy as np


class MiniBatchIterator:

    def __init__(self, idx_start, bat_size, num_sample, train_phase=True, is_permute=True):

        self._bat_size = bat_size
        self._idx_start = idx_start
        self._num_sample = num_sample
        self._train_phase = train_phase
        self._is_permute = is_permute

        if self._is_permute:
            self._idx_sample = np.random.permutation(self._num_sample)
        else:
            self._idx_sample = np.array(range(self._num_sample))

    @property
    def idx_start(self):
        return self._idx_start

    @property
    def bat_size(self):
        return self._bat_size

    @property
    def num_sample(self):
        return self._num_sample

    @property
    def train_phase(self):
        return self._train_phase

    @property
    def is_permute(self):
        return self._is_permute

    def get_batch(self):
        """ Get indices of a mini-batch """

        if self._idx_start + self._bat_size > self._num_sample:
            if self._train_phase:
                idx_out = self._idx_sample[self._idx_start:]

                if self._is_permute:
                    self._idx_sample = np.random.permutation(self._num_sample)

                count = self._bat_size - (self._num_sample - self._idx_start)
                idx_out = np.concatenate((idx_out, self._idx_sample[: count]))

                self._idx_start = count
            else:
                idx_out = self._idx_sample[self._idx_start:]
                self._idx_start = 0
        else:
            idx_out = self._idx_sample[
                self._idx_start: self._idx_start + self._bat_size]
            self._idx_start = (self._idx_start +
                               self._bat_size) % self._num_sample

        return idx_out

    def reset_iterator(self, idx_start=0):
        if idx_start < 0:
            raise ValueError('Sample index should be non-negative!')

        self._idx_start = idx_start

# unit test
if __name__ == '__main__':

    myIter = MiniBatchIterator(
        idx_start=0, bat_size=256, num_sample=5994, train_phase=True, is_permute=True)

    for i in xrange(25):
        idx = myIter.get_batch()
        print idx
