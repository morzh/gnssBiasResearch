from pylops import LinearOperator

class Diagonal(LinearOperator):

    def __init__(self, d, dtype=None):
        self.d = d.flatten()
        self.shape = (len(self.d), len(self.d))
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        return self.d * x

    def _rmatvec(self, x):
        return self.d * x
