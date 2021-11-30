
class NoiseGenerator:
    def __init__(self, amplitude=0.1):
        self.amplitude = amplitude


class NoisyBiasedFunction:
    def __init__(self):
        self.noise = None
        self.bias = None
        self.fn = None