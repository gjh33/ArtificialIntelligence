import math

class Sigmoid:
    def apply(self, val):
        return (1/(1 + math.exp(val)))

    def derivative(self, val):
        return (val * (1 - val))
