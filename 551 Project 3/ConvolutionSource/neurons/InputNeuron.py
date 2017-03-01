# Simple neuron to relay input
class InputNeuron:
    def __init__(self):
        self.input_value = 0
        self.last_prediction = 0.5

    def set_input_value(self, value):
        self.input_value = value

    def predict(self):
        self.last_prediction = self.input_value
        return self.input_value
