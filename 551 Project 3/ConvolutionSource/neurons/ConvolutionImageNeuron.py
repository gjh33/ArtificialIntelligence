import random

from neurons.InnerNeuron import InnerNeuron
from statistic_functions.Sigmoid import Sigmoid

# This class acts as a container for the pixel neurons of an image. This is
# technically a layer not a neuron, however it will act like a neuron since
# all the neurons in this layer share a weight set.
class ConvolutionImageNeuron:
    # Filter size is a tuple (x, y)
    # Shift is how many spaces the filter is moved each iteration
    # zpad is the zero padding around the image
    def __init__(self, statistic_function, filter_size, shift_size=1, zpad=(0, 0)):
        self.inputs = []
        self.statistic_function = statistic_function
        self.delta = 0
        self.pixel_neurons = []
        self.input_size = (0, 0)
        self.filter_size = filter_size
        self.output_size = (0, 0)
        self.shift_size = shift_size
        self.zero_padding = zpad
        self.filter_parameters = []

    # Takes a ConvolutionImageNeuron or PoolingImageNeuron
    def add_input(self, image_neuron):
        # If we haven't added an input yet then initalize everything based on
        # this input
        if self.input_size == (0,0):
            self.input_size = image_neuron.output_size
            self.calculate_output_size()
            self.build_pixel_neurons()
        if self.input_size != image_neuron.output_size:
            print("Invalid input. Dimensions of all inputs must be the same!")
        else:
            self.inputs.append(image_neuron)
            self.map_pixel_neurons(image_neuron)

    def get_pixel_neuron_at(self, x, y):
        return self.pixel_neurons[x][y]

    def build_pixel_neurons(self):
        for i in range(0, output_size[0]):
            column = []
            for j in range(0, output_size[1]):
                column.append(InnerNeuron(Sigmoid()))
            self.pixel_neurons.append(column)

    def calculate_output_size(self):
        x = (self.input_size[0] - self.filter_size[0] + 1 + (2 * self.zero_padding[0]))
        x = int(x/shift_size)
        y = (self.input_size[1] - self.filter_size[1] + 1 + (2 * self.zero_padding[1]))
        y = int(y/shift_size)
        self.output_size = (x, y)

    def map_pixel_neurons(self, image_neuron):
        # Using computed values, map each pixel neuron to a the neurons from
        # the input that the filter should affect
        # i.e. pixel_neuron[0] with a nxn filter would map to the first nxn
        # square of pixels from the input
        # make sure to map additively (append, don't equal) becuase there may
        # be multiple input images
