from numpy import ndarray


class Operation(object):
    '''
    Base class for an "opeation" in a neural network
    '''

    def __init__(self):
        pass

    def forward(self, input_: ndarray):
        '''
        Stores input in the self.input_ instance variable
        Calls the self._output() function
        '''

        self.input_ = input_
        self.output = self._output()
        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        assert_same_shape(self.output, output_grad)



    def _output(self) -> ndarray:
        raise NotImplementedError()

