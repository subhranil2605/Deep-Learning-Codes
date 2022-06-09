from numpy import ndarray


class Operation(object):
    '''
    Base class for an 'operation' in a neural network
    '''

    def __init__(self):
        pass

    def forward(self, input_: ndarray):
        '''
        Stores input in the self._input instance variable
        Calls the self._output() function
        '''
        self.input_ = input_

        self.output = self._output()

        return self.output

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Calls the self._input_grad() function.
        Checks that the appropiate shapes match
        '''
        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        return self.input_grad

    def _output(self) -> ndarray:
        '''
        The _output method must be defined for each operation.
        '''
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        The _input_grad method must be defined for each operation.
        '''
        raise NotImplementedError()
