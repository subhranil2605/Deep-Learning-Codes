from numpy import ndarray


def assert_same_shape(array: ndarray, array_grad: ndarray):
    assert array.shape == array_grad.shape, \
        f'''
        Two ndarrays should have the same shape;
        instead, first array's shape is {tuple(array.shape)}
        and second array's shape is {tuple(array_grad.shape)}
    '''
    return None


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


class ParamOperation(Operation):
    '''
    An Operation with parameters.
    '''

    def __int__(self, param: ndarray) -> ndarray:
        """
        The ParamOperation method
        """
        super().__init__()
        self.param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        """
        Calls self._input_grad and self._param_grad.
        Checks appropriate shapes.
        """

        assert_same_shape(self.output, output_grad)
        self.input_grad = self._input_grad(output_grad)
        self.param_grad = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad)
        assert_same_shape(self.param, self.param_grad)

        return self.input_grad

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        """
        Every subclass of ParamOperation must implement _param_grad.
        """
        raise NotImplementedError()
