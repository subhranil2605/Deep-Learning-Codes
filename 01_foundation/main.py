import numpy as np
from numpy import ndarray
from typing import Callable, List

# A function takes in a ndarray as argument and produces an ndarray
Array_function = Callable[[ndarray], ndarray]

# A chain is a list of functions
Chain = List[Array_function]


# Our module class
class DeepLearningMath:
    def __init__(self):
        pass

    # derivative function
    def deriv(self, func: Array_function,
              input_: np.ndarray,
              delta: float = 0.001) -> np.ndarray:
        '''
        Evaluates the derivative of a function "func" at every element
        '''
        return (func(input_ + delta) - func(input_ - delta)) / (delta * 2)

    # chain of two functions
    def chain_length_2(self, chain: Chain,
                       x: ndarray) -> ndarray:
        '''
        Evaluates two functions in a row, in a "Chain"
        '''

        assert len(chain) == 2, "Length of input 'chain' should be 2"

        f1 = chain[0]
        f2 = chain[1]

        return f2(f1(x))

    # chain of three functions
    def chain_length_3(self, chain: Chain,
                       x: ndarray) -> ndarray:
        '''
        Evaluates three functions in a row, in a "Chain"
        '''

        assert len(chain) == 3, "Length of input 'chain' should be 3"

        f1 = chain[0]
        f2 = chain[1]
        f3 = chain[2]

        return f3(f2(f1(x)))

    # Chain Derivative
    def chain_deriv_2(self, chain: Chain,
                      input_range: ndarray) -> ndarray:
        '''
        Uses the chain rule to compute the derivative of the composite functions:
        (f2(f1(x)))' = f2'(f1(x)) * f1'(x)
        '''

        assert len(chain) == 2, "This function requires two functions"

        assert input_range.ndim == 1, "1D array of elements"

        f1 = chain[0]
        f2 = chain[1]

        # f1(x)
        f1_of_x = f1(input_range)

        # f1'(x)
        df1dx = self.deriv(f1, input_range)

        # f2'(f1(x))
        df2du = self.deriv(f2, f1_of_x)

        return df2du * df1dx

    # derivative of three chained functions
    def chain_deriv_3(self, chain: Chain,
                      input_range: ndarray) -> ndarray:
        '''
        Uses the chain rule to compute the derivative of the composite functions:
        (f3(f2(f1(x))))' = f3'(f2(f1(x))) * f2'(f1(x)) * f1'(x)
        '''

        assert len(chain) == 3, "This function requires two functions"

        assert input_range.ndim == 1, "1D array of elements"

        f1 = chain[0]
        f2 = chain[1]
        f3 = chain[2]

        # f1(x)
        f1_x = f1(input_range)

        # f2(f1(x))
        f2_f1_x = f2(f1_x)

        # df3du
        df3du = self.deriv(f3, f2_f1_x)

        # df2du
        df2du = self.deriv(f2, f1_x)

        # df1dx
        df1dx = self.deriv(f1, input_range)

        return df1dx * df2du * df3du

    # multiple variable functions
    def multiple_inputs_add(self, x: np.ndarray,
                            y: np.ndarray,
                            sigma: Array_function) -> ndarray:
        '''
        Function with multiple inputs and addition, forward pass.
        '''

        assert x.shape == y.shape

        a = x + y
        return sigma(a)

    # derivative of the multiple input function
    def multiple_input_derivative(self, x: np.ndarray,
                                  y: np.ndarray,
                                  sigma: Array_function):
        '''
        Computes the derivative of this simple function with respect to both inputs
        '''

        # compute the "forward pass"
        a = x + y

        # compute the derivatives
        dsda = self.deriv(sigma, a)

        dadx = dady = 1

        return dsda * dadx, dsda * dady

    # matrix multiplication
    def matmul_forward(self,
                       X: ndarray,
                       W: ndarray) -> ndarray:
        '''
        Computes the forward pass of a matrix multiplication
        '''

        assert X.shape[1] == W.shape[0], \
            '''
            For matrix multiplication, the number of columns in the first array should
            match the number of rows in the second;
            '''

        # matrix multiplication
        N = np.dot(X, W)

        return N

    # matrix multiplication with function
    def matrix_forward_extra(self,
                             X: ndarray,
                             W: ndarray,
                             sigma: Array_function) -> ndarray:
        '''
        Computes the forward pass of a function involving matrix multiplication
        with extra function
        '''

        # matrix multiplication
        N = np.dot(X, W)

        # feeding the output of the matrix multiplication through sigma
        S = sigma(N)

        return S

    # matrix multiplication derivative
    def matmul_backward_first(self,
                              X: ndarray,
                              W: ndarray) -> ndarray:
        '''
        Computes the backward pass of a matrix multiplication with respect
        to the first element
        '''

        # backward pass
        dNdX = np.transpose(W, (1, 0))

        return dNdX

    # matrix multiplication function derivative
    def matrix_function_backward(self,
                                 X: ndarray,
                                 W: ndarray,
                                 sigma: Array_function) -> ndarray:
        '''
        Computes the derivative backward pass of a function involving matrix multiplication
        with extra function
        '''

        # matrix multiplication
        N = np.dot(X, W)

        # feeding the output of the matrix multiplication through sigma
        S = sigma(N)

        # backward calculation
        dSdN = self.deriv(sigma, N)

        # dNdX
        dNdX = np.transpose(W, (1, 0))

        # multiply them together: since dNdX is 1x1 here, order doesn't matter
        return np.dot(dSdN, dNdX)

    # matrix function forward sum of 2D array
    def matrix_funtion_forward_sum_2D(self,
                                      X: ndarray,
                                      W: ndarray,
                                      sigma: Array_function) -> float:
        '''
        Computing the result of the forward pass of this function with
        input ndarrays X and W and function sigma.
        '''

        assert X.shape[1] == W.shape[0]

        # matrix multiplication
        N = np.dot(X, W)

        # feeding the output of the matrix multiplication through sigma
        S = sigma(N)

        # sum of all the elements
        L = np.sum(S)

        return L

    # 2D matrix derivative
    def matrix_function_backward_sum_2D(self,
                                        X: ndarray,
                                        W: ndarray,
                                        sigma: Array_function) -> ndarray:
        '''
        Compute derivative of matrix function with a sum with respect to the
        first matrix input
        '''

        assert X.shape[1] == W.shape[0]

        # matrix multiplication
        N = np.dot(X, W)

        # feeding the output the function
        S = sigma(N)

        # sum of all elements
        L = np.sum(S)

        dLdS = np.ones_like(S)

        # dSdN
        dSdN = self.deriv(sigma, N)

        # dLdN
        dLdN = dLdS * dSdN

        # dNdX
        dNdX = np.transpose(W, (1, 0))

        # dLdX
        dLdX = np.dot(dSdN, dNdX)

        return dLdX
