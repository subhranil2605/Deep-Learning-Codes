
def deriv(func, input_, delta=0.001):
    return (func(input_ + delta) - func(input_ - delta)) / (2 * delta)

