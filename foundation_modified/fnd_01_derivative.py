

def deriv(func, x, delta = 0.001):
	return (func(x + delta) - func(x-delta))/(2*delta)
