import autograd.numpy as np   # Thinly-wrapped version of Numpy
from autograd import grad
from autograd import elementwise_grad

## For vectors

def hard_function(x):
    ans = np.ones(len(x))
    for i in range(len(x)):
        ans = ans * np.roll(x, i)
    return ans
# If the result is a vector, then we need the function `elementwise_grad`. This
# returns the gradient vector function of `hard_function`
jac_hard = jacobian(hard_function)

vec = np.array(range(10))+1
print("Jacobian of hard_function is", jac_hard(vec))


## For matrices
def mat(x):
    ans = np.ones(x.shape)
    ans = x * np.roll(x,1,0)
    return ans
jac_mat = elementwise_grad(mat)

def mat2(phi):
    ans = -2 * 1.0 * phi* (np.roll(phi, 1, 0) + np.roll(phi, 1, 1))+ (1 - 2 * 2.0) * phi**2 + 2.0 * phi**4
    return ans
jac_mat2 = elementwise_grad(mat2)

phi = np.random.random((2,2))
jac_mat2(x)
