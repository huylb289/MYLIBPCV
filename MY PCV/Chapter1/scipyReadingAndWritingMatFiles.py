##SciPy comes with some useful modules for input and output. Two of them are io
##and misc .
##
##Reading and writing .mat files

##http://docs.scipy.org/doc/scipy/reference/io.html
from scipy import io

data = {}
data['x'] = "this is test"

io.savemat('test.mat',data)

data = io.loadmat('test.mat')



