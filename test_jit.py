from numba import jit
import time
#@jit
def fib(n):
    a,b = 1,0
    for i in range(n):
        a,b = b,a+b
t0 = time.time()
n = 1000000
fib(n)
t1 = time.time()
print('catime:'+str(t1-t0))