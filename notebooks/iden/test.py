from __future__ import print_function
import numpy as np
import numba

m = 1
n = 3
Tf = 100
Delta_T = 0.0001

TimeSpan = np.arange(0, Tf, Delta_T)
nt = TimeSpan.shape[0] # mt do not use in python 

u = np.fromiter( [ 500 * np.sin(0.00004*i) * np.cos(0.000009*i+1.15) for i in range(nt) ], float )

print("init done")

#@numba.jit(nopython=True)
@numba.jit
def init_x(nt, n, u, A_s, B):
    randoms = np.random.random_sample(3*nt).reshape(nt,-3) * np.array([70, 80, 90])
    x = np.zeros((nt,n))

    for j in range(n):
        x[0][j] = 2*(j+1)

    rng = nt - 1
    for i in range(rng):
        cur_x = x[i][:]
        cur_u = u[i]
        cur_rand = randoms[i]

        rand = Delta_T * cur_rand
        fact_A = Delta_T * np.dot(A_s, cur_x)#np.matmul(A_s, cur_x) # normal func
        #fact_A = Delta_T * (A_s * cur_x)[0] # numba.jit func
        fact_B = Delta_T * B * cur_u
        fact = cur_x + fact_A + fact_B + rand

        x[i+1][:] = fact
    
    return x


A_s = 0.001 * np.diag([-5,-7,-6])
B = np.array([0, 0, 1])

x = init_x(nt, n, u, A_s, B)

print("All done")
