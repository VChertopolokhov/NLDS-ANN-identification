import numpy as np

from scipy import linalg
from scipy import integrate

from scipy.interpolate import interp1d


class LearningLaw:
    """
    Class with associated base learning law process.
    ...

    Attributes
    ----------
    exposure : float
        Exposure in seconds.

    Methods
    -------
    colorspace(c='rgb')
        Represent the photo in the given colorspace.
    """
    def __init__(self,
                nn_obj, time: np.array, init_cond: np.array,
                K1: np.matrix, K2: np.matrix):

            self.nn = nn_obj
            self.time = time
            self.init_cond = init_cond 
            self.matrix_K1 = K1 
            self.matrix_K2 = K2
    
    @property
    def step(self):
        """ Calculate discrete step of time.
        Returns
        -------
        float
            step value
        """
        return self.time[1] - self.time[0]

    @staticmethod
    def func(t, y, z):
        return y+z[0][0]*z[1][0]

    def integrate(self): 
        # Define parametres of dt and output value  
        dt = self.step
        states = np.empty_like(self.time)

        W1, W2 = self.nn.get_weights

        #r = integrate.ode(self.func).set_integrator("dop853")
        r = integrate.ode(self.nn.ident).set_integrator("dop853")

        for i in range(len(self.time)):
            #print([W1[i], W2[i]])

            r.set_initial_value(*self.init_cond).set_f_params([W1[0], W2[1]])
            r.set_f_params([W1[0], W2[1]])

            r.integrate(r.t+dt)
            states[i] = r.y

        return states

class DiffNN:
    """
    Class with associated base differential neural network.
    ...

    Attributes
    ----------
    exposure : float
        Exposure in seconds.

    Methods
    -------
    colorspace(c='rgb')
        Represent the photo in the given colorspace.
    """
    def __init__(self,
                    A=None, B=None, R=None, Q=None, Q_0=None, # matrix 
                    eta_u=None, eta_sigma=None, eta_phi=None, # constants
                    u=None, phi=None, sigma=None): # functions

        # Initialization A matrix
        if A is None:
            self.matrix_A = -10*np.diag([1, 1])
        else:
            self.matrix_A = A

        # Initialization B matrix
        if B is None:
            self.matrix_B = np.ones(4).reshape(-2, 2)
        else:
            self.matrix_B = B

        # Initialization R matrix
        if R is None:
            self.matrix_R = np.array([4, 1, 1, 4]).reshape(-2, 2)
        else:
            self.matrix_R = R

        # Initialization Q matrix
        if Q is None:
            self.matrix_Q = np.array([3, 1, 1, 3]).reshape(-2, 2)
        else:
            self.matrix_Q = Q

        # Initialization Q_0 matrix
        if Q_0 is None:
            self.matrix_Q_0 = np.diag([1, 1])
        else:
            self.matrix_Q_0 = Q_0

        # Initialization eta_u value
        if eta_u is None:
            self.eta_u = 1.
        else:
            self.eta_u = eta_u

        # Initialization eta_sigma value
        if eta_sigma is None:
            self.eta_sigma = 2.
        else:
            self.eta_sigma = eta_sigma

        # Initialization eta_phi value
        if eta_phi is None:
            self.eta_phi = 2.
        else:
            self.eta_phi = eta_phi

        # Initialization sigma activation function
        if sigma is None:
            self.sigma = lambda x: 1 / (1 + np.exp(-1*x))
        else:
            self.sigma = sigma

        # Initialization phi activation function
        if phi is None:
            self.phi = lambda x: 1 / (1 + np.exp(-1*x))
        else:
            self.phi = phi 

        # Initialization u signal function
        if u is None:
            self.u = lambda t: np.sin(t)
        else:
            self.u = u

        # Find P as solution of matrix Riccati equation
        self.matrix_P = linalg.solve_continuous_are(
                                                    self.matrix_A, 
                                                    self.matrix_B, 
                                                    self.matrix_Q,
                                                    self.matrix_R)
        # Find sqrt P for dead zone function 
        self.sqrt_matrix_P = np.sqrt(
                                    self.matrix_P.astype(np.complex))

        # Find nu via property method
        self.nu = self.calc_nu

        ##############################################
        ######## ALL go to learning base class
        ##############################################

        # Difine  weights matrices
        #self.matrix_W1 = np.ones(4).reshape(-2, 2)
        #self.matrix_W2 = np.ones(4).reshape(-2, 2)

        # Define numerical states of matrices W1 and W2
        self.states_W1 = np.ones(2000).reshape(-2, 2)
        self.states_W2 = np.ones(2000).reshape(-2, 2)

        # Difine K1 and K2  matrices
        self.matrix_K1 = np.ones(4).reshape(-2, 2)
        self.matrix_K2 = np.ones(4).reshape(-2, 2)

        # Set time from 0 to 100 
        self.time = np.linspace(0,100,1000) 

        # Set delta 
        #self.delta = np.random.random(1000)

        # Kostyl to find value in descret time step 
        self.delta_value = interp1d(self.time, self.delta,fill_value='extrapolate')
        self.W1_value = interp1d(self.time, self.delta,fill_value='extrapolate')
        self.W2_value = interp1d(self.time, self.delta,fill_value='extrapolate')

    # Method for print(DiffNN instance)
    def __str__(self):
        return \
            'Parameters of DiffNN instance\
            \nA:\
            \n{0}\
            \nR:\
            \n{1}\
            \nQ:\
            \n{2}\
            \nQ_0:\
            \n{3}\
            \nP:\
            \n{4}\
            \neta_u: {5} \
            \neta_sigma: {6} \
            \neta_phi: {7} \
            \nnu: {8} \
            '.format(
                    self.matrix_A, 
                    self.matrix_B,
                    self.matrix_Q,
                    self.matrix_Q_0,
                    self.matrix_P,
                    self.eta_u,
                    self.eta_sigma,
                    self.eta_phi,
                    self.nu)
        

    # TODO: check work of np.min and linalg.eigvals
    @property
    def calc_nu(self):
        """ Calculate nu-constant by formula:
            nu = (eta_sigma + eta_phi*eta_u) / lambda_min(P^(-1/2)*Q_0*P^(-1/2)), 
            where lambda_min = min(eigvals of matrix) 
        Returns
        -------
        float
            nu-constant value
        """
        P_inv = np.linalg.inv(self.matrix_P) # find P^(-1)
        P_inv_sqr = np.sqrt(P_inv.astype(np.complex)) # P^(-1/2)

        tmp_matrix = np.matmul(np.matmul(P_inv_sqr, self.matrix_Q), P_inv_sqr) # P^(-1/2)*Q*P^(-1/2)
        lambda_min = np.min(linalg.eigvals(tmp_matrix)).real # min eigvals of matrix P
        
        nu = (self.eta_sigma + self.eta_phi*self.eta_u) / lambda_min
        return nu

    @property
    def get_weights(self):
        """ Return current weights of matrices W1 and W2.
        Returns
        -------
        tuple
            (W1, W2)
        """
        return (self.states_W1, self.states_W2)

    # TODO: rewrite this method in more np-style
    @staticmethod
    def sign_plus(z: float):
        """ Calculate z value by formula:
            z if z >= 0, else 0
        Parameters
        -------
            z: float
        Returns
        -------
        float
            non-negative z value
        """
        if z >= 0:
            return z
        else:
            return 0
    
    
    def s(self, delta: float):
        """ Calculate s value by formula:
            s = sign_plus( 1 - nu/norm(P^(1/2)*delta) ), 
            where sign_plus(z) define as z if z >= 0, else 0
        Parameters
        -------
            delta: float
        Returns
        -------
        float
            s value
        """
        norm_value = linalg.norm(self.sqrt_matrix_P * delta) # find matrix norm
        z_value = 1 - self.nu/norm_value # find z by formula 

        z_value = self.sign_plus(-z_value) # apply sign_plus function 

        return z_value

    def W1(self, delta, x, t):
        v1 = np.matmul(self.matrix_P, self.matrix_K1)
        print(v1, delta, x)
        v2 = np.matmul(delta, v1)
        v3 = np.matmul(self.sigma(x).reshape(-1, 1), v2.reshape(1,-1))

        return v3

    def W2(self, delta, x, t):
        v1 = np.matmul(self.matrix_P, self.matrix_K1)
        v2 = np.matmul(delta, v1)
        v3 = np.matmul(self.sigma(x).reshape(-1, 1), v2.reshape(1,-1))
        
        return v3


    def ident(self, z, t):
        #W1 = self.W1_value(t)
        #W2 = self.W2_value(t)
        delta = self.delta_value(t) #self.delta[index]
        t_vec = np.array([t, t])
        x  = z[0:2]
        W1 = z[2:6].reshape(-2, 2)
        W2 = z[6:10].reshape(-2, 2)

        z0 = np.matmul(self.matrix_A, x) + \
             np.matmul(W1, self.sigma(x)) + \
             np.matmul(np.matmul(W2, self.phi(x)), self.u(t_vec)) # [z_01, z_02]

        z1 = -1 * self.s(delta) * self.W1(delta, x, t_vec) # [[z_11, z_12], [z_13, z_14]]

        z2 = -1 * self.s(delta) * self.W1(delta, x, t_vec) #self.sigma(x)) # [[z_21, z_22], [z_23, z_24]]

        return  np.concatenate((z0, z1, z2), axis=None)

        #[z_01, z_02, z_11, z_12, z_13, z_14, z_21, z_22, z_23, z_24].T

        ############################
        #z_01 z_02
        #           z_11 z_12
        #           z_13 z_14
        #                       z_21 z_22
        #                       z_23 z_24


    def fit(self, time: np.array, K1: np.matrix, K2: np.matrix, 
            init_cond: np.array, states_system: np.array, epoch=5):

        # Get a time (np.array)
        time = self.time 
        #Set matrices K1 and K2 like attributes
        self.matrix_K1 = K2
        self.matrix_K2 = K2

        for i in range(epoch):
            print('Epoch #{0} start'.format(i))
            # Find states of matrices W1 and W2 
            states_ident, info_W1 = integrate.odeint(self.ident, init_cond, time, full_output=True)
            #states_W2, info_W2 = integrate.odeint(self.W_2, init_cond, time, full_output=True)

            #print(info_W1, info_W2)

            #Set states of matrices W1 and W2 like attributes
            #self.states_W1 = states_W1
            #self.states_W2 = states_W2 

            #states_ident = ll.integrate()

            self.delta =  np.abs(states_ident - states_system.T[0])
            print('Epoch #{0} error: {1}'.format(i, np.mean(self.delta)))
            break 
        #return states_ident, self.delta

        #return (states_W1, states_W2)











    