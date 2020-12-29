from Library.import_library import *
from Updates.fout_gout_dgout_sign_sign import FoutGoutDgout_sign_sign
from Updates.fw_fc import FW_FC_gaussian, FW_FC_binary
from Functions.generic_functions import load_object, save_object
from Functions.annex_functions import print_matrix


class Teacher(object):
    def __init__(self, N=1000, alpha=1, K=1,
                 weights_distrib='gaussian',
                 channel='sign-sign',
                 Delta=1e-2, sigma_w=1,
                 parity=False, verbose=False, seed=False):

        # Parameters
        self.N = N
        self.alpha = alpha
        self.M = int(self.alpha * self.N)
        self.K = K
        self.channel = channel
        self.parity = parity
        self.Delta = Delta

        # Seed
        self.seed = seed
        if self.seed:
            np.random.seed(0)

        # Options
        self.verbose = verbose

        # Init
        self.weights_distrib = weights_distrib
        self.sigma_w = sigma_w

        self.weights = self.init_weights(
            weights_distrib=self.weights_distrib,
            sigma_w=self.sigma_w
        )
        self.dataset = self.generate_dataset()

    def sample_X(self, M_samples, mean_x=0, var_x=1):
        """
        Generates iid Gaussian matrix of size N x M_samples
        with mean=mean_x, var=var_x
        """
        X = np.random.normal(mean_x, var_x, (self.N, M_samples))
        if self.verbose:
            print('X=', X, '\n')
        return X

    def init_weights(self, weights_distrib='gaussian', sigma_w=1):
        """
        Generates teacher weights W^* with prior depending on self.weights
        """
        # gaussian
        if weights_distrib == 'gaussian':
            W1 = np.zeros((self.K, self.N))
            lambda_w = np.zeros((self.K))
            Sigma_w = sigma_w * np.identity(self.K)
            for i in range(self.N):
                W1[:, i] = np.random.multivariate_normal(
                    lambda_w, Sigma_w)
                # binary
        elif weights_distrib == 'binary':
            W1 = 2 * np.random.randint(2, size=(self.K, self.N)) - 1
            # otherwise
        else:
            raise NameError('Weights distribution undefined')

            # weights in the second layer fixed to 1
        W2 = np.ones((1, self.K))

        if self.verbose:
            print('W1=', W1, '\n')

        return {"W1": W1, "W2": W2}

    def phi_2(self, z1, parity=False):
        if parity:
            resul = 1*(z1 > 0) + 1*(z1 < 0) + 0*(z1 == 0)
        else:
            resul = 1*(z1 > 0) - 1*(z1 < 0) + 0*(z1 == 0)
        return resul

    def phi_1(self, z2):
        resul = 1*(z2 > 0) - 1*(z2 < 0) + 0*(z2 == 0)
        return resul

    def phi_out(self, z, channel, Delta):
        """
        Returns phi(z) + N(0, Delta)
        """
        if channel == 'sign-sign':
            resul = self.phi_2(self.weights["W2"].dot(
                self.phi_1(z)), parity=self.parity).reshape((z.shape[1]))
        else:
            raise NameError('Channel undefined')

        if Delta > 0:
            size = resul.shape
            noise = np.random.normal(0, sqrt(Delta), size).astype("int64")
            resul += noise
        return resul

    def forward(self, W, X):
        """
                Generate output y = phi(1/sqrt(N) * X w)
        """
        N, M = X.shape
        Z = W["W1"].dot(X) / sqrt(self.N)
        Y = self.phi_out(Z, channel=self.channel,
                         Delta=self.Delta)

        if self.verbose:
            print('Y=', Y, '\n')
            print(f'X={X.shape} W1={W["W1"].shape} Z={Z.shape} Y={Y.shape}')
            print(f'N={N} M={M} K={self.K}')
        return Y

    def generate_dataset(self, alpha_test=10):
        """
        - Generates data matrix X
        - Generates teacher weights W
        - Generates Y
        """
        # training set
        X_train = self.sample_X(self.M)
        y_train = self.forward(self.weights, X_train)

        # test set
        X_test = self.sample_X(int(alpha_test * self.N))
        y_test = self.forward(self.weights, X_test)

        # dataset
        dataset = {
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test
        }
        return dataset


class ApproximateMessagePassing(object):
    def __init__(self, N=1000, alpha=1, K=1,
                 weights_distrib='gaussian', channel='sign-sign', Delta=0,
                 seed=True, verbose=True, verbose_init=False, parity=False):
        # Parameters
        self.K = K
        self.N = N
        self.alpha = alpha
        self.M = int(self.N*self.alpha)
        self.channel = channel

        # Seed
        self.seed = seed
        if self.seed:
            np.random.seed(0)

        # Weights
        self.weights_distrib = weights_distrib
        self.T_W = np.zeros((K))  # mean for gaussian
        self.Sigma_W = np.identity(K)  # covariance for gaussian

        # Options
        self.verbose = verbose
        self.verbose_init = verbose_init
        self.verbose_update = False
        self.Nishimori_identity = True
        self.parity = parity

       # Options print

        # Convergence of the algorithm
        self.convergence_params = {"precision_overlap": 1e-3, "precision_weights": 1e-2,
                                   "steps_min": 5, "steps_max": 100, "overlap_max": 0.995}

        # Damping
        self.damping = False
        self.damping_coef = 0.5

        # Initialization
        self.initialization_mode = "random"

        # Callbacks
        self.callbacks = {"overlaps": {"q": [], "m": []}}

        self.args = dict(vars(self))

    # Initialization
    def initialization(self, teacher):
        # Data set
        print("Create dataset")
        self.X = teacher.dataset["X_train"]
        self.Y = teacher.dataset["y_train"]
        self.teacher_weights = teacher.weights
        self.W1 = self.teacher_weights["W1"]
        self.W2 = self.teacher_weights["W2"]

        # Messages
        print("Initialization")
        # Initialization What, Chat
        self.W_hat, self.C_hat = self.initialization_What_Chat()
        # Initialization V, omega
        self.initialization_V_omega()
        # Initialization gout, dgout
        self.initialization_gout_dgout()
        # Initialization Sigma, T
        self.initialization_Sigma_T()

    def initialization_What_Chat(self):
        W_hat = np.zeros((self.K, self.N))
        C_hat = np.zeros((self.K, self.K, self.N))
        for i in range(self.N):
            if self.initialization_mode == "ground_truth":
                noise = 0.01
                W_hat[:, i] = self.W2[:, i] + noise * np.random.randn(self.K)
                C_hat[:, :, i] = W_hat[:, i].dot(W_hat[:, i].transpose())
            elif self.initialization_mode == "random":
                W_hat[:, i] = self.initialization_What()
                C_hat[:, :, i] = self.initialization_Chat()
            else:
                raise NotImplementedError("initialization_mode undefined")

        if self.verbose_init:
            print('### What, Chat Initialized ###')
            print_matrix(W_hat, 'What =')
            print_matrix(C_hat, 'Chat =')

        return W_hat, C_hat

    def initialization_What(self):
        W_hat = np.zeros((self.K))
        if self.weights_distrib == 'gaussian':
            W_hat = self.T_W
        elif self.weights_distrib == 'binary':
            W_hat = np.zeros((self.K))
        else:
            raise NotImplementedError("weights_distrib undefined")
        return W_hat

    def initialization_Chat(self):
        C_hat = np.zeros((self.K, self.K))
        if self.weights_distrib == 'gaussian':
            C_hat = self.Sigma_W
        if self.weights_distrib == 'binary':
            C_hat = np.identity(self.K)
        return C_hat

    def initialization_V_omega(self, initialization_V="spd", initialization_omega="ones"):
        V = np.zeros((self.K, self.K, self.M))
        V_inv = np.zeros((self.K, self.K, self.M))

        for l in range(self.M):
            # SPD
            if initialization_V == "spd":
                V[:, :, l] = make_spd_matrix(self.K)
            else:
                raise NotImplementedError("wrong initialization_omega")

            # Inverse
            V_inv[:, :, l] = inv(V[:, :, l])

        # Ones
        if initialization_omega == "ones":
            omega = np.ones((self.K, self.M))
        # Random
        elif initialization_omega == "random":
            omega = np.random.randn(self.K, self.M)
        else:
            raise NotImplementedError("Wrong initialization_omega")

        self.V = V
        self.V_inv = V_inv
        self.omega = omega

        if self.verbose_init:
            print('### V, V_inv, omega Initialized ###')
            print_matrix(self.V, 'V =')
            print_matrix(self.V_inv, 'V_inv =')
            print_matrix(self.omega, 'omega =')

    def initialization_gout_dgout(self):
        (self.gout, self.dgout) = self.update_gout_dgout()
        if self.verbose_init:
            print('### gout, dgout Computed ###')
            print_matrix(self.gout, 'gout =')
            print_matrix(self.dgout, 'dgout =')
            print('\n')

    def initialization_Sigma_T(self):
        self.Sigma_inv = self.update_Sigma_inv()
        self.Sigma = self.update_Sigma()
        self.T = self.update_T()
        if self.verbose_init:
            print('### S, T Computed ###')
            print_matrix(self.Sigma, 'Sigma =')
            print_matrix(self.Sigma_inv, 'Sigma_inv =')
            print_matrix(self.T, 'T =')
        return (self.Sigma, self.Sigma_inv, self.T)

    def initialization_q(self):
        self.q = np.zeros((self.K, self.K))
        (self.callbacks["overlaps"]["q"]).append(self.q)
        return self.q

    # Updates
    def damping(self, X_new, X_self):
        return (1-self.damping_coef) * (X_self) + (self.damping_coef) * X_new

    def update_Sigma_inv(self):
        Sigma_inv = - np.einsum('ijl,kl->ijk', self.dgout, np.square(self.X))
        if self.damping:
            Sigma_inv = self.damping(
                Sigma_inv, copy.copy(self.Sigma_inv))

        if self.verbose_update:
            print_matrix(Sigma_inv, 'Sigma_inv =')

        return Sigma_inv

    def update_Sigma(self):
        Sigma = np.moveaxis(inv(np.moveaxis(self.Sigma_inv, -1, 0)), 0, -1)

        if self.verbose_update:
            print_matrix(Sigma, 'Sigma =')

        return Sigma

    def update_T(self):
        T = np.einsum('ijl,jl->il', self.Sigma,
                      np.einsum('ij,lj->il', self.gout, self.X)) \
            + np.einsum('ijk,jk->ik', self.Sigma, np.einsum('ijk,jk->ik', -
                                                            np.einsum('ijk,lk->ijl', self.dgout, np.square(self.X)), self.W_hat))
        if self.damping:
            T = self.damping(T, copy.copy(self.T))

        if self.verbose_update:
            print_matrix(T, 'T =')

        return T

    # Compute/update gout, dgout
    def update_gout_dgout(self):
        gout = np.zeros((self.K, self.M))
        dgout = np.zeros((self.K, self.K, self.M))
        for l in range(self.M):
            gout[:, l], dgout[:, :, l] = self.compute_gout_dgout(l)

        if self.damping:
            gout = self.damping(gout, copy.copy(self.gout))
            dgout = self.damping(dgout, copy.copy(self.dgout))

        if self.verbose_update:
            print_matrix(gout, 'gout =')
            print_matrix(dgout, 'dgout =')

        return gout, dgout

    def compute_gout_dgout(self, l):
        # Sign-Sign
        if self.channel == 'sign-sign':
            V = self.V[:, :, l]
            V_inv = self.V_inv[:, :, l]
            omega = self.omega[:, l]
            y = self.Y[l]

            # for K = 1, K=2
            if (self.K == 1 or self.K == 2):
                fout_gout_dgout_sign_sign = FoutGoutDgout_sign_sign(
                    K=self.K, y=y, omega=omega, V=V, V_inv=V_inv, parity=self.parity)
                gout, dgout = fout_gout_dgout_sign_sign.gout_dgout()
                # Nishimori identity
                if self.Nishimori_identity:
                    gout_ = gout.reshape(self.K, 1)
                    dgout = - gout_.dot(gout_.T)
            # for larger K
            else:
                raise NotImplemented("")

        else:
            raise NotImplementedError("Undefined channel")

        return gout, dgout

    # What (K,N) , Chat (K,K,N)
    def update_What_Chat(self):  # Update messages
        W_hat = np.zeros((self.K, self.N))
        C_hat = np.zeros((self.K, self.K, self.N))
        for i in range(self.N):
            W_hat[:, i], C_hat[:, :, i] = self.compute_fW_fC(i)

        if self.damping:
            W_hat = self.damping(W_hat, copy.copy(self.W_hat))
            C_hat = self.damping(W_hat, copy.copy(self.C_hat))

        if self.verbose_update:
            print_matrix(W_hat, 'What =')
            print_matrix(C_hat, 'Chat =')

        return W_hat, C_hat

    def compute_fW_fC(self, i):
        Sigma = self.Sigma[:, :, i]
        Sigma_inv = self.Sigma_inv[:, :, i]
        T = self.T[:, i]
        fW = np.zeros((self.K))
        fC = np.zeros((self.K, self.K))

        # Gaussian case
        if self.weights_distrib == 'gaussian':
            fw_fc_gaussian = FW_FC_gaussian(
                T=T, Sigma_inv=Sigma_inv, T_W=self.T_W, Sigma_W=self.Sigma_W)
            fW, fC = fw_fc_gaussian.fw_fc()

        # Binary for K =1
        elif self.weights_distrib == 'binary':
            fw_fc_binary = FW_FC_binary(
                T=T, Sigma_inv=Sigma_inv, K=self.K)
            fW, fC = fw_fc_binary.fw_fc()

        else:
            print('weights_distrib not defined')

        return fW, fC

    # V, Vinv (K,K,P)
    def update_V(self):
        V = np.einsum('ijl,lk->ijk', self.C_hat, np.square(self.X))

        if self.damping:
            V = self.damping(V, copy.copy(self.V))

        if self.verbose_update:
            print_matrix(V, 'V =')

        return V

    def update_V_inv(self):
        V_inv = np.moveaxis(np.linalg.inv(np.moveaxis(self.V, -1, 0)), 0, -1)
        if self.verbose_update:
            print_matrix(V_inv, 'V_inv =')
        return V_inv

    # Omega (K,P)
    def update_omega(self):
        omega = np.einsum('ij,jl->il', self.W_hat, self.X) \
            - np.einsum('ijl,jl->il', np.einsum('ijk,jl->ikl',
                                                self.C_hat, self.gout), np.square(self.X))

        if self.damping:
            omega = self.damping(omega, copy.copy(self.omega))

        if self.verbose_update:
            print_matrix(omega, 'omega =')

        return omega

    # iteration
    def AMP_step(self):
        # Update of What, Chat
        self.W_hat, self.C_hat = self.update_What_Chat()

        # Update of V and V_inv
        self.V = self.update_V()
        self.V_inv = self.update_V_inv()

        # Update of omega
        self.omega = self.update_omega()

        # Update gout, dgout
        self.gout, self.dgout = self.update_gout_dgout()

        # Update Sigma, Sigma_inv
        self.Sigma_inv = self.update_Sigma_inv()
        self.Sigma = self.update_Sigma()

        # Update T
        self.T = self.update_T()

    def AMP_iteration(self):
        print(self.args)
        difference, step, stop = 10, 0, False

        q = self.initialization_q()
        step += 1

        while step < self.convergence_params["steps_max"] and stop == False and difference > self.convergence_params["precision_overlap"]:
            if self.verbose:
                print(f'Step={step}')
            step += 1

            q = copy.copy(self.callbacks["overlaps"]["q"][-1])
            W_hat = copy.copy(self.W_hat)

            if np.amax(self.q) > 0.99:
                try:
                    self.AMP_step()
                except:
                    print("Break")
                    break
            else:
                self.AMP_step()

            # Compute difference for convergence
            self.compute_overlap()
            difference_q = norm(self.q-q)
            difference_What = norm(self.W_hat-W_hat)
            if self.verbose:
                print('difference q =', difference_q)
                print('difference What =', difference_What)
            if step > self.convergence_params["steps_min"]:
                difference = difference_q
            self.difference = difference

            if self.weights_distrib == 'binary':
                if np.amax(self.q) > self.convergence_params["overlap_max"]:
                    stop = True

            if self.weights_distrib == 'gaussian':
                if np.amax(self.q) > self.convergence_params["overlap_max"] and step > 25:
                    stop = True

    def compute_overlap(self):
        What = self.W_hat
        W1 = self.W1

        q = 1/self.N * What.dot(What.transpose())
        m = 1/self.N * What.dot(W1.transpose())

        (self.callbacks["overlaps"]["q"]).append(q)
        (self.callbacks["overlaps"]["m"]).append(m)
        self.q = q
        self.m = m

        if self.verbose:
            print_matrix(self.q, 'q_AMP=')
            print_matrix(self.m, 'm_AMP=')
