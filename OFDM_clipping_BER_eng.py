#############
#
# Complex-field TISTA
#
#############

# coding: utf-8

import torch
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
import numpy as np
import time

device = torch.device('cuda') # 'cpu' or 'cuda'


######
# Some definitions of basic operations for  complex-valued tensors
#
# In this code, complex tensor is defined as a pair of real tensors corresponding to real and imaginary parts, i.e.,
# for tensor Z with complex elements is represented by Z = (A, B) where A(=Re Z) and B(=Im Z) are (real) torch.tensor.
#
######

# tensor addition (X + Y)
def c_add(X, Y):
    return (X[0] + Y[0], X[1] + Y[1])

# tensor subtraction (X - Y)
def c_sub(X, Y):
    return (X[0] - Y[0], X[1] - Y[1])

# Hermitian transpose
def c_ht(X):
    return (X[0].t(), -X[1].t())

# tensor multiplication
def c_mm(X, Y):
    Z_re = torch.mm(X[0], Y[0]) - torch.mm(X[1], Y[1])
    Z_im = torch.mm(X[0], Y[1]) + torch.mm(X[1], Y[0])
    return (Z_re, Z_im) 

# scalar multiplication (a; real scholar)
def c_scalar_mul(a, X):
    return (a * X[0], a * X[1])

# matrix inverse
def c_inverse(X):
    X_re = X[0]
    X_im = X[1]
    X_re_inv = torch.inverse(X_re)
    tmp = torch.mm(X_im, X_re_inv)
    tmp = torch.mm(tmp, X_im)
    Z_re = torch.inverse(X_re + tmp)
    tmp = - torch.mm(X_re_inv, X_im)
    Z_im = torch.mm(tmp, Z_re)
    return (Z_re, Z_im)

# random matrix with normal distribution (circular (complex) Gaussian distribution) with standard deviation "stdv"

def c_normal(m,n,stdv):
    Z_re = torch.normal(torch.zeros(m,n), std = stdv/math.sqrt(2.0)).to(device)
    Z_im = torch.normal(torch.zeros(m,n), std = stdv/math.sqrt(2.0)).to(device)
    return (Z_re, Z_im)


# pseudo inverse
def c_pseudo_inverse(X):
    m = X[0].size()[0]
    n = X[0].size()[1]
    X_ht = c_ht(X)
    tmp = c_inverse(c_mm(X_ht, X))
    tmp2 = c_inverse(c_mm(X, X_ht))
    if n < m:
        return c_mm(tmp, X_ht)
    else:
        return c_mm(X_ht, tmp2)
    
# Squared error 
def c_squared_error(X, Y):
    Z_re = X[0] - Y[0]
    Z_im = X[1] - Y[1]
    return ((Z_re**2).sum() + (Z_im**2).sum()).item()

# normalization
def c_normalize(X):
    return (X[0]/torch.sqrt(X[0]**2 + X[1]**2), X[1]/torch.sqrt(X[0]**2 + X[1]**2))

# zero tensor
def c_zeros(m,n):
    return (torch.zeros(m,n).to(device), torch.zeros(m,n).to(device))


#squared norm
def c_norm(X):
    return X[0]**2.0+ X[1]**2.0

# hadamard product
def c_hadamard_prod(X, Y):
    return (X[0]*Y[0]-X[1]*Y[1],X[1]*Y[0]+X[0]*Y[1])

# trace
def c_trace(X):
    return (torch.trace(X[0]),torch.trace(X[1]))

# transpose
def c_t(X):
    return (X[0].t(),X[1].t())

# conjugate
def c_conj(X):
    return (X[0],-X[1])

###########


# DFT matrix
def dft(n):
    B_re = torch.zeros(n, n)
    B_im = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            B_re[i][j] = math.cos(-2.0*math.pi*i*j/n)/math.sqrt(n)
            B_im[i][j] = math.sin(-2.0*math.pi*i*j/n)/math.sqrt(n)
    return (B_re.to(device), B_im.to(device))

# IDFT matrix
def idft(n):
    B_re = torch.zeros(n, n)
    B_im = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            B_re[i][j] = math.cos(2.0*math.pi*i*j/n)/math.sqrt(n)
            B_im[i][j] = math.sin(2.0*math.pi*i*j/n)/math.sqrt(n)
    return (B_re.to(device), B_im.to(device))


########## global variables
# system and alg. params.
m = 128
n = 128 
max_itr = 10

# clipping params. 
clip_mode = 1 # clipping mode selection
# 0: no clipping (linear observation)
# 1: amplitude clipping; clip(x) = x if |x|<=c, cx/|x| if |x|>c (c := clip_limit)
# 2: element-wise clipping; clip(x) = x1+ j*x2, where x1 = hardtanh(Re(x), c), x2 = hardtanh(Im(x), c)

clip_limit = 1.0 # strength of clipping effect 


# param. for numerical derivative
diff_delta = 0.001 

# learning params.
mbs_learn = 200 # minibatch size
num_batch = 500 # number of batches
learning_rate = 0.0005 # learning rate of optimizer

# BER measurement params.
n_loops = 100
mbs_measure = 1000

# SNRs
SNR = np.arange(15.0, 25.1, 5.0)

# random seeds
seed=12345
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
########## end of global variables

t_start = time.time()

A = idft(n)
A = (A[0][0:n :], A[1][0:n, :])
Ah = c_ht(A)
W = c_pseudo_inverse(A)

B = c_mm(Ah,A) # B=A^HA

taa = c_trace((c_mm(A, Ah)))[0]


print("#_ ", "n=", n, "m=",m, "max_itr=", max_itr, "mbs_learn=", mbs_learn, "num_batch=", num_batch,
      "learning_rate=", learning_rate, "n_loops=",n_loops, "mbs_measure=", mbs_measure)


mbs = mbs_learn

############### define (non-)linear element-wise function here
def c_measure_func(X):
    if clip_mode==0: # no clipping
        return X

    if clip_mode==1: # amplitude clipping; clip(x) = x if |x|<=c, cx/|x| if |x|>c
        eps = 1.0e-16
        norm = torch.pow(c_norm(X)+eps, 0.5) # |x|
        coeff = clip_limit/(norm)
        
        X_re = torch.where(norm<clip_limit, X[0], X[0]*coeff) # norm_i > clip_limit => X[0]_i = X[0]_i*coeff
        X_im = torch.where(norm<clip_limit, X[1], X[1]*coeff) 
        
        return ( X_re, X_im )
    
    if clip_mode==2: # element-wise clipping; clip(x) = x1+ j*x2, where x1 = hardtanh(Re(x), c), x2 = hardtanh(Im(x), c)
        htanh  = nn.Hardtanh(-clip_limit,clip_limit)
        return ( htanh(X[0]), htanh(X[1]) ) # element_wise
    
    else:
        print("Select proper clip_mode")
        return None

############### End of function definition
    

############### Derivative of c_measure_func(X)
##### Here, numerical Wirtinger derivative is implemented for general functions.
##### Using explicit Wirtinger derivative of your function is encouraged for time efficiency.

def c_diff(f, X): # input: function f=(f[0],f[1]), complex tensor X=(X[0],X[1])
    # neighbors of a given point X
    X_xplus  = (X[0]+diff_delta,X[1])
    X_xminus = (X[0]-diff_delta,X[1])
    X_yplus  = (X[0],X[1]+diff_delta)
    X_yminus = (X[0],X[1]-diff_delta)
    
    # numerical derivative
    g_x = (f(X_xplus)[0]-f(X_xminus)[0])/(2.0*diff_delta)
    h_x = (f(X_xplus)[1]-f(X_xminus)[1])/(2.0*diff_delta)
    g_y = (f(X_yplus)[0]-f(X_yminus)[0])/(2.0*diff_delta)
    h_y = (f(X_yplus)[1]-f(X_yminus)[1])/(2.0*diff_delta)
    
    df_dz_ast = ( 0.5*(g_x-h_y), 0.5*(h_x+g_y) )
    df_ast_dz_ast = ( 0.5*(g_x+h_y), 0.5*(-h_x+g_y) )
    
    return ( df_dz_ast, df_ast_dz_ast ) # output: ( df/dz^\ast, df^\ast/dz^\ast )

############### End of derivative definition


############### constellation
# 16QAM
M = 16
point = torch.tensor([
    [3.0,-3.0],
    [3.0,-1.0],
    [3.0,1.0],
    [3.0,3.0],
    [1.0,-3.0],
    [1.0,-1.0],
    [1.0,1.0],
    [1.0,3.0],
    [-1.0,-3.0],
    [-1.0,-1.0],
    [-1.0,1.0],
    [-1.0,3.0],
    [-3.0,-3.0],
    [-3.0,-1.0],
    [-3.0,1.0],
    [-3.0,3.0]
])
################


def gen_minibatch():
    re = []
    im = []
    for i in range(mbs):
        rindex = torch.randint(M, (1,n)).view(n)
        tmp = torch.index_select(point, 0, rindex)
        re.append(tmp[:,0].to(device))
        im.append(tmp[:,1].to(device))
    return (torch.stack(re), torch.stack(im)) 

def gen_word_batch():
    re = []
    im = []
    rind = []
    for i in range(mbs_measure):
        rindex = torch.randint(M, (1,n)).view(n)
        rind.append(rindex)
        tmp = torch.index_select(point, 0, rindex)
        re.append(tmp[:,0].to(device))
        im.append(tmp[:,1].to(device))
    return torch.stack(rind).to(device), (torch.stack(re), torch.stack(im))


def min_dist(X):
    re = X[0].view(mbs_measure,n,1).repeat(1,1,M) # dim = mbs*n*M
    im = X[1].view(mbs_measure,n,1).repeat(1,1,M)
    dist = (re - point[:,0].to(device))**2 + (im - point[:,1].to(device))**2

    return torch.argmin(dist, 2) # dim = mbs*n



# C-ISTA
class C_nonlin_ISTA(nn.Module):
    def __init__(self, max_itr):
        super(C_nonlin_ISTA, self).__init__()
        self.beta = nn.Parameter(0.1*torch.ones(max_itr))
        self.a    = nn.Parameter(1.0*torch.ones(max_itr))
        self.b    = nn.Parameter(0.1*torch.ones(max_itr))
        
    def c_shrinkage(self, x, var_mat):
        eps = 1e-10
        var_mat[var_mat<=0] = eps
        num_re = torch.zeros(mbs, n).to(device)
        num_im = torch.zeros(mbs, n).to(device)
        deno   = torch.zeros(mbs, n).to(device) + eps
        for i in range(M):
            r = (x[0] - point[i][0])**2 + (x[1] - point[i][1])**2
            f = torch.exp(-r/var_mat)
            num_re += point[i][0] * f
            num_im += point[i][1] * f
            deno += f
        return (num_re/deno, num_im/deno)
    
    def forward(self, num_itr):
        s = c_mm(y, dft(m)) # initial value: idft
        for i in range(num_itr):
            tmp0 = c_mm(s, A) # sA dim=mbs*m
            tmp = c_sub(y, c_measure_func(tmp0)) # y-f(sA) dim=mbs*m
            var = c_norm(tmp).sum(1)
            var = (var.expand(n, mbs)).t() # error variance
            diff = c_diff(c_measure_func,tmp0)
            par1 = c_hadamard_prod(c_conj(tmp), diff[0] ) # (y-f(sA))^\ast \cdot \frac{\partial f}{\partial z^\ast}
            par2 = c_hadamard_prod(tmp, diff[1]) # (y-f(sA)) \cdot \frac{\partial f^\ast}{\partial z^\ast}
            tmp2 = c_scalar_mul( self.beta[i]**2.0, c_mm(c_add(par1,par2),W) ) 
            r = c_add(s, tmp2)
            s = self.c_shrinkage(r, self.a[i]*var/taa + self.b[i])
            
        return s


# main loop
for snr in SNR:
    # calculation of signam for given SNR
    Es = 0.0
    for i in range(100):
        x = gen_minibatch()
        t = c_mm(x, A)
        t = c_measure_func(t)
        Es += (c_norm(t).sum()/(mbs*m)).detach().cpu().item()
    Es = Es / 100	

    sigma = math.sqrt(Es * 10.0**(-snr/10.0))
    print("\n #### SNR=", snr)
    
    mbs = mbs_learn
    loss_func = nn.MSELoss()
    model_non = C_nonlin_ISTA(max_itr).to(device)
    opt_non   = optim.Adam(model_non.parameters(), lr=learning_rate) 

    
    # learning C-TISTA
    for gen in (range(max_itr)):
        print("#_ Incremental training gen=", gen+1)
        for i in range(num_batch):
            x = gen_minibatch()
            w = c_normal(mbs, m, sigma)
            t = c_mm(x, A)
            t = c_measure_func(t)
            y = c_add(t, w)

            opt_non.zero_grad()
            with torch.autograd.detect_anomaly():
                x_hat = model_non(gen + 1)
                loss_non  = loss_func(x_hat[0], x[0]) + loss_func(x_hat[1], x[1])
                loss_non.backward()
                opt_non.step()
        
            if (i+1) % 100 ==0:
                print("C-TISTA loss:", gen, i, loss_non.item())


        # measure MSE and SER
        total_syms = 0
        sq_error_non = 0.0
        sq_error_idft = 0.0
        error_syms_non = 0
        error_syms_idft = 0

        dftm = dft(m)
        mbs = mbs_measure

        for i in tqdm(range(n_loops)):
            with torch.no_grad():# set require_grad = False      
                rind, x = gen_word_batch()
                w = c_normal(mbs_measure, m, sigma)
                t = c_mm(x, A)
                t = c_measure_func(t)
                y = c_add(t, w)
       
                # C-TISTA
                s = model_non(gen+1)
                sq_error_non += c_norm(c_sub(s, x)).sum().detach().item()
                error_syms_non += (min_dist(s) != rind).sum()
        
                # IDFT (baseline algorithm here)
                s = c_mm(y, dftm)
                sq_error_idft += c_norm(c_sub(s, x)).sum().detach().item()
                error_syms_idft += (min_dist(s) != rind).sum()
        
                total_syms += n*mbs_measure

            sq_error_non /= n_loops * n * mbs_measure
            sq_error_idft /= n_loops * n * mbs_measure


        print("\n SER, MSE vs. t: t, SER (C-TISTA), MSE (C-TISTA), SER (IDFT), MSE (IDFT)")
        print("{0} {1:e} {2:e} {3:e} {4:e}".format(gen+1,\
        error_syms_non.cpu().numpy()/total_syms, sq_error_non,\
        error_syms_idft.cpu().item(), error_syms_idft.cpu().numpy()/total_syms, sq_error_idft),flush = True)


    print("\n Trained params.: t, beta[t], a[t], b[t]")
    for gen in range(max_itr):
        print("{0} {1} {2:8.7f} {3:8.7f} {4:8.7f}".format(snr, gen,  model_non.beta[gen].detach(), model_non.a[gen].detach(), model_non.b[gen].detach() ) ,flush = True)

    print("#### Final results:")
    print('C-TISTA {0} [dB]'.format(snr))
    print('total_syms = ', total_syms)
    print('error_syms = ', error_syms_non.cpu().item())
    print('error prob = ', error_syms_non.cpu().numpy()/total_syms)
    print('squared error = ', sq_error_non)

    print('IDFT {0} [dB]'.format(snr))
    print('total_syms = ', total_syms)
    print('error_syms = ', error_syms_idft.cpu().item())
    print('error prob = ', error_syms_idft.cpu().numpy()/total_syms)
    print('squared error = ', sq_error_idft)
    print("###########")
    

t_end = time.time()
print("#_ time {0}".format(t_end-t_start))


