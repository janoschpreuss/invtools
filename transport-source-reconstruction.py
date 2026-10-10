# ------------------------------ LOAD LIBRARIES -------------------------------
from netgen.geom2d import unit_square
from ngsolve import *
from xfem import *
from math import pi
#from invtools import *
ngsglobals.msg_level = 1

solver = "pardiso"
#solver = "umfpack"

# -------------------------------- PARAMETERS ---------------------------------
# Space finite element order
order = 3
# Time finite element order
k_t = order  
# Final simulation time
tend = 2.0
N = 1
#N = 32
# Time step
delta_t = tend / N 
gamma = 10 
gamma_data = 100
gamma_stab = 1000
# ----------------------------------- MAIN ------------------------------------
#mesh = Mesh(unit_square.GenerateMesh(maxh=0.03, quad_dominated=False))
mesh = Mesh(unit_square.GenerateMesh(maxh=0.07, quad_dominated=False))

dst = delta_t * dxtref(mesh, time_order=2*k_t + 2, skeleton=True, vb=BND)

H_coeff = CoefficientFunction((1.0 ,0.0))
p_coeff = CoefficientFunction(0.0)



def dt(u):
    return 1.0 / delta_t * dtref(u)

def ddt(u):
    return (1.0 / delta_t**2) * ddtref(u)

def dt_grad(u):
    return 1.0 / delta_t * dtref_grad(u)

def L_diffop(u):
    return dt(u) + InnerProduct(H_coeff,grad(u)) + p_coeff * u

def dtL_diffop(u):
    return ddt(u) + InnerProduct(H_coeff,dt_grad(u)) + p_coeff * dt(u)


def Solve_least_squares(mesh,N):
       
    dxt = delta_t * dxtref(mesh, time_order=2*k_t)
    dxold = dmesh(mesh, tref=0)

    told = Parameter(0)
    t = told + delta_t * tref
    t_slice = [n*delta_t + delta_t*tref for n in range(N)]
    

    u_exact_slice = [ sin(pi * t_slice[n]) * cos(x) * cos(y)  for n in range(N)]
    ut_exact_slice = [ pi * cos(pi * t_slice[n]) * cos(x) * cos(y)   for n in range(N)]
    ut0_exact_slice =  pi * cos(x) * cos(y) 
    utt_exact_slice = [ (-1) * pi**2 * sin(pi * t_slice[n]) * cos(x) * cos(y)  for n in range(N)]
    gradu_exact_slice = [ CoefficientFunction(( (-1) * sin(pi * t_slice[n]) * sin(x) * cos(y)  , (-1) * sin(pi * t_slice[n]) * cos(x) * sin(y) ))  for n in range(N)]
    gradut_exact_slice = [ CoefficientFunction(( (-1) * pi * cos(pi * t_slice[n]) * sin(x) * cos(y)  , (-1) * pi * cos(pi * t_slice[n]) * cos(x) * sin(y) ))  for n in range(N)]
     
    R_slice = [ pi * cos( pi * t_slice[n] ) * cos(x) - H_coeff[0] * sin(pi * t_slice[n]) * sin(x) + p_coeff * sin(pi * t_slice[n]) * cos(x) for n in range(N) ]
    Rt_slice = [ (-1) * pi**2 * sin( pi * t_slice[n] ) * cos(x) - pi * H_coeff[0] * cos(pi * t_slice[n]) * sin(x) + pi * p_coeff * cos(pi * t_slice[n]) * cos(x) for n in range(N) ]
    f_spatial = cos(y) 

    f_slice  = [ f_spatial * R_slice[n]  for n in range(N) ] 
    ft_slice = [ f_spatial * Rt_slice[n]  for n in range(N) ] 

    V = H1(mesh, order=order, dirichlet=[])
    tfe = ScalarTimeFE(k_t) 
    st_fes = tfe * V
    Xfes = H1CompoundSpaceTimeFESpace(st_fes,N)
    u, v = Xfes.TnT()

    a = BilinearForm(Xfes)
     
    a += (1/delta_t) * fix_Dtref(u[0],time=0) * fix_tref(v[0],time=0) * dmesh(mesh) 
    for n in range(N):
        a += L_diffop(u[n]) *  L_diffop(v[n]) * dxt
        a += dtL_diffop(u[n]) * dtL_diffop(v[n]) * dxt
        a += gamma * (1.0/delta_t) *  u[n] * v[n]  * dst
        
        if n > 0: 
            a += InnerProduct( fix_Dtref_Grad(u[n],time=0)-fix_Dtref_Grad(u[n-1],time=1),  fix_Dtref_Grad(v[n],time=0)-fix_Dtref_Grad(v[n-1],time=1)) * dmesh(mesh) # ok
            a += (1.0/delta_t**2) * (fix_Dtref(u[n],time=0)-fix_Dtref(u[n-1],time=1)) * (fix_Dtref(v[n],time=0)-fix_Dtref(v[n-1],time=1)) * dmesh(mesh)                                # ok
            a += (fix_Dtref(u[n],time=0,element_vb=BND)-fix_Dtref(u[n-1],time=1,element_vb=BND)) * (fix_Dtref(v[n],time=0,element_vb=BND)-fix_Dtref(v[n-1],time=1,element_vb=BND)) * dsmesh(mesh, element_vb=BND)  # ok

    a.Assemble()

    f = LinearForm(Xfes)
    f +=  ut0_exact_slice * fix_tref(v[0],time=0) * dmesh(mesh) 
    for n in range(N):
        f +=  f_slice[n] * L_diffop(v[n]) * dxt
        f +=  ft_slice[n] * dtL_diffop(v[n]) * dxt
        f += gamma * (1.0/delta_t) * u_exact_slice[n] * v[n] * dst

    f.Assemble()

    gfu = GridFunction(Xfes)
    gfu.vec.data = a.mat.Inverse(Xfes.FreeDofs(), inverse=solver) * f.vec

    # calculate L2-error
    l2_error_sum = 0.0 
    for n in range(N): 
        l2error = Integrate((u_exact_slice[n] - gfu.components[n])**2 * dxt, mesh)
        l2_error_sum += l2error
    print("l2_error = ", sqrt( l2_error_sum ))

#Solve_least_squares(mesh,N)


def Solve_for_source(mesh,N):

    dxt = delta_t * dxtref(mesh, time_order=2*k_t+2)
    dSigma = dst 
    dxold = dmesh(mesh, tref=0)

    told = Parameter(0)
    t = told + delta_t * tref
    t_slice = [n*delta_t + delta_t*tref for n in range(N)]
    
    u_exact_slice = [ sin(pi * t_slice[n]) * cos(x) * cos(y)  for n in range(N)]
    ut_exact_slice = [ pi * cos(pi * t_slice[n]) * cos(x) * cos(y)   for n in range(N)]
    ut0_exact_slice =  pi * cos(x) * cos(y) 
    utt_exact_slice = [ (-1) * pi**2 * sin(pi * t_slice[n]) * cos(x) * cos(y)  for n in range(N)]
    gradu_exact_slice = [ CoefficientFunction(( (-1) * sin(pi * t_slice[n]) * sin(x) * cos(y)  , (-1) * sin(pi * t_slice[n]) * cos(x) * sin(y) ))  for n in range(N)]
    gradut_exact_slice = [ CoefficientFunction(( (-1) * pi * cos(pi * t_slice[n]) * sin(x) * cos(y)  , (-1) * pi * cos(pi * t_slice[n]) * cos(x) * sin(y) ))  for n in range(N)]
     
    R_slice = [ pi * cos( pi * t_slice[n] ) * cos(x) - H_coeff[0] * sin(pi * t_slice[n]) * sin(x) + p_coeff * sin(pi * t_slice[n]) * cos(x) for n in range(N) ]
    Rt_slice = [ (-1) * pi**2 * sin( pi * t_slice[n] ) * cos(x) - pi * H_coeff[0] * cos(pi * t_slice[n]) * sin(x) + pi * p_coeff * cos(pi * t_slice[n]) * cos(x) for n in range(N) ]
    f_spatial = cos(y) 

    #f_slice  = [ f_spatial * R_slice[n]  for n in range(N) ] 
    #ft_slice = [ f_spatial * Rt_slice[n]  for n in range(N) ] 

    V = H1(mesh, order=order, dirichlet=[])
    tfe = ScalarTimeFE(k_t) 
    st_fes = tfe * V
    stfes_global = H1CompoundSpaceTimeFESpace(st_fes,N)
    Xfes =  stfes_global * V     

    (u,fh), (v,gh) = Xfes.TnT()

    a = BilinearForm(Xfes)
     
    a +=  fix_tref(u[0],time=0) * fix_tref(v[0],time=0) * dmesh(mesh)
    a +=  InnerProduct(fix_tref_Grad(u[0],time=0), fix_tref_Grad(v[0],time=0) ) * dmesh(mesh)
    #a += (1/delta_t) * fix_Dtref(u[0],time=0) * fix_tref(v[0],time=0) * dmesh(mesh) 
    for n in range(N):
        a += (L_diffop(u[n]) - R_slice[n] * fh) * (L_diffop(v[n]) - R_slice[n] * gh) * dxt
        a += (dtL_diffop(u[n]) - Rt_slice[n] * fh) * (dtL_diffop(v[n]) - Rt_slice[n] * gh)  * dxt
        a += gamma_data *  dt(u[n]) * dt(v[n]) * dSigma 
        
        if n > 0:
            #pass
            #print("hello")
            # probably need different scaling below. Note that time derivative is taken on reference interval
            a += gamma_stab * InnerProduct( fix_Dtref_Grad(u[n],time=0)-fix_Dtref_Grad(u[n-1],time=1),  fix_Dtref_Grad(v[n],time=0)-fix_Dtref_Grad(v[n-1],time=1)) * dmesh(mesh) # ok
            a += gamma_stab * (1.0/delta_t**2) * (fix_Dtref(u[n],time=0)-fix_Dtref(u[n-1],time=1)) * (fix_Dtref(v[n],time=0)-fix_Dtref(v[n-1],time=1)) * dmesh(mesh)                                # ok
            #a +=  (fix_Dtref(u[n],time=0)-fix_Dtref(u[n-1],time=1)) * (fix_Dtref(v[n],time=0)-fix_Dtref(v[n-1],time=1)) * dmesh(mesh)                                # ok
            a +=  gamma_stab * (fix_Dtref(u[n],time=0,element_vb=BND)-fix_Dtref(u[n-1],time=1,element_vb=BND)) * (fix_Dtref(v[n],time=0,element_vb=BND)-fix_Dtref(v[n-1],time=1,element_vb=BND)) * dsmesh(mesh, element_vb=BND)  # ok

    a.Assemble()

    f = LinearForm(Xfes)
    #f +=  ut0_exact_slice * fix_tref(v[0],time=0) * dmesh(mesh) 
    for n in range(N):
        #f +=  f_slice[n] * L_diffop(v[n]) * dxt
        #f +=  ft_slice[n] * dtL_diffop(v[n]) * dxt
        f += gamma_data * ut_exact_slice[n] * dt(v[n]) * dSigma 

    f.Assemble()

    gfuX = GridFunction(Xfes)
    gfuX.vec.data = a.mat.Inverse(Xfes.FreeDofs(), inverse=solver) * f.vec
    gfu,gff = gfuX.components

    # calculate L2-error
    l2_error_sum = 0.0 
    for n in range(N): 
        l2error = Integrate((u_exact_slice[n] - gfu.components[n])**2 * dxt, mesh)
        l2_error_sum += l2error
    print("l2_error u= ", sqrt( l2_error_sum ))

    l2_error_sum = 0.0 
    for n in range(N): 
        l2error = Integrate((u_exact_slice[n] - gfu.components[n])**2 * dSigma, mesh)
        l2_error_sum += l2error
    print("l2_error u data= ", sqrt( l2_error_sum ))


    l2_error_source = sqrt(Integrate((f_spatial - gff)**2 * dx, mesh))
    print("l2_error source= ", sqrt( l2_error_source ))

Solve_for_source(mesh,N)

