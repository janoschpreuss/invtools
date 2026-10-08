# ------------------------------ LOAD LIBRARIES -------------------------------
from netgen.geom2d import unit_square
from ngsolve import *
from xfem import *
from math import pi
from invtools import *
ngsglobals.msg_level = 1

solver = "pardiso"
#solver = "umfpack"

# -------------------------------- PARAMETERS ---------------------------------
# Space finite element order
order = 3
# Time finite element order
k_t = order  
# Final simulation time
tend = 1.0
N = 8
#N = 32
# Time step
delta_t = 1 / N 
gamma = 10 


# ----------------------------------- MAIN ------------------------------------
#mesh = Mesh(unit_square.GenerateMesh(maxh=0.05, quad_dominated=False))
#mesh = Mesh(unit_square.GenerateMesh(maxh=0.05, quad_dominated=False))
#mesh = Mesh(unit_square.GenerateMesh(maxh=0.015, quad_dominated=False))
#mesh = Mesh(unit_square.GenerateMesh(maxh=0.03, quad_dominated=False))
mesh = Mesh(unit_square.GenerateMesh(maxh=0.03, quad_dominated=False))

H_coeff = CoefficientFunction((1.0 ,1.0))
H4_coeff = CoefficientFunction((1.0 ,1.0,1.0,1.0))
p_coeff = CoefficientFunction(1.0)

dst = delta_t * dxtref(mesh, time_order=2*k_t, skeleton=True, vb=BND)

solution_flags = ["bnd-zero", "bnd-nonzero"]
solution_flag = solution_flags[1] 

if solution_flag  == "bnd-zero": 
    H_coeff = CoefficientFunction((1.0 ,1.0))
    p_coeff = CoefficientFunction(1.0)
elif solution_flag  == "bnd-nonzero":
    H_coeff = CoefficientFunction((1.0 ,0.0))
    p_coeff = CoefficientFunction(0.0)
else:
    raise Exception("solution_flag not recognized")



def dt(u):
    return 1.0 / delta_t * dtref(u)

def ddt(u):
    return (1.0 / delta_t**2) * ddtref(u)

def dt_grad(u):
    #return 1.0 / delta_t * dtref(grad(u))
    #print("u.shape = ", u.shape)
    #print("grad(u).shape = ", grad(u).shape)
    #print("dtref(grad(u)).dim = ", dtref_grad(u).dim )
    #print("dtref(grad(u)).shape = ", dtref_grad(u).shape )
    #input("")
    #help(dtref_grad(u))
    return 1.0 / delta_t * dtref_grad(u)
    #return dtref_grad(u)

def L_diffop(u):
    return dt(u) + InnerProduct(H_coeff,grad(u)) + p_coeff * u

#def L_diffop(u):
#    return dt(u) + InnerProduct(H_coeff,dt_grad(u)) + p_coeff * u


def dtL_diffop(u):
    #return ddt(u) + InnerProduct(H_coeff,dt_grad(u)) + p_coeff * dt(u)
    return ddt(u) + InnerProduct(H_coeff,dt_grad(u)) + p_coeff * dt(u)



def Solve(mesh,N):

    dxt = delta_t * dxtref(mesh, time_order=2*k_t)
    dxold = dmesh(mesh, tref=0)

    told = Parameter(0)
    t = told + delta_t * tref
    t_slice = [n*delta_t + delta_t*tref for n in range(N)]
    u_exact_slice = [sin(pi * t_slice[n]) * sin(pi * x)**2 * sin(pi * y)**2 for n in range(N)]
    ut_exact_slice = [ pi * cos(pi * t_slice[n]) * sin(pi * x)**2 * sin(pi * y)**2 for n in range(N)]
    utt_exact_slice = [ (-1) * pi**2 * sin(pi * t_slice[n]) * sin(pi * x)**2 * sin(pi * y)**2 for n in range(N)]

    gradu_exact_slice = [ CoefficientFunction(( 2 * pi * sin(pi * t_slice[n]) * cos(pi * x) * sin(pi * x) * sin(pi * y)**2, 2 * pi * sin(pi * t_slice[n]) * sin(pi * x)**2 *  cos(pi * y) * sin(pi * y)  ))  for n in range(N)]

    gradut_exact_slice = [ CoefficientFunction(( 2 * pi**2 * cos(pi * t_slice[n]) * cos(pi * x) * sin(pi * x) * sin(pi * y)**2, 2 * pi**2 * cos(pi * t_slice[n]) * sin(pi * x)**2 *  cos(pi * y) * sin(pi * y)  ))  for n in range(N)]

    f_slice = [ ut_exact_slice[n] + InnerProduct(H_coeff, gradu_exact_slice[n]) + p_coeff * u_exact_slice[n]  for n in range(N) ] 
    ft_slice = [ utt_exact_slice[n] + InnerProduct(H_coeff, gradut_exact_slice[n]) + p_coeff * ut_exact_slice[n]  for n in range(N) ] 

    V = H1(mesh, order=order, dirichlet=".*")
    ndofspace = V.ndof
    tfe = ScalarTimeFE(k_t) 
    st_fes = tfe * V
    #Xfes = FESpace([st_fes  for n in range(N)])
    Xfes = GlobalCompoundSpaceTimeSpace(st_fes,N,ndofspace)
    u, v = Xfes.TnT()

    a = BilinearForm(Xfes)
    for n in range(N):
        
        a += dt(u[n]) * v[n] * dxt
        a += InnerProduct(grad(u[n]), H_coeff)  * v[n] * dxt
        a +=  p_coeff * u[n] * v[n] * dxt

        if n > 0:
            pass
            #a += InnerProduct( fix_Dtref_Grad(u[n],time=0)-fix_Dtref_Grad(u[n-1],time=1),  fix_Dtref_Grad(v[n],time=0)-fix_Dtref_Grad(v[n-1],time=1)) * dmesh(mesh) # ok
            #a += (fix_Dtref(u[n],time=0)-fix_Dtref(u[n-1],time=1)) * (fix_Dtref(v[n],time=0)-fix_Dtref(v[n-1],time=1)) * dmesh(mesh)                                # ok
            #a += (fix_Dtref(u[n],time=0,element_vb=BND)-fix_Dtref(u[n-1],time=1,element_vb=BND)) * (fix_Dtref(v[n],time=0,element_vb=BND)-fix_Dtref(v[n-1],time=1,element_vb=BND)) * dsmesh(mesh, element_vb=BND)  # ok

    a.Assemble()

    f = LinearForm(Xfes)
    for n in range(N):
        f +=  f_slice[n] * v[n] * dxt
        
        #f += u_exact_slice[n] * v[n] * dxold # cheating 
        
        #f +=  u_exact_slice[n] * v[n] * dxt
    f.Assemble()

    gfu = GridFunction(Xfes)
    gfu.vec.data = a.mat.Inverse(Xfes.FreeDofs(), inverse=solver) * f.vec

    # calculate L2-error
    l2_error_sum = 0.0 
    for n in range(N): 
        l2error = Integrate((u_exact_slice[n] - gfu.components[n])**2 * dxt, mesh)
        l2_error_sum += l2error
    print("l2_error = ", sqrt( l2_error_sum ))

#Solve(mesh,N)


def Solve_least_squares(mesh,N):
    
     


    dxt = delta_t * dxtref(mesh, time_order=2*k_t)
    dxold = dmesh(mesh, tref=0)

    told = Parameter(0)
    t = told + delta_t * tref
    t_slice = [n*delta_t + delta_t*tref for n in range(N)]
    

    if solution_flag  == "bnd-zero": 
        u_exact_slice = [sin(pi * t_slice[n]) * sin(pi * x)**2 * sin(pi * y)**2 for n in range(N)]
        ut_exact_slice = [ pi * cos(pi * t_slice[n]) * sin(pi * x)**2 * sin(pi * y)**2 for n in range(N)]
        ut0_exact_slice = pi * sin(pi * x)**2 * sin(pi * y)**2 
        utt_exact_slice = [ (-1) * pi**2 * sin(pi * t_slice[n]) * sin(pi * x)**2 * sin(pi * y)**2 for n in range(N)]
        gradu_exact_slice = [ CoefficientFunction(( 2 * pi * sin(pi * t_slice[n]) * cos(pi * x) * sin(pi * x) * sin(pi * y)**2, 2 * pi * sin(pi * t_slice[n]) * sin(pi * x)**2 *  cos(pi * y) * sin(pi * y)  ))  for n in range(N)]
        gradut_exact_slice = [ CoefficientFunction(( 2 * pi**2 * cos(pi * t_slice[n]) * cos(pi * x) * sin(pi * x) * sin(pi * y)**2, 2 * pi**2 * cos(pi * t_slice[n]) * sin(pi * x)**2 *  cos(pi * y) * sin(pi * y)  ))  for n in range(N)]
        f_slice  = [ ut_exact_slice[n] + InnerProduct(H_coeff, gradu_exact_slice[n]) + p_coeff * u_exact_slice[n]  for n in range(N) ] 
        ft_slice = [ utt_exact_slice[n] + InnerProduct(H_coeff, gradut_exact_slice[n]) + p_coeff * ut_exact_slice[n]  for n in range(N) ] 
        ut_grad_H_exact_slice = [ InnerProduct(gradut_exact_slice[n],H_coeff)  for n in range(N)]
    elif solution_flag  == "bnd-nonzero":
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
        #ut_grad_H_exact_slice = [ InnerProduct(gradut_exact_slice[n],H_coeff)  for n in range(N)]

    else:
        raise Exception("solution_flag not recognized")


    #f_slice  = [ ut_exact_slice[n] + InnerProduct(H_coeff, gradu_exact_slice[n]) + p_coeff * u_exact_slice[n]  for n in range(N) ] 


    if solution_flag  == "bnd-zero": 
        #V = H1(mesh, order=order, dirichlet=".*")
        V = H1(mesh, order=order, dirichlet=[])
        
        
    elif solution_flag  == "bnd-nonzero":
        V = H1(mesh, order=order, dirichlet=[])
    else:
        raise Exception("solution_flag not recognized")

    ndofspace = V.ndof
    tfe = ScalarTimeFE(k_t) 
    st_fes = tfe * V
    #Xfes = FESpace([st_fes  for n in range(N)])
    Xfes = GlobalCompoundSpaceTimeSpace(st_fes,N,ndofspace)
    u, v = Xfes.TnT()



    a = BilinearForm(Xfes)
     
    a += (1/delta_t) * fix_Dtref(u[0],time=0) * fix_tref(v[0],time=0) * dmesh(mesh) 
    for n in range(N):
        a += L_diffop(u[n]) *  L_diffop(v[n]) * dxt
        a += dtL_diffop(u[n]) * dtL_diffop(v[n]) * dxt
        #a += (1.0/delta_t) *  u[n] * v[n]  * dst
        a += gamma * (1.0/delta_t) *  u[n] * v[n]  * dst
        
        #a +=  InnerProduct(dt_grad(u[n]), H_coeff) * v[n] * dxt 
        #a +=  InnerProduct(grad(u[n]), H_coeff) * v[n] * dxt 
        #a +=  dt(u[n])  * v[n] * dxt 
        #a += dt(u[n]) * v[n] * dxt
        #a += InnerProduct(grad(u[n]), H_coeff)  * v[n] * dxt
        #a += InnerProduct(dt_grad(u[n]), H_coeff)  * v[n] * dxt
        #a += p_coeff * u[n] * v[n] * dxt
        
        #a += InnerProduct(dt_grad(u[n]), dt_grad(v[n]))  * dxt

        #a += ddt(u[n]) * v[n] * dxt
        #a += dt(u[n]) * v[n] * dxt
        #a += grad(u[n]) * v[n] * dxt
        #a += InnerProduct(grad(u[n]), H_coeff)  * v[n] * dxt
        #a += p_coeff * u[n] * v[n] * dxt

        #print(" fix_Dtref_Grad(u[n],time=0).shape = ", fix_Dtref_Grad(u[n],time=0).shape )
        if n > 0:
            #pass
            #a += InnerProduct( fix_Dtref_Grad(u[n],time=0)-fix_Dtref_Grad(u[n-1],time=1),  fix_Dtref_Grad(v[n],time=0)-fix_Dtref_Grad(v[n-1],time=1)) * dmesh(mesh) # ok
            
            a += InnerProduct( fix_Dtref_Grad(u[n],time=0)-fix_Dtref_Grad(u[n-1],time=1),  fix_Dtref_Grad(v[n],time=0)-fix_Dtref_Grad(v[n-1],time=1)) * dmesh(mesh) # ok
            a += (1.0/delta_t**2) * (fix_Dtref(u[n],time=0)-fix_Dtref(u[n-1],time=1)) * (fix_Dtref(v[n],time=0)-fix_Dtref(v[n-1],time=1)) * dmesh(mesh)                                # ok
            
            a += (fix_Dtref(u[n],time=0,element_vb=BND)-fix_Dtref(u[n-1],time=1,element_vb=BND)) * (fix_Dtref(v[n],time=0,element_vb=BND)-fix_Dtref(v[n-1],time=1,element_vb=BND)) * dsmesh(mesh, element_vb=BND)  # ok

    a.Assemble()

    f = LinearForm(Xfes)
    f +=  ut0_exact_slice * fix_tref(v[0],time=0) * dmesh(mesh) 
    for n in range(N):
        #f +=  f_slice[n] * v[n] * dxt
        #f +=  f_slice[n] * v[n] * dxt

        #f +=  InnerProduct(gradut_exact_slice[n], H_coeff) * v[n] * dxt

        f +=  f_slice[n] * L_diffop(v[n]) * dxt

        #f += utt_exact_slice[n] * v[n] * dxt

        f +=  ft_slice[n] * dtL_diffop(v[n]) * dxt
        
        #f += (1.0/delta_t) * u_exact_slice[n] * v[n] * dst
        f += gamma * (1.0/delta_t) * u_exact_slice[n] * v[n] * dst

        #f += u_exact_slice[n] * v[n] * dxold # cheating 
        
        #f +=  ut_exact_slice[n] * v[n] * dxt
        #f +=  p_coeff * u_exact_slice[n] * v[n] * dxt

        #f += InnerProduct( gradut_exact_slice[n] , dt_grad(v[n]))  * dxt


    f.Assemble()

    gfu = GridFunction(Xfes)
    gfu.vec.data = a.mat.Inverse(Xfes.FreeDofs(), inverse=solver) * f.vec

    # calculate L2-error
    l2_error_sum = 0.0 
    for n in range(N): 
        l2error = Integrate((u_exact_slice[n] - gfu.components[n])**2 * dxt, mesh)
        l2_error_sum += l2error
    print("l2_error = ", sqrt( l2_error_sum ))

Solve_least_squares(mesh,N)




