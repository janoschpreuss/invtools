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
#gamma = 0.0 
gamma_data = 1e4
#gamma_stab = 1.0
gamma_stab = 100.0
# ----------------------------------- MAIN ------------------------------------
#mesh = Mesh(unit_square.GenerateMesh(maxh=0.03, quad_dominated=False))
#mesh = Mesh(unit_square.GenerateMesh(maxh=0.04, quad_dominated=False))
mesh = Mesh(unit_square.GenerateMesh(maxh=0.05, quad_dominated=False))
h = specialcf.mesh_size
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

#def dtL_diffop(u):
#    return InnerProduct(H_coeff,dt_grad(u)) 

#def dtL_diffop(u):
#    return dt(u) 


#def dtL_diffop(u):
#    return ddt(u)

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


    #u_exact_slice = [sin(pi * t_slice[n]) * sin(pi * x)**2 * sin(pi * y)**2 for n in range(N)]
    #ut_exact_slice = [ pi * cos(pi * t_slice[n]) * sin(pi * x)**2 * sin(pi * y)**2 for n in range(N)]
    #ut0_exact_slice = pi * sin(pi * x)**2 * sin(pi * y)**2 
    #utt_exact_slice = [ (-1) * pi**2 * sin(pi * t_slice[n]) * sin(pi * x)**2 * sin(pi * y)**2 for n in range(N)]
    #gradu_exact_slice = [ CoefficientFunction(( 2 * pi * sin(pi * t_slice[n]) * cos(pi * x) * sin(pi * x) * sin(pi * y)**2, 2 * pi * sin(pi * t_slice[n]) * sin(pi * x)**2 *  cos(pi * y) * sin(pi * y)  ))  for n in range(N)]
    #gradut_exact_slice = [ CoefficientFunction(( 2 * pi**2 * cos(pi * t_slice[n]) * cos(pi * x) * sin(pi * x) * sin(pi * y)**2, 2 * pi**2 * cos(pi * t_slice[n]) * sin(pi * x)**2 *  cos(pi * y) * sin(pi * y)  ))  for n in range(N)]
    #f_slice  = [ ut_exact_slice[n] + InnerProduct(H_coeff, gradu_exact_slice[n]) + p_coeff * u_exact_slice[n]  for n in range(N) ] 
    #ft_slice = [ utt_exact_slice[n] + InnerProduct(H_coeff, gradut_exact_slice[n]) + p_coeff * ut_exact_slice[n]  for n in range(N) ] 
    #ut_grad_H_exact_slice = [ InnerProduct(gradut_exact_slice[n],H_coeff)  for n in range(N)]

    f_slice  = [ f_spatial * R_slice[n]  for n in range(N) ] 
    ft_slice = [ f_spatial * Rt_slice[n]  for n in range(N) ] 
    

    #V = H1(mesh, order=order, dirichlet=".*")
    V = H1(mesh, order=order, dirichlet=[])
    tfe = ScalarTimeFE(k_t) 
    st_fes = tfe * V
    Xfes = H1CompoundSpaceTimeFESpace(st_fes,N)
    u, v = Xfes.TnT()

    a = BilinearForm(Xfes)
     
    #a += (1/delta_t) * fix_Dtref(u[0],time=0) * fix_tref(v[0],time=0) * dmesh(mesh) 
    a +=  fix_tref(u[0],time=0) * fix_tref(v[0],time=0) * dmesh(mesh) 
    a += (1/delta_t)**2 * fix_Dtref(u[0],time=0) * fix_Dtref(v[0],time=0) * dmesh(mesh) 
    #a +=  (1/delta_t) * fix_Dtref(u[0],time=0) * fix_Dtref(v[0],time=0) * dmesh(mesh) 
    for n in range(N):
        #a += u[n] * v[n] * dxt
        a += L_diffop(u[n]) *  L_diffop(v[n]) * dxt
        #a += ddt(u[n]) * ddt(v[n]) * dxt 
        a += dtL_diffop(u[n]) * dtL_diffop(v[n]) * dxt 
        
        a += gamma * (1.0/h) *  u[n] * v[n]  * dst
        #a += gamma *  u[n] * v[n]  * dst
        
        if n > 0: 
            a += gamma_stab * (1.0/delta_t**2) * InnerProduct( fix_Dtref_Grad(u[n],time=0)-fix_Dtref_Grad(u[n-1],time=1),  fix_Dtref_Grad(v[n],time=0)-fix_Dtref_Grad(v[n-1],time=1)) * dmesh(mesh) # ok
            #a += gamma_stab * (1.0/delta_t**2) * (fix_Dtref(u[n],time=0)-fix_Dtref(u[n-1],time=1)) * (fix_Dtref(v[n],time=0)-fix_Dtref(v[n-1],time=1)) * dmesh(mesh)                                # ok
            a += gamma_stab * (1.0/delta_t**4) * (fix_Dtref(u[n],time=0)-fix_Dtref(u[n-1],time=1)) * (fix_Dtref(v[n],time=0)-fix_Dtref(v[n-1],time=1)) * dmesh(mesh)                                # ok
            #a += gamma_stab * (1.0/delta_t**2) * (fix_tref(u[n],time=0)-fix_tref(u[n-1],time=1)) * (fix_tref(v[n],time=0)-fix_tref(v[n-1],time=1)) * dmesh(mesh)                                # ok
            a += gamma_stab * (1.0/delta_t**2) * (fix_Dtref(u[n],time=0,element_vb=BND)-fix_Dtref(u[n-1],time=1,element_vb=BND)) * (fix_Dtref(v[n],time=0,element_vb=BND)-fix_Dtref(v[n-1],time=1,element_vb=BND)) * dsmesh(mesh, element_vb=BND)  # ok

    a.Assemble()

    f = LinearForm(Xfes)
    #f +=  ut0_exact_slice * fix_tref(v[0],time=0) * dmesh(mesh) 
    #f +=  u0_exact_slice * fix_tref(v[0],time=0) * dmesh(mesh) 
    f += (1/delta_t) *  ut0_exact_slice * fix_Dtref(v[0],time=0) * dmesh(mesh) 
    #f += ut0_exact_slice * fix_Dtref(v[0],time=0) * dmesh(mesh) 
    for n in range(N):
        #f +=  u_exact_slice[n] * v[n] * dxt
        f +=  f_slice[n] * L_diffop(v[n]) * dxt
        f +=  ft_slice[n] * dtL_diffop(v[n]) * dxt
        #f +=  ut_exact_slice[n] * dtl_diffop(v[n]) * dxt
        #f +=  utt_exact_slice[n] * ddt(v[n]) * dxt 
        #f +=  InnerProduct(gradut_exact_slice[n], H_coeff) * dtL_diffop(v[n]) * dxt 
        f += gamma * (1.0/h)  * u_exact_slice[n] * v[n] * dst
        #f += gamma * u_exact_slice[n] * v[n] * dst

    f.Assemble()

    gfu = GridFunction(Xfes)
    gfu.vec.data = a.mat.Inverse(Xfes.FreeDofs(), inverse=solver) * f.vec

    # calculate L2-error
    l2_error_sum = 0.0 
    for n in range(N): 
        l2error = Integrate((u_exact_slice[n] - gfu.components[n])**2 * dxt, mesh)
        l2_error_sum += l2error
    print("l2_error = ", sqrt( l2_error_sum ))

    # calculate residual error
    l2_error_sum = 0.0 
    for n in range(N): 
        res_n = L_diffop(gfu.components[n]) - f_slice[n] 
        l2error = Integrate( res_n**2 * dxt, mesh)
        l2_error_sum += l2error
    print("res-error= ", sqrt( l2_error_sum ))

    l2_error_sum = 0.0 
    for n in range(N):
        #res_n =  dtL_diffop(gfu.components[n]) - utt_exact_slice[n]
        #res_n = dtL_diffop(gfu.components[n]) -  InnerProduct(gradut_exact_slice[n], H_coeff) 
        #res_n = dtL_diffop(gfu.components[n]) -  ut_exact_slice[n]
        res_n =  dtL_diffop(gfu.components[n]) - ft_slice[n] 
        l2error = Integrate( res_n**2 * dxt, mesh)
        l2_error_sum += l2error
    print("dt-res-error= ", sqrt( l2_error_sum ))

    # jump error
    l2_error_sum = 0.0 
    for n in range(N):
        if n > 0:
            res_n =  (fix_Dtref(gfu.components[n],time=0)-fix_Dtref(gfu.components[n-1],time=1))  
            l2error = Integrate( res_n**2 * dxt, mesh)
            l2_error_sum += l2error
    print("dt-ref-jump-error= ", sqrt( l2_error_sum ))


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
    #a += (1/delta_t)**2 * fix_Dtref(u[0],time=0) * fix_Dtref(v[0],time=0)  * dmesh(mesh) 
    for n in range(N):
        a += (L_diffop(u[n]) - R_slice[n] * fh) * (L_diffop(v[n]) - R_slice[n] * gh) * dxt
        a += (dtL_diffop(u[n]) - Rt_slice[n] * fh) * (dtL_diffop(v[n]) - Rt_slice[n] * gh)  * dxt
        a += gamma_data *  dt(u[n]) * dt(v[n]) * dSigma 
        
        if n > 0:
            a += gamma_stab * (1.0/delta_t**2) * InnerProduct( fix_Dtref_Grad(u[n],time=0)-fix_Dtref_Grad(u[n-1],time=1),  fix_Dtref_Grad(v[n],time=0)-fix_Dtref_Grad(v[n-1],time=1)) * dmesh(mesh) # ok
            a += gamma_stab * (1.0/delta_t**4) * (fix_Dtref(u[n],time=0)-fix_Dtref(u[n-1],time=1)) * (fix_Dtref(v[n],time=0)-fix_Dtref(v[n-1],time=1)) * dmesh(mesh)                                # ok
            a += gamma_stab * (1.0/delta_t**2) * (fix_Dtref(u[n],time=0,element_vb=BND)-fix_Dtref(u[n-1],time=1,element_vb=BND)) * (fix_Dtref(v[n],time=0,element_vb=BND)-fix_Dtref(v[n-1],time=1,element_vb=BND)) * dsmesh(mesh, element_vb=BND)  # ok

            #pass
            #print("hello")
            # probably need different scaling below. Note that time derivative is taken on reference interval
            #a += gamma_stab * InnerProduct( fix_Dtref_Grad(u[n],time=0)-fix_Dtref_Grad(u[n-1],time=1),  fix_Dtref_Grad(v[n],time=0)-fix_Dtref_Grad(v[n-1],time=1)) * dmesh(mesh) # ok
            #a += gamma_stab * (1.0/delta_t**2) * (fix_Dtref(u[n],time=0)-fix_Dtref(u[n-1],time=1)) * (fix_Dtref(v[n],time=0)-fix_Dtref(v[n-1],time=1)) * dmesh(mesh)                                # ok
            #a +=  (fix_Dtref(u[n],time=0)-fix_Dtref(u[n-1],time=1)) * (fix_Dtref(v[n],time=0)-fix_Dtref(v[n-1],time=1)) * dmesh(mesh)                                # ok
            #a +=  gamma_stab * (fix_Dtref(u[n],time=0,element_vb=BND)-fix_Dtref(u[n-1],time=1,element_vb=BND)) * (fix_Dtref(v[n],time=0,element_vb=BND)-fix_Dtref(v[n-1],time=1,element_vb=BND)) * dsmesh(mesh, element_vb=BND)  # ok

    a.Assemble()

    f = LinearForm(Xfes)
    #f +=  (1/delta_t) * ut0_exact_slice * fix_Dtref(v[0],time=0)  * dmesh(mesh) 
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
        #l2error = Integrate((u_exact_slice[n] - gfu.components[n])**2 * dxt, mesh)
        l2error = Integrate((u_exact_slice[n] - gfu.components[n])**2 * dxt, mesh)
        #l2error = Integrate(( InnerProduct(dt_grad(gfu.components[n]),  dt_grad(gfu.components[n])))**2 * dxt, mesh)
        l2_error_sum += l2error
    print("l2_error u= ", sqrt( l2_error_sum ))

    # calculate data-error
    l2_error_sum = 0.0 
    for n in range(N): 
        l2error = Integrate((u_exact_slice[n] - gfu.components[n])**2 * dSigma, mesh)
        l2_error_sum += l2error
    print("l2_error u data= ", sqrt( l2_error_sum ))

    # calculate residual error
    l2_error_sum = 0.0 
    for n in range(N): 
        res_n = L_diffop(gfu.components[n]) - R_slice[n] * gff
        l2error = Integrate( res_n**2 * dxt, mesh)
        l2_error_sum += l2error
    print("res-error= ", sqrt( l2_error_sum ))

    l2_error_sum = 0.0 
    for n in range(N): 
        res_n =  dtL_diffop(gfu.components[n]) - Rt_slice[n] * gff
        l2error = Integrate( res_n**2 * dxt, mesh)
        l2_error_sum += l2error
    print("dt-res-error= ", sqrt( l2_error_sum ))


    l2_error_source = sqrt(Integrate((f_spatial - gff)**2 * dx, mesh))
    print("l2_error source= ", sqrt( l2_error_source ))

Solve_for_source(mesh,N)


'''
order = 3
# Time finite element order
k_t = order   
# Final simulation time
tend = 2.0
N = 16
# Time step
delta_t = tend / N 
gamma = 10 
gamma_data = 1e4
gamma_stab = 100.0

l2_error u
l2_errors = [ 0.3224683047427635,0.5274009054078607,0.04560999294416,0.0032809181513545854,0.0002172026904258848]
eoc = [-0.7097426822397304, 3.5314781958061823, 3.797178438322321, 3.9169857249716076]

res-error
l2_errors = [ 0.7691030476836759,0.7743065912984066,0.08159575427526455,0.00862934696441242,0.0008788882837092856]
eoc = [-0.009728013642602593, 3.2463389325952607, 3.2411707943947445, 3.295499686102978]

dt-res-error
l2_errors = [4.185204676785611,  2.094584619238267, 0.6301639488725059, 0.16380527099368938, 0.041176378664597696  ]
eoc = [0.9986340103807234, 1.7328650423880139, 1.9437454400337943, 1.9920929212767104]

l2_errors_source
l2_errors = [ 0.627724931308285,0.8687594991889946 ,0.2419126449349382,0.060606255236560956,0.01541128686826275 ]
eoc =  [-0.4688243380445567, 1.844470665605347, 1.996947573321636, 1.9754793691497956]
'''
