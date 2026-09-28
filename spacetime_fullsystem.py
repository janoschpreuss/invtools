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
order = 1
# Time finite element order
k_t = 1
# Final simulation time
tend = 1.0
N = 16
# Time step
delta_t = 1 / N 


# ----------------------------------- MAIN ------------------------------------
#mesh = Mesh(unit_square.GenerateMesh(maxh=0.05, quad_dominated=False))
mesh = Mesh(unit_square.GenerateMesh(maxh=0.05, quad_dominated=False))


def dt(u):
    return 1.0 / delta_t * dtref(u)

def Solve(mesh,N):
   
    dxt = delta_t * dxtref(mesh, time_order=2)
    dxold = dmesh(mesh, tref=0)

    told = Parameter(0)
    t = told + delta_t * tref
    t_slice = [n*delta_t + delta_t*tref for n in range(N)]
    u_exact_slice = [sin(pi * t_slice[n]) * sin(pi * x)**2 * sin(pi * y)**2 for n in range(N)]
    #f_slice = [ u_exact_slice[n].Diff(t) - (u_exact_slice[n].Diff(x).Diff(x) + u_exact_slice[n].Diff(y).Diff(y)) for n in range(N) ]
    f_slice = [ pi*cos(pi * t_slice[n]) * sin(pi * x)**2 * sin(pi * y)**2 - 2*pi**2* sin(pi * t_slice[n]) * ( cos(2*pi*x) * sin(pi * y)**2 + sin(pi * x)**2 * cos(2 * pi *y ))  for n in range(N)] 


    V = H1(mesh, order=order, dirichlet=".*")
    ndofspace = V.ndof
    tfe = ScalarTimeFE(k_t) 
    st_fes = tfe * V
    #Xfes = FESpace([st_fes  for n in range(N)])
    Xfes = GlobalCompoundSpaceTimeSpace(st_fes,N,ndofspace)
    u, v = Xfes.TnT()

    a = BilinearForm(Xfes)
    for n in range(N):
        
        #a += u[n] * v[n] * dxt
        a += grad(u[n]) * grad(v[n]) * dxt
        #a += u * v * dxold
        a += dt(u[n]) * v[n] * dxt
        #if n > 0:
        #    (u[n]-u[n-1]) * v[n] * dxold
        
        #a += u[n] * v[n] * dxold # cheating 
        #if n > 0:
        #    print("adding jump term")
        #    #a += (fix_tref_proxy(dt(u[n]),0)-fix_tref_proxy(dt(u[n-1]),1)) * fix_tref(v[n],0) * dmesh(mesh)


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

Solve(mesh,N)

# Fitted heat equation example
'''
tnew = 0
t = told + delta_t * tref

u_exact = sin(pi * t) * sin(pi * x)**2 * sin(pi * y)**2

coeff_f = u_exact.Diff(t) - (u_exact.Diff(x).Diff(x) + u_exact.Diff(y).Diff(y))
coeff_f = coeff_f.Compile()

gfu = GridFunction(st_fes)
u_last = CreateTimeRestrictedGF(gfu, 1)

dxnew = dmesh(mesh, tref=1)




a = BilinearForm(st_fes, symmetric=False)
a += grad(u) * grad(v) * dxt
a += u * v * dxold
a += dt(u) * v * dxt
a.Assemble()

f = LinearForm(st_fes)
f += coeff_f * v * dxt
f += u_last * v * dxold

u_last.Set(fix_tref(u_exact, 0))
Draw(u_last, mesh, "u")






while tend - told.Get() > delta_t / 2:
    f.Assemble()
    gfu.vec.data = a.mat.Inverse(st_fes.FreeDofs(), "") * f.vec
    RestrictGFInTime(spacetime_gf=gfu, reference_time=1.0, space_gf=u_last)
    l2error = sqrt(Integrate((u_exact - gfu)**2 * dxnew, mesh))
    Redraw()
    told.Set(told.Get() + delta_t)
    print("\rt = {0:12.9f}, L2 error = {1:12.9e}".format(told.Get(), l2error))
'''




