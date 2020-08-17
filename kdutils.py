import numpy as np
import time
import itertools
from joblib import Parallel, delayed
import multiprocessing
import os

#
# Roottest.m --> Frank S. Tsung, Dec, 2002
#

def croots( func,xmin,xmax,ymin,ymax, nx=100, ny=100, eps=1e-10, nroot_tol=0.9 ):
    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny

    x=np.linspace(xmin,xmax,nx)
    y=np.linspace(ymin,ymax,ny)

    f = np.zeros((nx,ny),dtype=complex)
    g = np.zeros((nx,ny))

    for m in range(nx):
        for n in range(ny):
            f[m,n] = func(x[m]+1j*y[n])
            if (np.abs(f[m,n]) < eps):
                g[m,n]= 1/eps
            else:
                g[m,n]= 1.0/np.abs(f[m,n])

    # plotting something?
    # imagesc(x,y,g)

    # I set the maximum number of roots to 30, if I found more than 30, Ill shut it off
    maxroots=50

    aroots=0
    eroots=0

    aroots_r=np.zeros(maxroots)
    aroots_i=np.zeros(maxroots)

    eroots_r=np.zeros(maxroots)
    eroots_i=np.zeros(maxroots)

    # find exact and approximate roots
    for m in range(1,nx-1):
        for n in range(1,ny-1):
            # print('g={}, 1/eps={}, =? {}'.format(g[m,n],1/eps,g[m,n] == 1/eps))
            if (g[m,n] == 1/eps):
                print('Found an exact root (you lucky dog), z = {} + {} i'.format(x[m],y[n]) )
                eroots_r[eroots]=x[m]
                eroots_i[eroots]=y[n]
                eroots=eroots+1
                if eroots==maxroots-1:
                    print('found too many eroots ({}), shutting off'.format(maxroots))
                    exit(-1) 
            elif( g[m+1,n]<g[m,n] and g[m-1,n]<g[m,n] and g[m,n+1]<g[m,n] and g[m,n-1]<g[m,n] ):
                print('Found an approximate root, z = {} + {}i'.format(x[m],y[n]) )
                aroots_r[aroots]=x[m]
                aroots_i[aroots]=y[n]
                aroots=aroots+1
                if aroots==maxroots-1:
                    print('found too many aroots ({}), shutting off'.format(maxroots))
                    exit(-1) 

    n_iters = 50

    # refine approximate roots
    if (aroots > 0):
        for m in range(aroots):
            # print('')
            print('Refining aroot {} of {}'.format(m,aroots))
            dx_temp=dx/2
            dy_temp=dy/2
            for n in range(n_iters):
                # print('f(aroot) = f({} + {} i) = {}'.format(aroots_r[m],aroots_i[m], np.abs(func(aroots_r[m]+1j*aroots_i[m]))))
                x0_t=aroots_r[m]-dx_temp
                y0_t=aroots_i[m]-dy_temp
                x1_t=aroots_r[m]+dx_temp
                y1_t=aroots_i[m]-dy_temp
                x2_t=aroots_r[m]-dx_temp
                y2_t=aroots_i[m]+dy_temp
                x3_t=aroots_r[m]+dx_temp
                y3_t=aroots_i[m]+dy_temp

                f0=np.abs(func(x0_t+1j*y0_t))
                f1=np.abs(func(x1_t+1j*y1_t))
                f2=np.abs(func(x2_t+1j*y2_t))
                f3=np.abs(func(x3_t+1j*y3_t))

                if(f0 < eps):
                    print('found an exact root z = {} + {} i'.format(x0_t,y0_t) )
                    # print('')
                    eroots_r[eroots]=x0_t
                    eroots_i[eroots]=y0_t
                    eroots=eroots+1
                    break
                elif (f1 < eps):
                    print('found an exact root z = {} + {} i'.format(x1_t,y1_t) )
                    # print('')
                    eroots_r[eroots]=x1_t
                    eroots_i[eroots]=y1_t
                    eroots=eroots+1
                    break
                elif (f2 < eps):
                    print('found an exact root z = {} + {} i'.format(x2_t,y2_t) )
                    # print('')
                    eroots_r[eroots]=x2_t
                    eroots_i[eroots]=y2_t
                    eroots=eroots+1
                    break
                elif (f3 < eps):
                    print('found an exact root z = {} + {} i'.format(x3_t,y3_t) )
                    # print('')
                    eroots_r[eroots]=x3_t
                    eroots_i[eroots]=y3_t
                    eroots=eroots+1
                    break
                elif ((f0 < f1) and (f0 < f2) and (f0 < f3) ):
                    aroots_r[m]=x0_t
                    aroots_i[m]=y0_t
                    dx_temp=dx_temp/2.0
                    dy_temp=dy_temp/2.0
                elif ((f1 < f2) and (f1 < f3)):
                    aroots_r[m]=x1_t
                    aroots_i[m]=y1_t
                    dx_temp=dx_temp/2.0
                    dy_temp=dy_temp/2.0   
                elif (f2 < f3):
                    aroots_r[m]=x2_t
                    aroots_i[m]=y2_t
                    dx_temp=dx_temp/2.0
                    dy_temp=dy_temp/2.0
                else:
                    aroots_r[m]=x3_t
                    aroots_i[m]=y3_t
                    dx_temp=dx_temp/2.0
                    dy_temp=dy_temp/2.0

            if (n==n_iters-1):
                print('did not find an eroot for this aroot')
                # print('')
            
            # don't think we need this, that's what the first four if/elifs do.
            # if (n >= n_iters and np.abs(func(aroots_r[m]+1j*aroots_i[m])) < 1e-5)
            #     eroots_r(eroots+1)=aroots_r[m]
            #     eroots_i(eroots+1)=aroots_i[m]
            #     eroots=eroots+1

    vroots = 0
    if (eroots > 0):

        roots = np.zeros(eroots,dtype=complex)
        for m in range(eroots):
            print('validating root {} of {}, z = {} + {}i'.format(m,eroots,eroots_r[m],eroots_i[m]))

            n = nroots(func, eroots_r[m]+1j*eroots_i[m], .01)
            
            if n>nroot_tol:
                print('  VALIDATED, nroots={} > {}'.format(n,nroot_tol))
                roots[vroots] = eroots_r[m]+1j*eroots_i[m]
                vroots += 1
            else:
                print('  NOT FUNNY, nroots={} <= {}'.format(n,nroot_tol))
                # just jam it in there anyway for the time being
                # roots[vroots] = eroots_r[m]+1j*eroots_i[m]

    # print('t0 = {}s, t1 = {}s, t2 = {}s, t3 = {}s'.format(t0,t1,t2,t3))

    if vroots > 0:
        return roots[0:vroots]
    else:
        return None

def croots_2( func,xmin,xmax,ymin,ymax, nx=100, ny=100, eps=1e-10, nroot_tol=0.9 ):

    val_radius = 0.02

    dx = (xmax-xmin)/nx
    dy = (ymax-ymin)/ny

    x=np.linspace(xmin,xmax,nx)
    y=np.linspace(ymin,ymax,ny)

    f = np.zeros((nx,ny),dtype=complex)
    g = np.zeros((nx,ny))

    for m in range(nx):
        for n in range(ny):
            f[m,n] = func(x[m]+1j*y[n])
            if (np.abs(f[m,n]) < eps):
                g[m,n]= 1/eps
            else:
                g[m,n]= 1.0/np.abs(f[m,n])

    # plotting something?
    # imagesc(x,y,g)

    # I set the maximum number of roots to 30, if I found more than 30, Ill shut it off
    maxroots=50

    aroots=0
    eroots=0

    aroots_r=np.zeros(maxroots)
    aroots_i=np.zeros(maxroots)

    eroots_r=np.zeros(maxroots)
    eroots_i=np.zeros(maxroots)

    # find exact and approximate roots
    for m in range(1,nx-1):
        for n in range(1,ny-1):
            # print('g={}, 1/eps={}, =? {}'.format(g[m,n],1/eps,g[m,n] == 1/eps))
            if (g[m,n] == 1/eps):
                print('Found an exact root (you lucky dog), z = {} + {} i'.format(x[m],y[n]) )
                eroots_r[eroots]=x[m]
                eroots_i[eroots]=y[n]
                eroots=eroots+1
                if eroots==maxroots-1:
                    print('found too many eroots ({}), shutting off'.format(maxroots))
                    exit(-1) 
            elif( g[m+1,n]<g[m,n] and g[m-1,n]<g[m,n] and g[m,n+1]<g[m,n] and g[m,n-1]<g[m,n] ):
                print('Found an approximate root, z = {} + {}i'.format(x[m],y[n]) )
                aroots_r[aroots]=x[m]
                aroots_i[aroots]=y[n]
                aroots=aroots+1
                if aroots==maxroots-1:
                    print('found too many aroots ({}), shutting off'.format(maxroots))
                    exit(-1) 

    n_iters = 50

    # refine approximate roots
    if (aroots > 0):
        for m in range(aroots):
            # print('')
            print('Refining aroot {} of {}'.format(m,aroots))
            dx_temp=dx/2
            dy_temp=dy/2
            for n in range(n_iters):
                # print('f(aroot) = f({} + {} i) = {}'.format(aroots_r[m],aroots_i[m], np.abs(func(aroots_r[m]+1j*aroots_i[m]))))
                x0_t=aroots_r[m]-dx_temp
                y0_t=aroots_i[m]-dy_temp
                x1_t=aroots_r[m]+dx_temp
                y1_t=aroots_i[m]-dy_temp
                x2_t=aroots_r[m]-dx_temp
                y2_t=aroots_i[m]+dy_temp
                x3_t=aroots_r[m]+dx_temp
                y3_t=aroots_i[m]+dy_temp

                f0=np.abs(func(x0_t+1j*y0_t))
                f1=np.abs(func(x1_t+1j*y1_t))
                f2=np.abs(func(x2_t+1j*y2_t))
                f3=np.abs(func(x3_t+1j*y3_t))

                if(f0 < eps):
                    print('found an exact root z = {} + {} i'.format(x0_t,y0_t) )
                    # print('')
                    eroots_r[eroots]=x0_t
                    eroots_i[eroots]=y0_t
                    eroots=eroots+1
                    break
                elif (f1 < eps):
                    print('found an exact root z = {} + {} i'.format(x1_t,y1_t) )
                    # print('')
                    eroots_r[eroots]=x1_t
                    eroots_i[eroots]=y1_t
                    eroots=eroots+1
                    break
                elif (f2 < eps):
                    print('found an exact root z = {} + {} i'.format(x2_t,y2_t) )
                    # print('')
                    eroots_r[eroots]=x2_t
                    eroots_i[eroots]=y2_t
                    eroots=eroots+1
                    break
                elif (f3 < eps):
                    print('found an exact root z = {} + {} i'.format(x3_t,y3_t) )
                    # print('')
                    eroots_r[eroots]=x3_t
                    eroots_i[eroots]=y3_t
                    eroots=eroots+1
                    break

                elif ((f0 < f1) and (f0 < f2) and (f0 < f3) ):
                    aroots_r[m]=x0_t
                    aroots_i[m]=y0_t
                    dx_temp=dx_temp/2.0
                    dy_temp=dy_temp/2.0
                elif ((f1 < f2) and (f1 < f3)):
                    aroots_r[m]=x1_t
                    aroots_i[m]=y1_t
                    dx_temp=dx_temp/2.0
                    dy_temp=dy_temp/2.0   
                elif (f2 < f3):
                    aroots_r[m]=x2_t
                    aroots_i[m]=y2_t
                    dx_temp=dx_temp/2.0
                    dy_temp=dy_temp/2.0
                elif (f3 < f2):
                    aroots_r[m]=x3_t
                    aroots_i[m]=y3_t
                    dx_temp=dx_temp/2.0
                    dy_temp=dy_temp/2.0
                else:  # we are here because it is undecided
                    dx_temp=1.5*dx_temp
                    dy_temp=1.5*dy_temp


            if (n==n_iters-1):
                print('did not find an eroot for this aroot')
                # print('')
            
            # don't think we need this, that's what the first four if/elifs do.
            # if (n >= n_iters and np.abs(func(aroots_r[m]+1j*aroots_i[m])) < 1e-5)
            #     eroots_r(eroots+1)=aroots_r[m]
            #     eroots_i(eroots+1)=aroots_i[m]
            #     eroots=eroots+1

    vroots = 0
    if (eroots > 0):

        roots = np.zeros(eroots,dtype=complex)
        for m in range(eroots):
            print('validating root {} of {}, z = {} + {}i'.format(m,eroots,eroots_r[m],eroots_i[m]))

            n = nroots(func, eroots_r[m]+1j*eroots_i[m], val_radius)
            
            if n>nroot_tol:
                print('  VALIDATED, nroots={} > {}'.format(n,nroot_tol))
                roots[vroots] = eroots_r[m]+1j*eroots_i[m]
                vroots += 1
            else:
                print('  NOT FUNNY, nroots={} <= {}'.format(n,nroot_tol))
                # just jam it in there anyway for the time being
                # roots[vroots] = eroots_r[m]+1j*eroots_i[m]

    # print('t0 = {}s, t1 = {}s, t2 = {}s, t3 = {}s'.format(t0,t1,t2,t3))

    if vroots > 0:
        return roots[0:vroots]
    else:
        return None


# Cauchy's argument principle
# This function returns the number of zeros of f inside the countour, a circle of radius
# rad centered around z0
# These routines are dumb they are super expensive to run 
def nroots(f,z0,rad,npoints=1000):
    dtheta  = 2*np.pi/npoints
    fminus  = f(z0+rad)
    s       = np.complex(0,0)
    for i in range(npoints):
        fplus = f( z0 + rad * (np.cos(i*dtheta)+1j*np.sin(i*dtheta)) )
        s    += (fplus-fminus) /(fplus+fminus)
        fminus = fplus

    return np.real( s / (1j*np.pi) ) # throw away small imag part


def test_nroot():
    def test(z):
       return (z-1)**2*(z-1j)

    print(nroots(test,1j,3))
    print(nroots(test,1,.01))
    print(nroots(test,1j,.01))

def test_croot():
    def test(z):
       return (z-1)**2*(z-1j)*(z-0.5-0.5*1j)

    roots = croots( test,-1,2,-1,2,nx=100,ny=100,eps=1e-5,nroot_tol=0.8 )
