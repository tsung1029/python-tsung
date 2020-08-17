import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import mpmath
import matplotlib
import warnings
import multiprocessing
import itertools
import time
from joblib import Parallel, delayed
import os

from kddef import *
import kdutils


# process warnings as exceptions
warnings.filterwarnings("error")


def simple_test():
    
    # contrived example that I already know the answer to just to test out the root finder

    k = 1
    vth_e = 1
    wp_e = 0
    vth_i = 1
    wp_i = 0
    k0=0
    w0=0
    gamma_p=0
    gamma_m = 0
    c = 1
    v0=1

    wr = np.sqrt(0.5)
    wi = 0

    e = f_epsilon_ri(wi, wr, k, k0, w0, gamma_p, gamma_m, c, v0, wp_e = 0, X_e=1, X_i=0)

    print('w = {} + {}i; epsilon(w) = {}'.format(wr,wi,e))

    wi_guess = 0.1

    wi_root = srs_dispersion_ri(wi_guess, wr, k, k0, w0, gamma_p, gamma_m, c, v0, wp_e = 0, X_e=1, X_i=0)

    print('root finder result: wi_root = {}'.format(wi_root))


def w_rwpe_vs_k_parallel(func, k_arr, r_wp_arr, root_trial_arr, vth, vosc, solver, \
        cpu_count):

    nw = r_wp_arr.shape[0]
    nk = k_arr.shape[0]

    paramlist = list(itertools.product(range(nk),range(nw)))

    def roots_on_grid(params):
        print(params)
        k_val      = k_arr[params[0]]
        wp_val     = 1.0/r_wp_arr[params[1]]

        def epsilon(omega):
            return func(omega,k_val,wp_val,vth,vosc)

        # w0 = np.sqrt(1+wp*wp)
        wr_min = np.float(landau_damping( k_arr[0], wp_val, vth ) / 2)
        wr_max = np.float(landau_damping( k_arr[-1], wp_val, vth ) * 2)
        wi_min = -.1
        wi_max = 1  #vosc * wp**2 / (2 * np.sqrt(2) * w0) # kruer 7.25 or something

        newroot = kdutils.croots(epsilon,wr_min,wr_max,wi_min,wi_max, nx=250, ny=250)

        # keep only one root, root with largest wi
        r = 0
        rm = 1
        if newroot is not None:
            rm = 0
            r = newroot[0]
            for l in range(1,len(newroot)):
                if newroot[l].imag>newroot[l-1].imag:
                    r = newroot[l]

        return (r, rm)

    r = Parallel(n_jobs=cpu_count)( delayed(roots_on_grid)(param) for param in paramlist)

    rr = np.reshape( np.array([ i[0].real for i in r ]), (nk,nw) )
    ri = np.reshape( np.array([ i[0].imag for i in r ]), (nk,nw) )
    rm = np.sum( np.array([ i[1] for i in r ]) )

    return rr, ri, rm


def w_rwpe_vs_k(func, k_arr, r_wp_arr, root_trial_arr, vth, vosc, solver):

    # plot wi in 1/wpe vs k space

    nw = r_wp_arr.shape[0]
    nk = k_arr.shape[0]

    ri = np.zeros((nk,nw))
    rr = np.zeros((nk,nw))

    rm = 0

    # root_trial = 0

    for i in range(nk):
        for j in range(nw):
            k_val      = k_arr[i]
            wp_val     = 1.0/r_wp_arr[j]
            def epsilon(omega):
                return func(omega,k_val,wp_val,vth,vosc)
            if solver=='mpmath':
                try:
                    root_trial = root_trial_arr[i,j]
                    newroot=mpmath.findroot(epsilon,root_trial,solver='muller',tol=1e-18)
                    ri[i,j] = newroot.imag
                    rr[i,j] = newroot.real
                    ###
                    # use current root as guess for next root
                    #root_trial = newroot
                    ###
                    print('[i,j] = {:3}, {:3}; (r_wp,k) = {:5.2f}, {:5.2f}; newroot = {}'.format(i, j, r_wp_arr[j],k_val,newroot))
                except ValueError:
                    rm += 1
                    print('')
                    print('mpmath.findroot raised ValueError @ [i,j] = {:3}, {:3}; (r_wp,k) = {:5.2f}, {:5.2f}'.format(i, j, r_wp_arr[j],k_val))
                    print('')
                except RuntimeWarning:
                    rm += 1
                    print('')
                    print('mpmath.findroot raised RuntimeWarning @ [i,j] = {:3}, {:3}; (r_wp,k) = {:5.2f}, {:5.2f}'.format(i, j, r_wp_arr[j],k_val))
                    print('')

            elif solver=='croot':
                print('[i,j] = {:3}, {:3}; (r_wp,k) = {:5.2f}, {:5.2f}'.format(i,j,r_wp_arr[j],k_val))

                # w0 = np.sqrt(1+wp*wp)
                wr_min = np.float(landau_damping( k_arr[0], wp_val, vth ) / 2)
                wr_max = np.float(landau_damping( k_arr[-1], wp_val, vth ) * 2)
                wi_min = -.1
                wi_max = 1  #vosc * wp**2 / (2 * np.sqrt(2) * w0) # kruer 7.25 or something

                newroot = kdutils.croots(epsilon,wr_min,wr_max,wi_min,wi_max, nx=250, ny=250)
                print(newroot)
                # keep only one root, root with largest wi
                if newroot is not None:
                    rr[i,j] = newroot[0].real
                    ri[i,j] = newroot[0].imag
                    for l in range(1,len(newroot)):
                        if newroot[l].imag>newroot[l-1].imag:
                            rr[i,j] = newroot[l].real
                            ri[i,j] = newroot[l].imag
                else:
                    rm += 1

                print('')

    return rr, ri, rm


def wi_rwpe_vs_k_complex(func,k_arr,r_wp_arr,wr_arr,root_trial_arr,vth,vosc):

    # plot wi in 1/wpe vs k space
    # func to test epsilon_manc

    nw=r_wp_arr.shape[0]
    nk=k_arr.shape[0]

    rr = np.zeros((nk,nw))

    # root_trial = 0
    for i_mode in range(0,nk):
        for j_mode in range(0,nw):
            k_val      = k_arr[i_mode]
            wp_val     = 1.0/r_wp_arr[j_mode]
            root_trial = root_trial_arr[i_mode,j_mode]
            wr_val     = wr_arr[i_mode,j_mode]

            def epsilon(wi):
                return func(wi,wr_val,k_val,wp_val,vth,vosc)

            newroot=mpmath.findroot(epsilon,root_trial,solver='muller',tol=1e-18)
            # newroot=mpmath.findroot(epsilon,root_trial,solver='halley',tol=1e-5) # have had some luck with this. Can't remember what the tolerance was tho. Actually not, I had to give it the answer in order for it to get it...stupid root solver algoritm. Also it doesn't keep it complex.

            # # performs well for epsilon_fluid_complex()
            # if np.abs(newroot.imag) < 5e-1:
            #     root_trial = newroot.real
            #     rr[i_mode,j_mode] = newroot.imag

            # performs well for epsilon_kinetic_complex()
            rr[i_mode,j_mode] = newroot.real
            # root_trial = newroot.real

            print('[i,j] = {}, {} newroot = {}'.format(i_mode,j_mode,newroot))

    return rr


def wi_k_vs_wr(func, k_arr, wr_arr, root_trial_arr, wp, vth, vosc):

    # plot wi in k vs wr space

    nw = wr_arr.shape[0]
    nk = k_arr.shape[0]

    rr = np.zeros((nw,nk))
    ri = np.zeros((nw,nk))

    root_trial = 0
    for i in range(0,nw):
        for j in range(0,nk):
            wr_val     = wr_arr[i]
            k_val      = k_arr[j]

            # only use root trial if it's not 0. Otherwise use previous root as guess.
            if root_trial_arr[i,j]!=0:
                print('return')
                exit(1)
                root_trial = root_trial_arr[i,j]

            def epsilon(wi):
                return func(wi,wr_val,k_val,wp,vth,vosc)

            newroot = mpmath.findroot(epsilon,root_trial,solver='muller',tol=1e-18)
            # newroot = mpmath.findroot(epsilon,root_trial,solver='halley',tol=1e-2)
            # newroot = scipy.optimize.newton(epsilon,root_trial,tol=1e-18)

            if np.abs(newroot.imag) < 1e-1:
                # if newroot.real < 0: # should be doing this but not working so great
                rr[i,j] = newroot.real
            ri[i,j] = newroot.imag

            root_trial = newroot.real

            print('[i,j] = {}, {} newroot = {}'.format(i,j,newroot.real))


    return rr, ri


def landau_damping( k, wp, vth ):
    def epsilon(omega):
        return epsilon_landau_damping(omega,k,wp,vth)
    newroot=mpmath.findroot(epsilon,0,solver='muller',tol=1e-18)
    return newroot.real


def bohm_gross( k, wp, vth ):
    rr = np.sqrt( wp**2 + 3 * vth**2 * k**2 )
    return rr


def imshow_w_contours(vals, vosc, vth, wnorm, save='imshow_w_contours.png', c_xarr=None, c_yarr=None, **kwargs):
    plt.figure(figsize=[10,10])

    plt.imshow(vals,origin='lower',**kwargs)
    plt.colorbar(label='$w_{{imag}} [{}]$'.format(wnorm))

    if (c_xarr is not None and c_yarr is not None):
        contours = plt.contour(c_xarr,c_yarr,vals,colors='k',levels=[-7.5e-2,-4.5e-2,-1.5e-2,1.5e-2,4.5e-2,7.5e-2])
        plt.clabel(contours,inline=True, fontsize=8)

    plt.gca().set_aspect('auto')
    plt.xlabel('$k_0 c / w_{pe}$')
    plt.ylabel('$k/k_0$')
    plt.title('vosc={}, vth={}'.format(vosc,vth))
    plt.savefig(save)
    plt.close('all')


def fig2a(nw=25, nk=25, r_wp_ext=[1.6,3], k_ext=[.2,1.8], vth=0.1, vosc=0.04, \
        epsilon=epsilon_kinetic,wnorm='ck0',cpu_count=None):

    # This solves the dispersion relation for Parametric Instabilities in Forslund et al
    # here we try to recover Figure 2(a) from Forslund
    # Normalization is k0=1 c=1
    
    k_arr = np.linspace(k_ext[0], k_ext[1], num=nk)
    r_wp_arr = np.linspace(r_wp_ext[0], r_wp_ext[1],num=nw)

    ##
    ## fiddling around with different intial geusses and stuff
    ##
    # trial_arr = np.zeros([np.size(k_arr,0),np.size(r_wp_arr,0)],dtype='complex')
    # for i in range(np.size(k_arr,0)):
    #     for j in range(np.size(r_wp_arr,0)):
    #         # wp = 1.0/r_wp_arr[j]
    #         # trial_arr[i,j] = np.sqrt(wp*wp+3.0*vth*vth*k_arr[i]*k_arr[i])
    #         trial_arr[i,j] = 1e-1*1j

    # use fluid as guess
    # rr,ri,rm = w_rwpe_vs_k(epsilon_fluid, k_arr, r_wp_arr, trial_arr, vth, vosc)
    # ri = np.abs(ri)
    # trial_arr = 1j * ri

    # rr,ri,rm = w_rwpe_vs_k(epsilon_kinetic, k_arr, r_wp_arr, trial_arr, vth, vosc)
    ##

    trial_arr = np.zeros([np.size(k_arr,0),np.size(r_wp_arr,0)],dtype='complex')

    if cpu_count is not None:
        rr,ri,rm = w_rwpe_vs_k_parallel(epsilon, k_arr, r_wp_arr, trial_arr, vth, vosc, \
                solver='croot',cpu_count=cpu_count)
    else:
        rr,ri,rm = w_rwpe_vs_k(epsilon, k_arr, r_wp_arr, trial_arr, vth, vosc, \
                solver='croot')


    print('roots missed: {}'.format(rm))

    ## post process

    # normalize by wpe
    if wnorm=='wpe':
        rr = np.multiply(r_wp_arr,rr)
        ri = np.multiply(r_wp_arr,ri)

    extent_ = [r_wp_ext[0],r_wp_ext[1],k_ext[0],k_ext[1]]

    cmap_ = 'RdYlBu'

    # plot wi
    kwargs_ = {'norm':matplotlib.colors.SymLogNorm(1e-15), 'extent':extent_, 'cmap':cmap_, \
        'vmin':-np.max(np.abs(ri)),'vmax':np.max(np.abs(ri)) }
    imshow_w_contours(ri,vosc,vth,wnorm,save='fig2a_imag.png',c_xarr=r_wp_arr, c_yarr=k_arr,**kwargs_)


    # plot wr
    imshow_w_contours(rr,vosc,vth,wnorm,save='fig2a_real.png',extent=extent_)

    # wr vs k scatterplot with wi colorbar at fixed wpe
    kwargs_ = {'norm':matplotlib.colors.SymLogNorm(1e-10), \
               'cmap':'coolwarm_r', \
               'vmin':-np.max(np.abs(ri)),'vmax':np.max(np.abs(ri)) }
               # 'vmin':-5e-2,'vmax':5e-2 }
               # 'vmin':-1e-1,'vmax':1e-1 }

    r_wp_idx = int(.5*len(r_wp_arr))

    bg = [ bohm_gross    (k,1.0/r_wp_arr[r_wp_idx],vth) for k in k_arr ]
    ld = [ landau_damping(k,1.0/r_wp_arr[r_wp_idx],vth) for k in k_arr ]
    if (wnorm=='wpe'):
        ld = np.multiply(r_wp_arr[r_wp_idx], ld)
        bg = np.multiply(r_wp_arr[r_wp_idx], bg)
        rr = np.multiply(r_wp_arr[r_wp_idx], rr)
        ri = np.multiply(r_wp_arr[r_wp_idx], ri)

    plt.figure()
    plt.plot( k_arr, bg, label='bohm gross')
    plt.plot( k_arr, ld, label='landau damping' )
    plt.scatter(k_arr, rr[:,r_wp_idx], c=ri[:,r_wp_idx], **kwargs_)
    plt.colorbar(label='$w_{{imag}} [{}]$'.format(wnorm))
    plt.title('vosc={:.3f}, vth={:.3f}, 1/wp={:.3f}, lambdad={:.3f}'.format(vosc,vth,r_wp_arr[r_wp_idx],vth*r_wp_arr[r_wp_idx]))
    plt.ylabel('$\omega_{{real}} [{}]$'.format(wnorm))
    plt.xlabel('$k/k_0$')
    plt.legend(loc=0)
    plt.savefig('fig2a_real_lineout.png')
    plt.close('all')


def test_croot_1d(nk=25, r_wp=2.3, k_ext=[.2,1.8], vth=0.1, vosc=0.04 ,wnorm='ck0'):

    k_arr = np.linspace(k_ext[0], k_ext[1], num=nk)
    r_wp_arr = np.array([r_wp])

    rr = np.zeros(0)
    ri = np.zeros(0)
    kp = np.zeros(0)

    wp = 1.0/r_wp
    w0 = np.sqrt(1+wp*wp)

    wr_min = np.float(landau_damping( k_arr[0], wp, vth ) / 2)
    wr_max = np.float(landau_damping( k_arr[-1], wp, vth ) * 2)
    wi_min = -.1
    wi_max = .5  #vosc * wp**2 / (2 * np.sqrt(2) * w0) # kruer 7.25 or something

    for i in range(nk):
        k_val = k_arr[i]

        def epsilon(omega):
            return epsilon_kinetic(omega,k_val,wp,vth,vosc)

        print('k = {}'.format(k_val))

        newroot = kdutils.croots(epsilon,wr_min,wr_max,wi_min,wi_max, nx=250, ny=250)

        print(newroot)

        # keep only one root, root with largest wi
        if newroot is not None:
            kp = np.append(kp,k_val)
            rr = np.append(rr,newroot[0].real)
            ri = np.append(ri,newroot[0].imag)
            for l in range(1,len(newroot)):
                if newroot[l].imag>newroot[l-1].imag:
                    rr[-1] = newroot[l].real
                    ri[-1] = newroot[l].imag

        # keep all returned roots
        # if newroot is not None:
        #     for l in range(len(newroot)):
        #         kp = np.append(kp,k_val)
        #         rr = np.append(rr,newroot[l].real)
        #         ri = np.append(ri,newroot[l].imag)

        print('')

    # wr vs k scatterplot with wi colorbar at fixed wpe
    kwargs_ = {'norm':matplotlib.colors.SymLogNorm(1e-5), \
               'cmap':'coolwarm_r', \
               'vmin':-np.max(np.abs(ri)),'vmax':np.max(np.abs(ri)) }

    bg = [ bohm_gross(k,r_wp,vth) for k in k_arr ]
    ld = [ landau_damping(k,wp,vth) for k in k_arr ]

    plt.figure()
    plt.plot( k_arr, bg, label='bohm gross')
    plt.plot( k_arr, ld, label='landau damping' )
    plt.scatter(kp, rr, c=ri, **kwargs_)
    plt.colorbar(label='$w_{{imag}} [{}]$'.format(wnorm))
    plt.title('vosc={:.3f}, vth={:.3f}, 1/wp={:.3f}, lambdad={:.3f}'.format(vosc,vth,r_wp,vth*r_wp))
    plt.ylabel('$\omega_{{real}} [{}]$'.format(wnorm))
    plt.xlabel('$k/k_0$')
    plt.legend(loc=0)
    plt.savefig('test_croot_1d.png')
    plt.close('all')


def wi_bohm_gross():

    # try to plot omega_imag in omega_real vs k space
    # Normalization is k0=1 c=1

    n = 25
    k_extent  = [.2,1.8]
    wr_extent = [.2,1.5]#[0.3,0.6]#[0.9,1.1]
    k_arr = np.linspace(k_extent[0], k_extent[1], num=n)
    wr_arr = np.linspace(wr_extent[0], wr_extent[1],num=n)
    r_wp_arr = np.array([2.7])

    wp = 1.0/r_wp_arr[0]
    vth = np.sqrt(3000/511000)
    vosc = 0.1

    bohm_gross = [np.sqrt((wp**2 + 3*vth**2*k**2)) for k in k_arr]

    trial_arr = np.empty([np.size(wr_arr,0),np.size(k_arr,0)])
    for i in range(np.size(wr_arr,0)):
        for j in range(np.size(k_arr,0)):
            trial_arr[i,j] = 0

    seedr,seedi = w_rwpe_vs_k(epsilon_fluid, k_arr, r_wp_arr, trial_arr, vth, vosc)

    wi_vals,wi_error = wi_k_vs_wr(epsilon_fluid_complex, k_arr, wr_arr, trial_arr, wp, vth, vosc)

    iextent = [k_extent[0],k_extent[1],wr_extent[0],wr_extent[1]]
    icmap = 'viridis'
    ikwargs = {'origin':'lower','extent':iextent, \
            'cmap':icmap,'norm':matplotlib.colors.LogNorm()}#,'vmax':1e-28,'vmin':1e-40}
            # 'cmap':icmap}#,'norm':matplotlib.colors.LogNorm(),'vmax':1e-20,'vmin':1e-40}

    skwargs = {'cmap':'copper'}#,norm=matplotlib.colors.LogNorm())

    print('wr:')
    print(seedr)
    print('wi:')
    print(seedi)
    print('')

    # for i in range(np.size(wr_arr,0)):
    #     for j in range(np.size(k_arr,0)):
    #         if wi_vals[i,j]!=0:
    #             wi_vals[i,j] = np.log(np.abs(wi_vals[i,j]))

    # plot wi
    plt.figure(figsize=[12,10])
    # plt.plot( k_arr, bohm_gross )
    plt.imshow(np.abs(wi_vals),**ikwargs)
    # plt.colorbar(label='$c^{-1}k_0^{-1}$')
    # plt.scatter(k_arr, seedr, c=np.abs(seedi), **skwargs)
    plt.colorbar()
    # plt.gca().set_aspect('auto')
    plt.xlabel('$k/k_0$')
    plt.ylabel('$Real(\omega)/\omega_0$')
    plt.title('wp={}, vosc={}, vth={}, kld={}'.format(wp,vosc,vth,2*vth/wp))
    plt.savefig('wi_bohm_gross.png')

    # plot wi_error
    plt.figure(figsize=[10,10])
    plt.plot( k_arr, bohm_gross )
    plt.imshow(wi_error,**{'origin':'lower', 'extent':iextent, 'cmap':'bwr','vmin':-.5,'vmax':.5}) #'norm':matplotlib.colors.LogNorm(),\
    plt.colorbar()
    plt.gca().set_aspect('auto')
    plt.xlabel('$k/k_0$')
    plt.ylabel('$Real(\omega)/\omega_0$')
    plt.title('wp={}, vosc={}, vth={}, kld={}'.format(wp,vosc,vth,2*vth/wp))
    plt.savefig('wi_bohm_grosserror.png')


def test_eps_complex():

    # test epsilon_fluid_complex/epsilon_cmult by comparing to epsilon_fluid used to make fig2a

    n = 25
    kextent = [.2,1.8]
    r_om_extent = [1.6,3.0]
    k_array = np.linspace(kextent[0], kextent[1], num=n)
    r_om_array = np.linspace(r_om_extent[0], r_om_extent[1],num=n)

    vth = 0.1
    vosc = 0.2

    # first get solution using original fig2a routines
    trial_arr = np.empty([np.size(k_array,0),np.size(r_om_array,0)])
    for i in range(np.size(k_array,0)):
        for j in range(np.size(r_om_array,0)):
            trial_arr[i,j] = 0

    [rr,ri] = w_rwpe_vs_k(epsilon_kinetic, k_array, r_om_array, trial_arr, vth, vosc)

    # next use those values to test complex epsilon
    ri2 = wi_rwpe_vs_k_complex(epsilon_kinetic_complex, k_array, r_om_array, rr, trial_arr, vth, vosc)

    extent_ = [r_om_extent[0],r_om_extent[1],kextent[0],kextent[1]]
    cmap_ = 'viridis'
    kwargs_ = {'origin':'lower', 'norm':matplotlib.colors.LogNorm(),'extent':extent_, \
               'cmap':cmap_,'vmin':1e-3}

    plt.figure(figsize=[10,10])
    plt.imshow(np.abs(ri2),**kwargs_)
    plt.colorbar()
    plt.gca().set_aspect('auto')
    plt.xlabel('$k_0 c / w_{pe}$')
    plt.ylabel('$k/k_0$')
    plt.title('vosc={}, vth={}'.format(vosc,vth))
    plt.savefig('test_eps_complex.png')


def epscomplex_vs_epscmult():
    # figure out what the hell is the difference between these two complex epsilons
    n = 25
    kextent = [.2,1.8]
    r_om_extent = [1.6,3.0]
    k_array = np.linspace(kextent[0], kextent[1], num=n)
    r_om_array = np.linspace(r_om_extent[0], r_om_extent[1],num=n)

    vth = 0.1
    vosc = 0.3

    e1 = np.empty([np.size(k_array,0),np.size(r_om_array,0)])
    e2 = np.empty([np.size(k_array,0),np.size(r_om_array,0)])
    for i in range(np.size(k_array,0)):
        for j in range(np.size(r_om_array,0)):
            wp = 1/r_om_array[j]
            k = k_array[i]
            e1[i,j] = epsilon_fluid_complex(0,0,k,wp,vth,vosc)
            e2[i,j] = epsilon_cmult(0,0,k,wp,vth,vosc)

    plt.figure()
    plt.imshow(e1-e2,vmin=-1e-20,vmax=1e-20)
    plt.colorbar()
    plt.show()


def test_cmult():
    a = np.random.randint(0,9) + np.random.randint(0,9)*1j
    aa = [a.real,a.imag]
    b = np.random.randint(0,9) + np.random.randint(0,9)*1j
    bb = [b.real,b.imag]
    print('a',a,'b',b)
    print('a+b',a+b,np.add(aa,bb))
    print('a*b',a*b,cmult(aa,bb))
    print('const*a',5*a,np.multiply(5,aa))
    print('|a|',np.abs(a), np.sqrt(aa[0]**2+aa[1]**2))


def lim_vth_to_0():
    # func to compare kinetic and fluid epsilon in the limit vth goes to 0
    for vth in np.linspace(.001,.1,15):
        print('{:.3f}'.format(vth))
        fig2a(n=50,vth=vth,epsilon=epsilon_fluid,r_wp_ext=[1,5],k_ext=[.2,3])

    ## put this in func fig2a()
    [rr,rik] = w_rwpe_vs_k(epsilon_kinetic, k_arr, r_wp_arr, trial_arr, vth, vosc)

    [rr,rif] = w_rwpe_vs_k(epsilon_fluid, k_arr, r_wp_arr, trial_arr, vth, vosc)
    rif = -np.abs(rif)

    rid = np.abs(rik) + rif

    kwargs_ = {'norm':matplotlib.colors.SymLogNorm(1e-5), 'extent':extent_, 'cmap':cmap_, \
        'vmin':-np.max(np.abs(rid)),'vmax':np.max(np.abs(rid)) }
    imshow_w_contours(rid,vosc,vth,save='fig2a_{}_imag_vth{:.3f}.png'.format('diff',vth),\
        **kwargs_)

    kwargs_ = {'norm':matplotlib.colors.SymLogNorm(1e-5), 'extent':extent_, 'cmap':cmap_, \
        'vmin':-np.max(np.abs(rif)),'vmax':np.max(np.abs(rif)) }
    imshow_w_contours(rif,vosc,vth,save='fig2a_{}_imag_vth{:.3f}.png'.format('fluid',vth),\
        **kwargs_)

    kwargs_ = {'norm':matplotlib.colors.SymLogNorm(1e-5), 'extent':extent_, 'cmap':cmap_, \
        'vmin':-np.max(np.abs(rik)),'vmax':np.max(np.abs(rik)) }
    imshow_w_contours(rik,vosc,vth,save='fig2a_{}_imag_vth{:.3f}.png'.format('kinetic',vth),\
        **kwargs_)
    ##



# scan29
# fig2a(n=25,r_wp_ext=[1.6,3],k_ext=[.4,1.8],vth=.07,vosc=.017)
# fig2a(n=25,r_wp_ext=[1.6,3],k_ext=[.4,3],vth=.07,vosc=.0)

# fig2a
# fig2a(vosc=.1)

# fig2a(n=25,r_wp_ext=[1.6,5],k_ext=[.4,5],vth=.1,vosc=.05)

# fig2a(vth=.1,vosc=.05)

# fig2a(nk=3,nw=3,vosc=.1)

fig2a(nk=4,nw=4,cpu_count=os.cpu_count())
