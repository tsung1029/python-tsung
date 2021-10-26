import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import mpmath


def plasma_dispersion(value):
    z = scipy.special.wofz(value)
    z = z * np.sqrt(np.pi)*np.complex(0,1)
    return z


def plasma_dispersion_prime(value):
    ## the line below is needed for the root finder, which uses MP (multi-precision) variables
    ## instead of real and/or complex, so the first step is to convert the variable "z" from
    ## of a type complex to a type MP
    z = plasma_dispersion(complex(value.real,value.imag))
    # z = plasma_dispersion(value)

    return -2.0*(1.0+value*z)


def epsilon_fluid(w,k,wp,vth,vosc):
    # eqn 19 forslund, fluid chi, chi_i=0, gamma terms dropped, rewritten as polynomial
    # Normalization is k0=1 c=1
    w0      = np.sqrt(1+wp*wp)
    w_minus = w - w0
    w_plus  = w + w0
    kminus  = k - 1.0
    kplus   = k + 1.0

    chi_epw = (w*w-3*k*k*vth*vth)
    d_minus = (w_minus * w_minus - wp * wp - kminus * kminus)
    d_plus  = (w_plus  * w_plus  - wp * wp - kplus  * kplus)

    epsilon = chi_epw * d_minus * d_plus \
              - wp**2 * d_minus * d_plus \
              - 1.0/2 * wp**2 * k**2 * vosc**2 * ( d_minus + d_plus )

    return epsilon


def epsilon_kinetic(w,k,wp,vth,vosc):
    # eqn 19 forslund, kinetic chi, chi_i=0, gamma terms dropped, rewritten as polynomial
    w0      = np.sqrt(1+wp*wp)
    w_minus = w - w0
    w_plus  = w + w0
    kminus  = k - 1.0
    kplus   = k + 1.0

    z_prime = plasma_dispersion_prime( w / (np.sqrt(2) * k * vth) )
    d_minus = (w_minus * w_minus - wp * wp - kminus * kminus)
    d_plus  = (w_plus  * w_plus  - wp * wp - kplus  * kplus)

    # # polynomial
    # epsilon = 2 * k**2 * vth**2 * d_plus * d_minus \
    #           - wp**2 * d_plus * d_minus * z_prime \
    #           - 1.0/2 * wp**2 * k**2 * vosc**2 * ( d_minus + d_plus ) * z_prime

    # # works better when you want vosc = 0, ie landau damping
    # epsilon = 2 * k**2 * vth**2 - wp**2 * z_prime
    
    # also works well for landau damping but contains all the lpi stuff
    epsilon = 2 * k**2 * vth**2 \
              - wp**2 * z_prime \
              - 1.0/2 * wp**2 * k**2 * vosc**2 * ( d_minus + d_plus ) * z_prime / (d_plus * d_minus)

    return epsilon


def epsilon_landau_damping(w,k,wp,vth):
    w0      = np.sqrt(1+wp*wp)
    w_minus = w - w0
    w_plus  = w + w0
    kminus  = k - 1.0
    kplus   = k + 1.0

    z_prime = plasma_dispersion_prime( w / (np.sqrt(2) * k * vth) )
    d_minus = (w_minus * w_minus - wp * wp - kminus * kminus)
    d_plus  = (w_plus  * w_plus  - wp * wp - kplus  * kplus)

    epsilon = 2 * k**2 * vth**2 - wp**2 * z_prime

    return epsilon



def wi_srs2(k, v0, wp_e, wr, w0):
    # Im(omega) for srs "farther above threshold where dampin unimportant" (forslund eq 32)
    # to be used as guess for root finder
    wi = - k*v0 / ( 2*np.sqrt(2) ) * wp_e / np.sqrt( wr*(w0-wr) )
    return wi


def epsilon_fluid_complex(wi,wr,k,wp,vth,vosc):
    # eqn 19 forslund, fluid chi, chi_i=0, gamma terms dropped, rewritten as polynomial
    # should be identicle to epsilon_fluid() just with wi and wr separated in arg list
    w      = wr + 1j*wi
    w0     = np.sqrt(1+wp*wp)
    w_mnus = w - w0
    w_plus = w + w0
    k_mnus = k - 1.0
    k_plus = k + 1.0

    chi_epw = (w*w-3*k*k*vth*vth)
    d_minus = np.add(w_mnus * w_mnus, -1 * wp * wp - k_mnus * k_mnus)
    d_plus  = np.add(w_plus * w_plus, -1 * wp * wp - k_plus  * k_plus)

    term1 = np.multiply( chi_epw, np.multiply(d_minus, d_plus) )
    term2 = np.multiply( -wp**2, np.multiply(d_minus, d_plus) )
    term3 = np.multiply( -1.0/2 * wp**2 * k**2 * vosc**2, np.multiply(chi_epw, np.add(d_minus,d_plus)) )

    epsilon = np.add(term1, np.add(term2, term3))

    # return (epsilon.real**2+epsilon.imag**2) # what I was doing when it wasn't working. I guess the solver has an easier time when you return a complex number...
    # return epsilon**2 # total garbage but finds some more roots, not all of which are legit. Kind of like the cmult version
    return epsilon


def epsilon_kinetic_complex(wi,wr,k,wp,vth,vosc):
    # eqn 19 forslund, kinetic chi, chi_i=0, gamma terms dropped, rewritten as polynomial
    # should be identicle to epsilon_kinetic() just with wi and wr separated in arg list
    w       = wr + 1j*wi
    w0      = np.sqrt(1+wp*wp)
    w_minus = w - w0
    w_plus  = w + w0
    kminus  = k - 1.0
    kplus   = k + 1.0

    z_prime = plasma_dispersion_prime( w / (np.sqrt(2) * k * vth) )
    d_minus = (w_minus * w_minus - wp * wp - kminus * kminus)
    d_plus  = (w_plus  * w_plus  - wp * wp - kplus  * kplus)

    epsilon = 2 * k**2 * vth**2 * d_plus * d_minus \
              - wp**2 * d_plus * d_minus * z_prime \
              - 1.0/2 * wp**2 * k**2 * vosc**2 * ( d_minus + d_plus ) * z_prime
    
    return epsilon


def epsilon_fluid_cmult(wi,wr,k,wp,vth,vosc):
    # eqn 19 forslund, fluid chi, chi_i=0, gamma terms dropped, rewritten as polynomial

    # complex multiplication
    def cmult(a,b):
        c = [ np.subtract(np.multiply(a[0],b[0]),np.multiply(a[1],b[1])), \
              np.add(np.multiply(a[0],b[1]),np.multiply(a[1],b[0])) ]
        return c

    w0     = np.sqrt(1+wp*wp)
    w_mnus = [wr-w0, wi]
    w_plus = [wr+w0, wi]
    k_mnus = k - 1.0
    k_plus = k + 1.0

    chi_epw = np.add( cmult([wr,wi], [wr,wi]), [-3*k*k*vth*vth, 0] )
    d_minus  = np.add( cmult(w_mnus, w_mnus), [-1*wp*wp - k_mnus*k_mnus,0] )
    d_plus  = np.add( cmult(w_plus, w_plus), [-1*wp*wp - k_plus*k_plus,0] )

    term1 = cmult(chi_epw, cmult(d_minus, d_plus))
    term2 = np.multiply(-1*wp**2,cmult(d_minus,d_plus))
    term3 = np.multiply(-1./2*wp**2*k**2*vosc**2, cmult(chi_epw, np.add(d_minus,d_plus)))

    epsilon = np.add( term1, np.add( term2, term3 ) )
    
    # return (np.abs(epsilon[0])**2 + np.abs(epsilon[1])**2) # crashes in the same way as epsilon_complex (the version where I had return (epsilon.real**2+epsilon.imag**2)). Save as secret weapon.
    # return np.abs(np.sum(epsilon)) # also crashes like the non working version of epsilon complex
    # return np.sum(epsilon) # should be the most legit of all of them, but gives totally whack results
    return (epsilon[0]**2 + epsilon[1]**2) # "works" but is totally incorrect


def chi(omega,omegap,k,vth):
    # fluid chi
    chi = - omegap*omegap/(omega*omega - 3 * k * k * vth*vth)
    return(chi)


def d_minus(omega,k,omegap,vth):
    # frank
    #! here we assume k0=1, c=1
    k0=np.sqrt(1-omegap*omegap)
    kminus = 1 - k
    omega0=np.sqrt(omegap*omegap+3*k*k*vth*vth)+np.sqrt(omegap*omegap+kminus*kminus)
    d_minus=1./((omega-omega0)**2-omegap*omegap-kminus*kminus)


def epsilon_frank1(omega,k,omegap,vth,vosc):
    # Frank modifications on forslund eqn 19 (maybe like kruer)
    # this produced nice plots but the equation is kind of not right
    omega0=1.0/np.sqrt(1-omegap*omegap)
    omega_minus = omega0-omega
    kminus=k-1.0
    chi_ = chi(omega,omegap,k,vth)
    # letting epsilon0 = 0:
    epsilon = -1.0 * chi_ * (k*k*vosc*vosc)/ \
              (omega_minus*omega_minus-omegap*omegap-kminus*kminus)
    # with epsilon0:
    # epsilon = -1.0 * chi_ * (k*k*vosc*vosc)/ \
    #           (omega_minus*omega_minus-omegap*omegap-kminus*kminus)
    #           (omega_minus*omega_minus-omegap*omegap-kminus*kminus) - 1.0 - chi_  
    return epsilon


def epsilon_frank2(omega,k,omegap,vth,vosc):
    # eqn 19 forslund, fluid chi, chi_i=0, ignore forward scattered waves, rewritten as polynomial
    # gamma terms dropped
    omega0=np.sqrt(1+omegap*omegap)
    omega_minus = omega - omega0
    kminus= k - 1.0
    chi_epw=(omega*omega-3*k*k*vth*vth)
    d_minus = (omega_minus*omega_minus - omegap*omegap - kminus*kminus)

    epsilon = chi_epw * d_minus - omegap**2 * d_minus - 1./2 * omegap**2 * k**2 * vosc**2
    
    return epsilon


def epsilon_eqn28_forslund(omega,k,omegap,vth,vosc):
    # this never worked.
    kminus = k - 1.0
    omega0 = np.sqrt(omegap**2+1.0)
    epsilon = 0.5*omegap**2*k**2*vosc**2 / \
              (omega**2 - omegap**2 - 3*k**2*vth**2) / \
              ( (omega-omega0**2)**2 - kminus**2 - omegap**2) - 1.0
    return epsilon


def zfunc(z):
    # Franks zfunc
    a = scipy.special.wofz(z)
    a *= np.sqrt(np.pi)*complex(0,1)
    return a


def zprime(z):
    # frank's zprime

    ## the line below is needed for the root finder, which uses MP (multi-precision) variables
    ## instead of real and/or complex, so the first step is to convert the variable "z" from
    ## of a type complex to a type MP
    arg= complex(z.real,z.imag)

    value= zfunc(arg)
    return(-2.0*(1.0+z*value))
