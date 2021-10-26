import numpy
import cmath
mecgs = 9.1094e-28
qecgs = 4.8032e-10
ccgs = 29979000000
mypi = 3.1415926

def omegap(density):
    result=numpy.sqrt(4*mypi*qecgs*qecgs*density/mecgs)
    return result

def ncrit(lambdamic):
    omega0=ccgs*2.0*mypi/(lambdamic/10000)
    result=omega0*omega0*mecgs/(4*mypi*qecgs*qecgs)
    return result

def ve(te):
    teerg = te*1.602e-12
    result=numpy.sqrt(teerg/mecgs*3)
    return result

def vth(te):
    teerg = te*1.602e-12
    result=numpy.sqrt(teerg/mecgs)
    return result


def betawig(te, ionefour,lnmic,nppmp):
    result = numpy.sqrt(.75)*numpy.power((ve(te)/ccgs),2)/vowiggle(ionefour,lnmic,nppmp)
    return result

def vowiggle(ionefour,lambdamic,nppmp):
    result=numpy.sqrt(nppmp)*numpy.sqrt(1-nppmp)/2*vosc(ionefour,lambdamic);
    return result

def vosc(ionefour,lambdamic):
    inorm=ionefour*10**21
    result=numpy.sqrt((inorm*qecgs*qecgs*8.0*mypi)/
    ((ccgs*2.0*mypi/(lambdamic*1e-4))**2*mecgs*mecgs*ccgs))/ccgs;
    return result

def ch_hyb(tev,ionefour,lnmic,lambdamic):
    result=0.5*numpy.sqrt(0.75)*3*vth(tev)*vth(tev)/(ccgs*ccgs)*betawig(tev,ionefour,lambdamic,0.25)
    return result

def cinh_hyb(ionefour,lnmic,lambdamic):
    result=0.5*epsln(lnmic,lambdamic,0.25,1)/vowiggle(ionefour,lambdamic,0.25)**1.5
    return result

def epsln(lnmic,lambdamic,nppmp,order):
    result=nppmp**(1/order)/(2*mypi*lnmic/lambdamic)
    return result

def gamma_real_hyb(eta,tev,ionefour,lnmic,lambdamic,fac):
    vowig_int=vowiggle(ionefour,lambdamic,0.25)
    ch_int=ch_hyb(tev,ionefour,lnmic,lambdamic)
    tau = eta*eta/vowig_int
    cinh_int=cinh_hyb(ionefour,lnmic,lambdamic)

    result=vowig_int*(1.0-ch_int * tau - cinh_int/((1+tau*tau)**(0.25))*numpy.sin(mypi/4+0.5*numpy.arctan(tau)))
    return result

def cmult(tev,ionefour,lnmic,lambdamic):
    result=18.4*(ionefour*lambdamic**2)*(lnmic/lambdamic)/tev
    return result

def cmult_srs(ionefour,lnmic,lambdamic):
    result=(ionefour**0.75*lambdamic**0.5)*(lnmic)/155.0
    return result



def gamma_real(eta,tev,ionefour,lnmic,lambdamic,state):

    cmult_int=cmult(tev,ionefour,lnmic,lambdamic)
    beta_int = betawig(tev,ionefour,lambdamic,0.25)
    vo_int=vowiggle(ionefour,lambdamic,0.25)
    result=vo_int*(1.0-0.5*beta_int*beta_int*eta*eta-(2*state+1)*((2/3)**(1.5))/(cmult_int*eta*beta_int))
    return result

def envelope(t,t_delay,t_rise,t_flat,t_fall):

    if (t<t_delay):
        result=0.0
    elif (t < (t_delay+t_rise)):
        result=envelope_norm((t-t_delay)/t_rise)
    elif (t <= (t_delay+t_rise+t_flat)):
        result = 1.0
    elif (t < (t_delay+t_rise+t_flat+t_fall)):
        result = envelope_norm_rev((t-t_delay-t_rise-t_flat)/t_fall)
    else:
        result = 0.0
    

    return result


def envelope_norm(tau):
    if (tau<=1.0):
        result = 10*tau**3-15*tau**4+6*tau**5
    else:
        result = 0.0
    return result

def envelope_norm_rev(tau):
    if (tau<=1.0):
        result = envelope_norm(1.0-tau)
    else:
        result = 0.0
    return result

def kplus_para(kperp):

    epsoverfour= 0.75/4.0

    return numpy.sqrt(kperp*kperp + epsoverfour) + numpy.sqrt(epsoverfour) 

def kminus_para(kperp):

    epsoverfour = 0.75/4.0

    return numpy.sqrt(kperp*kperp + epsoverfour) - numpy.sqrt(epsoverfour)

def kplus_sq(kperp):

    epsoverfour = 0.75/4.0

    kplus =  numpy.sqrt(kperp*kperp + epsoverfour) + numpy.sqrt(epsoverfour)

    return kplus*kplus + kperp*kperp

def kminus_sq(kperp):

    epsoverfour = 0.75/4.0
    kminus =  numpy.sqrt(kperp*kperp + epsoverfour) - numpy.sqrt(epsoverfour)

    return kminus*kminus + kperp*kperp


def density_match(kperp,tev):

    mecgs = 9.1094e-28
    qecgs = 4.8032e-10
    ccgs = 29979000000
    mypi = 3.1415926

    ve=vth(tev)/ccgs

    return (0.25 * (1+numpy.sqrt(1-12*ve*ve*(kplus_sq(kperp)+kminus_sq(kperp)))))**2.0


def tpd_match_func(wp,eta,tev):

    ve=vth(tev)/ccgs
    # DEBUG
    # print(ve)
    # print(kplus_sq(eta))
    # print(kminus_sq(eta))
    # DEBUG
    return numpy.sqrt(wp*wp+3*ve*ve*kplus_sq(eta)) + numpy.sqrt(wp*wp+3*ve*ve*kminus_sq(eta)) - 1.0



def density_match_exact(kperp,tev):
    from scipy import optimize

    def test(wp):

        return tpd_match_func(wp,kperp,tev)

    test2=optimize.root_scalar(test,x0=0.50,x1=0.49,method='secant')
    # DEBUG
    # print(test2)
    # DEBUG
    return test2.root*test2.root
