
# ************************************************************
# ************************************************************
# ************************************************************
#
#         this file contains commonly used scripts 
#         for PIC's, such as shape function and 
#         smoother and compensators
#         Frank S. Tsung
#
# ************************************************************
# ************************************************************
# ************************************************************



import numpy
import cmath
mecgs = 9.1094e-28
qecgs = 4.8032e-10
ccgs = 29979000000
mypi = 3.1415926

sixth = 0.1666666667
half = 0.5
quarter = 0.25


def cubic_spline(x):

    result = 0
    if(x<=2 and x > -2):
        result = (x<=-1 and x > -2) * (2+x)**3 * sixth + (x <=0 and x > -1) * sixth * (-3*x**3 - 6*x**2 + 4 ) + (x <=1 and x > 0) * (3*x**3-6*x**2 +4)*sixth + (x <=2 and x > 1) * (2-x)**3*sixth

    return result


def ngc(x):
    result = 0
    # if(x<0.5 and x > -0.5):
    #     result = 1
    result = (np.abs(x)<0.5)
    return result

def linear_spline(x):
    result = 0
    if (np.abs(x)<1):
        result=1-np.abs(x)
    return result

def quad_spline(x):
    result = 0
    if(np.abs(x)<1.5):
        result=(np.abs(x)<0.5)*(-x**2+.75)+(np.abs(x)>=0.5)*(np.abs(x)-1.5)**2*0.5
    return result

def os_env(t):
    result = 10*t**3 - 15*t**4 +6*t**5
    return result

def r_spline(k,order):

    result = 1
    result = (2**(order+1)) * (abs(k)>=1e-6) *(np.sin(k*half)/k)**(order+1) + (np.abs(k)<1e-6) * 1.0

    return result
    
