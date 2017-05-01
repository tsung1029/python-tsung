import numpy as np

import matplotlib.pyplot as plt

import warnings

#

# This script shows an 1D laser antenna propagating at locations near the plane

# of best focus. The propagation is modeled by evaluating the diffraction

# integral.

#

# normalized quantities:

# # distance from focal plane

# Z  = z/z0 = z / (\lambda * f^2 / d^2)

# F  = f/z0

# # transverse coordinate on the focal plane

# X  = x/x0 = x / (\lambda * f / d)

# # transverse coordinate on aperture plane

# X' = x'/d

# T  = t/t0 = t / (\lambda * f^2 / d^2 / c)

# ==============================================================================



# input starts

# laser wave length, in micron

lam = 0.351

# focal length, in meter

f = 7.7

# aperture of the focal lens, in meter

d = f / 8.0

# half length (unit is z0) of the simulation box in propagation direction

hz = 22.0

# total number of beamlets

m = 25

# transverse size (unit is x0) of the simulation box

lx = 30

# laser bandwidth due to phase modulation, defined as FWHM, in laser frequency

# 117 --> 1THz
# lsr_bw = 0.00117
lsr_bw = 0.00

# amplitude of phase modulation, should be no smaller than pi

pm_am = np.pi

# number of color cycles for the highest fm frequency

ncc = 6.4

# number of fm frequencies

multi_fm = 3

# number of grid points in propagation direction

nz = 128 * 2

# number of grid points in transverse direction

nx = 25

# number of time steps for the movie

tnmax = 100

# type of phase modulation, currently support:

#   "AR": ISI with phase modeled with AR(1) process

#   "GS": ISI with Gaussian PSD phase modulation

#   "FM":  SSD with sinusoidal phase modulation

#   "RPM": SSD with random phase modulation

phmod_type = "fm"

# simple moving average for AR process to get close to Gaussian profile (experimental)

if_sma = False

# ------------------------------------------------------------------------------

# input ends



# normalization factors

lam *= 1e-6

# normalize t0 to 1/\omega

t0 = 2.0 * np.pi * (f * f) / (d * d)

x0 = lam * f / d

z0 = lam * (f * f) / (d * d)

#

# time between two positions in propagation direction. the unit is 1/\omega_0

tu = 2.0 * hz * t0 / nz

F = f / z0

Z = np.arange(-hz, hz, (2.0*hz)/nz)

X = np.arange(-0.5*lx, 0.5*lx, float(lx)/nx)

Xp = np.arange(-0.5, 0.5, 1.0/m)

if nx <= 1:

    X = 0.5 * lx * np.random.normal()

plt.rc('axes', titlesize=24)

plt.rc('axes', labelsize=24)

plt.rc('xtick', labelsize=18)

plt.rc('ytick', labelsize=18)





def plot_2d_xz(fld):

    """ save the absolute values of fld as a png file

    """

    plt.figure(figsize=(16, 9))

    plt.imshow(abs(fld), extent=[-hz * z0 * 1e6, hz * z0 * 1e6,

                                 -0.5 * lx * x0 * 1e6, 0.5 * lx * x0 * 1e6],

               aspect='auto', vmin=0, vmax=3)

    plt.title('Envelope of E field ('+phmod_type +

              ', t='+"{0:.3f}".format(tn*tu*1.86e-4)+'ps)')

    plt.xlabel('$z (\mu m)$')

    plt.ylabel('$x (\mu m)$')

    plt.colorbar()

    plt.savefig('test'+"{0:0>4}".format(tn)+'.png')

    plt.close()





def plot_1d_z(fld):

    """ save the absolute values of fld as a png file

    """

    plt.figure(figsize=(10.5, 4.5))

    plt.plot(Z * z0 * 1e6, abs(np.squeeze(fld)))

    plt.gca().set_ylim([0, 3])

    plt.title(phmod_type+', t=' + "{0:.3f}".format(tn * tu * 1.86e-4) + 'ps')

    plt.xlabel('$z (\mu m)$')

    plt.ylabel('Envelope')

    plt.savefig('test' + "{0:0>4}".format(tn) + '.png')

    plt.close()





func_dict = {

    True: plot_2d_xz,

    False: plot_1d_z,

}

save_plot = func_dict[nx > 1]





# ISI (order 1 autoregressive)

def ar1(b, sigma, pha, num=m):

    return b * pha + sigma * np.random.normal(size=num)





# SMA

def sma2d(pha):

    ret = np.cumsum(pha, axis=-1)

    ret[:, smn:] = ret[:, smn:] - ret[:, :-smn]

    return ret / smn



# static random phase

rph = np.random.uniform(-np.pi, np.pi, m)



rph_t = np.zeros((m, nz))

efld = np.zeros((np.size(X), nz))

ph_pro = np.zeros((nx, nz))

ft_co = np.zeros((m, nz))



# caculate the constant factors of the diffraction integral

fac = np.pi * d * d / (lam * f)

ph_pro_st = np.mod(2 * np.pi * F / lam, 2 * np.pi)

amp = F / (F - Z) / np.sqrt(m)

phi = np.zeros((m, nz))

phmod_type = phmod_type.upper()

ph_shift = np.zeros((m, nz))

tn_all = tnmax + nz

omega = np.fft.fftshift(np.fft.fftfreq(tn_all, d=tu))

if tu * lsr_bw > 0 and if_sma:

    smn = int(1.0 / (tu * lsr_bw))

else:

    smn = 1

rph_buff = np.zeros((m, nz + smn - 1))



# # calculate the first frame

# generate the phase array for the whole frame

if phmod_type == 'FM':

    pm_bw = 0.5 * lsr_bw / pm_am / multi_fm

    if pm_bw > 0:

        s = 2 * np.pi * ncc

    else:

        s = 0

    ss = np.arange(1, multi_fm + 1) * s / multi_fm

    fm_am = pm_am / np.sqrt(multi_fm)

    for si in range(1, multi_fm + 1):

        for zi in range(0, nz):

            rph_t[:, nz - zi - 1] += (fm_am * np.sin(pm_bw * si * zi * tu -

                                                     ss[si - 1] * Xp))

    for zi in range(0, nz):

        rph_t[:, zi] += rph

else:

    if (tu * lsr_bw) > (np.pi * 0.05):

        warnings.warn('Speckle pattern is changing too fast. '

                      'Decrease the value of (pm_bw*pm_am) or '

                      'increase the value of nz')

    if phmod_type == 'AR':

        pm_bw = 0.5 * lsr_bw / (pm_am * pm_am)

        arcoeff1 = np.exp(- tu * pm_bw)

        arcoeff2 = np.sqrt(1 - arcoeff1 * arcoeff1) * pm_am

        rph_buff[:, nz + smn - 2] = ar1(arcoeff1, arcoeff2, rph)

        for zi in range(1, nz + smn - 1):

            rph_buff[:, nz + smn - zi - 2] = ar1(arcoeff1, arcoeff2,

                                                 rph_buff[:, nz + smn - zi - 1])

        rph_t[:, nz - 1] = np.mean(rph_buff[:, nz-1:nz+smn-1], axis=-1)

        for zi in range(1, nz):

            rph_t[:, nz - zi - 1] = rph_t[:, nz - zi] + (rph_buff[:, nz - zi - 1] -

                                                         rph_buff[:, nz + smn - zi - 1]) / smn

    if phmod_type == 'GS':

        pm_bw = 0.5 * lsr_bw / pm_am

        spec_ph = np.zeros((m, tn_all), dtype=complex)

        if pm_bw > 0:

            # numpy fft is defined in terms of frequency not angular frequency

            psd = np.exp(

                -np.log(2) * 0.5 * np.square(omega / pm_bw * 2 * np.pi))

            psd *= np.sqrt(2 * tn_all) / np.sqrt(np.mean(np.square(psd))) * pm_am

            for mi in range(0, m):

                mth_ph = np.random.normal(scale=np.pi, size=tn_all)

                spec_ph[mi, :] = psd * (np.cos(mth_ph) + 1j * np.sin(mth_ph))

        phase_all = np.real(np.fft.ifft(np.fft.ifftshift(spec_ph, axes=-1)))

        rph_t = phase_all[:, tn_all-nz:tn_all]

    if phmod_type == 'RPM':

        pm_bw = 0.5 * lsr_bw / (pm_am * pm_am)

        arcoeff1 = np.exp(- tu * pm_bw)

        arcoeff2 = np.sqrt(1 - arcoeff1 * arcoeff1) * pm_am

        s = 2 * np.pi * ncc

        x_t = np.arange(0, s, s / m)

        sz_queue = int(np.ceil(s / (pm_bw * tu)))

        if sz_queue < m:

            warnings.warn('Phase bandwidth too large. Reduce pm_bw or increase nz')

        ph_xt = np.arange(0, s, s / sz_queue)

        ph_seq = np.zeros(sz_queue)

        ph_seq[0] = np.random.uniform(-np.pi, np.pi, 1)

        for szq in range(1, sz_queue):

            ph_seq[szq] = ar1(arcoeff1, arcoeff2, ph_seq[szq - 1], num=1)

        for zi in range(0, nz):

            ph_seq = np.roll(ph_seq, -1)

            ph_seq[sz_queue - 1] = ar1(arcoeff1, arcoeff2,

                                       ph_seq[sz_queue - 2], num=1)

            rph_t[:, nz - zi - 1] = np.interp(x_t, ph_xt, ph_seq) + rph

# more factors for the diffraction integral

for zi in range(0, nz):

    ph_shift[:, zi] = fac * Z[zi] * Xp * Xp / (F - Z[zi])

    ph_pro[:, zi] = ph_pro_st + np.square(X) * np.pi / (F - Z[zi])

    ft_co[:, zi] = 2 * np.pi * d * d / (lam * f) * Xp / (F - Z[zi])



for tn in range(0, tnmax):

    efld.fill(0)

    # calculate the diffraction integral

    for zi in range(0, nz):

        for i in range(0, m):

            efld[:, zi] += np.sin(rph_t[i, zi] + ph_pro[:, zi] -

                                  ft_co[i, zi] * X + ph_shift[i, zi])

        # efld[:, zi] *= (amp[zi] * np.sinc(np.pi * X / m))

        efld[:, zi] *= amp[zi]

    # shift the phase array by one element (advancing in time), generate new phases

    rph_t = np.roll(rph_t, 1, axis=1)

    if phmod_type == 'AR':

        rph_buff = np.roll(rph_buff, 1, axis=1)

        rph_buff[:, 0] = ar1(arcoeff1, arcoeff2, rph_buff[:, 1])

        rph_t[:, 0] = rph_t[:, 1] + (rph_buff[:, 1] - rph_buff[:, 2]) / smn

    elif phmod_type == 'FM':

        rph_t[:, 0] = fm_am * np.sin(pm_bw * (nz + tn) * tu - ss[0] * Xp) + rph

        for si in range(2, multi_fm + 1):

            rph_t[:, 0] += fm_am * np.sin(pm_bw * si * (nz + tn) * tu -

                                          ss[si - 1] * Xp)

    elif phmod_type == 'GS':

        rph_t[:, 0] = phase_all[:, tn_all-nz-tn-1]

    else:  # 'RPM'

        ph_seq = np.roll(ph_seq, -1)

        ph_seq[sz_queue - 1] = ar1(arcoeff1, arcoeff2,

                                   ph_seq[sz_queue - 2], num=1)

        rph_t[:, 0] = np.interp(x_t, ph_xt, ph_seq) + rph

    save_plot(efld)

    # if tn % nz == 0:

    #     omega = np.fft.fftshift(np.fft.fftfreq(np.size(efld, axis=-1), d=tu))

    #     spec_am = np.sum(np.square(np.abs(np.fft.fftshift(np.fft.fft(

    #         efld, axis=-1)))), axis=0)

    #     plt.plot(omega, spec_am, label=phmod_type)

    #     plt.show()





