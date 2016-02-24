import numpy as np

cpdef double integrand(double u, double H, double D, double Dy, double N_a, double N_m, double sigma_a, double sigma_R, double c, double A, double z, double az):
    cdef double beta = (az/180 * np.pi + np.arcsin(Dy/D) - np.pi/2)
    cdef double h = u*np.cos(z/180 * np.pi) + A
    cdef double K = N_a*sigma_a / (11.11*N_m*sigma_R*np.exp(-c*H))
    cdef double a = 0.657 + 0.059*K
    cdef double l = np.sqrt(D**2 + A**2)
    cdef double theta = np.arccos( (D*np.sin(z/180 * np.pi)*np.cos(beta) - A*np.cos(z/180 * np.pi)) /l )
    cdef double s = np.sqrt(u**2 + l**2 - 2.*u*l*np.cos(theta))
    cdef double psi = np.arccos(h/s)
    cdef double phi = np.arccos( (l - np.cos(theta)) /s )
    cdef double thetapsi = theta + psi
    cdef double gamma = 1/3.
    if thetapsi > np.pi:
        thetapsi = np.arccos(np.cos(thetapsi))
    thetapsi = thetapsi * 180 / np.pi
    cdef double f
    if thetapsi >= 0 and thetapsi <= 10: # degrees
        f = 7.0*np.exp(-0.2462*thetapsi)
    elif thetapsi > 10 and thetapsi <= 90:
        f = 0.9124*np.exp(-0.04245*thetapsi)
    elif thetapsi > 90 and thetapsi <= 180:
        f = 0.02
    cdef double p = (1./c)*(1 - np.exp(-c*h)) + 11.778*K*(1./a)*(1 - np.exp(-a*h))
    cdef double EF_XQ = np.exp(-N_m*sigma_R*np.exp(-c*H)*p*1./(np.cos(psi))) # fractional reduction
    p = (c**(-1)*(np.exp(-c*A) - np.exp(-c*h))
    + 11.778*K*a**(-1)*(np.exp(-a*A) - np.exp(-a*h)))
    cdef double EF_QO = np.exp(-N_m * sigma_R * np.exp(-c*H)*p*1./(np.cos(z/180 * np.pi)))
    cdef double DS = 1. + (N_a*sigma_a* (1 - np.exp(-a*s*np.cos(psi))) )/ (a*np.cos(psi))
    + gamma*N_m*sigma_R*np.exp(-c*H)*(1. - np.exp(-c*s*np.cos(psi)))/(c*np.cos(psi))
    cdef double integrand = ( (s**(-2)) * EF_XQ * EF_QO * DS
    * (np.exp(-c*h) * 3. * (1 + np.cos(theta + psi)**2)/(16*np.pi)
    + np.exp(-a*h) * 11.11 * K * f) )
    return integrand