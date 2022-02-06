import ROOT as r
import numpy as np
import matplotlib.pyplot as plt

from FourVector import Lorentz_Boost
from Constants import EMMU, EMASS, pi, twopi, fine_structure_const
from Constants import michel_rho, michel_delta, michel_xsi, michel_eta, W_mue, x0, x0_squared


# radiative correction codes from geant4
# https://gitlab.cern.ch/geant4/geant4/-/blob/master/source/particles/management/include/G4MuonDecayChannelWithSpin.hh
# https://gitlab.cern.ch/geant4/geant4/-/blob/master/source/externals/clhep/include/CLHEP/Units/PhysicalConstants.h
def F_c(x,x0,omega):
    f_c =0;
    f_c = (5.+17.*x-34.*x*x)*(omega+np.log(x))-22.*x+34.*x*x;
    f_c = (1.-x)/(3.*x*x)*f_c;
    f_c = (6.-4.*x)*R_c(x,omega)+(6.-6.*x)*np.log(x) + f_c;
    f_c = (fine_structure_const/twopi) * (x*x-x0*x0) * f_c;

    return f_c;


def F_theta(x,x0, omega):
    f_theta=0;

    f_theta = (1.+x+34*x*x)*(omega+np.log(x))+3.-7.*x-32.*x*x;
    f_theta = f_theta + ((4.*(1.-x)*(1.-x))/x)*np.log(1.-x);
    f_theta = (1.-x)/(3.*x*x) * f_theta;
    f_theta = (2.-4.*x)*R_c(x,omega)+(2.-6.*x)*np.log(x)-f_theta;
    f_theta = (fine_structure_const/twopi) * (x*x-x0*x0) * f_theta;

    return f_theta;


def R_c(x,omega):
    n_max = int(100.*x);

    if(n_max<10):n_max=10;

    L2 = 0.0;

    for n in range(n_max+1)[1:]:
        L2 += pow(x,n)/(n*n);

    r_c=0.0;

    r_c = 2.*L2-(pi*pi/3.)-2.;
    r_c = r_c + omega * (1.5+2.*np.log((1.-x)/x));
    r_c = r_c - np.log(x)*(2.*np.log(x)-1.);
    r_c = r_c + (3.*np.log(x)-1.-1./x)*np.log(1.-x);

    return r_c;


def GenerateMichelKinematics():
    """
    this is the code adapted from geant4 
      // ***************************************************
      //     x0 <= x <= 1.   and   -1 <= y <= 1
      //
      //     F(x,y) = f(x)*g(x,y);   g(x,y) = 1.+g(x)*y
      // ***************************************************

      // ***** sampling F(x,y) directly (brute force) *****

    Generate kinematic variables of positrons from muon decay (in the muon rest frame)

    Parameters
    ---------------
    None, this function was designed to run once
    
    Returns
    ---------------
    x 
        the ratio of positron energy to its maximum decay energy (52.8 MeV)
    ctheta 
        the cosine of polar angle (theta, relative to z-axis) of ejected positron
    """

    # initialize variables during michel sampling
    MAX_LOOP = 1000 # maximum number of trials (before giving up the guess)
    count = 0
    rndm =0
    x =0
    ctheta =0
    FG = 0
    FG_max = 2.00

    while (count<MAX_LOOP):
        rndm = np.random.uniform(0,1);

        x = x0 + rndm*(1.-x0); # throw out a guess for x

        x_squared = x*x;

        F_IS=0; F_AS=0; G_IS=0; G_AS=0

        F_IS = 1./6.*(-2.*x_squared+3.*x-x0_squared);
        F_AS = 1./6.*np.sqrt(x_squared-x0_squared)*(2.*x-2.+np.sqrt(1.-x0_squared));

        G_IS = 2./9.*(michel_rho-0.75)*(4.*x_squared-3.*x-x0_squared);
        G_IS = G_IS + michel_eta*(1.-x)*x0;

        G_AS = 3.*(michel_xsi-1.)*(1.-x);
        G_AS = G_AS+2.*(michel_xsi*michel_delta-0.75)*(4.*x-4.+np.sqrt(1.-x0_squared));
        G_AS = 1./9.*np.sqrt(x_squared-x0_squared)*G_AS;

        F_IS = F_IS + G_IS;
        F_AS = F_AS + G_AS;

    # / *** Radiative Corrections ***
        omega =  np.log(EMMU/EMASS);
        R_IS = F_c(x,x0,omega);

        F = 6.*F_IS + R_IS/np.sqrt(x_squared-x0_squared);

    # // *** Radiative Corrections ***

        R_AS = F_theta(x,x0,omega);

        rndm = np.random.uniform(0,1);

        ctheta = 2.*rndm-1.; # throw out a guess for theta (cosine of angle between spin and momentum of positron, range from -1 to 1)

        G = 6.*F_AS - R_AS/np.sqrt(x_squared-x0_squared);


    # combine separate parts

        FG = np.sqrt(x_squared-x0_squared)*F*(1.+(G/F)*ctheta);

        if(FG>FG_max):
            Print("JustWarning, Problem in Muon Decay: FG > FG_max");
            FG_max = FG;

        rndm = np.random.uniform(0,1); # throw out a guess for THOR
        count +=1

        if (FG >= rndm*FG_max): # points ?? above ?? the surface of distribution get sampled ?????????????
            return x, ctheta
            break




def GeneratePositron_MRF(x,ctheta,theta_s):
    """
    generate positrons energy, PX, PY, PZ in muon rest frame
    Parameters
    ---------------
    x 
        the ratio of positron energy to its maximum decay energy (52.8 MeV)

    ctheta 
        the cosine of polar angle (theta, relative to z-axis) of ejected positron
    
    theta_s
        spin precession angle of the muon, relative to its momentum
    
    Returns
    ---------------
    energy, px, py, pz, stheta_s, muDecayPolY, ctheta_s
    """
    # https://gitlab.cern.ch/geant4/geant4/-/blob/master/source/externals/clhep/include/CLHEP/Vector/LorentzVector.h
    # get michel x and cos(theta) from the argument
    energy = x*W_mue
    if(energy < EMASS):energy = EMASS;

    stheta = np.sqrt(1.-ctheta*ctheta);
    
    three_momentum = np.sqrt(energy*energy - EMASS*EMASS);
    
    # generate a random angle phi to fix muon momentum in x,y,z
    rndm = np.random.uniform(0,1) 
    phi = twopi * rndm;
    cphi = np.cos(phi);
    sphi = np.sin(phi);
    
    # momentum of the decay positron with respect to the muon spin
    px0 = stheta*cphi*three_momentum ;
    py0 = stheta*sphi*three_momentum ;
    pz0 = ctheta*three_momentum ;
    
    # calculate spin precession based on input theta_s
    ctheta_s = np.cos(theta_s); # sin(theta_s) # muDecayPolZ i not yet understand, but polarization involve counting, so it is lorentz transformed into itself
    stheta_s = np.sin(theta_s); # cos(theta_s) # muDecayPolX ref: https://arxiv.org/pdf/hep-ph/0409166.pdf
    muDecayPolY = 0 # not yet include EDM !
    
    # rotate components px py pz according to spin precession angle theta_s
    px = ctheta_s*px0 + stheta_s*pz0
    py = py0
    pz = -stheta_s*px0 + ctheta_s*pz0
    
    return energy, px, py, pz, stheta_s, muDecayPolY, ctheta_s # stheta_s is muDecayPolX     # ctheta_s is muDecayPolZ




def Calculate_Phase(y1, x1, y2, x2):
    """
    Compute the angle difference between vector1 and vector2 given their x,y components.
    Parameters: y1, x1, y2, x2
    ---------------
    note: to calculate g2phase angle, I use: muDecayPX,muDecayPZ,muDecayPolX,muDecayPolZ

    Returns
    ---------------
    theta
    """

    theta1 = np.arctan2(y1,x1);  # angle of vector 1, muDecayP
    theta2 = np.arctan2(y2,x2);  # angle of vector 2, muDecayPol
    theta = theta1 - theta2;

    if(theta<0):
        theta = theta + 2*np.pi;

    return theta;




def GeneratePositron_LAB(E_primed, px_primed, py_primed, pz_primed, theta_c):
    """
    generate positrons PosiInitE, PosiInitPX, PosiInitPY, PosiInitPZ in Lab frame according to g-2 experiment setting
    Parameters
    ---------------
    E_primed
        positron energy in muon rest frame
    
    px_primed, py_primed, pz_primed
        component of positron momentum in muon rest frame

    theta_c
        cyclotron angle in the x-z plane, to fix the velocity components of the moving muon frame 
    
    Returns
    ---------------
    PosiInitE, PosiInitPX, PosiInitPY, PosiInitPZ, muDecayPX, muDecayPY, muDecayPZ
    """
    # fixed parameters, lorentz boost parameter (in gm2 experiment)
    p_mu_magic = 3.1 # GeV/c (PRL)
    gamma_mu = 29.3 # (PRL)
    beta_mu_squared = 1-(1/gamma_mu)*(1/gamma_mu) # check the redundancy !
    beta_mu = np.sqrt(beta_mu_squared) # calculated from gamma_mu
    gamma_tau_mu = 64.4e-6 # sec (PRD)

    # calculate cyclotron motion for muon using argument theta_c
    ctheta_c = np.cos(theta_c);
    stheta_c = np.sin(theta_c);

    # calculate the velocity components of muon frame in LAB frame
    beta_mu_y = 0 # no EDM !!!!
    beta_mu_x = beta_mu*stheta_c
    beta_mu_z = beta_mu*ctheta_c
    

    # calculate the momentum components of the muon 
    # ???? muInitPX or muDecayPX ????
    # problem: how to get momentum from beta ????
    #     muInitPY = gamma_mu*EMMU*beta_mu_y 
    muDecayPY = 0 # no EDM !!!!
    muDecayPX = gamma_mu*EMMU*beta_mu_x
    muDecayPZ = gamma_mu*EMMU*beta_mu_z

    # lorentz boost
    PosiInitE,PosiInitPX,PosiInitPY,PosiInitPZ = Lorentz_Boost(E_primed,px_primed,py_primed,pz_primed,gamma_mu,beta_mu_x,0,beta_mu_z,inverse=True) 
    
    return PosiInitE, PosiInitPX, PosiInitPY, PosiInitPZ, muDecayPX, muDecayPY, muDecayPZ






