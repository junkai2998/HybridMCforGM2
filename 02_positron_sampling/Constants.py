import numpy as np

# Global constants
EMMU = 105.6 # mass of muon [MeV] (from PDG)
EMASS = 0.511 # mass of positron [MeV] 
pi = np.pi
twopi = 2*pi
fine_structure_const = 1/137


# Constants for Michel sampling 
# Michel Parameters
michel_rho   = 0.75 # Standard Model Michel rho
michel_delta = 0.75 # Standard Model Michel delta
michel_xsi   = 1.00 # Standard Model Michel xsi
michel_eta   = 0.00 # Standard Model eta


# maximum positron energy from muon decay # shared 
# gives the maximum energy of emitted positron (52.8 MeV), neglecting neutrino mass
W_mue  = (EMMU*EMMU+EMASS*EMASS)/(2.*EMMU) 
x0     =  EMASS/W_mue # maximum positron energy in terms of ratio
x0_squared = x0*x0
