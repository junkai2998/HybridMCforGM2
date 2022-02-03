def Lorentz_Boost(x_0,x_1,x_2,x_3,gamma,beta1,beta2,beta3,inverse=False):
    '''
    Implementation of general lorentz boost, from Jackson (11.98)
    x is a 4-vector, with components x_0, x_1, x_2, x_3 (minkowski notation with 0,1,2,3 represent ct,x,y,z)
    x can be in K or K' frame, the inverse parameter will take care the correct transformation
    beta1, beta2, beta3 are the components of the velocity of the moving frame K' (K and K' axes are taken to be all parallel to each other)
    
    '''

    if(inverse==True):    # invserse transformation from K' back to K (lab), beta changes sign
        beta1 = -beta1
        beta2 = -beta2
        beta3 = -beta3

    beta_squared = (gamma*gamma - 1) / (gamma*gamma)
        
    #calculate all matrix element of the transformation
    B_00 = gamma

    B_01 = -gamma*beta1

    B_02 = -gamma*beta2

    B_03 = -gamma*beta3


    B_10 = -gamma*beta1

    B_11 = 1+(gamma-1)*(beta1*beta1)/(beta_squared)

    B_12 = (gamma-1)*(beta1*beta2)/(beta_squared)

    B_13 = (gamma-1)*(beta1*beta3)/(beta_squared)


    B_20 = -gamma*beta2

    B_21 = (gamma-1)*(beta2*beta1)/(beta_squared)

    B_22 = 1+(gamma-1)*(beta2*beta2)/(beta_squared)

    B_23 = (gamma-1)*(beta2*beta3)/(beta_squared)


    B_30 = -gamma*beta3

    B_31 = (gamma-1)*(beta3*beta1)/(beta_squared)

    B_32 = (gamma-1)*(beta3*beta2)/(beta_squared)

    B_33 = 1+(gamma-1)*(beta3*beta3)/(beta_squared)


    # calculate the results
    x0_transformed = B_00*x_0 + B_01*x_1 + B_02*x_2 + B_03*x_3

    x1_transformed = B_10*x_0 + B_11*x_1 + B_12*x_2 + B_13*x_3

    x2_transformed = B_20*x_0 + B_21*x_1 + B_22*x_2 + B_23*x_3

    x3_transformed = B_30*x_0 + B_31*x_1 + B_32*x_2 + B_33*x_3

    return x0_transformed, x1_transformed, x2_transformed, x3_transformed
