# function for generating theoretical scaling values with only first-order derivatives.
def theoretical_scaling_order1(Dimensionality, L=5, asymptotic=False):

    # Defining variables based on problem's dimensionality
    M = Dimensionality - 1
    p = 10**(Dimensionality)
    d = Dimensionality
    m = 50**(Dimensionality-1)
    e = 20*(Dimensionality)

    # Variables for both Architectures
    T1 = 6*p*d*e 
    T2 = (L-1)*e**2
    T3 = 4*M*p*e

    # Variables for Architecture I
    T4 = 3*M*m*p*e
    T5 = 3*p*(M+2)

    # Variables for Architecture II
    T6 = 3*M*m*e
    T7 = 3*(M+2*p)

    numerator = T4 + T1 + T5*T2 + T3
    if not asymptotic:
        denominator = T6 + T1 + T7*T2 + T3
    elif asymptotic:
        denominator = T1 + T7*T2 + T3

    return numerator/denominator

# function for generating theoretical scaling values with only second-order derivatives.
def theoretical_scaling_order2(Dimensionality, L=5, asymptotic=False):

    # Defining variables based on problem's dimensionality
    M = Dimensionality - 1
    p = 10**(Dimensionality)
    d = Dimensionality
    m = 50**(Dimensionality-1)
    e = 20*(Dimensionality)

    # Variables for both Architectures
    T1 = 9*p*d*e 
    T2 = (L-1)*e**2
    T3 = 6*M*p*e

    # Variables for Architecture I
    T4 = 3*M*m*p*e
    T5 = 3*p*(M+3)

    # Variables for Architecture II
    T6 = 3*M*m*e
    T7 = 3*(M+3*p)

    numerator = T4 + T1 + T5*T2 + T3
    if not asymptotic:
        denominator = T6 + T1 + T7*T2 + T3
    elif asymptotic:
        denominator = T1 + T7*T2 + T3

    return numerator/denominator

# 1D, first order derivatives
Dimensionality = 2
theoretical_scaling_values = theoretical_scaling_order1(Dimensionality)
asymptotic_scaling_values = theoretical_scaling_order1(Dimensionality, asymptotic=True)
print(f"Theoretical scaling value for {Dimensionality-1}D is: {theoretical_scaling_values}")
print(f"Asymptotic scaling value for {Dimensionality-1}D is: {asymptotic_scaling_values}")

# 2D first order derivatives
Dimensionality = 3
theoretical_scaling_values = theoretical_scaling_order1(Dimensionality)
asymptotic_scaling_values = theoretical_scaling_order1(Dimensionality, asymptotic=True)
print(f"Theoretical scaling value for {Dimensionality-1}D is: {theoretical_scaling_values}")
print(f"Asymptotic scaling value for {Dimensionality-1}D is: {asymptotic_scaling_values}")

# 3D first order derivatives
Dimensionality = 4
theoretical_scaling_values = theoretical_scaling_order1(Dimensionality)
asymptotic_scaling_values = theoretical_scaling_order1(Dimensionality, asymptotic=True)
print(f"Theoretical scaling value for {Dimensionality-1}D is: {theoretical_scaling_values}")
print(f"Asymptotic scaling value for {Dimensionality-1}D is: {asymptotic_scaling_values}")

# 2D second order derivatives
Dimensionality = 3
theoretical_scaling_values = theoretical_scaling_order2(Dimensionality)
asymptotic_scaling_values = theoretical_scaling_order2(Dimensionality, asymptotic=True)
print(f"Theoretical scaling value for {Dimensionality-1}D is: {theoretical_scaling_values}")
print(f"Asymptotic scaling value for {Dimensionality-1}D is: {asymptotic_scaling_values}")