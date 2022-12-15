function [rv_mul] = preprocessing(Scenario, M_R, N_T, T, IDD)
%{
preprocessing calculates the multiplication count of the preprocessing
Inputs:
    Scenario    "MF_BACKSUB" for matched filtering w/ backsubstitution, 
                "EXPL_INV" for explicitly calculating the inverse of A, 
                "EXPL_FILTER" for explicitly calculating the filter
    M_R         Number of BS antennas
    N_T         Number of Transmitters
    T           Number of time slots per coherence interval (block fading)
    IDD         boolean, true if IDD
Outputs:
    rv_mul      Number of real valued multiplications
%}
mul = 0;

% calculating the Gram matrix
mul = mul + M_R*N_T^2;

rv_mul = 4*mul;

end

 