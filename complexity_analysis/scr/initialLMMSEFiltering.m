function [rv_mul] = initialLMMSEFiltering(Scenario, M_R, N_T, T, Q)
%{
initialLMMSEFiltering calculates the multiplication count of the initial
LMMSE Detection, considering an efficient implementation of
textbook-algorithms, applying several low-complexity tricks.
Inputs:
    Scenario    "MF_BACKSUB" for matched filtering w/ backsubstitution,
                "EXPL_INV" for explicitly calculating the inverse of A, 
                "EXPL_FILTER" for explicitly calculating the filter
    M_R         Number of BS antennas
    N_T         Number of Transmitters
    T           Number of time slots per coherence interval (block fading)
    Q           number of bits, positive is general case, negative value
    applies Gray Labeling QAM trick (negative Q \in {-1,-2,-4,-6})
Outputs:
    rv_mul      Number of real valued multiplications
%}
init_h

mul = 0;
if Scenario == QR_BACKSUB || Scenario == QR_EXPLICIT
    % calculate reduced QR decomposition of [H; \sigma_n I_M_R]
    % considering that the lower part corresponding to \sigma_n
    % I_M_R is lower triangular with real values on the diagonal.
    mul = M_R.*(N_T.^2) + 1/3*N_T.^3 - 1/4*N_T.^2 -5/12*N_T;

    % \tilde{Q}_a is M_R x N_T (part that corresponds to H), 1/sigma_n Q_b
    % is R^-1
    % R_red^-1 (\tilde{Q}_a^H y)
    
    if Scenario == QR_BACKSUB
        % \tilde{Q}_a^H y
        mul = mul + N_T*M_R*T;
        % R_red^-1 (equivalent: \tilde{Q}_b) w/ back substitution
        % (backsub is the same complexity as multiplying
        % upper triangular matrix with vector, diag R_red is real)
        mul = mul + T*1/2*(N_T.^2);
    elseif Scenario == QR_EXPLICIT
        % 1/sigma_n Q_b
        mul = mul + N_T.^2/2;
        % W = (1/sigma_n Q_b) \tilde{Q}_a^H
        mul = mul + 1/2*N_T.^2*M_R;
        % filtering W*y
        mul = mul + N_T*M_R*T;
    end

    % NPI calculatoin: mu = diag(I - Q_b Q_b^H) (Q_b upper triangular, diag is real)
    mul = mul + 1/4*(N_T.^2);
 else
    % Gram Matrix calculation
    mul = mul + gramMatrix(M_R, N_T);
    % Cholesky Factorization of A = (G + n0*eye(M_T)) = L D L^H
    mul = mul + 1/6*N_T.^3 - 1/2*N_T.^2 - 1/3*N_T;
    % A = L D L^H
end

% Filtering x_hat = A \ H' y
switch Scenario
    case MF_BACKSUB
        % matched filtering y_MF = H' * y
        mul = mul + T*N_T*M_R;
        % Forward Substitution L^-1 * y_MF (same complexity as directly
        % multiplying y with L^-1, diag of L is 1)
        mul = mul + T*(1/2)*N_T.*(N_T-1);
        % Scale w D^-1 (real)
        mul = mul + 1/2*N_T*T;
        % Backsubstitutioin L^-H (diag of L is one)
        mul = mul + T*(1/2)*N_T.*(N_T-1);

        % Calculate mu (for filter normalizatioin and NPI calculatoin)
        % mu_i = diag(L^-H (D^-1 (L^-1 G)))     
        % by inverting L explicitly (diag L is one)
        mul = mul + 1/6*N_T.^3 - 1/6*N_T;
        % calculate all mu_i
        % mu_i = 1- \sigma_n^2 \sum_j=i^N_T D_jj^-1 |L^-1_ji|^2
        % (diag L is one)
        mul = mul + 1/4*(3/2 * N_T.^2 - 1/2*N_T);
    case EXPL_INV
        % matched filtering y_MF = H' * y
        mul = mul + T*N_T*M_R;
        % Forward Substitution (calc inverse) L^-1 (diag of L is one)
        mul = mul+ 1/6*N_T.^3 - 1/6*N_T;
        % Scaling D^-1 L^-1 (D is real, diag L is one)
        mul = mul + N_T.*(N_T-1)/4;
        % Multiplication of two triangular matrices L^-H (D L^-1) (diagonal
        % elements of right part are real, left part one, we only need to 
        % calculate upper triangular part, because A is hermitian)
        mul = mul + 1/6*N_T.^3 + 1/4*N_T.^2 - 5/12*N_T;
        % Now, we have A^-1

        % Filtering A^-1 y_MF
        mul = mul + T*N_T.^2;

        % Calculate mu (for filter normalizatioin and NPI calculatoin)
        % mu_i = 1-\sigma_n^2 (A^-1)_ii (real)
        mul = mul + 1/4*N_T;
    case EXPL_FILTER
        % No Matched Filtering!!

        % W = L^-H (D^-1 (L^-1 H^H))
        % forward substitution M_R times (diag L is one)
        mul = mul + 1/2*(N_T.*(N_T-1))*M_R;
        % scale with D^-1 (real)
        mul = mul + N_T*M_R/2;
        % backsub L^-H  M_R times (diag L is one)
        mul = mul + 1/2*(N_T.*(N_T-1))*M_R;

        % filtering W y
        mul = mul + N_T*M_R*T;
                
        % calculate mu_i = w_i' * h_i
        mul = mul + N_T*M_R;
    case QR_BACKSUB
    case QR_EXPLICIT
    otherwise
        error('Unknown Scenario')
end

% scale equalized symbols for unbiasedness (i.e., \tilde{x}_i =
% \hat{x_i}/mu_i and NPI calculation (rho_i = mu_i/(1-mu_i))
mul = mul + (T+1/4)*N_T;

rv_mul = 4*mul;

% LLR calculation
switch Q
    case -1
        rv_mul = rv_mul + 1*N_T*T;
    case -2
        rv_mul = rv_mul + 2*N_T*T;
    case -4
        rv_mul = rv_mul + 4*N_T*T;
    case -6
        rv_mul = rv_mul + 6*N_T*T;
    otherwise
        rv_mul = rv_mul + 3*T*N_T*2^Q;
end
%Multiply lambda_b my rho_i
rv_mul = rv_mul + T*Q*N_T;

% rv_mul = int64(rv_mul);

end

 