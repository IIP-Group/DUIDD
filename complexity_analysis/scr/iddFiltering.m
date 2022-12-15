function [rv_mul] = iddFiltering(Detector, Scenario, M_R, N_T, T, Q, L)
%{
iddFiltering calculates the multiplication count of the IDD Detection,
considering an efficient implementation of textbook algorithms.

Inputs:
    Detector    "Low-Complexity-MF-LMMSE" or "MMSE-PIC"
    Scenario    "EXPL-INV" or "EXPL-FILTER"
    M_R         Number of BS antennas
    N_T         Number of Transmitters
    T           Number of time slots per coherence interval (block fading)
    Q           number of bits, positive is general case, negative values
    apply low-complexity gray-labeling QAM trick
Outputs:
    rv_mul      Number of real valued multiplications

%}

init_h

%% Soft Symbol Calculation (applying the Gray labeling trick, best
% implemenetation)
rv_mul = 0;

% p0 = 0.5 * (1 - tf.math.tanh(0.5 * llr_a))
rv_mul = rv_mul + N_T*T*Q*2;
switch Q
    case -1
        rv_mul = rv_mul + 1*N_T*T;
    case -2
        rv_mul = rv_mul + 4*N_T*T;
    case -4
        rv_mul = rv_mul + 10*N_T*T;
    case -6
        rv_mul = rv_mul + 22*N_T*T;
    otherwise
        % calculate symbol probability
        rv_mul = rv_mul + N_T*T*(2^Q)*Q;
        % s_hat = tf.reduce_sum(points_reshaped * tf.cast(P_C,
        % tf.complex64), axis=-1) complex!!
        rv_mul = rv_mul + 4*T*N_T*2^Q;
        % Calculate Squared Error
        rv_mul = rv_mul + 2*4*2^Q * N_T * T;
        % Calculate Error Variance
        rv_mul = rv_mul + 2^Q * N_T * T;
end

%% Parallel Interference Cancellation
compl_mul = 0;
switch Scenario
    case EXPL_INV
        % y_MF, G already calculated
        % efficient implementation of PIC with y_MF
        compl_mul = compl_mul + T*N_T.^2;
    case EXPL_FILTER
        % no y_MF and no G... really perform PIC on y (not y_MF)
        compl_mul = compl_mul + T*N_T*M_R;
    otherwise
end

%% Equalization
switch Detector
    case MMSE_PIC
        % has to calculate everything (also inverse) for every t
        switch Scenario
            case EXPL_INV
                % needs to do LU Decomposition instead of Cholesky becaues
                % A is not hermitian
                
                % calculation of A
                % G*diag(error_var) (Gram matrix is Hermitian, only
                % calculate upper triangular muls)
                compl_mul = compl_mul + T*1/2*N_T.*(N_T+1);

                % LU-decomposition
                compl_mul = compl_mul + T*(1/3*N_T.^3 + 1/4*N_T.^2 + 3/4*N_T - 1/2);

                % Forward substitution of L (invert L explicitly, diag is one)
                compl_mul = compl_mul + T*(1/6*N_T.^3 - 1/6*N_T);

                % back-substitution of U (N_T times vector backsub, nothing
                % is real)
                compl_mul = compl_mul + T*N_T.*((1/2)*(N_T.^2 + N_T));

                % calculate mu_i
                % mu_i = a_i^H g_i = 1- \sigma_n^2(A^-1)_ii (real)
                compl_mul = compl_mul + T*1/4*N_T;

                % calculate MMSE output 
                % Filtering A^-1 y_MF
                compl_mul = compl_mul + T*N_T.^2;

                % NPI calculation (rho_i = mu_i/(1-mu_i))
                compl_mul = compl_mul + (T/4)*N_T;
            case EXPL_FILTER
                error('not implemented')
            otherwise
                error('not implemented')
        end
    case MF_LMMSE
        switch Scenario
            case EXPL_INV
                % y_MF, G, mu_i and A^-1 already calculated
                % calculating filter matrix Z_bar
                compl_mul = compl_mul + N_T.^2 + 2*N_T;
                
                % NPI calculation (assuming SNR_Heuristic 4)
                rv_mul = rv_mul + T*(N_T.^2);
                
                % Filtering
                compl_mul = compl_mul + T*N_T.^2;
            case EXPL_FILTER
                error('not implemented')
            otherwise
                error('not implemented')
        end
    otherwise
        error('not implemented')
end

%% LLR calculation
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

rv_mul = rv_mul + 4*compl_mul;

%% L IDD iterations (i.e. L-1 times this filtering)
rv_mul = (L-1)*rv_mul;

% include initial squared and abs Gram Matrix calculation
if Scenario == EXPL_INV && Detector == MF_LMMSE && L >= 2
    rv_mul = rv_mul + 2*N_T.^2;
end

% rv_mul = int64(rv_mul);

end

