function [mul_compl] = gramMatrix(M_R, N_T)
%GRAMMATRIX calculating the Gram matrix:
% we only need to calculate the upper
% triangular part becaues the Gram is hermitian
mul_compl = M_R*(N_T+1).*N_T/2;
end

