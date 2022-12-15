% Computational Complexity Analysis Script
% 3 scenarios: 16, 32, 64 and 128 BS Antennas
% Sweeps with N_T=1:M_R
% Three scenarios for T=1,14 and sweep from 1:100
%
% This analysis counts the number of real valued multiplications for detection
%

clear all
close all

addpath ./scr

QR_BACKSUB = "QR-BACKSUB";
QR_EXPLICIT = "QR-EXPLICIT";
MF_BACKSUB = "MF-BACKSUB";
EXPL_INV = "EXPL-INV";
EXPL_FILTER = "EXPL-FILTER";

MMSE_PIC = "MMSE-PIC";
MF_LMMSE = "MF-LMMSE";

Q = -4;

%% Analyzing Initial LMMSE Detection (to determine which method is optimal)

for T = [1, 10, 14]
    for M_R = [4, 8, 16, 32, 64, 128]
        figure
        hold on
        title(sprintf('Initial Filtering: UE-Sweep w. T = %d M_R = %d', T, M_R))
        N_T = 1:M_R;
        plot(N_T, initialLMMSEFiltering(QR_BACKSUB, M_R, N_T, T, Q),'b-','Linewidth',2)
        plot(N_T, initialLMMSEFiltering(QR_EXPLICIT, M_R, N_T, T, Q),'c-','Linewidth',2)
        plot(N_T, initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'r-','Linewidth',2)
        plot(N_T, initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q),'m-','Linewidth',2)
        plot(N_T, initialLMMSEFiltering(EXPL_FILTER, M_R, N_T, T, Q),'g-','Linewidth',2)
        legend(QR_BACKSUB, QR_EXPLICIT, "Chol. " + MF_BACKSUB, "Chol. " + EXPL_INV, "Chol. " + EXPL_FILTER, Location="northwest")
        xlabel("N_T")
        ylabel("Real Valued Multiplications")
        hold off
    end
end

% Sweeping over T
T=1:1000;
settings = {[4,4], [16,4], [16,8], [32,8], [64,16], [128,32]};
for i=1:length(settings)
    M_R= settings{i}(1);
    N_T= settings{i}(2);
    figure
    loglog(T, initialLMMSEFiltering(QR_BACKSUB, M_R, N_T, T, Q),'b-','Linewidth',2)
    title(sprintf('Initial Filtering: T-Sweep M_R = %d N_T = %d', M_R, N_T))
    hold on
    loglog(T, initialLMMSEFiltering(QR_EXPLICIT, M_R, N_T, T, Q),'c-','Linewidth',2)
    loglog(T, initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'r-','Linewidth',2)
    loglog(T, initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q),'m-','Linewidth',2)
    loglog(T, initialLMMSEFiltering(EXPL_FILTER, M_R, N_T, T, Q),'g-','Linewidth',2)
    legend(QR_BACKSUB, QR_EXPLICIT, "Chol. " + MF_BACKSUB, "Chol. " + EXPL_INV, "Chol. " + EXPL_FILTER, Location="northwest")
    xlabel("T")
    ylabel("Real Valued Multiplications")
    hold off
end

%% Analyzing the IDD Detection Complexity Increase

for T = [1, 10, 14]
    for M_R = [16, 32, 64, 128]
        figure
        hold on
        title(sprintf('IDD and Baseline EXPL-INV: UE-Sweep w. T = %d M_R = %d', T, M_R))
        N_T = 1:M_R;
        plot(N_T, (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 1) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q),'b-','Linewidth',2)
        plot(N_T, (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 1) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q),'r-','Linewidth',2)
        plot(N_T, (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 2) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q),'b-.','Linewidth',2)
        plot(N_T, (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 2) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q),'r-.','Linewidth',2)
        plot(N_T, (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 3) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q),'b--','Linewidth',2)
        plot(N_T, (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 3) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q),'r--','Linewidth',2)
        plot(N_T, (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 4) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q),'b:','Linewidth',2)
        plot(N_T, (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 4) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q),'r:','Linewidth',2)
        legend(MF_LMMSE+"I=1", MMSE_PIC+"I=1", MF_LMMSE+"I=2", MMSE_PIC+"I=2", MF_LMMSE+"I=3", MMSE_PIC+"I=3", MF_LMMSE+"I=4", MMSE_PIC+"I=4", Location="northwest")
        xlabel("N_T")
        ylabel("(#MUL IDD Iter)/(#MUL Baseline LMMSE)")
        hold off
    end
end

for T = [1, 10, 14]
    for M_R = [16, 32, 64, 128]
        figure
        hold on
        title(sprintf('IDD w/ EXPL-INV, Baseline MF-BACKSUB: UE-Sweep w. T = %d M_R = %d', T, M_R))
        N_T = 1:M_R;
        plot(N_T, (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 1) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'b-','Linewidth',2)
        plot(N_T, (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 1) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'r-','Linewidth',2)
        plot(N_T, (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 2) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'b-.','Linewidth',2)
        plot(N_T, (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 2) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'r-.','Linewidth',2)
        plot(N_T, (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 3) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'b--','Linewidth',2)
        plot(N_T, (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 3) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'r--','Linewidth',2)
        plot(N_T, (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 4) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'b:','Linewidth',2)
        plot(N_T, (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 4) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'r:','Linewidth',2)
        legend(MF_LMMSE+"I=1", MMSE_PIC+"I=1", MF_LMMSE+"I=2", MMSE_PIC+"I=2", MF_LMMSE+"I=3", MMSE_PIC+"I=3", MF_LMMSE+"I=4", MMSE_PIC+"I=4", Location="northwest")
        xlabel("N_T")
        ylabel("(#MUL IDD Iter)/(#MUL Baseline LMMSE)")
        hold off
    end
end

% Sweeping over T
T=1:1000;
settings = {[4,4], [16,4], [16,8], [32,8], [64,16], [128,32]};
for i=1:length(settings)
    M_R= settings{i}(1);
    N_T= settings{i}(2);
    figure
    semilogx(T, (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 1) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'b-','Linewidth',2)
    hold on
    semilogx(T, (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 1) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'r-','Linewidth',2)
    semilogx(T, (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 2) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'b-.','Linewidth',2)
    semilogx(T, (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 2) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'r-.','Linewidth',2)
    semilogx(T, (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 3) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'b--','Linewidth',2)
    semilogx(T, (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 3) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'r--','Linewidth',2)
    semilogx(T, (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 4) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'b:','Linewidth',2)
    semilogx(T, (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 4) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q))./initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q),'r:','Linewidth',2)
    title(sprintf('IDD-Complexity/Initial Filtering: T-Sweep M_R = %d N_T = %d', M_R, N_T))
    legend(MF_LMMSE+"I=1", MMSE_PIC+"I=1", MF_LMMSE+"I=2", MMSE_PIC+"I=2", MF_LMMSE+"I=3", MMSE_PIC+"I=3", MF_LMMSE+"I=4", MMSE_PIC+"I=4", Location="northwest")
    xlabel("T")
    ylabel("(#MUL IDD Iter)/(#MUL Baseline LMMSE)")
    hold off
end

%% Complexity of considered scenario
M_R = 16;
N_T = 4;
T_arr = [1, 10, 14];
for T=T_arr
    baseline_lmmse = initialLMMSEFiltering(MF_BACKSUB, M_R, N_T, T, Q);
    mmse_pic_1 = (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 1) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    mmse_pic_2 = (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 2) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    mmse_pic_3 = (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 3) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    mmse_pic_4 = (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 4) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    lmmse_mf_1 = (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 1) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    lmmse_mf_2 = (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 2) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    lmmse_mf_3 = (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 3) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    lmmse_mf_4 = (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 4) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    
    data_table_ = table(baseline_lmmse, mmse_pic_1, mmse_pic_2, mmse_pic_3, mmse_pic_4, lmmse_mf_1, lmmse_mf_2, lmmse_mf_3, lmmse_mf_4);
    data_table_.Properties.VariableNames = ["LMMSE", MMSE_PIC+"1", MMSE_PIC+"2", MMSE_PIC+"3", MMSE_PIC+"4",  MF_LMMSE+"1", MF_LMMSE+"2", MF_LMMSE+"3", MF_LMMSE+"4"];
    dataset_title = sprintf('IDD-Complexity M_R = %d N_T = %d T=%d', M_R, N_T, T);
    writetable(data_table_,"./data/"+string(dataset_title)+".csv",'Delimiter',',','QuoteStrings',true)
end

M_R = 8;
N_T = 4;
T_arr = [10];
for T=T_arr
    baseline_lmmse = initialLMMSEFiltering(QR_EXPLICIT, M_R, N_T, T, Q);    % QR Explicit more efficient for 8x4 and T=10
    mmse_pic_1 = (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 1) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    mmse_pic_2 = (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 2) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    mmse_pic_3 = (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 3) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    mmse_pic_4 = (iddFiltering(MMSE_PIC, EXPL_INV, M_R, N_T, T, Q, 4) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    lmmse_mf_1 = (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 1) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    lmmse_mf_2 = (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 2) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    lmmse_mf_3 = (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 3) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));
    lmmse_mf_4 = (iddFiltering(MF_LMMSE, EXPL_INV, M_R, N_T, T, Q, 4) + initialLMMSEFiltering(EXPL_INV, M_R, N_T, T, Q));

    data_table_ = table(baseline_lmmse, mmse_pic_1, mmse_pic_2, mmse_pic_3, mmse_pic_4, lmmse_mf_1, lmmse_mf_2, lmmse_mf_3, lmmse_mf_4);
    data_table_.Properties.VariableNames = ["LMMSE", MMSE_PIC+"1", MMSE_PIC+"2", MMSE_PIC+"3", MMSE_PIC+"4",  MF_LMMSE+"1", MF_LMMSE+"2", MF_LMMSE+"3", MF_LMMSE+"4"];
    dataset_title = sprintf('IDD-Complexity M_R = %d N_T = %d T=%d', M_R, N_T, T);
    writetable(data_table_,"./data/"+string(dataset_title)+".csv",'Delimiter',',','QuoteStrings',true)
end

