%% compare_q_learning_improved.m
%   IMPROVED Structure-Aware Q-Learning 
%
%   Key improvements:
%   1. Increased iterations and better learning rate schedules
%   2. State visitation tracking with exploration bonuses
%   3. Average reward updates only on greedy actions
%   4. Comprehensive diagnostics and convergence analysis
%   5. Policy comparison and gap analysis
%
%   MDP Model:
%   - State: S = (T_r, T_s), where T_r in {0,...,K}, T_s in {1,...,K}
%   - Action: a in {0, 1} (silent or send)
%   - Transition on success (a=1, Z=1): T_r' = T_s
%   - Transition otherwise: T_r' = max(T_r - 1, 0)
%   - Sender update: T_s' = max(T_s - 1, Tilde_T_s)
%   - Reward: r * 1{T_r' > 0} - c * a
%   - Structural constraint: If T_r > T_s, then a = 0
% --------------------------------------------------------------------------
clear; clc; rng(42);
close all;

% ========== GLOBAL PARAMETERS =============================================
K      = 20;       % Max lifetime
r      = 1.0;      % Reward
c0     = 0.5;      % Cost
ps0    = 0.5;      % Success probability
STEPS  = 2e4;      % Learning iterations
RUNS   = 300;      % Monte Carlo runs

% IMPROVED: Better learning rate parameters
eps0   = 0.3;               % initial exploration rate (reduced for stability)
alpha0 = 0.3;               % initial Q-learning rate
beta0  = 0.05;              % initial average-reward rate (reduced for stability)
DECAY_START = 200000;       % steps before decay begins (increased)

% Exploration bonus parameters
USE_EXPLORATION_BONUS = true;
BONUS_SCALE = 0.05;

% Only update rho on greedy actions
GREEDY_RHO_UPDATE = true;

SAFEK  = 3000;              % do exact DP only if K <= SAFEK

% Sweep grids
psVals = 0 : 0.1 : 1;           % 11 points for p_s
cVals  = linspace(0, r, 11);    % 11 points for c (0 to r)

% Diagnostic settings
SAVE_DIAGNOSTICS = true;
PLOT_DIAGNOSTICS = true;

% ========== DEFINE T_s DISTRIBUTIONS ======================================
distributions = struct();

% 1. Uniform distribution
distributions(1).name = 'Uniform';
distributions(1).PT = ones(1,K)/K;
distributions(1).folder = 'results_uniform';

% % 2. Geometric-like (heavier on small values)
% geo_param = 0.7;
% geo_unnorm = geo_param.^(0:K-1);
% distributions(2).name = 'Geometric';
% distributions(2).PT = geo_unnorm / sum(geo_unnorm);
% distributions(2).folder = 'results_geometric';
% 
% % 3. Concentrated at K (always max TTL)
% concentrated_K = zeros(1, K);
% concentrated_K(K) = 1;
% distributions(3).name = 'Concentrated_at_K';
% distributions(3).PT = concentrated_K;
% distributions(3).folder = 'results_concentrated_K';
% 
% % 4. Concentrated at 1 (always min TTL)
% concentrated_1 = zeros(1, K);
% concentrated_1(1) = 1;
% distributions(4).name = 'Concentrated_at_1';
% distributions(4).PT = concentrated_1;
% distributions(4).folder = 'results_concentrated_1';
% 
% % 5. Bimodal (peaks at 1 and K)
% bimodal = zeros(1, K);
% bimodal(1) = 0.4;
% bimodal(K) = 0.4;
% bimodal(round(K/2)) =
% 0.2;
% distributions(5).name = 'Bimodal';
% distributions(5).PT = bimodal / sum(bimodal);
% distributions(5).folder = 'results_bimodal';
% 
% % 6. Truncated Gaussian (peak in middle)
% x = 1:K;
% mu = K/2;
% sigma = K/6;
% gauss_unnorm = exp(-0.5*((x-mu)/sigma).^2);
% distributions(6).name = 'Gaussian';
% distributions(6).PT = gauss_unnorm / sum(gauss_unnorm);
% distributions(6).folder = 'results_gaussian';

% ========== MAIN LOOP OVER DISTRIBUTIONS ==================================
for dist_idx = 1:length(distributions)
    PT = distributions(dist_idx).PT;
    dist_name = distributions(dist_idx).name;
    folder_name = distributions(dist_idx).folder;
    
    fprintf('\n============================================================\n');
    fprintf('Processing Distribution: %s\n', dist_name);
    fprintf('============================================================\n');
    
    % Create output folder
    if ~exist(folder_name, 'dir')
        mkdir(folder_name);
    end
    
    % Create diagnostics subfolder
    diag_folder = fullfile(folder_name, 'diagnostics');
    if ~exist(diag_folder, 'dir')
        mkdir(diag_folder);
    end
    
    % ========== LEARNING PARAMETERS STRUCT ================================
    learnParams = struct();
    learnParams.eps0 = eps0;
    learnParams.alpha0 = alpha0;
    learnParams.beta0 = beta0;
    learnParams.DECAY_START = DECAY_START;
    learnParams.USE_EXPLORATION_BONUS = USE_EXPLORATION_BONUS;
    learnParams.BONUS_SCALE = BONUS_SCALE;
    learnParams.GREEDY_RHO_UPDATE = GREEDY_RHO_UPDATE;
    
    % ========== 1. DEMO LEARNING CURVES (Fig1) =============================
    fprintf('\n-- Fig1: learning curves (ps=%.2f, c/r=%.2f) --\n', ps0, c0/r);
    [rho_opt, rho_base, rho_struct, gBaseCurve, gStructCurve, piOpt, piLearned, diagnostics] = ...
        simulateSetting(K, ps0, r, c0, PT, STEPS, RUNS, learnParams, SAFEK, false);
    
    % ========== DIAGNOSTIC ANALYSIS ========================================
    fprintf('\n-- Analyzing convergence --\n');
    [policyAnalysis] = analyzePolicyDifference(piOpt, piLearned, diagnostics.Qs_final, K);
    
    if PLOT_DIAGNOSTICS
        % Plot diagnostic figures
        figDiag = plotDiagnosticFigures(diagnostics, K, rho_opt, policyAnalysis, dist_name);
        saveFigures(figDiag, diag_folder, 'Diagnostics_Overview');
        close(figDiag);
    end
    
    if SAVE_DIAGNOSTICS
        % Save diagnostics to file
        save(fullfile(diag_folder, 'diagnostics.mat'), 'diagnostics', 'policyAnalysis');
    end
    
    % Plot learning curves
    t = 1:STEPS;
    C = get(groot, 'defaultAxesColorOrder');
    
    fig1 = figure('Name', sprintf('Fig1: Learning Curves (%s)', dist_name));
    hold on; box on;
    plot(t, gBaseCurve,   'Color', C(1,:), 'LineWidth', 1.3);
    plot(t, gStructCurve, 'Color', C(2,:), 'LineWidth', 1.3);
    
    if ~isnan(rho_opt)
        yline(rho_opt, ':k', 'LineWidth', 1.6);
    end
    xlabel('Iteration $n$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('Running avg reward', 'FontSize', 12);
    title(sprintf('Learning Curves (%s): K=%d, p_s=%.2f, c/r=%.2f', ...
          dist_name, K, ps0, c0/r));
    legend({'Baseline Q', 'Struct-Aware Q', 'Optimal $\rho^*$'}, ...
           'Interpreter', 'latex', 'Location', 'southeast');
    grid on; ylim([0 r]);
    
    saveFigures(fig1, folder_name, 'Fig1_Learning_Curves');
    
    % ========== 2. Q-LEARNING SWEEPS (Fig2 & Fig3) =========================
    nPs = numel(psVals);
    nC  = numel(cVals);
    
    rhoOpt_ps  = zeros(1, nPs);
    rhoBase_ps = zeros(1, nPs);
    rhoStr_ps  = zeros(1, nPs);
    
    rhoOpt_c   = zeros(1, nC);
    rhoBase_c  = zeros(1, nC);
    rhoStr_c   = zeros(1, nC);
    
    % Sweep p_s (fixed c = c0)
    fprintf('\n-- Sweeping p_s for Q-learning comparison --\n');
    for i = 1:nPs
        ps = psVals(i);
        fprintf('  p_s = %.2f (%d/%d)\n', ps, i, nPs);
        [rhoOpt_ps(i), rhoBase_ps(i), rhoStr_ps(i), ~, ~, ~, ~, ~] = ...
            simulateSetting(K, ps, r, c0, PT, STEPS, RUNS, learnParams, SAFEK, false);
    end
    
    % Sweep c/r (fixed ps = ps0)
    fprintf('\n-- Sweeping c/r for Q-learning comparison --\n');
    for j = 1:nC
        c = cVals(j);
        fprintf('  c/r = %.2f (%d/%d)\n', c/r, j, nC);
        [rhoOpt_c(j), rhoBase_c(j), rhoStr_c(j), ~, ~, ~, ~, ~] = ...
            simulateSetting(K, ps0, r, c, PT, STEPS, RUNS, learnParams, SAFEK, false);
    end
    
    % ----- Fig2 : avg reward vs p_s ----------------------------------------
    fig2 = figure('Name', sprintf('Fig2: Avg Reward vs p_s (%s)', dist_name));
    hold on; box on;
    plot(psVals, rhoBase_ps, '-o', 'LineWidth', 1.4, 'MarkerSize', 6);
    plot(psVals, rhoStr_ps,  '-s', 'LineWidth', 1.4, 'MarkerSize', 6);
    plot(psVals, rhoOpt_ps,  ':k', 'LineWidth', 1.8);
    xlabel('$p_s$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('Average reward', 'FontSize', 12);
    title(sprintf('Q-Learning (%s): K=%d, c/r=%.2f', dist_name, K, c0/r), ...
          'Interpreter', 'latex');
    legend({'Baseline Q', 'Struct-Aware Q', 'Optimal $\rho^*$'}, ...
           'Interpreter', 'latex', 'Location', 'southeast');
    grid on;
    
    saveFigures(fig2, folder_name, 'Fig2_AvgReward_vs_ps');
    
    % ----- Fig3 : avg reward vs c/r ----------------------------------------
    fig3 = figure('Name', sprintf('Fig3: Avg Reward vs c/r (%s)', dist_name));
    hold on; box on;
    plot(cVals/r, rhoBase_c, '-o', 'LineWidth', 1.4, 'MarkerSize', 6);
    plot(cVals/r, rhoStr_c,  '-s', 'LineWidth', 1.4, 'MarkerSize', 6);
    plot(cVals/r, rhoOpt_c,  ':k', 'LineWidth', 1.8);
    xlabel('$c/r$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('Average reward', 'FontSize', 12);
    title(sprintf('Q-Learning (%s): K=%d, p_s=%.2f', dist_name, K, ps0), ...
          'Interpreter', 'latex');
    legend({'Baseline Q', 'Struct-Aware Q', 'Optimal $\rho^*$'}, ...
           'Interpreter', 'latex', 'Location', 'northeast');
    grid on; xlim([0 1]);
    
    saveFigures(fig3, folder_name, 'Fig3_AvgReward_vs_cr');
    
    % ========== 3. POLICY COMPARISON CURVES (Fig4) =========================
    fprintf('\n-- Fig4: Policy comparison curves --\n');
    [gAlwaysCurve, ~]  = simulatePolicy('always', K, ps0, r, c0, PT, STEPS, RUNS);
    [gJITCurve, ~]     = simulatePolicy('JIT',    K, ps0, r, c0, PT, STEPS, RUNS);
    [gThreshCurve, ~]  = simulatePolicy('threshold', K, ps0, r, c0, PT, STEPS, RUNS);
    
    fig4 = figure('Name', sprintf('Fig4: Policy Comparison (%s)', dist_name));
    hold on; box on;
    plot(t, gStructCurve, '-',  'LineWidth', 1.3);
    plot(t, gAlwaysCurve, '--', 'LineWidth', 1.3);
    plot(t, gJITCurve,    '-.', 'LineWidth', 1.3);
    plot(t, gThreshCurve, ':',  'LineWidth', 1.3);
    if ~isnan(rho_opt)
        yline(rho_opt, ':k', 'LineWidth', 1.6);
    end
    xlabel('Iteration $n$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('Running avg reward', 'FontSize', 12);
    title(sprintf('Policy Comparison (%s): K=%d, p_s=%.2f, c/r=%.2f', ...
          dist_name, K, ps0, c0/r));
    legend({'Struct-Aware Q', 'Always send', 'JIT', 'Threshold', 'Optimal'}, ...
           'Interpreter', 'latex', 'Location', 'southeast');
    grid on; ylim([0 r]);
    
    saveFigures(fig4, folder_name, 'Fig4_Policy_Comparison');
    
    % ========== 4. POLICY SWEEPS (Fig5 & Fig6) =============================
    rhoStr_p      = zeros(1, nPs);
    rhoAlways_p   = zeros(1, nPs);
    rhoJIT_p      = zeros(1, nPs);
    rhoThresh_p   = zeros(1, nPs);
    
    rhoStr_cPol   = zeros(1, nC);
    rhoAlways_c   = zeros(1, nC);
    rhoJIT_c      = zeros(1, nC);
    rhoThresh_c   = zeros(1, nC);
    
    % Sweep p_s for policies
    fprintf('\n-- Sweeping p_s for policy comparison --\n');
    for i = 1:nPs
        ps = psVals(i);
        fprintf('  p_s = %.2f (%d/%d)\n', ps, i, nPs);
        [~, ~, rhoStr_p(i), ~, ~, ~, ~, ~] = ...
            simulateSetting(K, ps, r, c0, PT, STEPS, RUNS, learnParams, SAFEK, false);
        [~, rhoAlways_p(i)]  = simulatePolicy('always',    K, ps, r, c0, PT, STEPS, RUNS);
        [~, rhoJIT_p(i)]     = simulatePolicy('JIT',       K, ps, r, c0, PT, STEPS, RUNS);
        [~, rhoThresh_p(i)]  = simulatePolicy('threshold', K, ps, r, c0, PT, STEPS, RUNS);
    end
    
    % Sweep c/r for policies
    fprintf('\n-- Sweeping c/r for policy comparison --\n');
    for j = 1:nC
        c = cVals(j);
        fprintf('  c/r = %.2f (%d/%d)\n', c/r, j, nC);
        [~, ~, rhoStr_cPol(j), ~, ~, ~, ~, ~] = ...
            simulateSetting(K, ps0, r, c, PT, STEPS, RUNS, learnParams, SAFEK, false);
        [~, rhoAlways_c(j)]  = simulatePolicy('always',    K, ps0, r, c, PT, STEPS, RUNS);
        [~, rhoJIT_c(j)]     = simulatePolicy('JIT',       K, ps0, r, c, PT, STEPS, RUNS);
        [~, rhoThresh_c(j)]  = simulatePolicy('threshold', K, ps0, r, c, PT, STEPS, RUNS);
    end
    
    % ----- Fig5 : policies vs p_s ------------------------------------------
    fig5 = figure('Name', sprintf('Fig5: Policies vs p_s (%s)', dist_name));
    hold on; box on;
    plot(psVals, rhoStr_p,    '-s', 'LineWidth', 1.4, 'MarkerSize', 6);
    plot(psVals, rhoAlways_p, '-o', 'LineWidth', 1.4, 'MarkerSize', 6);
    plot(psVals, rhoJIT_p,    '-^', 'LineWidth', 1.4, 'MarkerSize', 6);
    plot(psVals, rhoThresh_p, '-d', 'LineWidth', 1.4, 'MarkerSize', 6);
    plot(psVals, rhoOpt_ps,   ':k', 'LineWidth', 1.8);
    xlabel('$p_s$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('Average reward', 'FontSize', 12);
    title(sprintf('Policies vs $p_s$ (%s): K=%d, c/r=%.2f', dist_name, K, c0/r), ...
          'Interpreter', 'latex');
    legend({'Struct-Aware Q', 'Always send', 'JIT', 'Threshold', 'Optimal'}, ...
           'Interpreter', 'latex', 'Location', 'southeast');
    grid on;
    
    saveFigures(fig5, folder_name, 'Fig5_Policies_vs_ps');
    
    % ----- Fig6 : policies vs c/r ------------------------------------------
    fig6 = figure('Name', sprintf('Fig6: Policies vs c/r (%s)', dist_name));
    hold on; box on;
    plot(cVals/r, rhoStr_cPol, '-s', 'LineWidth', 1.4, 'MarkerSize', 6);
    plot(cVals/r, rhoAlways_c, '-o', 'LineWidth', 1.4, 'MarkerSize', 6);
    plot(cVals/r, rhoJIT_c,    '-^', 'LineWidth', 1.4, 'MarkerSize', 6);
    plot(cVals/r, rhoThresh_c, '-d', 'LineWidth', 1.4, 'MarkerSize', 6);
    plot(cVals/r, rhoOpt_c,    ':k', 'LineWidth', 1.8);
    xlabel('$c/r$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('Average reward', 'FontSize', 12);
    title(sprintf('Policies vs $c/r$ (%s): K=%d, p_s=%.2f', dist_name, K, ps0), ...
          'Interpreter', 'latex');
    legend({'Struct-Aware Q', 'Always send', 'JIT', 'Threshold', 'Optimal'}, ...
           'Interpreter', 'latex', 'Location', 'northeast');
    grid on; xlim([0 1]);
    
    saveFigures(fig6, folder_name, 'Fig6_Policies_vs_cr');
    
    % ========== 5. POLICY HEATMAPS (Fig7 & Fig8) ===========================
    fprintf('\n-- Generating policy heatmaps --\n');
    
    % ----- Fig7: Optimal Policy Heatmap (from DP) --------------------------
    if ~isempty(piOpt) && ~any(isnan(piOpt(:)))
        fig7 = figure('Name', sprintf('Fig7: Optimal Policy (%s)', dist_name));
        plotPolicyHeatmap(piOpt, K, sprintf('Optimal Policy (DP) - %s', dist_name), ps0, c0, r);
        saveFigures(fig7, folder_name, 'Fig7_Optimal_Policy_Heatmap');
    end
    
    % ----- Fig8: Learned Policy Heatmap (from Struct-Aware Q) --------------
    if ~isempty(piLearned)
        fig8 = figure('Name', sprintf('Fig8: Learned Policy (%s)', dist_name));
        plotPolicyHeatmap(piLearned, K, sprintf('Learned Policy (Struct-Aware Q) - %s', dist_name), ps0, c0, r);
        saveFigures(fig8, folder_name, 'Fig8_Learned_Policy_Heatmap');
    end
    
    % ----- Fig8b: Policy Difference Heatmap --------------------------------
    if ~isempty(piOpt) && ~any(isnan(piOpt(:))) && ~isempty(piLearned)
        fig8b = figure('Name', sprintf('Fig8b: Policy Difference (%s)', dist_name));
        plotPolicyDifferenceHeatmap(piOpt, piLearned, diagnostics.Qs_final, K, ...
            sprintf('Policy Difference - %s', dist_name), ps0, c0, r);
        saveFigures(fig8b, folder_name, 'Fig8b_Policy_Difference_Heatmap');
    end
    
    % ========== 6. HEATMAPS FOR DIFFERENT c/r VALUES =======================
    fprintf('\n-- Generating heatmaps for different c/r values --\n');
    
    cValsHeatmap = [0, 0.25, 0.5, 0.75];
    fig9 = figure('Name', sprintf('Fig9: Policy Heatmaps vs c/r (%s)', dist_name));
    fig9.Position = [100 100 1200 900];
    
    for idx = 1:length(cValsHeatmap)
        c_val = cValsHeatmap(idx);
        fprintf('  Generating heatmap for c/r = %.2f\n', c_val/r);
        [~, ~, ~, ~, ~, piOpt_c, ~, ~] = ...
            simulateSetting(K, ps0, r, c_val, PT, STEPS, 1, learnParams, SAFEK, false);
        
        subplot(2, 2, idx);
        plotPolicyHeatmapSubplot(piOpt_c, K, sprintf('$c/r = %.2f$', c_val/r));
    end
    sgtitle(sprintf('Optimal Policy Regions (%s): K=%d, $p_s$=%.2f', dist_name, K, ps0), ...
            'Interpreter', 'latex', 'FontSize', 14);
    
    saveFigures(fig9, folder_name, 'Fig9_Policy_Heatmaps_vs_cr');
    
    % ========== 7. HEATMAPS FOR DIFFERENT p_s VALUES =======================
    fprintf('\n-- Generating heatmaps for different p_s values --\n');
    
    psValsHeatmap = [0.25, 0.5, 0.75, 1.0];
    fig10 = figure('Name', sprintf('Fig10: Policy Heatmaps vs p_s (%s)', dist_name));
    fig10.Position = [100 100 1200 900];
    
    for idx = 1:length(psValsHeatmap)
        ps_val = psValsHeatmap(idx);
        fprintf('  Generating heatmap for p_s = %.2f\n', ps_val);
        [~, ~, ~, ~, ~, piOpt_ps, ~, ~] = ...
            simulateSetting(K, ps_val, r, c0, PT, STEPS, 1, learnParams, SAFEK, false);
        
        subplot(2, 2, idx);
        plotPolicyHeatmapSubplot(piOpt_ps, K, sprintf('$p_s = %.2f$', ps_val));
    end
    sgtitle(sprintf('Optimal Policy Regions (%s): K=%d, $c/r$=%.2f', dist_name, K, c0/r), ...
            'Interpreter', 'latex', 'FontSize', 14);
    
    saveFigures(fig10, folder_name, 'Fig10_Policy_Heatmaps_vs_ps');
    
    % ========== 8. PLOT THE DISTRIBUTION ITSELF ============================
    fig11 = figure('Name', sprintf('Fig11: Distribution (%s)', dist_name));
    bar(1:K, PT, 'FaceColor', [0.3 0.5 0.8]);
    xlabel('$\tilde{T}_s$ (New packet TTL)', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('Probability', 'FontSize', 12);
    title(sprintf('Distribution of $\\tilde{T}_s$: %s', dist_name), 'Interpreter', 'latex');
    grid on;
    xlim([0 K+1]);
    
    saveFigures(fig11, folder_name, 'Fig11_Ts_Distribution');
    
    % ========== 9. Q-VALUE ANALYSIS FIGURE =================================
    if ~isempty(diagnostics.Qs_final)
        fig12 = figure('Name', sprintf('Fig12: Q-Value Analysis (%s)', dist_name));
        plotQValueAnalysis(diagnostics.Qs_final, K, dist_name, ps0, c0, r);
        saveFigures(fig12, folder_name, 'Fig12_QValue_Analysis');
    end
    
    fprintf('\n-- Completed distribution: %s --\n', dist_name);
    fprintf('   Figures saved in folder: %s\n', folder_name);
    fprintf('   Policy agreement: %.1f%%\n', policyAnalysis.agreement_rate * 100);
    
    % Close figures to save memory
    close all;
end

fprintf('\n============================================================\n');
fprintf('All distributions processed. Figures saved in respective folders.\n');
fprintf('============================================================\n');


%% ==========================================================================
%  HELPER FUNCTION: SAVE FIGURES
% ==========================================================================

function saveFigures(fig, folder, filename)
% SAVEFIGURES  Save figure in eps, fig, and pdf formats

% Save as .fig
savefig(fig, fullfile(folder, [filename '.fig']));

% Save as .eps
print(fig, fullfile(folder, [filename '.eps']), '-depsc2', '-r300');

% Save as .pdf
try
    exportgraphics(fig, fullfile(folder, [filename '.pdf']), 'ContentType', 'vector');
catch
    % Fallback for older MATLAB versions
    print(fig, fullfile(folder, [filename '.pdf']), '-dpdf', '-r300');
end

fprintf('    Saved: %s (.fig, .eps, .pdf)\n', filename);

end


%% ==========================================================================
%  MAIN SIMULATION FUNCTION (IMPROVED)
% ==========================================================================

function [rho_opt, rho_base, rho_struct, gBaseAvg, gStructAvg, piOpt, piLearned, diagnostics] = ...
         simulateSetting(K, ps, r, c, PT, STEPS, RUNS, learnParams, SAFEK, makeFig1)
% SIMULATESETTING  Run DP and Q-learning for a given setting (IMPROVED)

% Extract learning parameters
eps0 = learnParams.eps0;
alpha0 = learnParams.alpha0;
beta0 = learnParams.beta0;
DECAY_START = learnParams.DECAY_START;
USE_EXPLORATION_BONUS = learnParams.USE_EXPLORATION_BONUS;
BONUS_SCALE = learnParams.BONUS_SCALE;
GREEDY_RHO_UPDATE = learnParams.GREEDY_RHO_UPDATE;

% Helper: sample expiration time from PT
sampleTs = @() find(rand < cumsum(PT), 1);

% Initialize policy matrices
piOpt     = NaN(K+1, K);
piLearned = zeros(K+1, K);

% Initialize diagnostics
diagnostics = struct();
diagnostics.Qs_final = [];
diagnostics.visitCount = zeros(K+1, K);
diagnostics.td_errors_struct = [];
diagnostics.rho_history_struct = [];
diagnostics.rho_history_base = [];

% =========================================================================
%  PART 1: Build MDP and solve via Relative Value Iteration
% =========================================================================
S   = (K+1) * K;  % Total number of states
s2i = @(Tr, Ts) Tr*K + Ts;  % State-to-index mapping

% Transition matrices and reward vectors
P0 = sparse(S, S);
P1 = sparse(S, S);
R0 = zeros(S, 1);
R1 = zeros(S, 1);

for Tr = 0:K
    for Ts = 1:K
        s = s2i(Tr, Ts);
        
        % === Action 0: Silent ===
        Tr_next_silent = max(Tr - 1, 0);
        R0(s) = r * (Tr_next_silent > 0);
        
        for Tsp = 1:K
            Ts_next = max(Ts - 1, Tsp);
            sp = s2i(Tr_next_silent, Ts_next);
            P0(s, sp) = P0(s, sp) + PT(Tsp);
        end
        
        % === Action 1: Send ===
        Tr_next_fail = max(Tr - 1, 0);
        R1(s) = ps * r + (1 - ps) * r * (Tr_next_fail > 0) - c;
        
        for Tsp = 1:K
            Ts_next = max(Ts - 1, Tsp);
            
            % Failure
            sp_fail = s2i(Tr_next_fail, Ts_next);
            P1(s, sp_fail) = P1(s, sp_fail) + (1 - ps) * PT(Tsp);
            
            % Success: Tr' = Ts
            Tr_next_succ = Ts;
            sp_succ = s2i(Tr_next_succ, Ts_next);
            P1(s, sp_succ) = P1(s, sp_succ) + ps * PT(Tsp);
        end
    end
end

% ---- Relative Value Iteration for optimal ρ* ----------------------------
rho_opt = NaN;
if K <= SAFEK
    tol  = 1e-10;
    maxI = 10000;
    h    = zeros(S, 1);
    ref  = s2i(0, 1);
    
    for it = 1:maxI
        V0 = R0 + P0 * h;
        V1 = R1 + P1 * h;
        
        A = zeros(S, 1);
        for Tr = 0:K
            for Ts = 1:K
                s = s2i(Tr, Ts);
                if Tr > Ts
                    A(s) = V0(s);
                    piOpt(Tr+1, Ts) = 0;
                else
                    if V1(s) > V0(s)
                        A(s) = V1(s);
                        piOpt(Tr+1, Ts) = 1;
                    else
                        A(s) = V0(s);
                        piOpt(Tr+1, Ts) = 0;
                    end
                end
            end
        end
        
        hNew = A - A(ref);
        
        if max(abs(hNew - h)) < tol
            break;
        end
        h = hNew;
    end
    rho_opt = A(ref);
end

% =========================================================================
%  PART 2: Q-Learning (Baseline and Struct-Aware) - IMPROVED
% =========================================================================
gBase   = zeros(RUNS, STEPS);
gStruct = zeros(RUNS, STEPS);
Qs_final = zeros(K+1, K, 2);
visitCount_final = zeros(K+1, K);

% For diagnostics, store from last run
td_errors_struct = zeros(STEPS, 1);
rho_history_struct = zeros(STEPS, 1);
rho_history_base = zeros(STEPS, 1);

for run = 1:RUNS
    % Initialize Baseline Q-learning
    Qb  = zeros(K+1, K, 2);
    rhoB = 0;
    Tr_b = 0;
    Ts_b = sampleTs();
    cumRwdB = 0;
    visitCountB = zeros(K+1, K);
    
    % Initialize Struct-Aware Q-learning
    Qs  = zeros(K+1, K, 2);
    rhoS = 0;
    Tr_s = 0;
    Ts_s = sampleTs();
    cumRwdS = 0;
    visitCountS = zeros(K+1, K);
    
    for n = 1:STEPS
        % ========== IMPROVED Learning rate schedule ==========
        if n <= DECAY_START
            eps   = eps0;
            alpha = alpha0;
            beta  = beta0;
        else
            decay_n = n - DECAY_START;
            % Polynomial decay (more stable than exponential)
            eps   = eps0 / (1 + decay_n / DECAY_START)^0.5;
            alpha = alpha0 / (1 + decay_n / DECAY_START)^0.6;
            beta  = beta0 / (1 + decay_n / DECAY_START)^0.8;
        end
        
        % Common randomness for fair comparison
        Tsp_new = sampleTs();
        Z_common = rand;
        explore_rand_b = rand;
        explore_rand_s = rand;
        action_rand_b = rand;
        action_rand_s = rand;
        
        % === BASELINE Q-LEARNING ===
        Tr = Tr_b;
        Ts = Ts_b;
        visitCountB(Tr+1, Ts) = visitCountB(Tr+1, Ts) + 1;
        
        % Exploration bonus for baseline
        if USE_EXPLORATION_BONUS
            bonus_b = BONUS_SCALE / sqrt(visitCountB(Tr+1, Ts));
            effective_eps_b = min(eps + bonus_b, 0.5);
        else
            effective_eps_b = eps;
        end
        
        exploring_b = false;
        if explore_rand_b < effective_eps_b
            aB = floor(action_rand_b * 2);  % 0 or 1
            exploring_b = true;
        else
            [~, idx] = max(Qb(Tr+1, Ts, :));
            aB = idx - 1;
        end
        
        Zb = (Z_common < ps);
        
        if aB == 1 && Zb == 1
            Tr_next = Ts;
        else
            Tr_next = max(Tr - 1, 0);
        end
        
        Ts_next = max(Ts - 1, Tsp_new);
        rwdB = r * (Tr_next > 0) - c * aB;
        
        V_next_B = max(Qb(Tr_next+1, Ts_next, :));
        td_error_b = rwdB - rhoB + V_next_B - Qb(Tr+1, Ts, aB+1);
        Qb(Tr+1, Ts, aB+1) = Qb(Tr+1, Ts, aB+1) + alpha * td_error_b;
        
        % IMPROVED: Update rho only on greedy actions
        if GREEDY_RHO_UPDATE
            if ~exploring_b
                rhoB = rhoB + beta * (rwdB - rhoB);
            end
        else
            rhoB = rhoB + beta * (rwdB - rhoB);
        end
        
        Tr_b = Tr_next;
        Ts_b = Ts_next;
        
        cumRwdB = cumRwdB + rwdB;
        gBase(run, n) = cumRwdB / n;
        
        % === STRUCTURE-AWARE Q-LEARNING ===
        Tr = Tr_s;
        Ts = Ts_s;
        visitCountS(Tr+1, Ts) = visitCountS(Tr+1, Ts) + 1;
        
        % Exploration bonus for struct-aware
        if USE_EXPLORATION_BONUS
            bonus_s = BONUS_SCALE / sqrt(visitCountS(Tr+1, Ts));
            effective_eps_s = min(eps + bonus_s, 0.5);
        else
            effective_eps_s = eps;
        end
        
        exploring_s = false;
        if Tr > Ts
            % Structural constraint: must wait
            aS = 0;
        elseif explore_rand_s < effective_eps_s
            aS = floor(action_rand_s * 2);  % 0 or 1
            exploring_s = true;
        else
            [~, idx] = max(Qs(Tr+1, Ts, :));
            aS = idx - 1;
        end
        
        Zs = (Z_common < ps);
        
        if aS == 1 && Zs == 1
            Tr_next = Ts;
        else
            Tr_next = max(Tr - 1, 0);
        end
        
        Ts_next = max(Ts - 1, Tsp_new);
        rwdS = r * (Tr_next > 0) - c * aS;
        
        % IMPROVED: Constrained value bootstrap
        if Tr_next > Ts_next
            V_next_S = Qs(Tr_next+1, Ts_next, 1);  % Only a=0 feasible
        else
            V_next_S = max(Qs(Tr_next+1, Ts_next, :));
        end
        
        td_error_s = rwdS - rhoS + V_next_S - Qs(Tr+1, Ts, aS+1);
        Qs(Tr+1, Ts, aS+1) = Qs(Tr+1, Ts, aS+1) + alpha * td_error_s;
        
        % IMPROVED: Update rho only on greedy actions
        if GREEDY_RHO_UPDATE
            if ~exploring_s
                rhoS = rhoS + beta * (rwdS - rhoS);
            end
        else
            rhoS = rhoS + beta * (rwdS - rhoS);
        end
        
        Tr_s = Tr_next;
        Ts_s = Ts_next;
        
        cumRwdS = cumRwdS + rwdS;
        gStruct(run, n) = cumRwdS / n;
        
        % Store diagnostics for last run
        if run == RUNS
            td_errors_struct(n) = td_error_s;
            rho_history_struct(n) = rhoS;
            rho_history_base(n) = rhoB;
        end
    end
    
    % Store final Q-table and visit counts from last run
    if run == RUNS
        Qs_final = Qs;
        visitCount_final = visitCountS;
    end
end

% Extract learned policy with tie-breaking
for Tr = 0:K
    for Ts = 1:K
        if Tr > Ts
            piLearned(Tr+1, Ts) = 0;
        else
            Q0 = Qs_final(Tr+1, Ts, 1);
            Q1 = Qs_final(Tr+1, Ts, 2);
            % Use small threshold for tie-breaking
            if Q1 > Q0 + 1e-6
                piLearned(Tr+1, Ts) = 1;
            else
                piLearned(Tr+1, Ts) = 0;
            end
        end
    end
end

rho_base   = mean(gBase(:, end));
rho_struct = mean(gStruct(:, end));
gBaseAvg   = mean(gBase, 1);
gStructAvg = mean(gStruct, 1);

% Store diagnostics
diagnostics.Qs_final = Qs_final;
diagnostics.visitCount = visitCount_final;
diagnostics.td_errors_struct = td_errors_struct;
diagnostics.rho_history_struct = rho_history_struct;
diagnostics.rho_history_base = rho_history_base;
diagnostics.final_rho_struct = rho_struct;
diagnostics.final_rho_base = rho_base;

if makeFig1
    t = 1:STEPS;
    C = get(groot, 'defaultAxesColorOrder');
    
    figure('Name', 'Fig1: Learning Curves'); hold on; box on;
    plot(t, gBaseAvg,   'Color', C(1,:), 'LineWidth', 1.3);
    plot(t, gStructAvg, 'Color', C(2,:), 'LineWidth', 1.3);
    if ~isnan(rho_opt)
        yline(rho_opt, ':k', 'LineWidth', 1.6);
    end
    xlabel('Iteration $n$', 'Interpreter', 'latex', 'FontSize', 12);
    ylabel('Running avg reward', 'FontSize', 12);
    title(sprintf('Learning Curves: K=%d, p_s=%.2f, c/r=%.2f', K, ps, c/r));
    legend({'Baseline Q', 'Struct-Aware Q', 'Optimal $\rho^*$'}, ...
           'Interpreter', 'latex', 'Location', 'southeast');
    grid on; ylim([0 r]);
end

end


%% ==========================================================================
%  POLICY SIMULATION FUNCTION (IMPROVED)
% ==========================================================================

function [gCurve, finalAvg] = simulatePolicy(policyType, K, ps, r, c, PT, STEPS, RUNS)
% SIMULATEPOLICY  Simulate a fixed policy and compute average reward

sampleTs = @() find(rand < cumsum(PT), 1);

gRuns = zeros(RUNS, STEPS);

% Compute optimal threshold for threshold policy
if strcmp(policyType, 'threshold')
    % Simple heuristic: threshold at K/2
    thresh_k = max(1, floor(K * 0.3));
end

for run = 1:RUNS
    Tr = 0;
    Ts = sampleTs();
    cumRwd = 0;
    
    for n = 1:STEPS
        Tsp_new = sampleTs();
        
        switch policyType
            case 'always'
                % Respect structural constraint
                if Tr > Ts
                    a = 0;
                else
                    a = 1;
                end
                
            case 'JIT'
                % Transmit only at expiration
                if Tr > Ts
                    a = 0;
                else
                    a = (Tr == 0);
                end
                
            case 'threshold'
                % Transmit when Tr <= threshold
                if Tr > Ts
                    a = 0;
                else
                    a = (Tr <= thresh_k);
                end
                
            otherwise
                error('Unknown policy type: %s', policyType);
        end
        
        Z = (rand < ps);
        
        if a == 1 && Z == 1
            Tr_next = Ts;
        else
            Tr_next = max(Tr - 1, 0);
        end
        
        Ts_next = max(Ts - 1, Tsp_new);
        rwd = r * (Tr_next > 0) - c * a;
        
        cumRwd = cumRwd + rwd;
        gRuns(run, n) = cumRwd / n;
        
        Tr = Tr_next;
        Ts = Ts_next;
    end
end

gCurve   = mean(gRuns, 1);
finalAvg = gCurve(end);

end


%% ==========================================================================
%  POLICY ANALYSIS FUNCTION
% ==========================================================================

function [analysis] = analyzePolicyDifference(piOpt, piLearned, Qs, K)
% ANALYZEPOLICYDIFFERENCE  Analyze differences between optimal and learned policy

analysis = struct();

if isempty(piOpt) || any(isnan(piOpt(:)))
    analysis.agreement_rate = NaN;
    analysis.disagree_states = [];
    analysis.q_gaps = [];
    analysis.significant_disagree = 0;
    return;
end

agree_count = 0;
disagree_count = 0;
disagree_states = [];
q_gaps = [];

for Tr = 0:K
    for Ts = 1:K
        if Tr <= Ts  % Feasible region only
            Q0 = Qs(Tr+1, Ts, 1);
            Q1 = Qs(Tr+1, Ts, 2);
            gap = abs(Q1 - Q0);
            q_gaps = [q_gaps; Tr, Ts, gap, Q0, Q1];
            
            if piOpt(Tr+1, Ts) == piLearned(Tr+1, Ts)
                agree_count = agree_count + 1;
            else
                disagree_count = disagree_count + 1;
                disagree_states = [disagree_states; Tr, Ts, piOpt(Tr+1, Ts), piLearned(Tr+1, Ts), gap];
            end
        end
    end
end

total = agree_count + disagree_count;
analysis.agreement_rate = agree_count / total;
analysis.disagree_states = disagree_states;
analysis.q_gaps = q_gaps;

% Count significant disagreements (where Q-gap > threshold)
gap_threshold = 0.05;
if ~isempty(disagree_states)
    analysis.significant_disagree = sum(disagree_states(:, 5) > gap_threshold);
else
    analysis.significant_disagree = 0;
end

% Print summary
fprintf('\n===== POLICY ANALYSIS =====\n');
fprintf('Agreement rate: %.1f%% (%d/%d states)\n', 100*analysis.agreement_rate, agree_count, total);
fprintf('Disagreement states: %d\n', disagree_count);
fprintf('Significant disagreements (gap > %.2f): %d\n', gap_threshold, analysis.significant_disagree);

if ~isempty(disagree_states)
    fprintf('\nTop 10 disagreements by Q-gap:\n');
    [~, sort_idx] = sort(disagree_states(:, 5), 'descend');
    for i = 1:min(10, size(disagree_states, 1))
        idx = sort_idx(i);
        fprintf('  (Tr=%2d, Ts=%2d): opt=%d, learned=%d, gap=%.4f\n', ...
            disagree_states(idx, 1), disagree_states(idx, 2), ...
            disagree_states(idx, 3), disagree_states(idx, 4), ...
            disagree_states(idx, 5));
    end
end

end


%% ==========================================================================
%  DIAGNOSTIC PLOTTING FUNCTION
% ==========================================================================

function fig = plotDiagnosticFigures(diagnostics, K, rho_opt, policyAnalysis, dist_name)
% PLOTDIAGNOSTICFIGURES  Create comprehensive diagnostic plots

fig = figure('Position', [100 100 1600 900], 'Name', sprintf('Diagnostics: %s', dist_name));

STEPS = length(diagnostics.td_errors_struct);

% 1. Visit count heatmap
subplot(2, 3, 1);
imagesc(1:K, 0:K, log10(diagnostics.visitCount + 1));
set(gca, 'YDir', 'normal');
colorbar;
xlabel('$T_s$', 'Interpreter', 'latex');
ylabel('$T_r$', 'Interpreter', 'latex');
title('$\log_{10}$(Visit Count + 1)', 'Interpreter', 'latex');
hold on;
plot([1, K], [1, K], 'w--', 'LineWidth', 1.5);
hold off;

% 2. TD error convergence
subplot(2, 3, 2);
window = min(10000, STEPS/10);
smoothed_td = movmean(abs(diagnostics.td_errors_struct), window);
semilogy(1:STEPS, smoothed_td, 'LineWidth', 1.2);
xlabel('Iteration', 'FontSize', 10);
ylabel('$|$TD Error$|$ (smoothed)', 'Interpreter', 'latex');
title('TD Error Convergence');
grid on;

% 3. Average reward convergence
subplot(2, 3, 3);
plot(1:STEPS, diagnostics.rho_history_struct, 'b', 'LineWidth', 1.2);
hold on;
plot(1:STEPS, diagnostics.rho_history_base, 'r', 'LineWidth', 1.2);
if ~isnan(rho_opt)
    yline(rho_opt, 'k--', 'LineWidth', 1.5);
end
xlabel('Iteration', 'FontSize', 10);
ylabel('$\rho$', 'Interpreter', 'latex');
legend({'Struct-Aware', 'Baseline', 'Optimal'}, 'Location', 'southeast');
title('Average Reward Convergence');
grid on;
hold off;

% 4. Q-value gap distribution
subplot(2, 3, 4);
if ~isempty(policyAnalysis.q_gaps)
    histogram(policyAnalysis.q_gaps(:, 3), 30, 'FaceColor', [0.3 0.5 0.8]);
    xlabel('$|Q(s,1) - Q(s,0)|$', 'Interpreter', 'latex');
    ylabel('Count');
    title('Q-Value Gap Distribution');
    
    % Mark threshold
    xline(0.05, 'r--', 'LineWidth', 1.5);
    text(0.06, max(ylim)*0.9, 'Threshold', 'Color', 'r');
    grid on;
end

% 5. Disagreement analysis
subplot(2, 3, 5);
if ~isempty(policyAnalysis.disagree_states)
    scatter(policyAnalysis.disagree_states(:, 2), policyAnalysis.disagree_states(:, 1), ...
        50, policyAnalysis.disagree_states(:, 5), 'filled');
    colorbar;
    xlabel('$T_s$', 'Interpreter', 'latex');
    ylabel('$T_r$', 'Interpreter', 'latex');
    title('Disagreement States (color = Q-gap)');
    xlim([0.5, K+0.5]);
    ylim([-0.5, K+0.5]);
    hold on;
    plot([1, K], [1, K], 'k--', 'LineWidth', 1);
    hold off;
    grid on;
else
    text(0.5, 0.5, 'No Disagreements!', 'HorizontalAlignment', 'center', ...
        'FontSize', 14, 'FontWeight', 'bold');
    axis([0 1 0 1]);
end

% 6. Summary statistics
subplot(2, 3, 6);
axis off;

stats_text = {
    sprintf('Distribution: %s', dist_name), ...
    '', ...
    sprintf('Agreement Rate: %.1f%%', policyAnalysis.agreement_rate * 100), ...
    sprintf('Total Disagreements: %d', size(policyAnalysis.disagree_states, 1)), ...
    sprintf('Significant (gap>0.05): %d', policyAnalysis.significant_disagree), ...
    '', ...
    sprintf('Min Visit Count: %d', min(diagnostics.visitCount(diagnostics.visitCount > 0))), ...
    sprintf('Max Visit Count: %d', max(diagnostics.visitCount(:))), ...
    '', ...
    sprintf('Final \\rho (Struct): %.4f', diagnostics.final_rho_struct), ...
    sprintf('Final \\rho (Base): %.4f', diagnostics.final_rho_base), ...
    sprintf('Optimal \\rho*: %.4f', rho_opt)
};

text(0.1, 0.9, stats_text, 'FontSize', 11, 'VerticalAlignment', 'top', ...
    'FontName', 'FixedWidth');
title('Summary Statistics');

sgtitle(sprintf('Convergence Diagnostics: %s', dist_name), 'FontSize', 14);

end


%% ==========================================================================
%  Q-VALUE ANALYSIS PLOTTING
% ==========================================================================

function plotQValueAnalysis(Qs, K, dist_name, ps, c, r)
% PLOTQVALUEANALYSIS  Visualize Q-values and optimal actions

subplot(1, 2, 1);
Q_diff = Qs(:, :, 2) - Qs(:, :, 1);  % Q(s,1) - Q(s,0)
imagesc(1:K, 0:K, Q_diff);
set(gca, 'YDir', 'normal');
colorbar;
colormap(gca, 'jet');

% Center colormap at zero
max_abs = max(abs(Q_diff(:)));
if max_abs > 0
    caxis([-max_abs, max_abs]);
end

xlabel('$T_s$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$T_r$', 'Interpreter', 'latex', 'FontSize', 12);
title('$Q(s,1) - Q(s,0)$', 'Interpreter', 'latex');

hold on;
% Mark zero contour (decision boundary)
contour(1:K, 0:K, Q_diff, [0 0], 'k-', 'LineWidth', 2);
% Mark structural constraint
plot([1, K], [1, K], 'w--', 'LineWidth', 1.5);
hold off;

subplot(1, 2, 2);
% Show absolute Q-values for action 1
imagesc(1:K, 0:K, Qs(:, :, 2));
set(gca, 'YDir', 'normal');
colorbar;
xlabel('$T_s$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$T_r$', 'Interpreter', 'latex', 'FontSize', 12);
title('$Q(s, a=1)$', 'Interpreter', 'latex');

hold on;
plot([1, K], [1, K], 'w--', 'LineWidth', 1.5);
hold off;

sgtitle(sprintf('Q-Value Analysis (%s): $p_s$=%.2f, $c/r$=%.2f', dist_name, ps, c/r), ...
    'Interpreter', 'latex', 'FontSize', 12);

end


%% ==========================================================================
%  POLICY DIFFERENCE HEATMAP
% ==========================================================================

function plotPolicyDifferenceHeatmap(piOpt, piLearned, Qs, K, titleStr, ps, c, r)
% PLOTPOLICYDIFFERENCEHEATMAP  Show where policies differ and why

% Create difference matrix: 0=agree, 1=disagree (opt=0,learn=1), -1=disagree (opt=1,learn=0)
diff_matrix = zeros(K+1, K);
gap_matrix = zeros(K+1, K);

for Tr = 0:K
    for Ts = 1:K
        gap_matrix(Tr+1, Ts) = abs(Qs(Tr+1, Ts, 2) - Qs(Tr+1, Ts, 1));
        
        if Tr > Ts
            diff_matrix(Tr+1, Ts) = NaN;  % Structural constraint region
        elseif piOpt(Tr+1, Ts) == piLearned(Tr+1, Ts)
            diff_matrix(Tr+1, Ts) = 0;  % Agreement
        elseif piOpt(Tr+1, Ts) == 0 && piLearned(Tr+1, Ts) == 1
            diff_matrix(Tr+1, Ts) = 1;  % Opt says wait, learned says send
        else
            diff_matrix(Tr+1, Ts) = -1;  % Opt says send, learned says wait
        end
    end
end

% Create custom colormap: blue (learned=send,opt=wait), white (agree), red (learned=wait,opt=send)
subplot(1, 2, 1);
imagesc(1:K, 0:K, diff_matrix);
set(gca, 'YDir', 'normal');

% Custom colormap
cmap = [0.2 0.4 0.8;    % Blue: learned=1, opt=0
        0.95 0.95 0.95; % White: agreement
        0.8 0.2 0.2];   % Red: learned=0, opt=1
colormap(gca, cmap);
caxis([-1 1]);

cb = colorbar;
cb.Ticks = [-0.67, 0, 0.67];
cb.TickLabels = {'Learned Wait, Opt Send', 'Agree', 'Learned Send, Opt Wait'};

xlabel('$T_s$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$T_r$', 'Interpreter', 'latex', 'FontSize', 12);
title('Policy Difference');

hold on;
plot([1, K], [1, K], 'k--', 'LineWidth', 1.5);
hold off;

% Show Q-gaps at disagreement locations
subplot(1, 2, 2);
% Mask agreement regions
masked_gaps = gap_matrix;
masked_gaps(diff_matrix == 0) = NaN;

imagesc(1:K, 0:K, masked_gaps);
set(gca, 'YDir', 'normal');
colorbar;
xlabel('$T_s$', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$T_r$', 'Interpreter', 'latex', 'FontSize', 12);
title('Q-Gap at Disagreement States');

hold on;
plot([1, K], [1, K], 'k--', 'LineWidth', 1.5);
hold off;

sgtitle({titleStr, sprintf('$K=%d$, $p_s=%.2f$, $c/r=%.2f$', K, ps, c/r)}, ...
    'Interpreter', 'latex', 'FontSize', 11);

end


%% ==========================================================================
%  HEATMAP PLOTTING FUNCTIONS
% ==========================================================================

function plotPolicyHeatmap(policy, K, titleStr, ps, c, r)
% PLOTPOLICYHEATMAP  Plot a heatmap of transmission/waiting regions

Ts_vals = 1:K;
Tr_vals = 0:K;

imagesc(Ts_vals, Tr_vals, policy);
set(gca, 'YDir', 'normal');

% Custom colormap
cmap = [0.2 0.4 0.8;   % Blue for Wait
        0.9 0.2 0.2];  % Red for Transmit
colormap(gca, cmap);
caxis([0 1]);

cb = colorbar;
cb.Ticks = [0.25, 0.75];
cb.TickLabels = {'Wait (a=0)', 'Transmit (a=1)'};
cb.FontSize = 10;

xlabel('$T_s$ (Sender TTL)', 'Interpreter', 'latex', 'FontSize', 12);
ylabel('$T_r$ (Receiver TTL)', 'Interpreter', 'latex', 'FontSize', 12);

title({titleStr, sprintf('$K=%d$, $p_s=%.2f$, $c/r=%.2f$', K, ps, c/r)}, ...
      'Interpreter', 'latex', 'FontSize', 11);

xlim([0.5, K+0.5]);
ylim([-0.5, K+0.5]);

hold on;

% Draw diagonal line (structural constraint boundary)
plot([1, K], [1, K], 'k--', 'LineWidth', 2);

% Add text annotations
text(K*0.3, K*0.85, '$T_r > T_s$: Forced Wait', 'Interpreter', 'latex', ...
     'FontSize', 10, 'Color', 'w', 'FontWeight', 'bold', ...
     'BackgroundColor', [0.2 0.2 0.2 0.7], 'EdgeColor', 'k');

text(K*0.6, K*0.2, '$T_r \leq T_s$: Decision Region', 'Interpreter', 'latex', ...
     'FontSize', 10, 'Color', 'k', 'FontWeight', 'bold', ...
     'BackgroundColor', [1 1 1 0.7], 'EdgeColor', 'k');

% Draw light grid
for i = 0:K
    plot([0.5, K+0.5], [i+0.5, i+0.5], '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.3);
end
for j = 1:K
    plot([j+0.5, j+0.5], [-0.5, K+0.5], '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.3);
end

hold off;
box on;

set(gca, 'XTick', 1:2:K);
set(gca, 'YTick', 0:2:K);

end


function plotPolicyHeatmapSubplot(policy, K, titleStr)
% PLOTPOLICYHEATMAPSUBPLOT  Simplified heatmap for subplots

if isempty(policy) || any(isnan(policy(:)))
    text(0.5, 0.5, 'No data', 'HorizontalAlignment', 'center');
    title(titleStr, 'Interpreter', 'latex');
    axis([0 1 0 1]);
    return;
end

Ts_vals = 1:K;
Tr_vals = 0:K;

imagesc(Ts_vals, Tr_vals, policy);
set(gca, 'YDir', 'normal');

cmap = [0.2 0.4 0.8;   % Blue for Wait
        0.9 0.2 0.2];  % Red for Transmit
colormap(gca, cmap);
caxis([0 1]);

xlabel('$T_s$', 'Interpreter', 'latex', 'FontSize', 10);
ylabel('$T_r$', 'Interpreter', 'latex', 'FontSize', 10);
title(titleStr, 'Interpreter', 'latex', 'FontSize', 11);

xlim([0.5, K+0.5]);
ylim([-0.5, K+0.5]);

hold on;
plot([1, K], [1, K], 'k--', 'LineWidth', 1.5);
hold off;

box on;

set(gca, 'XTick', [1, K/2, K]);
set(gca, 'YTick', [0, K/2, K]);

end