%% TP1 — Motor Sentinel · Simulation + plots + data export
%
% Usage: cd to this folder, then >> TP1_Complet

clear; clc; close all;

%% Paths
rootDir   = fileparts(mfilename('fullpath'));
if strcmp(rootDir, fileparts(rootDir)) || contains(rootDir, 'MATLAB-Course') && ...
        ~contains(rootDir, 'Projet_DL')
    rootDir = fullfile(fileparts(rootDir), 'Projet_DL_Guiafaing_Ronald');
end
courseDir = fileparts(rootDir);
modelName = 'Moteur_Asynchrone_Base2019bG';
modelFile = fullfile(courseDir, [modelName '.slx']);
dataDir   = fullfile(rootDir, 'data');
figDir    = fullfile(rootDir, 'figures');

if ~exist(dataDir, 'dir'), mkdir(dataDir); end
if ~exist(figDir, 'dir'), mkdir(figDir); end

if ~isfile(modelFile)
    error('Model not found: %s', modelFile);
end

addpath(fullfile(rootDir, 'helpers'));
cd(courseDir);

fprintf('=== TP1 Motor Sentinel ===\n');
fprintf('Model: %s\n\n', modelFile);

load_system(modelName);
set_param(modelName, 'StopTime', '1.0');

if isempty(find_system(modelName, 'SearchDepth', 1, 'Name', 'Scenario_TP1'))
    fprintf('Upgrading Simulink model (Scenario_TP1)...\n');
    upgrade_modele_TP1();
    load_system(modelName);
end
if exist('repair_simulink_wiring', 'file') == 2
    % skip auto-repair loop — call repair_simulink_wiring manually if needed
end

scenarioPath = [modelName '/Scenario_TP1'];

%% ----- Section 3: model analysis -----
fprintf('--- Section 3: Model analysis ---\n');
fprintf('1. Three-phase source: 50 Hz, 380 V (line-line RMS, powerlib)\n');
fprintf('2. Inertia/load: MoteurAsync block (J, F) + torque Tm via Scenario_TP1\n');
fprintf('3. BusSelect: extracts is_a, is_b, is_c, wm from motor measurements\n');
fprintf('4. To Workspace: simout (4 mux signals) + t (clock) — Structure With Time\n\n');

%% ----- Simulation 1: HEALTHY -----
fprintf('Simulation 1/3 — HEALTHY (Scenario_TP1 = 1)...\n');
set_param(scenarioPath, 'Value', '1');
simout = run_simulation(modelName);
data_healthy = simout;
assignin('base', 'data_healthy', simout);
[t_s, ia_s, ~, ~, wm_s] = extract_simout(simout);
save(fullfile(dataDir, 'TP1_Data.mat'), 'data_healthy', '-v7.3');

%% ----- Simulation 2: UNBALANCE -----
fprintf('Simulation 2/3 — UNBALANCE (Scenario_TP1 = 2, f=157 rad/s)...\n');
set_param(scenarioPath, 'Value', '2');
simout = run_simulation(modelName);
data_fault_unbalance = simout;
assignin('base', 'data_fault_unbalance', simout);
[t_u, ia_u, ~, ~, wm_u] = extract_simout(simout);
data_fault = simout;
save(fullfile(dataDir, 'TP1_Data.mat'), 'data_healthy', 'data_fault', '-append');

%% ----- Simulation 3: BEARING FAULT -----
fprintf('Simulation 3/3 — BEARING FAULT (Scenario_TP1 = 3, f=800 rad/s)...\n');
set_param(scenarioPath, 'Value', '3');
simout = run_simulation(modelName);
data_fault_bearing = simout;
assignin('base', 'data_fault_bearing', simout);
[t_b, ia_b, ~, ~, wm_b] = extract_simout(simout);
save(fullfile(dataDir, 'TP1_Data.mat'), 'data_fault_bearing', '-append');

fprintf('\nData saved: %s\n\n', fullfile(dataDir, 'TP1_Data.mat'));

%% ----- Section 4: healthy case plots -----
fprintf('--- Section 4: Nominal case figures ---\n');
fprintf('simout size (healthy): %d samples x %d signals\n', ...
    size(data_healthy.signals.values, 1), size(data_healthy.signals.values, 2));

fig1 = figure('Name', 'TP1_Fig01_Healthy_PhaseA', 'Color', 'w', 'Position', [100 100 900 400]);
plot(t_s, ia_s, 'b', 'LineWidth', 1.1);
grid on; xlabel('Time (s)'); ylabel('i_{s,a} (A)');
title('HEALTHY — Stator current phase A (startup + steady state)');
xline(0.1, 'r--', 'Breaker closure ~0.04–0.1 s', 'LabelVerticalAlignment', 'bottom');
saveas(fig1, fullfile(figDir, 'Fig01_Healthy_PhaseA_Current.png'));

fig2 = figure('Name', 'TP1_Fig02_Startup_Zoom', 'Color', 'w', 'Position', [120 120 900 400]);
idxStart = t_s <= 0.25;
plot(t_s(idxStart), ia_s(idxStart), 'b', 'LineWidth', 1.2);
grid on; xlabel('Time (s)'); ylabel('i_{s,a} (A)');
title('Startup zoom (0–0.25 s) — inrush current (locked rotor / max slip)');
saveas(fig2, fullfile(figDir, 'Fig02_Startup_Zoom.png'));

fig3 = figure('Name', 'TP1_Fig03_Steady_State', 'Color', 'w', 'Position', [140 140 900 400]);
idxSS = t_s >= 0.3;
plot(t_s(idxSS), ia_s(idxSS), 'b', 'LineWidth', 1.1);
grid on; xlabel('Time (s)'); ylabel('i_{s,a} (A)');
title('Steady state — quasi-sinusoidal current');
saveas(fig3, fullfile(figDir, 'Fig03_Steady_State.png'));

fig4 = figure('Name', 'TP1_Fig04_Rotor_Speed', 'Color', 'w', 'Position', [160 160 900 350]);
plot(t_s, wm_s, 'k', 'LineWidth', 1.1);
grid on; xlabel('Time (s)'); ylabel('\omega_m (rad/s)');
title('Rotor speed — stabilization to steady state');
saveas(fig4, fullfile(figDir, 'Fig04_Rotor_Speed_Healthy.png'));

%% ----- Section 5: fault comparison -----
fig5 = figure('Name', 'TP1_Fig05_Healthy_vs_Unbalance', 'Color', 'w', 'Position', [180 180 950 450]);
plot(t_u, ia_s, 'b', 'DisplayName', 'Healthy'); hold on;
plot(t_u, ia_u, 'r', 'DisplayName', 'Unbalance (f=157 rad/s)');
grid on; xlabel('Time (s)'); ylabel('i_{s,a} (A)');
title('Section 5.1 — Current amplitude: healthy vs unbalance (visible ripple)');
legend('Location', 'best');
saveas(fig5, fullfile(figDir, 'Fig05_Healthy_vs_Unbalance.png'));

idxSSu = t_u >= 0.35;
idxSSs = t_s >= 0.35;
fig6 = figure('Name', 'TP1_Fig06_Unbalance_Envelope', 'Color', 'w', 'Position', [200 200 950 400]);
plot(t_u(idxSSu), ia_u(idxSSu), 'r', 'LineWidth', 1.1); hold on;
plot(t_s(idxSSs), ia_s(idxSSs), 'b--', 'LineWidth', 0.9);
grid on; xlabel('Time (s)'); ylabel('i_{s,a} (A)');
title('Steady state — amplitude modulation (mechanical unbalance)');
legend('Fault', 'Healthy (ref.)', 'Location', 'best');
saveas(fig6, fullfile(figDir, 'Fig06_Unbalance_Envelope.png'));

fig7 = figure('Name', 'TP1_Fig07_Bearing_Time', 'Color', 'w', 'Position', [220 220 950 400]);
plot(t_b, ia_b, 'Color', [0.85 0.33 0.10], 'LineWidth', 1.0);
grid on; xlabel('Time (s)'); ylabel('i_{s,a} (A)');
title('Section 5.2 — Bearing fault (f=800 rad/s) — time domain');
saveas(fig7, fullfile(figDir, 'Fig07_Bearing_Time_Domain.png'));

idxZoom = t_b >= 0.5 & t_b <= 0.55;
fig8 = figure('Name', 'TP1_Fig08_Bearing_Zoom', 'Color', 'w', 'Position', [240 240 950 400]);
plot(t_b(idxZoom), ia_b(idxZoom), 'Color', [0.85 0.33 0.10], 'LineWidth', 1.2);
grid on; xlabel('Time (s)'); ylabel('i_{s,a} (A)');
title('Time zoom — HF fault hard to read by eye');
saveas(fig8, fullfile(figDir, 'Fig08_Bearing_Zoom.png'));

%% FFT (Section 5.2)
dt = mean(diff(t_b));
Fs = 1 / dt;
idxFFT_b = t_b >= 0.4;
sigB = ia_b(idxFFT_b) - mean(ia_b(idxFFT_b));
Nb = numel(sigB);
YfB = abs(fft(sigB));
fB  = (0:Nb-1) * (Fs / Nb);
halfB = 1:floor(Nb/2);

fig9 = figure('Name', 'TP1_Fig09_FFT_Bearing', 'Color', 'w', 'Position', [260 260 950 400]);
plot(fB(halfB), YfB(halfB), 'k', 'LineWidth', 1.0);
grid on; xlim([0 min(2000, Fs/2)]);
xlabel('Frequency (Hz)'); ylabel('|FFT|');
title('Spectrum — bearing fault more visible in frequency than on Scope');
saveas(fig9, fullfile(figDir, 'Fig09_FFT_Bearing.png'));

fig10 = figure('Name', 'TP1_Fig10_FFT_Comparison', 'Color', 'w', 'Position', [280 280 950 450]);
idxFFT_s = t_s >= 0.4;
idxFFT_u = t_u >= 0.4;
sigS = ia_s(idxFFT_s) - mean(ia_s(idxFFT_s));
sigU = ia_u(idxFFT_u) - mean(ia_u(idxFFT_u));
dt2 = mean(diff(t_s));
Fs2 = 1 / dt2;
Ns = numel(sigS);
Nu = numel(sigU);
fS = (0:Ns-1) * (Fs2 / Ns);
fU = (0:Nu-1) * (Fs2 / Nu);
halfS = 1:floor(Ns/2);
halfU = 1:floor(Nu/2);
YfS = abs(fft(sigS));
YfU = abs(fft(sigU));
plot(fS(halfS), YfS(halfS), 'b', 'DisplayName', 'Healthy'); hold on;
plot(fU(halfU), YfU(halfU), 'r', 'DisplayName', 'Unbalance');
grid on; xlim([0 200]); xlabel('Frequency (Hz)'); ylabel('|FFT|');
title('Spectral comparison (0–200 Hz) — healthy vs unbalance');
legend('Location', 'best');
saveas(fig10, fullfile(figDir, 'Fig10_FFT_Healthy_vs_Unbalance.png'));

%% ----- Section 7: synthesis (console) -----
fprintf('\n--- Section 7: Synthesis (copy into report) ---\n');
fprintf(['Stator current reflects mechanical state because electromagnetic torque\n', ...
    'couples the electrical and mechanical domains: any load fluctuation\n', ...
    '(unbalance, bearing fault) changes the resisting torque seen by the rotor,\n', ...
    'which modulates inrush current and the stator component (MCSA).\n\n']);

fprintf('--- Section 8: Automation ---\n');
fprintf('Use this script + sim() + set_param() + for loop for N samples.\n');
fprintf('No need to click Run 1000 times.\n\n');

fprintf('=== TP1 complete ===\n');
fprintf('Figures: %s\n', figDir);
fprintf('Data: %s\n', fullfile(dataDir, 'TP1_Data.mat'));
