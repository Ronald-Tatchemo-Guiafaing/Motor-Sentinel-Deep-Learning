%% generer_dataset.m — TP2 Exercice 3 et 5
% Taille via config_dataset.m (defaut : 3000 = 3 x 1000)
% Compatible modèle classique (Fault_Switch) ou upgradé (Scenario_TP1)

clearvars -except MODE_VISIBLE
clc; close all;

cfg = config_dataset();
N_echantillons = cfg.N_ECHANTILLONS_PAR_CLASSE;
N_total = cfg.N_TOTAL;
nom_modele = 'Moteur_Asynchrone_Base2019bG';
dt = 50e-6;   % SampleTime powergui

chemin_cours = fileparts(fileparts(mfilename('fullpath')));
dossier_projet = fileparts(mfilename('fullpath'));
cd(chemin_cours);

figDir = fullfile(dossier_projet, 'figures_TP2');
if ~exist(figDir, 'dir'), mkdir(figDir); end
if ~exist(fullfile(dossier_projet, 'data'), 'dir')
    mkdir(fullfile(dossier_projet, 'data'));
end
if ~exist(fullfile(chemin_cours, 'data'), 'dir')
    mkdir(fullfile(chemin_cours, 'data'));
end

Dataset = cell(N_echantillons * 3, 1);
Labels  = zeros(N_echantillons * 3, 1);

load_system(nom_modele);
useScenario = ~isempty(find_system(nom_modele, 'SearchDepth', 1, 'Name', 'Scenario_TP1'));

eta_min = round(N_total * 0.25);  % ~15 s/sim -> estimation grossiere
fprintf('Debut generation — %d simulations (~%d-%d min estimes)...\n', ...
    N_total, round(eta_min * 0.6), round(eta_min * 1.5));
t_gen = tic;
if useScenario
    disp('Modele detecte : Scenario_TP1 (upgrade TP1)');
else
    disp('Modele detecte : Fault_Switch (classique)');
end

for classe = 0:2
    for i = 1:N_echantillons
        charge_val = 10 + rand();
        set_param([nom_modele '/Load_Nominal'], 'Value', num2str(charge_val));

        if useScenario
            set_param([nom_modele '/Scenario_TP1'], 'Value', num2str(classe + 1));
        else
            if classe == 0
                set_param([nom_modele '/Fault_Switch'], 'sw', '0');
            else
                set_param([nom_modele '/Fault_Switch'], 'sw', '1');
                if classe == 1
                    set_param([nom_modele '/Vibration_Defaut'], 'Frequency', '157');
                else
                    set_param([nom_modele '/Vibration_Defaut'], 'Frequency', '800');
                end
            end
        end

        simOut = sim(nom_modele, 'ReturnWorkspaceOutputs', 'on');
        signal = simOut.simout.signals.values(:, 1);
        signal = appliquer_signature_defaut(signal, classe, dt);

        idx = classe * N_echantillons + i;
        Dataset{idx} = signal;
        Labels(idx) = classe;

        if mod(i, 50) == 0 || i == N_echantillons
            done = classe * N_echantillons + i;
            elapsed = toc(t_gen);
            eta = elapsed / done * (N_total - done);
            fprintf('  [%d/%d] %.1f%% — ETA %.0f min\n', ...
                done, N_total, 100 * done / N_total, eta / 60);
        end
    end
    noms = {'Sain', 'Desequilibre', 'Roulement'};
    fprintf('Classe %s : OK\n', noms{classe + 1});
end

N_echantillons_par_classe = N_echantillons;
date_generation = datestr(now);
meta = struct('N_echantillons_par_classe', N_echantillons_par_classe, ...
    'N_total', N_total, 'date_generation', date_generation);
save(fullfile(dossier_projet, 'data', 'Dataset_Moteur.mat'), ...
    'Dataset', 'Labels', 'N_echantillons_par_classe', 'N_total', 'date_generation', '-v7.3');
save(fullfile(chemin_cours, 'data', 'Dataset_Moteur.mat'), ...
    'Dataset', 'Labels', 'N_echantillons_par_classe', 'N_total', 'date_generation', '-v7.3');
fprintf('Fichier cree : Dataset_Moteur.mat (%d signaux, -v7.3)\n', numel(Dataset));
fprintf('Duree generation : %.1f min\n', toc(t_gen) / 60);

c12 = corr(Dataset{N_echantillons + 1}(:), Dataset{2 * N_echantillons + 1}(:));
fprintf('Verification : corr classe1 vs classe2 = %.4f (doit etre < 0.99)\n', c12);

bdclose(nom_modele);

%% Exercice 5 : validation
figure('Visible', 'off');
subplot(3,1,1);
plot(Dataset{1});
title(['Classe 0 (Sain) - Label : ' num2str(Labels(1))]);
subplot(3,1,2);
plot(Dataset{N_echantillons + 1});
title(['Classe 1 (Desequilibre) - Label : ' num2str(Labels(N_echantillons + 1))]);
subplot(3,1,3);
plot(Dataset{2 * N_echantillons + 1});
title(['Classe 2 (Roulement) - Label : ' num2str(Labels(2 * N_echantillons + 1))]);
saveas(gcf, fullfile(figDir, 'TP2_Validation_Dataset.png'));
close(gcf);

disp('Labels uniques :');
disp(unique(Labels)');

function signal = appliquer_signature_defaut(signal, classe, dt)
% Signature physique des defauts (TP1) si Simulink ne les separe pas dans simout
    if classe == 0, return; end
    t = (0:numel(signal) - 1)' * dt;
    amp = 0.18 * rms(signal);
    if classe == 1
        signal = signal + amp * sin(157 * t);   % desequilibre (rad/s)
    elseif classe == 2
        signal = signal + amp * sin(800 * t);   % roulement (rad/s)
    end
end
