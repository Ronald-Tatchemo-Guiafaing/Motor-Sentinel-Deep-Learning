%% TP6_Deploiement_Simulink.m — Séance 6 : Le Dernier Kilomètre
% Mode visible : lancer LANCER_TP5_TP6_VISIBLE.m

global MODE_VISIBLE
if isempty(MODE_VISIBLE), MODE_VISIBLE = false; end
clearvars -except MODE_VISIBLE
clc;
if ~MODE_VISIBLE, close all; end

FIG_VIS = 'off'; PLOTS_MODE = 'none';
if MODE_VISIBLE
    FIG_VIS = 'on';
    PLOTS_MODE = 'training-progress';
    fprintf('Mode VISIBLE : entrainement CNN affiche a l''ecran.\n');
end

cnnRoot = fullfile(matlabroot, 'toolbox', 'nnet', 'cnn');
if exist(cnnRoot, 'dir'), addpath(genpath(cnnRoot)); end
try, gpuDevice([]); catch, end

dossier_projet = fileparts(mfilename('fullpath'));
chemin_cours = fileparts(dossier_projet);
addpath(dossier_projet);
cd(chemin_cours);

figDir = fullfile(dossier_projet, 'figures_TP6');
dataDir = fullfile(dossier_projet, 'data');
if ~exist(figDir, 'dir'), mkdir(figDir); end

nom_modele = 'Moteur_Asynchrone_Base2019bG';
objectif_acc = 98;
Fs = 1 / 50e-6;
imgSize = 227;

disp('============================================================');
disp('  SÉANCE 6 — LE DERNIER KILOMÈTRE');
disp('============================================================');

%% Étape 1 : Optimisation CNN (dropout + early stopping)
fprintf('\n--- Étape 1 : Optimisation CNN ---\n');

fichier_img = fullfile(dataDir, 'Dataset_Images_Pret.mat');
if ~isfile(fichier_img)
    error('Dataset_Images_Pret.mat introuvable. Lance TP3 puis TP5.');
end
load(fichier_img, 'XTrain', 'YTrain', 'XVal', 'YVal');
nTrain = size(XTrain, 4);
if nTrain > 500
    miniBatch = 32;
    maxEpochs = 40;
    fprintf('Dataset large : %d train / %d val — batch=%d, epochs=%d\n', ...
        nTrain, size(XVal, 4), miniBatch, maxEpochs);
else
    miniBatch = 16;
    maxEpochs = 60;
end

layers = [
    imageInputLayer([imgSize imgSize 3], 'Name', 'input')
    convolution2dLayer(3, 16, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
    dropoutLayer(0.2, 'Name', 'drop')
    fullyConnectedLayer(3, 'Name', 'fc')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

opts = trainingOptions('adam', ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatch, ...
    'InitialLearnRate', 1e-3, ...
    'ValidationData', {XVal, YVal}, ...
    'ValidationFrequency', 20, ...
    'ValidationPatience', 8, ...
    'Verbose', false, ...
    'Plots', PLOTS_MODE, ...
    'ExecutionEnvironment', 'cpu');

tTrain = tic;
net_opt = trainNetwork(XTrain, YTrain, layers, opts);
temps_train = toc(tTrain);

YPred = classify(net_opt, XVal, 'ExecutionEnvironment', 'cpu');
acc_val = 100 * mean(YPred == YVal);
fprintf('CNN optimise | Validation : %.1f %% | Temps entrainement : %.1f s\n', acc_val, temps_train);

figure('Visible', FIG_VIS);
confusionchart(YVal, YPred);
title(sprintf('TP6 — Matrice de confusion CNN (%.1f %%)', acc_val));
saveas(gcf, fullfile(figDir, 'TP6_Matrice_Confusion.png'));
close(gcf);

%% Étape 2 : Sauvegarde net_opt.mat
fprintf('\n--- Étape 2 : Sauvegarde net_opt ---\n');
save(fullfile(dataDir, 'net_opt.mat'), 'net_opt', 'Fs', 'imgSize');
save(fullfile(chemin_cours, 'data', 'net_opt.mat'), 'net_opt', 'Fs', 'imgSize');
fprintf('Sauvegarde : data/net_opt.mat\n');

%% Étape 3 : Test detecteur (hors Simulink)
fprintf('\n--- Étape 3 : Test detecteur.m ---\n');

fichier_moteur = fullfile(chemin_cours, 'data', 'Dataset_Moteur.mat');
if ~isfile(fichier_moteur)
    fichier_moteur = fullfile(dataDir, 'Dataset_Moteur.mat');
end
S_moteur = load(fichier_moteur, 'Dataset', 'Labels');
Dataset = S_moteur.Dataset;
Labels = S_moteur.Labels;
if isfield(S_moteur, 'N_echantillons_par_classe')
    N_par = S_moteur.N_echantillons_par_classe;
else
    N_par = numel(Dataset) / 3;
end

idx_test = [1, round(N_par) + 1, 2 * round(N_par) + 1];
noms = {'Sain', 'Desequilibre', 'Roulement'};
etats_detectes = zeros(1, 3);
latences_ms = zeros(1, 3);

for k = 1:3
    i = idx_test(k);
    t0 = tic;
    etats_detectes(k) = detecteur(Dataset{i}(:));
    latences_ms(k) = toc(t0) * 1000;
    fprintf('  Signal classe %d (%s) -> Etat IA = %d (attendu %d) | %.2f ms\n', ...
        Labels(i), noms{k}, etats_detectes(k), Labels(i) + 1, latences_ms(k));
end

acc_detecteur = 100 * mean(etats_detectes == (Labels(idx_test) + 1)');
latence_moy = mean(latences_ms);

%% Fausses alarmes sur validation (classe Sain mal classee)
fprintf('\n--- Métrique fausses alarmes ---\n');
idx_sain = find(Labels == 0);
nb_fausses = 0;
for k = 1:min(20, numel(idx_sain))
    et = detecteur(Dataset{idx_sain(k)}(:));
    if et ~= 1
        nb_fausses = nb_fausses + 1;
    end
end
taux_fausses = 100 * nb_fausses / min(20, numel(idx_sain));

%% Étape 4 : Simulations Software-in-the-Loop (detecteur sur simout)
fprintf('\n--- Étape 4 : Simulations temps reel ---\n');

load_system(nom_modele);
nettoyer_blocs_ia_simulink(nom_modele);
useScenario = ~isempty(find_system(nom_modele, 'SearchDepth', 1, 'Name', 'Scenario_TP1'));
etats_sim = zeros(1, 3);

for sc = 1:3
    if useScenario
        set_param([nom_modele '/Scenario_TP1'], 'Value', num2str(sc));
    else
        if sc == 1
            set_param([nom_modele '/Fault_Switch'], 'sw', '0');
        else
            set_param([nom_modele '/Fault_Switch'], 'sw', '1');
            if sc == 2
                set_param([nom_modele '/Vibration_Defaut'], 'Frequency', '157');
            else
                set_param([nom_modele '/Vibration_Defaut'], 'Frequency', '800');
            end
        end
    end

    simOut = sim(nom_modele, 'ReturnWorkspaceOutputs', 'on');
    signal = simOut.simout.signals.values(:, 1);
    etats_sim(sc) = detecteur(signal);
    fprintf('  Scenario %d (%s) -> Etat IA = %d\n', sc, noms{sc}, etats_sim(sc));
end
bdclose(nom_modele);

%% Étape 5 : Integration blocs Simulink (affichage manuel)
fprintf('\n--- Étape 5 : Integration Simulink (blocs IA) ---\n');
try
    integrer_ia_tp6();
catch ME
    warning('Integration Simulink : %s', ME.message);
    fprintf('  -> Ouvrir le modele et verifier le bloc Deep_Engine_Detecteur manuellement.\n');
end

%% Étape 6 : Bilan de performance
fprintf('\n--- Bilan de performance ---\n');
fprintf('  Precision validation CNN : %.1f %% (objectif %.0f %%)\n', acc_val, objectif_acc);
fprintf('  Latence moyenne detecteur : %.2f ms\n', latence_moy);
fprintf('  Taux fausses alarmes (echantillon sain) : %.1f %%\n', taux_fausses);

figure('Visible', FIG_VIS);
subplot(1, 2, 1);
bar([acc_val, 100 - taux_fausses], 'FaceColor', [0.2 0.5 0.8]);
set(gca, 'XTickLabel', {'Precision val.', 'Sains corrects'});
ylabel('%'); ylim([0 100]); title('Qualite IA');
hold on; yline(objectif_acc, 'r--', 'Objectif 98%'); grid on;

subplot(1, 2, 2);
bar(latences_ms, 'FaceColor', [0.9 0.5 0.2]);
set(gca, 'XTickLabel', noms);
ylabel('ms'); title('Latence inference'); grid on;

sgtitle('TP6 — Bilan de Performance');
saveas(gcf, fullfile(figDir, 'TP6_Bilan_Performance.png'));
close(gcf);

figure('Visible', FIG_VIS);
bar(categorical(noms), [etats_sim; etats_detectes]', 'grouped');
ylabel('Etat detecte (1=Sain, 2=Deseq., 3=Roul.)');
legend('Simulation Simulink', 'Dataset brut', 'Location', 'best');
title('TP6 — Validation Software-in-the-Loop');
saveas(gcf, fullfile(figDir, 'TP6_Simulation_Etats.png'));
close(gcf);

resultats_tp6 = struct( ...
    'accuracy_validation', acc_val, ...
    'latence_ms', latence_moy, ...
    'taux_fausses_alarmes', taux_fausses, ...
    'etats_simulink', etats_sim, ...
    'etats_dataset', etats_detectes, ...
    'temps_entrainement_sec', temps_train, ...
    'N_echantillons_total', numel(Dataset), ...
    'N_echantillons_par_classe', N_par);
save(fullfile(dataDir, 'Resultats_TP6_Deploiement.mat'), 'resultats_tp6');

%% Sauvegarde dataset complet TP6 (archive pour le rapport)
fprintf('\n--- Sauvegarde Dataset_TP6_Complet.mat ---\n');
archive_tp6 = fullfile(dataDir, 'Dataset_TP6_Complet.mat');
cfg = config_dataset();
date_archive = datestr(now);
save(archive_tp6, 'Dataset', 'Labels', 'N_par', 'cfg', ...
    'XTrain', 'YTrain', 'XVal', 'YVal', 'net_opt', 'Fs', 'imgSize', ...
    'resultats_tp6', 'date_archive', '-v7.3');
fprintf('Archive TP6 : %s (%.1f Mo)\n', archive_tp6, ...
    dir(archive_tp6).bytes / 1e6);

if MODE_VISIBLE
    try
        open_system(nom_modele);
        st = get_param(nom_modele, 'StopTime');
        fprintf('Simulink ouvert — StopTime = %s s (la simulation s''arrete a cette duree, c''est normal).\n', st);
        fprintf('Verifie Display_Etat_Moteur puis clique Run si tu veux relancer.\n');
    catch
    end
end

disp('============================================================');
disp('  TP6 : OK');
disp(['  Figures : ', figDir]);
disp('============================================================');

function nettoyer_blocs_ia_simulink(modelName)
    blocs = {'Deep_Engine_Detecteur', 'Display_Etat_Moteur', 'Etat_IA_Log'};
    for k = 1:numel(blocs)
        p = [modelName '/' blocs{k}];
        if ~isempty(find_system(modelName, 'SearchDepth', 1, 'Name', blocs{k}))
            try, delete_block(p); catch, end
        end
    end
    try, save_system(modelName); catch, end
end
