%% TP4_Feature_Engineering.m — Moteur Sentinel
% Fiche TP4 : Feature Engineering (RMS, Kurtosis, Peak)
% Prerequis : Dataset_Moteur.mat (TP2)

clear; clc; close all;

dossier_projet = fileparts(mfilename('fullpath'));
chemin_cours = fileparts(dossier_projet);
cd(chemin_cours);

figDir = fullfile(dossier_projet, 'figures_TP4');
if ~exist(figDir, 'dir'), mkdir(figDir); end
if ~exist(fullfile(dossier_projet, 'data'), 'dir')
    mkdir(fullfile(dossier_projet, 'data'));
end

%% Preparation — chargement du dataset
fichier_data = fullfile(chemin_cours, 'data', 'Dataset_Moteur.mat');
if ~isfile(fichier_data)
    fichier_data = fullfile(dossier_projet, 'data', 'Dataset_Moteur.mat');
end
if ~isfile(fichier_data)
    error('Dataset_Moteur.mat introuvable. Lance generer_dataset.m (TP2) d abord.');
end

load(fichier_data);   % Dataset, Labels

numSignals = length(Dataset);
Features = zeros(numSignals, 3);
Labels_num = zeros(numSignals, 1);

%% ============================================================
%% EXERCICE 1 : Calcul manuel des indicateurs
%% ============================================================

disp('Calcul des features en cours...');

for i = 1:numSignals
    signal = Dataset{i};
    signal = signal(:);

    % TODO 1 — RMS (indice d'energie)
    val_rms = sqrt(mean(signal .^ 2));

    % TODO 2 — Peak (valeur max absolue)
    val_peak = max(abs(signal));

    % TODO 3 — Kurtosis (indice de chocs)
    % MATLAB : kurtosis d'exces (loi normale -> 0)
    val_kurt = kurtosis(signal);

    Features(i, :) = [val_rms, val_kurt, val_peak];
    Labels_num(i) = Labels(i);
end

disp('Calcul termine.');
fprintf('Exemple signal 1 : RMS=%.3f  Kurtosis=%.3f  Peak=%.3f\n', Features(1, :));

%% ============================================================
%% EXERCICE 2 : Visualisation des clusters
%% ============================================================

% Affichage graphique 2D (RMS vs Kurtosis)
figure('Name', 'TP4 - Separation des classes');
gscatter(Features(:, 1), Features(:, 2), Labels_num, 'brg', 'ox+');
xlabel('RMS (Energie)');
ylabel('Kurtosis (Chocs)');
title('Separation des classes par Features');
legend('Classe 0 (Sain)', 'Classe 1 (Desequilibre)', 'Classe 2 (Roulement)');
grid on;
saveas(gcf, fullfile(figDir, 'TP4_Separation_Classes.png'));

% Complement : RMS vs Kurtosis (vue clusters)
figure('Name', 'TP4 - Clusters RMS vs Kurtosis');
gscatter(Features(:, 1), Features(:, 2), Labels_num, 'brg', 'o');
xlabel('RMS (Energie)'); ylabel('Kurtosis (Chocs)');
title('Clusters — RMS vs Kurtosis');
legend('Classe 0 (Sain)', 'Classe 1 (Desequilibre)', 'Classe 2 (Roulement)');
grid on;
saveas(gcf, fullfile(figDir, 'TP4_Clusters_RMS_Kurtosis.png'));

% Complement : RMS vs Peak
figure('Name', 'TP4 - Clusters RMS vs Peak');
gscatter(Features(:, 1), Features(:, 3), Labels_num, 'brg', 'o');
xlabel('RMS (Energie)'); ylabel('Peak (Crete)');
title('Clusters — RMS vs Peak');
legend('Classe 0 (Sain)', 'Classe 1 (Desequilibre)', 'Classe 2 (Roulement)');
grid on;
saveas(gcf, fullfile(figDir, 'TP4_Clusters_RMS_Peak.png'));

%% ============================================================
%% EXERCICE 3 : Standardisation Z-Score
%% ============================================================

% Calcul de la moyenne (mu) et de l'ecart-type (sigma) sur tout le dataset
mu = mean(Features);
sigma = std(Features);

% Application de la formule Z-Score
Features_Norm = (Features - mu) ./ sigma;

% Verification
disp('--- Verification Normalisation ---');
fprintf('Moyenne RMS (devrait etre ~0) : %.4f\n', mean(Features_Norm(:, 1)));
fprintf('Ecart-type RMS (devrait etre ~1) : %.4f\n', std(Features_Norm(:, 1)));

% Clusters apres normalisation Z-Score
figure('Name', 'TP4 - Clusters normalises');
gscatter(Features_Norm(:, 1), Features_Norm(:, 2), Labels_num, 'brg', 'o');
xlabel('RMS normalise'); ylabel('Kurtosis normalise');
title('Clusters apres Z-Score');
legend('Classe 0 (Sain)', 'Classe 1 (Desequilibre)', 'Classe 2 (Roulement)');
grid on;
saveas(gcf, fullfile(figDir, 'TP4_Clusters_Normalises.png'));

%% Question fiche TP4 :
% Pourquoi interdit d'utiliser les donnees de Test pour mu et sigma ?
% -> Pour eviter la fuite d'information (data leakage) sur le futur.

%% ============================================================
%% EXERCICE 4 : Conversion en One-Hot
%% ============================================================

numClasses = 3;
Labels_OneHot = zeros(numSignals, numClasses);

for i = 1:numSignals
    % Labels de 0 a 2 -> index MATLAB a partir de 1
    currentLabel = Labels_num(i) + 1;
    Labels_OneHot(i, currentLabel) = 1;
end

% Visualisation pour verifier
disp('Premiers labels originaux :');
disp(Labels_num(1:5));
disp('Correspondance en One-Hot :');
disp(Labels_OneHot(1:5, :));

%% ============================================================
%% SECTION 6 : Sauvegarde et conclusion
%% ============================================================

save(fullfile(dossier_projet, 'data', 'Dataset_Features_Pret.mat'), ...
    'Features_Norm', 'Labels_OneHot', 'Labels_num');
disp('Dataset Features sauvegarde.');

disp('TP4 : OK');
