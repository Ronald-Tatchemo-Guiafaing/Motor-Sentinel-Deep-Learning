%% TP3_Traitement_Signal.m — Moteur Sentinel
% Du signal physique à l'image (Data Engineering)
% Prérequis : Dataset_Moteur.mat (TP2)

clearvars -except MODE_VISIBLE
clc; close all;

dossier_projet = fileparts(mfilename('fullpath'));
chemin_cours = fileparts(dossier_projet);
cd(chemin_cours);

figDir = fullfile(dossier_projet, 'figures_TP3');
if ~exist(figDir, 'dir'), mkdir(figDir); end
if ~exist(fullfile(dossier_projet, 'data'), 'dir')
    mkdir(fullfile(dossier_projet, 'data'));
end

%% Chargement du dataset TP2
fichier_data = fullfile(chemin_cours, 'data', 'Dataset_Moteur.mat');
if ~isfile(fichier_data)
    fichier_data = fullfile(dossier_projet, 'data', 'Dataset_Moteur.mat');
end
if ~isfile(fichier_data)
    error('Dataset_Moteur.mat introuvable. Lance generer_dataset.m (TP2) d abord.');
end
load(fichier_data, 'Dataset', 'Labels');
fprintf('Dataset charge : %d signaux\n', numel(Dataset));
N_par_classe = numel(Dataset) / 3;
idx_roulement = 2 * N_par_classe + 1;

Fs = 1 / 50e-6;   % 20 kHz — SampleTime powergui du modele Simulink
imgSize = 227;    % taille image pour CNN (ex. AlexNet)

% puissance -> dB avant normalisation (sinon 1 pic domine et image noire)
toSpectroImage = @(S) imresize(mat2gray(10*log10(S + eps)), [imgSize imgSize]);

% Exemples pour l analyse (1 = sain, 2*N+1 = roulement)
signal_sain = Dataset{1}(:);
signal_defaut = Dataset{idx_roulement}(:);
N = length(signal_sain);
t = (0:N-1) / Fs;

%% ============================================================
%% EXERCICE 1 : Temporel vs Frequentiel (FFT)
%% ============================================================

% --- Domaine temporel ---
figure('Name', 'TP3 - Temporel');
subplot(2,1,1);
plot(t, signal_sain);
title('Signal Sain (classe 0)');
xlabel('Temps (s)'); ylabel('Courant (A)'); grid on;

subplot(2,1,2);
plot(t, signal_defaut);
title('Signal Defaut Roulement (classe 2)');
xlabel('Temps (s)'); ylabel('Courant (A)'); grid on;
saveas(gcf, fullfile(figDir, 'TP3_Temporel_Sain_vs_Defaut.png'));

% --- Domaine frequentiel (FFT) ---
Y_sain = fft(signal_sain);
Y_defaut = fft(signal_defaut);
f = (0:N-1) * (Fs/N);

% on garde la moitie positive du spectre
demi = 1:floor(N/2);

figure('Name', 'TP3 - FFT');
subplot(2,1,1);
plot(f(demi), abs(Y_sain(demi)));
title('FFT - Moteur Sain');
xlabel('Frequence (Hz)'); ylabel('Amplitude'); grid on;
xlim([0 200]);

subplot(2,1,2);
plot(f(demi), abs(Y_defaut(demi)));
title('FFT - Defaut Roulement');
xlabel('Frequence (Hz)'); ylabel('Amplitude'); grid on;
xlim([0 200]);
saveas(gcf, fullfile(figDir, 'TP3_FFT_Sain_vs_Defaut.png'));

%% ============================================================
%% EXERCICE 2 : Spectrogrammes (images 2D)
%% ============================================================

[S_sain, ~, ~] = pspectrum(signal_sain, Fs, 'spectrogram');
[S_defaut, ~, ~] = pspectrum(signal_defaut, Fs, 'spectrogram');

img_sain = toSpectroImage(S_sain);
img_defaut = toSpectroImage(S_defaut);

figure('Name', 'TP3 - Spectrogrammes');
subplot(1,2,1);
imagesc(img_sain); colormap jet; colorbar;
title('Spectrogramme Sain');

subplot(1,2,2);
imagesc(img_defaut); colormap jet; colorbar;
title('Spectrogramme Defaut');
saveas(gcf, fullfile(figDir, 'TP3_Spectrogrammes_Sain_vs_Defaut.png'));

%% ============================================================
%% EXERCICE 3 : Boucle — N signaux -> images RGB
%% ============================================================

numSignals = length(Dataset);
imgArray = zeros(imgSize, imgSize, 3, numSignals, 'uint8');

disp('Traitement des signaux en cours...');
t_tp3 = tic;
for i = 1:numSignals
    signal = Dataset{i}(:);

    [S, ~, ~] = pspectrum(signal, Fs, 'spectrogram');
    img = toSpectroImage(S);

    % RGB : 3 canaux identiques (CNN pre-entraines)
    imgRGB = cat(3, img, img, img);
    imgRGB = im2uint8(imgRGB);

    imgArray(:,:,:,i) = imgRGB;

    if mod(i, 200) == 0 || i == numSignals
        fprintf('  TP3 : %d/%d images (%.1f min)\n', i, numSignals, toc(t_tp3) / 60);
    end
end
disp('Traitement termine.');

%% ============================================================
%% SECTION 6 : Training set (80% / 20%)
%% ============================================================

classes = ["Sain", "Desequilibre", "Roulement"];
Y = categorical(Labels, [0, 1, 2], classes);

cv = cvpartition(Y, 'HoldOut', 0.2);
idxTrain = training(cv);
idxVal = test(cv);

XTrain = imgArray(:,:,:, idxTrain);
YTrain = Y(idxTrain);
XVal = imgArray(:,:,:, idxVal);
YVal = Y(idxVal);

save(fullfile(dossier_projet, 'data', 'Dataset_Images_Pret.mat'), ...
    'XTrain', 'YTrain', 'XVal', 'YVal', 'classes', 'imgSize', 'Fs', ...
    'N_par_classe', 'numSignals', '-v7.3');
disp('Dataset sauvegarde. Pret pour l IA !');
fprintf('Train : %d images | Validation : %d images\n', sum(idxTrain), sum(idxVal));

%% Apercu grille (3 classes)
figure('Name', 'TP3 - Apercu images');
for k = 1:3
    idx = find(Labels == k-1, 1, 'first');
    subplot(1,3,k);
    imshow(imgArray(:,:,:,idx));
    title(sprintf('Classe %d - %s', k-1, classes(k)));
end
saveas(gcf, fullfile(figDir, 'TP3_Apercu_3_Classes.png'));

disp('TP3 : OK');
