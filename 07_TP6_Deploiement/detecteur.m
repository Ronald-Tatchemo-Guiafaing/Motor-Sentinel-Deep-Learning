function Etat_Moteur = detecteur(Signal_Courant)
%DETECTEUR — Inférence temps réel (Séance 6)
% Entrée  : courant Phase A (scalaire flux Simulink ou vecteur complet)
% Sortie  : 1 = Sain | 2 = Déséquilibre | 3 = Roulement

    persistent net buf bufIdx lastEtat Fs imgSize

    WIN = 10000;
    if isempty(net)
        [net, Fs, imgSize] = charger_reseau();
        buf = zeros(WIN, 1);
        bufIdx = 1;
        lastEtat = uint8(1);
    end

    if numel(Signal_Courant) > 1
        Etat_Moteur = predire_depuis_signal(net, Signal_Courant(:), Fs, imgSize);
        return;
    end

    buf(bufIdx) = double(Signal_Courant);
    bufIdx = bufIdx + 1;

    if bufIdx > WIN
        Etat_Moteur = predire_depuis_signal(net, buf, Fs, imgSize);
        lastEtat = Etat_Moteur;
        bufIdx = 1;
    else
        Etat_Moteur = lastEtat;
    end
end

function [net, Fs, imgSize] = charger_reseau()
    dossier = fileparts(mfilename('fullpath'));
    chemin_cours = fileparts(dossier);
    fichiers = {
        fullfile(dossier, 'data', 'net_opt.mat')
        fullfile(chemin_cours, 'data', 'net_opt.mat')
    };
    for k = 1:numel(fichiers)
        if isfile(fichiers{k})
            S = load(fichiers{k});
            net = S.net_opt;
            if isfield(S, 'Fs'), Fs = S.Fs; else, Fs = 20000; end
            if isfield(S, 'imgSize'), imgSize = S.imgSize; else, imgSize = 227; end
            return;
        end
    end
    error('net_opt.mat introuvable. Lance TP6_Deploiement_Simulink.m.');
end

function etat = predire_depuis_signal(net, signal, Fs, imgSize)
    signal = signal(:);
    signal = signal(isfinite(signal));
    if numel(signal) < 256
        etat = uint8(1);
        return;
    end

    [S, ~, ~] = pspectrum(signal, Fs, 'spectrogram');
    img = imresize(mat2gray(10 * log10(S + eps)), [imgSize imgSize]);
    imgRGB = im2uint8(cat(3, img, img, img));

    pred = classify(net, imgRGB, 'ExecutionEnvironment', 'cpu');
    noms = {'Sain', 'Desequilibre', 'Roulement'};
    idx = find(strcmp(string(pred), noms), 1);
    if isempty(idx), idx = 1; end
    etat = uint8(idx);
end
