%% Sprint_Modeling.m — TP5 Moteur Sentinel
% Sprint : MLP vs CNN vs LSTM — objectif professeur : >= 98 %% accuracy
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
    fprintf('Mode VISIBLE : courbes d''entrainement affichees.\n');
end

init_deep_learning_paths();

dossier_projet = fileparts(mfilename('fullpath'));
chemin_cours = fileparts(dossier_projet);
cd(chemin_cours);

figDir = fullfile(dossier_projet, 'figures_TP5');
if ~exist(figDir, 'dir'), mkdir(figDir); end

dataDir = fullfile(dossier_projet, 'data');
classes = ["Sain", "Desequilibre", "Roulement"];
objectif_acc = 98;

verifier_qualite_dataset(chemin_cours, dossier_projet);

disp('============================================================');
disp('  TP5 — SPRINT DE MODÉLISATION (MLP / CNN / LSTM)');
disp(sprintf('  Objectif accuracy validation : >= %.0f %%', objectif_acc));
disp('============================================================');

resultats = struct();
resultats.MLP = run_MLP_baseline(dataDir, classes, figDir, objectif_acc);
resultats.CNN = run_CNN_heavy(dataDir, figDir, objectif_acc);
resultats.LSTM = run_LSTM_sequence(chemin_cours, dossier_projet, classes, figDir, objectif_acc);

afficher_synthese(resultats, figDir, objectif_acc);

resultats_save = resultats;
champs_net = {'net', 'info'};
for k = 1:numel(champs_net)
    if isfield(resultats_save.MLP, champs_net{k}), resultats_save.MLP = rmfield(resultats_save.MLP, champs_net{k}); end
    if isfield(resultats_save.CNN, champs_net{k}), resultats_save.CNN = rmfield(resultats_save.CNN, champs_net{k}); end
    if isfield(resultats_save.LSTM, champs_net{k}), resultats_save.LSTM = rmfield(resultats_save.LSTM, champs_net{k}); end
end
save(fullfile(dataDir, 'Resultats_TP5_Sprint.mat'), 'resultats_save');
disp(['Résultats sauvegardés : ', fullfile(dataDir, 'Resultats_TP5_Sprint.mat')]);
disp('TP5 : OK');

%% ============================================================
%% Fonctions locales
%% ============================================================

function init_deep_learning_paths()
    cnnRoot = fullfile(matlabroot, 'toolbox', 'nnet', 'cnn');
    if exist(cnnRoot, 'dir')
        addpath(genpath(cnnRoot));
    end
    try
        gpuDevice([]);  % force CPU (evite blocage recompilation CUDA)
    catch
    end
end

function verifier_qualite_dataset(chemin_cours, dossier_projet)
    fichier = fullfile(chemin_cours, 'data', 'Dataset_Moteur.mat');
    if ~isfile(fichier)
        fichier = fullfile(dossier_projet, 'data', 'Dataset_Moteur.mat');
    end
    if ~isfile(fichier), return; end
    S = load(fichier, 'Dataset', 'N_echantillons_par_classe');
    if isfield(S, 'N_echantillons_par_classe')
        N = S.N_echantillons_par_classe;
    else
        N = numel(S.Dataset) / 3;
    end
    c = corr(S.Dataset{N + 1}(:), S.Dataset{2 * N + 1}(:));
    if c > 0.99
        warning(['Dataset : classes 1 et 2 quasi identiques (corr=%.3f). ', ...
            'Lance generer_dataset.m (f=800) puis TP3 et TP4 pour viser 98%%.'], c);
    end
end

function res = run_MLP_baseline(dataDir, classes, figDir, objectif_acc)
    fprintf('\n--- PARTIE A : BASELINE MLP ---\n');

    load(fullfile(dataDir, 'Dataset_Features_Pret.mat'), 'Features_Norm', 'Labels_num');

    Y = categorical(Labels_num, [0 1 2], classes);
    cv = cvpartition(Y, 'HoldOut', 0.2);
    idxTrain = training(cv);
    idxVal = test(cv);

    XTrain = Features_Norm(idxTrain, :);
    YTrain = Labels_num(idxTrain);
    XVal = Features_Norm(idxVal, :);
    YVal = Labels_num(idxVal);

    configs = {[64 32], [128 64], [128 64 32], [256 128 64]};
    epochs_list = [300, 500, 800];

    best_acc = 0;
    best_net = [];
    best_temps = 0;
    best_cfg = [];
    best_info = [];

    tGlobal = tic;
    for cfg = configs
        for ep = epochs_list
            net = patternnet(cfg{1});
            net.trainParam.showWindow = false;
            net.trainParam.showCommandLine = false;
            net.trainParam.epochs = ep;
            net.trainParam.goal = 1e-6;
            net.divideParam.trainRatio = 1;
            net.divideParam.valRatio = 0;
            net.divideParam.testRatio = 0;

            Ytr_oh = ind2vec(YTrain' + 1);
            net = train(net, XTrain', Ytr_oh);

            Ypred = vec2ind(net(XVal'));
            acc = 100 * mean(Ypred' - 1 == YVal);

            if acc > best_acc
                best_acc = acc;
                best_net = net;
                best_cfg = cfg{1};
            end
            fprintf('  MLP essai %s / %d epochs -> %.1f %%\n', mat2str(cfg{1}), ep, acc);

            if acc >= objectif_acc, break; end
        end
        if best_acc >= objectif_acc, break; end
    end
    temps = toc(tGlobal);

    if best_acc < objectif_acc
        fprintf('  >> MLP sous %.0f %% : features limitées (voir TP4). CNN recommandé.\n', objectif_acc);
    end

    Ypred = vec2ind(best_net(XVal'));
    YVal_cat = categorical(YVal, [0 1 2], classes);
    YPred_cat = categorical(Ypred' - 1, [0 1 2], classes);

    figure('Name', 'TP5 - MLP Confusion', 'Visible', get_fig_vis());
    confusionchart(YVal_cat, YPred_cat);
    title(sprintf('MLP — Accuracy : %.1f %% (objectif %.0f %%)', best_acc, objectif_acc));
    saveas(gcf, fullfile(figDir, 'TP5_MLP_Confusion.png'));
    close(gcf);

  sauver_courbe_patternnet(best_net, figDir, 'TP5_MLP_Training.png');

    nParams = compter_parametres_patternnet(best_net);
    fprintf('MLP FINAL | Accuracy : %.1f %% | Temps : %.1f s | Paramètres : %d\n', best_acc, temps, nParams);

    res.modele = 'MLP';
    res.type_donnees = 'Features (Tabulaire / RMS, Kurtosis, Peak)';
    res.accuracy = best_acc;
    res.temps_sec = temps;
    res.nb_params = nParams;
    res.complexite = 'Faible';
    res.config = best_cfg;
    res.objectif_atteint = best_acc >= objectif_acc;
    res.commentaire = 'Baseline rapide ; dépend des features ingénieur (TP4).';
    res.net = best_net;
end

function res = run_CNN_heavy(dataDir, figDir, objectif_acc)
    fprintf('\n--- PARTIE B : GROS CANON CNN ---\n');

    load(fullfile(dataDir, 'Dataset_Images_Pret.mat'), 'XTrain', 'YTrain', 'XVal', 'YVal', 'imgSize');
    nTrain = size(XTrain, 4);
    if nTrain > 500
        miniBatch = 32;
        configs = {struct('epochs', 40, 'lr', 1e-3)};
        fprintf('  Dataset large (%d train) — 1 config CNN, batch=%d\n', nTrain, miniBatch);
    else
        miniBatch = 16;
        configs = {
            struct('epochs', 40, 'lr', 1e-3)
            struct('epochs', 60, 'lr', 5e-4)
            struct('epochs', 80, 'lr', 2e-4)
        };
    end

    best_acc = 0;
    best_net = [];
    best_cfg = [];
    best_info = [];

    tGlobal = tic;
    for k = 1:numel(configs)
        c = configs{k};
        layers = [
            imageInputLayer([imgSize imgSize 3], 'Name', 'input')
            convolution2dLayer(5, 32, 'Padding', 'same', 'Name', 'conv1')
            batchNormalizationLayer('Name', 'bn1')
            reluLayer('Name', 'relu1')
            maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
            convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
            batchNormalizationLayer('Name', 'bn2')
            reluLayer('Name', 'relu2')
            maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
            fullyConnectedLayer(128, 'Name', 'fc1')
            reluLayer('Name', 'fc_relu')
            fullyConnectedLayer(3, 'Name', 'fc_out')
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'output')
        ];

        opts = trainingOptions('adam', ...
            'MaxEpochs', c.epochs, ...
            'MiniBatchSize', miniBatch, ...
            'InitialLearnRate', c.lr, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.5, ...
            'LearnRateDropPeriod', max(10, floor(c.epochs / 2)), ...
            'Shuffle', 'every-epoch', ...
            'ValidationData', {XVal, YVal}, ...
            'ValidationFrequency', 20, ...
            'Verbose', false, ...
            'Plots', get_plots_mode(), ...
            'ExecutionEnvironment', 'cpu');

        net = trainNetwork(XTrain, YTrain, layers, opts);
        YPred = classify(net, XVal, 'ExecutionEnvironment', 'cpu');
        acc = 100 * mean(YPred == YVal);

        fprintf('  CNN essai %d / epochs=%d -> %.1f %%\n', k, c.epochs, acc);

        if acc > best_acc
            best_acc = acc;
            best_net = net;
            best_cfg = c;
        end
        if acc >= objectif_acc, break; end
    end
    temps = toc(tGlobal);

    if best_acc < objectif_acc
        fprintf('  >> CNN sous %.0f %% : relancer TP3 après generer_dataset.m.\n', objectif_acc);
    end

    figure('Name', 'TP5 - CNN Confusion', 'Visible', get_fig_vis());
    confusionchart(YVal, classify(best_net, XVal, 'ExecutionEnvironment', 'cpu'));
    title(sprintf('CNN — Accuracy : %.1f %% (objectif %.0f %%)', best_acc, objectif_acc));
    saveas(gcf, fullfile(figDir, 'TP5_CNN_Confusion.png'));
    close(gcf);

    sauver_courbe_trainnet(best_info, figDir, 'TP5_CNN_Training.png', 'CNN');

    nParams = compter_parametres(best_net);
    fprintf('CNN FINAL | Accuracy : %.1f %% | Temps : %.1f s | Paramètres : %d\n', best_acc, temps, nParams);

    res.modele = 'CNN';
    res.type_donnees = 'Images (Spectrogrammes 227x227)';
    res.accuracy = best_acc;
    res.temps_sec = temps;
    res.nb_params = nParams;
    res.complexite = 'Élevée';
    res.config = best_cfg;
    res.objectif_atteint = best_acc >= objectif_acc;
    res.commentaire = 'Gros canon — motifs visuels sur spectrogramme.';
    res.net = best_net;
    res.info = best_info;
end

function C = sequences_to_cell(X, idx)
    n = sum(idx);
    seqLen = size(X, 2);
    C = cell(n, 1);
    k = 1;
    for i = find(idx)'
        C{k} = reshape(single(X(i, :, 1)), 1, seqLen);
        k = k + 1;
    end
end

function res = run_LSTM_sequence(chemin_cours, dossier_projet, classes, figDir, objectif_acc)
    fprintf('\n--- PARTIE C : SPÉCIALISTE LSTM ---\n');

    fichier = fullfile(dossier_projet, 'data', 'Dataset_Moteur.mat');
    if ~isfile(fichier)
        fichier = fullfile(chemin_cours, 'data', 'Dataset_Moteur.mat');
    end
    load(fichier, 'Dataset', 'Labels');

    seqLen = 1000;
    numSignals = numel(Dataset);
    X = zeros(numSignals, seqLen, 1, 'single');

    for i = 1:numSignals
        sig = single(Dataset{i}(:));
        sig = sig - mean(sig);
        if std(sig) > 0, sig = sig / std(sig); end
        n = min(numel(sig), seqLen);
        X(i, 1:n, 1) = sig(1:n);
    end

    Y = categorical(Labels, [0 1 2], classes);
    cv = cvpartition(Y, 'HoldOut', 0.2);
    idxTrain = training(cv);
    idxVal = test(cv);

    XTrain = sequences_to_cell(X, idxTrain);
    XVal = sequences_to_cell(X, idxVal);
    YTrain = Y(idxTrain);
    YVal = Y(idxVal);

    configs = {
        struct('units', 64, 'epochs', 40)
    };
    if numSignals <= 500
        configs = {
            struct('units', 64, 'epochs', 40)
            struct('units', 128, 'epochs', 60)
            struct('units', 128, 'epochs', 80)
        };
    else
        fprintf('  Dataset large (%d signaux) — 1 config LSTM\n', numSignals);
    end

    best_acc = 0;
    best_net = [];
    best_temps = 0;
    best_cfg = [];
    best_info = [];

    tGlobal = tic;
    for k = 1:numel(configs)
        c = configs{k};
        layers = [
            sequenceInputLayer(1, 'Name', 'input')
            lstmLayer(c.units, 'OutputMode', 'last', 'Name', 'lstm1')
            dropoutLayer(0.3, 'Name', 'drop')
            fullyConnectedLayer(64, 'Name', 'fc1')
            reluLayer('Name', 'relu1')
            fullyConnectedLayer(3, 'Name', 'fc_out')
            softmaxLayer('Name', 'softmax')
            classificationLayer('Name', 'output')
        ];

        opts = trainingOptions('adam', ...
            'MaxEpochs', c.epochs, ...
            'MiniBatchSize', 32, ...
            'InitialLearnRate', 1e-3, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropFactor', 0.5, ...
            'LearnRateDropPeriod', 20, ...
            'Shuffle', 'every-epoch', ...
            'ValidationData', {XVal, YVal}, ...
            'ValidationFrequency', 15, ...
            'GradientThreshold', 1, ...
            'Verbose', false, ...
            'Plots', get_plots_mode(), ...
            'ExecutionEnvironment', 'cpu');

        net = trainNetwork(XTrain, YTrain, layers, opts);
        YPred = classify(net, XVal, 'ExecutionEnvironment', 'cpu');
        acc = 100 * mean(YPred == YVal);

        fprintf('  LSTM essai %d / units=%d epochs=%d -> %.1f %%\n', k, c.units, c.epochs, acc);

        if acc > best_acc
            best_acc = acc;
            best_net = net;
            best_cfg = c;
            best_info = [];
        end
        if acc >= objectif_acc, break; end
    end
    temps = toc(tGlobal);

    figure('Name', 'TP5 - LSTM Confusion', 'Visible', get_fig_vis());
    confusionchart(YVal, classify(best_net, XVal, 'ExecutionEnvironment', 'cpu'));
    title(sprintf('LSTM — Accuracy : %.1f %% (objectif %.0f %%)', best_acc, objectif_acc));
    saveas(gcf, fullfile(figDir, 'TP5_LSTM_Confusion.png'));
    close(gcf);

    sauver_courbe_trainnet(best_info, figDir, 'TP5_LSTM_Training.png', 'LSTM');

    nParams = compter_parametres(best_net);
    fprintf('LSTM FINAL | Accuracy : %.1f %% | Temps : %.1f s | Paramètres : %d\n', best_acc, temps, nParams);

    res.modele = 'LSTM';
    res.type_donnees = 'Séquence (Signal brut, 1000 pts)';
    res.accuracy = best_acc;
    res.temps_sec = temps;
    res.nb_params = nParams;
    res.complexite = 'Moyenne';
    res.config = best_cfg;
    res.objectif_atteint = best_acc >= objectif_acc;
    res.commentaire = 'Séquence temporelle ; plus lent à entraîner.';
    res.net = best_net;
    res.info = best_info;
end

function n = compter_parametres(net)
    n = 0;
    if ~isprop(net, 'Layers'), return; end
    for k = 1:numel(net.Layers)
        L = net.Layers(k);
        if isprop(L, 'Weights') && ~isempty(L.Weights)
            n = n + numel(L.Weights);
        end
        if isprop(L, 'Bias') && ~isempty(L.Bias)
            n = n + numel(L.Bias);
        end
    end
end

function n = compter_parametres_patternnet(net)
    n = 0;
    for k = 1:numel(net.IW)
        n = n + numel(net.IW{k});
    end
    for k = 1:numel(net.LW)
        n = n + numel(net.LW{k});
    end
    for k = 1:numel(net.b)
        n = n + numel(net.b{k});
    end
end

function sauver_courbe_trainnet(info, figDir, nomFichier, titre)
    if isempty(info) || ~isfield(info, 'TrainingAccuracy'), return; end
    figure('Visible', get_fig_vis());
    subplot(2,1,1);
    plot(info.TrainingAccuracy, 'b'); hold on;
    if isfield(info, 'ValidationAccuracy')
        valIdx = info.ValidationFrequency:info.ValidationFrequency:numel(info.TrainingAccuracy);
        valIdx = valIdx(valIdx <= numel(info.ValidationAccuracy));
        plot(valIdx, info.ValidationAccuracy(1:numel(valIdx)), 'r.-');
        legend('Train', 'Validation', 'Location', 'best');
    end
    ylabel('Accuracy (%)'); title([titre ' — Convergence']); grid on;

    subplot(2,1,2);
    plot(info.TrainingLoss, 'b'); hold on;
    if isfield(info, 'ValidationLoss')
        plot(valIdx, info.ValidationLoss(1:numel(valIdx)), 'r.-');
    end
    ylabel('Loss'); xlabel('Iteration'); grid on;
    saveas(gcf, fullfile(figDir, nomFichier));
    close(gcf);
end

function sauver_courbe_patternnet(net, figDir, nomFichier)
    if ~isfield(net, 'perf') || isempty(net.perf), return; end
    figure('Visible', get_fig_vis());
    plot(net.perf, 'b-o', 'LineWidth', 1.2);
    xlabel('Epoch'); ylabel('MSE'); title('MLP — Courbe d''apprentissage');
    grid on;
    saveas(gcf, fullfile(figDir, nomFichier));
    close(gcf);
end

function afficher_synthese(resultats, figDir, objectif_acc)
    noms = {'MLP', 'CNN', 'LSTM'};
    fprintf('\n============================================================\n');
    fprintf('  SYNTHÈSE — COMPTE-RENDU DE SPRINT (objectif %.0f %%)\n', objectif_acc);
    fprintf('============================================================\n');
    fprintf('%-6s | %8s | %10s | %12s | %s\n', 'Modèle', 'Préc.%', 'Paramètres', 'Temps (s)', 'Objectif');
    fprintf('%s\n', repmat('-', 1, 70));

    for i = 1:numel(noms)
        r = resultats.(noms{i});
        ok = 'OK';
        if ~r.objectif_atteint, ok = 'OPTIMISER'; end
        fprintf('%-6s | %7.1f | %10d | %12.1f | %s\n', ...
            r.modele, r.accuracy, r.nb_params, r.temps_sec, ok);
    end

    [~, idxRapide] = min([resultats.MLP.temps_sec, resultats.CNN.temps_sec, resultats.LSTM.temps_sec]);
    accs = [resultats.MLP.accuracy, resultats.CNN.accuracy, resultats.LSTM.accuracy];
    [bestAcc, idxPrecis] = max(accs);

    fprintf('\n--- Questions de synthèse ---\n');
    fprintf('1. Plus rapide : %s (%.1f s)\n', noms{idxRapide}, resultats.(noms{idxRapide}).temps_sec);
    fprintf('2. Meilleure précision : %s (%.1f %%)\n', noms{idxPrecis}, bestAcc);
    fprintf('3. Microcontrôleur : MLP (3 features, inférence légère).\n');

    temps = [resultats.MLP.temps_sec, resultats.CNN.temps_sec, resultats.LSTM.temps_sec];

    figure('Name', 'TP5 - Comparaison Sprint', 'Visible', get_fig_vis());
    subplot(1, 2, 1);
    b = bar(accs, 0.6, 'FaceColor', 'flat');
    b.CData = [0.2 0.7 0.3; 0.2 0.4 0.9; 0.9 0.6 0.1];
    hold on;
    yline(objectif_acc, 'r--', sprintf('Objectif %.0f%%', objectif_acc), 'LineWidth', 1.5);
    set(gca, 'XTickLabel', noms);
    ylabel('Accuracy (%)'); title('Précision validation');
    ylim([0 100]); grid on;

    subplot(1, 2, 2);
    bar(temps, 'FaceColor', [0.9 0.5 0.2]);
    set(gca, 'XTickLabel', noms);
    ylabel('Temps (s)'); title('Temps d''entraînement'); grid on;

    sgtitle('TP5 — Bilan du Code-Off');
    saveas(gcf, fullfile(figDir, 'TP5_Comparaison_Sprint.png'));
    close(gcf);
end

function v = get_fig_vis()
    global MODE_VISIBLE
    if ~isempty(MODE_VISIBLE) && MODE_VISIBLE
        v = 'on';
    else
        v = 'off';
    end
end

function p = get_plots_mode()
    global MODE_VISIBLE
    if ~isempty(MODE_VISIBLE) && MODE_VISIBLE
        p = 'training-progress';
    else
        p = 'none';
    end
end
