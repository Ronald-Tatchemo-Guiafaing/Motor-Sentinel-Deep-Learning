%% test_auto.m - TP2 Exercices 1 et 2 (compatible Scenario_TP1)
clc; close all;

nom_modele = 'Moteur_Asynchrone_Base2019bG';
dossier_script = fileparts(mfilename('fullpath'));
chemin_cours = fileparts(dossier_script);
cd(chemin_cours);

figDir = fullfile(dossier_script, 'figures_TP2');
if ~exist(figDir, 'dir'), mkdir(figDir); end

load_system(nom_modele);
useScenario = ~isempty(find_system(nom_modele, 'SearchDepth', 1, 'Name', 'Scenario_TP1'));

if useScenario
    set_param([nom_modele '/Scenario_TP1'], 'Value', '1');
else
    set_param([nom_modele '/Fault_Switch'], 'sw', '0');
end

simOut = sim(nom_modele, 'ReturnWorkspaceOutputs', 'on');
temps = simOut.simout.time;
courant_A = simOut.simout.signals.values(:, 1);

figure('Visible', 'off');
plot(temps, courant_A);
title('Automatique : Moteur Sain (dataset x20)');
xlabel('Temps (s)'); ylabel('Courant Ia (A)'); grid on;
saveas(gcf, fullfile(figDir, 'TP2_Moteur_Sain.png'));
close(gcf);

niveau_bruit = 0.8;
courant_noisy = courant_A + niveau_bruit * randn(length(courant_A), 1);
figure('Visible', 'off');
subplot(2,1,1); plot(temps, courant_A); title('Signal Simulink (Propre)');
subplot(2,1,2); plot(temps, courant_noisy); title('Signal Reel (Bruite)');
saveas(gcf, fullfile(figDir, 'TP2_Propre_vs_Bruite.png'));
close(gcf);

bdclose(nom_modele);
fprintf('test_auto : figures TP2 regenerees (%s)\n', datestr(now));
