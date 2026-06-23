function Etat_Moteur = Deep_Engine_Detecteur(Signal_Courant)
%DEEP_ENGINE_DETECTEUR — Bloc Simulink (Séance 6)
% Compatible simulation : appel extrinsic vers detecteur.m

    coder.extrinsic('detecteur');
    Etat_Moteur = uint8(1);

    if isempty(Signal_Courant)
        return;
    end

    tmp = detecteur(Signal_Courant);
    if ~isempty(tmp)
        Etat_Moteur = uint8(tmp);
    end
