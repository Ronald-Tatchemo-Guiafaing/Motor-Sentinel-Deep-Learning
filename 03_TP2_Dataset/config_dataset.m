function cfg = config_dataset()
%CONFIG_DATASET — Taille centralisee du dataset (modifiable ici)
    cfg.N_BASE = 50;
    cfg.FACTEUR = 20;
    cfg.N_ECHANTILLONS_PAR_CLASSE = cfg.N_BASE * cfg.FACTEUR;  % 1000
    cfg.N_TOTAL = cfg.N_ECHANTILLONS_PAR_CLASSE * 3;           % 3000
end
