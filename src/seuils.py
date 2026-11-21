"""Seuils et métriques d'alerte fixés A PRIORI (avant de voir les résultats).

Règle (chantier #7 de docs/ROADMAP.md) : aucun seuil de décision n'est choisi
après avoir vu les performances. Toute modification de ces constantes doit être
justifiée dans docs/PASSATION.md AVANT le backtest qui l'utilise.
"""

# Seuil de probabilité au-delà duquel on émet une alerte de couverture insuffisante.
SEUIL_ALERTE_RISQUE = 0.5

# Métriques d'évaluation retenues a priori pour les tensions :
# - courbe PR (précision-rappel) et PR-AUC ;
# - nombre de cas détectés à nombre de fausses alertes fixé (1, 2, 3) ;
# - priorité aux NOUVELLES dégradations (départ en couverture acceptable/satisfaisante).
METRIQUES_TENSIONS = ["pr_auc", "detections_a_cout_fixe", "fausses_alertes", "degradations_detectees"]

# Métriques retenues a priori pour les volumes : MAE, WAPE, biais signé, erreur max,
# calculés sur le total ET par forme (le total peut masquer des erreurs).
METRIQUES_VOLUMES = ["mae", "wape", "biais", "erreur_max"]

# Seuils de qualité des quantiles : couverture visée de la bande [q10, q90].
COUVERTURE_VISEE = 0.80
