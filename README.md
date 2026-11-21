# Prévisions hospitalières et pharmaceutiques

Les projections utilisent les poids officiels de [Timestral-209M de Mistral AI](https://huggingface.co/mistralai/Timestral-209M), avec la bibliothèque d’inférence officielle sur CPU. Les méthodes historiques servent de références dans le backtest. Les calculs d’équivalents de lits MCO restent des scénarios déterministes fondés sur des hypothèses d’activité et d’occupation.

## Exécuter les analyses

```bash
python3.13 -m venv .venv-timestral
.venv-timestral/bin/python -m pip install -r requirements.txt
.venv-timestral/bin/python main.py --skip-fetch
.venv-timestral/bin/python main_mensuel.py --skip-fetch
.venv-timestral/bin/python main_pharma.py
```

Avec `--skip-fetch`, les scripts utilisent les archives de `data/raw/`. Sans cette option, ils interrogent data.gouv.fr pour repérer les ressources. Prévoir environ 837 Mo pour le checkpoint. Le code charge les fichiers de `data/models/Timestral-209M/` lorsqu’ils sont présents. Sinon, il les récupère auprès de Hugging Face à la révision épinglée dans le code. Les données de santé sont traitées localement.

## Lire les résultats

| Analyse | Résultat | À quoi il sert |
|---|---|---|
| Capacité nationale | [Tableau annuel](outputs/dashboard_lits.html), [projections 2026–2035](outputs/prediction_lits_2026_2035.csv) | Comparer Timestral aux références et afficher ses quantiles 10 % et 90 %. |
| Charge MCO | [Scénarios nationaux](outputs/scenarios_capacite_mco.csv), [scénarios par territoire](outputs/scenarios_capacite_mco_regions.csv) | Faire varier l’activité de −5 %, 0 % ou +5 % et tester des hypothèses d’occupation de 80 %, 85 % et 90 %. Les résultats expriment une charge moyenne annuelle en équivalents de lits ouverts. |
| Occupation avec SARS-CoV-2 | [Tableau mensuel](outputs/dashboard_mensuel.html), [backtest](outputs/backtest_mensuel.csv) | Comparer les performances à plusieurs horizons sur l’archive SPF. La projection de juillet 2023 à juin 2025 illustre le comportement du modèle après la dernière observation. |
| Pathologies chroniques | [Prévalences par territoire](outputs/prevalence_projections_2030.csv) | Explorer la prolongation des tendances 2015–2024 à 2030. |
| Antibiotiques pédiatriques | [Tableau des tensions](outputs/dashboard_pharma.html), [backtest](outputs/backtest_pharma.csv) | Tester la prévision d’une couverture insuffisante chez les grossistes à 14 et 28 jours sur six formes buvables. |
| Boîtes d’antibiotiques remboursées | [Prévisions mensuelles](outputs/dashboard_pharma.html#boites), [prévisions CSV](outputs/previsions_boites_pharma.csv) | Estimer le volume remboursé des six formes à 1, 2 et 3 mois et comparer Timestral à la référence saisonnière. |

Les tableaux Plotly sont des fichiers HTML autonomes, ouvrables hors ligne.
Dans **Observé et prédit**, choisis un modèle et un horizon. Le tableau mensuel permet aussi de choisir la France entière ou une région. Les courbes, les écarts et le tableau des valeurs se mettent à jour ensemble.

### Exemple MCO

L’activité de référence 2025 est de **51,25 millions de journées**. À activité identique et avec une hypothèse d’occupation de 85 %, le calcul donne **165 190 équivalents de lits ouverts** en moyenne. Ce calcul sert à comparer des hypothèses. Pour dimensionner un service, il faut des données locales sur les pics d’activité, la capacité ouverte, les effectifs et les flux de patients.

La validation annuelle porte sur des horizons de **1 à 3 ans**. Les chiffres jusqu’en 2035 décrivent des scénarios au-delà de ces horizons. L’étude mensuelle mesure l’erreur à **1, 3, 6 et 12 mois**, avec peu de dates de départ à 12 mois.

Voir [la documentation des données et des contrôles](docs/DATA.md) et [la passation pour reprendre les expérimentations](docs/PASSATION.md).
