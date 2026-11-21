# Roadmap

Objectif : rendre les résultats **démontrables** (fiabilité) et **utilisables** (décision), et publier les tableaux de bord en ligne. Les priorités viennent de l'analyse de `README.md`, `docs/DATA.md`, `docs/PASSATION.md` et du code (`src/`). Chaque chantier est « fait » quand son code, ses tests et la doc sont à jour.

Suivi : goal actif de la session + cases à cocher ci-dessous.

## Phase 0 — Publication des dashboards (Vercel) — Option A retenue

Les trois dashboards (`outputs/dashboard_lits.html`, `dashboard_mensuel.html`, `dashboard_pharma.html`, 4,6–5,4 Mo chacun) sont des fichiers HTML **autonomes** (Plotly embarqué) : un hébergement 100 % statique suffit, Vercel est adapté (gratuit en Hobby).

- [x] `vercel.json` à la racine (réécritures `/`, `/lits`, `/mensuel`, `/pharma` vers `outputs/`).
- [x] Branche de travail de l'agent Mistral : `mistral` (pushée sur origin), pour différencier des autres agents.
- [ ] Déployer (à faire côté compte Vercel de l'utilisateur) : sur [vercel.com/new](https://vercel.com/new), importer `RisqueSept/TimePharm`, Framework Preset = **Other**, Build Command et Output Directory **vides** ; puis Settings → Git → **Production Branch = `mistral`**. Chaque push sur `mistral` redéploie ; les autres branches ont une preview.
- [x] Procédure de mise à jour : `./scripts/publier.sh` (régénère les 3 dashboards avec `--skip-fetch`, lance les tests, commit + push des `outputs/`). Variante CI GitHub Actions avec `VERCEL_TOKEN` à étudier ensuite (checkpoint de 837 Mo à mettre en cache Hugging Face).
- [ ] Vérifier en ligne : les 3 dashboards + les CSV/PNG de `outputs/` accessibles, navigation entre pages.
- [ ] (Optionnel) domaine personnalisé dans les réglages Vercel.

## Phase 1 — Fiabilité (fondations, à faire en premier)

| # | Chantier | Pourquoi (constat) | Fichiers concernés | Fait quand |
|---|---|---|---|---|
| 1 | **Test anti-fuite temporelle** : modifier toutes les données postérieures à une origine et vérifier que la prévision à cette origine est inchangée | Le masquage ANSM/Sentinelles est codé à la main, jamais prouvé | `tests/test_fuites_pharma.py` (nouveau), `src/modele_pharma.py`, `src/modele_volumes_pharma.py` | ✅ fait — 4 tests anti-fuite, `unittest discover` au vert (24 tests) |
| 2 | **Tests unitaires manquants** sur la chronologie des relevés (horizons 14/28 j) et des volumes (délai, horizons 1–3 mois) | `tests/` couvre lits/mensuel mais pas le cas prioritaire pharma | `tests/test_fuites_pharma.py` (nouveau) | ✅ fait — fenêtre cible ±3 j, `derniere_cible_connue < origine`, cutoff volumes, structure horizons 1–3 |
| 3 | **Manifeste de run** : hash des CSV d'entrée, révision du modèle, paramètres, date → `outputs/manifest.json` par exécution | « les résultats ont changé » n'est pas diagnosable aujourd'hui | `src/manifest.py` (nouveau), `main.py`, `main_mensuel.py`, `main_pharma.py`, `tests/test_manifest.py` | ✅ fait — `manifest.json` + historique `outputs/manifests/`, écrit par les 3 pipelines |
| 4 | **Calibration mesurée** : couverture empirique des quantiles 10–90 %, Brier + courbe de fiabilité des scores de tension, biais signé par forme (pas seulement le total) | DATA.md : « la couverture réelle dépend du domaine », « la calibration reste à mesurer » | `src/modele_pharma.py`, `src/modele_volumes_pharma.py`, dashboards | métriques affichées dans les dashboards/CSV |
| 5 | **Vrai calendrier de délai Medic'AM** par mois à partir de `data/research/pharma/medicam_dates_archives.csv`, utilisé dans le backtest courant | délai simulé 2 mois = WAPE 9,74 % ; archives vérifiables = 13,0–13,2 % | `src/modele_volumes_pharma.py`, `src/recherche_archives_pharma.py` | backtest courant utilise le délai réel |
| 6 | **Période de vérification réellement réservée** + intervalles de confiance du gain (block bootstrap / Diebold-Mariano) au lieu du rééchantillonnage « indicatif » | sept. 2024–août 2026 déjà inclus dans les stats globales ; poids du mélange choisi sur le développement | `src/recherche_volumes_pharma.py`, `src/recherche_delai_pharma.py` | mois futurs exclus de tout réglage, IC publiés |
| 7 | **Seuils d'alerte fixés a priori** (coût fausse alerte/détection manquée, PR-AUC) avant de voir les résultats | seuil 0,5 et « min de fausses alertes » choisis a posteriori sur 14 cas | `src/modele_pharma.py`, `docs/PASSATION.md` | seuils écrits dans la doc avant le prochain backtest |
| 8 | **Comparaison total direct vs somme des prévisions** (France mensuelle = somme des régions ; pharma = somme des 6 formes) | listé comme piste non faite dans PASSATION | `src/modele_mensuel.py`, `src/modele_volumes_pharma.py` | écart mesuré et publié |

## Phase 2 — Utilité (décision)

| # | Chantier | Fait quand |
|---|---|---|
| 9 | **Alertes actionnables** dans `dashboard_pharma.html` : produit, horizon 14/28 j, niveau de risque, « nouvelle dégradation » vs « tension connue », seuil affiché | liste d'alertes visible dans le dashboard |
| 10 | **Indicateur de fraîcheur** en en-tête des 3 dashboards : dernier mois Medic'AM, dernier relevé ANSM, date de génération, prochaine mise à jour prévue | bandeau « données du … » présent |
| 11 | **Scénarios MCO avec distribution saisonnière** par territoire (pas seulement la moyenne annuelle en lits équivalents) | `scenarios_capacite_mco_regions.csv` + dashboard mensuel par territoire |
| 12 | **Indicateur de besoin relatif par région** (volumes prévus × prévalences projetées / part des 75 ans+) | nouvelle vue ou CSV croisant pharma et prévalences |
| 13 | **Horizons hors validation grisés** (24 mois mensuel, 2035 annuel) dans les dashboards | légende « hors validation » affichée |

## Phase 3 — Organisation et performance

| # | Chantier | Fait quand |
|---|---|---|
| 14 | **Cache des inférences Timestral** (clé = produit/origine/horizon/révision) pour accélérer les backtests | temps du backtest pharma divisé, cache hit > 50 % en rejoue |
| 15 | **CLI unifiée** (`python -m timestral run|backtest|recherche|dashboard …`) + dates de split et paramètres dans une config versionnée (au lieu de constantes dispersées) | une seule commande par pipeline |
| 16 | **Détection de ruptures de définition** sur les séries d'entrée (ANSM, Sentinelles, Medic'AM, Odissé) + flag « série révisée » dans les sorties | alerte sur changement de variance/moyenne testée |
| 17 | **JSON sidecar** par sortie (date de génération, hash entrées, révision modèle) pour réutilisation traçable | sidecar écrit à chaque export |

## Règles de travail

- Un chantier = une branche/session dédiée, tests et doc à jour avant intégration.
- Rien n'est réglé sur la période de vérification réservée (chantier 6).
- Chaque déploiement Vercel est vérifié manuellement (Phase 0).
