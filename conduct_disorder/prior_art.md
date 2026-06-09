# Prior-art landscape — #038 Conduct Disorder
Pair: **perospirone + oxymetazoline**
Mechanism: DRD2 antagonism + α2A agonism
Tier: Mid  ·  Market: ~$400M (pediatric off-label)  ·  FTO: High

Source: local FAISS index over US/EP patents 1999→present, queried via the
`patents-rag` MCP server. Scores are cosine similarity (higher = closer).
Top hits per query are listed below; full JSON in `prior_art.json`.

## 01_pair_combo
`perospirone oxymetazoline combination pharmaceutical composition method of treatment`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.618 | 1574491 | Pharmaceutical cream compositions and methods of use |
| 2 | 0.602 | 830016 | Pharmaceutical composition containing imidazoline derivative |
| 3 | 0.602 | 824156 | Pharmaceutical composition containing imidazoline derivative |
| 4 | 0.590 | 1177766 | Single solid oral dosage forms for treating |
| 5 | 0.576 | 1208955 | Formulation comprising benzothiazolone compound |
| 6 | 0.576 | 1237430 | Compositions and methods for treating or preventing diseases associated with oxidative stress |
| 7 | 0.572 | 454842 | Compositions and methods for treating or preventing diseases associated with oxidative stress |
| 8 | 0.568 | 3498670 | Pharmaceutical composition for treatment of hypertrophic cardiomyopathy and treatment method using same composition |

## 02_pair_in_disorder
`perospirone oxymetazoline Conduct Disorder treatment combination therapy`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.491 | 1814344 | Treatment for paresis/paralysis |
| 2 | 0.473 | 1075568 | Peptides as oxytocin agonists |
| 3 | 0.467 | 2529004 | Use of quetiapine |
| 4 | 0.465 | 3077354 | Treatment of oppositional defiant disorder and conduct disorder with 5-aminoalkyl-4,5,6,7-tetrahydro-4-oxyindolones |
| 5 | 0.458 | 1830360 | 1,4-diaza-bicyclo[3.2.2]nonyl oxadiazolyl compounds and their medical use |
| 6 | 0.458 | 1972155 | 1,4-diaza-bicyclo[3.2.2]nonyl oxadiazolyl compounds and their medical use |
| 7 | 0.457 | 2063481 | Oxadiazole derivatives and their medical use |
| 8 | 0.456 | 2005802 | 1,4-diaza-bicyclo[3.2.2]nonyl oxadiazolyl derivatives useful as modulator of nicotinic acetylcholine receptors |

## 03_medicationA_in_disorder
`perospirone Conduct Disorder method of treatment use`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.513 | 2380381 | Treatment of sexual disorders |
| 2 | 0.506 | 799994 | Treatment of impulsivity-related disorders |
| 3 | 0.481 | 1215331 | Methods of treating Prader-Willi Syndrome |
| 4 | 0.481 | 965266 | Methods of treating Prader-Willi syndrome |
| 5 | 0.474 | 76894 | Treatment of addiction and dependency |
| 6 | 0.473 | 1445622 | Use of botulinum neurotoxin to treat substance addictions |
| 7 | 0.472 | 2612688 | Vesicular monoamine transporter-2 ligands and their use in the treatment of psychostimulant abuse |
| 8 | 0.472 | 255488 | Vesicular monoamine transporter-2 ligands and their use in the treatment of psychostimulant abuse |

## 04_medicationB_in_disorder
`oxymetazoline Conduct Disorder method of treatment use`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.523 | 3077354 | Treatment of oppositional defiant disorder and conduct disorder with 5-aminoalkyl-4,5,6,7-tetrahydro-4-oxyindolones |
| 2 | 0.515 | 740948 | Oxymetazoline compositions |
| 3 | 0.515 | 713037 | Oxymetazoline compositions |
| 4 | 0.515 | 639667 | Oxymetazoline compositions |
| 5 | 0.515 | 544757 | Oxymetazoline compositions |
| 6 | 0.515 | 438369 | Oxymetazoline compositions |
| 7 | 0.512 | 2267027 | Thiazoline and oxazoline derivatives and their methods of use |
| 8 | 0.512 | 1787091 | Thiazoline and oxazoline derivatives and their methods of use |

## 05_mechanism_combo
`DRD2 antagonism + α2A agonism combination treatment Conduct Disorder`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.565 | 160455 | Single nucleotide polymorphic alleles of human DP-2 gene for detection of susceptibility to hair growth inhibition by PGD2 |
| 2 | 0.539 | 1135000 | Method of treatment of aggression |
| 3 | 0.539 | 961718 | Method of treatment of aggression |
| 4 | 0.539 | 399854 | Method of treatment of aggression |
| 5 | 0.534 | 1814326 | Hexahydro-1H-4,7-methanoisoindole-1,3-dione compounds |
| 6 | 0.523 | 1176663 | Nucleic acids encoding polypeptides which disrupt D1-D2 dopamine receptor coupling and modulate function |
| 7 | 0.523 | 1807822 | Polypeptides and methods for modulating D1-D2 dopamine receptor interaction and function |
| 8 | 0.523 | 1376529 | Methods for modulating D1-D2 dopamine receptor coupling and function |

## 06_mechanism_general
`DRD2 antagonism + α2A agonism pharmaceutical composition method`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.621 | 2582200 | Dual antagonist of PGD |
| 2 | 0.609 | 2199001 | Cardiovascular protection using anti-aldosteronic progestins |
| 3 | 0.595 | 155699 | Modulators of diacyglycerol acyltransferase 2 (DGAT2) |
| 4 | 0.595 | 546176 | Modulators of diacyglycerol acyltransferase 2 (DGAT2) |
| 5 | 0.582 | 667765 | Treatment of a subtype of ASD |
| 6 | 0.581 | 2057332 | Platelet aggregation inhibitor composition |
| 7 | 0.581 | 1465258 | Treating inflammation and inflammatory pain in mucosa using mucosal prolonged release bioadhesive therapeutic carriers |
| 8 | 0.580 | 1987142 | PAR-2 agonist |

## 07_disorder_general
`Conduct Disorder pharmaceutical treatment method novel combination`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.650 | 3077354 | Treatment of oppositional defiant disorder and conduct disorder with 5-aminoalkyl-4,5,6,7-tetrahydro-4-oxyindolones |
| 2 | 0.616 | 2529004 | Use of quetiapine |
| 3 | 0.603 | 2639348 | Heteroaromatic phenylimidazole derivatives as PDE10A enzyme inhibitors |
| 4 | 0.603 | 1581912 | Phenylimidazole derivatives as PDE10A enzyme inhibitors |
| 5 | 0.603 | 1504499 | Phenylimidazole derivatives as PDE10A enzyme inhibitors |
| 6 | 0.603 | 1462576 | Heteroaromatic phenylimidazole derivatives as PDE10A enzyme inhibitors |
| 7 | 0.596 | 541433 | Method of treatment of attention deficit/hyperactivity disorder (ADHD) |
| 8 | 0.596 | 469768 | Method of treatment of attention deficit/hyperactivity disorder (ADHD) |

## 08_medicationA_alone
`perospirone pharmaceutical composition formulation`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.707 | 2619582 | No Title |
| 2 | 0.707 | 945134 | Pharmaceutical composition based on Centella asiatica ( |
| 3 | 0.692 | 544828 | Pharmaceutical compositions containing cannabis, uses thereof and methods for improving energy levels and/or alleviating fatigue |
| 4 | 0.687 | 1017493 | Pharmaceutical composition of fused aminodihydrothiazine derivative |
| 5 | 0.687 | 1004254 | Pharmaceutical composition of fused aminodihydrothiazine derivative |
| 6 | 0.682 | 607494 | Pharmaceutical compositions for treating cystic fibrosis |
| 7 | 0.682 | 2639383 | Use of kaempferia parviflora wall. ex. baker extracts or flavone compound for preventing or treating muscle diseases, or improving muscle function |
| 8 | 0.682 | 969547 | Use of Kaempferia parviflora wall. ex. baker extracts or flavone compound for preventing or treating muscle diseases, or improving muscle function |

## 09_medicationB_alone
`oxymetazoline pharmaceutical composition formulation`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.718 | 740948 | Oxymetazoline compositions |
| 2 | 0.718 | 713037 | Oxymetazoline compositions |
| 3 | 0.718 | 639667 | Oxymetazoline compositions |
| 4 | 0.718 | 544757 | Oxymetazoline compositions |
| 5 | 0.718 | 438369 | Oxymetazoline compositions |
| 6 | 0.716 | 830016 | Pharmaceutical composition containing imidazoline derivative |
| 7 | 0.716 | 824156 | Pharmaceutical composition containing imidazoline derivative |
| 8 | 0.696 | 1574491 | Pharmaceutical cream compositions and methods of use |

## 10_class_combo
`DRD2 antagonism + α2A agonism synergistic combination central nervous system`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.542 | 160455 | Single nucleotide polymorphic alleles of human DP-2 gene for detection of susceptibility to hair growth inhibition by PGD2 |
| 2 | 0.510 | 3271988 | US006214615B1 (12) United States Patent (10) Patent No.: US 6,214,615 B1 Brann et al. (45) Date of P |
| 3 | 0.506 | 770906 | Selectable marker genes |
| 4 | 0.505 | 844691 | Peptide or pharmaceutically acceptable salt thereof, or promedication thereof |
| 5 | 0.503 | 1807822 | Polypeptides and methods for modulating D1-D2 dopamine receptor interaction and function |
| 6 | 0.503 | 1376529 | Methods for modulating D1-D2 dopamine receptor coupling and function |
| 7 | 0.503 | 1176663 | Nucleic acids encoding polypeptides which disrupt D1-D2 dopamine receptor coupling and modulate function |
| 8 | 0.502 | 2791731 | Nucleic acid encoding calcyon, a D-1 like dopamine receptor activity modifying protein |
