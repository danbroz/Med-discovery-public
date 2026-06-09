# Prior-art landscape — #033 Delirium
Pair: **pitolisant + xanomeline**
Mechanism: H3 antagonism + M1 agonism
Tier: Mid  ·  Market: ~$500M (off-label antipsychotics)  ·  FTO: Mixed

Source: local FAISS index over US/EP patents 1999→present, queried via the
`patents-rag` MCP server. Scores are cosine similarity (higher = closer).
Top hits per query are listed below; full JSON in `prior_art.json`.

## 01_pair_combo
`pitolisant xanomeline combination pharmaceutical composition method of treatment`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.595 | 1177766 | Single solid oral dosage forms for treating |
| 2 | 0.547 | 636228 | Inhibitors of glucocorticoid receptor |
| 3 | 0.547 | 596692 | Inhibitors of glucocorticoid receptor |
| 4 | 0.546 | 1604508 | Alfentanil composition for the treatment of acute pain |
| 5 | 0.546 | 1145855 | Alfentanil composition for the treatment of acute pain |
| 6 | 0.543 | 795542 | Compositions comprising melatonin |
| 7 | 0.543 | 1408178 | Pharmaceutical composition for oral administration |
| 8 | 0.541 | 863745 | Inhibitors of glucocorticoid receptor |

## 02_pair_in_disorder
`pitolisant xanomeline Delirium treatment combination therapy`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.530 | 610016 | Prophylactic or therapeutic agent for delirium |
| 2 | 0.518 | 2550740 | Method and composition for treating post operative conditions |
| 3 | 0.500 | 479937 | Method and system for at least reducing or preventing delirium in a patient |
| 4 | 0.494 | 890848 | Methods and compositions for treatment of disorders ameliorated by muscarinic receptor activation |
| 5 | 0.483 | 787280 | Methods and compositions for treatment of disorders ameliorated by muscarinic receptor activation |
| 6 | 0.470 | 2838915 | Methods for treating delirium glucocorticoid receptor-specific antagonists |
| 7 | 0.470 | 2145059 | Methods for treating delirium using glucocorticoid receptor-specific antagonists |
| 8 | 0.469 | 1456593 | Compounds and methods for treating aberrant adrenocartical cell disorders |

## 03_medicationA_in_disorder
`pitolisant Delirium method of treatment use`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.590 | 479937 | Method and system for at least reducing or preventing delirium in a patient |
| 2 | 0.573 | 2550740 | Method and composition for treating post operative conditions |
| 3 | 0.550 | 2838915 | Methods for treating delirium glucocorticoid receptor-specific antagonists |
| 4 | 0.550 | 2145059 | Methods for treating delirium using glucocorticoid receptor-specific antagonists |
| 5 | 0.547 | 610016 | Prophylactic or therapeutic agent for delirium |
| 6 | 0.527 | 925043 | Reversal of general anesthesia by administration of methylphenidate, amphetamine, modafinil, amantadine, and/or caffeine |
| 7 | 0.526 | 432113 | Reversal of general anesthesia by administration of methylphenidate, amphetamine, modafinil, amantadine, and/or caffeine |
| 8 | 0.523 | 367950 | Reversal of general anesthesia by administration of methylphenidate, amphetamine, modafinil, amantadine, and/or caffeine |

## 04_medicationB_in_disorder
`xanomeline Delirium method of treatment use`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.564 | 2550740 | Method and composition for treating post operative conditions |
| 2 | 0.559 | 610016 | Prophylactic or therapeutic agent for delirium |
| 3 | 0.556 | 890848 | Methods and compositions for treatment of disorders ameliorated by muscarinic receptor activation |
| 4 | 0.554 | 479937 | Method and system for at least reducing or preventing delirium in a patient |
| 5 | 0.548 | 890847 | Methods and compositions for treatment of disorders ameliorated by muscarinic receptor activation |
| 6 | 0.545 | 787280 | Methods and compositions for treatment of disorders ameliorated by muscarinic receptor activation |
| 7 | 0.523 | 129888 | Analogs of xanomeline |
| 8 | 0.516 | 3037839 | Use of R (+)-α-(2,3-dimethoxyphenyl)-1-[2-(4-fluorophenyl)ethyl]-4-piperidinemethanol for the treatment of substance induced insomnia |

## 05_mechanism_combo
`H3 antagonism + M1 agonism combination treatment Delirium`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.589 | 610016 | Prophylactic or therapeutic agent for delirium |
| 2 | 0.562 | 584319 | Combinations of a H3 antagonist and a noradrenaline reuptake inhibitor, and the therapeutical uses thereof |
| 3 | 0.533 | 228360 | Combinations of a H3 antagonist and a noradrenaline reuptake inhibitor, and the therapeutical uses thereof |
| 4 | 0.520 | 2623688 | Use of H3K9me3 modulation for enhancing cognitive function |
| 5 | 0.520 | 727882 | Use of H3K9me3 modulation for enhancing cognitive function |
| 6 | 0.520 | 544793 | Use of H3K9me3 modulation for enhancing cognitive function |
| 7 | 0.516 | 2550740 | Method and composition for treating post operative conditions |
| 8 | 0.513 | 2708734 | Serotonin 5-HT3 receptor agonist |

## 06_mechanism_general
`H3 antagonism + M1 agonism pharmaceutical composition method`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.643 | 817463 | Compounds as neuronal histamine receptor-3 antagonists and uses thereof |
| 2 | 0.643 | 691282 | Compounds as neuronal histamine receptor-3 antagonists and uses thereof |
| 3 | 0.643 | 487540 | Compounds as neuronal histamine receptor-3 antagonists and uses thereof |
| 4 | 0.643 | 177596 | Compounds as neuronal histamine receptor-3 antagonists and uses thereof |
| 5 | 0.640 | 2171880 | Histamine H3 receptor inhibitors, preparation and therapeutic uses |
| 6 | 0.629 | 2681473 | Pyrrolidine derivatives as histamine H3 receptor antagonists |
| 7 | 0.625 | 1957614 | Histamine H3 receptor agents, preparation and therapeutic uses |
| 8 | 0.624 | 2132966 | Histamine H3 receptor antagonists, preparation and therapeutic uses |

## 07_disorder_general
`Delirium pharmaceutical treatment method novel combination`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.692 | 610016 | Prophylactic or therapeutic agent for delirium |
| 2 | 0.646 | 2550740 | Method and composition for treating post operative conditions |
| 3 | 0.639 | 479937 | Method and system for at least reducing or preventing delirium in a patient |
| 4 | 0.639 | 2838915 | Methods for treating delirium glucocorticoid receptor-specific antagonists |
| 5 | 0.639 | 2145059 | Methods for treating delirium using glucocorticoid receptor-specific antagonists |
| 6 | 0.627 | 367950 | Reversal of general anesthesia by administration of methylphenidate, amphetamine, modafinil, amantadine, and/or caffeine |
| 7 | 0.623 | 925043 | Reversal of general anesthesia by administration of methylphenidate, amphetamine, modafinil, amantadine, and/or caffeine |
| 8 | 0.622 | 432113 | Reversal of general anesthesia by administration of methylphenidate, amphetamine, modafinil, amantadine, and/or caffeine |

## 08_medicationA_alone
`pitolisant pharmaceutical composition formulation`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.722 | 544828 | Pharmaceutical compositions containing cannabis, uses thereof and methods for improving energy levels and/or alleviating fatigue |
| 2 | 0.716 | 2619582 | No Title |
| 3 | 0.716 | 945134 | Pharmaceutical composition based on Centella asiatica ( |
| 4 | 0.705 | 2690432 | Self emulsifying drug delivery system |
| 5 | 0.697 | 1115228 | Pharmaceutical composition containing extract of houttuynia cordata as active ingredient for preventing and treating dementia, parkinson's disease, or epilepsy |
| 6 | 0.694 | 1177766 | Single solid oral dosage forms for treating |
| 7 | 0.690 | 2770984 | Pharmaceutical compositions for inhalation containing an anticholinergic, corticosteroid and betamimetic |
| 8 | 0.690 | 2531353 | Long-acting drug combinations for the treatment of respiratory complaints |

## 09_medicationB_alone
`xanomeline pharmaceutical composition formulation`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.662 | 1667622 | Xanthurenic acid derivative pharmaceutical compositions and methods related thereto |
| 2 | 0.659 | 701392 | Compositions and methods for treating disorders ameliorated by muscarinic receptor activation |
| 3 | 0.659 | 2619582 | No Title |
| 4 | 0.659 | 945134 | Pharmaceutical composition based on Centella asiatica ( |
| 5 | 0.654 | 544828 | Pharmaceutical compositions containing cannabis, uses thereof and methods for improving energy levels and/or alleviating fatigue |
| 6 | 0.650 | 305567 | Compositions and methods for treating disorders ameliorated by muscarinic receptor activation |
| 7 | 0.650 | 704223 | Compositions and methods for treatment of disorders ameliorated by muscarinic receptor activation |
| 8 | 0.650 | 472423 | Compositions and methods for treating disorders ameliorated by muscarinic receptor activation |

## 10_class_combo
`H3 antagonism + M1 agonism synergistic combination central nervous system`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.561 | 727936 | H3.3 CTL peptides and uses thereof |
| 2 | 0.561 | 289705 | H3.3 CTL peptides and uses thereof |
| 3 | 0.539 | 2708734 | Serotonin 5-HT3 receptor agonist |
| 4 | 0.526 | 2745121 | Constitutively active histamine H3 receptor mutants and uses thereof |
| 5 | 0.521 | 2640540 | 5-HT3 receptor antagonists |
| 6 | 0.521 | 1001732 | 5-HT3 receptor antagonists |
| 7 | 0.514 | 228360 | Combinations of a H3 antagonist and a noradrenaline reuptake inhibitor, and the therapeutical uses thereof |
| 8 | 0.501 | 584319 | Combinations of a H3 antagonist and a noradrenaline reuptake inhibitor, and the therapeutical uses thereof |
