# Prior-art landscape — #022 Lewy Body Disease
Pair: **pitolisant + xanomeline**
Mechanism: H3 antagonism + M1 agonism
Tier: Large  ·  Market: ~$1B (Nuplazid PDP-led)  ·  FTO: Mixed

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
`pitolisant xanomeline Lewy Body Disease treatment combination therapy`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.541 | 544772 | Lewy body disease therapeutic agent containing pyrazoloquinoline derivative |
| 2 | 0.538 | 2235898 | Treatment and delay of outset of Parkinson's disease |
| 3 | 0.538 | 2117018 | Treatment and delay of onset of synucleinopathic and amyloidogenic disease |
| 4 | 0.538 | 2014139 | Prevention and treatment of synucleinopathic and amyloidogenic disease |
| 5 | 0.538 | 1769984 | Prevention and treatment of synucleinopathic and amyloidogenic disease |
| 6 | 0.538 | 1667031 | Prevention and treatment of synucleinopathic and amyloidogenic disease |
| 7 | 0.538 | 1494783 | Treatment and delay of outset of synucleinopathic and amyloidogenic disease |
| 8 | 0.505 | 254382 | Methods of treating parkinson's disease and/or lewy body disease or disorder(s) |

## 03_medicationA_in_disorder
`pitolisant Lewy Body Disease method of treatment use`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.606 | 2235898 | Treatment and delay of outset of Parkinson's disease |
| 2 | 0.606 | 2117018 | Treatment and delay of onset of synucleinopathic and amyloidogenic disease |
| 3 | 0.606 | 2014139 | Prevention and treatment of synucleinopathic and amyloidogenic disease |
| 4 | 0.606 | 1769984 | Prevention and treatment of synucleinopathic and amyloidogenic disease |
| 5 | 0.606 | 1667031 | Prevention and treatment of synucleinopathic and amyloidogenic disease |
| 6 | 0.606 | 1494783 | Treatment and delay of outset of synucleinopathic and amyloidogenic disease |
| 7 | 0.587 | 254382 | Methods of treating parkinson's disease and/or lewy body disease or disorder(s) |
| 8 | 0.552 | 544772 | Lewy body disease therapeutic agent containing pyrazoloquinoline derivative |

## 04_medicationB_in_disorder
`xanomeline Lewy Body Disease method of treatment use`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.575 | 544772 | Lewy body disease therapeutic agent containing pyrazoloquinoline derivative |
| 2 | 0.563 | 254382 | Methods of treating parkinson's disease and/or lewy body disease or disorder(s) |
| 3 | 0.553 | 2235898 | Treatment and delay of outset of Parkinson's disease |
| 4 | 0.553 | 2117018 | Treatment and delay of onset of synucleinopathic and amyloidogenic disease |
| 5 | 0.553 | 2014139 | Prevention and treatment of synucleinopathic and amyloidogenic disease |
| 6 | 0.553 | 1769984 | Prevention and treatment of synucleinopathic and amyloidogenic disease |
| 7 | 0.553 | 1667031 | Prevention and treatment of synucleinopathic and amyloidogenic disease |
| 8 | 0.553 | 1494783 | Treatment and delay of outset of synucleinopathic and amyloidogenic disease |

## 05_mechanism_combo
`H3 antagonism + M1 agonism combination treatment Lewy Body Disease`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.643 | 544772 | Lewy body disease therapeutic agent containing pyrazoloquinoline derivative |
| 2 | 0.623 | 1618854 | Humanized antibodies that recognize alpha-synuclein |
| 3 | 0.623 | 1402293 | Humanized antibodies that recognize alpha-synuclein |
| 4 | 0.623 | 1391782 | Humanized antibodies that recognize alpha-synuclein |
| 5 | 0.623 | 1005332 | Humanized antibodies that recognize alpha-synuclein |
| 6 | 0.623 | 828412 | Humanized antibodies that recognize alpha-synuclein |
| 7 | 0.623 | 724823 | Humanized antibodies that recognize alpha-synuclein |
| 8 | 0.623 | 2640583 | Humanized antibodies that recognize alpha-synuclein |

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
`Lewy Body Disease pharmaceutical treatment method novel combination`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.737 | 544772 | Lewy body disease therapeutic agent containing pyrazoloquinoline derivative |
| 2 | 0.703 | 2235898 | Treatment and delay of outset of Parkinson's disease |
| 3 | 0.703 | 2117018 | Treatment and delay of onset of synucleinopathic and amyloidogenic disease |
| 4 | 0.703 | 2014139 | Prevention and treatment of synucleinopathic and amyloidogenic disease |
| 5 | 0.703 | 1769984 | Prevention and treatment of synucleinopathic and amyloidogenic disease |
| 6 | 0.703 | 1667031 | Prevention and treatment of synucleinopathic and amyloidogenic disease |
| 7 | 0.703 | 1494783 | Treatment and delay of outset of synucleinopathic and amyloidogenic disease |
| 8 | 0.678 | 254382 | Methods of treating parkinson's disease and/or lewy body disease or disorder(s) |

## 08_medicationA_alone
`pitolisant pharmaceutical composition formulation`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.722 | 544828 | Pharmaceutical compositions containing cannabis, uses thereof and methods for improving energy levels and/or alleviating fatigue |
| 2 | 0.716 | 2619582 | No Title |
| 3 | 0.716 | 945134 | Pharmaceutical composition based on Centella asiatica ( |
| 4 | 0.705 | 2690432 | Self emulsifying medication delivery system |
| 5 | 0.697 | 1115228 | Pharmaceutical composition containing extract of houttuynia cordata as active ingredient for preventing and treating dementia, parkinson's disease, or epilepsy |
| 6 | 0.694 | 1177766 | Single solid oral dosage forms for treating |
| 7 | 0.690 | 2770984 | Pharmaceutical compositions for inhalation containing an anticholinergic, corticosteroid and betamimetic |
| 8 | 0.690 | 2531353 | Long-acting medication combinations for the treatment of respiratory complaints |

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
