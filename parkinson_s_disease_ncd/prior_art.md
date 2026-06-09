# Prior-art landscape — #010 Parkinson's Disease NCD
Pair: **pitolisant + xanomeline**
Mechanism: H3 antagonism + M1 agonism
Tier: Large  ·  Market: ~$4B Parkinson's Rx  ·  FTO: Mixed

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
`pitolisant xanomeline Parkinson's Disease NCD treatment combination therapy`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.549 | 890848 | Methods and compositions for treatment of disorders ameliorated by muscarinic receptor activation |
| 2 | 0.548 | 2713167 | Sporoderm-broken germination-activated ganoderma lucidum spores for protection of dopaminergic neurons and treatment of Parkinson's disease |
| 3 | 0.532 | 685486 | Delta opioid agonist mu opioid antagonist compositions and methods for treating Parkinsons disease |
| 4 | 0.528 | 2569561 | Use of levodopa, carbidopa and entacapone for treating Parkinson's disease |
| 5 | 0.528 | 15410 | Use of levodopa, carbidopa and entacapone for treating Parkinson's disease |
| 6 | 0.528 | 12090 | Use of levodopa, carbidopa and entacapone for treating Parkinson's disease |
| 7 | 0.523 | 1522423 | Transdermal pharmaceutical preparation containing active substance combinations, for treating Parkinson's disease |
| 8 | 0.522 | 1326392 | Methods for treating Parkinson's disease |

## 03_medicationA_in_disorder
`pitolisant Parkinson's Disease NCD method of treatment use`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.599 | 2569561 | Use of levodopa, carbidopa and entacapone for treating Parkinson's disease |
| 2 | 0.599 | 15410 | Use of levodopa, carbidopa and entacapone for treating Parkinson's disease |
| 3 | 0.599 | 12090 | Use of levodopa, carbidopa and entacapone for treating Parkinson's disease |
| 4 | 0.588 | 2713167 | Sporoderm-broken germination-activated ganoderma lucidum spores for protection of dopaminergic neurons and treatment of Parkinson's disease |
| 5 | 0.587 | 1873606 | Methods for preventing and/or treating degenerative disorders of the central nervous system |
| 6 | 0.577 | 1326392 | Methods for treating Parkinson's disease |
| 7 | 0.574 | 438339 | Method for treating or mitigating Parkinson's disease using nicotine inhaler or nicotine nasal spray |
| 8 | 0.573 | 1269947 | Preparation for improving the action of receptors |

## 04_medicationB_in_disorder
`xanomeline Parkinson's Disease NCD method of treatment use`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.606 | 890848 | Methods and compositions for treatment of disorders ameliorated by muscarinic receptor activation |
| 2 | 0.581 | 2569561 | Use of levodopa, carbidopa and entacapone for treating Parkinson's disease |
| 3 | 0.581 | 15410 | Use of levodopa, carbidopa and entacapone for treating Parkinson's disease |
| 4 | 0.581 | 12090 | Use of levodopa, carbidopa and entacapone for treating Parkinson's disease |
| 5 | 0.576 | 787280 | Methods and compositions for treatment of disorders ameliorated by muscarinic receptor activation |
| 6 | 0.569 | 2713167 | Sporoderm-broken germination-activated ganoderma lucidum spores for protection of dopaminergic neurons and treatment of Parkinson's disease |
| 7 | 0.565 | 129888 | Analogs of xanomeline |
| 8 | 0.563 | 438339 | Method for treating or mitigating Parkinson's disease using nicotine inhaler or nicotine nasal spray |

## 05_mechanism_combo
`H3 antagonism + M1 agonism combination treatment Parkinson's Disease NCD`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.640 | 2117192 | 14-3-3 proteins for diagnosis of Parkinson's disease |
| 2 | 0.589 | 637397 | Nato3 mutant polypeptides and uses thereof |
| 3 | 0.573 | 2069214 | Transgenic non-human animal for use in research models for studying Parkinson's disease |
| 4 | 0.564 | 706759 | Treatment of synucleinopathies |
| 5 | 0.562 | 479760 | Levodopa fractionated dose composition and use |
| 6 | 0.562 | 335043 | Levodopa fractionated dose composition and use |
| 7 | 0.562 | 123144 | NK3 modulators and uses thereof |
| 8 | 0.559 | 685486 | Delta opioid agonist mu opioid antagonist compositions and methods for treating Parkinsons disease |

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
`Parkinson's Disease NCD pharmaceutical treatment method novel combination`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.761 | 685486 | Delta opioid agonist mu opioid antagonist compositions and methods for treating Parkinsons disease |
| 2 | 0.734 | 1317303 | Compositions for preventing and/or treating degenerative disorders of the central nervous system |
| 3 | 0.734 | 1721454 | Compositions for preventing and/or treating degenerative disorders of the central nervous system |
| 4 | 0.732 | 1452914 | Composition containing jetbead extracts |
| 5 | 0.730 | 2569561 | Use of levodopa, carbidopa and entacapone for treating Parkinson's disease |
| 6 | 0.730 | 15410 | Use of levodopa, carbidopa and entacapone for treating Parkinson's disease |
| 7 | 0.730 | 12090 | Use of levodopa, carbidopa and entacapone for treating Parkinson's disease |
| 8 | 0.725 | 1326392 | Methods for treating Parkinson's disease |

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
