# Prior-art landscape — #001 Major Depressive Disorder
Pair: **agomelatine + vilazodone**
Mechanism: MT1/MT2 agonism + HTR1A agonism
Tier: Mega  ·  Market: ~$18B global antidepressants  ·  FTO: High

Source: local FAISS index over US/EP patents 1999→present, queried via the
`patents-rag` MCP server. Scores are cosine similarity (higher = closer).
Top hits per query are listed below; full JSON in `prior_art.json`.

## 01_pair_combo
`agomelatine vilazodone combination pharmaceutical composition method of treatment`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.661 | 1258045 | Crystal form VII of agomelatine, preparation method and use thereof and pharmaceutical composition containing same |
| 2 | 0.660 | 2140171 | Orodispersible pharmaceutical composition for oromucosal or sublingual administration of agomelatine |
| 3 | 0.633 | 1177766 | Single solid oral dosage forms for treating |
| 4 | 0.629 | 1905908 | Process for the synthesis of (methoxy-1-naphthyl) acetonitrile and application in the synthesis of agomelatine |
| 5 | 0.622 | 1649426 | Agomelatine hydrochloride hydrate and preparation thereof |
| 6 | 0.608 | 1158158 | Imatinib mesylate oral pharmaceutical composition and process for preparation thereof |
| 7 | 0.594 | 2565223 | Pharmaceutical compositions comprising (s)-4-(4-(4- (((2-(2,6-dioxopiperidin-3-yl)-1-oxoisoindolin-4-yl)oxy)methyl)benzyl)piperazin-1-yl)-3-fluorobenzonitrtle and methods of using the |
| 8 | 0.594 | 225356 | Pharmaceutical compositions comprising (S)-4-(4-(4-(((2-(2,6-dioxopiperidin-3-yl)-1-oxoisoindolin-4-yl)oxy)methyl)benzyl)piperazin-1-yl)-3-fluorobenzonitrile and methods of using the same |

## 02_pair_in_disorder
`agomelatine vilazodone Major Depressive Disorder treatment combination therapy`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.636 | 148661 | Method of treatment and selection of patients benefiting from agomelatine based on EEG measurements |
| 2 | 0.591 | 2103060 | Use of agomelatine in obtaining medicaments intended for the treatment of bipolar disorders |
| 3 | 0.552 | 2318192 | Association between agomelatine and a noradrenaline reuptake inhibitor, and pharmaceutical compositions containing it |
| 4 | 0.528 | 945093 | Methods for treating depressive symptoms |
| 5 | 0.528 | 776055 | Methods for treating depressive symptoms |
| 6 | 0.528 | 441948 | Methods for treating depressive symptoms |
| 7 | 0.528 | 132929 | Methods for treating depressive symptoms |
| 8 | 0.517 | 3020813 | Combination therapy for treatment of refractory depression |

## 03_medicationA_in_disorder
`agomelatine Major Depressive Disorder method of treatment use`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.732 | 148661 | Method of treatment and selection of patients benefiting from agomelatine based on EEG measurements |
| 2 | 0.687 | 2103060 | Use of agomelatine in obtaining medicaments intended for the treatment of bipolar disorders |
| 3 | 0.612 | 2318192 | Association between agomelatine and a noradrenaline reuptake inhibitor, and pharmaceutical compositions containing it |
| 4 | 0.602 | 2132955 | Association between agomelatine and a thymoregulatory agent and pharmaceutical compositions containing it |
| 5 | 0.579 | 1660931 | Co-crystals of agomelatine, a process for there preparation and pharmaceutical compositions containing them |
| 6 | 0.518 | 1946086 | Methods for the treatment of major depressive disorder using glucocorticoid receptor antagonists |
| 7 | 0.508 | 439626 | Modulators of melanocortin receptors for the treatment of depression and anxiety |
| 8 | 0.507 | 1258045 | Crystal form VII of agomelatine, preparation method and use thereof and pharmaceutical composition containing same |

## 04_medicationB_in_disorder
`vilazodone Major Depressive Disorder method of treatment use`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.605 | 945093 | Methods for treating depressive symptoms |
| 2 | 0.605 | 776055 | Methods for treating depressive symptoms |
| 3 | 0.605 | 441948 | Methods for treating depressive symptoms |
| 4 | 0.605 | 132929 | Methods for treating depressive symptoms |
| 5 | 0.595 | 3020813 | Combination therapy for treatment of refractory depression |
| 6 | 0.582 | 2632743 | Oxabicycloheptanes and oxabicycloheptenes for the treatment of depressive and stress disorders |
| 7 | 0.582 | 876194 | Oxabicycloheptanes and oxabicycloheptenes for the treatment of depressive and stress disorders |
| 8 | 0.564 | 619516 | Method of treatment for ketamine infusion |

## 05_mechanism_combo
`MT1/MT2 agonism + HTR1A agonism combination treatment Major Depressive Disorder`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.608 | 265682 | Methods of treating acute depression and/or acute anxiety |
| 2 | 0.608 | 62132 | Methods of treating acute depression |
| 3 | 0.608 | 24850 | Methods of treating acute depression and anxiety |
| 4 | 0.599 | 2554120 | Methods |
| 5 | 0.557 | 1250874 | Assays and methods for selecting a treatment regimen for a subject with depression |
| 6 | 0.544 | 1362677 | Bicyclo (3.1.0) hexane-2, 6-dicarboxylic acid derivatives as mGlu2 receptor agonist |
| 7 | 0.521 | 403567 | mTORC1 modulators and uses thereof |
| 8 | 0.507 | 2939499 | Mutations of the 5′ region of the human 5-HT1A gene |

## 06_mechanism_general
`MT1/MT2 agonism + HTR1A agonism pharmaceutical composition method`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.657 | 2639351 | Enhanced treatment regimens using mTOR inhibitors |
| 2 | 0.657 | 1420513 | Enhanced treatment regimens using mTor inhibitors |
| 3 | 0.655 | 335034 | Rapamycin analogs and uses thereof |
| 4 | 0.652 | 368247 | Methods of treatment using an mTORC1 modulator |
| 5 | 0.652 | 35503 | Methods of treatment using an mTORC1 modulator |
| 6 | 0.620 | 403567 | mTORC1 modulators and uses thereof |
| 7 | 0.615 | 746375 | mTORC1 modulators |
| 8 | 0.607 | 529015 | Polymorphic compounds and uses thereof |

## 07_disorder_general
`Major Depressive Disorder pharmaceutical treatment method novel combination`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.724 | 765065 | Combination methods and compositions including sleep therapeutics for treating mood |
| 2 | 0.724 | 2585571 | Combination methods and compositions including sleep therapeutics for treating mood |
| 3 | 0.723 | 243175 | Combination methods and compositions including sleep therapeutics for treating mood |
| 4 | 0.723 | 105577 | Combination methods and compositions including sleep therapeutics for treating mood |
| 5 | 0.712 | 945093 | Methods for treating depressive symptoms |
| 6 | 0.712 | 776055 | Methods for treating depressive symptoms |
| 7 | 0.712 | 441948 | Methods for treating depressive symptoms |
| 8 | 0.712 | 132929 | Methods for treating depressive symptoms |

## 08_medicationA_alone
`agomelatine pharmaceutical composition formulation`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.781 | 1258045 | Crystal form VII of agomelatine, preparation method and use thereof and pharmaceutical composition containing same |
| 2 | 0.742 | 2140171 | Orodispersible pharmaceutical composition for oromucosal or sublingual administration of agomelatine |
| 3 | 0.736 | 1714500 | Crystalline form VI of agomelatine, preparation method and application thereof |
| 4 | 0.735 | 1905908 | Process for the synthesis of (methoxy-1-naphthyl) acetonitrile and application in the synthesis of agomelatine |
| 5 | 0.718 | 1552617 | Pharmaceutically acceptable cocrystals of N-[2-(7-methoxyl-1-naphtyl)ethyl]acetamide and methods of their preparation |
| 6 | 0.694 | 1158158 | Imatinib mesylate oral pharmaceutical composition and process for preparation thereof |
| 7 | 0.687 | 1649426 | Agomelatine hydrochloride hydrate and preparation thereof |
| 8 | 0.680 | 1192031 | Forms of co-crystals of agomelatine and p toluenesulphonic acid, a process for their preparation and pharmaceutical compositions containing them |

## 09_medicationB_alone
`vilazodone pharmaceutical composition formulation`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.742 | 1270976 | Solid state forms of vilazodone and vilazodone hydrochloride |
| 2 | 0.727 | 2619582 | No Title |
| 3 | 0.727 | 945134 | Pharmaceutical composition based on Centella asiatica ( |
| 4 | 0.724 | 1728293 | Pharmaceutical composition comprising N-[3-chloro-4-[(3-fluorophenyl)methoxy]phenyl]-6-[5[[[2-(methylsulfonyl)ethyl]amino]methyl]-quinazolinamine |
| 5 | 0.721 | 544828 | Pharmaceutical compositions containing cannabis, uses thereof and methods for improving energy levels and/or alleviating fatigue |
| 6 | 0.713 | 607494 | Pharmaceutical compositions for treating cystic fibrosis |
| 7 | 0.704 | 1208955 | Formulation comprising benzothiazolone compound |
| 8 | 0.702 | 1177766 | Single solid oral dosage forms for treating |

## 10_class_combo
`MT1/MT2 agonism + HTR1A agonism synergistic combination central nervous system`

| Rank | Score | Patent ID | Title |
|---:|---:|---:|---|
| 1 | 0.542 | 157947 | Targeting P18 for mTOR-related disorders |
| 2 | 0.502 | 529015 | Polymorphic compounds and uses thereof |
| 3 | 0.502 | 137332 | Polymorphic compounds and uses thereof |
| 4 | 0.488 | 1002881 | Methods for identifying compounds as modulators of SLC38A9 interactions |
| 5 | 0.483 | 35503 | Methods of treatment using an mTORC1 modulator |
| 6 | 0.483 | 368247 | Methods of treatment using an mTORC1 modulator |
| 7 | 0.480 | 403567 | mTORC1 modulators and uses thereof |
| 8 | 0.475 | 1820872 | L-methylfolate treatment for psychiatric or neurologic disorders |
