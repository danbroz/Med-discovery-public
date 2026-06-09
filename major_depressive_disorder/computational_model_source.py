from __future__ import annotations

"""
Major Depressive Disorder siibra scaffold.

This script turns a chapter-level biological summary of Major Depressive
Disorder (MDD) into a conservative, atlas-grounded mechanistic scaffold using
siibra where available. It is intended for research prototyping and transparent
hypothesis exploration only. It is not a diagnostic, prognostic, or treatment
recommendation tool.

Core modeling choices from the chapter:
- MDD is modeled as an interaction between inherited/familial vulnerability,
  environmental stress, and dysregulated monoaminergic plus GABA/glutamate
  neurobiology.
- Serotonergic and noradrenergic dysregulation are treated as primary latent
  biology, with GABA/glutamate imbalance and stress sensitization as important
  interacting processes.
- Region anchors are conservative: prefrontal control, hippocampus, amygdala,
  thalamus proxy, and basal-ganglia proxy, because the chapter names these
  regions directly while keeping many mechanisms at systems level.
- Default mode network overconnectivity is modeled as a latent network-level
  process rather than over-forcing a single parcel, because the chapter links
  it to rumination at the network level.
- The simulator is intentionally simple and acyclic:
  inputs -> latent biology -> regional dysregulation -> symptoms -> phenotypes

The script degrades gracefully:
- If siibra is not installed, or if a feature/modality is unavailable, the
  atlas-backed parts stay empty instead of crashing.
- The simulator still runs even without atlas data.

Source chapter themes encoded here include:
- serotonergic and noradrenergic antidepressant evidence,
- GABA/glutamate dysregulation and ketamine-linked rapid antidepressant effects,
- heritable and familial vulnerability,
- stress-triggered episode liability,
- prefrontal, hippocampal, amygdalar, thalamic, and basal-ganglia circuitry,
- default-mode overconnectivity associated with rumination.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
    _SIIBRA_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore
    _SIIBRA_IMPORT_ERROR = exc


DEFAULT_GENE_PANEL = [
    # Serotonergic signaling
    "SLC6A4",
    "TPH2",
    "HTR1A",
    "HTR2A",
    # Noradrenergic / monoamine metabolism
    "SLC6A2",
    "ADRA2A",
    "MAOA",
    "COMT",
    # GABA / glutamate balance
    "GAD1",
    "GABRA2",
    "GRIN2B",
    "SLC1A2",
    # Neuroplasticity and stress responsivity
    "BDNF",
    "FKBP5",
    "NR3C1",
    "CRHR1",
]


class MajorDepressiveDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Major Depressive Disorder.

    Notes
    -----
    - This is a research scaffold, not a validated disease model.
    - Regional values in `simulate()` quantify dysregulation / burden rather
      than healthy activation.
    - Proxies are used where the chapter stays systems-level or where a stable
      Julich label may vary across environments.
    - DMN-related rumination is kept as a latent network process because the
      chapter describes it as altered resting-state connectivity rather than as
      a single cytoarchitectonic parcel.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        self.atlas = None
        self.parcellation = None
        self.space = None
        self._atlas_available = False

        if siibra is None:
            warnings.warn(
                f"siibra could not be imported ({_SIIBRA_IMPORT_ERROR}). "
                "Atlas-backed methods will remain available only as graceful stubs; "
                "the simulator still works."
            )
        else:
            try:
                self.atlas = siibra.atlases.get(atlas_spec)
                self.parcellation = (
                    self.atlas.get_parcellation(parcellation_spec)
                    if hasattr(self.atlas, "get_parcellation")
                    else self.atlas.parcellations.get(parcellation_spec)
                )
                self.space = (
                    self.atlas.get_space(space_spec)
                    if hasattr(self.atlas, "get_space")
                    else self.atlas.spaces.get(space_spec)
                )
                self._atlas_available = (
                    self.atlas is not None and self.parcellation is not None and self.space is not None
                )
            except Exception as exc:
                warnings.warn(
                    f"Could not initialize siibra atlas resources: {exc}. "
                    "Atlas-backed helpers will degrade gracefully."
                )

        # Conservative region anchors from the chapter's explicitly named
        # circuitry. Proxies are preferred when the chapter is broader than a
        # single stable cytoarchitectonic label.
        self.region_candidates: Dict[str, List[str]] = {
            "pfc_control": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "Area 8Av left",
                "dorsolateral prefrontal cortex left",
                "prefrontal cortex left",
                "prefrontal cortex",
            ],
            "hippocampus": [
                "CA1 left",
                "CA left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "LA (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "thalamus_proxy": [
                "mediodorsal thalamus left",
                "thalamus left",
                "thalamus",
            ],
            "basal_ganglia_proxy": [
                "caudate nucleus left",
                "caudate left",
                "putamen left",
                "striatum left",
                "basal ganglia left",
                "basal ganglia",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "pfc_control": (
                "Prefrontal control proxy for executive regulation, affective control, "
                "and cognitive governance disrupted in MDD."
            ),
            "hippocampus": (
                "Hippocampal memory and neuroplasticity anchor relevant to stress-related "
                "cognitive burden and depressive vulnerability."
            ),
            "amygdala": (
                "Amygdala anchor for limbic salience, negative affect, and stress-linked "
                "emotional bias."
            ),
            "thalamus_proxy": (
                "Thalamic proxy for distributed relay, arousal, and thalamocortical "
                "integration burden implicated by the chapter."
            ),
            "basal_ganglia_proxy": (
                "Basal-ganglia / striatal proxy for motivational, psychomotor, and "
                "reward-related burden in depression."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "polygenic_liability": (
                "Inherited risk distributed across monoamine signaling, stress response, "
                "and neuroplasticity pathways."
            ),
            "parental_depression_burden": (
                "Familial loading from parental mood disorder, contributing both genetic "
                "vulnerability and potentially adverse emotional environment."
            ),
            "adverse_life_stress": (
                "Acute or episodic adverse life events capable of precipitating depressive episodes."
            ),
            "chronic_stress_load": (
                "Sustained stress burden that sensitizes stress biology and cognitive-affective circuits."
            ),
            "adolescent_onset_vulnerability": (
                "Developmental vulnerability reflecting the stronger genetic contribution often "
                "reported for adolescent-onset depression."
            ),
            "monoamine_treatment_support": (
                "Protective monoamine-targeted support representing SSRI, SNRI, or TCA-like "
                "restoration of serotonergic and noradrenergic tone."
            ),
            "glutamatergic_treatment_support": (
                "Protective glutamatergic support representing ketamine-like reduction of "
                "glutamatergic dysregulation."
            ),
            "recovery_support": (
                "Protective treatment adherence, structure, and psychosocial support that "
                "reduces stress burden and supports recovery."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_dysregulation": (
                "Reduced or maladaptively regulated serotonin signaling consistent with the "
                "monoamine evidence base in MDD."
            ),
            "noradrenergic_dysregulation": (
                "Altered norepinephrine tone affecting arousal, attention, energy, and mood."
            ),
            "gaba_glutamate_imbalance": (
                "Disrupted inhibitory-excitatory balance linking anxiety comorbidity, insomnia, "
                "and cognitive burden."
            ),
            "hpa_stress_sensitization": (
                "Stress-linked neuroendocrine sensitization that lowers threshold for depressive episodes."
            ),
            "hippocampal_neuroplasticity_burden": (
                "Stress-related hippocampal memory/plasticity burden contributing to cognitive symptoms."
            ),
            "frontolimbic_dysregulation": (
                "Disrupted coordination between prefrontal control systems and limbic emotion systems."
            ),
            "dmn_overconnectivity": (
                "Excessive default mode network coupling supporting negative self-referential thought "
                "and rumination."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "depressed_affect": (
                "Persistent depressed mood, negative affect, and emotional suffering."
            ),
            "anhedonia_motivation_loss": (
                "Reduced reward responsivity, diminished motivation, and loss of interest."
            ),
            "rumination": (
                "Persistent, intrusive, negative self-focused thought associated with DMN overconnectivity."
            ),
            "cognitive_impairment": (
                "Attention, memory, and executive inefficiency linked to hippocampal and prefrontal burden."
            ),
            "anxiety_insomnia_burden": (
                "Anxious arousal and sleep disturbance linked to GABA/glutamate and stress dysregulation."
            ),
            "psychomotor_somatic_burden": (
                "Psychomotor slowing, fatigue, and broader somatic burden emerging from distributed circuit dysfunction."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "polygenic_liability",
                "target": "serotonergic_dysregulation",
                "relation": "heritable variation increases serotonergic vulnerability",
                "mdd_change": "increased",
            },
            {
                "source": "polygenic_liability",
                "target": "noradrenergic_dysregulation",
                "relation": "heritable variation increases noradrenergic vulnerability",
                "mdd_change": "increased",
            },
            {
                "source": "polygenic_liability",
                "target": "gaba_glutamate_imbalance",
                "relation": "genetic liability contributes to inhibitory-excitatory instability",
                "mdd_change": "increased",
            },
            {
                "source": "parental_depression_burden",
                "target": "hpa_stress_sensitization",
                "relation": "familial mood-disorder loading contributes to a lower stress threshold",
                "mdd_change": "increased",
            },
            {
                "source": "parental_depression_burden",
                "target": "frontolimbic_dysregulation",
                "relation": "familial burden increases vulnerability of mood-regulation circuits",
                "mdd_change": "increased",
            },
            {
                "source": "adverse_life_stress",
                "target": "hpa_stress_sensitization",
                "relation": "adverse life events can precipitate episodes through stress biology",
                "mdd_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "hpa_stress_sensitization",
                "relation": "sustained stress sensitizes neuroendocrine burden",
                "mdd_change": "increased",
            },
            {
                "source": "adolescent_onset_vulnerability",
                "target": "frontolimbic_dysregulation",
                "relation": "developmental vulnerability increases susceptibility of mood-regulation networks",
                "mdd_change": "increased",
            },
            {
                "source": "monoamine_treatment_support",
                "target": "serotonergic_dysregulation",
                "relation": "monoamine-targeted treatment partially restores serotonin signaling",
                "mdd_change": "decreased",
            },
            {
                "source": "monoamine_treatment_support",
                "target": "noradrenergic_dysregulation",
                "relation": "monoamine-targeted treatment partially restores norepinephrine signaling",
                "mdd_change": "decreased",
            },
            {
                "source": "glutamatergic_treatment_support",
                "target": "gaba_glutamate_imbalance",
                "relation": "NMDA-targeted treatment may reduce glutamatergic burden",
                "mdd_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "hpa_stress_sensitization",
                "relation": "protective support buffers stress amplification",
                "mdd_change": "decreased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "frontolimbic_dysregulation",
                "relation": "serotonergic disruption weakens affective regulation across prefrontal-limbic circuits",
                "mdd_change": "increased",
            },
            {
                "source": "noradrenergic_dysregulation",
                "target": "frontolimbic_dysregulation",
                "relation": "noradrenergic disruption perturbs arousal and cognitive control systems",
                "mdd_change": "increased",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "frontolimbic_dysregulation",
                "relation": "excitatory-inhibitory imbalance destabilizes mood-regulation circuitry",
                "mdd_change": "increased",
            },
            {
                "source": "hpa_stress_sensitization",
                "target": "hippocampal_neuroplasticity_burden",
                "relation": "stress biology compromises hippocampal plasticity and memory systems",
                "mdd_change": "increased",
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "dmn_overconnectivity",
                "relation": "weak control networks permit excessive self-referential default-mode coupling",
                "mdd_change": "increased",
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "pfc_control",
                "relation": "frontolimbic disruption compromises prefrontal regulation",
                "mdd_change": "dysregulated",
            },
            {
                "source": "hippocampal_neuroplasticity_burden",
                "target": "hippocampus",
                "relation": "stress-related neuroplastic burden disrupts hippocampal function",
                "mdd_change": "dysregulated",
            },
            {
                "source": "hpa_stress_sensitization",
                "target": "amygdala",
                "relation": "stress sensitization heightens limbic emotional reactivity",
                "mdd_change": "hyperreactive",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "thalamus_proxy",
                "relation": "inhibitory-excitatory imbalance perturbs thalamocortical relay processes",
                "mdd_change": "dysregulated",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "basal_ganglia_proxy",
                "relation": "monoaminergic disruption burdens motivational and motor-affective circuits",
                "mdd_change": "dysregulated",
            },
            {
                "source": "noradrenergic_dysregulation",
                "target": "basal_ganglia_proxy",
                "relation": "noradrenergic dysregulation contributes to psychomotor and motivational slowing",
                "mdd_change": "dysregulated",
            },
            {
                "source": "amygdala",
                "target": "depressed_affect",
                "relation": "limbic negative-emotion bias amplifies depressed affect",
                "mdd_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "anhedonia_motivation_loss",
                "relation": "striatal/basal-ganglia dysfunction contributes to low motivation and diminished reward",
                "mdd_change": "increased",
            },
            {
                "source": "dmn_overconnectivity",
                "target": "rumination",
                "relation": "default-mode overconnectivity supports intrusive negative self-focused thought",
                "mdd_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "cognitive_impairment",
                "relation": "hippocampal burden contributes to memory and cognitive inefficiency",
                "mdd_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "cognitive_impairment",
                "relation": "prefrontal dysfunction weakens executive and attentional control",
                "mdd_change": "increased",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "anxiety_insomnia_burden",
                "relation": "inhibitory-excitatory imbalance contributes to anxious arousal and insomnia",
                "mdd_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "psychomotor_somatic_burden",
                "relation": "thalamocortical dysregulation contributes to psychomotor and somatic burden",
                "mdd_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "psychomotor_somatic_burden",
                "relation": "basal-ganglia burden contributes to fatigue and psychomotor slowing",
                "mdd_change": "increased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        self._pmap = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    def _modality_candidates(self, kind: str) -> List[Any]:
        if siibra is None:
            return []

        cands: List[Any] = []
        try:
            if kind == "receptor":
                cands.append(siibra.features.molecular.ReceptorDensityFingerprint)
            elif kind == "gene":
                cands.append(siibra.features.molecular.GeneExpressions)
            elif kind == "connectivity":
                cands.append(siibra.features.connectivity.StreamlineCounts)
        except Exception:
            pass

        if kind == "receptor":
            cands.append("receptor density fingerprint")
        elif kind == "gene":
            cands.append("gene expressions")
        elif kind == "connectivity":
            cands.append("StreamlineCounts")
        return cands

    def _safe_features_any(self, concept: Any, modalities: Sequence[Any], **kwargs: Any) -> List[Any]:
        if siibra is None or concept is None:
            return []
        for modality in modalities:
            try:
                with siibra.QUIET:
                    feats = siibra.features.get(concept, modality, **kwargs)
                if feats:
                    return list(feats)
            except Exception:
                continue
        return []

    def _julich_matches(self, query: str) -> List[Any]:
        if self.atlas is None:
            return []
        try:
            matches = self.atlas.find_regions(
                query,
                all_versions=False,
                filter_children=False,
                find_topmost=False,
            )
        except Exception:
            return []

        out = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower():
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        """
        Lower tuples rank better.
        Prefer left hemisphere, specific labels, and Julich labels over generic names.
        """
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "prefrontal cortex",
            "hippocampus",
            "amygdala",
            "thalamus",
            "basal ganglia",
            "striatum",
        } else 0
        proxy_penalty = 1 if "proxy" in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if self.atlas is None or self.parcellation is None:
            return None

        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation)
            except Exception:
                pass
            try:
                return self.parcellation.get_region(spec)
            except Exception:
                pass
            matches = self._julich_matches(spec)
            if matches:
                matches = sorted(matches, key=self._region_rank)
                return matches[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows = []
        seen = set()
        matches = self._julich_matches(keyword)
        for region in sorted(matches, key=self._region_rank):
            row = (
                self._name_of(region),
                getattr(region, "identifier", None),
                getattr(getattr(region, "parcellation", None), "name", ""),
            )
            if row in seen:
                continue
            seen.add(row)
            rows.append(
                {"name": row[0], "identifier": row[1], "parcellation": row[2]}
            )
            if len(rows) >= limit:
                break
        return pd.DataFrame(rows)

    def _spatial_props_list(self, region: Any) -> List[Any]:
        if region is None or self.space is None:
            return []
        try:
            props = region.spatial_props(space=self.space)
        except Exception:
            return []
        if props is None:
            return []
        if isinstance(props, dict):
            return list(props.values())
        if isinstance(props, (list, tuple)):
            return list(props)
        if hasattr(props, "components"):
            return list(getattr(props, "components", []))
        return [props]

    def _main_component(
        self, region: Any
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        volume_mm3 = float(getattr(main, "volume", float("nan")))
        return centroid_xyz, volume_mm3

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        try:
            df = feats[0].data.copy().reset_index()
            if "index" in df.columns and "receptor" not in df.columns:
                df = df.rename(columns={"index": "receptor"})
            return df
        except Exception:
            return pd.DataFrame()

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        if not genes:
            return pd.DataFrame()

        feats = self._safe_features_any(
            region,
            self._modality_candidates("gene"),
            gene=list(genes),
        )
        if not feats:
            return pd.DataFrame()
        try:
            df = feats[0].data.copy()
        except Exception:
            return pd.DataFrame()

        lower_cols = {c.lower(): c for c in df.columns}
        required = {"gene", "level", "zscore"}
        if required.issubset(lower_cols):
            gene_col = lower_cols["gene"]
            level_col = lower_cols["level"]
            zscore_col = lower_cols["zscore"]
            return (
                df.groupby(gene_col, dropna=False)
                .agg(
                    level_mean=(level_col, "mean"),
                    level_std=(level_col, "std"),
                    probe_count=(level_col, "count"),
                    zscore_mean=(zscore_col, "mean"),
                    zscore_std=(zscore_col, "std"),
                )
                .reset_index()
                .rename(columns={gene_col: "gene"})
                .sort_values("gene")
                .reset_index(drop=True)
            )
        return df.reset_index(drop=True)

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix
        if self.parcellation is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        feats = self._safe_features_any(
            self.parcellation,
            self._modality_candidates("connectivity"),
        )
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort),
            feats[0],
        )

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            self._connectivity_matrix = compound[0].data.copy()
        except Exception:
            self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        if region is None:
            return None

        for x in labels:
            if x is region:
                return x

        exact = [x for x in labels if self._name_of(x) == self._name_of(region)]
        if exact:
            return exact[0]

        rn = self._name_of(region).lower()
        fuzzy = [
            x
            for x in labels
            if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn
        ]
        return fuzzy[0] if fuzzy else None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        label = self._match_region_label(list(matrix.index), region)
        axis = "index"
        if label is None:
            label = self._match_region_label(list(matrix.columns), region)
            axis = "columns"
        if label is None:
            return pd.DataFrame()

        try:
            series = matrix.loc[label] if axis == "index" else matrix[label]
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != self._name_of(region)].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return a within-model connectivity submatrix for resolved region nodes,
        using fuzzy matching against the selected connectivity matrix.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        selected_labels: Dict[str, Any] = {}
        for node_key, region in self.region_objects.items():
            row_label = self._match_region_label(list(matrix.index), region)
            col_label = self._match_region_label(list(matrix.columns), region)
            if row_label is not None and col_label is not None:
                if row_label in matrix.index and col_label in matrix.columns:
                    selected_labels[node_key] = row_label

        if not selected_labels:
            return pd.DataFrame()

        common = [lbl for lbl in selected_labels.values() if lbl in matrix.index and lbl in matrix.columns]
        if not common:
            return pd.DataFrame()

        try:
            sub = matrix.loc[common, common].copy()
        except Exception:
            return pd.DataFrame()

        inverse = {v: k for k, v in selected_labels.items()}
        sub.index = [inverse.get(x, self._name_of(x)) for x in sub.index]
        sub.columns = [inverse.get(x, self._name_of(x)) for x in sub.columns]
        return sub

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []

        for key, desc in self.input_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "input",
                    "description": desc,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            desc = self.region_node_descriptions.get(key, "Atlas-backed circuit node")
            if region is None:
                if self._atlas_available:
                    warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.upper(),
                        "node_type": "region",
                        "description": f"{desc} (unresolved in this environment)",
                        "atlas_region": None,
                        "region_identifier": None,
                        "centroid_mni": None,
                        "volume_mm3": None,
                        "feature_summary": "unresolved",
                    }
                )
                continue

            self.region_objects[key] = region
            centroid_mni, volume_mm3 = self._main_component(region)
            receptor_df = self._receptor_table(region)
            gene_df = self._gene_table(region, gene_panel)
            conn_df = self._connectivity_profile(region, max_rows=connectivity_rows)
            self.receptors[key] = receptor_df
            self.genes[key] = gene_df
            self.connectivity_profiles[key] = conn_df

            nodes.append(
                {
                    "key": key,
                    "label": self._name_of(region),
                    "node_type": "region",
                    "description": desc,
                    "atlas_region": self._name_of(region),
                    "region_identifier": getattr(region, "identifier", None),
                    "centroid_mni": centroid_mni,
                    "volume_mm3": volume_mm3,
                    "feature_summary": (
                        f"receptors={'yes' if not receptor_df.empty else 'no'}; "
                        f"genes={'yes' if not gene_df.empty else 'no'}; "
                        f"connectivity={'yes' if not conn_df.empty else 'no'}"
                    ),
                }
            )

        for key, desc in self.latent_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "latent_biology",
                    "description": desc,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, desc in self.symptom_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "symptom",
                    "description": desc,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        self.nodes_df = pd.DataFrame(nodes)
        self.edges_df = pd.DataFrame(self.edge_table)
        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": self.circuit_connectivity(),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Assign an MNI152 coordinate to Julich regions using a statistical map.

        Returns an empty dataframe if siibra or a statistical map is unavailable.
        """
        if siibra is None or self.parcellation is None:
            return pd.DataFrame()

        if self._pmap is None:
            try:
                with siibra.QUIET:
                    try:
                        self._pmap = siibra.get_map(
                            parcellation=self.parcellation_spec,
                            space=self.assignment_space,
                            maptype="statistical",
                        )
                    except Exception:
                        self._pmap = self.atlas.get_map(
                            parcellation=self.parcellation,
                            space=self.assignment_space,
                            maptype="statistical",
                        )
            except Exception:
                return pd.DataFrame()

        try:
            point = siibra.Point(tuple(xyz), space=self.assignment_space)
            with siibra.QUIET:
                assignments = self._pmap.assign(point)
        except Exception:
            return pd.DataFrame()

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str, maptype: str = "labelled") -> Any:
        """
        Return a regional mask/volume object for a resolved region node, or None.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        try:
            if hasattr(region, "get_regional_mask"):
                return region.get_regional_mask(self.assignment_space, maptype=maptype)
        except Exception:
            pass
        try:
            if hasattr(region, "fetch_regional_map"):
                return region.fetch_regional_map(self.assignment_space, maptype=maptype)
        except Exception:
            pass
        return None

    def simulate(
        self,
        polygenic_liability: float = 0.50,
        parental_depression_burden: float = 0.45,
        adverse_life_stress: float = 0.55,
        chronic_stress_load: float = 0.60,
        adolescent_onset_vulnerability: float = 0.30,
        monoamine_treatment_support: float = 0.20,
        glutamatergic_treatment_support: float = 0.05,
        recovery_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Values are clipped to [0, 1]. Higher regional-state values mean greater
        dysregulation / burden in that circuit node.
        """
        inputs = pd.Series(
            {
                "polygenic_liability": self._clip01(polygenic_liability),
                "parental_depression_burden": self._clip01(parental_depression_burden),
                "adverse_life_stress": self._clip01(adverse_life_stress),
                "chronic_stress_load": self._clip01(chronic_stress_load),
                "adolescent_onset_vulnerability": self._clip01(adolescent_onset_vulnerability),
                "monoamine_treatment_support": self._clip01(monoamine_treatment_support),
                "glutamatergic_treatment_support": self._clip01(glutamatergic_treatment_support),
                "recovery_support": self._clip01(recovery_support),
            },
            name="value",
        )

        # Inputs -> latent biology
        serotonergic_dysregulation = self._clip01(
            0.30 * inputs["polygenic_liability"]
            + 0.20 * inputs["parental_depression_burden"]
            + 0.15 * inputs["adverse_life_stress"]
            + 0.10 * inputs["chronic_stress_load"]
            + 0.10 * inputs["adolescent_onset_vulnerability"]
            - 0.30 * inputs["monoamine_treatment_support"]
            - 0.10 * inputs["recovery_support"]
        )

        noradrenergic_dysregulation = self._clip01(
            0.25 * inputs["polygenic_liability"]
            + 0.20 * inputs["chronic_stress_load"]
            + 0.15 * inputs["parental_depression_burden"]
            + 0.15 * inputs["adverse_life_stress"]
            + 0.10 * inputs["adolescent_onset_vulnerability"]
            - 0.25 * inputs["monoamine_treatment_support"]
            - 0.10 * inputs["recovery_support"]
        )

        gaba_glutamate_imbalance = self._clip01(
            0.25 * inputs["chronic_stress_load"]
            + 0.20 * inputs["adverse_life_stress"]
            + 0.15 * inputs["polygenic_liability"]
            + 0.15 * noradrenergic_dysregulation
            + 0.10 * serotonergic_dysregulation
            - 0.20 * inputs["glutamatergic_treatment_support"]
            - 0.10 * inputs["recovery_support"]
        )

        hpa_stress_sensitization = self._clip01(
            0.35 * inputs["chronic_stress_load"]
            + 0.25 * inputs["adverse_life_stress"]
            + 0.15 * inputs["parental_depression_burden"]
            + 0.10 * inputs["adolescent_onset_vulnerability"]
            + 0.10 * inputs["polygenic_liability"]
            - 0.15 * inputs["recovery_support"]
        )

        hippocampal_neuroplasticity_burden = self._clip01(
            0.35 * hpa_stress_sensitization
            + 0.20 * inputs["chronic_stress_load"]
            + 0.15 * serotonergic_dysregulation
            + 0.10 * gaba_glutamate_imbalance
            + 0.05 * inputs["parental_depression_burden"]
            - 0.10 * inputs["recovery_support"]
        )

        frontolimbic_dysregulation = self._clip01(
            0.25 * serotonergic_dysregulation
            + 0.20 * noradrenergic_dysregulation
            + 0.20 * hpa_stress_sensitization
            + 0.15 * gaba_glutamate_imbalance
            + 0.10 * hippocampal_neuroplasticity_burden
            + 0.05 * inputs["adolescent_onset_vulnerability"]
            - 0.15 * inputs["monoamine_treatment_support"]
            - 0.10 * inputs["recovery_support"]
        )

        dmn_overconnectivity = self._clip01(
            0.30 * frontolimbic_dysregulation
            + 0.25 * hpa_stress_sensitization
            + 0.20 * inputs["chronic_stress_load"]
            + 0.10 * serotonergic_dysregulation
            + 0.10 * inputs["adolescent_onset_vulnerability"]
            - 0.15 * inputs["recovery_support"]
        )

        latents = pd.Series(
            {
                "serotonergic_dysregulation": serotonergic_dysregulation,
                "noradrenergic_dysregulation": noradrenergic_dysregulation,
                "gaba_glutamate_imbalance": gaba_glutamate_imbalance,
                "hpa_stress_sensitization": hpa_stress_sensitization,
                "hippocampal_neuroplasticity_burden": hippocampal_neuroplasticity_burden,
                "frontolimbic_dysregulation": frontolimbic_dysregulation,
                "dmn_overconnectivity": dmn_overconnectivity,
            },
            name="value",
        )

        # Latent biology -> regional dysregulation burden
        regional_state = pd.Series(
            {
                "pfc_control": self._clip01(
                    0.35 * frontolimbic_dysregulation
                    + 0.20 * dmn_overconnectivity
                    + 0.15 * serotonergic_dysregulation
                    + 0.10 * noradrenergic_dysregulation
                    + 0.10 * hpa_stress_sensitization
                    - 0.20 * inputs["monoamine_treatment_support"]
                    - 0.10 * inputs["recovery_support"]
                ),
                "hippocampus": self._clip01(
                    0.40 * hippocampal_neuroplasticity_burden
                    + 0.25 * hpa_stress_sensitization
                    + 0.15 * inputs["chronic_stress_load"]
                    + 0.10 * frontolimbic_dysregulation
                    - 0.10 * inputs["recovery_support"]
                ),
                "amygdala": self._clip01(
                    0.35 * hpa_stress_sensitization
                    + 0.25 * frontolimbic_dysregulation
                    + 0.20 * gaba_glutamate_imbalance
                    + 0.10 * inputs["adverse_life_stress"]
                ),
                "thalamus_proxy": self._clip01(
                    0.30 * noradrenergic_dysregulation
                    + 0.25 * gaba_glutamate_imbalance
                    + 0.15 * frontolimbic_dysregulation
                    + 0.10 * hpa_stress_sensitization
                    + 0.10 * serotonergic_dysregulation
                    - 0.10 * inputs["recovery_support"]
                ),
                "basal_ganglia_proxy": self._clip01(
                    0.30 * serotonergic_dysregulation
                    + 0.25 * noradrenergic_dysregulation
                    + 0.20 * frontolimbic_dysregulation
                    + 0.10 * gaba_glutamate_imbalance
                    + 0.10 * hpa_stress_sensitization
                    - 0.15 * inputs["monoamine_treatment_support"]
                ),
            },
            name="value",
        )

        # Regional dysregulation -> symptoms
        depressed_affect = self._clip01(
            0.30 * frontolimbic_dysregulation
            + 0.25 * regional_state["amygdala"]
            + 0.20 * serotonergic_dysregulation
            + 0.15 * noradrenergic_dysregulation
            + 0.05 * dmn_overconnectivity
        )

        anhedonia_motivation_loss = self._clip01(
            0.35 * regional_state["basal_ganglia_proxy"]
            + 0.20 * noradrenergic_dysregulation
            + 0.15 * serotonergic_dysregulation
            + 0.10 * regional_state["pfc_control"]
            + 0.05 * regional_state["thalamus_proxy"]
            - 0.10 * inputs["monoamine_treatment_support"]
        )

        rumination = self._clip01(
            0.40 * dmn_overconnectivity
            + 0.25 * regional_state["pfc_control"]
            + 0.15 * hpa_stress_sensitization
            + 0.10 * frontolimbic_dysregulation
            + 0.05 * depressed_affect
        )

        cognitive_impairment = self._clip01(
            0.30 * regional_state["hippocampus"]
            + 0.25 * regional_state["pfc_control"]
            + 0.15 * regional_state["thalamus_proxy"]
            + 0.15 * gaba_glutamate_imbalance
            + 0.10 * dmn_overconnectivity
            - 0.10 * inputs["glutamatergic_treatment_support"]
        )

        anxiety_insomnia_burden = self._clip01(
            0.30 * gaba_glutamate_imbalance
            + 0.25 * regional_state["amygdala"]
            + 0.20 * hpa_stress_sensitization
            + 0.10 * noradrenergic_dysregulation
            + 0.05 * frontolimbic_dysregulation
            - 0.10 * inputs["glutamatergic_treatment_support"]
            - 0.05 * inputs["recovery_support"]
        )

        psychomotor_somatic_burden = self._clip01(
            0.25 * regional_state["thalamus_proxy"]
            + 0.25 * regional_state["basal_ganglia_proxy"]
            + 0.15 * noradrenergic_dysregulation
            + 0.15 * depressed_affect
            + 0.10 * anxiety_insomnia_burden
            + 0.05 * regional_state["pfc_control"]
        )

        symptoms = pd.Series(
            {
                "depressed_affect": depressed_affect,
                "anhedonia_motivation_loss": anhedonia_motivation_loss,
                "rumination": rumination,
                "cognitive_impairment": cognitive_impairment,
                "anxiety_insomnia_burden": anxiety_insomnia_burden,
                "psychomotor_somatic_burden": psychomotor_somatic_burden,
            },
            name="value",
        )

        # Symptom bundles / phenotype summaries
        ruminative_mdd_profile = self._clip01(
            0.35 * rumination
            + 0.25 * depressed_affect
            + 0.20 * dmn_overconnectivity
            + 0.10 * regional_state["pfc_control"]
        )

        anxious_depression_profile = self._clip01(
            0.35 * anxiety_insomnia_burden
            + 0.25 * depressed_affect
            + 0.15 * regional_state["amygdala"]
            + 0.10 * gaba_glutamate_imbalance
            + 0.05 * hpa_stress_sensitization
        )

        cognitive_affective_mdd_profile = self._clip01(
            0.30 * cognitive_impairment
            + 0.25 * depressed_affect
            + 0.20 * rumination
            + 0.10 * regional_state["hippocampus"]
            + 0.10 * regional_state["pfc_control"]
        )

        anhedonic_motivational_profile = self._clip01(
            0.35 * anhedonia_motivation_loss
            + 0.25 * regional_state["basal_ganglia_proxy"]
            + 0.15 * psychomotor_somatic_burden
            + 0.10 * noradrenergic_dysregulation
            + 0.05 * serotonergic_dysregulation
        )

        stress_sensitized_mdd_profile = self._clip01(
            0.30 * hpa_stress_sensitization
            + 0.25 * hippocampal_neuroplasticity_burden
            + 0.20 * depressed_affect
            + 0.10 * regional_state["amygdala"]
            + 0.10 * regional_state["hippocampus"]
        )

        global_mdd_burden = self._clip01(
            0.22 * depressed_affect
            + 0.18 * anhedonia_motivation_loss
            + 0.16 * rumination
            + 0.14 * cognitive_impairment
            + 0.15 * anxiety_insomnia_burden
            + 0.15 * psychomotor_somatic_burden
        )

        phenotypes = pd.Series(
            {
                "ruminative_mdd_profile": ruminative_mdd_profile,
                "anxious_depression_profile": anxious_depression_profile,
                "cognitive_affective_mdd_profile": cognitive_affective_mdd_profile,
                "anhedonic_motivational_profile": anhedonic_motivational_profile,
                "stress_sensitized_mdd_profile": stress_sensitized_mdd_profile,
                "global_mdd_burden": global_mdd_burden,
            },
            name="value",
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }


if __name__ == "__main__":
    model = MajorDepressiveDisorderModel()

    print("\n=== Building atlas-backed scaffold ===")
    scaffold = model.build(connectivity_rows=10)

    node_cols = ["key", "node_type", "atlas_region", "feature_summary"]
    print("\nNodes:")
    print(scaffold["nodes"][node_cols].to_string(index=False))

    print("\nEdges (first 20):")
    print(scaffold["edges"].head(20).to_string(index=False))

    if "amygdala" in scaffold["receptors"] and not scaffold["receptors"]["amygdala"].empty:
        print("\nAmygdala receptor fingerprint:")
        print(scaffold["receptors"]["amygdala"].head().to_string(index=False))
    else:
        print("\nNo amygdala receptor table available in this environment.")

    if "pfc_control" in scaffold["genes"] and not scaffold["genes"]["pfc_control"].empty:
        print("\nPFC-control gene summary:")
        print(scaffold["genes"]["pfc_control"].head().to_string(index=False))
    else:
        print("\nNo PFC-control gene table available in this environment.")

    if (
        "hippocampus" in scaffold["connectivity_profiles"]
        and not scaffold["connectivity_profiles"]["hippocampus"].empty
    ):
        print("\nHippocampus connectivity profile:")
        print(scaffold["connectivity_profiles"]["hippocampus"].head().to_string(index=False))
    else:
        print("\nNo hippocampus connectivity profile available in this environment.")

    if not scaffold["circuit_connectivity"].empty:
        print("\nWithin-model circuit connectivity:")
        print(scaffold["circuit_connectivity"].round(3).to_string())
    else:
        print("\nNo within-model circuit connectivity matrix available in this environment.")

    print("\n=== Simulation example: stress-sensitized ruminative depression ===")
    sim = model.simulate(
        polygenic_liability=0.55,
        parental_depression_burden=0.60,
        adverse_life_stress=0.65,
        chronic_stress_load=0.70,
        adolescent_onset_vulnerability=0.35,
        monoamine_treatment_support=0.20,
        glutamatergic_treatment_support=0.05,
        recovery_support=0.25,
    )
    for name, series in sim.items():
        print(f"\n{name}:")
        print(series.sort_values(ascending=False).to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((-10, -18, -12)).head())

    # Example regional mask retrieval:
    # mask = model.region_mask("hippocampus")
    # if mask is not None:
    #     print(mask)
