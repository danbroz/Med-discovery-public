from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Bipolar II-oriented panel:
# - norepinephrine / arousal / hypomania
# - GABA / inhibitory brake
# - stress / HPA-axis
# - circadian rhythm biology
# - inflammatory / thyroid-related endophenotype pressure
# - neurotrophic support
DEFAULT_GENE_PANEL = [
    "SLC6A2",   # norepinephrine transporter
    "DBH",      # dopamine beta hydroxylase
    "ADRA2A",   # adrenergic receptor
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor
    "GABRA2",   # GABA-A receptor
    "GABRB2",   # GABA-A receptor
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
    "CRHR1",    # CRH signaling
    "BDNF",     # neurotrophic support
    "CLOCK",    # circadian rhythm
    "ARNTL",    # BMAL1 circadian rhythm
    "PER3",     # circadian rhythm
    "THRA",     # thyroid hormone receptor alpha
    "THRB",     # thyroid hormone receptor beta
    "IL6",      # inflammatory signaling
    "TNF",      # inflammatory signaling
]


class BipolarIIDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Bipolar II Disorder.

    What it does:
      1) Resolves chapter-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates hypomania, bipolar depression, mixed-state pressure,
         cognitive burden, and stress-circadian instability.

    This is a research scaffold, not a clinical diagnostic tool.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.atlas = siibra.atlases.get(atlas_spec)
        self.parcellation_spec = parcellation_spec
        self.parcellation = self.atlas.parcellations.get(parcellation_spec)
        self.space_spec = space_spec
        self.space = self.atlas.spaces.get(space_spec)
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        # The chapter emphasizes orbital, dorsolateral, and medial frontal cortex,
        # ACC, thalamus, basal ganglia, hippocampus, and amygdala.
        # Striatum/thalamus are kept as proxies because exact subnuclei coverage can vary.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "DG (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "hippocampus",
            ],
            "acc": [
                "Area 33 (ACC) left",
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "ACC",
            ],
            "dlpfc": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "dorsolateral prefrontal",
                "middle frontal",
                "superior frontal",
            ],
            "mpfc": [
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "medial prefrontal",
                "pACC",
                "sACC",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "orbitofrontal",
                "Fo4",
                "Fo3",
            ],
            "striatum_proxy": [
                "caudate",
                "putamen",
                "accumbens",
                "basal ganglia",
                "striatum",
            ],
            "thalamus_proxy": [
                "thalamus",
                "anterior thalamus",
                "mediodorsal thalamus",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_liability": "Heritable liability for bipolar-spectrum illness",
            "psychosocial_stress": "Stressful life events that precipitate episodes",
            "circadian_disruption": "Sleep-wake and circadian rhythm instability",
            "inflammatory_burden": "Inflammatory burden contributing to mood dysregulation",
            "recovery_support": "Protective treatment, rhythm stabilization, and recovery structure",
        }

        self.latent_nodes: Dict[str, str] = {
            "noradrenergic_instability": "Arousal-system instability linked to hypomania and depression",
            "gaba_disinhibition": "Reduced inhibitory brake on neural excitability",
            "hpa_axis_dysregulation": "Stress-axis sensitization across mood states",
            "circadian_rhythm_instability": "Oscillatory instability linked to episode switching",
            "inflammatory_endophenotype": "Immune-inflammatory contribution to bipolar vulnerability",
            "thyroid_immune_overlap": "Autoimmune-thyroid-related endophenotype pressure",
            "bdnf_reduction": "Reduced neurotrophic support in stress-sensitive structures",
            "hypofrontality": "Reduced frontal recruitment / control, especially in depressive burden",
            "cortico_limbic_striatal_thalamic_instability": "Distributed mood-circuit dysfunction",
        }

        self.symptom_nodes: Dict[str, str] = {
            "hypomania_activation": "Hypomanic activation, high energy, and reduced need for sleep",
            "bipolar_depression": "Depressive burden, fatigue, and low drive",
            "affective_instability": "Shifts in mood intensity and emotional reactivity",
            "impulsivity_disinhibition": "Disinhibition, pressured behavior, and weak restraint",
            "psychomotor_shift": "Activation or slowing across mood states",
            "cognitive_burden": "Executive and cognitive impairment",
            "emotional_memory_bias": "Mood-congruent emotional-memory burden",
            "mixed_state_pressure": "Concurrent activation and depressive burden",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_liability",
                "target": "noradrenergic_instability",
                "relation": "raises vulnerability in arousal and attention systems",
                "bipolar_ii_change": "increased susceptibility",
            },
            {
                "source": "genetic_liability",
                "target": "gaba_disinhibition",
                "relation": "raises vulnerability in inhibitory control of excitability",
                "bipolar_ii_change": "increased susceptibility",
            },
            {
                "source": "genetic_liability",
                "target": "hpa_axis_dysregulation",
                "relation": "raises vulnerability of stress-response regulation",
                "bipolar_ii_change": "increased susceptibility",
            },
            {
                "source": "genetic_liability",
                "target": "circadian_rhythm_instability",
                "relation": "raises vulnerability of rhythm-based homeostatic control",
                "bipolar_ii_change": "increased susceptibility",
            },
            {
                "source": "genetic_liability",
                "target": "thyroid_immune_overlap",
                "relation": "supports immune-thyroid endophenotype liability",
                "bipolar_ii_change": "increased susceptibility",
            },
            {
                "source": "psychosocial_stress",
                "target": "hpa_axis_dysregulation",
                "relation": "precipitates and amplifies episode-related stress biology",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "psychosocial_stress",
                "target": "bdnf_reduction",
                "relation": "stress load can reduce trophic support",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "circadian_disruption",
                "target": "circadian_rhythm_instability",
                "relation": "destabilizes mood homeostasis and energy regulation",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "circadian_disruption",
                "target": "hypomania_activation",
                "relation": "biases toward activation and reduced sleep need",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "inflammatory_burden",
                "target": "inflammatory_endophenotype",
                "relation": "raises inflammatory signaling linked to affective burden",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "recovery_support",
                "target": "circadian_rhythm_instability",
                "relation": "buffers rhythm instability",
                "bipolar_ii_change": "protective",
            },
            {
                "source": "recovery_support",
                "target": "hpa_axis_dysregulation",
                "relation": "buffers stress sensitization",
                "bipolar_ii_change": "protective",
            },
            {
                "source": "recovery_support",
                "target": "hypomania_activation",
                "relation": "reduces escalation into severe activation",
                "bipolar_ii_change": "protective",
            },
            {
                "source": "recovery_support",
                "target": "bipolar_depression",
                "relation": "buffers depressive burden",
                "bipolar_ii_change": "protective",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "noradrenergic_instability",
                "relation": "stress hormones destabilize arousal systems",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "bdnf_reduction",
                "relation": "repeated stress lowers neurotrophic support",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "circadian_rhythm_instability",
                "target": "hypomania_activation",
                "relation": "supports high energy and reduced need for sleep",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "circadian_rhythm_instability",
                "target": "affective_instability",
                "relation": "destabilizes mood switching and homeostasis",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "inflammatory_endophenotype",
                "target": "thyroid_immune_overlap",
                "relation": "links immune dysfunction to thyroid-related bipolar vulnerability",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "inflammatory_endophenotype",
                "target": "bdnf_reduction",
                "relation": "inflammatory signaling suppresses neurotrophic support",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "inflammatory_endophenotype",
                "target": "bipolar_depression",
                "relation": "raises depressive symptom pressure",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "noradrenergic_instability",
                "target": "hypomania_activation",
                "relation": "supports hyperarousal, increased energy, and reduced sleep need",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "noradrenergic_instability",
                "target": "bipolar_depression",
                "relation": "deficit states contribute to lassitude, fatigue, and poor concentration",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "gaba_disinhibition",
                "target": "hypomania_activation",
                "relation": "reduced inhibition supports racing thoughts and pressured behavior",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "gaba_disinhibition",
                "target": "impulsivity_disinhibition",
                "relation": "reduced neural braking supports impulsive behavior",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "bdnf_reduction",
                "target": "hippocampus",
                "relation": "reduced trophic support burdens hippocampal plasticity",
                "bipolar_ii_change": "reduced support",
            },
            {
                "source": "hypofrontality",
                "target": "dlpfc",
                "relation": "reduces executive recruitment in cognitive control networks",
                "bipolar_ii_change": "reduced function",
            },
            {
                "source": "hypofrontality",
                "target": "cognitive_burden",
                "relation": "supports executive and motivational impairment",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "amygdala",
                "relation": "destabilizes limbic emotional reactivity",
                "bipolar_ii_change": "increased dysregulation",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "hippocampus",
                "relation": "destabilizes contextual and emotional-memory circuitry",
                "bipolar_ii_change": "increased dysregulation",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "acc",
                "relation": "burdens conflict monitoring and emotion integration",
                "bipolar_ii_change": "reduced function",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "dlpfc",
                "relation": "burdens executive control over mood and cognition",
                "bipolar_ii_change": "reduced function",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "mpfc",
                "relation": "burdens medial frontal emotional and self-referential regulation",
                "bipolar_ii_change": "reduced function",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "ofc",
                "relation": "burdens orbital valuation and impulse-control functions",
                "bipolar_ii_change": "reduced function",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "striatum_proxy",
                "relation": "destabilizes psychomotor and motivational circuitry",
                "bipolar_ii_change": "increased dysregulation",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "thalamus_proxy",
                "relation": "burdens relay/integration across mood circuits",
                "bipolar_ii_change": "increased dysregulation",
            },
            {
                "source": "dlpfc",
                "target": "cognitive_burden",
                "relation": "reduced dorsolateral control impairs executive function",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "ofc",
                "target": "impulsivity_disinhibition",
                "relation": "reduced orbital control weakens inhibition and judgment",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "mpfc",
                "target": "affective_instability",
                "relation": "reduced medial regulation destabilizes mood intensity",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "acc",
                "target": "mixed_state_pressure",
                "relation": "weakened conflict monitoring destabilizes mixed affective pressure",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "affective_instability",
                "relation": "amplifies emotional reactivity and instability",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "emotional_memory_bias",
                "relation": "hippocampal-temporal dysfunction biases emotional memory processing",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "emotional_memory_bias",
                "relation": "limbic emotional salience shapes mood-congruent memory bias",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "striatum_proxy",
                "target": "psychomotor_shift",
                "relation": "striatal dysfunction contributes to psychomotor change",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "cognitive_burden",
                "relation": "thalamic relay dysfunction weakens coordinated cognitive-affective control",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "hypomania_activation",
                "target": "mixed_state_pressure",
                "relation": "activation can coexist with depressive burden",
                "bipolar_ii_change": "increased",
            },
            {
                "source": "bipolar_depression",
                "target": "mixed_state_pressure",
                "relation": "depressive burden can coexist with activation pressure",
                "bipolar_ii_change": "increased",
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "thalamus",
            "striatum",
            "basal ganglia",
            "prefrontal cortex",
            "anterior cingulate",
        } else 0
        return (left_bonus, right_penalty, generic_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation_spec)
            except Exception:
                pass

            try:
                matches = self.parcellation.find(spec, filter_children=False)
                if matches:
                    matches = sorted(matches, key=self._region_rank)
                    return matches[0]
            except Exception:
                pass

        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        """
        Helper for refining proxy searches against the atlas.
        Useful for tuning striatal or thalamic candidates.
        """
        try:
            matches = self.parcellation.find(keyword, filter_children=False)
        except Exception:
            return pd.DataFrame(columns=["name", "identifier"])

        rows = []
        seen = set()
        for region in sorted(matches, key=self._region_rank):
            row = (self._name_of(region), getattr(region, "identifier", None))
            if row in seen:
                continue
            seen.add(row)
            rows.append(
                {
                    "name": row[0],
                    "identifier": row[1],
                }
            )
            if len(rows) >= limit:
                break

        return pd.DataFrame(rows)

    def _spatial_props_list(self, region: Any) -> List[Any]:
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
        self,
        region: Any,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None

        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)

        centroid = getattr(main, "centroid", None)
        if centroid is not None:
            try:
                centroid_xyz = tuple(float(x) for x in centroid)
            except Exception:
                centroid_xyz = None
        else:
            centroid_xyz = None

        try:
            volume_mm3 = float(getattr(main, "volume", float("nan")))
        except Exception:
            volume_mm3 = None

        return centroid_xyz, volume_mm3

    def _safe_features(self, concept: Any, modality: Any, **kwargs: Any) -> List[Any]:
        try:
            with siibra.QUIET:
                feats = siibra.features.get(concept, modality, **kwargs)
            return list(feats) if feats else []
        except Exception:
            return []

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features(
            region,
            siibra.features.molecular.ReceptorDensityFingerprint,
        )
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
        feats = self._safe_features(
            region,
            siibra.features.molecular.GeneExpressions,
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

            summary = (
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
            )
            return summary

        return df.reset_index(drop=True)

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features(
            self.parcellation,
            siibra.features.connectivity.StreamlineCounts,
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
        exact = [x for x in labels if self._name_of(x) == region.name]
        if exact:
            return exact[0]

        rn = region.name.lower()
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
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(
        self,
        node_keys: Sequence[str] = (
            "amygdala",
            "hippocampus",
            "acc",
            "dlpfc",
            "mpfc",
            "ofc",
            "striatum_proxy",
            "thalamus_proxy",
        ),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the Bipolar II circuit.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        labels = []
        names = []

        for key in node_keys:
            region = self.region_objects.get(key)
            if region is None:
                continue

            match = self._match_region_label(list(matrix.index), region)
            if match is None:
                continue

            labels.append(match)
            names.append(region.name)

        if not labels:
            return pd.DataFrame()

        try:
            sub = matrix.loc[labels, labels].copy()
            sub.index = names
            sub.columns = names
            return sub
        except Exception:
            return pd.DataFrame()

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
            if region is None:
                warnings.warn(f"Could not resolve a Julich region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.upper(),
                        "node_type": "region",
                        "description": "Atlas-backed node (unresolved in this environment)",
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
                    "label": region.name,
                    "node_type": "region",
                    "description": "Atlas-backed Bipolar II circuit node",
                    "atlas_region": region.name,
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

    def simulate(
        self,
        genetic_liability: float,
        psychosocial_stress: float,
        circadian_disruption: float,
        inflammatory_burden: float,
        recovery_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while recovery_support is protective.
        """
        g = self._clip01(genetic_liability)
        s = self._clip01(psychosocial_stress)
        c = self._clip01(circadian_disruption)
        i = self._clip01(inflammatory_burden)
        r = self._clip01(recovery_support)

        # Latent biology
        circadian_rhythm_instability = self._clip01(
            0.45 * c + 0.20 * g - 0.20 * r
        )
        hpa_axis_dysregulation = self._clip01(
            0.35 * s + 0.20 * g + 0.15 * circadian_rhythm_instability - 0.20 * r
        )
        inflammatory_endophenotype = self._clip01(
            0.50 * i + 0.15 * g + 0.10 * s - 0.10 * r
        )
        thyroid_immune_overlap = self._clip01(
            0.35 * inflammatory_endophenotype + 0.20 * g
        )
        bdnf_reduction = self._clip01(
            0.35 * hpa_axis_dysregulation + 0.25 * inflammatory_endophenotype
        )
        noradrenergic_instability = self._clip01(
            0.30 * g
            + 0.25 * circadian_rhythm_instability
            + 0.20 * hpa_axis_dysregulation
            + 0.10 * thyroid_immune_overlap
            - 0.10 * r
        )
        gaba_disinhibition = self._clip01(
            0.30 * g + 0.20 * circadian_rhythm_instability + 0.15 * s - 0.10 * r
        )
        hypofrontality = self._clip01(
            0.35 * bdnf_reduction
            + 0.25 * noradrenergic_instability
            + 0.10 * inflammatory_endophenotype
            - 0.20 * r
        )
        cortico_limbic_striatal_thalamic_instability = self._clip01(
            0.20 * noradrenergic_instability
            + 0.20 * hypofrontality
            + 0.15 * circadian_rhythm_instability
            + 0.15 * inflammatory_endophenotype
            + 0.15 * hpa_axis_dysregulation
            + 0.10 * gaba_disinhibition
            - 0.20 * r
        )

        # Regional state proxies
        amygdala = self._clip01(
            0.35 * cortico_limbic_striatal_thalamic_instability + 0.20 * hpa_axis_dysregulation
        )
        hippocampus = self._clip01(
            0.40 * bdnf_reduction + 0.20 * hpa_axis_dysregulation
        )
        acc = self._clip01(
            0.30 * cortico_limbic_striatal_thalamic_instability + 0.20 * hypofrontality
        )
        dlpfc = self._clip01(
            0.40 * hypofrontality + 0.15 * cortico_limbic_striatal_thalamic_instability - 0.15 * r
        )
        mpfc = self._clip01(
            0.35 * hypofrontality + 0.20 * hpa_axis_dysregulation - 0.15 * r
        )
        ofc = self._clip01(
            0.30 * hypofrontality + 0.20 * noradrenergic_instability - 0.10 * r
        )
        striatum_proxy = self._clip01(
            0.35 * noradrenergic_instability + 0.20 * cortico_limbic_striatal_thalamic_instability
        )
        thalamus_proxy = self._clip01(
            0.30 * cortico_limbic_striatal_thalamic_instability + 0.10 * circadian_rhythm_instability
        )

        # Symptoms / behavior
        hypomania_activation = self._clip01(
            0.35 * noradrenergic_instability
            + 0.25 * gaba_disinhibition
            + 0.20 * circadian_rhythm_instability
            + 0.10 * striatum_proxy
            - 0.15 * r
        )
        bipolar_depression = self._clip01(
            0.35 * bdnf_reduction
            + 0.25 * hpa_axis_dysregulation
            + 0.15 * inflammatory_endophenotype
            + 0.10 * hippocampus
            - 0.10 * r
        )
        affective_instability = self._clip01(
            0.30 * amygdala
            + 0.20 * mpfc
            + 0.20 * circadian_rhythm_instability
            + 0.10 * hypomania_activation
        )
        impulsivity_disinhibition = self._clip01(
            0.30 * gaba_disinhibition
            + 0.25 * ofc
            + 0.20 * hypomania_activation
            + 0.10 * dlpfc
            - 0.10 * r
        )
        psychomotor_shift = self._clip01(
            0.30 * striatum_proxy
            + 0.20 * thalamus_proxy
            + 0.15 * hypomania_activation
            + 0.10 * bipolar_depression
        )
        cognitive_burden = self._clip01(
            0.35 * dlpfc + 0.20 * acc + 0.15 * thalamus_proxy + 0.10 * hippocampus
        )
        emotional_memory_bias = self._clip01(
            0.35 * hippocampus + 0.25 * amygdala + 0.15 * bipolar_depression
        )
        mixed_state_pressure = self._clip01(
            0.35 * hypomania_activation
            + 0.30 * bipolar_depression
            + 0.15 * affective_instability
            + 0.10 * noradrenergic_instability
        )

        return {
            "inputs": pd.Series(
                {
                    "genetic_liability": g,
                    "psychosocial_stress": s,
                    "circadian_disruption": c,
                    "inflammatory_burden": i,
                    "recovery_support": r,
                }
            ),
            "latents": pd.Series(
                {
                    "cortico_limbic_striatal_thalamic_instability": cortico_limbic_striatal_thalamic_instability,
                    "noradrenergic_instability": noradrenergic_instability,
                    "gaba_disinhibition": gaba_disinhibition,
                    "hpa_axis_dysregulation": hpa_axis_dysregulation,
                    "circadian_rhythm_instability": circadian_rhythm_instability,
                    "inflammatory_endophenotype": inflammatory_endophenotype,
                    "thyroid_immune_overlap": thyroid_immune_overlap,
                    "bdnf_reduction": bdnf_reduction,
                    "hypofrontality": hypofrontality,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "amygdala": amygdala,
                    "hippocampus": hippocampus,
                    "acc": acc,
                    "dlpfc": dlpfc,
                    "mpfc": mpfc,
                    "ofc": ofc,
                    "striatum_proxy": striatum_proxy,
                    "thalamus_proxy": thalamus_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "hypomania_activation": hypomania_activation,
                    "bipolar_depression": bipolar_depression,
                    "mixed_state_pressure": mixed_state_pressure,
                    "affective_instability": affective_instability,
                    "impulsivity_disinhibition": impulsivity_disinhibition,
                    "psychomotor_shift": psychomotor_shift,
                    "cognitive_burden": cognitive_burden,
                    "emotional_memory_bias": emotional_memory_bias,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "hypomania_dominant_profile": self._clip01(
                        0.45 * hypomania_activation
                        + 0.25 * impulsivity_disinhibition
                        + 0.15 * psychomotor_shift
                    ),
                    "bipolar_depression_dominant_profile": self._clip01(
                        0.45 * bipolar_depression
                        + 0.25 * cognitive_burden
                        + 0.15 * bdnf_reduction
                    ),
                    "mixed_instability_profile": self._clip01(
                        0.50 * mixed_state_pressure
                        + 0.20 * affective_instability
                        + 0.15 * emotional_memory_bias
                    ),
                    "stress_circadian_profile": self._clip01(
                        0.35 * hpa_axis_dysregulation
                        + 0.30 * circadian_rhythm_instability
                        + 0.15 * affective_instability
                        + 0.10 * hypomania_activation
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-20, 24, -10)).head(10)
        """
        if self._pmap is None:
            with siibra.QUIET:
                self._pmap = siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=self.assignment_space,
                    maptype="statistical",
                )

        point = siibra.Point(tuple(xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments

    def region_mask(self, node_key: str):
        """
        Return a siibra regional mask object for a resolved node.
        Use .fetch() to obtain the NIfTI image.
        """
        region = self.region_objects[node_key]
        return region.get_regional_mask(self.space, maptype="labelled")


if __name__ == "__main__":
    model = BipolarIIDisorderModel()

    # Build atlas-backed graph + evidence tables
    bundle = model.build()

    print("\n=== NODES ===")
    print(
        bundle["nodes"][
            ["key", "node_type", "atlas_region", "centroid_mni", "feature_summary"]
        ].to_string(index=False)
    )

    print("\n=== EDGES ===")
    print(
        bundle["edges"][["source", "target", "relation", "bipolar_ii_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["amygdala", "hippocampus", "acc", "dlpfc", "mpfc", "ofc", "striatum_proxy", "thalamus_proxy"]:
        print(f"\n=== {key.upper()} : receptor fingerprint ===")
        if not bundle["receptors"][key].empty:
            print(bundle["receptors"][key].head(10).to_string(index=False))
        else:
            print("No receptor fingerprint available for this node.")

        print(f"\n=== {key.upper()} : gene panel summary ===")
        if not bundle["genes"][key].empty:
            print(bundle["genes"][key].to_string(index=False))
        else:
            print("No gene-expression summary available for this node.")

        print(f"\n=== {key.upper()} : top structural connectivity ===")
        if not bundle["connectivity_profiles"][key].empty:
            print(bundle["connectivity_profiles"][key].head(10).to_string(index=False))
        else:
            print("No connectivity profile available for this node.")

    # Example simulation
    sim = model.simulate(
        genetic_liability=0.80,
        psychosocial_stress=0.70,
        circadian_disruption=0.85,
        inflammatory_burden=0.45,
        recovery_support=0.20,
    )

    print("\n=== LATENT BIOLOGY ===")
    print(sim["latents"].to_string())

    print("\n=== REGIONAL STATE ===")
    print(sim["regional_state"].to_string())

    print("\n=== SYMPTOMS ===")
    print(sim["symptoms"].to_string())

    print("\n=== PHENOTYPES ===")
    print(sim["phenotypes"].to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((-20, 24, -10)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("thalamus").to_string(index=False))
    # print(model.suggest_regions("striatum").to_string(index=False))
