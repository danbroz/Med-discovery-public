from __future__ import annotations

"""
Pyromania atlas-grounded siibra scaffold.

This script translates a chapter-level biological discussion of Pyromania into
an explicit, research-oriented mechanistic graph. The chapter emphasizes that
Pyromania is biologically under-studied, so the scaffold is intentionally
inferential and conservative. It centers the following chapter themes:

- pre-fire tension and autonomic arousal,
- stress-system dysregulation involving norepinephrine and cortisol / HPA-axis load,
- impaired top-down inhibition from prefrontal control systems,
- amygdala-centered salience hyper-reactivity to fire-related cues,
- reward / relief learning that reinforces the tension-discharge cycle,
- frontal-basal-ganglia involvement in action gating and response inhibition.

Important:
- This is a research scaffold, not a diagnostic or treatment tool.
- It is not an operational guide for setting fires or evading detection.
- Higher simulated values indicate greater dysregulation burden or symptom
  pressure unless explicitly noted otherwise.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore[assignment]
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_PYROMANIA_GENE_PANEL = [
    "SLC6A2",   # norepinephrine transporter
    "DBH",      # dopamine beta-hydroxylase / catecholamine balance
    "COMT",     # catecholamine degradation / executive control
    "MAOA",     # monoamine degradation / impulsive aggression literature
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor 1A
    "HTR2A",    # serotonin receptor 2A
    "SLC6A3",   # dopamine transporter
    "DRD2",     # dopamine receptor D2
    "DRD4",     # dopamine receptor D4 / novelty seeking literature
    "CRHR1",    # corticotropin-releasing hormone receptor 1
    "FKBP5",    # stress sensitivity / trauma-related regulation
    "NR3C1",    # glucocorticoid receptor
    "BDNF",     # neuroplastic adaptation under chronic stress
]


class PyromaniaModel:
    """
    Atlas-grounded research scaffold for Pyromania.

    The motivating chapter frames Pyromania as an interaction among escalating
    stress-arousal, poor impulse inhibition, limbic cue-reactivity to fire, and
    relief-based reinforcement after the act. Because the chapter is explicitly
    theoretical and not tied to a settled lesion model, this scaffold mixes
    atlas-backed regions with clearly labeled proxy nodes.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        if siibra is None:
            raise ImportError(
                "siibra is required to use this scaffold. Install it in your Python "
                "environment before running the model."
            ) from _SIIBRA_IMPORT_ERROR

        self.atlas = siibra.atlases.get(atlas_spec)
        self.parcellation_spec = parcellation_spec
        self.parcellation = (
            self.atlas.get_parcellation(parcellation_spec)
            if hasattr(self.atlas, "get_parcellation")
            else self.atlas.parcellations.get(parcellation_spec)
        )
        self.space_spec = space_spec
        self.space = (
            self.atlas.get_space(space_spec)
            if hasattr(self.atlas, "get_space")
            else self.atlas.spaces.get(space_spec)
        )
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        # Conservative atlas anchors. vmPFC is modeled as a proxy because exact
        # Julich labels vary across medial/orbitomedial subregions. The chapter
        # explicitly names vmPFC, dlPFC, amygdala, ventral striatum, and a
        # frontal-basal-ganglia inhibitory network.
        self.region_candidates: Dict[str, List[str]] = {
            "vmpfc_proxy": [
                "Area p32pr left",
                "Area s32 left",
                "Area p24pr left",
                "Area 25 left",
                "Area 14m left",
                "ventromedial prefrontal cortex",
                "medial prefrontal cortex",
            ],
            "dlpfc": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 9 left",
                "Area 45 left",
                "dorsolateral prefrontal cortex",
                "prefrontal cortex",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "ventral_striatum_proxy": [
                "nucleus accumbens left",
                "caudate left",
                "putamen left",
                "striatum",
                "basal ganglia",
            ],
            "basal_ganglia_proxy": [
                "caudate left",
                "putamen left",
                "pallidum left",
                "basal ganglia",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "vmpfc_proxy": (
                "Proxy for ventromedial prefrontal control circuitry integrating affective value, consequence "
                "prediction, and inhibition of destructive impulses."
            ),
            "dlpfc": (
                "Dorsolateral prefrontal control node representing executive control, cognitive stopping, and "
                "response inhibition."
            ),
            "amygdala": (
                "Amygdala node representing emotional salience, arousal amplification, and fire-cue reactivity."
            ),
            "ventral_striatum_proxy": (
                "Proxy for nucleus-accumbens / ventral-striatal reinforcement circuitry implicated in reward and "
                "relief learning around the act of fire-setting."
            ),
            "basal_ganglia_proxy": (
                "Proxy for frontal-basal-ganglia action-gating circuitry involved in inhibitory control and the "
                "processing of sudden, highly salient events."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_temperamental_liability": (
                "Inherited and temperamental vulnerability affecting aggression, negative affect, impulsivity, and "
                "stress responsivity."
            ),
            "early_life_stress_trauma": (
                "Early adversity that may bias later amygdala-prefrontal development and stress-reactive gene expression."
            ),
            "chronic_stress_load": (
                "Sustained stress burden that increases baseline irritability, anxiety, and physiological arousal."
            ),
            "acute_tension_arousal": (
                "Immediate build-up of subjective tension, restlessness, or arousal preceding the act."
            ),
            "fire_cue_exposure": (
                "Exposure to fire-related cues, imagery, or contextual triggers that potentiate salience and fascination."
            ),
            "relief_reinforcement_history": (
                "Prior learning that fire-setting or witnessing the aftermath produces gratification or relief."
            ),
            "inhibitory_support": (
                "Protective executive scaffolding from treatment, supervision, structure, or practiced impulse-control support."
            ),
            "stress_regulation_support": (
                "Protective support that lowers autonomic arousal and improves tension regulation before urges escalate."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "noradrenergic_hyperarousal": (
                "Catecholamine-driven hyperarousal state contributing to fight-or-flight activation and escalating tension."
            ),
            "hpa_axis_stress_sensitization": (
                "Cortisol-linked stress sensitization that maintains a chronically dysregulated arousal baseline."
            ),
            "frontolimbic_disconnection": (
                "Weak functional coupling between prefrontal regulation systems and limbic emotion-generation systems."
            ),
            "prefrontal_inhibitory_failure": (
                "Failure of vmPFC/dlPFC systems to suppress destructive impulses and integrate consequences."
            ),
            "amygdala_fire_salience_hyperreactivity": (
                "Exaggerated limbic response to fire-related cues, internal tension states, and dramatic sensory salience."
            ),
            "reward_relief_reinforcement_bias": (
                "Bias toward reward and negative reinforcement in which fire becomes linked to gratification or relief."
            ),
            "frontal_basal_ganglia_gating_dysfunction": (
                "Dysfunction in frontal-basal-ganglia response-gating networks that should interrupt impulsive action sequences."
            ),
            "tension_discharge_learning_loop": (
                "Learned expectation that setting or observing a fire will discharge escalating tension and restore relief."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "fire_fascination_preoccupation": (
                "Persistent fascination or preoccupation with fire and its contextual elements."
            ),
            "pre_fire_tension_arousal": (
                "Rising tension, anxiety, or physiological arousal before fire-setting."
            ),
            "fire_setting_urge": (
                "Intensifying urge to initiate fire-setting as arousal and fascination converge."
            ),
            "impaired_resistance_to_fire_setting": (
                "Diminished capacity to inhibit the urge despite awareness of danger or consequences."
            ),
            "recurrent_deliberate_fire_setting": (
                "Recurrent, deliberate, noninstrumental fire-setting behavior."
            ),
            "post_fire_relief_gratification": (
                "Pleasure, gratification, or relief after ignition or observing the aftermath."
            ),
            "functional_impairment_risk": (
                "Overall impairment burden and risk associated with repeated fire-setting and preoccupation."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_temperamental_liability",
                "target": "noradrenergic_hyperarousal",
                "relation": "raises inherited vulnerability in arousal and threat-response systems",
                "pyromania_change": "increased",
            },
            {
                "source": "genetic_temperamental_liability",
                "target": "prefrontal_inhibitory_failure",
                "relation": "raises trait-level vulnerability in impulse inhibition and emotional regulation",
                "pyromania_change": "increased",
            },
            {
                "source": "early_life_stress_trauma",
                "target": "hpa_axis_stress_sensitization",
                "relation": "can produce long-term stress sensitization and epigenetic biasing",
                "pyromania_change": "increased",
            },
            {
                "source": "early_life_stress_trauma",
                "target": "frontolimbic_disconnection",
                "relation": "can weaken amygdala-prefrontal coupling through developmental stress effects",
                "pyromania_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "noradrenergic_hyperarousal",
                "relation": "maintains elevated fight-or-flight readiness",
                "pyromania_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "hpa_axis_stress_sensitization",
                "relation": "raises stress-hormone burden and lowers regulatory reserve",
                "pyromania_change": "increased",
            },
            {
                "source": "acute_tension_arousal",
                "target": "noradrenergic_hyperarousal",
                "relation": "directly amplifies short-term arousal pressure before the act",
                "pyromania_change": "increased",
            },
            {
                "source": "acute_tension_arousal",
                "target": "pre_fire_tension_arousal",
                "relation": "contributes directly to the subjective buildup that precedes fire-setting",
                "pyromania_change": "increased",
            },
            {
                "source": "fire_cue_exposure",
                "target": "amygdala_fire_salience_hyperreactivity",
                "relation": "provides salient trigger cues that amplify emotional and sensory reactivity",
                "pyromania_change": "increased",
            },
            {
                "source": "fire_cue_exposure",
                "target": "reward_relief_reinforcement_bias",
                "relation": "strengthens associative attraction to fire-related cues and context",
                "pyromania_change": "increased",
            },
            {
                "source": "relief_reinforcement_history",
                "target": "reward_relief_reinforcement_bias",
                "relation": "strengthens reward and negative-reinforcement expectations",
                "pyromania_change": "increased",
            },
            {
                "source": "relief_reinforcement_history",
                "target": "tension_discharge_learning_loop",
                "relation": "conditions fire-setting as a learned route to tension discharge",
                "pyromania_change": "increased",
            },
            {
                "source": "inhibitory_support",
                "target": "prefrontal_inhibitory_failure",
                "relation": "buffers control failure by supporting cognitive stopping and consequence monitoring",
                "pyromania_change": "decreased",
            },
            {
                "source": "stress_regulation_support",
                "target": "hpa_axis_stress_sensitization",
                "relation": "buffers chronic arousal and lowers stress-linked escalation",
                "pyromania_change": "decreased",
            },
            {
                "source": "stress_regulation_support",
                "target": "noradrenergic_hyperarousal",
                "relation": "reduces sympathetic overactivation and acute tension build-up",
                "pyromania_change": "decreased",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "prefrontal_inhibitory_failure",
                "relation": "impairs reflective inhibition under high arousal",
                "pyromania_change": "increased",
            },
            {
                "source": "hpa_axis_stress_sensitization",
                "target": "frontolimbic_disconnection",
                "relation": "sustains poor emotion regulation and top-down control",
                "pyromania_change": "increased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "amygdala_fire_salience_hyperreactivity",
                "relation": "permits more poorly modulated limbic reactivity",
                "pyromania_change": "increased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "prefrontal_inhibitory_failure",
                "relation": "reduces top-down control over arousal and action impulses",
                "pyromania_change": "increased",
            },
            {
                "source": "reward_relief_reinforcement_bias",
                "target": "tension_discharge_learning_loop",
                "relation": "strengthens expectation that the act will be rewarding or relieving",
                "pyromania_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_failure",
                "target": "frontal_basal_ganglia_gating_dysfunction",
                "relation": "disrupts the inhibitory network that should stop destructive action sequences",
                "pyromania_change": "increased",
            },
            {
                "source": "amygdala_fire_salience_hyperreactivity",
                "target": "vmpfc_proxy",
                "relation": "increases dysfunction burden in affective valuation and regulatory control circuitry",
                "pyromania_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_failure",
                "target": "vmpfc_proxy",
                "relation": "raises dysfunction burden in ventromedial control circuitry",
                "pyromania_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_failure",
                "target": "dlpfc",
                "relation": "raises dysfunction burden in executive control circuitry",
                "pyromania_change": "increased",
            },
            {
                "source": "amygdala_fire_salience_hyperreactivity",
                "target": "amygdala",
                "relation": "raises limbic arousal burden during cue exposure and tension states",
                "pyromania_change": "increased",
            },
            {
                "source": "reward_relief_reinforcement_bias",
                "target": "ventral_striatum_proxy",
                "relation": "increases dysregulation burden in reward and incentive-salience circuitry",
                "pyromania_change": "increased",
            },
            {
                "source": "frontal_basal_ganglia_gating_dysfunction",
                "target": "basal_ganglia_proxy",
                "relation": "raises action-gating dysregulation within frontal-basal-ganglia loops",
                "pyromania_change": "increased",
            },
            {
                "source": "reward_relief_reinforcement_bias",
                "target": "fire_fascination_preoccupation",
                "relation": "increases motivational pull and preoccupation with fire-related stimuli",
                "pyromania_change": "increased",
            },
            {
                "source": "amygdala_fire_salience_hyperreactivity",
                "target": "fire_fascination_preoccupation",
                "relation": "amplifies emotional salience of fire and contextual cues",
                "pyromania_change": "increased",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "pre_fire_tension_arousal",
                "relation": "drives bodily and subjective escalation before the act",
                "pyromania_change": "increased",
            },
            {
                "source": "tension_discharge_learning_loop",
                "target": "fire_setting_urge",
                "relation": "turns rising tension into a learned impulse toward fire-setting",
                "pyromania_change": "increased",
            },
            {
                "source": "fire_fascination_preoccupation",
                "target": "fire_setting_urge",
                "relation": "adds cue-driven motivational pull to the urge state",
                "pyromania_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_failure",
                "target": "impaired_resistance_to_fire_setting",
                "relation": "weakens the ability to suppress the urge",
                "pyromania_change": "increased",
            },
            {
                "source": "frontal_basal_ganglia_gating_dysfunction",
                "target": "impaired_resistance_to_fire_setting",
                "relation": "reduces effective stopping of the behavioral sequence",
                "pyromania_change": "increased",
            },
            {
                "source": "fire_setting_urge",
                "target": "recurrent_deliberate_fire_setting",
                "relation": "provides direct motivational pressure toward repeated fire-setting",
                "pyromania_change": "increased",
            },
            {
                "source": "impaired_resistance_to_fire_setting",
                "target": "recurrent_deliberate_fire_setting",
                "relation": "allows urges to progress into overt acts",
                "pyromania_change": "increased",
            },
            {
                "source": "recurrent_deliberate_fire_setting",
                "target": "post_fire_relief_gratification",
                "relation": "produces gratification or relief after ignition or observation",
                "pyromania_change": "increased",
            },
            {
                "source": "recurrent_deliberate_fire_setting",
                "target": "functional_impairment_risk",
                "relation": "increases clinical and real-world burden through repeated destructive acts",
                "pyromania_change": "increased",
            },
            {
                "source": "fire_fascination_preoccupation",
                "target": "functional_impairment_risk",
                "relation": "increases preoccupation burden and downstream impairment",
                "pyromania_change": "increased",
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
        try:
            matches = self.atlas.find_regions(
                query,
                all_versions=False,
                filter_children=False,
                find_topmost=False,
            )
        except Exception:
            return []

        out: List[Any] = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower():
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "prefrontal cortex",
            "medial prefrontal cortex",
            "striatum",
            "basal ganglia",
        } else 0
        proxy_penalty = 1 if "proxy" in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                region = self.atlas.get_region(spec, parcellation=self.parcellation)
                if region is not None:
                    return region
            except Exception:
                pass
            try:
                region = self.parcellation.get_region(spec)
                if region is not None:
                    return region
            except Exception:
                pass
            matches = self._julich_matches(spec)
            if matches:
                return sorted(matches, key=self._region_rank)[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
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
                {
                    "name": row[0],
                    "identifier": row[1],
                    "parcellation": row[2],
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

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz = tuple(float(v) for v in centroid) if centroid is not None else None
        volume = getattr(main, "volume", None)
        volume_mm3 = float(volume) if volume is not None else None
        return centroid_xyz, volume_mm3

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()

        for feat in feats:
            try:
                df = feat.data.copy().reset_index()
                if "index" in df.columns and "receptor" not in df.columns:
                    df = df.rename(columns={"index": "receptor"})
                return df
            except Exception:
                continue
        return pd.DataFrame()

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()

        for feat in feats:
            try:
                df = feat.data.copy()
            except Exception:
                continue
            lower_cols = {str(c).lower(): c for c in df.columns}
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

        return pd.DataFrame()

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), feats[0])

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
        try:
            object_exact = [x for x in labels if x == region]
            if object_exact:
                return object_exact[0]
        except Exception:
            pass

        exact = [x for x in labels if self._name_of(x) == region.name]
        if exact:
            return exact[0]
        rn = region.name.lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        return fuzzy[0] if fuzzy else None

    @staticmethod
    def _coerce_numeric_scalar(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            pass

        try:
            if isinstance(value, pd.Series):
                arr = pd.to_numeric(value, errors="coerce").dropna().to_numpy()
            elif isinstance(value, pd.DataFrame):
                arr = pd.to_numeric(pd.Series(value.to_numpy().ravel()), errors="coerce").dropna().to_numpy()
            else:
                arr = pd.to_numeric(pd.Series([value]), errors="coerce").dropna().to_numpy()
        except Exception:
            return None

        if len(arr) == 0:
            return None
        return float(arr.mean())

    def _connectivity_selection_to_series(self, selection: Any, axis: str) -> pd.Series:
        if isinstance(selection, pd.Series):
            series = selection.copy()
        elif isinstance(selection, pd.DataFrame):
            # Duplicate row/column labels can make a scalar lookup or profile query
            # return a DataFrame instead of a Series. Collapse conservatively.
            series = selection.mean(axis=0) if axis == "index" else selection.mean(axis=1)
        else:
            scalar = self._coerce_numeric_scalar(selection)
            if scalar is None:
                return pd.Series(dtype=float)
            return pd.Series([scalar])

        series = pd.to_numeric(series, errors="coerce").dropna()
        if series.empty:
            return pd.Series(dtype=float)

        named_index = [self._name_of(x) for x in series.index]
        series.index = named_index
        if series.index.has_duplicates:
            series = series.groupby(level=0).mean()
        return series.sort_values(ascending=False)

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
            selection = matrix.loc[label] if axis == "index" else matrix[label]
            series = self._connectivity_selection_to_series(selection, axis=axis)
            if series.empty:
                return pd.DataFrame()
            df = series.reset_index()
            df.columns = ["connected_region", "value"]
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return pairwise connectivity values among the scaffold's resolved circuit nodes.
        Missing or unresolved regions are skipped.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        matched_rows: Dict[str, Any] = {}
        matched_cols: Dict[str, Any] = {}

        for key, region in self.region_objects.items():
            row_label = self._match_region_label(list(matrix.index), region)
            col_label = self._match_region_label(list(matrix.columns), region)
            if row_label is not None:
                matched_rows[key] = row_label
            if col_label is not None:
                matched_cols[key] = col_label

        keys = list(self.region_objects.keys())
        for src in keys:
            for dst in keys:
                if src == dst:
                    continue
                row_label = matched_rows.get(src)
                col_label = matched_cols.get(dst)
                if row_label is None or col_label is None:
                    continue
                try:
                    value = matrix.loc[row_label, col_label]
                except Exception:
                    continue
                scalar = self._coerce_numeric_scalar(value)
                if scalar is None:
                    continue
                rows.append(
                    {
                        "source_key": src,
                        "source_region": self.region_objects[src].name,
                        "target_key": dst,
                        "target_region": self.region_objects[dst].name,
                        "value": scalar,
                    }
                )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_PYROMANIA_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
        """
        Build the atlas-grounded graph and collect multimodal summaries.
        """
        nodes: List[Dict[str, Any]] = []
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
        self._connectivity_matrix = None

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
            desc = self.region_descriptions.get(key, "Atlas-backed circuit node")

            if region is None:
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region",
                        "description": f"{desc} Unresolved in this siibra environment; keep as an explicit proxy.",
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
                    "description": desc,
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
        genetic_temperamental_liability: float = 0.50,
        early_life_stress_trauma: float = 0.35,
        chronic_stress_load: float = 0.55,
        acute_tension_arousal: float = 0.60,
        fire_cue_exposure: float = 0.50,
        relief_reinforcement_history: float = 0.40,
        inhibitory_support: float = 0.30,
        stress_regulation_support: float = 0.30,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator for the scaffold.

        Inputs are clipped to 0..1. Protective variables subtract from
        dysregulation. The calculation is acyclic:
            inputs -> latent biology -> regional burden -> symptoms -> phenotypes
        """
        inputs = {
            "genetic_temperamental_liability": self._clip01(genetic_temperamental_liability),
            "early_life_stress_trauma": self._clip01(early_life_stress_trauma),
            "chronic_stress_load": self._clip01(chronic_stress_load),
            "acute_tension_arousal": self._clip01(acute_tension_arousal),
            "fire_cue_exposure": self._clip01(fire_cue_exposure),
            "relief_reinforcement_history": self._clip01(relief_reinforcement_history),
            "inhibitory_support": self._clip01(inhibitory_support),
            "stress_regulation_support": self._clip01(stress_regulation_support),
        }

        latents = {
            "noradrenergic_hyperarousal": self._clip01(
                0.34 * inputs["acute_tension_arousal"]
                + 0.24 * inputs["chronic_stress_load"]
                + 0.16 * inputs["genetic_temperamental_liability"]
                + 0.08 * inputs["fire_cue_exposure"]
                - 0.18 * inputs["stress_regulation_support"]
            ),
            "hpa_axis_stress_sensitization": self._clip01(
                0.38 * inputs["early_life_stress_trauma"]
                + 0.30 * inputs["chronic_stress_load"]
                + 0.16 * inputs["acute_tension_arousal"]
                - 0.24 * inputs["stress_regulation_support"]
            ),
            "reward_relief_reinforcement_bias": self._clip01(
                0.34 * inputs["relief_reinforcement_history"]
                + 0.22 * inputs["fire_cue_exposure"]
                + 0.16 * inputs["acute_tension_arousal"]
                + 0.12 * inputs["genetic_temperamental_liability"]
                - 0.12 * inputs["inhibitory_support"]
            ),
        }

        latents["frontolimbic_disconnection"] = self._clip01(
            0.38 * inputs["early_life_stress_trauma"]
            + 0.24 * latents["hpa_axis_stress_sensitization"]
            + 0.14 * inputs["genetic_temperamental_liability"]
            - 0.18 * inputs["inhibitory_support"]
            - 0.10 * inputs["stress_regulation_support"]
        )
        latents["prefrontal_inhibitory_failure"] = self._clip01(
            0.30 * latents["noradrenergic_hyperarousal"]
            + 0.24 * latents["frontolimbic_disconnection"]
            + 0.18 * latents["hpa_axis_stress_sensitization"]
            + 0.12 * inputs["acute_tension_arousal"]
            + 0.08 * inputs["genetic_temperamental_liability"]
            - 0.28 * inputs["inhibitory_support"]
        )
        latents["amygdala_fire_salience_hyperreactivity"] = self._clip01(
            0.32 * inputs["fire_cue_exposure"]
            + 0.24 * latents["noradrenergic_hyperarousal"]
            + 0.20 * latents["frontolimbic_disconnection"]
            + 0.10 * inputs["acute_tension_arousal"]
            - 0.12 * inputs["stress_regulation_support"]
        )
        latents["frontal_basal_ganglia_gating_dysfunction"] = self._clip01(
            0.30 * latents["prefrontal_inhibitory_failure"]
            + 0.22 * latents["amygdala_fire_salience_hyperreactivity"]
            + 0.16 * inputs["acute_tension_arousal"]
            + 0.12 * inputs["fire_cue_exposure"]
            + 0.10 * latents["reward_relief_reinforcement_bias"]
            - 0.12 * inputs["inhibitory_support"]
        )
        latents["tension_discharge_learning_loop"] = self._clip01(
            0.30 * latents["reward_relief_reinforcement_bias"]
            + 0.24 * inputs["acute_tension_arousal"]
            + 0.20 * latents["hpa_axis_stress_sensitization"]
            + 0.14 * inputs["relief_reinforcement_history"]
            + 0.08 * latents["amygdala_fire_salience_hyperreactivity"]
            - 0.10 * inputs["stress_regulation_support"]
        )

        regional_state = {
            "vmpfc_proxy": self._clip01(
                0.48 * latents["prefrontal_inhibitory_failure"]
                + 0.22 * latents["frontolimbic_disconnection"]
                + 0.12 * latents["hpa_axis_stress_sensitization"]
                + 0.08 * inputs["acute_tension_arousal"]
                - 0.22 * inputs["inhibitory_support"]
            ),
            "dlpfc": self._clip01(
                0.46 * latents["prefrontal_inhibitory_failure"]
                + 0.24 * latents["noradrenergic_hyperarousal"]
                + 0.14 * latents["frontolimbic_disconnection"]
                + 0.08 * inputs["acute_tension_arousal"]
                - 0.24 * inputs["inhibitory_support"]
            ),
            "amygdala": self._clip01(
                0.54 * latents["amygdala_fire_salience_hyperreactivity"]
                + 0.20 * latents["frontolimbic_disconnection"]
                + 0.12 * inputs["fire_cue_exposure"]
                - 0.10 * inputs["stress_regulation_support"]
            ),
            "ventral_striatum_proxy": self._clip01(
                0.50 * latents["reward_relief_reinforcement_bias"]
                + 0.24 * latents["tension_discharge_learning_loop"]
                + 0.14 * inputs["fire_cue_exposure"]
                - 0.10 * inputs["inhibitory_support"]
            ),
            "basal_ganglia_proxy": self._clip01(
                0.46 * latents["frontal_basal_ganglia_gating_dysfunction"]
                + 0.18 * latents["prefrontal_inhibitory_failure"]
                + 0.12 * inputs["acute_tension_arousal"]
                + 0.10 * latents["reward_relief_reinforcement_bias"]
                - 0.10 * inputs["inhibitory_support"]
            ),
        }

        symptoms = {
            "fire_fascination_preoccupation": self._clip01(
                0.34 * regional_state["amygdala"]
                + 0.26 * regional_state["ventral_striatum_proxy"]
                + 0.16 * inputs["fire_cue_exposure"]
                + 0.14 * latents["reward_relief_reinforcement_bias"]
                + 0.06 * inputs["relief_reinforcement_history"]
            ),
            "pre_fire_tension_arousal": self._clip01(
                0.34 * latents["noradrenergic_hyperarousal"]
                + 0.28 * latents["hpa_axis_stress_sensitization"]
                + 0.20 * inputs["acute_tension_arousal"]
                + 0.10 * regional_state["amygdala"]
                - 0.14 * inputs["stress_regulation_support"]
            ),
        }

        symptoms["fire_setting_urge"] = self._clip01(
            0.32 * symptoms["fire_fascination_preoccupation"]
            + 0.24 * symptoms["pre_fire_tension_arousal"]
            + 0.20 * latents["tension_discharge_learning_loop"]
            + 0.14 * regional_state["ventral_striatum_proxy"]
            + 0.06 * inputs["fire_cue_exposure"]
        )
        symptoms["impaired_resistance_to_fire_setting"] = self._clip01(
            0.32 * latents["prefrontal_inhibitory_failure"]
            + 0.24 * latents["frontal_basal_ganglia_gating_dysfunction"]
            + 0.20 * symptoms["fire_setting_urge"]
            + 0.10 * regional_state["dlpfc"]
            + 0.06 * regional_state["vmpfc_proxy"]
            - 0.12 * inputs["inhibitory_support"]
        )
        symptoms["recurrent_deliberate_fire_setting"] = self._clip01(
            0.30 * symptoms["fire_setting_urge"]
            + 0.24 * symptoms["impaired_resistance_to_fire_setting"]
            + 0.18 * latents["tension_discharge_learning_loop"]
            + 0.12 * symptoms["fire_fascination_preoccupation"]
            + 0.10 * regional_state["basal_ganglia_proxy"]
        )
        symptoms["post_fire_relief_gratification"] = self._clip01(
            0.42 * symptoms["recurrent_deliberate_fire_setting"]
            + 0.30 * latents["tension_discharge_learning_loop"]
            + 0.18 * latents["reward_relief_reinforcement_bias"]
            + 0.06 * symptoms["pre_fire_tension_arousal"]
        )
        symptoms["functional_impairment_risk"] = self._clip01(
            0.28 * symptoms["recurrent_deliberate_fire_setting"]
            + 0.22 * symptoms["impaired_resistance_to_fire_setting"]
            + 0.18 * symptoms["fire_fascination_preoccupation"]
            + 0.16 * symptoms["pre_fire_tension_arousal"]
            + 0.08 * inputs["chronic_stress_load"]
        )

        phenotypes = {
            "pyromania_core_profile": self._clip01(
                (
                    symptoms["fire_setting_urge"]
                    + symptoms["impaired_resistance_to_fire_setting"]
                    + symptoms["recurrent_deliberate_fire_setting"]
                    + symptoms["post_fire_relief_gratification"]
                ) / 4.0
            ),
            "stress_discharging_fire_setting_profile": self._clip01(
                (
                    latents["noradrenergic_hyperarousal"]
                    + latents["hpa_axis_stress_sensitization"]
                    + symptoms["pre_fire_tension_arousal"]
                    + symptoms["post_fire_relief_gratification"]
                ) / 4.0
            ),
            "cue_reactive_fire_fascination_profile": self._clip01(
                (
                    symptoms["fire_fascination_preoccupation"]
                    + latents["amygdala_fire_salience_hyperreactivity"]
                    + regional_state["ventral_striatum_proxy"]
                    + inputs["fire_cue_exposure"]
                ) / 4.0
            ),
            "frontal_control_failure_profile": self._clip01(
                (
                    latents["prefrontal_inhibitory_failure"]
                    + latents["frontal_basal_ganglia_gating_dysfunction"]
                    + regional_state["vmpfc_proxy"]
                    + regional_state["dlpfc"]
                ) / 4.0
            ),
            "reinforced_tension_relief_cycle_profile": self._clip01(
                (
                    latents["tension_discharge_learning_loop"]
                    + latents["reward_relief_reinforcement_bias"]
                    + symptoms["fire_setting_urge"]
                    + symptoms["post_fire_relief_gratification"]
                ) / 4.0
            ),
        }

        return {
            "inputs": pd.Series(inputs, name="inputs"),
            "latents": pd.Series(latents, name="latents"),
            "regional_state": pd.Series(regional_state, name="regional_state"),
            "symptoms": pd.Series(symptoms, name="symptoms"),
            "phenotypes": pd.Series(phenotypes, name="phenotypes"),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Assign an MNI coordinate to Julich regions using a statistical map.
        """
        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception:
                with siibra.QUIET:
                    self._pmap = self.atlas.get_map(
                        parcellation=self.parcellation,
                        space=self.atlas.get_space(self.assignment_space),
                        maptype="statistical",
                    )

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break

        if "region" in assignments.columns:
            assignments = assignments.copy()
            assignments["region"] = assignments["region"].map(self._name_of)
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        """
        Return a regional mask object for a resolved region node.

        Call `.fetch()` on the returned object to obtain the underlying image.
        Returns None for unresolved nodes.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(space=self.space, maptype="labelled")
            except Exception:
                return None


if __name__ == "__main__":
    if siibra is None:
        print(
            "siibra is not installed in this environment. Install siibra, then rerun "
            "this script to build the atlas-grounded Pyromania scaffold."
        )
        raise SystemExit(0)

    model = PyromaniaModel()
    built = model.build(connectivity_rows=10)

    print("\n=== NODES ===")
    print(built["nodes"][["key", "node_type", "label", "atlas_region", "feature_summary"]].to_string(index=False))

    print("\n=== EDGES ===")
    print(built["edges"][["source", "target", "relation", "pyromania_change"]].to_string(index=False))

    print("\n=== REGION RESOLUTION ===")
    if model.region_objects:
        for key, region in model.region_objects.items():
            print(f"{key}: {region.name}")
    else:
        print("No regions resolved in this environment.")

    for node_key in ["amygdala", "vmpfc_proxy", "dlpfc", "ventral_striatum_proxy", "basal_ganglia_proxy"]:
        receptor_df = built["receptors"].get(node_key, pd.DataFrame())
        gene_df = built["genes"].get(node_key, pd.DataFrame())
        conn_df = built["connectivity_profiles"].get(node_key, pd.DataFrame())

        print(f"\n=== FEATURES: {node_key} ===")
        print("Receptors:")
        print(receptor_df.head().to_string(index=False) if not receptor_df.empty else "<none>")
        print("Genes:")
        print(gene_df.head().to_string(index=False) if not gene_df.empty else "<none>")
        print("Connectivity:")
        print(conn_df.head().to_string(index=False) if not conn_df.empty else "<none>")

    circuit_df = built["circuit_connectivity"]
    print("\n=== CIRCUIT CONNECTIVITY ===")
    print(circuit_df.head(12).to_string(index=False) if not circuit_df.empty else "<none>")

    sim = model.simulate(
        genetic_temperamental_liability=0.60,
        early_life_stress_trauma=0.45,
        chronic_stress_load=0.70,
        acute_tension_arousal=0.80,
        fire_cue_exposure=0.75,
        relief_reinforcement_history=0.55,
        inhibitory_support=0.25,
        stress_regulation_support=0.20,
    )

    print("\n=== SIMULATION: INPUTS ===")
    print(sim["inputs"].to_string())
    print("\n=== SIMULATION: LATENTS ===")
    print(sim["latents"].sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: REGIONAL STATE ===")
    print(sim["regional_state"].sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: SYMPTOMS ===")
    print(sim["symptoms"].sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: PHENOTYPES ===")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((0, 28, -8)).head())
