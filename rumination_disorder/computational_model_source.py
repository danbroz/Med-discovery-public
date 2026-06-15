from __future__ import annotations

"""
Rumination Disorder siibra scaffold.

This script turns a chapter-level biological summary of Rumination Disorder
(Rumination Syndrome) into a transparent, atlas-grounded mechanistic scaffold
using siibra. It is intended for research prototyping, feature exploration,
and educational modeling.

It is not a diagnostic, prognostic, or treatment tool.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore[assignment]
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_RUMINATION_GENE_PANEL = [
    # Dopaminergic motor / habit signaling
    "SLC6A3",
    "DRD1",
    "DRD2",
    "COMT",
    # GABAergic inhibitory control
    "GAD1",
    "GABRA1",
    "SLC6A1",
    # Noradrenergic arousal / stress response
    "SLC6A2",
    "ADRA2A",
    # Postprandial peptide and gut-motor sensitivity
    "CCK",
    "CCKAR",
    "CCKBR",
    # Immune, mast-cell, and mucosal barrier candidates
    "IL6",
    "TNF",
    "KIT",
    "TJP1",
    # Broad neuroplastic / neurodevelopmental support
    "BDNF",
]


class RuminationDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Rumination Disorder.

    Conceptual flow
    ---------------
    inputs -> latent biology -> regional state -> symptoms -> phenotype summaries

    Modeling choices
    ----------------
    - Treats the chapter as a gut-brain interaction disorder with strong
      interoceptive and affective components.
    - Keeps dopamine, noradrenergic arousal, GABAergic inhibition, postprandial
      peptide signaling, and immune/mucosal biology primarily as latent nodes.
    - Atlas-anchors only regions the chapter names or strongly implies:
      insula, ACC, amygdala, a conservative PFC-control proxy, and a basal
      ganglia proxy for involuntary motor-program release / habit maintenance.
    - Models repeated regurgitation as an interaction among visceral
      hypersensitivity, postprandial trigger sensitivity, deficient inhibitory
      control, and habit-like motor release.
    - Uses normalized 0..1 values for interpretability.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        assignment_space_spec: str = "mni152",
        connectivity_cohort: str = "HCP",
    ) -> None:
        if siibra is None:  # pragma: no cover - depends on runtime environment
            raise ImportError(
                "siibra is required to use RuminationDisorderModel. "
                "Install siibra in your environment before running this scaffold."
            ) from _SIIBRA_IMPORT_ERROR

        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space_spec = assignment_space_spec
        self.connectivity_cohort = connectivity_cohort

        # Compatibility-first initialization.
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
        try:
            self.assignment_space = (
                self.atlas.get_space(assignment_space_spec)
                if hasattr(self.atlas, "get_space")
                else self.atlas.spaces.get(assignment_space_spec)
            )
        except Exception:
            self.assignment_space = assignment_space_spec

        self.region_candidates: Dict[str, List[str]] = {
            "insula": [
                "Area Ig2 (Insula) left",
                "Area Ig1 (Insula) left",
                "Area Id7 (Insula) left",
                "insula left",
                "insula",
            ],
            "acc": [
                "Area p24pr left",
                "Area p24ab left",
                "Area a24pr left",
                "anterior cingulate cortex left",
                "anterior cingulate",
                "acc",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "pfc_control": [
                "Area Fp1 left",
                "Area Fp2 left",
                "Area 46 left",
                "Area 9 left",
                "prefrontal cortex left",
                "prefrontal cortex",
            ],
            "basal_ganglia_proxy": [
                "caudate nucleus left",
                "putamen left",
                "globus pallidus left",
                "basal ganglia left",
                "basal ganglia",
                "striatum left",
                "striatum",
            ],
        }

        self.region_node_notes: Dict[str, str] = {
            "insula": "Primary interoceptive cortex for visceral sensations such as gastric distension, nausea, and taste.",
            "acc": "Affective-visceral conflict and urge-monitoring node relevant to discomfort and premonitory sensation.",
            "amygdala": "Affective salience and stress-linked distress amplification node.",
            "pfc_control": "Conservative prefrontal inhibitory-control proxy for top-down suppression of prepotent motor responses.",
            "basal_ganglia_proxy": "Conservative basal-ganglia proxy for dopaminergic inhibitory-control failure and habit-like rumination motor release.",
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Broad inherited liability affecting transmitter function, immune regulation, excitability, and visceral sensitivity."
            ),
            "neurodevelopmental_vulnerability": (
                "Developmental liability reflecting the chapter's overlap with neurodevelopmental disorders and altered control systems."
            ),
            "psychosocial_stress": (
                "Stress and anxiety burden that can trigger or exacerbate rumination episodes."
            ),
            "upper_gi_dysfunction": (
                "Co-occurring upper gastrointestinal dysfunction such as GERD, dyspepsia, or delayed gastric emptying."
            ),
            "visceral_mechanosensory_load": (
                "Mechanical/interoceptive load from gastric distension, esophageal sensation, nausea, and related gut signals."
            ),
            "postprandial_trigger_load": (
                "Meal-related trigger burden, including satiety and gastric-motility signaling after eating."
            ),
            "immune_mucosal_dysregulation": (
                "Low-grade inflammation, mast-cell activation, gut-barrier dysfunction, and immune-triggered visceral sensitization."
            ),
            "learned_habit_strength": (
                "Established habit-like repetition of the rumination motor program once the behavior has become entrained."
            ),
            "behavioral_regulation_support": (
                "Protective behavioral structure and regulation support that can reduce arousal and interrupt the motor pattern."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "gut_brain_axis_dysregulation": (
                "Dysregulated bidirectional gut-brain signaling linking upper-GI dysfunction to central processing."
            ),
            "interoceptive_hypersensitivity": (
                "Heightened sensitivity to normal or mildly aversive internal gastrointestinal sensations."
            ),
            "dopamine_habit_signal_dysregulation": (
                "Dopaminergic disturbance affecting motor control, motivation, and compulsive habit maintenance."
            ),
            "noradrenergic_arousal": (
                "Stress-linked arousal bias influencing central vigilance and autonomic gut reactivity."
            ),
            "gabaergic_disinhibition": (
                "Insufficient inhibitory braking that weakens suppression of unwanted motor patterns."
            ),
            "cck_gut_motor_trigger_sensitivity": (
                "Postprandial peptide and gastric-motility sensitivity that can trigger the abnormal motor response."
            ),
            "immune_neuroinflammatory_signal": (
                "Immune/mucosal inflammatory burden altering enteric signaling and central discomfort processing."
            ),
            "visceral_affective_salience": (
                "Aversive tagging of gut sensations as unpleasant, urgent, or distressing."
            ),
            "acc_urge_monitoring_bias": (
                "Conflict-monitoring and urge-tracking bias for premonitory visceral sensations."
            ),
            "prefrontal_inhibitory_failure": (
                "Weak top-down suppression of prepotent rumination-related responses."
            ),
            "basal_ganglia_motor_release": (
                "Release of the involuntary / habit-like rumination motor program."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "premonitory_visceral_urge": (
                "Aversive internal sensation or urge that precedes regurgitation."
            ),
            "postprandial_regurgitation": (
                "Meal-linked regurgitation and rumination behavior."
            ),
            "visceral_distress_anxiety": (
                "Anxiety, discomfort, or distress coupled to gut sensations."
            ),
            "hypervigilance_to_gut_sensations": (
                "Excess attention to internal gastrointestinal sensations."
            ),
            "stress_triggered_exacerbation": (
                "Worsening of rumination episodes under stress or anxiety."
            ),
            "compulsive_habit_maintenance": (
                "Persistence of the behavior through reinforced habit-like repetition."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "dopamine_habit_signal_dysregulation",
                "relation": "loads liability in transmitter systems shaping motor control and habit maintenance",
                "rumination_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "gabaergic_disinhibition",
                "relation": "raises trait vulnerability to weak inhibitory control",
                "rumination_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "immune_neuroinflammatory_signal",
                "relation": "loads immune and mucosal vulnerability",
                "rumination_change": "increased",
            },
            {
                "source": "neurodevelopmental_vulnerability",
                "target": "interoceptive_hypersensitivity",
                "relation": "contributes to atypical sensory processing of bodily signals",
                "rumination_change": "increased",
            },
            {
                "source": "neurodevelopmental_vulnerability",
                "target": "prefrontal_inhibitory_failure",
                "relation": "can weaken developmental establishment of top-down control",
                "rumination_change": "increased",
            },
            {
                "source": "psychosocial_stress",
                "target": "noradrenergic_arousal",
                "relation": "amplifies stress-linked central arousal and autonomic reactivity",
                "rumination_change": "increased",
            },
            {
                "source": "psychosocial_stress",
                "target": "immune_neuroinflammatory_signal",
                "relation": "can disrupt the gut barrier and increase immune activation",
                "rumination_change": "increased",
            },
            {
                "source": "psychosocial_stress",
                "target": "prefrontal_inhibitory_failure",
                "relation": "weakens contextual regulation of prepotent responses",
                "rumination_change": "increased",
            },
            {
                "source": "upper_gi_dysfunction",
                "target": "gut_brain_axis_dysregulation",
                "relation": "provides the gastrointestinal dysfunction background shared with GERD and dyspepsia",
                "rumination_change": "increased",
            },
            {
                "source": "visceral_mechanosensory_load",
                "target": "interoceptive_hypersensitivity",
                "relation": "provides gastric and esophageal signals that can be perceived as aversive",
                "rumination_change": "increased",
            },
            {
                "source": "postprandial_trigger_load",
                "target": "cck_gut_motor_trigger_sensitivity",
                "relation": "loads postprandial peptide and gastric-motility triggering",
                "rumination_change": "increased",
            },
            {
                "source": "immune_mucosal_dysregulation",
                "target": "immune_neuroinflammatory_signal",
                "relation": "raises inflammatory and mast-cell mediated visceral sensitization",
                "rumination_change": "increased",
            },
            {
                "source": "immune_mucosal_dysregulation",
                "target": "gut_brain_axis_dysregulation",
                "relation": "perturbs enteric signaling and barrier integrity within the gut-brain axis",
                "rumination_change": "increased",
            },
            {
                "source": "learned_habit_strength",
                "target": "dopamine_habit_signal_dysregulation",
                "relation": "stabilizes the behavior as a compulsive or habit-driven action",
                "rumination_change": "increased",
            },
            {
                "source": "learned_habit_strength",
                "target": "basal_ganglia_motor_release",
                "relation": "facilitates re-expression of the rumination motor program",
                "rumination_change": "increased",
            },
            {
                "source": "behavioral_regulation_support",
                "target": "noradrenergic_arousal",
                "relation": "can reduce stress-linked hyperarousal",
                "rumination_change": "decreased",
            },
            {
                "source": "behavioral_regulation_support",
                "target": "prefrontal_inhibitory_failure",
                "relation": "can improve inhibitory regulation of maladaptive responses",
                "rumination_change": "decreased",
            },
            {
                "source": "behavioral_regulation_support",
                "target": "basal_ganglia_motor_release",
                "relation": "can interrupt expression of the maladaptive motor pattern",
                "rumination_change": "decreased",
            },
            {
                "source": "gut_brain_axis_dysregulation",
                "target": "interoceptive_hypersensitivity",
                "relation": "amplifies central responsiveness to GI signals",
                "rumination_change": "increased",
            },
            {
                "source": "interoceptive_hypersensitivity",
                "target": "insula",
                "relation": "loads the interoceptive cortex with amplified visceral signals",
                "rumination_change": "increased",
            },
            {
                "source": "immune_neuroinflammatory_signal",
                "target": "visceral_affective_salience",
                "relation": "sensitizes discomfort processing and affective unpleasantness",
                "rumination_change": "increased",
            },
            {
                "source": "noradrenergic_arousal",
                "target": "amygdala",
                "relation": "heightens threat-linked distress to visceral cues",
                "rumination_change": "increased",
            },
            {
                "source": "visceral_affective_salience",
                "target": "acc",
                "relation": "engages cingulate affective processing of discomfort and urge",
                "rumination_change": "increased",
            },
            {
                "source": "visceral_affective_salience",
                "target": "amygdala",
                "relation": "adds emotional salience and anxiety to gut sensations",
                "rumination_change": "increased",
            },
            {
                "source": "acc_urge_monitoring_bias",
                "target": "acc",
                "relation": "focuses monitoring on premonitory sensations and discomfort",
                "rumination_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_failure",
                "target": "pfc_control",
                "relation": "reduces top-down suppression of prepotent motor and emotional responses",
                "rumination_change": "decreased",
            },
            {
                "source": "dopamine_habit_signal_dysregulation",
                "target": "basal_ganglia_motor_release",
                "relation": "permits expression and maintenance of the habit-like motor program",
                "rumination_change": "increased",
            },
            {
                "source": "gabaergic_disinhibition",
                "target": "basal_ganglia_motor_release",
                "relation": "weakens suppression of unwanted motor patterns",
                "rumination_change": "increased",
            },
            {
                "source": "cck_gut_motor_trigger_sensitivity",
                "target": "premonitory_visceral_urge",
                "relation": "contributes meal-linked gut-motor triggering before regurgitation",
                "rumination_change": "increased",
            },
            {
                "source": "insula",
                "target": "premonitory_visceral_urge",
                "relation": "translates amplified gut signals into a conscious internal urge",
                "rumination_change": "increased",
            },
            {
                "source": "acc",
                "target": "premonitory_visceral_urge",
                "relation": "adds affective urgency and monitoring to the visceral sensation",
                "rumination_change": "increased",
            },
            {
                "source": "basal_ganglia_motor_release",
                "target": "basal_ganglia_proxy",
                "relation": "loads motor gating and habit circuitry",
                "rumination_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "postprandial_regurgitation",
                "relation": "permits execution of the rumination motor pattern",
                "rumination_change": "increased",
            },
            {
                "source": "premonitory_visceral_urge",
                "target": "postprandial_regurgitation",
                "relation": "acts as the immediate subjective precursor to the regurgitation episode",
                "rumination_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "visceral_distress_anxiety",
                "relation": "drives distress and anxiety around the gut sensation",
                "rumination_change": "increased",
            },
            {
                "source": "insula",
                "target": "hypervigilance_to_gut_sensations",
                "relation": "supports excessive interoceptive attention to gut sensations",
                "rumination_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "stress_triggered_exacerbation",
                "relation": "helps stress transform visceral cues into worsened episodes",
                "rumination_change": "increased",
            },
            {
                "source": "dopamine_habit_signal_dysregulation",
                "target": "compulsive_habit_maintenance",
                "relation": "helps maintain the repetitive behavior as a habit-driven act",
                "rumination_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "compulsive_habit_maintenance",
                "relation": "supports repetition of the motor routine once established",
                "rumination_change": "increased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        self._pmap: Optional[Any] = None
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

        out = []
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
            "insula",
            "amygdala",
            "anterior cingulate cortex",
            "prefrontal cortex",
            "basal ganglia",
            "striatum",
        } else 0
        proxy_penalty = 1 if any(token in name for token in ["cortex", "brain", "lobe"]) and "area" not in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
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
        rows: List[Dict[str, Any]] = []
        seen = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
            record = (
                self._name_of(region),
                getattr(region, "identifier", None),
                getattr(getattr(region, "parcellation", None), "name", ""),
            )
            if record in seen:
                continue
            seen.add(record)
            rows.append(
                {
                    "name": record[0],
                    "identifier": record[1],
                    "parcellation": record[2],
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
        centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        volume_mm3 = getattr(main, "volume", None)
        return centroid_xyz, (float(volume_mm3) if volume_mm3 is not None else None)

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
                out = (
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
                return out.reset_index(drop=True)
            return df.reset_index(drop=True)
        return pd.DataFrame()

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (
                f
                for f in feats
                if str(getattr(f, "cohort", "")).lower() == self.connectivity_cohort.lower()
            ),
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
            first = compound[0]
            first_data = getattr(first, "data", None)
            if isinstance(first_data, pd.DataFrame):
                self._connectivity_matrix = first_data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._name_of(region)
        region_id = getattr(region, "identifier", None)

        exact = [x for x in labels if self._name_of(x) == region_name]
        if exact:
            return exact[0]

        if region_id is not None:
            id_matches = [x for x in labels if getattr(x, "identifier", None) == region_id]
            if id_matches:
                return id_matches[0]

        rn = region_name.lower()
        fuzzy = [
            x
            for x in labels
            if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn
        ]
        if fuzzy:
            return fuzzy[0]

        return None

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
            df = pd.DataFrame(
                {
                    "connected_region": [self._name_of(idx) for idx in series.index],
                    "value": pd.to_numeric(series.values, errors="coerce"),
                }
            )
            df = df[df["connected_region"] != region.name]
            df = df.dropna(subset=["value"]).sort_values("value", ascending=False)
            return df.head(max_rows).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """Return a compact pairwise connectivity view for resolved circuit nodes."""
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        keys = list(self.region_objects.keys())
        for i, src_key in enumerate(keys):
            src = self.region_objects[src_key]
            src_label = self._match_region_label(list(matrix.index), src)
            if src_label is None:
                src_label = self._match_region_label(list(matrix.columns), src)
            if src_label is None:
                continue

            for dst_key in keys[i + 1 :]:
                dst = self.region_objects[dst_key]
                dst_label = self._match_region_label(list(matrix.columns), dst)
                if dst_label is None:
                    dst_label = self._match_region_label(list(matrix.index), dst)
                if dst_label is None:
                    continue

                value = None
                try:
                    value = matrix.loc[src_label, dst_label]
                except Exception:
                    try:
                        value = matrix.loc[dst_label, src_label]
                    except Exception:
                        value = None
                if value is None:
                    continue
                try:
                    value = float(value)
                except Exception:
                    continue

                rows.append(
                    {
                        "source_key": src_key,
                        "source_region": self._name_of(src),
                        "target_key": dst_key,
                        "target_region": self._name_of(dst),
                        "value": value,
                    }
                )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_RUMINATION_GENE_PANEL,
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
            note = self.region_node_notes.get(key, "Atlas-backed circuit node or conservative proxy.")
            region = self._resolve_region(candidates)
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
                        "description": f"{note} Unresolved in this runtime.",
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
                    "description": note,
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

    def region_mask(self, node_key: str) -> Any:
        """Return a regional mask if the node resolved to an atlas region."""
        region = self.region_objects.get(node_key)
        if region is None:
            return None

        space_candidates = [self.assignment_space, self.assignment_space_spec, self.space, self.space_spec]
        for space in space_candidates:
            try:
                return region.get_regional_mask(space)
            except Exception:
                continue
        return None

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Probabilistically assign an MNI coordinate to Julich regions.

        Uses a statistical/probabilistic parcellation map when available.
        """
        if len(xyz) != 3:
            raise ValueError("xyz must contain exactly three coordinates.")

        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space_spec,
                        maptype="statistical",
                    )
            except Exception:
                with siibra.QUIET:
                    self._pmap = self.atlas.get_map(
                        space=self.assignment_space,
                        parcellation=self.parcellation,
                        maptype="statistical",
                    )

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space_spec)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        if isinstance(assignments, pd.DataFrame):
            for candidate in ("map value", "correlation", "intersection over union"):
                if candidate in assignments.columns:
                    return assignments.sort_values(candidate, ascending=False).reset_index(drop=True)
            return assignments.reset_index(drop=True)

        try:
            df = pd.DataFrame(assignments)
            for candidate in ("map value", "correlation", "intersection over union"):
                if candidate in df.columns:
                    return df.sort_values(candidate, ascending=False).reset_index(drop=True)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def simulate(
        self,
        genetic_vulnerability: float = 0.40,
        neurodevelopmental_vulnerability: float = 0.25,
        psychosocial_stress: float = 0.55,
        upper_gi_dysfunction: float = 0.60,
        visceral_mechanosensory_load: float = 0.65,
        postprandial_trigger_load: float = 0.70,
        immune_mucosal_dysregulation: float = 0.35,
        learned_habit_strength: float = 0.50,
        behavioral_regulation_support: float = 0.20,
    ) -> Dict[str, pd.Series]:
        """
        Run a transparent normalized simulation of chapter-derived Rumination Disorder biology.

        Parameters are normalized to the 0..1 range. Larger values represent more of the
        named input except for behavioral_regulation_support, which is protective.
        """
        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "neurodevelopmental_vulnerability": self._clip01(neurodevelopmental_vulnerability),
                "psychosocial_stress": self._clip01(psychosocial_stress),
                "upper_gi_dysfunction": self._clip01(upper_gi_dysfunction),
                "visceral_mechanosensory_load": self._clip01(visceral_mechanosensory_load),
                "postprandial_trigger_load": self._clip01(postprandial_trigger_load),
                "immune_mucosal_dysregulation": self._clip01(immune_mucosal_dysregulation),
                "learned_habit_strength": self._clip01(learned_habit_strength),
                "behavioral_regulation_support": self._clip01(behavioral_regulation_support),
            },
            name="value",
        )

        gut_brain_axis_dysregulation = self._clip01(
            0.40 * inputs["upper_gi_dysfunction"]
            + 0.25 * inputs["visceral_mechanosensory_load"]
            + 0.20 * inputs["immune_mucosal_dysregulation"]
            + 0.15 * inputs["psychosocial_stress"]
            - 0.20 * inputs["behavioral_regulation_support"]
        )

        interoceptive_hypersensitivity = self._clip01(
            0.35 * gut_brain_axis_dysregulation
            + 0.25 * inputs["visceral_mechanosensory_load"]
            + 0.20 * inputs["postprandial_trigger_load"]
            + 0.10 * inputs["neurodevelopmental_vulnerability"]
            + 0.10 * inputs["immune_mucosal_dysregulation"]
        )

        dopamine_habit_signal_dysregulation = self._clip01(
            0.35 * inputs["learned_habit_strength"]
            + 0.25 * inputs["genetic_vulnerability"]
            + 0.20 * inputs["psychosocial_stress"]
            + 0.20 * inputs["postprandial_trigger_load"]
        )

        noradrenergic_arousal = self._clip01(
            0.50 * inputs["psychosocial_stress"]
            + 0.20 * inputs["visceral_mechanosensory_load"]
            + 0.15 * inputs["immune_mucosal_dysregulation"]
            + 0.15 * inputs["upper_gi_dysfunction"]
            - 0.25 * inputs["behavioral_regulation_support"]
        )

        gabaergic_disinhibition = self._clip01(
            0.35 * inputs["genetic_vulnerability"]
            + 0.25 * inputs["psychosocial_stress"]
            + 0.20 * inputs["neurodevelopmental_vulnerability"]
            + 0.20 * inputs["learned_habit_strength"]
            - 0.20 * inputs["behavioral_regulation_support"]
        )

        cck_gut_motor_trigger_sensitivity = self._clip01(
            0.55 * inputs["postprandial_trigger_load"]
            + 0.25 * inputs["upper_gi_dysfunction"]
            + 0.20 * gut_brain_axis_dysregulation
        )

        immune_neuroinflammatory_signal = self._clip01(
            0.40 * inputs["immune_mucosal_dysregulation"]
            + 0.25 * inputs["psychosocial_stress"]
            + 0.20 * inputs["upper_gi_dysfunction"]
            + 0.15 * inputs["genetic_vulnerability"]
        )

        visceral_affective_salience = self._clip01(
            0.35 * interoceptive_hypersensitivity
            + 0.25 * noradrenergic_arousal
            + 0.20 * immune_neuroinflammatory_signal
            + 0.20 * gabaergic_disinhibition
        )

        acc_urge_monitoring_bias = self._clip01(
            0.35 * visceral_affective_salience
            + 0.30 * interoceptive_hypersensitivity
            + 0.20 * noradrenergic_arousal
            + 0.15 * inputs["learned_habit_strength"]
        )

        prefrontal_inhibitory_failure = self._clip01(
            0.35 * inputs["psychosocial_stress"]
            + 0.20 * noradrenergic_arousal
            + 0.20 * gabaergic_disinhibition
            + 0.15 * visceral_affective_salience
            + 0.10 * inputs["neurodevelopmental_vulnerability"]
            - 0.30 * inputs["behavioral_regulation_support"]
        )

        basal_ganglia_motor_release = self._clip01(
            0.35 * dopamine_habit_signal_dysregulation
            + 0.25 * gabaergic_disinhibition
            + 0.20 * cck_gut_motor_trigger_sensitivity
            + 0.20 * inputs["learned_habit_strength"]
            - 0.20 * inputs["behavioral_regulation_support"]
        )

        latents = pd.Series(
            {
                "gut_brain_axis_dysregulation": gut_brain_axis_dysregulation,
                "interoceptive_hypersensitivity": interoceptive_hypersensitivity,
                "dopamine_habit_signal_dysregulation": dopamine_habit_signal_dysregulation,
                "noradrenergic_arousal": noradrenergic_arousal,
                "gabaergic_disinhibition": gabaergic_disinhibition,
                "cck_gut_motor_trigger_sensitivity": cck_gut_motor_trigger_sensitivity,
                "immune_neuroinflammatory_signal": immune_neuroinflammatory_signal,
                "visceral_affective_salience": visceral_affective_salience,
                "acc_urge_monitoring_bias": acc_urge_monitoring_bias,
                "prefrontal_inhibitory_failure": prefrontal_inhibitory_failure,
                "basal_ganglia_motor_release": basal_ganglia_motor_release,
            },
            name="value",
        )

        regional_state = pd.Series(
            {
                "insula": self._clip01(
                    0.45 * interoceptive_hypersensitivity
                    + 0.25 * gut_brain_axis_dysregulation
                    + 0.15 * cck_gut_motor_trigger_sensitivity
                    + 0.15 * immune_neuroinflammatory_signal
                ),
                "acc": self._clip01(
                    0.40 * acc_urge_monitoring_bias
                    + 0.30 * visceral_affective_salience
                    + 0.15 * noradrenergic_arousal
                    + 0.15 * interoceptive_hypersensitivity
                ),
                "amygdala": self._clip01(
                    0.45 * visceral_affective_salience
                    + 0.35 * noradrenergic_arousal
                    + 0.20 * immune_neuroinflammatory_signal
                ),
                "pfc_control": self._clip01(
                    0.75
                    - 0.55 * prefrontal_inhibitory_failure
                    - 0.15 * noradrenergic_arousal
                    + 0.25 * inputs["behavioral_regulation_support"]
                ),
                "basal_ganglia_proxy": self._clip01(
                    0.50 * basal_ganglia_motor_release
                    + 0.30 * dopamine_habit_signal_dysregulation
                    + 0.20 * gabaergic_disinhibition
                ),
            },
            name="value",
        )

        premonitory_visceral_urge = self._clip01(
            0.40 * regional_state["insula"]
            + 0.25 * regional_state["acc"]
            + 0.20 * cck_gut_motor_trigger_sensitivity
            + 0.15 * regional_state["basal_ganglia_proxy"]
        )

        postprandial_regurgitation = self._clip01(
            0.35 * premonitory_visceral_urge
            + 0.25 * regional_state["basal_ganglia_proxy"]
            + 0.20 * inputs["upper_gi_dysfunction"]
            + 0.10 * cck_gut_motor_trigger_sensitivity
            + 0.10 * (1.0 - regional_state["pfc_control"])
        )

        visceral_distress_anxiety = self._clip01(
            0.35 * regional_state["amygdala"]
            + 0.25 * regional_state["acc"]
            + 0.20 * regional_state["insula"]
            + 0.20 * (1.0 - regional_state["pfc_control"])
        )

        hypervigilance_to_gut_sensations = self._clip01(
            0.45 * regional_state["insula"]
            + 0.25 * interoceptive_hypersensitivity
            + 0.20 * noradrenergic_arousal
            + 0.10 * regional_state["acc"]
        )

        stress_triggered_exacerbation = self._clip01(
            0.35 * inputs["psychosocial_stress"]
            + 0.25 * regional_state["amygdala"]
            + 0.20 * noradrenergic_arousal
            + 0.20 * (1.0 - regional_state["pfc_control"])
        )

        compulsive_habit_maintenance = self._clip01(
            0.40 * regional_state["basal_ganglia_proxy"]
            + 0.30 * dopamine_habit_signal_dysregulation
            + 0.20 * inputs["learned_habit_strength"]
            + 0.10 * regional_state["acc"]
        )

        symptoms = pd.Series(
            {
                "premonitory_visceral_urge": premonitory_visceral_urge,
                "postprandial_regurgitation": postprandial_regurgitation,
                "visceral_distress_anxiety": visceral_distress_anxiety,
                "hypervigilance_to_gut_sensations": hypervigilance_to_gut_sensations,
                "stress_triggered_exacerbation": stress_triggered_exacerbation,
                "compulsive_habit_maintenance": compulsive_habit_maintenance,
            },
            name="value",
        )

        phenotypes = pd.Series(
            {
                "interoceptive_rumination_profile": self._clip01(
                    (
                        premonitory_visceral_urge
                        + postprandial_regurgitation
                        + hypervigilance_to_gut_sensations
                    )
                    / 3.0
                ),
                "stress_reactive_rumination_profile": self._clip01(
                    (
                        stress_triggered_exacerbation
                        + visceral_distress_anxiety
                        + regional_state["amygdala"]
                    )
                    / 3.0
                ),
                "habit_maintained_profile": self._clip01(
                    (
                        compulsive_habit_maintenance
                        + postprandial_regurgitation
                        + regional_state["basal_ganglia_proxy"]
                    )
                    / 3.0
                ),
                "immune_visceral_sensitivity_profile": self._clip01(
                    (
                        immune_neuroinflammatory_signal
                        + gut_brain_axis_dysregulation
                        + regional_state["insula"]
                        + visceral_distress_anxiety
                    )
                    / 4.0
                ),
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


if __name__ == "__main__":  # pragma: no cover - example usage
    try:
        model = RuminationDisorderModel()
    except ImportError as exc:
        print(exc)
        raise SystemExit(1)

    print("Building Rumination Disorder scaffold...\n")
    bundle = model.build(connectivity_rows=10)

    print("NODES")
    print(bundle["nodes"][["key", "node_type", "atlas_region", "feature_summary"]].to_string(index=False))

    print("\nEDGES")
    print(bundle["edges"][["source", "target", "relation", "rumination_change"]].to_string(index=False))

    if "insula" in bundle["receptors"] and not bundle["receptors"]["insula"].empty:
        print("\nINSULA RECEPTOR FINGERPRINT (head)")
        print(bundle["receptors"]["insula"].head().to_string(index=False))

    if "acc" in bundle["genes"] and not bundle["genes"]["acc"].empty:
        print("\nACC GENE SUMMARY (head)")
        print(bundle["genes"]["acc"].head().to_string(index=False))

    if not bundle["circuit_connectivity"].empty:
        print("\nCIRCUIT CONNECTIVITY")
        print(bundle["circuit_connectivity"].head(20).to_string(index=False))

    print("\nSIMULATION EXAMPLE")
    sim = model.simulate(
        genetic_vulnerability=0.45,
        neurodevelopmental_vulnerability=0.30,
        psychosocial_stress=0.75,
        upper_gi_dysfunction=0.80,
        visceral_mechanosensory_load=0.70,
        postprandial_trigger_load=0.85,
        immune_mucosal_dysregulation=0.45,
        learned_habit_strength=0.65,
        behavioral_regulation_support=0.25,
    )
    for name, series in sim.items():
        print(f"\n{name.upper()}")
        print(series.sort_values(ascending=False).to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((-32, 18, 6)).head())
