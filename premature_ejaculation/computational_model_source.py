from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Premature (Early) Ejaculation (PE).

This script turns a chapter-level biological narrative into a transparent,
reusable mechanistic model. It is intended for research scaffolding and model
inspection, not diagnosis or treatment.

Key modeling choices for this chapter:
- Keep serotonergic inhibitory tone, sensory amplification, stress arousal, and
  ejaculatory reflex timing as latent biology rather than forcing uncertain
  chemistry into a single parcel.
- Anchor only anatomy that the chapter clearly names or strongly implies:
  amygdala, hippocampus, cingulate gyrus, temporal-limbic cortex, and frontal
  control systems.
- Represent pelvic floor, glans hypersensitivity, accelerated conduction, and
  cortical amplification as mechanistic inputs because they are chapter-level
  biological theories rather than established atlas parcels.
- Tolerate partial atlas / feature availability and siibra API variation.
"""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


PREMATURE_EJACULATION_GENE_PANEL = [
    "SLC6A4",
    "HTR1A",
    "HTR1B",
    "HTR2C",
    "TPH2",
    "MAOA",
    "COMT",
    "BDNF",
]


class PrematureEjaculationModel:
    """
    Mechanistic siibra scaffold for Premature (Early) Ejaculation.

    Conceptual flow:
        inputs -> latent biology -> regional burden/state -> symptoms -> phenotypes

    This is a conservative interpretation of the supplied chapter. The chapter
    emphasizes a biopsychosocial framework with especially strong biological
    weight on reduced serotonergic inhibition, genital/somatic trigger
    hypersensitivity, accelerated neural conduction, cortical amplification of
    genital stimuli, and limbic-frontal dysregulation interacting with anxiety
    and stress.

    Important limitation:
    The chapter does not provide PE-specific functional imaging with precise
    cytoarchitectonic localization, so this scaffold uses conservative atlas
    anchors plus clearly labeled proxies where needed.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
        aggregate_connectivity_subjects: int = 8,
    ) -> None:
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort
        self.aggregate_connectivity_subjects = max(1, int(aggregate_connectivity_subjects))

        # Compatibility-first atlas/parcellation/space lookup.
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

        self.disorder_name = "Premature (Early) Ejaculation"
        self.disorder_key = "premature_ejaculation"

        # Conservative region resolution: the chapter names limbic structures,
        # cingulate gyrus, temporal lobes, and frontal lobes, but does not justify
        # more aggressive parcel-level localization.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "SF (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "CA2 (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "Subiculum left",
                "hippocampus",
            ],
            "cingulate_gyrus": [
                "Area p24pr left",
                "Area p24ab left",
                "anterior cingulate cortex",
                "cingulate gyrus",
                "cingulate cortex",
            ],
            "temporal_limbic_cortex_proxy": [
                "Area TG (Temporal Pole) left",
                "Temporal pole left",
                "Area TE 1.0 (TE) left",
                "Area TE 1.2 (TE) left",
                "temporal lobe",
                "temporal cortex",
            ],
            "frontal_control_proxy": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 9 left",
                "middle frontal gyrus",
                "frontal lobe",
                "prefrontal cortex",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "polygenic_vulnerability": (
                "Constitutional liability affecting serotonergic tone, receptor balance, "
                "stress reactivity, and ejaculatory threshold"
            ),
            "stress_anxiety_load": (
                "Acute or chronic anxiety/stress burden that can amplify autonomic "
                "arousal and weaken control over the ejaculatory response"
            ),
            "glans_hypersensitivity": (
                "Peripheral genital hypersensitivity that raises trigger salience and "
                "speeds reflex engagement"
            ),
            "pelvic_floor_dysfunction": (
                "Pelvic floor alteration that may destabilize timing and coordination of "
                "the ejaculatory response"
            ),
            "accelerated_neural_conduction": (
                "Faster sensory-reflex conduction bias that can shorten latency to "
                "ejaculatory completion"
            ),
            "cortical_stimulus_amplification": (
                "Amplified cortical processing of genital stimuli, increasing the impact "
                "of sensory triggers"
            ),
            "ssri_treatment_support": (
                "Protective serotonergic treatment support modeled on the chapter's "
                "observation that SSRIs delay ejaculation"
            ),
            "stress_regulation_support": (
                "Protective support that lowers stress-driven arousal amplification"
            ),
            "behavioral_control_support": (
                "Protective support for inhibitory control and emotional regulation"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "reduced_serotonergic_inhibition": (
                "Reduced inhibitory 5-HT modulation of ejaculation, lowering reflex restraint"
            ),
            "somatosensory_trigger_gain": (
                "Heightened sensory gain from genital/peripheral and cortical stimulus channels"
            ),
            "stress_arousal_amplification": (
                "Stress-coupled autonomic and emotional arousal that accelerates ejaculation"
            ),
            "limbic_temporal_dysregulation": (
                "Disrupted limbic-temporal regulation of sexual behaviour, emotion, and autonomic patterning"
            ),
            "frontal_inhibitory_control_failure": (
                "Weakened frontal and cingulate inhibitory control over urges and response timing"
            ),
            "rapid_reflex_propagation": (
                "Fast progression from sensory trigger to ejaculatory motor/autonomic completion"
            ),
            "low_ejaculatory_threshold": (
                "Constitutionally or acquired low threshold for ejaculatory completion"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "rapid_ejaculation": "Markedly shortened ejaculatory latency",
            "loss_of_control": "Subjective inability to delay or regulate ejaculation",
            "anticipatory_anxiety": "Pre-sexual or in-situation anxiety about rapid ejaculation",
            "emotional_distress": "Distress linked to repeated rapid ejaculation and poor control",
            "relationship_burden": "Interpersonal and sexual burden secondary to the symptom pattern",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "polygenic_vulnerability",
                "target": "reduced_serotonergic_inhibition",
                "relation": "can bias baseline inhibitory serotonergic control downward",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "polygenic_vulnerability",
                "target": "low_ejaculatory_threshold",
                "relation": "can establish a lifelong low threshold for rapid completion",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "glans_hypersensitivity",
                "target": "somatosensory_trigger_gain",
                "relation": "raises salience of genital triggers",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "cortical_stimulus_amplification",
                "target": "somatosensory_trigger_gain",
                "relation": "amplifies cortical impact of genital stimuli",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "accelerated_neural_conduction",
                "target": "rapid_reflex_propagation",
                "relation": "speeds sensory-reflex transmission",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "pelvic_floor_dysfunction",
                "target": "rapid_reflex_propagation",
                "relation": "destabilizes ejaculatory timing and coordination",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "stress_anxiety_load",
                "target": "stress_arousal_amplification",
                "relation": "raises autonomic and emotional arousal around sexual performance",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "stress_arousal_amplification",
                "target": "amygdala",
                "relation": "heightens limbic threat/salience reactivity",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "stress_arousal_amplification",
                "target": "limbic_temporal_dysregulation",
                "relation": "destabilizes emotional-autonomic coordination in temporal-limbic systems",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "limbic_temporal_dysregulation",
                "target": "temporal_limbic_cortex_proxy",
                "relation": "loads the temporal-limbic response network",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "limbic_temporal_dysregulation",
                "target": "hippocampus",
                "relation": "alters contextual-emotional regulation within the limbic system",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "limbic_temporal_dysregulation",
                "target": "frontal_inhibitory_control_failure",
                "relation": "reduces effective top-down restraint from frontal systems",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "frontal_inhibitory_control_failure",
                "target": "frontal_control_proxy",
                "relation": "expresses as weakened frontal inhibitory state",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "frontal_inhibitory_control_failure",
                "target": "cingulate_gyrus",
                "relation": "loads cingulate control and conflict-monitoring systems",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "somatosensory_trigger_gain",
                "target": "rapid_reflex_propagation",
                "relation": "converts amplified sensory input into faster reflex progression",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "reduced_serotonergic_inhibition",
                "target": "low_ejaculatory_threshold",
                "relation": "removes inhibitory delay from the ejaculatory process",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "rapid_reflex_propagation",
                "target": "low_ejaculatory_threshold",
                "relation": "further shortens the path to completion",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "low_ejaculatory_threshold",
                "target": "rapid_ejaculation",
                "relation": "directly shortens latency",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "rapid_reflex_propagation",
                "target": "rapid_ejaculation",
                "relation": "speeds the sensory-autonomic transition to ejaculation",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "frontal_inhibitory_control_failure",
                "target": "loss_of_control",
                "relation": "weakens voluntary delay and inhibitory regulation",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "stress_arousal_amplification",
                "target": "anticipatory_anxiety",
                "relation": "promotes performance-linked anxious expectation",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "rapid_ejaculation",
                "target": "emotional_distress",
                "relation": "persistent rapid latency increases distress burden",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "loss_of_control",
                "target": "emotional_distress",
                "relation": "subjective lack of control increases burden",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "rapid_ejaculation",
                "target": "relationship_burden",
                "relation": "rapid ejaculation can impair sexual and relational functioning",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "emotional_distress",
                "target": "relationship_burden",
                "relation": "distress amplifies interpersonal strain",
                "premature_ejaculation_change": "increased",
            },
            {
                "source": "ssri_treatment_support",
                "target": "reduced_serotonergic_inhibition",
                "relation": "restores inhibitory serotonergic restraint",
                "premature_ejaculation_change": "decreased",
            },
            {
                "source": "stress_regulation_support",
                "target": "stress_arousal_amplification",
                "relation": "reduces stress-driven autonomic escalation",
                "premature_ejaculation_change": "decreased",
            },
            {
                "source": "behavioral_control_support",
                "target": "frontal_inhibitory_control_failure",
                "relation": "supports top-down control and emotional regulation",
                "premature_ejaculation_change": "decreased",
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
        self._build_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _mean_clip(values: Iterable[float]) -> float:
        vals = [float(v) for v in values]
        if not vals:
            return 0.0
        return max(0.0, min(1.0, round(sum(vals) / len(vals), 4)))

    @staticmethod
    def _series_from(mapping: Dict[str, float], name: str) -> pd.Series:
        return pd.Series({k: round(float(v), 4) for k, v in mapping.items()}, name=name)

    def _modality_candidates(self, kind: str) -> List[Any]:
        candidates: List[Any] = []
        try:
            if kind == "receptor":
                candidates.append(siibra.features.molecular.ReceptorDensityFingerprint)
            elif kind == "gene":
                candidates.append(siibra.features.molecular.GeneExpressions)
            elif kind == "connectivity":
                candidates.append(siibra.features.connectivity.StreamlineCounts)
        except Exception:
            pass

        if kind == "receptor":
            candidates.append("receptor density fingerprint")
        elif kind == "gene":
            candidates.append("gene expressions")
        elif kind == "connectivity":
            candidates.append("StreamlineCounts")
        return candidates

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
            "hippocampus",
            "cingulate gyrus",
            "cingulate cortex",
            "temporal lobe",
            "temporal cortex",
            "frontal lobe",
            "prefrontal cortex",
        } else 0
        subregion_bonus = 0 if any(
            token in name for token in ("area ", "ca1", "ca2", "ca3", "subiculum", "lb", "cm", "sf", "p24")
        ) else 1
        return (left_bonus, right_penalty, generic_penalty, subregion_bonus)

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
                return sorted(matches, key=self._region_rank)[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        seen = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
            item = (
                self._name_of(region),
                getattr(region, "identifier", None),
                getattr(getattr(region, "parcellation", None), "name", ""),
            )
            if item in seen:
                continue
            seen.add(item)
            rows.append(
                {
                    "name": item[0],
                    "identifier": item[1],
                    "parcellation": item[2],
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
        volume_mm3 = getattr(main, "volume", None)
        try:
            volume_mm3 = float(volume_mm3) if volume_mm3 is not None else None
        except Exception:
            volume_mm3 = None
        return centroid_xyz, volume_mm3

    def _extract_tabular_data(self, feature: Any) -> pd.DataFrame:
        try:
            data = getattr(feature, "data", None)
            if isinstance(data, pd.DataFrame):
                return data.copy()
            if isinstance(data, pd.Series):
                return data.to_frame().reset_index(drop=False)
            if isinstance(data, dict):
                return pd.DataFrame(data)
        except Exception:
            pass
        return pd.DataFrame()

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        try:
            df = self._extract_tabular_data(feats[0]).reset_index(drop=False)
            if "index" in df.columns and "receptor" not in df.columns:
                df = df.rename(columns={"index": "receptor"})
            return df
        except Exception:
            return pd.DataFrame()

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()
        try:
            df = self._extract_tabular_data(feats[0])
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        lower_cols = {str(c).lower(): c for c in df.columns}
        required = {"gene", "level", "zscore"}
        if required.issubset(lower_cols):
            gene_col = lower_cols["gene"]
            level_col = lower_cols["level"]
            zscore_col = lower_cols["zscore"]
            try:
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
            except Exception:
                return df.reset_index(drop=True)
        return df.reset_index(drop=True)

    def _candidate_connectivity_feature(self) -> Any:
        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            return None
        cohort_match = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), None)
        return cohort_match if cohort_match is not None else feats[0]

    def _matrix_like(self, obj: Any) -> pd.DataFrame:
        try:
            data = getattr(obj, "data", None)
            if isinstance(data, pd.DataFrame):
                return data.copy()
        except Exception:
            pass
        return pd.DataFrame()

    def _average_matrices(self, matrices: Sequence[pd.DataFrame]) -> pd.DataFrame:
        usable = [m for m in matrices if isinstance(m, pd.DataFrame) and not m.empty]
        if not usable:
            return pd.DataFrame()
        if len(usable) == 1:
            return usable[0].copy()
        try:
            running = usable[0].copy()
            for m in usable[1:]:
                running = running.add(m, fill_value=0.0)
            return running / float(len(usable))
        except Exception:
            return usable[0].copy()

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feat = self._candidate_connectivity_feature()
        if feat is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        matrices: List[pd.DataFrame] = []
        try:
            for idx, element in enumerate(feat):
                matrices.append(self._matrix_like(element))
                if idx + 1 >= self.aggregate_connectivity_subjects:
                    break
        except Exception:
            pass

        if not matrices:
            direct = self._matrix_like(feat)
            if not direct.empty:
                matrices = [direct]

        self._connectivity_matrix = self._average_matrices(matrices)
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        exact = [x for x in labels if self._name_of(x) == getattr(region, "name", self._name_of(region))]
        if exact:
            return exact[0]

        rn = getattr(region, "name", self._name_of(region)).lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        if fuzzy:
            return fuzzy[0]

        region_tokens = {t for t in rn.replace("(", " ").replace(")", " ").replace("-", " ").split() if len(t) > 2}
        best = None
        best_score = 0
        for label in labels:
            ln = self._name_of(label).lower()
            label_tokens = {t for t in ln.replace("(", " ").replace(")", " ").replace("-", " ").split() if len(t) > 2}
            score = len(region_tokens & label_tokens)
            if score > best_score:
                best = label
                best_score = score
        return best if best_score > 0 else None

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
            df = df[df["connected_region"] != getattr(region, "name", self._name_of(region))]
            return df.head(max_rows).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Public atlas helpers
    # ------------------------------------------------------------------
    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if self._pmap is None:
            with siibra.QUIET:
                self._pmap = siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=self.assignment_space,
                    maptype="statistical",
                )
        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        if isinstance(assignments, pd.DataFrame):
            for candidate in ("map value", "correlation", "intersection over union"):
                if candidate in assignments.columns:
                    return assignments.sort_values(candidate, ascending=False).reset_index(drop=True)
        return assignments

    def region_mask(self, node_key: str) -> Any:
        region = self.region_objects.get(node_key)
        if region is None:
            raise KeyError(f"No resolved region object for node '{node_key}'")

        for method_name in ("get_regional_mask", "fetch_regional_map"):
            method = getattr(region, method_name, None)
            if method is None:
                continue
            try:
                if method_name == "get_regional_mask":
                    return method(space=self.assignment_space, maptype="labelled")
                return method(space=self.assignment_space, maptype="labelled")
            except TypeError:
                try:
                    return method(self.assignment_space)
                except Exception:
                    continue
            except Exception:
                continue
        raise RuntimeError(f"Could not fetch a region mask for node '{node_key}'")

    # ------------------------------------------------------------------
    # Build and connectivity summaries
    # ------------------------------------------------------------------
    def circuit_connectivity(self, max_rows_per_seed: int = 10) -> pd.DataFrame:
        if not self.region_objects:
            if self.nodes_df.empty:
                self.build(connectivity_rows=max_rows_per_seed)
            else:
                return pd.DataFrame()

        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        labels_index = list(matrix.index)
        labels_columns = list(matrix.columns)

        for seed_key, seed_region in self.region_objects.items():
            seed_label = self._match_region_label(labels_index, seed_region)
            axis = "index"
            if seed_label is None:
                seed_label = self._match_region_label(labels_columns, seed_region)
                axis = "columns"
            if seed_label is None:
                continue

            try:
                series = matrix.loc[seed_label] if axis == "index" else matrix[seed_label]
                series = series.sort_values(ascending=False)
            except Exception:
                continue

            taken = 0
            for target_label, value in series.items():
                target_name = self._name_of(target_label)
                if target_name == getattr(seed_region, "name", self._name_of(seed_region)):
                    continue
                target_key = None
                for k, region in self.region_objects.items():
                    if self._name_of(region) == target_name:
                        target_key = k
                        break
                rows.append(
                    {
                        "seed_key": seed_key,
                        "seed_region": getattr(seed_region, "name", self._name_of(seed_region)),
                        "target_key": target_key,
                        "target_region": target_name,
                        "value": float(value),
                    }
                )
                taken += 1
                if taken >= max_rows_per_seed:
                    break

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["seed_key", "value"], ascending=[True, False]).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = PREMATURE_EJACULATION_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
        # Clear prior resolved objects so repeated builds stay deterministic.
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
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
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region_proxy" if key.endswith("_proxy") else "region",
                        "description": "Conservative atlas-backed proxy node unresolved in this environment",
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
                    "label": getattr(region, "name", self._name_of(region)),
                    "node_type": "region_proxy" if key.endswith("_proxy") else "region",
                    "description": (
                        "Conservative atlas-backed proxy node"
                        if key.endswith("_proxy")
                        else "Atlas-backed circuit node"
                    ),
                    "atlas_region": getattr(region, "name", self._name_of(region)),
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
        out = {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": self.circuit_connectivity(max_rows_per_seed=max(5, min(12, connectivity_rows))),
        }
        self._build_cache = out
        return out

    # ------------------------------------------------------------------
    # Transparent one-pass simulator
    # ------------------------------------------------------------------
    def simulate(
        self,
        polygenic_vulnerability: float = 0.5,
        stress_anxiety_load: float = 0.5,
        glans_hypersensitivity: float = 0.5,
        pelvic_floor_dysfunction: float = 0.4,
        accelerated_neural_conduction: float = 0.5,
        cortical_stimulus_amplification: float = 0.5,
        ssri_treatment_support: float = 0.0,
        stress_regulation_support: float = 0.0,
        behavioral_control_support: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized simulation.

        Inputs are clipped to [0, 1].
        Higher symptom values indicate stronger expression of the modeled
        pathway, not clinical certainty.
        """

        inputs = {
            "polygenic_vulnerability": self._clip01(polygenic_vulnerability),
            "stress_anxiety_load": self._clip01(stress_anxiety_load),
            "glans_hypersensitivity": self._clip01(glans_hypersensitivity),
            "pelvic_floor_dysfunction": self._clip01(pelvic_floor_dysfunction),
            "accelerated_neural_conduction": self._clip01(accelerated_neural_conduction),
            "cortical_stimulus_amplification": self._clip01(cortical_stimulus_amplification),
            "ssri_treatment_support": self._clip01(ssri_treatment_support),
            "stress_regulation_support": self._clip01(stress_regulation_support),
            "behavioral_control_support": self._clip01(behavioral_control_support),
        }

        latents = {
            "reduced_serotonergic_inhibition": self._clip01(
                0.42 * inputs["polygenic_vulnerability"]
                + 0.20 * inputs["stress_anxiety_load"]
                + 0.12 * inputs["accelerated_neural_conduction"]
                + 0.10 * inputs["glans_hypersensitivity"]
                + 0.06 * inputs["cortical_stimulus_amplification"]
                + 0.10 * inputs["pelvic_floor_dysfunction"]
                - 0.40 * inputs["ssri_treatment_support"]
            ),
            "somatosensory_trigger_gain": self._clip01(
                0.40 * inputs["glans_hypersensitivity"]
                + 0.20 * inputs["accelerated_neural_conduction"]
                + 0.25 * inputs["cortical_stimulus_amplification"]
                + 0.10 * inputs["stress_anxiety_load"]
                + 0.05 * inputs["pelvic_floor_dysfunction"]
            ),
            "stress_arousal_amplification": self._clip01(
                0.55 * inputs["stress_anxiety_load"]
                + 0.20 * inputs["polygenic_vulnerability"]
                + 0.10 * inputs["cortical_stimulus_amplification"]
                + 0.05 * inputs["glans_hypersensitivity"]
                - 0.30 * inputs["stress_regulation_support"]
            ),
        }

        latents["limbic_temporal_dysregulation"] = self._clip01(
            0.35 * latents["stress_arousal_amplification"]
            + 0.25 * latents["reduced_serotonergic_inhibition"]
            + 0.20 * latents["somatosensory_trigger_gain"]
            + 0.20 * inputs["polygenic_vulnerability"]
        )
        latents["frontal_inhibitory_control_failure"] = self._clip01(
            0.34 * latents["stress_arousal_amplification"]
            + 0.28 * latents["limbic_temporal_dysregulation"]
            + 0.20 * latents["reduced_serotonergic_inhibition"]
            + 0.10 * latents["somatosensory_trigger_gain"]
            + 0.08 * inputs["polygenic_vulnerability"]
            - 0.30 * inputs["behavioral_control_support"]
        )
        latents["rapid_reflex_propagation"] = self._clip01(
            0.35 * inputs["accelerated_neural_conduction"]
            + 0.25 * latents["somatosensory_trigger_gain"]
            + 0.20 * inputs["pelvic_floor_dysfunction"]
            + 0.10 * latents["reduced_serotonergic_inhibition"]
            + 0.10 * latents["stress_arousal_amplification"]
        )
        latents["low_ejaculatory_threshold"] = self._clip01(
            0.35 * latents["reduced_serotonergic_inhibition"]
            + 0.25 * latents["rapid_reflex_propagation"]
            + 0.20 * latents["somatosensory_trigger_gain"]
            + 0.20 * latents["frontal_inhibitory_control_failure"]
        )

        regional_state = {
            "amygdala": self._clip01(
                0.45 * latents["stress_arousal_amplification"]
                + 0.30 * latents["limbic_temporal_dysregulation"]
                + 0.15 * latents["somatosensory_trigger_gain"]
                + 0.10 * inputs["polygenic_vulnerability"]
            ),
            "hippocampus": self._clip01(
                0.35 * latents["limbic_temporal_dysregulation"]
                + 0.25 * latents["stress_arousal_amplification"]
                + 0.20 * inputs["polygenic_vulnerability"]
                + 0.20 * latents["somatosensory_trigger_gain"]
            ),
            "cingulate_gyrus": self._clip01(
                0.40 * latents["frontal_inhibitory_control_failure"]
                + 0.30 * latents["stress_arousal_amplification"]
                + 0.15 * latents["limbic_temporal_dysregulation"]
                + 0.15 * latents["low_ejaculatory_threshold"]
            ),
            "temporal_limbic_cortex_proxy": self._clip01(
                0.40 * latents["limbic_temporal_dysregulation"]
                + 0.25 * latents["somatosensory_trigger_gain"]
                + 0.20 * latents["stress_arousal_amplification"]
                + 0.15 * latents["rapid_reflex_propagation"]
            ),
            "frontal_control_proxy": self._clip01(
                0.45 * latents["frontal_inhibitory_control_failure"]
                + 0.20 * latents["stress_arousal_amplification"]
                + 0.20 * latents["low_ejaculatory_threshold"]
                + 0.15 * latents["limbic_temporal_dysregulation"]
            ),
        }

        symptoms = {
            "rapid_ejaculation": self._clip01(
                0.45 * latents["low_ejaculatory_threshold"]
                + 0.25 * latents["rapid_reflex_propagation"]
                + 0.20 * latents["somatosensory_trigger_gain"]
                + 0.10 * latents["stress_arousal_amplification"]
            ),
            "loss_of_control": self._clip01(
                0.42 * latents["frontal_inhibitory_control_failure"]
                + 0.28 * latents["low_ejaculatory_threshold"]
                + 0.20 * latents["stress_arousal_amplification"]
                + 0.10 * latents["limbic_temporal_dysregulation"]
            ),
        }
        symptoms["anticipatory_anxiety"] = self._clip01(
            0.48 * latents["stress_arousal_amplification"]
            + 0.20 * symptoms["rapid_ejaculation"]
            + 0.17 * symptoms["loss_of_control"]
            + 0.15 * regional_state["amygdala"]
        )
        symptoms["emotional_distress"] = self._clip01(
            0.32 * symptoms["rapid_ejaculation"]
            + 0.22 * symptoms["loss_of_control"]
            + 0.26 * symptoms["anticipatory_anxiety"]
            + 0.20 * regional_state["cingulate_gyrus"]
        )
        symptoms["relationship_burden"] = self._clip01(
            0.42 * symptoms["rapid_ejaculation"]
            + 0.18 * symptoms["loss_of_control"]
            + 0.20 * symptoms["emotional_distress"]
            + 0.20 * symptoms["anticipatory_anxiety"]
        )

        phenotypes = {
            "constitutional_rapid_reflex_profile": self._mean_clip(
                [
                    latents["reduced_serotonergic_inhibition"],
                    latents["rapid_reflex_propagation"],
                    latents["low_ejaculatory_threshold"],
                ]
            ),
            "stress_reactive_pe_profile": self._mean_clip(
                [
                    latents["stress_arousal_amplification"],
                    regional_state["amygdala"],
                    symptoms["anticipatory_anxiety"],
                    symptoms["rapid_ejaculation"],
                ]
            ),
            "sensory_hyperreactive_profile": self._mean_clip(
                [
                    latents["somatosensory_trigger_gain"],
                    latents["rapid_reflex_propagation"],
                    regional_state["temporal_limbic_cortex_proxy"],
                    symptoms["rapid_ejaculation"],
                ]
            ),
            "control_breakdown_profile": self._mean_clip(
                [
                    latents["frontal_inhibitory_control_failure"],
                    regional_state["frontal_control_proxy"],
                    symptoms["loss_of_control"],
                    symptoms["emotional_distress"],
                ]
            ),
        }

        return {
            "inputs": self._series_from(inputs, name="inputs"),
            "latents": self._series_from(latents, name="latents"),
            "regional_state": self._series_from(regional_state, name="regional_state"),
            "symptoms": self._series_from(symptoms, name="symptoms"),
            "phenotypes": self._series_from(phenotypes, name="phenotypes"),
        }


if __name__ == "__main__":
    model = PrematureEjaculationModel()
    built = model.build(connectivity_rows=8)

    print("\n=== Nodes ===")
    print(built["nodes"][["key", "node_type", "label", "atlas_region", "feature_summary"]].to_string(index=False))

    print("\n=== Edges ===")
    print(built["edges"][["source", "target", "relation", "premature_ejaculation_change"]].to_string(index=False))

    print("\n=== Example receptor profile: amygdala ===")
    print(model.receptors.get("amygdala", pd.DataFrame()).head(10).to_string(index=False))

    print("\n=== Example gene profile: frontal_control_proxy ===")
    print(model.genes.get("frontal_control_proxy", pd.DataFrame()).head(10).to_string(index=False))

    print("\n=== Example connectivity profile: cingulate_gyrus ===")
    print(model.connectivity_profiles.get("cingulate_gyrus", pd.DataFrame()).head(10).to_string(index=False))

    print("\n=== Circuit connectivity summary ===")
    print(built["circuit_connectivity"].head(20).to_string(index=False))

    sim = model.simulate(
        polygenic_vulnerability=0.65,
        stress_anxiety_load=0.75,
        glans_hypersensitivity=0.80,
        pelvic_floor_dysfunction=0.55,
        accelerated_neural_conduction=0.75,
        cortical_stimulus_amplification=0.70,
        ssri_treatment_support=0.35,
        stress_regulation_support=0.20,
        behavioral_control_support=0.20,
    )

    print("\n=== Simulation: inputs ===")
    print(sim["inputs"].to_string())
    print("\n=== Simulation: latents ===")
    print(sim["latents"].to_string())
    print("\n=== Simulation: regional_state ===")
    print(sim["regional_state"].to_string())
    print("\n=== Simulation: symptoms ===")
    print(sim["symptoms"].to_string())
    print("\n=== Simulation: phenotypes ===")
    print(sim["phenotypes"].to_string())

    # Optional interactive examples for real siibra environments:
    # print(model.suggest_regions("temporal"))
    # print(model.assign_mni_point((24, -4, -18)).head())
