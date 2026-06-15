from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Delayed Ejaculation (DE).

This script turns a chapter-level biological narrative into a transparent,
reusable mechanistic model. It is intended for research scaffolding and model
inspection, not diagnosis or treatment.

Key modeling choices for this chapter:
- Keep serotonin, dopamine, autonomic output, and hypothalamic-pituitary
  influences as latent biology rather than pretending they map cleanly to a
  single cytoarchitectonic parcel.
- Anchor only the chapter's clearly named limbic regions directly and use
  explicit proxies where the chapter is systems-level (for example temporal
  lobe and hypothalamic control).
- Tolerate partial atlas / feature availability and siibra API variation.
"""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


DELAYED_EJACULATION_GENE_PANEL = [
    "HTR2C",
    "HTR1A",
    "HTR1B",
    "SLC6A4",
    "TPH2",
    "DRD2",
    "DRD4",
    "COMT",
    "MAOA",
    "BDNF",
]


class DelayedEjaculationModel:
    """
    Mechanistic siibra scaffold for Delayed Ejaculation.

    Conceptual flow:
        inputs -> latent biology -> regional burden/state -> symptoms -> phenotypes

    This is a conservative interpretation of the supplied chapter. The chapter
    centers serotonergic inhibition, 5-HT receptor subtype balance,
    dopaminergic under-facilitation, endocrine influences, peripheral nerve
    injury, and limbic/temporal dysfunction as plausible contributors to DE.

    Important limitation:
    The chapter explicitly notes that disorder-specific structural/functional
    neuroimaging evidence for idiopathic delayed ejaculation is sparse, so this
    scaffold uses conservative atlas anchors plus clearly labeled proxies.
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

        self.disorder_name = "Delayed Ejaculation"
        self.disorder_key = "delayed_ejaculation"

        # Region resolution is conservative: direct anchors only where the chapter
        # actually names anatomy, proxies where the chapter is broader or more
        # systems-level.
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
            "temporal_limbic_cortex_proxy": [
                "Area TG (Temporal Pole) left",
                "Temporal pole left",
                "Area TE 1.0 (TE) left",
                "Area TE 1.2 (TE) left",
                "temporal lobe",
                "temporal cortex",
            ],
            "hypothalamus_proxy": [
                "hypothalamus left",
                "hypothalamus",
                "hypothalamic",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "polygenic_vulnerability": (
                "Constitutional liability affecting baseline serotonergic tone, "
                "5-HT receptor balance, dopaminergic responsivity, and reflex threshold"
            ),
            "ssri_load": (
                "Iatrogenic serotonergic load from SSRI exposure or other strong "
                "serotonin-raising antidepressant effects"
            ),
            "limbic_epileptiform_burden": (
                "Temporal-lobe / limbic epileptiform activity capable of disrupting "
                "sexual, emotional, and autonomic regulation"
            ),
            "neurological_injury_load": (
                "Neurological disease burden such as stroke, multiple sclerosis, or "
                "other central lesions affecting ejaculatory coordination"
            ),
            "peripheral_nerve_damage": (
                "Peripheral neuropathic burden that can disrupt sensory or autonomic "
                "components of the ejaculatory reflex"
            ),
            "endocrine_burden": (
                "Endocrine or hormonal dysregulation capable of modulating sexual "
                "function through hypothalamic-pituitary pathways"
            ),
            "serotonin_sparing_adjustment": (
                "Protective reduction of serotonergic inhibition, modeled after "
                "switching or augmenting away from SSRI-heavy states"
            ),
            "dopamine_support": (
                "Protective facilitatory support for catecholaminergic drive; a "
                "mechanistic scaffold variable, not a treatment recommendation"
            ),
            "neurologic_endocrine_correction": (
                "Protective correction of reversible endocrine or neurological "
                "contributors to ejaculatory reflex failure"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_hyperactivity": (
                "Excess inhibitory central 5-HT tone that prolongs ejaculatory latency"
            ),
            "reduced_5ht1a_facilitation": (
                "Failure of the pro-ejaculatory / pro-orgasmic 5-HT1A pathway to "
                "adequately reduce serotonergic inhibition"
            ),
            "dopaminergic_hyporesponsivity": (
                "Under-responsive dopaminergic facilitation that raises the threshold "
                "for orgasmic and ejaculatory completion"
            ),
            "limbic_temporal_dysregulation": (
                "Dysregulated limbic-temporal coordination of sexual behaviour, "
                "emotion, and autonomic patterning"
            ),
            "hypothalamic_pituitary_dysregulation": (
                "Disordered hormonal modulation of sexual function through limbic, "
                "hypothalamic, and pituitary interactions"
            ),
            "autonomic_ejaculatory_output_failure": (
                "Breakdown of coordinated central-autonomic-peripheral execution of "
                "the ejaculatory reflex"
            ),
            "orgasm_generation_failure": (
                "Failure to reach or discharge the integrated orgasmic state despite "
                "arousal or sexual activity"
            ),
            "elevated_ejaculatory_threshold": (
                "Constitutionally or acquired high threshold for ejaculatory completion"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "delayed_ejaculation": "Marked prolongation of ejaculatory latency",
            "anejaculation": "Failure to ejaculate despite sustained stimulation",
            "delayed_orgasm": "Delayed or blunted orgasmic completion",
            "anorgasmia": "Absent orgasm, representing a severe DE phenotype",
            "sexual_distress": "Subjective burden or distress secondary to persistent delay/inhibition",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "ssri_load",
                "target": "serotonergic_hyperactivity",
                "relation": "raises synaptic serotonin and strengthens inhibitory 5-HT tone",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "polygenic_vulnerability",
                "target": "serotonergic_hyperactivity",
                "relation": "can bias baseline serotonergic tone upward",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "polygenic_vulnerability",
                "target": "dopaminergic_hyporesponsivity",
                "relation": "can lower facilitatory catecholaminergic drive",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "polygenic_vulnerability",
                "target": "reduced_5ht1a_facilitation",
                "relation": "can alter receptor/transporter balance and pro-ejaculatory serotonergic feedback",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "serotonergic_hyperactivity",
                "target": "reduced_5ht1a_facilitation",
                "relation": "overwhelms or counteracts pro-ejaculatory 5-HT1A-mediated release reduction",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "serotonergic_hyperactivity",
                "target": "elevated_ejaculatory_threshold",
                "relation": "prolongs latency via inhibitory receptor balance",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "dopaminergic_hyporesponsivity",
                "target": "elevated_ejaculatory_threshold",
                "relation": "weakens facilitatory motivational drive toward completion",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "limbic_epileptiform_burden",
                "target": "limbic_temporal_dysregulation",
                "relation": "disrupts limbic cortex organization of sexual and autonomic responses",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "neurological_injury_load",
                "target": "limbic_temporal_dysregulation",
                "relation": "adds distributed central burden to the sexual response network",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "limbic_temporal_dysregulation",
                "target": "hypothalamic_pituitary_dysregulation",
                "relation": "can perturb hormonal release and neuroendocrine control",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "endocrine_burden",
                "target": "hypothalamic_pituitary_dysregulation",
                "relation": "alters hormonal modulation of sexual function",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "neurological_injury_load",
                "target": "autonomic_ejaculatory_output_failure",
                "relation": "impairs central-autonomic coordination of emission and expulsion",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "peripheral_nerve_damage",
                "target": "autonomic_ejaculatory_output_failure",
                "relation": "disrupts peripheral sensory/autonomic signaling needed for reflex completion",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "hypothalamic_pituitary_dysregulation",
                "target": "autonomic_ejaculatory_output_failure",
                "relation": "destabilizes hormonal modulation of reflex execution",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "autonomic_ejaculatory_output_failure",
                "target": "elevated_ejaculatory_threshold",
                "relation": "forces more stimulation and stronger drive for completion",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "reduced_5ht1a_facilitation",
                "target": "orgasm_generation_failure",
                "relation": "weakens a pathway implicated in orgasm generation",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "dopaminergic_hyporesponsivity",
                "target": "orgasm_generation_failure",
                "relation": "reduces facilitatory reward/salience drive needed for orgasmic discharge",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "serotonergic_hyperactivity",
                "target": "delayed_ejaculation",
                "relation": "directly prolongs ejaculatory latency",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "elevated_ejaculatory_threshold",
                "target": "delayed_ejaculation",
                "relation": "requires more drive and time to complete ejaculation",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "autonomic_ejaculatory_output_failure",
                "target": "anejaculation",
                "relation": "can prevent coordinated reflex completion entirely",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "orgasm_generation_failure",
                "target": "delayed_orgasm",
                "relation": "slows the integrated orgasmic discharge",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "orgasm_generation_failure",
                "target": "anorgasmia",
                "relation": "can eliminate orgasmic completion in severe states",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "delayed_ejaculation",
                "target": "sexual_distress",
                "relation": "persistent ejaculatory delay can become subjectively burdensome",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "anejaculation",
                "target": "sexual_distress",
                "relation": "complete inhibition commonly increases burden",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "delayed_orgasm",
                "target": "sexual_distress",
                "relation": "prolonged or blunted orgasm increases distress burden",
                "delayed_ejaculation_change": "increased",
            },
            {
                "source": "serotonin_sparing_adjustment",
                "target": "serotonergic_hyperactivity",
                "relation": "reduces excess inhibitory serotonergic load",
                "delayed_ejaculation_change": "decreased",
            },
            {
                "source": "dopamine_support",
                "target": "dopaminergic_hyporesponsivity",
                "relation": "partially restores facilitatory catecholaminergic support",
                "delayed_ejaculation_change": "decreased",
            },
            {
                "source": "neurologic_endocrine_correction",
                "target": "hypothalamic_pituitary_dysregulation",
                "relation": "reduces reversible endocrine contribution to dysfunction",
                "delayed_ejaculation_change": "decreased",
            },
            {
                "source": "neurologic_endocrine_correction",
                "target": "autonomic_ejaculatory_output_failure",
                "relation": "reduces reversible neurogenic or endocrine burden on the reflex",
                "delayed_ejaculation_change": "decreased",
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

    # ---------------------------------------------------------------------
    # Utility helpers
    # ---------------------------------------------------------------------
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
            "temporal lobe",
            "temporal cortex",
            "hypothalamus",
        } else 0
        subregion_bonus = 0 if any(token in name for token in ("area ", "ca1", "ca2", "ca3", "lb", "cm", "sf", "subiculum")) else 1
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

        # Sometimes the connectivity query yields a compound feature containing
        # subject-level elements. Average a small subset for a stable scaffold,
        # but gracefully fall back to the first accessible matrix.
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

        # Loose token overlap fallback for generic parent/child naming differences.
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

    # ---------------------------------------------------------------------
    # Public atlas helpers
    # ---------------------------------------------------------------------
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

        # Current and legacy siibra API fallbacks.
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

    # ---------------------------------------------------------------------
    # Build and connectivity summaries
    # ---------------------------------------------------------------------
    def circuit_connectivity(self, max_rows_per_seed: int = 10) -> pd.DataFrame:
        if not self.region_objects:
            self.build(connectivity_rows=max_rows_per_seed)

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
        gene_panel: Sequence[str] = DELAYED_EJACULATION_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
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
        circuit_df = self.circuit_connectivity(max_rows_per_seed=connectivity_rows)

        self._build_cache = {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": circuit_df,
        }
        return self._build_cache

    # ---------------------------------------------------------------------
    # Simulator
    # ---------------------------------------------------------------------
    def simulate(
        self,
        polygenic_vulnerability: float = 0.35,
        ssri_load: float = 0.0,
        limbic_epileptiform_burden: float = 0.0,
        neurological_injury_load: float = 0.0,
        peripheral_nerve_damage: float = 0.0,
        endocrine_burden: float = 0.0,
        serotonin_sparing_adjustment: float = 0.0,
        dopamine_support: float = 0.0,
        neurologic_endocrine_correction: float = 0.0,
    ) -> Dict[str, pd.Series]:
        inp = {
            "polygenic_vulnerability": self._clip01(polygenic_vulnerability),
            "ssri_load": self._clip01(ssri_load),
            "limbic_epileptiform_burden": self._clip01(limbic_epileptiform_burden),
            "neurological_injury_load": self._clip01(neurological_injury_load),
            "peripheral_nerve_damage": self._clip01(peripheral_nerve_damage),
            "endocrine_burden": self._clip01(endocrine_burden),
            "serotonin_sparing_adjustment": self._clip01(serotonin_sparing_adjustment),
            "dopamine_support": self._clip01(dopamine_support),
            "neurologic_endocrine_correction": self._clip01(neurologic_endocrine_correction),
        }

        lat = {}
        lat["serotonergic_hyperactivity"] = self._clip01(
            0.46 * inp["ssri_load"]
            + 0.26 * inp["polygenic_vulnerability"]
            + 0.08 * inp["endocrine_burden"]
            - 0.24 * inp["serotonin_sparing_adjustment"]
        )
        lat["reduced_5ht1a_facilitation"] = self._clip01(
            0.38 * lat["serotonergic_hyperactivity"]
            + 0.24 * inp["polygenic_vulnerability"]
            + 0.08 * inp["neurological_injury_load"]
            - 0.12 * inp["serotonin_sparing_adjustment"]
        )
        lat["dopaminergic_hyporesponsivity"] = self._clip01(
            0.34 * inp["polygenic_vulnerability"]
            + 0.18 * inp["ssri_load"]
            + 0.14 * inp["neurological_injury_load"]
            - 0.24 * inp["dopamine_support"]
        )
        lat["limbic_temporal_dysregulation"] = self._clip01(
            0.42 * inp["limbic_epileptiform_burden"]
            + 0.28 * inp["neurological_injury_load"]
            + 0.10 * inp["polygenic_vulnerability"]
            - 0.10 * inp["neurologic_endocrine_correction"]
        )
        lat["hypothalamic_pituitary_dysregulation"] = self._clip01(
            0.40 * inp["endocrine_burden"]
            + 0.25 * lat["limbic_temporal_dysregulation"]
            + 0.12 * inp["neurological_injury_load"]
            - 0.24 * inp["neurologic_endocrine_correction"]
        )
        lat["autonomic_ejaculatory_output_failure"] = self._clip01(
            0.32 * inp["neurological_injury_load"]
            + 0.26 * inp["peripheral_nerve_damage"]
            + 0.18 * lat["hypothalamic_pituitary_dysregulation"]
            + 0.16 * lat["limbic_temporal_dysregulation"]
            - 0.22 * inp["neurologic_endocrine_correction"]
        )
        lat["orgasm_generation_failure"] = self._clip01(
            0.30 * lat["reduced_5ht1a_facilitation"]
            + 0.24 * lat["dopaminergic_hyporesponsivity"]
            + 0.16 * lat["serotonergic_hyperactivity"]
            + 0.12 * lat["autonomic_ejaculatory_output_failure"]
            + 0.05 * inp["limbic_epileptiform_burden"]
        )
        lat["elevated_ejaculatory_threshold"] = self._clip01(
            0.34 * lat["serotonergic_hyperactivity"]
            + 0.22 * lat["dopaminergic_hyporesponsivity"]
            + 0.20 * lat["autonomic_ejaculatory_output_failure"]
            + 0.12 * lat["hypothalamic_pituitary_dysregulation"]
            + 0.08 * inp["polygenic_vulnerability"]
        )

        regional = {
            "amygdala": self._clip01(
                0.48 * lat["limbic_temporal_dysregulation"]
                + 0.18 * lat["hypothalamic_pituitary_dysregulation"]
                + 0.10 * inp["polygenic_vulnerability"]
            ),
            "hippocampus": self._clip01(
                0.44 * lat["limbic_temporal_dysregulation"]
                + 0.12 * lat["hypothalamic_pituitary_dysregulation"]
                + 0.12 * inp["polygenic_vulnerability"]
            ),
            "temporal_limbic_cortex_proxy": self._clip01(
                0.54 * lat["limbic_temporal_dysregulation"]
                + 0.12 * inp["neurological_injury_load"]
                + 0.10 * inp["limbic_epileptiform_burden"]
            ),
            "hypothalamus_proxy": self._clip01(
                0.56 * lat["hypothalamic_pituitary_dysregulation"]
                + 0.18 * lat["limbic_temporal_dysregulation"]
                + 0.10 * inp["endocrine_burden"]
            ),
        }

        symptoms = {}
        symptoms["delayed_ejaculation"] = self._clip01(
            0.37 * lat["elevated_ejaculatory_threshold"]
            + 0.19 * lat["serotonergic_hyperactivity"]
            + 0.18 * lat["autonomic_ejaculatory_output_failure"]
            + 0.12 * lat["dopaminergic_hyporesponsivity"]
            + 0.06 * regional["hypothalamus_proxy"]
        )
        symptoms["anejaculation"] = self._clip01(
            0.35 * lat["autonomic_ejaculatory_output_failure"]
            + 0.22 * lat["elevated_ejaculatory_threshold"]
            + 0.16 * inp["neurological_injury_load"]
            + 0.10 * inp["peripheral_nerve_damage"]
            + 0.06 * regional["hypothalamus_proxy"]
        )
        symptoms["delayed_orgasm"] = self._clip01(
            0.24 * lat["serotonergic_hyperactivity"]
            + 0.22 * lat["reduced_5ht1a_facilitation"]
            + 0.22 * lat["orgasm_generation_failure"]
            + 0.12 * lat["dopaminergic_hyporesponsivity"]
            + 0.10 * lat["elevated_ejaculatory_threshold"]
        )
        symptoms["anorgasmia"] = self._clip01(
            0.32 * lat["orgasm_generation_failure"]
            + 0.24 * lat["reduced_5ht1a_facilitation"]
            + 0.16 * lat["serotonergic_hyperactivity"]
            + 0.10 * lat["autonomic_ejaculatory_output_failure"]
            + 0.10 * lat["dopaminergic_hyporesponsivity"]
        )
        symptoms["sexual_distress"] = self._clip01(
            0.34 * symptoms["delayed_ejaculation"]
            + 0.24 * symptoms["delayed_orgasm"]
            + 0.18 * symptoms["anejaculation"]
            + 0.14 * symptoms["anorgasmia"]
        )

        phenotypes = {
            "iatrogenic_serotonergic_de_profile": self._mean_clip(
                [
                    inp["ssri_load"],
                    lat["serotonergic_hyperactivity"],
                    symptoms["delayed_ejaculation"],
                    symptoms["delayed_orgasm"],
                ]
            ),
            "limbic_neurological_de_profile": self._mean_clip(
                [
                    inp["limbic_epileptiform_burden"],
                    lat["limbic_temporal_dysregulation"],
                    regional["temporal_limbic_cortex_proxy"],
                    symptoms["anejaculation"],
                ]
            ),
            "endocrine_autonomic_de_profile": self._mean_clip(
                [
                    inp["endocrine_burden"],
                    lat["hypothalamic_pituitary_dysregulation"],
                    lat["autonomic_ejaculatory_output_failure"],
                    symptoms["delayed_ejaculation"],
                ]
            ),
            "severe_anorgasmia_profile": self._mean_clip(
                [
                    lat["reduced_5ht1a_facilitation"],
                    lat["orgasm_generation_failure"],
                    symptoms["delayed_orgasm"],
                    symptoms["anorgasmia"],
                ]
            ),
            "constitutional_high_threshold_profile": self._mean_clip(
                [
                    inp["polygenic_vulnerability"],
                    lat["serotonergic_hyperactivity"],
                    lat["dopaminergic_hyporesponsivity"],
                    lat["elevated_ejaculatory_threshold"],
                ]
            ),
        }

        return {
            "inputs": self._series_from(inp, "inputs"),
            "latents": self._series_from(lat, "latents"),
            "regional_state": self._series_from(regional, "regional_state"),
            "symptoms": self._series_from(symptoms, "symptoms"),
            "phenotypes": self._series_from(phenotypes, "phenotypes"),
        }


if __name__ == "__main__":
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)

    model = DelayedEjaculationModel()
    bundle = model.build()

    print("\n=== NODE TABLE (key fields) ===")
    print(
        bundle["nodes"][["key", "node_type", "atlas_region", "region_identifier", "feature_summary"]]
        .fillna("")
        .to_string(index=False)
    )

    print("\n=== EDGE TABLE ===")
    print(bundle["edges"].to_string(index=False))

    for region_key in ["amygdala", "hippocampus", "temporal_limbic_cortex_proxy", "hypothalamus_proxy"]:
        receptor_df = bundle["receptors"].get(region_key, pd.DataFrame())
        gene_df = bundle["genes"].get(region_key, pd.DataFrame())
        conn_df = bundle["connectivity_profiles"].get(region_key, pd.DataFrame())

        print(f"\n=== REGION: {region_key} ===")
        print("Receptors:")
        print(receptor_df.head(10).to_string(index=False) if not receptor_df.empty else "<no receptor fingerprint available>")
        print("\nGenes:")
        print(gene_df.head(10).to_string(index=False) if not gene_df.empty else "<no gene-expression table available>")
        print("\nConnectivity:")
        print(conn_df.head(10).to_string(index=False) if not conn_df.empty else "<no connectivity profile available>")

    print("\n=== CIRCUIT CONNECTIVITY ===")
    circuit_df = bundle["circuit_connectivity"]
    print(circuit_df.head(25).to_string(index=False) if not circuit_df.empty else "<no circuit connectivity available>")

    example = model.simulate(
        polygenic_vulnerability=0.55,
        ssri_load=0.80,
        limbic_epileptiform_burden=0.10,
        neurological_injury_load=0.10,
        peripheral_nerve_damage=0.05,
        endocrine_burden=0.20,
        serotonin_sparing_adjustment=0.15,
        dopamine_support=0.10,
        neurologic_endocrine_correction=0.20,
    )

    print("\n=== SIMULATION: example serotonergic / high-threshold phenotype ===")
    for name, series in example.items():
        print(f"\n{name.upper()}")
        print(series.to_string())

    # Example coordinate usage:
    # assignments = model.assign_mni_point((0, -12, -8))
    # print(assignments.head(10).to_string(index=False))
