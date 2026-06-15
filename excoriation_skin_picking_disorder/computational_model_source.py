from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Excoriation (Skin-Picking) Disorder.

This script translates a chapter-level biological narrative into a transparent,
reusable mechanistic model. It is intended for research scaffolding and model
inspection, not diagnosis or treatment.

Key modeling choices for this chapter:
- Treat Excoriation Disorder as a body-focused repetitive behavior rooted in
  interacting corticostriatal, affective, sensory-salience, and habit systems,
  not as a primary dermatologic condition.
- Keep serotonin, dopamine, glutamate, and HPA-axis stress biology as latent
  processes unless the chapter clearly localizes them.
- Anchor the chapter's named circuitry conservatively with OFC, ACC, DLPFC,
  amygdala, and striatal proxies.
- Tolerate partial atlas / feature availability and siibra API variation.
"""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


EXCORIATION_SKIN_PICKING_GENE_PANEL = [
    "SLC1A1",
    "GRIN2B",
    "GRM5",
    "SLC6A4",
    "HTR2A",
    "DRD2",
    "COMT",
    "DLGAP3",
    "SLITRK5",
    "BDNF",
    "FKBP5",
    "NR3C1",
]


class ExcoriationSkinPickingDisorderModel:
    """
    Mechanistic siibra scaffold for Excoriation (Skin-Picking) Disorder.

    Conceptual flow:
        inputs -> latent biology -> regional burden/state -> symptoms -> phenotypes

    This is a conservative interpretation of the supplied chapter. The chapter
    emphasizes body-focused repetitive behavior, CSTC dysfunction,
    glutamatergic dysregulation, impaired impulse control, sensory triggering,
    and emotion-regulation failure.

    Important limitation:
    The chapter relies heavily on inference from OCD, trichotillomania, and
    related OCRDs rather than a large SPD-specific neuroimaging literature, so
    this scaffold uses conservative atlas anchors plus clearly labeled proxies.
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

        self.disorder_name = "Excoriation (Skin-Picking) Disorder"
        self.disorder_key = "excoriation_skin_picking_disorder"

        # Direct atlas anchors where the chapter is specific; proxies where the
        # chapter names systems or subcortical structures more generically.
        self.region_candidates: Dict[str, List[str]] = {
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Fo4 left",
                "Fo3 left",
                "orbitofrontal cortex left",
                "orbitofrontal cortex",
            ],
            "acc": [
                "Area p24ab left",
                "Area a24pr left",
                "Area p32 left",
                "anterior cingulate cortex left",
                "anterior cingulate cortex",
            ],
            "dlpfc": [
                "Area 9/46d left",
                "Area 46 left",
                "Area 9/46v left",
                "dorsolateral prefrontal cortex left",
                "dorsolateral prefrontal cortex",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "SF (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala",
            ],
            "caudate_proxy": [
                "caudate nucleus left",
                "caudate left",
                "caudate nucleus",
                "caudate",
            ],
            "nucleus_accumbens_proxy": [
                "nucleus accumbens left",
                "accumbens left",
                "nucleus accumbens",
                "ventral striatum",
                "accumbens",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Polygenic liability affecting corticostriatal control, neurotransmitter balance, "
                "stress responsivity, and habit susceptibility"
            ),
            "early_life_stress": (
                "Early stress or trauma burden capable of producing durable stress-system and "
                "emotion-regulation vulnerability"
            ),
            "negative_affect_load": (
                "Current tension, anxiety, or negative emotional pressure that can precipitate picking"
            ),
            "boredom_understimulation": (
                "Low-stimulation internal state that can increase repetitive self-directed behavior"
            ),
            "sensory_irregularity_cues": (
                "External sensory triggers such as perceived minor skin imperfections or tactile irregularities"
            ),
            "glutamate_homeostasis_support": (
                "Protective normalization of compulsive-drive circuitry modeled after glutamate-homeostasis "
                "restoration, not as a treatment recommendation"
            ),
            "top_down_control_support": (
                "Protective executive-control support that improves urge inhibition and response selection"
            ),
            "stress_regulation_support": (
                "Protective support that dampens HPA-axis amplification and negative-affect escalation"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "glutamatergic_cstc_hyperactivity": (
                "Hyperexcitable cortico-striato-thalamo-cortical signaling and impaired synaptic regulation"
            ),
            "serotonergic_regulatory_imbalance": (
                "Altered serotonergic modulation of impulsivity, affect regulation, and compulsive behavior"
            ),
            "dopaminergic_reinforcement_bias": (
                "Relief / gratification reinforcement bias that strengthens repetitive picking habits"
            ),
            "hpa_axis_sensitization": (
                "Stress-system sensitization that prolongs vulnerability to affect-driven picking"
            ),
            "limbic_negative_affect_drive": (
                "Amygdala-cingulate negative affect burden that motivates maladaptive self-regulation"
            ),
            "sensory_salience_amplification": (
                "Overweighting of minor skin irregularities and cue-triggered action readiness"
            ),
            "frontostriatal_control_failure": (
                "Failure of prefrontal top-down inhibition over striatal urge and habit systems"
            ),
            "compulsive_habit_reinforcement": (
                "Entrenched urge-action-relief loop that converts picking into a repetitive habit"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "urge_to_pick": "Escalating urge or pull to manipulate or pick at the skin",
            "repetitive_skin_picking": "Persistent repetitive picking behavior despite negative consequences",
            "loss_of_control_over_picking": "Failure to resist or stop picking once urges emerge",
            "tension_relief_cycle": "Pre-picking tension followed by gratification or relief after picking",
            "skin_damage": "Tissue damage / excoriation resulting from recurrent picking",
            "distress_impairment": "Clinical distress or impairment associated with persistent picking behavior",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "glutamatergic_cstc_hyperactivity",
                "relation": "can bias CSTC excitability and compulsive-loop vulnerability upward",
                "excoriation_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_regulatory_imbalance",
                "relation": "can alter baseline serotonergic regulation of impulsivity and affect",
                "excoriation_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "frontostriatal_control_failure",
                "relation": "can weaken inherited executive-control efficiency",
                "excoriation_change": "increased",
            },
            {
                "source": "early_life_stress",
                "target": "hpa_axis_sensitization",
                "relation": "can create enduring stress-response amplification via epigenetic and endocrine pathways",
                "excoriation_change": "increased",
            },
            {
                "source": "negative_affect_load",
                "target": "limbic_negative_affect_drive",
                "relation": "raises the aversive emotional state that can motivate picking",
                "excoriation_change": "increased",
            },
            {
                "source": "boredom_understimulation",
                "target": "dopaminergic_reinforcement_bias",
                "relation": "can increase repetitive self-stimulating habit selection",
                "excoriation_change": "increased",
            },
            {
                "source": "boredom_understimulation",
                "target": "sensory_salience_amplification",
                "relation": "can heighten attention to body-focused sensory cues",
                "excoriation_change": "increased",
            },
            {
                "source": "sensory_irregularity_cues",
                "target": "sensory_salience_amplification",
                "relation": "promotes over-detection of minor skin imperfections as action triggers",
                "excoriation_change": "increased",
            },
            {
                "source": "hpa_axis_sensitization",
                "target": "limbic_negative_affect_drive",
                "relation": "intensifies anxiety/tension states that precede picking",
                "excoriation_change": "increased",
            },
            {
                "source": "glutamatergic_cstc_hyperactivity",
                "target": "frontostriatal_control_failure",
                "relation": "overloads CSTC control loops and impairs inhibitory gating",
                "excoriation_change": "increased",
            },
            {
                "source": "serotonergic_regulatory_imbalance",
                "target": "frontostriatal_control_failure",
                "relation": "reduces stable modulation of impulsivity and compulsive restraint",
                "excoriation_change": "increased",
            },
            {
                "source": "sensory_salience_amplification",
                "target": "frontostriatal_control_failure",
                "relation": "biases attention toward urges and away from inhibitory control",
                "excoriation_change": "increased",
            },
            {
                "source": "glutamatergic_cstc_hyperactivity",
                "target": "compulsive_habit_reinforcement",
                "relation": "strengthens repetitive motor tendencies and compulsive loop persistence",
                "excoriation_change": "increased",
            },
            {
                "source": "dopaminergic_reinforcement_bias",
                "target": "compulsive_habit_reinforcement",
                "relation": "links relief / gratification to repeated picking",
                "excoriation_change": "increased",
            },
            {
                "source": "limbic_negative_affect_drive",
                "target": "compulsive_habit_reinforcement",
                "relation": "drives picking as maladaptive emotional self-regulation",
                "excoriation_change": "increased",
            },
            {
                "source": "sensory_salience_amplification",
                "target": "urge_to_pick",
                "relation": "converts minor irregularities into action urges",
                "excoriation_change": "increased",
            },
            {
                "source": "limbic_negative_affect_drive",
                "target": "urge_to_pick",
                "relation": "negative affect increases body-focused urge intensity",
                "excoriation_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "loss_of_control_over_picking",
                "relation": "weakens the ability to resist or stop picking",
                "excoriation_change": "increased",
            },
            {
                "source": "compulsive_habit_reinforcement",
                "target": "repetitive_skin_picking",
                "relation": "stabilizes repeated urge-driven picking behavior",
                "excoriation_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "repetitive_skin_picking",
                "relation": "allows subcortical urges to manifest as compulsive action",
                "excoriation_change": "increased",
            },
            {
                "source": "repetitive_skin_picking",
                "target": "skin_damage",
                "relation": "recurrent picking produces excoriation and tissue injury",
                "excoriation_change": "increased",
            },
            {
                "source": "limbic_negative_affect_drive",
                "target": "tension_relief_cycle",
                "relation": "drives the pre-picking tension state",
                "excoriation_change": "increased",
            },
            {
                "source": "dopaminergic_reinforcement_bias",
                "target": "tension_relief_cycle",
                "relation": "supports gratification / relief after picking",
                "excoriation_change": "increased",
            },
            {
                "source": "loss_of_control_over_picking",
                "target": "distress_impairment",
                "relation": "persistent inability to resist picking increases burden",
                "excoriation_change": "increased",
            },
            {
                "source": "skin_damage",
                "target": "distress_impairment",
                "relation": "visible tissue injury increases impairment and distress",
                "excoriation_change": "increased",
            },
            {
                "source": "tension_relief_cycle",
                "target": "distress_impairment",
                "relation": "recurrent tension-relief dynamics help sustain clinical burden",
                "excoriation_change": "increased",
            },
            {
                "source": "glutamate_homeostasis_support",
                "target": "glutamatergic_cstc_hyperactivity",
                "relation": "restores excitatory balance and reduces compulsive drive pressure",
                "excoriation_change": "decreased",
            },
            {
                "source": "top_down_control_support",
                "target": "frontostriatal_control_failure",
                "relation": "improves inhibitory control over picking urges",
                "excoriation_change": "decreased",
            },
            {
                "source": "stress_regulation_support",
                "target": "hpa_axis_sensitization",
                "relation": "reduces chronic stress amplification",
                "excoriation_change": "decreased",
            },
            {
                "source": "stress_regulation_support",
                "target": "limbic_negative_affect_drive",
                "relation": "lowers affective pressure that precipitates picking",
                "excoriation_change": "decreased",
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
            "orbitofrontal cortex",
            "anterior cingulate cortex",
            "dorsolateral prefrontal cortex",
            "amygdala",
            "caudate nucleus",
            "caudate",
            "nucleus accumbens",
            "ventral striatum",
        } else 0
        subregion_bonus = 0 if any(
            token in name
            for token in (
                "area ",
                "fo3",
                "fo4",
                "24",
                "32",
                "46",
                "9/46",
                "lb",
                "cm",
                "sf",
            )
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

        # Connectivity queries may return compound features containing
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
        gene_panel: Sequence[str] = EXCORIATION_SKIN_PICKING_GENE_PANEL,
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

    # ------------------------------------------------------------------
    # Simulator
    # ------------------------------------------------------------------
    def simulate(
        self,
        genetic_vulnerability: float = 0.40,
        early_life_stress: float = 0.25,
        negative_affect_load: float = 0.30,
        boredom_understimulation: float = 0.20,
        sensory_irregularity_cues: float = 0.30,
        glutamate_homeostasis_support: float = 0.0,
        top_down_control_support: float = 0.0,
        stress_regulation_support: float = 0.0,
    ) -> Dict[str, pd.Series]:
        inp = {
            "genetic_vulnerability": self._clip01(genetic_vulnerability),
            "early_life_stress": self._clip01(early_life_stress),
            "negative_affect_load": self._clip01(negative_affect_load),
            "boredom_understimulation": self._clip01(boredom_understimulation),
            "sensory_irregularity_cues": self._clip01(sensory_irregularity_cues),
            "glutamate_homeostasis_support": self._clip01(glutamate_homeostasis_support),
            "top_down_control_support": self._clip01(top_down_control_support),
            "stress_regulation_support": self._clip01(stress_regulation_support),
        }

        lat: Dict[str, float] = {}
        lat["glutamatergic_cstc_hyperactivity"] = self._clip01(
            0.30 * inp["genetic_vulnerability"]
            + 0.18 * inp["negative_affect_load"]
            + 0.16 * inp["sensory_irregularity_cues"]
            + 0.12 * inp["boredom_understimulation"]
            + 0.08 * inp["early_life_stress"]
            - 0.28 * inp["glutamate_homeostasis_support"]
        )
        lat["serotonergic_regulatory_imbalance"] = self._clip01(
            0.28 * inp["genetic_vulnerability"]
            + 0.20 * inp["negative_affect_load"]
            + 0.16 * inp["early_life_stress"]
            - 0.16 * inp["stress_regulation_support"]
        )
        lat["dopaminergic_reinforcement_bias"] = self._clip01(
            0.24 * inp["boredom_understimulation"]
            + 0.22 * inp["negative_affect_load"]
            + 0.18 * inp["sensory_irregularity_cues"]
            + 0.12 * inp["genetic_vulnerability"]
            + 0.10 * lat["glutamatergic_cstc_hyperactivity"]
            - 0.12 * inp["top_down_control_support"]
        )
        lat["hpa_axis_sensitization"] = self._clip01(
            0.40 * inp["early_life_stress"]
            + 0.20 * inp["negative_affect_load"]
            + 0.12 * inp["genetic_vulnerability"]
            - 0.28 * inp["stress_regulation_support"]
        )
        lat["limbic_negative_affect_drive"] = self._clip01(
            0.34 * inp["negative_affect_load"]
            + 0.24 * lat["hpa_axis_sensitization"]
            + 0.14 * lat["serotonergic_regulatory_imbalance"]
            + 0.10 * inp["early_life_stress"]
            - 0.22 * inp["stress_regulation_support"]
        )
        lat["sensory_salience_amplification"] = self._clip01(
            0.40 * inp["sensory_irregularity_cues"]
            + 0.20 * inp["boredom_understimulation"]
            + 0.16 * lat["limbic_negative_affect_drive"]
            + 0.12 * lat["glutamatergic_cstc_hyperactivity"]
            - 0.12 * inp["top_down_control_support"]
        )
        lat["frontostriatal_control_failure"] = self._clip01(
            0.28 * lat["glutamatergic_cstc_hyperactivity"]
            + 0.22 * lat["serotonergic_regulatory_imbalance"]
            + 0.20 * lat["limbic_negative_affect_drive"]
            + 0.12 * lat["sensory_salience_amplification"]
            + 0.10 * inp["genetic_vulnerability"]
            - 0.26 * inp["top_down_control_support"]
            - 0.08 * inp["glutamate_homeostasis_support"]
        )
        lat["compulsive_habit_reinforcement"] = self._clip01(
            0.26 * lat["glutamatergic_cstc_hyperactivity"]
            + 0.22 * lat["dopaminergic_reinforcement_bias"]
            + 0.20 * lat["sensory_salience_amplification"]
            + 0.18 * lat["limbic_negative_affect_drive"]
            + 0.14 * lat["frontostriatal_control_failure"]
            - 0.14 * inp["glutamate_homeostasis_support"]
            - 0.10 * inp["top_down_control_support"]
        )

        regional = {
            "ofc": self._clip01(
                0.56 * lat["frontostriatal_control_failure"]
                + 0.18 * lat["glutamatergic_cstc_hyperactivity"]
                + 0.08 * inp["genetic_vulnerability"]
            ),
            "acc": self._clip01(
                0.34 * lat["frontostriatal_control_failure"]
                + 0.28 * lat["limbic_negative_affect_drive"]
                + 0.16 * lat["hpa_axis_sensitization"]
                + 0.08 * lat["glutamatergic_cstc_hyperactivity"]
            ),
            "dlpfc": self._clip01(
                0.52 * lat["frontostriatal_control_failure"]
                + 0.14 * lat["limbic_negative_affect_drive"]
                + 0.08 * inp["genetic_vulnerability"]
            ),
            "amygdala": self._clip01(
                0.50 * lat["limbic_negative_affect_drive"]
                + 0.26 * lat["hpa_axis_sensitization"]
                + 0.10 * inp["negative_affect_load"]
            ),
            "caudate_proxy": self._clip01(
                0.50 * lat["glutamatergic_cstc_hyperactivity"]
                + 0.22 * lat["compulsive_habit_reinforcement"]
                + 0.12 * lat["frontostriatal_control_failure"]
            ),
            "nucleus_accumbens_proxy": self._clip01(
                0.46 * lat["dopaminergic_reinforcement_bias"]
                + 0.30 * lat["compulsive_habit_reinforcement"]
                + 0.10 * lat["sensory_salience_amplification"]
            ),
        }

        symptoms: Dict[str, float] = {}
        symptoms["urge_to_pick"] = self._clip01(
            0.30 * lat["sensory_salience_amplification"]
            + 0.24 * lat["limbic_negative_affect_drive"]
            + 0.16 * lat["compulsive_habit_reinforcement"]
            + 0.10 * lat["frontostriatal_control_failure"]
            + 0.06 * regional["amygdala"]
            + 0.06 * regional["nucleus_accumbens_proxy"]
        )
        symptoms["repetitive_skin_picking"] = self._clip01(
            0.32 * symptoms["urge_to_pick"]
            + 0.26 * lat["compulsive_habit_reinforcement"]
            + 0.20 * lat["frontostriatal_control_failure"]
            + 0.10 * regional["caudate_proxy"]
            + 0.06 * regional["ofc"]
        )
        symptoms["loss_of_control_over_picking"] = self._clip01(
            0.42 * lat["frontostriatal_control_failure"]
            + 0.18 * lat["compulsive_habit_reinforcement"]
            + 0.14 * symptoms["urge_to_pick"]
            + 0.10 * regional["ofc"]
            + 0.08 * regional["dlpfc"]
        )
        symptoms["tension_relief_cycle"] = self._clip01(
            0.32 * lat["limbic_negative_affect_drive"]
            + 0.28 * lat["dopaminergic_reinforcement_bias"]
            + 0.22 * lat["compulsive_habit_reinforcement"]
            + 0.10 * symptoms["repetitive_skin_picking"]
        )
        symptoms["skin_damage"] = self._clip01(
            0.46 * symptoms["repetitive_skin_picking"]
            + 0.18 * symptoms["loss_of_control_over_picking"]
            + 0.16 * symptoms["tension_relief_cycle"]
            + 0.10 * lat["sensory_salience_amplification"]
        )
        symptoms["distress_impairment"] = self._clip01(
            0.26 * symptoms["repetitive_skin_picking"]
            + 0.24 * symptoms["skin_damage"]
            + 0.18 * symptoms["loss_of_control_over_picking"]
            + 0.16 * symptoms["tension_relief_cycle"]
            + 0.10 * symptoms["urge_to_pick"]
        )

        phenotypes = {
            "affect_regulation_picking_profile": self._mean_clip(
                [
                    inp["negative_affect_load"],
                    lat["hpa_axis_sensitization"],
                    lat["limbic_negative_affect_drive"],
                    symptoms["tension_relief_cycle"],
                ]
            ),
            "cue_reactive_picking_profile": self._mean_clip(
                [
                    inp["sensory_irregularity_cues"],
                    lat["sensory_salience_amplification"],
                    symptoms["urge_to_pick"],
                    symptoms["repetitive_skin_picking"],
                ]
            ),
            "compulsive_cstc_profile": self._mean_clip(
                [
                    lat["glutamatergic_cstc_hyperactivity"],
                    lat["frontostriatal_control_failure"],
                    regional["caudate_proxy"],
                    symptoms["loss_of_control_over_picking"],
                ]
            ),
            "habit_reward_picking_profile": self._mean_clip(
                [
                    inp["boredom_understimulation"],
                    lat["dopaminergic_reinforcement_bias"],
                    lat["compulsive_habit_reinforcement"],
                    regional["nucleus_accumbens_proxy"],
                ]
            ),
            "severe_skin_damage_profile": self._mean_clip(
                [
                    symptoms["repetitive_skin_picking"],
                    symptoms["skin_damage"],
                    symptoms["distress_impairment"],
                    symptoms["loss_of_control_over_picking"],
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
    pd.set_option("display.width", 150)

    model = ExcoriationSkinPickingDisorderModel()
    bundle = model.build()

    print("\n=== NODE TABLE (key fields) ===")
    print(
        bundle["nodes"][["key", "node_type", "atlas_region", "region_identifier", "feature_summary"]]
        .fillna("")
        .to_string(index=False)
    )

    print("\n=== EDGE TABLE ===")
    print(bundle["edges"].to_string(index=False))

    for region_key in ["ofc", "acc", "dlpfc", "amygdala", "caudate_proxy", "nucleus_accumbens_proxy"]:
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
        genetic_vulnerability=0.55,
        early_life_stress=0.45,
        negative_affect_load=0.70,
        boredom_understimulation=0.35,
        sensory_irregularity_cues=0.65,
        glutamate_homeostasis_support=0.20,
        top_down_control_support=0.15,
        stress_regulation_support=0.20,
    )

    print("\n=== SIMULATION: example affect-loaded cue-reactive picking phenotype ===")
    for name, series in example.items():
        print(f"\n{name.upper()}")
        print(series.to_string())

    # Example coordinate usage:
    # assignments = model.assign_mni_point((-8, 44, -12))
    # print(assignments.head(10).to_string(index=False))
