from __future__ import annotations

"""
Reactive Attachment Disorder atlas-grounded siibra scaffold.

This script translates a short neurobiological chapter on Reactive Attachment
Disorder (RAD) into a transparent research scaffold. It is intended for
mechanistic exploration and atlas-backed feature querying, not for diagnosis,
treatment selection, or prediction for any individual.

Core chapter logic encoded here:
- early caregiver deprivation and maltreatment disrupt attachment formation and
  self-regulation during critical developmental periods,
- HPA-axis dysregulation biologically embeds chronic early stress,
- mesocorticolimbic reward dysfunction blunts motivation for comfort-seeking
  and positive social engagement,
- GABA/glutamate imbalance can contribute to hyperarousal and stress-linked
  neural burden,
- epigenetic adaptation and altered neurodevelopment can stabilize long-term
  changes in stress, reward, and social-bonding systems,
- amygdala threat bias and weakened prefrontal control contribute to persistent
  fear, anxiety, irritability, and emotionally withdrawn attachment behavior.
"""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "This scaffold requires the 'siibra' package. Install siibra-python and rerun."
    ) from exc


RAD_GENE_PANEL = [
    "NR3C1",
    "FKBP5",
    "CRHR1",
    "CRHBP",
    "OXTR",
    "CD38",
    "SLC6A3",
    "DRD2",
    "DRD4",
    "COMT",
    "SLC6A4",
    "HTR1A",
    "BDNF",
    "GAD1",
    "GABRA2",
    "GRIN1",
    "GRIN2B",
    "SLC1A2",
]


class ReactiveAttachmentDisorderModel:
    """
    Atlas-grounded research scaffold for Reactive Attachment Disorder.

    Notes
    -----
    - The simulator is intentionally simple, normalized, and acyclic.
    - Region mappings are conservative. Broad or uncertain systems are modeled
      as explicit proxies rather than forced into overly precise parcels.
    - The model captures a mechanistic interpretation of the uploaded chapter,
      not a validated clinical biomarker model.
    - Multimodal feature lookup degrades gracefully when receptor, gene, or
      connectivity data are missing.
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

        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "Amygdala left",
                "amygdala",
            ],
            "pfc_control": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 9/46v left",
                "dorsolateral prefrontal",
                "prefrontal cortex",
            ],
            "nucleus_accumbens_proxy": [
                "nucleus accumbens",
                "accumbens",
                "ventral striatum",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "amygdala": "Threat-detection and fear-salience node capturing exaggerated bottom-up reactivity.",
            "pfc_control": "Representative left prefrontal control node for dlPFC / broader prefrontal regulatory burden.",
            "nucleus_accumbens_proxy": "Reward-salience proxy for mesocorticolimbic motivation and social approach circuitry.",
        }

        self.input_nodes: Dict[str, str] = {
            "early_caregiver_deprivation": "Failure to establish a secure attachment with a consistent responsive caregiver during critical developmental periods.",
            "maltreatment_trauma_load": "Neglect, abuse, and chronic developmental stress burden affecting arousal and neural development.",
            "repeated_adversity": "Cumulative and recurrent exposure to deprivation or trauma that reinforces biological embedding.",
            "genetic_epigenetic_vulnerability": "Baseline susceptibility in stress, reward, and social-bonding biology that can interact with adversity.",
            "stable_caregiving_support": "Protective stable responsive caregiving and relational safety that can buffer dysregulation.",
            "developmental_enrichment_support": "Protective developmental support, environmental enrichment, and corrective scaffolding for regulation.",
        }

        self.latent_nodes: Dict[str, str] = {
            "hpa_axis_dysregulation": "Stress-system dysregulation that heightens physiological and emotional reactivity.",
            "dopaminergic_social_reward_blunting": "Mesocorticolimbic reward dysfunction reducing pleasure, initiative, and social motivation.",
            "social_bonding_signal_disruption": "Potential disruption of affiliative attachment signaling relevant to comfort-seeking and social approach.",
            "gaba_glutamate_imbalance": "Reduced inhibitory buffering and/or excessive excitatory drive contributing to hyperarousal and neural burden.",
            "epigenetic_embedding_of_adversity": "Stable biological embedding of early adversity through enduring molecular regulation changes.",
            "altered_neurodevelopmental_architecture": "Stress-linked disruption of pruning, myelination, and gray-matter development in regulatory systems.",
            "frontolimbic_regulatory_failure": "Compromised prefrontal-amygdala regulation producing weak top-down control over emotional arousal.",
        }

        self.symptom_nodes: Dict[str, str] = {
            "failure_to_seek_comfort": "Reduced tendency to seek comfort or support when distressed.",
            "limited_positive_affect": "Restricted positive affect, social apathy, or anhedonic emotional expression.",
            "social_withdrawal": "Emotionally withdrawn social behavior and low engagement with caregivers or others.",
            "fear_anxiety_irritability": "Persistent fearfulness, anxiety, irritability, or hyperaroused affect.",
            "emotion_regulation_failure": "Core self-regulation impairment spanning physiological and emotional control.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "early_caregiver_deprivation",
                "target": "hpa_axis_dysregulation",
                "relation": "disrupts early self-regulation and stress calibration during attachment formation",
                "rad_change": "increased",
            },
            {
                "source": "early_caregiver_deprivation",
                "target": "social_bonding_signal_disruption",
                "relation": "weakens the development of affiliative and comfort-seeking signaling",
                "rad_change": "increased",
            },
            {
                "source": "early_caregiver_deprivation",
                "target": "altered_neurodevelopmental_architecture",
                "relation": "burdens critical-period neural development in regulatory systems",
                "rad_change": "increased",
            },
            {
                "source": "maltreatment_trauma_load",
                "target": "hpa_axis_dysregulation",
                "relation": "drives chronic stress-system dysregulation and cortisol-related burden",
                "rad_change": "increased",
            },
            {
                "source": "maltreatment_trauma_load",
                "target": "gaba_glutamate_imbalance",
                "relation": "shifts inhibitory-excitatory balance toward hyperarousal and potential excitotoxic stress",
                "rad_change": "increased",
            },
            {
                "source": "maltreatment_trauma_load",
                "target": "epigenetic_embedding_of_adversity",
                "relation": "promotes stable adversity-linked molecular regulation changes",
                "rad_change": "increased",
            },
            {
                "source": "repeated_adversity",
                "target": "epigenetic_embedding_of_adversity",
                "relation": "reinforces durable biological embedding of early stress",
                "rad_change": "increased",
            },
            {
                "source": "repeated_adversity",
                "target": "altered_neurodevelopmental_architecture",
                "relation": "progressively burdens distributed neural maturation",
                "rad_change": "increased",
            },
            {
                "source": "genetic_epigenetic_vulnerability",
                "target": "hpa_axis_dysregulation",
                "relation": "raises baseline susceptibility in developmental stress-reactivity systems",
                "rad_change": "increased",
            },
            {
                "source": "genetic_epigenetic_vulnerability",
                "target": "dopaminergic_social_reward_blunting",
                "relation": "contributes to vulnerability in reward and social motivation systems",
                "rad_change": "increased",
            },
            {
                "source": "stable_caregiving_support",
                "target": "hpa_axis_dysregulation",
                "relation": "buffers stress calibration and reduces chronic physiological dysregulation",
                "rad_change": "decreased",
            },
            {
                "source": "stable_caregiving_support",
                "target": "social_bonding_signal_disruption",
                "relation": "supports affiliative safety and comfort-seeking salience",
                "rad_change": "decreased",
            },
            {
                "source": "developmental_enrichment_support",
                "target": "altered_neurodevelopmental_architecture",
                "relation": "supports corrective maturation of regulatory circuitry",
                "rad_change": "decreased",
            },
            {
                "source": "developmental_enrichment_support",
                "target": "dopaminergic_social_reward_blunting",
                "relation": "supports approach motivation and positive affective engagement",
                "rad_change": "decreased",
            },
            {
                "source": "epigenetic_embedding_of_adversity",
                "target": "hpa_axis_dysregulation",
                "relation": "stabilizes altered stress reactivity over time",
                "rad_change": "increased",
            },
            {
                "source": "epigenetic_embedding_of_adversity",
                "target": "dopaminergic_social_reward_blunting",
                "relation": "contributes to enduring reward-circuit blunting in adversity-exposed development",
                "rad_change": "increased",
            },
            {
                "source": "epigenetic_embedding_of_adversity",
                "target": "social_bonding_signal_disruption",
                "relation": "can alter attachment-relevant reward and oxytocin-linked systems",
                "rad_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "frontolimbic_regulatory_failure",
                "relation": "amplifies stress-linked imbalance between threat reactivity and top-down control",
                "rad_change": "increased",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "frontolimbic_regulatory_failure",
                "relation": "increases arousal instability and undermines regulation circuits",
                "rad_change": "increased",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "altered_neurodevelopmental_architecture",
                "relation": "adds excitatory stress burden to ongoing neural maturation",
                "rad_change": "increased",
            },
            {
                "source": "altered_neurodevelopmental_architecture",
                "target": "pfc_control",
                "relation": "maps developmental burden onto representative prefrontal control circuitry",
                "rad_change": "increased",
            },
            {
                "source": "altered_neurodevelopmental_architecture",
                "target": "frontolimbic_regulatory_failure",
                "relation": "weakens structural support for effective top-down regulation",
                "rad_change": "increased",
            },
            {
                "source": "frontolimbic_regulatory_failure",
                "target": "amygdala",
                "relation": "permits exaggerated threat reactivity under weak top-down constraint",
                "rad_change": "increased",
            },
            {
                "source": "frontolimbic_regulatory_failure",
                "target": "pfc_control",
                "relation": "burdens prefrontal recruitment during emotion regulation",
                "rad_change": "increased",
            },
            {
                "source": "dopaminergic_social_reward_blunting",
                "target": "nucleus_accumbens_proxy",
                "relation": "maps reduced social motivation and wanting onto reward-circuit proxy burden",
                "rad_change": "increased",
            },
            {
                "source": "social_bonding_signal_disruption",
                "target": "failure_to_seek_comfort",
                "relation": "reduces the drive to seek comfort from caregivers or others",
                "rad_change": "increased",
            },
            {
                "source": "nucleus_accumbens_proxy",
                "target": "limited_positive_affect",
                "relation": "reward blunting supports restricted positive affect and anhedonic social engagement",
                "rad_change": "increased",
            },
            {
                "source": "nucleus_accumbens_proxy",
                "target": "social_withdrawal",
                "relation": "reduced reward salience weakens positive social approach",
                "rad_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "fear_anxiety_irritability",
                "relation": "heightens threat sensitivity and distress reactivity",
                "rad_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "emotion_regulation_failure",
                "relation": "bottom-up reactivity overwhelms fragile regulation capacity",
                "rad_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "emotion_regulation_failure",
                "relation": "prefrontal under-recruitment weakens top-down emotional control",
                "rad_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "failure_to_seek_comfort",
                "relation": "poor regulatory recruitment limits flexible approach to comfort and support",
                "rad_change": "increased",
            },
            {
                "source": "dopaminergic_social_reward_blunting",
                "target": "limited_positive_affect",
                "relation": "reduces motivation and pleasure for positive social engagement",
                "rad_change": "increased",
            },
            {
                "source": "failure_to_seek_comfort",
                "target": "social_withdrawal",
                "relation": "low comfort-seeking reinforces emotionally withdrawn social behavior",
                "rad_change": "increased",
            },
            {
                "source": "limited_positive_affect",
                "target": "social_withdrawal",
                "relation": "restricted positive affect supports persistent disengagement",
                "rad_change": "increased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        self._pmap: Any = None
        self._connectivity_feature: Any = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        vals = [float(v) for v in values]
        return 0.0 if not vals else sum(vals) / len(vals)

    @staticmethod
    def _region_identifier(region: Any) -> Optional[str]:
        return getattr(region, "identifier", getattr(region, "id", None))

    def _modality_candidates(self, kind: str) -> List[Any]:
        candidates: List[Any] = []
        feature_spaces = [
            getattr(siibra.features, "molecular", None),
            getattr(siibra.features, "tabular", None),
            getattr(siibra, "features", None),
        ]

        if kind == "receptor":
            names = ["ReceptorDensityFingerprint"]
        elif kind == "gene":
            names = ["GeneExpressions"]
        elif kind == "connectivity":
            feature_spaces.append(getattr(siibra.features, "connectivity", None))
            names = ["StreamlineCounts"]
        else:
            names = []

        for space in feature_spaces:
            if space is None:
                continue
            for name in names:
                try:
                    obj = getattr(space, name)
                except Exception:
                    obj = None
                if obj is not None and obj not in candidates:
                    candidates.append(obj)

        if kind == "receptor":
            candidates.extend([
                "receptor density fingerprint",
                "ReceptorDensityFingerprint",
            ])
        elif kind == "gene":
            candidates.extend([
                "gene expressions",
                "GeneExpressions",
            ])
        elif kind == "connectivity":
            candidates.extend([
                "StreamlineCounts",
                "streamline counts",
            ])
        return candidates

    def _safe_features_any(
        self,
        concept: Any,
        modalities: Sequence[Any],
        **kwargs: Any,
    ) -> List[Any]:
        for modality in modalities:
            try:
                with siibra.QUIET:
                    features = siibra.features.get(concept, modality, **kwargs)
                if features:
                    return list(features)
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
        generic_penalty = 1 if name in {"amygdala", "prefrontal cortex", "ventral striatum"} else 0
        parent_penalty = 1 if any(token in name for token in ["lobe", "cortex"]) and "area" not in name else 0
        return (left_bonus, right_penalty, generic_penalty, parent_penalty)

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
            row = (
                self._name_of(region),
                self._region_identifier(region),
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

    def _main_component(
        self, region: Any
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_raw = getattr(centroid, "coordinate", centroid)
        centroid_xyz = None
        if centroid_raw is not None:
            try:
                centroid_xyz = tuple(float(x) for x in centroid_raw)
            except Exception:
                centroid_xyz = None
        volume_mm3 = getattr(main, "volume", None)
        volume = float(volume_mm3) if volume_mm3 is not None else None
        return centroid_xyz, volume

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        features = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not features:
            return pd.DataFrame()
        try:
            data = features[0].data.copy()
            if isinstance(data, pd.Series):
                df = data.reset_index()
                if len(df.columns) == 2:
                    df.columns = ["receptor", "value"]
                return df
            if isinstance(data, pd.DataFrame):
                df = data.reset_index()
                if "index" in df.columns and "receptor" not in df.columns:
                    df = df.rename(columns={"index": "receptor"})
                return df
        except Exception:
            pass
        return pd.DataFrame()

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        features = self._safe_features_any(
            region,
            self._modality_candidates("gene"),
            gene=list(genes),
        )
        if not features:
            features = self._safe_features_any(
                region,
                self._modality_candidates("gene"),
                genes=list(genes),
            )
        if not features:
            return pd.DataFrame()
        try:
            df = features[0].data.copy()
        except Exception:
            return pd.DataFrame()
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()

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
                .reset_index(drop=True)
            )
            return out
        return df.reset_index(drop=True)

    def _pick_connectivity_feature(self, features: Sequence[Any]) -> Optional[Any]:
        if not features:
            return None
        wanted = str(self.connectivity_cohort).lower()
        for feat in features:
            cohort = str(getattr(feat, "cohort", "")).lower()
            if cohort == wanted:
                return feat
        return features[0]

    @staticmethod
    def _coerce_scalar(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            pass
        if isinstance(value, pd.Series):
            numeric = pd.to_numeric(value, errors="coerce").dropna()
            return float(numeric.mean()) if not numeric.empty else None
        if isinstance(value, pd.DataFrame):
            numeric = value.apply(pd.to_numeric, errors="coerce").stack().dropna()
            return float(numeric.mean()) if not numeric.empty else None
        if isinstance(value, (list, tuple)):
            numeric = [ReactiveAttachmentDisorderModel._coerce_scalar(v) for v in value]
            numeric = [v for v in numeric if v is not None]
            return float(sum(numeric) / len(numeric)) if numeric else None
        return None

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        features = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not features and self.region_objects:
            first_region = next(iter(self.region_objects.values()))
            features = self._safe_features_any(first_region, self._modality_candidates("connectivity"))

        feature = self._pick_connectivity_feature(features)
        self._connectivity_feature = feature
        if feature is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        try:
            data = getattr(feature, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            for element in feature:
                data = getattr(element, "data", None)
                if isinstance(data, pd.DataFrame):
                    self._connectivity_matrix = data.copy()
                    return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        target = self._name_of(region)
        exact = [label for label in labels if self._name_of(label) == target]
        if exact:
            return exact[0]
        target_lower = target.lower()
        fuzzy = [
            label
            for label in labels
            if target_lower in self._name_of(label).lower()
            or self._name_of(label).lower() in target_lower
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
            if not isinstance(series, pd.Series):
                return pd.DataFrame()
            numeric = pd.to_numeric(series, errors="coerce").dropna().sort_values(ascending=False)
            if numeric.empty:
                return pd.DataFrame()
            df = numeric.reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        labels_index = list(matrix.index)
        labels_columns = list(matrix.columns)
        keys = list(self.region_objects.keys())
        rows: List[Dict[str, Any]] = []

        for i, source_key in enumerate(keys):
            source_region = self.region_objects[source_key]
            source_idx = self._match_region_label(labels_index, source_region)
            source_col = self._match_region_label(labels_columns, source_region)
            for target_key in keys[i + 1 :]:
                target_region = self.region_objects[target_key]
                target_idx = self._match_region_label(labels_index, target_region)
                target_col = self._match_region_label(labels_columns, target_region)

                value = None
                try:
                    if source_idx is not None and target_col is not None:
                        value = self._coerce_scalar(matrix.loc[source_idx, target_col])
                except Exception:
                    value = None
                if value is None:
                    try:
                        if target_idx is not None and source_col is not None:
                            value = self._coerce_scalar(matrix.loc[target_idx, source_col])
                    except Exception:
                        value = None

                rows.append(
                    {
                        "source_key": source_key,
                        "source_region": source_region.name,
                        "target_key": target_key,
                        "target_region": target_region.name,
                        "value": value,
                        "cohort": self.connectivity_cohort,
                        "modality": "StreamlineCounts",
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty and "value" in df.columns:
            df = df.sort_values("value", ascending=False, na_position="last").reset_index(drop=True)
        return df

    def build(
        self,
        gene_panel: Sequence[str] = RAD_GENE_PANEL,
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
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region",
                        "description": self.region_node_descriptions.get(
                            key,
                            "Atlas-backed region node unresolved in this environment.",
                        ),
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
            connectivity_df = self._connectivity_profile(region, max_rows=connectivity_rows)
            self.receptors[key] = receptor_df
            self.genes[key] = gene_df
            self.connectivity_profiles[key] = connectivity_df

            nodes.append(
                {
                    "key": key,
                    "label": region.name,
                    "node_type": "region",
                    "description": self.region_node_descriptions.get(key, "Atlas-backed circuit node."),
                    "atlas_region": region.name,
                    "region_identifier": self._region_identifier(region),
                    "centroid_mni": centroid_mni,
                    "volume_mm3": volume_mm3,
                    "feature_summary": (
                        f"receptors={'yes' if not receptor_df.empty else 'no'}; "
                        f"genes={'yes' if not gene_df.empty else 'no'}; "
                        f"connectivity={'yes' if not connectivity_df.empty else 'no'}"
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
        circuit_df = self.circuit_connectivity()
        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": circuit_df,
            "metadata": {
                "disorder": "Reactive Attachment Disorder",
                "gene_panel": list(gene_panel),
                "atlas": getattr(self.atlas, "name", str(self.atlas)),
                "parcellation": getattr(self.parcellation, "name", str(self.parcellation)),
                "space": getattr(self.space, "name", str(self.space)),
                "assignment_space": self.assignment_space,
                "connectivity_cohort": self.connectivity_cohort,
                "notes": "Research scaffold only; not a clinical tool.",
            },
        }

    def simulate(
        self,
        genetic_epigenetic_vulnerability: float = 0.35,
        early_caregiver_deprivation: float = 0.80,
        maltreatment_trauma_load: float = 0.65,
        repeated_adversity: float = 0.55,
        stable_caregiving_support: float = 0.20,
        developmental_enrichment_support: float = 0.20,
    ) -> Dict[str, pd.Series]:
        """
        Run a transparent normalized Reactive Attachment Disorder simulation.

        All inputs are clipped to [0, 1]. Higher values generally represent
        more burden, except the two protective support variables.
        """

        gev = self._clip01(genetic_epigenetic_vulnerability)
        ecd = self._clip01(early_caregiver_deprivation)
        mtl = self._clip01(maltreatment_trauma_load)
        ra = self._clip01(repeated_adversity)
        scs = self._clip01(stable_caregiving_support)
        des = self._clip01(developmental_enrichment_support)

        inputs = pd.Series(
            {
                "genetic_epigenetic_vulnerability": gev,
                "early_caregiver_deprivation": ecd,
                "maltreatment_trauma_load": mtl,
                "repeated_adversity": ra,
                "stable_caregiving_support": scs,
                "developmental_enrichment_support": des,
            },
            name="inputs",
        )

        epigenetic_embedding_of_adversity = self._clip01(
            0.30 * ecd + 0.30 * mtl + 0.20 * ra + 0.15 * gev - 0.15 * scs - 0.05 * des
        )
        hpa_axis_dysregulation = self._clip01(
            0.30 * ecd
            + 0.25 * mtl
            + 0.15 * ra
            + 0.10 * gev
            + 0.20 * epigenetic_embedding_of_adversity
            - 0.25 * scs
            - 0.10 * des
        )
        dopaminergic_social_reward_blunting = self._clip01(
            0.30 * ecd
            + 0.20 * mtl
            + 0.10 * gev
            + 0.20 * epigenetic_embedding_of_adversity
            + 0.10 * hpa_axis_dysregulation
            - 0.20 * scs
            - 0.15 * des
        )
        social_bonding_signal_disruption = self._clip01(
            0.30 * ecd
            + 0.15 * mtl
            + 0.15 * ra
            + 0.20 * epigenetic_embedding_of_adversity
            + 0.05 * gev
            - 0.25 * scs
            - 0.10 * des
        )
        gaba_glutamate_imbalance = self._clip01(
            0.25 * mtl
            + 0.20 * hpa_axis_dysregulation
            + 0.15 * ra
            + 0.10 * gev
            + 0.10 * epigenetic_embedding_of_adversity
            - 0.10 * scs
            - 0.05 * des
        )
        altered_neurodevelopmental_architecture = self._clip01(
            0.25 * ecd
            + 0.20 * mtl
            + 0.20 * epigenetic_embedding_of_adversity
            + 0.15 * hpa_axis_dysregulation
            + 0.10 * gaba_glutamate_imbalance
            + 0.05 * ra
            - 0.15 * scs
            - 0.10 * des
        )
        frontolimbic_regulatory_failure = self._clip01(
            0.25 * hpa_axis_dysregulation
            + 0.20 * gaba_glutamate_imbalance
            + 0.20 * altered_neurodevelopmental_architecture
            + 0.15 * dopaminergic_social_reward_blunting
            + 0.10 * social_bonding_signal_disruption
            + 0.05 * epigenetic_embedding_of_adversity
            - 0.15 * scs
            - 0.10 * des
        )

        latents = pd.Series(
            {
                "hpa_axis_dysregulation": hpa_axis_dysregulation,
                "dopaminergic_social_reward_blunting": dopaminergic_social_reward_blunting,
                "social_bonding_signal_disruption": social_bonding_signal_disruption,
                "gaba_glutamate_imbalance": gaba_glutamate_imbalance,
                "epigenetic_embedding_of_adversity": epigenetic_embedding_of_adversity,
                "altered_neurodevelopmental_architecture": altered_neurodevelopmental_architecture,
                "frontolimbic_regulatory_failure": frontolimbic_regulatory_failure,
            },
            name="latents",
        )

        amygdala_threat_bias = self._clip01(
            0.35 * hpa_axis_dysregulation
            + 0.25 * gaba_glutamate_imbalance
            + 0.20 * frontolimbic_regulatory_failure
            + 0.10 * altered_neurodevelopmental_architecture
            + 0.10 * mtl
            - 0.10 * scs
        )
        pfc_control_dysfunction = self._clip01(
            0.35 * frontolimbic_regulatory_failure
            + 0.30 * altered_neurodevelopmental_architecture
            + 0.15 * hpa_axis_dysregulation
            + 0.10 * epigenetic_embedding_of_adversity
            - 0.20 * scs
            - 0.10 * des
        )
        nucleus_accumbens_social_reward_deficit = self._clip01(
            0.40 * dopaminergic_social_reward_blunting
            + 0.25 * social_bonding_signal_disruption
            + 0.15 * epigenetic_embedding_of_adversity
            + 0.10 * hpa_axis_dysregulation
            - 0.20 * scs
            - 0.10 * des
        )

        regional_state = pd.Series(
            {
                "amygdala_threat_bias": amygdala_threat_bias,
                "pfc_control_dysfunction": pfc_control_dysfunction,
                "nucleus_accumbens_social_reward_deficit": nucleus_accumbens_social_reward_deficit,
            },
            name="regional_state",
        )

        failure_to_seek_comfort = self._clip01(
            0.35 * social_bonding_signal_disruption
            + 0.25 * nucleus_accumbens_social_reward_deficit
            + 0.15 * pfc_control_dysfunction
            + 0.10 * amygdala_threat_bias
            + 0.15 * ecd
            - 0.10 * scs
        )
        limited_positive_affect = self._clip01(
            0.45 * nucleus_accumbens_social_reward_deficit
            + 0.20 * dopaminergic_social_reward_blunting
            + 0.15 * social_bonding_signal_disruption
            + 0.10 * hpa_axis_dysregulation
            - 0.10 * des
            - 0.10 * scs
        )
        social_withdrawal = self._clip01(
            0.30 * failure_to_seek_comfort
            + 0.25 * limited_positive_affect
            + 0.20 * amygdala_threat_bias
            + 0.15 * pfc_control_dysfunction
            + 0.10 * frontolimbic_regulatory_failure
            - 0.05 * scs
        )
        fear_anxiety_irritability = self._clip01(
            0.35 * amygdala_threat_bias
            + 0.25 * hpa_axis_dysregulation
            + 0.20 * gaba_glutamate_imbalance
            + 0.10 * frontolimbic_regulatory_failure
            + 0.10 * pfc_control_dysfunction
            - 0.10 * scs
        )
        emotion_regulation_failure = self._clip01(
            0.35 * pfc_control_dysfunction
            + 0.25 * amygdala_threat_bias
            + 0.20 * frontolimbic_regulatory_failure
            + 0.10 * hpa_axis_dysregulation
            + 0.10 * altered_neurodevelopmental_architecture
            - 0.15 * scs
            - 0.10 * des
        )

        symptoms = pd.Series(
            {
                "failure_to_seek_comfort": failure_to_seek_comfort,
                "limited_positive_affect": limited_positive_affect,
                "social_withdrawal": social_withdrawal,
                "fear_anxiety_irritability": fear_anxiety_irritability,
                "emotion_regulation_failure": emotion_regulation_failure,
            },
            name="symptoms",
        )

        emotionally_withdrawn_attachment_profile = self._clip01(
            self._mean([failure_to_seek_comfort, limited_positive_affect, social_withdrawal])
        )
        threat_hyperarousal_profile = self._clip01(
            self._mean([fear_anxiety_irritability, emotion_regulation_failure, amygdala_threat_bias])
        )
        social_reward_detachment_profile = self._clip01(
            self._mean(
                [
                    nucleus_accumbens_social_reward_deficit,
                    limited_positive_affect,
                    social_withdrawal,
                ]
            )
        )
        developmental_trauma_embedding_profile = self._clip01(
            self._mean(
                [
                    epigenetic_embedding_of_adversity,
                    altered_neurodevelopmental_architecture,
                    hpa_axis_dysregulation,
                ]
            )
        )

        phenotypes = pd.Series(
            {
                "emotionally_withdrawn_attachment_profile": emotionally_withdrawn_attachment_profile,
                "threat_hyperarousal_profile": threat_hyperarousal_profile,
                "social_reward_detachment_profile": social_reward_detachment_profile,
                "developmental_trauma_embedding_profile": developmental_trauma_embedding_profile,
            },
            name="phenotypes",
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = self.atlas.get_map(
                        space=self.assignment_space,
                        parcellation=self.parcellation,
                        maptype="statistical",
                    )
            except Exception:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )

        point = siibra.Point(tuple(float(x) for x in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in (
            "map value",
            "value",
            "correlation",
            "intersection over union",
            "contains",
            "contained",
        ):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(
        self,
        node_key: str,
        maptype: str = "labelled",
        threshold: float = 0.05,
        fetch: bool = True,
    ) -> Any:
        region = self.region_objects.get(node_key)
        if region is None and node_key in self.region_candidates:
            region = self._resolve_region(self.region_candidates[node_key])
            if region is not None:
                self.region_objects[node_key] = region
        if region is None:
            warnings.warn(f"Unknown or unresolved region node: {node_key}")
            return None

        errors: List[Exception] = []

        if hasattr(region, "get_regional_mask"):
            try:
                mask_obj = region.get_regional_mask(space=self.assignment_space, maptype=maptype)
                return mask_obj.fetch() if fetch and hasattr(mask_obj, "fetch") else mask_obj
            except Exception as exc:
                errors.append(exc)

        if hasattr(region, "get_regional_map"):
            try:
                map_obj = region.get_regional_map(self.assignment_space, maptype)
                return map_obj.fetch() if fetch and hasattr(map_obj, "fetch") else map_obj
            except Exception as exc:
                errors.append(exc)

        if hasattr(region, "fetch_regional_map"):
            try:
                return region.fetch_regional_map(
                    space=self.assignment_space,
                    maptype=maptype,
                    threshold=threshold,
                )
            except Exception as exc:
                errors.append(exc)

        if errors:
            warnings.warn(f"Could not fetch mask for '{node_key}': {errors[-1]}")
        return None


if __name__ == "__main__":
    model = ReactiveAttachmentDisorderModel()
    bundle = model.build(connectivity_rows=10)

    print("\n=== Nodes ===")
    print(
        bundle["nodes"][
            [
                "key",
                "node_type",
                "atlas_region",
                "centroid_mni",
                "feature_summary",
            ]
        ].to_string(index=False)
    )

    print("\n=== Edge sample ===")
    print(bundle["edges"].head(12).to_string(index=False))

    print("\n=== Resolved regions ===")
    for key, region in bundle["regions"].items():
        print(f"- {key}: {region.name}")

    print("\n=== Example multimodal summaries ===")
    for node_key in ["amygdala", "pfc_control", "nucleus_accumbens_proxy"]:
        receptor_df = bundle["receptors"].get(node_key, pd.DataFrame())
        gene_df = bundle["genes"].get(node_key, pd.DataFrame())
        conn_df = bundle["connectivity_profiles"].get(node_key, pd.DataFrame())
        print(
            f"\n[{node_key}] receptors rows={len(receptor_df)}, "
            f"genes rows={len(gene_df)}, connectivity rows={len(conn_df)}"
        )
        if not gene_df.empty:
            print(gene_df.head(5).to_string(index=False))
        elif not conn_df.empty:
            print(conn_df.head(5).to_string(index=False))

    if not bundle["circuit_connectivity"].empty:
        print("\n=== Circuit connectivity ===")
        print(bundle["circuit_connectivity"].head(10).to_string(index=False))

    sim = model.simulate(
        genetic_epigenetic_vulnerability=0.40,
        early_caregiver_deprivation=0.85,
        maltreatment_trauma_load=0.70,
        repeated_adversity=0.65,
        stable_caregiving_support=0.15,
        developmental_enrichment_support=0.20,
    )

    print("\n=== Simulation: inputs ===")
    print(sim["inputs"].to_string())
    print("\n=== Simulation: latents ===")
    print(sim["latents"].sort_values(ascending=False).to_string())
    print("\n=== Simulation: regional state ===")
    print(sim["regional_state"].sort_values(ascending=False).to_string())
    print("\n=== Simulation: symptoms ===")
    print(sim["symptoms"].sort_values(ascending=False).to_string())
    print("\n=== Simulation: phenotypes ===")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((-24, -4, -18)).head())
