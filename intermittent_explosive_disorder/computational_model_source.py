from __future__ import annotations

"""
Intermittent Explosive Disorder atlas-grounded siibra scaffold.

This script translates a short neurobiological chapter on Intermittent Explosive
Disorder (IED) into a transparent research scaffold. It is intended for
mechanistic exploration and atlas-backed feature querying, not for diagnosis,
treatment selection, or risk prediction in individual patients.

Core chapter logic encoded here:
- serotonergic hypofunction contributes to impulsive aggression,
- stress and HPA-axis dysregulation heighten reactivity,
- repeated outbursts can sensitize the system over time,
- amygdala bottom-up drive can overpower prefrontal top-down control,
- traumatic/frontal injury can worsen self-regulation.
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


IED_GENE_PANEL = [
    "SLC6A4",
    "TPH2",
    "HTR1B",
    "HTR1A",
    "HTR2A",
    "MAOA",
    "COMT",
    "DRD2",
    "SLC6A3",
    "BDNF",
    "NR3C1",
    "FKBP5",
    "CRHR1",
]


class IntermittentExplosiveDisorderModel:
    """
    Atlas-grounded research scaffold for Intermittent Explosive Disorder.

    Notes
    -----
    - The simulator is intentionally simple, normalized, and acyclic.
    - Region mappings are conservative. Where the chapter is anatomically broad,
      the scaffold uses explicit proxies rather than overclaiming precision.
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
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Fo4",
                "Fo3",
                "orbitofrontal",
            ],
            "vmpfc_proxy": [
                "Area p32 (pACC) left",
                "Area s32 (ACC) left",
                "Area 14M",
                "p32",
                "s32",
                "medial orbitofrontal",
                "ventromedial prefrontal",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "amygdala": "Threat-reactive limbic node representing bottom-up emotional drive.",
            "ofc": "Orbitofrontal control node relevant to behavioral inhibition and consequence evaluation.",
            "vmpfc_proxy": "Atlas-resolved proxy for ventromedial prefrontal control over emotional valuation and self-regulation.",
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": "Heritable liability affecting stress reactivity, impulse control, and aggressive temperament.",
            "chronic_stress_load": "Chronic or traumatic stress burden that can dysregulate HPA function and amplify reactivity.",
            "acute_stressor": "Immediate provocation or stressor that can trigger bottom-up escalation.",
            "traumatic_brain_injury_burden": "Frontal injury burden or related structural compromise affecting regulation.",
            "repeated_outburst_history": "Past cycles of tension and aggressive release that may sensitize the system over time.",
            "serotonergic_treatment_support": "Protective serotonergic support, such as SSRI-mediated dampening of impulsive aggression.",
            "recovery_support": "Protective psychosocial structure, treatment adherence, and recovery support.",
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_hypofunction": "Reduced serotonergic inhibitory control over mood, anxiety, and impulse regulation.",
            "hpa_axis_dysregulation": "Stress-system dysregulation increasing reactivity to later provocation.",
            "neurosensitization": "Progressive sensitization from repeated stress, injury, or recurrent outbursts.",
            "reward_impulse_circuit_adaptation": "Lasting adaptation in reward and impulse-control circuits after repeated outbursts.",
            "frontolimbic_disinhibition": "Failure of prefrontal control to contain bottom-up limbic signals.",
        }

        self.symptom_nodes: Dict[str, str] = {
            "impulsive_aggressive_outbursts": "Explosive, disproportionate aggressive episodes.",
            "poor_impulse_control": "Difficulty resisting aggressive impulses once arousal escalates.",
            "emotional_lability": "Rapid, poorly contained swings in affective intensity.",
            "outburst_recurrence": "Chronic relapsing tendency toward repeated outbursts.",
            "interpersonal_conflict": "Family, relationship, and social conflict related to explosive behavior.",
            "social_occupational_impairment": "Functional impairment in work and social roles.",
            "legal_consequence_risk": "Risk of legal or disciplinary consequences secondary to aggressive behavior.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_hypofunction",
                "relation": "raises trait-level vulnerability in inhibitory control systems",
                "ied_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "hpa_axis_dysregulation",
                "relation": "drives stress-system dysregulation and cortisol-related reactivity",
                "ied_change": "increased",
            },
            {
                "source": "acute_stressor",
                "target": "hpa_axis_dysregulation",
                "relation": "acutely pushes the stress system toward hyperreactivity",
                "ied_change": "increased",
            },
            {
                "source": "traumatic_brain_injury_burden",
                "target": "frontolimbic_disinhibition",
                "relation": "disrupts frontal self-regulation and top-down control",
                "ied_change": "increased",
            },
            {
                "source": "repeated_outburst_history",
                "target": "neurosensitization",
                "relation": "sensitizes central stress and aggression mechanisms over time",
                "ied_change": "increased",
            },
            {
                "source": "repeated_outburst_history",
                "target": "reward_impulse_circuit_adaptation",
                "relation": "reinforces recurrent tension-release circuitry",
                "ied_change": "increased",
            },
            {
                "source": "serotonergic_treatment_support",
                "target": "serotonergic_hypofunction",
                "relation": "restores inhibitory serotonergic tone",
                "ied_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "hpa_axis_dysregulation",
                "relation": "buffers stress-system escalation",
                "ied_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "neurosensitization",
                "relation": "reduces reinforcement of chronic relapse dynamics",
                "ied_change": "decreased",
            },
            {
                "source": "serotonergic_hypofunction",
                "target": "frontolimbic_disinhibition",
                "relation": "weakens inhibitory control over impulsive aggressive drives",
                "ied_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "frontolimbic_disinhibition",
                "relation": "amplifies stress-linked dyscontrol within regulation circuits",
                "ied_change": "increased",
            },
            {
                "source": "neurosensitization",
                "target": "frontolimbic_disinhibition",
                "relation": "lowers the threshold for exaggerated aggressive responding",
                "ied_change": "increased",
            },
            {
                "source": "frontolimbic_disinhibition",
                "target": "amygdala",
                "relation": "permits greater bottom-up emotional dominance",
                "ied_change": "increased",
            },
            {
                "source": "frontolimbic_disinhibition",
                "target": "ofc",
                "relation": "erodes orbitofrontal inhibitory control",
                "ied_change": "increased",
            },
            {
                "source": "frontolimbic_disinhibition",
                "target": "vmpfc_proxy",
                "relation": "erodes ventromedial prefrontal control over valuation and restraint",
                "ied_change": "increased",
            },
            {
                "source": "traumatic_brain_injury_burden",
                "target": "ofc",
                "relation": "increases orbitofrontal dysfunction burden",
                "ied_change": "increased",
            },
            {
                "source": "traumatic_brain_injury_burden",
                "target": "vmpfc_proxy",
                "relation": "increases ventromedial prefrontal dysfunction burden",
                "ied_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "amygdala",
                "relation": "heightens threat responsivity and stress-linked emotional drive",
                "ied_change": "increased",
            },
            {
                "source": "neurosensitization",
                "target": "amygdala",
                "relation": "amplifies reactivity to minor provocation",
                "ied_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "impulsive_aggressive_outbursts",
                "relation": "drives bottom-up emotional escalation into explosive aggression",
                "ied_change": "increased",
            },
            {
                "source": "ofc",
                "target": "poor_impulse_control",
                "relation": "orbitofrontal dysfunction impairs restraint and consequence evaluation",
                "ied_change": "increased",
            },
            {
                "source": "vmpfc_proxy",
                "target": "poor_impulse_control",
                "relation": "ventromedial prefrontal dysfunction weakens top-down regulation",
                "ied_change": "increased",
            },
            {
                "source": "serotonergic_hypofunction",
                "target": "poor_impulse_control",
                "relation": "reduces inhibitory control over impulsive action",
                "ied_change": "increased",
            },
            {
                "source": "reward_impulse_circuit_adaptation",
                "target": "outburst_recurrence",
                "relation": "supports chronic relapsing tension-release patterns",
                "ied_change": "increased",
            },
            {
                "source": "impulsive_aggressive_outbursts",
                "target": "interpersonal_conflict",
                "relation": "disrupts family and social relationships",
                "ied_change": "increased",
            },
            {
                "source": "impulsive_aggressive_outbursts",
                "target": "social_occupational_impairment",
                "relation": "undermines work and social functioning",
                "ied_change": "increased",
            },
            {
                "source": "impulsive_aggressive_outbursts",
                "target": "legal_consequence_risk",
                "relation": "raises risk of legal or disciplinary fallout",
                "ied_change": "increased",
            },
            {
                "source": "emotional_lability",
                "target": "interpersonal_conflict",
                "relation": "destabilizes relationships and social interactions",
                "ied_change": "increased",
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
        generic_penalty = 1 if name in {"amygdala", "prefrontal cortex", "orbitofrontal"} else 0
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
            numeric = [IntermittentExplosiveDisorderModel._coerce_scalar(v) for v in value]
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
        gene_panel: Sequence[str] = IED_GENE_PANEL,
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
                "disorder": "Intermittent Explosive Disorder",
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
        genetic_vulnerability: float = 0.35,
        chronic_stress_load: float = 0.35,
        acute_stressor: float = 0.25,
        traumatic_brain_injury_burden: float = 0.0,
        repeated_outburst_history: float = 0.25,
        serotonergic_treatment_support: float = 0.0,
        recovery_support: float = 0.2,
    ) -> Dict[str, pd.Series]:
        """
        Run a transparent normalized IED simulation.

        All inputs are clipped to [0, 1]. Higher values generally represent more
        burden, except the two protective supports.
        """

        gv = self._clip01(genetic_vulnerability)
        csl = self._clip01(chronic_stress_load)
        ast = self._clip01(acute_stressor)
        tbi = self._clip01(traumatic_brain_injury_burden)
        roh = self._clip01(repeated_outburst_history)
        sts = self._clip01(serotonergic_treatment_support)
        rs = self._clip01(recovery_support)

        inputs = pd.Series(
            {
                "genetic_vulnerability": gv,
                "chronic_stress_load": csl,
                "acute_stressor": ast,
                "traumatic_brain_injury_burden": tbi,
                "repeated_outburst_history": roh,
                "serotonergic_treatment_support": sts,
                "recovery_support": rs,
            },
            name="inputs",
        )

        serotonergic_hypofunction = self._clip01(
            0.45 * gv + 0.15 * csl + 0.10 * tbi + 0.10 * roh - 0.35 * sts - 0.10 * rs
        )
        hpa_axis_dysregulation = self._clip01(
            0.35 * csl + 0.25 * ast + 0.10 * gv + 0.10 * roh - 0.20 * rs
        )
        neurosensitization = self._clip01(
            0.35 * roh + 0.25 * csl + 0.20 * tbi + 0.10 * gv - 0.15 * rs
        )
        reward_impulse_circuit_adaptation = self._clip01(
            0.30 * roh + 0.25 * gv + 0.15 * csl + 0.10 * serotonergic_hypofunction - 0.15 * rs
        )
        frontolimbic_disinhibition = self._clip01(
            0.35 * serotonergic_hypofunction
            + 0.25 * hpa_axis_dysregulation
            + 0.20 * tbi
            + 0.15 * neurosensitization
            - 0.15 * sts
            - 0.10 * rs
        )

        latents = pd.Series(
            {
                "serotonergic_hypofunction": serotonergic_hypofunction,
                "hpa_axis_dysregulation": hpa_axis_dysregulation,
                "neurosensitization": neurosensitization,
                "reward_impulse_circuit_adaptation": reward_impulse_circuit_adaptation,
                "frontolimbic_disinhibition": frontolimbic_disinhibition,
            },
            name="latents",
        )

        amygdala_reactivity = self._clip01(
            0.45 * hpa_axis_dysregulation
            + 0.25 * neurosensitization
            + 0.20 * serotonergic_hypofunction
            + 0.10 * ast
        )
        ofc_dyscontrol = self._clip01(
            0.35 * frontolimbic_disinhibition
            + 0.25 * tbi
            + 0.15 * gv
            + 0.10 * neurosensitization
            - 0.15 * sts
            - 0.15 * rs
        )
        vmpfc_proxy_dyscontrol = self._clip01(
            0.35 * frontolimbic_disinhibition
            + 0.20 * tbi
            + 0.15 * hpa_axis_dysregulation
            + 0.15 * neurosensitization
            - 0.10 * sts
            - 0.15 * rs
        )
        pfc_top_down_failure = self._clip01(0.55 * ofc_dyscontrol + 0.45 * vmpfc_proxy_dyscontrol)

        regional_state = pd.Series(
            {
                "amygdala_reactivity": amygdala_reactivity,
                "ofc_dyscontrol": ofc_dyscontrol,
                "vmpfc_proxy_dyscontrol": vmpfc_proxy_dyscontrol,
                "pfc_top_down_failure": pfc_top_down_failure,
            },
            name="regional_state",
        )

        impulsive_aggressive_outbursts = self._clip01(
            0.40 * amygdala_reactivity
            + 0.35 * pfc_top_down_failure
            + 0.15 * reward_impulse_circuit_adaptation
            + 0.10 * ast
        )
        poor_impulse_control = self._clip01(
            0.45 * pfc_top_down_failure
            + 0.25 * serotonergic_hypofunction
            + 0.15 * tbi
            + 0.15 * reward_impulse_circuit_adaptation
        )
        emotional_lability = self._clip01(
            0.45 * amygdala_reactivity
            + 0.25 * hpa_axis_dysregulation
            + 0.20 * neurosensitization
            + 0.10 * tbi
        )
        outburst_recurrence = self._clip01(
            0.35 * impulsive_aggressive_outbursts
            + 0.30 * neurosensitization
            + 0.20 * reward_impulse_circuit_adaptation
            + 0.15 * csl
        )
        interpersonal_conflict = self._clip01(
            0.45 * impulsive_aggressive_outbursts
            + 0.20 * emotional_lability
            + 0.20 * poor_impulse_control
            + 0.15 * outburst_recurrence
        )
        social_occupational_impairment = self._clip01(
            0.40 * impulsive_aggressive_outbursts
            + 0.20 * poor_impulse_control
            + 0.20 * interpersonal_conflict
            + 0.20 * outburst_recurrence
        )
        legal_consequence_risk = self._clip01(
            0.55 * impulsive_aggressive_outbursts
            + 0.20 * poor_impulse_control
            + 0.15 * tbi
            + 0.10 * emotional_lability
        )

        symptoms = pd.Series(
            {
                "impulsive_aggressive_outbursts": impulsive_aggressive_outbursts,
                "poor_impulse_control": poor_impulse_control,
                "emotional_lability": emotional_lability,
                "outburst_recurrence": outburst_recurrence,
                "interpersonal_conflict": interpersonal_conflict,
                "social_occupational_impairment": social_occupational_impairment,
                "legal_consequence_risk": legal_consequence_risk,
            },
            name="symptoms",
        )

        phenotypes = pd.Series(
            {
                "explosive_outburst_profile": self._clip01(
                    self._mean(
                        [
                            impulsive_aggressive_outbursts,
                            emotional_lability,
                            poor_impulse_control,
                        ]
                    )
                ),
                "stress_sensitized_profile": self._clip01(
                    self._mean(
                        [
                            hpa_axis_dysregulation,
                            neurosensitization,
                            amygdala_reactivity,
                            outburst_recurrence,
                        ]
                    )
                ),
                "injury_amplified_profile": self._clip01(
                    self._mean(
                        [
                            tbi,
                            ofc_dyscontrol,
                            vmpfc_proxy_dyscontrol,
                            poor_impulse_control,
                        ]
                    )
                ),
                "functional_impairment_profile": self._clip01(
                    self._mean(
                        [
                            interpersonal_conflict,
                            social_occupational_impairment,
                            legal_consequence_risk,
                        ]
                    )
                ),
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
    model = IntermittentExplosiveDisorderModel()
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
    for node_key in ["amygdala", "ofc", "vmpfc_proxy"]:
        receptor_df = bundle["receptors"].get(node_key, pd.DataFrame())
        gene_df = bundle["genes"].get(node_key, pd.DataFrame())
        conn_df = bundle["connectivity_profiles"].get(node_key, pd.DataFrame())
        print(f"\n[{node_key}] receptors rows={len(receptor_df)}, genes rows={len(gene_df)}, connectivity rows={len(conn_df)}")
        if not gene_df.empty:
            print(gene_df.head(5).to_string(index=False))
        if not conn_df.empty:
            print(conn_df.head(5).to_string(index=False))

    print("\n=== Circuit connectivity ===")
    circuit_df = bundle["circuit_connectivity"]
    if circuit_df.empty:
        print("No pairwise circuit connectivity matrix available in this environment.")
    else:
        print(circuit_df.to_string(index=False))

    print("\n=== Example simulation ===")
    sim = model.simulate(
        genetic_vulnerability=0.70,
        chronic_stress_load=0.80,
        acute_stressor=0.65,
        traumatic_brain_injury_burden=0.20,
        repeated_outburst_history=0.75,
        serotonergic_treatment_support=0.30,
        recovery_support=0.20,
    )
    for name, series in sim.items():
        print(f"\n{name.upper()}")
        print(series.to_string())

    # Optional helpers for interactive use:
    # print(model.suggest_regions("orbitofrontal").head(10).to_string(index=False))
    # print(model.assign_mni_point((-6.0, 42.0, -10.0)).head(10).to_string(index=False))
    # mask_img = model.region_mask("amygdala")
