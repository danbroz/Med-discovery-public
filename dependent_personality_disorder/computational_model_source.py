from __future__ import annotations

"""
Dependent Personality Disorder siibra scaffold.

This script translates a chapter-level biological summary of Dependent Personality
Disorder (DPD) into a transparent, atlas-grounded mechanistic scaffold using
siibra. It is intended for research exploration, educational use, and model
refinement. It is not a validated disease model, and it must not be used for
clinical diagnosis or treatment decisions.

Design notes
------------
- The chapter is systems-level rather than parcel-precise, so several nodes are
  represented as clearly labeled proxies (for example ventral striatum and
  vmPFC) instead of forcing false anatomical precision.
- Higher values in ``regional_state`` reflect *disorder-related circuit burden or
  dysregulation*, not raw BOLD activation.
- The simulator is intentionally acyclic and normalized to 0..1 to keep the
  direction of causality transparent: inputs -> latent biology -> regional state
  -> symptoms -> phenotype summaries.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import siibra
except ImportError as exc:  # pragma: no cover - import guard for environments without siibra
    raise ImportError(
        "This scaffold requires siibra-python. Install it in your environment "
        "before running this script."
    ) from exc


DEFAULT_DEPENDENT_PERSONALITY_GENE_PANEL: List[str] = [
    "OXTR",
    "AVPR1A",
    "SLC6A4",
    "HTR1A",
    "HTR2A",
    "DRD2",
    "DRD4",
    "SLC6A3",
    "COMT",
    "BDNF",
    "FKBP5",
    "NR3C1",
    "CRHR1",
    "MAOA",
]


class DependentPersonalityDisorderModel:
    """
    Atlas-grounded research scaffold for Dependent Personality Disorder.

    The model captures chapter-level claims linking heritable anxious/inhibited
    temperament, attachment adversity, trauma, social-threat processing,
    reward-based reassurance seeking, and impaired autonomous decision control.
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

        # Atlas-backed nodes and conservative proxies.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "Amygdala left",
                "amygdala",
            ],
            "dlpfc": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "Area 9 left",
                "Area 9/46d",
                "Area 9/46v",
                "Area 46",
                "dorsolateral prefrontal cortex left",
                "dorsolateral prefrontal cortex",
                "middle frontal gyrus left",
                "dlpfc",
            ],
            "insula": [
                "Area Id1 left",
                "Area Id2 left",
                "Area Ig2 left",
                "insula left",
                "insula",
            ],
            "acc": [
                "Area p24ab left",
                "Area p24pr left",
                "Area a24pr left",
                "anterior cingulate cortex left",
                "ACC",
            ],
            "vmpfc_proxy": [
                "Area 14m left",
                "Area 32 left",
                "Area s32 left",
                "ventromedial prefrontal cortex left",
                "vmPFC",
            ],
            "ventral_striatum_proxy": [
                "Nucleus accumbens left",
                "Acb left",
                "ventral striatum left",
                "accumbens",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_temperamental_liability": (
                "Polygenic and temperamental loading for timidity, harm avoidance, "
                "anxiety proneness, and rejection sensitivity"
            ),
            "early_attachment_adversity": (
                "Uncaring, inconsistent, or dependency-shaping attachment environment"
            ),
            "childhood_trauma": (
                "Abusive or otherwise sensitizing early adverse experiences"
            ),
            "rejection_cue_load": (
                "Current criticism, disapproval, abandonment signals, or perceived social threat"
            ),
            "supportive_relationship_context": (
                "Protective secure-base experiences and stable relational validation"
            ),
            "autonomy_skills_support": (
                "Therapy, coaching, or structured support that increases independent decision making"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "inhibited_anxious_temperament": (
                "Early-emerging inhibited or anxious style that amplifies threat sensitivity"
            ),
            "attachment_insecurity": (
                "Persistent expectation that security and regulation depend on another person"
            ),
            "negative_self_schema": (
                "Low self-worth and fragile self-representation requiring external support"
            ),
            "amygdala_threat_bias": (
                "Hyper-reactive social-threat processing, especially for disapproval and abandonment cues"
            ),
            "interoceptive_social_alarm": (
                "Body-state alarm and heightened felt urgency during interpersonal threat"
            ),
            "dopaminergic_reassurance_craving": (
                "Reward-linked drive to obtain reassurance and approval as transient relief"
            ),
            "frontolimbic_regulatory_failure": (
                "Reduced top-down control over fear and attachment-driven behavior"
            ),
            "dependency_self_representation": (
                "Self-definition shifted toward helplessness and reliance on others"
            ),
            "impaired_autonomous_decision_control": (
                "Underpowered independent planning, choice, and self-directed problem solving"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "fear_of_abandonment": "Marked distress about rejection, separation, or loss of support",
            "reassurance_seeking": "Compulsive checking, validation seeking, and need for approval",
            "indecisiveness": "Difficulty making ordinary decisions without excessive advice or reassurance",
            "submissive_help_seeking": "Reliance on others to assume responsibility and guide action",
            "avoidance_of_independence": "Avoidance of autonomy because it feels unsafe or overwhelming",
            "low_self_efficacy": "Persistent sense of personal inadequacy and inability to cope alone",
            "clingy_relationship_pattern": "Over-attachment and relationship-preserving behavior driven by fear",
        }

        self.edge_table: List[Dict[str, Any]] = [
            {
                "source": "genetic_temperamental_liability",
                "target": "inhibited_anxious_temperament",
                "relation": "raises early inhibition, timidity, and rejection sensitivity",
                "dependent_pd_change": "increased",
                "weight": 0.42,
            },
            {
                "source": "early_attachment_adversity",
                "target": "attachment_insecurity",
                "relation": "promotes insecure attachment and reliance on external regulation",
                "dependent_pd_change": "increased",
                "weight": 0.42,
            },
            {
                "source": "early_attachment_adversity",
                "target": "negative_self_schema",
                "relation": "weakens self-worth and stable self-representation",
                "dependent_pd_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "childhood_trauma",
                "target": "amygdala_threat_bias",
                "relation": "sensitizes threat detection and abandonment alarm",
                "dependent_pd_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "childhood_trauma",
                "target": "negative_self_schema",
                "relation": "consolidates helpless self-appraisals",
                "dependent_pd_change": "increased",
                "weight": 0.33,
            },
            {
                "source": "rejection_cue_load",
                "target": "amygdala_threat_bias",
                "relation": "acutely drives reactivity to criticism and disapproval",
                "dependent_pd_change": "increased",
                "weight": 0.28,
            },
            {
                "source": "inhibited_anxious_temperament",
                "target": "attachment_insecurity",
                "relation": "biases the person toward seeking external safety",
                "dependent_pd_change": "increased",
                "weight": 0.18,
            },
            {
                "source": "inhibited_anxious_temperament",
                "target": "amygdala_threat_bias",
                "relation": "primes heightened vigilance for interpersonal threat",
                "dependent_pd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "amygdala",
                "target": "amygdala_threat_bias",
                "relation": "threat-detection node likely shows exaggerated response to rejection cues",
                "dependent_pd_change": "hyper-reactive",
                "weight": 0.30,
            },
            {
                "source": "insula",
                "target": "interoceptive_social_alarm",
                "relation": "amplifies felt social alarm and internal distress signals",
                "dependent_pd_change": "increased",
                "weight": 0.26,
            },
            {
                "source": "acc",
                "target": "frontolimbic_regulatory_failure",
                "relation": "weaker conflict monitoring and emotion regulation worsens dependence",
                "dependent_pd_change": "increased dysfunction",
                "weight": 0.25,
            },
            {
                "source": "dlpfc",
                "target": "impaired_autonomous_decision_control",
                "relation": "reduced executive control weakens independent choice and planning",
                "dependent_pd_change": "increased dysfunction",
                "weight": 0.34,
            },
            {
                "source": "ventral_striatum_proxy",
                "target": "dopaminergic_reassurance_craving",
                "relation": "social approval becomes a potent reward that reinforces reassurance seeking",
                "dependent_pd_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "vmpfc_proxy",
                "target": "dependency_self_representation",
                "relation": "self-valuation becomes overly tied to external support and approval",
                "dependent_pd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "attachment_insecurity",
                "target": "dependency_self_representation",
                "relation": "organizes self-other expectations around dependence",
                "dependent_pd_change": "increased",
                "weight": 0.32,
            },
            {
                "source": "negative_self_schema",
                "target": "dependency_self_representation",
                "relation": "low self-worth is stabilized by defining the self as unable to cope alone",
                "dependent_pd_change": "increased",
                "weight": 0.28,
            },
            {
                "source": "negative_self_schema",
                "target": "dopaminergic_reassurance_craving",
                "relation": "external validation is recruited to repair fragile self-worth",
                "dependent_pd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "amygdala_threat_bias",
                "target": "interoceptive_social_alarm",
                "relation": "threat bias spills into felt urgency and alarm",
                "dependent_pd_change": "increased",
                "weight": 0.36,
            },
            {
                "source": "amygdala_threat_bias",
                "target": "frontolimbic_regulatory_failure",
                "relation": "fear-driven bottom-up pressure overwhelms control systems",
                "dependent_pd_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "frontolimbic_regulatory_failure",
                "target": "impaired_autonomous_decision_control",
                "relation": "poor top-down control reduces self-directed problem solving",
                "dependent_pd_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "attachment_insecurity",
                "target": "fear_of_abandonment",
                "relation": "creates persistent separation and rejection fear",
                "dependent_pd_change": "increased",
                "weight": 0.38,
            },
            {
                "source": "interoceptive_social_alarm",
                "target": "fear_of_abandonment",
                "relation": "bodily alarm escalates perceived threat of losing support",
                "dependent_pd_change": "increased",
                "weight": 0.26,
            },
            {
                "source": "dopaminergic_reassurance_craving",
                "target": "reassurance_seeking",
                "relation": "approval and validation are pursued as rewarding relief",
                "dependent_pd_change": "increased",
                "weight": 0.43,
            },
            {
                "source": "dependency_self_representation",
                "target": "submissive_help_seeking",
                "relation": "the self is cast as needing others to take responsibility",
                "dependent_pd_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "impaired_autonomous_decision_control",
                "target": "indecisiveness",
                "relation": "reduces capacity to decide without reassurance",
                "dependent_pd_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "impaired_autonomous_decision_control",
                "target": "avoidance_of_independence",
                "relation": "autonomous action feels unsafe and effortful",
                "dependent_pd_change": "increased",
                "weight": 0.38,
            },
            {
                "source": "negative_self_schema",
                "target": "low_self_efficacy",
                "relation": "consolidates beliefs about being unable to cope alone",
                "dependent_pd_change": "increased",
                "weight": 0.46,
            },
            {
                "source": "fear_of_abandonment",
                "target": "clingy_relationship_pattern",
                "relation": "relationship-preserving behavior is driven by fear of loss",
                "dependent_pd_change": "increased",
                "weight": 0.32,
            },
            {
                "source": "reassurance_seeking",
                "target": "clingy_relationship_pattern",
                "relation": "repeated validation-seeking reinforces over-attachment",
                "dependent_pd_change": "increased",
                "weight": 0.28,
            },
            {
                "source": "supportive_relationship_context",
                "target": "attachment_insecurity",
                "relation": "secure-base experiences soften dependency-driven insecurity",
                "dependent_pd_change": "decreased",
                "weight": -0.28,
            },
            {
                "source": "supportive_relationship_context",
                "target": "amygdala_threat_bias",
                "relation": "predictable safety cues dampen social-threat reactivity",
                "dependent_pd_change": "decreased",
                "weight": -0.18,
            },
            {
                "source": "autonomy_skills_support",
                "target": "impaired_autonomous_decision_control",
                "relation": "structured practice improves independent planning and choice",
                "dependent_pd_change": "decreased",
                "weight": -0.34,
            },
            {
                "source": "autonomy_skills_support",
                "target": "avoidance_of_independence",
                "relation": "graded mastery reduces avoidance of self-directed action",
                "dependent_pd_change": "decreased",
                "weight": -0.20,
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

    @staticmethod
    def _coerce_xyz(point_like: Any) -> Optional[Tuple[float, float, float]]:
        if point_like is None:
            return None
        if hasattr(point_like, "coordinate"):
            coords = getattr(point_like, "coordinate")
        elif isinstance(point_like, (list, tuple)):
            coords = point_like
        else:
            try:
                coords = tuple(point_like)
            except Exception:
                return None
        try:
            xyz = tuple(float(v) for v in coords[:3])
        except Exception:
            return None
        return xyz if len(xyz) == 3 else None

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
        proxy_penalty = 1 if "proxy" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "insula",
            "prefrontal cortex",
        } else 0
        specificity_penalty = 1 if len(name) < 10 else 0
        return (left_bonus, proxy_penalty, generic_penalty, specificity_penalty)

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
            row_key = (
                self._name_of(region),
                getattr(region, "identifier", None),
                getattr(getattr(region, "parcellation", None), "name", ""),
            )
            if row_key in seen:
                continue
            seen.add(row_key)
            rows.append(
                {
                    "name": row_key[0],
                    "identifier": row_key[1],
                    "parcellation": row_key[2],
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
        centroid_xyz = self._coerce_xyz(getattr(main, "centroid", None))
        volume_mm3: Optional[float]
        try:
            volume_mm3 = float(getattr(main, "volume", float("nan")))
        except Exception:
            volume_mm3 = None
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
                grouped = (
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
                return grouped
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

        # Common case: compound feature has a .data dataframe.
        for attr in ("data", "matrix"):
            try:
                candidate = getattr(compound, attr, None)
                if isinstance(candidate, pd.DataFrame):
                    self._connectivity_matrix = candidate.copy()
                    return self._connectivity_matrix
            except Exception:
                pass

        # Fallback: inspect first contained element if accessible.
        for idx in (0,):
            try:
                element = compound[idx]
                candidate = getattr(element, "data", None)
                if isinstance(candidate, pd.DataFrame):
                    self._connectivity_matrix = candidate.copy()
                    return self._connectivity_matrix
            except Exception:
                continue

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = region.name.lower()
        region_id = str(getattr(region, "identifier", "")).lower()

        def as_name(label: Any) -> str:
            return self._name_of(label).lower()

        def as_id(label: Any) -> str:
            return str(getattr(label, "identifier", "")).lower()

        # Exact name.
        for label in labels:
            if as_name(label) == region_name:
                return label

        # Exact identifier.
        for label in labels:
            label_id = as_id(label)
            if region_id and (region_id == label_id or region_id == as_name(label)):
                return label

        # Fuzzy containment using names and identifiers.
        fuzzy = [
            label
            for label in labels
            if (
                region_name in as_name(label)
                or as_name(label) in region_name
                or (region_id and region_id in as_name(label))
                or (region_id and as_id(label) and (region_id in as_id(label) or as_id(label) in region_id))
            )
        ]
        if fuzzy:
            return fuzzy[0]
        return None

    @staticmethod
    def _numeric_scalar(value: Any) -> Optional[float]:
        if isinstance(value, pd.DataFrame):
            numeric = value.apply(pd.to_numeric, errors="coerce").stack().dropna()
            return float(numeric.mean()) if not numeric.empty else None
        if isinstance(value, pd.Series):
            numeric = pd.to_numeric(value, errors="coerce").dropna()
            return float(numeric.mean()) if not numeric.empty else None
        try:
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _connectivity_series(selection: Any, aggregate_axis: int) -> pd.Series:
        if isinstance(selection, pd.DataFrame):
            numeric = selection.apply(pd.to_numeric, errors="coerce")
            series = numeric.mean(axis=aggregate_axis, skipna=True)
        elif isinstance(selection, pd.Series):
            series = pd.to_numeric(selection, errors="coerce")
        else:
            series = pd.Series(dtype=float)
        return series.dropna()

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        row_label = self._match_region_label(list(matrix.index), region)
        if row_label is not None:
            try:
                series = self._connectivity_series(matrix.loc[row_label], aggregate_axis=0)
                if not series.empty:
                    df = series.sort_values(ascending=False).reset_index()
                    df.columns = ["connected_region", "value"]
                    df["connected_region"] = df["connected_region"].map(self._name_of)
                    df = df[df["connected_region"] != region.name].head(max_rows)
                    return df.reset_index(drop=True)
            except Exception:
                pass

        col_label = self._match_region_label(list(matrix.columns), region)
        if col_label is not None:
            try:
                series = self._connectivity_series(matrix[col_label], aggregate_axis=1)
                if not series.empty:
                    df = series.sort_values(ascending=False).reset_index()
                    df.columns = ["connected_region", "value"]
                    df["connected_region"] = df["connected_region"].map(self._name_of)
                    df = df[df["connected_region"] != region.name].head(max_rows)
                    return df.reset_index(drop=True)
            except Exception:
                pass

        return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        resolved = {k: r for k, r in self.region_objects.items() if r is not None}
        if len(resolved) < 2:
            return pd.DataFrame()

        row_labels = list(matrix.index)
        col_labels = list(matrix.columns)
        matched_rows = {k: self._match_region_label(row_labels, r) for k, r in resolved.items()}
        matched_cols = {k: self._match_region_label(col_labels, r) for k, r in resolved.items()}

        rows: List[Dict[str, Any]] = []
        keys = list(resolved.keys())
        for i, src_key in enumerate(keys):
            for tgt_key in keys[i + 1 :]:
                vals: List[float] = []
                src_row = matched_rows.get(src_key)
                tgt_row = matched_rows.get(tgt_key)
                src_col = matched_cols.get(src_key)
                tgt_col = matched_cols.get(tgt_key)

                try:
                    if src_row is not None and tgt_col is not None:
                        scalar = self._numeric_scalar(matrix.loc[src_row, tgt_col])
                        if scalar is not None:
                            vals.append(scalar)
                except Exception:
                    pass
                try:
                    if tgt_row is not None and src_col is not None:
                        scalar = self._numeric_scalar(matrix.loc[tgt_row, src_col])
                        if scalar is not None:
                            vals.append(scalar)
                except Exception:
                    pass

                if not vals:
                    continue

                mean_value = float(np.nanmean(vals))
                if not np.isfinite(mean_value):
                    continue

                rows.append(
                    {
                        "source_key": src_key,
                        "source_region": resolved[src_key].name,
                        "target_key": tgt_key,
                        "target_region": resolved[tgt_key].name,
                        "mean_streamline_count": round(mean_value, 6),
                    }
                )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "source_key",
                    "source_region",
                    "target_key",
                    "target_region",
                    "mean_streamline_count",
                ]
            )

        return pd.DataFrame(rows).sort_values("mean_streamline_count", ascending=False).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_DEPENDENT_PERSONALITY_GENE_PANEL,
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
                        "node_type": "region_proxy" if key.endswith("_proxy") else "region",
                        "description": "Atlas-backed circuit node or proxy (unresolved in this environment)",
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
                    "node_type": "region_proxy" if key.endswith("_proxy") else "region",
                    "description": "Atlas-backed circuit node" if not key.endswith("_proxy") else "Atlas-matched proxy node",
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

    def region_mask(self, node_key: str, maptype: str = "labelled") -> Any:
        region = self.region_objects.get(node_key)
        if region is None:
            raise KeyError(f"No resolved region is available for node '{node_key}'.")
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype=maptype)
        except Exception:
            return region.get_regional_mask(space=self.space, maptype=maptype)

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if self._pmap is None:
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
        return assignments.reset_index(drop=True)

    def simulate(
        self,
        genetic_temperamental_liability: float = 0.50,
        early_attachment_adversity: float = 0.50,
        childhood_trauma: float = 0.50,
        rejection_cue_load: float = 0.50,
        supportive_relationship_context: float = 0.30,
        autonomy_skills_support: float = 0.30,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized one-pass simulation.

        Inputs are clipped to [0, 1]. Larger values indicate greater load except
        for the two explicitly protective supports.
        """

        inputs = pd.Series(
            {
                "genetic_temperamental_liability": self._clip01(genetic_temperamental_liability),
                "early_attachment_adversity": self._clip01(early_attachment_adversity),
                "childhood_trauma": self._clip01(childhood_trauma),
                "rejection_cue_load": self._clip01(rejection_cue_load),
                "supportive_relationship_context": self._clip01(supportive_relationship_context),
                "autonomy_skills_support": self._clip01(autonomy_skills_support),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["inhibited_anxious_temperament"] = self._clip01(
            0.65 * inputs["genetic_temperamental_liability"]
            + 0.20 * inputs["childhood_trauma"]
            + 0.10 * inputs["early_attachment_adversity"]
        )
        latents["attachment_insecurity"] = self._clip01(
            0.42 * inputs["early_attachment_adversity"]
            + 0.20 * inputs["childhood_trauma"]
            + 0.20 * latents["inhibited_anxious_temperament"]
            + 0.12 * inputs["rejection_cue_load"]
            - 0.28 * inputs["supportive_relationship_context"]
            - 0.08 * inputs["autonomy_skills_support"]
        )
        latents["negative_self_schema"] = self._clip01(
            0.38 * inputs["early_attachment_adversity"]
            + 0.30 * inputs["childhood_trauma"]
            + 0.17 * inputs["rejection_cue_load"]
            + 0.08 * latents["attachment_insecurity"]
            - 0.20 * inputs["supportive_relationship_context"]
        )
        latents["amygdala_threat_bias"] = self._clip01(
            0.33 * inputs["childhood_trauma"]
            + 0.24 * inputs["rejection_cue_load"]
            + 0.24 * latents["inhibited_anxious_temperament"]
            + 0.13 * latents["attachment_insecurity"]
            - 0.18 * inputs["supportive_relationship_context"]
        )
        latents["interoceptive_social_alarm"] = self._clip01(
            0.45 * latents["amygdala_threat_bias"]
            + 0.28 * latents["attachment_insecurity"]
            + 0.12 * inputs["rejection_cue_load"]
            - 0.10 * inputs["supportive_relationship_context"]
        )
        latents["dopaminergic_reassurance_craving"] = self._clip01(
            0.34 * latents["negative_self_schema"]
            + 0.28 * latents["attachment_insecurity"]
            + 0.18 * inputs["rejection_cue_load"]
            + 0.10 * latents["amygdala_threat_bias"]
            - 0.12 * inputs["supportive_relationship_context"]
        )
        latents["frontolimbic_regulatory_failure"] = self._clip01(
            0.38 * latents["amygdala_threat_bias"]
            + 0.22 * latents["negative_self_schema"]
            + 0.14 * inputs["rejection_cue_load"]
            + 0.10 * latents["interoceptive_social_alarm"]
            - 0.24 * inputs["autonomy_skills_support"]
            - 0.08 * inputs["supportive_relationship_context"]
        )
        latents["dependency_self_representation"] = self._clip01(
            0.34 * latents["attachment_insecurity"]
            + 0.28 * latents["negative_self_schema"]
            + 0.18 * latents["dopaminergic_reassurance_craving"]
            + 0.10 * latents["frontolimbic_regulatory_failure"]
            - 0.14 * inputs["supportive_relationship_context"]
            - 0.08 * inputs["autonomy_skills_support"]
        )
        latents["impaired_autonomous_decision_control"] = self._clip01(
            0.42 * latents["frontolimbic_regulatory_failure"]
            + 0.25 * latents["dependency_self_representation"]
            + 0.18 * latents["amygdala_threat_bias"]
            + 0.05 * latents["negative_self_schema"]
            - 0.34 * inputs["autonomy_skills_support"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["amygdala"] = self._clip01(
            0.82 * latents["amygdala_threat_bias"] + 0.10 * inputs["rejection_cue_load"]
        )
        regional_state["insula"] = self._clip01(
            0.52 * latents["interoceptive_social_alarm"]
            + 0.22 * latents["amygdala_threat_bias"]
            + 0.12 * inputs["rejection_cue_load"]
        )
        regional_state["acc"] = self._clip01(
            0.42 * latents["frontolimbic_regulatory_failure"]
            + 0.22 * latents["amygdala_threat_bias"]
            + 0.16 * latents["interoceptive_social_alarm"]
        )
        regional_state["dlpfc"] = self._clip01(
            0.46 * latents["impaired_autonomous_decision_control"]
            + 0.24 * latents["frontolimbic_regulatory_failure"]
            + 0.10 * inputs["rejection_cue_load"]
            - 0.20 * inputs["autonomy_skills_support"]
        )
        regional_state["ventral_striatum_proxy"] = self._clip01(
            0.54 * latents["dopaminergic_reassurance_craving"]
            + 0.18 * latents["attachment_insecurity"]
            + 0.12 * inputs["rejection_cue_load"]
            - 0.10 * inputs["supportive_relationship_context"]
        )
        regional_state["vmpfc_proxy"] = self._clip01(
            0.42 * latents["dependency_self_representation"]
            + 0.28 * latents["negative_self_schema"]
            + 0.10 * latents["attachment_insecurity"]
            - 0.10 * inputs["supportive_relationship_context"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["fear_of_abandonment"] = self._clip01(
            0.44 * latents["attachment_insecurity"]
            + 0.24 * latents["amygdala_threat_bias"]
            + 0.18 * latents["interoceptive_social_alarm"]
            + 0.08 * inputs["rejection_cue_load"]
            - 0.16 * inputs["supportive_relationship_context"]
        )
        symptoms["reassurance_seeking"] = self._clip01(
            0.40 * latents["dopaminergic_reassurance_craving"]
            + 0.22 * latents["dependency_self_representation"]
            + 0.18 * latents["negative_self_schema"]
            + 0.10 * symptoms["fear_of_abandonment"]
            - 0.14 * inputs["supportive_relationship_context"]
        )
        symptoms["indecisiveness"] = self._clip01(
            0.50 * latents["impaired_autonomous_decision_control"]
            + 0.18 * latents["negative_self_schema"]
            + 0.12 * latents["amygdala_threat_bias"]
            - 0.24 * inputs["autonomy_skills_support"]
        )
        symptoms["submissive_help_seeking"] = self._clip01(
            0.34 * latents["dependency_self_representation"]
            + 0.24 * latents["impaired_autonomous_decision_control"]
            + 0.18 * latents["attachment_insecurity"]
            + 0.12 * latents["dopaminergic_reassurance_craving"]
            - 0.10 * inputs["autonomy_skills_support"]
        )
        symptoms["avoidance_of_independence"] = self._clip01(
            0.34 * latents["impaired_autonomous_decision_control"]
            + 0.28 * symptoms["fear_of_abandonment"]
            + 0.14 * latents["dependency_self_representation"]
            - 0.18 * inputs["autonomy_skills_support"]
        )
        symptoms["low_self_efficacy"] = self._clip01(
            0.46 * latents["negative_self_schema"]
            + 0.20 * latents["dependency_self_representation"]
            + 0.16 * latents["impaired_autonomous_decision_control"]
            - 0.24 * inputs["autonomy_skills_support"]
            - 0.08 * inputs["supportive_relationship_context"]
        )
        symptoms["clingy_relationship_pattern"] = self._clip01(
            0.32 * symptoms["fear_of_abandonment"]
            + 0.26 * symptoms["reassurance_seeking"]
            + 0.20 * symptoms["submissive_help_seeking"]
            + 0.10 * latents["attachment_insecurity"]
            - 0.12 * inputs["supportive_relationship_context"]
        )

        phenotypes = pd.Series(dtype=float, name="phenotypes")
        phenotypes["dependent_profile"] = self._clip01(
            float(
                np.mean(
                    [
                        symptoms["fear_of_abandonment"],
                        symptoms["reassurance_seeking"],
                        symptoms["submissive_help_seeking"],
                        symptoms["avoidance_of_independence"],
                    ]
                )
            )
        )
        phenotypes["autonomy_collapse_risk"] = self._clip01(
            float(
                np.mean(
                    [
                        symptoms["indecisiveness"],
                        symptoms["low_self_efficacy"],
                        latents["impaired_autonomous_decision_control"],
                    ]
                )
            )
        )
        phenotypes["social_threat_sensitivity"] = self._clip01(
            float(
                np.mean(
                    [
                        latents["amygdala_threat_bias"],
                        latents["interoceptive_social_alarm"],
                        symptoms["fear_of_abandonment"],
                    ]
                )
            )
        )
        phenotypes["validation_craving_profile"] = self._clip01(
            float(
                np.mean(
                    [
                        latents["dopaminergic_reassurance_craving"],
                        symptoms["reassurance_seeking"],
                        regional_state["ventral_striatum_proxy"],
                        latents["dependency_self_representation"],
                    ]
                )
            )
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    model = DependentPersonalityDisorderModel()
    bundle = model.build()

    print("\n=== Nodes ===")
    print(
        bundle["nodes"][
            ["key", "node_type", "atlas_region", "centroid_mni", "feature_summary"]
        ].to_string(index=False)
    )

    print("\n=== Edges ===")
    print(
        bundle["edges"][["source", "target", "relation", "dependent_pd_change", "weight"]]
        .to_string(index=False)
    )

    if not bundle["circuit_connectivity"].empty:
        print("\n=== Circuit Connectivity (resolved region pairs) ===")
        print(bundle["circuit_connectivity"].head(15).to_string(index=False))

    for region_key in ("amygdala", "dlpfc", "ventral_striatum_proxy"):
        receptor_df = bundle["receptors"].get(region_key, pd.DataFrame())
        gene_df = bundle["genes"].get(region_key, pd.DataFrame())
        conn_df = bundle["connectivity_profiles"].get(region_key, pd.DataFrame())

        if not receptor_df.empty:
            print(f"\n=== Receptor fingerprint preview: {region_key} ===")
            print(receptor_df.head(10).to_string(index=False))

        if not gene_df.empty:
            print(f"\n=== Gene expression preview: {region_key} ===")
            print(gene_df.head(10).to_string(index=False))

        if not conn_df.empty:
            print(f"\n=== Connectivity profile preview: {region_key} ===")
            print(conn_df.head(10).to_string(index=False))

    sim = model.simulate(
        genetic_temperamental_liability=0.70,
        early_attachment_adversity=0.80,
        childhood_trauma=0.72,
        rejection_cue_load=0.64,
        supportive_relationship_context=0.25,
        autonomy_skills_support=0.20,
    )

    print("\n=== Simulation outputs ===")
    for section_name, series in sim.items():
        print(f"\n[{section_name}]")
        print(series.sort_values(ascending=False).to_string())

    # Optional exploration examples:
    # print(model.suggest_regions("amygdala"))
    # print(model.assign_mni_point((-8, -4, -18)).head())
    # amygdala_mask = model.region_mask("amygdala")
    # print(amygdala_mask)
