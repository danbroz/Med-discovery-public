from __future__ import annotations

"""
Narcissistic Personality Disorder siibra scaffold.

This script turns a chapter-level biological summary of Narcissistic
Personality Disorder (NPD) into a small, interpretable, atlas-grounded
mechanistic model. It is a research scaffold only. It is not a diagnostic,
prognostic, or treatment system.

Design choices follow the source chapter conservatively:
- explicit anatomy is only used where the chapter names structures or strongly
  implies them (amygdala, hippocampus, hypothalamus, prefrontal cortex,
  reward circuitry),
- transmitter and peptide systems such as dopamine, serotonin, oxytocin, and
  vasopressin are kept primarily as latent biology unless a stable atlas parcel
  is obvious,
- hypothalamic and ventral-striatal concepts are modeled as clearly labeled
  proxy nodes because Julich availability varies across siibra environments,
- the simulator exposes one transparent direction of influence:
  inputs -> latent biology -> regional state -> symptoms -> phenotype summaries.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except ImportError as exc:  # pragma: no cover - import guard for portability
    raise ImportError(
        "This scaffold requires the 'siibra' package. Install siibra in your "
        "Python environment before running this script."
    ) from exc


NARCISSISTIC_PERSONALITY_DISORDER_GENE_PANEL = [
    # Oxytocin / vasopressin bonding and social attachment
    "OXTR",
    "OXT",
    "CD38",
    "AVPR1A",
    "AVP",
    # Dopamine / reward sensitivity
    "DRD2",
    "DRD4",
    "SLC6A3",
    "TH",
    "COMT",
    # Serotonin / affect regulation
    "SLC6A4",
    "HTR1A",
    "HTR2A",
    "MAOA",
    # Stress, trauma, and neuroplasticity
    "NR3C1",
    "FKBP5",
    "CRHR1",
    "BDNF",
]

NARCISSISTIC_PERSONALITY_GENE_PANEL = NARCISSISTIC_PERSONALITY_DISORDER_GENE_PANEL
DEFAULT_GENE_PANEL = NARCISSISTIC_PERSONALITY_DISORDER_GENE_PANEL


class NarcissisticPersonalityDisorderModel:
    """
    Atlas-grounded research scaffold for Narcissistic Personality Disorder.

    Higher values in the simulation generally indicate more dysregulation or
    symptom burden, except the protective support input where higher values
    indicate stronger stabilizing support.
    """

    disorder_name = "Narcissistic Personality Disorder"
    abbreviation = "narcissistic_personality_disorder"

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

        # Disorder-specific graph content
        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Inherited liability affecting social reward sensitivity, bonding, "
                "stress reactivity, and regulatory control."
            ),
            "early_empathic_deprivation": (
                "Developmental environment lacking warmth, empathy, and stable mirroring."
            ),
            "developmental_trauma_load": (
                "Trauma burden that sensitizes limbic threat and defensive responding."
            ),
            "admiration_reward_exposure": (
                "Frequent reinforcement of status, admiration, and superiority-seeking."
            ),
            "social_status_threat": (
                "Perceived criticism, humiliation, or withdrawal of validation that threatens self-esteem."
            ),
            "reflective_support": (
                "Protective reflective or empathic support included as an exploratory stabilizing counterweight."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "epigenetic_social_bonding_scar": (
                "Enduring gene-expression style change consistent with early deprivation or trauma-related social-bonding impairment."
            ),
            "oxytocin_vasopressin_bonding_deficit": (
                "Deficient social-bonding and affiliative signaling linked to empathy impairment."
            ),
            "dopaminergic_admiration_dependence": (
                "Pathological reward dependence on admiration, status, and social approval."
            ),
            "serotonergic_affect_regulation_instability": (
                "Reduced serotonergic contribution to emotional regulation and restraint."
            ),
            "amygdala_threat_sensitization": (
                "Trauma- and threat-linked limbic hyperreactivity that amplifies defensive emotion."
            ),
            "reward_deficit_rage_liability": (
                "Negative affect state triggered when narcissistic supply is threatened or withdrawn."
            ),
            "hippocampal_stress_memory_burden": (
                "Stress-memory burden that reinforces defensive self-protection."
            ),
            "frontolimbic_control_failure": (
                "Compromised top-down regulation of limbic drives and emotional outbursts."
            ),
            "false_self_defensive_schema": (
                "Stable defensive grandiosity that suppresses vulnerable self-states."
            ),
            "neuroplastic_defensive_entrenchment": (
                "Habit-like consolidation of grandiose scanning, threat monitoring, and disregard for others."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "admiration_dependence": (
                "Persistent need for admiration and external validation."
            ),
            "grandiosity": "Inflated self-importance and superiority beliefs.",
            "empathy_deficit": "Reduced capacity for reciprocal empathy and social attunement.",
            "entitlement_superiority": (
                "Expectations of special status and privileged treatment."
            ),
            "fragile_self_esteem_reactivity": (
                "Marked reactivity to criticism, shame, or status loss."
            ),
            "envy_rage": "Envy and narcissistic rage when self-esteem is threatened.",
            "impulsive_emotional_outbursts": (
                "Poor impulse control and poorly modulated emotional expression."
            ),
            "interpersonal_exploitiveness": (
                "Instrumental use of others with limited reciprocal concern."
            ),
        }

        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 left",
                "CA3 left",
                "DG left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "hypothalamus_proxy": [
                "hypothalamus left",
                "hypothalamus",
                "preoptic",
            ],
            "pfc_control": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 9/46v left",
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "orbitofrontal",
                "prefrontal cortex",
            ],
            "ventral_striatum_proxy": [
                "nucleus accumbens left",
                "accumbens",
                "ventral striatum",
                "striatum",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "amygdala": "Atlas-backed limbic threat and aggression-reactivity node.",
            "hippocampus": "Atlas-backed stress-memory and contextualization node.",
            "hypothalamus_proxy": (
                "Proxy for hypothalamic attachment, instinctive drive, and neurohormonal integration."
            ),
            "pfc_control": "Atlas-backed or proxy prefrontal top-down regulatory control node.",
            "ventral_striatum_proxy": (
                "Proxy for reward-salience circuitry sustaining admiration dependence."
            ),
        }

        self.edge_table: List[Dict[str, Any]] = [
            {
                "source": "genetic_vulnerability",
                "target": "epigenetic_social_bonding_scar",
                "relation": "increases susceptibility to lasting social-bonding dysregulation",
                "npd_change": "increased",
                "weight": 0.18,
            },
            {
                "source": "genetic_vulnerability",
                "target": "dopaminergic_admiration_dependence",
                "relation": "biases reward sensitivity toward admiration and status cues",
                "npd_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_affect_regulation_instability",
                "relation": "reduces affective regulatory reserve",
                "npd_change": "increased",
                "weight": 0.18,
            },
            {
                "source": "early_empathic_deprivation",
                "target": "epigenetic_social_bonding_scar",
                "relation": "writes early deprivation into enduring defensive biology",
                "npd_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "early_empathic_deprivation",
                "target": "false_self_defensive_schema",
                "relation": "promotes defensive grandiose self-organization",
                "npd_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "developmental_trauma_load",
                "target": "amygdala_threat_sensitization",
                "relation": "sensitizes limbic threat and aggression circuitry",
                "npd_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "developmental_trauma_load",
                "target": "hippocampal_stress_memory_burden",
                "relation": "loads stress-memory systems and contextual fear learning",
                "npd_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "admiration_reward_exposure",
                "target": "dopaminergic_admiration_dependence",
                "relation": "reinforces reward seeking for approval and superiority",
                "npd_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "social_status_threat",
                "target": "reward_deficit_rage_liability",
                "relation": "creates reward-deficit states when admiration is threatened",
                "npd_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "social_status_threat",
                "target": "amygdala_threat_sensitization",
                "relation": "amplifies self-esteem threat processing",
                "npd_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "reflective_support",
                "target": "frontolimbic_control_failure",
                "relation": "buffers dysregulated top-down control",
                "npd_change": "decreased",
                "weight": -0.25,
            },
            {
                "source": "reflective_support",
                "target": "false_self_defensive_schema",
                "relation": "reduces rigid defensive self-organization",
                "npd_change": "decreased",
                "weight": -0.12,
            },
            {
                "source": "epigenetic_social_bonding_scar",
                "target": "oxytocin_vasopressin_bonding_deficit",
                "relation": "reduces affiliative and empathy-supporting signaling",
                "npd_change": "increased",
                "weight": 0.50,
            },
            {
                "source": "oxytocin_vasopressin_bonding_deficit",
                "target": "hypothalamus_proxy",
                "relation": "maps to disturbed hypothalamic attachment-neurohormonal state",
                "npd_change": "increased",
                "weight": 0.55,
            },
            {
                "source": "oxytocin_vasopressin_bonding_deficit",
                "target": "empathy_deficit",
                "relation": "weakens social bonding and empathic reciprocity",
                "npd_change": "increased",
                "weight": 0.50,
            },
            {
                "source": "dopaminergic_admiration_dependence",
                "target": "ventral_striatum_proxy",
                "relation": "maps to reward and salience circuitry for validation seeking",
                "npd_change": "increased",
                "weight": 0.60,
            },
            {
                "source": "dopaminergic_admiration_dependence",
                "target": "admiration_dependence",
                "relation": "creates reliance on external validation for self-esteem maintenance",
                "npd_change": "increased",
                "weight": 0.55,
            },
            {
                "source": "serotonergic_affect_regulation_instability",
                "target": "frontolimbic_control_failure",
                "relation": "reduces affective restraint and regulatory stability",
                "npd_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "amygdala_threat_sensitization",
                "target": "amygdala",
                "relation": "maps to hyper-reactive limbic threat state",
                "npd_change": "increased",
                "weight": 0.65,
            },
            {
                "source": "reward_deficit_rage_liability",
                "target": "envy_rage",
                "relation": "translates threatened supply into rage and envy",
                "npd_change": "increased",
                "weight": 0.50,
            },
            {
                "source": "hippocampal_stress_memory_burden",
                "target": "hippocampus",
                "relation": "maps to contextual stress-memory burden",
                "npd_change": "increased",
                "weight": 0.60,
            },
            {
                "source": "frontolimbic_control_failure",
                "target": "pfc_control",
                "relation": "maps to impaired prefrontal top-down control",
                "npd_change": "increased",
                "weight": 0.60,
            },
            {
                "source": "pfc_control",
                "target": "impulsive_emotional_outbursts",
                "relation": "permits poorly inhibited emotional and behavioral responses",
                "npd_change": "increased",
                "weight": 0.50,
            },
            {
                "source": "false_self_defensive_schema",
                "target": "grandiosity",
                "relation": "supports defensive superiority and self-inflation",
                "npd_change": "increased",
                "weight": 0.50,
            },
            {
                "source": "false_self_defensive_schema",
                "target": "entitlement_superiority",
                "relation": "supports rigid expectations of exceptional status",
                "npd_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "neuroplastic_defensive_entrenchment",
                "target": "interpersonal_exploitiveness",
                "relation": "stabilizes habitual disregard for others' feelings",
                "npd_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "social_status_threat",
                "target": "fragile_self_esteem_reactivity",
                "relation": "reveals unstable self-esteem under criticism or shame",
                "npd_change": "increased",
                "weight": 0.35,
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame(self.edge_table)
        self._pmap: Any = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

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
            candidates.extend([
                "receptor density fingerprint",
                "receptor density profile",
            ])
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
                    feats = siibra.features.get(concept, modality, **kwargs)
                if feats:
                    return list(feats)
            except Exception:
                continue
        return []

    def _julich_matches(self, query: str) -> List[Any]:
        attempts: List[Tuple[Any, str, bool]] = [
            (self.parcellation, "find", True),
            (self.parcellation, "find_regions", True),
            (self.atlas, "find_regions", False),
        ]
        matches: List[Any] = []
        from_julich_source = False
        for obj, method_name, is_julich_source in attempts:
            method = getattr(obj, method_name, None)
            if method is None:
                continue
            try:
                result = method(
                    query,
                    all_versions=False,
                    filter_children=False,
                    find_topmost=False,
                )
            except TypeError:
                try:
                    result = method(query, filter_children=False, find_topmost=False)
                except TypeError:
                    try:
                        result = method(query)
                    except Exception:
                        continue
            except Exception:
                continue
            if result:
                matches = list(result)
                from_julich_source = is_julich_source
                break

        out: List[Any] = []
        seen = set()
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if not from_julich_source and "julich" not in str(parc_name).lower():
                continue
            identifier = getattr(region, "identifier", None)
            key = identifier or self._name_of(region)
            if key in seen:
                continue
            seen.add(key)
            out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "hypothalamus",
            "striatum",
            "prefrontal cortex",
        } else 0
        proxy_penalty = 1 if "gapmap" in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            for getter in (
                lambda s: self.atlas.get_region(s, parcellation=self.parcellation),
                lambda s: self.parcellation.get_region(s),
                lambda s: siibra.get_region(self.parcellation_spec, s),
            ):
                try:
                    return getter(spec)
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
        components = getattr(props, "components", None)
        if components is not None:
            return list(components)
        return [props]

    def _main_component(
        self, region: Any
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda item: float(getattr(item, "volume", 0.0) or 0.0))
        centroid = getattr(main, "centroid", None)
        centroid_xyz: Optional[Tuple[float, float, float]] = None
        if centroid is not None:
            try:
                centroid_xyz = tuple(float(v) for v in centroid)
            except Exception:
                coordinate = getattr(centroid, "coordinate", None)
                if coordinate is not None:
                    centroid_xyz = tuple(float(v) for v in coordinate)
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
        try:
            df = feats[0].data.copy().reset_index()
        except Exception:
            return pd.DataFrame()

        lower_cols = {str(c).lower(): c for c in df.columns}
        if "index" in lower_cols and "receptor" not in lower_cols:
            df = df.rename(columns={lower_cols["index"]: "receptor"})
        elif df.columns.tolist():
            first_col = df.columns[0]
            if str(first_col).lower() not in {"receptor", "name"}:
                df = df.rename(columns={first_col: "receptor"})
        return df.reset_index(drop=True)

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
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

        lower_cols = {str(c).lower(): c for c in df.columns}
        gene_col = lower_cols.get("gene")
        level_col = lower_cols.get("level") or lower_cols.get("expression level")
        zscore_col = lower_cols.get("zscore") or lower_cols.get("z-score")
        if gene_col and level_col:
            agg_spec: Dict[str, Tuple[str, str]] = {
                "level_mean": (level_col, "mean"),
                "level_std": (level_col, "std"),
                "probe_count": (level_col, "count"),
            }
            if zscore_col:
                agg_spec["zscore_mean"] = (zscore_col, "mean")
                agg_spec["zscore_std"] = (zscore_col, "std")
            grouped = (
                df.groupby(gene_col, dropna=False)
                .agg(**agg_spec)
                .reset_index()
                .rename(columns={gene_col: "gene"})
            )
            grouped["gene"] = grouped["gene"].astype(str).str.upper()
            return grouped.sort_values("gene").reset_index(drop=True)

        return df.reset_index(drop=True)

    def _normalize_connectivity_matrix(self, matrix: Any) -> pd.DataFrame:
        if not isinstance(matrix, pd.DataFrame) or matrix.empty:
            return pd.DataFrame()

        out = matrix.copy()

        try:
            out.index = [self._name_of(item) for item in out.index]
        except Exception:
            pass
        try:
            out.columns = [self._name_of(item) for item in out.columns]
        except Exception:
            pass

        try:
            out = out.apply(pd.to_numeric, errors="coerce")
        except Exception:
            try:
                out = out.astype(float)
            except Exception:
                return pd.DataFrame()

        try:
            if getattr(out.index, "has_duplicates", False):
                out = out.groupby(level=0).mean()
        except Exception:
            pass

        try:
            if getattr(out.columns, "has_duplicates", False):
                out = out.T.groupby(level=0).mean().T
        except Exception:
            pass

        return out

    @staticmethod
    def _coerce_scalar(value: Any) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, pd.DataFrame):
            flat = pd.to_numeric(pd.Series(value.to_numpy().ravel()), errors="coerce").dropna()
            return float(flat.mean()) if not flat.empty else None

        if isinstance(value, pd.Series):
            flat = pd.to_numeric(value, errors="coerce").dropna()
            return float(flat.mean()) if not flat.empty else None

        try:
            if pd.isna(value):
                return None
        except Exception:
            pass

        try:
            return float(value)
        except Exception:
            return None

    def _coerce_connectivity_series(self, value: Any, prefer_axis: str = "row") -> pd.Series:
        if value is None:
            return pd.Series(dtype=float)

        if isinstance(value, pd.Series):
            series = pd.to_numeric(value, errors="coerce").dropna()
            try:
                series.index = [self._name_of(item) for item in series.index]
            except Exception:
                pass
            if getattr(series.index, "has_duplicates", False):
                series = series.groupby(level=0).mean()
            return series

        if isinstance(value, pd.DataFrame):
            try:
                numeric_df = value.apply(pd.to_numeric, errors="coerce")
            except Exception:
                return pd.Series(dtype=float)

            if prefer_axis == "column":
                series = numeric_df.mean(axis=1)
                try:
                    series.index = [self._name_of(item) for item in numeric_df.index]
                except Exception:
                    pass
            else:
                series = numeric_df.mean(axis=0)
                try:
                    series.index = [self._name_of(item) for item in numeric_df.columns]
                except Exception:
                    pass

            series = pd.to_numeric(series, errors="coerce").dropna()
            if getattr(series.index, "has_duplicates", False):
                series = series.groupby(level=0).mean()
            return series

        return pd.Series(dtype=float)

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(
            self.parcellation,
            self._modality_candidates("connectivity"),
        )
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        chosen = next(
            (
                f
                for f in feats
                if str(getattr(f, "cohort", "")).lower() == self.connectivity_cohort.lower()
            ),
            feats[0],
        )

        candidate_data = getattr(chosen, "data", None)
        normalized = self._normalize_connectivity_matrix(candidate_data)
        if not normalized.empty:
            self._connectivity_matrix = normalized
            return self._connectivity_matrix

        if hasattr(chosen, "__getitem__"):
            try:
                element = chosen[0]
                candidate_data = getattr(element, "data", None)
                normalized = self._normalize_connectivity_matrix(candidate_data)
                if not normalized.empty:
                    self._connectivity_matrix = normalized
                    return self._connectivity_matrix
            except Exception:
                pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._name_of(region)
        region_name_l = region_name.lower()

        exact = [item for item in labels if self._name_of(item) == region_name]
        if exact:
            return exact[0]

        fuzzy = [
            item
            for item in labels
            if region_name_l in self._name_of(item).lower()
            or self._name_of(item).lower() in region_name_l
        ]
        if fuzzy:
            return fuzzy[0]

        shortened = (
            region_name_l.replace("area ", "")
            .replace(" (gapmap)", "")
            .replace(" left", "")
            .replace(" right", "")
        )
        loose = [
            item
            for item in labels
            if shortened and shortened in self._name_of(item).lower().replace("area ", "")
        ]
        return loose[0] if loose else None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        row_label = self._match_region_label(list(matrix.index), region)
        col_label = self._match_region_label(list(matrix.columns), region)

        series = pd.Series(dtype=float)
        if row_label is not None:
            try:
                series = self._coerce_connectivity_series(matrix.loc[row_label], prefer_axis="row")
            except Exception:
                series = pd.Series(dtype=float)
        if series.empty and col_label is not None:
            try:
                series = self._coerce_connectivity_series(matrix[col_label], prefer_axis="column")
            except Exception:
                series = pd.Series(dtype=float)
        if series.empty:
            return pd.DataFrame()

        df = series.sort_values(ascending=False).reset_index()
        df.columns = ["connected_region", "value"]
        df["connected_region"] = df["connected_region"].map(str)
        df = df[df["connected_region"] != self._name_of(region)]
        return df.head(max_rows).reset_index(drop=True)

    def circuit_connectivity(self) -> pd.DataFrame:
        """Return a pairwise connectivity table among resolved circuit nodes."""
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        region_items = list(self.region_objects.items())
        for src_key, src_region in region_items:
            for dst_key, dst_region in region_items:
                if src_key == dst_key:
                    continue
                src_label = self._match_region_label(list(matrix.index), src_region)
                dst_label = self._match_region_label(list(matrix.columns), dst_region)
                if src_label is None or dst_label is None:
                    continue

                value: Optional[float] = None
                try:
                    value = self._coerce_scalar(matrix.loc[src_label, dst_label])
                except Exception:
                    value = None
                if value is None:
                    try:
                        value = self._coerce_scalar(matrix.loc[dst_label, src_label])
                    except Exception:
                        value = None
                if value is None:
                    continue

                rows.append(
                    {
                        "source_key": src_key,
                        "source_region": src_region.name,
                        "target_key": dst_key,
                        "target_region": dst_region.name,
                        "value": value,
                    }
                )
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .sort_values(["source_key", "value"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}

        for key, description in self.input_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "input",
                    "description": description,
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
                warnings.warn(
                    f"Could not resolve a Julich region for '{key}'. Keeping it as a proxy node."
                )
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region_proxy",
                        "description": self.region_descriptions.get(
                            key,
                            "Proxy region node unresolved in this siibra environment.",
                        ),
                        "atlas_region": None,
                        "region_identifier": None,
                        "centroid_mni": None,
                        "volume_mm3": None,
                        "feature_summary": "unresolved proxy",
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
                    "description": self.region_descriptions.get(key, "Atlas-backed circuit node."),
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

        for key, description in self.latent_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "latent_biology",
                    "description": description,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, description in self.symptom_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "symptom",
                    "description": description,
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
        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(
        self,
        node_key: str,
        space: Optional[str] = None,
        maptype: str = "labelled",
        fetch: bool = True,
    ) -> Any:
        region = self.region_objects.get(node_key)
        if region is None:
            candidates = self.region_candidates.get(node_key)
            if candidates:
                region = self._resolve_region(candidates)
                if region is not None:
                    self.region_objects[node_key] = region
        if region is None:
            return None

        target_space = space or self.assignment_space

        try:
            mask = region.get_regional_mask(space=target_space, maptype=maptype)
            if fetch and hasattr(mask, "fetch"):
                return mask.fetch()
            return mask
        except Exception:
            pass

        try:
            return region.fetch_regional_map(space=target_space, maptype=maptype)
        except Exception:
            pass

        try:
            mask = region.get_regional_map(target_space, maptype)
            if fetch and hasattr(mask, "fetch"):
                return mask.fetch()
            return mask
        except Exception:
            return None

    def simulate(
        self,
        genetic_vulnerability: float = 0.45,
        early_empathic_deprivation: float = 0.60,
        developmental_trauma_load: float = 0.45,
        admiration_reward_exposure: float = 0.55,
        social_status_threat: float = 0.50,
        reflective_support: float = 0.20,
    ) -> Dict[str, pd.Series]:
        """Run a transparent normalized simulation for the NPD scaffold."""
        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "early_empathic_deprivation": self._clip01(early_empathic_deprivation),
                "developmental_trauma_load": self._clip01(developmental_trauma_load),
                "admiration_reward_exposure": self._clip01(admiration_reward_exposure),
                "social_status_threat": self._clip01(social_status_threat),
                "reflective_support": self._clip01(reflective_support),
            },
            name="inputs",
        )

        # Inputs -> latent biology
        epigenetic_social_bonding_scar = self._clip01(
            0.42 * inputs["early_empathic_deprivation"]
            + 0.33 * inputs["developmental_trauma_load"]
            + 0.18 * inputs["genetic_vulnerability"]
            - 0.20 * inputs["reflective_support"]
        )

        oxytocin_vasopressin_bonding_deficit = self._clip01(
            0.50 * epigenetic_social_bonding_scar
            + 0.18 * inputs["genetic_vulnerability"]
            + 0.10 * inputs["social_status_threat"]
            - 0.18 * inputs["reflective_support"]
        )

        dopaminergic_admiration_dependence = self._clip01(
            0.42 * inputs["admiration_reward_exposure"]
            + 0.20 * inputs["genetic_vulnerability"]
            + 0.18 * inputs["early_empathic_deprivation"]
            + 0.10 * inputs["social_status_threat"]
            - 0.12 * inputs["reflective_support"]
        )

        serotonergic_affect_regulation_instability = self._clip01(
            0.30 * inputs["developmental_trauma_load"]
            + 0.22 * inputs["genetic_vulnerability"]
            + 0.18 * inputs["social_status_threat"]
            - 0.15 * inputs["reflective_support"]
        )

        amygdala_threat_sensitization = self._clip01(
            0.40 * inputs["developmental_trauma_load"]
            + 0.25 * inputs["social_status_threat"]
            + 0.15 * serotonergic_affect_regulation_instability
            + 0.10 * epigenetic_social_bonding_scar
            - 0.10 * inputs["reflective_support"]
        )

        reward_deficit_rage_liability = self._clip01(
            0.38 * inputs["social_status_threat"]
            + 0.30 * dopaminergic_admiration_dependence
            + 0.15 * amygdala_threat_sensitization
            - 0.10 * inputs["reflective_support"]
        )

        hippocampal_stress_memory_burden = self._clip01(
            0.35 * inputs["developmental_trauma_load"]
            + 0.22 * amygdala_threat_sensitization
            + 0.18 * inputs["social_status_threat"]
        )

        frontolimbic_control_failure = self._clip01(
            0.32 * amygdala_threat_sensitization
            + 0.20 * serotonergic_affect_regulation_instability
            + 0.18 * oxytocin_vasopressin_bonding_deficit
            + 0.15 * dopaminergic_admiration_dependence
            + 0.10 * hippocampal_stress_memory_burden
            - 0.25 * inputs["reflective_support"]
        )

        false_self_defensive_schema = self._clip01(
            0.35 * inputs["early_empathic_deprivation"]
            + 0.25 * dopaminergic_admiration_dependence
            + 0.18 * reward_deficit_rage_liability
            + 0.15 * oxytocin_vasopressin_bonding_deficit
            + 0.12 * inputs["social_status_threat"]
            - 0.10 * inputs["reflective_support"]
        )

        neuroplastic_defensive_entrenchment = self._clip01(
            0.42 * false_self_defensive_schema
            + 0.22 * dopaminergic_admiration_dependence
            + 0.18 * amygdala_threat_sensitization
            + 0.12 * hippocampal_stress_memory_burden
        )

        latents = pd.Series(
            {
                "epigenetic_social_bonding_scar": epigenetic_social_bonding_scar,
                "oxytocin_vasopressin_bonding_deficit": oxytocin_vasopressin_bonding_deficit,
                "dopaminergic_admiration_dependence": dopaminergic_admiration_dependence,
                "serotonergic_affect_regulation_instability": serotonergic_affect_regulation_instability,
                "amygdala_threat_sensitization": amygdala_threat_sensitization,
                "reward_deficit_rage_liability": reward_deficit_rage_liability,
                "hippocampal_stress_memory_burden": hippocampal_stress_memory_burden,
                "frontolimbic_control_failure": frontolimbic_control_failure,
                "false_self_defensive_schema": false_self_defensive_schema,
                "neuroplastic_defensive_entrenchment": neuroplastic_defensive_entrenchment,
            },
            name="latents",
        )

        # Latent biology -> regional state
        ventral_striatum_proxy = self._clip01(
            0.65 * dopaminergic_admiration_dependence
            + 0.20 * inputs["admiration_reward_exposure"]
            + 0.10 * false_self_defensive_schema
        )

        amygdala = self._clip01(
            0.65 * amygdala_threat_sensitization
            + 0.20 * reward_deficit_rage_liability
        )

        hippocampus = self._clip01(
            0.65 * hippocampal_stress_memory_burden
            + 0.15 * inputs["developmental_trauma_load"]
        )

        hypothalamus_proxy = self._clip01(
            0.60 * oxytocin_vasopressin_bonding_deficit
            + 0.15 * amygdala_threat_sensitization
            + 0.10 * epigenetic_social_bonding_scar
        )

        pfc_control = self._clip01(
            0.65 * frontolimbic_control_failure
            + 0.15 * neuroplastic_defensive_entrenchment
            + 0.10 * serotonergic_affect_regulation_instability
        )

        regional_state = pd.Series(
            {
                "ventral_striatum_proxy": ventral_striatum_proxy,
                "amygdala": amygdala,
                "hippocampus": hippocampus,
                "hypothalamus_proxy": hypothalamus_proxy,
                "pfc_control": pfc_control,
            },
            name="regional_state",
        )

        # Regional state -> symptoms
        admiration_dependence = self._clip01(
            0.55 * ventral_striatum_proxy
            + 0.22 * false_self_defensive_schema
            + 0.12 * inputs["social_status_threat"]
        )

        grandiosity = self._clip01(
            0.42 * false_self_defensive_schema
            + 0.25 * dopaminergic_admiration_dependence
            + 0.20 * admiration_dependence
            + 0.10 * neuroplastic_defensive_entrenchment
        )

        empathy_deficit = self._clip01(
            0.50 * oxytocin_vasopressin_bonding_deficit
            + 0.20 * hypothalamus_proxy
            + 0.18 * frontolimbic_control_failure
            + 0.08 * neuroplastic_defensive_entrenchment
        )

        entitlement_superiority = self._clip01(
            0.35 * grandiosity
            + 0.25 * admiration_dependence
            + 0.20 * neuroplastic_defensive_entrenchment
            + 0.10 * oxytocin_vasopressin_bonding_deficit
        )

        fragile_self_esteem_reactivity = self._clip01(
            0.40 * reward_deficit_rage_liability
            + 0.25 * admiration_dependence
            + 0.20 * hippocampal_stress_memory_burden
            + 0.10 * inputs["social_status_threat"]
        )

        envy_rage = self._clip01(
            0.38 * amygdala
            + 0.26 * reward_deficit_rage_liability
            + 0.18 * inputs["social_status_threat"]
            + 0.10 * admiration_dependence
        )

        impulsive_emotional_outbursts = self._clip01(
            0.42 * pfc_control
            + 0.25 * envy_rage
            + 0.15 * amygdala
        )

        interpersonal_exploitiveness = self._clip01(
            0.35 * empathy_deficit
            + 0.28 * entitlement_superiority
            + 0.18 * admiration_dependence
        )

        symptoms = pd.Series(
            {
                "admiration_dependence": admiration_dependence,
                "grandiosity": grandiosity,
                "empathy_deficit": empathy_deficit,
                "entitlement_superiority": entitlement_superiority,
                "fragile_self_esteem_reactivity": fragile_self_esteem_reactivity,
                "envy_rage": envy_rage,
                "impulsive_emotional_outbursts": impulsive_emotional_outbursts,
                "interpersonal_exploitiveness": interpersonal_exploitiveness,
            },
            name="symptoms",
        )

        phenotypes = pd.Series(
            {
                "grandiose_narcissism_profile": self._clip01(
                    (grandiosity + admiration_dependence + entitlement_superiority) / 3.0
                ),
                "vulnerable_reactivity_profile": self._clip01(
                    (
                        fragile_self_esteem_reactivity
                        + envy_rage
                        + impulsive_emotional_outbursts
                    )
                    / 3.0
                ),
                "empathy_attachment_deficit_profile": self._clip01(
                    (empathy_deficit + interpersonal_exploitiveness) / 2.0
                ),
                "narcissistic_rage_profile": self._clip01(
                    (envy_rage + impulsive_emotional_outbursts + reward_deficit_rage_liability)
                    / 3.0
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


if __name__ == "__main__":
    model = NarcissisticPersonalityDisorderModel()
    bundle = model.build(connectivity_rows=10)

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 12)

    print("\n=== NODES ===")
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

    print("\n=== EDGES ===")
    print(bundle["edges"][["source", "target", "relation", "weight"]].to_string(index=False))

    for key in ["amygdala", "ventral_striatum_proxy", "pfc_control"]:
        print(f"\n=== REGION SUMMARY: {key} ===")
        region = bundle["regions"].get(key)
        print(f"Resolved region: {getattr(region, 'name', 'unresolved proxy')}")
        receptor_df = bundle["receptors"].get(key, pd.DataFrame())
        gene_df = bundle["genes"].get(key, pd.DataFrame())
        conn_df = bundle["connectivity_profiles"].get(key, pd.DataFrame())
        print("Receptors:")
        print(receptor_df.head(8).to_string(index=False) if not receptor_df.empty else "  <none>")
        print("Genes:")
        print(gene_df.head(8).to_string(index=False) if not gene_df.empty else "  <none>")
        print("Connectivity:")
        print(conn_df.head(8).to_string(index=False) if not conn_df.empty else "  <none>")

    circuit_df = bundle["circuit_connectivity"]
    print("\n=== CIRCUIT CONNECTIVITY ===")
    print(circuit_df.head(20).to_string(index=False) if not circuit_df.empty else "<none>")

    sim = model.simulate(
        genetic_vulnerability=0.50,
        early_empathic_deprivation=0.70,
        developmental_trauma_load=0.55,
        admiration_reward_exposure=0.65,
        social_status_threat=0.60,
        reflective_support=0.20,
    )

    for section, series in sim.items():
        print(f"\n=== {section.upper()} ===")
        print(series.to_string())

    # Optional interactive checks:
    # print(model.suggest_regions("amygdala").head(10))
    # print(model.assign_mni_point((20, -6, -14)).head(10))
    # mask = model.region_mask("amygdala")
    # print(type(mask))
