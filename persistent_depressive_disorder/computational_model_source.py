from __future__ import annotations

"""
Persistent Depressive Disorder (PDD) siibra scaffold.

This script turns a chapter-level biological summary of Persistent Depressive
Disorder into a small, interpretable, atlas-grounded mechanistic model. It is
intended as a research scaffold only. It is not a diagnostic, prognostic, or
therapeutic system.

Design choices follow the chapter conservatively:
- explicit anatomy is limited to regions named in the chapter or strongly tied
  to the cited mood-regulation network (prefrontal cortex, anterior cingulate
  cortex, hippocampus, amygdala),
- dopamine, glutamate, GABA, stress biology, and neuroplasticity are kept as
  latent processes rather than being forced into artificial atlas parcels,
- early- versus late-onset heterogeneity is modeled as input modifiers instead
  of pretending they map to unique cytoarchitectonic regions.
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


PERSISTENT_DEPRESSIVE_DISORDER_GENE_PANEL = [
    # Dopamine / reward / motivation
    "DRD2",
    "DRD3",
    "SLC6A3",
    "COMT",
    # Glutamate / GABA balance
    "GRIN1",
    "GRIN2A",
    "GRIN2B",
    "GAD1",
    "GAD2",
    "GABRA1",
    "GABRB2",
    # Stress responsivity / neuroplasticity
    "NR3C1",
    "FKBP5",
    "BDNF",
]

DEFAULT_GENE_PANEL = PERSISTENT_DEPRESSIVE_DISORDER_GENE_PANEL


class PersistentDepressiveDisorderModel:
    """
    Atlas-grounded research scaffold for Persistent Depressive Disorder.

    The simulator is deliberately simple and normalized to 0..1. Higher values
    generally indicate more dysregulation or symptom burden, except the support
    inputs where higher values indicate stronger protective support.
    """

    disorder_name = "Persistent Depressive Disorder"
    abbreviation = "pdd"

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

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Heritable vulnerability contributing to persistent mood dysregulation and chronic depressive burden."
            ),
            "chronic_psychosocial_stress": (
                "Sustained psychosocial stress load that can maintain chronic depressive physiology."
            ),
            "adverse_life_event_burden": (
                "Accumulated adverse experiences that amplify mood-disorder risk in vulnerable individuals."
            ),
            "early_onset_trait_load": (
                "Trait-like early-onset dysthymic vulnerability linked to neuroticism and enduring negative affectivity."
            ),
            "late_onset_somatic_burden": (
                "Later-life somatic or cerebrovascular burden that can worsen chronic depressive circuitry."
            ),
            "recovery_support": (
                "Protective treatment, structure, and psychosocial support that can reduce chronic dysregulation."
            ),
            "dopamine_targeted_support": (
                "Reward- and motivation-oriented support consistent with dopamine-modulating treatment effects."
            ),
            "glutamatergic_plasticity_support": (
                "Plasticity-oriented support consistent with NMDA/glutamatergic modulation."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "hpa_stress_sensitization": (
                "Sustained stress-system sensitization that can maintain chronic depressive states."
            ),
            "dopaminergic_reward_deficit": (
                "Reduced reward, pleasure, and incentive salience consistent with anhedonia and apathy."
            ),
            "glutamate_gaba_imbalance": (
                "Persistent excitation-inhibition imbalance affecting mood-regulating circuits."
            ),
            "neuroplasticity_burden": (
                "Reduced adaptive plasticity linked to chronic stress and neurotransmitter dysregulation."
            ),
            "trait_negative_affect_bias": (
                "Enduring negative-affect style consistent with early-onset dysthymic traits."
            ),
            "frontolimbic_dysregulation": (
                "Distributed dysfunction across cortical and limbic mood-regulation circuits."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "chronic_depressed_mood": (
                "Persistently low mood or chronic depressive affect."
            ),
            "anhedonia": "Reduced pleasure or interest in normally rewarding activities.",
            "apathy_low_motivation": (
                "Low drive, low initiative, and apathy linked to reward-system dysfunction."
            ),
            "affective_cognitive_control_difficulty": (
                "Difficulty regulating mood, cognition, and conflict processing."
            ),
            "stress_sensitive_persistence": (
                "Tendency for depressive burden to persist under stress or adversity."
            ),
            "functional_impairment": (
                "Disabling functional burden associated with persistent depressive symptoms."
            ),
        }

        self.region_candidates: Dict[str, List[str]] = {
            "pfc_control": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 9/46v left",
                "Area Fp1 (FPole) left",
                "prefrontal cortex",
            ],
            "acc": [
                "Area p32 left",
                "Area p24ab left",
                "Area p24c left",
                "Area a24pr left",
                "anterior cingulate",
                "cingulate cortex",
            ],
            "hippocampus": [
                "CA1 left",
                "CA3 left",
                "DG left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "pfc_control": "Atlas-backed or proxy prefrontal regulatory control node.",
            "acc": "Anterior cingulate affective/conflict regulation node.",
            "hippocampus": "Atlas-backed hippocampal stress-memory and context node.",
            "amygdala": "Atlas-backed limbic salience and negative-affect node.",
        }

        self.edge_table: List[Dict[str, Any]] = [
            {
                "source": "genetic_vulnerability",
                "target": "dopaminergic_reward_deficit",
                "relation": "raises persistent reward-system vulnerability",
                "pdd_change": "increased",
                "weight": 0.26,
            },
            {
                "source": "genetic_vulnerability",
                "target": "glutamate_gaba_imbalance",
                "relation": "increases baseline neurotransmitter-system vulnerability",
                "pdd_change": "increased",
                "weight": 0.18,
            },
            {
                "source": "genetic_vulnerability",
                "target": "hpa_stress_sensitization",
                "relation": "amplifies biological susceptibility to chronic stress effects",
                "pdd_change": "increased",
                "weight": 0.16,
            },
            {
                "source": "chronic_psychosocial_stress",
                "target": "hpa_stress_sensitization",
                "relation": "sensitizes sustained stress physiology",
                "pdd_change": "increased",
                "weight": 0.42,
            },
            {
                "source": "adverse_life_event_burden",
                "target": "hpa_stress_sensitization",
                "relation": "adds cumulative adversity burden to chronic stress systems",
                "pdd_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "early_onset_trait_load",
                "target": "trait_negative_affect_bias",
                "relation": "reinforces enduring negative-affect style",
                "pdd_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "late_onset_somatic_burden",
                "target": "frontolimbic_dysregulation",
                "relation": "adds somatic and cerebrovascular burden to mood circuitry",
                "pdd_change": "increased",
                "weight": 0.22,
            },
            {
                "source": "recovery_support",
                "target": "hpa_stress_sensitization",
                "relation": "reduces physiological perpetuation of depressive burden",
                "pdd_change": "decreased",
                "weight": -0.20,
            },
            {
                "source": "dopamine_targeted_support",
                "target": "dopaminergic_reward_deficit",
                "relation": "partly compensates for reward and motivation deficits",
                "pdd_change": "decreased",
                "weight": -0.30,
            },
            {
                "source": "glutamatergic_plasticity_support",
                "target": "glutamate_gaba_imbalance",
                "relation": "partly offsets excitation-inhibition dysregulation",
                "pdd_change": "decreased",
                "weight": -0.30,
            },
            {
                "source": "hpa_stress_sensitization",
                "target": "neuroplasticity_burden",
                "relation": "erodes adaptive plasticity under chronic stress",
                "pdd_change": "increased",
                "weight": 0.28,
            },
            {
                "source": "hpa_stress_sensitization",
                "target": "glutamate_gaba_imbalance",
                "relation": "biases excitation-inhibition balance toward dysregulation",
                "pdd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "dopaminergic_reward_deficit",
                "target": "frontolimbic_dysregulation",
                "relation": "reduces effective reward-related control within mood circuits",
                "pdd_change": "increased",
                "weight": 0.22,
            },
            {
                "source": "glutamate_gaba_imbalance",
                "target": "frontolimbic_dysregulation",
                "relation": "destabilizes cortical-limbic regulation",
                "pdd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "trait_negative_affect_bias",
                "target": "frontolimbic_dysregulation",
                "relation": "loads persistent negative-affect processing onto mood circuits",
                "pdd_change": "increased",
                "weight": 0.18,
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "pfc_control",
                "relation": "maps to chronic prefrontal regulatory burden",
                "pdd_change": "increased",
                "weight": 0.55,
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "acc",
                "relation": "maps to chronic anterior cingulate affective/conflict burden",
                "pdd_change": "increased",
                "weight": 0.52,
            },
            {
                "source": "neuroplasticity_burden",
                "target": "hippocampus",
                "relation": "maps to chronic hippocampal plasticity and context burden",
                "pdd_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "hpa_stress_sensitization",
                "target": "amygdala",
                "relation": "maps to heightened negative-affect salience burden",
                "pdd_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "dopaminergic_reward_deficit",
                "target": "anhedonia",
                "relation": "drives persistent loss of interest and pleasure",
                "pdd_change": "increased",
                "weight": 0.50,
            },
            {
                "source": "dopaminergic_reward_deficit",
                "target": "apathy_low_motivation",
                "relation": "drives low motivation and apathy",
                "pdd_change": "increased",
                "weight": 0.48,
            },
            {
                "source": "amygdala",
                "target": "chronic_depressed_mood",
                "relation": "amplifies sustained negative affective tone",
                "pdd_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "acc",
                "target": "affective_cognitive_control_difficulty",
                "relation": "increases affective conflict and regulation burden",
                "pdd_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "pfc_control",
                "target": "affective_cognitive_control_difficulty",
                "relation": "increases executive and regulatory burden",
                "pdd_change": "increased",
                "weight": 0.38,
            },
            {
                "source": "hippocampus",
                "target": "functional_impairment",
                "relation": "contributes memory and context-related burden to chronic disability",
                "pdd_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "hpa_stress_sensitization",
                "target": "stress_sensitive_persistence",
                "relation": "sustains depressive burden under repeated stress",
                "pdd_change": "increased",
                "weight": 0.40,
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
            "prefrontal cortex",
            "anterior cingulate cortex",
            "hippocampus",
            "amygdala",
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
        if isinstance(candidate_data, pd.DataFrame):
            self._connectivity_matrix = candidate_data.copy()
            return self._connectivity_matrix

        if hasattr(chosen, "__getitem__"):
            try:
                element = chosen[0]
                candidate_data = getattr(element, "data", None)
                if isinstance(candidate_data, pd.DataFrame):
                    self._connectivity_matrix = candidate_data.copy()
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

    @staticmethod
    def _connectivity_series_from_selection(
        selection: Any,
        *,
        average_axis: int,
    ) -> pd.Series:
        """
        Coerce a connectivity selection to a numeric Series.

        Duplicate row or column labels in siibra-derived matrices can make a
        single label lookup return a DataFrame rather than a Series. In that
        case we average across the duplicated axis to obtain one profile.
        """
        if isinstance(selection, pd.DataFrame):
            numeric = selection.apply(pd.to_numeric, errors="coerce")
            series = numeric.mean(axis=average_axis)
        elif isinstance(selection, pd.Series):
            series = pd.to_numeric(selection, errors="coerce")
        else:
            return pd.Series(dtype=float)

        if not series.index.is_unique:
            series = series.groupby(level=0, sort=False).mean()
        return series.dropna()

    @staticmethod
    def _connectivity_scalar(value: Any) -> Optional[float]:
        """
        Reduce a connectivity lookup to a scalar float.

        When both the row and column labels are duplicated, pandas can return a
        DataFrame. We average all numeric entries so the circuit table remains
        sortable and comparable.
        """
        if isinstance(value, pd.DataFrame):
            flattened = pd.to_numeric(
                pd.Series(value.to_numpy().reshape(-1)),
                errors="coerce",
            ).dropna()
            if flattened.empty:
                return None
            return float(flattened.mean())

        if isinstance(value, pd.Series):
            numeric = pd.to_numeric(value, errors="coerce").dropna()
            if numeric.empty:
                return None
            return float(numeric.mean())

        try:
            return float(value)
        except Exception:
            return None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        row_label = self._match_region_label(list(matrix.index), region)
        col_label = self._match_region_label(list(matrix.columns), region)

        series = None
        if row_label is not None:
            try:
                selection = matrix.loc[row_label]
                series = self._connectivity_series_from_selection(
                    selection,
                    average_axis=0,
                )
            except Exception:
                series = None
        if (series is None or series.empty) and col_label is not None:
            try:
                selection = matrix[col_label]
                series = self._connectivity_series_from_selection(
                    selection,
                    average_axis=1,
                )
            except Exception:
                series = None
        if series is None or series.empty:
            return pd.DataFrame()

        try:
            df = series.sort_values(ascending=False).reset_index()
        except Exception:
            return pd.DataFrame()
        df.columns = ["connected_region", "value"]
        df["connected_region"] = df["connected_region"].map(self._name_of)
        df = df[df["connected_region"] != region.name]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
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

                value = None
                try:
                    value = self._connectivity_scalar(matrix.loc[src_label, dst_label])
                except Exception:
                    value = None
                if value is None:
                    try:
                        value = self._connectivity_scalar(matrix.loc[dst_label, src_label])
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

        df = pd.DataFrame(rows)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        if df.empty:
            return pd.DataFrame()
        return df.sort_values(["source_key", "value"], ascending=[True, False]).reset_index(drop=True)

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
        genetic_vulnerability: float = 0.50,
        chronic_psychosocial_stress: float = 0.60,
        adverse_life_event_burden: float = 0.50,
        early_onset_trait_load: float = 0.45,
        late_onset_somatic_burden: float = 0.20,
        recovery_support: float = 0.35,
        dopamine_targeted_support: float = 0.15,
        glutamatergic_plasticity_support: float = 0.10,
    ) -> Dict[str, pd.Series]:
        """
        Run a transparent normalized simulation.

        Higher values indicate more burden for the risk inputs and stronger
        support for the protective inputs.
        """
        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "chronic_psychosocial_stress": self._clip01(chronic_psychosocial_stress),
                "adverse_life_event_burden": self._clip01(adverse_life_event_burden),
                "early_onset_trait_load": self._clip01(early_onset_trait_load),
                "late_onset_somatic_burden": self._clip01(late_onset_somatic_burden),
                "recovery_support": self._clip01(recovery_support),
                "dopamine_targeted_support": self._clip01(dopamine_targeted_support),
                "glutamatergic_plasticity_support": self._clip01(glutamatergic_plasticity_support),
            },
            name="inputs",
        )

        hpa_stress_sensitization = self._clip01(
            0.42 * inputs["chronic_psychosocial_stress"]
            + 0.28 * inputs["adverse_life_event_burden"]
            + 0.16 * inputs["genetic_vulnerability"]
            - 0.20 * inputs["recovery_support"]
        )

        dopaminergic_reward_deficit = self._clip01(
            0.30 * inputs["genetic_vulnerability"]
            + 0.22 * hpa_stress_sensitization
            + 0.16 * inputs["chronic_psychosocial_stress"]
            + 0.12 * inputs["early_onset_trait_load"]
            - 0.28 * inputs["dopamine_targeted_support"]
            - 0.08 * inputs["recovery_support"]
        )

        glutamate_gaba_imbalance = self._clip01(
            0.26 * hpa_stress_sensitization
            + 0.22 * inputs["genetic_vulnerability"]
            + 0.18 * inputs["chronic_psychosocial_stress"]
            + 0.14 * inputs["late_onset_somatic_burden"]
            - 0.28 * inputs["glutamatergic_plasticity_support"]
            - 0.08 * inputs["recovery_support"]
        )

        neuroplasticity_burden = self._clip01(
            0.30 * dopaminergic_reward_deficit
            + 0.28 * glutamate_gaba_imbalance
            + 0.22 * hpa_stress_sensitization
            + 0.12 * inputs["late_onset_somatic_burden"]
            - 0.20 * inputs["glutamatergic_plasticity_support"]
            - 0.08 * inputs["recovery_support"]
        )

        trait_negative_affect_bias = self._clip01(
            0.40 * inputs["early_onset_trait_load"]
            + 0.18 * inputs["genetic_vulnerability"]
            + 0.18 * hpa_stress_sensitization
            + 0.10 * inputs["adverse_life_event_burden"]
            - 0.12 * inputs["recovery_support"]
        )

        frontolimbic_dysregulation = self._clip01(
            0.24 * dopaminergic_reward_deficit
            + 0.24 * glutamate_gaba_imbalance
            + 0.20 * hpa_stress_sensitization
            + 0.18 * trait_negative_affect_bias
            + 0.10 * inputs["late_onset_somatic_burden"]
            - 0.14 * inputs["recovery_support"]
            - 0.08 * inputs["glutamatergic_plasticity_support"]
        )

        latents = pd.Series(
            {
                "hpa_stress_sensitization": hpa_stress_sensitization,
                "dopaminergic_reward_deficit": dopaminergic_reward_deficit,
                "glutamate_gaba_imbalance": glutamate_gaba_imbalance,
                "neuroplasticity_burden": neuroplasticity_burden,
                "trait_negative_affect_bias": trait_negative_affect_bias,
                "frontolimbic_dysregulation": frontolimbic_dysregulation,
            },
            name="latents",
        )

        regional_state = pd.Series(
            {
                "pfc_control": self._clip01(
                    0.55 * frontolimbic_dysregulation
                    + 0.20 * neuroplasticity_burden
                    + 0.10 * inputs["late_onset_somatic_burden"]
                    + 0.08 * hpa_stress_sensitization
                    - 0.10 * inputs["recovery_support"]
                ),
                "acc": self._clip01(
                    0.48 * frontolimbic_dysregulation
                    + 0.22 * glutamate_gaba_imbalance
                    + 0.12 * trait_negative_affect_bias
                    + 0.08 * inputs["late_onset_somatic_burden"]
                    - 0.08 * inputs["recovery_support"]
                ),
                "hippocampus": self._clip01(
                    0.48 * neuroplasticity_burden
                    + 0.24 * hpa_stress_sensitization
                    + 0.10 * inputs["chronic_psychosocial_stress"]
                    + 0.08 * inputs["late_onset_somatic_burden"]
                    - 0.08 * inputs["recovery_support"]
                ),
                "amygdala": self._clip01(
                    0.42 * hpa_stress_sensitization
                    + 0.24 * trait_negative_affect_bias
                    + 0.18 * frontolimbic_dysregulation
                    + 0.10 * inputs["adverse_life_event_burden"]
                    - 0.08 * inputs["recovery_support"]
                ),
            },
            name="regional_state",
        )

        chronic_depressed_mood = self._clip01(
            0.30 * regional_state["amygdala"]
            + 0.22 * regional_state["acc"]
            + 0.18 * hpa_stress_sensitization
            + 0.14 * regional_state["hippocampus"]
            + 0.10 * trait_negative_affect_bias
            - 0.10 * inputs["recovery_support"]
        )

        anhedonia = self._clip01(
            0.42 * dopaminergic_reward_deficit
            + 0.16 * regional_state["pfc_control"]
            + 0.14 * regional_state["acc"]
            + 0.10 * neuroplasticity_burden
            - 0.18 * inputs["dopamine_targeted_support"]
            - 0.06 * inputs["recovery_support"]
        )

        apathy_low_motivation = self._clip01(
            0.38 * dopaminergic_reward_deficit
            + 0.20 * regional_state["pfc_control"]
            + 0.12 * regional_state["hippocampus"]
            + 0.10 * chronic_depressed_mood
            - 0.14 * inputs["dopamine_targeted_support"]
            - 0.06 * inputs["recovery_support"]
        )

        affective_cognitive_control_difficulty = self._clip01(
            0.34 * regional_state["pfc_control"]
            + 0.28 * regional_state["acc"]
            + 0.16 * regional_state["hippocampus"]
            + 0.10 * neuroplasticity_burden
            + 0.06 * chronic_depressed_mood
        )

        stress_sensitive_persistence = self._clip01(
            0.28 * hpa_stress_sensitization
            + 0.24 * regional_state["amygdala"]
            + 0.18 * trait_negative_affect_bias
            + 0.14 * chronic_depressed_mood
            + 0.08 * inputs["adverse_life_event_burden"]
            - 0.08 * inputs["recovery_support"]
        )

        functional_impairment = self._clip01(
            0.26 * chronic_depressed_mood
            + 0.24 * anhedonia
            + 0.20 * affective_cognitive_control_difficulty
            + 0.14 * apathy_low_motivation
            + 0.10 * stress_sensitive_persistence
            + 0.06 * regional_state["hippocampus"]
        )

        symptoms = pd.Series(
            {
                "chronic_depressed_mood": chronic_depressed_mood,
                "anhedonia": anhedonia,
                "apathy_low_motivation": apathy_low_motivation,
                "affective_cognitive_control_difficulty": affective_cognitive_control_difficulty,
                "stress_sensitive_persistence": stress_sensitive_persistence,
                "functional_impairment": functional_impairment,
            },
            name="symptoms",
        )

        phenotypes = pd.Series(
            {
                "anhedonic_pdd_profile": self._clip01(
                    0.42 * anhedonia
                    + 0.30 * apathy_low_motivation
                    + 0.28 * dopaminergic_reward_deficit
                ),
                "frontolimbic_pdd_profile": self._clip01(
                    0.22 * regional_state["pfc_control"]
                    + 0.22 * regional_state["acc"]
                    + 0.18 * regional_state["amygdala"]
                    + 0.16 * regional_state["hippocampus"]
                    + 0.22 * affective_cognitive_control_difficulty
                ),
                "stress_sensitized_pdd_profile": self._clip01(
                    0.34 * hpa_stress_sensitization
                    + 0.24 * regional_state["amygdala"]
                    + 0.22 * stress_sensitive_persistence
                    + 0.20 * chronic_depressed_mood
                ),
                "early_onset_trait_profile": self._clip01(
                    0.34 * inputs["early_onset_trait_load"]
                    + 0.26 * trait_negative_affect_bias
                    + 0.20 * chronic_depressed_mood
                    + 0.20 * stress_sensitive_persistence
                ),
                "late_onset_somatic_profile": self._clip01(
                    0.40 * inputs["late_onset_somatic_burden"]
                    + 0.20 * regional_state["pfc_control"]
                    + 0.18 * regional_state["hippocampus"]
                    + 0.22 * functional_impairment
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
    pd.set_option("display.width", 150)
    pd.set_option("display.max_columns", 20)

    model = PersistentDepressiveDisorderModel()
    built = model.build(
        gene_panel=PERSISTENT_DEPRESSIVE_DISORDER_GENE_PANEL,
        connectivity_rows=10,
    )

    print("\n=== NODES (head) ===")
    print(built["nodes"].head(12).to_string(index=False))

    print("\n=== EDGES (head) ===")
    print(built["edges"].head(12).to_string(index=False))

    print("\n=== RESOLVED REGIONS ===")
    if built["regions"]:
        for key, region in built["regions"].items():
            print(f"- {key}: {region.name}")
    else:
        print("No regions resolved in this environment.")

    for key in ("pfc_control", "acc", "hippocampus", "amygdala"):
        receptor_df = built["receptors"].get(key, pd.DataFrame())
        gene_df = built["genes"].get(key, pd.DataFrame())
        conn_df = built["connectivity_profiles"].get(key, pd.DataFrame())

        print(f"\n=== FEATURE SNAPSHOT: {key} ===")
        print("Receptors:")
        print(receptor_df.head(5).to_string(index=False) if not receptor_df.empty else "<none>")
        print("Genes:")
        print(gene_df.head(5).to_string(index=False) if not gene_df.empty else "<none>")
        print("Connectivity:")
        print(conn_df.head(5).to_string(index=False) if not conn_df.empty else "<none>")

    sim = model.simulate(
        genetic_vulnerability=0.60,
        chronic_psychosocial_stress=0.70,
        adverse_life_event_burden=0.55,
        early_onset_trait_load=0.50,
        late_onset_somatic_burden=0.20,
        recovery_support=0.35,
        dopamine_targeted_support=0.20,
        glutamatergic_plasticity_support=0.10,
    )

    print("\n=== SIMULATION: LATENTS ===")
    print(sim["latents"].sort_values(ascending=False).to_string())

    print("\n=== SIMULATION: SYMPTOMS ===")
    print(sim["symptoms"].sort_values(ascending=False).to_string())

    print("\n=== SIMULATION: PHENOTYPES ===")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Example coordinate assignment:
    # assignments = model.assign_mni_point((0, 32, 18))
    # print(assignments.head())

    # Example region mask retrieval:
    # mask_img = model.region_mask("acc")
    # print(mask_img)
