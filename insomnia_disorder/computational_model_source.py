from __future__ import annotations

"""
Insomnia Disorder siibra scaffold.

This script turns a chapter-level biological summary of Insomnia Disorder into a
small, interpretable, atlas-grounded mechanistic model. It is a research
scaffold only. It is not a diagnostic, prognostic, or treatment system.

Design choices follow the source chapter conservatively:
- explicit anatomy is only used where the chapter names structures or strongly
  implies them (hippocampus, prefrontal cortex, hypothalamic sleep/circadian
  hubs),
- systems such as orexin, histaminergic, and monoaminergic arousal are kept as
  latent biology unless a stable atlas parcel is obvious,
- hypothalamic entities like VLPO and SCN are modeled as clearly labeled proxy
  nodes because Julich availability varies across siibra environments.
"""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except ImportError as exc:  # pragma: no cover - import guard for portability
    raise ImportError(
        "This scaffold requires the 'siibra' package. Install siibra in your "
        "Python environment before running this script."
    ) from exc


INSOMNIA_GENE_PANEL = [
    # GABAergic sleep-promotion
    "GAD1",
    "GABRA1",
    "GABRA2",
    "GABRB2",
    "GABRG2",
    "GAL",
    # Orexin / arousal
    "HCRT",
    "HCRTR1",
    "HCRTR2",
    "HDC",
    "SLC6A4",
    "MAOA",
    # Circadian clock genes
    "CLOCK",
    "ARNTL",
    "PER1",
    "PER2",
    "PER3",
    "CRY1",
    "CRY2",
    # Stress / neuroplasticity
    "NR3C1",
    "FKBP5",
    "BDNF",
]

DEFAULT_GENE_PANEL = INSOMNIA_GENE_PANEL


class InsomniaDisorderModel:
    """
    Atlas-grounded research scaffold for Insomnia Disorder.

    The simulator is deliberately simple and normalized to 0..1. Higher values
    generally indicate more dysregulation or symptom burden, except the support
    inputs where higher values indicate stronger protective support.
    """

    disorder_name = "Insomnia Disorder"
    abbreviation = "insomnia"

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
                "Inherited liability affecting hyperarousal, circadian stability, and sleep continuity."
            ),
            "circadian_gene_disruption": (
                "Clock-gene related vulnerability or circadian misalignment burden."
            ),
            "psychosocial_stress_load": (
                "Stress burden that amplifies arousal and HPA-axis sensitization."
            ),
            "comorbid_sleep_disorder_load": (
                "Burden from other sleep-disrupting conditions considered in differential diagnosis, such as RLS, breathing disorders, or parasomnias."
            ),
            "behavioral_sleep_support": (
                "Protective behavioral structure and sleep-supportive routines."
            ),
            "circadian_stabilization": (
                "Protective circadian regularization and zeitgeber alignment."
            ),
            "gabaergic_hypnotic_support": (
                "Short-term GABA-A enhancing hypnotic support; modeled as sleep-promoting but with downstream trade-off risk."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "circadian_instability": (
                "Disordered timing signal consistent with clock-gene and SCN-related dysregulation."
            ),
            "hpa_stress_sensitization": (
                "Stress-linked physiological sensitization that can sustain insomnia vulnerability."
            ),
            "ascending_arousal_drive": (
                "Persistent orexinergic, histaminergic, and monoaminergic wake-promoting pressure."
            ),
            "gaba_sleep_drive_failure": (
                "Reduced effectiveness of VLPO-centered GABAergic sleep-promoting inhibition."
            ),
            "central_hyperarousal": (
                "State-level CNS hyperarousal spanning wake and sleep."
            ),
            "sleep_state_misperception": (
                "Mismatch between subjective sleep complaint and modest PSG findings."
            ),
            "hippocampal_stress_memory_burden": (
                "Memory-stress burden linked to poor sleep and hippocampal vulnerability."
            ),
            "prefrontal_regulatory_burden": (
                "Executive and emotion-regulation burden consistent with prefrontal dysfunction."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "sleep_initiation_difficulty": "Difficulty initiating sleep despite opportunity.",
            "sleep_maintenance_difficulty": "Difficulty maintaining consolidated sleep.",
            "fragmented_poor_quality_sleep": "Poor continuity or perceived poor sleep quality.",
            "nonrestorative_sleep": "Sleep that fails to restore daytime function.",
            "daytime_impairment": "Daytime functional impairment attributable to insomnia.",
            "executive_emotional_dysfunction": (
                "Executive and emotion-regulation problems associated with insomnia burden."
            ),
            "subjective_objective_sleep_discrepancy": (
                "Subjective complaint exceeds standard PSG abnormalities."
            ),
            "hypnotic_tolerance_side_effect_risk": (
                "Longer-term trade-off risk from hypnotic exposure, including tolerance, dependence, or cognitive side effects."
            ),
        }

        self.region_candidates: Dict[str, List[str]] = {
            "vlpo_hypothalamus_proxy": [
                "ventrolateral preoptic",
                "VLPO",
                "preoptic",
                "hypothalamus",
            ],
            "scn_proxy": [
                "suprachiasmatic",
                "SCN",
                "hypothalamus",
            ],
            "hippocampus": [
                "CA1 left",
                "CA3 left",
                "DG left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "pfc_control": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 9/46v left",
                "Area Fp1 (FPole) left",
                "Area Fp2 (FPole) left",
                "prefrontal cortex",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "vlpo_hypothalamus_proxy": (
                "Proxy for VLPO-centered hypothalamic sleep-promotion."
            ),
            "scn_proxy": "Proxy for suprachiasmatic circadian timing hub.",
            "hippocampus": "Atlas-backed hippocampal memory/stress node.",
            "pfc_control": "Atlas-backed or proxy prefrontal regulatory control node.",
        }

        self.edge_table: List[Dict[str, Any]] = [
            {
                "source": "genetic_vulnerability",
                "target": "circadian_instability",
                "relation": "increases susceptibility to clock-related sleep timing dysregulation",
                "insomnia_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "genetic_vulnerability",
                "target": "ascending_arousal_drive",
                "relation": "biases the system toward wake-promoting hyperarousal",
                "insomnia_change": "increased",
                "weight": 0.18,
            },
            {
                "source": "circadian_gene_disruption",
                "target": "circadian_instability",
                "relation": "destabilizes molecular clock timing",
                "insomnia_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "psychosocial_stress_load",
                "target": "hpa_stress_sensitization",
                "relation": "sensitizes stress physiology",
                "insomnia_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "psychosocial_stress_load",
                "target": "ascending_arousal_drive",
                "relation": "boosts wake-promoting arousal systems",
                "insomnia_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "comorbid_sleep_disorder_load",
                "target": "sleep_maintenance_difficulty",
                "relation": "disrupts sleep continuity",
                "insomnia_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "behavioral_sleep_support",
                "target": "central_hyperarousal",
                "relation": "reduces behavioral and physiological perpetuation of insomnia",
                "insomnia_change": "decreased",
                "weight": -0.20,
            },
            {
                "source": "circadian_stabilization",
                "target": "circadian_instability",
                "relation": "regularizes timing signals and entrainment",
                "insomnia_change": "decreased",
                "weight": -0.30,
            },
            {
                "source": "gabaergic_hypnotic_support",
                "target": "gaba_sleep_drive_failure",
                "relation": "partly compensates for impaired sleep-promoting inhibition",
                "insomnia_change": "decreased",
                "weight": -0.35,
            },
            {
                "source": "ascending_arousal_drive",
                "target": "gaba_sleep_drive_failure",
                "relation": "opposes VLPO-centered inhibitory sleep drive",
                "insomnia_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "circadian_instability",
                "target": "scn_proxy",
                "relation": "maps to dysfunctional circadian timing hub state",
                "insomnia_change": "increased",
                "weight": 0.55,
            },
            {
                "source": "gaba_sleep_drive_failure",
                "target": "vlpo_hypothalamus_proxy",
                "relation": "maps to impaired sleep-promoting hypothalamic switch function",
                "insomnia_change": "increased",
                "weight": 0.60,
            },
            {
                "source": "central_hyperarousal",
                "target": "pfc_control",
                "relation": "burdens prefrontal regulation during chronic insomnia",
                "insomnia_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "hpa_stress_sensitization",
                "target": "hippocampus",
                "relation": "increases hippocampal stress-memory burden",
                "insomnia_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "vlpo_hypothalamus_proxy",
                "target": "sleep_initiation_difficulty",
                "relation": "impairs transition into sleep",
                "insomnia_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "scn_proxy",
                "target": "sleep_initiation_difficulty",
                "relation": "mistimed circadian drive delays or destabilizes sleep onset",
                "insomnia_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "central_hyperarousal",
                "target": "sleep_maintenance_difficulty",
                "relation": "sustains awakenings and shallow sleep",
                "insomnia_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "sleep_maintenance_difficulty",
                "target": "fragmented_poor_quality_sleep",
                "relation": "reduces continuity and perceived quality",
                "insomnia_change": "increased",
                "weight": 0.50,
            },
            {
                "source": "fragmented_poor_quality_sleep",
                "target": "nonrestorative_sleep",
                "relation": "prevents restorative sleep",
                "insomnia_change": "increased",
                "weight": 0.55,
            },
            {
                "source": "hippocampus",
                "target": "daytime_impairment",
                "relation": "contributes to memory-related daytime burden",
                "insomnia_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "pfc_control",
                "target": "executive_emotional_dysfunction",
                "relation": "drives executive and emotion-regulation burden",
                "insomnia_change": "increased",
                "weight": 0.50,
            },
            {
                "source": "sleep_state_misperception",
                "target": "subjective_objective_sleep_discrepancy",
                "relation": "amplifies mismatch between complaint and standard PSG measures",
                "insomnia_change": "increased",
                "weight": 0.60,
            },
            {
                "source": "gabaergic_hypnotic_support",
                "target": "hypnotic_tolerance_side_effect_risk",
                "relation": "raises longer-term trade-off risk with sustained exposure",
                "insomnia_change": "increased",
                "weight": 0.60,
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
        # Prefer parcellation-native search, then atlas-level fallbacks.
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
        generic_penalty = 1 if name in {"hippocampus", "prefrontal cortex", "hypothalamus"} else 0
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

        # Depending on siibra version, the compound feature can expose data
        # directly, or individual elements can.
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

        Duplicate row/column labels in siibra-derived matrices can make a single
        label lookup return a DataFrame rather than a Series. In that case we
        average across the duplicated axis to obtain one profile.
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

        When both the row and column labels are duplicated, pandas returns a
        DataFrame. We average all numeric entries so the circuit table stays
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
                warnings.warn(f"Could not resolve a Julich region for '{key}'. Keeping it as a proxy node.")
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
            # Attempt late resolution for convenience.
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
        circadian_gene_disruption: float = 0.40,
        psychosocial_stress_load: float = 0.55,
        comorbid_sleep_disorder_load: float = 0.20,
        behavioral_sleep_support: float = 0.35,
        circadian_stabilization: float = 0.30,
        gabaergic_hypnotic_support: float = 0.10,
    ) -> Dict[str, pd.Series]:
        """
        Run a transparent normalized simulation.

        Higher values indicate more burden for the risk inputs and stronger
        support for the protective inputs.
        """
        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "circadian_gene_disruption": self._clip01(circadian_gene_disruption),
                "psychosocial_stress_load": self._clip01(psychosocial_stress_load),
                "comorbid_sleep_disorder_load": self._clip01(comorbid_sleep_disorder_load),
                "behavioral_sleep_support": self._clip01(behavioral_sleep_support),
                "circadian_stabilization": self._clip01(circadian_stabilization),
                "gabaergic_hypnotic_support": self._clip01(gabaergic_hypnotic_support),
            },
            name="inputs",
        )

        # Inputs -> latent biology
        circadian_instability = self._clip01(
            0.50 * inputs["circadian_gene_disruption"]
            + 0.22 * inputs["genetic_vulnerability"]
            + 0.12 * inputs["comorbid_sleep_disorder_load"]
            - 0.28 * inputs["circadian_stabilization"]
            - 0.10 * inputs["behavioral_sleep_support"]
        )

        hpa_stress_sensitization = self._clip01(
            0.48 * inputs["psychosocial_stress_load"]
            + 0.20 * inputs["genetic_vulnerability"]
            + 0.10 * circadian_instability
            - 0.18 * inputs["behavioral_sleep_support"]
        )

        ascending_arousal_drive = self._clip01(
            0.30 * inputs["psychosocial_stress_load"]
            + 0.25 * circadian_instability
            + 0.20 * inputs["genetic_vulnerability"]
            + 0.15 * inputs["comorbid_sleep_disorder_load"]
            - 0.12 * inputs["behavioral_sleep_support"]
            - 0.08 * inputs["circadian_stabilization"]
        )

        gaba_sleep_drive_failure = self._clip01(
            0.36 * ascending_arousal_drive
            + 0.18 * hpa_stress_sensitization
            + 0.12 * circadian_instability
            - 0.38 * inputs["gabaergic_hypnotic_support"]
            - 0.10 * inputs["behavioral_sleep_support"]
        )

        central_hyperarousal = self._clip01(
            0.34 * ascending_arousal_drive
            + 0.28 * hpa_stress_sensitization
            + 0.18 * circadian_instability
            + 0.10 * inputs["comorbid_sleep_disorder_load"]
            - 0.15 * inputs["behavioral_sleep_support"]
            - 0.10 * inputs["gabaergic_hypnotic_support"]
        )

        sleep_state_misperception = self._clip01(
            0.42 * central_hyperarousal
            + 0.22 * hpa_stress_sensitization
            + 0.10 * circadian_instability
            - 0.10 * inputs["behavioral_sleep_support"]
        )

        hippocampal_stress_memory_burden = self._clip01(
            0.35 * hpa_stress_sensitization
            + 0.25 * central_hyperarousal
            + 0.12 * circadian_instability
            - 0.10 * inputs["behavioral_sleep_support"]
        )

        prefrontal_regulatory_burden = self._clip01(
            0.35 * central_hyperarousal
            + 0.18 * circadian_instability
            + 0.20 * hpa_stress_sensitization
            - 0.18 * inputs["behavioral_sleep_support"]
        )

        latents = pd.Series(
            {
                "circadian_instability": circadian_instability,
                "hpa_stress_sensitization": hpa_stress_sensitization,
                "ascending_arousal_drive": ascending_arousal_drive,
                "gaba_sleep_drive_failure": gaba_sleep_drive_failure,
                "central_hyperarousal": central_hyperarousal,
                "sleep_state_misperception": sleep_state_misperception,
                "hippocampal_stress_memory_burden": hippocampal_stress_memory_burden,
                "prefrontal_regulatory_burden": prefrontal_regulatory_burden,
            },
            name="latents",
        )

        # Latent biology -> regional dysregulation state
        regional_state = pd.Series(
            {
                "vlpo_hypothalamus_proxy": self._clip01(
                    0.48 * gaba_sleep_drive_failure
                    + 0.22 * ascending_arousal_drive
                    + 0.10 * central_hyperarousal
                    - 0.12 * inputs["gabaergic_hypnotic_support"]
                ),
                "scn_proxy": self._clip01(
                    0.60 * circadian_instability
                    + 0.12 * central_hyperarousal
                    - 0.18 * inputs["circadian_stabilization"]
                ),
                "hippocampus": self._clip01(
                    0.55 * hippocampal_stress_memory_burden
                    + 0.12 * central_hyperarousal
                    + 0.08 * inputs["genetic_vulnerability"]
                ),
                "pfc_control": self._clip01(
                    0.58 * prefrontal_regulatory_burden
                    + 0.12 * central_hyperarousal
                    + 0.08 * circadian_instability
                ),
            },
            name="regional_state",
        )

        # Regional burden -> symptoms
        sleep_initiation_difficulty = self._clip01(
            0.36 * regional_state["vlpo_hypothalamus_proxy"]
            + 0.22 * regional_state["scn_proxy"]
            + 0.24 * ascending_arousal_drive
            + 0.08 * central_hyperarousal
            - 0.10 * inputs["gabaergic_hypnotic_support"]
        )

        sleep_maintenance_difficulty = self._clip01(
            0.36 * central_hyperarousal
            + 0.22 * inputs["comorbid_sleep_disorder_load"]
            + 0.18 * regional_state["scn_proxy"]
            + 0.10 * regional_state["vlpo_hypothalamus_proxy"]
            - 0.08 * inputs["gabaergic_hypnotic_support"]
        )

        fragmented_poor_quality_sleep = self._clip01(
            0.44 * sleep_maintenance_difficulty
            + 0.18 * circadian_instability
            + 0.12 * central_hyperarousal
            + 0.12 * inputs["comorbid_sleep_disorder_load"]
        )

        nonrestorative_sleep = self._clip01(
            0.40 * fragmented_poor_quality_sleep
            + 0.18 * central_hyperarousal
            + 0.12 * sleep_initiation_difficulty
        )

        executive_emotional_dysfunction = self._clip01(
            0.40 * regional_state["pfc_control"]
            + 0.18 * regional_state["hippocampus"]
            + 0.16 * hpa_stress_sensitization
            + 0.10 * nonrestorative_sleep
        )

        daytime_impairment = self._clip01(
            0.34 * nonrestorative_sleep
            + 0.24 * executive_emotional_dysfunction
            + 0.18 * regional_state["hippocampus"]
            + 0.08 * sleep_maintenance_difficulty
        )

        subjective_objective_sleep_discrepancy = self._clip01(
            0.52 * sleep_state_misperception
            + 0.18 * central_hyperarousal
            + 0.08 * sleep_initiation_difficulty
        )

        hypnotic_tolerance_side_effect_risk = self._clip01(
            0.62 * inputs["gabaergic_hypnotic_support"]
            + 0.10 * daytime_impairment
            - 0.08 * inputs["behavioral_sleep_support"]
        )

        symptoms = pd.Series(
            {
                "sleep_initiation_difficulty": sleep_initiation_difficulty,
                "sleep_maintenance_difficulty": sleep_maintenance_difficulty,
                "fragmented_poor_quality_sleep": fragmented_poor_quality_sleep,
                "nonrestorative_sleep": nonrestorative_sleep,
                "daytime_impairment": daytime_impairment,
                "executive_emotional_dysfunction": executive_emotional_dysfunction,
                "subjective_objective_sleep_discrepancy": subjective_objective_sleep_discrepancy,
                "hypnotic_tolerance_side_effect_risk": hypnotic_tolerance_side_effect_risk,
            },
            name="symptoms",
        )

        phenotypes = pd.Series(
            {
                "hyperarousal_insomnia_profile": self._clip01(
                    0.35 * central_hyperarousal
                    + 0.25 * sleep_initiation_difficulty
                    + 0.20 * sleep_maintenance_difficulty
                    + 0.20 * nonrestorative_sleep
                ),
                "circadian_insomnia_profile": self._clip01(
                    0.40 * circadian_instability
                    + 0.20 * regional_state["scn_proxy"]
                    + 0.20 * sleep_initiation_difficulty
                    + 0.20 * fragmented_poor_quality_sleep
                ),
                "cognitive_daytime_burden": self._clip01(
                    0.40 * daytime_impairment
                    + 0.30 * executive_emotional_dysfunction
                    + 0.15 * regional_state["pfc_control"]
                    + 0.15 * regional_state["hippocampus"]
                ),
                "sleep_state_misperception_profile": self._clip01(
                    0.45 * subjective_objective_sleep_discrepancy
                    + 0.35 * sleep_state_misperception
                    + 0.20 * central_hyperarousal
                ),
                "hypnotic_tradeoff_profile": self._clip01(
                    0.50 * hypnotic_tolerance_side_effect_risk
                    + 0.25 * sleep_initiation_difficulty
                    + 0.25 * sleep_maintenance_difficulty
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
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    model = InsomniaDisorderModel()
    built = model.build(gene_panel=INSOMNIA_GENE_PANEL[:10], connectivity_rows=10)

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

    for key in ("hippocampus", "pfc_control", "vlpo_hypothalamus_proxy", "scn_proxy"):
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
        genetic_vulnerability=0.55,
        circadian_gene_disruption=0.60,
        psychosocial_stress_load=0.70,
        comorbid_sleep_disorder_load=0.20,
        behavioral_sleep_support=0.30,
        circadian_stabilization=0.25,
        gabaergic_hypnotic_support=0.15,
    )

    print("\n=== SIMULATION: LATENTS ===")
    print(sim["latents"].sort_values(ascending=False).to_string())

    print("\n=== SIMULATION: SYMPTOMS ===")
    print(sim["symptoms"].sort_values(ascending=False).to_string())

    print("\n=== SIMULATION: PHENOTYPES ===")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Example coordinate assignment:
    # assignments = model.assign_mni_point((0, -10, -8))
    # print(assignments.head())

    # Example region mask retrieval:
    # mask_img = model.region_mask("hippocampus")
    # print(mask_img)
