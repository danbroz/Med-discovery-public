from __future__ import annotations

"""
Separation Anxiety Disorder siibra scaffold.

This script turns a chapter-level biological summary of Separation Anxiety
Disorder into a small, interpretable, atlas-grounded mechanistic model. It is
intended as a research scaffold only. It is not a diagnostic, prognostic, or
therapeutic system.

Design choices follow the chapter conservatively:
- direct SAD neurobiology is limited in the source text, so the scaffold uses
  anxiety, fear, and stress analogues only where the chapter explicitly does so,
- explicit anatomy is limited to structures named in the chapter or in the
  chapter's functional-imaging hypothesis (amygdala, hippocampus, mPFC/PFC),
- serotonergic, GABAergic, stress/epigenetic, and attachment-threat mechanisms
  are kept as latent processes rather than forced into artificial parcels,
- cortico-striato-thalamo-cortical involvement is represented with clearly
  labeled proxy nodes because stable Julich matches vary across environments.
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


SEPARATION_ANXIETY_DISORDER_GENE_PANEL = [
    # Serotonin signaling / reuptake
    "SLC6A4",
    "HTR1A",
    "HTR2A",
    "TPH2",
    # GABAergic inhibition
    "GAD1",
    "GAD2",
    "GABRA1",
    "GABRB2",
    "GABRG2",
    # Stress responsivity / epigenetic sensitivity / plasticity
    "NR3C1",
    "FKBP5",
    "CRHR1",
    "BDNF",
]

DEFAULT_GENE_PANEL = SEPARATION_ANXIETY_DISORDER_GENE_PANEL


class SeparationAnxietyDisorderModel:
    """
    Atlas-grounded research scaffold for Separation Anxiety Disorder.

    The simulator is deliberately simple and normalized to 0..1. Higher values
    generally indicate more dysregulation or symptom burden, except the support
    inputs where higher values indicate stronger protective support.
    """

    disorder_name = "Separation Anxiety Disorder"
    abbreviation = "separation_anxiety"

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
                "Polygenic liability increasing vulnerability to fear, anxiety, and stress dysregulation."
            ),
            "early_life_adversity": (
                "Early trauma, neglect, abuse, or loss that sensitizes later anxiety risk."
            ),
            "stressful_family_environment": (
                "Parental conflict, instability, or loss-related family stress that amplifies separation threat."
            ),
            "anxious_temperament": (
                "Inherited or developmentally shaped anxious temperament predisposing to separation fear."
            ),
            "separation_cue_load": (
                "Current burden of separation-related cues, transitions, or attachment threat signals."
            ),
            "family_stability_support": (
                "Protective caregiving stability, reassurance, and environmental safety support."
            ),
            "serotonergic_support": (
                "Protective support consistent with serotonergic treatment effects, such as SSRI-related normalization."
            ),
            "gabaergic_acute_support": (
                "Acute GABA-A enhancing anxiolytic support modeled as short-term fear inhibition with trade-off risk."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_dysregulation": (
                "Relative serotonergic dysfunction contributing to worry, hypervigilance, and dysphoric affect."
            ),
            "gabaergic_fear_inhibition_failure": (
                "Reduced inhibitory gating of fear and arousal consistent with GABAergic deficit."
            ),
            "stress_epigenetic_sensitization": (
                "Stress-linked and epigenetic sensitization through which adversity shapes later vulnerability."
            ),
            "attachment_threat_bias": (
                "Threat-biased attachment processing that makes separation cues feel excessively dangerous."
            ),
            "frontolimbic_control_failure": (
                "Weak top-down regulation of limbic fear circuitry, especially amygdala control by mPFC/PFC."
            ),
            "cstc_worry_perseveration": (
                "Persistent worry/perseveration pressure linked to cortico-striato-thalamo-cortical dysfunction."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "separation_related_fear": (
                "Excessive fear or distress linked to actual or anticipated separation."
            ),
            "anticipatory_anxiety": (
                "Excessive anticipatory anxiety when separation is expected."
            ),
            "hypervigilance_persistent_worry": (
                "Heightened vigilance and persistent worry around attachment threat and loss."
            ),
            "restlessness_tension": (
                "Tension, arousal, and restlessness associated with impaired fear inhibition."
            ),
            "dysphoric_distress": (
                "Dysphoric mood or distress coupled to anxiety and separation threat."
            ),
            "clinging_avoidant_behavior": (
                "Clinging, avoidance, or behavioral disruption aimed at preventing separation."
            ),
            "functional_impairment": (
                "Clinically significant interference in developmentally expected functioning."
            ),
            "benzodiazepine_tradeoff_risk": (
                "Longer-term risk from GABAergic acute support, including dependence or cognitive side effects."
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
            "pfc_control": [
                "Area s32 (MFC) left",
                "Area p32 (MFC) left",
                "Area 24dv (ACC) left",
                "Area Fp1 (FPole) left",
                "medial prefrontal cortex",
                "prefrontal cortex",
            ],
            "striatum_proxy": [
                "striatum left",
                "caudate left",
                "putamen left",
                "striatum",
            ],
            "thalamus_proxy": [
                "thalamus left",
                "thalamus",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "amygdala": "Atlas-backed fear and threat salience node.",
            "hippocampus": "Atlas-backed contextual memory and adversity-linked threat-generalization node.",
            "pfc_control": "Atlas-backed or proxy medial/prefrontal top-down control node.",
            "striatum_proxy": "Proxy for CSTC striatal involvement in persistent worry/perseveration.",
            "thalamus_proxy": "Proxy for CSTC thalamic relay involvement in persistent worry/perseveration.",
        }

        self.edge_table: List[Dict[str, Any]] = [
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_dysregulation",
                "relation": "raises liability for serotonergic anxiety dysregulation",
                "separation_anxiety_change": "increased",
                "weight": 0.28,
            },
            {
                "source": "genetic_vulnerability",
                "target": "gabaergic_fear_inhibition_failure",
                "relation": "raises liability for impaired inhibitory fear control",
                "separation_anxiety_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "genetic_vulnerability",
                "target": "stress_epigenetic_sensitization",
                "relation": "confers vulnerability that interacts with adversity",
                "separation_anxiety_change": "increased",
                "weight": 0.18,
            },
            {
                "source": "early_life_adversity",
                "target": "stress_epigenetic_sensitization",
                "relation": "sensitizes stress biology and long-term anxiety vulnerability",
                "separation_anxiety_change": "increased",
                "weight": 0.42,
            },
            {
                "source": "stressful_family_environment",
                "target": "stress_epigenetic_sensitization",
                "relation": "sustains stress-linked biological sensitization",
                "separation_anxiety_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "anxious_temperament",
                "target": "attachment_threat_bias",
                "relation": "biases the child toward perceiving separation as threatening",
                "separation_anxiety_change": "increased",
                "weight": 0.38,
            },
            {
                "source": "separation_cue_load",
                "target": "attachment_threat_bias",
                "relation": "amplifies salience of attachment-loss cues",
                "separation_anxiety_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "family_stability_support",
                "target": "attachment_threat_bias",
                "relation": "reduces perceived separation threat and environmental danger",
                "separation_anxiety_change": "decreased",
                "weight": -0.25,
            },
            {
                "source": "serotonergic_support",
                "target": "serotonergic_dysregulation",
                "relation": "partly compensates for serotonergic dysregulation",
                "separation_anxiety_change": "decreased",
                "weight": -0.32,
            },
            {
                "source": "gabaergic_acute_support",
                "target": "gabaergic_fear_inhibition_failure",
                "relation": "partly compensates for deficient inhibitory fear control",
                "separation_anxiety_change": "decreased",
                "weight": -0.34,
            },
            {
                "source": "stress_epigenetic_sensitization",
                "target": "serotonergic_dysregulation",
                "relation": "pushes anxiety circuitry toward persistent serotonergic imbalance",
                "separation_anxiety_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "stress_epigenetic_sensitization",
                "target": "gabaergic_fear_inhibition_failure",
                "relation": "weakens inhibitory control of arousal and fear",
                "separation_anxiety_change": "increased",
                "weight": 0.22,
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "frontolimbic_control_failure",
                "relation": "weakens regulation of worry and negative affect",
                "separation_anxiety_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "gabaergic_fear_inhibition_failure",
                "target": "frontolimbic_control_failure",
                "relation": "permits excessive fear arousal that overwhelms cortical regulation",
                "separation_anxiety_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "attachment_threat_bias",
                "target": "frontolimbic_control_failure",
                "relation": "loads separation threat into limbic-control circuitry",
                "separation_anxiety_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "cstc_worry_perseveration",
                "relation": "biases CSTC processing toward persistent worry",
                "separation_anxiety_change": "increased",
                "weight": 0.28,
            },
            {
                "source": "attachment_threat_bias",
                "target": "cstc_worry_perseveration",
                "relation": "sustains repetitive separation-related worry",
                "separation_anxiety_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "frontolimbic_control_failure",
                "target": "amygdala",
                "relation": "maps to heightened amygdala reactivity under failed top-down control",
                "separation_anxiety_change": "increased",
                "weight": 0.42,
            },
            {
                "source": "stress_epigenetic_sensitization",
                "target": "hippocampus",
                "relation": "maps adversity burden into contextual memory/threat circuitry",
                "separation_anxiety_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "frontolimbic_control_failure",
                "target": "pfc_control",
                "relation": "maps to mPFC/PFC regulatory burden or hypo-control",
                "separation_anxiety_change": "increased",
                "weight": 0.55,
            },
            {
                "source": "cstc_worry_perseveration",
                "target": "striatum_proxy",
                "relation": "maps to corticostriatal worry-perseveration burden",
                "separation_anxiety_change": "increased",
                "weight": 0.52,
            },
            {
                "source": "cstc_worry_perseveration",
                "target": "thalamus_proxy",
                "relation": "maps to thalamic relay burden in repetitive worry",
                "separation_anxiety_change": "increased",
                "weight": 0.48,
            },
            {
                "source": "amygdala",
                "target": "separation_related_fear",
                "relation": "drives fear responses to actual or anticipated separation",
                "separation_anxiety_change": "increased",
                "weight": 0.48,
            },
            {
                "source": "pfc_control",
                "target": "anticipatory_anxiety",
                "relation": "weakened top-down control amplifies anticipatory anxiety",
                "separation_anxiety_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "striatum_proxy",
                "target": "hypervigilance_persistent_worry",
                "relation": "supports repetitive worry and vigilance",
                "separation_anxiety_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "thalamus_proxy",
                "target": "hypervigilance_persistent_worry",
                "relation": "supports relay and reinforcement of worry-related threat processing",
                "separation_anxiety_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "gabaergic_fear_inhibition_failure",
                "target": "restlessness_tension",
                "relation": "produces tension and restlessness through impaired inhibition",
                "separation_anxiety_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "dysphoric_distress",
                "relation": "links anxiety with dysphoric mood burden",
                "separation_anxiety_change": "increased",
                "weight": 0.32,
            },
            {
                "source": "hippocampus",
                "target": "clinging_avoidant_behavior",
                "relation": "contributes contextual threat memory to avoidance and clinging",
                "separation_anxiety_change": "increased",
                "weight": 0.22,
            },
            {
                "source": "anticipatory_anxiety",
                "target": "functional_impairment",
                "relation": "interferes with age-expected functioning",
                "separation_anxiety_change": "increased",
                "weight": 0.28,
            },
            {
                "source": "clinging_avoidant_behavior",
                "target": "functional_impairment",
                "relation": "directly restricts daily participation and independence",
                "separation_anxiety_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "gabaergic_acute_support",
                "target": "benzodiazepine_tradeoff_risk",
                "relation": "raises longer-term dependence and cognitive side-effect risk",
                "separation_anxiety_change": "increased",
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
            "prefrontal cortex",
            "medial prefrontal cortex",
            "striatum",
            "thalamus",
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
        self,
        region: Any,
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
        average_axis: int,
    ) -> pd.Series:
        """
        Coerce a connectivity lookup to a numeric Series.

        Duplicate labels in connectivity matrices can make pandas return a
        DataFrame instead of a Series. We average across the duplicated axis so
        downstream sorting remains stable.
        """
        if isinstance(selection, pd.Series):
            series = pd.to_numeric(selection, errors="coerce")
            return series.dropna()

        if isinstance(selection, pd.DataFrame):
            numeric = selection.apply(pd.to_numeric, errors="coerce")
            if average_axis == 0:
                series = numeric.mean(axis=0)
            else:
                series = numeric.mean(axis=1)
            if not series.index.is_unique:
                series = series.groupby(level=0, sort=False).mean()
            return series.dropna()

        series = pd.Series(selection, dtype=float)
        if not series.index.is_unique:
            series = series.groupby(level=0, sort=False).mean()
        return series.dropna()

    @staticmethod
    def _connectivity_scalar(value: Any) -> Optional[float]:
        """
        Reduce a connectivity lookup to a scalar float.

        When both row and column labels are duplicated, pandas can return a
        DataFrame. We average numeric entries so the circuit table remains
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
        early_life_adversity: float = 0.45,
        stressful_family_environment: float = 0.50,
        anxious_temperament: float = 0.55,
        separation_cue_load: float = 0.60,
        family_stability_support: float = 0.35,
        serotonergic_support: float = 0.20,
        gabaergic_acute_support: float = 0.10,
    ) -> Dict[str, pd.Series]:
        """
        Run a transparent normalized simulation.

        Higher values indicate more burden for the risk inputs and stronger
        support for the protective inputs.
        """
        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "early_life_adversity": self._clip01(early_life_adversity),
                "stressful_family_environment": self._clip01(stressful_family_environment),
                "anxious_temperament": self._clip01(anxious_temperament),
                "separation_cue_load": self._clip01(separation_cue_load),
                "family_stability_support": self._clip01(family_stability_support),
                "serotonergic_support": self._clip01(serotonergic_support),
                "gabaergic_acute_support": self._clip01(gabaergic_acute_support),
            },
            name="inputs",
        )

        stress_epigenetic_sensitization = self._clip01(
            0.40 * inputs["early_life_adversity"]
            + 0.30 * inputs["stressful_family_environment"]
            + 0.18 * inputs["genetic_vulnerability"]
            + 0.08 * inputs["anxious_temperament"]
            - 0.18 * inputs["family_stability_support"]
        )

        serotonergic_dysregulation = self._clip01(
            0.30 * inputs["genetic_vulnerability"]
            + 0.20 * stress_epigenetic_sensitization
            + 0.18 * inputs["anxious_temperament"]
            + 0.10 * inputs["separation_cue_load"]
            - 0.30 * inputs["serotonergic_support"]
            - 0.08 * inputs["family_stability_support"]
        )

        gabaergic_fear_inhibition_failure = self._clip01(
            0.28 * stress_epigenetic_sensitization
            + 0.22 * inputs["anxious_temperament"]
            + 0.18 * inputs["genetic_vulnerability"]
            + 0.10 * inputs["separation_cue_load"]
            - 0.34 * inputs["gabaergic_acute_support"]
            - 0.08 * inputs["family_stability_support"]
        )

        attachment_threat_bias = self._clip01(
            0.30 * inputs["anxious_temperament"]
            + 0.24 * inputs["separation_cue_load"]
            + 0.18 * inputs["early_life_adversity"]
            + 0.16 * inputs["stressful_family_environment"]
            + 0.08 * inputs["genetic_vulnerability"]
            - 0.22 * inputs["family_stability_support"]
        )

        frontolimbic_control_failure = self._clip01(
            0.26 * serotonergic_dysregulation
            + 0.24 * gabaergic_fear_inhibition_failure
            + 0.22 * stress_epigenetic_sensitization
            + 0.18 * attachment_threat_bias
            - 0.12 * inputs["family_stability_support"]
            - 0.06 * inputs["serotonergic_support"]
        )

        cstc_worry_perseveration = self._clip01(
            0.30 * serotonergic_dysregulation
            + 0.22 * attachment_threat_bias
            + 0.18 * frontolimbic_control_failure
            + 0.12 * inputs["separation_cue_load"]
            - 0.10 * inputs["serotonergic_support"]
            - 0.06 * inputs["family_stability_support"]
        )

        latents = pd.Series(
            {
                "stress_epigenetic_sensitization": stress_epigenetic_sensitization,
                "serotonergic_dysregulation": serotonergic_dysregulation,
                "gabaergic_fear_inhibition_failure": gabaergic_fear_inhibition_failure,
                "attachment_threat_bias": attachment_threat_bias,
                "frontolimbic_control_failure": frontolimbic_control_failure,
                "cstc_worry_perseveration": cstc_worry_perseveration,
            },
            name="latents",
        )

        regional_state = pd.Series(
            {
                "amygdala": self._clip01(
                    0.42 * attachment_threat_bias
                    + 0.20 * frontolimbic_control_failure
                    + 0.18 * gabaergic_fear_inhibition_failure
                    + 0.10 * stress_epigenetic_sensitization
                    + 0.08 * inputs["separation_cue_load"]
                    - 0.08 * inputs["family_stability_support"]
                ),
                "hippocampus": self._clip01(
                    0.34 * stress_epigenetic_sensitization
                    + 0.24 * attachment_threat_bias
                    + 0.16 * frontolimbic_control_failure
                    + 0.10 * inputs["early_life_adversity"]
                    + 0.08 * inputs["separation_cue_load"]
                    - 0.06 * inputs["family_stability_support"]
                ),
                "pfc_control": self._clip01(
                    0.56 * frontolimbic_control_failure
                    + 0.14 * serotonergic_dysregulation
                    + 0.10 * cstc_worry_perseveration
                    + 0.08 * stress_epigenetic_sensitization
                    - 0.08 * inputs["family_stability_support"]
                    - 0.06 * inputs["serotonergic_support"]
                ),
                "striatum_proxy": self._clip01(
                    0.46 * cstc_worry_perseveration
                    + 0.18 * serotonergic_dysregulation
                    + 0.12 * attachment_threat_bias
                    + 0.06 * frontolimbic_control_failure
                ),
                "thalamus_proxy": self._clip01(
                    0.44 * cstc_worry_perseveration
                    + 0.16 * frontolimbic_control_failure
                    + 0.12 * stress_epigenetic_sensitization
                    + 0.06 * attachment_threat_bias
                ),
            },
            name="regional_state",
        )

        separation_related_fear = self._clip01(
            0.34 * regional_state["amygdala"]
            + 0.22 * attachment_threat_bias
            + 0.14 * regional_state["hippocampus"]
            + 0.12 * inputs["separation_cue_load"]
            + 0.08 * gabaergic_fear_inhibition_failure
            - 0.08 * inputs["family_stability_support"]
        )

        anticipatory_anxiety = self._clip01(
            0.28 * separation_related_fear
            + 0.22 * regional_state["pfc_control"]
            + 0.18 * regional_state["amygdala"]
            + 0.14 * cstc_worry_perseveration
            + 0.08 * stress_epigenetic_sensitization
            - 0.06 * inputs["family_stability_support"]
        )

        hypervigilance_persistent_worry = self._clip01(
            0.24 * regional_state["amygdala"]
            + 0.22 * regional_state["striatum_proxy"]
            + 0.18 * regional_state["thalamus_proxy"]
            + 0.16 * serotonergic_dysregulation
            + 0.10 * regional_state["pfc_control"]
            - 0.10 * inputs["serotonergic_support"]
        )

        restlessness_tension = self._clip01(
            0.34 * gabaergic_fear_inhibition_failure
            + 0.22 * hypervigilance_persistent_worry
            + 0.16 * anticipatory_anxiety
            + 0.10 * stress_epigenetic_sensitization
            - 0.08 * inputs["gabaergic_acute_support"]
        )

        dysphoric_distress = self._clip01(
            0.24 * serotonergic_dysregulation
            + 0.24 * anticipatory_anxiety
            + 0.18 * regional_state["amygdala"]
            + 0.10 * regional_state["pfc_control"]
            + 0.08 * separation_related_fear
            - 0.10 * inputs["serotonergic_support"]
        )

        clinging_avoidant_behavior = self._clip01(
            0.28 * separation_related_fear
            + 0.22 * anticipatory_anxiety
            + 0.18 * regional_state["hippocampus"]
            + 0.14 * attachment_threat_bias
            + 0.08 * hypervigilance_persistent_worry
            - 0.08 * inputs["family_stability_support"]
        )

        functional_impairment = self._clip01(
            0.30 * clinging_avoidant_behavior
            + 0.22 * anticipatory_anxiety
            + 0.18 * dysphoric_distress
            + 0.14 * hypervigilance_persistent_worry
            + 0.08 * restlessness_tension
        )

        benzodiazepine_tradeoff_risk = self._clip01(
            0.62 * inputs["gabaergic_acute_support"]
            + 0.10 * functional_impairment
            - 0.08 * inputs["family_stability_support"]
        )

        symptoms = pd.Series(
            {
                "separation_related_fear": separation_related_fear,
                "anticipatory_anxiety": anticipatory_anxiety,
                "hypervigilance_persistent_worry": hypervigilance_persistent_worry,
                "restlessness_tension": restlessness_tension,
                "dysphoric_distress": dysphoric_distress,
                "clinging_avoidant_behavior": clinging_avoidant_behavior,
                "functional_impairment": functional_impairment,
                "benzodiazepine_tradeoff_risk": benzodiazepine_tradeoff_risk,
            },
            name="symptoms",
        )

        phenotypes = pd.Series(
            {
                "separation_anxiety_core_profile": self._clip01(
                    0.30 * separation_related_fear
                    + 0.24 * anticipatory_anxiety
                    + 0.22 * clinging_avoidant_behavior
                    + 0.24 * functional_impairment
                ),
                "fear_hyperarousal_profile": self._clip01(
                    0.26 * separation_related_fear
                    + 0.22 * hypervigilance_persistent_worry
                    + 0.20 * restlessness_tension
                    + 0.16 * regional_state["amygdala"]
                    + 0.16 * stress_epigenetic_sensitization
                ),
                "worry_perseveration_profile": self._clip01(
                    0.30 * hypervigilance_persistent_worry
                    + 0.22 * cstc_worry_perseveration
                    + 0.18 * anticipatory_anxiety
                    + 0.16 * regional_state["striatum_proxy"]
                    + 0.14 * regional_state["thalamus_proxy"]
                ),
                "attachment_threat_profile": self._clip01(
                    0.30 * attachment_threat_bias
                    + 0.24 * separation_related_fear
                    + 0.22 * regional_state["amygdala"]
                    + 0.12 * regional_state["hippocampus"]
                    + 0.12 * clinging_avoidant_behavior
                ),
                "medication_tradeoff_profile": self._clip01(
                    0.50 * benzodiazepine_tradeoff_risk
                    + 0.20 * anticipatory_anxiety
                    + 0.15 * hypervigilance_persistent_worry
                    + 0.15 * restlessness_tension
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
    pd.set_option("display.max_columns", 24)

    model = SeparationAnxietyDisorderModel()
    built = model.build(gene_panel=SEPARATION_ANXIETY_DISORDER_GENE_PANEL[:10], connectivity_rows=10)

    print("\n=== NODES (head) ===")
    print(built["nodes"].head(14).to_string(index=False))

    print("\n=== EDGES (head) ===")
    print(built["edges"].head(14).to_string(index=False))

    print("\n=== RESOLVED REGIONS ===")
    if built["regions"]:
        for key, region in built["regions"].items():
            print(f"- {key}: {region.name}")
    else:
        print("No regions resolved in this environment.")

    for key in ("amygdala", "hippocampus", "pfc_control", "striatum_proxy", "thalamus_proxy"):
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
        early_life_adversity=0.45,
        stressful_family_environment=0.60,
        anxious_temperament=0.65,
        separation_cue_load=0.70,
        family_stability_support=0.30,
        serotonergic_support=0.20,
        gabaergic_acute_support=0.10,
    )

    print("\n=== SIMULATION: LATENTS ===")
    print(sim["latents"].sort_values(ascending=False).to_string())

    print("\n=== SIMULATION: SYMPTOMS ===")
    print(sim["symptoms"].sort_values(ascending=False).to_string())

    print("\n=== SIMULATION: PHENOTYPES ===")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Example coordinate assignment:
    # assignments = model.assign_mni_point((-24, -4, -18))
    # print(assignments.head())

    # Example region mask retrieval:
    # mask_img = model.region_mask("amygdala")
    # print(mask_img)
