from __future__ import annotations

"""
Obstructive sleep apnea hypopnea (OSAH) siibra scaffold.

This research scaffold translates a chapter-level biological summary of
obstructive sleep apnea hypopnea into a transparent mechanistic graph with
optional atlas-backed region resolution via siibra.

The model emphasizes that OSAH is not only an airway mechanics problem, but a
state-dependent disorder of central nervous system control over upper-airway
patency and respiratory stability. It is intended for exploratory modeling,
not diagnosis or treatment.
"""

import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore

    SIIBRA_AVAILABLE = True
except Exception:  # pragma: no cover - depends on runtime environment
    siibra = None  # type: ignore
    SIIBRA_AVAILABLE = False


DEFAULT_GENE_PANEL = [
    "HTR2A",
    "HTR2C",
    "SLC6A4",
    "SLC6A2",
    "DBH",
    "TH",
    "DRD2",
    "GABRA1",
    "GABRB3",
    "LEP",
    "LEPR",
    "PPARG",
    "APOE",
    "BDNF",
    "HIF1A",
    "NFE2L2",
    "IL6",
    "TNF",
]


class ObstructiveSleepApneaHypopneaModel:
    """
    Atlas-grounded scaffold for obstructive sleep apnea hypopnea biology.

    Chapter logic encoded here emphasizes:
    - sleep-state withdrawal of serotonergic and noradrenergic drive from the
      brainstem, which reduces upper-airway dilator tone,
    - GABAergic and hypnotic muscle-relaxant burden that can worsen airway
      collapsibility,
    - genetic contributions to craniofacial risk, body fat distribution,
      neuromuscular control, ventilatory stability, and arousal threshold,
    - chronic intermittent hypoxia causing inflammation, oxidative stress, and
      small-vessel / white-matter injury,
    - downstream hippocampal and frontal-system vulnerability associated with
      attention, executive, memory, mood, and daytime sleepiness symptoms.

    Several brainstem and airway-control constructs remain explicit proxies,
    because the source chapter is systems-level rather than nucleus-specific.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.disorder_name = "Obstructive Sleep Apnea Hypopnea"
        self.domain_key = "osah"
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec

        self.atlas = None
        self.parcellation = None
        self.space = None

        if SIIBRA_AVAILABLE:
            try:
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
            except Exception as exc:  # pragma: no cover - runtime dependent
                warnings.warn(
                    f"siibra atlas initialization failed; atlas-backed features will be limited: {exc}"
                )
        else:
            warnings.warn(
                "siibra is not installed in this environment; atlas-backed region resolution, "
                "feature queries, and coordinate assignments will return empty results. "
                "The simulator remains usable."
            )

        self.region_candidates: Dict[str, List[str]] = {
            "brainstem_monoaminergic_proxy": [
                "brainstem",
                "pons",
                "medulla oblongata",
                "midbrain",
                "mesencephalon",
            ],
            "frontal_control_proxy": [
                "Area 9/46d (DLPFC) left",
                "Area 9/46v (DLPFC) left",
                "Area 46 left",
                "Area 9 left",
                "middle frontal gyrus",
                "dorsolateral prefrontal cortex",
                "frontal lobe",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "CA2 (Hippocampus) left",
                "DG (Hippocampus) left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "upper_airway_anatomical_load": (
                "Anatomical narrowing burden from obesity, fat distribution, or craniofacial susceptibility."
            ),
            "genetic_vulnerability": (
                "Polygenic susceptibility influencing anatomy, neuromuscular airway control, loop gain, and arousal threshold."
            ),
            "sleep_state_monoamine_withdrawal": (
                "Loss of wake-like serotonergic and noradrenergic excitatory drive during NREM and especially REM sleep."
            ),
            "gabaergic_hypnotic_load": (
                "Muscle-relaxant and inhibitory burden from benzodiazepine-like or Z-medication-like GABAergic enhancement."
            ),
            "ventilatory_instability_load": (
                "Instability of respiratory control, including loop-gain related susceptibility to recurring events."
            ),
            "intermittent_hypoxia_load": (
                "Cumulative nocturnal oxygen desaturation burden reflecting repeated hypoxic stress."
            ),
            "airway_patency_support": (
                "Protective support from interventions or behaviors that help maintain airway patency and reduce hypoxic burden."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "upper_airway_dilator_failure": (
                "Failure of upper-airway motor output to maintain pharyngeal patency during sleep."
            ),
            "respiratory_control_instability": (
                "Unstable respiratory control that amplifies recurring obstructive events."
            ),
            "arousal_threshold_dysregulation": (
                "Trait-level dysregulation of the arousal threshold affecting event termination and sleep continuity."
            ),
            "sleep_state_airway_collapsibility": (
                "State-dependent increase in airway collapsibility across sleep stages."
            ),
            "intermittent_hypoxia_inflammatory_stress": (
                "Inflammatory and oxidative stress burden generated by chronic intermittent hypoxia."
            ),
            "small_vessel_white_matter_injury": (
                "White-matter hyperintensity and axonal/demyelinating burden consistent with small-vessel ischemic injury."
            ),
            "frontal_hippocampal_network_vulnerability": (
                "Vulnerability of memory and executive networks centered on hippocampal and frontal systems."
            ),
            "distributed_osah_neural_burden": (
                "Overall central nervous system burden integrating airway-control failure, hypoxia, and network injury."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "apnea_hypopnea_event_burden": "Burden of recurring obstructive apneas and hypopneas during sleep.",
            "sleep_fragmentation": "Repeated sleep interruption and unstable sleep continuity due to respiratory events and arousals.",
            "excessive_daytime_sleepiness": "Daytime sleepiness emerging from fragmented sleep and arousal-network burden.",
            "fatigue": "Persistent fatigue and reduced daytime energy.",
            "concentration_impairment": "Impaired concentration and sustained attention.",
            "memory_executive_impairment": "Memory and executive dysfunction linked to hypoxia and disconnection burden.",
            "mood_disturbance": "Mood symptoms associated with sleep disruption and chronic neural stress.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "upper_airway_anatomical_load",
                "target": "upper_airway_dilator_failure",
                "relation": "increases the mechanical burden that neuromuscular airway support must overcome",
                "osah_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "upper_airway_dilator_failure",
                "relation": "raises susceptibility through anatomy and neural control endophenotypes",
                "osah_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "respiratory_control_instability",
                "relation": "contributes to inherited variation in ventilatory control stability",
                "osah_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "arousal_threshold_dysregulation",
                "relation": "contributes to inherited differences in arousal threshold traits",
                "osah_change": "increased",
            },
            {
                "source": "sleep_state_monoamine_withdrawal",
                "target": "upper_airway_dilator_failure",
                "relation": "withdraws serotonergic and noradrenergic excitation from airway motor neurons during sleep",
                "osah_change": "increased",
            },
            {
                "source": "gabaergic_hypnotic_load",
                "target": "upper_airway_dilator_failure",
                "relation": "further relaxes upper-airway dilator tone",
                "osah_change": "increased",
            },
            {
                "source": "gabaergic_hypnotic_load",
                "target": "arousal_threshold_dysregulation",
                "relation": "alters arousal dynamics while increasing inhibitory burden",
                "osah_change": "increased",
            },
            {
                "source": "ventilatory_instability_load",
                "target": "respiratory_control_instability",
                "relation": "raises loop-gain-like respiratory instability",
                "osah_change": "increased",
            },
            {
                "source": "upper_airway_dilator_failure",
                "target": "brainstem_monoaminergic_proxy",
                "relation": "maps loss of central monoaminergic support onto brainstem airway-control circuitry",
                "osah_change": "increased",
            },
            {
                "source": "upper_airway_dilator_failure",
                "target": "sleep_state_airway_collapsibility",
                "relation": "permits pharyngeal collapse during sleep",
                "osah_change": "increased",
            },
            {
                "source": "respiratory_control_instability",
                "target": "sleep_state_airway_collapsibility",
                "relation": "amplifies recurring instability of breathing events",
                "osah_change": "increased",
            },
            {
                "source": "arousal_threshold_dysregulation",
                "target": "sleep_state_airway_collapsibility",
                "relation": "modulates how easily obstructive events terminate or recur",
                "osah_change": "increased",
            },
            {
                "source": "sleep_state_airway_collapsibility",
                "target": "apnea_hypopnea_event_burden",
                "relation": "produces the recurrent respiratory events that define syndrome severity",
                "osah_change": "increased",
            },
            {
                "source": "sleep_state_airway_collapsibility",
                "target": "sleep_fragmentation",
                "relation": "triggers repeated arousals and disrupted sleep continuity",
                "osah_change": "increased",
            },
            {
                "source": "intermittent_hypoxia_load",
                "target": "intermittent_hypoxia_inflammatory_stress",
                "relation": "drives chronic inflammatory and oxidative stress pathways",
                "osah_change": "increased",
            },
            {
                "source": "apnea_hypopnea_event_burden",
                "target": "intermittent_hypoxia_inflammatory_stress",
                "relation": "adds recurrent desaturation burden to inflammatory and oxidative stress cascades",
                "osah_change": "increased",
            },
            {
                "source": "intermittent_hypoxia_inflammatory_stress",
                "target": "small_vessel_white_matter_injury",
                "relation": "promotes demyelinating and axonal white-matter injury",
                "osah_change": "increased",
            },
            {
                "source": "small_vessel_white_matter_injury",
                "target": "frontal_hippocampal_network_vulnerability",
                "relation": "disconnects executive and memory networks",
                "osah_change": "increased",
            },
            {
                "source": "frontal_hippocampal_network_vulnerability",
                "target": "frontal_control_proxy",
                "relation": "maps executive vulnerability onto frontal control systems",
                "osah_change": "increased",
            },
            {
                "source": "frontal_hippocampal_network_vulnerability",
                "target": "hippocampus",
                "relation": "maps hypoxic-ischemic vulnerability onto memory circuitry",
                "osah_change": "increased",
            },
            {
                "source": "upper_airway_dilator_failure",
                "target": "distributed_osah_neural_burden",
                "relation": "adds central neuromuscular airway-control failure to overall disease burden",
                "osah_change": "increased",
            },
            {
                "source": "respiratory_control_instability",
                "target": "distributed_osah_neural_burden",
                "relation": "adds unstable respiratory control to overall disease burden",
                "osah_change": "increased",
            },
            {
                "source": "intermittent_hypoxia_inflammatory_stress",
                "target": "distributed_osah_neural_burden",
                "relation": "adds inflammatory and oxidative neural burden",
                "osah_change": "increased",
            },
            {
                "source": "small_vessel_white_matter_injury",
                "target": "distributed_osah_neural_burden",
                "relation": "adds connectivity injury to overall burden",
                "osah_change": "increased",
            },
            {
                "source": "distributed_osah_neural_burden",
                "target": "excessive_daytime_sleepiness",
                "relation": "contributes to daytime sleepiness and vigilance failure",
                "osah_change": "increased",
            },
            {
                "source": "sleep_fragmentation",
                "target": "excessive_daytime_sleepiness",
                "relation": "reduces restorative sleep and increases daytime somnolence",
                "osah_change": "increased",
            },
            {
                "source": "sleep_fragmentation",
                "target": "fatigue",
                "relation": "produces persistent daytime fatigue",
                "osah_change": "increased",
            },
            {
                "source": "frontal_control_proxy",
                "target": "concentration_impairment",
                "relation": "impairs attention and executive control",
                "osah_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "memory_executive_impairment",
                "relation": "contributes memory-system vulnerability",
                "osah_change": "increased",
            },
            {
                "source": "frontal_control_proxy",
                "target": "memory_executive_impairment",
                "relation": "contributes executive vulnerability",
                "osah_change": "increased",
            },
            {
                "source": "distributed_osah_neural_burden",
                "target": "mood_disturbance",
                "relation": "links chronic sleep disruption and neural stress with affective symptoms",
                "osah_change": "increased",
            },
            {
                "source": "airway_patency_support",
                "target": "upper_airway_dilator_failure",
                "relation": "partially offsets airway patency failure",
                "osah_change": "decreased",
            },
            {
                "source": "airway_patency_support",
                "target": "sleep_state_airway_collapsibility",
                "relation": "reduces the tendency of the airway to collapse",
                "osah_change": "decreased",
            },
            {
                "source": "airway_patency_support",
                "target": "intermittent_hypoxia_inflammatory_stress",
                "relation": "reduces downstream hypoxic stress",
                "osah_change": "decreased",
            },
            {
                "source": "airway_patency_support",
                "target": "excessive_daytime_sleepiness",
                "relation": "reduces daytime symptom burden",
                "osah_change": "decreased",
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

    def _quiet_context(self):
        return siibra.QUIET if SIIBRA_AVAILABLE else nullcontext()

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    def _modality_candidates(self, kind: str) -> List[Any]:
        cands: List[Any] = []
        if SIIBRA_AVAILABLE:
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
        if not SIIBRA_AVAILABLE or concept is None:
            return []
        for modality in modalities:
            try:
                with self._quiet_context():
                    feats = siibra.features.get(concept, modality, **kwargs)
                if feats:
                    return list(feats)
            except Exception:
                continue
        return []

    def _julich_matches(self, query: str) -> List[Any]:
        if self.atlas is None:
            return []
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {"hippocampus", "frontal lobe", "brainstem", "cerebral white matter"} else 0
        return (left_bonus, right_penalty, generic_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if self.atlas is None or self.parcellation is None:
            return None
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
                matches = sorted(matches, key=self._region_rank)
                return matches[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        seen = set()
        matches = self._julich_matches(keyword)
        for region in sorted(matches, key=self._region_rank):
            row = (
                self._name_of(region),
                getattr(region, "identifier", None),
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
        if region is None or self.space is None:
            return []
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
        centroid_xyz = None
        if centroid is not None:
            try:
                centroid_xyz = tuple(float(x) for x in centroid)
            except Exception:
                try:
                    centroid_xyz = tuple(float(x) for x in centroid.coordinate)
                except Exception:
                    centroid_xyz = None
        volume = getattr(main, "volume", None)
        volume_mm3 = float(volume) if volume is not None else None
        return centroid_xyz, volume_mm3

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        try:
            df = feats[0].data.copy().reset_index()
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
            df = feats[0].data.copy()
        except Exception:
            return pd.DataFrame()

        lower_cols = {str(c).lower(): c for c in df.columns}
        required = {"gene", "level", "zscore"}
        if required.issubset(lower_cols):
            gene_col = lower_cols["gene"]
            level_col = lower_cols["level"]
            zscore_col = lower_cols["zscore"]
            return (
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
        return df.reset_index(drop=True)

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix
        if self.parcellation is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort),
            feats[0],
        )
        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass
        try:
            self._connectivity_matrix = compound[0].data.copy()
        except Exception:
            self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        exact = [x for x in labels if self._name_of(x) == getattr(region, "name", None)]
        if exact:
            return exact[0]
        region_name = getattr(region, "name", "").lower()
        fuzzy = [
            x
            for x in labels
            if region_name and (region_name in self._name_of(x).lower() or self._name_of(x).lower() in region_name)
        ]
        return fuzzy[0] if fuzzy else None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or region is None:
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
            df = df[df["connected_region"] != getattr(region, "name", None)].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _pairwise_connectivity_value(self, matrix: pd.DataFrame, region_a: Any, region_b: Any) -> Optional[float]:
        row_a = self._match_region_label(list(matrix.index), region_a)
        row_b = self._match_region_label(list(matrix.index), region_b)
        col_a = self._match_region_label(list(matrix.columns), region_a)
        col_b = self._match_region_label(list(matrix.columns), region_b)

        values: List[float] = []
        for row_label, col_label in ((row_a, col_b), (row_b, col_a)):
            if row_label is None or col_label is None:
                continue
            try:
                value = matrix.loc[row_label, col_label]
                if pd.notna(value):
                    values.append(float(value))
            except Exception:
                continue
        if not values:
            return None
        return round(sum(values) / len(values), 6)

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or len(self.region_objects) < 2:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        keys = list(self.region_objects.keys())
        for i, src_key in enumerate(keys):
            for tgt_key in keys[i + 1 :]:
                src_region = self.region_objects[src_key]
                tgt_region = self.region_objects[tgt_key]
                value = self._pairwise_connectivity_value(matrix, src_region, tgt_region)
                if value is None:
                    continue
                rows.append(
                    {
                        "source_key": src_key,
                        "source_region": getattr(src_region, "name", src_key),
                        "target_key": tgt_key,
                        "target_region": getattr(tgt_region, "name", tgt_key),
                        "value": value,
                    }
                )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
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
                warnings.warn(f"Could not resolve a Julich region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region",
                        "description": "Atlas-backed node or proxy unresolved in this environment.",
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
                    "label": getattr(region, "name", key),
                    "node_type": "region",
                    "description": "Atlas-backed circuit node.",
                    "atlas_region": getattr(region, "name", None),
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
        circuit_conn = self.circuit_connectivity()
        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": circuit_conn,
        }

    def simulate(
        self,
        upper_airway_anatomical_load: float = 0.55,
        genetic_vulnerability: float = 0.35,
        sleep_state_monoamine_withdrawal: float = 0.60,
        gabaergic_hypnotic_load: float = 0.20,
        ventilatory_instability_load: float = 0.40,
        intermittent_hypoxia_load: float = 0.50,
        airway_patency_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """
        One-pass normalized simulator.

        Values are interpreted as burden or support on a 0..1 scale.
        Higher regional-state values reflect greater dysfunction or burden.
        """
        inputs = pd.Series(
            {
                "upper_airway_anatomical_load": self._clip01(upper_airway_anatomical_load),
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "sleep_state_monoamine_withdrawal": self._clip01(sleep_state_monoamine_withdrawal),
                "gabaergic_hypnotic_load": self._clip01(gabaergic_hypnotic_load),
                "ventilatory_instability_load": self._clip01(ventilatory_instability_load),
                "intermittent_hypoxia_load": self._clip01(intermittent_hypoxia_load),
                "airway_patency_support": self._clip01(airway_patency_support),
            },
            name="inputs",
        )

        latents = pd.Series(
            {
                "upper_airway_dilator_failure": self._clip01(
                    0.25 * inputs["upper_airway_anatomical_load"]
                    + 0.20 * inputs["sleep_state_monoamine_withdrawal"]
                    + 0.15 * inputs["gabaergic_hypnotic_load"]
                    + 0.15 * inputs["genetic_vulnerability"]
                    - 0.20 * inputs["airway_patency_support"]
                ),
                "respiratory_control_instability": self._clip01(
                    0.40 * inputs["ventilatory_instability_load"]
                    + 0.20 * inputs["genetic_vulnerability"]
                    + 0.15 * inputs["intermittent_hypoxia_load"]
                    - 0.10 * inputs["airway_patency_support"]
                ),
                "arousal_threshold_dysregulation": self._clip01(
                    0.30 * inputs["genetic_vulnerability"]
                    + 0.20 * inputs["ventilatory_instability_load"]
                    + 0.20 * inputs["gabaergic_hypnotic_load"]
                    + 0.10 * inputs["sleep_state_monoamine_withdrawal"]
                    - 0.05 * inputs["airway_patency_support"]
                ),
            },
            name="latents",
        )

        latents.loc["sleep_state_airway_collapsibility"] = self._clip01(
            0.35 * latents["upper_airway_dilator_failure"]
            + 0.20 * inputs["upper_airway_anatomical_load"]
            + 0.15 * latents["respiratory_control_instability"]
            + 0.10 * latents["arousal_threshold_dysregulation"]
            + 0.10 * inputs["sleep_state_monoamine_withdrawal"]
            - 0.20 * inputs["airway_patency_support"]
        )
        latents.loc["intermittent_hypoxia_inflammatory_stress"] = self._clip01(
            0.35 * inputs["intermittent_hypoxia_load"]
            + 0.20 * latents["sleep_state_airway_collapsibility"]
            + 0.10 * latents["respiratory_control_instability"]
            - 0.15 * inputs["airway_patency_support"]
        )
        latents.loc["small_vessel_white_matter_injury"] = self._clip01(
            0.40 * latents["intermittent_hypoxia_inflammatory_stress"]
            + 0.20 * inputs["intermittent_hypoxia_load"]
            + 0.10 * inputs["genetic_vulnerability"]
            - 0.10 * inputs["airway_patency_support"]
        )
        latents.loc["frontal_hippocampal_network_vulnerability"] = self._clip01(
            0.30 * latents["small_vessel_white_matter_injury"]
            + 0.25 * latents["intermittent_hypoxia_inflammatory_stress"]
            + 0.15 * latents["sleep_state_airway_collapsibility"]
            + 0.10 * inputs["genetic_vulnerability"]
        )
        latents.loc["distributed_osah_neural_burden"] = self._clip01(
            0.20 * latents["upper_airway_dilator_failure"]
            + 0.15 * latents["respiratory_control_instability"]
            + 0.15 * latents["sleep_state_airway_collapsibility"]
            + 0.20 * latents["intermittent_hypoxia_inflammatory_stress"]
            + 0.15 * latents["small_vessel_white_matter_injury"]
            + 0.15 * latents["frontal_hippocampal_network_vulnerability"]
            - 0.10 * inputs["airway_patency_support"]
        )

        regional_state = pd.Series(
            {
                "brainstem_monoaminergic_proxy": self._clip01(
                    0.45 * latents["upper_airway_dilator_failure"]
                    + 0.20 * latents["respiratory_control_instability"]
                    + 0.15 * inputs["sleep_state_monoamine_withdrawal"]
                    + 0.10 * inputs["gabaergic_hypnotic_load"]
                    - 0.20 * inputs["airway_patency_support"]
                ),
                "frontal_control_proxy": self._clip01(
                    0.40 * latents["small_vessel_white_matter_injury"]
                    + 0.25 * latents["frontal_hippocampal_network_vulnerability"]
                    + 0.15 * latents["intermittent_hypoxia_inflammatory_stress"]
                    + 0.05 * latents["distributed_osah_neural_burden"]
                ),
                "hippocampus": self._clip01(
                    0.45 * latents["frontal_hippocampal_network_vulnerability"]
                    + 0.20 * latents["intermittent_hypoxia_inflammatory_stress"]
                    + 0.10 * latents["small_vessel_white_matter_injury"]
                    + 0.05 * inputs["intermittent_hypoxia_load"]
                ),
            },
            name="regional_state",
        )

        symptoms = pd.Series(
            {
                "apnea_hypopnea_event_burden": self._clip01(
                    0.40 * latents["sleep_state_airway_collapsibility"]
                    + 0.20 * latents["upper_airway_dilator_failure"]
                    + 0.15 * latents["respiratory_control_instability"]
                    + 0.10 * latents["arousal_threshold_dysregulation"]
                    - 0.20 * inputs["airway_patency_support"]
                ),
                "sleep_fragmentation": self._clip01(
                    0.30 * latents["sleep_state_airway_collapsibility"]
                    + 0.20 * latents["arousal_threshold_dysregulation"]
                    + 0.20 * latents["respiratory_control_instability"]
                    + 0.10 * regional_state["brainstem_monoaminergic_proxy"]
                    - 0.15 * inputs["airway_patency_support"]
                ),
            },
            name="symptoms",
        )
        symptoms.loc["excessive_daytime_sleepiness"] = self._clip01(
            0.30 * symptoms["sleep_fragmentation"]
            + 0.25 * latents["distributed_osah_neural_burden"]
            + 0.15 * latents["intermittent_hypoxia_inflammatory_stress"]
            + 0.10 * regional_state["brainstem_monoaminergic_proxy"]
            - 0.20 * inputs["airway_patency_support"]
        )
        symptoms.loc["fatigue"] = self._clip01(
            0.30 * symptoms["excessive_daytime_sleepiness"]
            + 0.20 * symptoms["sleep_fragmentation"]
            + 0.20 * latents["intermittent_hypoxia_inflammatory_stress"]
            + 0.10 * latents["distributed_osah_neural_burden"]
            - 0.10 * inputs["airway_patency_support"]
        )
        symptoms.loc["concentration_impairment"] = self._clip01(
            0.30 * regional_state["frontal_control_proxy"]
            + 0.20 * symptoms["excessive_daytime_sleepiness"]
            + 0.15 * latents["small_vessel_white_matter_injury"]
            + 0.10 * symptoms["sleep_fragmentation"]
        )
        symptoms.loc["memory_executive_impairment"] = self._clip01(
            0.25 * regional_state["frontal_control_proxy"]
            + 0.20 * regional_state["hippocampus"]
            + 0.15 * latents["small_vessel_white_matter_injury"]
            + 0.15 * latents["intermittent_hypoxia_inflammatory_stress"]
            + 0.10 * symptoms["concentration_impairment"]
        )
        symptoms.loc["mood_disturbance"] = self._clip01(
            0.25 * symptoms["fatigue"]
            + 0.20 * symptoms["excessive_daytime_sleepiness"]
            + 0.15 * latents["distributed_osah_neural_burden"]
            + 0.10 * regional_state["frontal_control_proxy"]
        )

        phenotypes = pd.Series(
            {
                "neuromuscular_collapse_dominant_profile": self._clip01(
                    0.35 * symptoms["apnea_hypopnea_event_burden"]
                    + 0.25 * latents["upper_airway_dilator_failure"]
                    + 0.20 * inputs["sleep_state_monoamine_withdrawal"]
                    + 0.10 * inputs["upper_airway_anatomical_load"]
                ),
                "hypoxic_white_matter_injury_profile": self._clip01(
                    0.35 * latents["small_vessel_white_matter_injury"]
                    + 0.25 * latents["intermittent_hypoxia_inflammatory_stress"]
                    + 0.20 * symptoms["concentration_impairment"]
                    + 0.20 * symptoms["memory_executive_impairment"]
                ),
                "daytime_neurobehavioral_impairment_profile": self._clip01(
                    0.25 * symptoms["excessive_daytime_sleepiness"]
                    + 0.20 * symptoms["fatigue"]
                    + 0.20 * symptoms["concentration_impairment"]
                    + 0.15 * symptoms["mood_disturbance"]
                    + 0.20 * symptoms["memory_executive_impairment"]
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
        if not SIIBRA_AVAILABLE:
            return pd.DataFrame()
        if self._pmap is None:
            try:
                with self._quiet_context():
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception:
                try:
                    with self._quiet_context():
                        self._pmap = self.atlas.get_map(
                            parcellation=self.parcellation,
                            space=self.assignment_space,
                            maptype="statistical",
                        )
                except Exception:
                    return pd.DataFrame()

        try:
            point = siibra.Point(tuple(float(x) for x in xyz), space=self.assignment_space)
            with self._quiet_context():
                assignments = self._pmap.assign(point)
        except Exception:
            return pd.DataFrame()

        if not isinstance(assignments, pd.DataFrame):
            try:
                assignments = pd.DataFrame(assignments)
            except Exception:
                return pd.DataFrame()

        for candidate in (
            "map value",
            "correlation",
            "intersection over union",
            "intersection_over_union",
            "probability",
        ):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        region = self.region_objects.get(node_key)
        if region is None or not SIIBRA_AVAILABLE:
            return None
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(space=self.assignment_space)
            except Exception:
                return None


if __name__ == "__main__":
    pd.set_option("display.width", 150)
    pd.set_option("display.max_columns", 12)

    model = ObstructiveSleepApneaHypopneaModel()
    bundle = model.build()

    print("\n=== Nodes (first 20) ===")
    print(bundle["nodes"].head(20).to_string(index=False))

    print("\n=== Edges ===")
    print(bundle["edges"].to_string(index=False))

    print("\n=== Resolved regions ===")
    if bundle["regions"]:
        for key, region in bundle["regions"].items():
            print(f"- {key}: {getattr(region, 'name', key)}")
    else:
        print("No atlas-backed regions resolved in this runtime.")

    print("\n=== Example multimodal tables ===")
    for region_key in list(bundle["regions"].keys())[:2]:
        print(f"\nRegion: {region_key}")
        print("Receptors:")
        print(bundle["receptors"].get(region_key, pd.DataFrame()).head().to_string(index=False))
        print("Genes:")
        print(bundle["genes"].get(region_key, pd.DataFrame()).head().to_string(index=False))
        print("Connectivity profile:")
        print(bundle["connectivity_profiles"].get(region_key, pd.DataFrame()).head().to_string(index=False))

    print("\n=== Circuit connectivity ===")
    print(bundle["circuit_connectivity"].head(10).to_string(index=False))

    print("\n=== Example simulation ===")
    sim = model.simulate(
        upper_airway_anatomical_load=0.70,
        genetic_vulnerability=0.50,
        sleep_state_monoamine_withdrawal=0.75,
        gabaergic_hypnotic_load=0.30,
        ventilatory_instability_load=0.45,
        intermittent_hypoxia_load=0.65,
        airway_patency_support=0.20,
    )
    for key, series in sim.items():
        print(f"\n[{key}]")
        print(series.to_string())

    # Example coordinate assignment if siibra is available:
    # print(model.assign_mni_point((-28, -18, -16)).head())
