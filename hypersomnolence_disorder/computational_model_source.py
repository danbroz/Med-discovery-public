from __future__ import annotations

"""
Hypersomnolence disorder siibra scaffold.

This research scaffold translates a chapter-level biological summary of
hypersomnolence disorder into a transparent mechanistic graph with optional
atlas-backed region resolution via siibra.

It is intended for exploratory modeling, not diagnosis or treatment.
"""

import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore

    SIIBRA_AVAILABLE = True
except Exception:  # pragma: no cover - import environment dependent
    siibra = None  # type: ignore
    SIIBRA_AVAILABLE = False


DEFAULT_GENE_PANEL = [
    "HCRT",
    "HCRTR2",
    "HDC",
    "HRH1",
    "SLC6A3",
    "DRD2",
    "SLC6A2",
    "DBH",
    "SLC6A4",
    "TPH2",
    "CHAT",
    "SLC5A7",
    "ARNTL",
    "PER2",
    "HLA-DQB1",
    "TCRA",
]


class HypersomnolenceDisorderModel:
    """
    Atlas-grounded scaffold for central hypersomnolence biology.

    Chapter logic encoded here emphasizes:
    - orexin/hypocretin-related wake instability,
    - broader monoaminergic, histaminergic, and cholinergic wake failure,
    - thalamocortical gating dysfunction and sleep fragmentation,
    - salience-network dysregulation contributing to poor alertness,
    - downstream excessive daytime sleepiness and unstable wakefulness.

    The model deliberately uses conservative proxies for hypothalamus and
    brainstem arousal systems because the source text is systems-level rather
    than parcel-specific.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.disorder_name = "Hypersomnolence Disorder"
        self.domain_key = "hypersomnolence"
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
            except Exception as exc:  # pragma: no cover - depends on siibra runtime
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
            "hypothalamus_proxy": [
                "hypothalamus",
                "lateral hypothalamus",
            ],
            "brainstem_arousal_proxy": [
                "brainstem",
                "midbrain",
                "pons",
                "mesencephalon",
            ],
            "thalamus_proxy": [
                "thalamus",
                "thalamic",
            ],
            "insula": [
                "Area Id1 (Insula) left",
                "Area Id2 (Insula) left",
                "Area Ig1 (Insula) left",
                "insula",
            ],
            "acc": [
                "Area p24ab left",
                "Area p24ab",
                "Area 32 left",
                "anterior cingulate",
                "cingulate",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Familial or inherited susceptibility to central hypersomnolence phenotypes."
            ),
            "orexin_deficit_load": (
                "Upstream hypocretin/orexin deficiency burden, especially relevant to narcolepsy type 1."
            ),
            "arousal_transmitter_imbalance_load": (
                "Burden of dysregulation across dopamine, norepinephrine, serotonin, histamine, and acetylcholine systems."
            ),
            "sleep_fragmentation_load": (
                "Burden of fragmented sleep and repeated arousals that destabilize consolidated wake-sleep control."
            ),
            "wake_promoting_treatment_support": (
                "Protective effect of wake-promoting therapy or structured support, such as stimulant or modafinil-like treatment."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "hypocretin_wake_instability": (
                "Failure of stable wake-state maintenance due to insufficient orexin/hypocretin signaling."
            ),
            "monoaminergic_arousal_failure": (
                "Reduced arousal support from dopamine, norepinephrine, and serotonin systems."
            ),
            "histaminergic_cholinergic_wake_failure": (
                "Reduced wake-promoting support from histamine and acetylcholine systems."
            ),
            "thalamocortical_gating_dysfunction": (
                "Disrupted thalamocortical gating and unstable sleep-wake transitions that increase fragmentation."
            ),
            "salience_network_dysregulation": (
                "Reduced ability of salience circuitry to sustain alert responses to internal and external stimuli."
            ),
            "distributed_arousal_network_dysregulation": (
                "Network-level failure across hypothalamic, brainstem, thalamic, and salience-related wake systems."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "excessive_daytime_sleepiness": "Excessive daytime sleepiness driven by unstable wake maintenance.",
            "unstable_wakefulness": "Inability to consistently achieve and sustain wakefulness.",
            "sleep_fragmentation": "Disrupted sleep continuity with repeated fragmentation or arousals.",
            "sleep_drunkenness": "Difficulty transitioning from sleep to wake with prolonged grogginess or inertia.",
            "vigilance_impairment": "Reduced sustained alertness and impaired detection of relevant stimuli.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "hypocretin_wake_instability",
                "relation": "raises vulnerability to inherited wake-regulation instability",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "orexin_deficit_load",
                "target": "hypocretin_wake_instability",
                "relation": "directly destabilizes global wake-promoting tone",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "arousal_transmitter_imbalance_load",
                "target": "monoaminergic_arousal_failure",
                "relation": "weakens dopamine, norepinephrine, and serotonin mediated vigilance support",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "arousal_transmitter_imbalance_load",
                "target": "histaminergic_cholinergic_wake_failure",
                "relation": "weakens histamine and acetylcholine mediated wake promotion",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "sleep_fragmentation_load",
                "target": "thalamocortical_gating_dysfunction",
                "relation": "disrupts sensory gating and sleep continuity",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "hypocretin_wake_instability",
                "target": "hypothalamus_proxy",
                "relation": "maps orexin-related instability onto hypothalamic wake control",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "monoaminergic_arousal_failure",
                "target": "brainstem_arousal_proxy",
                "relation": "reduces ascending arousal support",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "histaminergic_cholinergic_wake_failure",
                "target": "brainstem_arousal_proxy",
                "relation": "lowers wake-promoting brainstem tone",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "thalamocortical_gating_dysfunction",
                "target": "thalamus_proxy",
                "relation": "impairs thalamic gating of sensory flow across sleep-wake states",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "salience_network_dysregulation",
                "target": "insula",
                "relation": "reduces internal and external salience registration",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "salience_network_dysregulation",
                "target": "acc",
                "relation": "reduces salience-related control over alerting responses",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "hypocretin_wake_instability",
                "target": "distributed_arousal_network_dysregulation",
                "relation": "propagates instability across the global arousal system",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "monoaminergic_arousal_failure",
                "target": "distributed_arousal_network_dysregulation",
                "relation": "reduces wake-stabilizing neuromodulatory support",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "histaminergic_cholinergic_wake_failure",
                "target": "distributed_arousal_network_dysregulation",
                "relation": "removes additional wake-promoting reinforcement",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "thalamocortical_gating_dysfunction",
                "target": "distributed_arousal_network_dysregulation",
                "relation": "destabilizes network transitions between sleep and wakefulness",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "distributed_arousal_network_dysregulation",
                "target": "unstable_wakefulness",
                "relation": "prevents sustained wake-state maintenance",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "distributed_arousal_network_dysregulation",
                "target": "excessive_daytime_sleepiness",
                "relation": "shifts the system toward daytime sleepiness",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "thalamocortical_gating_dysfunction",
                "target": "sleep_fragmentation",
                "relation": "promotes fragmented sleep architecture",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "salience_network_dysregulation",
                "target": "vigilance_impairment",
                "relation": "reduces responsiveness to relevant stimuli",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "unstable_wakefulness",
                "target": "sleep_drunkenness",
                "relation": "makes sleep-to-wake transitions more difficult and prolonged",
                "hypersomnolence_change": "increased",
            },
            {
                "source": "wake_promoting_treatment_support",
                "target": "monoaminergic_arousal_failure",
                "relation": "partially offsets monoaminergic wake failure",
                "hypersomnolence_change": "decreased",
            },
            {
                "source": "wake_promoting_treatment_support",
                "target": "histaminergic_cholinergic_wake_failure",
                "relation": "partially offsets other wake-promoting neurotransmitter deficits",
                "hypersomnolence_change": "decreased",
            },
            {
                "source": "wake_promoting_treatment_support",
                "target": "excessive_daytime_sleepiness",
                "relation": "reduces daytime sleepiness burden",
                "hypersomnolence_change": "decreased",
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
        generic_penalty = 1 if name in {"amygdala", "hippocampus", "prefrontal cortex", "cingulate cortex"} else 0
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
            rows.append({
                "name": row[0],
                "identifier": row[1],
                "parcellation": row[2],
            })
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
        genetic_vulnerability: float = 0.35,
        orexin_deficit_load: float = 0.40,
        arousal_transmitter_imbalance_load: float = 0.45,
        sleep_fragmentation_load: float = 0.35,
        wake_promoting_treatment_support: float = 0.20,
    ) -> Dict[str, pd.Series]:
        """
        One-pass normalized simulator.

        Values are interpreted as burden or support on a 0..1 scale.
        Higher regional-state values reflect greater dysfunction burden,
        not healthier activity.
        """

        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "orexin_deficit_load": self._clip01(orexin_deficit_load),
                "arousal_transmitter_imbalance_load": self._clip01(arousal_transmitter_imbalance_load),
                "sleep_fragmentation_load": self._clip01(sleep_fragmentation_load),
                "wake_promoting_treatment_support": self._clip01(wake_promoting_treatment_support),
            },
            name="inputs",
        )

        latents = pd.Series(
            {
                "hypocretin_wake_instability": self._clip01(
                    0.55 * inputs["orexin_deficit_load"]
                    + 0.25 * inputs["genetic_vulnerability"]
                    + 0.10 * inputs["sleep_fragmentation_load"]
                    - 0.25 * inputs["wake_promoting_treatment_support"]
                ),
                "monoaminergic_arousal_failure": self._clip01(
                    0.45 * inputs["arousal_transmitter_imbalance_load"]
                    + 0.20 * inputs["genetic_vulnerability"]
                    + 0.15 * inputs["orexin_deficit_load"]
                    + 0.10 * inputs["sleep_fragmentation_load"]
                    - 0.30 * inputs["wake_promoting_treatment_support"]
                ),
                "histaminergic_cholinergic_wake_failure": self._clip01(
                    0.45 * inputs["arousal_transmitter_imbalance_load"]
                    + 0.20 * inputs["orexin_deficit_load"]
                    + 0.10 * inputs["sleep_fragmentation_load"]
                    - 0.20 * inputs["wake_promoting_treatment_support"]
                ),
                "thalamocortical_gating_dysfunction": self._clip01(
                    0.50 * inputs["sleep_fragmentation_load"]
                    + 0.15 * inputs["arousal_transmitter_imbalance_load"]
                    + 0.15 * inputs["genetic_vulnerability"]
                    - 0.10 * inputs["wake_promoting_treatment_support"]
                ),
                "salience_network_dysregulation": self._clip01(
                    0.35 * inputs["arousal_transmitter_imbalance_load"]
                    + 0.25 * inputs["sleep_fragmentation_load"]
                    + 0.20 * inputs["orexin_deficit_load"]
                    - 0.10 * inputs["wake_promoting_treatment_support"]
                ),
            },
            name="latents",
        )
        latents.loc["distributed_arousal_network_dysregulation"] = self._clip01(
            0.30 * latents["hypocretin_wake_instability"]
            + 0.25 * latents["monoaminergic_arousal_failure"]
            + 0.20 * latents["histaminergic_cholinergic_wake_failure"]
            + 0.15 * latents["thalamocortical_gating_dysfunction"]
            + 0.10 * latents["salience_network_dysregulation"]
        )

        regional_state = pd.Series(
            {
                "hypothalamus_proxy": self._clip01(
                    0.70 * latents["hypocretin_wake_instability"]
                    + 0.15 * latents["distributed_arousal_network_dysregulation"]
                    + 0.05 * inputs["genetic_vulnerability"]
                    - 0.15 * inputs["wake_promoting_treatment_support"]
                ),
                "brainstem_arousal_proxy": self._clip01(
                    0.45 * latents["monoaminergic_arousal_failure"]
                    + 0.35 * latents["histaminergic_cholinergic_wake_failure"]
                    + 0.15 * latents["distributed_arousal_network_dysregulation"]
                    - 0.20 * inputs["wake_promoting_treatment_support"]
                ),
                "thalamus_proxy": self._clip01(
                    0.60 * latents["thalamocortical_gating_dysfunction"]
                    + 0.15 * latents["distributed_arousal_network_dysregulation"]
                    + 0.10 * inputs["sleep_fragmentation_load"]
                ),
                "insula": self._clip01(
                    0.55 * latents["salience_network_dysregulation"]
                    + 0.15 * latents["distributed_arousal_network_dysregulation"]
                    + 0.10 * latents["monoaminergic_arousal_failure"]
                ),
                "acc": self._clip01(
                    0.55 * latents["salience_network_dysregulation"]
                    + 0.15 * latents["thalamocortical_gating_dysfunction"]
                    + 0.10 * latents["distributed_arousal_network_dysregulation"]
                ),
            },
            name="regional_state",
        )

        symptoms = pd.Series(
            {
                "unstable_wakefulness": self._clip01(
                    0.35 * latents["distributed_arousal_network_dysregulation"]
                    + 0.25 * regional_state["hypothalamus_proxy"]
                    + 0.20 * regional_state["brainstem_arousal_proxy"]
                    + 0.10 * regional_state["thalamus_proxy"]
                    - 0.20 * inputs["wake_promoting_treatment_support"]
                ),
                "excessive_daytime_sleepiness": self._clip01(
                    0.35 * latents["distributed_arousal_network_dysregulation"]
                    + 0.20 * regional_state["brainstem_arousal_proxy"]
                    + 0.15 * regional_state["thalamus_proxy"]
                    + 0.10 * latents["monoaminergic_arousal_failure"]
                    - 0.20 * inputs["wake_promoting_treatment_support"]
                ),
                "sleep_fragmentation": self._clip01(
                    0.45 * latents["thalamocortical_gating_dysfunction"]
                    + 0.20 * regional_state["thalamus_proxy"]
                    + 0.15 * inputs["sleep_fragmentation_load"]
                    + 0.10 * regional_state["brainstem_arousal_proxy"]
                    - 0.10 * inputs["wake_promoting_treatment_support"]
                ),
                "sleep_drunkenness": self._clip01(
                    0.35 * regional_state["thalamus_proxy"]
                    + 0.25 * latents["distributed_arousal_network_dysregulation"]
                    + 0.20 * inputs["sleep_fragmentation_load"]
                    + 0.10 * regional_state["hypothalamus_proxy"]
                    - 0.10 * inputs["wake_promoting_treatment_support"]
                ),
            },
            name="symptoms",
        )
        symptoms.loc["vigilance_impairment"] = self._clip01(
            0.30 * regional_state["insula"]
            + 0.20 * regional_state["acc"]
            + 0.20 * latents["distributed_arousal_network_dysregulation"]
            + 0.15 * symptoms["excessive_daytime_sleepiness"]
            - 0.10 * inputs["wake_promoting_treatment_support"]
        )

        phenotypes = pd.Series(
            {
                "narcolepsy_type1_like_profile": self._clip01(
                    0.45 * inputs["orexin_deficit_load"]
                    + 0.25 * latents["hypocretin_wake_instability"]
                    + 0.20 * symptoms["unstable_wakefulness"]
                    + 0.10 * symptoms["excessive_daytime_sleepiness"]
                ),
                "idiopathic_hypersomnia_like_profile": self._clip01(
                    0.35 * symptoms["sleep_drunkenness"]
                    + 0.25 * symptoms["excessive_daytime_sleepiness"]
                    + 0.20 * symptoms["vigilance_impairment"]
                    + 0.10 * latents["thalamocortical_gating_dysfunction"]
                    + 0.10 * (1.0 - inputs["orexin_deficit_load"])
                ),
                "fragmented_hypersomnolence_profile": self._clip01(
                    0.40 * symptoms["sleep_fragmentation"]
                    + 0.25 * latents["thalamocortical_gating_dysfunction"]
                    + 0.20 * symptoms["unstable_wakefulness"]
                    + 0.15 * symptoms["vigilance_impairment"]
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
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 12)

    model = HypersomnolenceDisorderModel()
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

    sim = model.simulate(
        genetic_vulnerability=0.45,
        orexin_deficit_load=0.75,
        arousal_transmitter_imbalance_load=0.55,
        sleep_fragmentation_load=0.40,
        wake_promoting_treatment_support=0.30,
    )

    print("\n=== Simulation ===")
    for name, series in sim.items():
        print(f"\n{name.upper()}")
        print(series.to_string())

    if SIIBRA_AVAILABLE:
        print("\n=== Example region suggestions for 'thalamus' ===")
        print(model.suggest_regions("thalamus", limit=10).to_string(index=False))

        # Example coordinate assignment:
        # print(model.assign_mni_point((-8, -20, 8)).head().to_string(index=False))
