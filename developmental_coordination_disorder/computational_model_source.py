from __future__ import annotations

"""
DevelopmentalCoordinationDisorderModel
=====================================

Atlas-grounded research scaffold for Developmental Coordination Disorder (DCD).

This script translates a chapter-level biological summary of DCD into a simple,
transparent mechanistic model that can be explored with siibra when atlas data
are available. It is intended for research scaffolding and hypothesis tracing,
not diagnosis or treatment.

Chapter-derived themes emphasized here:
- DCD is modeled as a neurodevelopmental motor-planning / coordination disorder.
- Catecholaminergic hypotheses are kept latent because the chapter does not
  localize dopamine or norepinephrine effects to a single parcel.
- The motor-control network is represented conservatively using atlas-backed
  cortical anchors plus explicit cerebellar and basal-ganglia proxies.
- White-matter and developmental-wiring burden are modeled as latent processes
  because the chapter describes distributed connectivity risk rather than one
  isolated lesion.

The scaffold degrades gracefully when siibra is unavailable: the simulator still
runs, while atlas-backed feature retrieval methods return empty tables.
"""

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_GENE_PANEL = [
    # Catecholaminergic / ADHD-overlap probes
    "DRD2",
    "DRD4",
    "SLC6A3",
    "SLC6A2",
    "DBH",
    "COMT",
    "SNAP25",
    # Neurodevelopment, migration, timing, and synaptic maturation
    "BDNF",
    "RELN",
    "DCX",
    "CNTNAP2",
    "DCDC2",
    "KIAA0319",
    "ROBO1",
    # Myelination / axonal support
    "MBP",
    "PLP1",
]


class DevelopmentalCoordinationDisorderModel:
    """
    Research scaffold for Developmental Coordination Disorder (DCD).

    The graph is a chapter-faithful interpretation of DCD as a disorder of
    motor planning, sequencing, execution, and developmental circuit formation.
    Atlas-backed nodes are limited to cortical motor-parietal structures that
    the chapter names or strongly implies. Basal ganglia and cerebellum are
    retained as proxies because exact Julich mappings may vary by siibra version
    and environment.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.siibra_available = siibra is not None
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        self.atlas = None
        self.parcellation = None
        self.space = None
        self._pmap = None
        self._labelmap = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

        if self.siibra_available:
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
            except Exception as exc:
                warnings.warn(
                    "siibra is installed but atlas resources could not be initialized; "
                    f"atlas-backed methods will degrade gracefully. Detail: {exc}"
                )
                self.atlas = None
                self.parcellation = None
                self.space = None
        else:
            warnings.warn(
                "siibra is not installed in this environment. Atlas-backed build(), "
                "feature lookup, and anatomical assignment will return partial outputs, "
                "but simulate() remains usable."
            )

        # Conservative region mapping from chapter text.
        self.region_candidates: Dict[str, List[str]] = {
            "primary_motor_cortex": [
                "Area 4a left",
                "Area 4p left",
                "primary motor cortex",
                "Area 4",
            ],
            "premotor_cortex": [
                "Area 6 left",
                "premotor cortex",
                "Area 6",
            ],
            "supplementary_motor_area_proxy": [
                "supplementary motor area",
                "pre-supplementary motor area",
                "SMA",
                "Area 6 left",
            ],
            "posterior_parietal_cortex": [
                "Area 7A left",
                "Area 7PC left",
                "Area hIP3 left",
                "Area 5L left",
                "posterior parietal cortex",
                "intraparietal",
            ],
            # Explicit proxies: these may or may not resolve in Julich.
            "basal_ganglia_proxy": [
                "putamen left",
                "caudate nucleus left",
                "striatum left",
                "basal ganglia",
            ],
            "cerebellum_proxy": [
                "cerebellum left",
                "cerebellum",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Polygenic neurodevelopmental liability shared in part with ADHD, ASD, "
                "and dyslexia vulnerability patterns discussed in the chapter"
            ),
            "developmental_wiring_load": (
                "Burden on neuronal migration, synaptogenesis, and myelination implied by the chapter"
            ),
            "adhd_comorbidity_load": (
                "ADHD-overlap burden used as a catecholaminergic and frontostriatal hypothesis bridge"
            ),
            "autism_dyslexia_liability": (
                "Shared neurodevelopmental liability from ASD/dyslexia comorbidity context"
            ),
            "attention_control_burden": (
                "Impaired sustained attention, vigilance, and processing-speed burden that can worsen motor performance"
            ),
            "motor_task_complexity": (
                "Demand imposed by multi-step, precision, balance, or dual-task motor requirements"
            ),
            "compensatory_support": (
                "Optional generic scaffold control for structured support, practice, or environmental adaptation; "
                "included for simulation transparency rather than as an explicit chapter claim"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "polygenic_neurodevelopmental_liability": (
                "Shared developmental liability spanning DCD and commonly comorbid neurodevelopmental conditions"
            ),
            "neuronal_migration_synaptogenesis_myelination_variation": (
                "Distributed developmental-wiring variation affecting circuit formation and white-matter maturation"
            ),
            "catecholaminergic_support_deficit": (
                "Hypothesized dopamine/norepinephrine support deficit influencing attention, motor learning, and behavioral control"
            ),
            "frontostriatal_attention_control_dysfunction": (
                "Reduced cortical-striatal-thalamic-cortical efficiency supporting attention and action selection"
            ),
            "cerebellar_prediction_error_dysfunction": (
                "Reduced cerebellar support for calibration, prediction, timing, and online error correction"
            ),
            "sensorimotor_integration_burden": (
                "Impaired integration of body-state, spatial, and action-planning information"
            ),
            "white_matter_connectivity_disruption": (
                "Distributed tract-level inefficiency linking motor, parietal, cerebellar, and subcortical systems"
            ),
            "motor_sequence_planning_failure": (
                "Core planning, sequencing, and execution difficulty described by the chapter"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "impaired_motor_coordination": "Global motor coordination difficulty across everyday actions",
            "balance_and_gait_difficulty": "Gross-motor problems with balance, running, and postural control",
            "fine_motor_impairment": "Fine-motor inefficiency affecting manual precision tasks",
            "handwriting_self_care_impairment": (
                "Functional fine-motor consequences in handwriting, dressing, and self-care tasks such as tying shoelaces"
            ),
            "slow_motor_learning": "Reduced acquisition and automatization of complex motor routines",
            "dual_task_motor_breakdown": (
                "Disproportionate worsening of movement when attentional demand and motor demand must be managed together"
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "polygenic_neurodevelopmental_liability",
                "relation": "contributes shared neurodevelopmental risk",
                "dcd_change": "increased",
            },
            {
                "source": "adhd_comorbidity_load",
                "target": "polygenic_neurodevelopmental_liability",
                "relation": "indexes overlap with heritable neurodevelopmental vulnerability",
                "dcd_change": "increased",
            },
            {
                "source": "autism_dyslexia_liability",
                "target": "polygenic_neurodevelopmental_liability",
                "relation": "reinforces shared developmental-risk context",
                "dcd_change": "increased",
            },
            {
                "source": "developmental_wiring_load",
                "target": "neuronal_migration_synaptogenesis_myelination_variation",
                "relation": "amplifies developmental circuit-formation burden",
                "dcd_change": "increased",
            },
            {
                "source": "polygenic_neurodevelopmental_liability",
                "target": "neuronal_migration_synaptogenesis_myelination_variation",
                "relation": "biases distributed neurodevelopmental organization",
                "dcd_change": "increased",
            },
            {
                "source": "adhd_comorbidity_load",
                "target": "catecholaminergic_support_deficit",
                "relation": "motivates dopamine/norepinephrine hypothesis transfer from ADHD overlap",
                "dcd_change": "increased",
            },
            {
                "source": "attention_control_burden",
                "target": "catecholaminergic_support_deficit",
                "relation": "raises need for attentional neuromodulatory support",
                "dcd_change": "increased",
            },
            {
                "source": "catecholaminergic_support_deficit",
                "target": "frontostriatal_attention_control_dysfunction",
                "relation": "weakens action selection and sustained attentional control",
                "dcd_change": "increased",
            },
            {
                "source": "neuronal_migration_synaptogenesis_myelination_variation",
                "target": "white_matter_connectivity_disruption",
                "relation": "impairs long-range circuit efficiency",
                "dcd_change": "increased",
            },
            {
                "source": "white_matter_connectivity_disruption",
                "target": "sensorimotor_integration_burden",
                "relation": "reduces efficient communication across motor and parietal systems",
                "dcd_change": "increased",
            },
            {
                "source": "frontostriatal_attention_control_dysfunction",
                "target": "motor_sequence_planning_failure",
                "relation": "degrades planning and sequencing of actions",
                "dcd_change": "increased",
            },
            {
                "source": "cerebellar_prediction_error_dysfunction",
                "target": "motor_sequence_planning_failure",
                "relation": "limits error correction and calibration during skilled movement",
                "dcd_change": "increased",
            },
            {
                "source": "sensorimotor_integration_burden",
                "target": "motor_sequence_planning_failure",
                "relation": "impairs spatial and body-state updating for action assembly",
                "dcd_change": "increased",
            },
            {
                "source": "motor_sequence_planning_failure",
                "target": "primary_motor_cortex",
                "relation": "increases downstream burden on execution circuitry",
                "dcd_change": "increased",
            },
            {
                "source": "motor_sequence_planning_failure",
                "target": "premotor_cortex",
                "relation": "increases preparatory motor-control burden",
                "dcd_change": "increased",
            },
            {
                "source": "motor_sequence_planning_failure",
                "target": "supplementary_motor_area_proxy",
                "relation": "impairs sequencing and internally guided action assembly",
                "dcd_change": "increased",
            },
            {
                "source": "sensorimotor_integration_burden",
                "target": "posterior_parietal_cortex",
                "relation": "raises spatial-integration burden in parietal action systems",
                "dcd_change": "increased",
            },
            {
                "source": "frontostriatal_attention_control_dysfunction",
                "target": "basal_ganglia_proxy",
                "relation": "matches hypothesized frontostriatal action-selection inefficiency",
                "dcd_change": "increased",
            },
            {
                "source": "cerebellar_prediction_error_dysfunction",
                "target": "cerebellum_proxy",
                "relation": "matches hypothesized cerebellar timing and calibration burden",
                "dcd_change": "increased",
            },
            {
                "source": "primary_motor_cortex",
                "target": "impaired_motor_coordination",
                "relation": "contributes execution instability",
                "dcd_change": "increased",
            },
            {
                "source": "cerebellum_proxy",
                "target": "balance_and_gait_difficulty",
                "relation": "contributes balance and locomotor instability",
                "dcd_change": "increased",
            },
            {
                "source": "premotor_cortex",
                "target": "fine_motor_impairment",
                "relation": "degrades precision preparation and hand-action control",
                "dcd_change": "increased",
            },
            {
                "source": "fine_motor_impairment",
                "target": "handwriting_self_care_impairment",
                "relation": "produces functional manual difficulties",
                "dcd_change": "increased",
            },
            {
                "source": "cerebellar_prediction_error_dysfunction",
                "target": "slow_motor_learning",
                "relation": "reduces efficient skill acquisition and adaptation",
                "dcd_change": "increased",
            },
            {
                "source": "frontostriatal_attention_control_dysfunction",
                "target": "dual_task_motor_breakdown",
                "relation": "impairs concurrent attention and motor management",
                "dcd_change": "increased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    def _modality_candidates(self, kind: str) -> List[Any]:
        if not self.siibra_available:
            return []

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
        if not self.siibra_available or concept is None:
            return []

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
        if not self.siibra_available or self.atlas is None:
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

        out = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower() or self.parcellation is None:
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        proxy_penalty = 1 if "proxy" in name else 0
        generic_penalty = 1 if name in {"cerebellum", "amygdala", "hippocampus", "putamen"} else 0
        return (left_bonus, right_penalty, proxy_penalty, generic_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if not self.siibra_available or self.atlas is None:
            return None

        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation)
            except Exception:
                pass
            try:
                if self.parcellation is not None and hasattr(self.parcellation, "get_region"):
                    return self.parcellation.get_region(spec)
            except Exception:
                pass
            matches = self._julich_matches(spec)
            if matches:
                matches = sorted(matches, key=self._region_rank)
                return matches[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows = []
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
        centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        volume_mm3 = float(getattr(main, "volume", float("nan")))
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

        compound = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), feats[0])

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            first = compound[0]
            if hasattr(first, "data") and isinstance(first.data, pd.DataFrame):
                self._connectivity_matrix = first.data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._name_of(region)
        exact = [x for x in labels if self._name_of(x) == region_name]
        if exact:
            return exact[0]

        rn = region_name.lower()
        fuzzy = [
            x
            for x in labels
            if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn
        ]
        if fuzzy:
            return fuzzy[0]

        compact_rn = rn.replace(" left", "").replace(" right", "")
        for x in labels:
            xl = self._name_of(x).lower().replace(" left", "").replace(" right", "")
            if compact_rn == xl or compact_rn in xl or xl in compact_rn:
                return x
        return None

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
            df = pd.DataFrame(series).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = (
                df.dropna(subset=["value"])
                .sort_values("value", ascending=False)
                .query("connected_region != @region.name", engine="python")
                .head(max_rows)
                .reset_index(drop=True)
            )
            return df
        except Exception:
            return pd.DataFrame()

    def _lookup_connectivity_value(
        self,
        matrix: pd.DataFrame,
        src_region: Any,
        dst_region: Any,
    ) -> Tuple[float, Optional[str], Optional[str]]:
        if matrix.empty:
            return float("nan"), None, None

        src_row = self._match_region_label(list(matrix.index), src_region)
        src_col = self._match_region_label(list(matrix.columns), src_region)
        dst_row = self._match_region_label(list(matrix.index), dst_region)
        dst_col = self._match_region_label(list(matrix.columns), dst_region)

        # Most common case: square matrix with the same labels on both axes.
        if src_row is not None and dst_col is not None:
            try:
                value = float(pd.to_numeric(pd.Series([matrix.loc[src_row, dst_col]]), errors="coerce").iloc[0])
                return value, self._name_of(src_row), self._name_of(dst_col)
            except Exception:
                pass

        if dst_row is not None and src_col is not None:
            try:
                value = float(pd.to_numeric(pd.Series([matrix.loc[dst_row, src_col]]), errors="coerce").iloc[0])
                return value, self._name_of(src_col), self._name_of(dst_row)
            except Exception:
                pass

        return float("nan"), None, None

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        rows: List[Dict[str, Any]] = []

        for src_key, src_region in self.region_objects.items():
            for dst_key, dst_region in self.region_objects.items():
                if src_key == dst_key:
                    continue
                value, src_label, dst_label = self._lookup_connectivity_value(matrix, src_region, dst_region)
                rows.append(
                    {
                        "source_key": src_key,
                        "target_key": dst_key,
                        "source_region": self._name_of(src_region),
                        "target_region": self._name_of(dst_region),
                        "matrix_source_label": src_label,
                        "matrix_target_label": dst_label,
                        "value": value,
                    }
                )

        return pd.DataFrame(rows)

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
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region_proxy" if key.endswith("_proxy") else "region",
                        "description": "Atlas-backed circuit node unresolved in this environment; proxy retained",
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
                    "description": "Atlas-backed circuit node",
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

    def simulate(
        self,
        genetic_vulnerability: float = 0.55,
        developmental_wiring_load: float = 0.60,
        adhd_comorbidity_load: float = 0.45,
        autism_dyslexia_liability: float = 0.35,
        attention_control_burden: float = 0.55,
        motor_task_complexity: float = 0.60,
        compensatory_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """
        Run a transparent one-pass normalized simulation.

        Inputs are clipped to [0, 1]. Higher values indicate greater burden,
        except compensatory_support, which is protective.
        """

        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "developmental_wiring_load": self._clip01(developmental_wiring_load),
                "adhd_comorbidity_load": self._clip01(adhd_comorbidity_load),
                "autism_dyslexia_liability": self._clip01(autism_dyslexia_liability),
                "attention_control_burden": self._clip01(attention_control_burden),
                "motor_task_complexity": self._clip01(motor_task_complexity),
                "compensatory_support": self._clip01(compensatory_support),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["polygenic_neurodevelopmental_liability"] = self._clip01(
            0.45 * inputs["genetic_vulnerability"]
            + 0.25 * inputs["adhd_comorbidity_load"]
            + 0.15 * inputs["autism_dyslexia_liability"]
            + 0.15 * inputs["developmental_wiring_load"]
        )
        latents["neuronal_migration_synaptogenesis_myelination_variation"] = self._clip01(
            0.50 * inputs["developmental_wiring_load"]
            + 0.25 * latents["polygenic_neurodevelopmental_liability"]
            + 0.10 * inputs["autism_dyslexia_liability"]
            + 0.05 * inputs["genetic_vulnerability"]
            - 0.20 * inputs["compensatory_support"]
        )
        latents["catecholaminergic_support_deficit"] = self._clip01(
            0.45 * inputs["adhd_comorbidity_load"]
            + 0.30 * inputs["attention_control_burden"]
            + 0.15 * inputs["genetic_vulnerability"]
            - 0.20 * inputs["compensatory_support"]
        )
        latents["frontostriatal_attention_control_dysfunction"] = self._clip01(
            0.45 * latents["catecholaminergic_support_deficit"]
            + 0.25 * latents["polygenic_neurodevelopmental_liability"]
            + 0.20 * inputs["attention_control_burden"]
            + 0.10 * inputs["motor_task_complexity"]
            - 0.20 * inputs["compensatory_support"]
        )
        latents["cerebellar_prediction_error_dysfunction"] = self._clip01(
            0.40 * latents["neuronal_migration_synaptogenesis_myelination_variation"]
            + 0.20 * latents["polygenic_neurodevelopmental_liability"]
            + 0.20 * inputs["motor_task_complexity"]
            + 0.10 * latents["catecholaminergic_support_deficit"]
            - 0.15 * inputs["compensatory_support"]
        )
        latents["sensorimotor_integration_burden"] = self._clip01(
            0.35 * latents["neuronal_migration_synaptogenesis_myelination_variation"]
            + 0.25 * latents["frontostriatal_attention_control_dysfunction"]
            + 0.20 * inputs["attention_control_burden"]
            + 0.20 * inputs["motor_task_complexity"]
            - 0.15 * inputs["compensatory_support"]
        )
        latents["white_matter_connectivity_disruption"] = self._clip01(
            0.50 * latents["neuronal_migration_synaptogenesis_myelination_variation"]
            + 0.20 * latents["polygenic_neurodevelopmental_liability"]
            + 0.15 * inputs["adhd_comorbidity_load"]
            + 0.05 * inputs["autism_dyslexia_liability"]
            - 0.10 * inputs["compensatory_support"]
        )
        latents["motor_sequence_planning_failure"] = self._clip01(
            0.35 * latents["frontostriatal_attention_control_dysfunction"]
            + 0.25 * latents["cerebellar_prediction_error_dysfunction"]
            + 0.20 * latents["sensorimotor_integration_burden"]
            + 0.20 * inputs["motor_task_complexity"]
            - 0.15 * inputs["compensatory_support"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["primary_motor_cortex"] = self._clip01(
            0.40 * latents["motor_sequence_planning_failure"]
            + 0.30 * latents["sensorimotor_integration_burden"]
            + 0.30 * latents["white_matter_connectivity_disruption"]
        )
        regional_state["premotor_cortex"] = self._clip01(
            0.40 * latents["motor_sequence_planning_failure"]
            + 0.35 * latents["frontostriatal_attention_control_dysfunction"]
            + 0.25 * inputs["attention_control_burden"]
        )
        regional_state["supplementary_motor_area_proxy"] = self._clip01(
            0.45 * latents["motor_sequence_planning_failure"]
            + 0.30 * latents["frontostriatal_attention_control_dysfunction"]
            + 0.25 * inputs["motor_task_complexity"]
        )
        regional_state["posterior_parietal_cortex"] = self._clip01(
            0.45 * latents["sensorimotor_integration_burden"]
            + 0.30 * latents["white_matter_connectivity_disruption"]
            + 0.25 * inputs["attention_control_burden"]
        )
        regional_state["basal_ganglia_proxy"] = self._clip01(
            0.45 * latents["frontostriatal_attention_control_dysfunction"]
            + 0.30 * latents["catecholaminergic_support_deficit"]
            + 0.25 * latents["motor_sequence_planning_failure"]
        )
        regional_state["cerebellum_proxy"] = self._clip01(
            0.50 * latents["cerebellar_prediction_error_dysfunction"]
            + 0.25 * latents["sensorimotor_integration_burden"]
            + 0.25 * latents["white_matter_connectivity_disruption"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["impaired_motor_coordination"] = self._clip01(
            0.25 * regional_state["primary_motor_cortex"]
            + 0.20 * regional_state["premotor_cortex"]
            + 0.20 * regional_state["supplementary_motor_area_proxy"]
            + 0.20 * regional_state["posterior_parietal_cortex"]
            + 0.15 * regional_state["cerebellum_proxy"]
        )
        symptoms["balance_and_gait_difficulty"] = self._clip01(
            0.45 * regional_state["cerebellum_proxy"]
            + 0.20 * regional_state["posterior_parietal_cortex"]
            + 0.20 * latents["white_matter_connectivity_disruption"]
            + 0.15 * inputs["motor_task_complexity"]
        )
        symptoms["fine_motor_impairment"] = self._clip01(
            0.30 * regional_state["primary_motor_cortex"]
            + 0.25 * regional_state["premotor_cortex"]
            + 0.20 * regional_state["basal_ganglia_proxy"]
            + 0.15 * regional_state["posterior_parietal_cortex"]
            + 0.10 * inputs["attention_control_burden"]
        )
        symptoms["handwriting_self_care_impairment"] = self._clip01(
            0.50 * symptoms["fine_motor_impairment"]
            + 0.20 * latents["motor_sequence_planning_failure"]
            + 0.20 * inputs["attention_control_burden"]
            + 0.10 * symptoms["impaired_motor_coordination"]
        )
        symptoms["slow_motor_learning"] = self._clip01(
            0.35 * regional_state["cerebellum_proxy"]
            + 0.25 * latents["catecholaminergic_support_deficit"]
            + 0.20 * latents["frontostriatal_attention_control_dysfunction"]
            + 0.20 * latents["sensorimotor_integration_burden"]
        )
        symptoms["dual_task_motor_breakdown"] = self._clip01(
            0.35 * latents["frontostriatal_attention_control_dysfunction"]
            + 0.30 * inputs["attention_control_burden"]
            + 0.20 * symptoms["fine_motor_impairment"]
            + 0.15 * inputs["motor_task_complexity"]
        )

        phenotypes = pd.Series(dtype=float, name="phenotypes")
        phenotypes["gross_motor_profile"] = self._clip01(
            0.50 * symptoms["balance_and_gait_difficulty"]
            + 0.50 * symptoms["impaired_motor_coordination"]
        )
        phenotypes["fine_motor_profile"] = self._clip01(
            0.55 * symptoms["fine_motor_impairment"]
            + 0.45 * symptoms["handwriting_self_care_impairment"]
        )
        phenotypes["attention_motor_interference_profile"] = self._clip01(
            0.55 * symptoms["dual_task_motor_breakdown"]
            + 0.45 * symptoms["slow_motor_learning"]
        )
        phenotypes["lifespan_functional_impact_profile"] = self._clip01(
            0.30 * symptoms["impaired_motor_coordination"]
            + 0.25 * symptoms["handwriting_self_care_impairment"]
            + 0.20 * symptoms["balance_and_gait_difficulty"]
            + 0.15 * symptoms["slow_motor_learning"]
            + 0.10 * symptoms["dual_task_motor_breakdown"]
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Assign an MNI152 coordinate to atlas regions using a statistical map.
        """
        if not self.siibra_available or self.parcellation is None:
            return pd.DataFrame(
                [{
                    "region": None,
                    "detail": "siibra unavailable or parcellation not initialized",
                    "xyz": tuple(xyz),
                }]
            )

        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception:
                try:
                    with siibra.QUIET:
                        self._pmap = self.atlas.get_map(
                            parcellation=self.parcellation,
                            space=self.atlas.get_space(self.assignment_space)
                            if hasattr(self.atlas, "get_space")
                            else self.assignment_space,
                            maptype="statistical",
                        )
                except Exception as exc:
                    return pd.DataFrame(
                        [{
                            "region": None,
                            "detail": f"could not create statistical map: {exc}",
                            "xyz": tuple(xyz),
                        }]
                    )

        try:
            point = siibra.Point(tuple(xyz), space=self.assignment_space)
            with siibra.QUIET:
                assignments = self._pmap.assign(point)
        except Exception as exc:
            return pd.DataFrame(
                [{
                    "region": None,
                    "detail": f"assignment failed: {exc}",
                    "xyz": tuple(xyz),
                }]
            )

        for candidate in ("map value", "correlation", "intersection over union", "value"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Any:
        """
        Return a region-specific map or mask when possible.

        Depending on siibra version, this may be a fetched statistical map,
        a labelled map fragment, or None if no suitable accessor exists.
        """
        region = self.region_objects.get(node_key)
        if region is None or not self.siibra_available:
            return None

        # Preferred: statistical regional map fetched from a probabilistic map.
        try:
            if self._pmap is None:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            return self._pmap.fetch(region)
        except Exception:
            pass

        # Fallback: labelled map fetch.
        try:
            if self._labelmap is None:
                with siibra.QUIET:
                    self._labelmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="labelled",
                    )
            return self._labelmap.fetch(region)
        except Exception:
            pass

        # Last-resort fallbacks for alternate APIs.
        for attr in ("get_regional_map", "fetch"):
            try:
                fn = getattr(region, attr)
            except Exception:
                continue
            try:
                return fn(self.space) if attr == "get_regional_map" else fn()
            except Exception:
                continue
        return None


if __name__ == "__main__":
    model = DevelopmentalCoordinationDisorderModel()
    build = model.build()

    print("\n=== Nodes (first 12) ===")
    print(build["nodes"].head(12).to_string(index=False))

    print("\n=== Edges (first 12) ===")
    print(build["edges"].head(12).to_string(index=False))

    print("\n=== Resolved region keys ===")
    print(list(build["regions"].keys()))

    sample_key = "primary_motor_cortex"
    print(f"\n=== Receptor table: {sample_key} ===")
    print(build["receptors"].get(sample_key, pd.DataFrame()).head().to_string(index=False))

    print(f"\n=== Gene table: {sample_key} ===")
    print(build["genes"].get(sample_key, pd.DataFrame()).head().to_string(index=False))

    print(f"\n=== Connectivity profile: {sample_key} ===")
    print(build["connectivity_profiles"].get(sample_key, pd.DataFrame()).head().to_string(index=False))

    print("\n=== Circuit connectivity ===")
    print(build["circuit_connectivity"].head(12).to_string(index=False))

    sim = model.simulate(
        genetic_vulnerability=0.65,
        developmental_wiring_load=0.70,
        adhd_comorbidity_load=0.55,
        autism_dyslexia_liability=0.40,
        attention_control_burden=0.60,
        motor_task_complexity=0.75,
        compensatory_support=0.25,
    )

    print("\n=== Latents ===")
    print(sim["latents"].sort_values(ascending=False).to_string())

    print("\n=== Symptoms ===")
    print(sim["symptoms"].sort_values(ascending=False).to_string())

    print("\n=== Phenotypes ===")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Optional coordinate assignment example when siibra is available:
    # print(model.assign_mni_point((-30, -20, 55)).head())
