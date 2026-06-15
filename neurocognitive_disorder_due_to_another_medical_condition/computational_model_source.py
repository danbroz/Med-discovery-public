from __future__ import annotations

"""
Neurocognitive disorder due to another medical condition siibra scaffold.

This research scaffold translates a chapter-level summary of Major or Mild
Neurocognitive Disorder Due to Another Medical Condition into an
atlas-grounded, transparent mechanistic model.

It is intended for exploratory modeling and region retuning, not diagnosis or
clinical decision-making.
"""

import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore

    SIIBRA_AVAILABLE = True
except Exception:  # pragma: no cover - depends on runtime
    siibra = None  # type: ignore
    SIIBRA_AVAILABLE = False


DEFAULT_GENE_PANEL = [
    "APOE",
    "GRIN1",
    "GRIN2B",
    "SLC1A2",
    "GAD1",
    "GABRA1",
    "SCN1A",
    "BDNF",
    "MAPT",
    "NOTCH3",
    "MBP",
    "GFAP",
    "IL6",
    "TNF",
]


class NeurocognitiveDisorderDueToAnotherMedicalConditionModel:
    """
    Atlas-grounded scaffold for heterogeneous medical-cause neurocognitive disorder.

    Chapter logic emphasized here:
    - glutamatergic excitotoxicity as a common injury pathway,
    - GABAergic failure and seizure-related network hyperexcitability,
    - vascular white-matter disconnection and traumatic axonal injury,
    - strategic focal lesion burden affecting thalamic and frontal-subcortical circuits,
    - metabolic encephalopathy and inflammatory / infectious limbic-temporal injury,
    - broad downstream cognitive decline with amnestic, dysexecutive, confusional,
      and behavioral phenotypes.

    Because the chapter is intentionally heterogeneous, several region nodes are
    conservative proxies rather than over-precise parcel claims.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.disorder_name = "Major or Mild Neurocognitive Disorder Due to Another Medical Condition"
        self.domain_key = "medical_ncd"
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
            except Exception as exc:  # pragma: no cover - environment dependent
                warnings.warn(
                    f"siibra atlas initialization failed; atlas-backed features will be limited: {exc}"
                )
        else:
            warnings.warn(
                "siibra is not installed in this environment; atlas-backed region resolution, "
                "feature queries, and coordinate assignments will return empty results. "
                "The mechanistic simulator remains usable."
            )

        self.region_candidates: Dict[str, List[str]] = {
            "thalamus_proxy": [
                "thalamus left",
                "thalamus",
                "thalamic",
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
            "frontal_subcortical_proxy": [
                "caudate nucleus left",
                "caudate",
                "putamen left",
                "putamen",
                "basal ganglia",
                "subcortical",
            ],
            "limbic_temporal_proxy": [
                "CA1 (Hippocampus) left",
                "DG (Hippocampus) left",
                "hippocampus left",
                "amygdala left",
                "parahippocampal",
                "temporal lobe",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "vascular_ischemic_injury_load": (
                "Burden of stroke, lacunes, microangiopathy, or other ischemic cerebrovascular injury."
            ),
            "traumatic_axonal_injury_load": (
                "Burden of traumatic brain injury and diffuse axonal disconnection."
            ),
            "metabolic_systemic_encephalopathy_load": (
                "Burden of systemic, metabolic, or dysmetabolic encephalopathy producing diffuse slowing or delirium."
            ),
            "inflammatory_infectious_cns_load": (
                "Burden of inflammatory, autoimmune, demyelinating, or infectious CNS injury."
            ),
            "mass_effect_hydrocephalus_load": (
                "Burden of tumors, subdural collections, hydrocephalus, or other space-occupying lesions."
            ),
            "seizure_burden": (
                "Burden of seizures or nonconvulsive epileptic activity contributing to fluctuating cognition."
            ),
            "genetic_susceptibility": (
                "Background genetic susceptibility that lowers the threshold for medical or environmental brain injury."
            ),
            "medical_reversal_rehabilitation_support": (
                "Protective effect of correcting the medical cause, stabilizing physiology, and rehabilitation support."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "injury_susceptibility": (
                "Lowered reserve or resilience that makes brain networks more vulnerable to diverse medical insults."
            ),
            "glutamate_excitotoxicity": (
                "Excess glutamatergic receptor activation driving neuronal injury after ischemic or traumatic insult."
            ),
            "gaba_inhibitory_failure": (
                "Reduced inhibitory stability permitting hyperexcitability, seizures, and fluctuating cognition."
            ),
            "strategic_focal_lesion_burden": (
                "Cognitive impact of focal lesions in small but strategically important regions or circuits."
            ),
            "white_matter_network_disconnection": (
                "Executive and processing-speed burden caused by vascular white matter disease or axonal injury."
            ),
            "metabolic_encephalopathic_slowing": (
                "Diffuse slowing and global inefficiency produced by systemic or dysmetabolic brain dysfunction."
            ),
            "limbic_temporal_injury": (
                "Medial temporal and limbic burden from autoimmune, infectious, inflammatory, or mass lesions."
            ),
            "executive_control_circuit_failure": (
                "Breakdown of frontal and frontal-subcortical control circuits needed for executive performance."
            ),
            "distributed_cognitive_network_failure": (
                "System-level failure of large-scale cortical-subcortical networks supporting cognition."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "memory_impairment": "Amnestic dysfunction or memory inefficiency.",
            "executive_dysfunction": "Reduced planning, set shifting, inhibition, or working control.",
            "processing_speed_slowing": "Cognitive slowing typical of diffuse white matter or metabolic burden.",
            "social_cognitive_change": "Altered social cognition or interpretation of interpersonal cues.",
            "behavioral_apathy_personality_change": "Apathy, personality change, or frontal-behavioral syndrome.",
            "confusional_state_or_delirium": "Acute or fluctuating confusion, often worsened by seizures or metabolic failure.",
            "global_cognitive_decline": "Overall neurocognitive deterioration across multiple domains.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_susceptibility",
                "target": "injury_susceptibility",
                "relation": "lowers reserve and the threshold for medical insults to produce decline",
                "medical_ncd_change": "increased",
            },
            {
                "source": "vascular_ischemic_injury_load",
                "target": "glutamate_excitotoxicity",
                "relation": "drives ischemia-related excitotoxic neuronal damage",
                "medical_ncd_change": "increased",
            },
            {
                "source": "vascular_ischemic_injury_load",
                "target": "strategic_focal_lesion_burden",
                "relation": "creates strategic infarcts affecting cognition out of proportion to lesion size",
                "medical_ncd_change": "increased",
            },
            {
                "source": "vascular_ischemic_injury_load",
                "target": "white_matter_network_disconnection",
                "relation": "produces subcortical white-matter disease and disconnection syndromes",
                "medical_ncd_change": "increased",
            },
            {
                "source": "traumatic_axonal_injury_load",
                "target": "glutamate_excitotoxicity",
                "relation": "adds traumatic excitotoxic burden after brain injury",
                "medical_ncd_change": "increased",
            },
            {
                "source": "traumatic_axonal_injury_load",
                "target": "white_matter_network_disconnection",
                "relation": "disrupts large-scale connectivity through diffuse axonal injury",
                "medical_ncd_change": "increased",
            },
            {
                "source": "metabolic_systemic_encephalopathy_load",
                "target": "metabolic_encephalopathic_slowing",
                "relation": "produces diffuse slowing and confusional inefficiency",
                "medical_ncd_change": "increased",
            },
            {
                "source": "inflammatory_infectious_cns_load",
                "target": "limbic_temporal_injury",
                "relation": "injures limbic and temporal systems in autoimmune or infectious encephalitic states",
                "medical_ncd_change": "increased",
            },
            {
                "source": "inflammatory_infectious_cns_load",
                "target": "white_matter_network_disconnection",
                "relation": "adds demyelinating or inflammatory disconnection burden",
                "medical_ncd_change": "increased",
            },
            {
                "source": "mass_effect_hydrocephalus_load",
                "target": "strategic_focal_lesion_burden",
                "relation": "compresses or distorts focal cognitive circuits",
                "medical_ncd_change": "increased",
            },
            {
                "source": "seizure_burden",
                "target": "gaba_inhibitory_failure",
                "relation": "reflects inhibitory breakdown and hyperexcitability",
                "medical_ncd_change": "increased",
            },
            {
                "source": "injury_susceptibility",
                "target": "distributed_cognitive_network_failure",
                "relation": "reduces network resilience across medical etiologies",
                "medical_ncd_change": "increased",
            },
            {
                "source": "glutamate_excitotoxicity",
                "target": "distributed_cognitive_network_failure",
                "relation": "propagates neuronal loss into broad cognitive circuit failure",
                "medical_ncd_change": "increased",
            },
            {
                "source": "white_matter_network_disconnection",
                "target": "executive_control_circuit_failure",
                "relation": "disconnects frontal-subcortical and frontoparietal control loops",
                "medical_ncd_change": "increased",
            },
            {
                "source": "strategic_focal_lesion_burden",
                "target": "thalamus_proxy",
                "relation": "captures thalamic and nearby strategic lesion effects",
                "medical_ncd_change": "increased",
            },
            {
                "source": "strategic_focal_lesion_burden",
                "target": "frontal_subcortical_proxy",
                "relation": "captures disruption of frontal-subcortical loops",
                "medical_ncd_change": "increased",
            },
            {
                "source": "executive_control_circuit_failure",
                "target": "frontal_control_proxy",
                "relation": "maps dysexecutive burden onto frontal control cortex",
                "medical_ncd_change": "increased",
            },
            {
                "source": "limbic_temporal_injury",
                "target": "limbic_temporal_proxy",
                "relation": "maps inflammatory or infectious burden onto medial temporal-limbic circuitry",
                "medical_ncd_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "processing_speed_slowing",
                "relation": "strategic thalamic dysfunction slows relay-dependent cognition",
                "medical_ncd_change": "increased",
            },
            {
                "source": "frontal_control_proxy",
                "target": "executive_dysfunction",
                "relation": "frontal control failure produces dysexecutive symptoms",
                "medical_ncd_change": "increased",
            },
            {
                "source": "frontal_control_proxy",
                "target": "behavioral_apathy_personality_change",
                "relation": "frontal lesions can drive apathy and personality change",
                "medical_ncd_change": "increased",
            },
            {
                "source": "frontal_subcortical_proxy",
                "target": "executive_dysfunction",
                "relation": "frontal-subcortical loop disruption weakens executive performance",
                "medical_ncd_change": "increased",
            },
            {
                "source": "limbic_temporal_proxy",
                "target": "memory_impairment",
                "relation": "limbic-temporal damage produces amnestic burden",
                "medical_ncd_change": "increased",
            },
            {
                "source": "limbic_temporal_proxy",
                "target": "social_cognitive_change",
                "relation": "limbic-temporal dysfunction can alter social and emotional interpretation",
                "medical_ncd_change": "increased",
            },
            {
                "source": "gaba_inhibitory_failure",
                "target": "confusional_state_or_delirium",
                "relation": "hyperexcitability and seizure activity can present as fluctuating confusion",
                "medical_ncd_change": "increased",
            },
            {
                "source": "distributed_cognitive_network_failure",
                "target": "global_cognitive_decline",
                "relation": "network-level dysfunction yields multidomain cognitive deterioration",
                "medical_ncd_change": "increased",
            },
            {
                "source": "medical_reversal_rehabilitation_support",
                "target": "distributed_cognitive_network_failure",
                "relation": "correction of the underlying condition can partially buffer network dysfunction",
                "medical_ncd_change": "decreased",
            },
            {
                "source": "medical_reversal_rehabilitation_support",
                "target": "confusional_state_or_delirium",
                "relation": "medical stabilization can reduce acute confusional burden",
                "medical_ncd_change": "decreased",
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
        generic_penalty = 1 if name in {
            "thalamus",
            "hippocampus",
            "amygdala",
            "frontal lobe",
            "subcortical",
        } else 0
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
                        "description": "Atlas-backed node or conservative proxy unresolved in this environment.",
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
                    "description": "Atlas-backed circuit node or proxy anchor.",
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
        vascular_ischemic_injury_load: float = 0.45,
        traumatic_axonal_injury_load: float = 0.25,
        metabolic_systemic_encephalopathy_load: float = 0.20,
        inflammatory_infectious_cns_load: float = 0.15,
        mass_effect_hydrocephalus_load: float = 0.10,
        seizure_burden: float = 0.15,
        genetic_susceptibility: float = 0.30,
        medical_reversal_rehabilitation_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """
        One-pass normalized simulator.

        Values represent burden or support on a 0..1 scale.
        Higher regional-state values reflect greater dysfunction burden,
        not healthier activity.
        """

        inputs = pd.Series(
            {
                "vascular_ischemic_injury_load": self._clip01(vascular_ischemic_injury_load),
                "traumatic_axonal_injury_load": self._clip01(traumatic_axonal_injury_load),
                "metabolic_systemic_encephalopathy_load": self._clip01(metabolic_systemic_encephalopathy_load),
                "inflammatory_infectious_cns_load": self._clip01(inflammatory_infectious_cns_load),
                "mass_effect_hydrocephalus_load": self._clip01(mass_effect_hydrocephalus_load),
                "seizure_burden": self._clip01(seizure_burden),
                "genetic_susceptibility": self._clip01(genetic_susceptibility),
                "medical_reversal_rehabilitation_support": self._clip01(
                    medical_reversal_rehabilitation_support
                ),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")

        # Compute latents sequentially to keep causal ordering explicit.
        latents.loc["injury_susceptibility"] = self._clip01(
            0.60 * inputs["genetic_susceptibility"]
            + 0.15 * inputs["vascular_ischemic_injury_load"]
            + 0.10 * inputs["metabolic_systemic_encephalopathy_load"]
            + 0.05 * inputs["inflammatory_infectious_cns_load"]
            - 0.15 * inputs["medical_reversal_rehabilitation_support"]
        )
        latents.loc["glutamate_excitotoxicity"] = self._clip01(
            0.40 * inputs["vascular_ischemic_injury_load"]
            + 0.30 * inputs["traumatic_axonal_injury_load"]
            + 0.20 * latents["injury_susceptibility"]
            + 0.05 * inputs["inflammatory_infectious_cns_load"]
            - 0.15 * inputs["medical_reversal_rehabilitation_support"]
        )
        latents.loc["gaba_inhibitory_failure"] = self._clip01(
            0.50 * inputs["seizure_burden"]
            + 0.20 * inputs["metabolic_systemic_encephalopathy_load"]
            + 0.15 * inputs["inflammatory_infectious_cns_load"]
            + 0.10 * inputs["traumatic_axonal_injury_load"]
            + 0.05 * latents["injury_susceptibility"]
            - 0.10 * inputs["medical_reversal_rehabilitation_support"]
        )
        latents.loc["strategic_focal_lesion_burden"] = self._clip01(
            0.35 * inputs["vascular_ischemic_injury_load"]
            + 0.20 * inputs["mass_effect_hydrocephalus_load"]
            + 0.15 * inputs["inflammatory_infectious_cns_load"]
            + 0.15 * inputs["traumatic_axonal_injury_load"]
            + 0.15 * latents["injury_susceptibility"]
            - 0.10 * inputs["medical_reversal_rehabilitation_support"]
        )
        latents.loc["white_matter_network_disconnection"] = self._clip01(
            0.40 * inputs["vascular_ischemic_injury_load"]
            + 0.30 * inputs["traumatic_axonal_injury_load"]
            + 0.10 * inputs["inflammatory_infectious_cns_load"]
            + 0.10 * inputs["metabolic_systemic_encephalopathy_load"]
            + 0.10 * latents["injury_susceptibility"]
            - 0.15 * inputs["medical_reversal_rehabilitation_support"]
        )
        latents.loc["metabolic_encephalopathic_slowing"] = self._clip01(
            0.60 * inputs["metabolic_systemic_encephalopathy_load"]
            + 0.15 * inputs["seizure_burden"]
            + 0.10 * inputs["mass_effect_hydrocephalus_load"]
            + 0.10 * inputs["inflammatory_infectious_cns_load"]
            + 0.05 * latents["injury_susceptibility"]
            - 0.20 * inputs["medical_reversal_rehabilitation_support"]
        )
        latents.loc["limbic_temporal_injury"] = self._clip01(
            0.40 * inputs["inflammatory_infectious_cns_load"]
            + 0.15 * inputs["mass_effect_hydrocephalus_load"]
            + 0.15 * latents["glutamate_excitotoxicity"]
            + 0.15 * latents["strategic_focal_lesion_burden"]
            + 0.15 * latents["injury_susceptibility"]
            - 0.10 * inputs["medical_reversal_rehabilitation_support"]
        )
        latents.loc["executive_control_circuit_failure"] = self._clip01(
            0.35 * latents["white_matter_network_disconnection"]
            + 0.25 * latents["strategic_focal_lesion_burden"]
            + 0.15 * latents["metabolic_encephalopathic_slowing"]
            + 0.15 * latents["gaba_inhibitory_failure"]
            + 0.10 * latents["injury_susceptibility"]
            - 0.10 * inputs["medical_reversal_rehabilitation_support"]
        )
        latents.loc["distributed_cognitive_network_failure"] = self._clip01(
            0.20 * latents["injury_susceptibility"]
            + 0.20 * latents["glutamate_excitotoxicity"]
            + 0.20 * latents["white_matter_network_disconnection"]
            + 0.15 * latents["metabolic_encephalopathic_slowing"]
            + 0.15 * latents["limbic_temporal_injury"]
            + 0.10 * latents["gaba_inhibitory_failure"]
            - 0.20 * inputs["medical_reversal_rehabilitation_support"]
        )

        regional_state = pd.Series(
            {
                "thalamus_proxy": self._clip01(
                    0.45 * latents["strategic_focal_lesion_burden"]
                    + 0.25 * latents["metabolic_encephalopathic_slowing"]
                    + 0.15 * latents["white_matter_network_disconnection"]
                    + 0.15 * latents["distributed_cognitive_network_failure"]
                ),
                "frontal_control_proxy": self._clip01(
                    0.45 * latents["executive_control_circuit_failure"]
                    + 0.20 * latents["white_matter_network_disconnection"]
                    + 0.15 * latents["metabolic_encephalopathic_slowing"]
                    + 0.10 * inputs["mass_effect_hydrocephalus_load"]
                    + 0.10 * latents["distributed_cognitive_network_failure"]
                ),
                "frontal_subcortical_proxy": self._clip01(
                    0.35 * latents["white_matter_network_disconnection"]
                    + 0.30 * latents["strategic_focal_lesion_burden"]
                    + 0.15 * latents["gaba_inhibitory_failure"]
                    + 0.10 * inputs["traumatic_axonal_injury_load"]
                    + 0.10 * latents["distributed_cognitive_network_failure"]
                ),
                "limbic_temporal_proxy": self._clip01(
                    0.50 * latents["limbic_temporal_injury"]
                    + 0.20 * latents["glutamate_excitotoxicity"]
                    + 0.15 * inputs["inflammatory_infectious_cns_load"]
                    + 0.10 * inputs["mass_effect_hydrocephalus_load"]
                    + 0.05 * latents["distributed_cognitive_network_failure"]
                ),
            },
            name="regional_state",
        )

        symptoms = pd.Series(
            {
                "memory_impairment": self._clip01(
                    0.45 * regional_state["limbic_temporal_proxy"]
                    + 0.20 * regional_state["thalamus_proxy"]
                    + 0.20 * latents["distributed_cognitive_network_failure"]
                    + 0.15 * latents["metabolic_encephalopathic_slowing"]
                ),
                "executive_dysfunction": self._clip01(
                    0.40 * regional_state["frontal_control_proxy"]
                    + 0.25 * regional_state["frontal_subcortical_proxy"]
                    + 0.20 * latents["white_matter_network_disconnection"]
                    + 0.15 * regional_state["thalamus_proxy"]
                ),
                "processing_speed_slowing": self._clip01(
                    0.40 * latents["white_matter_network_disconnection"]
                    + 0.30 * latents["metabolic_encephalopathic_slowing"]
                    + 0.20 * regional_state["thalamus_proxy"]
                    + 0.10 * latents["distributed_cognitive_network_failure"]
                ),
                "social_cognitive_change": self._clip01(
                    0.30 * regional_state["frontal_control_proxy"]
                    + 0.25 * regional_state["limbic_temporal_proxy"]
                    + 0.25 * latents["distributed_cognitive_network_failure"]
                    + 0.20 * inputs["inflammatory_infectious_cns_load"]
                ),
                "behavioral_apathy_personality_change": self._clip01(
                    0.35 * regional_state["frontal_control_proxy"]
                    + 0.25 * regional_state["frontal_subcortical_proxy"]
                    + 0.20 * inputs["mass_effect_hydrocephalus_load"]
                    + 0.20 * latents["distributed_cognitive_network_failure"]
                ),
                "confusional_state_or_delirium": self._clip01(
                    0.35 * latents["metabolic_encephalopathic_slowing"]
                    + 0.25 * latents["gaba_inhibitory_failure"]
                    + 0.15 * regional_state["thalamus_proxy"]
                    + 0.15 * inputs["seizure_burden"]
                    + 0.10 * inputs["mass_effect_hydrocephalus_load"]
                    - 0.15 * inputs["medical_reversal_rehabilitation_support"]
                ),
            },
            name="symptoms",
        )
        symptoms.loc["global_cognitive_decline"] = self._clip01(
            0.25 * symptoms["memory_impairment"]
            + 0.25 * symptoms["executive_dysfunction"]
            + 0.15 * symptoms["processing_speed_slowing"]
            + 0.10 * symptoms["social_cognitive_change"]
            + 0.10 * symptoms["behavioral_apathy_personality_change"]
            + 0.15 * latents["distributed_cognitive_network_failure"]
        )

        phenotypes = pd.Series(
            {
                "vascular_disconnection_profile": self._clip01(
                    0.30 * inputs["vascular_ischemic_injury_load"]
                    + 0.30 * latents["white_matter_network_disconnection"]
                    + 0.20 * symptoms["executive_dysfunction"]
                    + 0.20 * symptoms["processing_speed_slowing"]
                ),
                "post_traumatic_dysexecutive_profile": self._clip01(
                    0.30 * inputs["traumatic_axonal_injury_load"]
                    + 0.25 * latents["glutamate_excitotoxicity"]
                    + 0.25 * regional_state["frontal_subcortical_proxy"]
                    + 0.20 * symptoms["executive_dysfunction"]
                ),
                "metabolic_encephalopathy_profile": self._clip01(
                    0.35 * inputs["metabolic_systemic_encephalopathy_load"]
                    + 0.25 * latents["metabolic_encephalopathic_slowing"]
                    + 0.20 * symptoms["confusional_state_or_delirium"]
                    + 0.20 * symptoms["global_cognitive_decline"]
                ),
                "limbic_temporal_amnestic_profile": self._clip01(
                    0.30 * inputs["inflammatory_infectious_cns_load"]
                    + 0.30 * regional_state["limbic_temporal_proxy"]
                    + 0.25 * symptoms["memory_impairment"]
                    + 0.15 * symptoms["social_cognitive_change"]
                ),
                "reversible_structural_mass_effect_profile": self._clip01(
                    0.35 * inputs["mass_effect_hydrocephalus_load"]
                    + 0.25 * symptoms["behavioral_apathy_personality_change"]
                    + 0.20 * symptoms["confusional_state_or_delirium"]
                    + 0.20 * symptoms["global_cognitive_decline"]
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

    model = NeurocognitiveDisorderDueToAnotherMedicalConditionModel()
    bundle = model.build()

    print("\n=== Nodes (first 24) ===")
    print(bundle["nodes"].head(24).to_string(index=False))

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
        vascular_ischemic_injury_load=0.65,
        traumatic_axonal_injury_load=0.25,
        metabolic_systemic_encephalopathy_load=0.20,
        inflammatory_infectious_cns_load=0.15,
        mass_effect_hydrocephalus_load=0.10,
        seizure_burden=0.20,
        genetic_susceptibility=0.35,
        medical_reversal_rehabilitation_support=0.25,
    )

    print("\n=== Simulation ===")
    for name, series in sim.items():
        print(f"\n{name.upper()}")
        print(series.to_string())

    if SIIBRA_AVAILABLE:
        print("\n=== Example region suggestions for 'thalamus' ===")
        print(model.suggest_regions("thalamus", limit=10).to_string(index=False))

        # Example coordinate assignment:
        # print(model.assign_mni_point((-8, -14, 8)).head().to_string(index=False))
