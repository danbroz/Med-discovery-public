
from __future__ import annotations

"""
AlzheimersDiseaseNeurocognitiveDisorderModel
============================================

Atlas-grounded research scaffold for Major or Mild Neurocognitive Disorder
Due to Alzheimer's Disease.

This script translates a chapter-level biological summary of Alzheimer's
disease (AD) into a simple, transparent mechanistic model that can be
explored with siibra when atlas data are available. It is intended for
research scaffolding and hypothesis tracing, not diagnosis or treatment.

Chapter-derived themes emphasized here:
- Amyloid-beta (Aβ) oligomer burden is treated as an early upstream driver
  of synaptic toxicity and circuit disconnection.
- Tau / neurofibrillary tangle burden is modeled as the process most closely
  tracking neurodegeneration and clinical severity.
- Medial temporal structures are anchored conservatively, with hippocampus
  and entorhinal cortex as the core early-memory circuit nodes.
- Default-mode network disruption is represented using posterior cingulate /
  precuneus and medial-prefrontal proxies because the chapter is systems-level
  and exact Julich labels can vary.
- Later temporal and parietal cortical involvement is modeled as distributed
  cortical atrophy rather than as a single focal lesion.

The scaffold degrades gracefully when siibra is unavailable: the simulator
still runs, while atlas-backed feature retrieval methods return empty tables.
"""

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
    # Core AD risk and amyloid processing
    "APOE",
    "APP",
    "PSEN1",
    "PSEN2",
    "BACE1",
    "SORL1",
    # Tau and axonal / trafficking biology
    "MAPT",
    "BIN1",
    "CLU",
    "PICALM",
    # Microglial / inflammatory and late-onset risk biology
    "TREM2",
    "CR1",
    "ABCA7",
    # Synaptic and excitatory / inhibitory balance
    "SYN1",
    "DLG4",
    "GRIN1",
    "GRIN2B",
    "SLC17A7",
    "GAD1",
    "GAD2",
    # Plasticity / resilience
    "BDNF",
]


class AlzheimersDiseaseNeurocognitiveDisorderModel:
    """
    Research scaffold for Alzheimer's disease neurocognitive disorder.

    The graph is a chapter-faithful interpretation of AD as a progression from
    amyloid accumulation to tau-associated neurodegeneration, synaptic
    disconnection, medial-temporal memory-network failure, and later distributed
    cortical and default-mode network disruption.
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

        # Conservative region mapping from the chapter.
        self.region_candidates: Dict[str, List[str]] = {
            "hippocampus": [
                "CA1 left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "entorhinal_cortex": [
                "Area 28 left",
                "Area 28/34 left",
                "Area 34 left",
                "entorhinal cortex left",
                "entorhinal cortex",
                "entorhinal",
            ],
            "posterior_cingulate_precuneus_proxy": [
                "Area 31 left",
                "Area 23c left",
                "precuneus left",
                "posterior cingulate cortex",
                "posterior cingulate",
                "precuneus",
            ],
            "medial_prefrontal_cortex_proxy": [
                "Area p32 (pACC) left",
                "Area s32 left",
                "Area 10m left",
                "medial prefrontal cortex",
                "anterior cingulate cortex",
                "frontopolar cortex",
            ],
            "lateral_temporal_cortex_proxy": [
                "Area TE 3 left",
                "Area TE 1.0 left",
                "Area TE 1.1 left",
                "middle temporal gyrus",
                "superior temporal gyrus",
                "temporal cortex",
            ],
            "inferior_parietal_cortex_proxy": [
                "Area PGp left",
                "Area PGa left",
                "Area PFm left",
                "inferior parietal lobule",
                "angular gyrus",
                "supramarginal gyrus",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "late_life_sporadic_risk": (
                "Late-onset sporadic Alzheimer's risk load reflecting the complex interaction "
                "of aging, genetics, and environment described in the chapter"
            ),
            "apoe_e4_load": (
                "APOE ε4-associated burden affecting amyloid aggregation, clearance, synaptic "
                "function, and neuroinflammation"
            ),
            "familial_amyloidogenic_mutation_load": (
                "Rare APP / PSEN1 / PSEN2 mutation burden used to model high-confidence "
                "amyloid-pathway causality"
            ),
            "amyloid_clearance_production_imbalance": (
                "Imbalance between Aβ production and clearance, especially for Aβ42-prone accumulation"
            ),
            "pathology_duration": (
                "Accumulated preclinical-to-clinical disease duration capturing gradual burden accrual"
            ),
            "cognitive_reserve_support": (
                "Protective reserve / compensatory capacity moderating translation from pathology "
                "to overt clinical impairment"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "amyloidogenic_genetic_liability": (
                "Inherited amyloid-pathway risk spanning APOE-mediated clearance burden and rare APP/PSEN mutations"
            ),
            "amyloid_beta_oligomer_burden": (
                "Neurotoxic soluble Aβ oligomer accumulation producing early synaptotoxic effects"
            ),
            "tau_neurofibrillary_tangle_burden": (
                "Tau hyperphosphorylation / NFT burden more closely associated with neurodegeneration and symptoms"
            ),
            "neuroinflammation_clearance_stress": (
                "Inflammatory and clearance stress linked to Aβ accumulation and ApoE-mediated vulnerability"
            ),
            "synaptic_disconnection": (
                "Loss of efficient communication across circuits due to synaptotoxicity and synapse failure"
            ),
            "inhibitory_excitatory_synapse_loss": (
                "Loss of both glutamatergic and GABAergic synaptic integrity in affected networks"
            ),
            "network_hyperexcitability": (
                "Hyperexcitable network state promoted especially by inhibitory synapse loss"
            ),
            "medial_temporal_neurodegeneration": (
                "Hippocampal / entorhinal degenerative burden underlying early memory impairment"
            ),
            "default_mode_network_disconnection": (
                "Disrupted posterior-cingulate / precuneus / medial-prefrontal / hippocampal network coupling"
            ),
            "distributed_cortical_atrophy": (
                "Later temporal and parietal cortical atrophy extending beyond the medial temporal lobe"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "episodic_memory_impairment": (
                "Difficulty forming or retrieving episodic memories driven by medial-temporal pathology"
            ),
            "memory_consolidation_failure": (
                "Breakdown of consolidation and relay functions associated with entorhinal-hippocampal compromise"
            ),
            "default_mode_cognitive_disruption": (
                "Impaired internally directed cognition and network integration associated with DMN failure"
            ),
            "memory_led_mci": (
                "Memory-predominant mild cognitive impairment phenotype emerging from early circuit damage"
            ),
            "global_cognitive_decline": (
                "Broader decline as cortical atrophy and network dysfunction spread"
            ),
            "functional_impairment": (
                "Loss of day-to-day functional independence consistent with major neurocognitive disorder"
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "late_life_sporadic_risk",
                "target": "amyloidogenic_genetic_liability",
                "relation": "contributes background late-onset vulnerability",
                "ad_change": "increased",
            },
            {
                "source": "apoe_e4_load",
                "target": "amyloidogenic_genetic_liability",
                "relation": "raises inherited burden through impaired clearance and aggregation bias",
                "ad_change": "increased",
            },
            {
                "source": "familial_amyloidogenic_mutation_load",
                "target": "amyloidogenic_genetic_liability",
                "relation": "adds rare high-penetrance amyloid-pathway burden",
                "ad_change": "increased",
            },
            {
                "source": "amyloid_clearance_production_imbalance",
                "target": "amyloid_beta_oligomer_burden",
                "relation": "permits progressive Aβ accumulation",
                "ad_change": "increased",
            },
            {
                "source": "amyloidogenic_genetic_liability",
                "target": "amyloid_beta_oligomer_burden",
                "relation": "promotes aggregation and persistence of toxic Aβ species",
                "ad_change": "increased",
            },
            {
                "source": "pathology_duration",
                "target": "amyloid_beta_oligomer_burden",
                "relation": "captures long preclinical accrual of amyloid pathology",
                "ad_change": "increased",
            },
            {
                "source": "apoe_e4_load",
                "target": "neuroinflammation_clearance_stress",
                "relation": "worsens inflammatory and clearance burden",
                "ad_change": "increased",
            },
            {
                "source": "amyloid_beta_oligomer_burden",
                "target": "neuroinflammation_clearance_stress",
                "relation": "triggers inflammatory and injury cascades",
                "ad_change": "increased",
            },
            {
                "source": "amyloid_beta_oligomer_burden",
                "target": "tau_neurofibrillary_tangle_burden",
                "relation": "drives downstream tau pathology in the cascade model",
                "ad_change": "increased",
            },
            {
                "source": "pathology_duration",
                "target": "tau_neurofibrillary_tangle_burden",
                "relation": "allows tau burden and clinical relevance to accumulate over time",
                "ad_change": "increased",
            },
            {
                "source": "familial_amyloidogenic_mutation_load",
                "target": "tau_neurofibrillary_tangle_burden",
                "relation": "supports downstream tau burden through amyloidogenic mechanisms",
                "ad_change": "increased",
            },
            {
                "source": "amyloid_beta_oligomer_burden",
                "target": "synaptic_disconnection",
                "relation": "directly disrupts synaptic transmission and plasticity",
                "ad_change": "increased",
            },
            {
                "source": "tau_neurofibrillary_tangle_burden",
                "target": "synaptic_disconnection",
                "relation": "tracks neuronal dysfunction and loss more closely than plaque load",
                "ad_change": "increased",
            },
            {
                "source": "neuroinflammation_clearance_stress",
                "target": "synaptic_disconnection",
                "relation": "adds injury pressure to already vulnerable circuits",
                "ad_change": "increased",
            },
            {
                "source": "synaptic_disconnection",
                "target": "inhibitory_excitatory_synapse_loss",
                "relation": "progresses into broader glutamatergic and GABAergic synapse loss",
                "ad_change": "increased",
            },
            {
                "source": "tau_neurofibrillary_tangle_burden",
                "target": "inhibitory_excitatory_synapse_loss",
                "relation": "worsens synaptic and neuronal compromise",
                "ad_change": "increased",
            },
            {
                "source": "inhibitory_excitatory_synapse_loss",
                "target": "network_hyperexcitability",
                "relation": "loss of inhibitory balance promotes hyperexcitable states",
                "ad_change": "increased",
            },
            {
                "source": "amyloid_beta_oligomer_burden",
                "target": "network_hyperexcitability",
                "relation": "contributes to abnormal excitability and circuit instability",
                "ad_change": "increased",
            },
            {
                "source": "tau_neurofibrillary_tangle_burden",
                "target": "medial_temporal_neurodegeneration",
                "relation": "drives hippocampal and entorhinal degeneration",
                "ad_change": "increased",
            },
            {
                "source": "synaptic_disconnection",
                "target": "medial_temporal_neurodegeneration",
                "relation": "progresses from synaptic failure to neurodegenerative loss",
                "ad_change": "increased",
            },
            {
                "source": "medial_temporal_neurodegeneration",
                "target": "hippocampus",
                "relation": "loads the hippocampal memory system with degenerative burden",
                "ad_change": "increased",
            },
            {
                "source": "medial_temporal_neurodegeneration",
                "target": "entorhinal_cortex",
                "relation": "loads the entorhinal gateway with early degenerative burden",
                "ad_change": "increased",
            },
            {
                "source": "synaptic_disconnection",
                "target": "default_mode_network_disconnection",
                "relation": "reduces communication efficiency within the default mode network",
                "ad_change": "increased",
            },
            {
                "source": "medial_temporal_neurodegeneration",
                "target": "default_mode_network_disconnection",
                "relation": "disconnects hippocampal memory circuitry from cortical hubs",
                "ad_change": "increased",
            },
            {
                "source": "network_hyperexcitability",
                "target": "default_mode_network_disconnection",
                "relation": "adds unstable signaling to already compromised network coupling",
                "ad_change": "increased",
            },
            {
                "source": "default_mode_network_disconnection",
                "target": "posterior_cingulate_precuneus_proxy",
                "relation": "burdens a core posterior DMN hub",
                "ad_change": "increased",
            },
            {
                "source": "default_mode_network_disconnection",
                "target": "medial_prefrontal_cortex_proxy",
                "relation": "burdens a core anterior DMN hub",
                "ad_change": "increased",
            },
            {
                "source": "tau_neurofibrillary_tangle_burden",
                "target": "distributed_cortical_atrophy",
                "relation": "drives later widespread cortical degeneration",
                "ad_change": "increased",
            },
            {
                "source": "pathology_duration",
                "target": "distributed_cortical_atrophy",
                "relation": "permits spread beyond early medial-temporal disease",
                "ad_change": "increased",
            },
            {
                "source": "default_mode_network_disconnection",
                "target": "distributed_cortical_atrophy",
                "relation": "tracks failure across higher-order cortical association networks",
                "ad_change": "increased",
            },
            {
                "source": "distributed_cortical_atrophy",
                "target": "lateral_temporal_cortex_proxy",
                "relation": "captures later temporal cortical involvement",
                "ad_change": "increased",
            },
            {
                "source": "distributed_cortical_atrophy",
                "target": "inferior_parietal_cortex_proxy",
                "relation": "captures later parietal cortical involvement",
                "ad_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "episodic_memory_impairment",
                "relation": "drives core episodic-memory deficits",
                "ad_change": "increased",
            },
            {
                "source": "entorhinal_cortex",
                "target": "memory_consolidation_failure",
                "relation": "disrupts relay and consolidation functions linked to memory formation",
                "ad_change": "increased",
            },
            {
                "source": "default_mode_network_disconnection",
                "target": "default_mode_cognitive_disruption",
                "relation": "reduces episodic-memory and resting-network integrity",
                "ad_change": "increased",
            },
            {
                "source": "synaptic_disconnection",
                "target": "default_mode_cognitive_disruption",
                "relation": "undermines distributed network communication",
                "ad_change": "increased",
            },
            {
                "source": "episodic_memory_impairment",
                "target": "memory_led_mci",
                "relation": "promotes amnestic mild cognitive impairment",
                "ad_change": "increased",
            },
            {
                "source": "memory_consolidation_failure",
                "target": "memory_led_mci",
                "relation": "adds encoding and consolidation failure to early clinical expression",
                "ad_change": "increased",
            },
            {
                "source": "default_mode_cognitive_disruption",
                "target": "global_cognitive_decline",
                "relation": "widens impairment beyond pure memory failure",
                "ad_change": "increased",
            },
            {
                "source": "distributed_cortical_atrophy",
                "target": "global_cognitive_decline",
                "relation": "broadens decline as cortical disease spreads",
                "ad_change": "increased",
            },
            {
                "source": "global_cognitive_decline",
                "target": "functional_impairment",
                "relation": "translates cognitive burden into day-to-day disability",
                "ad_change": "increased",
            },
            {
                "source": "cognitive_reserve_support",
                "target": "memory_led_mci",
                "relation": "buffers clinical expression despite biological burden",
                "ad_change": "decreased",
            },
            {
                "source": "cognitive_reserve_support",
                "target": "global_cognitive_decline",
                "relation": "partly offsets conversion of pathology into overt decline",
                "ad_change": "decreased",
            },
            {
                "source": "cognitive_reserve_support",
                "target": "functional_impairment",
                "relation": "buffers daily-function consequences of pathology",
                "ad_change": "decreased",
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
            if "julich" in str(parc_name).lower():
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {"hippocampus", "entorhinal cortex", "temporal cortex"} else 0
        return (left_bonus, right_penalty, generic_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if not self.siibra_available or self.atlas is None:
            return None

        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation)
            except Exception:
                pass
            try:
                return self.parcellation.get_region(spec) if self.parcellation is not None else None
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
            rows.append({"name": row[0], "identifier": row[1], "parcellation": row[2]})
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
            return df.reset_index(drop=True)
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

        if not self.siibra_available or self.parcellation is None:
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
            self._connectivity_matrix = compound[0].data.copy()
        except Exception:
            self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        exact = [x for x in labels if self._name_of(x) == region.name]
        if exact:
            return exact[0]
        rn = region.name.lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
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
            df = series.sort_values(ascending=False).reset_index()
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
                        "description": (
                            "Atlas-backed circuit proxy unresolved in this environment"
                            if key.endswith("_proxy")
                            else "Atlas-backed circuit node unresolved in this environment"
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
            conn_df = self._connectivity_profile(region, max_rows=connectivity_rows)

            self.receptors[key] = receptor_df
            self.genes[key] = gene_df
            self.connectivity_profiles[key] = conn_df

            nodes.append(
                {
                    "key": key,
                    "label": region.name,
                    "node_type": "region_proxy" if key.endswith("_proxy") else "region",
                    "description": (
                        "Atlas-backed circuit proxy"
                        if key.endswith("_proxy")
                        else "Atlas-backed circuit node"
                    ),
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
        late_life_sporadic_risk: float = 0.55,
        apoe_e4_load: float = 0.50,
        familial_amyloidogenic_mutation_load: float = 0.05,
        amyloid_clearance_production_imbalance: float = 0.65,
        pathology_duration: float = 0.60,
        cognitive_reserve_support: float = 0.30,
    ) -> Dict[str, pd.Series]:
        """
        Run a transparent one-pass normalized simulation.

        Inputs are clipped to [0, 1]. Higher values indicate greater burden,
        except cognitive_reserve_support, which is protective.
        """

        inputs = pd.Series(
            {
                "late_life_sporadic_risk": self._clip01(late_life_sporadic_risk),
                "apoe_e4_load": self._clip01(apoe_e4_load),
                "familial_amyloidogenic_mutation_load": self._clip01(familial_amyloidogenic_mutation_load),
                "amyloid_clearance_production_imbalance": self._clip01(amyloid_clearance_production_imbalance),
                "pathology_duration": self._clip01(pathology_duration),
                "cognitive_reserve_support": self._clip01(cognitive_reserve_support),
            }
        )

        latents = pd.Series(
            {
                "amyloidogenic_genetic_liability": self._clip01(
                    0.35 * inputs["apoe_e4_load"]
                    + 0.30 * inputs["familial_amyloidogenic_mutation_load"]
                    + 0.20 * inputs["late_life_sporadic_risk"]
                    + 0.10 * inputs["amyloid_clearance_production_imbalance"]
                ),
                "amyloid_beta_oligomer_burden": self._clip01(
                    0.30 * inputs["amyloid_clearance_production_imbalance"]
                    + 0.25 * inputs["apoe_e4_load"]
                    + 0.20 * inputs["pathology_duration"]
                    + 0.15 * inputs["familial_amyloidogenic_mutation_load"]
                    + 0.10 * inputs["late_life_sporadic_risk"]
                ),
            }
        )

        latents["neuroinflammation_clearance_stress"] = self._clip01(
            0.35 * latents["amyloid_beta_oligomer_burden"]
            + 0.25 * inputs["apoe_e4_load"]
            + 0.15 * inputs["late_life_sporadic_risk"]
            + 0.10 * inputs["pathology_duration"]
        )

        latents["tau_neurofibrillary_tangle_burden"] = self._clip01(
            0.40 * latents["amyloid_beta_oligomer_burden"]
            + 0.25 * inputs["pathology_duration"]
            + 0.15 * latents["neuroinflammation_clearance_stress"]
            + 0.10 * inputs["familial_amyloidogenic_mutation_load"]
            + 0.05 * inputs["apoe_e4_load"]
        )

        latents["synaptic_disconnection"] = self._clip01(
            0.35 * latents["amyloid_beta_oligomer_burden"]
            + 0.30 * latents["tau_neurofibrillary_tangle_burden"]
            + 0.15 * latents["neuroinflammation_clearance_stress"]
            + 0.10 * inputs["pathology_duration"]
        )

        latents["inhibitory_excitatory_synapse_loss"] = self._clip01(
            0.45 * latents["synaptic_disconnection"]
            + 0.25 * latents["tau_neurofibrillary_tangle_burden"]
            + 0.10 * latents["amyloid_beta_oligomer_burden"]
            + 0.10 * inputs["pathology_duration"]
        )

        latents["network_hyperexcitability"] = self._clip01(
            0.45 * latents["inhibitory_excitatory_synapse_loss"]
            + 0.20 * latents["amyloid_beta_oligomer_burden"]
            + 0.15 * latents["synaptic_disconnection"]
            + 0.05 * inputs["pathology_duration"]
        )

        latents["medial_temporal_neurodegeneration"] = self._clip01(
            0.40 * latents["tau_neurofibrillary_tangle_burden"]
            + 0.30 * latents["synaptic_disconnection"]
            + 0.15 * inputs["pathology_duration"]
            + 0.10 * latents["neuroinflammation_clearance_stress"]
        )

        latents["default_mode_network_disconnection"] = self._clip01(
            0.30 * latents["synaptic_disconnection"]
            + 0.25 * latents["medial_temporal_neurodegeneration"]
            + 0.15 * latents["network_hyperexcitability"]
            + 0.10 * latents["tau_neurofibrillary_tangle_burden"]
            - 0.10 * inputs["cognitive_reserve_support"]
        )

        latents["distributed_cortical_atrophy"] = self._clip01(
            0.35 * latents["tau_neurofibrillary_tangle_burden"]
            + 0.25 * latents["medial_temporal_neurodegeneration"]
            + 0.20 * inputs["pathology_duration"]
            + 0.10 * latents["default_mode_network_disconnection"]
        )

        regional_state = pd.Series(
            {
                "hippocampus": self._clip01(
                    0.45 * latents["medial_temporal_neurodegeneration"]
                    + 0.25 * latents["tau_neurofibrillary_tangle_burden"]
                    + 0.15 * latents["synaptic_disconnection"]
                    + 0.10 * inputs["pathology_duration"]
                ),
                "entorhinal_cortex": self._clip01(
                    0.50 * latents["medial_temporal_neurodegeneration"]
                    + 0.25 * latents["tau_neurofibrillary_tangle_burden"]
                    + 0.10 * latents["amyloid_beta_oligomer_burden"]
                    + 0.10 * latents["synaptic_disconnection"]
                ),
                "posterior_cingulate_precuneus_proxy": self._clip01(
                    0.35 * latents["default_mode_network_disconnection"]
                    + 0.20 * latents["synaptic_disconnection"]
                    + 0.20 * latents["distributed_cortical_atrophy"]
                    + 0.10 * latents["network_hyperexcitability"]
                ),
                "medial_prefrontal_cortex_proxy": self._clip01(
                    0.30 * latents["default_mode_network_disconnection"]
                    + 0.20 * latents["synaptic_disconnection"]
                    + 0.20 * latents["distributed_cortical_atrophy"]
                    + 0.10 * latents["network_hyperexcitability"]
                    - 0.05 * inputs["cognitive_reserve_support"]
                ),
                "lateral_temporal_cortex_proxy": self._clip01(
                    0.40 * latents["distributed_cortical_atrophy"]
                    + 0.20 * latents["tau_neurofibrillary_tangle_burden"]
                    + 0.15 * latents["default_mode_network_disconnection"]
                    + 0.10 * latents["synaptic_disconnection"]
                ),
                "inferior_parietal_cortex_proxy": self._clip01(
                    0.35 * latents["distributed_cortical_atrophy"]
                    + 0.25 * latents["default_mode_network_disconnection"]
                    + 0.15 * latents["synaptic_disconnection"]
                    + 0.10 * latents["tau_neurofibrillary_tangle_burden"]
                ),
            }
        )

        symptoms = pd.Series(
            {
                "episodic_memory_impairment": self._clip01(
                    0.40 * regional_state["hippocampus"]
                    + 0.30 * regional_state["entorhinal_cortex"]
                    + 0.15 * latents["synaptic_disconnection"]
                    + 0.10 * latents["default_mode_network_disconnection"]
                    - 0.25 * inputs["cognitive_reserve_support"]
                ),
                "memory_consolidation_failure": self._clip01(
                    0.35 * regional_state["entorhinal_cortex"]
                    + 0.25 * regional_state["hippocampus"]
                    + 0.15 * latents["tau_neurofibrillary_tangle_burden"]
                    + 0.10 * latents["synaptic_disconnection"]
                    - 0.15 * inputs["cognitive_reserve_support"]
                ),
                "default_mode_cognitive_disruption": self._clip01(
                    0.40 * latents["default_mode_network_disconnection"]
                    + 0.25 * regional_state["posterior_cingulate_precuneus_proxy"]
                    + 0.20 * regional_state["medial_prefrontal_cortex_proxy"]
                    + 0.05 * latents["network_hyperexcitability"]
                    - 0.15 * inputs["cognitive_reserve_support"]
                ),
            }
        )

        symptoms["memory_led_mci"] = self._clip01(
            0.35 * symptoms["episodic_memory_impairment"]
            + 0.25 * symptoms["memory_consolidation_failure"]
            + 0.15 * regional_state["hippocampus"]
            + 0.10 * latents["default_mode_network_disconnection"]
            - 0.20 * inputs["cognitive_reserve_support"]
        )

        symptoms["global_cognitive_decline"] = self._clip01(
            0.30 * latents["distributed_cortical_atrophy"]
            + 0.20 * symptoms["default_mode_cognitive_disruption"]
            + 0.15 * regional_state["lateral_temporal_cortex_proxy"]
            + 0.15 * regional_state["inferior_parietal_cortex_proxy"]
            + 0.10 * symptoms["memory_led_mci"]
            - 0.20 * inputs["cognitive_reserve_support"]
        )

        symptoms["functional_impairment"] = self._clip01(
            0.40 * symptoms["global_cognitive_decline"]
            + 0.25 * symptoms["memory_led_mci"]
            + 0.15 * symptoms["episodic_memory_impairment"]
            + 0.10 * symptoms["default_mode_cognitive_disruption"]
            - 0.20 * inputs["cognitive_reserve_support"]
        )

        phenotypes = pd.Series(
            {
                "preclinical_biological_ad_profile": self._clip01(
                    (
                        latents["amyloid_beta_oligomer_burden"]
                        + latents["tau_neurofibrillary_tangle_burden"]
                        + latents["medial_temporal_neurodegeneration"]
                        + latents["default_mode_network_disconnection"]
                    )
                    / 4.0
                    - 0.30 * symptoms["memory_led_mci"]
                    - 0.20 * symptoms["functional_impairment"]
                    + 0.10 * inputs["cognitive_reserve_support"]
                ),
                "amnestic_mci_profile": self._clip01(
                    (
                        symptoms["episodic_memory_impairment"]
                        + symptoms["memory_consolidation_failure"]
                        + symptoms["memory_led_mci"]
                        + regional_state["hippocampus"]
                        + regional_state["entorhinal_cortex"]
                    )
                    / 5.0
                ),
                "major_neurocognitive_disorder_profile": self._clip01(
                    (
                        symptoms["global_cognitive_decline"]
                        + symptoms["functional_impairment"]
                        + latents["distributed_cortical_atrophy"]
                        + symptoms["default_mode_cognitive_disruption"]
                    )
                    / 4.0
                ),
                "hyperexcitable_disconnective_profile": self._clip01(
                    (
                        latents["network_hyperexcitability"]
                        + latents["synaptic_disconnection"]
                        + latents["inhibitory_excitatory_synapse_loss"]
                        + latents["default_mode_network_disconnection"]
                    )
                    / 4.0
                ),
            }
        )

        return {
            "inputs": inputs,
            "latents": latents.sort_index(),
            "regional_state": regional_state.sort_index(),
            "symptoms": symptoms.sort_index(),
            "phenotypes": phenotypes.sort_index(),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Assign an MNI152 coordinate to the most likely parcellation regions.

        Returns a dataframe sorted by the most informative scoring column that is
        available in the current siibra environment.
        """
        if not self.siibra_available or self.atlas is None or self.parcellation is None:
            return pd.DataFrame(
                [{
                    "region": None,
                    "detail": "siibra atlas resources unavailable in this environment",
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

        for attr in ("get_regional_map", "fetch"):
            try:
                fn = getattr(region, attr)
            except Exception:
                continue
            try:
                if attr == "get_regional_map":
                    return fn("mni152", "statistical")
                return fn()
            except Exception:
                continue
        return None


if __name__ == "__main__":
    model = AlzheimersDiseaseNeurocognitiveDisorderModel()
    build = model.build()

    print("\n=== Nodes (first 14) ===")
    print(build["nodes"].head(14).to_string(index=False))

    print("\n=== Edges (first 14) ===")
    print(build["edges"].head(14).to_string(index=False))

    print("\n=== Resolved region keys ===")
    print(list(build["regions"].keys()))

    sample_key = "hippocampus"
    print(f"\n=== Receptor table: {sample_key} ===")
    print(build["receptors"].get(sample_key, pd.DataFrame()).head().to_string(index=False))

    print(f"\n=== Gene table: {sample_key} ===")
    print(build["genes"].get(sample_key, pd.DataFrame()).head().to_string(index=False))

    print(f"\n=== Connectivity profile: {sample_key} ===")
    print(build["connectivity_profiles"].get(sample_key, pd.DataFrame()).head().to_string(index=False))

    sim = model.simulate(
        late_life_sporadic_risk=0.65,
        apoe_e4_load=0.70,
        familial_amyloidogenic_mutation_load=0.05,
        amyloid_clearance_production_imbalance=0.75,
        pathology_duration=0.70,
        cognitive_reserve_support=0.25,
    )

    print("\n=== Simulated latents ===")
    print(sim["latents"].to_string())

    print("\n=== Simulated symptoms ===")
    print(sim["symptoms"].to_string())

    print("\n=== Phenotype summary ===")
    print(sim["phenotypes"].to_string())

    # Optional example:
    # print(model.assign_mni_point((-24, -18, -18)).head().to_string(index=False))
