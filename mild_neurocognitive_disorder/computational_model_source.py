from __future__ import annotations

"""
Mild Neurocognitive Disorder siibra scaffold.

This script converts a chapter-level biological summary of Mild Neurocognitive
Disorder (MND) into an atlas-grounded, inspectable research scaffold organized
around siibra idioms. It is intended for hypothesis generation, teaching, and
reproducible multimodal exploration of candidate regions and circuit burden. It
is not a diagnostic or treatment tool.

Conceptual choices used here:
- The chapter explicitly frames MND as a heterogeneous clinical syndrome rather
  than a single disease, so the model keeps etiologies separate at the input
  level (neurodegenerative, cerebrovascular, traumatic, substance/medication,
  systemic medical, and genetic risk).
- The chapter gives specific anatomic emphasis to hippocampus and entorhinal
  cortex in amnestic presentations, and to posterior cingulate/precuneus,
  medial prefrontal cortex, and medial temporal structures in the default mode
  network. Those are kept as direct anchors where plausible or as conservative
  proxies when the chapter stays network-level.
- Frontotemporal and temporal-parietal spread are modeled as distinct latent
  pathways because the chapter separates amnestic/Alzheimer-like progression,
  frontotemporal presentations, and vascular cognitive impairment.
- The gene panel is deliberately broad and chapter-aligned: Alzheimer-related
  risk genes, frontotemporal degeneration genes, and glutamatergic/synaptic
  plasticity genes are included as mechanistic probes rather than as a validated
  disease-specific biomarker panel.

The scaffold degrades gracefully when siibra or individual multimodal features
are unavailable. Atlas-backed nodes remain visible in the graph, but unresolved
feature tables are returned empty instead of raising errors.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception:  # pragma: no cover - optional dependency in some environments
    siibra = None


DEFAULT_GENE_PANEL = [
    # Alzheimer-related genetic liability / prodromal amnestic pathways
    "APOE",
    "APP",
    "PSEN1",
    "PSEN2",
    # Frontotemporal degeneration pathways
    "MAPT",
    "GRN",
    "C9orf72",
    # Glutamatergic / synaptic plasticity pathways stressed in the chapter
    "GRIN1",
    "GRIN2B",
    "SLC1A2",
    "BDNF",
]


class MildNeurocognitiveDisorderModel:
    """
    Atlas-grounded research scaffold for Mild Neurocognitive Disorder.

    The simulator keeps a transparent one-pass causal flow:
        inputs -> latent biology -> regional burden -> symptoms -> phenotypes

    Because the chapter describes a syndrome with multiple etiologies, the model
    emphasizes mixed-pathway vulnerability rather than a single disease process.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        assignment_space: str = "mni152",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.siibra = siibra
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = assignment_space
        self.connectivity_cohort = connectivity_cohort

        self.atlas = None
        self.parcellation = None
        self.space = None
        self._pmap = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

        if self.siibra is None:
            warnings.warn(
                "siibra is not installed in this environment. Atlas-backed build "
                "steps will degrade gracefully, returning unresolved region nodes "
                "and empty feature tables until siibra is available."
            )
        else:
            try:
                self.atlas = self.siibra.atlases.get(atlas_spec)
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
            except Exception as exc:  # pragma: no cover - depends on local siibra data
                warnings.warn(
                    f"Could not initialize atlas/parcellation resources: {exc!r}. "
                    "Atlas-backed methods may return empty outputs."
                )

        self.region_candidates: Dict[str, List[str]] = {
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "HC-Subiculum (Hippocampus) left",
                "DG (Hippocampus) left",
                "hippocampus left",
                "hippocampus",
            ],
            "entorhinal_cortex": [
                "Entorhinal Cortex left",
                "entorhinal cortex",
                "Area 28 left",
                "Area 28/34 left",
                "entorhinal",
            ],
            "posterior_cingulate_precuneus_proxy": [
                "Area 7M (SPL) left",
                "Area 7P (SPL) left",
                "Area hPO1 (POS) left",
                "precuneus",
                "posterior cingulate",
            ],
            "medial_prefrontal_proxy": [
                "Area s32 (sACC) left",
                "Area p32 (pACC) left",
                "Area Fp1 (FPole) left",
                "Area 9 left",
                "medial prefrontal cortex",
            ],
            "parietal_association_proxy": [
                "Area PGp (IPL) left",
                "Area PFm (IPL) left",
                "Area 7P (SPL) left",
                "Area 7M (SPL) left",
                "inferior parietal",
                "parietal association cortex",
            ],
            "frontal_lobe_proxy": [
                "Area 46 left",
                "Area 9 left",
                "Area 45 (IFG) left",
                "Area 44 (IFG) left",
                "frontal lobe",
            ],
            "temporal_lobe_proxy": [
                "Area Ph1 (PhG) left",
                "Area Ph2 (PhG) left",
                "Area TE 3 (STG) left",
                "Area TE 2.2 (STG) left",
                "temporal lobe",
            ],
        }

        self.proxy_region_nodes = {
            "posterior_cingulate_precuneus_proxy",
            "medial_prefrontal_proxy",
            "parietal_association_proxy",
            "frontal_lobe_proxy",
            "temporal_lobe_proxy",
        }

        self.region_descriptions: Dict[str, str] = {
            "hippocampus": (
                "Direct atlas anchor for medial temporal memory circuitry highlighted "
                "as especially vulnerable in amnestic MND and prodromal Alzheimer-like disease."
            ),
            "entorhinal_cortex": (
                "Direct atlas anchor for entorhinal involvement emphasized in early "
                "amnestic presentations and strongly linked to episodic memory decline."
            ),
            "posterior_cingulate_precuneus_proxy": (
                "Conservative default-mode proxy for the posterior cingulate/precuneus "
                "hub named in the chapter."
            ),
            "medial_prefrontal_proxy": (
                "Conservative default-mode proxy for medial prefrontal involvement in "
                "internally directed cognition and network-level disruption."
            ),
            "parietal_association_proxy": (
                "Proxy for the temporal-parietal association-cortex spread described in "
                "amnestic/Alzheimer-like progression."
            ),
            "frontal_lobe_proxy": (
                "Proxy for disproportionate frontal lobe burden in frontotemporal variants "
                "and executive inefficiency."
            ),
            "temporal_lobe_proxy": (
                "Proxy for temporal lobe burden outside the hippocampus/entorhinal cortex, "
                "including frontotemporal and language-relevant involvement."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "neurodegenerative_burden": (
                "Aggregate burden from Alzheimer-like, Lewy-body, or other progressive neurodegenerative pathology."
            ),
            "cerebrovascular_burden": (
                "Burden from lacunes, ischemic injury, white matter disease, or strategic infarcts."
            ),
            "traumatic_brain_injury_burden": (
                "History or residual burden from traumatic brain injury contributing to later cognitive decline."
            ),
            "substance_medication_burden": (
                "Cognitive burden from alcohol, other medications, or medication effects noted as possible etiologies."
            ),
            "systemic_medical_burden": (
                "Burden from systemic medical conditions capable of degrading cognition."
            ),
            "polygenic_liability": (
                "Broad heritable risk for neurodegenerative and cognitive-vulnerability pathways."
            ),
            "apoe_related_ad_risk": (
                "Alzheimer-related inherited risk signal centered on late-onset susceptibility."
            ),
            "frontotemporal_mutation_load": (
                "Familial frontotemporal degeneration liability when a causative mutation is present."
            ),
            "nmda_protective_treatment": (
                "Protective NMDA-modulating treatment support, analogous to memantine-style anti-excitotoxic buffering."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "glutamatergic_excitotoxicity": (
                "Excess glutamatergic drive and excitotoxic injury affecting vulnerable neurons and circuits."
            ),
            "synaptic_plasticity_failure": (
                "Failure of efficient synaptic plasticity and neuronal communication needed for cognition."
            ),
            "medial_temporal_degeneration": (
                "Atrophy and dysfunction centered on hippocampal-entorhinal memory systems."
            ),
            "dmn_disconnectivity": (
                "Reduced connectivity within the default mode network, especially in amnestic presentations."
            ),
            "temporal_parietal_association_spread": (
                "Spread of pathology from medial temporal structures into temporal and parietal association cortices."
            ),
            "frontotemporal_lobar_degeneration": (
                "Disproportionate frontal and/or temporal degeneration linked to frontotemporal phenotypes."
            ),
            "vascular_network_injury": (
                "Circuit inefficiency driven by infarcts, ischemic injury, and vascular lesions in cognitively relevant regions."
            ),
            "white_matter_burden": (
                "Structural disconnection associated with white matter hyperintensities or diffuse vascular injury."
            ),
            "multidomain_network_inefficiency": (
                "Net loss of efficient communication across memory, executive, and association networks."
            ),
            "episodic_memory_circuit_failure": (
                "Failure of memory encoding and retrieval circuitry centered on medial temporal and DMN interactions."
            ),
            "executive_language_network_failure": (
                "Failure of frontal-temporal systems supporting executive control, language, and behavior regulation."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "episodic_memory_impairment": (
                "Reduced episodic memory performance typical of amnestic MND presentations."
            ),
            "executive_inefficiency": (
                "Reduced executive efficiency, judgment, and complex task organization."
            ),
            "language_behavior_change": (
                "Language or behavioral change characteristic of frontotemporal-spectrum presentations."
            ),
            "processing_speed_attention_deficit": (
                "Slowed processing speed and attentional inefficiency, often prominent in vascular burden."
            ),
            "multidomain_cognitive_decline": (
                "Mild but clinically meaningful decline across more than one cognitive domain."
            ),
            "progression_vulnerability": (
                "Risk that early impairment reflects ongoing disease progression toward broader decline."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "neurodegenerative_burden",
                "target": "glutamatergic_excitotoxicity",
                "relation": "degenerative pathology can amplify excitotoxic neuronal stress",
                "mnd_change": "increased",
            },
            {
                "source": "cerebrovascular_burden",
                "target": "glutamatergic_excitotoxicity",
                "relation": "ischemic injury contributes to glutamate-linked excitotoxic damage",
                "mnd_change": "increased",
            },
            {
                "source": "traumatic_brain_injury_burden",
                "target": "glutamatergic_excitotoxicity",
                "relation": "traumatic injury can drive excitotoxic cascades",
                "mnd_change": "increased",
            },
            {
                "source": "nmda_protective_treatment",
                "target": "glutamatergic_excitotoxicity",
                "relation": "NMDA-modulating treatment can buffer excitotoxic burden",
                "mnd_change": "decreased",
            },
            {
                "source": "glutamatergic_excitotoxicity",
                "target": "synaptic_plasticity_failure",
                "relation": "excitotoxic stress impairs the synaptic plasticity needed for learning and memory",
                "mnd_change": "increased",
            },
            {
                "source": "apoe_related_ad_risk",
                "target": "medial_temporal_degeneration",
                "relation": "Alzheimer-related genetic liability raises early medial temporal vulnerability",
                "mnd_change": "increased",
            },
            {
                "source": "neurodegenerative_burden",
                "target": "medial_temporal_degeneration",
                "relation": "prodromal Alzheimer-like pathology drives hippocampal and entorhinal atrophy",
                "mnd_change": "increased",
            },
            {
                "source": "synaptic_plasticity_failure",
                "target": "medial_temporal_degeneration",
                "relation": "synaptic failure undermines medial temporal memory systems",
                "mnd_change": "increased",
            },
            {
                "source": "medial_temporal_degeneration",
                "target": "hippocampus",
                "relation": "medial temporal pathology burdens the hippocampus",
                "mnd_change": "increased",
            },
            {
                "source": "medial_temporal_degeneration",
                "target": "entorhinal_cortex",
                "relation": "medial temporal pathology burdens the entorhinal cortex",
                "mnd_change": "increased",
            },
            {
                "source": "medial_temporal_degeneration",
                "target": "dmn_disconnectivity",
                "relation": "medial temporal pathology weakens the default mode network",
                "mnd_change": "increased",
            },
            {
                "source": "vascular_network_injury",
                "target": "dmn_disconnectivity",
                "relation": "vascular lesions can degrade large-scale network coherence",
                "mnd_change": "increased",
            },
            {
                "source": "dmn_disconnectivity",
                "target": "posterior_cingulate_precuneus_proxy",
                "relation": "default mode disruption burdens posterior cingulate/precuneus hub function",
                "mnd_change": "increased",
            },
            {
                "source": "dmn_disconnectivity",
                "target": "medial_prefrontal_proxy",
                "relation": "default mode disruption burdens medial prefrontal hub function",
                "mnd_change": "increased",
            },
            {
                "source": "medial_temporal_degeneration",
                "target": "temporal_parietal_association_spread",
                "relation": "amnestic pathology spreads from medial temporal structures to association cortex",
                "mnd_change": "increased",
            },
            {
                "source": "temporal_parietal_association_spread",
                "target": "parietal_association_proxy",
                "relation": "spread engages temporal-parietal association cortex",
                "mnd_change": "increased",
            },
            {
                "source": "frontotemporal_mutation_load",
                "target": "frontotemporal_lobar_degeneration",
                "relation": "familial mutation burden increases frontotemporal vulnerability",
                "mnd_change": "increased",
            },
            {
                "source": "neurodegenerative_burden",
                "target": "frontotemporal_lobar_degeneration",
                "relation": "frontotemporal-spectrum pathology burdens frontal and temporal systems",
                "mnd_change": "increased",
            },
            {
                "source": "frontotemporal_lobar_degeneration",
                "target": "frontal_lobe_proxy",
                "relation": "degeneration burdens frontal circuits",
                "mnd_change": "increased",
            },
            {
                "source": "frontotemporal_lobar_degeneration",
                "target": "temporal_lobe_proxy",
                "relation": "degeneration burdens temporal circuits",
                "mnd_change": "increased",
            },
            {
                "source": "cerebrovascular_burden",
                "target": "vascular_network_injury",
                "relation": "vascular lesions impair cognitively relevant circuits",
                "mnd_change": "increased",
            },
            {
                "source": "vascular_network_injury",
                "target": "white_matter_burden",
                "relation": "vascular burden is associated with white matter hyperintensity and disconnection load",
                "mnd_change": "increased",
            },
            {
                "source": "white_matter_burden",
                "target": "multidomain_network_inefficiency",
                "relation": "white matter damage reduces efficient network communication",
                "mnd_change": "increased",
            },
            {
                "source": "synaptic_plasticity_failure",
                "target": "episodic_memory_circuit_failure",
                "relation": "plasticity failure degrades memory encoding and retrieval circuitry",
                "mnd_change": "increased",
            },
            {
                "source": "dmn_disconnectivity",
                "target": "episodic_memory_circuit_failure",
                "relation": "DMN disruption undermines autobiographical and episodic memory support",
                "mnd_change": "increased",
            },
            {
                "source": "frontotemporal_lobar_degeneration",
                "target": "executive_language_network_failure",
                "relation": "frontotemporal burden weakens executive, language, and behavior networks",
                "mnd_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "episodic_memory_impairment",
                "relation": "hippocampal burden impairs episodic memory",
                "mnd_change": "increased",
            },
            {
                "source": "entorhinal_cortex",
                "target": "episodic_memory_impairment",
                "relation": "entorhinal burden impairs early memory encoding and retrieval support",
                "mnd_change": "increased",
            },
            {
                "source": "frontal_lobe_proxy",
                "target": "executive_inefficiency",
                "relation": "frontal burden degrades executive organization and control",
                "mnd_change": "increased",
            },
            {
                "source": "temporal_lobe_proxy",
                "target": "language_behavior_change",
                "relation": "temporal burden contributes to language and behavioral change",
                "mnd_change": "increased",
            },
            {
                "source": "white_matter_burden",
                "target": "processing_speed_attention_deficit",
                "relation": "disconnection burden slows processing and attention",
                "mnd_change": "increased",
            },
            {
                "source": "multidomain_network_inefficiency",
                "target": "multidomain_cognitive_decline",
                "relation": "global network inefficiency broadens impairment across domains",
                "mnd_change": "increased",
            },
            {
                "source": "frontotemporal_lobar_degeneration",
                "target": "progression_vulnerability",
                "relation": "ongoing degeneration raises risk of progression",
                "mnd_change": "increased",
            },
            {
                "source": "vascular_network_injury",
                "target": "progression_vulnerability",
                "relation": "vascular injury can push impairment into broader decline",
                "mnd_change": "increased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}
        self.region_resolution: Dict[str, Dict[str, Any]] = {}
        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    def _modality_candidates(self, kind: str) -> List[Any]:
        if self.siibra is None:
            return []
        cands: List[Any] = []
        try:
            if kind == "receptor":
                cands.append(self.siibra.features.molecular.ReceptorDensityFingerprint)
            elif kind == "gene":
                cands.append(self.siibra.features.molecular.GeneExpressions)
            elif kind == "connectivity":
                cands.append(self.siibra.features.connectivity.StreamlineCounts)
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
        if self.siibra is None:
            return []
        for modality in modalities:
            try:
                with self.siibra.QUIET:
                    feats = self.siibra.features.get(concept, modality, **kwargs)
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

        out = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower():
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {"hippocampus", "entorhinal cortex", "frontal lobe", "temporal lobe"} else 0
        proxy_penalty = 1 if "proxy" in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

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
        centroid_xyz = None
        if centroid is not None:
            try:
                centroid_xyz = tuple(float(x) for x in centroid)
            except Exception:
                centroid_xyz = None
        volume = getattr(main, "volume", None)
        try:
            volume_mm3 = float(volume) if volume is not None else None
        except Exception:
            volume_mm3 = None
        return centroid_xyz, volume_mm3

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            data = getattr(feat, "data", None)
            if isinstance(data, pd.DataFrame):
                df = data.copy().reset_index()
                if "index" in df.columns and "receptor" not in df.columns:
                    df = df.rename(columns={"index": "receptor"})
                return df
            try:
                df = pd.DataFrame(data).reset_index()
                if not df.empty:
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
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            lower_cols = {str(c).lower(): c for c in df.columns}
            required = {"gene", "level", "zscore"}
            if required.issubset(lower_cols):
                gene_col = lower_cols["gene"]
                level_col = lower_cols["level"]
                zscore_col = lower_cols["zscore"]
                try:
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
                    )
                except Exception:
                    return df.reset_index(drop=True)

            return df.reset_index(drop=True)
        return pd.DataFrame()

    def _extract_connectivity_data(self, feature: Any) -> pd.DataFrame:
        data = getattr(feature, "data", None)
        if isinstance(data, pd.DataFrame):
            return data.copy()
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()

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

        chosen = None
        for feat in feats:
            if getattr(feat, "cohort", None) == self.connectivity_cohort:
                chosen = feat
                break
        if chosen is None:
            chosen = feats[0]

        df = self._extract_connectivity_data(chosen)
        if df.empty:
            try:
                df = self._extract_connectivity_data(chosen[0])
            except Exception:
                df = pd.DataFrame()

        self._connectivity_matrix = df
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        if region is None:
            return None
        exact = [x for x in labels if self._name_of(x) == region.name]
        if exact:
            return exact[0]
        rn = region.name.lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        if fuzzy:
            return fuzzy[0]
        region_tokens = {tok for tok in rn.replace("(", " ").replace(")", " ").replace("-", " ").split() if len(tok) > 2}
        best = None
        best_score = 0
        for label in labels:
            label_name = self._name_of(label).lower()
            label_tokens = {tok for tok in label_name.replace("(", " ").replace(")", " ").replace("-", " ").split() if len(tok) > 2}
            score = len(region_tokens & label_tokens)
            if score > best_score:
                best = label
                best_score = score
        return best if best_score >= 2 else None

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
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = pd.Series(series).sort_values(ascending=False)
            df = series.reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name]
            return df.head(max_rows).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return a submatrix among resolved disorder-relevant regional nodes.

        Rows and columns are renamed to the model's stable node keys so the
        result stays readable even if atlas labels differ slightly across
        siibra versions.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        selected_index = []
        node_keys = []
        for node_key, region in self.region_objects.items():
            label = self._match_region_label(list(matrix.index), region)
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
            if label is None:
                continue
            selected_index.append(label)
            node_keys.append(node_key)

        if not selected_index:
            return pd.DataFrame()

        try:
            sub = matrix.loc[selected_index, selected_index].copy()
        except Exception:
            return pd.DataFrame()

        rename_map = {label: key for label, key in zip(selected_index, node_keys)}
        sub.index = [rename_map.get(idx, self._name_of(idx)) for idx in sub.index]
        sub.columns = [rename_map.get(col, self._name_of(col)) for col in sub.columns]
        return sub

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        nodes = []
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
        self.region_resolution = {}

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
            description = self.region_descriptions.get(key, "Atlas-backed proxy node")
            if region is None:
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                self.region_resolution[key] = {
                    "node_key": key,
                    "resolved": False,
                    "requested_candidates": list(candidates),
                    "resolved_name": None,
                    "identifier": None,
                }
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region_proxy" if key in self.proxy_region_nodes else "region",
                        "description": f"{description} Unresolved in this environment.",
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
            self.region_resolution[key] = {
                "node_key": key,
                "resolved": True,
                "requested_candidates": list(candidates),
                "resolved_name": region.name,
                "identifier": getattr(region, "identifier", None),
            }
            nodes.append(
                {
                    "key": key,
                    "label": region.name,
                    "node_type": "region_proxy" if key in self.proxy_region_nodes else "region",
                    "description": description,
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
            "region_resolution": pd.DataFrame(self.region_resolution.values()),
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": self.circuit_connectivity(),
        }

    def simulate(
        self,
        neurodegenerative_burden: float = 0.55,
        cerebrovascular_burden: float = 0.35,
        traumatic_brain_injury_burden: float = 0.15,
        substance_medication_burden: float = 0.10,
        systemic_medical_burden: float = 0.15,
        polygenic_liability: float = 0.40,
        apoe_related_ad_risk: float = 0.45,
        frontotemporal_mutation_load: float = 0.05,
        nmda_protective_treatment: float = 0.15,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        All values are clipped to [0, 1]. The arithmetic is intentionally simple
        and inspectable. This is a mechanistic scaffold, not a validated disease
        progression model.
        """
        inputs = {
            "neurodegenerative_burden": self._clip01(neurodegenerative_burden),
            "cerebrovascular_burden": self._clip01(cerebrovascular_burden),
            "traumatic_brain_injury_burden": self._clip01(traumatic_brain_injury_burden),
            "substance_medication_burden": self._clip01(substance_medication_burden),
            "systemic_medical_burden": self._clip01(systemic_medical_burden),
            "polygenic_liability": self._clip01(polygenic_liability),
            "apoe_related_ad_risk": self._clip01(apoe_related_ad_risk),
            "frontotemporal_mutation_load": self._clip01(frontotemporal_mutation_load),
            "nmda_protective_treatment": self._clip01(nmda_protective_treatment),
        }

        x = inputs

        latents = {
            "glutamatergic_excitotoxicity": self._clip01(
                0.28 * x["neurodegenerative_burden"]
                + 0.25 * x["cerebrovascular_burden"]
                + 0.18 * x["traumatic_brain_injury_burden"]
                + 0.10 * x["substance_medication_burden"]
                + 0.07 * x["systemic_medical_burden"]
                + 0.05 * x["polygenic_liability"]
                - 0.20 * x["nmda_protective_treatment"]
            ),
        }
        latents["synaptic_plasticity_failure"] = self._clip01(
            0.30 * latents["glutamatergic_excitotoxicity"]
            + 0.25 * x["neurodegenerative_burden"]
            + 0.10 * x["substance_medication_burden"]
            + 0.10 * x["systemic_medical_burden"]
            + 0.10 * x["polygenic_liability"]
            - 0.10 * x["nmda_protective_treatment"]
        )
        latents["medial_temporal_degeneration"] = self._clip01(
            0.35 * x["neurodegenerative_burden"]
            + 0.20 * x["apoe_related_ad_risk"]
            + 0.20 * latents["synaptic_plasticity_failure"]
            + 0.08 * x["substance_medication_burden"]
        )
        latents["vascular_network_injury"] = self._clip01(
            0.45 * x["cerebrovascular_burden"]
            + 0.12 * x["traumatic_brain_injury_burden"]
            + 0.10 * x["systemic_medical_burden"]
            + 0.08 * latents["glutamatergic_excitotoxicity"]
        )
        latents["dmn_disconnectivity"] = self._clip01(
            0.35 * latents["medial_temporal_degeneration"]
            + 0.20 * latents["synaptic_plasticity_failure"]
            + 0.20 * latents["vascular_network_injury"]
            + 0.10 * x["neurodegenerative_burden"]
        )
        latents["temporal_parietal_association_spread"] = self._clip01(
            0.35 * latents["medial_temporal_degeneration"]
            + 0.25 * x["neurodegenerative_burden"]
            + 0.20 * latents["dmn_disconnectivity"]
        )
        latents["frontotemporal_lobar_degeneration"] = self._clip01(
            0.30 * x["neurodegenerative_burden"]
            + 0.28 * x["frontotemporal_mutation_load"]
            + 0.12 * latents["synaptic_plasticity_failure"]
            + 0.08 * x["traumatic_brain_injury_burden"]
        )
        latents["white_matter_burden"] = self._clip01(
            0.45 * latents["vascular_network_injury"]
            + 0.12 * x["cerebrovascular_burden"]
            + 0.10 * x["traumatic_brain_injury_burden"]
            + 0.08 * x["substance_medication_burden"]
        )
        latents["multidomain_network_inefficiency"] = self._clip01(
            0.25 * latents["dmn_disconnectivity"]
            + 0.25 * latents["white_matter_burden"]
            + 0.20 * latents["synaptic_plasticity_failure"]
            + 0.12 * latents["frontotemporal_lobar_degeneration"]
            + 0.08 * x["substance_medication_burden"]
            + 0.05 * x["systemic_medical_burden"]
        )
        latents["episodic_memory_circuit_failure"] = self._clip01(
            0.40 * latents["medial_temporal_degeneration"]
            + 0.25 * latents["dmn_disconnectivity"]
            + 0.15 * latents["synaptic_plasticity_failure"]
        )
        latents["executive_language_network_failure"] = self._clip01(
            0.35 * latents["frontotemporal_lobar_degeneration"]
            + 0.20 * latents["white_matter_burden"]
            + 0.20 * latents["multidomain_network_inefficiency"]
        )

        regional_state = {
            "hippocampus": self._clip01(
                0.55 * latents["medial_temporal_degeneration"]
                + 0.20 * latents["episodic_memory_circuit_failure"]
                + 0.10 * latents["glutamatergic_excitotoxicity"]
            ),
            "entorhinal_cortex": self._clip01(
                0.50 * latents["medial_temporal_degeneration"]
                + 0.20 * latents["synaptic_plasticity_failure"]
                + 0.10 * x["apoe_related_ad_risk"]
            ),
            "posterior_cingulate_precuneus_proxy": self._clip01(
                0.45 * latents["dmn_disconnectivity"]
                + 0.20 * latents["temporal_parietal_association_spread"]
                + 0.10 * latents["vascular_network_injury"]
            ),
            "medial_prefrontal_proxy": self._clip01(
                0.40 * latents["dmn_disconnectivity"]
                + 0.20 * latents["white_matter_burden"]
                + 0.15 * latents["frontotemporal_lobar_degeneration"]
            ),
            "parietal_association_proxy": self._clip01(
                0.45 * latents["temporal_parietal_association_spread"]
                + 0.20 * latents["vascular_network_injury"]
                + 0.10 * latents["dmn_disconnectivity"]
            ),
            "frontal_lobe_proxy": self._clip01(
                0.45 * latents["frontotemporal_lobar_degeneration"]
                + 0.25 * latents["white_matter_burden"]
                + 0.10 * latents["multidomain_network_inefficiency"]
            ),
            "temporal_lobe_proxy": self._clip01(
                0.35 * latents["frontotemporal_lobar_degeneration"]
                + 0.30 * latents["temporal_parietal_association_spread"]
                + 0.15 * latents["medial_temporal_degeneration"]
            ),
        }

        symptoms = {
            "episodic_memory_impairment": self._clip01(
                0.35 * regional_state["hippocampus"]
                + 0.25 * regional_state["entorhinal_cortex"]
                + 0.20 * latents["dmn_disconnectivity"]
                + 0.10 * latents["episodic_memory_circuit_failure"]
            ),
            "executive_inefficiency": self._clip01(
                0.35 * regional_state["frontal_lobe_proxy"]
                + 0.20 * latents["white_matter_burden"]
                + 0.20 * latents["multidomain_network_inefficiency"]
                + 0.10 * regional_state["medial_prefrontal_proxy"]
            ),
            "language_behavior_change": self._clip01(
                0.30 * regional_state["temporal_lobe_proxy"]
                + 0.30 * regional_state["frontal_lobe_proxy"]
                + 0.25 * latents["executive_language_network_failure"]
            ),
            "processing_speed_attention_deficit": self._clip01(
                0.35 * latents["white_matter_burden"]
                + 0.25 * latents["vascular_network_injury"]
                + 0.15 * regional_state["parietal_association_proxy"]
            ),
        }
        symptoms["multidomain_cognitive_decline"] = self._clip01(
            (
                symptoms["episodic_memory_impairment"]
                + symptoms["executive_inefficiency"]
                + symptoms["processing_speed_attention_deficit"]
                + 0.8 * symptoms["language_behavior_change"]
                + latents["multidomain_network_inefficiency"]
            )
            / 4.8
        )
        symptoms["progression_vulnerability"] = self._clip01(
            0.28 * x["neurodegenerative_burden"]
            + 0.20 * latents["vascular_network_injury"]
            + 0.18 * symptoms["multidomain_cognitive_decline"]
            + 0.15 * latents["dmn_disconnectivity"]
            + 0.10 * latents["frontotemporal_lobar_degeneration"]
        )

        phenotypes = {
            "amnestic_mnd_profile": self._clip01(
                (
                    symptoms["episodic_memory_impairment"]
                    + regional_state["hippocampus"]
                    + regional_state["entorhinal_cortex"]
                    + latents["dmn_disconnectivity"]
                )
                / 4.0
            ),
            "frontotemporal_mnd_profile": self._clip01(
                (
                    symptoms["language_behavior_change"]
                    + regional_state["frontal_lobe_proxy"]
                    + regional_state["temporal_lobe_proxy"]
                    + latents["frontotemporal_lobar_degeneration"]
                )
                / 4.0
            ),
            "vascular_mnd_profile": self._clip01(
                (
                    symptoms["processing_speed_attention_deficit"]
                    + latents["vascular_network_injury"]
                    + latents["white_matter_burden"]
                    + regional_state["parietal_association_proxy"]
                )
                / 4.0
            ),
            "default_mode_disconnectivity_profile": self._clip01(
                (
                    symptoms["episodic_memory_impairment"]
                    + regional_state["posterior_cingulate_precuneus_proxy"]
                    + regional_state["medial_prefrontal_proxy"]
                    + latents["dmn_disconnectivity"]
                )
                / 4.0
            ),
            "mixed_multidomain_profile": self._clip01(
                (
                    symptoms["multidomain_cognitive_decline"]
                    + symptoms["progression_vulnerability"]
                    + latents["multidomain_network_inefficiency"]
                )
                / 3.0
            ),
        }

        return {
            "inputs": pd.Series(inputs, name="value"),
            "latents": pd.Series(latents, name="value"),
            "regional_state": pd.Series(regional_state, name="value"),
            "symptoms": pd.Series(symptoms, name="value"),
            "phenotypes": pd.Series(phenotypes, name="value"),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Probabilistically assign an MNI coordinate to Julich regions.

        Returns an empty dataframe if siibra or map resources are unavailable.
        """
        if self.siibra is None or self.parcellation is None:
            warnings.warn("siibra resources are unavailable; returning empty assignment.")
            return pd.DataFrame()

        try:
            if self._pmap is None:
                with self.siibra.QUIET:
                    self._pmap = self.siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )

            point = self.siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
            with self.siibra.QUIET:
                assignments = self._pmap.assign(point)

            if not isinstance(assignments, pd.DataFrame):
                assignments = pd.DataFrame(assignments)

            for candidate in ("map value", "correlation", "intersection over union"):
                if candidate in assignments.columns:
                    assignments = assignments.sort_values(candidate, ascending=False)
                    break
            return assignments.reset_index(drop=True)
        except Exception as exc:
            warnings.warn(f"Coordinate assignment failed: {exc!r}")
            return pd.DataFrame()

    def region_mask(self, node_key: str) -> Any:
        """
        Fetch a regional mask for a resolved node.

        Returns a Nifti image if available, otherwise None.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            warnings.warn(f"No resolved region available for '{node_key}'.")
            return None
        try:
            mask = region.get_regional_mask(space=self.assignment_space, maptype="labelled")
            return mask.fetch() if hasattr(mask, "fetch") else mask
        except Exception as exc:
            warnings.warn(f"Could not fetch region mask for '{node_key}': {exc!r}")
            return None


if __name__ == "__main__":
    model = MildNeurocognitiveDisorderModel()
    bundle = model.build()

    print("\n=== Nodes (head) ===")
    print(bundle["nodes"].head(12).to_string(index=False))

    print("\n=== Edges (head) ===")
    print(bundle["edges"].head(12).to_string(index=False))

    print("\n=== Region resolution ===")
    rr = bundle["region_resolution"]
    if isinstance(rr, pd.DataFrame) and not rr.empty:
        print(rr.to_string(index=False))
    else:
        print("No region resolution information available.")

    print("\n=== Example connectivity subgraph ===")
    cc = bundle["circuit_connectivity"]
    if isinstance(cc, pd.DataFrame) and not cc.empty:
        print(cc.to_string())
    else:
        print("No circuit connectivity matrix available in this environment.")

    print("\n=== Example receptor table (first non-empty region) ===")
    printed = False
    for key, df in bundle["receptors"].items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"[{key}]")
            print(df.head(10).to_string(index=False))
            printed = True
            break
    if not printed:
        print("No receptor fingerprint available in this environment.")

    print("\n=== Example gene table (first non-empty region) ===")
    printed = False
    for key, df in bundle["genes"].items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"[{key}]")
            print(df.head(10).to_string(index=False))
            printed = True
            break
    if not printed:
        print("No gene-expression table available in this environment.")

    print("\n=== Example connectivity profile (first non-empty region) ===")
    printed = False
    for key, df in bundle["connectivity_profiles"].items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"[{key}]")
            print(df.head(10).to_string(index=False))
            printed = True
            break
    if not printed:
        print("No connectivity profile available in this environment.")

    print("\n=== Simulation example ===")
    sim = model.simulate(
        neurodegenerative_burden=0.65,
        cerebrovascular_burden=0.30,
        traumatic_brain_injury_burden=0.10,
        substance_medication_burden=0.08,
        systemic_medical_burden=0.12,
        polygenic_liability=0.45,
        apoe_related_ad_risk=0.55,
        frontotemporal_mutation_load=0.05,
        nmda_protective_treatment=0.20,
    )
    for name, series in sim.items():
        print(f"\n[{name}]")
        print(series.sort_values(ascending=False).to_string())

    # Example coordinate assignment, if siibra is installed and resources are available:
    # print(model.assign_mni_point((-26, -18, -20)).head())
    # print(model.suggest_regions("entorhinal").head(10).to_string(index=False))
