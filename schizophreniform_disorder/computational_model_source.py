from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore
    _SIIBRA_IMPORT_ERROR = exc
else:
    _SIIBRA_IMPORT_ERROR = None


SCHIZOPHRENIFORM_DISORDER_GENE_PANEL = [
    "COMT",
    "DRD2",
    "DRD1",
    "HTR2A",
    "GRIN2A",
    "GRM3",
    "GAD1",
    "RELN",
    "SLC6A3",
]


class SchizophreniformDisorderModel:
    """
    Atlas-grounded siibra scaffold for Schizophreniform Disorder.

    This script is a research scaffold, not a diagnostic, prognostic, or treatment
    tool. It translates a chapter-level description of Schizophreniform Disorder
    into a transparent mechanistic graph and a simple normalized simulator.

    Important modeling constraint:
    - The chapter explicitly states that disorder-specific neurobiological evidence
      is limited and that the best current model is extrapolated from the broader
      schizophrenia literature. This scaffold therefore treats the biology as a
      schizophrenia-spectrum proxy model rather than as a validated, specific
      biomarker model for Schizophreniform Disorder.

    Chapter-driven design choices:
    - Dopamine remains central, but serotonin, glutamate, and GABA are modeled as
      co-determining latent systems because the chapter emphasizes a distributed
      neurotransmission imbalance rather than a single-transmitter disorder.
    - Frontal cortex subregions, ACC, hippocampus, thalamus, temporal cortex,
      basal ganglia, and cerebellum are used as the main circuit anchors because
      the chapter names them repeatedly in structural and PET findings.
    - Basal ganglia, thalamus, temporal cortex, and cerebellum are retained as
      conservative proxies where exact Julich parcel labels can vary by siibra
      environment.
    - Atypical antipsychotic support is represented as a protective input because
      the chapter highlights combined D2 and 5-HT2A antagonism as biologically
      meaningful to symptom relief.

    Higher regional-state values produced by ``simulate()`` represent greater
    dysfunction burden in that node, not healthier activation.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        if siibra is None:  # pragma: no cover - environment dependent
            raise ImportError(
                "siibra is required to use SchizophreniformDisorderModel. "
                "Install siibra-python in your runtime before building the scaffold."
            ) from _SIIBRA_IMPORT_ERROR

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

        self.region_candidates: Dict[str, List[str]] = {
            "dlpfc": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "dorsolateral prefrontal cortex left",
                "dorsolateral prefrontal cortex",
            ],
            "ofc": [
                "Area Fo4 left",
                "Area Fo3 left",
                "orbitofrontal cortex left",
                "orbitofrontal cortex",
                "orbital frontal cortex",
            ],
            "mpfc": [
                "Area 10 left",
                "medial prefrontal cortex left",
                "medial prefrontal cortex",
                "frontal pole",
            ],
            "acc": [
                "Area p32 (pACC) left",
                "Area s24 (pACC) left",
                "Area p24ab left",
                "anterior cingulate cortex left",
                "anterior cingulate cortex",
                "anterior cingulate",
            ],
            "hippocampus": [
                "CA1 left",
                "CA2 left",
                "CA3 left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "thalamus_proxy": [
                "thalamus left",
                "thalamus",
            ],
            "basal_ganglia_proxy": [
                "caudate nucleus left",
                "putamen left",
                "nucleus accumbens left",
                "basal ganglia left",
                "basal ganglia",
            ],
            "temporal_cortex_proxy": [
                "superior temporal cortex left",
                "superior temporal gyrus left",
                "Area TE 1.0 left",
                "Area TE 1.2 left",
                "temporal cortex left",
                "temporal cortex",
            ],
            "cerebellum_proxy": [
                "cerebellum left",
                "cerebellum",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "dlpfc": "Dorsolateral prefrontal executive-control node implicated in planning, working memory, and behavioral regulation.",
            "ofc": "Orbitofrontal frontal-subregion node used as a proxy for frontal valuation and behavioral regulation burden.",
            "mpfc": "Medial prefrontal frontal-subregion node used as a proxy for medial frontal integration and self-referential control burden.",
            "acc": "Anterior cingulate node implicated in distributed frontal-limbic-thalamic dysfunction.",
            "hippocampus": "Medial temporal memory node repeatedly implicated in schizophrenia-spectrum structural and inhibitory abnormalities.",
            "thalamus_proxy": "Proxy relay node for thalamic gating and cortical-subcortical integration deficits.",
            "basal_ganglia_proxy": "Proxy node for striatal and broader basal ganglia involvement in dopaminergic dysregulation.",
            "temporal_cortex_proxy": "Proxy node for temporal-lobe subregion involvement in psychosis-relevant distributed dysconnectivity.",
            "cerebellum_proxy": "Proxy node for cerebellar participation in brain-wide dysconnectivity and cognitive coordination burden.",
        }

        self.input_nodes: Dict[str, str] = {
            "familial_genetic_liability": (
                "Inherited schizophrenia-spectrum risk conveyed through family loading and polygenic vulnerability"
            ),
            "neurodevelopmental_vulnerability": (
                "Subtle developmental vulnerability affecting synaptic organization, connectivity, and later psychosis risk"
            ),
            "environmental_stress_load": (
                "Environmental or developmental stress burden capable of amplifying latent psychosis vulnerability"
            ),
            "resilience_support": (
                "Protective resilience and stabilizing factors that may buffer expression of spectrum liability"
            ),
            "atypical_antipsychotic_support": (
                "Protective D2 and 5-HT2A antagonism approximating atypical antipsychotic support"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "schizotaxic_neurointegrative_defect": (
                "Underlying schizophrenia-spectrum neurointegrative vulnerability linking genetics, development, and later psychosis"
            ),
            "serotonergic_modulatory_dysregulation": (
                "5-HT dysfunction that modulates frontal and striatal dopaminergic tone and perceptual instability"
            ),
            "glutamatergic_excitation_pressure": (
                "Excitatory dysregulation contributing to cortical instability and transmitter imbalance"
            ),
            "gaba_interneuron_disinhibition": (
                "Reduced inhibitory tone in hippocampal and prefrontal circuits with downstream cortical disorganization"
            ),
            "dopaminergic_salience_dysregulation": (
                "Abnormal dopamine-mediated salience assignment central to psychotic expression"
            ),
            "early_information_processing_deficit": (
                "Early stimulus-processing and signal-filtering deficit linked to poor cortical signal-to-noise"
            ),
            "frontotemporal_thalamocerebellar_dysconnectivity": (
                "Brain-wide dysconnectivity across frontal, temporal, thalamic, subcortical, and cerebellar systems"
            ),
            "negative_symptom_circuit_burden": (
                "Cortical-subcortical burden associated with blunted motivation, diminished affective expression, and poor initiation"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "positive_psychosis_burden": (
                "Burden of hallucination, delusion, and overt psychotic salience distortion"
            ),
            "negative_symptom_burden": (
                "Burden of reduced motivation, flattened affect, and impoverished engagement"
            ),
            "cognitive_disorganization": (
                "Formal disorganization and poor coherence of thought or information integration"
            ),
            "attention_working_memory_impairment": (
                "Difficulty with attention, working memory, and on-line mental maintenance"
            ),
            "executive_dysfunction": (
                "Reduced planning, flexibility, inhibition, and strategic problem solving"
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "familial_genetic_liability",
                "target": "schizotaxic_neurointegrative_defect",
                "relation": "loads inherited schizophrenia-spectrum vulnerability into a latent neurointegrative defect",
                "schizophreniform_change": "increased",
            },
            {
                "source": "neurodevelopmental_vulnerability",
                "target": "schizotaxic_neurointegrative_defect",
                "relation": "contributes developmental instability affecting connectivity and circuit maturation",
                "schizophreniform_change": "increased",
            },
            {
                "source": "environmental_stress_load",
                "target": "schizotaxic_neurointegrative_defect",
                "relation": "adds stress pressure that can unmask or intensify latent spectrum vulnerability",
                "schizophreniform_change": "increased",
            },
            {
                "source": "resilience_support",
                "target": "schizotaxic_neurointegrative_defect",
                "relation": "buffers clinical expression of the underlying neurointegrative defect",
                "schizophreniform_change": "decreased",
            },
            {
                "source": "schizotaxic_neurointegrative_defect",
                "target": "serotonergic_modulatory_dysregulation",
                "relation": "creates susceptibility to dysregulated serotonin modulation of cortical and striatal systems",
                "schizophreniform_change": "increased",
            },
            {
                "source": "schizotaxic_neurointegrative_defect",
                "target": "glutamatergic_excitation_pressure",
                "relation": "creates excitatory vulnerability through altered synaptic and receptor-level development",
                "schizophreniform_change": "increased",
            },
            {
                "source": "schizotaxic_neurointegrative_defect",
                "target": "gaba_interneuron_disinhibition",
                "relation": "contributes to reduced inhibitory tone in hippocampal and prefrontal circuits",
                "schizophreniform_change": "increased",
            },
            {
                "source": "glutamatergic_excitation_pressure",
                "target": "gaba_interneuron_disinhibition",
                "relation": "adds excitatory stress that worsens inhibitory failure and cortical instability",
                "schizophreniform_change": "increased",
            },
            {
                "source": "serotonergic_modulatory_dysregulation",
                "target": "dopaminergic_salience_dysregulation",
                "relation": "modulates frontal and striatal dopamine pathways toward dysregulation",
                "schizophreniform_change": "increased",
            },
            {
                "source": "glutamatergic_excitation_pressure",
                "target": "dopaminergic_salience_dysregulation",
                "relation": "feeds dopaminergic instability through broader excitation-inhibition imbalance",
                "schizophreniform_change": "increased",
            },
            {
                "source": "gaba_interneuron_disinhibition",
                "target": "dopaminergic_salience_dysregulation",
                "relation": "reduces inhibitory buffering and permits downstream salience overload",
                "schizophreniform_change": "increased",
            },
            {
                "source": "atypical_antipsychotic_support",
                "target": "serotonergic_modulatory_dysregulation",
                "relation": "reduces 5-HT2A-linked dysregulation through atypical antipsychotic antagonism",
                "schizophreniform_change": "decreased",
            },
            {
                "source": "atypical_antipsychotic_support",
                "target": "dopaminergic_salience_dysregulation",
                "relation": "reduces D2-linked salience overload through antipsychotic blockade",
                "schizophreniform_change": "decreased",
            },
            {
                "source": "gaba_interneuron_disinhibition",
                "target": "early_information_processing_deficit",
                "relation": "reduces signal-to-noise and impairs very early information processing",
                "schizophreniform_change": "increased",
            },
            {
                "source": "glutamatergic_excitation_pressure",
                "target": "early_information_processing_deficit",
                "relation": "adds cortical instability that worsens early processing precision",
                "schizophreniform_change": "increased",
            },
            {
                "source": "schizotaxic_neurointegrative_defect",
                "target": "frontotemporal_thalamocerebellar_dysconnectivity",
                "relation": "creates distributed vulnerability across frontal, temporal, thalamic, and cerebellar systems",
                "schizophreniform_change": "increased",
            },
            {
                "source": "gaba_interneuron_disinhibition",
                "target": "frontotemporal_thalamocerebellar_dysconnectivity",
                "relation": "destabilizes long-range cortical-subcortical coordination",
                "schizophreniform_change": "increased",
            },
            {
                "source": "dopaminergic_salience_dysregulation",
                "target": "frontotemporal_thalamocerebellar_dysconnectivity",
                "relation": "pushes distributed salience and relay systems toward further dysconnectivity",
                "schizophreniform_change": "increased",
            },
            {
                "source": "resilience_support",
                "target": "frontotemporal_thalamocerebellar_dysconnectivity",
                "relation": "partially buffers distributed dysconnectivity burden",
                "schizophreniform_change": "decreased",
            },
            {
                "source": "frontotemporal_thalamocerebellar_dysconnectivity",
                "target": "negative_symptom_circuit_burden",
                "relation": "burdens distributed cortical-subcortical systems tied to initiation and motivation",
                "schizophreniform_change": "increased",
            },
            {
                "source": "serotonergic_modulatory_dysregulation",
                "target": "negative_symptom_circuit_burden",
                "relation": "contributes to broader cortical-subcortical symptom burden beyond positive psychosis",
                "schizophreniform_change": "increased",
            },
            {
                "source": "frontotemporal_thalamocerebellar_dysconnectivity",
                "target": "dlpfc",
                "relation": "maps dysconnectivity onto dorsolateral prefrontal executive circuits",
                "schizophreniform_change": "increased",
            },
            {
                "source": "frontotemporal_thalamocerebellar_dysconnectivity",
                "target": "ofc",
                "relation": "maps dysconnectivity onto orbital frontal regulatory circuits",
                "schizophreniform_change": "increased",
            },
            {
                "source": "frontotemporal_thalamocerebellar_dysconnectivity",
                "target": "mpfc",
                "relation": "maps dysconnectivity onto medial frontal integration circuits",
                "schizophreniform_change": "increased",
            },
            {
                "source": "frontotemporal_thalamocerebellar_dysconnectivity",
                "target": "acc",
                "relation": "maps dysconnectivity onto anterior cingulate monitoring and integration circuits",
                "schizophreniform_change": "increased",
            },
            {
                "source": "gaba_interneuron_disinhibition",
                "target": "hippocampus",
                "relation": "burdens hippocampal inhibitory control and contributes to medial temporal dysfunction",
                "schizophreniform_change": "increased",
            },
            {
                "source": "frontotemporal_thalamocerebellar_dysconnectivity",
                "target": "thalamus_proxy",
                "relation": "maps distributed relay and gating dysfunction onto thalamic systems",
                "schizophreniform_change": "increased",
            },
            {
                "source": "dopaminergic_salience_dysregulation",
                "target": "basal_ganglia_proxy",
                "relation": "maps dopaminergic imbalance onto striatal and basal ganglia circuitry",
                "schizophreniform_change": "increased",
            },
            {
                "source": "frontotemporal_thalamocerebellar_dysconnectivity",
                "target": "temporal_cortex_proxy",
                "relation": "maps network burden onto temporal lobe subregions implicated in psychosis",
                "schizophreniform_change": "increased",
            },
            {
                "source": "frontotemporal_thalamocerebellar_dysconnectivity",
                "target": "cerebellum_proxy",
                "relation": "maps distributed dysconnectivity onto cerebellar coordination systems",
                "schizophreniform_change": "increased",
            },
            {
                "source": "dopaminergic_salience_dysregulation",
                "target": "positive_psychosis_burden",
                "relation": "drives aberrant salience and psychotic expression",
                "schizophreniform_change": "increased",
            },
            {
                "source": "negative_symptom_circuit_burden",
                "target": "negative_symptom_burden",
                "relation": "drives blunting, amotivation, and reduced initiation",
                "schizophreniform_change": "increased",
            },
            {
                "source": "early_information_processing_deficit",
                "target": "cognitive_disorganization",
                "relation": "reduces coherent filtering and integration of incoming information",
                "schizophreniform_change": "increased",
            },
            {
                "source": "dlpfc",
                "target": "attention_working_memory_impairment",
                "relation": "prefrontal dysfunction burdens attention and working memory",
                "schizophreniform_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "attention_working_memory_impairment",
                "relation": "medial temporal dysfunction burdens memory contribution to cognition",
                "schizophreniform_change": "increased",
            },
            {
                "source": "dlpfc",
                "target": "executive_dysfunction",
                "relation": "frontal dysfunction weakens planning, flexibility, and inhibition",
                "schizophreniform_change": "increased",
            },
            {
                "source": "acc",
                "target": "executive_dysfunction",
                "relation": "cingulate dysfunction burdens conflict monitoring and cognitive control",
                "schizophreniform_change": "increased",
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

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    def _modality_candidates(self, kind: str) -> List[Any]:
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
        matches: List[Any] = []
        try:
            matches.extend(
                list(
                    self.atlas.find_regions(
                        query,
                        all_versions=False,
                        filter_children=False,
                        find_topmost=False,
                    )
                )
            )
        except Exception:
            pass
        try:
            if hasattr(self.parcellation, "find"):
                found = self.parcellation.find(query, filter_children=False)
                if found:
                    matches.extend(list(found))
        except Exception:
            pass

        out: List[Any] = []
        seen = set()
        for region in matches:
            key = (self._name_of(region), getattr(region, "identifier", None))
            if key in seen:
                continue
            seen.add(key)
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower() or self.parcellation.name in str(parc_name):
                out.append(region)
                continue
            if not parc_name:
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_terms = {
            "dorsolateral prefrontal cortex",
            "orbitofrontal cortex",
            "medial prefrontal cortex",
            "anterior cingulate cortex",
            "anterior cingulate",
            "hippocampus",
            "thalamus",
            "basal ganglia",
            "temporal cortex",
            "cerebellum",
            "frontal pole",
        }
        generic_penalty = 1 if name in generic_terms else 0
        parent_penalty = 1 if name.count("(") == 0 and "area" not in name else 0
        return (left_bonus, right_penalty, generic_penalty, parent_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
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
            series = pd.to_numeric(series, errors="coerce").dropna().sort_values(ascending=False)
            df = series.reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        matched: Dict[str, Any] = {}
        all_labels = list(dict.fromkeys(list(matrix.index) + list(matrix.columns)))
        for key, region in self.region_objects.items():
            label = self._match_region_label(all_labels, region)
            if label is not None:
                matched[key] = label

        if not matched:
            return pd.DataFrame()

        rows: List[pd.Series] = []
        for src_key, src_label in matched.items():
            row: Dict[str, float] = {}
            for dst_key, dst_label in matched.items():
                value = pd.NA
                try:
                    value = matrix.loc[src_label, dst_label]
                except Exception:
                    try:
                        value = matrix[dst_label].loc[src_label]
                    except Exception:
                        value = pd.NA
                row[dst_key] = value
            rows.append(pd.Series(row, name=src_key))
        return pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")

    def build(
        self,
        gene_panel: Sequence[str] = SCHIZOPHRENIFORM_DISORDER_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
        nodes = []
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
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region",
                        "description": self.region_node_descriptions.get(
                            key, "Atlas-backed region or proxy node (unresolved in this environment)"
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
                    "node_type": "region",
                    "description": self.region_node_descriptions.get(key, "Atlas-backed circuit node"),
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
            "metadata": {
                "model_name": self.__class__.__name__,
                "gene_panel": list(gene_panel),
                "connectivity_cohort": self.connectivity_cohort,
                "space": self.space_spec,
                "parcellation": self.parcellation_spec,
                "notes": (
                    "Research scaffold only. This is a schizophrenia-spectrum proxy model for Schizophreniform Disorder "
                    "because the chapter notes that disorder-specific neurobiological evidence is limited."
                ),
            },
        }

    def simulate(
        self,
        familial_genetic_liability: float = 0.72,
        neurodevelopmental_vulnerability: float = 0.65,
        environmental_stress_load: float = 0.45,
        resilience_support: float = 0.20,
        atypical_antipsychotic_support: float = 0.15,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized one-pass Schizophreniform Disorder simulation.

        All inputs are expected on a 0..1 scale. Higher support values are protective;
        all other inputs represent higher burden or risk. Returned regional-state
        values reflect dysfunction burden in the anchored system.
        """

        inputs = {
            "familial_genetic_liability": self._clip01(familial_genetic_liability),
            "neurodevelopmental_vulnerability": self._clip01(neurodevelopmental_vulnerability),
            "environmental_stress_load": self._clip01(environmental_stress_load),
            "resilience_support": self._clip01(resilience_support),
            "atypical_antipsychotic_support": self._clip01(atypical_antipsychotic_support),
        }

        latents = {
            "schizotaxic_neurointegrative_defect": 0.0,
            "serotonergic_modulatory_dysregulation": 0.0,
            "glutamatergic_excitation_pressure": 0.0,
            "gaba_interneuron_disinhibition": 0.0,
            "dopaminergic_salience_dysregulation": 0.0,
            "early_information_processing_deficit": 0.0,
            "frontotemporal_thalamocerebellar_dysconnectivity": 0.0,
            "negative_symptom_circuit_burden": 0.0,
        }

        latents["schizotaxic_neurointegrative_defect"] = self._clip01(
            0.40 * inputs["familial_genetic_liability"]
            + 0.34 * inputs["neurodevelopmental_vulnerability"]
            + 0.12 * inputs["environmental_stress_load"]
            - 0.18 * inputs["resilience_support"]
        )

        latents["serotonergic_modulatory_dysregulation"] = self._clip01(
            0.34 * latents["schizotaxic_neurointegrative_defect"]
            + 0.16 * inputs["environmental_stress_load"]
            - 0.28 * inputs["atypical_antipsychotic_support"]
        )

        latents["glutamatergic_excitation_pressure"] = self._clip01(
            0.34 * latents["schizotaxic_neurointegrative_defect"]
            + 0.18 * inputs["environmental_stress_load"]
            - 0.08 * inputs["atypical_antipsychotic_support"]
        )

        latents["gaba_interneuron_disinhibition"] = self._clip01(
            0.34 * latents["schizotaxic_neurointegrative_defect"]
            + 0.24 * latents["glutamatergic_excitation_pressure"]
            + 0.10 * inputs["environmental_stress_load"]
            - 0.10 * inputs["resilience_support"]
        )

        latents["dopaminergic_salience_dysregulation"] = self._clip01(
            0.28 * latents["schizotaxic_neurointegrative_defect"]
            + 0.22 * latents["serotonergic_modulatory_dysregulation"]
            + 0.18 * latents["glutamatergic_excitation_pressure"]
            + 0.14 * latents["gaba_interneuron_disinhibition"]
            + 0.08 * inputs["environmental_stress_load"]
            - 0.32 * inputs["atypical_antipsychotic_support"]
        )

        latents["early_information_processing_deficit"] = self._clip01(
            0.42 * latents["gaba_interneuron_disinhibition"]
            + 0.18 * latents["glutamatergic_excitation_pressure"]
            + 0.12 * latents["schizotaxic_neurointegrative_defect"]
            - 0.10 * inputs["resilience_support"]
        )

        latents["frontotemporal_thalamocerebellar_dysconnectivity"] = self._clip01(
            0.34 * latents["schizotaxic_neurointegrative_defect"]
            + 0.24 * latents["gaba_interneuron_disinhibition"]
            + 0.18 * latents["dopaminergic_salience_dysregulation"]
            + 0.10 * latents["glutamatergic_excitation_pressure"]
            + 0.08 * inputs["environmental_stress_load"]
            - 0.14 * inputs["resilience_support"]
        )

        latents["negative_symptom_circuit_burden"] = self._clip01(
            0.30 * latents["frontotemporal_thalamocerebellar_dysconnectivity"]
            + 0.22 * latents["serotonergic_modulatory_dysregulation"]
            + 0.16 * latents["early_information_processing_deficit"]
            + 0.10 * latents["schizotaxic_neurointegrative_defect"]
            - 0.08 * inputs["atypical_antipsychotic_support"]
            - 0.08 * inputs["resilience_support"]
        )

        regional_state = {
            "dlpfc": self._clip01(
                0.42 * latents["frontotemporal_thalamocerebellar_dysconnectivity"]
                + 0.22 * latents["gaba_interneuron_disinhibition"]
                + 0.14 * latents["early_information_processing_deficit"]
                + 0.10 * latents["negative_symptom_circuit_burden"]
            ),
            "ofc": self._clip01(
                0.34 * latents["frontotemporal_thalamocerebellar_dysconnectivity"]
                + 0.22 * latents["dopaminergic_salience_dysregulation"]
                + 0.16 * latents["serotonergic_modulatory_dysregulation"]
            ),
            "mpfc": self._clip01(
                0.36 * latents["frontotemporal_thalamocerebellar_dysconnectivity"]
                + 0.18 * latents["negative_symptom_circuit_burden"]
                + 0.14 * latents["serotonergic_modulatory_dysregulation"]
            ),
            "acc": self._clip01(
                0.38 * latents["frontotemporal_thalamocerebellar_dysconnectivity"]
                + 0.22 * latents["early_information_processing_deficit"]
                + 0.12 * latents["negative_symptom_circuit_burden"]
            ),
            "hippocampus": self._clip01(
                0.36 * latents["gaba_interneuron_disinhibition"]
                + 0.22 * latents["glutamatergic_excitation_pressure"]
                + 0.18 * latents["frontotemporal_thalamocerebellar_dysconnectivity"]
            ),
            "thalamus_proxy": self._clip01(
                0.40 * latents["frontotemporal_thalamocerebellar_dysconnectivity"]
                + 0.18 * latents["dopaminergic_salience_dysregulation"]
                + 0.12 * latents["early_information_processing_deficit"]
            ),
            "basal_ganglia_proxy": self._clip01(
                0.44 * latents["dopaminergic_salience_dysregulation"]
                + 0.18 * latents["serotonergic_modulatory_dysregulation"]
                + 0.10 * latents["frontotemporal_thalamocerebellar_dysconnectivity"]
            ),
            "temporal_cortex_proxy": self._clip01(
                0.34 * latents["frontotemporal_thalamocerebellar_dysconnectivity"]
                + 0.22 * latents["early_information_processing_deficit"]
                + 0.14 * latents["glutamatergic_excitation_pressure"]
            ),
            "cerebellum_proxy": self._clip01(
                0.32 * latents["frontotemporal_thalamocerebellar_dysconnectivity"]
                + 0.16 * latents["early_information_processing_deficit"]
                + 0.10 * latents["negative_symptom_circuit_burden"]
            ),
        }

        symptoms = {
            "positive_psychosis_burden": self._clip01(
                0.32 * latents["dopaminergic_salience_dysregulation"]
                + 0.16 * latents["serotonergic_modulatory_dysregulation"]
                + 0.12 * regional_state["temporal_cortex_proxy"]
                + 0.10 * regional_state["thalamus_proxy"]
                + 0.10 * regional_state["hippocampus"]
                + 0.08 * regional_state["basal_ganglia_proxy"]
                - 0.16 * inputs["atypical_antipsychotic_support"]
            ),
            "negative_symptom_burden": self._clip01(
                0.40 * latents["negative_symptom_circuit_burden"]
                + 0.16 * regional_state["mpfc"]
                + 0.14 * regional_state["dlpfc"]
                + 0.08 * regional_state["cerebellum_proxy"]
                - 0.08 * inputs["atypical_antipsychotic_support"]
            ),
            "cognitive_disorganization": self._clip01(
                0.30 * latents["early_information_processing_deficit"]
                + 0.18 * regional_state["dlpfc"]
                + 0.16 * regional_state["acc"]
                + 0.12 * regional_state["thalamus_proxy"]
                + 0.10 * regional_state["temporal_cortex_proxy"]
            ),
            "attention_working_memory_impairment": self._clip01(
                0.30 * regional_state["dlpfc"]
                + 0.20 * regional_state["hippocampus"]
                + 0.18 * latents["early_information_processing_deficit"]
                + 0.12 * regional_state["acc"]
            ),
            "executive_dysfunction": self._clip01(
                0.34 * regional_state["dlpfc"]
                + 0.18 * regional_state["ofc"]
                + 0.16 * regional_state["acc"]
                + 0.10 * latents["frontotemporal_thalamocerebellar_dysconnectivity"]
            ),
        }

        phenotypes = {
            "acute_schizophreniform_profile": self._clip01(
                (
                    symptoms["positive_psychosis_burden"]
                    + symptoms["cognitive_disorganization"]
                    + symptoms["executive_dysfunction"]
                )
                / 3.0
            ),
            "cortical_disorganization_profile": self._clip01(
                (
                    symptoms["cognitive_disorganization"]
                    + symptoms["attention_working_memory_impairment"]
                    + symptoms["executive_dysfunction"]
                )
                / 3.0
            ),
            "negative_symptom_profile": self._clip01(
                (
                    symptoms["negative_symptom_burden"]
                    + regional_state["mpfc"]
                    + regional_state["dlpfc"]
                )
                / 3.0
            ),
            "dysconnectivity_profile": self._clip01(
                (
                    latents["frontotemporal_thalamocerebellar_dysconnectivity"]
                    + regional_state["thalamus_proxy"]
                    + regional_state["cerebellum_proxy"]
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
        if self._pmap is None:
            with siibra.QUIET:
                self._pmap = siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=self.assignment_space,
                    maptype="statistical",
                )

        point = siibra.Point(tuple(float(x) for x in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        out = assignments.copy()
        if "region" in out.columns:
            out["region"] = out["region"].map(self._name_of)
        for candidate in (
            "map value",
            "correlation",
            "intersection over union",
            "contained",
            "contains",
        ):
            if candidate in out.columns:
                out = out.sort_values(candidate, ascending=False)
                break
        return out.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Any:
        region = self.region_objects.get(node_key)
        if region is None and node_key in self.region_candidates:
            region = self._resolve_region(self.region_candidates[node_key])
        if region is None:
            return None
        try:
            with siibra.QUIET:
                return region.get_regional_mask(self.space, maptype="labelled")
        except Exception:
            return None


if __name__ == "__main__":
    if siibra is None:  # pragma: no cover - environment dependent
        print("siibra is not installed in this runtime. Install siibra-python to build and query the scaffold.")
        raise SystemExit(0)

    model = SchizophreniformDisorderModel()
    built = model.build(connectivity_rows=10)

    print("\n=== NODES ===")
    print(
        built["nodes"][
            [
                "key",
                "node_type",
                "atlas_region",
                "region_identifier",
                "feature_summary",
            ]
        ].to_string(index=False)
    )

    print("\n=== EDGES ===")
    print(built["edges"].to_string(index=False))

    print("\n=== CIRCUIT CONNECTIVITY ===")
    circuit_df = built["circuit_connectivity"]
    if circuit_df.empty:
        print("No circuit connectivity matrix available in this environment.")
    else:
        print(circuit_df.round(3).to_string())

    for region_key in (
        "dlpfc",
        "ofc",
        "mpfc",
        "acc",
        "hippocampus",
        "thalamus_proxy",
        "basal_ganglia_proxy",
        "temporal_cortex_proxy",
        "cerebellum_proxy",
    ):
        receptor_df = built["receptors"].get(region_key, pd.DataFrame())
        gene_df = built["genes"].get(region_key, pd.DataFrame())
        conn_df = built["connectivity_profiles"].get(region_key, pd.DataFrame())

        print(f"\n=== RECEPTOR TABLE: {region_key} ===")
        if receptor_df.empty:
            print("No receptor fingerprint available.")
        else:
            print(receptor_df.head(10).to_string(index=False))

        print(f"\n=== GENE TABLE: {region_key} ===")
        if gene_df.empty:
            print("No gene-expression summary available.")
        else:
            print(gene_df.head(10).to_string(index=False))

        print(f"\n=== CONNECTIVITY PROFILE: {region_key} ===")
        if conn_df.empty:
            print("No connectivity profile available.")
        else:
            print(conn_df.to_string(index=False))

    print("\n=== SIMULATION EXAMPLE ===")
    simulated = model.simulate(
        familial_genetic_liability=0.78,
        neurodevelopmental_vulnerability=0.70,
        environmental_stress_load=0.52,
        resilience_support=0.18,
        atypical_antipsychotic_support=0.20,
    )
    for name, series in simulated.items():
        print(f"\n--- {name.upper()} ---")
        print(series.sort_values(ascending=False).round(3).to_string())

    # Example coordinate assignments in MNI152:
    # print(model.assign_mni_point((-42, 36, 28)).head())   # dorsolateral prefrontal vicinity
    # print(model.assign_mni_point((2, 30, 20)).head())     # anterior cingulate / medial prefrontal vicinity
    # print(model.assign_mni_point((-24, -18, -16)).head()) # hippocampal vicinity
