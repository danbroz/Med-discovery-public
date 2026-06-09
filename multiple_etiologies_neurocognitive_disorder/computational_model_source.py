from __future__ import annotations

"""
Major or Mild Neurocognitive Disorder Due to Multiple Etiologies siibra scaffold.

This script turns a biologically broad chapter into a transparent, atlas-grounded
research scaffold. The source chapter emphasizes cumulative and interacting
insults rather than a single lesion or a single molecular pathway, so the model
is intentionally organized around mixed burden:

- cerebrovascular disease,
- traumatic brain injury,
- metabolic / inflammatory stress,
- chronic psychosocial stress,
- psychiatric / neurodevelopmental vulnerability,
- and mixed neurodegenerative contributions.

The chapter provides enough anatomical content to justify cautious anchors for
hippocampus, orbitofrontal cortex, prefrontal control systems, temporal
association cortex, basal ganglia, and cerebellum. However, diffuse white matter
hyperintensities and axonal injury are better modeled as latent distributed
processes rather than over-forced into single parcels.

This is a research scaffold for mechanistic exploration only. It is not a
validated disease model, not a diagnostic tool, and not a treatment guide.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception:  # pragma: no cover - import guard for portable execution
    siibra = None  # type: ignore


DEFAULT_GENE_PANEL = [
    "GRIN2B",   # glutamatergic excitotoxicity / synaptic plasticity
    "GRIA2",    # AMPA receptor signaling
    "SLC1A2",   # astrocytic glutamate clearance
    "GAD1",     # GABA synthesis
    "GABRA1",   # inhibitory signaling proxy
    "BDNF",     # plasticity / stress vulnerability
    "APOE",     # neurodegenerative / vascular risk proxy
    "MAPT",     # frontotemporal degeneration proxy
    "COMT",     # prefrontal dopamine regulation
    "DRD2",     # striatal / frontal-subcortical dopamine signaling
    "SLC6A4",   # stress-affective serotonergic proxy
    "FKBP5",    # stress sensitivity / diathesis-stress proxy
    "IL6",      # inflammatory signaling
    "TNF",      # inflammatory signaling
    "MBP",      # white-matter / myelin integrity proxy
]


class MultipleEtiologiesNeurocognitiveDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Major or Mild Neurocognitive
    Disorder Due to Multiple Etiologies.

    Conceptual reading of the chapter:
    - The syndrome is treated as a final common pathway of distributed brain
      dysfunction produced by cumulative and interacting etiologies.
    - Excitation / inhibition imbalance, excitotoxic burden, stress-related
      hippocampal vulnerability, white-matter disconnection, and frontal-
      subcortical loop failure are central latent processes.
    - Memory, executive, psychomotor, and behavioral symptoms are modeled as
      partially separable downstream phenotypes that can coexist in mixed
      presentations.
    - White matter hyperintensities and diffuse axonal injury remain latent
      because the chapter describes them as distributed imaging findings rather
      than parcel-specific lesions.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        self.atlas = None
        self.parcellation = None
        self.space = None

        if siibra is None:
            warnings.warn(
                "siibra is not installed in this environment. Atlas lookups, "
                "feature queries, masks, and coordinate assignment will return "
                "empty results, but build() and simulate() remain usable."
            )
        else:
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
                    f"Could not initialize siibra atlas resources: {exc}. "
                    "The scaffold will continue with unresolved atlas nodes."
                )
                self.atlas = None
                self.parcellation = None
                self.space = None

        # The chapter justifies these anchors directly or via strong implication:
        # - hippocampus from medial temporal lobe atrophy / stress vulnerability
        # - OFC from trauma-related and behavioral-disinhibition framing
        # - prefrontal control from frontal-subcortical executive circuits
        # - temporal association cortex from frontotemporal involvement
        # - basal ganglia and cerebellum because the chapter names them explicitly
        self.region_candidates: Dict[str, List[str]] = {
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "CA2 (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "DG (Hippocampus) left",
                "hippocampus left",
                "hippocampus",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Area Fo2 (OFC) left",
                "Area Fo1 (OFC) left",
                "orbitofrontal",
                "ofc",
            ],
            "pfc_control": [
                "Area 46 left",
                "Area 9 left",
                "Area 45 (IFG) left",
                "Area 44 (IFG) left",
                "dlpfc",
                "prefrontal cortex",
                "frontal cortex",
            ],
            "temporal_association_proxy": [
                "Area TE 3 (STG) left",
                "Area TE 2.1 (STG) left",
                "Area TE 1.0 (HESCHL) left",
                "temporal cortex",
                "temporal lobe",
                "temporal",
            ],
            "basal_ganglia_proxy": [
                "Caudate nucleus left",
                "Putamen left",
                "striatum left",
                "basal ganglia left",
                "basal ganglia",
            ],
            "cerebellum_proxy": [
                "Cerebellum left",
                "cerebellum left",
                "cerebellum",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "cerebrovascular_disease_burden": (
                "Vascular lesion burden and chronic small-vessel injury contributing to "
                "white-matter damage and subcortical cognitive slowing"
            ),
            "traumatic_brain_injury_burden": (
                "Focal contusion and diffuse axonal injury burden from traumatic brain injury"
            ),
            "metabolic_inflammatory_burden": (
                "Metabolic disturbance and neuroinflammatory load increasing distributed neural stress"
            ),
            "chronic_stress_burden": (
                "Chronic psychosocial stress that can dysregulate glutamatergic signaling and weaken cognition"
            ),
            "psychiatric_genetic_liability": (
                "Inherited liability from affective, psychotic, personality, or substance-related comorbid risk"
            ),
            "neurodevelopmental_liability": (
                "Neurodevelopmental vulnerability from autism / ADHD-like developmental circuitry differences"
            ),
            "substance_use_burden": (
                "Addictive or neurotoxic exposure burden worsening cognitive and motivational outcomes"
            ),
            "medial_temporal_neurodegenerative_burden": (
                "Amnestic / Alzheimer-like medial temporal process contributing to memory decline"
            ),
            "frontotemporal_neurodegenerative_burden": (
                "Frontal and temporal degenerative burden contributing to behavioral and executive change"
            ),
            "cognitive_reserve": (
                "Protective reserve buffering clinical expression of mixed pathology"
            ),
            "etiology_treatment_support": (
                "Correction or management of underlying medical, psychiatric, and lifestyle contributors"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "mixed_pathology_accumulation": (
                "Cumulative burden produced by interacting vascular, traumatic, metabolic, stress, and psychiatric insults"
            ),
            "glutamatergic_excitotoxic_pressure": (
                "Excitatory overload and excitotoxic damage pressure from stroke, TBI, inflammation, and stress"
            ),
            "gabaergic_inhibitory_loss": (
                "Reduced inhibitory control and excitation / inhibition imbalance worsening cognitive regulation"
            ),
            "white_matter_disconnection": (
                "Distributed disconnection from white-matter disease and diffuse axonal injury"
            ),
            "stress_affective_load": (
                "Stress-linked affective and motivational burden interacting with cognitive decline"
            ),
            "medial_temporal_memory_system_failure": (
                "Hippocampal / medial temporal dysfunction supporting memory impairment"
            ),
            "frontosubcortical_loop_failure": (
                "Breakdown of frontal-subcortical loops supporting executive control, motivation, and psychomotor speed"
            ),
            "frontotemporal_behavioral_control_failure": (
                "Failure of frontal-temporal behavioral regulation associated with disinhibition and social change"
            ),
            "distributed_circuit_burden": (
                "Net systems-level circuit dysfunction representing the mixed-etiology final common pathway"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "episodic_memory_impairment": (
                "Memory deficits in the amnestic / medial temporal domain"
            ),
            "executive_dysfunction": (
                "Impairment in planning, set shifting, organization, and cognitive control"
            ),
            "psychomotor_slowing": (
                "Subcortical-style slowing of mental and motor processing"
            ),
            "behavioral_disinhibition": (
                "Social / behavioral dyscontrol often linked to frontal-temporal dysfunction"
            ),
            "apathy_depression": (
                "Apathy and depressive burden linked to basal ganglia, stress, and distributed circuit dysfunction"
            ),
            "global_cognitive_impairment": (
                "Final common clinical expression of broad mixed cognitive burden"
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "cerebrovascular_disease_burden",
                "target": "mixed_pathology_accumulation",
                "relation": "adds vascular lesion burden to the cumulative syndrome load",
                "ncdme_change": "increased",
            },
            {
                "source": "traumatic_brain_injury_burden",
                "target": "mixed_pathology_accumulation",
                "relation": "adds focal and diffuse injury burden to the cumulative syndrome load",
                "ncdme_change": "increased",
            },
            {
                "source": "metabolic_inflammatory_burden",
                "target": "mixed_pathology_accumulation",
                "relation": "adds systemic physiological stress to the cumulative syndrome load",
                "ncdme_change": "increased",
            },
            {
                "source": "chronic_stress_burden",
                "target": "mixed_pathology_accumulation",
                "relation": "adds chronic psychosocial stress to the cumulative disease burden",
                "ncdme_change": "increased",
            },
            {
                "source": "psychiatric_genetic_liability",
                "target": "mixed_pathology_accumulation",
                "relation": "contributes inherited vulnerability from comorbid psychiatric risk",
                "ncdme_change": "increased",
            },
            {
                "source": "substance_use_burden",
                "target": "mixed_pathology_accumulation",
                "relation": "adds addictive / neurotoxic burden that worsens later-life cognition",
                "ncdme_change": "increased",
            },
            {
                "source": "cerebrovascular_disease_burden",
                "target": "glutamatergic_excitotoxic_pressure",
                "relation": "vascular injury can drive glutamate-mediated excitotoxic damage",
                "ncdme_change": "increased",
            },
            {
                "source": "traumatic_brain_injury_burden",
                "target": "glutamatergic_excitotoxic_pressure",
                "relation": "TBI can produce excitotoxic glutamatergic injury",
                "ncdme_change": "increased",
            },
            {
                "source": "chronic_stress_burden",
                "target": "glutamatergic_excitotoxic_pressure",
                "relation": "chronic stress dysregulates glutamate and increases hippocampal vulnerability",
                "ncdme_change": "increased",
            },
            {
                "source": "psychiatric_genetic_liability",
                "target": "gabaergic_inhibitory_loss",
                "relation": "comorbid psychiatric liability can weaken inhibitory control systems",
                "ncdme_change": "increased",
            },
            {
                "source": "neurodevelopmental_liability",
                "target": "gabaergic_inhibitory_loss",
                "relation": "neurodevelopmental liability contributes to excitation / inhibition imbalance",
                "ncdme_change": "increased",
            },
            {
                "source": "cerebrovascular_disease_burden",
                "target": "white_matter_disconnection",
                "relation": "white-matter hyperintensities and vascular injury disrupt large-scale communication",
                "ncdme_change": "increased",
            },
            {
                "source": "traumatic_brain_injury_burden",
                "target": "white_matter_disconnection",
                "relation": "diffuse axonal injury disconnects distributed circuits",
                "ncdme_change": "increased",
            },
            {
                "source": "chronic_stress_burden",
                "target": "stress_affective_load",
                "relation": "sustains affective and neuroendocrine burden that worsens cognition",
                "ncdme_change": "increased",
            },
            {
                "source": "psychiatric_genetic_liability",
                "target": "stress_affective_load",
                "relation": "genetic diathesis can amplify stress-linked affective burden",
                "ncdme_change": "increased",
            },
            {
                "source": "medial_temporal_neurodegenerative_burden",
                "target": "medial_temporal_memory_system_failure",
                "relation": "loads amnestic medial temporal dysfunction",
                "ncdme_change": "increased",
            },
            {
                "source": "glutamatergic_excitotoxic_pressure",
                "target": "medial_temporal_memory_system_failure",
                "relation": "excitotoxic stress worsens hippocampal memory circuitry",
                "ncdme_change": "increased",
            },
            {
                "source": "white_matter_disconnection",
                "target": "frontosubcortical_loop_failure",
                "relation": "disconnects frontal-subcortical executive loops",
                "ncdme_change": "increased",
            },
            {
                "source": "gabaergic_inhibitory_loss",
                "target": "frontosubcortical_loop_failure",
                "relation": "weakens inhibitory control within executive circuits",
                "ncdme_change": "increased",
            },
            {
                "source": "frontotemporal_neurodegenerative_burden",
                "target": "frontotemporal_behavioral_control_failure",
                "relation": "loads frontal and temporal systems supporting behavioral regulation",
                "ncdme_change": "increased",
            },
            {
                "source": "stress_affective_load",
                "target": "frontotemporal_behavioral_control_failure",
                "relation": "affective burden worsens behavioral regulation and social control",
                "ncdme_change": "increased",
            },
            {
                "source": "medial_temporal_memory_system_failure",
                "target": "hippocampus",
                "relation": "maps medial temporal dysfunction onto hippocampal state",
                "ncdme_change": "increased",
            },
            {
                "source": "frontotemporal_behavioral_control_failure",
                "target": "ofc",
                "relation": "loads orbitofrontal behavioral control burden",
                "ncdme_change": "increased",
            },
            {
                "source": "frontosubcortical_loop_failure",
                "target": "pfc_control",
                "relation": "loads frontal executive-control circuitry",
                "ncdme_change": "increased",
            },
            {
                "source": "frontotemporal_behavioral_control_failure",
                "target": "temporal_association_proxy",
                "relation": "loads temporal systems involved in frontotemporal presentations",
                "ncdme_change": "increased",
            },
            {
                "source": "frontosubcortical_loop_failure",
                "target": "basal_ganglia_proxy",
                "relation": "loads basal ganglia hubs within executive and motivational loops",
                "ncdme_change": "increased",
            },
            {
                "source": "white_matter_disconnection",
                "target": "cerebellum_proxy",
                "relation": "propagates distributed disconnection into cerebellar-cognitive systems",
                "ncdme_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "episodic_memory_impairment",
                "relation": "hippocampal dysfunction drives memory deficits",
                "ncdme_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "executive_dysfunction",
                "relation": "prefrontal control failure impairs executive function",
                "ncdme_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "psychomotor_slowing",
                "relation": "basal ganglia dysfunction contributes to psychomotor slowing",
                "ncdme_change": "increased",
            },
            {
                "source": "ofc",
                "target": "behavioral_disinhibition",
                "relation": "orbitofrontal dysfunction contributes to disinhibited behavior",
                "ncdme_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "apathy_depression",
                "relation": "basal ganglia and frontal-subcortical dysfunction contributes to apathy and depression",
                "ncdme_change": "increased",
            },
            {
                "source": "distributed_circuit_burden",
                "target": "global_cognitive_impairment",
                "relation": "mixed systems burden converges on broad cognitive impairment",
                "ncdme_change": "increased",
            },
            {
                "source": "etiology_treatment_support",
                "target": "distributed_circuit_burden",
                "relation": "reduces overall burden by correcting or buffering contributing etiologies",
                "ncdme_change": "decreased",
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

    @staticmethod
    def _identifier_of(obj: Any) -> Optional[str]:
        return getattr(obj, "identifier", getattr(obj, "id", None))

    def _modality_candidates(self, kind: str) -> List[Any]:
        if siibra is None:
            return []

        cands: List[Any] = []
        try:
            if kind == "receptor":
                if hasattr(siibra.features, "molecular") and hasattr(
                    siibra.features.molecular, "ReceptorDensityFingerprint"
                ):
                    cands.append(siibra.features.molecular.ReceptorDensityFingerprint)
                if hasattr(siibra.features, "tabular") and hasattr(
                    siibra.features.tabular, "ReceptorDensityFingerprint"
                ):
                    cands.append(siibra.features.tabular.ReceptorDensityFingerprint)
            elif kind == "gene":
                if hasattr(siibra.features, "molecular") and hasattr(
                    siibra.features.molecular, "GeneExpressions"
                ):
                    cands.append(siibra.features.molecular.GeneExpressions)
                if hasattr(siibra.features, "tabular") and hasattr(
                    siibra.features.tabular, "GeneExpressions"
                ):
                    cands.append(siibra.features.tabular.GeneExpressions)
            elif kind == "connectivity":
                if hasattr(siibra.features, "connectivity") and hasattr(
                    siibra.features.connectivity, "StreamlineCounts"
                ):
                    cands.append(siibra.features.connectivity.StreamlineCounts)
        except Exception:
            pass

        if kind == "receptor":
            cands.extend(["receptor density fingerprint", "ReceptorDensityFingerprint"])
        elif kind == "gene":
            cands.extend(["gene expressions", "GeneExpressions"])
        elif kind == "connectivity":
            cands.extend(["StreamlineCounts"])
        return cands

    def _safe_features_any(self, concept: Any, modalities: Sequence[Any], **kwargs: Any) -> List[Any]:
        if siibra is None or concept is None:
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
        if self.atlas is None:
            return []
        matches: List[Any] = []

        try:
            matches = list(
                self.atlas.find_regions(
                    query,
                    all_versions=False,
                    filter_children=False,
                    find_topmost=False,
                )
            )
        except Exception:
            matches = []

        if not matches and self.parcellation is not None:
            try:
                matches = list(self.parcellation.find(query, filter_children=False, find_topmost=False))
            except Exception:
                matches = []

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
        generic_penalty = 1 if name in {
            "hippocampus",
            "orbitofrontal cortex",
            "prefrontal cortex",
            "temporal cortex",
            "basal ganglia",
            "cerebellum",
        } else 0
        specificity_penalty = 1 if "area " not in name and "(" not in name and "ca" not in name else 0
        return (left_bonus, right_penalty, generic_penalty, specificity_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if self.atlas is None and self.parcellation is None:
            return None

        for spec in candidates:
            if self.atlas is not None:
                try:
                    return self.atlas.get_region(spec, parcellation=self.parcellation)
                except Exception:
                    pass
            if self.parcellation is not None:
                try:
                    return self.parcellation.get_region(spec)
                except Exception:
                    pass
            matches = self._julich_matches(spec)
            if matches:
                return sorted(matches, key=self._region_rank)[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows = []
        seen = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
            row = (
                self._name_of(region),
                self._identifier_of(region),
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
        if hasattr(props, "components"):
            return list(getattr(props, "components", []))
        if isinstance(props, dict):
            return list(props.values())
        if isinstance(props, (list, tuple)):
            return list(props)
        return [props]

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz: Optional[Tuple[float, float, float]] = None
        if centroid is not None:
            try:
                centroid_xyz = tuple(float(x) for x in tuple(centroid))  # type: ignore[arg-type]
            except Exception:
                try:
                    centroid_xyz = tuple(float(x) for x in centroid.coordinate)  # type: ignore[attr-defined]
                except Exception:
                    centroid_xyz = None
        volume_mm3 = getattr(main, "volume", None)
        return centroid_xyz, (float(volume_mm3) if volume_mm3 is not None else None)

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

        for feat in feats:
            try:
                df = feat.data.copy().reset_index(drop=True)
            except Exception:
                continue
            lower_cols = {str(c).lower(): c for c in df.columns}
            if {"gene", "level", "zscore"}.issubset(lower_cols):
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
            return df
        return pd.DataFrame()

    def _select_connectivity_compound(self) -> Optional[Any]:
        if self.parcellation is None:
            return None
        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            return None

        preferred = [f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort]
        return preferred[0] if preferred else feats[0]

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        compound = self._select_connectivity_compound()
        if compound is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            if len(compound) > 0:
                self._connectivity_matrix = compound[0].data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        if region is None:
            return None
        region_names = {
            self._name_of(region).lower(),
            str(getattr(region, "name", "")).lower(),
        }

        exact = [x for x in labels if self._name_of(x).lower() in region_names]
        if exact:
            return exact[0]

        rn = self._name_of(region).lower()
        fuzzy = [
            x
            for x in labels
            if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn
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
            df = df[df["connected_region"] != self._name_of(region)].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self, node_keys: Optional[Sequence[str]] = None) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame(columns=["source", "target", "value"])

        keys = list(node_keys) if node_keys is not None else list(self.region_objects.keys())
        rows: List[Dict[str, Any]] = []

        for src_key in keys:
            src_region = self.region_objects.get(src_key)
            if src_region is None:
                continue
            src_label = self._match_region_label(list(matrix.index), src_region)
            if src_label is None:
                src_label = self._match_region_label(list(matrix.columns), src_region)
            if src_label is None:
                continue

            for dst_key in keys:
                if src_key == dst_key:
                    continue
                dst_region = self.region_objects.get(dst_key)
                if dst_region is None:
                    continue
                dst_label = self._match_region_label(list(matrix.columns), dst_region)
                if dst_label is None:
                    dst_label = self._match_region_label(list(matrix.index), dst_region)
                if dst_label is None:
                    continue

                value = None
                try:
                    value = matrix.loc[src_label, dst_label]
                except Exception:
                    try:
                        value = matrix.loc[dst_label, src_label]
                    except Exception:
                        value = None
                if value is None:
                    continue
                try:
                    rows.append(
                        {
                            "source": src_key,
                            "target": dst_key,
                            "value": float(value),
                        }
                    )
                except Exception:
                    continue

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(["source", "value"], ascending=[True, False]).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
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
                warnings.warn(
                    f"Could not resolve a region for node '{key}'. This is acceptable for a proxy scaffold."
                )
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " "),
                        "node_type": "region_proxy",
                        "description": "Proxy circuit node unresolved in this siibra environment",
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
                    "label": self._name_of(region),
                    "node_type": "region_proxy",
                    "description": "Atlas-backed proxy node for a chapter-grounded circuit",
                    "atlas_region": self._name_of(region),
                    "region_identifier": self._identifier_of(region),
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
        cerebrovascular_disease_burden: float = 0.55,
        traumatic_brain_injury_burden: float = 0.35,
        metabolic_inflammatory_burden: float = 0.45,
        chronic_stress_burden: float = 0.45,
        psychiatric_genetic_liability: float = 0.50,
        neurodevelopmental_liability: float = 0.30,
        substance_use_burden: float = 0.25,
        medial_temporal_neurodegenerative_burden: float = 0.40,
        frontotemporal_neurodegenerative_burden: float = 0.30,
        cognitive_reserve: float = 0.45,
        etiology_treatment_support: float = 0.40,
    ) -> Dict[str, pd.Series]:
        """
        Transparent one-pass normalized simulator.

        The simulation reflects chapter directionality:
        inputs -> latent mixed biology -> regional state -> symptoms -> phenotype summaries

        All values are clipped to [0, 1].
        """

        inputs = pd.Series(
            {
                "cerebrovascular_disease_burden": self._clip01(cerebrovascular_disease_burden),
                "traumatic_brain_injury_burden": self._clip01(traumatic_brain_injury_burden),
                "metabolic_inflammatory_burden": self._clip01(metabolic_inflammatory_burden),
                "chronic_stress_burden": self._clip01(chronic_stress_burden),
                "psychiatric_genetic_liability": self._clip01(psychiatric_genetic_liability),
                "neurodevelopmental_liability": self._clip01(neurodevelopmental_liability),
                "substance_use_burden": self._clip01(substance_use_burden),
                "medial_temporal_neurodegenerative_burden": self._clip01(medial_temporal_neurodegenerative_burden),
                "frontotemporal_neurodegenerative_burden": self._clip01(frontotemporal_neurodegenerative_burden),
                "cognitive_reserve": self._clip01(cognitive_reserve),
                "etiology_treatment_support": self._clip01(etiology_treatment_support),
            },
            name="value",
        )

        latents = pd.Series(dtype=float, name="value")
        latents["mixed_pathology_accumulation"] = self._clip01(
            0.16 * inputs["cerebrovascular_disease_burden"]
            + 0.14 * inputs["traumatic_brain_injury_burden"]
            + 0.12 * inputs["metabolic_inflammatory_burden"]
            + 0.10 * inputs["chronic_stress_burden"]
            + 0.10 * inputs["psychiatric_genetic_liability"]
            + 0.08 * inputs["neurodevelopmental_liability"]
            + 0.08 * inputs["substance_use_burden"]
            + 0.12 * inputs["medial_temporal_neurodegenerative_burden"]
            + 0.10 * inputs["frontotemporal_neurodegenerative_burden"]
            - 0.10 * inputs["etiology_treatment_support"]
        )
        latents["glutamatergic_excitotoxic_pressure"] = self._clip01(
            0.28 * inputs["cerebrovascular_disease_burden"]
            + 0.25 * inputs["traumatic_brain_injury_burden"]
            + 0.18 * inputs["metabolic_inflammatory_burden"]
            + 0.16 * inputs["chronic_stress_burden"]
            + 0.08 * inputs["substance_use_burden"]
            - 0.12 * inputs["etiology_treatment_support"]
        )
        latents["gabaergic_inhibitory_loss"] = self._clip01(
            0.26 * inputs["psychiatric_genetic_liability"]
            + 0.20 * inputs["neurodevelopmental_liability"]
            + 0.16 * inputs["substance_use_burden"]
            + 0.14 * inputs["chronic_stress_burden"]
            + 0.12 * inputs["metabolic_inflammatory_burden"]
            + 0.08 * inputs["traumatic_brain_injury_burden"]
            - 0.12 * inputs["cognitive_reserve"]
        )
        latents["white_matter_disconnection"] = self._clip01(
            0.36 * inputs["cerebrovascular_disease_burden"]
            + 0.30 * inputs["traumatic_brain_injury_burden"]
            + 0.16 * inputs["metabolic_inflammatory_burden"]
            + 0.08 * inputs["substance_use_burden"]
            - 0.14 * inputs["etiology_treatment_support"]
        )
        latents["stress_affective_load"] = self._clip01(
            0.36 * inputs["chronic_stress_burden"]
            + 0.24 * inputs["psychiatric_genetic_liability"]
            + 0.14 * inputs["substance_use_burden"]
            + 0.10 * inputs["metabolic_inflammatory_burden"]
            + 0.06 * inputs["frontotemporal_neurodegenerative_burden"]
            - 0.18 * inputs["etiology_treatment_support"]
        )
        latents["medial_temporal_memory_system_failure"] = self._clip01(
            0.42 * inputs["medial_temporal_neurodegenerative_burden"]
            + 0.22 * latents["glutamatergic_excitotoxic_pressure"]
            + 0.14 * inputs["chronic_stress_burden"]
            + 0.10 * inputs["metabolic_inflammatory_burden"]
            + 0.08 * inputs["traumatic_brain_injury_burden"]
            - 0.18 * inputs["cognitive_reserve"]
        )
        latents["frontosubcortical_loop_failure"] = self._clip01(
            0.32 * latents["white_matter_disconnection"]
            + 0.22 * latents["gabaergic_inhibitory_loss"]
            + 0.16 * inputs["cerebrovascular_disease_burden"]
            + 0.10 * inputs["traumatic_brain_injury_burden"]
            + 0.10 * latents["mixed_pathology_accumulation"]
            - 0.18 * inputs["cognitive_reserve"]
        )
        latents["frontotemporal_behavioral_control_failure"] = self._clip01(
            0.38 * inputs["frontotemporal_neurodegenerative_burden"]
            + 0.18 * latents["stress_affective_load"]
            + 0.16 * latents["gabaergic_inhibitory_loss"]
            + 0.12 * latents["glutamatergic_excitotoxic_pressure"]
            + 0.08 * inputs["substance_use_burden"]
            - 0.16 * inputs["etiology_treatment_support"]
        )
        latents["distributed_circuit_burden"] = self._clip01(
            0.18 * latents["mixed_pathology_accumulation"]
            + 0.18 * latents["white_matter_disconnection"]
            + 0.18 * latents["frontosubcortical_loop_failure"]
            + 0.16 * latents["medial_temporal_memory_system_failure"]
            + 0.16 * latents["frontotemporal_behavioral_control_failure"]
            + 0.14 * latents["stress_affective_load"]
            - 0.10 * inputs["etiology_treatment_support"]
        )

        regional_state = pd.Series(dtype=float, name="value")
        regional_state["hippocampus"] = self._clip01(
            0.48 * latents["medial_temporal_memory_system_failure"]
            + 0.22 * latents["glutamatergic_excitotoxic_pressure"]
            + 0.18 * inputs["chronic_stress_burden"]
            + 0.08 * inputs["metabolic_inflammatory_burden"]
            - 0.18 * inputs["cognitive_reserve"]
        )
        regional_state["ofc"] = self._clip01(
            0.42 * latents["frontotemporal_behavioral_control_failure"]
            + 0.20 * latents["stress_affective_load"]
            + 0.14 * inputs["substance_use_burden"]
            + 0.10 * inputs["traumatic_brain_injury_burden"]
            + 0.08 * latents["mixed_pathology_accumulation"]
            - 0.14 * inputs["etiology_treatment_support"]
        )
        regional_state["pfc_control"] = self._clip01(
            0.38 * latents["frontosubcortical_loop_failure"]
            + 0.24 * latents["white_matter_disconnection"]
            + 0.18 * latents["gabaergic_inhibitory_loss"]
            + 0.10 * inputs["chronic_stress_burden"]
            - 0.18 * inputs["cognitive_reserve"]
        )
        regional_state["temporal_association_proxy"] = self._clip01(
            0.40 * latents["frontotemporal_behavioral_control_failure"]
            + 0.22 * inputs["frontotemporal_neurodegenerative_burden"]
            + 0.16 * latents["medial_temporal_memory_system_failure"]
            + 0.10 * inputs["traumatic_brain_injury_burden"]
            + 0.08 * latents["mixed_pathology_accumulation"]
        )
        regional_state["basal_ganglia_proxy"] = self._clip01(
            0.34 * latents["frontosubcortical_loop_failure"]
            + 0.24 * inputs["cerebrovascular_disease_burden"]
            + 0.14 * inputs["substance_use_burden"]
            + 0.12 * inputs["psychiatric_genetic_liability"]
            + 0.10 * inputs["traumatic_brain_injury_burden"]
            - 0.12 * inputs["etiology_treatment_support"]
        )
        regional_state["cerebellum_proxy"] = self._clip01(
            0.28 * latents["white_matter_disconnection"]
            + 0.22 * latents["frontosubcortical_loop_failure"]
            + 0.16 * inputs["cerebrovascular_disease_burden"]
            + 0.16 * inputs["traumatic_brain_injury_burden"]
            + 0.10 * inputs["metabolic_inflammatory_burden"]
            - 0.10 * inputs["cognitive_reserve"]
        )

        symptoms = pd.Series(dtype=float, name="value")
        symptoms["episodic_memory_impairment"] = self._clip01(
            0.56 * regional_state["hippocampus"]
            + 0.20 * latents["medial_temporal_memory_system_failure"]
            + 0.12 * regional_state["temporal_association_proxy"]
            + 0.08 * latents["distributed_circuit_burden"]
            - 0.12 * inputs["cognitive_reserve"]
        )
        symptoms["executive_dysfunction"] = self._clip01(
            0.42 * regional_state["pfc_control"]
            + 0.22 * regional_state["basal_ganglia_proxy"]
            + 0.14 * regional_state["cerebellum_proxy"]
            + 0.14 * latents["white_matter_disconnection"]
            + 0.06 * latents["distributed_circuit_burden"]
            - 0.12 * inputs["cognitive_reserve"]
        )
        symptoms["psychomotor_slowing"] = self._clip01(
            0.36 * regional_state["basal_ganglia_proxy"]
            + 0.24 * regional_state["cerebellum_proxy"]
            + 0.20 * latents["white_matter_disconnection"]
            + 0.10 * inputs["metabolic_inflammatory_burden"]
            + 0.06 * latents["distributed_circuit_burden"]
        )
        symptoms["behavioral_disinhibition"] = self._clip01(
            0.48 * regional_state["ofc"]
            + 0.20 * regional_state["temporal_association_proxy"]
            + 0.16 * latents["frontotemporal_behavioral_control_failure"]
            + 0.10 * inputs["substance_use_burden"]
            - 0.12 * inputs["etiology_treatment_support"]
        )
        symptoms["apathy_depression"] = self._clip01(
            0.28 * regional_state["basal_ganglia_proxy"]
            + 0.26 * latents["stress_affective_load"]
            + 0.16 * regional_state["cerebellum_proxy"]
            + 0.14 * latents["frontosubcortical_loop_failure"]
            + 0.10 * inputs["psychiatric_genetic_liability"]
            - 0.12 * inputs["etiology_treatment_support"]
        )
        symptoms["global_cognitive_impairment"] = self._clip01(
            0.26 * symptoms["episodic_memory_impairment"]
            + 0.26 * symptoms["executive_dysfunction"]
            + 0.16 * symptoms["psychomotor_slowing"]
            + 0.12 * symptoms["behavioral_disinhibition"]
            + 0.10 * symptoms["apathy_depression"]
            + 0.14 * latents["distributed_circuit_burden"]
            - 0.08 * inputs["cognitive_reserve"]
        )

        phenotypes = pd.Series(dtype=float, name="value")
        phenotypes["amnestic_mixed_profile"] = self._clip01(
            (
                symptoms["episodic_memory_impairment"]
                + symptoms["global_cognitive_impairment"]
                + regional_state["hippocampus"]
            )
            / 3.0
        )
        phenotypes["subcortical_vascular_profile"] = self._clip01(
            (
                symptoms["executive_dysfunction"]
                + symptoms["psychomotor_slowing"]
                + regional_state["basal_ganglia_proxy"]
                + latents["white_matter_disconnection"]
            )
            / 4.0
        )
        phenotypes["frontotemporal_behavioral_profile"] = self._clip01(
            (
                symptoms["behavioral_disinhibition"]
                + symptoms["apathy_depression"]
                + regional_state["ofc"]
                + regional_state["temporal_association_proxy"]
            )
            / 4.0
        )
        phenotypes["stress_complicated_mixed_ncd_profile"] = self._clip01(
            (
                symptoms["global_cognitive_impairment"]
                + symptoms["apathy_depression"]
                + latents["stress_affective_load"]
                + latents["mixed_pathology_accumulation"]
            )
            / 4.0
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if siibra is None or self.parcellation is None:
            warnings.warn("siibra/parcellation unavailable; returning empty assignment table.")
            return pd.DataFrame()

        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception as exc:
                warnings.warn(f"Could not load probabilistic map: {exc}")
                return pd.DataFrame()

        try:
            point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
            with siibra.QUIET:
                assignments = self._pmap.assign(point)
        except Exception as exc:
            warnings.warn(f"Coordinate assignment failed: {exc}")
            return pd.DataFrame()

        if not isinstance(assignments, pd.DataFrame):
            try:
                assignments = pd.DataFrame(assignments)
            except Exception:
                return pd.DataFrame()

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                return assignments.sort_values(candidate, ascending=False).reset_index(drop=True)
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Any:
        region = self.region_objects.get(node_key)
        if region is None or self.space is None:
            return None

        for method_name in ("get_regional_mask", "fetch_regional_map"):
            method = getattr(region, method_name, None)
            if callable(method):
                try:
                    return method(self.space, maptype="labelled")
                except TypeError:
                    try:
                        return method(space=self.space, maptype="labelled")
                    except Exception:
                        continue
                except Exception:
                    continue
        return None


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    model = MultipleEtiologiesNeurocognitiveDisorderModel()
    bundle = model.build()

    print("\n=== Nodes ===")
    print(bundle["nodes"].head(30).to_string(index=False))

    print("\n=== Edges ===")
    print(bundle["edges"].to_string(index=False))

    print("\n=== Resolved regions ===")
    if bundle["regions"]:
        for key, region in bundle["regions"].items():
            print(f"- {key}: {getattr(region, 'name', region)}")
    else:
        print("No atlas regions resolved in this environment.")

    print("\n=== Feature availability ===")
    for key in model.region_candidates:
        print(
            f"- {key}: receptors={not model.receptors.get(key, pd.DataFrame()).empty}, "
            f"genes={not model.genes.get(key, pd.DataFrame()).empty}, "
            f"connectivity={not model.connectivity_profiles.get(key, pd.DataFrame()).empty}"
        )

    print("\n=== Example connectivity among resolved circuit nodes ===")
    circuit_df = bundle["circuit_connectivity"]
    if isinstance(circuit_df, pd.DataFrame) and not circuit_df.empty:
        print(circuit_df.head(20).to_string(index=False))
    else:
        print("No circuit connectivity matrix available in this environment.")

    for key in ("hippocampus", "ofc", "pfc_control"):
        receptor_df = model.receptors.get(key, pd.DataFrame())
        gene_df = model.genes.get(key, pd.DataFrame())
        conn_df = model.connectivity_profiles.get(key, pd.DataFrame())

        if not receptor_df.empty:
            print(f"\n=== Receptor fingerprint sample: {key} ===")
            print(receptor_df.head(10).to_string(index=False))
            break

        if not gene_df.empty:
            print(f"\n=== Gene-expression sample: {key} ===")
            print(gene_df.head(10).to_string(index=False))
            break

        if not conn_df.empty:
            print(f"\n=== Connectivity profile sample: {key} ===")
            print(conn_df.head(10).to_string(index=False))
            break

    print("\n=== Example simulation ===")
    sim = model.simulate(
        cerebrovascular_disease_burden=0.70,
        traumatic_brain_injury_burden=0.35,
        metabolic_inflammatory_burden=0.55,
        chronic_stress_burden=0.50,
        psychiatric_genetic_liability=0.55,
        neurodevelopmental_liability=0.30,
        substance_use_burden=0.25,
        medial_temporal_neurodegenerative_burden=0.50,
        frontotemporal_neurodegenerative_burden=0.35,
        cognitive_reserve=0.45,
        etiology_treatment_support=0.40,
    )
    for name, series in sim.items():
        print(f"\n{name.upper()}")
        print(series.sort_values(ascending=False).to_string())

    # Optional coordinate test in a siibra-enabled environment:
    # print(model.assign_mni_point((-24, -18, -18)).head(10))
