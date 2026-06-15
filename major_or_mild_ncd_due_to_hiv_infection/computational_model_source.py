from __future__ import annotations

"""
Major or Mild Neurocognitive Disorder Due to HIV Infection atlas-grounded siibra scaffold.

This script translates a chapter-level discussion of HIV-associated neurocognitive
injury into a transparent, research-oriented mechanistic graph. The chapter title
maps onto the DSM framing of Major or Mild Neurocognitive Disorder Due to HIV
Infection, while the biological text emphasizes the broader HIV-associated
neurocognitive disorders (HAND) continuum spanning asymptomatic neurocognitive
impairment (ANI), mild neurocognitive disorder (MND), and HIV-associated dementia
(HAD).

The scaffold centers on the chapter's main biological themes:
- early CNS invasion and persistence of HIV,
- host chemokine-related susceptibility affecting monocyte trafficking,
- neuroinflammation and neurotoxic viral/inflammatory mediators,
- synaptic and neuronal dysfunction rather than direct neuronal infection,
- preferential burden on deep gray matter and cerebral white matter,
- cerebellar and brainstem pathway involvement in some cases,
- downstream psychomotor, attentional/executive, motor, and behavioral changes.

Important:
- This is a research scaffold, not a diagnostic or treatment tool.
- The simulator is a transparent heuristic model, not a validated disease model.
- Higher simulated values indicate greater dysregulation, injury burden, or
  symptom pressure unless explicitly noted otherwise.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore[assignment]
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_HIV_NCD_GENE_PANEL = [
    "CCL2",    # MCP-1; chapter-linked chemokine susceptibility marker
    "CCR2",    # chapter-linked receptor polymorphism
    "CCR5",    # HIV / chemokine-system relevance
    "IL6",     # inflammatory signaling
    "TNF",     # inflammatory mediator / neurotoxicity context
    "CXCL10",  # neuroinflammatory chemokine context
    "GFAP",    # astroglial reactivity context
    "MBP",     # white-matter / myelin context
    "SYN1",    # synaptic integrity context
    "BDNF",    # neuroplasticity / neuronal resilience
    "SLC6A3",  # subcortical dopaminergic system relevance
    "GRIN1",   # excitatory synaptic function context
]


class MajorOrMildNeurocognitiveDisorderDueToHivInfectionModel:
    """
    Atlas-grounded research scaffold for HIV-associated neurocognitive disorder.

    The chapter emphasizes a continuum of HIV-related neurocognitive injury built
    on early CNS invasion, chronic inflammatory and neurotoxic cascades, and a
    subcortical pattern of network dysfunction with possible cerebellar/brainstem
    extension. Because the chapter is systems-level rather than parcel-specific,
    this scaffold combines atlas-backed regions with explicit proxy nodes.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        if siibra is None:
            raise ImportError(
                "siibra is required to use this scaffold. Install it in your Python "
                "environment before running the model."
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

        # Conservative atlas anchors reflecting the chapter's topography. Deep
        # gray matter is represented by basal ganglia and thalamus nodes. White
        # matter, cerebellar, and brainstem involvement are kept as explicit
        # proxies when exact Julich labels vary or may be unavailable.
        self.region_candidates: Dict[str, List[str]] = {
            "basal_ganglia_proxy": [
                "caudate left",
                "putamen left",
                "striatum",
                "basal ganglia",
            ],
            "thalamus_proxy": [
                "thalamus left",
                "thalamus",
            ],
            "cerebral_white_matter_proxy": [
                "white matter",
                "internal capsule left",
                "corona radiata left",
                "corpus callosum",
            ],
            "cerebellum_proxy": [
                "cerebellum left",
                "cerebellar hemisphere left",
                "cerebellum",
                "vermis",
            ],
            "brainstem_oculomotor_proxy": [
                "brainstem",
                "midbrain",
                "pons",
                "superior colliculus left",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "basal_ganglia_proxy": (
                "Proxy for deep gray matter fronto-striatal circuitry emphasized in the chapter's subcortical "
                "dementia model."
            ),
            "thalamus_proxy": (
                "Proxy for thalamic relay circuitry contributing to attentional and executive slowing in the "
                "subcortical pattern."
            ),
            "cerebral_white_matter_proxy": (
                "Proxy for cerebral white-matter tracts whose disruption is central to the chapter's distributed "
                "network injury model."
            ),
            "cerebellum_proxy": (
                "Proxy for cerebellar circuitry involved in ataxia, limb clumsiness, and broader cerebro-cerebellar "
                "disruption."
            ),
            "brainstem_oculomotor_proxy": (
                "Proxy for brainstem oculomotor pathways implicated by fixation, saccade, and smooth-pursuit "
                "abnormalities."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "acute_persistent_cns_hiv_burden": (
                "Early neuroinvasion and persistent HIV burden within the CNS compartment across the disease course."
            ),
            "chemokine_monocyte_infiltration_liability": (
                "Host susceptibility linked to chemokine-related variation influencing monocyte trafficking into the CNS."
            ),
            "chronic_immune_activation": (
                "Ongoing immune activation that sustains inflammatory pressure within the nervous system."
            ),
            "disease_severity_load": (
                "Overall illness burden associated with greater cumulative neurological vulnerability."
            ),
            "antiretroviral_cns_control_support": (
                "Protective reduction in CNS viral and inflammatory burden from effective control of HIV."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "monocyte_cns_trafficking": (
                "Chemokine-biased entry and persistence of monocyte/macrophage lineage cells that seed CNS inflammation."
            ),
            "chronic_neuroinflammation": (
                "Sustained neuroinflammatory state driven by infected or activated immune cells and inflammatory mediators."
            ),
            "viral_protein_neurotoxicity": (
                "Indirect neuronal injury from viral proteins and inflammatory mediators rather than direct neuronal infection."
            ),
            "synaptic_neuronal_dysfunction": (
                "Widespread synaptic and neuronal dysfunction underlying cognitive, motor, and behavioral abnormalities."
            ),
            "subcortical_transmitter_imbalance": (
                "Heuristic neurotransmitter-system perturbation affecting psychomotor speed, motivation, and executive control."
            ),
            "subcortical_white_matter_disconnection": (
                "Disruption of white-matter tracts and distributed subcortical network communication."
            ),
            "deep_gray_matter_network_failure": (
                "Failure of basal ganglia-thalamic subcortical processing as injury accumulates in deep gray matter."
            ),
            "cerebellar_brainstem_pathway_disruption": (
                "Extension of pathology into cerebellar and brainstem pathways affecting coordination and oculomotor control."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "psychomotor_slowing": (
                "Subcortical psychomotor slowing characteristic of HIV-associated neurocognitive impairment."
            ),
            "attentional_executive_dysfunction": (
                "Attention and executive-control deficits that dominate the classic subcortical profile."
            ),
            "behavioral_motivation_change": (
                "Behavioral and motivational change within the cognitive-motor-behavioral triad of severe HAND."
            ),
            "motor_incoordination_ataxia": (
                "Ataxia, clumsiness, or broader motor coordination disturbance linked to cerebellar/circuit involvement."
            ),
            "oculomotor_abnormalities": (
                "Abnormal fixation, saccades, or smooth pursuit suggesting brainstem-cerebellar pathway dysfunction."
            ),
            "daily_function_interference": (
                "Real-world interference in functioning across the HAND continuum from mild to severe states."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "acute_persistent_cns_hiv_burden",
                "target": "monocyte_cns_trafficking",
                "relation": "establishes and maintains CNS infection conditions that permit inflammatory cell entry and persistence",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "acute_persistent_cns_hiv_burden",
                "target": "viral_protein_neurotoxicity",
                "relation": "raises the burden of viral products contributing to indirect neural injury",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "chemokine_monocyte_infiltration_liability",
                "target": "monocyte_cns_trafficking",
                "relation": "biases monocyte recruitment into the CNS through chemokine-system susceptibility",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "chronic_immune_activation",
                "target": "chronic_neuroinflammation",
                "relation": "sustains inflammatory signaling within the CNS",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "disease_severity_load",
                "target": "viral_protein_neurotoxicity",
                "relation": "raises cumulative neurotoxic burden as illness severity advances",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "disease_severity_load",
                "target": "subcortical_white_matter_disconnection",
                "relation": "increases vulnerability of distributed deep-gray and white-matter systems",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "antiretroviral_cns_control_support",
                "target": "monocyte_cns_trafficking",
                "relation": "reduces the effective inflammatory seeding burden in the CNS",
                "hiv_ncd_change": "decreased",
            },
            {
                "source": "antiretroviral_cns_control_support",
                "target": "chronic_neuroinflammation",
                "relation": "buffers neuroinflammatory drive by reducing ongoing viral activity",
                "hiv_ncd_change": "decreased",
            },
            {
                "source": "antiretroviral_cns_control_support",
                "target": "viral_protein_neurotoxicity",
                "relation": "lowers the burden of indirect viral neurotoxicity",
                "hiv_ncd_change": "decreased",
            },
            {
                "source": "monocyte_cns_trafficking",
                "target": "chronic_neuroinflammation",
                "relation": "initiates and sustains inflammatory cascades in the CNS",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "chronic_neuroinflammation",
                "target": "synaptic_neuronal_dysfunction",
                "relation": "drives widespread synaptic and neuronal dysfunction",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "viral_protein_neurotoxicity",
                "target": "synaptic_neuronal_dysfunction",
                "relation": "adds direct toxic stress on neural signaling and viability",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "chronic_neuroinflammation",
                "target": "subcortical_transmitter_imbalance",
                "relation": "perturbs subcortical transmitter systems relevant to cognition and motor control",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "synaptic_neuronal_dysfunction",
                "target": "subcortical_transmitter_imbalance",
                "relation": "destabilizes neurotransmitter-dependent subcortical processing",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "synaptic_neuronal_dysfunction",
                "target": "subcortical_white_matter_disconnection",
                "relation": "weakens communication across distributed subcortical networks",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "subcortical_white_matter_disconnection",
                "target": "deep_gray_matter_network_failure",
                "relation": "disconnects basal ganglia-thalamic processing loops",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "subcortical_transmitter_imbalance",
                "target": "deep_gray_matter_network_failure",
                "relation": "worsens processing efficiency in deep gray matter circuits",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "chronic_neuroinflammation",
                "target": "cerebellar_brainstem_pathway_disruption",
                "relation": "extends pathological burden into cerebellar and brainstem pathways",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "subcortical_white_matter_disconnection",
                "target": "cerebellar_brainstem_pathway_disruption",
                "relation": "interrupts cerebro-cerebellar and brainstem-related communication pathways",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "deep_gray_matter_network_failure",
                "target": "basal_ganglia_proxy",
                "relation": "increases dysfunction burden in basal ganglia circuitry",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "deep_gray_matter_network_failure",
                "target": "thalamus_proxy",
                "relation": "increases dysfunction burden in thalamic relay circuitry",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "subcortical_white_matter_disconnection",
                "target": "cerebral_white_matter_proxy",
                "relation": "increases injury burden in distributed white-matter pathways",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "cerebellar_brainstem_pathway_disruption",
                "target": "cerebellum_proxy",
                "relation": "raises dysfunction burden in cerebellar coordination circuitry",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "cerebellar_brainstem_pathway_disruption",
                "target": "brainstem_oculomotor_proxy",
                "relation": "raises dysfunction burden in brainstem oculomotor pathways",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "psychomotor_slowing",
                "relation": "contributes to psychomotor slowing in the subcortical dementia pattern",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "cerebral_white_matter_proxy",
                "target": "psychomotor_slowing",
                "relation": "slows distributed information transfer and motor-cognitive throughput",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "attentional_executive_dysfunction",
                "relation": "weakens relay and attentional control functions",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "cerebral_white_matter_proxy",
                "target": "attentional_executive_dysfunction",
                "relation": "disconnects distributed executive and attentional networks",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "behavioral_motivation_change",
                "relation": "contributes to behavioral and motivational slowing or alteration",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "cerebellum_proxy",
                "target": "motor_incoordination_ataxia",
                "relation": "drives ataxia and coordination deficits when cerebellar pathways are affected",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "brainstem_oculomotor_proxy",
                "target": "oculomotor_abnormalities",
                "relation": "contributes to abnormal fixation, saccades, and smooth pursuit",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "psychomotor_slowing",
                "target": "daily_function_interference",
                "relation": "increases day-to-day burden through slowed cognition and action",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "attentional_executive_dysfunction",
                "target": "daily_function_interference",
                "relation": "reduces efficiency in everyday functioning",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "motor_incoordination_ataxia",
                "target": "daily_function_interference",
                "relation": "increases functional burden through motor impairment",
                "hiv_ncd_change": "increased",
            },
            {
                "source": "behavioral_motivation_change",
                "target": "daily_function_interference",
                "relation": "adds behavioral and motivational impairment to the syndrome burden",
                "hiv_ncd_change": "increased",
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "thalamus",
            "cerebellum",
            "brainstem",
            "striatum",
            "basal ganglia",
            "white matter",
        } else 0
        proxy_penalty = 1 if "proxy" in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                region = self.atlas.get_region(spec, parcellation=self.parcellation)
                if region is not None:
                    return region
            except Exception:
                pass
            try:
                region = self.parcellation.get_region(spec)
                if region is not None:
                    return region
            except Exception:
                pass
            matches = self._julich_matches(spec)
            if matches:
                return sorted(matches, key=self._region_rank)[0]
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
        centroid_xyz = tuple(float(v) for v in centroid) if centroid is not None else None
        volume = getattr(main, "volume", None)
        volume_mm3 = float(volume) if volume is not None else None
        return centroid_xyz, volume_mm3

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()

        for feat in feats:
            try:
                df = feat.data.copy().reset_index()
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

        return pd.DataFrame()

    def _deduplicate_connectivity_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize connectivity labels to names and average duplicate row/column
        labels. Some siibra connectivity matrices can expose repeated labels,
        which makes label-based selection return DataFrames instead of scalars.
        """
        if matrix.empty:
            return matrix

        out = matrix.copy()
        out.index = [self._name_of(x) for x in out.index]
        out.columns = [self._name_of(x) for x in out.columns]
        out = out.apply(pd.to_numeric, errors="coerce")

        if not out.index.is_unique:
            out = out.groupby(level=0, sort=False).mean()
        if not out.columns.is_unique:
            out = out.T.groupby(level=0, sort=False).mean().T
        return out

    @staticmethod
    def _connectivity_selection_to_series(selection: Any, axis: str) -> pd.Series:
        if isinstance(selection, pd.DataFrame):
            series = selection.mean(axis=0 if axis == "index" else 1)
        elif isinstance(selection, pd.Series):
            series = selection
        else:
            return pd.Series(dtype=float)
        return pd.to_numeric(series, errors="coerce").dropna()

    @staticmethod
    def _connectivity_selection_to_scalar(selection: Any) -> Optional[float]:
        if isinstance(selection, pd.DataFrame):
            stacked = selection.stack(dropna=True)
            numeric = pd.to_numeric(stacked, errors="coerce").dropna()
            if numeric.empty:
                return None
            return float(numeric.mean())
        if isinstance(selection, pd.Series):
            numeric = pd.to_numeric(selection, errors="coerce").dropna()
            if numeric.empty:
                return None
            return float(numeric.mean())
        try:
            value = float(selection)
        except Exception:
            return None
        return None if pd.isna(value) else value

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
                self._connectivity_matrix = self._deduplicate_connectivity_matrix(data)
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            data = getattr(compound[0], "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = self._deduplicate_connectivity_matrix(data)
            else:
                self._connectivity_matrix = pd.DataFrame()
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
            selection = matrix.loc[label] if axis == "index" else matrix[label]
            series = self._connectivity_selection_to_series(selection, axis=axis)
            if series.empty:
                return pd.DataFrame()
            if not series.index.is_unique:
                series = series.groupby(level=0, sort=False).mean()
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return pairwise connectivity values among the scaffold's resolved circuit nodes.
        Missing or unresolved regions are skipped.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        matched_rows: Dict[str, Any] = {}
        matched_cols: Dict[str, Any] = {}

        for key, region in self.region_objects.items():
            row_label = self._match_region_label(list(matrix.index), region)
            col_label = self._match_region_label(list(matrix.columns), region)
            if row_label is not None:
                matched_rows[key] = row_label
            if col_label is not None:
                matched_cols[key] = col_label

        keys = list(self.region_objects.keys())
        for src in keys:
            for dst in keys:
                if src == dst:
                    continue
                row_label = matched_rows.get(src)
                col_label = matched_cols.get(dst)
                if row_label is None or col_label is None:
                    continue
                try:
                    selection = matrix.loc[row_label, col_label]
                except Exception:
                    continue
                value = self._connectivity_selection_to_scalar(selection)
                if value is None:
                    continue
                rows.append(
                    {
                        "source_key": src,
                        "source_region": self.region_objects[src].name,
                        "target_key": dst,
                        "target_region": self.region_objects[dst].name,
                        "value": value,
                    }
                )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_HIV_NCD_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
        """
        Build the atlas-grounded graph and collect multimodal summaries.
        """
        nodes: List[Dict[str, Any]] = []
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
        self._connectivity_matrix = None

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
            desc = self.region_descriptions.get(key, "Atlas-backed circuit node")

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
                        "description": f"{desc} Unresolved in this siibra environment; keep as an explicit proxy.",
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
                    "description": desc,
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
        acute_persistent_cns_hiv_burden: float = 0.65,
        chemokine_monocyte_infiltration_liability: float = 0.50,
        chronic_immune_activation: float = 0.60,
        disease_severity_load: float = 0.50,
        antiretroviral_cns_control_support: float = 0.45,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator for the scaffold.

        Inputs are clipped to 0..1. Protective variables subtract from
        dysregulation. The calculation is acyclic:
            inputs -> latent biology -> regional burden -> symptoms -> phenotypes
        """
        inputs = {
            "acute_persistent_cns_hiv_burden": self._clip01(acute_persistent_cns_hiv_burden),
            "chemokine_monocyte_infiltration_liability": self._clip01(chemokine_monocyte_infiltration_liability),
            "chronic_immune_activation": self._clip01(chronic_immune_activation),
            "disease_severity_load": self._clip01(disease_severity_load),
            "antiretroviral_cns_control_support": self._clip01(antiretroviral_cns_control_support),
        }

        latents = {
            "monocyte_cns_trafficking": self._clip01(
                0.40 * inputs["acute_persistent_cns_hiv_burden"]
                + 0.32 * inputs["chemokine_monocyte_infiltration_liability"]
                + 0.18 * inputs["chronic_immune_activation"]
                + 0.08 * inputs["disease_severity_load"]
                - 0.22 * inputs["antiretroviral_cns_control_support"]
            ),
            "chronic_neuroinflammation": self._clip01(
                0.34 * inputs["chronic_immune_activation"]
                + 0.28 * inputs["acute_persistent_cns_hiv_burden"]
                + 0.24 * inputs["disease_severity_load"]
                + 0.16 * inputs["chemokine_monocyte_infiltration_liability"]
                - 0.24 * inputs["antiretroviral_cns_control_support"]
            ),
            "viral_protein_neurotoxicity": self._clip01(
                0.42 * inputs["acute_persistent_cns_hiv_burden"]
                + 0.24 * inputs["disease_severity_load"]
                + 0.18 * inputs["chronic_immune_activation"]
                + 0.08 * inputs["chemokine_monocyte_infiltration_liability"]
                - 0.22 * inputs["antiretroviral_cns_control_support"]
            ),
        }

        latents["synaptic_neuronal_dysfunction"] = self._clip01(
            0.34 * latents["chronic_neuroinflammation"]
            + 0.32 * latents["viral_protein_neurotoxicity"]
            + 0.18 * inputs["disease_severity_load"]
            + 0.10 * latents["monocyte_cns_trafficking"]
            - 0.12 * inputs["antiretroviral_cns_control_support"]
        )
        latents["subcortical_transmitter_imbalance"] = self._clip01(
            0.34 * latents["synaptic_neuronal_dysfunction"]
            + 0.28 * latents["chronic_neuroinflammation"]
            + 0.18 * latents["viral_protein_neurotoxicity"]
            + 0.08 * inputs["disease_severity_load"]
            - 0.10 * inputs["antiretroviral_cns_control_support"]
        )
        latents["subcortical_white_matter_disconnection"] = self._clip01(
            0.34 * latents["synaptic_neuronal_dysfunction"]
            + 0.30 * latents["chronic_neuroinflammation"]
            + 0.18 * inputs["disease_severity_load"]
            + 0.12 * latents["monocyte_cns_trafficking"]
            - 0.12 * inputs["antiretroviral_cns_control_support"]
        )
        latents["deep_gray_matter_network_failure"] = self._clip01(
            0.34 * latents["synaptic_neuronal_dysfunction"]
            + 0.24 * latents["subcortical_transmitter_imbalance"]
            + 0.20 * latents["subcortical_white_matter_disconnection"]
            + 0.14 * latents["viral_protein_neurotoxicity"]
            + 0.08 * inputs["disease_severity_load"]
        )
        latents["cerebellar_brainstem_pathway_disruption"] = self._clip01(
            0.30 * latents["chronic_neuroinflammation"]
            + 0.26 * latents["subcortical_white_matter_disconnection"]
            + 0.20 * inputs["disease_severity_load"]
            + 0.14 * latents["synaptic_neuronal_dysfunction"]
            + 0.10 * latents["viral_protein_neurotoxicity"]
            - 0.10 * inputs["antiretroviral_cns_control_support"]
        )

        regional_state = {
            "basal_ganglia_proxy": self._clip01(
                0.46 * latents["deep_gray_matter_network_failure"]
                + 0.24 * latents["subcortical_transmitter_imbalance"]
                + 0.14 * latents["synaptic_neuronal_dysfunction"]
                + 0.08 * latents["subcortical_white_matter_disconnection"]
                - 0.10 * inputs["antiretroviral_cns_control_support"]
            ),
            "thalamus_proxy": self._clip01(
                0.42 * latents["deep_gray_matter_network_failure"]
                + 0.22 * latents["subcortical_white_matter_disconnection"]
                + 0.16 * latents["chronic_neuroinflammation"]
                + 0.10 * latents["subcortical_transmitter_imbalance"]
                - 0.10 * inputs["antiretroviral_cns_control_support"]
            ),
            "cerebral_white_matter_proxy": self._clip01(
                0.52 * latents["subcortical_white_matter_disconnection"]
                + 0.22 * latents["chronic_neuroinflammation"]
                + 0.12 * inputs["disease_severity_load"]
                - 0.10 * inputs["antiretroviral_cns_control_support"]
            ),
            "cerebellum_proxy": self._clip01(
                0.50 * latents["cerebellar_brainstem_pathway_disruption"]
                + 0.18 * latents["chronic_neuroinflammation"]
                + 0.10 * inputs["disease_severity_load"]
                - 0.08 * inputs["antiretroviral_cns_control_support"]
            ),
            "brainstem_oculomotor_proxy": self._clip01(
                0.48 * latents["cerebellar_brainstem_pathway_disruption"]
                + 0.18 * latents["subcortical_white_matter_disconnection"]
                + 0.12 * latents["synaptic_neuronal_dysfunction"]
                - 0.08 * inputs["antiretroviral_cns_control_support"]
            ),
        }

        symptoms = {
            "psychomotor_slowing": self._clip01(
                0.34 * regional_state["basal_ganglia_proxy"]
                + 0.28 * regional_state["cerebral_white_matter_proxy"]
                + 0.16 * regional_state["thalamus_proxy"]
                + 0.12 * latents["subcortical_transmitter_imbalance"]
                + 0.08 * inputs["disease_severity_load"]
            ),
            "attentional_executive_dysfunction": self._clip01(
                0.30 * regional_state["cerebral_white_matter_proxy"]
                + 0.24 * regional_state["thalamus_proxy"]
                + 0.20 * regional_state["basal_ganglia_proxy"]
                + 0.16 * latents["synaptic_neuronal_dysfunction"]
                + 0.06 * inputs["disease_severity_load"]
            ),
            "behavioral_motivation_change": self._clip01(
                0.28 * regional_state["basal_ganglia_proxy"]
                + 0.24 * latents["synaptic_neuronal_dysfunction"]
                + 0.18 * latents["chronic_neuroinflammation"]
                + 0.12 * regional_state["thalamus_proxy"]
                + 0.10 * inputs["disease_severity_load"]
            ),
            "motor_incoordination_ataxia": self._clip01(
                0.40 * regional_state["cerebellum_proxy"]
                + 0.24 * regional_state["brainstem_oculomotor_proxy"]
                + 0.18 * regional_state["cerebral_white_matter_proxy"]
                + 0.08 * inputs["disease_severity_load"]
            ),
            "oculomotor_abnormalities": self._clip01(
                0.42 * regional_state["brainstem_oculomotor_proxy"]
                + 0.28 * regional_state["cerebellum_proxy"]
                + 0.16 * regional_state["cerebral_white_matter_proxy"]
                + 0.08 * inputs["disease_severity_load"]
            ),
        }

        symptoms["daily_function_interference"] = self._clip01(
            0.28 * symptoms["psychomotor_slowing"]
            + 0.28 * symptoms["attentional_executive_dysfunction"]
            + 0.18 * symptoms["motor_incoordination_ataxia"]
            + 0.14 * symptoms["behavioral_motivation_change"]
            + 0.08 * symptoms["oculomotor_abnormalities"]
            + 0.04 * inputs["disease_severity_load"]
        )

        phenotypes = {
            "hand_continuum_burden": self._clip01(
                (
                    symptoms["psychomotor_slowing"]
                    + symptoms["attentional_executive_dysfunction"]
                    + symptoms["behavioral_motivation_change"]
                    + symptoms["daily_function_interference"]
                ) / 4.0
            ),
            "subcortical_dementia_profile": self._clip01(
                (
                    symptoms["psychomotor_slowing"]
                    + symptoms["attentional_executive_dysfunction"]
                    + regional_state["basal_ganglia_proxy"]
                    + regional_state["thalamus_proxy"]
                    + regional_state["cerebral_white_matter_proxy"]
                ) / 5.0
            ),
            "motor_cerebellar_profile": self._clip01(
                (
                    symptoms["motor_incoordination_ataxia"]
                    + symptoms["oculomotor_abnormalities"]
                    + regional_state["cerebellum_proxy"]
                    + regional_state["brainstem_oculomotor_proxy"]
                ) / 4.0
            ),
            "ani_pattern_index": self._clip01(
                0.40 * symptoms["attentional_executive_dysfunction"]
                + 0.30 * symptoms["psychomotor_slowing"]
                + 0.16 * regional_state["cerebral_white_matter_proxy"]
                - 0.26 * symptoms["daily_function_interference"]
            ),
            "mnd_pattern_index": self._clip01(
                0.28 * symptoms["attentional_executive_dysfunction"]
                + 0.22 * symptoms["psychomotor_slowing"]
                + 0.26 * symptoms["daily_function_interference"]
                + 0.14 * symptoms["behavioral_motivation_change"]
                + 0.10 * symptoms["motor_incoordination_ataxia"]
            ),
            "had_like_subcortical_dementia_profile": self._clip01(
                0.24 * symptoms["psychomotor_slowing"]
                + 0.22 * symptoms["attentional_executive_dysfunction"]
                + 0.16 * symptoms["behavioral_motivation_change"]
                + 0.14 * symptoms["motor_incoordination_ataxia"]
                + 0.18 * symptoms["daily_function_interference"]
                + 0.06 * symptoms["oculomotor_abnormalities"]
            ),
        }

        return {
            "inputs": pd.Series(inputs, name="inputs"),
            "latents": pd.Series(latents, name="latents"),
            "regional_state": pd.Series(regional_state, name="regional_state"),
            "symptoms": pd.Series(symptoms, name="symptoms"),
            "phenotypes": pd.Series(phenotypes, name="phenotypes"),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Assign an MNI coordinate to Julich regions using a statistical map.
        """
        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception:
                with siibra.QUIET:
                    self._pmap = self.atlas.get_map(
                        parcellation=self.parcellation,
                        space=self.atlas.get_space(self.assignment_space),
                        maptype="statistical",
                    )

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break

        if "region" in assignments.columns:
            assignments = assignments.copy()
            assignments["region"] = assignments["region"].map(self._name_of)
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        """
        Return a regional mask object for a resolved region node.

        Call `.fetch()` on the returned object to obtain the underlying image.
        Returns None for unresolved nodes.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(space=self.space, maptype="labelled")
            except Exception:
                return None


if __name__ == "__main__":
    if siibra is None:
        print(
            "siibra is not installed in this environment. Install siibra, then rerun "
            "this script to build the atlas-grounded HIV neurocognitive disorder scaffold."
        )
        raise SystemExit(0)

    model = MajorOrMildNeurocognitiveDisorderDueToHivInfectionModel()
    built = model.build(connectivity_rows=10)

    print("\n=== NODES ===")
    print(built["nodes"][["key", "node_type", "label", "atlas_region", "feature_summary"]].to_string(index=False))

    print("\n=== EDGES ===")
    print(built["edges"][["source", "target", "relation", "hiv_ncd_change"]].to_string(index=False))

    print("\n=== REGION RESOLUTION ===")
    if model.region_objects:
        for key, region in model.region_objects.items():
            print(f"{key}: {region.name}")
    else:
        print("No regions resolved in this environment.")

    for node_key in [
        "basal_ganglia_proxy",
        "thalamus_proxy",
        "cerebral_white_matter_proxy",
        "cerebellum_proxy",
        "brainstem_oculomotor_proxy",
    ]:
        receptor_df = built["receptors"].get(node_key, pd.DataFrame())
        gene_df = built["genes"].get(node_key, pd.DataFrame())
        conn_df = built["connectivity_profiles"].get(node_key, pd.DataFrame())

        print(f"\n=== FEATURES: {node_key} ===")
        print("Receptors:")
        print(receptor_df.head().to_string(index=False) if not receptor_df.empty else "<none>")
        print("Genes:")
        print(gene_df.head().to_string(index=False) if not gene_df.empty else "<none>")
        print("Connectivity:")
        print(conn_df.head().to_string(index=False) if not conn_df.empty else "<none>")

    circuit_df = built["circuit_connectivity"]
    print("\n=== CIRCUIT CONNECTIVITY ===")
    print(circuit_df.head(12).to_string(index=False) if not circuit_df.empty else "<none>")

    sim = model.simulate(
        acute_persistent_cns_hiv_burden=0.78,
        chemokine_monocyte_infiltration_liability=0.58,
        chronic_immune_activation=0.72,
        disease_severity_load=0.64,
        antiretroviral_cns_control_support=0.42,
    )

    print("\n=== SIMULATION: INPUTS ===")
    print(sim["inputs"].round(3).to_string())
    print("\n=== SIMULATION: LATENTS ===")
    print(sim["latents"].round(3).sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: REGIONAL STATE ===")
    print(sim["regional_state"].round(3).sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: SYMPTOMS ===")
    print(sim["symptoms"].round(3).sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: PHENOTYPES ===")
    print(sim["phenotypes"].round(3).sort_values(ascending=False).to_string())

    # Example coordinate assignment for future use:
    # print(model.assign_mni_point((-12, 8, 8)).head())
    # mask = model.region_mask("thalamus_proxy")
    # if mask is not None:
    #     nii = mask.fetch()
