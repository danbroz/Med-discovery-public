from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Major or Mild Neurocognitive Disorder Due to
Prion Disease.

This script turns a chapter-level biological summary of prion disease into a
transparent, research-facing mechanistic scaffold. It is intended for atlas-
grounded exploration and refinement, not as a validated disease model,
diagnostic instrument, or treatment tool.

Design choices:
- The chapter centers on protein misfolding, spongiform degeneration, gliosis,
  and distributed circuit failure. Those processes are therefore kept as latent
  biology rather than over-localized into a single parcel.
- Deep-gray structures (caudate, putamen, thalamus), cerebellar burden, and
  brainstem monoaminergic nuclei are represented as explicit proxies because
  atlas label availability can vary across siibra environments.
- Frontal and temporal cortical burden are represented with conservative
  cortical proxy anchors because the chapter links rapidly progressive dementia,
  executive/language/visuospatial impairment, personality change, apathy, and
  disinhibition to frontal and temporal involvement.
- The simulator is explicit and one-pass:
  inputs -> latent pathology -> regional burden -> symptoms / imaging markers
  -> phenotype summaries.

Higher values in ``regional_state`` indicate higher inferred tissue burden or
functional dysregulation, not healthy activation.
"""

import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:  # pragma: no cover - environment dependent
    import siibra  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore


DEFAULT_GENE_PANEL = [
    "PRNP",
    "DRD2",
    "SLC6A3",   # DAT1
    "TH",
    "SLC6A4",
    "TPH2",
    "HTR1A",
    "SLC6A2",
    "DBH",
    "CHAT",
    "ACHE",
    "CHRNA4",
    "BDNF",
]


class PrionDiseaseNeurocognitiveDisorderModel:
    """
    Research scaffold for chapter-driven prion-disease neurobiology.

    The build() method resolves conservative atlas regions (or explicit proxy
    nodes), retrieves receptor / gene / connectivity summaries where available,
    and returns stable node and edge tables.

    The simulate() method models how prion misfolding burden, genetic liability,
    disease stage, and regional involvement propagate into distributed tissue
    injury, monoaminergic disruption, cholinergic cognitive failure, and
    frontotemporal breakdown, yielding movement, cognitive, mood, sleep, and
    imaging phenotypes.
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

        self.siibra_available = siibra is not None
        self.atlas = None
        self.parcellation = None
        self.space = None
        self._quiet = nullcontext()

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
                self._quiet = getattr(siibra, "QUIET", nullcontext())
            except Exception as exc:
                warnings.warn(
                    f"siibra atlas initialization failed ({exc!r}). "
                    "Atlas-backed retrieval will be unavailable, but simulate() remains usable."
                )
                self.siibra_available = False
        else:
            warnings.warn(
                "siibra is not installed in this environment. "
                "build() will return empty atlas-backed feature tables, but simulate() still works."
            )

        # Deep gray and brainstem structures are represented conservatively as
        # proxies, since exact Julich label availability varies across siibra
        # environments. Frontal and temporal cortex are proxy-anchored to
        # representative left-hemisphere cortical labels where possible.
        self.region_candidates: Dict[str, List[str]] = {
            "caudate_proxy": [
                "caudate nucleus left",
                "caudate left",
                "caudate nucleus",
                "caudate",
            ],
            "putamen_proxy": [
                "putamen left",
                "putamen",
            ],
            "thalamus_proxy": [
                "thalamus left",
                "thalamus",
            ],
            "frontal_cortex_proxy": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 8Av left",
                "frontal cortex",
                "frontal lobe",
            ],
            "temporal_cortex_proxy": [
                "Area TE 3 left",
                "Area TE 1.0 left",
                "Area TE 2.1 left",
                "Area TGd left",
                "temporal cortex",
                "temporal lobe",
            ],
            "cerebellum_proxy": [
                "cerebellum left",
                "cerebellum",
            ],
            "raphe_proxy": [
                "raphe nuclei",
                "brainstem raphe",
                "raphe",
            ],
            "locus_coeruleus_proxy": [
                "locus coeruleus",
                "coeruleus",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "prion_misfolding_load": (
                "Burden of misfolded prion protein (PrPSc) driving downstream tissue injury."
            ),
            "prnp_mutation_liability": (
                "Hereditary PRNP mutation burden increasing the probability of unstable prion protein conformation."
            ),
            "codon_129_susceptibility": (
                "PRNP codon-129 susceptibility load shaping disease risk and a faster course."
            ),
            "disease_stage": (
                "Overall stage / progression burden from earlier mild impairment toward major neurocognitive disorder."
            ),
            "cortical_involvement": (
                "Degree of cortical pathology burden, especially the frontal and temporal cortex."
            ),
            "deep_gray_involvement": (
                "Degree of deep-gray pathology burden involving caudate, putamen, and thalamic systems."
            ),
            "brainstem_involvement": (
                "Extent of pathology burden in brainstem nuclei such as raphe and locus coeruleus."
            ),
            "cerebellar_involvement": (
                "Extent of cerebellar burden contributing to ataxic presentation."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "protein_misfolding_neurodegeneration": (
                "Prion-conformation driven neuronal loss and network vulnerability."
            ),
            "spongiform_synaptic_gliotic_injury": (
                "Combined synaptic dysfunction, spongiform change, and gliosis burden."
            ),
            "nigrostriatal_dopaminergic_disruption": (
                "Basal-ganglia dopaminergic failure linked to parkinsonism, apathy, and depression."
            ),
            "serotonergic_noradrenergic_decline": (
                "Brainstem monoaminergic decline associated with mood, anxiety, and sleep-wake disturbance."
            ),
            "cholinergic_attention_memory_failure": (
                "Central cholinergic compromise affecting attention and memory."
            ),
            "frontotemporal_network_breakdown": (
                "Frontal-temporal cortical disintegration underlying rapidly progressive dementia and behavioral change."
            ),
            "sleep_wake_circuit_instability": (
                "Instability in sleep-wake regulation driven by brainstem monoaminergic injury and thalamic burden."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "rapid_cognitive_decline": (
                "Rapidly progressive decline toward mild or major neurocognitive disorder."
            ),
            "executive_language_visuospatial_impairment": (
                "Distributed higher-order cognitive deficits spanning executive, language, and visuospatial domains."
            ),
            "parkinsonism": "Rigidity, bradykinesia, and hypokinetic motor slowing.",
            "cerebellar_ataxia": "Ataxic motor phenotype linked to cerebellar burden.",
            "myoclonus": "Myoclonic jerks emerging from distributed tissue injury.",
            "apathy_depression": "Apathy and depressive symptoms tied to dopaminergic and cortical disruption.",
            "anxiety_behavioral_disturbance": (
                "Anxiety and behavioral / neuropsychiatric disturbance linked to monoaminergic decline."
            ),
            "sleep_wake_disturbance": "Disturbed sleep-wake regulation, including insomnia-like phenotypes.",
            "personality_change_disinhibition": (
                "Personality change, apathy, or disinhibition associated with frontal-temporal disease."
            ),
            "daily_function_interference": (
                "Interference with day-to-day functioning consistent with major neurocognitive disorder severity."
            ),
            "cortical_ribboning_mri_pattern": (
                "Supportive MRI cortical-ribboning pattern on DWI / FLAIR."
            ),
            "deep_gray_mri_pattern": (
                "Supportive MRI deep-gray hyperintensity pattern involving caudate, putamen, and/or thalamus."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "prion_misfolding_load",
                "target": "protein_misfolding_neurodegeneration",
                "relation": "misfolded PrP burden drives neuronal death and network failure",
                "prion_change": "increased",
            },
            {
                "source": "prnp_mutation_liability",
                "target": "protein_misfolding_neurodegeneration",
                "relation": "pathogenic PRNP variation destabilizes protein conformation",
                "prion_change": "increased",
            },
            {
                "source": "codon_129_susceptibility",
                "target": "protein_misfolding_neurodegeneration",
                "relation": "susceptibility polymorphism increases risk and may accelerate course",
                "prion_change": "increased",
            },
            {
                "source": "disease_stage",
                "target": "spongiform_synaptic_gliotic_injury",
                "relation": "later stage corresponds to greater spongiform degeneration and gliosis burden",
                "prion_change": "increased",
            },
            {
                "source": "protein_misfolding_neurodegeneration",
                "target": "spongiform_synaptic_gliotic_injury",
                "relation": "misfolding pathology propagates synaptic dysfunction, spongiform change, and gliosis",
                "prion_change": "increased",
            },
            {
                "source": "deep_gray_involvement",
                "target": "nigrostriatal_dopaminergic_disruption",
                "relation": "basal-ganglia burden disrupts dopaminergic motor circuitry",
                "prion_change": "increased",
            },
            {
                "source": "spongiform_synaptic_gliotic_injury",
                "target": "nigrostriatal_dopaminergic_disruption",
                "relation": "degeneration worsens nigrostriatal signaling failure",
                "prion_change": "increased",
            },
            {
                "source": "brainstem_involvement",
                "target": "serotonergic_noradrenergic_decline",
                "relation": "raphe and locus-coeruleus involvement weakens serotonin and noradrenaline systems",
                "prion_change": "increased",
            },
            {
                "source": "spongiform_synaptic_gliotic_injury",
                "target": "serotonergic_noradrenergic_decline",
                "relation": "distributed injury degrades monoaminergic brainstem function",
                "prion_change": "increased",
            },
            {
                "source": "cortical_involvement",
                "target": "cholinergic_attention_memory_failure",
                "relation": "cortical disease burden compromises attention and memory systems",
                "prion_change": "increased",
            },
            {
                "source": "spongiform_synaptic_gliotic_injury",
                "target": "cholinergic_attention_memory_failure",
                "relation": "synaptic and gliotic injury worsens cholinergic cognitive failure",
                "prion_change": "increased",
            },
            {
                "source": "cortical_involvement",
                "target": "frontotemporal_network_breakdown",
                "relation": "frontal and temporal pathology disrupts higher-order cortical networks",
                "prion_change": "increased",
            },
            {
                "source": "spongiform_synaptic_gliotic_injury",
                "target": "frontotemporal_network_breakdown",
                "relation": "tissue injury propagates rapidly progressive cortical dysfunction",
                "prion_change": "increased",
            },
            {
                "source": "serotonergic_noradrenergic_decline",
                "target": "sleep_wake_circuit_instability",
                "relation": "brainstem monoaminergic decline destabilizes sleep-wake regulation",
                "prion_change": "increased",
            },
            {
                "source": "nigrostriatal_dopaminergic_disruption",
                "target": "caudate_proxy",
                "relation": "dopaminergic network burden is expressed in deep-gray striatal systems",
                "prion_change": "increased",
            },
            {
                "source": "nigrostriatal_dopaminergic_disruption",
                "target": "putamen_proxy",
                "relation": "motor striatal systems bear dopaminergic and deep-gray disease burden",
                "prion_change": "increased",
            },
            {
                "source": "spongiform_synaptic_gliotic_injury",
                "target": "thalamus_proxy",
                "relation": "deep-gray and thalamic burden reflects progressing spongiform injury",
                "prion_change": "increased",
            },
            {
                "source": "frontotemporal_network_breakdown",
                "target": "frontal_cortex_proxy",
                "relation": "frontal cortical burden indexes executive and behavioral network failure",
                "prion_change": "increased",
            },
            {
                "source": "frontotemporal_network_breakdown",
                "target": "temporal_cortex_proxy",
                "relation": "temporal cortical burden indexes language and broader cortical network failure",
                "prion_change": "increased",
            },
            {
                "source": "cerebellar_involvement",
                "target": "cerebellum_proxy",
                "relation": "cerebellar burden supports ataxic presentation",
                "prion_change": "increased",
            },
            {
                "source": "serotonergic_noradrenergic_decline",
                "target": "raphe_proxy",
                "relation": "serotonergic brainstem decline is represented by raphe proxy burden",
                "prion_change": "increased",
            },
            {
                "source": "serotonergic_noradrenergic_decline",
                "target": "locus_coeruleus_proxy",
                "relation": "noradrenergic brainstem decline is represented by locus-coeruleus proxy burden",
                "prion_change": "increased",
            },
            {
                "source": "cholinergic_attention_memory_failure",
                "target": "rapid_cognitive_decline",
                "relation": "cholinergic compromise contributes to dementia-level cognitive decline",
                "prion_change": "increased",
            },
            {
                "source": "frontotemporal_network_breakdown",
                "target": "executive_language_visuospatial_impairment",
                "relation": "cortical network failure drives multi-domain cognitive deficits",
                "prion_change": "increased",
            },
            {
                "source": "nigrostriatal_dopaminergic_disruption",
                "target": "parkinsonism",
                "relation": "deep-gray dopaminergic failure produces rigidity and bradykinesia",
                "prion_change": "increased",
            },
            {
                "source": "nigrostriatal_dopaminergic_disruption",
                "target": "apathy_depression",
                "relation": "dopamine dysfunction contributes to apathy and depression",
                "prion_change": "increased",
            },
            {
                "source": "serotonergic_noradrenergic_decline",
                "target": "anxiety_behavioral_disturbance",
                "relation": "monoamine failure contributes to anxiety and neuropsychiatric symptoms",
                "prion_change": "increased",
            },
            {
                "source": "sleep_wake_circuit_instability",
                "target": "sleep_wake_disturbance",
                "relation": "brainstem sleep-wake instability drives insomnia-like and circadian symptoms",
                "prion_change": "increased",
            },
            {
                "source": "frontal_cortex_proxy",
                "target": "personality_change_disinhibition",
                "relation": "frontal damage contributes to apathy, personality change, and disinhibition",
                "prion_change": "increased",
            },
            {
                "source": "temporal_cortex_proxy",
                "target": "personality_change_disinhibition",
                "relation": "temporal cortical involvement contributes to behavioral and cognitive change",
                "prion_change": "increased",
            },
            {
                "source": "cerebellum_proxy",
                "target": "cerebellar_ataxia",
                "relation": "cerebellar burden yields ataxic presentation",
                "prion_change": "increased",
            },
            {
                "source": "spongiform_synaptic_gliotic_injury",
                "target": "myoclonus",
                "relation": "distributed injury contributes to myoclonic motor phenomena",
                "prion_change": "increased",
            },
            {
                "source": "cortical_involvement",
                "target": "cortical_ribboning_mri_pattern",
                "relation": "cortical disease burden supports ribboning pattern on MRI",
                "prion_change": "increased",
            },
            {
                "source": "deep_gray_involvement",
                "target": "deep_gray_mri_pattern",
                "relation": "caudate, putamen, and thalamic burden supports deep-gray MRI abnormalities",
                "prion_change": "increased",
            },
            {
                "source": "rapid_cognitive_decline",
                "target": "daily_function_interference",
                "relation": "progressive cognitive failure compromises independent daily functioning",
                "prion_change": "increased",
            },
            {
                "source": "parkinsonism",
                "target": "daily_function_interference",
                "relation": "motor slowing and rigidity worsen functional dependence",
                "prion_change": "increased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        self._pmap: Any = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    def _modality_candidates(self, kind: str) -> List[Any]:
        cands: List[Any] = []
        if siibra is not None:
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

    def _safe_features(self, concept: Any, modality: Any, **kwargs: Any) -> List[Any]:
        if not self.siibra_available or concept is None:
            return []
        try:
            with self._quiet:
                feats = siibra.features.get(concept, modality, **kwargs)
            return list(feats) if feats else []
        except Exception:
            return []

    def _safe_features_any(self, concept: Any, modalities: Sequence[Any], **kwargs: Any) -> List[Any]:
        for modality in modalities:
            feats = self._safe_features(concept, modality, **kwargs)
            if feats:
                return feats
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
        proxy_penalty = 1 if "proxy" in name else 0
        generic_penalty = 1 if name in {
            "caudate nucleus",
            "putamen",
            "thalamus",
            "cerebellum",
            "frontal cortex",
            "temporal cortex",
        } else 0
        return (left_bonus, right_penalty, proxy_penalty, generic_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if not self.siibra_available:
            return None
        for spec in candidates:
            try:
                if self.atlas is not None:
                    return self.atlas.get_region(spec, parcellation=self.parcellation)
            except Exception:
                pass
            try:
                if self.parcellation is not None:
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
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
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
        if region is None or not self.siibra_available:
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
        volume_mm3 = getattr(main, "volume", None)
        return centroid_xyz, (float(volume_mm3) if volume_mm3 is not None else None)

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            try:
                data = getattr(feat, "data", None)
                if isinstance(data, pd.DataFrame):
                    df = data.copy().reset_index()
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
                data = getattr(feat, "data", None)
                if not isinstance(data, pd.DataFrame):
                    continue
                df = data.copy()
                lower_cols = {str(c).lower(): c for c in df.columns}
                required = {"gene", "level", "zscore"}
                if required.issubset(lower_cols):
                    gene_col = lower_cols["gene"]
                    level_col = lower_cols["level"]
                    zscore_col = lower_cols["zscore"]
                    out = (
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
                    return out
                return df.reset_index(drop=True)
            except Exception:
                continue
        return pd.DataFrame()

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

        preferred = None
        for feat in feats:
            cohort = getattr(feat, "cohort", None)
            if cohort == self.connectivity_cohort:
                preferred = feat
                break
        compound = preferred or feats[0]

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            sub = compound[0]
            data = getattr(sub, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        if region is None:
            return None
        exact = [x for x in labels if self._name_of(x) == getattr(region, "name", None)]
        if exact:
            return exact[0]

        rn = getattr(region, "name", "").lower()
        fuzzy = [
            x for x in labels
            if rn and (
                rn in self._name_of(x).lower()
                or self._name_of(x).lower() in rn
            )
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
            if not isinstance(series, pd.Series):
                series = pd.Series(series)
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            region_name = getattr(region, "name", "")
            df = df[df["connected_region"] != region_name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _feature_summary(self, receptor_df: pd.DataFrame, gene_df: pd.DataFrame, conn_df: pd.DataFrame) -> str:
        return (
            f"receptors={'yes' if not receptor_df.empty else 'no'}; "
            f"genes={'yes' if not gene_df.empty else 'no'}; "
            f"connectivity={'yes' if not conn_df.empty else 'no'}"
        )

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return a within-circuit connectivity submatrix for resolved scaffold nodes.

        Rows and columns are renamed to scaffold node keys so the output remains
        stable even when exact siibra labels differ.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        matched_labels: Dict[str, Any] = {}
        for node_key, region in self.region_objects.items():
            label = self._match_region_label(list(matrix.index), region)
            axis = "index"
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
                axis = "columns"
            if label is None or axis != "index":
                continue
            matched_labels[node_key] = label

        if len(matched_labels) < 2:
            return pd.DataFrame()

        labels = list(matched_labels.values())
        try:
            sub = matrix.loc[labels, labels].copy()
            rename_map = {label: key for key, label in matched_labels.items()}
            sub = sub.rename(index=rename_map, columns=rename_map)
            return sub
        except Exception:
            return pd.DataFrame()

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
            region_is_proxy = key.endswith("_proxy")

            if region is None:
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region_proxy" if region_is_proxy else "region",
                        "description": (
                            "Explicit proxy node unresolved in this environment"
                            if region_is_proxy else
                            "Atlas-backed node unresolved in this environment"
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
                    "label": getattr(region, "name", key),
                    "node_type": "region_proxy" if region_is_proxy else "region",
                    "description": (
                        "Atlas-backed proxy node for a chapter-level system"
                        if region_is_proxy else
                        "Atlas-backed circuit node"
                    ),
                    "atlas_region": getattr(region, "name", None),
                    "region_identifier": getattr(region, "identifier", None),
                    "centroid_mni": centroid_mni,
                    "volume_mm3": volume_mm3,
                    "feature_summary": self._feature_summary(receptor_df, gene_df, conn_df),
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
        prion_misfolding_load: float = 0.70,
        prnp_mutation_liability: float = 0.20,
        codon_129_susceptibility: float = 0.50,
        disease_stage: float = 0.60,
        cortical_involvement: float = 0.65,
        deep_gray_involvement: float = 0.60,
        brainstem_involvement: float = 0.35,
        cerebellar_involvement: float = 0.30,
    ) -> Dict[str, pd.Series]:
        """
        Transparent 0..1 simulator.

        Inputs describe pathology burden and anatomical involvement rather than
        probabilities of diagnosis. Higher values indicate greater inferred
        disease burden.

        Returns a dictionary of pandas Series:
        - inputs
        - latents
        - regional_state
        - symptoms
        - phenotypes
        """
        inputs = {
            "prion_misfolding_load": self._clip01(prion_misfolding_load),
            "prnp_mutation_liability": self._clip01(prnp_mutation_liability),
            "codon_129_susceptibility": self._clip01(codon_129_susceptibility),
            "disease_stage": self._clip01(disease_stage),
            "cortical_involvement": self._clip01(cortical_involvement),
            "deep_gray_involvement": self._clip01(deep_gray_involvement),
            "brainstem_involvement": self._clip01(brainstem_involvement),
            "cerebellar_involvement": self._clip01(cerebellar_involvement),
        }

        regional_burden_mean = self._clip01(
            0.34 * inputs["cortical_involvement"]
            + 0.28 * inputs["deep_gray_involvement"]
            + 0.20 * inputs["brainstem_involvement"]
            + 0.18 * inputs["cerebellar_involvement"]
        )

        latents = {
            # Upstream protein-conformation pathology.
            "protein_misfolding_neurodegeneration": self._clip01(
                0.44 * inputs["prion_misfolding_load"]
                + 0.18 * inputs["prnp_mutation_liability"]
                + 0.14 * inputs["codon_129_susceptibility"]
                + 0.18 * inputs["disease_stage"]
            ),
            # Histopathological hallmarks: synaptic dysfunction, spongiform change, gliosis.
            "spongiform_synaptic_gliotic_injury": self._clip01(
                0.40 * regional_burden_mean
                + 0.28 * inputs["disease_stage"]
                + 0.22 * (
                    0.44 * inputs["prion_misfolding_load"]
                    + 0.18 * inputs["prnp_mutation_liability"]
                    + 0.14 * inputs["codon_129_susceptibility"]
                    + 0.18 * inputs["disease_stage"]
                )
            ),
        }

        latents["nigrostriatal_dopaminergic_disruption"] = self._clip01(
            0.42 * inputs["deep_gray_involvement"]
            + 0.30 * latents["spongiform_synaptic_gliotic_injury"]
            + 0.16 * latents["protein_misfolding_neurodegeneration"]
            + 0.08 * inputs["disease_stage"]
        )

        latents["serotonergic_noradrenergic_decline"] = self._clip01(
            0.42 * inputs["brainstem_involvement"]
            + 0.28 * latents["spongiform_synaptic_gliotic_injury"]
            + 0.16 * latents["protein_misfolding_neurodegeneration"]
            + 0.08 * inputs["disease_stage"]
        )

        latents["cholinergic_attention_memory_failure"] = self._clip01(
            0.36 * inputs["cortical_involvement"]
            + 0.28 * latents["spongiform_synaptic_gliotic_injury"]
            + 0.22 * latents["protein_misfolding_neurodegeneration"]
            + 0.08 * inputs["disease_stage"]
        )

        latents["frontotemporal_network_breakdown"] = self._clip01(
            0.40 * inputs["cortical_involvement"]
            + 0.30 * latents["spongiform_synaptic_gliotic_injury"]
            + 0.20 * latents["protein_misfolding_neurodegeneration"]
            + 0.08 * inputs["disease_stage"]
        )

        latents["sleep_wake_circuit_instability"] = self._clip01(
            0.44 * latents["serotonergic_noradrenergic_decline"]
            + 0.20 * inputs["brainstem_involvement"]
            + 0.18 * inputs["deep_gray_involvement"]
            + 0.10 * inputs["disease_stage"]
        )

        regional_state = {
            "caudate_proxy": self._clip01(
                0.48 * latents["nigrostriatal_dopaminergic_disruption"]
                + 0.22 * inputs["deep_gray_involvement"]
                + 0.16 * latents["spongiform_synaptic_gliotic_injury"]
            ),
            "putamen_proxy": self._clip01(
                0.52 * latents["nigrostriatal_dopaminergic_disruption"]
                + 0.22 * inputs["deep_gray_involvement"]
                + 0.14 * latents["spongiform_synaptic_gliotic_injury"]
            ),
            "thalamus_proxy": self._clip01(
                0.34 * inputs["deep_gray_involvement"]
                + 0.30 * latents["spongiform_synaptic_gliotic_injury"]
                + 0.16 * latents["sleep_wake_circuit_instability"]
                + 0.08 * inputs["disease_stage"]
            ),
            "frontal_cortex_proxy": self._clip01(
                0.48 * latents["frontotemporal_network_breakdown"]
                + 0.22 * latents["cholinergic_attention_memory_failure"]
                + 0.12 * inputs["cortical_involvement"]
            ),
            "temporal_cortex_proxy": self._clip01(
                0.44 * latents["frontotemporal_network_breakdown"]
                + 0.24 * latents["cholinergic_attention_memory_failure"]
                + 0.12 * inputs["cortical_involvement"]
            ),
            "cerebellum_proxy": self._clip01(
                0.48 * inputs["cerebellar_involvement"]
                + 0.24 * latents["spongiform_synaptic_gliotic_injury"]
                + 0.12 * latents["protein_misfolding_neurodegeneration"]
            ),
            "raphe_proxy": self._clip01(
                0.54 * latents["serotonergic_noradrenergic_decline"]
                + 0.20 * inputs["brainstem_involvement"]
            ),
            "locus_coeruleus_proxy": self._clip01(
                0.50 * latents["serotonergic_noradrenergic_decline"]
                + 0.22 * inputs["brainstem_involvement"]
            ),
        }

        symptoms = {
            "rapid_cognitive_decline": self._clip01(
                0.34 * latents["cholinergic_attention_memory_failure"]
                + 0.26 * latents["frontotemporal_network_breakdown"]
                + 0.18 * latents["spongiform_synaptic_gliotic_injury"]
                + 0.14 * inputs["disease_stage"]
            ),
            "executive_language_visuospatial_impairment": self._clip01(
                0.30 * latents["frontotemporal_network_breakdown"]
                + 0.24 * latents["cholinergic_attention_memory_failure"]
                + 0.14 * regional_state["frontal_cortex_proxy"]
                + 0.14 * regional_state["temporal_cortex_proxy"]
            ),
            "parkinsonism": self._clip01(
                0.34 * latents["nigrostriatal_dopaminergic_disruption"]
                + 0.20 * regional_state["putamen_proxy"]
                + 0.16 * regional_state["caudate_proxy"]
                + 0.12 * regional_state["thalamus_proxy"]
            ),
            "cerebellar_ataxia": self._clip01(
                0.46 * regional_state["cerebellum_proxy"]
                + 0.18 * latents["spongiform_synaptic_gliotic_injury"]
                + 0.16 * inputs["cerebellar_involvement"]
            ),
            "myoclonus": self._clip01(
                0.30 * latents["spongiform_synaptic_gliotic_injury"]
                + 0.18 * regional_state["thalamus_proxy"]
                + 0.16 * regional_state["frontal_cortex_proxy"]
                + 0.12 * regional_state["temporal_cortex_proxy"]
            ),
            "apathy_depression": self._clip01(
                0.28 * latents["nigrostriatal_dopaminergic_disruption"]
                + 0.28 * latents["serotonergic_noradrenergic_decline"]
                + 0.14 * regional_state["frontal_cortex_proxy"]
                + 0.10 * inputs["disease_stage"]
            ),
            "anxiety_behavioral_disturbance": self._clip01(
                0.34 * latents["serotonergic_noradrenergic_decline"]
                + 0.16 * regional_state["locus_coeruleus_proxy"]
                + 0.14 * regional_state["raphe_proxy"]
                + 0.10 * regional_state["temporal_cortex_proxy"]
            ),
            "sleep_wake_disturbance": self._clip01(
                0.42 * latents["sleep_wake_circuit_instability"]
                + 0.18 * regional_state["raphe_proxy"]
                + 0.16 * regional_state["locus_coeruleus_proxy"]
                + 0.10 * regional_state["thalamus_proxy"]
            ),
            "personality_change_disinhibition": self._clip01(
                0.28 * latents["frontotemporal_network_breakdown"]
                + 0.20 * regional_state["frontal_cortex_proxy"]
                + 0.18 * regional_state["temporal_cortex_proxy"]
                + 0.10 * latents["serotonergic_noradrenergic_decline"]
            ),
            "daily_function_interference": self._clip01(
                0.28 * (
                    0.34 * latents["cholinergic_attention_memory_failure"]
                    + 0.26 * latents["frontotemporal_network_breakdown"]
                    + 0.18 * latents["spongiform_synaptic_gliotic_injury"]
                    + 0.14 * inputs["disease_stage"]
                )
                + 0.20 * (
                    0.30 * latents["frontotemporal_network_breakdown"]
                    + 0.24 * latents["cholinergic_attention_memory_failure"]
                    + 0.14 * regional_state["frontal_cortex_proxy"]
                    + 0.14 * regional_state["temporal_cortex_proxy"]
                )
                + 0.12 * (
                    0.34 * latents["nigrostriatal_dopaminergic_disruption"]
                    + 0.20 * regional_state["putamen_proxy"]
                    + 0.16 * regional_state["caudate_proxy"]
                    + 0.12 * regional_state["thalamus_proxy"]
                )
                + 0.10 * (
                    0.46 * regional_state["cerebellum_proxy"]
                    + 0.18 * latents["spongiform_synaptic_gliotic_injury"]
                    + 0.16 * inputs["cerebellar_involvement"]
                )
                + 0.10 * (
                    0.42 * latents["sleep_wake_circuit_instability"]
                    + 0.18 * regional_state["raphe_proxy"]
                    + 0.16 * regional_state["locus_coeruleus_proxy"]
                    + 0.10 * regional_state["thalamus_proxy"]
                )
            ),
            "cortical_ribboning_mri_pattern": self._clip01(
                0.40 * inputs["cortical_involvement"]
                + 0.26 * latents["spongiform_synaptic_gliotic_injury"]
                + 0.18 * regional_state["frontal_cortex_proxy"]
                + 0.10 * regional_state["temporal_cortex_proxy"]
            ),
            "deep_gray_mri_pattern": self._clip01(
                0.36 * inputs["deep_gray_involvement"]
                + 0.26 * latents["spongiform_synaptic_gliotic_injury"]
                + 0.14 * regional_state["caudate_proxy"]
                + 0.14 * regional_state["putamen_proxy"]
                + 0.06 * regional_state["thalamus_proxy"]
            ),
        }

        phenotypes = {
            "mild_neurocognitive_disorder_profile": self._clip01(
                0.30 * symptoms["rapid_cognitive_decline"]
                + 0.24 * symptoms["executive_language_visuospatial_impairment"]
                + 0.12 * symptoms["apathy_depression"]
                + 0.10 * symptoms["sleep_wake_disturbance"]
                + 0.12 * (1.0 - symptoms["daily_function_interference"])
                + 0.08 * (1.0 - inputs["disease_stage"])
            ),
            "major_neurocognitive_disorder_profile": self._clip01(
                0.30 * symptoms["rapid_cognitive_decline"]
                + 0.24 * symptoms["executive_language_visuospatial_impairment"]
                + 0.22 * symptoms["daily_function_interference"]
                + 0.12 * symptoms["personality_change_disinhibition"]
                + 0.08 * inputs["disease_stage"]
            ),
            "rapidly_progressive_dementia_profile": self._clip01(
                0.32 * symptoms["rapid_cognitive_decline"]
                + 0.22 * symptoms["daily_function_interference"]
                + 0.18 * symptoms["cortical_ribboning_mri_pattern"]
                + 0.12 * symptoms["deep_gray_mri_pattern"]
                + 0.08 * inputs["codon_129_susceptibility"]
            ),
            "parkinsonian_prion_profile": self._clip01(
                0.40 * symptoms["parkinsonism"]
                + 0.18 * symptoms["apathy_depression"]
                + 0.16 * symptoms["deep_gray_mri_pattern"]
                + 0.10 * inputs["deep_gray_involvement"]
            ),
            "brainstem_affective_sleep_profile": self._clip01(
                0.34 * symptoms["sleep_wake_disturbance"]
                + 0.24 * symptoms["anxiety_behavioral_disturbance"]
                + 0.18 * symptoms["apathy_depression"]
                + 0.12 * inputs["brainstem_involvement"]
            ),
            "cerebellar_prion_profile": self._clip01(
                0.48 * symptoms["cerebellar_ataxia"]
                + 0.18 * symptoms["myoclonus"]
                + 0.14 * inputs["cerebellar_involvement"]
            ),
            "imaging_support_profile": self._clip01(
                0.44 * symptoms["cortical_ribboning_mri_pattern"]
                + 0.34 * symptoms["deep_gray_mri_pattern"]
                + 0.10 * inputs["disease_stage"]
            ),
        }

        return {
            "inputs": pd.Series(inputs, dtype=float),
            "latents": pd.Series(latents, dtype=float),
            "regional_state": pd.Series(regional_state, dtype=float),
            "symptoms": pd.Series(symptoms, dtype=float),
            "phenotypes": pd.Series(phenotypes, dtype=float),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Assign an MNI152 point to the statistical map of the selected parcellation.

        Returns an empty DataFrame-like message when siibra is unavailable or
        assignment fails.
        """
        if not self.siibra_available or siibra is None:
            return pd.DataFrame(
                [{"message": "siibra unavailable; coordinate assignment not possible in this environment."}]
            )

        if self._pmap is None:
            try:
                with self._quiet:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception as exc:
                return pd.DataFrame([{"message": f"Could not load statistical map: {exc!r}"}])

        try:
            point = siibra.Point(tuple(xyz), space=self.assignment_space)
            with self._quiet:
                assignments = self._pmap.assign(point)
            if assignments is None:
                return pd.DataFrame()

            sort_candidates = [
                "map value",
                "correlation",
                "intersection over union",
                "score",
                "probability",
            ]
            for col in sort_candidates:
                if col in assignments.columns:
                    assignments = assignments.sort_values(col, ascending=False)
                    break
            return assignments.reset_index(drop=True)
        except Exception as exc:
            return pd.DataFrame([{"message": f"Point assignment failed: {exc!r}"}])

    def region_mask(self, node_key: str) -> Any:
        """
        Return a regional map / mask-like object when possible.

        Because siibra APIs vary across versions, this helper attempts a few
        compatible access patterns and otherwise returns None.
        """
        region = self.region_objects.get(node_key)
        if region is None or not self.siibra_available:
            return None

        if hasattr(region, "fetch_regional_map"):
            for kwargs in (
                {"space": self.space, "maptype": "labelled"},
                {"space": self.space, "maptype": "statistical"},
                {"space": self.space},
            ):
                try:
                    return region.fetch_regional_map(**kwargs)
                except Exception:
                    continue

        if self._pmap is None:
            try:
                with self._quiet:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.space_spec,
                        maptype="statistical",
                    )
            except Exception:
                self._pmap = None

        if self._pmap is not None:
            try:
                if hasattr(self._pmap, "fetch"):
                    return self._pmap.fetch(region)
            except Exception:
                pass

        return None


if __name__ == "__main__":
    model = PrionDiseaseNeurocognitiveDisorderModel()
    built = model.build(connectivity_rows=10)

    print("\n=== NODE TABLE ===")
    print(
        built["nodes"][["key", "node_type", "atlas_region", "feature_summary"]].to_string(index=False)
    )

    print("\n=== EDGE TABLE (first 14 rows) ===")
    print(built["edges"].head(14).to_string(index=False))

    print("\n=== WITHIN-CIRCUIT CONNECTIVITY ===")
    circuit_df = built["circuit_connectivity"]
    if isinstance(circuit_df, pd.DataFrame) and not circuit_df.empty:
        print(circuit_df.to_string())
    else:
        print("No circuit connectivity matrix available in this environment.")

    print("\n=== AVAILABLE REGION-SPECIFIC TABLES ===")
    for key in ("frontal_cortex_proxy", "temporal_cortex_proxy", "thalamus_proxy"):
        receptor_rows = len(built["receptors"].get(key, pd.DataFrame()))
        gene_rows = len(built["genes"].get(key, pd.DataFrame()))
        conn_rows = len(built["connectivity_profiles"].get(key, pd.DataFrame()))
        print(f"{key}: receptors={receptor_rows}, genes={gene_rows}, connectivity_rows={conn_rows}")

    sim = model.simulate(
        prion_misfolding_load=0.82,
        prnp_mutation_liability=0.30,
        codon_129_susceptibility=0.70,
        disease_stage=0.78,
        cortical_involvement=0.80,
        deep_gray_involvement=0.72,
        brainstem_involvement=0.40,
        cerebellar_involvement=0.36,
    )

    print("\n=== SIMULATION: LATENTS ===")
    print(sim["latents"].sort_values(ascending=False).to_string())

    print("\n=== SIMULATION: SYMPTOMS ===")
    print(sim["symptoms"].sort_values(ascending=False).to_string())

    print("\n=== SIMULATION: PHENOTYPES ===")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Example exploratory helpers in a siibra-enabled environment:
    # print(model.suggest_regions("thalamus").head(10))
    # print(model.assign_mni_point((0, -16, 8)).head(10))
    # mask = model.region_mask("frontal_cortex_proxy")
