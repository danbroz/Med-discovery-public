
from __future__ import annotations

"""
Major or Mild Neurocognitive Disorder Due to Huntington's Disease siibra scaffold.

This script translates a Huntington's disease (HD) neurocognitive-disorder chapter
into a transparent, atlas-grounded research scaffold using siibra. It is a
mechanistic interpretation of chapter logic, not a validated disease model,
diagnostic tool, or treatment recommender.

The source chapter frames the disorder as a paradigmatic monogenic,
progressively neurodegenerative condition caused by a pathogenic CAG-repeat
expansion in HTT. The chapter emphasizes striatal and basal-ganglia pathology,
dopaminergic/GABAergic/glutamatergic imbalance, disruption of the
caudate-prefrontal executive circuit, and later thalamocortical/frontotemporal
network breakdown. To stay conservative, the scaffold anchors only regions that
are explicitly named or strongly implied, and uses proxy region keys where
Julich labels may differ across siibra versions.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except Exception as exc:  # pragma: no cover - depends on user environment
    siibra = None
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on user environment
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_HUNTINGTONS_DISEASE_GENE_PANEL = [
    "HTT",      # causative huntingtin gene
    "BDNF",     # corticostriatal trophic support, commonly discussed in HD biology
    "DRD1",     # dopamine receptor signaling
    "DRD2",     # dopamine receptor signaling
    "SLC6A3",   # dopamine transporter
    "COMT",     # catecholamine metabolism
    "GAD1",     # GABA synthesis
    "GAD2",     # GABA synthesis
    "SLC32A1",  # vesicular GABA transporter
    "GRIN2B",   # glutamatergic signaling
    "SLC1A2",   # glutamate transport
    "GRIA2",    # glutamatergic AMPA signaling
]


class MajorOrMildNeurocognitiveDisorderDueToHuntingtonsDiseaseModel:
    """
    Atlas-grounded research scaffold for neurocognitive disorder due to HD.

    The model follows a one-pass transparent order:
        inputs -> latent biology -> regional dysfunction burden -> symptoms -> phenotypes

    Important caveats
    -----------------
    - Higher `regional_state` values indicate modeled dysfunction burden, not
      healthy activation.
    - `symptomatic_management_support` can modestly reduce modeled symptom burden
      but does not reverse the monogenic or neurodegenerative core process,
      consistent with the chapter's emphasis that current interventions are
      largely symptomatic.
    - This is a chapter-faithful research scaffold, not a clinically validated
      causal model.
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
                "siibra is required for this scaffold. Install siibra-python in the "
                "target environment before instantiating the model."
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
            "caudate_nucleus": [
                "Caudate nucleus left",
                "caudate left",
                "caudate nucleus",
                "caudate",
            ],
            "striatum_proxy": [
                "Putamen left",
                "putamen left",
                "striatum left",
                "putamen",
                "striatum",
                "basal ganglia",
            ],
            "thalamus_proxy": [
                "Thalamus left",
                "thalamus left",
                "thalamus",
                "thalamic nuclei",
            ],
            "pfc_control": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "MFG1 left",
                "MFG2 left",
                "middle frontal gyrus left",
                "dorsolateral prefrontal cortex",
                "prefrontal cortex",
            ],
            "temporal_association_cortex": [
                "Area TE 2.1 (STG) left",
                "Area TGd (Temporal pole) left",
                "Area TGv (Temporal pole) left",
                "superior temporal gyrus left",
                "temporal cortex left",
                "temporal cortex",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "caudate_nucleus": (
                "Caudate-prefrontal executive-circuit node emphasized by the chapter "
                "as a key substrate of dysexecutive decline."
            ),
            "striatum_proxy": (
                "Broader striatal / basal-ganglia proxy capturing the primary site of "
                "HD pathology when exact Julich labels vary."
            ),
            "thalamus_proxy": (
                "Thalamic relay proxy representing downstream hypometabolic or "
                "hypoperfused cognitive-motor relay burden."
            ),
            "pfc_control": (
                "Prefrontal control proxy, using left-sided frontal associative / "
                "dorsolateral territory to model executive dysfunction."
            ),
            "temporal_association_cortex": (
                "Temporal associative cortical proxy representing frontotemporal "
                "network involvement in later-stage cognitive decline."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "pathogenic_htt_burden": (
                "Pathogenic HTT CAG-repeat expansion burden, the core inherited driver "
                "of Huntington's disease biology."
            ),
            "disease_stage_burden": (
                "Progressive stage burden reflecting accumulation of neurodegeneration "
                "and broader network failure over time."
            ),
            "juvenile_rigid_bias": (
                "Bias toward the juvenile akinetic-rigid phenotype rather than the "
                "more typical adult choreic presentation."
            ),
            "cognitive_reserve_support": (
                "Protective reserve, environmental structure, and compensatory support "
                "that can buffer cognitive expression without removing the mutation."
            ),
            "symptomatic_management_support": (
                "Symptomatic pharmacologic and supportive management that can reduce "
                "motor-cognitive interference but is not disease modifying."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "mutant_huntingtin_toxicity": (
                "Toxic effect of polyglutamine-expanded huntingtin protein on neurons."
            ),
            "progressive_striatal_degeneration": (
                "Ongoing basal-ganglia / striatal neurodegeneration central to HD."
            ),
            "glutamate_gaba_imbalance": (
                "Circuit imbalance involving glutamatergic drive and GABAergic loss."
            ),
            "dopaminergic_state_shift": (
                "Stage- and phenotype-sensitive dopaminergic dysregulation linked to "
                "chorea versus akinetic-rigid motor expression."
            ),
            "thalamocortical_disconnection": (
                "Breakdown of thalamic relay integration with cortical cognitive-motor "
                "systems."
            ),
            "frontostriatal_executive_failure": (
                "Disruption of the caudate-prefrontal executive circuit causing "
                "planning, flexibility, and impulse-control deficits."
            ),
            "diffuse_cortical_network_breakdown": (
                "Later-stage cortical and frontotemporal network deterioration "
                "contributing to global dementia."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "executive_dysfunction": (
                "Dysexecutive burden including planning, cognitive flexibility, and "
                "impulse-control problems."
            ),
            "attention_processing_speed_decline": (
                "Attention and processing-speed burden linked to frontothalamic "
                "network dysfunction."
            ),
            "memory_language_decline": (
                "Memory and language burden accompanying later diffuse cortical "
                "decline."
            ),
            "behavioral_disinhibition": (
                "Behavioral regulation and impulse-control disturbance."
            ),
            "movement_disorder_burden": (
                "Motor burden spanning choreiform versus akinetic-rigid expression."
            ),
            "global_neurocognitive_burden": (
                "Overall major/mild neurocognitive impairment burden."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "pathogenic_htt_burden",
                "target": "mutant_huntingtin_toxicity",
                "relation": "CAG-repeat expansion drives toxic mutant huntingtin biology",
                "hd_ncd_change": "increased",
            },
            {
                "source": "pathogenic_htt_burden",
                "target": "progressive_striatal_degeneration",
                "relation": "the causative mutation promotes progressive striatal neurodegeneration",
                "hd_ncd_change": "increased",
            },
            {
                "source": "disease_stage_burden",
                "target": "progressive_striatal_degeneration",
                "relation": "later disease stages accumulate neuronal loss in core basal-ganglia circuits",
                "hd_ncd_change": "increased",
            },
            {
                "source": "disease_stage_burden",
                "target": "diffuse_cortical_network_breakdown",
                "relation": "progression expands pathology beyond striatum into widespread cortical networks",
                "hd_ncd_change": "increased",
            },
            {
                "source": "juvenile_rigid_bias",
                "target": "dopaminergic_state_shift",
                "relation": "juvenile phenotype alters dopamine-state expression toward akinetic-rigid patterns",
                "hd_ncd_change": "increased",
            },
            {
                "source": "cognitive_reserve_support",
                "target": "frontostriatal_executive_failure",
                "relation": "reserve can buffer functional expression of executive impairment",
                "hd_ncd_change": "decreased",
            },
            {
                "source": "cognitive_reserve_support",
                "target": "diffuse_cortical_network_breakdown",
                "relation": "reserve can modestly reduce the expressed impact of cortical burden",
                "hd_ncd_change": "decreased",
            },
            {
                "source": "symptomatic_management_support",
                "target": "dopaminergic_state_shift",
                "relation": "symptomatic therapy can partly modulate dopamine-linked motor expression",
                "hd_ncd_change": "decreased",
            },
            {
                "source": "symptomatic_management_support",
                "target": "movement_disorder_burden",
                "relation": "supportive treatment may reduce motor symptom expression without reversing pathology",
                "hd_ncd_change": "decreased",
            },
            {
                "source": "mutant_huntingtin_toxicity",
                "target": "progressive_striatal_degeneration",
                "relation": "toxic huntingtin promotes ongoing neuronal degeneration in HD-sensitive circuits",
                "hd_ncd_change": "increased",
            },
            {
                "source": "mutant_huntingtin_toxicity",
                "target": "glutamate_gaba_imbalance",
                "relation": "mutant huntingtin disturbs inhibitory-excitatory circuit balance",
                "hd_ncd_change": "increased",
            },
            {
                "source": "mutant_huntingtin_toxicity",
                "target": "diffuse_cortical_network_breakdown",
                "relation": "the toxic protein burden contributes to broader cortical network failure",
                "hd_ncd_change": "increased",
            },
            {
                "source": "progressive_striatal_degeneration",
                "target": "caudate_nucleus",
                "relation": "striatal degeneration heavily involves the caudate nucleus",
                "hd_ncd_change": "increased",
            },
            {
                "source": "progressive_striatal_degeneration",
                "target": "striatum_proxy",
                "relation": "primary pathology is centered on the basal ganglia / striatum",
                "hd_ncd_change": "increased",
            },
            {
                "source": "progressive_striatal_degeneration",
                "target": "thalamocortical_disconnection",
                "relation": "striatal damage propagates through thalamocortical relay loops",
                "hd_ncd_change": "increased",
            },
            {
                "source": "glutamate_gaba_imbalance",
                "target": "dopaminergic_state_shift",
                "relation": "GABAergic, glutamatergic, and dopaminergic systems interact to shape motor-cognitive imbalance",
                "hd_ncd_change": "increased",
            },
            {
                "source": "glutamate_gaba_imbalance",
                "target": "thalamocortical_disconnection",
                "relation": "transmitter imbalance destabilizes relay integration across cognitive-motor loops",
                "hd_ncd_change": "increased",
            },
            {
                "source": "thalamocortical_disconnection",
                "target": "thalamus_proxy",
                "relation": "thalamic relay burden emerges with disrupted subcortical-cortical integration",
                "hd_ncd_change": "increased",
            },
            {
                "source": "thalamocortical_disconnection",
                "target": "pfc_control",
                "relation": "disconnection undermines frontal associative processing",
                "hd_ncd_change": "increased",
            },
            {
                "source": "diffuse_cortical_network_breakdown",
                "target": "pfc_control",
                "relation": "cortical atrophy and dysfunction burden frontal associative cortex",
                "hd_ncd_change": "increased",
            },
            {
                "source": "diffuse_cortical_network_breakdown",
                "target": "temporal_association_cortex",
                "relation": "frontotemporal cortical involvement contributes to later-stage cognitive decline",
                "hd_ncd_change": "increased",
            },
            {
                "source": "caudate_nucleus",
                "target": "frontostriatal_executive_failure",
                "relation": "caudate damage disrupts prefrontal executive loops",
                "hd_ncd_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "frontostriatal_executive_failure",
                "relation": "frontal associative impairment worsens dysexecutive network failure",
                "hd_ncd_change": "increased",
            },
            {
                "source": "frontostriatal_executive_failure",
                "target": "executive_dysfunction",
                "relation": "caudate-prefrontal disruption drives planning and flexibility deficits",
                "hd_ncd_change": "increased",
            },
            {
                "source": "frontostriatal_executive_failure",
                "target": "behavioral_disinhibition",
                "relation": "executive-circuit breakdown reduces impulse control",
                "hd_ncd_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "attention_processing_speed_decline",
                "relation": "thalamic and relay dysfunction slows cognitive processing",
                "hd_ncd_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "executive_dysfunction",
                "relation": "frontal associative dysfunction worsens executive performance",
                "hd_ncd_change": "increased",
            },
            {
                "source": "temporal_association_cortex",
                "target": "memory_language_decline",
                "relation": "temporal cortical involvement contributes to memory and language deterioration",
                "hd_ncd_change": "increased",
            },
            {
                "source": "striatum_proxy",
                "target": "movement_disorder_burden",
                "relation": "striatal degeneration produces major motor manifestations",
                "hd_ncd_change": "increased",
            },
            {
                "source": "dopaminergic_state_shift",
                "target": "movement_disorder_burden",
                "relation": "dopamine-state changes shape choreiform versus rigid motor burden",
                "hd_ncd_change": "increased",
            },
            {
                "source": "executive_dysfunction",
                "target": "global_neurocognitive_burden",
                "relation": "dysexecutive decline contributes centrally to neurocognitive disorder",
                "hd_ncd_change": "increased",
            },
            {
                "source": "attention_processing_speed_decline",
                "target": "global_neurocognitive_burden",
                "relation": "slowed cognition contributes to overall neurocognitive burden",
                "hd_ncd_change": "increased",
            },
            {
                "source": "memory_language_decline",
                "target": "global_neurocognitive_burden",
                "relation": "progressive memory/language decline contributes to major NCD severity",
                "hd_ncd_change": "increased",
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
            tabular = getattr(siibra.features, "tabular", None)
            if tabular is not None:
                if kind == "receptor" and hasattr(tabular, "ReceptorDensityFingerprint"):
                    cands.append(getattr(tabular, "ReceptorDensityFingerprint"))
                elif kind == "gene" and hasattr(tabular, "GeneExpressions"):
                    cands.append(getattr(tabular, "GeneExpressions"))
        except Exception:
            pass

        try:
            molecular = getattr(siibra.features, "molecular", None)
            if molecular is not None:
                if kind == "receptor" and hasattr(molecular, "ReceptorDensityFingerprint"):
                    cands.append(getattr(molecular, "ReceptorDensityFingerprint"))
                elif kind == "gene" and hasattr(molecular, "GeneExpressions"):
                    cands.append(getattr(molecular, "GeneExpressions"))
        except Exception:
            pass

        try:
            connectivity = getattr(siibra.features, "connectivity", None)
            if kind == "connectivity" and connectivity is not None and hasattr(connectivity, "StreamlineCounts"):
                cands.append(getattr(connectivity, "StreamlineCounts"))
        except Exception:
            pass

        if kind == "receptor":
            cands.extend(["receptor density fingerprint", "ReceptorDensityFingerprint"])
        elif kind == "gene":
            cands.extend(["gene expressions", "GeneExpressions"])
        elif kind == "connectivity":
            cands.extend(["StreamlineCounts", "streamline counts"])
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
            "caudate nucleus",
            "caudate",
            "striatum",
            "basal ganglia",
            "thalamus",
            "prefrontal cortex",
            "middle frontal gyrus",
            "temporal cortex",
            "superior temporal gyrus",
        } else 0
        gapmap_penalty = 1 if "gapmap" in name else 0
        return (left_bonus, right_penalty, generic_penalty, gapmap_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation)
            except Exception:
                pass
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation_spec)
            except Exception:
                pass
            try:
                return self.parcellation.get_region(spec)
            except Exception:
                pass
            try:
                return siibra.get_region(self.parcellation_spec, spec)
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
        for space_arg in (self.space, self.space_spec):
            try:
                props = region.spatial_props(space=space_arg)
            except Exception:
                continue
            if props is None:
                continue
            if isinstance(props, dict):
                return list(props.values())
            if isinstance(props, (list, tuple)):
                return list(props)
            if hasattr(props, "components"):
                return list(getattr(props, "components", []))
            return [props]
        return []

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
        try:
            volume_mm3 = float(getattr(main, "volume", float("nan")))
        except Exception:
            volume_mm3 = None
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
                        .reset_index(drop=True)
                    )
                except Exception:
                    return df.reset_index(drop=True)
            return df.reset_index(drop=True)
        return pd.DataFrame()

    @staticmethod
    def _cohort_of(feature: Any) -> str:
        return str(getattr(feature, "cohort", "")).upper().replace(" ", "")

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        wanted = str(self.connectivity_cohort).upper().replace(" ", "")
        compound = next((f for f in feats if self._cohort_of(f) == wanted), feats[0])

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            first = compound[0]
            data = getattr(first, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        exact = [x for x in labels if self._name_of(x) == region.name]
        if exact:
            return exact[0]

        rn = region.name.lower()
        fuzzy = [
            x for x in labels
            if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn
        ]
        if fuzzy:
            return fuzzy[0]

        region_key = next((k for k, v in self.region_objects.items() if v == region), "")
        for candidate in self.region_candidates.get(region_key, []):
            candidate_lower = candidate.lower()
            fuzzy = [x for x in labels if candidate_lower in self._name_of(x).lower()]
            if fuzzy:
                return fuzzy[0]
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
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """Return pairwise structural connectivity among resolved circuit nodes."""
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame(columns=["source", "target", "value"])

        rows: List[Dict[str, Any]] = []
        for src_key, src_region in self.region_objects.items():
            src_label = self._match_region_label(list(matrix.index), src_region)
            axis = "index"
            if src_label is None:
                src_label = self._match_region_label(list(matrix.columns), src_region)
                axis = "columns"
            if src_label is None:
                continue

            try:
                src_series = matrix.loc[src_label] if axis == "index" else matrix[src_label]
            except Exception:
                continue

            for dst_key, dst_region in self.region_objects.items():
                if src_key == dst_key:
                    continue
                dst_label = self._match_region_label(list(src_series.index), dst_region)
                if dst_label is None:
                    continue
                try:
                    value = float(src_series.loc[dst_label])
                except Exception:
                    continue
                rows.append({"source": src_key, "target": dst_key, "value": value})

        if not rows:
            return pd.DataFrame(columns=["source", "target", "value"])
        return (
            pd.DataFrame(rows)
            .sort_values(["source", "value"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_HUNTINGTONS_DISEASE_GENE_PANEL,
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
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region",
                        "description": self.region_descriptions.get(key, "Atlas-backed region or proxy node."),
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
                    "description": self.region_descriptions.get(key, "Atlas-backed region or proxy node."),
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

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """Probabilistically assign an MNI coordinate to Julich regions."""
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
                        space=self.assignment_space,
                        maptype="statistical",
                    )

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Any:
        """Return a best-effort statistical regional map or mask for a resolved node."""
        region = self.region_objects.get(node_key)
        if region is None:
            return None

        for space_arg in (self.assignment_space, self.space, self.space_spec):
            try:
                return region.get_regional_map(space_arg, "statistical")
            except Exception:
                pass
            try:
                return region.get_regional_map(space=space_arg, maptype="statistical")
            except Exception:
                pass
            try:
                return region.get_regional_mask(space_arg)
            except Exception:
                pass
        return None

    def simulate(
        self,
        pathogenic_htt_burden: float = 0.95,
        disease_stage_burden: float = 0.60,
        juvenile_rigid_bias: float = 0.20,
        cognitive_reserve_support: float = 0.25,
        symptomatic_management_support: float = 0.30,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized HD neurocognitive-disorder simulation.

        Parameters are 0..1 normalized knobs. Higher symptom values indicate greater
        modeled burden.
        """
        inputs = pd.Series(
            {
                "pathogenic_htt_burden": self._clip01(pathogenic_htt_burden),
                "disease_stage_burden": self._clip01(disease_stage_burden),
                "juvenile_rigid_bias": self._clip01(juvenile_rigid_bias),
                "cognitive_reserve_support": self._clip01(cognitive_reserve_support),
                "symptomatic_management_support": self._clip01(symptomatic_management_support),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["mutant_huntingtin_toxicity"] = self._clip01(
            0.72 * inputs["pathogenic_htt_burden"]
            + 0.16 * inputs["disease_stage_burden"]
        )
        latents["progressive_striatal_degeneration"] = self._clip01(
            0.42 * latents["mutant_huntingtin_toxicity"]
            + 0.30 * inputs["disease_stage_burden"]
            + 0.12 * inputs["pathogenic_htt_burden"]
        )
        latents["glutamate_gaba_imbalance"] = self._clip01(
            0.36 * latents["progressive_striatal_degeneration"]
            + 0.22 * latents["mutant_huntingtin_toxicity"]
            + 0.12 * inputs["disease_stage_burden"]
        )
        latents["dopaminergic_state_shift"] = self._clip01(
            0.28 * latents["progressive_striatal_degeneration"]
            + 0.20 * latents["glutamate_gaba_imbalance"]
            + 0.18 * inputs["disease_stage_burden"]
            + 0.12 * inputs["juvenile_rigid_bias"]
            - 0.10 * inputs["symptomatic_management_support"]
        )
        latents["thalamocortical_disconnection"] = self._clip01(
            0.30 * latents["progressive_striatal_degeneration"]
            + 0.22 * latents["glutamate_gaba_imbalance"]
            + 0.16 * inputs["disease_stage_burden"]
        )
        latents["diffuse_cortical_network_breakdown"] = self._clip01(
            0.30 * inputs["disease_stage_burden"]
            + 0.24 * latents["mutant_huntingtin_toxicity"]
            + 0.18 * latents["progressive_striatal_degeneration"]
            - 0.14 * inputs["cognitive_reserve_support"]
        )
        latents["frontostriatal_executive_failure"] = self._clip01(
            0.34 * latents["progressive_striatal_degeneration"]
            + 0.24 * latents["thalamocortical_disconnection"]
            + 0.18 * latents["diffuse_cortical_network_breakdown"]
            - 0.18 * inputs["cognitive_reserve_support"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["caudate_nucleus"] = self._clip01(
            0.52 * latents["progressive_striatal_degeneration"]
            + 0.18 * latents["glutamate_gaba_imbalance"]
            + 0.10 * latents["mutant_huntingtin_toxicity"]
        )
        regional_state["striatum_proxy"] = self._clip01(
            0.44 * latents["progressive_striatal_degeneration"]
            + 0.26 * latents["glutamate_gaba_imbalance"]
            + 0.16 * latents["dopaminergic_state_shift"]
            - 0.10 * inputs["symptomatic_management_support"]
        )
        regional_state["thalamus_proxy"] = self._clip01(
            0.46 * latents["thalamocortical_disconnection"]
            + 0.18 * latents["progressive_striatal_degeneration"]
            + 0.10 * latents["diffuse_cortical_network_breakdown"]
        )
        regional_state["pfc_control"] = self._clip01(
            0.44 * latents["frontostriatal_executive_failure"]
            + 0.18 * latents["diffuse_cortical_network_breakdown"]
            + 0.10 * latents["thalamocortical_disconnection"]
            - 0.16 * inputs["cognitive_reserve_support"]
        )
        regional_state["temporal_association_cortex"] = self._clip01(
            0.40 * latents["diffuse_cortical_network_breakdown"]
            + 0.14 * inputs["disease_stage_burden"]
            + 0.12 * latents["thalamocortical_disconnection"]
            - 0.12 * inputs["cognitive_reserve_support"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["executive_dysfunction"] = self._clip01(
            0.40 * regional_state["caudate_nucleus"]
            + 0.28 * regional_state["pfc_control"]
            + 0.18 * latents["frontostriatal_executive_failure"]
        )
        symptoms["attention_processing_speed_decline"] = self._clip01(
            0.34 * regional_state["thalamus_proxy"]
            + 0.24 * regional_state["pfc_control"]
            + 0.20 * latents["thalamocortical_disconnection"]
        )
        symptoms["memory_language_decline"] = self._clip01(
            0.32 * regional_state["temporal_association_cortex"]
            + 0.26 * latents["diffuse_cortical_network_breakdown"]
            + 0.12 * inputs["disease_stage_burden"]
        )
        symptoms["behavioral_disinhibition"] = self._clip01(
            0.34 * regional_state["caudate_nucleus"]
            + 0.24 * regional_state["pfc_control"]
            + 0.16 * latents["dopaminergic_state_shift"]
        )
        symptoms["movement_disorder_burden"] = self._clip01(
            0.34 * regional_state["striatum_proxy"]
            + 0.24 * latents["dopaminergic_state_shift"]
            + 0.20 * latents["glutamate_gaba_imbalance"]
            - 0.12 * inputs["symptomatic_management_support"]
        )
        symptoms["global_neurocognitive_burden"] = self._clip01(
            0.28 * symptoms["executive_dysfunction"]
            + 0.22 * symptoms["attention_processing_speed_decline"]
            + 0.22 * symptoms["memory_language_decline"]
            + 0.10 * symptoms["behavioral_disinhibition"]
            + 0.08 * inputs["disease_stage_burden"]
            - 0.10 * inputs["cognitive_reserve_support"]
        )

        adult_choreic_pressure = self._clip01(
            latents["dopaminergic_state_shift"] * (1.0 - inputs["juvenile_rigid_bias"])
        )
        juvenile_rigid_pressure = self._clip01(
            latents["dopaminergic_state_shift"] * inputs["juvenile_rigid_bias"]
        )

        phenotypes = pd.Series(dtype=float, name="phenotypes")
        phenotypes["dysexecutive_mild_ncd_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        symptoms["executive_dysfunction"],
                        symptoms["attention_processing_speed_decline"],
                        regional_state["caudate_nucleus"],
                        regional_state["pfc_control"],
                    ]
                ).mean()
            )
        )
        phenotypes["advanced_major_ncd_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        symptoms["global_neurocognitive_burden"],
                        symptoms["memory_language_decline"],
                        latents["diffuse_cortical_network_breakdown"],
                        regional_state["temporal_association_cortex"],
                    ]
                ).mean()
            )
        )
        phenotypes["corticostriatal_disconnect_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        regional_state["caudate_nucleus"],
                        regional_state["pfc_control"],
                        regional_state["thalamus_proxy"],
                        latents["frontostriatal_executive_failure"],
                    ]
                ).mean()
            )
        )
        phenotypes["adult_choreiform_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        symptoms["movement_disorder_burden"],
                        adult_choreic_pressure,
                        regional_state["striatum_proxy"],
                    ]
                ).mean()
            )
        )
        phenotypes["juvenile_akinetic_rigid_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        symptoms["movement_disorder_burden"],
                        juvenile_rigid_pressure,
                        latents["progressive_striatal_degeneration"],
                    ]
                ).mean()
            )
        )

        return {
            "inputs": inputs,
            "latents": latents.round(4),
            "regional_state": regional_state.round(4),
            "symptoms": symptoms.round(4),
            "phenotypes": phenotypes.round(4),
        }


if __name__ == "__main__":  # pragma: no cover - example usage
    try:
        model = MajorOrMildNeurocognitiveDisorderDueToHuntingtonsDiseaseModel()
        scaffold = model.build()

        print("\n=== Nodes ===")
        print(scaffold["nodes"][["key", "node_type", "atlas_region", "feature_summary"]].to_string(index=False))

        print("\n=== Edges (first 15) ===")
        print(scaffold["edges"].head(15).to_string(index=False))

        print("\n=== Circuit connectivity ===")
        circuit_conn = scaffold["circuit_connectivity"]
        if circuit_conn.empty:
            print("No circuit connectivity matrix available in this environment.")
        else:
            print(circuit_conn.head(15).to_string(index=False))

        for node_key in ("caudate_nucleus", "pfc_control", "thalamus_proxy"):
            print(f"\n=== {node_key} receptor fingerprint ===")
            rec = scaffold["receptors"].get(node_key, pd.DataFrame())
            print(rec.head(10).to_string(index=False) if not rec.empty else "No receptor data available.")

            print(f"\n=== {node_key} gene expression summary ===")
            genes = scaffold["genes"].get(node_key, pd.DataFrame())
            print(genes.head(10).to_string(index=False) if not genes.empty else "No gene-expression data available.")

            print(f"\n=== {node_key} connectivity profile ===")
            conn = scaffold["connectivity_profiles"].get(node_key, pd.DataFrame())
            print(conn.head(10).to_string(index=False) if not conn.empty else "No connectivity profile available.")

        print("\n=== Simulation example ===")
        sim = model.simulate(
            pathogenic_htt_burden=0.98,
            disease_stage_burden=0.65,
            juvenile_rigid_bias=0.15,
            cognitive_reserve_support=0.30,
            symptomatic_management_support=0.35,
        )
        for key, series in sim.items():
            print(f"\n[{key}]")
            print(series.to_string())

        # Example coordinate assignment:
        # print(model.assign_mni_point((-12, 12, 8)).head())
    except Exception as exc:
        print(
            "Example execution could not be completed in this environment. "
            f"Reason: {exc}"
        )
