from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Major or Mild Neurocognitive Disorder Due to
Traumatic Brain Injury (TBI).

This script translates a chapter-level biological summary into a transparent
research scaffold. It is not a diagnostic or treatment tool. The model keeps a
simple, readable causal flow:

    inputs -> latent biology -> regional state -> symptoms -> phenotype summaries

The chapter motivating this scaffold emphasizes:
- heterogeneous focal and diffuse injury burden,
- microstructural axonal injury that may be missed on conventional CT or MRI,
- monoaminergic disruption affecting mood, anxiety, motivation, and sleep,
- hypothalamic histaminergic injury linked to hypersomnia,
- interaction between TBI and genetic vulnerability, especially APOE-related
  neurodegenerative risk,
- post-traumatic epileptogenesis,
- diaschisis and large-scale network disconnection involving default-mode and
  fronto-parietal executive systems.

Conservative atlas choices:
- frontal and parietal network burden are anchored to Julich cortical regions,
- basal ganglia, brainstem monoaminergic nuclei, hypothalamus, diffuse axonal
  injury, and network-level DMN dysfunction remain proxy or latent constructs
  because the source text is systems-level rather than cytoarchitectonically
  precise.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


DEFAULT_GENE_PANEL = [
    "APOE",   # strongest chapter-linked risk signal for poorer outcome / dementia risk
    "MAPT",   # tau-related chronic neurodegenerative vulnerability / CTE context
    "BDNF",   # plasticity and recovery capacity
    "COMT",   # frontal dopamine regulation
    "DRD2",   # dopaminergic frontostriatal signaling
    "SLC6A3", # dopamine transporter
    "SLC6A4", # serotonin transporter
    "DBH",    # norepinephrine synthesis
    "HDC",    # histamine synthesis / tuberomammillary context
    "SCN1A",  # excitability / epileptogenesis context
]


class TraumaticBrainInjuryNeurocognitiveDisorderModel:
    """
    Research scaffold for Major or Mild Neurocognitive Disorder Due to TBI.

    This model interprets the chapter as a mechanistic graph centered on:
    - diffuse and focal injury burden,
    - microstructural white-matter disconnection,
    - monoaminergic and histaminergic dysregulation,
    - diaschisis and large-scale network disruption,
    - downstream cognitive, affective, motivational, sleep, and seizure-related
      consequences.

    The scaffold is intentionally conservative. It uses atlas-backed cortical
    anchors where the chapter clearly supports them and keeps brainstem,
    hypothalamic, basal-ganglia, and diffuse network phenomena as latent or
    proxy constructs.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
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

        # Conservative cortical anchors reflecting chapter-level frontal and
        # fronto-parietal executive network burden. Brainstem, hypothalamus,
        # basal ganglia, and whole-network constructs remain latent/proxy.
        self.region_candidates: Dict[str, List[str]] = {
            "pfc_control": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 9/46v left",
                "middle frontal gyrus",
                "prefrontal cortex",
            ],
            "inferior_parietal": [
                "PGa left",
                "PGp left",
                "PFm left",
                "inferior parietal",
                "parietal cortex",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "injury_severity": "Overall burden of traumatic injury, spanning mild concussion through more severe traumatic insult.",
            "diffuse_axonal_injury_load": "Microscopic axonal and white-matter tract damage consistent with diffuse axonal injury.",
            "focal_circuit_lesion_load": "Focal lesion burden affecting specific cortical or subcortical circuits and their downstream projections.",
            "brainstem_projection_injury": "Damage to brainstem monoaminergic projection systems affecting dopamine, norepinephrine, and serotonin signaling.",
            "hypothalamic_arousal_injury": "Damage to histaminergic tuberomammillary and related hypothalamic wake-promoting systems.",
            "repetitive_tbi_exposure": "Repeated TBI or concussion exposure that can amplify chronic network and neurodegenerative burden.",
            "apoe_neurodegenerative_risk": "APOE-related and broader genetic vulnerability for poorer outcome and late-life neurodegeneration.",
            "epileptogenic_vulnerability": "Predisposition affecting neuronal excitability and risk for post-traumatic epilepsy.",
            "neuropsychiatric_predisposition": "Pre-existing vulnerability that TBI may unmask, including latent mood, psychotic, or dementing liability.",
            "cognitive_reserve": "Premorbid reserve that buffers functional impairment despite injury burden.",
            "rehabilitation_support": "Protective rehabilitative and recovery support that can reduce downstream disability.",
        }

        self.latent_nodes: Dict[str, str] = {
            "axonal_microstructural_disconnection": "Microscopic axonal injury producing distributed communication failure even when conventional imaging looks normal.",
            "diaschisis_remote_suppression": "Remote functional depression in connected regions after focal injury and deafferentation.",
            "monoaminergic_dysregulation": "Dopamine, norepinephrine, and serotonin disruption contributing to cognitive and neuropsychiatric sequelae.",
            "histaminergic_arousal_loss": "Reduced hypothalamic histaminergic wake drive linked to post-traumatic hypersomnia.",
            "frontostriatal_motivation_disruption": "Dopaminergic and circuit-level disruption of frontal-basal-ganglia motivational control.",
            "default_mode_network_disruption": "Disrupted functional integration within the default mode network after TBI.",
            "frontoparietal_control_network_disruption": "Disrupted executive control network connectivity supporting attention, processing speed, and executive function.",
            "sleep_wake_network_instability": "Combined wake-promoting and arousal-system disturbance after injury.",
            "neurotransmitter_switching_plasticity": "Post-injury circuit reconfiguration where chronic stimulation alters transmitter identity or polarity.",
            "tau_neurodegenerative_acceleration": "Second-hit acceleration of progressive neurodegenerative burden after TBI, especially with repetitive injury and APOE-related risk.",
            "post_traumatic_epileptogenesis": "Injury-driven shift toward seizure susceptibility shaped by excitability and plasticity vulnerabilities.",
        }

        self.symptom_nodes: Dict[str, str] = {
            "executive_dysfunction": "Impairment of planning, inhibition, set-shifting, and higher-order control.",
            "attention_dyscontrol": "Instability of sustained and selective attention due to network disconnection.",
            "processing_speed_reduction": "Slowed integration and cognitive throughput associated with diffuse white-matter injury.",
            "memory_impairment": "Reduced memory performance from distributed network dysfunction and possible progressive neurodegenerative processes.",
            "depression_anxiety_burden": "Post-injury depression and anxiety related to monoaminergic and network dysregulation.",
            "apathy_motivation_loss": "Reduced initiative and motivational drive linked to frontostriatal dysfunction.",
            "hypersomnia_sleep_wake_disturbance": "Excessive sleepiness or disturbed arousal after TBI, especially with hypothalamic injury.",
            "personality_change_disinhibition": "Behavioral and personality changes after frontal and network-level disruption.",
            "seizure_susceptibility": "Clinical liability toward post-traumatic seizure expression.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "injury_severity",
                "target": "axonal_microstructural_disconnection",
                "relation": "increases diffuse communication failure and microscopic axonal burden",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "diffuse_axonal_injury_load",
                "target": "axonal_microstructural_disconnection",
                "relation": "directly loads distributed white-matter disconnection",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "focal_circuit_lesion_load",
                "target": "diaschisis_remote_suppression",
                "relation": "causes remote functional depression through deafferentation of connected regions",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "brainstem_projection_injury",
                "target": "monoaminergic_dysregulation",
                "relation": "disrupts dopamine, norepinephrine, and serotonin projection systems",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "hypothalamic_arousal_injury",
                "target": "histaminergic_arousal_loss",
                "relation": "reduces wake-promoting histaminergic drive",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "repetitive_tbi_exposure",
                "target": "tau_neurodegenerative_acceleration",
                "relation": "raises chronic tau-related neurodegenerative risk after repeated injury",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "apoe_neurodegenerative_risk",
                "target": "tau_neurodegenerative_acceleration",
                "relation": "increases poorer long-term outcome and dementia vulnerability after TBI",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "epileptogenic_vulnerability",
                "target": "post_traumatic_epileptogenesis",
                "relation": "amplifies seizure risk from injury-driven excitability changes",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "repetitive_tbi_exposure",
                "target": "neurotransmitter_switching_plasticity",
                "relation": "chronic stimulation may alter circuit chemistry and polarity over time",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "neuropsychiatric_predisposition",
                "target": "depression_anxiety_burden",
                "relation": "can be unmasked or amplified by traumatic injury",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "axonal_microstructural_disconnection",
                "target": "default_mode_network_disruption",
                "relation": "reduces coherent communication across the default mode network",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "axonal_microstructural_disconnection",
                "target": "frontoparietal_control_network_disruption",
                "relation": "reduces executive control network integration",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "diaschisis_remote_suppression",
                "target": "default_mode_network_disruption",
                "relation": "extends focal damage into connected large-scale networks",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "diaschisis_remote_suppression",
                "target": "frontoparietal_control_network_disruption",
                "relation": "depresses remote executive-control nodes after focal lesions",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "monoaminergic_dysregulation",
                "target": "frontostriatal_motivation_disruption",
                "relation": "impairs dopaminergic motivational and executive circuitry",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "monoaminergic_dysregulation",
                "target": "sleep_wake_network_instability",
                "relation": "contributes to arousal and sleep disturbances after injury",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "histaminergic_arousal_loss",
                "target": "sleep_wake_network_instability",
                "relation": "reduces stable wakefulness and promotes hypersomnia",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "frontostriatal_motivation_disruption",
                "target": "pfc_control",
                "relation": "reduces frontal control and motivated goal-directed regulation",
                "tbi_ncd_change": "dysregulated",
            },
            {
                "source": "frontoparietal_control_network_disruption",
                "target": "pfc_control",
                "relation": "degrades frontal executive control within the executive network",
                "tbi_ncd_change": "dysregulated",
            },
            {
                "source": "frontoparietal_control_network_disruption",
                "target": "inferior_parietal",
                "relation": "degrades parietal executive-network participation",
                "tbi_ncd_change": "dysregulated",
            },
            {
                "source": "pfc_control",
                "target": "executive_dysfunction",
                "relation": "frontal control failure impairs planning, inhibition, and set-shifting",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "inferior_parietal",
                "target": "attention_dyscontrol",
                "relation": "parietal executive-network dysfunction impairs attention allocation",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "inferior_parietal",
                "target": "processing_speed_reduction",
                "relation": "network inefficiency slows distributed information integration",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "default_mode_network_disruption",
                "target": "memory_impairment",
                "relation": "disrupts broad cognitive integration supporting memory performance",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "monoaminergic_dysregulation",
                "target": "depression_anxiety_burden",
                "relation": "contributes to post-injury mood and anxiety syndromes",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "frontostriatal_motivation_disruption",
                "target": "apathy_motivation_loss",
                "relation": "reduces motivated behavior and initiative",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "sleep_wake_network_instability",
                "target": "hypersomnia_sleep_wake_disturbance",
                "relation": "produces clinically evident hypersomnia and arousal instability",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "neurotransmitter_switching_plasticity",
                "target": "personality_change_disinhibition",
                "relation": "may contribute to long-term behavioral change through altered circuit polarity",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "tau_neurodegenerative_acceleration",
                "target": "memory_impairment",
                "relation": "adds progressive neurodegenerative pressure to cognitive decline",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "post_traumatic_epileptogenesis",
                "target": "seizure_susceptibility",
                "relation": "raises the probability of post-traumatic seizure expression",
                "tbi_ncd_change": "increased",
            },
            {
                "source": "cognitive_reserve",
                "target": "frontoparietal_control_network_disruption",
                "relation": "buffers manifest disability despite injury",
                "tbi_ncd_change": "decreased",
            },
            {
                "source": "rehabilitation_support",
                "target": "frontoparietal_control_network_disruption",
                "relation": "supports recovery and partial functional compensation",
                "tbi_ncd_change": "decreased",
            },
            {
                "source": "rehabilitation_support",
                "target": "hypersomnia_sleep_wake_disturbance",
                "relation": "can reduce downstream disability from sleep-wake disruption",
                "tbi_ncd_change": "decreased",
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
        candidates: List[Any] = []
        try:
            if kind == "receptor":
                candidates.append(siibra.features.molecular.ReceptorDensityFingerprint)
            elif kind == "gene":
                candidates.append(siibra.features.molecular.GeneExpressions)
            elif kind == "connectivity":
                candidates.append(siibra.features.connectivity.StreamlineCounts)
        except Exception:
            pass

        if kind == "receptor":
            candidates.append("receptor density fingerprint")
        elif kind == "gene":
            candidates.append("gene expressions")
        elif kind == "connectivity":
            candidates.append("StreamlineCounts")
        return candidates

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
        proxy_penalty = 1 if any(term in name for term in ["cortex", "brain", "region"]) else 0
        generic_penalty = 1 if name in {"prefrontal cortex", "parietal cortex", "inferior parietal"} else 0
        return (left_bonus, right_penalty, proxy_penalty, generic_penalty)

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
        main = max(props, key=lambda item: getattr(item, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        volume_mm3 = getattr(main, "volume", None)
        volume_value = float(volume_mm3) if volume_mm3 is not None else None
        return centroid_xyz, volume_value

    def _to_dataframe(self, obj: Any) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        if obj is None:
            return pd.DataFrame()
        if hasattr(obj, "to_dataframe"):
            try:
                return obj.to_dataframe().copy()
            except Exception:
                pass
        if hasattr(obj, "data"):
            data = getattr(obj, "data")
            if isinstance(data, pd.DataFrame):
                return data.copy()
            if hasattr(data, "to_dataframe"):
                try:
                    return data.to_dataframe().copy()
                except Exception:
                    pass
            try:
                return pd.DataFrame(data)
            except Exception:
                return pd.DataFrame()
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame()

    def _numeric_scalar(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, pd.DataFrame):
            if value.empty:
                return None
            flattened = pd.to_numeric(pd.Series(value.to_numpy().ravel()), errors="coerce").dropna()
            return float(flattened.mean()) if not flattened.empty else None
        if isinstance(value, pd.Series):
            if value.empty:
                return None
            flattened = pd.to_numeric(value, errors="coerce").dropna()
            return float(flattened.mean()) if not flattened.empty else None
        try:
            return float(value)
        except Exception:
            try:
                flattened = pd.to_numeric(pd.Series(list(value)), errors="coerce").dropna()
                return float(flattened.mean()) if not flattened.empty else None
            except Exception:
                return None

    def _normalize_connectivity_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(matrix, pd.DataFrame) or matrix.empty:
            return pd.DataFrame()

        df = matrix.copy()
        try:
            df.index = [self._name_of(label) for label in df.index]
        except Exception:
            pass
        try:
            df.columns = [self._name_of(label) for label in df.columns]
        except Exception:
            pass

        try:
            mapper = getattr(df, "map", None)
            df = mapper(self._numeric_scalar) if callable(mapper) else df.applymap(self._numeric_scalar)
        except Exception:
            pass
        try:
            df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
        except Exception:
            return pd.DataFrame()

        try:
            if not df.index.is_unique:
                df = df.groupby(level=0, sort=False).mean()
        except Exception:
            pass
        try:
            if not df.columns.is_unique:
                df = df.T.groupby(level=0, sort=False).mean().T
        except Exception:
            pass
        return df

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        df = self._to_dataframe(feats[0])
        if df.empty:
            return df
        df = df.reset_index()
        if "index" in df.columns and "receptor" not in df.columns:
            df = df.rename(columns={"index": "receptor"})
        return df

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        for feat in feats:
            df = self._to_dataframe(feat)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        lower_cols = {str(col).lower(): col for col in df.columns}
        required = {"gene", "level", "zscore"}
        if required.issubset(lower_cols):
            gene_col = lower_cols["gene"]
            level_col = lower_cols["level"]
            zscore_col = lower_cols["zscore"]
            grouped = (
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
            return grouped
        return df.reset_index(drop=True)

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next((feat for feat in feats if getattr(feat, "cohort", None) == self.connectivity_cohort), feats[0])

        def _accept(candidate: Any) -> Optional[pd.DataFrame]:
            if isinstance(candidate, pd.DataFrame):
                df = self._normalize_connectivity_matrix(candidate)
                return df if not df.empty else None
            return None

        for candidate in (compound, getattr(compound, "data", None)):
            accepted = _accept(candidate)
            if accepted is not None:
                self._connectivity_matrix = accepted
                return self._connectivity_matrix

        try:
            accepted = _accept(compound[0].data)
            if accepted is not None:
                self._connectivity_matrix = accepted
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            for item in compound:
                accepted = _accept(getattr(item, "data", None))
                if accepted is not None:
                    self._connectivity_matrix = accepted
                    return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        exact = [label for label in labels if self._name_of(label) == region.name]
        if exact:
            return exact[0]

        region_name = region.name.lower()
        fuzzy = [
            label
            for label in labels
            if region_name in self._name_of(label).lower() or self._name_of(label).lower() in region_name
        ]
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
            if isinstance(series, pd.DataFrame):
                series = series.mean(axis=0 if axis == "index" else 1, numeric_only=True)
            if not isinstance(series, pd.Series):
                series = pd.Series(series)
            series = pd.to_numeric(series, errors="coerce").dropna()
            if series.empty:
                return pd.DataFrame()
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame(columns=["source", "target", "value"])

        labels: Dict[str, Any] = {}
        for key, region in self.region_objects.items():
            label = self._match_region_label(list(matrix.index), region)
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
            if label is not None:
                labels[key] = label

        rows: List[Dict[str, Any]] = []
        for source_key, source_label in labels.items():
            for target_key, target_label in labels.items():
                if source_key == target_key:
                    continue
                value = None
                try:
                    value = matrix.loc[source_label, target_label]
                except Exception:
                    try:
                        value = matrix[target_label].loc[source_label]
                    except Exception:
                        value = None
                value_scalar = self._numeric_scalar(value)
                if value_scalar is None:
                    continue
                rows.append(
                    {
                        "source": source_key,
                        "target": target_key,
                        "value": value_scalar,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["source", "target", "value"])
        return pd.DataFrame(rows).sort_values(["source", "value"], ascending=[True, False]).reset_index(drop=True)

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

        for key, description in self.input_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "input",
                    "description": description,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            if key == "pfc_control":
                region_description = "Conservative Julich proxy for frontal executive-control burden described in the chapter."
            elif key == "inferior_parietal":
                region_description = "Conservative Julich anchor for parietal participation in the fronto-parietal executive network."
            else:
                region_description = "Atlas-backed circuit node."

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
                        "description": f"{region_description} Unresolved in this siibra environment.",
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
                    "description": region_description,
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

        for key, description in self.latent_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "latent_biology",
                    "description": description,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, description in self.symptom_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "symptom",
                    "description": description,
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
        injury_severity: float = 0.0,
        diffuse_axonal_injury_load: float = 0.0,
        focal_circuit_lesion_load: float = 0.0,
        brainstem_projection_injury: float = 0.0,
        hypothalamic_arousal_injury: float = 0.0,
        repetitive_tbi_exposure: float = 0.0,
        apoe_neurodegenerative_risk: float = 0.0,
        epileptogenic_vulnerability: float = 0.0,
        neuropsychiatric_predisposition: float = 0.0,
        cognitive_reserve: float = 0.0,
        rehabilitation_support: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        Simulate the chapter's logic with normalized inputs in [0, 1].

        Larger values mean more of that factor. Protective variables
        (``cognitive_reserve`` and ``rehabilitation_support``) reduce downstream
        burden where appropriate.
        """
        inputs = {
            "injury_severity": self._clip01(injury_severity),
            "diffuse_axonal_injury_load": self._clip01(diffuse_axonal_injury_load),
            "focal_circuit_lesion_load": self._clip01(focal_circuit_lesion_load),
            "brainstem_projection_injury": self._clip01(brainstem_projection_injury),
            "hypothalamic_arousal_injury": self._clip01(hypothalamic_arousal_injury),
            "repetitive_tbi_exposure": self._clip01(repetitive_tbi_exposure),
            "apoe_neurodegenerative_risk": self._clip01(apoe_neurodegenerative_risk),
            "epileptogenic_vulnerability": self._clip01(epileptogenic_vulnerability),
            "neuropsychiatric_predisposition": self._clip01(neuropsychiatric_predisposition),
            "cognitive_reserve": self._clip01(cognitive_reserve),
            "rehabilitation_support": self._clip01(rehabilitation_support),
        }

        protective_buffer = self._clip01(
            0.55 * inputs["cognitive_reserve"] + 0.45 * inputs["rehabilitation_support"]
        )

        latents = {
            "axonal_microstructural_disconnection": self._clip01(
                0.36 * inputs["injury_severity"]
                + 0.34 * inputs["diffuse_axonal_injury_load"]
                + 0.14 * inputs["repetitive_tbi_exposure"]
                + 0.10 * inputs["focal_circuit_lesion_load"]
                - 0.12 * inputs["rehabilitation_support"]
            ),
            "diaschisis_remote_suppression": self._clip01(
                0.40 * inputs["focal_circuit_lesion_load"]
                + 0.26 * inputs["injury_severity"]
                + 0.18 * inputs["diffuse_axonal_injury_load"]
                - 0.10 * inputs["cognitive_reserve"]
                - 0.06 * inputs["rehabilitation_support"]
            ),
            "monoaminergic_dysregulation": self._clip01(
                0.38 * inputs["brainstem_projection_injury"]
                + 0.18 * inputs["focal_circuit_lesion_load"]
                + 0.14 * inputs["diffuse_axonal_injury_load"]
                + 0.12 * inputs["injury_severity"]
                + 0.08 * inputs["repetitive_tbi_exposure"]
                - 0.10 * inputs["rehabilitation_support"]
            ),
            "histaminergic_arousal_loss": self._clip01(
                0.50 * inputs["hypothalamic_arousal_injury"]
                + 0.18 * inputs["injury_severity"]
                + 0.12 * inputs["brainstem_projection_injury"]
                - 0.10 * inputs["rehabilitation_support"]
            ),
        }

        latents["neurotransmitter_switching_plasticity"] = self._clip01(
            0.24 * inputs["repetitive_tbi_exposure"]
            + 0.20 * inputs["injury_severity"]
            + 0.18 * latents["monoaminergic_dysregulation"]
            + 0.10 * inputs["brainstem_projection_injury"]
            - 0.08 * inputs["rehabilitation_support"]
        )

        latents["frontostriatal_motivation_disruption"] = self._clip01(
            0.34 * latents["monoaminergic_dysregulation"]
            + 0.24 * inputs["focal_circuit_lesion_load"]
            + 0.18 * latents["axonal_microstructural_disconnection"]
            + 0.10 * latents["neurotransmitter_switching_plasticity"]
            - 0.10 * inputs["cognitive_reserve"]
            - 0.08 * inputs["rehabilitation_support"]
        )

        latents["default_mode_network_disruption"] = self._clip01(
            0.34 * latents["axonal_microstructural_disconnection"]
            + 0.22 * latents["diaschisis_remote_suppression"]
            + 0.12 * inputs["injury_severity"]
            + 0.10 * inputs["repetitive_tbi_exposure"]
            - 0.10 * inputs["cognitive_reserve"]
            - 0.08 * inputs["rehabilitation_support"]
        )

        latents["frontoparietal_control_network_disruption"] = self._clip01(
            0.34 * latents["axonal_microstructural_disconnection"]
            + 0.26 * latents["diaschisis_remote_suppression"]
            + 0.18 * latents["frontostriatal_motivation_disruption"]
            + 0.10 * inputs["injury_severity"]
            - 0.10 * inputs["cognitive_reserve"]
            - 0.10 * inputs["rehabilitation_support"]
        )

        latents["sleep_wake_network_instability"] = self._clip01(
            0.42 * latents["histaminergic_arousal_loss"]
            + 0.18 * latents["monoaminergic_dysregulation"]
            + 0.10 * latents["axonal_microstructural_disconnection"]
            + 0.08 * inputs["injury_severity"]
            - 0.10 * inputs["rehabilitation_support"]
        )

        latents["tau_neurodegenerative_acceleration"] = self._clip01(
            0.28 * inputs["repetitive_tbi_exposure"]
            + 0.24 * inputs["apoe_neurodegenerative_risk"]
            + 0.16 * inputs["injury_severity"]
            + 0.14 * latents["axonal_microstructural_disconnection"]
            + 0.10 * inputs["neuropsychiatric_predisposition"]
            - 0.08 * inputs["cognitive_reserve"]
        )

        latents["post_traumatic_epileptogenesis"] = self._clip01(
            0.30 * inputs["injury_severity"]
            + 0.24 * inputs["epileptogenic_vulnerability"]
            + 0.20 * latents["axonal_microstructural_disconnection"]
            + 0.10 * inputs["repetitive_tbi_exposure"]
            + 0.08 * inputs["focal_circuit_lesion_load"]
            - 0.08 * inputs["rehabilitation_support"]
        )

        regional_state = {
            # Interpreted as dysfunction burden rather than simple activation.
            "pfc_control": self._clip01(
                0.42 * latents["frontoparietal_control_network_disruption"]
                + 0.24 * latents["frontostriatal_motivation_disruption"]
                + 0.16 * latents["diaschisis_remote_suppression"]
                + 0.08 * latents["monoaminergic_dysregulation"]
                - 0.14 * inputs["cognitive_reserve"]
                - 0.10 * inputs["rehabilitation_support"]
            ),
            "inferior_parietal": self._clip01(
                0.44 * latents["frontoparietal_control_network_disruption"]
                + 0.22 * latents["default_mode_network_disruption"]
                + 0.18 * latents["axonal_microstructural_disconnection"]
                + 0.10 * latents["diaschisis_remote_suppression"]
                - 0.12 * inputs["cognitive_reserve"]
                - 0.08 * inputs["rehabilitation_support"]
            ),
        }

        symptoms = {
            "executive_dysfunction": self._clip01(
                0.40 * regional_state["pfc_control"]
                + 0.28 * latents["frontostriatal_motivation_disruption"]
                + 0.18 * latents["frontoparietal_control_network_disruption"]
                + 0.08 * latents["monoaminergic_dysregulation"]
                - 0.10 * inputs["cognitive_reserve"]
            ),
            "attention_dyscontrol": self._clip01(
                0.34 * latents["frontoparietal_control_network_disruption"]
                + 0.24 * regional_state["inferior_parietal"]
                + 0.18 * latents["default_mode_network_disruption"]
                + 0.10 * latents["monoaminergic_dysregulation"]
                - 0.10 * inputs["cognitive_reserve"]
            ),
            "processing_speed_reduction": self._clip01(
                0.36 * latents["axonal_microstructural_disconnection"]
                + 0.24 * regional_state["inferior_parietal"]
                + 0.20 * latents["frontoparietal_control_network_disruption"]
                + 0.10 * latents["diaschisis_remote_suppression"]
                - 0.10 * inputs["cognitive_reserve"]
            ),
            "memory_impairment": self._clip01(
                0.34 * latents["default_mode_network_disruption"]
                + 0.20 * latents["diaschisis_remote_suppression"]
                + 0.18 * latents["tau_neurodegenerative_acceleration"]
                + 0.12 * latents["axonal_microstructural_disconnection"]
                + 0.08 * inputs["injury_severity"]
                - 0.10 * inputs["cognitive_reserve"]
            ),
            "depression_anxiety_burden": self._clip01(
                0.32 * latents["monoaminergic_dysregulation"]
                + 0.18 * latents["neurotransmitter_switching_plasticity"]
                + 0.16 * latents["default_mode_network_disruption"]
                + 0.10 * inputs["neuropsychiatric_predisposition"]
                - 0.10 * inputs["rehabilitation_support"]
            ),
            "apathy_motivation_loss": self._clip01(
                0.36 * latents["frontostriatal_motivation_disruption"]
                + 0.22 * latents["monoaminergic_dysregulation"]
                + 0.14 * regional_state["pfc_control"]
                + 0.10 * inputs["injury_severity"]
                - 0.10 * inputs["rehabilitation_support"]
            ),
            "hypersomnia_sleep_wake_disturbance": self._clip01(
                0.46 * latents["sleep_wake_network_instability"]
                + 0.18 * latents["histaminergic_arousal_loss"]
                + 0.12 * latents["monoaminergic_dysregulation"]
                + 0.08 * inputs["injury_severity"]
                - 0.10 * inputs["rehabilitation_support"]
            ),
            "personality_change_disinhibition": self._clip01(
                0.30 * regional_state["pfc_control"]
                + 0.18 * latents["neurotransmitter_switching_plasticity"]
                + 0.14 * latents["monoaminergic_dysregulation"]
                + 0.12 * inputs["focal_circuit_lesion_load"]
                + 0.10 * inputs["neuropsychiatric_predisposition"]
                - 0.08 * inputs["rehabilitation_support"]
            ),
            "seizure_susceptibility": self._clip01(
                0.46 * latents["post_traumatic_epileptogenesis"]
                + 0.16 * latents["axonal_microstructural_disconnection"]
                + 0.12 * inputs["injury_severity"]
                + 0.10 * inputs["repetitive_tbi_exposure"]
                - 0.08 * inputs["rehabilitation_support"]
            ),
        }

        phenotypes = {
            "multidomain_cognitive_profile": self._clip01(
                (
                    symptoms["executive_dysfunction"]
                    + symptoms["attention_dyscontrol"]
                    + symptoms["processing_speed_reduction"]
                    + symptoms["memory_impairment"]
                )
                / 4.0
            ),
            "affective_behavioral_profile": self._clip01(
                (
                    symptoms["depression_anxiety_burden"]
                    + symptoms["apathy_motivation_loss"]
                    + symptoms["personality_change_disinhibition"]
                )
                / 3.0
            ),
            "sleep_wake_profile": self._clip01(
                (
                    symptoms["hypersomnia_sleep_wake_disturbance"]
                    + latents["sleep_wake_network_instability"]
                )
                / 2.0
            ),
            "progressive_risk_profile": self._clip01(
                (
                    latents["tau_neurodegenerative_acceleration"]
                    + symptoms["memory_impairment"]
                    + symptoms["seizure_susceptibility"]
                )
                / 3.0
            ),
            "overall_tbi_neurocognitive_disorder_profile": self._clip01(
                (
                    symptoms["executive_dysfunction"]
                    + symptoms["attention_dyscontrol"]
                    + symptoms["processing_speed_reduction"]
                    + symptoms["memory_impairment"]
                    + symptoms["depression_anxiety_burden"]
                    + symptoms["apathy_motivation_loss"]
                    + symptoms["hypersomnia_sleep_wake_disturbance"]
                )
                / 7.0
            ),
            "functional_reserve_buffer": protective_buffer,
        }

        return {
            "inputs": pd.Series(inputs, name="inputs"),
            "latents": pd.Series(latents, name="latents"),
            "regional_state": pd.Series(regional_state, name="regional_state"),
            "symptoms": pd.Series(symptoms, name="symptoms"),
            "phenotypes": pd.Series(phenotypes, name="phenotypes"),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if self._pmap is None:
            with siibra.QUIET:
                self._pmap = siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=self.assignment_space,
                    maptype="statistical",
                )

        point = siibra.Point(tuple(xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Any:
        """
        Return a region-specific map or mask object when available.
        The exact returned type depends on the siibra version.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None

        for method_name in ("get_regional_map", "fetch", "build_mask"):
            method = getattr(region, method_name, None)
            if callable(method):
                try:
                    with siibra.QUIET:
                        if method_name == "get_regional_map":
                            return method(space=self.space)
                        return method()
                except Exception:
                    continue

        try:
            with siibra.QUIET:
                pmap = siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=self.space_spec,
                    maptype="labelled",
                )
            return pmap.fetch(region)
        except Exception:
            return None


if __name__ == "__main__":
    model = TraumaticBrainInjuryNeurocognitiveDisorderModel()
    bundle = model.build()

    print("\n=== Nodes ===")
    print(bundle["nodes"][["key", "node_type", "atlas_region", "feature_summary"]].to_string(index=False))

    print("\n=== Edges ===")
    print(bundle["edges"][["source", "target", "relation", "tbi_ncd_change"]].head(18).to_string(index=False))

    print("\n=== Region suggestions for 'parietal' ===")
    print(model.suggest_regions("parietal").head(10).to_string(index=False))

    print("\n=== Pairwise circuit connectivity ===")
    circuit_df = bundle["circuit_connectivity"]
    if circuit_df.empty:
        print("No connectivity matrix could be resolved in this environment.")
    else:
        print(circuit_df.to_string(index=False))

    print("\n=== Example receptor table: pfc_control ===")
    receptor_df = bundle["receptors"].get("pfc_control", pd.DataFrame())
    print(receptor_df.head(10).to_string(index=False) if not receptor_df.empty else "No receptor data available.")

    print("\n=== Example gene table: inferior_parietal ===")
    gene_df = bundle["genes"].get("inferior_parietal", pd.DataFrame())
    print(gene_df.head(10).to_string(index=False) if not gene_df.empty else "No gene-expression data available.")

    print("\n=== Example connectivity profile: pfc_control ===")
    conn_df = bundle["connectivity_profiles"].get("pfc_control", pd.DataFrame())
    print(conn_df.head(10).to_string(index=False) if not conn_df.empty else "No connectivity profile available.")

    print("\n=== Simulation example ===")
    result = model.simulate(
        injury_severity=0.72,
        diffuse_axonal_injury_load=0.78,
        focal_circuit_lesion_load=0.55,
        brainstem_projection_injury=0.42,
        hypothalamic_arousal_injury=0.48,
        repetitive_tbi_exposure=0.60,
        apoe_neurodegenerative_risk=0.58,
        epileptogenic_vulnerability=0.35,
        neuropsychiatric_predisposition=0.40,
        cognitive_reserve=0.38,
        rehabilitation_support=0.52,
    )
    for name, series in result.items():
        print(f"\n{name.upper()}")
        print(series.sort_values(ascending=False).to_string())

    # Optional manual example:
    # print(model.assign_mni_point((-42, 36, 24)).head())
