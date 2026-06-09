from __future__ import annotations

"""Atlas-grounded siibra scaffold for Posttraumatic Stress Disorder.

This script turns a short mechanistic chapter on PTSD into an interpretable
research scaffold. It is designed for exploratory modeling and multimodal atlas
profiling, not for diagnosis, prognosis, or treatment.

Conceptual interpretation of the chapter
----------------------------------------
The source text frames PTSD as a disorder of memory and fear regulation.
Traumatic experiences are modeled as becoming "hot" sensory-emotional traces
that are insufficiently integrated into coherent autobiographical memory. The
chapter further emphasizes:
1. persistent epinephrine / norepinephrine-driven hyperarousal,
2. over-consolidation of fear memories and resistance to extinction,
3. early-life trauma and glucocorticoid-related epigenetic embedding,
4. trauma-linked changes in immune activation, synaptic plasticity, and apoptosis,
5. stress-related dopaminergic dysregulation contributing to anhedonia and
   substance-use vulnerability,
6. disrupted fronto-limbic communication linking the amygdala, hippocampus,
   and prefrontal cortex.

The scaffold therefore models a one-pass causal chain:
inputs -> latent biology -> regional dysregulation -> symptoms -> phenotypes

Because the chapter names the prefrontal cortex broadly, the model uses a
conservative prefrontal control proxy rather than pretending the source text
specified a more precise cytoarchitectonic subdivision.
"""

import itertools
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


DEFAULT_GENE_PANEL = [
    "NR3C1",  # glucocorticoid receptor gene named in the chapter
    "BDNF",   # stress-sensitive plasticity / neurotrophic vulnerability
    "SLC6A2", # noradrenergic signaling
    "TH",     # catecholamine synthesis
    "DRD2",   # dopaminergic signaling
    "COMT",   # catecholamine regulation in cortex
    "IL6",    # immune activation proxy
    "TNF",    # immune activation proxy
    "GRIN2B", # synaptic plasticity / fear-learning relevance
    "SLC1A2", # excitatory homeostasis / plasticity support
    "BCL2",   # apoptosis regulation
    "CASP3",  # apoptosis regulation
]


class PosttraumaticStressDisorderModel:
    """Mechanistic siibra scaffold for Posttraumatic Stress Disorder.

    Notes
    -----
    - This is a research scaffold built from chapter-level claims.
    - Higher simulator values generally reflect greater biological burden or
      symptom severity.
    - Region nodes are atlas-grounded where possible; explicitly labeled proxies
      are used when the chapter stays at a systems level.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        assignment_space: str = "mni152",
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
        self.assignment_space = assignment_space
        self.connectivity_cohort = connectivity_cohort

        self.region_node_descriptions: Dict[str, str] = {
            "amygdala": (
                "Limbic threat and emotional-memory anchor for fear reactivity, "
                "hot memory triggering, and alarm amplification."
            ),
            "hippocampus": (
                "Contextual and autobiographical-memory anchor for integrating "
                "trauma into coherent narrative memory."
            ),
            "pfc_control_proxy": (
                "Conservative proxy for the chapter's broad prefrontal cortex "
                "reference, representing contextualization, inhibition, and "
                "top-down regulation over limbic alarm states."
            ),
            "ventral_striatum_proxy": (
                "Proxy for mesocorticolimbic reward / motivation circuitry used "
                "to represent stress-linked dopaminergic burden, anhedonia, and "
                "substance-use vulnerability."
            ),
        }

        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "DG (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "HC-Subiculum (Hippocampus) left",
                "hippocampus",
            ],
            "pfc_control_proxy": [
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "Area 9 left",
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "frontal pole",
                "prefrontal cortex",
            ],
            "ventral_striatum_proxy": [
                "nucleus accumbens",
                "ventral striatum",
                "striatum",
                "caudate",
                "putamen",
                "basal ganglia",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "traumatic_exposure_load": (
                "Burden of traumatic exposure driving fear-system activation and memory dysregulation."
            ),
            "early_life_trauma_load": (
                "Developmental trauma / childhood abuse burden that can embed long-term molecular vulnerability."
            ),
            "genetic_epigenetic_vulnerability": (
                "Liability affecting stress regulation, molecular embedding, and trauma sensitivity."
            ),
            "ongoing_stress_load": (
                "Persistent stress burden that sustains arousal and hampers recovery from fear responses."
            ),
            "recovery_support": (
                "Protective support factor representing contextual safety, treatment support, and stabilizing resources."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "glucocorticoid_epigenetic_embedding": (
                "Stress-embedded epigenetic burden affecting glucocorticoid regulation and long-term vulnerability."
            ),
            "catecholaminergic_hyperarousal": (
                "Persistent epinephrine / norepinephrine-driven physiological alarm bias."
            ),
            "neuroimmune_activation": (
                "Trauma-linked molecular burden involving immune activation and related neural/endocrine expression shifts."
            ),
            "synaptic_plasticity_apoptosis_shift": (
                "Trauma-related dysregulation of synaptic plasticity and apoptosis pathways relevant to lasting circuit change."
            ),
            "fear_memory_overconsolidation": (
                "Over-strengthening of fear memories with weak extinction and persistent hot memory traces."
            ),
            "dopaminergic_reward_dysregulation": (
                "Stress-linked dopaminergic disruption contributing to anhedonia and addiction vulnerability."
            ),
            "frontolimbic_disconnection": (
                "Weak functional communication between prefrontal control systems and limbic memory/fear systems."
            ),
            "autobiographical_context_failure": (
                "Failure to integrate trauma into coherent, cold autobiographical memory with proper context."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "physiological_alarm_symptoms": (
                "Palpitations, sweating, shortness of breath, and related somatic alarm signals."
            ),
            "danger_misattribution": (
                "Misinterpreting arousal sensations as evidence of immediate current danger."
            ),
            "fear_extinction_resistance": (
                "Persistent fear responding with weak extinction of traumatic associations."
            ),
            "intrusive_hot_memories": (
                "Context-free intrusive sensory / emotional trauma fragments."
            ),
            "present_time_reexperiencing": (
                "Re-experiencing the past as if it were occurring in the present."
            ),
            "anhedonia": (
                "Reduced capacity for reward and pleasure linked to stress-related dopaminergic dysfunction."
            ),
            "substance_use_vulnerability": (
                "Increased liability toward maladaptive substance use in the context of stress and reward dysregulation."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "traumatic_exposure_load",
                "target": "catecholaminergic_hyperarousal",
                "relation": "drives persistent physiological alarm activity",
                "ptsd_change": "increased",
            },
            {
                "source": "ongoing_stress_load",
                "target": "catecholaminergic_hyperarousal",
                "relation": "sustains chronic arousal and stress reactivity",
                "ptsd_change": "increased",
            },
            {
                "source": "early_life_trauma_load",
                "target": "glucocorticoid_epigenetic_embedding",
                "relation": "promotes durable trauma-linked methylation and stress-system embedding",
                "ptsd_change": "increased",
            },
            {
                "source": "genetic_epigenetic_vulnerability",
                "target": "glucocorticoid_epigenetic_embedding",
                "relation": "raises susceptibility to persistent stress-regulation changes",
                "ptsd_change": "increased",
            },
            {
                "source": "early_life_trauma_load",
                "target": "neuroimmune_activation",
                "relation": "amplifies trauma-related immune and molecular burden",
                "ptsd_change": "increased",
            },
            {
                "source": "traumatic_exposure_load",
                "target": "synaptic_plasticity_apoptosis_shift",
                "relation": "induces molecular shifts in plasticity and cell-survival pathways",
                "ptsd_change": "increased",
            },
            {
                "source": "glucocorticoid_epigenetic_embedding",
                "target": "synaptic_plasticity_apoptosis_shift",
                "relation": "translates embedded stress vulnerability into lasting circuit remodeling",
                "ptsd_change": "increased",
            },
            {
                "source": "neuroimmune_activation",
                "target": "synaptic_plasticity_apoptosis_shift",
                "relation": "adds molecular pressure on plasticity and structural integrity",
                "ptsd_change": "increased",
            },
            {
                "source": "catecholaminergic_hyperarousal",
                "target": "fear_memory_overconsolidation",
                "relation": "strengthens traumatic fear traces and weakens extinction",
                "ptsd_change": "increased",
            },
            {
                "source": "catecholaminergic_hyperarousal",
                "target": "dopaminergic_reward_dysregulation",
                "relation": "perturbs stress-sensitive dopaminergic reward circuitry",
                "ptsd_change": "increased",
            },
            {
                "source": "fear_memory_overconsolidation",
                "target": "frontolimbic_disconnection",
                "relation": "biases the circuit toward threat-dominant memory processing",
                "ptsd_change": "increased",
            },
            {
                "source": "synaptic_plasticity_apoptosis_shift",
                "target": "frontolimbic_disconnection",
                "relation": "contributes to altered connectivity and structural burden",
                "ptsd_change": "increased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "autobiographical_context_failure",
                "relation": "weakens contextual integration of emotional memory fragments",
                "ptsd_change": "increased",
            },
            {
                "source": "fear_memory_overconsolidation",
                "target": "autobiographical_context_failure",
                "relation": "allows hot memories to dominate over coherent narrative memory",
                "ptsd_change": "increased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "amygdala",
                "relation": "permits greater limbic alarm reactivity",
                "ptsd_change": "increased",
            },
            {
                "source": "autobiographical_context_failure",
                "target": "hippocampus",
                "relation": "reflects contextual-memory burden and impaired integration",
                "ptsd_change": "increased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "pfc_control_proxy",
                "relation": "reduces effective prefrontal contextualization and inhibition",
                "ptsd_change": "increased",
            },
            {
                "source": "dopaminergic_reward_dysregulation",
                "target": "ventral_striatum_proxy",
                "relation": "loads mesocorticolimbic reward circuitry",
                "ptsd_change": "increased",
            },
            {
                "source": "catecholaminergic_hyperarousal",
                "target": "physiological_alarm_symptoms",
                "relation": "produces sustained somatic alarm signals",
                "ptsd_change": "increased",
            },
            {
                "source": "physiological_alarm_symptoms",
                "target": "danger_misattribution",
                "relation": "encourages interpreting bodily alarm as immediate threat",
                "ptsd_change": "increased",
            },
            {
                "source": "fear_memory_overconsolidation",
                "target": "fear_extinction_resistance",
                "relation": "makes fear memories harder to extinguish",
                "ptsd_change": "increased",
            },
            {
                "source": "fear_memory_overconsolidation",
                "target": "intrusive_hot_memories",
                "relation": "generates intrusive context-free emotional/sensory traces",
                "ptsd_change": "increased",
            },
            {
                "source": "autobiographical_context_failure",
                "target": "present_time_reexperiencing",
                "relation": "prevents the traumatic event from being experienced as safely in the past",
                "ptsd_change": "increased",
            },
            {
                "source": "dopaminergic_reward_dysregulation",
                "target": "anhedonia",
                "relation": "reduces reward responsiveness and pleasure capacity",
                "ptsd_change": "increased",
            },
            {
                "source": "anhedonia",
                "target": "substance_use_vulnerability",
                "relation": "can increase maladaptive reward seeking and coping through substances",
                "ptsd_change": "increased",
            },
            {
                "source": "recovery_support",
                "target": "frontolimbic_disconnection",
                "relation": "buffers circuit-level dysregulation",
                "ptsd_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "autobiographical_context_failure",
                "relation": "supports safer contextualization of trauma memories",
                "ptsd_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "physiological_alarm_symptoms",
                "relation": "reduces destabilizing arousal burden",
                "ptsd_change": "decreased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame(self.edge_table)
        self._pmap: Any = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None
        self._connectivity_feature: Any = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _name_key(obj: Any) -> str:
        text = str(getattr(obj, "name", obj)).lower().replace("-", " ")
        return " ".join(text.split())

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
        name = self._name_key(region)
        left_penalty = 0 if " left" in f" {name}" else 1
        right_penalty = 1 if " right" in f" {name}" else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "prefrontal cortex",
            "frontal pole",
            "striatum",
            "basal ganglia",
            "caudate",
            "putamen",
        } else 0
        specificity_penalty = 0 if ("area " in name or "(" in name) else 1
        return (left_penalty, right_penalty, generic_penalty, specificity_penalty)

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
                return sorted(matches, key=self._region_rank)[0]
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
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        volume_mm3 = float(getattr(main, "volume", float("nan")))
        return centroid_xyz, volume_mm3

    def _region_volume(self, region: Any):
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(space=self.assignment_space)
            except Exception:
                return None

    def _feature_df(self, feature: Any) -> pd.DataFrame:
        try:
            data = getattr(feature, "data", None)
            if isinstance(data, pd.DataFrame):
                return data.copy()
            if data is not None:
                return pd.DataFrame(data)
        except Exception:
            pass
        return pd.DataFrame()

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            volume = self._region_volume(region)
            if volume is not None:
                feats = self._safe_features_any(volume, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        df = self._feature_df(feats[0]).reset_index(drop=False)
        if df.empty:
            return df
        lower_cols = {c.lower(): c for c in df.columns}
        if "index" in df.columns and "receptor" not in lower_cols:
            df = df.rename(columns={"index": "receptor"})
        return df.reset_index(drop=True)

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()
        df = self._feature_df(feats[0])
        if df.empty:
            return df

        lower_cols = {c.lower(): c for c in df.columns}
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

    def _get_connectivity_feature(self) -> Any:
        if self._connectivity_feature is not None:
            return self._connectivity_feature
        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_feature = None
            return None
        preferred = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), feats[0])
        self._connectivity_feature = preferred
        return preferred

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feature = self._get_connectivity_feature()
        if feature is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        df = self._feature_df(feature)
        if not df.empty:
            self._connectivity_matrix = df
            return self._connectivity_matrix

        element: Any = None
        try:
            element = feature[0]
        except Exception:
            element = None

        if element is None:
            try:
                first_index = feature.indices[0]
                element = feature.get_element(first_index)
            except Exception:
                element = None

        if element is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        df = self._feature_df(element)
        if df.empty:
            raw = getattr(element, "data", None)
            regions = getattr(element, "regions", None)
            try:
                if raw is not None and regions is not None:
                    region_list = list(regions)
                    df = pd.DataFrame(raw, index=region_list, columns=region_list)
            except Exception:
                df = pd.DataFrame()

        self._connectivity_matrix = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._name_key(region)

        exact = [x for x in labels if self._name_key(x) == region_name]
        if exact:
            return exact[0]

        fuzzy = [
            x for x in labels
            if region_name in self._name_key(x) or self._name_key(x) in region_name
        ]
        if fuzzy:
            return fuzzy[0]

        region_tokens = set(region_name.split())
        scored: List[Tuple[int, Any]] = []
        for x in labels:
            name = self._name_key(x)
            tokens = set(name.split())
            overlap = len(region_tokens & tokens)
            if overlap:
                scored.append((overlap, x))
        if scored:
            scored.sort(key=lambda pair: pair[0], reverse=True)
            return scored[0][1]
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
            series = pd.Series(series)
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name]
            df = df[df["value"].notna()]
            if "value" in df.columns:
                try:
                    df = df[df["value"] > 0]
                except Exception:
                    pass
            return df.head(max_rows).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame(columns=["source", "target", "value", "cohort"])

        rows: List[Dict[str, Any]] = []
        for src_key, dst_key in itertools.combinations(self.region_objects.keys(), 2):
            src = self.region_objects[src_key]
            dst = self.region_objects[dst_key]
            src_label = self._match_region_label(list(matrix.index), src) or self._match_region_label(list(matrix.columns), src)
            dst_label = self._match_region_label(list(matrix.columns), dst) or self._match_region_label(list(matrix.index), dst)
            if src_label is None or dst_label is None:
                continue

            value: Optional[float] = None
            try:
                value = float(matrix.loc[src_label, dst_label])
            except Exception:
                try:
                    value = float(matrix.loc[dst_label, src_label])
                except Exception:
                    value = None
            if value is None:
                continue
            rows.append(
                {
                    "source": src_key,
                    "target": dst_key,
                    "value": value,
                    "cohort": getattr(self._get_connectivity_feature(), "cohort", None),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["source", "target", "value", "cohort"])
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

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
                    "is_proxy": False,
                    "description": desc,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}

        for key, candidates in self.region_candidates.items():
            is_proxy = key.endswith("_proxy")
            desc = self.region_node_descriptions.get(key, "Atlas-backed circuit node")
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
                        "is_proxy": is_proxy,
                        "description": f"{desc} (unresolved in this siibra environment)",
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
                    "is_proxy": is_proxy,
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
                    "is_proxy": False,
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
                    "is_proxy": False,
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
        traumatic_exposure_load: float = 0.65,
        early_life_trauma_load: float = 0.45,
        genetic_epigenetic_vulnerability: float = 0.50,
        ongoing_stress_load: float = 0.55,
        recovery_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """Run a simple normalized PTSD simulator.

        All values are clipped to [0, 1]. Higher values indicate greater burden.
        Recovery support subtracts from several pathological pathways.
        """

        inputs = pd.Series(
            {
                "traumatic_exposure_load": self._clip01(traumatic_exposure_load),
                "early_life_trauma_load": self._clip01(early_life_trauma_load),
                "genetic_epigenetic_vulnerability": self._clip01(genetic_epigenetic_vulnerability),
                "ongoing_stress_load": self._clip01(ongoing_stress_load),
                "recovery_support": self._clip01(recovery_support),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["glucocorticoid_epigenetic_embedding"] = self._clip01(
            0.38 * inputs["early_life_trauma_load"]
            + 0.30 * inputs["genetic_epigenetic_vulnerability"]
            + 0.12 * inputs["traumatic_exposure_load"]
            + 0.10 * inputs["ongoing_stress_load"]
            - 0.18 * inputs["recovery_support"]
        )
        latents["catecholaminergic_hyperarousal"] = self._clip01(
            0.34 * inputs["traumatic_exposure_load"]
            + 0.26 * inputs["ongoing_stress_load"]
            + 0.20 * latents["glucocorticoid_epigenetic_embedding"]
            + 0.08 * inputs["early_life_trauma_load"]
            - 0.18 * inputs["recovery_support"]
        )
        latents["neuroimmune_activation"] = self._clip01(
            0.28 * inputs["traumatic_exposure_load"]
            + 0.24 * inputs["early_life_trauma_load"]
            + 0.20 * latents["glucocorticoid_epigenetic_embedding"]
            + 0.10 * inputs["ongoing_stress_load"]
            - 0.10 * inputs["recovery_support"]
        )
        latents["synaptic_plasticity_apoptosis_shift"] = self._clip01(
            0.28 * inputs["traumatic_exposure_load"]
            + 0.24 * latents["neuroimmune_activation"]
            + 0.20 * latents["glucocorticoid_epigenetic_embedding"]
            + 0.14 * latents["catecholaminergic_hyperarousal"]
            - 0.10 * inputs["recovery_support"]
        )
        latents["fear_memory_overconsolidation"] = self._clip01(
            0.38 * latents["catecholaminergic_hyperarousal"]
            + 0.24 * inputs["traumatic_exposure_load"]
            + 0.18 * latents["synaptic_plasticity_apoptosis_shift"]
            + 0.08 * inputs["ongoing_stress_load"]
            - 0.10 * inputs["recovery_support"]
        )
        latents["dopaminergic_reward_dysregulation"] = self._clip01(
            0.34 * latents["catecholaminergic_hyperarousal"]
            + 0.24 * latents["synaptic_plasticity_apoptosis_shift"]
            + 0.16 * inputs["ongoing_stress_load"]
            + 0.10 * inputs["traumatic_exposure_load"]
            - 0.10 * inputs["recovery_support"]
        )
        latents["frontolimbic_disconnection"] = self._clip01(
            0.34 * latents["fear_memory_overconsolidation"]
            + 0.24 * latents["synaptic_plasticity_apoptosis_shift"]
            + 0.18 * latents["glucocorticoid_epigenetic_embedding"]
            + 0.10 * latents["neuroimmune_activation"]
            - 0.20 * inputs["recovery_support"]
        )
        latents["autobiographical_context_failure"] = self._clip01(
            0.34 * latents["frontolimbic_disconnection"]
            + 0.30 * latents["fear_memory_overconsolidation"]
            + 0.18 * latents["synaptic_plasticity_apoptosis_shift"]
            + 0.08 * latents["glucocorticoid_epigenetic_embedding"]
            - 0.20 * inputs["recovery_support"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["amygdala"] = self._clip01(
            0.44 * latents["fear_memory_overconsolidation"]
            + 0.28 * latents["catecholaminergic_hyperarousal"]
            + 0.20 * latents["frontolimbic_disconnection"]
            - 0.14 * inputs["recovery_support"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.40 * latents["autobiographical_context_failure"]
            + 0.24 * latents["synaptic_plasticity_apoptosis_shift"]
            + 0.18 * latents["glucocorticoid_epigenetic_embedding"]
            + 0.10 * latents["fear_memory_overconsolidation"]
            - 0.14 * inputs["recovery_support"]
        )
        regional_state["pfc_control_proxy"] = self._clip01(
            0.48 * latents["frontolimbic_disconnection"]
            + 0.20 * latents["catecholaminergic_hyperarousal"]
            + 0.14 * latents["autobiographical_context_failure"]
            + 0.08 * latents["glucocorticoid_epigenetic_embedding"]
            - 0.24 * inputs["recovery_support"]
        )
        regional_state["ventral_striatum_proxy"] = self._clip01(
            0.44 * latents["dopaminergic_reward_dysregulation"]
            + 0.24 * latents["catecholaminergic_hyperarousal"]
            + 0.14 * latents["frontolimbic_disconnection"]
            - 0.10 * inputs["recovery_support"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["physiological_alarm_symptoms"] = self._clip01(
            0.44 * latents["catecholaminergic_hyperarousal"]
            + 0.24 * regional_state["amygdala"]
            + 0.14 * regional_state["pfc_control_proxy"]
            + 0.10 * inputs["ongoing_stress_load"]
            - 0.14 * inputs["recovery_support"]
        )
        symptoms["danger_misattribution"] = self._clip01(
            0.34 * symptoms["physiological_alarm_symptoms"]
            + 0.24 * regional_state["amygdala"]
            + 0.20 * latents["autobiographical_context_failure"]
            + 0.10 * regional_state["pfc_control_proxy"]
            - 0.10 * inputs["recovery_support"]
        )
        symptoms["fear_extinction_resistance"] = self._clip01(
            0.40 * latents["fear_memory_overconsolidation"]
            + 0.24 * regional_state["amygdala"]
            + 0.18 * latents["frontolimbic_disconnection"]
            + 0.10 * regional_state["hippocampus"]
            - 0.10 * inputs["recovery_support"]
        )
        symptoms["intrusive_hot_memories"] = self._clip01(
            0.34 * latents["fear_memory_overconsolidation"]
            + 0.24 * latents["autobiographical_context_failure"]
            + 0.20 * regional_state["amygdala"]
            + 0.10 * regional_state["hippocampus"]
            - 0.10 * inputs["recovery_support"]
        )
        symptoms["present_time_reexperiencing"] = self._clip01(
            0.34 * symptoms["intrusive_hot_memories"]
            + 0.24 * symptoms["danger_misattribution"]
            + 0.20 * latents["autobiographical_context_failure"]
            + 0.10 * regional_state["pfc_control_proxy"]
            - 0.10 * inputs["recovery_support"]
        )
        symptoms["anhedonia"] = self._clip01(
            0.44 * latents["dopaminergic_reward_dysregulation"]
            + 0.24 * regional_state["ventral_striatum_proxy"]
            + 0.14 * latents["frontolimbic_disconnection"]
            - 0.10 * inputs["recovery_support"]
        )
        symptoms["substance_use_vulnerability"] = self._clip01(
            0.34 * symptoms["anhedonia"]
            + 0.24 * latents["dopaminergic_reward_dysregulation"]
            + 0.18 * symptoms["physiological_alarm_symptoms"]
            + 0.10 * inputs["ongoing_stress_load"]
            - 0.10 * inputs["recovery_support"]
        )

        phenotypes = pd.Series(
            {
                "hyperarousal_profile": self._clip01(
                    (
                        symptoms["physiological_alarm_symptoms"]
                        + symptoms["danger_misattribution"]
                        + symptoms["fear_extinction_resistance"]
                    ) / 3.0
                ),
                "reexperiencing_profile": self._clip01(
                    (
                        symptoms["intrusive_hot_memories"]
                        + symptoms["present_time_reexperiencing"]
                        + latents["autobiographical_context_failure"]
                    ) / 3.0
                ),
                "developmental_embedding_profile": self._clip01(
                    (
                        latents["glucocorticoid_epigenetic_embedding"]
                        + latents["neuroimmune_activation"]
                        + latents["synaptic_plasticity_apoptosis_shift"]
                    ) / 3.0
                ),
                "reward_numbing_profile": self._clip01(
                    (
                        symptoms["anhedonia"]
                        + symptoms["substance_use_vulnerability"]
                        + latents["dopaminergic_reward_dysregulation"]
                    ) / 3.0
                ),
                "frontolimbic_disconnection_profile": self._clip01(
                    (
                        latents["frontolimbic_disconnection"]
                        + regional_state["pfc_control_proxy"]
                        + regional_state["amygdala"]
                        + regional_state["hippocampus"]
                    ) / 4.0
                ),
                "overall_ptsd_burden": self._clip01(
                    (
                        symptoms["present_time_reexperiencing"]
                        + symptoms["intrusive_hot_memories"]
                        + symptoms["physiological_alarm_symptoms"]
                        + symptoms["fear_extinction_resistance"]
                        + symptoms["anhedonia"]
                    ) / 5.0
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

    def region_mask(self, node_key: str):
        region = self.region_objects.get(node_key)
        if region is None:
            region = self._resolve_region(self.region_candidates.get(node_key, []))
        if region is None:
            return None
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(space=self.assignment_space)
            except Exception:
                return None


if __name__ == "__main__":
    model = PosttraumaticStressDisorderModel()
    built = model.build(connectivity_rows=10)

    print("\n=== NODES ===")
    node_cols = [
        "key",
        "node_type",
        "is_proxy",
        "atlas_region",
        "feature_summary",
    ]
    print(built["nodes"][node_cols].to_string(index=False))

    print("\n=== EDGES ===")
    edge_cols = ["source", "target", "relation", "ptsd_change"]
    print(built["edges"][edge_cols].to_string(index=False))

    print("\n=== CIRCUIT CONNECTIVITY (resolved region pairs) ===")
    circuit_df = built["circuit_connectivity"]
    if circuit_df.empty:
        print("No circuit connectivity matrix could be resolved in this environment.")
    else:
        print(circuit_df.head(10).to_string(index=False))

    for key in ["amygdala", "hippocampus", "pfc_control_proxy", "ventral_striatum_proxy"]:
        print(f"\n=== {key.upper()} GENE SUMMARY ===")
        gene_df = built["genes"].get(key, pd.DataFrame())
        print(gene_df.head(10).to_string(index=False) if not gene_df.empty else "No gene data available.")

        print(f"\n=== {key.upper()} CONNECTIVITY PROFILE ===")
        conn_df = built["connectivity_profiles"].get(key, pd.DataFrame())
        print(conn_df.head(10).to_string(index=False) if not conn_df.empty else "No connectivity profile available.")

    sim = model.simulate(
        traumatic_exposure_load=0.80,
        early_life_trauma_load=0.60,
        genetic_epigenetic_vulnerability=0.55,
        ongoing_stress_load=0.65,
        recovery_support=0.20,
    )

    print("\n=== SIMULATION OUTPUT ===")
    for name, series in sim.items():
        print(f"\n[{name}]")
        print(series.sort_values(ascending=False).to_string())

    print("\n=== OPTIONAL USAGE ===")
    print("# model.assign_mni_point((-24, -6, -18)).head()  # amygdala-like point")
    print("# mask = model.region_mask('hippocampus')")
