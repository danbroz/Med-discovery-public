
from __future__ import annotations

"""
Psychotic disorder due to another medical condition siibra scaffold.

This research scaffold translates a chapter-level summary of Psychotic
Disorder Due to Another Medical Condition into an atlas-grounded, transparent
mechanistic model.

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
    "DRD2",
    "COMT",
    "GRIN1",
    "GRIN2A",
    "GRIN2B",
    "GAD1",
    "GABRA1",
    "SLC1A2",
    "BDNF",
    "NRG1",
    "AKT1",
    "CACNA1C",
    "IL6",
    "TNF",
]


class PsychoticDisorderDueToAnotherMedicalConditionModel:
    """
    Atlas-grounded scaffold for psychosis emerging from medical and neurologic illness.

    Chapter logic emphasized here:
    - glutamatergic / GABAergic imbalance as a shared route from epilepsy,
      metabolic encephalopathy, hypoxia, and neuroinflammation into psychosis,
    - neuronal hyperexcitability as a key bridge from temporal lobe epilepsy to
      interictal, post-ictal, or alternative psychotic states,
    - frontotemporal network disruption after traumatic brain injury,
    - lesion-related perceptual and thought dysregulation after temporal,
      occipital, and subcortical / thalamic cerebrovascular injury,
    - a "second hit" model in which medical insults unmask a latent
      psychosis-prone biological vulnerability.

    Because the chapter is intentionally broad and medically heterogeneous,
    several atlas nodes are conservative proxies rather than precise parcel
    claims.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.disorder_name = "Psychotic Disorder Due to Another Medical Condition"
        self.domain_key = "medical_psychosis"
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
            "temporal_lobe_proxy": [
                "Area TE 2.1 (STG) left",
                "Area TE 3 (STG) left",
                "Area TE 1.2 (HESCHL) left",
                "CA1 (Hippocampus) left",
                "SF (Amygdala) left",
                "temporal lobe",
                "hippocampus left",
                "amygdala left",
            ],
            "frontal_lobe_proxy": [
                "Area 9/46d (DLPFC) left",
                "Area 9/46v (DLPFC) left",
                "Area 46 left",
                "Area 9 left",
                "middle frontal gyrus",
                "frontal lobe",
                "dorsolateral prefrontal cortex",
            ],
            "occipital_association_proxy": [
                "Area hOc5 (LOC) left",
                "Area hOc3v (LingG) left",
                "Area hOc4v (LingG) left",
                "Area hOc2 (V2, 18) left",
                "occipital lobe",
                "visual association cortex",
                "visual cortex",
            ],
            "subcortical_thalamic_proxy": [
                "thalamus left",
                "thalamus",
                "caudate nucleus left",
                "putamen left",
                "subcortical",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "epilepsy_hyperexcitability_load": (
                "Burden of epileptic hypersynchrony, especially temporal-lobe network instability."
            ),
            "metabolic_encephalopathy_hypoxia_load": (
                "Burden of metabolic encephalopathy, hypoxia, or other diffuse physiological brain stress."
            ),
            "neuroinflammatory_systemic_illness_load": (
                "Burden of neuroinflammatory, infectious, or systemic medical illness affecting the CNS."
            ),
            "traumatic_brain_injury_load": (
                "Burden of focal contusions and diffuse axonal injury after traumatic brain injury."
            ),
            "cerebrovascular_lesion_load": (
                "Burden of stroke, infarction, or silent cerebrovascular lesions affecting psychosis-relevant circuits."
            ),
            "genetic_psychosis_liability": (
                "Latent heritable vulnerability that lowers the threshold for psychosis when medical insults occur."
            ),
            "medical_stabilization_treatment_support": (
                "Protective effect of correcting the medical cause, reducing instability, and supportive treatment."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "psychosis_threshold_lowering": (
                "Reduced resilience or lowered threshold for psychosis expression after neurologic or systemic insult."
            ),
            "glutamate_gaba_imbalance": (
                "Disruption of excitatory-inhibitory balance linking epilepsy, inflammation, hypoxia, and delirious states."
            ),
            "neuronal_hyperexcitability": (
                "Hypersynchronous or overexcitable neural activity that can manifest as seizure-related psychosis."
            ),
            "metabolic_delirium_bridge": (
                "Diffuse physiological brain dysfunction bridging encephalopathy, delirium, and psychosis."
            ),
            "frontotemporal_network_disruption": (
                "Disconnection and circuit injury across frontal and temporal systems, especially after TBI."
            ),
            "focal_perceptual_release": (
                "Lesion-related perceptual disinhibition or release phenomena, particularly with temporal or occipital injury."
            ),
            "subcortical_reality_filtering_disruption": (
                "Breakdown of subcortical or thalamic filtering needed for stable thought and salience integration."
            ),
            "distributed_perception_thought_dysregulation": (
                "Large-scale disruption of networks governing perception, thought content, and emotional regulation."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "hallucinations": "Perceptual experiences without an external stimulus.",
            "delusional_paranoid_ideation": "Fixed false beliefs, suspiciousness, or paranoid interpretations.",
            "thought_disorganization": "Disordered thought form or behavior due to disrupted reality testing.",
            "affective_behavioral_dysregulation": (
                "Emotional lability or behavioral dyscontrol accompanying lesion-related psychosis."
            ),
            "fluctuating_confusional_psychosis": (
                "Psychosis embedded in delirious, encephalopathic, or rapidly fluctuating medical states."
            ),
            "global_psychosis_burden": "Overall severity of medically driven psychotic symptom expression.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_psychosis_liability",
                "target": "psychosis_threshold_lowering",
                "relation": "creates a latent vulnerability that can be unmasked by medical insults",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "epilepsy_hyperexcitability_load",
                "target": "glutamate_gaba_imbalance",
                "relation": "disrupts excitatory-inhibitory balance through epileptic network instability",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "epilepsy_hyperexcitability_load",
                "target": "neuronal_hyperexcitability",
                "relation": "directly produces hypersynchronous firing associated with seizure-linked psychosis",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "metabolic_encephalopathy_hypoxia_load",
                "target": "glutamate_gaba_imbalance",
                "relation": "destabilizes excitatory-inhibitory regulation during encephalopathy or hypoxia",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "metabolic_encephalopathy_hypoxia_load",
                "target": "metabolic_delirium_bridge",
                "relation": "links diffuse physiological brain dysfunction to delirium and psychosis",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "neuroinflammatory_systemic_illness_load",
                "target": "glutamate_gaba_imbalance",
                "relation": "adds inflammatory or infectious disruption to excitatory-inhibitory balance",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "neuroinflammatory_systemic_illness_load",
                "target": "metabolic_delirium_bridge",
                "relation": "contributes diffuse medical-brain dysfunction that can become psychotic or delirious",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "traumatic_brain_injury_load",
                "target": "frontotemporal_network_disruption",
                "relation": "damages frontal and temporal circuits through contusion and axonal injury",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "traumatic_brain_injury_load",
                "target": "psychosis_threshold_lowering",
                "relation": "acts as a potent environmental second hit for vulnerable individuals",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "cerebrovascular_lesion_load",
                "target": "focal_perceptual_release",
                "relation": "creates temporal or occipital lesions capable of producing hallucinations",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "cerebrovascular_lesion_load",
                "target": "subcortical_reality_filtering_disruption",
                "relation": "injures subcortical or thalamic systems that help stabilize thought and perception",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "glutamate_gaba_imbalance",
                "target": "neuronal_hyperexcitability",
                "relation": "permits psychosis-prone network overexcitation",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "glutamate_gaba_imbalance",
                "target": "metabolic_delirium_bridge",
                "relation": "helps connect physiological brain dysfunction with delirious psychotic states",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "neuronal_hyperexcitability",
                "target": "temporal_lobe_proxy",
                "relation": "is especially modeled through temporal-lobe seizure circuitry",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "frontotemporal_network_disruption",
                "target": "frontal_lobe_proxy",
                "relation": "burdens frontal systems involved in thought and behavioral control",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "frontotemporal_network_disruption",
                "target": "temporal_lobe_proxy",
                "relation": "burdens temporal systems implicated in post-TBI psychosis",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "focal_perceptual_release",
                "target": "occipital_association_proxy",
                "relation": "is represented through lesion-sensitive occipital perceptual association cortex",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "subcortical_reality_filtering_disruption",
                "target": "subcortical_thalamic_proxy",
                "relation": "is represented through subcortical-thalamic gating dysfunction",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "psychosis_threshold_lowering",
                "target": "distributed_perception_thought_dysregulation",
                "relation": "makes medical circuit disruption more likely to become overt psychosis",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "neuronal_hyperexcitability",
                "target": "distributed_perception_thought_dysregulation",
                "relation": "propagates unstable firing into broader psychosis-relevant networks",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "frontotemporal_network_disruption",
                "target": "distributed_perception_thought_dysregulation",
                "relation": "disorganizes circuits governing thought content and emotional regulation",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "focal_perceptual_release",
                "target": "hallucinations",
                "relation": "drives lesion-related hallucinations and perceptual distortions",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "subcortical_reality_filtering_disruption",
                "target": "delusional_paranoid_ideation",
                "relation": "impairs filtering and interpretation of internally generated or external signals",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "distributed_perception_thought_dysregulation",
                "target": "thought_disorganization",
                "relation": "disrupts coherent thought and behavior",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "frontal_lobe_proxy",
                "target": "affective_behavioral_dysregulation",
                "relation": "supports lesion-related emotional and behavioral dyscontrol",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "metabolic_delirium_bridge",
                "target": "fluctuating_confusional_psychosis",
                "relation": "produces rapidly fluctuating psychosis in medically unstable states",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "hallucinations",
                "target": "global_psychosis_burden",
                "relation": "contributes to overall psychosis severity",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "delusional_paranoid_ideation",
                "target": "global_psychosis_burden",
                "relation": "contributes to overall psychosis severity",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "thought_disorganization",
                "target": "global_psychosis_burden",
                "relation": "contributes to overall psychosis severity",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "fluctuating_confusional_psychosis",
                "target": "global_psychosis_burden",
                "relation": "contributes to overall psychosis severity",
                "medical_psychosis_change": "increased",
            },
            {
                "source": "medical_stabilization_treatment_support",
                "target": "glutamate_gaba_imbalance",
                "relation": "medical correction and stabilization can reduce physiologic destabilization",
                "medical_psychosis_change": "decreased",
            },
            {
                "source": "medical_stabilization_treatment_support",
                "target": "global_psychosis_burden",
                "relation": "treating the underlying condition can reduce psychotic expression",
                "medical_psychosis_change": "decreased",
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
        generic_penalty = 1 if name in {"amygdala", "hippocampus", "prefrontal cortex", "thalamus"} else 0
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
        circuit_df = self.circuit_connectivity()

        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": circuit_df,
        }

    def simulate(
        self,
        epilepsy_hyperexcitability_load: float = 0.35,
        metabolic_encephalopathy_hypoxia_load: float = 0.20,
        neuroinflammatory_systemic_illness_load: float = 0.15,
        traumatic_brain_injury_load: float = 0.20,
        cerebrovascular_lesion_load: float = 0.20,
        genetic_psychosis_liability: float = 0.30,
        medical_stabilization_treatment_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """
        One-pass normalized simulator.

        Values represent burden or support on a 0..1 scale.
        Higher regional-state values reflect greater dysfunction burden,
        not healthier activity.
        """

        inputs = pd.Series(
            {
                "epilepsy_hyperexcitability_load": self._clip01(epilepsy_hyperexcitability_load),
                "metabolic_encephalopathy_hypoxia_load": self._clip01(metabolic_encephalopathy_hypoxia_load),
                "neuroinflammatory_systemic_illness_load": self._clip01(neuroinflammatory_systemic_illness_load),
                "traumatic_brain_injury_load": self._clip01(traumatic_brain_injury_load),
                "cerebrovascular_lesion_load": self._clip01(cerebrovascular_lesion_load),
                "genetic_psychosis_liability": self._clip01(genetic_psychosis_liability),
                "medical_stabilization_treatment_support": self._clip01(
                    medical_stabilization_treatment_support
                ),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")

        latents.loc["psychosis_threshold_lowering"] = self._clip01(
            0.60 * inputs["genetic_psychosis_liability"]
            + 0.10 * inputs["traumatic_brain_injury_load"]
            + 0.10 * inputs["cerebrovascular_lesion_load"]
            + 0.10 * inputs["neuroinflammatory_systemic_illness_load"]
            + 0.10 * inputs["metabolic_encephalopathy_hypoxia_load"]
            - 0.20 * inputs["medical_stabilization_treatment_support"]
        )
        latents.loc["glutamate_gaba_imbalance"] = self._clip01(
            0.35 * inputs["epilepsy_hyperexcitability_load"]
            + 0.20 * inputs["metabolic_encephalopathy_hypoxia_load"]
            + 0.15 * inputs["neuroinflammatory_systemic_illness_load"]
            + 0.10 * inputs["traumatic_brain_injury_load"]
            + 0.10 * inputs["cerebrovascular_lesion_load"]
            + 0.10 * latents["psychosis_threshold_lowering"]
            - 0.15 * inputs["medical_stabilization_treatment_support"]
        )
        latents.loc["neuronal_hyperexcitability"] = self._clip01(
            0.45 * inputs["epilepsy_hyperexcitability_load"]
            + 0.25 * latents["glutamate_gaba_imbalance"]
            + 0.10 * inputs["metabolic_encephalopathy_hypoxia_load"]
            + 0.10 * inputs["neuroinflammatory_systemic_illness_load"]
            + 0.10 * latents["psychosis_threshold_lowering"]
            - 0.15 * inputs["medical_stabilization_treatment_support"]
        )
        latents.loc["metabolic_delirium_bridge"] = self._clip01(
            0.45 * inputs["metabolic_encephalopathy_hypoxia_load"]
            + 0.20 * inputs["neuroinflammatory_systemic_illness_load"]
            + 0.15 * latents["glutamate_gaba_imbalance"]
            + 0.10 * inputs["cerebrovascular_lesion_load"]
            + 0.10 * latents["psychosis_threshold_lowering"]
            - 0.20 * inputs["medical_stabilization_treatment_support"]
        )
        latents.loc["frontotemporal_network_disruption"] = self._clip01(
            0.40 * inputs["traumatic_brain_injury_load"]
            + 0.20 * latents["psychosis_threshold_lowering"]
            + 0.15 * inputs["cerebrovascular_lesion_load"]
            + 0.10 * inputs["neuroinflammatory_systemic_illness_load"]
            + 0.10 * latents["neuronal_hyperexcitability"]
            + 0.05 * inputs["metabolic_encephalopathy_hypoxia_load"]
            - 0.10 * inputs["medical_stabilization_treatment_support"]
        )
        latents.loc["focal_perceptual_release"] = self._clip01(
            0.35 * inputs["cerebrovascular_lesion_load"]
            + 0.20 * latents["neuronal_hyperexcitability"]
            + 0.15 * latents["metabolic_delirium_bridge"]
            + 0.15 * latents["frontotemporal_network_disruption"]
            + 0.15 * latents["psychosis_threshold_lowering"]
            - 0.10 * inputs["medical_stabilization_treatment_support"]
        )
        latents.loc["subcortical_reality_filtering_disruption"] = self._clip01(
            0.35 * inputs["cerebrovascular_lesion_load"]
            + 0.20 * latents["metabolic_delirium_bridge"]
            + 0.15 * latents["frontotemporal_network_disruption"]
            + 0.15 * latents["glutamate_gaba_imbalance"]
            + 0.15 * latents["psychosis_threshold_lowering"]
            - 0.10 * inputs["medical_stabilization_treatment_support"]
        )
        latents.loc["distributed_perception_thought_dysregulation"] = self._clip01(
            0.20 * latents["glutamate_gaba_imbalance"]
            + 0.20 * latents["neuronal_hyperexcitability"]
            + 0.20 * latents["frontotemporal_network_disruption"]
            + 0.15 * latents["focal_perceptual_release"]
            + 0.15 * latents["subcortical_reality_filtering_disruption"]
            + 0.10 * latents["metabolic_delirium_bridge"]
            - 0.20 * inputs["medical_stabilization_treatment_support"]
        )

        regional_state = pd.Series(
            {
                "temporal_lobe_proxy": self._clip01(
                    0.40 * latents["neuronal_hyperexcitability"]
                    + 0.30 * latents["frontotemporal_network_disruption"]
                    + 0.15 * latents["glutamate_gaba_imbalance"]
                    + 0.15 * latents["distributed_perception_thought_dysregulation"]
                ),
                "frontal_lobe_proxy": self._clip01(
                    0.45 * latents["frontotemporal_network_disruption"]
                    + 0.20 * latents["distributed_perception_thought_dysregulation"]
                    + 0.15 * latents["subcortical_reality_filtering_disruption"]
                    + 0.10 * inputs["traumatic_brain_injury_load"]
                    + 0.10 * latents["psychosis_threshold_lowering"]
                ),
                "occipital_association_proxy": self._clip01(
                    0.40 * latents["focal_perceptual_release"]
                    + 0.25 * inputs["cerebrovascular_lesion_load"]
                    + 0.20 * latents["distributed_perception_thought_dysregulation"]
                    + 0.15 * latents["metabolic_delirium_bridge"]
                ),
                "subcortical_thalamic_proxy": self._clip01(
                    0.45 * latents["subcortical_reality_filtering_disruption"]
                    + 0.20 * latents["metabolic_delirium_bridge"]
                    + 0.15 * inputs["cerebrovascular_lesion_load"]
                    + 0.10 * latents["distributed_perception_thought_dysregulation"]
                    + 0.10 * latents["psychosis_threshold_lowering"]
                ),
            },
            name="regional_state",
        )

        symptoms = pd.Series(
            {
                "hallucinations": self._clip01(
                    0.35 * regional_state["occipital_association_proxy"]
                    + 0.30 * regional_state["temporal_lobe_proxy"]
                    + 0.20 * latents["focal_perceptual_release"]
                    + 0.15 * latents["metabolic_delirium_bridge"]
                ),
                "delusional_paranoid_ideation": self._clip01(
                    0.35 * regional_state["subcortical_thalamic_proxy"]
                    + 0.25 * regional_state["frontal_lobe_proxy"]
                    + 0.20 * latents["distributed_perception_thought_dysregulation"]
                    + 0.20 * latents["psychosis_threshold_lowering"]
                ),
                "thought_disorganization": self._clip01(
                    0.35 * regional_state["frontal_lobe_proxy"]
                    + 0.25 * latents["distributed_perception_thought_dysregulation"]
                    + 0.20 * regional_state["subcortical_thalamic_proxy"]
                    + 0.20 * latents["metabolic_delirium_bridge"]
                ),
                "affective_behavioral_dysregulation": self._clip01(
                    0.35 * regional_state["frontal_lobe_proxy"]
                    + 0.20 * regional_state["temporal_lobe_proxy"]
                    + 0.20 * regional_state["subcortical_thalamic_proxy"]
                    + 0.15 * latents["frontotemporal_network_disruption"]
                    + 0.10 * latents["psychosis_threshold_lowering"]
                ),
                "fluctuating_confusional_psychosis": self._clip01(
                    0.45 * latents["metabolic_delirium_bridge"]
                    + 0.20 * regional_state["subcortical_thalamic_proxy"]
                    + 0.15 * latents["glutamate_gaba_imbalance"]
                    + 0.10 * inputs["neuroinflammatory_systemic_illness_load"]
                    + 0.10 * inputs["cerebrovascular_lesion_load"]
                ),
            },
            name="symptoms",
        )
        symptoms.loc["global_psychosis_burden"] = self._clip01(
            0.22 * symptoms["hallucinations"]
            + 0.22 * symptoms["delusional_paranoid_ideation"]
            + 0.20 * symptoms["thought_disorganization"]
            + 0.18 * symptoms["affective_behavioral_dysregulation"]
            + 0.18 * symptoms["fluctuating_confusional_psychosis"]
        )

        phenotypes = pd.Series(
            {
                "epilepsy_temporal_psychosis_profile": self._clip01(
                    0.35 * inputs["epilepsy_hyperexcitability_load"]
                    + 0.25 * latents["neuronal_hyperexcitability"]
                    + 0.20 * regional_state["temporal_lobe_proxy"]
                    + 0.20 * symptoms["hallucinations"]
                ),
                "post_tbi_frontotemporal_psychosis_profile": self._clip01(
                    0.35 * inputs["traumatic_brain_injury_load"]
                    + 0.25 * latents["frontotemporal_network_disruption"]
                    + 0.20 * regional_state["frontal_lobe_proxy"]
                    + 0.20 * symptoms["thought_disorganization"]
                ),
                "lesion_related_hallucinosis_profile": self._clip01(
                    0.35 * inputs["cerebrovascular_lesion_load"]
                    + 0.25 * regional_state["occipital_association_proxy"]
                    + 0.20 * latents["focal_perceptual_release"]
                    + 0.20 * symptoms["hallucinations"]
                ),
                "delirious_medical_psychosis_profile": self._clip01(
                    0.35 * latents["metabolic_delirium_bridge"]
                    + 0.20 * inputs["metabolic_encephalopathy_hypoxia_load"]
                    + 0.15 * inputs["neuroinflammatory_systemic_illness_load"]
                    + 0.15 * symptoms["fluctuating_confusional_psychosis"]
                    + 0.15 * symptoms["global_psychosis_burden"]
                ),
                "second_hit_vulnerability_profile": self._clip01(
                    0.35 * inputs["genetic_psychosis_liability"]
                    + 0.20 * latents["psychosis_threshold_lowering"]
                    + 0.15 * inputs["traumatic_brain_injury_load"]
                    + 0.15 * inputs["cerebrovascular_lesion_load"]
                    + 0.15 * symptoms["global_psychosis_burden"]
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

    model = PsychoticDisorderDueToAnotherMedicalConditionModel()
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
        epilepsy_hyperexcitability_load=0.55,
        metabolic_encephalopathy_hypoxia_load=0.20,
        neuroinflammatory_systemic_illness_load=0.10,
        traumatic_brain_injury_load=0.25,
        cerebrovascular_lesion_load=0.15,
        genetic_psychosis_liability=0.35,
        medical_stabilization_treatment_support=0.30,
    )

    print("\n=== Simulation ===")
    for name, series in sim.items():
        print(f"\n{name.upper()}")
        print(series.to_string())

    if SIIBRA_AVAILABLE:
        print("\n=== Example region suggestions for 'temporal' ===")
        print(model.suggest_regions("temporal", limit=10).to_string(index=False))

        # Example coordinate assignment:
        # print(model.assign_mni_point((-32, -22, -12)).head().to_string(index=False))
