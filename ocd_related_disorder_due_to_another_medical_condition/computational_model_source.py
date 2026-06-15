from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Obsessive-Compulsive and Related Disorder
Due to Another Medical Condition.

This script translates a chapter-level biological narrative into a transparent,
reusable mechanistic model. It is intended for research scaffolding and model
inspection, not diagnosis or treatment.

Key modeling choices for this chapter:
- Treat the syndrome as a medically precipitated OCRD phenotype rather than a
  primary OCD model.
- Keep glutamate, GABA, norepinephrine, HPA-axis, and epigenetic processes as
  latent biology unless the chapter clearly localizes them.
- Anchor the named CSTC circuit conservatively with OFC, ACC, caudate, and a
  thalamic proxy, and add amygdala, hippocampus, and cerebellar proxies as
  associated regions because the chapter discusses them as secondary but
  plausible contributors.
- Tolerate partial atlas / multimodal feature availability and siibra API
  variation.
"""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


OBSESSIVE_COMPULSIVE_RELATED_DISORDER_DUE_TO_ANOTHER_MEDICAL_CONDITION_GENE_PANEL = [
    "SLC1A1",
    "GRIN2B",
    "GRM5",
    "GAD1",
    "GABRA2",
    "SLC6A4",
    "HTR2A",
    "SLC6A2",
    "COMT",
    "BDNF",
    "FKBP5",
    "NR3C1",
    "CRHR1",
    "DLGAP3",
]


class ObsessiveCompulsiveRelatedDisorderDueToAnotherMedicalConditionModel:
    """
    Mechanistic siibra scaffold for medically precipitated obsessive-compulsive
    and related symptoms.

    Conceptual flow:
        inputs -> latent biology -> regional burden/state -> symptoms -> phenotypes

    This is a conservative interpretation of the supplied chapter. The chapter
    emphasizes medical-causal attribution, glutamatergic CSTC hyperactivity,
    reduced GABAergic restraint, stress-related epigenetic change,
    noradrenergic arousal, and associated limbic / hippocampal / cerebellar
    contributors.

    Important limitation:
    The chapter summarizes a heterogeneous syndrome caused by many possible
    medical conditions. This scaffold therefore models a shared pathophysiology
    template rather than a single lesion-specific disease entity.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
        aggregate_connectivity_subjects: int = 8,
    ) -> None:
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort
        self.aggregate_connectivity_subjects = max(1, int(aggregate_connectivity_subjects))

        # Compatibility-first atlas/parcellation/space lookup.
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

        self.disorder_name = "Obsessive-Compulsive and Related Disorder Due to Another Medical Condition"
        self.disorder_key = "ocrd_due_to_another_medical_condition"

        # Direct anchors where the chapter is specific; proxies where the
        # chapter describes systems, loops, or regions more generically.
        self.region_candidates: Dict[str, List[str]] = {
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Fo4 left",
                "Fo3 left",
                "orbitofrontal cortex left",
                "orbitofrontal cortex",
            ],
            "acc": [
                "Area p24ab left",
                "Area a24pr left",
                "Area p32 left",
                "anterior cingulate cortex left",
                "anterior cingulate cortex",
            ],
            "caudate_proxy": [
                "caudate nucleus left",
                "caudate left",
                "caudate nucleus",
                "caudate",
            ],
            "thalamus_proxy": [
                "thalamus left",
                "thalamus",
                "mediodorsal thalamus left",
                "mediodorsal thalamus",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "SF (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 left",
                "CA left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "cerebellum_proxy": [
                "cerebellum left",
                "cerebellum",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Background vulnerability in neurotransmitter regulation, plasticity, and stress responsivity"
            ),
            "focal_neurological_insult": (
                "Stroke, traumatic brain injury, neurodegeneration, or other focal neurological burden"
            ),
            "temporal_limbic_seizure_burden": (
                "Epileptic or limbic irritative burden capable of disturbing contextual and autonomic processing"
            ),
            "autoimmune_infectious_cns_process": (
                "Autoimmune, inflammatory, or infectious CNS process affecting distributed brain function"
            ),
            "systemic_metabolic_disturbance": (
                "Systemic metabolic or physiologic disturbance capable of destabilizing excitatory-inhibitory balance"
            ),
            "chronic_medical_stress": (
                "Sustained stress load related to medical illness, disability, uncertainty, or chronic physiological burden"
            ),
            "underlying_condition_treatment_support": (
                "Protective correction or stabilization of the precipitating medical condition"
            ),
            "behavioral_control_support": (
                "Protective executive and coping support that improves symptom containment"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "lesion_network_disconnection": (
                "Distributed network disruption produced by medical insult, seizure burden, or CNS disease"
            ),
            "glutamatergic_cstc_hyperactivity": (
                "Hyperexcitable glutamatergic signaling within cortico-striato-thalamo-cortical loops"
            ),
            "gabaergic_inhibitory_failure": (
                "Reduced inhibitory restraint that permits CSTC overactivity and compulsive output"
            ),
            "noradrenergic_arousal_dysregulation": (
                "Stress-linked arousal and vigilance dysregulation that amplifies anxiety and salience"
            ),
            "hpa_axis_epigenetic_sensitization": (
                "Stress-mediated endocrine / epigenetic sensitization sustaining altered network responsivity"
            ),
            "cstc_loop_dysfunction": (
                "Net impairment of cortico-striato-thalamo-cortical gating, selection, and termination processes"
            ),
            "frontolimbic_threat_bias": (
                "Amygdala-cingulate threat loading with insufficient top-down contextual regulation"
            ),
            "contextual_memory_triggering": (
                "Hippocampal / medial-temporal contextual cueing that can bind symptoms to specific triggers"
            ),
            "intrusive_salience_pressure": (
                "Persistent internally generated error / threat / salience signals promoting obsessions"
            ),
            "compulsive_action_gating_failure": (
                "Failure to inhibit or terminate repetitive acts once intrusive pressure is generated"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "intrusive_obsessions": "Recurrent intrusive thoughts, impulses, or preoccupations",
            "compulsive_rituals": "Repetitive behaviors or mental acts aimed at reducing distress",
            "anxiety_distress": "Aversive anxiety and distress surrounding symptoms or triggers",
            "response_inhibition_failure": "Failure to suppress or disengage from compulsive responding",
            "contextual_trigger_sensitivity": "Symptoms becoming linked to contextual or trauma-associated cues",
            "motoric_compulsion_pressure": "Motor urge and repetitive-action pressure associated with compulsive output",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "glutamatergic_cstc_hyperactivity",
                "relation": "can bias baseline excitatory CSTC tone upward",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "hpa_axis_epigenetic_sensitization",
                "relation": "can increase vulnerability to stress-mediated long-term regulatory change",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "focal_neurological_insult",
                "target": "lesion_network_disconnection",
                "relation": "directly disrupts distributed control circuits after TBI, stroke, or neurodegeneration",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "focal_neurological_insult",
                "target": "glutamatergic_cstc_hyperactivity",
                "relation": "can destabilize glutamatergic homeostasis and produce CSTC overdrive",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "temporal_limbic_seizure_burden",
                "target": "lesion_network_disconnection",
                "relation": "disrupts limbic and contextual processing networks",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "temporal_limbic_seizure_burden",
                "target": "contextual_memory_triggering",
                "relation": "can bind symptoms to temporal-limbic and contextual cue processing",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "temporal_limbic_seizure_burden",
                "target": "frontolimbic_threat_bias",
                "relation": "can increase limbic emotional loading and autonomic salience",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "autoimmune_infectious_cns_process",
                "target": "lesion_network_disconnection",
                "relation": "can create distributed inflammatory or infectious network burden",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "autoimmune_infectious_cns_process",
                "target": "glutamatergic_cstc_hyperactivity",
                "relation": "can disturb excitatory balance within OCD-relevant circuits",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "systemic_metabolic_disturbance",
                "target": "glutamatergic_cstc_hyperactivity",
                "relation": "can promote excitotoxic or excitatory-inhibitory imbalance",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "systemic_metabolic_disturbance",
                "target": "gabaergic_inhibitory_failure",
                "relation": "can reduce inhibitory buffering against CSTC overactivity",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "chronic_medical_stress",
                "target": "hpa_axis_epigenetic_sensitization",
                "relation": "can induce durable epigenetic and endocrine recalibration of stress systems",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "chronic_medical_stress",
                "target": "noradrenergic_arousal_dysregulation",
                "relation": "elevates stress-linked vigilance and arousal signaling",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "glutamatergic_cstc_hyperactivity",
                "target": "cstc_loop_dysfunction",
                "relation": "drives persistent hyperactivity of OCD-relevant CSTC loops",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "gabaergic_inhibitory_failure",
                "target": "cstc_loop_dysfunction",
                "relation": "removes inhibitory control over repetitive circuit output",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "noradrenergic_arousal_dysregulation",
                "target": "frontolimbic_threat_bias",
                "relation": "amplifies anxiety-linked emotional salience",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "hpa_axis_epigenetic_sensitization",
                "target": "frontolimbic_threat_bias",
                "relation": "sustains exaggerated stress responsivity in limbic-control networks",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "lesion_network_disconnection",
                "target": "cstc_loop_dysfunction",
                "relation": "disconnects or dyscoordinates core selection and gating circuits",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "cstc_loop_dysfunction",
                "target": "ofc",
                "relation": "maps network overactivity onto orbitofrontal error and valuation circuitry",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "cstc_loop_dysfunction",
                "target": "acc",
                "relation": "maps network overactivity onto cingulate monitoring and distress circuitry",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "cstc_loop_dysfunction",
                "target": "caudate_proxy",
                "relation": "maps network dysregulation onto striatal selection and habit circuitry",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "cstc_loop_dysfunction",
                "target": "thalamus_proxy",
                "relation": "maps loop dysregulation onto thalamocortical relay burden",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "frontolimbic_threat_bias",
                "target": "amygdala",
                "relation": "loads fear and salience processing circuitry",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "contextual_memory_triggering",
                "target": "hippocampus",
                "relation": "loads medial-temporal contextual memory circuits",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "compulsive_action_gating_failure",
                "target": "cerebellum_proxy",
                "relation": "can recruit motor-pattern support systems for repetitive acts",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "cstc_loop_dysfunction",
                "target": "intrusive_salience_pressure",
                "relation": "transforms network hyperactivity into persistent internal error / threat signaling",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "frontolimbic_threat_bias",
                "target": "intrusive_salience_pressure",
                "relation": "adds anxiety and emotional urgency to intrusive signal generation",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "intrusive_salience_pressure",
                "target": "intrusive_obsessions",
                "relation": "promotes intrusive obsessive thought content",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "cstc_loop_dysfunction",
                "target": "compulsive_action_gating_failure",
                "relation": "impairs action termination and repetitive behavior control",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "compulsive_action_gating_failure",
                "target": "compulsive_rituals",
                "relation": "permits repetitive behaviors and mental rituals to occur",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "frontolimbic_threat_bias",
                "target": "anxiety_distress",
                "relation": "increases distress that drives compulsive relief-seeking",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "contextual_memory_triggering",
                "target": "contextual_trigger_sensitivity",
                "relation": "links symptoms to trauma-associated or situational cues",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "compulsive_action_gating_failure",
                "target": "response_inhibition_failure",
                "relation": "reduces the ability to suppress repetitive responses",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "compulsive_action_gating_failure",
                "target": "motoric_compulsion_pressure",
                "relation": "can generate repetitive motoric pressure associated with compulsions",
                "ocrd_medical_change": "increased",
            },
            {
                "source": "underlying_condition_treatment_support",
                "target": "lesion_network_disconnection",
                "relation": "can partially reduce medically driven network burden when the precipitating condition is stabilized",
                "ocrd_medical_change": "decreased",
            },
            {
                "source": "underlying_condition_treatment_support",
                "target": "cstc_loop_dysfunction",
                "relation": "can reduce the circuit consequences of the precipitating medical process",
                "ocrd_medical_change": "decreased",
            },
            {
                "source": "behavioral_control_support",
                "target": "compulsive_action_gating_failure",
                "relation": "can strengthen compensatory inhibitory control and symptom containment",
                "ocrd_medical_change": "decreased",
            },
            {
                "source": "behavioral_control_support",
                "target": "anxiety_distress",
                "relation": "can reduce downstream distress and compulsive escalation",
                "ocrd_medical_change": "decreased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        self._build_cache: Optional[dict] = None
        self._pmap = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _mean_clip(values: Iterable[float]) -> float:
        vals = [float(v) for v in values]
        if not vals:
            return 0.0
        return max(0.0, min(1.0, round(sum(vals) / len(vals), 4)))

    @staticmethod
    def _series_from(mapping: Dict[str, float], name: str) -> pd.Series:
        return pd.Series(mapping, name=name, dtype=float)

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
        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "orbitofrontal cortex",
            "anterior cingulate cortex",
            "thalamus",
            "cerebellum",
            "caudate nucleus",
        } else 0
        subregion_bonus = 0 if any(tag in name for tag in ["area ", "lb", "sf", "cm", "ca1", "subiculum", "24", "32", "fo"]) else 1
        return (left_bonus, right_penalty, generic_penalty, subregion_bonus)

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
            item = (
                self._name_of(region),
                getattr(region, "identifier", None),
                getattr(getattr(region, "parcellation", None), "name", ""),
            )
            if item in seen:
                continue
            seen.add(item)
            rows.append(
                {
                    "name": item[0],
                    "identifier": item[1],
                    "parcellation": item[2],
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
        volume_mm3 = getattr(main, "volume", None)
        try:
            volume_mm3 = float(volume_mm3) if volume_mm3 is not None else None
        except Exception:
            volume_mm3 = None
        return centroid_xyz, volume_mm3

    def _extract_tabular_data(self, feature: Any) -> pd.DataFrame:
        try:
            data = getattr(feature, "data", None)
            if isinstance(data, pd.DataFrame):
                return data.copy()
            if isinstance(data, pd.Series):
                return data.to_frame().reset_index(drop=False)
            if isinstance(data, dict):
                return pd.DataFrame(data)
        except Exception:
            pass
        return pd.DataFrame()

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        try:
            df = self._extract_tabular_data(feats[0]).reset_index(drop=False)
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
            df = self._extract_tabular_data(feats[0])
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        lower_cols = {str(c).lower(): c for c in df.columns}
        required = {"gene", "level", "zscore"}
        if required.issubset(lower_cols):
            gene_col = lower_cols["gene"]
            level_col = lower_cols["level"]
            zscore_col = lower_cols["zscore"]
            try:
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
                )
                return out.reset_index(drop=True)
            except Exception:
                return df.reset_index(drop=True)
        return df.reset_index(drop=True)

    def _candidate_connectivity_feature(self) -> Any:
        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            return None
        cohort_match = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), None)
        return cohort_match if cohort_match is not None else feats[0]

    def _matrix_like(self, obj: Any) -> pd.DataFrame:
        try:
            data = getattr(obj, "data", None)
            if isinstance(data, pd.DataFrame):
                return data.copy()
        except Exception:
            pass
        return pd.DataFrame()

    def _average_matrices(self, matrices: Sequence[pd.DataFrame]) -> pd.DataFrame:
        usable = [m for m in matrices if isinstance(m, pd.DataFrame) and not m.empty]
        if not usable:
            return pd.DataFrame()
        if len(usable) == 1:
            return usable[0].copy()
        try:
            running = usable[0].copy()
            for m in usable[1:]:
                running = running.add(m, fill_value=0.0)
            return running / float(len(usable))
        except Exception:
            return usable[0].copy()

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feat = self._candidate_connectivity_feature()
        if feat is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        matrices: List[pd.DataFrame] = []

        # Connectivity queries may return compound features containing
        # subject-level elements. Average a small subset for a stable scaffold,
        # but gracefully fall back to the first accessible matrix.
        try:
            for idx, element in enumerate(feat):
                matrices.append(self._matrix_like(element))
                if idx + 1 >= self.aggregate_connectivity_subjects:
                    break
        except Exception:
            pass

        if not matrices:
            direct = self._matrix_like(feat)
            if not direct.empty:
                matrices = [direct]

        self._connectivity_matrix = self._average_matrices(matrices)
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        exact = [x for x in labels if self._name_of(x) == getattr(region, "name", self._name_of(region))]
        if exact:
            return exact[0]

        rn = getattr(region, "name", self._name_of(region)).lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        if fuzzy:
            return fuzzy[0]

        region_tokens = {t for t in rn.replace("(", " ").replace(")", " ").replace("-", " ").split() if len(t) > 2}
        best = None
        best_score = 0
        for label in labels:
            ln = self._name_of(label).lower()
            label_tokens = {t for t in ln.replace("(", " ").replace(")", " ").replace("-", " ").split() if len(t) > 2}
            score = len(region_tokens & label_tokens)
            if score > best_score:
                best = label
                best_score = score
        return best if best_score > 0 else None

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
            df = df[df["connected_region"] != getattr(region, "name", self._name_of(region))]
            return df.head(max_rows).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Public atlas helpers
    # ------------------------------------------------------------------
    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if self._pmap is None:
            with siibra.QUIET:
                self._pmap = siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=self.assignment_space,
                    maptype="statistical",
                )
        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        if isinstance(assignments, pd.DataFrame):
            for candidate in ("map value", "correlation", "intersection over union"):
                if candidate in assignments.columns:
                    return assignments.sort_values(candidate, ascending=False).reset_index(drop=True)
        return assignments

    def region_mask(self, node_key: str) -> Any:
        region = self.region_objects.get(node_key)
        if region is None:
            raise KeyError(f"No resolved region object for node '{node_key}'")

        for method_name in ("get_regional_mask", "fetch_regional_map"):
            method = getattr(region, method_name, None)
            if method is None:
                continue
            try:
                return method(space=self.assignment_space, maptype="labelled")
            except TypeError:
                try:
                    return method(self.assignment_space)
                except Exception:
                    continue
            except Exception:
                continue
        raise RuntimeError(f"Could not fetch a region mask for node '{node_key}'")

    # ------------------------------------------------------------------
    # Build and connectivity summaries
    # ------------------------------------------------------------------
    def circuit_connectivity(self, max_rows_per_seed: int = 10) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        labels_index = list(matrix.index)
        labels_columns = list(matrix.columns)

        for seed_key, seed_region in self.region_objects.items():
            seed_label = self._match_region_label(labels_index, seed_region)
            axis = "index"
            if seed_label is None:
                seed_label = self._match_region_label(labels_columns, seed_region)
                axis = "columns"
            if seed_label is None:
                continue

            try:
                series = matrix.loc[seed_label] if axis == "index" else matrix[seed_label]
                series = series.sort_values(ascending=False)
            except Exception:
                continue

            taken = 0
            for target_label, value in series.items():
                target_name = self._name_of(target_label)
                if target_name == getattr(seed_region, "name", self._name_of(seed_region)):
                    continue
                target_key = None
                for k, region in self.region_objects.items():
                    if self._name_of(region) == target_name:
                        target_key = k
                        break
                rows.append(
                    {
                        "seed_key": seed_key,
                        "seed_region": getattr(seed_region, "name", self._name_of(seed_region)),
                        "target_key": target_key,
                        "target_region": target_name,
                        "value": float(value),
                    }
                )
                taken += 1
                if taken >= max_rows_per_seed:
                    break

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["seed_key", "value"], ascending=[True, False]).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = OBSESSIVE_COMPULSIVE_RELATED_DISORDER_DUE_TO_ANOTHER_MEDICAL_CONDITION_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
        nodes: List[Dict[str, Any]] = []

        # Reset cached per-build state so repeated calls remain deterministic.
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
                        "label": key.upper(),
                        "node_type": "region",
                        "description": "Atlas-backed node (unresolved in this environment)",
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
                    "label": getattr(region, "name", self._name_of(region)),
                    "node_type": "region",
                    "description": "Atlas-backed circuit node",
                    "atlas_region": getattr(region, "name", self._name_of(region)),
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
        circuit_df = self.circuit_connectivity(max_rows_per_seed=connectivity_rows)

        self._build_cache = {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": circuit_df,
        }
        return self._build_cache

    # ------------------------------------------------------------------
    # Simulator
    # ------------------------------------------------------------------
    def simulate(
        self,
        genetic_vulnerability: float = 0.35,
        focal_neurological_insult: float = 0.35,
        temporal_limbic_seizure_burden: float = 0.20,
        autoimmune_infectious_cns_process: float = 0.20,
        systemic_metabolic_disturbance: float = 0.20,
        chronic_medical_stress: float = 0.35,
        underlying_condition_treatment_support: float = 0.15,
        behavioral_control_support: float = 0.10,
    ) -> Dict[str, pd.Series]:
        inp = {
            "genetic_vulnerability": self._clip01(genetic_vulnerability),
            "focal_neurological_insult": self._clip01(focal_neurological_insult),
            "temporal_limbic_seizure_burden": self._clip01(temporal_limbic_seizure_burden),
            "autoimmune_infectious_cns_process": self._clip01(autoimmune_infectious_cns_process),
            "systemic_metabolic_disturbance": self._clip01(systemic_metabolic_disturbance),
            "chronic_medical_stress": self._clip01(chronic_medical_stress),
            "underlying_condition_treatment_support": self._clip01(underlying_condition_treatment_support),
            "behavioral_control_support": self._clip01(behavioral_control_support),
        }

        lat: Dict[str, float] = {}
        lat["lesion_network_disconnection"] = self._clip01(
            0.34 * inp["focal_neurological_insult"]
            + 0.22 * inp["temporal_limbic_seizure_burden"]
            + 0.18 * inp["autoimmune_infectious_cns_process"]
            + 0.10 * inp["systemic_metabolic_disturbance"]
            + 0.08 * inp["genetic_vulnerability"]
            - 0.18 * inp["underlying_condition_treatment_support"]
        )
        lat["glutamatergic_cstc_hyperactivity"] = self._clip01(
            0.28 * inp["focal_neurological_insult"]
            + 0.22 * inp["systemic_metabolic_disturbance"]
            + 0.18 * inp["autoimmune_infectious_cns_process"]
            + 0.16 * inp["genetic_vulnerability"]
            + 0.10 * inp["temporal_limbic_seizure_burden"]
            - 0.10 * inp["underlying_condition_treatment_support"]
        )
        lat["gabaergic_inhibitory_failure"] = self._clip01(
            0.30 * lat["glutamatergic_cstc_hyperactivity"]
            + 0.18 * inp["systemic_metabolic_disturbance"]
            + 0.16 * inp["temporal_limbic_seizure_burden"]
            + 0.12 * inp["focal_neurological_insult"]
            + 0.08 * inp["genetic_vulnerability"]
            - 0.08 * inp["underlying_condition_treatment_support"]
        )
        lat["noradrenergic_arousal_dysregulation"] = self._clip01(
            0.34 * inp["chronic_medical_stress"]
            + 0.18 * inp["autoimmune_infectious_cns_process"]
            + 0.12 * inp["focal_neurological_insult"]
            + 0.10 * inp["genetic_vulnerability"]
            - 0.12 * inp["behavioral_control_support"]
        )
        lat["hpa_axis_epigenetic_sensitization"] = self._clip01(
            0.38 * inp["chronic_medical_stress"]
            + 0.16 * inp["genetic_vulnerability"]
            + 0.12 * inp["systemic_metabolic_disturbance"]
            + 0.10 * inp["autoimmune_infectious_cns_process"]
            - 0.14 * inp["behavioral_control_support"]
        )
        lat["cstc_loop_dysfunction"] = self._clip01(
            0.28 * lat["glutamatergic_cstc_hyperactivity"]
            + 0.22 * lat["gabaergic_inhibitory_failure"]
            + 0.20 * lat["lesion_network_disconnection"]
            + 0.10 * lat["noradrenergic_arousal_dysregulation"]
            - 0.16 * inp["underlying_condition_treatment_support"]
            - 0.10 * inp["behavioral_control_support"]
        )
        lat["frontolimbic_threat_bias"] = self._clip01(
            0.28 * lat["hpa_axis_epigenetic_sensitization"]
            + 0.24 * lat["noradrenergic_arousal_dysregulation"]
            + 0.16 * inp["temporal_limbic_seizure_burden"]
            + 0.12 * lat["lesion_network_disconnection"]
            - 0.12 * inp["behavioral_control_support"]
        )
        lat["contextual_memory_triggering"] = self._clip01(
            0.30 * inp["temporal_limbic_seizure_burden"]
            + 0.18 * lat["frontolimbic_threat_bias"]
            + 0.16 * lat["hpa_axis_epigenetic_sensitization"]
            + 0.12 * inp["autoimmune_infectious_cns_process"]
            + 0.10 * lat["lesion_network_disconnection"]
            - 0.08 * inp["behavioral_control_support"]
        )
        lat["intrusive_salience_pressure"] = self._clip01(
            0.34 * lat["cstc_loop_dysfunction"]
            + 0.24 * lat["frontolimbic_threat_bias"]
            + 0.14 * lat["glutamatergic_cstc_hyperactivity"]
            + 0.10 * lat["contextual_memory_triggering"]
            - 0.08 * inp["behavioral_control_support"]
        )
        lat["compulsive_action_gating_failure"] = self._clip01(
            0.34 * lat["cstc_loop_dysfunction"]
            + 0.22 * lat["gabaergic_inhibitory_failure"]
            + 0.16 * lat["intrusive_salience_pressure"]
            + 0.10 * lat["lesion_network_disconnection"]
            - 0.18 * inp["behavioral_control_support"]
            - 0.10 * inp["underlying_condition_treatment_support"]
        )

        regional: Dict[str, float] = {}
        regional["ofc"] = self._clip01(
            0.34 * lat["cstc_loop_dysfunction"]
            + 0.22 * lat["intrusive_salience_pressure"]
            + 0.14 * lat["glutamatergic_cstc_hyperactivity"]
            + 0.10 * lat["lesion_network_disconnection"]
        )
        regional["acc"] = self._clip01(
            0.30 * lat["cstc_loop_dysfunction"]
            + 0.24 * lat["frontolimbic_threat_bias"]
            + 0.14 * lat["hpa_axis_epigenetic_sensitization"]
            + 0.10 * lat["intrusive_salience_pressure"]
        )
        regional["caudate_proxy"] = self._clip01(
            0.30 * lat["cstc_loop_dysfunction"]
            + 0.20 * lat["gabaergic_inhibitory_failure"]
            + 0.16 * lat["compulsive_action_gating_failure"]
            + 0.12 * lat["lesion_network_disconnection"]
        )
        regional["thalamus_proxy"] = self._clip01(
            0.28 * lat["cstc_loop_dysfunction"]
            + 0.20 * lat["glutamatergic_cstc_hyperactivity"]
            + 0.16 * lat["lesion_network_disconnection"]
            + 0.10 * lat["intrusive_salience_pressure"]
        )
        regional["amygdala"] = self._clip01(
            0.34 * lat["frontolimbic_threat_bias"]
            + 0.22 * lat["noradrenergic_arousal_dysregulation"]
            + 0.18 * lat["hpa_axis_epigenetic_sensitization"]
            + 0.10 * lat["contextual_memory_triggering"]
        )
        regional["hippocampus"] = self._clip01(
            0.34 * lat["contextual_memory_triggering"]
            + 0.18 * lat["hpa_axis_epigenetic_sensitization"]
            + 0.14 * inp["temporal_limbic_seizure_burden"]
            + 0.10 * lat["frontolimbic_threat_bias"]
        )
        regional["cerebellum_proxy"] = self._clip01(
            0.22 * lat["compulsive_action_gating_failure"]
            + 0.18 * lat["lesion_network_disconnection"]
            + 0.16 * lat["cstc_loop_dysfunction"]
            + 0.10 * inp["focal_neurological_insult"]
        )

        symptoms: Dict[str, float] = {}
        symptoms["intrusive_obsessions"] = self._clip01(
            0.36 * lat["intrusive_salience_pressure"]
            + 0.22 * regional["ofc"]
            + 0.18 * regional["amygdala"]
            + 0.12 * regional["acc"]
        )
        symptoms["compulsive_rituals"] = self._clip01(
            0.34 * lat["compulsive_action_gating_failure"]
            + 0.22 * regional["caudate_proxy"]
            + 0.18 * symptoms["intrusive_obsessions"]
            + 0.12 * regional["thalamus_proxy"]
            - 0.10 * inp["behavioral_control_support"]
        )
        symptoms["anxiety_distress"] = self._clip01(
            0.34 * regional["amygdala"]
            + 0.22 * lat["hpa_axis_epigenetic_sensitization"]
            + 0.20 * symptoms["intrusive_obsessions"]
            + 0.10 * regional["acc"]
            - 0.10 * inp["behavioral_control_support"]
        )
        symptoms["response_inhibition_failure"] = self._clip01(
            0.30 * lat["compulsive_action_gating_failure"]
            + 0.24 * regional["ofc"]
            + 0.18 * regional["acc"]
            + 0.12 * regional["caudate_proxy"]
            - 0.16 * inp["behavioral_control_support"]
        )
        symptoms["contextual_trigger_sensitivity"] = self._clip01(
            0.32 * regional["hippocampus"]
            + 0.24 * regional["amygdala"]
            + 0.18 * lat["contextual_memory_triggering"]
            + 0.10 * symptoms["anxiety_distress"]
        )
        symptoms["motoric_compulsion_pressure"] = self._clip01(
            0.30 * regional["caudate_proxy"]
            + 0.22 * regional["cerebellum_proxy"]
            + 0.18 * lat["compulsive_action_gating_failure"]
            + 0.14 * symptoms["compulsive_rituals"]
        )

        phenotypes = {
            "medical_cstc_ocd_profile": self._mean_clip(
                [
                    lat["cstc_loop_dysfunction"],
                    regional["ofc"],
                    regional["caudate_proxy"],
                    symptoms["compulsive_rituals"],
                ]
            ),
            "limbic_contextual_ocd_profile": self._mean_clip(
                [
                    lat["frontolimbic_threat_bias"],
                    regional["amygdala"],
                    regional["hippocampus"],
                    symptoms["contextual_trigger_sensitivity"],
                ]
            ),
            "stress_sensitized_ocd_profile": self._mean_clip(
                [
                    inp["chronic_medical_stress"],
                    lat["noradrenergic_arousal_dysregulation"],
                    lat["hpa_axis_epigenetic_sensitization"],
                    symptoms["anxiety_distress"],
                ]
            ),
            "focal_lesion_ocd_profile": self._mean_clip(
                [
                    inp["focal_neurological_insult"],
                    lat["lesion_network_disconnection"],
                    lat["intrusive_salience_pressure"],
                    symptoms["response_inhibition_failure"],
                ]
            ),
            "mixed_medical_ocd_profile": self._mean_clip(
                [
                    symptoms["intrusive_obsessions"],
                    symptoms["compulsive_rituals"],
                    symptoms["anxiety_distress"],
                    symptoms["response_inhibition_failure"],
                ]
            ),
        }

        return {
            "inputs": self._series_from(inp, "inputs"),
            "latents": self._series_from(lat, "latents"),
            "regional_state": self._series_from(regional, "regional_state"),
            "symptoms": self._series_from(symptoms, "symptoms"),
            "phenotypes": self._series_from(phenotypes, "phenotypes"),
        }


if __name__ == "__main__":
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)

    model = ObsessiveCompulsiveRelatedDisorderDueToAnotherMedicalConditionModel()
    bundle = model.build()

    print("\n=== NODE TABLE (key fields) ===")
    print(
        bundle["nodes"][["key", "node_type", "atlas_region", "region_identifier", "feature_summary"]]
        .fillna("")
        .to_string(index=False)
    )

    print("\n=== EDGE TABLE ===")
    print(bundle["edges"].to_string(index=False))

    for region_key in [
        "ofc",
        "acc",
        "caudate_proxy",
        "thalamus_proxy",
        "amygdala",
        "hippocampus",
        "cerebellum_proxy",
    ]:
        receptor_df = bundle["receptors"].get(region_key, pd.DataFrame())
        gene_df = bundle["genes"].get(region_key, pd.DataFrame())
        conn_df = bundle["connectivity_profiles"].get(region_key, pd.DataFrame())

        print(f"\n=== REGION: {region_key} ===")
        print("Receptors:")
        print(receptor_df.head(10).to_string(index=False) if not receptor_df.empty else "<no receptor fingerprint available>")
        print("\nGenes:")
        print(gene_df.head(10).to_string(index=False) if not gene_df.empty else "<no gene-expression table available>")
        print("\nConnectivity:")
        print(conn_df.head(10).to_string(index=False) if not conn_df.empty else "<no connectivity profile available>")

    print("\n=== CIRCUIT CONNECTIVITY ===")
    circuit_df = bundle["circuit_connectivity"]
    print(circuit_df.head(25).to_string(index=False) if not circuit_df.empty else "<no circuit connectivity available>")

    example = model.simulate(
        genetic_vulnerability=0.45,
        focal_neurological_insult=0.55,
        temporal_limbic_seizure_burden=0.35,
        autoimmune_infectious_cns_process=0.25,
        systemic_metabolic_disturbance=0.20,
        chronic_medical_stress=0.60,
        underlying_condition_treatment_support=0.25,
        behavioral_control_support=0.20,
    )

    print("\n=== SIMULATION: example lesion-linked stress-sensitized medical OCRD phenotype ===")
    for name, series in example.items():
        print(f"\n{name.upper()}")
        print(series.to_string())

    # Example coordinate usage:
    # assignments = model.assign_mni_point((-6, 44, -10))
    # print(assignments.head(10).to_string(index=False))
