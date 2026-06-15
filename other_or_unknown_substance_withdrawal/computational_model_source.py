from __future__ import annotations

"""
Other (or Unknown) Substance Withdrawal siibra scaffold.

This script turns a chapter-level mechanistic summary of Other (or Unknown)
Substance Withdrawal into an atlas-grounded research scaffold built around
siibra idioms. It is meant for hypothesis generation, teaching, and
reproducible exploration of atlas-backed regional features. It is not a
clinical diagnostic or treatment tool.

Conceptual choices:
- Because the substance is unspecified, the scaffold models shared withdrawal
  biology rather than pretending to know a single receptor-level mechanism for
  every case.
- The chapter centers withdrawal as the unmasking of allostatic adaptations
  after chronic exposure is removed. The simulator therefore starts from
  chronic dependence and abrupt cessation, then flows through shared latent
  mechanisms such as cAMP/CREB-related adaptation, reward hypodopaminergia,
  stress-system hyperreactivity, prefrontal hypofunction, and
  GABA/glutamate-driven cortical hyperexcitability.
- The chapter explicitly names the striatum, amygdala, hippocampus, prefrontal
  cortex, insula, basal ganglia, thalamus, and medial temporal systems. Direct
  anchors are used where the chapter is sufficiently concrete (amygdala,
  hippocampus, insula). Conservative proxies are used where the chapter stays
  systems-level (prefrontal cortex, striatum, basal ganglia, thalamus,
  extended amygdala).
- The gene panel is broad and mechanism-focused rather than disorder-validated:
  dopamine, GABA/glutamate balance, stress responsivity, and cAMP/CREB-linked
  plasticity are included because they are directly aligned with the chapter.

The scaffold degrades gracefully when siibra or particular multimodal features
are unavailable. Atlas-backed nodes remain visible in the graph, but unresolved
feature tables are returned empty instead of raising an error.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception:  # pragma: no cover - optional dependency in some environments
    siibra = None


DEFAULT_GENE_PANEL = [
    # Dopamine and reward-salience biology
    "DRD2",
    "DRD3",
    "SLC6A3",
    "COMT",
    # GABA / glutamate balance and hyperexcitability
    "GABRA1",
    "GABRA2",
    "GAD1",
    "GRIN1",
    "GRIN2A",
    "GRIN2B",
    "SLC1A2",
    # Stress responsivity and cue-reactivity
    "CRHR1",
    "NR3C1",
    "SLC6A4",
    # Signal transduction / transcription / plasticity
    "ADCY1",
    "CREB1",
    "FOSB",
    "BDNF",
]


class OtherOrUnknownSubstanceWithdrawalModel:
    """
    Atlas-grounded research scaffold for Other (or Unknown) Substance Withdrawal.

    The simulator keeps a transparent one-pass causal flow:
        inputs -> latent biology -> regional burden -> symptoms -> phenotypes

    Since the chapter is intentionally trans-substance, the model prioritizes
    shared dependence-and-withdrawal mechanisms instead of medication-specific detail.
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
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "extended_amygdala_proxy": [
                "BST (Bed Nucleus) left",
                "LB (Amygdala) left",
                "SF (Amygdala) left",
                "bed nucleus",
                "extended amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "DG (Hippocampus) left",
                "HC-Subiculum (Hippocampus) left",
                "hippocampus left",
                "hippocampus",
            ],
            "insula": [
                "Area Id2 (Insula) left",
                "Area Id3 (Insula) left",
                "Area Id1 (Insula) left",
                "insula left",
                "insula",
            ],
            "prefrontal_cortex_proxy": [
                "Area 46 left",
                "Area 9 left",
                "Area 45 (IFG) left",
                "Area s32 (sACC) left",
                "prefrontal cortex",
            ],
            "striatum_proxy": [
                "putamen left",
                "caudate nucleus left",
                "striatum",
                "ventral striatum",
                "basal ganglia",
            ],
            "basal_ganglia_proxy": [
                "putamen left",
                "caudate nucleus left",
                "basal ganglia",
                "striatum",
            ],
            "thalamus_proxy": [
                "thalamus",
                "CGM (Metathalamus) left",
                "CGL (Metathalamus) left",
                "metathalamus",
                "thalamus left",
            ],
        }

        self.proxy_region_nodes = {
            "extended_amygdala_proxy",
            "prefrontal_cortex_proxy",
            "striatum_proxy",
            "basal_ganglia_proxy",
            "thalamus_proxy",
        }

        self.region_descriptions: Dict[str, str] = {
            "amygdala": (
                "Direct atlas anchor for reward-memory-emotion circuitry repeatedly "
                "named in cue-reactivity and withdrawal negative affect."
            ),
            "extended_amygdala_proxy": (
                "Conservative proxy for the extended amygdala and related stress "
                "circuitry driving dysphoria, anxiety, and relapse pressure."
            ),
            "hippocampus": (
                "Direct atlas anchor for memory-related circuitry implicated in cue "
                "reactivity, relapse, and withdrawal-associated memory problems."
            ),
            "insula": (
                "Direct atlas anchor for conscious urge awareness and interoceptive "
                "representation of craving."
            ),
            "prefrontal_cortex_proxy": (
                "Conservative proxy for the broad prefrontal control territory cited "
                "as hypofunctional during withdrawal."
            ),
            "striatum_proxy": (
                "Proxy for reward-circuit hypodopaminergia and anhedonia/craving burden "
                "described in withdrawal."
            ),
            "basal_ganglia_proxy": (
                "Proxy for basal-ganglia participation in psychomotor and catatonic "
                "withdrawal phenomena."
            ),
            "thalamus_proxy": (
                "Proxy for thalamic involvement in distributed catatonia and severe "
                "cortical hyperexcitability states."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "chronic_dependence_load": (
                "Degree of chronic dependence-producing substance exposure before withdrawal."
            ),
            "abrupt_cessation": (
                "Extent to which the substance has been suddenly removed, unmasking counteradaptations."
            ),
            "genetic_vulnerability": (
                "Shared heritable vulnerability to addiction, dependence, and withdrawal severity."
            ),
            "developmental_environmental_load": (
                "Environmental and developmental loading that shapes addiction vulnerability."
            ),
            "stress_sensitization": (
                "Stress responsivity that can amplify negative affect and relapse pressure."
            ),
            "cue_exposure": (
                "Exposure to medication-related cues that reactivate reward-memory circuitry during abstinence."
            ),
            "sedative_hypnotic_withdrawal_risk": (
                "Specific loading toward severe GABA/glutamate-rebound states such as seizures or delirium."
            ),
            "psychosis_liability": (
                "Underlying liability for mesolimbic dopaminergic dysregulation and psychotic symptoms."
            ),
            "recovery_support": (
                "Protective recovery structure, supervision, and relapse-prevention support."
            ),
            "medical_stabilization_support": (
                "Protective acute stabilization and monitoring that can buffer severe withdrawal physiology."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "allostatic_counteradaptation": (
                "Pathological homeostatic reset established during chronic substance exposure."
            ),
            "camp_creb_gene_adaptation": (
                "Second-messenger and transcriptional adaptation involving cAMP/CREB-linked plasticity."
            ),
            "reward_hypodopaminergia": (
                "Withdrawal-state reduction in reward-circuit dopamine tone, linked to anhedonia and craving."
            ),
            "stress_system_hyperreactivity": (
                "Overactivation of stress systems and extended amygdala during withdrawal."
            ),
            "white_matter_disconnection": (
                "Reduced efficiency of structural communication contributing to executive dysfunction and poor control."
            ),
            "prefrontal_hypofunction": (
                "Withdrawal-related reduction in prefrontal regulatory control."
            ),
            "frontostriatal_control_failure": (
                "Breakdown of control over reward-driven behavior and impulses."
            ),
            "cue_reactivity_craving_circuit": (
                "Reactivated reward-memory-motivation circuitry that drives craving during abstinence."
            ),
            "interoceptive_urge_awareness": (
                "Conscious urge-state representation associated with insular processing of craving."
            ),
            "gaba_glutamate_imbalance": (
                "Withdrawal imbalance between inhibitory and excitatory signaling."
            ),
            "cortical_hyperexcitability": (
                "Global failure of CNS inhibition leading to seizure/delirium vulnerability."
            ),
            "mesolimbic_dopamine_dysregulation": (
                "Mesolimbic dopaminergic dysregulation associated with psychotic symptoms."
            ),
            "basal_ganglia_thalamocortical_dysfunction": (
                "Distributed motor-regulatory dysfunction relevant to catatonic withdrawal states."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "withdrawal_negative_affect": (
                "Anxiety, irritability, and dysphoria linked to stress-circuit activation."
            ),
            "anhedonia": (
                "Reduced reward sensitivity and inability to feel pleasure during withdrawal."
            ),
            "craving": (
                "Intense urge to use the substance again during abstinence."
            ),
            "cognitive_impairment": (
                "Attention, concentration, and executive difficulties during withdrawal."
            ),
            "memory_impairment": (
                "Memory problems related to medial temporal dysfunction and cue-memory load."
            ),
            "executive_dyscontrol": (
                "Poor impulse control and weakened top-down regulation."
            ),
            "psychosis": (
                "Withdrawal-related hallucinations or delusions tied to mesolimbic dysregulation."
            ),
            "seizures_delirium": (
                "Severe hyperexcitable withdrawal state with generalized seizures or delirium."
            ),
            "catatonia": (
                "Psychomotor withdrawal syndrome involving basal ganglia-thalamocortical dysfunction."
            ),
            "relapse_vulnerability": (
                "High risk of renewed substance use driven by craving, negative affect, and poor control."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "chronic_dependence_load",
                "target": "allostatic_counteradaptation",
                "relation": "chronic exposure forces the brain into a new pathological equilibrium",
                "ousw_change": "increased",
            },
            {
                "source": "abrupt_cessation",
                "target": "allostatic_counteradaptation",
                "relation": "substance removal unmasks compensatory changes",
                "ousw_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "allostatic_counteradaptation",
                "relation": "heritable vulnerability shapes withdrawal susceptibility",
                "ousw_change": "increased",
            },
            {
                "source": "developmental_environmental_load",
                "target": "allostatic_counteradaptation",
                "relation": "environmental and developmental loading raises withdrawal vulnerability",
                "ousw_change": "increased",
            },
            {
                "source": "chronic_dependence_load",
                "target": "camp_creb_gene_adaptation",
                "relation": "chronic medication exposure induces intracellular signaling and transcriptional changes",
                "ousw_change": "increased",
            },
            {
                "source": "allostatic_counteradaptation",
                "target": "camp_creb_gene_adaptation",
                "relation": "pathological adaptation becomes stabilized through cAMP/CREB-linked plasticity",
                "ousw_change": "increased",
            },
            {
                "source": "camp_creb_gene_adaptation",
                "target": "reward_hypodopaminergia",
                "relation": "long-term synaptic changes help sustain a withdrawal reward deficit",
                "ousw_change": "increased",
            },
            {
                "source": "allostatic_counteradaptation",
                "target": "reward_hypodopaminergia",
                "relation": "withdrawal unmasks hypoactive reward signaling in the striatum",
                "ousw_change": "increased",
            },
            {
                "source": "stress_sensitization",
                "target": "stress_system_hyperreactivity",
                "relation": "stress responsivity amplifies withdrawal negative affect",
                "ousw_change": "increased",
            },
            {
                "source": "allostatic_counteradaptation",
                "target": "stress_system_hyperreactivity",
                "relation": "withdrawal exposes overactive stress circuitry",
                "ousw_change": "increased",
            },
            {
                "source": "chronic_dependence_load",
                "target": "white_matter_disconnection",
                "relation": "chronic substance exposure can degrade white-matter integrity",
                "ousw_change": "increased",
            },
            {
                "source": "camp_creb_gene_adaptation",
                "target": "white_matter_disconnection",
                "relation": "stable plasticity changes alter connectivity and synaptic organization",
                "ousw_change": "increased",
            },
            {
                "source": "white_matter_disconnection",
                "target": "prefrontal_hypofunction",
                "relation": "disrupted connectivity contributes to hypofrontality",
                "ousw_change": "increased",
            },
            {
                "source": "reward_hypodopaminergia",
                "target": "prefrontal_hypofunction",
                "relation": "reward deficit and motivational dysregulation weaken control systems",
                "ousw_change": "increased",
            },
            {
                "source": "prefrontal_hypofunction",
                "target": "frontostriatal_control_failure",
                "relation": "impaired top-down regulation erodes control over urges and impulses",
                "ousw_change": "increased",
            },
            {
                "source": "reward_hypodopaminergia",
                "target": "frontostriatal_control_failure",
                "relation": "low reward tone increases compulsive motivational pressure",
                "ousw_change": "increased",
            },
            {
                "source": "cue_exposure",
                "target": "cue_reactivity_craving_circuit",
                "relation": "medication cues reactivate reward, memory, and motivation circuits",
                "ousw_change": "increased",
            },
            {
                "source": "reward_hypodopaminergia",
                "target": "cue_reactivity_craving_circuit",
                "relation": "reward deficit intensifies cue-driven wanting",
                "ousw_change": "increased",
            },
            {
                "source": "camp_creb_gene_adaptation",
                "target": "cue_reactivity_craving_circuit",
                "relation": "long-term gene-expression changes stabilize craving circuitry",
                "ousw_change": "increased",
            },
            {
                "source": "cue_reactivity_craving_circuit",
                "target": "amygdala",
                "relation": "cue-reactivity recruits affective salience circuitry",
                "ousw_change": "increased",
            },
            {
                "source": "stress_system_hyperreactivity",
                "target": "extended_amygdala_proxy",
                "relation": "negative affect is linked to extended amygdala overactivation",
                "ousw_change": "increased",
            },
            {
                "source": "cue_reactivity_craving_circuit",
                "target": "hippocampus",
                "relation": "cue-related memory processing loads medial temporal systems",
                "ousw_change": "increased",
            },
            {
                "source": "cue_reactivity_craving_circuit",
                "target": "interoceptive_urge_awareness",
                "relation": "cue-driven craving becomes consciously represented as an urge",
                "ousw_change": "increased",
            },
            {
                "source": "interoceptive_urge_awareness",
                "target": "insula",
                "relation": "insula supports conscious awareness of craving",
                "ousw_change": "increased",
            },
            {
                "source": "prefrontal_hypofunction",
                "target": "prefrontal_cortex_proxy",
                "relation": "withdrawal burdens broad prefrontal control circuitry",
                "ousw_change": "increased",
            },
            {
                "source": "reward_hypodopaminergia",
                "target": "striatum_proxy",
                "relation": "reward circuit hypodopaminergia burdens striatal function",
                "ousw_change": "increased",
            },
            {
                "source": "psychosis_liability",
                "target": "mesolimbic_dopamine_dysregulation",
                "relation": "baseline vulnerability can magnify withdrawal-related psychotic risk",
                "ousw_change": "increased",
            },
            {
                "source": "reward_hypodopaminergia",
                "target": "mesolimbic_dopamine_dysregulation",
                "relation": "dopaminergic instability can shift toward psychosis-relevant dysregulation",
                "ousw_change": "increased",
            },
            {
                "source": "sedative_hypnotic_withdrawal_risk",
                "target": "gaba_glutamate_imbalance",
                "relation": "sedative-hypnotic withdrawal especially destabilizes inhibitory-excitatory balance",
                "ousw_change": "increased",
            },
            {
                "source": "allostatic_counteradaptation",
                "target": "gaba_glutamate_imbalance",
                "relation": "withdrawal reveals compensatory imbalance in transmitter systems",
                "ousw_change": "increased",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "cortical_hyperexcitability",
                "relation": "loss of inhibition produces widespread cortical hyperexcitability",
                "ousw_change": "increased",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "basal_ganglia_thalamocortical_dysfunction",
                "relation": "motor-regulatory circuits become unstable when inhibitory-excitatory balance collapses",
                "ousw_change": "increased",
            },
            {
                "source": "basal_ganglia_thalamocortical_dysfunction",
                "target": "basal_ganglia_proxy",
                "relation": "catatonia-relevant dysfunction burdens basal ganglia circuitry",
                "ousw_change": "increased",
            },
            {
                "source": "cortical_hyperexcitability",
                "target": "thalamus_proxy",
                "relation": "severe hyperexcitable states recruit distributed thalamocortical systems",
                "ousw_change": "increased",
            },
            {
                "source": "extended_amygdala_proxy",
                "target": "withdrawal_negative_affect",
                "relation": "extended amygdala stress activity drives dysphoria and irritability",
                "ousw_change": "increased",
            },
            {
                "source": "striatum_proxy",
                "target": "anhedonia",
                "relation": "striatal reward deficit contributes to anhedonia",
                "ousw_change": "increased",
            },
            {
                "source": "cue_reactivity_craving_circuit",
                "target": "craving",
                "relation": "reactivated reward-memory circuitry generates urge to use",
                "ousw_change": "increased",
            },
            {
                "source": "insula",
                "target": "craving",
                "relation": "insula contributes conscious urge awareness to craving",
                "ousw_change": "increased",
            },
            {
                "source": "white_matter_disconnection",
                "target": "cognitive_impairment",
                "relation": "poor connectivity weakens attention and executive efficiency",
                "ousw_change": "increased",
            },
            {
                "source": "prefrontal_cortex_proxy",
                "target": "executive_dyscontrol",
                "relation": "hypofrontality weakens impulse control",
                "ousw_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "memory_impairment",
                "relation": "medial temporal dysfunction contributes to memory problems",
                "ousw_change": "increased",
            },
            {
                "source": "mesolimbic_dopamine_dysregulation",
                "target": "psychosis",
                "relation": "mesolimbic dopaminergic dysregulation drives hallucinations and delusions",
                "ousw_change": "increased",
            },
            {
                "source": "cortical_hyperexcitability",
                "target": "seizures_delirium",
                "relation": "severe CNS disinhibition produces seizures and delirium",
                "ousw_change": "increased",
            },
            {
                "source": "basal_ganglia_thalamocortical_dysfunction",
                "target": "catatonia",
                "relation": "distributed motor-regulatory dysfunction can produce catatonic states",
                "ousw_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "relapse_vulnerability",
                "relation": "failed control over reward-driven behavior raises relapse risk",
                "ousw_change": "increased",
            },
            {
                "source": "recovery_support",
                "target": "cue_reactivity_craving_circuit",
                "relation": "support and structure can reduce cue-driven relapse pressure",
                "ousw_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "prefrontal_hypofunction",
                "relation": "supportive recovery can partially restore control capacity",
                "ousw_change": "decreased",
            },
            {
                "source": "medical_stabilization_support",
                "target": "cortical_hyperexcitability",
                "relation": "acute stabilization can buffer severe hyperexcitable withdrawal states",
                "ousw_change": "decreased",
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
        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "insula",
            "prefrontal cortex",
            "striatum",
            "thalamus",
            "basal ganglia",
        } else 0
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

        region_tokens = {tok for tok in rn.replace("(", " ").replace(")", " ").split() if len(tok) > 2}
        best = None
        best_score = 0
        for label in labels:
            label_name = self._name_of(label).lower()
            label_tokens = {tok for tok in label_name.replace("(", " ").replace(")", " ").split() if len(tok) > 2}
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
        result remains readable even when atlas labels differ across versions.
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
        region_resolution_df = pd.DataFrame(self.region_resolution.values())
        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "region_resolution": region_resolution_df,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": self.circuit_connectivity(),
        }

    def simulate(
        self,
        chronic_dependence_load: float = 0.7,
        abrupt_cessation: float = 0.8,
        genetic_vulnerability: float = 0.5,
        developmental_environmental_load: float = 0.4,
        stress_sensitization: float = 0.6,
        cue_exposure: float = 0.6,
        sedative_hypnotic_withdrawal_risk: float = 0.3,
        psychosis_liability: float = 0.3,
        recovery_support: float = 0.3,
        medical_stabilization_support: float = 0.2,
    ) -> Dict[str, pd.Series]:
        """
        Transparent one-pass withdrawal simulator.

        Inputs are normalized to [0, 1]. Protective factors subtract where the
        chapter clearly supports a buffering effect.
        """
        inputs = {
            "chronic_dependence_load": chronic_dependence_load,
            "abrupt_cessation": abrupt_cessation,
            "genetic_vulnerability": genetic_vulnerability,
            "developmental_environmental_load": developmental_environmental_load,
            "stress_sensitization": stress_sensitization,
            "cue_exposure": cue_exposure,
            "sedative_hypnotic_withdrawal_risk": sedative_hypnotic_withdrawal_risk,
            "psychosis_liability": psychosis_liability,
            "recovery_support": recovery_support,
            "medical_stabilization_support": medical_stabilization_support,
        }
        x = {k: self._clip01(v) for k, v in inputs.items()}

        latents: Dict[str, float] = {}
        latents["allostatic_counteradaptation"] = self._clip01(
            0.38 * x["chronic_dependence_load"]
            + 0.28 * x["abrupt_cessation"]
            + 0.10 * x["genetic_vulnerability"]
            + 0.08 * x["developmental_environmental_load"]
            + 0.06 * x["stress_sensitization"]
            - 0.10 * x["recovery_support"]
        )
        latents["camp_creb_gene_adaptation"] = self._clip01(
            0.42 * x["chronic_dependence_load"]
            + 0.28 * latents["allostatic_counteradaptation"]
            + 0.08 * x["genetic_vulnerability"]
            + 0.06 * x["developmental_environmental_load"]
            - 0.08 * x["recovery_support"]
        )
        latents["reward_hypodopaminergia"] = self._clip01(
            0.40 * latents["allostatic_counteradaptation"]
            + 0.24 * x["abrupt_cessation"]
            + 0.18 * latents["camp_creb_gene_adaptation"]
            + 0.08 * x["chronic_dependence_load"]
            - 0.10 * x["recovery_support"]
        )
        latents["stress_system_hyperreactivity"] = self._clip01(
            0.38 * latents["allostatic_counteradaptation"]
            + 0.26 * x["stress_sensitization"]
            + 0.12 * x["abrupt_cessation"]
            + 0.08 * x["cue_exposure"]
            - 0.10 * x["recovery_support"]
        )
        latents["white_matter_disconnection"] = self._clip01(
            0.30 * x["chronic_dependence_load"]
            + 0.20 * latents["camp_creb_gene_adaptation"]
            + 0.14 * x["developmental_environmental_load"]
            + 0.10 * latents["allostatic_counteradaptation"]
            - 0.06 * x["recovery_support"]
        )
        latents["prefrontal_hypofunction"] = self._clip01(
            0.34 * latents["reward_hypodopaminergia"]
            + 0.26 * latents["white_matter_disconnection"]
            + 0.18 * latents["stress_system_hyperreactivity"]
            + 0.08 * x["abrupt_cessation"]
            - 0.14 * x["recovery_support"]
        )
        latents["cue_reactivity_craving_circuit"] = self._clip01(
            0.30 * x["cue_exposure"]
            + 0.24 * latents["reward_hypodopaminergia"]
            + 0.20 * latents["stress_system_hyperreactivity"]
            + 0.12 * latents["camp_creb_gene_adaptation"]
            + 0.08 * x["abrupt_cessation"]
            - 0.12 * x["recovery_support"]
        )
        latents["frontostriatal_control_failure"] = self._clip01(
            0.40 * latents["prefrontal_hypofunction"]
            + 0.24 * latents["reward_hypodopaminergia"]
            + 0.18 * latents["cue_reactivity_craving_circuit"]
            + 0.12 * latents["white_matter_disconnection"]
            - 0.10 * x["recovery_support"]
        )
        latents["interoceptive_urge_awareness"] = self._clip01(
            0.46 * latents["cue_reactivity_craving_circuit"]
            + 0.22 * latents["stress_system_hyperreactivity"]
            + 0.10 * x["abrupt_cessation"]
        )
        latents["gaba_glutamate_imbalance"] = self._clip01(
            0.36 * latents["allostatic_counteradaptation"]
            + 0.24 * x["abrupt_cessation"]
            + 0.24 * x["sedative_hypnotic_withdrawal_risk"]
            - 0.16 * x["medical_stabilization_support"]
        )
        latents["cortical_hyperexcitability"] = self._clip01(
            0.48 * latents["gaba_glutamate_imbalance"]
            + 0.20 * x["sedative_hypnotic_withdrawal_risk"]
            + 0.12 * x["abrupt_cessation"]
            - 0.18 * x["medical_stabilization_support"]
        )
        latents["mesolimbic_dopamine_dysregulation"] = self._clip01(
            0.34 * latents["reward_hypodopaminergia"]
            + 0.24 * x["psychosis_liability"]
            + 0.16 * latents["cue_reactivity_craving_circuit"]
            + 0.10 * latents["stress_system_hyperreactivity"]
        )
        latents["basal_ganglia_thalamocortical_dysfunction"] = self._clip01(
            0.34 * latents["gaba_glutamate_imbalance"]
            + 0.22 * latents["cortical_hyperexcitability"]
            + 0.18 * latents["mesolimbic_dopamine_dysregulation"]
            + 0.12 * latents["white_matter_disconnection"]
            - 0.10 * x["medical_stabilization_support"]
        )

        regional_state = {
            "amygdala": self._clip01(
                0.42 * latents["cue_reactivity_craving_circuit"]
                + 0.28 * latents["stress_system_hyperreactivity"]
                + 0.08 * x["abrupt_cessation"]
            ),
            "extended_amygdala_proxy": self._clip01(
                0.52 * latents["stress_system_hyperreactivity"]
                + 0.16 * latents["cue_reactivity_craving_circuit"]
            ),
            "hippocampus": self._clip01(
                0.38 * latents["cue_reactivity_craving_circuit"]
                + 0.24 * latents["camp_creb_gene_adaptation"]
                + 0.16 * latents["stress_system_hyperreactivity"]
                + 0.08 * x["developmental_environmental_load"]
            ),
            "insula": self._clip01(
                0.54 * latents["interoceptive_urge_awareness"]
                + 0.18 * latents["stress_system_hyperreactivity"]
                + 0.08 * x["cue_exposure"]
            ),
            "prefrontal_cortex_proxy": self._clip01(
                0.44 * latents["prefrontal_hypofunction"]
                + 0.22 * latents["frontostriatal_control_failure"]
                + 0.18 * latents["white_matter_disconnection"]
            ),
            "striatum_proxy": self._clip01(
                0.52 * latents["reward_hypodopaminergia"]
                + 0.20 * latents["cue_reactivity_craving_circuit"]
                + 0.12 * latents["mesolimbic_dopamine_dysregulation"]
            ),
            "basal_ganglia_proxy": self._clip01(
                0.48 * latents["basal_ganglia_thalamocortical_dysfunction"]
                + 0.20 * latents["mesolimbic_dopamine_dysregulation"]
            ),
            "thalamus_proxy": self._clip01(
                0.38 * latents["cortical_hyperexcitability"]
                + 0.28 * latents["basal_ganglia_thalamocortical_dysfunction"]
                + 0.12 * latents["gaba_glutamate_imbalance"]
            ),
        }

        symptoms = {
            "withdrawal_negative_affect": self._clip01(
                0.42 * regional_state["extended_amygdala_proxy"]
                + 0.24 * latents["stress_system_hyperreactivity"]
                + 0.08 * x["abrupt_cessation"]
            ),
            "anhedonia": self._clip01(
                0.54 * latents["reward_hypodopaminergia"]
                + 0.20 * regional_state["striatum_proxy"]
            ),
            "craving": self._clip01(
                0.30 * latents["cue_reactivity_craving_circuit"]
                + 0.18 * regional_state["insula"]
                + 0.14 * regional_state["amygdala"]
                + 0.14 * regional_state["hippocampus"]
                + 0.10 * regional_state["striatum_proxy"]
                - 0.12 * x["recovery_support"]
            ),
            "cognitive_impairment": self._clip01(
                0.34 * regional_state["prefrontal_cortex_proxy"]
                + 0.26 * latents["white_matter_disconnection"]
                + 0.12 * regional_state["hippocampus"]
                + 0.10 * x["abrupt_cessation"]
            ),
            "memory_impairment": self._clip01(
                0.42 * regional_state["hippocampus"]
                + 0.18 * latents["stress_system_hyperreactivity"]
                + 0.12 * latents["camp_creb_gene_adaptation"]
            ),
            "executive_dyscontrol": self._clip01(
                0.40 * latents["frontostriatal_control_failure"]
                + 0.24 * regional_state["prefrontal_cortex_proxy"]
                + 0.14 * latents["white_matter_disconnection"]
            ),
            "psychosis": self._clip01(
                0.42 * latents["mesolimbic_dopamine_dysregulation"]
                + 0.18 * x["psychosis_liability"]
                + 0.14 * regional_state["hippocampus"]
                + 0.10 * latents["stress_system_hyperreactivity"]
            ),
            "seizures_delirium": self._clip01(
                0.48 * latents["cortical_hyperexcitability"]
                + 0.18 * regional_state["thalamus_proxy"]
                + 0.14 * x["sedative_hypnotic_withdrawal_risk"]
                - 0.18 * x["medical_stabilization_support"]
            ),
            "catatonia": self._clip01(
                0.30 * regional_state["basal_ganglia_proxy"]
                + 0.22 * regional_state["thalamus_proxy"]
                + 0.22 * latents["gaba_glutamate_imbalance"]
                + 0.10 * latents["cortical_hyperexcitability"]
                - 0.12 * x["medical_stabilization_support"]
            ),
        }
        symptoms["relapse_vulnerability"] = self._clip01(
            0.30 * symptoms["craving"]
            + 0.22 * symptoms["withdrawal_negative_affect"]
            + 0.16 * symptoms["executive_dyscontrol"]
            + 0.10 * latents["cue_reactivity_craving_circuit"]
            - 0.18 * x["recovery_support"]
        )

        phenotypes = {
            "negative_affect_craving_profile": self._clip01(
                (
                    symptoms["withdrawal_negative_affect"]
                    + symptoms["craving"]
                    + symptoms["anhedonia"]
                )
                / 3.0
            ),
            "cognitive_withdrawal_profile": self._clip01(
                (
                    symptoms["cognitive_impairment"]
                    + symptoms["memory_impairment"]
                    + symptoms["executive_dyscontrol"]
                )
                / 3.0
            ),
            "severe_hyperexcitable_withdrawal_profile": self._clip01(
                (
                    symptoms["seizures_delirium"]
                    + symptoms["catatonia"]
                    + latents["cortical_hyperexcitability"]
                )
                / 3.0
            ),
            "psychosis_vulnerable_withdrawal_profile": self._clip01(
                (
                    symptoms["psychosis"]
                    + latents["mesolimbic_dopamine_dysregulation"]
                    + regional_state["hippocampus"]
                )
                / 3.0
            ),
            "relapse_pressure_profile": self._clip01(
                (
                    symptoms["relapse_vulnerability"]
                    + symptoms["craving"]
                    + symptoms["withdrawal_negative_affect"]
                    + latents["frontostriatal_control_failure"]
                )
                / 4.0
            ),
        }

        return {
            "inputs": pd.Series(x, name="value"),
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
    model = OtherOrUnknownSubstanceWithdrawalModel()
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

    print("\n=== Simulated example ===")
    sim = model.simulate(
        chronic_dependence_load=0.85,
        abrupt_cessation=0.90,
        genetic_vulnerability=0.55,
        developmental_environmental_load=0.50,
        stress_sensitization=0.70,
        cue_exposure=0.75,
        sedative_hypnotic_withdrawal_risk=0.45,
        psychosis_liability=0.35,
        recovery_support=0.30,
        medical_stabilization_support=0.25,
    )
    for name, series in sim.items():
        print(f"\n-- {name} --")
        print(series.sort_values(ascending=False).to_string())

    # Example coordinate assignment once siibra is installed:
    # print(model.assign_mni_point((-32, 20, 6)).head())
