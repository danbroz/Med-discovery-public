from __future__ import annotations

"""
Trichotillomania (Hair-Pulling Disorder) siibra scaffold.

This script translates a chapter-level summary of trichotillomania (TTM) into
an atlas-grounded, transparent research scaffold. It is intended to be readable,
portable across evolving siibra APIs, and tolerant of partial multimodal data.

Important notes
---------------
- This is a research scaffold, not a diagnostic or treatment tool.
- The simulator is a mechanistic interpretation of the supplied chapter, not a
  validated disease model.
- The chapter frames TTM around three interacting domains: affect dysregulation,
  addiction / habit formation, and impulse dyscontrol. Those domains are kept
  explicit in the latent-biology layer and in the simulator.
- Reward, serotonin, dopamine, and glutamate mechanisms are modeled primarily as
  latent biology rather than being forced into single parcels.
- A generic self_regulation_support input is included as an explicit protective
  counterweight for simulation transparency, even though the chapter itself is
  primarily risk-focused.
- The gene panel is heuristic and intended to cover OCRD/BFRB vulnerability,
  reward/habit learning, glutamatergic control, stress responsivity, and
  synaptic plasticity that are discussed or strongly implied by the chapter.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception as exc:  # pragma: no cover - import guard for portability
    siibra = None
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial branch
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_GENE_PANEL = [
    "SLC6A4",
    "HTR2A",
    "HTR1B",
    "DRD2",
    "DRD3",
    "SLC6A3",
    "COMT",
    "OPRM1",
    "SLC1A1",
    "GRIN2B",
    "GRM5",
    "GAD1",
    "BDNF",
    "DLGAP3",
    "SLITRK5",
    "NR3C1",
    "FKBP5",
]


class TrichotillomaniaModel:
    """
    Atlas-grounded scaffold for trichotillomania (hair-pulling disorder).

    Chapter logic encoded here emphasizes:
    - multifactorial etiology combining genetic liability, environmental stress,
      psychiatric comorbidity, and neurochemical dysregulation;
    - OCRD/BFRB-spectrum overlap and a shared familial diathesis with other
      repetitive grooming behaviors;
    - serotonin, dopamine, and glutamate involvement in mood regulation,
      reinforcement, habit formation, and executive control;
    - a three-domain brain model built around affect dysregulation,
      addiction/habit formation, and impulse dyscontrol;
    - limbic trigger circuitry (amygdala, hippocampus, ACC), dorsal-striatal
      habit circuitry (caudate/putamen), ventral-striatal reinforcement, and
      prefrontal inhibitory-control failure (OFC/DLPFC/ACC);
    - stress-linked emotional memory and automatic pulling episodes with failed
      resistance, relief/reward reinforcement, and repetitive pulling burden.
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
                "siibra is required to instantiate TrichotillomaniaModel. "
                "Install siibra-python in your environment first."
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
            "ofc": [
                "Area Fo4 left",
                "Area Fo3 left",
                "Area Fo5 left",
                "orbitofrontal",
                "Fo4",
                "Fo3",
            ],
            "dlpfc": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 9/46v left",
                "dorsolateral prefrontal",
                "middle frontal",
            ],
            "acc": [
                "Area p24pr left",
                "Area a24pr left",
                "Area p24 left",
                "anterior cingulate",
                "cingulate",
            ],
            "caudate": [
                "Caudate nucleus left",
                "caudate nucleus",
                "caudate",
            ],
            "putamen": [
                "Putamen left",
                "putamen",
            ],
            "ventral_striatum_proxy": [
                "Nucleus accumbens left",
                "nucleus accumbens",
                "accumbens",
                "ventral striatum",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 left",
                "DG left",
                "Subiculum left",
                "hippocampus",
            ],
        }

        self.region_notes: Dict[str, str] = {
            "ofc": (
                "Orbitofrontal representative of top-down inhibitory control over "
                "urges and habit suppression."
            ),
            "dlpfc": (
                "Dorsolateral prefrontal representative of executive control and "
                "resistance to unwanted pulling."
            ),
            "acc": (
                "Anterior cingulate representative of affect regulation, conflict "
                "monitoring, and urge-related control burden."
            ),
            "caudate": (
                "Dorsal-striatal node for habit learning and action-selection bias "
                "in repetitive pulling."
            ),
            "putamen": (
                "Dorsal-striatal node for stereotyped motor habit expression and "
                "automatic pulling routines."
            ),
            "ventral_striatum_proxy": (
                "Proxy for relief / reward reinforcement that strengthens the "
                "hair-pulling habit loop."
            ),
            "amygdala": (
                "Limbic trigger node for stress-, anxiety-, and boredom-linked "
                "pulling episodes."
            ),
            "hippocampus": (
                "Stress-sensitive limbic node for emotional memory and negative "
                "affect sensitization relevant to recurrent pulling."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Heritable liability affecting OCRD/BFRB risk, transmitter balance, "
                "and habit-control circuits."
            ),
            "ocrd_bfrb_shared_liability": (
                "Shared obsessive-compulsive / body-focused repetitive behavior "
                "diathesis across grooming-spectrum conditions."
            ),
            "environmental_stress_load": (
                "Environmental stress burden that amplifies emotional triggers and "
                "habit expression."
            ),
            "negative_affect_trigger_load": (
                "Stress, anxiety, or other negative-affect trigger intensity before "
                "pulling episodes."
            ),
            "boredom_understimulation_load": (
                "Boredom / understimulation pressure that can precipitate pulling."
            ),
            "sensory_reinforcement_sensitivity": (
                "Sensitivity to sensory gratification, relief, or pleasurable pull "
                "feedback."
            ),
            "automatic_habit_proneness": (
                "Tendency toward stereotyped, low-awareness, automatic repetitive "
                "behavior."
            ),
            "depressive_anxiety_comorbidity": (
                "Comorbid depression / anxiety burden that shares and amplifies TTM "
                "pathways."
            ),
            "self_regulation_support": (
                "Generic protective self-regulation / behavioral support that can "
                "interrupt urges and reduce pulling expression."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_dysregulation": (
                "Serotonergic imbalance affecting mood regulation, anxiety, and urge "
                "modulation."
            ),
            "dopaminergic_reward_relief_bias": (
                "Dopamine-linked relief / reward reinforcement strengthening the "
                "habit loop after pulling."
            ),
            "glutamatergic_control_imbalance": (
                "Glutamate-linked control imbalance affecting executive regulation "
                "and action selection."
            ),
            "affect_dysregulation": (
                "Maladaptive emotional regulation that makes pulling a short-term "
                "relief strategy."
            ),
            "emotional_trigger_sensitivity": (
                "Heightened sensitivity to stress, anxiety, boredom, and other "
                "internal pulling triggers."
            ),
            "habit_loop_strengthening": (
                "Progressive reinforcement and consolidation of repetitive pulling "
                "into a deeply ingrained habit."
            ),
            "automaticity_awareness_decoupling": (
                "Separation between action execution and conscious monitoring during "
                "automatic pulling."
            ),
            "frontostriatal_dysconnectivity": (
                "Impaired connectivity between prefrontal control regions and the "
                "striatum."
            ),
            "frontolimbic_trigger_coupling": (
                "Tight coupling between emotional circuits and the urge-to-pull "
                "system."
            ),
            "hippocampal_stress_burden": (
                "Stress-related hippocampal burden contributing to negative-affect "
                "memory and sensitization."
            ),
            "impulse_control_failure": (
                "Failure of top-down inhibitory control over subcortically generated "
                "pulling urges."
            ),
            "pulling_urge_generation": (
                "Integrated urge state arising from affective triggers, reward bias, "
                "and weak inhibitory control."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "hair_pulling_urge": (
                "The urge or drive to pull hair."
            ),
            "failed_resistance": (
                "Difficulty resisting or suppressing the pulling impulse."
            ),
            "automatic_pulling": (
                "Low-awareness, automatic, stereotyped pulling episodes."
            ),
            "affect_triggered_pulling": (
                "Pulling linked to stress, anxiety, boredom, or other negative "
                "states."
            ),
            "relief_reinforced_pulling": (
                "Pulling maintained by pleasure, relief, or sensory reinforcement."
            ),
            "repetitive_hair_pulling": (
                "Overall repetitive pulling burden integrating urge, habit, and "
                "failed resistance."
            ),
            "distress_impairment": (
                "Downstream distress and functional burden from recurrent pulling."
            ),
        }

        self.edge_table: List[Dict[str, Any]] = [
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_dysregulation",
                "relation": "raises inherited transmitter vulnerability affecting mood and urges",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "genetic_vulnerability",
                "target": "dopaminergic_reward_relief_bias",
                "relation": "can increase sensitivity to reward and reinforcement",
                "ttm_change": "increased",
                "weight": 0.15,
            },
            {
                "source": "genetic_vulnerability",
                "target": "glutamatergic_control_imbalance",
                "relation": "raises liability for executive-control imbalance",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "ocrd_bfrb_shared_liability",
                "target": "frontostriatal_dysconnectivity",
                "relation": "links TTM to grooming-spectrum control-loop vulnerability",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "ocrd_bfrb_shared_liability",
                "target": "habit_loop_strengthening",
                "relation": "supports repetitive grooming-like habit consolidation",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "environmental_stress_load",
                "target": "affect_dysregulation",
                "relation": "stress amplifies the maladaptive emotion-regulation function of pulling",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "environmental_stress_load",
                "target": "hippocampal_stress_burden",
                "relation": "chronic stress loads hippocampal emotional-memory systems",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "negative_affect_trigger_load",
                "target": "affect_dysregulation",
                "relation": "negative internal states drive pulling as short-term relief",
                "ttm_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "negative_affect_trigger_load",
                "target": "emotional_trigger_sensitivity",
                "relation": "directly loads the affective trigger system",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "boredom_understimulation_load",
                "target": "emotional_trigger_sensitivity",
                "relation": "boredom can precipitate pulling episodes",
                "ttm_change": "increased",
                "weight": 0.15,
            },
            {
                "source": "boredom_understimulation_load",
                "target": "automaticity_awareness_decoupling",
                "relation": "understimulation increases low-awareness repetitive behavior",
                "ttm_change": "increased",
                "weight": 0.10,
            },
            {
                "source": "sensory_reinforcement_sensitivity",
                "target": "dopaminergic_reward_relief_bias",
                "relation": "sensory gratification and relief increase reinforcement value",
                "ttm_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "sensory_reinforcement_sensitivity",
                "target": "habit_loop_strengthening",
                "relation": "repeated sensory reward strengthens the pulling routine",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "automatic_habit_proneness",
                "target": "automaticity_awareness_decoupling",
                "relation": "predisposes to automatic, low-awareness pulling",
                "ttm_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "automatic_habit_proneness",
                "target": "habit_loop_strengthening",
                "relation": "supports stereotyped motor repetition",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "depressive_anxiety_comorbidity",
                "target": "serotonergic_dysregulation",
                "relation": "shared mood-anxiety biology increases serotonergic burden",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "depressive_anxiety_comorbidity",
                "target": "affect_dysregulation",
                "relation": "comorbid affective illness worsens emotional regulation failure",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "depressive_anxiety_comorbidity",
                "target": "hippocampal_stress_burden",
                "relation": "affective comorbidity increases hippocampal stress sensitivity",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "self_regulation_support",
                "target": "affect_dysregulation",
                "relation": "protective regulation reduces pulling-as-relief dependence",
                "ttm_change": "decreased",
                "weight": -0.20,
            },
            {
                "source": "self_regulation_support",
                "target": "impulse_control_failure",
                "relation": "protective control support helps resist urges",
                "ttm_change": "decreased",
                "weight": -0.20,
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "affect_dysregulation",
                "relation": "serotonin imbalance worsens mood and urge regulation",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "frontolimbic_trigger_coupling",
                "relation": "links transmitter instability to emotion-triggered pulling",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "dopaminergic_reward_relief_bias",
                "target": "habit_loop_strengthening",
                "relation": "relief and reward reinforce the motor habit loop",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "dopaminergic_reward_relief_bias",
                "target": "pulling_urge_generation",
                "relation": "reward salience increases urge intensity",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "glutamatergic_control_imbalance",
                "target": "frontostriatal_dysconnectivity",
                "relation": "control imbalance destabilizes prefrontal-striatal communication",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "glutamatergic_control_imbalance",
                "target": "impulse_control_failure",
                "relation": "reduces executive suppression of urges",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "affect_dysregulation",
                "target": "emotional_trigger_sensitivity",
                "relation": "increases sensitivity to stress, anxiety, and boredom cues",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "affect_dysregulation",
                "target": "frontolimbic_trigger_coupling",
                "relation": "strengthens coupling between emotional distress and pulling urges",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "emotional_trigger_sensitivity",
                "target": "pulling_urge_generation",
                "relation": "translates internal emotional triggers into urge pressure",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "habit_loop_strengthening",
                "target": "automaticity_awareness_decoupling",
                "relation": "turns pulling into an automatic stereotyped routine",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "habit_loop_strengthening",
                "target": "frontostriatal_dysconnectivity",
                "relation": "deep habit expression further biases striatal over prefrontal control",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "frontostriatal_dysconnectivity",
                "target": "impulse_control_failure",
                "relation": "PFC-striatal disconnection weakens inhibitory control",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "frontolimbic_trigger_coupling",
                "target": "acc",
                "relation": "loads conflict-monitoring and emotion-regulation circuitry",
                "ttm_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "frontolimbic_trigger_coupling",
                "target": "amygdala",
                "relation": "increases limbic trigger reactivity",
                "ttm_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "hippocampal_stress_burden",
                "target": "hippocampus",
                "relation": "stress-linked emotional-memory burden loads hippocampal state",
                "ttm_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "impulse_control_failure",
                "target": "ofc",
                "relation": "manifests as orbitofrontal inhibitory-control impairment",
                "ttm_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "impulse_control_failure",
                "target": "dlpfc",
                "relation": "manifests as weakened executive resistance to urges",
                "ttm_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "frontostriatal_dysconnectivity",
                "target": "caudate",
                "relation": "loads dorsal-striatal action-selection circuitry",
                "ttm_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "habit_loop_strengthening",
                "target": "putamen",
                "relation": "strengthens stereotyped motor-habit expression",
                "ttm_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "dopaminergic_reward_relief_bias",
                "target": "ventral_striatum_proxy",
                "relation": "loads the reward/relief reinforcement node",
                "ttm_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "pulling_urge_generation",
                "target": "hair_pulling_urge",
                "relation": "creates the conscious urge to pull",
                "ttm_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "impulse_control_failure",
                "target": "failed_resistance",
                "relation": "directly impairs suppression of the urge",
                "ttm_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "ofc",
                "target": "failed_resistance",
                "relation": "orbitofrontal control failure worsens inability to resist",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "dlpfc",
                "target": "failed_resistance",
                "relation": "weak executive control reduces successful urge suppression",
                "ttm_change": "increased",
                "weight": 0.15,
            },
            {
                "source": "automaticity_awareness_decoupling",
                "target": "automatic_pulling",
                "relation": "promotes low-awareness pulling episodes",
                "ttm_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "putamen",
                "target": "automatic_pulling",
                "relation": "motor habit circuitry promotes repetitive automatic pulling",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "caudate",
                "target": "automatic_pulling",
                "relation": "dorsal-striatal action bias contributes to stereotyped pulling",
                "ttm_change": "increased",
                "weight": 0.15,
            },
            {
                "source": "amygdala",
                "target": "affect_triggered_pulling",
                "relation": "limbic reactivity links distress to pulling episodes",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "acc",
                "target": "affect_triggered_pulling",
                "relation": "emotion/conflict burden contributes to pull-to-relieve behavior",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "ventral_striatum_proxy",
                "target": "relief_reinforced_pulling",
                "relation": "reward/relief circuitry reinforces future pulling",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "hair_pulling_urge",
                "target": "repetitive_hair_pulling",
                "relation": "urge pressure increases pulling frequency",
                "ttm_change": "increased",
                "weight": 0.15,
            },
            {
                "source": "failed_resistance",
                "target": "repetitive_hair_pulling",
                "relation": "inability to resist increases repeated pulling",
                "ttm_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "automatic_pulling",
                "target": "repetitive_hair_pulling",
                "relation": "automatic habit expression increases total pulling burden",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "affect_triggered_pulling",
                "target": "repetitive_hair_pulling",
                "relation": "negative-affect episodes increase pulling recurrence",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "relief_reinforced_pulling",
                "target": "repetitive_hair_pulling",
                "relation": "rewarded episodes strengthen future pulling",
                "ttm_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "repetitive_hair_pulling",
                "target": "distress_impairment",
                "relation": "higher pulling burden produces downstream distress and functional impairment",
                "ttm_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "depressive_anxiety_comorbidity",
                "target": "distress_impairment",
                "relation": "comorbid affective burden amplifies suffering and impairment",
                "ttm_change": "increased",
                "weight": 0.15,
            },
            {
                "source": "self_regulation_support",
                "target": "repetitive_hair_pulling",
                "relation": "protective support lowers total pulling expression",
                "ttm_change": "decreased",
                "weight": -0.15,
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
    def _float_or_none(x: Any) -> Optional[float]:
        try:
            value = float(x)
        except Exception:
            return None
        if pd.isna(value):
            return None
        return value

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

    def _region_rank(self, region: Any) -> Tuple[int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "orbitofrontal cortex",
            "prefrontal cortex",
            "anterior cingulate cortex",
            "striatum",
        } else 0
        return (left_bonus, right_penalty, generic_penalty)

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
        volume_mm3 = self._float_or_none(getattr(main, "volume", float("nan")))
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

    def _choose_connectivity_feature(self, concept: Any) -> Optional[Any]:
        feats = self._safe_features_any(concept, self._modality_candidates("connectivity"))
        if not feats:
            return None

        cohort_matches = []
        for feat in feats:
            cohort_text = str(getattr(feat, "cohort", ""))
            name_text = str(getattr(feat, "name", ""))
            if self.connectivity_cohort.lower() in cohort_text.lower() or self.connectivity_cohort.lower() in name_text.lower():
                cohort_matches.append(feat)
        return cohort_matches[0] if cohort_matches else feats[0]

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feature = self._choose_connectivity_feature(self.parcellation)
        if feature is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        try:
            data = getattr(feature, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            first = feature[0]
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
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        if fuzzy:
            return fuzzy[0]

        short_rn = (
            rn.replace("area ", "")
            .replace(" left", "")
            .replace(" right", "")
            .replace(" hemisphere", "")
            .replace(" proxy", "")
        )
        fuzzy_short = [
            x
            for x in labels
            if short_rn and short_rn in self._name_of(x).lower().replace("area ", "")
        ]
        return fuzzy_short[0] if fuzzy_short else None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        row_label = self._match_region_label(list(matrix.index), region)
        col_label = self._match_region_label(list(matrix.columns), region)
        series = None

        if row_label is not None:
            try:
                series = matrix.loc[row_label]
            except Exception:
                series = None

        if series is None and col_label is not None:
            try:
                series = matrix[col_label]
            except Exception:
                series = None

        if series is None:
            return pd.DataFrame()

        try:
            if isinstance(series, pd.DataFrame):
                series = series.mean(axis=0)
            numeric = pd.to_numeric(series, errors="coerce")
            numeric = numeric.dropna().sort_values(ascending=False)
            df = numeric.reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def build(self, gene_panel: Sequence[str] = DEFAULT_GENE_PANEL, connectivity_rows: int = 15) -> dict:
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
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region",
                        "description": self.region_notes.get(
                            key,
                            "Atlas-backed node that could not be resolved in this environment.",
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
                    "description": self.region_notes.get(key, "Atlas-backed circuit node."),
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

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        index_labels = list(matrix.index)
        column_labels = list(matrix.columns)

        for source_key, source_region in self.region_objects.items():
            src_row = self._match_region_label(index_labels, source_region)
            src_col = self._match_region_label(column_labels, source_region)
            for target_key, target_region in self.region_objects.items():
                if source_key == target_key:
                    continue
                tgt_row = self._match_region_label(index_labels, target_region)
                tgt_col = self._match_region_label(column_labels, target_region)
                value = None

                for lhs, rhs in ((src_row, tgt_col), (src_row, tgt_row), (src_col, tgt_col), (src_col, tgt_row)):
                    if lhs is None or rhs is None:
                        continue
                    try:
                        value = self._float_or_none(matrix.loc[lhs, rhs])
                    except Exception:
                        try:
                            value = self._float_or_none(matrix.loc[rhs, lhs])
                        except Exception:
                            value = None
                    if value is not None:
                        break

                if value is None:
                    continue
                rows.append(
                    {
                        "source_key": source_key,
                        "source_region": source_region.name,
                        "target_key": target_key,
                        "target_region": target_region.name,
                        "value": value,
                    }
                )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def simulate(
        self,
        genetic_vulnerability: float = 0.6,
        ocrd_bfrb_shared_liability: float = 0.55,
        environmental_stress_load: float = 0.6,
        negative_affect_trigger_load: float = 0.65,
        boredom_understimulation_load: float = 0.45,
        sensory_reinforcement_sensitivity: float = 0.6,
        automatic_habit_proneness: float = 0.6,
        depressive_anxiety_comorbidity: float = 0.5,
        self_regulation_support: float = 0.3,
    ) -> Dict[str, pd.Series]:
        """
        Run a one-pass normalized simulation.

        The calculation order is intentionally acyclic and transparent:
        inputs -> latent biology -> regional state -> symptoms -> phenotype summaries
        """

        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "ocrd_bfrb_shared_liability": self._clip01(ocrd_bfrb_shared_liability),
                "environmental_stress_load": self._clip01(environmental_stress_load),
                "negative_affect_trigger_load": self._clip01(negative_affect_trigger_load),
                "boredom_understimulation_load": self._clip01(boredom_understimulation_load),
                "sensory_reinforcement_sensitivity": self._clip01(sensory_reinforcement_sensitivity),
                "automatic_habit_proneness": self._clip01(automatic_habit_proneness),
                "depressive_anxiety_comorbidity": self._clip01(depressive_anxiety_comorbidity),
                "self_regulation_support": self._clip01(self_regulation_support),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["serotonergic_dysregulation"] = self._clip01(
            0.25 * inputs["genetic_vulnerability"]
            + 0.25 * inputs["depressive_anxiety_comorbidity"]
            + 0.15 * inputs["environmental_stress_load"]
            + 0.10 * inputs["negative_affect_trigger_load"]
            - 0.15 * inputs["self_regulation_support"]
        )
        latents["dopaminergic_reward_relief_bias"] = self._clip01(
            0.30 * inputs["sensory_reinforcement_sensitivity"]
            + 0.20 * inputs["automatic_habit_proneness"]
            + 0.15 * inputs["genetic_vulnerability"]
            + 0.10 * inputs["ocrd_bfrb_shared_liability"]
            + 0.10 * inputs["boredom_understimulation_load"]
            - 0.05 * inputs["self_regulation_support"]
        )
        latents["glutamatergic_control_imbalance"] = self._clip01(
            0.25 * inputs["genetic_vulnerability"]
            + 0.20 * inputs["ocrd_bfrb_shared_liability"]
            + 0.15 * inputs["environmental_stress_load"]
            + 0.15 * inputs["depressive_anxiety_comorbidity"]
            + 0.10 * inputs["negative_affect_trigger_load"]
            - 0.10 * inputs["self_regulation_support"]
        )
        latents["affect_dysregulation"] = self._clip01(
            0.30 * inputs["negative_affect_trigger_load"]
            + 0.20 * inputs["environmental_stress_load"]
            + 0.15 * inputs["depressive_anxiety_comorbidity"]
            + 0.15 * latents["serotonergic_dysregulation"]
            + 0.10 * inputs["boredom_understimulation_load"]
            - 0.20 * inputs["self_regulation_support"]
        )
        latents["emotional_trigger_sensitivity"] = self._clip01(
            0.25 * latents["affect_dysregulation"]
            + 0.20 * inputs["negative_affect_trigger_load"]
            + 0.15 * inputs["environmental_stress_load"]
            + 0.10 * inputs["depressive_anxiety_comorbidity"]
            + 0.10 * inputs["boredom_understimulation_load"]
        )
        latents["habit_loop_strengthening"] = self._clip01(
            0.25 * latents["dopaminergic_reward_relief_bias"]
            + 0.20 * inputs["automatic_habit_proneness"]
            + 0.20 * inputs["sensory_reinforcement_sensitivity"]
            + 0.15 * inputs["ocrd_bfrb_shared_liability"]
            + 0.10 * latents["affect_dysregulation"]
            - 0.10 * inputs["self_regulation_support"]
        )
        latents["automaticity_awareness_decoupling"] = self._clip01(
            0.30 * inputs["automatic_habit_proneness"]
            + 0.20 * latents["habit_loop_strengthening"]
            + 0.10 * inputs["boredom_understimulation_load"]
            + 0.10 * inputs["environmental_stress_load"]
            - 0.10 * inputs["self_regulation_support"]
        )
        latents["frontostriatal_dysconnectivity"] = self._clip01(
            0.25 * latents["glutamatergic_control_imbalance"]
            + 0.20 * inputs["ocrd_bfrb_shared_liability"]
            + 0.20 * latents["habit_loop_strengthening"]
            + 0.10 * inputs["environmental_stress_load"]
            + 0.10 * inputs["genetic_vulnerability"]
            - 0.15 * inputs["self_regulation_support"]
        )
        latents["frontolimbic_trigger_coupling"] = self._clip01(
            0.25 * latents["affect_dysregulation"]
            + 0.20 * latents["serotonergic_dysregulation"]
            + 0.20 * latents["emotional_trigger_sensitivity"]
            + 0.10 * inputs["depressive_anxiety_comorbidity"]
            - 0.10 * inputs["self_regulation_support"]
        )
        latents["hippocampal_stress_burden"] = self._clip01(
            0.25 * inputs["environmental_stress_load"]
            + 0.20 * inputs["depressive_anxiety_comorbidity"]
            + 0.15 * latents["affect_dysregulation"]
            + 0.10 * latents["emotional_trigger_sensitivity"]
        )
        latents["impulse_control_failure"] = self._clip01(
            0.25 * latents["frontostriatal_dysconnectivity"]
            + 0.20 * latents["glutamatergic_control_imbalance"]
            + 0.20 * latents["affect_dysregulation"]
            + 0.10 * latents["serotonergic_dysregulation"]
            + 0.10 * latents["automaticity_awareness_decoupling"]
            - 0.20 * inputs["self_regulation_support"]
        )
        latents["pulling_urge_generation"] = self._clip01(
            0.25 * latents["emotional_trigger_sensitivity"]
            + 0.20 * latents["dopaminergic_reward_relief_bias"]
            + 0.20 * latents["impulse_control_failure"]
            + 0.15 * latents["habit_loop_strengthening"]
            + 0.10 * latents["affect_dysregulation"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["ofc"] = self._clip01(
            0.35 * latents["impulse_control_failure"]
            + 0.20 * latents["frontostriatal_dysconnectivity"]
            + 0.10 * latents["serotonergic_dysregulation"]
            + 0.10 * inputs["environmental_stress_load"]
            - 0.15 * inputs["self_regulation_support"]
        )
        regional_state["dlpfc"] = self._clip01(
            0.30 * latents["impulse_control_failure"]
            + 0.25 * latents["frontostriatal_dysconnectivity"]
            + 0.10 * latents["glutamatergic_control_imbalance"]
            + 0.10 * inputs["depressive_anxiety_comorbidity"]
            - 0.15 * inputs["self_regulation_support"]
        )
        regional_state["acc"] = self._clip01(
            0.25 * latents["frontolimbic_trigger_coupling"]
            + 0.20 * latents["affect_dysregulation"]
            + 0.20 * latents["impulse_control_failure"]
            + 0.10 * inputs["environmental_stress_load"]
            - 0.10 * inputs["self_regulation_support"]
        )
        regional_state["caudate"] = self._clip01(
            0.30 * latents["habit_loop_strengthening"]
            + 0.20 * latents["frontostriatal_dysconnectivity"]
            + 0.10 * latents["automaticity_awareness_decoupling"]
            + 0.10 * inputs["ocrd_bfrb_shared_liability"]
        )
        regional_state["putamen"] = self._clip01(
            0.35 * latents["habit_loop_strengthening"]
            + 0.20 * latents["automaticity_awareness_decoupling"]
            + 0.15 * latents["frontostriatal_dysconnectivity"]
            + 0.10 * inputs["automatic_habit_proneness"]
        )
        regional_state["ventral_striatum_proxy"] = self._clip01(
            0.30 * latents["dopaminergic_reward_relief_bias"]
            + 0.20 * inputs["sensory_reinforcement_sensitivity"]
            + 0.20 * latents["habit_loop_strengthening"]
            + 0.10 * latents["pulling_urge_generation"]
        )
        regional_state["amygdala"] = self._clip01(
            0.30 * latents["frontolimbic_trigger_coupling"]
            + 0.20 * latents["emotional_trigger_sensitivity"]
            + 0.15 * latents["affect_dysregulation"]
            + 0.10 * inputs["environmental_stress_load"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.30 * latents["hippocampal_stress_burden"]
            + 0.20 * inputs["depressive_anxiety_comorbidity"]
            + 0.15 * latents["emotional_trigger_sensitivity"]
            + 0.10 * inputs["environmental_stress_load"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["hair_pulling_urge"] = self._clip01(
            0.30 * latents["pulling_urge_generation"]
            + 0.20 * regional_state["amygdala"]
            + 0.15 * regional_state["ventral_striatum_proxy"]
            + 0.10 * latents["affect_dysregulation"]
        )
        symptoms["failed_resistance"] = self._clip01(
            0.25 * latents["impulse_control_failure"]
            + 0.20 * regional_state["ofc"]
            + 0.15 * regional_state["dlpfc"]
            + 0.10 * regional_state["acc"]
            + 0.10 * symptoms["hair_pulling_urge"]
            - 0.15 * inputs["self_regulation_support"]
        )
        symptoms["automatic_pulling"] = self._clip01(
            0.30 * latents["automaticity_awareness_decoupling"]
            + 0.20 * regional_state["putamen"]
            + 0.15 * regional_state["caudate"]
            + 0.10 * latents["habit_loop_strengthening"]
        )
        symptoms["affect_triggered_pulling"] = self._clip01(
            0.25 * latents["emotional_trigger_sensitivity"]
            + 0.20 * regional_state["amygdala"]
            + 0.15 * regional_state["acc"]
            + 0.10 * symptoms["hair_pulling_urge"]
        )
        symptoms["relief_reinforced_pulling"] = self._clip01(
            0.25 * latents["dopaminergic_reward_relief_bias"]
            + 0.20 * regional_state["ventral_striatum_proxy"]
            + 0.15 * inputs["sensory_reinforcement_sensitivity"]
            + 0.10 * latents["habit_loop_strengthening"]
        )
        symptoms["repetitive_hair_pulling"] = self._clip01(
            0.20 * symptoms["failed_resistance"]
            + 0.20 * symptoms["automatic_pulling"]
            + 0.20 * symptoms["affect_triggered_pulling"]
            + 0.15 * symptoms["relief_reinforced_pulling"]
            + 0.10 * symptoms["hair_pulling_urge"]
            - 0.10 * inputs["self_regulation_support"]
        )
        symptoms["distress_impairment"] = self._clip01(
            0.25 * symptoms["repetitive_hair_pulling"]
            + 0.20 * symptoms["failed_resistance"]
            + 0.15 * inputs["depressive_anxiety_comorbidity"]
            + 0.10 * symptoms["affect_triggered_pulling"]
        )

        phenotypes = pd.Series(dtype=float, name="phenotypes")
        phenotypes["affect_regulation_profile"] = self._clip01(
            0.25 * symptoms["affect_triggered_pulling"]
            + 0.20 * symptoms["hair_pulling_urge"]
            + 0.15 * latents["affect_dysregulation"]
            + 0.10 * regional_state["amygdala"]
            + 0.10 * inputs["environmental_stress_load"]
        )
        phenotypes["automatic_habit_profile"] = self._clip01(
            0.25 * symptoms["automatic_pulling"]
            + 0.20 * regional_state["putamen"]
            + 0.15 * regional_state["caudate"]
            + 0.15 * latents["habit_loop_strengthening"]
            + 0.10 * latents["automaticity_awareness_decoupling"]
        )
        phenotypes["reward_relief_profile"] = self._clip01(
            0.25 * symptoms["relief_reinforced_pulling"]
            + 0.20 * regional_state["ventral_striatum_proxy"]
            + 0.15 * latents["dopaminergic_reward_relief_bias"]
            + 0.10 * inputs["sensory_reinforcement_sensitivity"]
            + 0.10 * symptoms["repetitive_hair_pulling"]
        )
        phenotypes["impulse_dyscontrol_profile"] = self._clip01(
            0.25 * symptoms["failed_resistance"]
            + 0.20 * regional_state["ofc"]
            + 0.15 * regional_state["dlpfc"]
            + 0.15 * latents["impulse_control_failure"]
            + 0.10 * latents["frontostriatal_dysconnectivity"]
        )
        phenotypes["overall_trichotillomania_burden"] = self._clip01(
            0.20 * symptoms["repetitive_hair_pulling"]
            + 0.20 * symptoms["distress_impairment"]
            + 0.15 * symptoms["failed_resistance"]
            + 0.15 * symptoms["automatic_pulling"]
            + 0.10 * symptoms["affect_triggered_pulling"]
            + 0.10 * symptoms["hair_pulling_urge"]
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

        point = siibra.Point(tuple(float(x) for x in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        sort_priority = [
            "map value",
            "value",
            "correlation",
            "contains",
            "contained",
            "intersection over union",
        ]
        lower_cols = {str(c).lower(): c for c in assignments.columns}
        for candidate in sort_priority:
            if candidate in lower_cols:
                assignments = assignments.sort_values(lower_cols[candidate], ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        region = self.region_objects.get(node_key)
        if region is None:
            candidates = self.region_candidates.get(node_key)
            if not candidates:
                return None
            region = self._resolve_region(candidates)
            if region is None:
                return None

        try:
            mask = region.get_regional_mask(space=self.assignment_space, maptype="labelled")
            return mask.fetch() if hasattr(mask, "fetch") else mask
        except Exception:
            pass

        try:
            return region.fetch_regional_map(space=self.assignment_space, maptype="labelled")
        except Exception:
            return None


if __name__ == "__main__":
    if siibra is None:
        print(
            "siibra is not installed in this environment. "
            "Install siibra-python to run the atlas-backed parts of this scaffold."
        )
    else:
        model = TrichotillomaniaModel()
        bundle = model.build(connectivity_rows=10)

        print("\n=== Nodes ===")
        print(
            bundle["nodes"][["key", "node_type", "atlas_region", "feature_summary"]]
            .fillna("")
            .to_string(index=False)
        )

        print("\n=== Edges (first 16) ===")
        print(bundle["edges"].head(16).to_string(index=False))

        print("\n=== Region suggestions for 'cingulate' ===")
        print(model.suggest_regions("cingulate").head(10).to_string(index=False))

        print("\n=== Example receptor / gene / connectivity tables ===")
        for region_key in (
            "ofc",
            "acc",
            "putamen",
            "ventral_striatum_proxy",
            "amygdala",
        ):
            print(f"\n-- {region_key} receptor fingerprint --")
            receptor_df = bundle["receptors"].get(region_key, pd.DataFrame())
            print(receptor_df.head(8).to_string(index=False) if not receptor_df.empty else "No receptor data found.")

            print(f"\n-- {region_key} gene summary --")
            gene_df = bundle["genes"].get(region_key, pd.DataFrame())
            print(gene_df.head(8).to_string(index=False) if not gene_df.empty else "No gene data found.")

            print(f"\n-- {region_key} connectivity profile --")
            conn_df = bundle["connectivity_profiles"].get(region_key, pd.DataFrame())
            print(conn_df.head(8).to_string(index=False) if not conn_df.empty else "No connectivity data found.")

        print("\n=== Circuit connectivity (top 20 rows) ===")
        circuit_df = bundle["circuit_connectivity"]
        print(circuit_df.head(20).to_string(index=False) if not circuit_df.empty else "No circuit connectivity matrix available.")

        print("\n=== Simulated TTM state ===")
        simulated = model.simulate(
            genetic_vulnerability=0.70,
            ocrd_bfrb_shared_liability=0.65,
            environmental_stress_load=0.75,
            negative_affect_trigger_load=0.80,
            boredom_understimulation_load=0.55,
            sensory_reinforcement_sensitivity=0.70,
            automatic_habit_proneness=0.68,
            depressive_anxiety_comorbidity=0.62,
            self_regulation_support=0.28,
        )
        for name, series in simulated.items():
            print(f"\n[{name}]")
            print(series.sort_values(ascending=False).to_string())

        # Example MNI coordinate assignment:
        # print(model.assign_mni_point((-8, 32, 22)).head(10).to_string(index=False))
