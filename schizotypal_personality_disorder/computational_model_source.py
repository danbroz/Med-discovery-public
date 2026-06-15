from __future__ import annotations

"""
Schizotypal Personality Disorder siibra scaffold.

This script translates a chapter on Schizotypal Personality Disorder (SPD)
into a transparent, atlas-grounded research scaffold.

Interpretive guardrails used here:
- The chapter is biologically rich but only moderately anatomically specific.
- SPD is placed on the schizophrenia spectrum, so several mechanisms are
  represented as cautious spectrum-informed latents rather than parcel-specific
  claims.
- Dopaminergic, glutamatergic, GABAergic, serotonergic, oxytocin-related, and
  endocannabinoid-related effects remain mostly latent biology, because the
  chapter does not localize them to a unique cortical or subcortical parcel.
- Frontal, temporal, thalamic, basal-ganglia, cerebellar, and parietal findings
  are anchored conservatively using Julich labels where possible and proxies
  where exact parcel resolution is not justified.
- Oxidative stress, low glutathione, inflammation, and mitochondrial burden are
  modeled as distributed injury-pressure latents rather than over-forced into a
  single structure.

This is a research scaffold for mechanistic exploration only. It is not a
validated disease model, not a diagnostic tool, and not a treatment guide.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception:  # pragma: no cover - portable fallback
    siibra = None  # type: ignore


DEFAULT_GENE_PANEL = [
    "COMT",      # 22q11.2 locus / prefrontal dopamine regulation proxy
    "DGCR8",     # 22q11.2 locus / neurodevelopmental RNA processing proxy
    "PRODH",     # 22q11.2 locus / glutamatergic-redox relevance proxy
    "SHANK3",    # synaptic scaffolding / autism-schizophrenia overlap
    "GTF2I",     # 7q11.23 locus proxy / social-cognitive developmental relevance
    "DRD2",      # dopamine salience signaling
    "SLC6A3",    # dopamine transporter
    "HTR2A",     # serotonergic modulation of salience / perception
    "SLC6A4",    # serotonin transporter
    "GRIN2A",    # glutamatergic receptor subunit
    "SLC1A2",    # glutamate transporter / excitatory regulation
    "GAD1",      # GABA synthesis
    "GABRB2",    # inhibitory signaling / schizophrenia-spectrum relevance
    "OXTR",      # social bonding / affiliative signaling proxy
    "CNR1",      # endocannabinoid modulation
    "GCLC",      # glutathione synthesis
    "GCLM",      # glutathione synthesis support
    "NFE2L2",    # oxidative stress defense / antioxidant response
]


class SchizotypalPersonalityDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Schizotypal Personality Disorder.

    Conceptual reading of the chapter:
    - SPD is modeled as a schizophrenia-spectrum neurodevelopmental condition
      with cognitive-perceptual, negative/interpersonal, and disorganized
      dimensions.
    - Shared synaptic-development liabilities are emphasized because the chapter
      highlights 22q11.2, SHANK3, 7q11.23-related mechanisms, and autism /
      schizophrenia overlap.
    - Dopamine, glutamate, GABA, and serotonin are the main transmitter-level
      latent systems. Oxytocin and endocannabinoid-related social-reward
      biology are represented through a social-bonding reward-blunting latent.
    - Oxidative stress, glutathione insufficiency, inflammation, and lower
      mitochondrial function are treated as distributed mechanisms increasing
      circuit inefficiency and psychosis-spectrum vulnerability.
    - The major circuit reading is frontotemporal-limbic dysregulation plus
      thalamocerebellar and parietal instability under cognitive demand.
    - Parietal involvement is weighted more strongly for attenuated psychotic
      features because the chapter explicitly ties emerging such features to
      altered parietal activation in high-risk populations.

    Anatomical caution:
    The chapter names lobes and broad circuit components more often than unique
    cytoarchitectonic parcels. Accordingly, several region nodes are explicit
    proxies and can be retuned with suggest_regions().
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
            except Exception as exc:  # pragma: no cover - depends on environment
                warnings.warn(
                    f"Could not initialize siibra atlas resources: {exc}. "
                    "The scaffold will continue with unresolved atlas nodes."
                )
                self.atlas = None
                self.parcellation = None
                self.space = None

        # Region choices follow the chapter's specificity level.
        # Frontal / temporal / parietal, thalamic, basal-ganglia, and cerebellar
        # nodes are expressed as proxies because the chapter is systems-level.
        self.region_candidates: Dict[str, List[str]] = {
            "frontal_cortex_proxy": [
                "Area 8v2 (MFG) left",
                "Area 8v1 (MFG) left",
                "MFG1 left",
                "MFG2 left",
                "frontal cortex",
                "prefrontal cortex",
                "frontal lobe",
            ],
            "temporal_cortex_proxy": [
                "Area TE 2.1 (STG) left",
                "Area TE 3 (STG) left",
                "Area TPJ (STG/SMG) left",
                "Area TE 1.2 (HESCHL) left",
                "Area TE 1.0 (HESCHL) left",
                "superior temporal gyrus",
                "temporal cortex",
                "temporal lobe",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "DG (Hippocampus) left",
                "hippocampus left",
                "hippocampus",
            ],
            "thalamus_proxy": [
                "CGM (Metathalamus) left",
                "CGL (Metathalamus) left",
                "thalamus left",
                "thalamus",
            ],
            "basal_ganglia_proxy": [
                "caudate nucleus left",
                "putamen left",
                "nucleus accumbens left",
                "basal ganglia",
                "striatum",
            ],
            "cerebellum_proxy": [
                "cerebellum left",
                "cerebellum",
            ],
            "parietal_cortex_proxy": [
                "Area PGp (IPL) left",
                "Area PGa (IPL) left",
                "Area 7A (SPL) left",
                "Area PFt (IPL) left",
                "Area PFcm (IPL) left",
                "parietal cortex",
                "parietal lobe",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_synaptic_liability": (
                "Shared schizophrenia-spectrum and autism-related synaptic-development liability "
                "including 22q11.2, SHANK3, and related neurodevelopmental pathways"
            ),
            "social_cognitive_developmental_load": (
                "Developmental burden on social cognition and communication consistent with the ASD-schizophrenia overlap discussed in the chapter"
            ),
            "oxidative_stress_load": (
                "Oxidative pressure contributing to neuronal and glial injury in psychosis-spectrum states"
            ),
            "proinflammatory_burden": (
                "Pro-inflammatory biological burden that can compromise neural signaling and integrity"
            ),
            "mitochondrial_energy_deficit": (
                "Reduced mitochondrial efficiency constraining energy metabolism in distributed circuits"
            ),
            "antioxidant_reserve": (
                "Protective glutathione and antioxidant buffering capacity"
            ),
            "cognitive_demand_load": (
                "Increasing cognitive demand that exposes latent frontal-thalamic-cerebellar inefficiency"
            ),
            "early_intervention_support": (
                "Protective early identification, support, and stabilizing intervention pressure"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "synaptic_development_disruption": (
                "Disturbed synaptic development and function spanning psychosis-spectrum and autism-related liabilities"
            ),
            "oxidative_inflammatory_neural_injury": (
                "Distributed injury pressure from oxidative stress, inflammation, and insufficient antioxidant buffering"
            ),
            "mitochondrial_constraint": (
                "Reduced bioenergetic capacity limiting neural resilience under cognitive demand"
            ),
            "glutamate_gaba_imbalance": (
                "Excitation-inhibition imbalance affecting plasticity, learning, and signal stability"
            ),
            "serotonin_dopamine_modulatory_shift": (
                "Serotonergic modulation altering salience, affective instability, and impulsive expression"
            ),
            "dopamine_salience_dysregulation": (
                "Abnormal salience assignment relevant to cognitive-perceptual oddity and psychosis-like experiences"
            ),
            "social_bonding_reward_blunting": (
                "Reduced affiliative and social reward tone consistent with oxytocin and endocannabinoid-related negative symptom biology"
            ),
            "frontotemporal_limbic_dysregulation": (
                "Distributed dysfunction across frontal, temporal, and limbic systems affecting executive control, emotion, and perception"
            ),
            "thalamocerebellar_coordination_failure": (
                "Inefficient frontal-thalamic-cerebellar coordination under increasing task demand"
            ),
            "parietal_signal_integration_shift": (
                "Altered parietal integration of sensory-cognitive signals linked to emerging attenuated psychotic features"
            ),
            "psychosis_transition_liability": (
                "Spectrum-level liability for transition toward more persistent attenuated or overt psychotic expression"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "odd_beliefs_eccentricity": (
                "Odd beliefs, eccentricity, and unusual cognitive-perceptual style"
            ),
            "unusual_perceptual_experiences": (
                "Psychotic-like or unusual perceptual experiences"
            ),
            "executive_dysfunction": (
                "Executive inefficiency and difficulty handling cognitive demands"
            ),
            "negative_interpersonal_detachment": (
                "Interpersonal distance, low social reciprocity, and discomfort with close relationships"
            ),
            "restricted_affect_asociality": (
                "Restricted affect and asocial negative-symptom expression"
            ),
            "disorganized_cognitive_style": (
                "Disorganized thought style and odd, difficult-to-follow cognition"
            ),
            "affective_instability_impulsivity": (
                "Affective instability and impulsive features that can be present in some individuals"
            ),
            "attenuated_psychotic_features": (
                "Subthreshold psychosis-spectrum features indicating increased transition vulnerability"
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_synaptic_liability",
                "target": "synaptic_development_disruption",
                "relation": "loads shared synaptic-development vulnerability across schizophrenia-spectrum and autism-related pathways",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "social_cognitive_developmental_load",
                "target": "synaptic_development_disruption",
                "relation": "adds developmental pressure on social-cognitive circuitry",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "oxidative_stress_load",
                "target": "oxidative_inflammatory_neural_injury",
                "relation": "raises redox-related neuronal and glial injury pressure",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "proinflammatory_burden",
                "target": "oxidative_inflammatory_neural_injury",
                "relation": "adds inflammatory disruption of neural signaling and integrity",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "mitochondrial_energy_deficit",
                "target": "oxidative_inflammatory_neural_injury",
                "relation": "lowers cellular resilience and magnifies distributed injury burden",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "mitochondrial_energy_deficit",
                "target": "mitochondrial_constraint",
                "relation": "reduces energy availability for distributed cognitive-emotional circuits",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "proinflammatory_burden",
                "target": "mitochondrial_constraint",
                "relation": "inflammation further compromises mitochondrial efficiency",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "antioxidant_reserve",
                "target": "oxidative_inflammatory_neural_injury",
                "relation": "buffers oxidative and glutathione-related injury pressure",
                "schizotypal_pd_change": "decreased",
            },
            {
                "source": "antioxidant_reserve",
                "target": "glutamate_gaba_imbalance",
                "relation": "reduces injury-driven excitation-inhibition destabilization",
                "schizotypal_pd_change": "decreased",
            },
            {
                "source": "early_intervention_support",
                "target": "psychosis_transition_liability",
                "relation": "buffers progression toward more persistent psychosis-spectrum expression",
                "schizotypal_pd_change": "decreased",
            },
            {
                "source": "early_intervention_support",
                "target": "thalamocerebellar_coordination_failure",
                "relation": "reduces destabilization during rising cognitive demand",
                "schizotypal_pd_change": "decreased",
            },
            {
                "source": "synaptic_development_disruption",
                "target": "glutamate_gaba_imbalance",
                "relation": "disordered synaptic development biases excitation-inhibition balance",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "synaptic_development_disruption",
                "target": "serotonin_dopamine_modulatory_shift",
                "relation": "developmental synaptic disruption changes modulatory transmitter coupling",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "synaptic_development_disruption",
                "target": "dopamine_salience_dysregulation",
                "relation": "supports abnormal salience assignment within the psychosis spectrum",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "social_cognitive_developmental_load",
                "target": "social_bonding_reward_blunting",
                "relation": "adds developmental pressure on affiliative and social reward processes",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "synaptic_development_disruption",
                "target": "social_bonding_reward_blunting",
                "relation": "weakens social reward and bonding-related processing",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "oxidative_inflammatory_neural_injury",
                "target": "glutamate_gaba_imbalance",
                "relation": "injury burden destabilizes plasticity and excitatory-inhibitory regulation",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "oxidative_inflammatory_neural_injury",
                "target": "frontotemporal_limbic_dysregulation",
                "relation": "distributed injury compromises cognitive-emotional circuitry",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "mitochondrial_constraint",
                "target": "frontotemporal_limbic_dysregulation",
                "relation": "bioenergetic limitation reduces efficient frontal-temporal-limbic coordination",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "glutamate_gaba_imbalance",
                "target": "frontotemporal_limbic_dysregulation",
                "relation": "excitation-inhibition imbalance destabilizes frontal, temporal, and limbic computations",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "serotonin_dopamine_modulatory_shift",
                "target": "dopamine_salience_dysregulation",
                "relation": "serotonergic modulation reshapes dopamine-driven salience attribution",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "serotonin_dopamine_modulatory_shift",
                "target": "affective_instability_impulsivity",
                "relation": "modulatory imbalance contributes to affective lability and impulsive expression",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "cognitive_demand_load",
                "target": "thalamocerebellar_coordination_failure",
                "relation": "increasing cognitive demands expose network inefficiency across frontal-thalamic-cerebellar loops",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "cognitive_demand_load",
                "target": "parietal_signal_integration_shift",
                "relation": "high demand exposes unstable parietal recruitment as symptoms intensify",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "glutamate_gaba_imbalance",
                "target": "thalamocerebellar_coordination_failure",
                "relation": "signal instability undermines large-scale cognitive coordination",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "mitochondrial_constraint",
                "target": "thalamocerebellar_coordination_failure",
                "relation": "energy limits reduce efficient network scaling under task demand",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "dopamine_salience_dysregulation",
                "target": "parietal_signal_integration_shift",
                "relation": "abnormal salience biases higher-order integration of perceptual signals",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "thalamocerebellar_coordination_failure",
                "target": "psychosis_transition_liability",
                "relation": "network inefficiency under demand raises transition vulnerability",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "parietal_signal_integration_shift",
                "target": "psychosis_transition_liability",
                "relation": "parietal instability is linked to emerging attenuated psychotic features",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "dopamine_salience_dysregulation",
                "target": "psychosis_transition_liability",
                "relation": "abnormal salience processing raises spectrum-level psychosis liability",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "frontotemporal_limbic_dysregulation",
                "target": "frontal_cortex_proxy",
                "relation": "maps executive and negative-symptom circuit burden onto frontal control systems",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "frontotemporal_limbic_dysregulation",
                "target": "temporal_cortex_proxy",
                "relation": "maps perceptual and memory-related circuit burden onto temporal cortex",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "frontotemporal_limbic_dysregulation",
                "target": "amygdala",
                "relation": "maps emotion-processing and social-cognitive burden onto amygdala state",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "frontotemporal_limbic_dysregulation",
                "target": "hippocampus",
                "relation": "maps memory and contextual processing burden onto hippocampal state",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "dopamine_salience_dysregulation",
                "target": "basal_ganglia_proxy",
                "relation": "loads subcortical salience and gating systems",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "glutamate_gaba_imbalance",
                "target": "basal_ganglia_proxy",
                "relation": "adds excitatory-inhibitory instability to subcortical regulation",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "thalamocerebellar_coordination_failure",
                "target": "thalamus_proxy",
                "relation": "maps demand-sensitive coordination failure onto thalamic relay systems",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "thalamocerebellar_coordination_failure",
                "target": "cerebellum_proxy",
                "relation": "maps coordination and timing inefficiency onto cerebellar systems",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "parietal_signal_integration_shift",
                "target": "parietal_cortex_proxy",
                "relation": "maps altered signal integration onto parietal recruitment",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "dopamine_salience_dysregulation",
                "target": "odd_beliefs_eccentricity",
                "relation": "abnormal salience promotes odd belief formation and eccentric interpretation",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "parietal_cortex_proxy",
                "target": "odd_beliefs_eccentricity",
                "relation": "altered parietal integration contributes to unusual meaning-making and cognitive-perceptual oddity",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "temporal_cortex_proxy",
                "target": "unusual_perceptual_experiences",
                "relation": "temporal abnormalities contribute to psychotic-like perceptual experiences",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "unusual_perceptual_experiences",
                "relation": "limbic-contextual disturbance contributes to unusual memory-perception integration",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "parietal_cortex_proxy",
                "target": "unusual_perceptual_experiences",
                "relation": "parietal recruitment changes amplify unusual perceptual-cognitive experience",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "frontal_cortex_proxy",
                "target": "executive_dysfunction",
                "relation": "frontal abnormalities impair executive control",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "thalamocerebellar_coordination_failure",
                "target": "executive_dysfunction",
                "relation": "coordination inefficiency under load worsens executive performance",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "cerebellum_proxy",
                "target": "executive_dysfunction",
                "relation": "cerebellar contribution to cognition supports executive dysfunction when impaired",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "executive_dysfunction",
                "relation": "subcortical gating deficits worsen executive efficiency",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "social_bonding_reward_blunting",
                "target": "negative_interpersonal_detachment",
                "relation": "reduced affiliative reward supports interpersonal distance and withdrawal",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "negative_interpersonal_detachment",
                "relation": "limbic social-emotional abnormalities contribute to impaired social cognition and distancing",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "negative_interpersonal_detachment",
                "relation": "motivational and gating dysfunction contribute to social disengagement",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "social_bonding_reward_blunting",
                "target": "restricted_affect_asociality",
                "relation": "affiliative blunting supports negative-symptom asociality and restricted affect",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "frontal_cortex_proxy",
                "target": "restricted_affect_asociality",
                "relation": "frontal abnormalities are linked to negative symptoms across the spectrum",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "frontal_cortex_proxy",
                "target": "disorganized_cognitive_style",
                "relation": "frontal inefficiency contributes to disorganized and eccentric thinking",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "thalamocerebellar_coordination_failure",
                "target": "disorganized_cognitive_style",
                "relation": "poor large-scale coordination destabilizes coherent cognitive sequencing",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "affective_instability_impulsivity",
                "relation": "limbic abnormalities contribute to emotional lability",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "psychosis_transition_liability",
                "target": "attenuated_psychotic_features",
                "relation": "transition liability expresses as subthreshold psychosis-spectrum symptoms",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "attenuated_psychotic_features",
                "relation": "thalamic network abnormality contributes to attenuated psychotic expression",
                "schizotypal_pd_change": "increased",
            },
            {
                "source": "parietal_cortex_proxy",
                "target": "attenuated_psychotic_features",
                "relation": "parietal activation shifts are linked to emerging attenuated psychotic features",
                "schizotypal_pd_change": "increased",
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
            "frontal cortex",
            "temporal cortex",
            "parietal cortex",
            "amygdala",
            "hippocampus",
            "thalamus",
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

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), feats[0])

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
        if region is None:
            return None
        exact = [x for x in labels if self._name_of(x) == getattr(region, "name", None)]
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
                    rows.append({"source": src_key, "target": dst_key, "value": float(value)})
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
            node_type = "region_proxy" if key.endswith("_proxy") else "region"
            if region is None:
                warnings.warn(
                    f"Could not resolve a region for node '{key}'. This is acceptable for a conservative scaffold."
                )
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " "),
                        "node_type": node_type,
                        "description": "Atlas-backed node unresolved in this siibra environment",
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
                    "node_type": node_type,
                    "description": "Atlas-backed circuit node for a chapter-grounded mechanism",
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
        genetic_synaptic_liability: float = 0.62,
        social_cognitive_developmental_load: float = 0.56,
        oxidative_stress_load: float = 0.50,
        proinflammatory_burden: float = 0.42,
        mitochondrial_energy_deficit: float = 0.40,
        antioxidant_reserve: float = 0.28,
        cognitive_demand_load: float = 0.58,
        early_intervention_support: float = 0.24,
    ) -> Dict[str, pd.Series]:
        """
        Transparent one-pass normalized simulator.

        Directionality follows the chapter's mechanistic reading:
        developmental liability and distributed biological burden -> latent
        biology -> regional state -> symptom dimensions -> phenotype summaries.

        All values are clipped to [0, 1].
        """

        inputs = pd.Series(
            {
                "genetic_synaptic_liability": self._clip01(genetic_synaptic_liability),
                "social_cognitive_developmental_load": self._clip01(social_cognitive_developmental_load),
                "oxidative_stress_load": self._clip01(oxidative_stress_load),
                "proinflammatory_burden": self._clip01(proinflammatory_burden),
                "mitochondrial_energy_deficit": self._clip01(mitochondrial_energy_deficit),
                "antioxidant_reserve": self._clip01(antioxidant_reserve),
                "cognitive_demand_load": self._clip01(cognitive_demand_load),
                "early_intervention_support": self._clip01(early_intervention_support),
            },
            name="value",
        )

        latents = pd.Series(dtype=float, name="value")
        latents["oxidative_inflammatory_neural_injury"] = self._clip01(
            0.34 * inputs["oxidative_stress_load"]
            + 0.26 * inputs["proinflammatory_burden"]
            + 0.16 * inputs["mitochondrial_energy_deficit"]
            + 0.10 * inputs["cognitive_demand_load"]
            - 0.20 * inputs["antioxidant_reserve"]
            - 0.08 * inputs["early_intervention_support"]
        )
        latents["mitochondrial_constraint"] = self._clip01(
            0.44 * inputs["mitochondrial_energy_deficit"]
            + 0.18 * inputs["proinflammatory_burden"]
            + 0.16 * inputs["oxidative_stress_load"]
            + 0.08 * inputs["cognitive_demand_load"]
            - 0.12 * inputs["antioxidant_reserve"]
        )
        latents["synaptic_development_disruption"] = self._clip01(
            0.40 * inputs["genetic_synaptic_liability"]
            + 0.24 * inputs["social_cognitive_developmental_load"]
            + 0.14 * latents["mitochondrial_constraint"]
            + 0.10 * inputs["proinflammatory_burden"]
            + 0.08 * latents["oxidative_inflammatory_neural_injury"]
            - 0.10 * inputs["early_intervention_support"]
        )
        latents["glutamate_gaba_imbalance"] = self._clip01(
            0.34 * latents["synaptic_development_disruption"]
            + 0.26 * latents["oxidative_inflammatory_neural_injury"]
            + 0.18 * latents["mitochondrial_constraint"]
            + 0.12 * inputs["cognitive_demand_load"]
            - 0.10 * inputs["antioxidant_reserve"]
        )
        latents["serotonin_dopamine_modulatory_shift"] = self._clip01(
            0.28 * inputs["genetic_synaptic_liability"]
            + 0.22 * latents["synaptic_development_disruption"]
            + 0.18 * latents["oxidative_inflammatory_neural_injury"]
            + 0.16 * inputs["social_cognitive_developmental_load"]
            + 0.08 * inputs["cognitive_demand_load"]
            - 0.10 * inputs["early_intervention_support"]
        )
        latents["dopamine_salience_dysregulation"] = self._clip01(
            0.34 * latents["synaptic_development_disruption"]
            + 0.24 * latents["glutamate_gaba_imbalance"]
            + 0.20 * latents["serotonin_dopamine_modulatory_shift"]
            + 0.12 * inputs["cognitive_demand_load"]
            + 0.08 * latents["oxidative_inflammatory_neural_injury"]
            - 0.10 * inputs["early_intervention_support"]
        )
        latents["social_bonding_reward_blunting"] = self._clip01(
            0.30 * inputs["social_cognitive_developmental_load"]
            + 0.26 * latents["synaptic_development_disruption"]
            + 0.18 * latents["serotonin_dopamine_modulatory_shift"]
            + 0.12 * latents["dopamine_salience_dysregulation"]
            + 0.08 * latents["oxidative_inflammatory_neural_injury"]
            - 0.12 * inputs["early_intervention_support"]
        )
        latents["frontotemporal_limbic_dysregulation"] = self._clip01(
            0.28 * latents["glutamate_gaba_imbalance"]
            + 0.22 * latents["dopamine_salience_dysregulation"]
            + 0.18 * latents["oxidative_inflammatory_neural_injury"]
            + 0.12 * latents["mitochondrial_constraint"]
            + 0.10 * inputs["social_cognitive_developmental_load"]
            + 0.10 * inputs["cognitive_demand_load"]
            - 0.12 * inputs["early_intervention_support"]
        )
        latents["thalamocerebellar_coordination_failure"] = self._clip01(
            0.34 * inputs["cognitive_demand_load"]
            + 0.22 * latents["glutamate_gaba_imbalance"]
            + 0.18 * latents["mitochondrial_constraint"]
            + 0.14 * latents["frontotemporal_limbic_dysregulation"]
            + 0.08 * latents["dopamine_salience_dysregulation"]
            - 0.12 * inputs["early_intervention_support"]
        )
        latents["parietal_signal_integration_shift"] = self._clip01(
            0.30 * inputs["cognitive_demand_load"]
            + 0.24 * latents["thalamocerebellar_coordination_failure"]
            + 0.18 * latents["dopamine_salience_dysregulation"]
            + 0.12 * latents["frontotemporal_limbic_dysregulation"]
            + 0.10 * inputs["social_cognitive_developmental_load"]
            - 0.10 * inputs["early_intervention_support"]
        )
        latents["psychosis_transition_liability"] = self._clip01(
            0.26 * latents["dopamine_salience_dysregulation"]
            + 0.22 * latents["thalamocerebellar_coordination_failure"]
            + 0.18 * latents["parietal_signal_integration_shift"]
            + 0.14 * latents["oxidative_inflammatory_neural_injury"]
            + 0.12 * latents["frontotemporal_limbic_dysregulation"]
            + 0.08 * inputs["cognitive_demand_load"]
            - 0.14 * inputs["early_intervention_support"]
        )

        regional_state = pd.Series(dtype=float, name="value")
        regional_state["frontal_cortex_proxy"] = self._clip01(
            0.40 * latents["frontotemporal_limbic_dysregulation"]
            + 0.24 * latents["glutamate_gaba_imbalance"]
            + 0.16 * latents["thalamocerebellar_coordination_failure"]
            + 0.10 * inputs["cognitive_demand_load"]
            + 0.06 * latents["synaptic_development_disruption"]
            - 0.14 * inputs["early_intervention_support"]
        )
        regional_state["temporal_cortex_proxy"] = self._clip01(
            0.36 * latents["frontotemporal_limbic_dysregulation"]
            + 0.26 * latents["dopamine_salience_dysregulation"]
            + 0.14 * latents["synaptic_development_disruption"]
            + 0.12 * latents["oxidative_inflammatory_neural_injury"]
            + 0.08 * latents["parietal_signal_integration_shift"]
            - 0.10 * inputs["early_intervention_support"]
        )
        regional_state["amygdala"] = self._clip01(
            0.34 * latents["frontotemporal_limbic_dysregulation"]
            + 0.24 * latents["serotonin_dopamine_modulatory_shift"]
            + 0.18 * latents["dopamine_salience_dysregulation"]
            + 0.12 * latents["oxidative_inflammatory_neural_injury"]
            + 0.08 * inputs["social_cognitive_developmental_load"]
            - 0.10 * inputs["early_intervention_support"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.32 * latents["frontotemporal_limbic_dysregulation"]
            + 0.26 * latents["oxidative_inflammatory_neural_injury"]
            + 0.18 * latents["mitochondrial_constraint"]
            + 0.14 * latents["synaptic_development_disruption"]
            + 0.08 * inputs["cognitive_demand_load"]
            - 0.10 * inputs["early_intervention_support"]
        )
        regional_state["thalamus_proxy"] = self._clip01(
            0.38 * latents["thalamocerebellar_coordination_failure"]
            + 0.24 * latents["parietal_signal_integration_shift"]
            + 0.16 * inputs["cognitive_demand_load"]
            + 0.12 * latents["dopamine_salience_dysregulation"]
            - 0.10 * inputs["early_intervention_support"]
        )
        regional_state["basal_ganglia_proxy"] = self._clip01(
            0.34 * latents["dopamine_salience_dysregulation"]
            + 0.24 * latents["social_bonding_reward_blunting"]
            + 0.18 * latents["frontotemporal_limbic_dysregulation"]
            + 0.14 * latents["glutamate_gaba_imbalance"]
            + 0.08 * inputs["cognitive_demand_load"]
            - 0.10 * inputs["early_intervention_support"]
        )
        regional_state["cerebellum_proxy"] = self._clip01(
            0.42 * latents["thalamocerebellar_coordination_failure"]
            + 0.24 * inputs["cognitive_demand_load"]
            + 0.14 * latents["mitochondrial_constraint"]
            + 0.10 * latents["glutamate_gaba_imbalance"]
            - 0.10 * inputs["early_intervention_support"]
        )
        regional_state["parietal_cortex_proxy"] = self._clip01(
            0.40 * latents["parietal_signal_integration_shift"]
            + 0.24 * latents["thalamocerebellar_coordination_failure"]
            + 0.18 * inputs["cognitive_demand_load"]
            + 0.10 * latents["dopamine_salience_dysregulation"]
            - 0.10 * inputs["early_intervention_support"]
        )

        symptoms = pd.Series(dtype=float, name="value")
        symptoms["odd_beliefs_eccentricity"] = self._clip01(
            0.26 * latents["dopamine_salience_dysregulation"]
            + 0.22 * regional_state["parietal_cortex_proxy"]
            + 0.20 * regional_state["temporal_cortex_proxy"]
            + 0.14 * regional_state["frontal_cortex_proxy"]
            + 0.10 * latents["psychosis_transition_liability"]
            + 0.08 * latents["serotonin_dopamine_modulatory_shift"]
            - 0.08 * inputs["early_intervention_support"]
        )
        symptoms["unusual_perceptual_experiences"] = self._clip01(
            0.30 * regional_state["temporal_cortex_proxy"]
            + 0.22 * regional_state["parietal_cortex_proxy"]
            + 0.16 * regional_state["thalamus_proxy"]
            + 0.12 * regional_state["hippocampus"]
            + 0.10 * latents["dopamine_salience_dysregulation"]
            + 0.10 * latents["psychosis_transition_liability"]
            - 0.08 * inputs["early_intervention_support"]
        )
        symptoms["executive_dysfunction"] = self._clip01(
            0.32 * regional_state["frontal_cortex_proxy"]
            + 0.22 * latents["thalamocerebellar_coordination_failure"]
            + 0.16 * regional_state["basal_ganglia_proxy"]
            + 0.14 * inputs["cognitive_demand_load"]
            + 0.08 * regional_state["cerebellum_proxy"]
            + 0.08 * regional_state["hippocampus"]
            - 0.10 * inputs["early_intervention_support"]
        )
        symptoms["negative_interpersonal_detachment"] = self._clip01(
            0.28 * latents["social_bonding_reward_blunting"]
            + 0.22 * regional_state["basal_ganglia_proxy"]
            + 0.18 * inputs["social_cognitive_developmental_load"]
            + 0.16 * regional_state["temporal_cortex_proxy"]
            + 0.10 * regional_state["amygdala"]
            - 0.12 * inputs["early_intervention_support"]
        )
        symptoms["restricted_affect_asociality"] = self._clip01(
            0.32 * latents["social_bonding_reward_blunting"]
            + 0.22 * regional_state["amygdala"]
            + 0.18 * regional_state["basal_ganglia_proxy"]
            + 0.12 * regional_state["frontal_cortex_proxy"]
            + 0.10 * inputs["social_cognitive_developmental_load"]
            - 0.12 * inputs["early_intervention_support"]
        )
        symptoms["disorganized_cognitive_style"] = self._clip01(
            0.28 * regional_state["frontal_cortex_proxy"]
            + 0.24 * latents["thalamocerebellar_coordination_failure"]
            + 0.20 * regional_state["parietal_cortex_proxy"]
            + 0.12 * latents["dopamine_salience_dysregulation"]
            + 0.10 * latents["glutamate_gaba_imbalance"]
            - 0.08 * inputs["early_intervention_support"]
        )
        symptoms["affective_instability_impulsivity"] = self._clip01(
            0.30 * latents["serotonin_dopamine_modulatory_shift"]
            + 0.22 * regional_state["amygdala"]
            + 0.18 * regional_state["basal_ganglia_proxy"]
            + 0.14 * latents["dopamine_salience_dysregulation"]
            + 0.10 * latents["frontotemporal_limbic_dysregulation"]
            - 0.08 * inputs["early_intervention_support"]
        )
        symptoms["attenuated_psychotic_features"] = self._clip01(
            0.26 * latents["psychosis_transition_liability"]
            + 0.22 * symptoms["unusual_perceptual_experiences"]
            + 0.18 * symptoms["odd_beliefs_eccentricity"]
            + 0.14 * regional_state["parietal_cortex_proxy"]
            + 0.12 * regional_state["thalamus_proxy"]
            + 0.08 * latents["dopamine_salience_dysregulation"]
            - 0.10 * inputs["early_intervention_support"]
        )

        phenotypes = pd.Series(dtype=float, name="value")
        phenotypes["cognitive_perceptual_schizotypy_profile"] = self._clip01(
            (
                symptoms["odd_beliefs_eccentricity"]
                + symptoms["unusual_perceptual_experiences"]
                + symptoms["attenuated_psychotic_features"]
                + regional_state["parietal_cortex_proxy"]
            )
            / 4.0
        )
        phenotypes["negative_interpersonal_schizotypy_profile"] = self._clip01(
            (
                symptoms["negative_interpersonal_detachment"]
                + symptoms["restricted_affect_asociality"]
                + latents["social_bonding_reward_blunting"]
                + regional_state["basal_ganglia_proxy"]
            )
            / 4.0
        )
        phenotypes["disorganized_executive_spectrum_profile"] = self._clip01(
            (
                symptoms["executive_dysfunction"]
                + symptoms["disorganized_cognitive_style"]
                + regional_state["frontal_cortex_proxy"]
                + latents["thalamocerebellar_coordination_failure"]
            )
            / 4.0
        )
        phenotypes["psychosis_transition_vulnerability_profile"] = self._clip01(
            (
                symptoms["attenuated_psychotic_features"]
                + latents["psychosis_transition_liability"]
                + regional_state["thalamus_proxy"]
                + regional_state["parietal_cortex_proxy"]
            )
            / 4.0
        )
        phenotypes["affective_labile_spectrum_profile"] = self._clip01(
            (
                symptoms["affective_instability_impulsivity"]
                + regional_state["amygdala"]
                + latents["serotonin_dopamine_modulatory_shift"]
                + latents["dopamine_salience_dysregulation"]
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
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 24)

    model = SchizotypalPersonalityDisorderModel()
    bundle = model.build()

    print("\n=== Nodes ===")
    print(bundle["nodes"].head(40).to_string(index=False))

    print("\n=== Edges ===")
    print(bundle["edges"].to_string(index=False))

    print("\n=== Resolved regions ===")
    if bundle["regions"]:
        for key, region in bundle["regions"].items():
            print(f"- {key}: {getattr(region, 'name', region)}")
    else:
        print("No atlas regions resolved in this environment.")

    print("\n=== Example receptor table: temporal_cortex_proxy ===")
    temporal_receptors = bundle["receptors"].get("temporal_cortex_proxy", pd.DataFrame())
    print(temporal_receptors.head(10).to_string(index=False) if not temporal_receptors.empty else "No receptor data available.")

    print("\n=== Example gene table: hippocampus ===")
    hippocampus_genes = bundle["genes"].get("hippocampus", pd.DataFrame())
    print(hippocampus_genes.head(10).to_string(index=False) if not hippocampus_genes.empty else "No gene-expression data available.")

    print("\n=== Example connectivity profile: frontal_cortex_proxy ===")
    frontal_conn = bundle["connectivity_profiles"].get("frontal_cortex_proxy", pd.DataFrame())
    print(frontal_conn.head(10).to_string(index=False) if not frontal_conn.empty else "No connectivity data available.")

    print("\n=== Circuit connectivity ===")
    circuit_df = bundle["circuit_connectivity"]
    print(circuit_df.head(20).to_string(index=False) if not circuit_df.empty else "No circuit connectivity matrix available.")

    print("\n=== Simulation example ===")
    sim = model.simulate(
        genetic_synaptic_liability=0.70,
        social_cognitive_developmental_load=0.62,
        oxidative_stress_load=0.56,
        proinflammatory_burden=0.44,
        mitochondrial_energy_deficit=0.46,
        antioxidant_reserve=0.24,
        cognitive_demand_load=0.68,
        early_intervention_support=0.22,
    )
    for name, series in sim.items():
        print(f"\n[{name}]")
        print(series.round(3).to_string())

    # Example coordinate query, when siibra resources are available:
    # print(model.assign_mni_point((-48, -20, 8)).head(10).to_string(index=False))
