from __future__ import annotations

"""
Major Neurocognitive Disorder siibra scaffold
===========================================

Research scaffold that translates a narrative chapter on Major
Neurocognitive Disorder (MND) into an atlas-grounded mechanistic model using
siibra.

This script is intentionally conservative:
- it treats MND as a syndrome-level umbrella rather than a single disease,
- it uses mixture inputs to represent Alzheimer-like, frontotemporal,
  vascular, synucleinopathy-related, sleep-related, and rarer genetic/subcortical
  contributors,
- it keeps distributed network concepts (default-mode, salience, executive,
  cortical-subcortical disconnection) partly latent or proxy-based rather than
  over-claiming parcel precision,
- it tolerates missing receptor, gene-expression, and connectivity features
  without crashing.

This is a research scaffold, not a diagnostic or treatment tool.
"""

import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This scaffold requires the 'siibra' package. "
        "Install it in your environment before running the script."
    ) from exc


DEFAULT_GENE_PANEL = [
    "APP",
    "PSEN1",
    "PSEN2",
    "APOE",
    "MAPT",
    "GRN",
    "C9ORF72",
    "SNCA",
    "PRKN",
    "HTT",
    "FMR1",
    "BDNF",
]


class MajorNeurocognitiveDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Major Neurocognitive Disorder.

    Chapter translation choices
    ---------------------------
    Inputs
        - genetic neurodegenerative liability
        - Alzheimer-like pathology load
        - frontotemporal pathology load
        - vascular / cerebrovascular burden
        - synucleinopathy burden
        - subcortical / rarer genetic neurodegenerative load
        - sleep apnea hypoxia burden
        - REM sleep behavior disorder burden
        - sleep hygiene support
        - disease-management support

    Latent biology
        - medial temporal degeneration
        - temporo-parietal association failure
        - frontotemporal network degeneration
        - cortical-subcortical disconnection
        - default mode network disconnectivity
        - salience / executive network failure
        - synucleinopathy sleep prodrome
        - diffuse neurocognitive decline

    Regions / proxies
        - hippocampus
        - entorhinal cortex
        - temporo-parietal association proxy
        - prefrontal control proxy
        - anterior temporal proxy
        - left temporal pole
        - left perisylvian language proxy
        - posterior cingulate / precuneus DMN proxy
        - ACC salience proxy
        - insula salience proxy

    Symptoms
        - episodic memory impairment
        - executive / attention impairment
        - behavioral / social conduct change
        - semantic knowledge loss
        - nonfluent language impairment
        - diffuse cognitive / functional decline

    Notes
    -----
    The chapter discusses Major Neurocognitive Disorder as a final common
    clinical syndrome arising from multiple etiologies rather than one unified
    mechanism. Accordingly, this scaffold uses etiology-mixture inputs and
    phenotype summaries for Alzheimer-like, frontotemporal, semantic,
    nonfluent-aphasic, vascular-disconnection, and synucleinopathy-prodromal
    profiles.
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

        self.region_candidates: Dict[str, List[str]] = {
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "DG (Hippocampus) left",
                "HC-Subiculum (Hippocampus) left",
                "hippocampus",
            ],
            "entorhinal_cortex": [
                "Area 28 left",
                "Area 28",
                "entorhinal cortex",
                "entorhinal",
                "EC",
            ],
            "temporoparietal_association_proxy": [
                "Area PGp (IPL) left",
                "Area PGa (IPL) left",
                "Area PFm (IPL) left",
                "inferior parietal lobule",
                "angular gyrus",
            ],
            "pfc_control_proxy": [
                "Area 9/46d left",
                "Area 46 left",
                "Area 9 left",
                "Area Fp2 (FPole) left",
                "prefrontal cortex",
            ],
            "anterior_temporal_proxy": [
                "Area TG (Temporal Pole) left",
                "anterior temporal",
                "temporal pole",
                "temporal lobe",
            ],
            "left_temporal_pole": [
                "Area TG (Temporal Pole) left",
                "temporal pole left",
                "temporal pole",
            ],
            "left_perisylvian_language_proxy": [
                "Area 44 (IFG) left",
                "Area 45 (IFG) left",
                "inferior frontal gyrus left",
                "perisylvian",
                "Broca",
            ],
            "pcc_precuneus_dmn_proxy": [
                "Area 31 (pCC) left",
                "Area 23 (pCC) left",
                "precuneus",
                "posterior cingulate cortex",
                "posterior cingulate",
            ],
            "acc_salience_proxy": [
                "Area p32 (pACC) left",
                "Area s24 (sACC) left",
                "Area 33 (ACC) left",
                "anterior cingulate cortex",
            ],
            "insula_salience_proxy": [
                "Area Id1 (Insula) left",
                "Area Id2 (Insula) left",
                "Area Ia1 (Insula) left",
                "insula",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "hippocampus": (
                "Atlas-backed medial temporal memory node highlighted by the chapter's "
                "discussion of hippocampal atrophy in Alzheimer disease and damage from "
                "sleep-apnea-related hypoxia."
            ),
            "entorhinal_cortex": (
                "Atlas-backed or near-atlas medial temporal gateway node corresponding "
                "to early entorhinal involvement in Alzheimer-like amnestic decline."
            ),
            "temporoparietal_association_proxy": (
                "Conservative proxy for temporo-parietal association cortex involved in "
                "later Alzheimer-like spread and broader association-network failure."
            ),
            "pfc_control_proxy": (
                "Conservative prefrontal control proxy capturing frontal atrophy, "
                "executive-control burden, and behavioral regulation deficits."
            ),
            "anterior_temporal_proxy": (
                "Conservative anterior temporal proxy for frontotemporal syndromes, "
                "especially behavioral and semantic variants."
            ),
            "left_temporal_pole": (
                "Left temporal-pole node emphasized by the chapter as a hallmark of "
                "semantic dementia."
            ),
            "left_perisylvian_language_proxy": (
                "Left perisylvian language proxy corresponding to progressive nonfluent "
                "aphasia patterns."
            ),
            "pcc_precuneus_dmn_proxy": (
                "Default-mode hub proxy centered on posterior cingulate / precuneus, "
                "which the chapter names as a key DMN node disrupted in Alzheimer disease."
            ),
            "acc_salience_proxy": (
                "ACC-centered salience-network proxy for attentional switching, conflict "
                "monitoring, and executive-affective control."
            ),
            "insula_salience_proxy": (
                "Insula-centered salience-network proxy, strongly implied by the chapter's "
                "salience-network discussion."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_neurodegenerative_liability": (
                "Aggregate genetic liability across familial neurodegenerative and related "
                "risk pathways."
            ),
            "alzheimer_like_pathology_load": (
                "Disease-load input emphasizing early medial temporal and later temporo-" 
                "parietal involvement characteristic of Alzheimer-like presentations."
            ),
            "frontotemporal_pathology_load": (
                "Disease-load input emphasizing frontal and anterior temporal degeneration "
                "as described for bvFTD and related syndromes."
            ),
            "vascular_cerebrovascular_burden": (
                "Cerebrovascular burden including infarcts and white-matter disease that "
                "can disrupt cortical-subcortical circuits."
            ),
            "synucleinopathy_burden": (
                "Burden related to Lewy-body / Parkinsonian synucleinopathy risk pathways."
            ),
            "subcortical_genetic_neurodegeneration_load": (
                "Rarer genetic or subcortical neurodegenerative burden, including patterns "
                "relevant to Huntington disease and other non-AD contributors."
            ),
            "sleep_apnea_hypoxia": (
                "Intermittent hypoxia burden from obstructive sleep apnea contributing to "
                "hippocampal and executive vulnerability."
            ),
            "rem_sleep_behavior_disorder_burden": (
                "REM sleep behavior disorder burden as a prodromal synucleinopathy signal."
            ),
            "sleep_hygiene_support": (
                "Protective non-pharmacological sleep-management support."
            ),
            "disease_management_support": (
                "Generic diagnostic, treatment, and supportive-care scaffold variable that "
                "reduces downstream burden."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "medial_temporal_degeneration": (
                "Medial temporal system degeneration linking hippocampal and entorhinal "
                "vulnerability to amnestic decline."
            ),
            "temporoparietal_association_failure": (
                "Association-cortex failure representing Alzheimer-like spread into "
                "temporo-parietal systems."
            ),
            "frontotemporal_network_degeneration": (
                "Frontal and anterior temporal degeneration underlying behavioral and "
                "language-variant neurocognitive syndromes."
            ),
            "cortical_subcortical_disconnection": (
                "Distributed circuit disconnection driven by vascular burden and other "
                "subcortical insults."
            ),
            "default_mode_network_disconnectivity": (
                "Reduced integrity of the default mode network, especially medial temporal "
                "and posterior cingulate / precuneus coupling."
            ),
            "salience_executive_network_failure": (
                "Breakdown in salience and executive-control network coordination, linked "
                "to attentional and executive deficits."
            ),
            "synucleinopathy_sleep_prodrome": (
                "Prodromal sleep-linked synucleinopathy burden emphasizing REM-sleep "
                "behavior disorder and downstream cognitive risk."
            ),
            "diffuse_neurocognitive_decline": (
                "Global convergence node summarizing widespread network degeneration and "
                "functional decline."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "episodic_memory_impairment": (
                "Prominent episodic-memory impairment aligned with hippocampal and "
                "entorhinal degeneration."
            ),
            "executive_attention_impairment": (
                "Attention and executive-function impairment linked to frontal, salience, "
                "and disconnection burdens."
            ),
            "behavioral_social_conduct_change": (
                "Changes in personality, social conduct, and behavioral regulation."
            ),
            "semantic_knowledge_loss": (
                "Loss of conceptual / semantic knowledge, especially in semantic-dementia-" 
                "like presentations."
            ),
            "nonfluent_language_impairment": (
                "Effortful, nonfluent language impairment consistent with left perisylvian "
                "involvement."
            ),
            "diffuse_cognitive_functional_decline": (
                "Broad global cognitive and functional decline reflecting the final common "
                "clinical syndrome of MND."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_neurodegenerative_liability",
                "target": "medial_temporal_degeneration",
                "relation": "genetic vulnerability can increase susceptibility to amnestic neurodegeneration",
                "mnd_change": "increased",
            },
            {
                "source": "genetic_neurodegenerative_liability",
                "target": "frontotemporal_network_degeneration",
                "relation": "familial and heritable factors load frontal and anterior temporal syndromes",
                "mnd_change": "increased",
            },
            {
                "source": "genetic_neurodegenerative_liability",
                "target": "synucleinopathy_sleep_prodrome",
                "relation": "familial synucleinopathy-related genes raise prodromal risk",
                "mnd_change": "increased",
            },
            {
                "source": "alzheimer_like_pathology_load",
                "target": "medial_temporal_degeneration",
                "relation": "Alzheimer-like disease burden first targets hippocampal and entorhinal systems",
                "mnd_change": "increased",
            },
            {
                "source": "alzheimer_like_pathology_load",
                "target": "temporoparietal_association_failure",
                "relation": "Alzheimer-like progression spreads toward temporo-parietal association cortex",
                "mnd_change": "increased",
            },
            {
                "source": "frontotemporal_pathology_load",
                "target": "frontotemporal_network_degeneration",
                "relation": "frontotemporal disease burden drives frontal and anterior temporal degeneration",
                "mnd_change": "increased",
            },
            {
                "source": "vascular_cerebrovascular_burden",
                "target": "cortical_subcortical_disconnection",
                "relation": "infarcts and white-matter disease disrupt cortical-subcortical circuits",
                "mnd_change": "increased",
            },
            {
                "source": "synucleinopathy_burden",
                "target": "synucleinopathy_sleep_prodrome",
                "relation": "synucleinopathy burden is linked to prodromal REM-sleep behavior disorder features",
                "mnd_change": "increased",
            },
            {
                "source": "subcortical_genetic_neurodegeneration_load",
                "target": "cortical_subcortical_disconnection",
                "relation": "subcortical and rarer genetic neurodegeneration add distributed circuit burden",
                "mnd_change": "increased",
            },
            {
                "source": "subcortical_genetic_neurodegeneration_load",
                "target": "diffuse_neurocognitive_decline",
                "relation": "rarer genetic and subcortical disorders can contribute directly to broad decline",
                "mnd_change": "increased",
            },
            {
                "source": "sleep_apnea_hypoxia",
                "target": "medial_temporal_degeneration",
                "relation": "intermittent hypoxia contributes to hippocampal damage",
                "mnd_change": "increased",
            },
            {
                "source": "sleep_apnea_hypoxia",
                "target": "salience_executive_network_failure",
                "relation": "sleep-apnea burden worsens attention and executive vulnerability",
                "mnd_change": "increased",
            },
            {
                "source": "rem_sleep_behavior_disorder_burden",
                "target": "synucleinopathy_sleep_prodrome",
                "relation": "RBD is modeled as a prodromal synucleinopathy signal",
                "mnd_change": "increased",
            },
            {
                "source": "sleep_hygiene_support",
                "target": "synucleinopathy_sleep_prodrome",
                "relation": "sleep-focused support can reduce downstream sleep-linked burden",
                "mnd_change": "decreased",
            },
            {
                "source": "sleep_hygiene_support",
                "target": "salience_executive_network_failure",
                "relation": "better sleep management can reduce executive strain from sleep disruption",
                "mnd_change": "decreased",
            },
            {
                "source": "disease_management_support",
                "target": "diffuse_neurocognitive_decline",
                "relation": "supportive care can reduce downstream functional burden",
                "mnd_change": "decreased",
            },
            {
                "source": "disease_management_support",
                "target": "salience_executive_network_failure",
                "relation": "management support can partially reduce executive strain",
                "mnd_change": "decreased",
            },
            {
                "source": "medial_temporal_degeneration",
                "target": "hippocampus",
                "relation": "medial temporal degeneration burdens hippocampal memory circuitry",
                "mnd_change": "increased",
            },
            {
                "source": "medial_temporal_degeneration",
                "target": "entorhinal_cortex",
                "relation": "medial temporal degeneration burdens entorhinal gateway cortex",
                "mnd_change": "increased",
            },
            {
                "source": "temporoparietal_association_failure",
                "target": "temporoparietal_association_proxy",
                "relation": "association-network failure loads temporo-parietal cortex",
                "mnd_change": "increased",
            },
            {
                "source": "frontotemporal_network_degeneration",
                "target": "pfc_control_proxy",
                "relation": "frontotemporal disease burden impairs frontal executive control systems",
                "mnd_change": "increased",
            },
            {
                "source": "frontotemporal_network_degeneration",
                "target": "anterior_temporal_proxy",
                "relation": "frontotemporal disease burden loads anterior temporal cortex",
                "mnd_change": "increased",
            },
            {
                "source": "frontotemporal_network_degeneration",
                "target": "left_temporal_pole",
                "relation": "semantic-dementia-like degeneration burdens the left temporal pole",
                "mnd_change": "increased",
            },
            {
                "source": "frontotemporal_network_degeneration",
                "target": "left_perisylvian_language_proxy",
                "relation": "nonfluent language variants burden left perisylvian systems",
                "mnd_change": "increased",
            },
            {
                "source": "default_mode_network_disconnectivity",
                "target": "pcc_precuneus_dmn_proxy",
                "relation": "DMN failure is centered on posterior cingulate / precuneus hubs",
                "mnd_change": "increased",
            },
            {
                "source": "default_mode_network_disconnectivity",
                "target": "hippocampus",
                "relation": "DMN failure couples with medial temporal dysfunction",
                "mnd_change": "increased",
            },
            {
                "source": "default_mode_network_disconnectivity",
                "target": "temporoparietal_association_proxy",
                "relation": "DMN breakdown extends into temporo-parietal association cortex",
                "mnd_change": "increased",
            },
            {
                "source": "salience_executive_network_failure",
                "target": "acc_salience_proxy",
                "relation": "salience-executive failure burdens ACC-centered control processes",
                "mnd_change": "increased",
            },
            {
                "source": "salience_executive_network_failure",
                "target": "insula_salience_proxy",
                "relation": "salience-executive failure burdens insular salience processing",
                "mnd_change": "increased",
            },
            {
                "source": "salience_executive_network_failure",
                "target": "pfc_control_proxy",
                "relation": "executive-network failure compounds frontal control burden",
                "mnd_change": "increased",
            },
            {
                "source": "cortical_subcortical_disconnection",
                "target": "pfc_control_proxy",
                "relation": "disconnection disrupts frontal-subcortical executive circuits",
                "mnd_change": "increased",
            },
            {
                "source": "cortical_subcortical_disconnection",
                "target": "temporoparietal_association_proxy",
                "relation": "disconnection degrades association-cortex communication",
                "mnd_change": "increased",
            },
            {
                "source": "cortical_subcortical_disconnection",
                "target": "left_perisylvian_language_proxy",
                "relation": "disconnection can impair distributed language-supporting circuits",
                "mnd_change": "increased",
            },
            {
                "source": "synucleinopathy_sleep_prodrome",
                "target": "insula_salience_proxy",
                "relation": "sleep-linked synucleinopathy burden contributes to salience-system vulnerability",
                "mnd_change": "increased",
            },
            {
                "source": "synucleinopathy_sleep_prodrome",
                "target": "acc_salience_proxy",
                "relation": "prodromal synucleinopathy burden can tax attentional control systems",
                "mnd_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "episodic_memory_impairment",
                "relation": "hippocampal burden drives episodic-memory deficits",
                "mnd_change": "increased",
            },
            {
                "source": "entorhinal_cortex",
                "target": "episodic_memory_impairment",
                "relation": "entorhinal dysfunction compounds memory encoding / retrieval failure",
                "mnd_change": "increased",
            },
            {
                "source": "pcc_precuneus_dmn_proxy",
                "target": "episodic_memory_impairment",
                "relation": "DMN hub disruption contributes to autobiographical-memory problems",
                "mnd_change": "increased",
            },
            {
                "source": "pfc_control_proxy",
                "target": "executive_attention_impairment",
                "relation": "frontal control failure impairs executive and attentional function",
                "mnd_change": "increased",
            },
            {
                "source": "acc_salience_proxy",
                "target": "executive_attention_impairment",
                "relation": "ACC burden impairs control allocation and conflict monitoring",
                "mnd_change": "increased",
            },
            {
                "source": "insula_salience_proxy",
                "target": "executive_attention_impairment",
                "relation": "insular salience dysfunction destabilizes attentional switching",
                "mnd_change": "increased",
            },
            {
                "source": "pfc_control_proxy",
                "target": "behavioral_social_conduct_change",
                "relation": "frontal degeneration alters inhibition, judgment, and social conduct",
                "mnd_change": "increased",
            },
            {
                "source": "anterior_temporal_proxy",
                "target": "behavioral_social_conduct_change",
                "relation": "anterior temporal degeneration contributes to social / personality change",
                "mnd_change": "increased",
            },
            {
                "source": "left_temporal_pole",
                "target": "semantic_knowledge_loss",
                "relation": "left temporal-pole atrophy is linked to semantic loss",
                "mnd_change": "increased",
            },
            {
                "source": "anterior_temporal_proxy",
                "target": "semantic_knowledge_loss",
                "relation": "anterior temporal degeneration degrades semantic knowledge systems",
                "mnd_change": "increased",
            },
            {
                "source": "left_perisylvian_language_proxy",
                "target": "nonfluent_language_impairment",
                "relation": "left perisylvian burden drives effortful nonfluent language deficits",
                "mnd_change": "increased",
            },
            {
                "source": "episodic_memory_impairment",
                "target": "diffuse_cognitive_functional_decline",
                "relation": "prominent memory failure contributes to global decline",
                "mnd_change": "increased",
            },
            {
                "source": "executive_attention_impairment",
                "target": "diffuse_cognitive_functional_decline",
                "relation": "executive dysfunction contributes to broad functional impairment",
                "mnd_change": "increased",
            },
            {
                "source": "behavioral_social_conduct_change",
                "target": "diffuse_cognitive_functional_decline",
                "relation": "behavioral dyscontrol adds to syndrome-level disability",
                "mnd_change": "increased",
            },
            {
                "source": "semantic_knowledge_loss",
                "target": "diffuse_cognitive_functional_decline",
                "relation": "semantic degradation contributes to generalized cognitive disability",
                "mnd_change": "increased",
            },
            {
                "source": "nonfluent_language_impairment",
                "target": "diffuse_cognitive_functional_decline",
                "relation": "language impairment contributes to broad neurocognitive disability",
                "mnd_change": "increased",
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
    def _quiet_context():
        return getattr(siibra, "QUIET", nullcontext())

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _norm(text: Any) -> str:
        return (
            str(text)
            .lower()
            .replace("area ", "")
            .replace("(hippocampus)", "")
            .replace("(temporal pole)", "")
            .replace("(ifg)", "")
            .replace("(pcc)", "")
            .replace("(acc)", "")
            .replace("(pacc)", "")
            .replace("(sacc)", "")
            .replace("(insula)", "")
            .replace("(ipl)", "")
            .replace("(fpole)", "")
            .replace("entorhinal cortex", "entorhinal")
            .replace("posterior cingulate cortex", "posterior cingulate")
            .replace("anterior cingulate cortex", "anterior cingulate")
            .replace("prefrontal cortex", "prefrontal")
            .replace("left", "l")
            .replace("right", "r")
            .replace("  ", " ")
            .strip()
        )

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
        if concept is None:
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
            "hippocampus",
            "entorhinal cortex",
            "temporal pole",
            "temporal lobe",
            "prefrontal cortex",
            "inferior frontal gyrus",
            "posterior cingulate cortex",
            "anterior cingulate cortex",
            "insula",
            "precuneus",
            "inferior parietal lobule",
        } else 0
        component_penalty = 1 if "component" in name else 0
        return (left_bonus, right_penalty, generic_penalty, component_penalty)

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
            )
        return df.reset_index(drop=True)

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats and self.region_objects:
            exemplar = next(iter(self.region_objects.values()))
            feats = self._safe_features_any(exemplar, self._modality_candidates("connectivity"))

        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (
                f
                for f in feats
                if self.connectivity_cohort.lower() in str(getattr(f, "cohort", "")).lower()
                or self.connectivity_cohort.lower() in str(getattr(f, "name", "")).lower()
            ),
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
            if isinstance(self._connectivity_matrix, pd.DataFrame):
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

        target = self._norm(getattr(region, "name", region))
        fuzzy = [x for x in labels if target in self._norm(self._name_of(x)) or self._norm(self._name_of(x)) in target]
        if fuzzy:
            return fuzzy[0]

        name = getattr(region, "name", "").lower()
        tokens = [t for t in name.replace("(", " ").replace(")", " ").replace(",", " ").split() if len(t) > 2]
        for label in labels:
            lname = self._name_of(label).lower()
            if any(tok in lname for tok in tokens[:4]):
                return label
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
            if not isinstance(series, pd.Series):
                return pd.DataFrame()
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Build a small square connectivity table among resolved circuit nodes.

        Rows and columns are node keys, not region labels.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        row_labels = list(matrix.index)
        col_labels = list(matrix.columns)

        matched_rows = {
            key: self._match_region_label(row_labels, region)
            for key, region in self.region_objects.items()
        }
        matched_cols = {
            key: self._match_region_label(col_labels, region)
            for key, region in self.region_objects.items()
        }

        usable = [k for k in self.region_objects if matched_rows.get(k) is not None and matched_cols.get(k) is not None]
        if not usable:
            return pd.DataFrame()

        out = pd.DataFrame(index=usable, columns=usable, dtype=float)
        for src in usable:
            for dst in usable:
                try:
                    out.loc[src, dst] = float(matrix.loc[matched_rows[src], matched_cols[dst]])
                except Exception:
                    out.loc[src, dst] = float("nan")
        return out

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
        """
        Resolve atlas regions, collect multimodal features, and assemble node / edge tables.
        """
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
        self._connectivity_matrix = None

        nodes = []

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
            region_desc = self.region_descriptions.get(key, "Atlas-backed circuit node")
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
                        "description": f"{region_desc} (unresolved in this environment)",
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
                    "description": region_desc,
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
        genetic_neurodegenerative_liability: float = 0.0,
        alzheimer_like_pathology_load: float = 0.0,
        frontotemporal_pathology_load: float = 0.0,
        vascular_cerebrovascular_burden: float = 0.0,
        synucleinopathy_burden: float = 0.0,
        subcortical_genetic_neurodegeneration_load: float = 0.0,
        sleep_apnea_hypoxia: float = 0.0,
        rem_sleep_behavior_disorder_burden: float = 0.0,
        sleep_hygiene_support: float = 0.0,
        disease_management_support: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        Transparent one-pass simulator using normalized 0..1 inputs.

        Higher values in `regional_state` indicate burden / dysregulation on the
        corresponding region node, not healthy function.
        """
        inputs = {
            "genetic_neurodegenerative_liability": self._clip01(genetic_neurodegenerative_liability),
            "alzheimer_like_pathology_load": self._clip01(alzheimer_like_pathology_load),
            "frontotemporal_pathology_load": self._clip01(frontotemporal_pathology_load),
            "vascular_cerebrovascular_burden": self._clip01(vascular_cerebrovascular_burden),
            "synucleinopathy_burden": self._clip01(synucleinopathy_burden),
            "subcortical_genetic_neurodegeneration_load": self._clip01(subcortical_genetic_neurodegeneration_load),
            "sleep_apnea_hypoxia": self._clip01(sleep_apnea_hypoxia),
            "rem_sleep_behavior_disorder_burden": self._clip01(rem_sleep_behavior_disorder_burden),
            "sleep_hygiene_support": self._clip01(sleep_hygiene_support),
            "disease_management_support": self._clip01(disease_management_support),
        }

        latents = {}
        latents["medial_temporal_degeneration"] = self._clip01(
            0.40 * inputs["alzheimer_like_pathology_load"]
            + 0.20 * inputs["genetic_neurodegenerative_liability"]
            + 0.20 * inputs["sleep_apnea_hypoxia"]
            + 0.10 * inputs["synucleinopathy_burden"]
            + 0.05 * inputs["subcortical_genetic_neurodegeneration_load"]
            - 0.05 * inputs["sleep_hygiene_support"]
            - 0.10 * inputs["disease_management_support"]
        )

        latents["cortical_subcortical_disconnection"] = self._clip01(
            0.45 * inputs["vascular_cerebrovascular_burden"]
            + 0.20 * inputs["subcortical_genetic_neurodegeneration_load"]
            + 0.10 * inputs["sleep_apnea_hypoxia"]
            + 0.10 * inputs["synucleinopathy_burden"]
            + 0.05 * inputs["genetic_neurodegenerative_liability"]
            - 0.10 * inputs["disease_management_support"]
        )

        latents["temporoparietal_association_failure"] = self._clip01(
            0.35 * inputs["alzheimer_like_pathology_load"]
            + 0.25 * latents["medial_temporal_degeneration"]
            + 0.15 * latents["cortical_subcortical_disconnection"]
            + 0.10 * inputs["genetic_neurodegenerative_liability"]
            + 0.05 * inputs["synucleinopathy_burden"]
            - 0.10 * inputs["disease_management_support"]
        )

        latents["frontotemporal_network_degeneration"] = self._clip01(
            0.45 * inputs["frontotemporal_pathology_load"]
            + 0.20 * inputs["genetic_neurodegenerative_liability"]
            + 0.10 * inputs["vascular_cerebrovascular_burden"]
            + 0.10 * inputs["subcortical_genetic_neurodegeneration_load"]
            + 0.05 * inputs["synucleinopathy_burden"]
            - 0.10 * inputs["disease_management_support"]
        )

        latents["synucleinopathy_sleep_prodrome"] = self._clip01(
            0.40 * inputs["synucleinopathy_burden"]
            + 0.35 * inputs["rem_sleep_behavior_disorder_burden"]
            + 0.10 * inputs["genetic_neurodegenerative_liability"]
            + 0.05 * inputs["subcortical_genetic_neurodegeneration_load"]
            - 0.10 * inputs["sleep_hygiene_support"]
            - 0.05 * inputs["disease_management_support"]
        )

        latents["default_mode_network_disconnectivity"] = self._clip01(
            0.35 * latents["medial_temporal_degeneration"]
            + 0.30 * latents["temporoparietal_association_failure"]
            + 0.15 * latents["cortical_subcortical_disconnection"]
            + 0.10 * inputs["alzheimer_like_pathology_load"]
            - 0.10 * inputs["disease_management_support"]
        )

        latents["salience_executive_network_failure"] = self._clip01(
            0.30 * latents["frontotemporal_network_degeneration"]
            + 0.25 * latents["cortical_subcortical_disconnection"]
            + 0.15 * inputs["sleep_apnea_hypoxia"]
            + 0.15 * latents["synucleinopathy_sleep_prodrome"]
            + 0.05 * inputs["alzheimer_like_pathology_load"]
            - 0.10 * inputs["sleep_hygiene_support"]
            - 0.10 * inputs["disease_management_support"]
        )

        latents["diffuse_neurocognitive_decline"] = self._clip01(
            0.20 * latents["medial_temporal_degeneration"]
            + 0.15 * latents["temporoparietal_association_failure"]
            + 0.15 * latents["frontotemporal_network_degeneration"]
            + 0.15 * latents["cortical_subcortical_disconnection"]
            + 0.15 * latents["default_mode_network_disconnectivity"]
            + 0.10 * latents["salience_executive_network_failure"]
            + 0.10 * latents["synucleinopathy_sleep_prodrome"]
            - 0.10 * inputs["disease_management_support"]
        )

        regional_state = {}
        regional_state["hippocampus"] = self._clip01(
            0.55 * latents["medial_temporal_degeneration"]
            + 0.20 * inputs["sleep_apnea_hypoxia"]
            + 0.10 * latents["default_mode_network_disconnectivity"]
            + 0.10 * inputs["genetic_neurodegenerative_liability"]
            + 0.05 * latents["diffuse_neurocognitive_decline"]
        )

        regional_state["entorhinal_cortex"] = self._clip01(
            0.60 * latents["medial_temporal_degeneration"]
            + 0.20 * inputs["alzheimer_like_pathology_load"]
            + 0.10 * latents["default_mode_network_disconnectivity"]
            + 0.05 * inputs["genetic_neurodegenerative_liability"]
            + 0.05 * latents["diffuse_neurocognitive_decline"]
        )

        regional_state["temporoparietal_association_proxy"] = self._clip01(
            0.55 * latents["temporoparietal_association_failure"]
            + 0.20 * latents["default_mode_network_disconnectivity"]
            + 0.15 * latents["cortical_subcortical_disconnection"]
            + 0.10 * latents["diffuse_neurocognitive_decline"]
        )

        regional_state["pfc_control_proxy"] = self._clip01(
            0.40 * latents["frontotemporal_network_degeneration"]
            + 0.25 * latents["salience_executive_network_failure"]
            + 0.20 * latents["cortical_subcortical_disconnection"]
            + 0.10 * inputs["sleep_apnea_hypoxia"]
            + 0.05 * latents["diffuse_neurocognitive_decline"]
        )

        regional_state["anterior_temporal_proxy"] = self._clip01(
            0.55 * latents["frontotemporal_network_degeneration"]
            + 0.15 * inputs["genetic_neurodegenerative_liability"]
            + 0.10 * latents["diffuse_neurocognitive_decline"]
            + 0.10 * latents["salience_executive_network_failure"]
            + 0.10 * latents["synucleinopathy_sleep_prodrome"]
        )

        regional_state["left_temporal_pole"] = self._clip01(
            0.60 * latents["frontotemporal_network_degeneration"]
            + 0.15 * inputs["genetic_neurodegenerative_liability"]
            + 0.10 * latents["diffuse_neurocognitive_decline"]
            + 0.10 * latents["temporoparietal_association_failure"]
            + 0.05 * latents["default_mode_network_disconnectivity"]
        )

        regional_state["left_perisylvian_language_proxy"] = self._clip01(
            0.40 * latents["frontotemporal_network_degeneration"]
            + 0.25 * latents["cortical_subcortical_disconnection"]
            + 0.15 * latents["salience_executive_network_failure"]
            + 0.10 * latents["diffuse_neurocognitive_decline"]
            + 0.10 * inputs["genetic_neurodegenerative_liability"]
        )

        regional_state["pcc_precuneus_dmn_proxy"] = self._clip01(
            0.55 * latents["default_mode_network_disconnectivity"]
            + 0.20 * latents["temporoparietal_association_failure"]
            + 0.15 * latents["cortical_subcortical_disconnection"]
            + 0.10 * latents["diffuse_neurocognitive_decline"]
        )

        regional_state["acc_salience_proxy"] = self._clip01(
            0.50 * latents["salience_executive_network_failure"]
            + 0.20 * latents["frontotemporal_network_degeneration"]
            + 0.15 * latents["cortical_subcortical_disconnection"]
            + 0.10 * latents["synucleinopathy_sleep_prodrome"]
            + 0.05 * latents["diffuse_neurocognitive_decline"]
        )

        regional_state["insula_salience_proxy"] = self._clip01(
            0.45 * latents["salience_executive_network_failure"]
            + 0.20 * latents["synucleinopathy_sleep_prodrome"]
            + 0.15 * latents["cortical_subcortical_disconnection"]
            + 0.10 * inputs["sleep_apnea_hypoxia"]
            + 0.10 * latents["diffuse_neurocognitive_decline"]
        )

        symptoms = {}
        symptoms["episodic_memory_impairment"] = self._clip01(
            0.45 * regional_state["hippocampus"]
            + 0.35 * regional_state["entorhinal_cortex"]
            + 0.15 * regional_state["pcc_precuneus_dmn_proxy"]
            + 0.05 * latents["diffuse_neurocognitive_decline"]
        )

        symptoms["executive_attention_impairment"] = self._clip01(
            0.30 * regional_state["pfc_control_proxy"]
            + 0.25 * regional_state["acc_salience_proxy"]
            + 0.20 * regional_state["insula_salience_proxy"]
            + 0.15 * latents["cortical_subcortical_disconnection"]
            + 0.10 * inputs["sleep_apnea_hypoxia"]
        )

        symptoms["behavioral_social_conduct_change"] = self._clip01(
            0.40 * regional_state["pfc_control_proxy"]
            + 0.30 * regional_state["anterior_temporal_proxy"]
            + 0.20 * regional_state["acc_salience_proxy"]
            + 0.10 * latents["diffuse_neurocognitive_decline"]
        )

        symptoms["semantic_knowledge_loss"] = self._clip01(
            0.45 * regional_state["left_temporal_pole"]
            + 0.25 * regional_state["anterior_temporal_proxy"]
            + 0.15 * regional_state["temporoparietal_association_proxy"]
            + 0.15 * latents["frontotemporal_network_degeneration"]
        )

        symptoms["nonfluent_language_impairment"] = self._clip01(
            0.45 * regional_state["left_perisylvian_language_proxy"]
            + 0.20 * regional_state["pfc_control_proxy"]
            + 0.15 * latents["cortical_subcortical_disconnection"]
            + 0.10 * regional_state["acc_salience_proxy"]
            + 0.10 * latents["diffuse_neurocognitive_decline"]
        )

        symptoms["diffuse_cognitive_functional_decline"] = self._clip01(
            0.25 * latents["diffuse_neurocognitive_decline"]
            + 0.20 * symptoms["episodic_memory_impairment"]
            + 0.20 * symptoms["executive_attention_impairment"]
            + 0.10 * symptoms["behavioral_social_conduct_change"]
            + 0.10 * symptoms["semantic_knowledge_loss"]
            + 0.10 * symptoms["nonfluent_language_impairment"]
            + 0.05 * latents["default_mode_network_disconnectivity"]
        )

        phenotypes = {}
        phenotypes["alzheimer_like_amnestic_profile"] = self._clip01(
            (
                0.40 * symptoms["episodic_memory_impairment"]
                + 0.25 * regional_state["hippocampus"]
                + 0.15 * regional_state["entorhinal_cortex"]
                + 0.10 * regional_state["temporoparietal_association_proxy"]
                + 0.10 * latents["default_mode_network_disconnectivity"]
            )
        )

        phenotypes["bvftd_like_behavioral_profile"] = self._clip01(
            (
                0.40 * symptoms["behavioral_social_conduct_change"]
                + 0.20 * regional_state["pfc_control_proxy"]
                + 0.20 * regional_state["anterior_temporal_proxy"]
                + 0.10 * symptoms["executive_attention_impairment"]
                + 0.10 * latents["frontotemporal_network_degeneration"]
            )
        )

        phenotypes["semantic_dementia_profile"] = self._clip01(
            (
                0.45 * symptoms["semantic_knowledge_loss"]
                + 0.30 * regional_state["left_temporal_pole"]
                + 0.15 * regional_state["anterior_temporal_proxy"]
                + 0.10 * latents["frontotemporal_network_degeneration"]
            )
        )

        phenotypes["progressive_nonfluent_aphasia_profile"] = self._clip01(
            (
                0.45 * symptoms["nonfluent_language_impairment"]
                + 0.25 * regional_state["left_perisylvian_language_proxy"]
                + 0.15 * regional_state["pfc_control_proxy"]
                + 0.15 * latents["frontotemporal_network_degeneration"]
            )
        )

        phenotypes["vascular_disconnection_profile"] = self._clip01(
            (
                0.35 * symptoms["executive_attention_impairment"]
                + 0.30 * latents["cortical_subcortical_disconnection"]
                + 0.20 * symptoms["diffuse_cognitive_functional_decline"]
                + 0.15 * latents["temporoparietal_association_failure"]
            )
        )

        phenotypes["synucleinopathy_sleep_prodrome_profile"] = self._clip01(
            (
                0.45 * latents["synucleinopathy_sleep_prodrome"]
                + 0.25 * inputs["rem_sleep_behavior_disorder_burden"]
                + 0.15 * symptoms["executive_attention_impairment"]
                + 0.15 * regional_state["insula_salience_proxy"]
            )
        )

        phenotypes["global_major_neurocognitive_profile"] = self._clip01(
            (
                0.30 * symptoms["diffuse_cognitive_functional_decline"]
                + 0.20 * symptoms["episodic_memory_impairment"]
                + 0.20 * symptoms["executive_attention_impairment"]
                + 0.10 * symptoms["behavioral_social_conduct_change"]
                + 0.10 * symptoms["semantic_knowledge_loss"]
                + 0.10 * symptoms["nonfluent_language_impairment"]
            )
        )

        return {
            "inputs": pd.Series(inputs, name="input"),
            "latents": pd.Series(latents, name="latent_biology"),
            "regional_state": pd.Series(regional_state, name="regional_burden"),
            "symptoms": pd.Series(symptoms, name="symptom"),
            "phenotypes": pd.Series(phenotypes, name="phenotype"),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Probabilistically assign an MNI152 coordinate to Julich regions.
        """
        if self._pmap is None:
            with self._quiet_context():
                self._pmap = siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=self.assignment_space,
                    maptype="statistical",
                )

        point = siibra.Point(tuple(float(x) for x in xyz), space=self.assignment_space)
        with self._quiet_context():
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        """
        Return a regional mask or map-like object for a resolved region node.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None

        attempts = [
            {"space": self.assignment_space, "maptype": "labelled"},
            {"space": self.assignment_space, "maptype": "statistical"},
            {"space": self.assignment_space},
        ]
        for kwargs in attempts:
            try:
                return region.get_regional_mask(**kwargs)
            except Exception:
                continue
        for args in [
            (self.assignment_space, "statistical"),
            (self.assignment_space,),
        ]:
            try:
                return region.get_regional_map(*args)
            except Exception:
                continue
        return None


if __name__ == "__main__":
    model = MajorNeurocognitiveDisorderModel()

    print("Building Major Neurocognitive Disorder scaffold...")
    scaffold = model.build(connectivity_rows=10)

    print("\nNodes:")
    node_cols = ["key", "node_type", "atlas_region", "feature_summary"]
    print(scaffold["nodes"][node_cols].to_string(index=False))

    print("\nEdges:")
    edge_cols = ["source", "target", "relation", "mnd_change"]
    print(scaffold["edges"][edge_cols].to_string(index=False))

    circuit_df = scaffold["circuit_connectivity"]
    print("\nCircuit connectivity among resolved region nodes:")
    if circuit_df.empty:
        print("No circuit connectivity matrix available in this environment.")
    else:
        print(circuit_df.round(4).to_string())

    for key in [
        "hippocampus",
        "entorhinal_cortex",
        "pcc_precuneus_dmn_proxy",
        "pfc_control_proxy",
    ]:
        receptor_df = scaffold["receptors"].get(key, pd.DataFrame())
        if not receptor_df.empty:
            print(f"\nExample receptor fingerprint for {key}:")
            print(receptor_df.head(10).to_string(index=False))
            break

    for key in [
        "hippocampus",
        "entorhinal_cortex",
        "left_temporal_pole",
        "pfc_control_proxy",
    ]:
        gene_df = scaffold["genes"].get(key, pd.DataFrame())
        if not gene_df.empty:
            print(f"\nExample gene-expression summary for {key}:")
            print(gene_df.head(10).to_string(index=False))
            break

    for key in [
        "hippocampus",
        "pcc_precuneus_dmn_proxy",
        "pfc_control_proxy",
        "temporoparietal_association_proxy",
    ]:
        conn_df = scaffold["connectivity_profiles"].get(key, pd.DataFrame())
        if not conn_df.empty:
            print(f"\nExample connectivity profile for {key}:")
            print(conn_df.head(10).to_string(index=False))
            break

    print("\nExample simulation (mixed Alzheimer-like + sleep-apnea + mild vascular burden):")
    sim = model.simulate(
        genetic_neurodegenerative_liability=0.55,
        alzheimer_like_pathology_load=0.75,
        frontotemporal_pathology_load=0.20,
        vascular_cerebrovascular_burden=0.35,
        synucleinopathy_burden=0.15,
        subcortical_genetic_neurodegeneration_load=0.10,
        sleep_apnea_hypoxia=0.45,
        rem_sleep_behavior_disorder_burden=0.10,
        sleep_hygiene_support=0.40,
        disease_management_support=0.35,
    )

    print("\nInputs:")
    print(sim["inputs"].to_string())
    print("\nLatent biology:")
    print(sim["latents"].sort_values(ascending=False).to_string())
    print("\nRegional burden:")
    print(sim["regional_state"].sort_values(ascending=False).to_string())
    print("\nSymptoms:")
    print(sim["symptoms"].sort_values(ascending=False).to_string())
    print("\nPhenotypes:")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Example coordinate checks for later interactive use:
    # print(model.assign_mni_point((-24, -16, -18)).head())   # medial temporal example
    # print(model.suggest_regions("entorhinal"))
    # mask = model.region_mask("hippocampus")
