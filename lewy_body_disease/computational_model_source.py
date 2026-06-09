from __future__ import annotations

"""
Lewy Body Disease atlas-grounded siibra scaffold.

This script translates a chapter-level biological summary of Lewy Body Disease
into a transparent mechanistic graph plus a simple normalized simulator.

Design choices
--------------
- The chapter is clinicopathological and systems-level.
- Some named nuclei, especially locus coeruleus and the dorsal vagal nucleus,
  do not reliably resolve as Julich regions in every siibra environment.
  They are therefore kept as *explicit proxy nodes* when no cytoarchitectonic
  anchor is available.
- Cortical disease burden is modeled with conservative association-cortex
  anchors rather than overly broad whole-lobe labels.

This is a research scaffold, not a diagnostic or treatment tool.
"""

import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore[assignment]
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_LBD_GENE_PANEL = [
    "SNCA",   # alpha-synuclein
    "GBA",    # lysosomal risk / glucocerebrosidase
    "LRRK2",  # familial PD/LBD spectrum risk
    "APOE",   # AD copathology burden / cognitive decline
    "MAPT",   # tau-related overlap
    "APP",    # amyloid precursor protein
    "PSEN1",  # amyloid processing
    "PSEN2",  # amyloid processing
    "SLC6A3", # dopamine transporter
    "DRD2",   # D2 signaling / neuroleptic sensitivity relevance
    "CHAT",   # acetylcholine synthesis
    "ACHE",   # acetylcholine breakdown
    "DBH",    # norepinephrine synthesis pathway relevance
]


class LewyBodyDiseaseModel:
    """
    Atlas-grounded mechanistic scaffold for Lewy Body Disease.

    Inputs
        - genetic vulnerability
        - alpha-synucleinopathy load
        - lysosomal clearance impairment
        - Alzheimer-type copathology load
        - dopaminergic replacement pressure
        - D2 antagonist exposure
        - cholinergic reserve support
        - low-D2-antagonism strategy

    Latent biology
        - lysosomal proteostasis failure
        - alpha-synuclein spread
        - mixed proteinopathy burden
        - cholinergic deficit
        - dopaminergic deficit
        - noradrenergic arousal instability
        - limbic/cortical synaptic dysfunction
        - autonomic network dysregulation
        - occipital alpha slowing
        - mesocorticolimbic psychosis liability
        - neuroleptic sensitivity

    Regions / proxies
        - substantia nigra proxy
        - striatum proxy
        - locus coeruleus proxy
        - dorsal vagal nucleus proxy
        - nucleus basalis / basal forebrain proxy
        - amygdala
        - parahippocampal-entorhinal proxy
        - cingulate proxy
        - insula
        - frontal cortex proxy
        - temporal cortex proxy
        - occipital visual cortex
        - sensorimotor cortex proxy

    Symptoms
        - parkinsonism
        - cognitive fluctuations
        - visual hallucinations
        - delusions / psychosis
        - executive dysfunction
        - visuospatial impairment
        - autonomic dysfunction
        - sleep-wake disturbance
        - neuroleptic sensitivity crisis
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
        aggregate_connectivity_subjects: int = 8,
    ) -> None:
        if siibra is None:
            raise ImportError(
                "siibra is required to use this scaffold. Install it in your Python "
                "environment before running the model."
            ) from _SIIBRA_IMPORT_ERROR

        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort
        self.aggregate_connectivity_subjects = max(1, int(aggregate_connectivity_subjects))

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

        self.disorder_name = "Lewy Body Disease"
        self.disorder_key = "lewy_body_disease"

        # Nuclei with weak or absent Julich coverage in many environments.
        # They remain important mechanistic nodes, but unresolved status should
        # not be treated as an error.
        self.proxy_only_nodes = {
            "locus_coeruleus_proxy",
            "dorsal_vagal_nucleus_proxy",
        }

        self.region_candidates: Dict[str, List[str]] = {
            "substantia_nigra_proxy": [
                "SNC (Midbrain, Substantia Nigra pars compacta) left",
                "substantia nigra pars compacta left",
                "substantia nigra left",
                "substantia nigra",
                "midbrain",
            ],
            "striatum_proxy": [
                "FuCd (Ventral Striatum, Fundus of Caudate Nucleus) left",
                "caudate nucleus left",
                "putamen left",
                "striatum left",
                "caudate",
                "putamen",
                "striatum",
            ],
            # Intentionally conservative proxy-only nodes. Leave empty so they do
            # not get spuriously forced onto generic pons/medulla labels.
            "locus_coeruleus_proxy": [],
            "dorsal_vagal_nucleus_proxy": [],
            "nucleus_basalis_meynert_proxy": [
                "Ch 4 (Basal Forebrain) left",
                "Ch 123 (Basal Forebrain) left",
                "basal forebrain",
                "nucleus basalis",
                "substantia innominata",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "SF (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala",
            ],
            "parahippocampal_proxy": [
                "Area EC (Hippocampal Region, Entorhinal Cortex) left",
                "entorhinal cortex left",
                "Area EC left",
                "parahippocampal gyrus left",
                "parahippocampal gyrus",
                "hippocampus",
            ],
            "cingulate_proxy": [
                "Area p24ab (pACC) left",
                "Area a24pr (ACC) left",
                "Area p32 (pACC) left",
                "cingulate gyrus left",
                "anterior cingulate cortex",
                "cingulate gyrus",
            ],
            "insula": [
                "Area Id1 (Insula) left",
                "Area Id2 (Insula) left",
                "Area Ig1 (Insula) left",
                "insula left",
                "insular cortex left",
                "insula",
            ],
            "frontal_cortex_proxy": [
                "Area 45 (IFG) left",
                "Area 46 left",
                "Area 9/46d left",
                "MFG1 left",
                "MFG2 left",
                "middle frontal gyrus left",
                "prefrontal cortex left",
                "frontal cortex",
            ],
            "temporal_cortex_proxy": [
                "Area Te 3 (STG) left",
                "Area Te 2.2 (STG) left",
                "Area TE 1.1 left",
                "superior temporal gyrus left",
                "middle temporal gyrus left",
                "temporal cortex left",
                "temporal cortex",
            ],
            "occipital_visual_cortex": [
                "Area hOc1 (V1, 17, CalcS) left",
                "Area hOc2 (V2, 18) left",
                "Area hOc3d (V3d) left",
                "visual cortex left",
                "occipital cortex left",
                "occipital cortex",
            ],
            "sensorimotor_cortex_proxy": [
                "Area 4a (PreCG) left",
                "Area 4p (PreCG) left",
                "Area 3b (PostCG) left",
                "precentral gyrus left",
                "sensorimotor cortex left",
                "sensorimotor cortex",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "substantia_nigra_proxy": "Proxy for nigrostriatal degeneration centered on substantia-nigra-type circuitry.",
            "striatum_proxy": "Proxy for striatal dopaminergic target systems contributing to motor and psychosis burden.",
            "locus_coeruleus_proxy": "Proxy for brainstem noradrenergic arousal regulation centered on locus-coeruleus-type systems.",
            "dorsal_vagal_nucleus_proxy": "Proxy for dorsal-vagal / medullary autonomic network burden.",
            "nucleus_basalis_meynert_proxy": "Proxy for basal forebrain cholinergic degeneration.",
            "amygdala": "Limbic node relevant to affective salience and psychosis burden.",
            "parahippocampal_proxy": "Proxy for medial temporal memory-related circuitry, especially entorhinal / parahippocampal burden.",
            "cingulate_proxy": "Proxy for cingulate attentional and executive integration burden.",
            "insula": "Interoceptive and autonomic integration node.",
            "frontal_cortex_proxy": "Proxy for frontal-executive cortical burden.",
            "temporal_cortex_proxy": "Proxy for temporal association-cortex burden.",
            "occipital_visual_cortex": "Posterior visual cortical anchor for occipital slowing and hallucination-related burden.",
            "sensorimotor_cortex_proxy": "Proxy for wider cortical motor-system burden beyond the nigrostriatal core.",
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": "Familial or polygenic vulnerability affecting synuclein handling, lysosomal biology, and neural reserve.",
            "alpha_synucleinopathy_load": "Burden of Lewy-body / Lewy-neurite pathology.",
            "lysosomal_clearance_impairment": "Reduced proteostatic and lysosomal clearance capacity that can worsen synuclein accumulation.",
            "alzheimer_copathology_load": "Concurrent amyloid/tau-type burden that accelerates cognitive decline.",
            "dopaminergic_replacement_pressure": "Medication-related dopaminergic pressure that may help motor signs but worsen psychosis burden.",
            "d2_antagonist_exposure": "Exposure to D2-blocking antipsychotic pressure, relevant to marked neuroleptic sensitivity.",
            "cholinergic_reserve_support": "Protective cholinergic reserve or support factor that buffers cognitive and perceptual symptoms.",
            "low_d2_antagonism_strategy": "Protective strategy minimizing D2 blockade intensity in a neuroleptic-sensitive disorder.",
        }

        self.latent_nodes: Dict[str, str] = {
            "lysosomal_proteostasis_failure": "Failure of lysosomal and proteostatic systems that worsens synuclein handling.",
            "alpha_synuclein_spread": "Topographic spread of Lewy pathology from brainstem/subcortex into limbic and cortical territories.",
            "mixed_proteinopathy_burden": "Combined synuclein plus Alzheimer-type proteinopathy burden.",
            "cholinergic_deficit": "Basal forebrain and cortical cholinergic signaling failure.",
            "dopaminergic_deficit": "Nigrostriatal and mesocorticolimbic dopamine depletion.",
            "noradrenergic_arousal_instability": "Brainstem arousal instability linked to noradrenergic degeneration.",
            "limbic_cortical_synaptic_dysfunction": "Distributed limbic and cortical synaptic dysfunction affecting cognition and neuropsychiatric symptoms.",
            "autonomic_network_dysregulation": "Brainstem-insular-cortical autonomic dysregulation.",
            "occipital_alpha_slowing": "Posterior slowing / visual-network dysfunction often described in DLB/LBD.",
            "mesocorticolimbic_psychosis_liability": "Psychosis vulnerability arising from dopaminergic, cholinergic, and cortical network dysfunction.",
            "neuroleptic_sensitivity": "Marked vulnerability to deterioration under D2 blockade.",
        }

        self.symptom_nodes: Dict[str, str] = {
            "parkinsonism": "Parkinsonian motor signs from nigrostriatal degeneration.",
            "cognitive_fluctuations": "Fluctuating attention and cognition across wakefulness states.",
            "visual_hallucinations": "Recurrent visual hallucinations linked to cholinergic and visual-network dysfunction.",
            "delusions_psychosis": "Delusional or broader psychotic symptom burden.",
            "executive_dysfunction": "Impaired frontal-executive control and goal management.",
            "visuospatial_impairment": "Visuospatial and visual-attention deficits linked to occipital dysfunction.",
            "autonomic_dysfunction": "Autonomic instability affecting visceral and regulatory functions.",
            "sleep_wake_disturbance": "Sleep/arousal instability emerging from brainstem and network dysfunction.",
            "neuroleptic_sensitivity_crisis": (
                "Severe adverse reaction with worsening parkinsonism, confusion, and autonomic burden under D2 blockade."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "lysosomal_proteostasis_failure",
                "relation": "can increase vulnerability of synuclein-handling and proteostasis systems",
                "lbd_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "alpha_synuclein_spread",
                "relation": "can bias baseline susceptibility to Lewy pathology progression",
                "lbd_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "cholinergic_deficit",
                "relation": "can reduce reserve in cholinergic systems",
                "lbd_change": "increased",
            },
            {
                "source": "lysosomal_clearance_impairment",
                "target": "lysosomal_proteostasis_failure",
                "relation": "directly worsens synuclein clearance capacity",
                "lbd_change": "increased",
            },
            {
                "source": "alpha_synucleinopathy_load",
                "target": "alpha_synuclein_spread",
                "relation": "drives widespread Lewy pathology and synaptic dysfunction",
                "lbd_change": "increased",
            },
            {
                "source": "alpha_synuclein_spread",
                "target": "dopaminergic_deficit",
                "relation": "degenerates nigrostriatal and mesocorticolimbic dopamine pathways",
                "lbd_change": "increased",
            },
            {
                "source": "alpha_synuclein_spread",
                "target": "cholinergic_deficit",
                "relation": "damages cholinergic nuclei and cortical cholinergic signaling",
                "lbd_change": "increased",
            },
            {
                "source": "alpha_synuclein_spread",
                "target": "noradrenergic_arousal_instability",
                "relation": "impairs brainstem arousal systems including locus-coeruleus-type regulation",
                "lbd_change": "increased",
            },
            {
                "source": "alpha_synuclein_spread",
                "target": "limbic_cortical_synaptic_dysfunction",
                "relation": "spreads into limbic and cortical territories including temporal, insular, cingulate, and frontal regions",
                "lbd_change": "increased",
            },
            {
                "source": "alpha_synuclein_spread",
                "target": "autonomic_network_dysregulation",
                "relation": "involves dorsal vagal and related autonomic nuclei",
                "lbd_change": "increased",
            },
            {
                "source": "alzheimer_copathology_load",
                "target": "mixed_proteinopathy_burden",
                "relation": "adds amyloid/tau pressure to synucleinopathy",
                "lbd_change": "increased",
            },
            {
                "source": "mixed_proteinopathy_burden",
                "target": "limbic_cortical_synaptic_dysfunction",
                "relation": "worsens cortical cognitive decline and disease severity",
                "lbd_change": "increased",
            },
            {
                "source": "mixed_proteinopathy_burden",
                "target": "executive_dysfunction",
                "relation": "accelerates cortical cognitive burden",
                "lbd_change": "increased",
            },
            {
                "source": "mixed_proteinopathy_burden",
                "target": "cognitive_fluctuations",
                "relation": "contributes to more severe and faster cognitive deterioration",
                "lbd_change": "increased",
            },
            {
                "source": "dopaminergic_deficit",
                "target": "substantia_nigra_proxy",
                "relation": "is anchored by degeneration of substantia-nigra-type circuitry",
                "lbd_change": "increased",
            },
            {
                "source": "dopaminergic_deficit",
                "target": "striatum_proxy",
                "relation": "reduces effective striatal dopaminergic signaling",
                "lbd_change": "increased",
            },
            {
                "source": "dopaminergic_deficit",
                "target": "parkinsonism",
                "relation": "produces parkinsonian motor signs through nigrostriatal loss",
                "lbd_change": "increased",
            },
            {
                "source": "dopaminergic_deficit",
                "target": "mesocorticolimbic_psychosis_liability",
                "relation": "mesocorticolimbic dopamine disruption contributes to psychiatric burden",
                "lbd_change": "increased",
            },
            {
                "source": "cholinergic_deficit",
                "target": "nucleus_basalis_meynert_proxy",
                "relation": "is anchored by basal forebrain cholinergic degeneration",
                "lbd_change": "increased",
            },
            {
                "source": "cholinergic_deficit",
                "target": "visual_hallucinations",
                "relation": "greater cholinergic depletion is linked to hallucination risk",
                "lbd_change": "increased",
            },
            {
                "source": "cholinergic_deficit",
                "target": "cognitive_fluctuations",
                "relation": "cholinergic failure destabilizes attention and cognition",
                "lbd_change": "increased",
            },
            {
                "source": "limbic_cortical_synaptic_dysfunction",
                "target": "amygdala",
                "relation": "burdens limbic structures implicated in psychiatric symptoms",
                "lbd_change": "increased",
            },
            {
                "source": "limbic_cortical_synaptic_dysfunction",
                "target": "parahippocampal_proxy",
                "relation": "burdens medial temporal regions implicated in memory-related dysfunction",
                "lbd_change": "increased",
            },
            {
                "source": "limbic_cortical_synaptic_dysfunction",
                "target": "cingulate_proxy",
                "relation": "burdens cingulate systems relevant to attention and executive function",
                "lbd_change": "increased",
            },
            {
                "source": "limbic_cortical_synaptic_dysfunction",
                "target": "insula",
                "relation": "burdens insular circuitry contributing to interoceptive and autonomic integration",
                "lbd_change": "increased",
            },
            {
                "source": "limbic_cortical_synaptic_dysfunction",
                "target": "frontal_cortex_proxy",
                "relation": "burdens frontal systems contributing to executive dysfunction and psychosis",
                "lbd_change": "increased",
            },
            {
                "source": "limbic_cortical_synaptic_dysfunction",
                "target": "temporal_cortex_proxy",
                "relation": "burdens temporal cortical systems implicated in cognition and behavior",
                "lbd_change": "increased",
            },
            {
                "source": "occipital_alpha_slowing",
                "target": "occipital_visual_cortex",
                "relation": "maps posterior slowing onto visual cortical network burden",
                "lbd_change": "increased",
            },
            {
                "source": "occipital_alpha_slowing",
                "target": "visuospatial_impairment",
                "relation": "posterior slowing contributes to attention and visuospatial dysfunction",
                "lbd_change": "increased",
            },
            {
                "source": "noradrenergic_arousal_instability",
                "target": "locus_coeruleus_proxy",
                "relation": "reflects brainstem arousal system degeneration",
                "lbd_change": "increased",
            },
            {
                "source": "noradrenergic_arousal_instability",
                "target": "sleep_wake_disturbance",
                "relation": "contributes to sleep and arousal instability",
                "lbd_change": "increased",
            },
            {
                "source": "autonomic_network_dysregulation",
                "target": "dorsal_vagal_nucleus_proxy",
                "relation": "reflects dorsal vagal/autonomic network burden",
                "lbd_change": "increased",
            },
            {
                "source": "autonomic_network_dysregulation",
                "target": "autonomic_dysfunction",
                "relation": "produces autonomic instability and visceral dysregulation",
                "lbd_change": "increased",
            },
            {
                "source": "mesocorticolimbic_psychosis_liability",
                "target": "delusions_psychosis",
                "relation": "drives hallucination-delusion spectrum symptoms",
                "lbd_change": "increased",
            },
            {
                "source": "d2_antagonist_exposure",
                "target": "neuroleptic_sensitivity",
                "relation": "raises risk of severe neuroleptic sensitivity in a dopamine-depleted disorder",
                "lbd_change": "increased",
            },
            {
                "source": "dopaminergic_replacement_pressure",
                "target": "parkinsonism",
                "relation": "can partially reduce motor burden by supporting dopamine tone",
                "lbd_change": "decreased",
            },
            {
                "source": "dopaminergic_replacement_pressure",
                "target": "visual_hallucinations",
                "relation": "can increase perceptual and psychotic burden in vulnerable patients",
                "lbd_change": "increased",
            },
            {
                "source": "cholinergic_reserve_support",
                "target": "cholinergic_deficit",
                "relation": "buffers cognitive-perceptual burden by preserving cholinergic reserve",
                "lbd_change": "decreased",
            },
            {
                "source": "low_d2_antagonism_strategy",
                "target": "neuroleptic_sensitivity",
                "relation": "reduces sensitivity-crisis risk by limiting D2 blockade burden",
                "lbd_change": "decreased",
            },
            {
                "source": "neuroleptic_sensitivity",
                "target": "neuroleptic_sensitivity_crisis",
                "relation": "can precipitate severe deterioration under D2 blockade",
                "lbd_change": "increased",
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
        self._build_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
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
    def _series_from(values: Dict[str, float], name: str) -> pd.Series:
        return pd.Series(values, name=name)

    def _mean_clip(self, values: Sequence[float]) -> float:
        usable = [float(v) for v in values if v is not None]
        if not usable:
            return 0.0
        return self._clip01(sum(usable) / float(len(usable)))

    @staticmethod
    def _norm(text: Any) -> str:
        return (
            str(text)
            .lower()
            .replace("area ", "")
            .replace("(insula)", "")
            .replace("(amygdala)", "")
            .replace("(hippocampal region, entorhinal cortex)", "entorhinal cortex")
            .replace("(pacc)", "")
            .replace("(acc)", "")
            .replace("(stg)", "")
            .replace("(v1, 17, calcs)", "v1")
            .replace("(midbrain, substantia nigra pars compacta)", "substantia nigra pars compacta")
            .replace("left", "l")
            .replace("right", "r")
            .replace("-", " ")
            .replace("_", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace(",", " ")
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

    # ------------------------------------------------------------------
    # Region resolution
    # ------------------------------------------------------------------
    def _julich_matches(self, query: str) -> List[Any]:
        matches: List[Any] = []

        try:
            if hasattr(self.parcellation, "find"):
                try:
                    matches = list(self.parcellation.find(query, filter_children=False))
                except TypeError:
                    matches = list(self.parcellation.find(query))
        except Exception:
            matches = []

        if matches:
            return matches

        try:
            atlas_matches = self.atlas.find_regions(
                query,
                all_versions=False,
                filter_children=False,
                find_topmost=False,
            )
        except Exception:
            return []

        out: List[Any] = []
        for region in atlas_matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower():
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "substantia nigra",
            "striatum",
            "caudate nucleus",
            "putamen",
            "basal forebrain",
            "amygdala",
            "hippocampus",
            "insula",
            "cingulate gyrus",
            "frontal cortex",
            "temporal cortex",
            "occipital cortex",
            "visual cortex",
            "sensorimotor cortex",
            "brainstem",
            "pons",
            "medulla",
        } else 0
        specific_bonus = 0 if any(
            token in name
            for token in (
                "area ",
                "snc",
                "lb",
                "sf",
                "cm",
                "ch ",
                "ec",
                "p24",
                "p32",
                "id1",
                "id2",
                "ig1",
                "te ",
                "hoc",
                "4a",
                "4p",
                "3b",
                "mfg",
                "ifg",
            )
        ) else 1
        return (left_bonus, right_penalty, generic_penalty, specific_bonus)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                region = self.atlas.get_region(spec, parcellation=self.parcellation)
                if region is not None:
                    return region
            except Exception:
                pass
            try:
                region = self.parcellation.get_region(spec)
                if region is not None:
                    return region
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

    # ------------------------------------------------------------------
    # Spatial and feature helpers
    # ------------------------------------------------------------------
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
            return df.reset_index(drop=True)
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

    # ------------------------------------------------------------------
    # Connectivity helpers
    # ------------------------------------------------------------------
    def _candidate_connectivity_feature(self) -> Optional[Any]:
        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats and self.region_objects:
            exemplar = next(iter(self.region_objects.values()))
            feats = self._safe_features_any(exemplar, self._modality_candidates("connectivity"))
        if not feats:
            return None
        for feat in feats:
            text = f"{getattr(feat, 'cohort', '')} {getattr(feat, 'name', '')}".lower()
            if self.connectivity_cohort.lower() in text:
                return feat
        return feats[0]

    def _matrix_like(self, obj: Any) -> pd.DataFrame:
        try:
            data = getattr(obj, "data", None)
            if isinstance(data, pd.DataFrame):
                return data.copy()
            if isinstance(data, pd.Series):
                return data.to_frame().copy()
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
        try:
            for idx, element in enumerate(feat):
                mat = self._matrix_like(element)
                if not mat.empty:
                    matrices.append(mat)
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
        if region is None:
            return None

        exact = [x for x in labels if self._name_of(x) == getattr(region, "name", self._name_of(region))]
        if exact:
            return exact[0]

        rn = getattr(region, "name", self._name_of(region)).lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        if fuzzy:
            return fuzzy[0]

        target = set(t for t in self._norm(rn).split() if len(t) > 2)
        best = None
        best_score = 0
        for label in labels:
            label_tokens = set(t for t in self._norm(self._name_of(label)).split() if len(t) > 2)
            score = len(target & label_tokens)
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

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Build a square connectivity table among resolved region nodes.
        Rows and columns are scaffold node keys.
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

        usable = [
            key
            for key in self.region_objects
            if matched_rows.get(key) is not None and matched_cols.get(key) is not None
        ]
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

    # ------------------------------------------------------------------
    # Build scaffold tables
    # ------------------------------------------------------------------
    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_LBD_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
        self._connectivity_matrix = None

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
                    "is_proxy": False,
                }
            )

        for key, candidates in self.region_candidates.items():
            region_desc = self.region_descriptions.get(key, "Atlas-backed circuit node")
            is_proxy = key.endswith("_proxy")
            region = self._resolve_region(candidates)

            if region is None:
                if key not in self.proxy_only_nodes:
                    warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region",
                        "description": (
                            f"{region_desc} Explicit proxy without a Julich anchor in this environment."
                            if key in self.proxy_only_nodes
                            else f"{region_desc} (unresolved in this environment)"
                        ),
                        "atlas_region": None,
                        "region_identifier": None,
                        "centroid_mni": None,
                        "volume_mm3": None,
                        "feature_summary": "proxy_only_unresolved" if key in self.proxy_only_nodes else "unresolved",
                        "is_proxy": is_proxy,
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
                    "is_proxy": is_proxy,
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
                    "is_proxy": False,
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
                    "is_proxy": False,
                }
            )

        self.nodes_df = pd.DataFrame(nodes)
        self.edges_df = pd.DataFrame(self.edge_table)
        circuit_df = self.circuit_connectivity()

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
        genetic_vulnerability: float = 0.40,
        alpha_synucleinopathy_load: float = 0.55,
        lysosomal_clearance_impairment: float = 0.35,
        alzheimer_copathology_load: float = 0.25,
        dopaminergic_replacement_pressure: float = 0.15,
        d2_antagonist_exposure: float = 0.05,
        cholinergic_reserve_support: float = 0.10,
        low_d2_antagonism_strategy: float = 0.10,
    ) -> Dict[str, pd.Series]:
        inp = {
            "genetic_vulnerability": self._clip01(genetic_vulnerability),
            "alpha_synucleinopathy_load": self._clip01(alpha_synucleinopathy_load),
            "lysosomal_clearance_impairment": self._clip01(lysosomal_clearance_impairment),
            "alzheimer_copathology_load": self._clip01(alzheimer_copathology_load),
            "dopaminergic_replacement_pressure": self._clip01(dopaminergic_replacement_pressure),
            "d2_antagonist_exposure": self._clip01(d2_antagonist_exposure),
            "cholinergic_reserve_support": self._clip01(cholinergic_reserve_support),
            "low_d2_antagonism_strategy": self._clip01(low_d2_antagonism_strategy),
        }

        lat: Dict[str, float] = {}
        lat["lysosomal_proteostasis_failure"] = self._clip01(
            0.42 * inp["lysosomal_clearance_impairment"]
            + 0.22 * inp["genetic_vulnerability"]
            + 0.12 * inp["alpha_synucleinopathy_load"]
        )
        lat["alpha_synuclein_spread"] = self._clip01(
            0.42 * inp["alpha_synucleinopathy_load"]
            + 0.24 * lat["lysosomal_proteostasis_failure"]
            + 0.16 * inp["genetic_vulnerability"]
            + 0.08 * inp["alzheimer_copathology_load"]
        )
        lat["mixed_proteinopathy_burden"] = self._clip01(
            0.46 * inp["alzheimer_copathology_load"]
            + 0.22 * lat["alpha_synuclein_spread"]
            + 0.12 * inp["genetic_vulnerability"]
        )
        lat["cholinergic_deficit"] = self._clip01(
            0.34 * lat["alpha_synuclein_spread"]
            + 0.24 * lat["mixed_proteinopathy_burden"]
            + 0.18 * inp["genetic_vulnerability"]
            + 0.08 * inp["alpha_synucleinopathy_load"]
            - 0.24 * inp["cholinergic_reserve_support"]
        )
        lat["dopaminergic_deficit"] = self._clip01(
            0.38 * lat["alpha_synuclein_spread"]
            + 0.24 * lat["lysosomal_proteostasis_failure"]
            + 0.14 * inp["genetic_vulnerability"]
            + 0.08 * inp["alpha_synucleinopathy_load"]
        )
        lat["noradrenergic_arousal_instability"] = self._clip01(
            0.30 * lat["alpha_synuclein_spread"]
            + 0.18 * lat["cholinergic_deficit"]
            + 0.14 * lat["mixed_proteinopathy_burden"]
        )
        lat["limbic_cortical_synaptic_dysfunction"] = self._clip01(
            0.30 * lat["alpha_synuclein_spread"]
            + 0.22 * lat["cholinergic_deficit"]
            + 0.18 * lat["mixed_proteinopathy_burden"]
            + 0.10 * lat["dopaminergic_deficit"]
            - 0.12 * inp["cholinergic_reserve_support"]
        )
        lat["autonomic_network_dysregulation"] = self._clip01(
            0.34 * lat["alpha_synuclein_spread"]
            + 0.22 * lat["noradrenergic_arousal_instability"]
            + 0.18 * lat["dopaminergic_deficit"]
            + 0.08 * lat["mixed_proteinopathy_burden"]
        )
        lat["occipital_alpha_slowing"] = self._clip01(
            0.32 * lat["cholinergic_deficit"]
            + 0.22 * lat["limbic_cortical_synaptic_dysfunction"]
            + 0.18 * lat["noradrenergic_arousal_instability"]
            + 0.10 * lat["mixed_proteinopathy_burden"]
            - 0.14 * inp["cholinergic_reserve_support"]
        )
        lat["mesocorticolimbic_psychosis_liability"] = self._clip01(
            0.26 * lat["cholinergic_deficit"]
            + 0.20 * lat["dopaminergic_deficit"]
            + 0.18 * lat["limbic_cortical_synaptic_dysfunction"]
            + 0.16 * lat["occipital_alpha_slowing"]
            + 0.14 * inp["dopaminergic_replacement_pressure"]
            - 0.08 * inp["cholinergic_reserve_support"]
        )
        lat["neuroleptic_sensitivity"] = self._clip01(
            0.34 * lat["dopaminergic_deficit"]
            + 0.24 * inp["d2_antagonist_exposure"]
            + 0.12 * lat["alpha_synuclein_spread"]
            + 0.10 * lat["autonomic_network_dysregulation"]
            + 0.10 * inp["d2_antagonist_exposure"] * lat["dopaminergic_deficit"]
            - 0.24 * inp["low_d2_antagonism_strategy"]
        )

        regional = {
            "substantia_nigra_proxy": self._clip01(
                0.60 * lat["dopaminergic_deficit"]
                + 0.18 * lat["alpha_synuclein_spread"]
                + 0.08 * lat["lysosomal_proteostasis_failure"]
            ),
            "striatum_proxy": self._clip01(
                0.40 * lat["dopaminergic_deficit"]
                + 0.22 * lat["mesocorticolimbic_psychosis_liability"]
                + 0.16 * lat["neuroleptic_sensitivity"]
                + 0.10 * inp["d2_antagonist_exposure"]
            ),
            "locus_coeruleus_proxy": self._clip01(
                0.58 * lat["noradrenergic_arousal_instability"]
                + 0.18 * lat["alpha_synuclein_spread"]
                + 0.10 * lat["autonomic_network_dysregulation"]
            ),
            "dorsal_vagal_nucleus_proxy": self._clip01(
                0.56 * lat["autonomic_network_dysregulation"]
                + 0.22 * lat["alpha_synuclein_spread"]
                + 0.10 * lat["noradrenergic_arousal_instability"]
            ),
            "nucleus_basalis_meynert_proxy": self._clip01(
                0.60 * lat["cholinergic_deficit"]
                + 0.18 * lat["alpha_synuclein_spread"]
                + 0.10 * lat["mixed_proteinopathy_burden"]
            ),
            "amygdala": self._clip01(
                0.34 * lat["alpha_synuclein_spread"]
                + 0.24 * lat["limbic_cortical_synaptic_dysfunction"]
                + 0.16 * lat["mesocorticolimbic_psychosis_liability"]
            ),
            "parahippocampal_proxy": self._clip01(
                0.32 * lat["alpha_synuclein_spread"]
                + 0.26 * lat["mixed_proteinopathy_burden"]
                + 0.20 * lat["limbic_cortical_synaptic_dysfunction"]
            ),
            "cingulate_proxy": self._clip01(
                0.30 * lat["limbic_cortical_synaptic_dysfunction"]
                + 0.22 * lat["cholinergic_deficit"]
                + 0.18 * lat["mesocorticolimbic_psychosis_liability"]
                + 0.10 * lat["mixed_proteinopathy_burden"]
            ),
            "insula": self._clip01(
                0.30 * lat["limbic_cortical_synaptic_dysfunction"]
                + 0.22 * lat["autonomic_network_dysregulation"]
                + 0.16 * lat["alpha_synuclein_spread"]
            ),
            "frontal_cortex_proxy": self._clip01(
                0.36 * lat["limbic_cortical_synaptic_dysfunction"]
                + 0.20 * lat["cholinergic_deficit"]
                + 0.18 * lat["dopaminergic_deficit"]
                + 0.14 * lat["mixed_proteinopathy_burden"]
            ),
            "temporal_cortex_proxy": self._clip01(
                0.30 * lat["limbic_cortical_synaptic_dysfunction"]
                + 0.24 * lat["alpha_synuclein_spread"]
                + 0.16 * lat["mixed_proteinopathy_burden"]
            ),
            "occipital_visual_cortex": self._clip01(
                0.54 * lat["occipital_alpha_slowing"]
                + 0.18 * lat["cholinergic_deficit"]
                + 0.10 * lat["limbic_cortical_synaptic_dysfunction"]
            ),
            "sensorimotor_cortex_proxy": self._clip01(
                0.30 * lat["dopaminergic_deficit"]
                + 0.24 * lat["noradrenergic_arousal_instability"]
                + 0.18 * lat["alpha_synuclein_spread"]
            ),
        }

        symptoms: Dict[str, float] = {}
        symptoms["parkinsonism"] = self._clip01(
            0.34 * lat["dopaminergic_deficit"]
            + 0.24 * regional["substantia_nigra_proxy"]
            + 0.18 * regional["striatum_proxy"]
            + 0.08 * lat["autonomic_network_dysregulation"]
            - 0.20 * inp["dopaminergic_replacement_pressure"]
        )
        symptoms["cognitive_fluctuations"] = self._clip01(
            0.24 * lat["cholinergic_deficit"]
            + 0.22 * lat["occipital_alpha_slowing"]
            + 0.16 * lat["noradrenergic_arousal_instability"]
            + 0.14 * lat["limbic_cortical_synaptic_dysfunction"]
            + 0.08 * regional["frontal_cortex_proxy"]
            + 0.06 * regional["occipital_visual_cortex"]
            - 0.12 * inp["cholinergic_reserve_support"]
        )
        symptoms["visual_hallucinations"] = self._clip01(
            0.28 * lat["cholinergic_deficit"]
            + 0.22 * lat["occipital_alpha_slowing"]
            + 0.20 * lat["mesocorticolimbic_psychosis_liability"]
            + 0.12 * regional["occipital_visual_cortex"]
            + 0.08 * regional["amygdala"]
            + 0.06 * inp["dopaminergic_replacement_pressure"]
        )
        symptoms["delusions_psychosis"] = self._clip01(
            0.28 * lat["mesocorticolimbic_psychosis_liability"]
            + 0.20 * symptoms["visual_hallucinations"]
            + 0.14 * regional["frontal_cortex_proxy"]
            + 0.10 * regional["cingulate_proxy"]
            + 0.08 * inp["dopaminergic_replacement_pressure"]
        )
        symptoms["executive_dysfunction"] = self._clip01(
            0.28 * regional["frontal_cortex_proxy"]
            + 0.22 * lat["cholinergic_deficit"]
            + 0.18 * lat["dopaminergic_deficit"]
            + 0.16 * lat["limbic_cortical_synaptic_dysfunction"]
            + 0.10 * lat["mixed_proteinopathy_burden"]
        )
        symptoms["visuospatial_impairment"] = self._clip01(
            0.32 * lat["occipital_alpha_slowing"]
            + 0.24 * regional["occipital_visual_cortex"]
            + 0.16 * symptoms["cognitive_fluctuations"]
            + 0.12 * regional["parahippocampal_proxy"]
            + 0.08 * lat["cholinergic_deficit"]
        )
        symptoms["autonomic_dysfunction"] = self._clip01(
            0.34 * lat["autonomic_network_dysregulation"]
            + 0.22 * regional["dorsal_vagal_nucleus_proxy"]
            + 0.14 * regional["insula"]
            + 0.10 * regional["locus_coeruleus_proxy"]
            + 0.08 * lat["neuroleptic_sensitivity"]
        )
        symptoms["sleep_wake_disturbance"] = self._clip01(
            0.28 * lat["noradrenergic_arousal_instability"]
            + 0.24 * lat["autonomic_network_dysregulation"]
            + 0.14 * regional["locus_coeruleus_proxy"]
            + 0.12 * symptoms["cognitive_fluctuations"]
            + 0.08 * lat["occipital_alpha_slowing"]
        )
        symptoms["neuroleptic_sensitivity_crisis"] = self._clip01(
            0.34 * lat["neuroleptic_sensitivity"]
            + 0.18 * inp["d2_antagonist_exposure"]
            + 0.16 * symptoms["parkinsonism"]
            + 0.14 * symptoms["autonomic_dysfunction"]
            + 0.10 * symptoms["delusions_psychosis"]
            - 0.12 * inp["low_d2_antagonism_strategy"]
        )

        phenotypes = {
            "parkinsonian_lbd_profile": self._mean_clip(
                [
                    lat["dopaminergic_deficit"],
                    regional["substantia_nigra_proxy"],
                    regional["striatum_proxy"],
                    symptoms["parkinsonism"],
                ]
            ),
            "cholinergic_visual_cognitive_profile": self._mean_clip(
                [
                    lat["cholinergic_deficit"],
                    lat["occipital_alpha_slowing"],
                    symptoms["visual_hallucinations"],
                    symptoms["visuospatial_impairment"],
                ]
            ),
            "fluctuating_cognition_profile": self._mean_clip(
                [
                    symptoms["cognitive_fluctuations"],
                    symptoms["executive_dysfunction"],
                    lat["occipital_alpha_slowing"],
                    regional["frontal_cortex_proxy"],
                ]
            ),
            "autonomic_sleep_burden_profile": self._mean_clip(
                [
                    lat["autonomic_network_dysregulation"],
                    symptoms["autonomic_dysfunction"],
                    symptoms["sleep_wake_disturbance"],
                    regional["dorsal_vagal_nucleus_proxy"],
                ]
            ),
            "neuroleptic_sensitive_psychosis_profile": self._mean_clip(
                [
                    lat["neuroleptic_sensitivity"],
                    symptoms["visual_hallucinations"],
                    symptoms["delusions_psychosis"],
                    symptoms["neuroleptic_sensitivity_crisis"],
                ]
            ),
            "mixed_proteinopathy_rapid_decline_profile": self._mean_clip(
                [
                    lat["mixed_proteinopathy_burden"],
                    symptoms["cognitive_fluctuations"],
                    symptoms["executive_dysfunction"],
                    symptoms["visuospatial_impairment"],
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

    # ------------------------------------------------------------------
    # Spatial query helpers
    # ------------------------------------------------------------------
    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if self._pmap is None:
            try:
                with self._quiet_context():
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception:
                with self._quiet_context():
                    self._pmap = self.atlas.get_map(
                        parcellation=self.parcellation,
                        space=(
                            self.atlas.get_space(self.assignment_space)
                            if hasattr(self.atlas, "get_space")
                            else self.assignment_space
                        ),
                        maptype="statistical",
                    )

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with self._quiet_context():
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break

        if "region" in assignments.columns:
            assignments = assignments.copy()
            assignments["region"] = assignments["region"].map(self._name_of)
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
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
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 160)

    if siibra is None:
        print(
            "siibra is not installed in this environment. Install siibra, then rerun "
            "this script to build the atlas-grounded Lewy Body Disease scaffold."
        )
        raise SystemExit(0)

    model = LewyBodyDiseaseModel()
    bundle = model.build()

    print("\n=== NODE TABLE (key fields) ===")
    print(
        bundle["nodes"][["key", "node_type", "atlas_region", "region_identifier", "feature_summary", "is_proxy"]]
        .fillna("")
        .to_string(index=False)
    )

    print("\n=== EDGE TABLE ===")
    print(bundle["edges"].to_string(index=False))

    for region_key in [
        "substantia_nigra_proxy",
        "striatum_proxy",
        "locus_coeruleus_proxy",
        "dorsal_vagal_nucleus_proxy",
        "nucleus_basalis_meynert_proxy",
        "amygdala",
        "parahippocampal_proxy",
        "cingulate_proxy",
        "insula",
        "frontal_cortex_proxy",
        "temporal_cortex_proxy",
        "occipital_visual_cortex",
        "sensorimotor_cortex_proxy",
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
        genetic_vulnerability=0.55,
        alpha_synucleinopathy_load=0.70,
        lysosomal_clearance_impairment=0.50,
        alzheimer_copathology_load=0.35,
        dopaminergic_replacement_pressure=0.30,
        d2_antagonist_exposure=0.15,
        cholinergic_reserve_support=0.15,
        low_d2_antagonism_strategy=0.25,
    )

    print("\n=== SIMULATION: example mixed motor-cognitive LBD phenotype ===")
    for name, series in example.items():
        print(f"\n{name.upper()}")
        print(series.to_string())

    # Example coordinate usage:
    # assignments = model.assign_mni_point((-8, -16, -10))
    # print(assignments.head(10).to_string(index=False))
