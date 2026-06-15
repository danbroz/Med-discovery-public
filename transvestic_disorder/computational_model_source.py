from __future__ import annotations

"""
Transvestic Disorder siibra scaffold.

This script converts a chapter-level biological summary of Transvestic Disorder
(TD) into an atlas-grounded research scaffold built around siibra idioms. It is
for hypothesis generation, teaching, and reproducible exploration of
atlas-backed regional features. It is not a clinical diagnostic or treatment
system.

Important scope note:
- This scaffold models the chapter's *disorder construct*, which explicitly
  requires clinically significant distress or impairment.
- It does not imply that cross-dressing, gender nonconformity, or varied gender
  expression are pathological in themselves.

Conceptual choices:
- The chapter describes a still-nascent biological literature, so the model is
  explicitly hypothesis-driven rather than biomarker-validated.
- Reward and motivation circuitry are treated as central for recurrent sexual
  arousal and reinforcement of the chapter-defined behavioral pattern.
- OCD-like orbitofrontal/anterior-cingulate/caudate circuitry is used
  conservatively to model compulsive repetition and impaired resistance to
  urges because the chapter specifically invokes impulse-control and habit-loop
  analogies from OCD.
- Chronic concealment, internal conflict, and stigma are modeled as major
  drivers of HPA-axis and noradrenergic stress dysregulation, with downstream
  impacts on prefrontal cortex, hippocampus, and amygdala-centered fear
  regulation.
- vmPFC and caudate are kept as clearly labeled proxies because the chapter is
  systems-level and atlas labels vary across siibra versions.
- The gene panel is hypothesis-driven around dopamine, serotonin,
  norepinephrine, stress-axis regulation, and plasticity.

The scaffold degrades gracefully when siibra or specific multimodal features are
unavailable. Atlas-backed nodes remain visible, while unresolved feature tables
are returned empty instead of raising errors.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception:  # pragma: no cover - optional dependency in some environments
    siibra = None


DEFAULT_GENE_PANEL = [
    # Dopamine / reward / motivational salience
    "DRD2",
    "DRD4",
    "SLC6A3",
    "COMT",
    # Serotonergic affect and compulsivity modulation
    "SLC6A4",
    "HTR2A",
    "HTR1A",
    "MAOA",
    # Noradrenergic arousal and stress responsivity
    "SLC6A2",
    "ADRA2A",
    # HPA-axis and stress embedding
    "CRHR1",
    "FKBP5",
    "NR3C1",
    # Plasticity / long-term adaptation
    "BDNF",
]


class TransvesticDisorderModel:
    """
    Atlas-grounded research scaffold for Transvestic Disorder.

    The simulator keeps a transparent one-pass causal flow:
        inputs -> latent biology -> regional burden -> symptoms -> phenotypes

    Because the empirical literature summarized in the chapter is sparse,
    several latent variables represent mechanistic hypotheses inferred from the
    chapter's stated reward, compulsivity, stress, and frontolimbic models.
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
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Fo4",
                "Fo3",
                "orbitofrontal",
            ],
            "acc": [
                "Area p24 left",
                "Area a24 left",
                "Area s24 left",
                "anterior cingulate cortex",
                "cingulate",
            ],
            "hippocampus": [
                "CA1 left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "vmpfc_proxy": [
                "Area 14M left",
                "Area 10 left",
                "Area 32 left",
                "ventromedial prefrontal cortex",
                "medial prefrontal cortex",
            ],
            "caudate_proxy": [
                "caudate nucleus left",
                "caudate left",
                "caudate nucleus",
                "caudate",
                "striatum",
            ],
        }

        self.proxy_region_nodes = {
            "vmpfc_proxy",
            "caudate_proxy",
        }

        self.region_descriptions: Dict[str, str] = {
            "amygdala": (
                "Direct atlas anchor for the chapter's fear, salience, and social-judgment circuitry."
            ),
            "ofc": (
                "Direct atlas anchor for orbitofrontal reward valuation and compulsive habit-loop control."
            ),
            "acc": (
                "Direct atlas anchor for conflict monitoring, distress processing, and control allocation."
            ),
            "hippocampus": (
                "Direct atlas anchor for stress-sensitive memory/context systems affected by chronic stress."
            ),
            "vmpfc_proxy": (
                "Conservative proxy for ventromedial prefrontal regulation used because the chapter is "
                "systems-level and atlas labels may vary."
            ),
            "caudate_proxy": (
                "Conservative proxy for caudate-centered habit and impulse-control circuitry referenced "
                "through OCD-like CSTC comparisons."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "familial_atypical_interest_loading": (
                "Possible familial loading for atypical sexual-interest patterns noted as a hypothesis in the chapter."
            ),
            "obsessive_compulsive_trait_loading": (
                "Familial or temperamental obsessive-compulsive traits that may bias habit formation and urge resistance."
            ),
            "arousal_reinforcement_history": (
                "History of repeated reinforcement of the disorder-defined arousal pattern, strengthening reward salience."
            ),
            "early_life_stress": (
                "Early stress or trauma that may produce durable biological and epigenetic effects."
            ),
            "chronic_concealment_stress": (
                "Sustained stress from concealment of the behavior pattern and fear of exposure."
            ),
            "internal_conflict": (
                "Persistent internal conflict around the behavior pattern and its meaning."
            ),
            "social_stigma_exposure": (
                "External stigma, fear of judgment, discovery, rejection, or social punishment."
            ),
            "affirming_support": (
                "Protective affirming and regulating support that can reduce concealment burden and chronic stress."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "familial_vulnerability": (
                "Integrated inherited vulnerability spanning atypical-interest loading and compulsive trait architecture."
            ),
            "epigenetic_stress_embedding": (
                "Long-lasting biological embedding of early or chronic stress through gene-regulatory mechanisms."
            ),
            "dopamine_reward_motivation_bias": (
                "Reward and motivation bias that strengthens arousal salience and incentive pull."
            ),
            "serotonergic_regulation_shift": (
                "Serotonergic modulation shift affecting compulsivity, affect regulation, and distress sensitivity."
            ),
            "hpa_axis_dysregulation": (
                "Stress-axis dysregulation arising from concealment, conflict, stigma, and prior stress exposure."
            ),
            "noradrenergic_hyperarousal": (
                "Noradrenergic hyperarousal supporting anxiety, hypervigilance, and exaggerated startle."
            ),
            "frontostriatal_compulsivity": (
                "OFC-ACC-striatal habit-loop engagement and reduced resistance to repetitive urges."
            ),
            "pfc_top_down_weakening": (
                "Reduced prefrontal control over emotional responses and compulsive behavioral output."
            ),
            "amygdala_social_threat_reactivity": (
                "Heightened amygdalar reactivity to judgment, discovery, rejection, and shame-related cues."
            ),
            "hippocampal_stress_burden": (
                "Stress-related burden on hippocampal contextual memory and emotional regulation functions."
            ),
            "shame_conflict_dysphoria": (
                "Integrated negative-affect state of shame, dysphoria, irritability, and emotional overload."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "recurrent_cross_dressing_arousal": (
                "Recurrent intense sexual arousal tied to cross-dressing within the chapter's disorder definition."
            ),
            "compulsive_cross_dressing_behavior": (
                "Repetitive enactment with difficulty resisting urges despite adverse consequences."
            ),
            "shame_distress": (
                "Clinically significant shame, distress, or overwhelm associated with the behavior pattern."
            ),
            "anxiety_hypervigilance": (
                "Generalized anxiety, hypervigilance, and scanning for discovery or judgment."
            ),
            "exaggerated_startle": (
                "Heightened startle and autonomic reactivity linked to chronic arousal of stress systems."
            ),
            "episodic_dysphoria_irritability": (
                "Episodes of dysphoria or irritability linked to volatile stress-system dynamics."
            ),
            "depressive_burden": (
                "Comorbid depressive burden supported by chronic stress, shame, and dysregulation."
            ),
            "functional_impairment": (
                "Social, occupational, or relational impairment arising from compulsion, distress, and concealment."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "familial_atypical_interest_loading",
                "target": "familial_vulnerability",
                "relation": "familial clustering of atypical sexual interests may contribute to baseline liability",
                "transvestic_change": "increased",
            },
            {
                "source": "obsessive_compulsive_trait_loading",
                "target": "familial_vulnerability",
                "relation": "familial obsessive-compulsive traits may bias urge-related vulnerability",
                "transvestic_change": "increased",
            },
            {
                "source": "early_life_stress",
                "target": "epigenetic_stress_embedding",
                "relation": "early stress can leave durable biological marks through epigenetic mechanisms",
                "transvestic_change": "increased",
            },
            {
                "source": "chronic_concealment_stress",
                "target": "epigenetic_stress_embedding",
                "relation": "chronic stress may reinforce lasting stress-system recalibration",
                "transvestic_change": "increased",
            },
            {
                "source": "familial_vulnerability",
                "target": "dopamine_reward_motivation_bias",
                "relation": "baseline vulnerability may bias reward and motivational salience systems",
                "transvestic_change": "increased",
            },
            {
                "source": "arousal_reinforcement_history",
                "target": "dopamine_reward_motivation_bias",
                "relation": "repeated reinforcement strengthens incentive salience and reward pull",
                "transvestic_change": "increased",
            },
            {
                "source": "familial_vulnerability",
                "target": "serotonergic_regulation_shift",
                "relation": "baseline liability may contribute to serotonergic regulation differences",
                "transvestic_change": "increased",
            },
            {
                "source": "epigenetic_stress_embedding",
                "target": "serotonergic_regulation_shift",
                "relation": "stress embedding can alter affective regulation systems",
                "transvestic_change": "increased",
            },
            {
                "source": "epigenetic_stress_embedding",
                "target": "hpa_axis_dysregulation",
                "relation": "stress embedding sensitizes the HPA axis",
                "transvestic_change": "increased",
            },
            {
                "source": "chronic_concealment_stress",
                "target": "hpa_axis_dysregulation",
                "relation": "concealment burden chronically activates stress physiology",
                "transvestic_change": "increased",
            },
            {
                "source": "internal_conflict",
                "target": "hpa_axis_dysregulation",
                "relation": "internal conflict sustains chronic stress-system activation",
                "transvestic_change": "increased",
            },
            {
                "source": "social_stigma_exposure",
                "target": "hpa_axis_dysregulation",
                "relation": "stigma and judgment exposure amplify stress-axis burden",
                "transvestic_change": "increased",
            },
            {
                "source": "affirming_support",
                "target": "hpa_axis_dysregulation",
                "relation": "affirming support can buffer chronic stress physiology",
                "transvestic_change": "decreased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "noradrenergic_hyperarousal",
                "relation": "chronic stress dysregulation recruits noradrenergic hyperarousal",
                "transvestic_change": "increased",
            },
            {
                "source": "social_stigma_exposure",
                "target": "noradrenergic_hyperarousal",
                "relation": "fear of judgment or discovery heightens arousal and vigilance",
                "transvestic_change": "increased",
            },
            {
                "source": "dopamine_reward_motivation_bias",
                "target": "frontostriatal_compulsivity",
                "relation": "reward-driven salience can strengthen repetitive habit loops",
                "transvestic_change": "increased",
            },
            {
                "source": "obsessive_compulsive_trait_loading",
                "target": "frontostriatal_compulsivity",
                "relation": "obsessive-compulsive trait loading increases CSTC-style compulsive pressure",
                "transvestic_change": "increased",
            },
            {
                "source": "serotonergic_regulation_shift",
                "target": "frontostriatal_compulsivity",
                "relation": "serotonergic dysregulation can weaken inhibition and increase compulsive repetition",
                "transvestic_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "pfc_top_down_weakening",
                "relation": "chronic stress impairs prefrontal executive and regulatory function",
                "transvestic_change": "increased",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "pfc_top_down_weakening",
                "relation": "hyperarousal weakens stable top-down control",
                "transvestic_change": "increased",
            },
            {
                "source": "frontostriatal_compulsivity",
                "target": "pfc_top_down_weakening",
                "relation": "compulsive drive makes inhibitory control less effective",
                "transvestic_change": "increased",
            },
            {
                "source": "social_stigma_exposure",
                "target": "amygdala_social_threat_reactivity",
                "relation": "judgment and rejection cues heighten amygdala threat reactivity",
                "transvestic_change": "increased",
            },
            {
                "source": "internal_conflict",
                "target": "amygdala_social_threat_reactivity",
                "relation": "internal conflict amplifies shame- and discovery-related salience",
                "transvestic_change": "increased",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "amygdala_social_threat_reactivity",
                "relation": "hyperarousal intensifies fear and salience responses",
                "transvestic_change": "increased",
            },
            {
                "source": "pfc_top_down_weakening",
                "target": "amygdala_social_threat_reactivity",
                "relation": "weak vmPFC/PFC control fails to dampen amygdala fear output",
                "transvestic_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "hippocampal_stress_burden",
                "relation": "chronic stress burdens hippocampal contextual and regulatory systems",
                "transvestic_change": "increased",
            },
            {
                "source": "early_life_stress",
                "target": "hippocampal_stress_burden",
                "relation": "earlier stress exposure sensitizes hippocampal stress vulnerability",
                "transvestic_change": "increased",
            },
            {
                "source": "amygdala_social_threat_reactivity",
                "target": "shame_conflict_dysphoria",
                "relation": "social-threat reactivity fuels shame and dysphoric affect",
                "transvestic_change": "increased",
            },
            {
                "source": "internal_conflict",
                "target": "shame_conflict_dysphoria",
                "relation": "conflict around the behavior pattern deepens dysphoria and irritability",
                "transvestic_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "shame_conflict_dysphoria",
                "relation": "stress-system volatility contributes to dysphoric and irritable episodes",
                "transvestic_change": "increased",
            },
            {
                "source": "affirming_support",
                "target": "shame_conflict_dysphoria",
                "relation": "supportive context may reduce shame and negative affect",
                "transvestic_change": "decreased",
            },
            {
                "source": "frontostriatal_compulsivity",
                "target": "ofc",
                "relation": "compulsive repetition burdens orbitofrontal control circuitry",
                "transvestic_change": "increased",
            },
            {
                "source": "frontostriatal_compulsivity",
                "target": "acc",
                "relation": "habit conflict and monitoring pressures burden anterior cingulate circuitry",
                "transvestic_change": "increased",
            },
            {
                "source": "frontostriatal_compulsivity",
                "target": "caudate_proxy",
                "relation": "habit-loop engagement burdens caudate-centered circuitry",
                "transvestic_change": "increased",
            },
            {
                "source": "amygdala_social_threat_reactivity",
                "target": "amygdala",
                "relation": "threat-reactive salience concentrates burden in amygdala-centered networks",
                "transvestic_change": "increased",
            },
            {
                "source": "pfc_top_down_weakening",
                "target": "vmpfc_proxy",
                "relation": "reduced top-down control manifests in vmPFC-centered regulation territory",
                "transvestic_change": "increased",
            },
            {
                "source": "hippocampal_stress_burden",
                "target": "hippocampus",
                "relation": "stress burden engages hippocampal contextual-memory circuitry",
                "transvestic_change": "increased",
            },
            {
                "source": "dopamine_reward_motivation_bias",
                "target": "recurrent_cross_dressing_arousal",
                "relation": "reward salience and motivation strengthen recurrent arousal",
                "transvestic_change": "increased",
            },
            {
                "source": "ofc",
                "target": "compulsive_cross_dressing_behavior",
                "relation": "orbitofrontal habit valuation contributes to repetitive enactment",
                "transvestic_change": "increased",
            },
            {
                "source": "acc",
                "target": "compulsive_cross_dressing_behavior",
                "relation": "conflict-monitoring burden can accompany compulsive repetition",
                "transvestic_change": "increased",
            },
            {
                "source": "caudate_proxy",
                "target": "compulsive_cross_dressing_behavior",
                "relation": "caudate-centered habit circuitry supports repetitive performance",
                "transvestic_change": "increased",
            },
            {
                "source": "shame_conflict_dysphoria",
                "target": "shame_distress",
                "relation": "integrated negative-affect load drives clinically significant distress",
                "transvestic_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "shame_distress",
                "relation": "threat and shame salience intensify distress experience",
                "transvestic_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "anxiety_hypervigilance",
                "relation": "amygdala threat bias drives anxiety and vigilance",
                "transvestic_change": "increased",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "anxiety_hypervigilance",
                "relation": "noradrenergic activation supports anxious hypervigilance",
                "transvestic_change": "increased",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "exaggerated_startle",
                "relation": "hyperarousal increases startle reactivity",
                "transvestic_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "episodic_dysphoria_irritability",
                "relation": "volatile stress-system dynamics support dysphoria and irritability",
                "transvestic_change": "increased",
            },
            {
                "source": "shame_conflict_dysphoria",
                "target": "episodic_dysphoria_irritability",
                "relation": "negative-affect load is expressed as dysphoric and irritable episodes",
                "transvestic_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "depressive_burden",
                "relation": "stress-related hippocampal dysfunction can contribute to depressive burden",
                "transvestic_change": "increased",
            },
            {
                "source": "shame_conflict_dysphoria",
                "target": "depressive_burden",
                "relation": "chronic shame and dysphoria contribute to depressive symptoms",
                "transvestic_change": "increased",
            },
            {
                "source": "compulsive_cross_dressing_behavior",
                "target": "functional_impairment",
                "relation": "repetition despite consequences can produce social and occupational impairment",
                "transvestic_change": "increased",
            },
            {
                "source": "shame_distress",
                "target": "functional_impairment",
                "relation": "distress and concealment burden contribute to functional disruption",
                "transvestic_change": "increased",
            },
            {
                "source": "anxiety_hypervigilance",
                "target": "functional_impairment",
                "relation": "persistent anxiety and hypervigilance interfere with functioning",
                "transvestic_change": "increased",
            },
            {
                "source": "depressive_burden",
                "target": "functional_impairment",
                "relation": "depressive symptoms add to overall impairment",
                "transvestic_change": "increased",
            },
            {
                "source": "affirming_support",
                "target": "functional_impairment",
                "relation": "support may reduce distress-driven impairment",
                "transvestic_change": "decreased",
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
            "orbitofrontal cortex",
            "anterior cingulate cortex",
            "caudate nucleus",
            "medial prefrontal cortex",
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
        familial_atypical_interest_loading: float = 0.45,
        obsessive_compulsive_trait_loading: float = 0.5,
        arousal_reinforcement_history: float = 0.65,
        early_life_stress: float = 0.35,
        chronic_concealment_stress: float = 0.7,
        internal_conflict: float = 0.65,
        social_stigma_exposure: float = 0.6,
        affirming_support: float = 0.2,
    ) -> Dict[str, pd.Series]:
        """
        Transparent one-pass simulator for disorder-level TD burden.

        Inputs are normalized to [0, 1]. Outputs reflect relative mechanistic
        burden, not prevalence, moral judgment, or probability of diagnosis.
        """
        inputs = {
            "familial_atypical_interest_loading": familial_atypical_interest_loading,
            "obsessive_compulsive_trait_loading": obsessive_compulsive_trait_loading,
            "arousal_reinforcement_history": arousal_reinforcement_history,
            "early_life_stress": early_life_stress,
            "chronic_concealment_stress": chronic_concealment_stress,
            "internal_conflict": internal_conflict,
            "social_stigma_exposure": social_stigma_exposure,
            "affirming_support": affirming_support,
        }
        x = {k: self._clip01(v) for k, v in inputs.items()}

        latents: Dict[str, float] = {}
        latents["familial_vulnerability"] = self._clip01(
            0.34 * x["familial_atypical_interest_loading"]
            + 0.34 * x["obsessive_compulsive_trait_loading"]
            + 0.08 * x["early_life_stress"]
            - 0.06 * x["affirming_support"]
        )
        latents["epigenetic_stress_embedding"] = self._clip01(
            0.40 * x["early_life_stress"]
            + 0.26 * x["chronic_concealment_stress"]
            + 0.14 * x["social_stigma_exposure"]
            - 0.10 * x["affirming_support"]
        )
        latents["dopamine_reward_motivation_bias"] = self._clip01(
            0.34 * latents["familial_vulnerability"]
            + 0.40 * x["arousal_reinforcement_history"]
            + 0.08 * x["obsessive_compulsive_trait_loading"]
            - 0.06 * x["affirming_support"]
        )
        latents["serotonergic_regulation_shift"] = self._clip01(
            0.26 * latents["familial_vulnerability"]
            + 0.30 * latents["epigenetic_stress_embedding"]
            + 0.18 * x["internal_conflict"]
            - 0.10 * x["affirming_support"]
        )
        latents["hpa_axis_dysregulation"] = self._clip01(
            0.26 * latents["epigenetic_stress_embedding"]
            + 0.26 * x["chronic_concealment_stress"]
            + 0.18 * x["internal_conflict"]
            + 0.18 * x["social_stigma_exposure"]
            - 0.18 * x["affirming_support"]
        )
        latents["noradrenergic_hyperarousal"] = self._clip01(
            0.48 * latents["hpa_axis_dysregulation"]
            + 0.18 * x["social_stigma_exposure"]
            + 0.10 * x["internal_conflict"]
            - 0.10 * x["affirming_support"]
        )
        latents["frontostriatal_compulsivity"] = self._clip01(
            0.28 * latents["dopamine_reward_motivation_bias"]
            + 0.28 * x["obsessive_compulsive_trait_loading"]
            + 0.18 * latents["serotonergic_regulation_shift"]
            + 0.08 * x["arousal_reinforcement_history"]
            - 0.08 * x["affirming_support"]
        )
        latents["pfc_top_down_weakening"] = self._clip01(
            0.34 * latents["hpa_axis_dysregulation"]
            + 0.24 * latents["noradrenergic_hyperarousal"]
            + 0.18 * latents["frontostriatal_compulsivity"]
            - 0.16 * x["affirming_support"]
        )
        latents["amygdala_social_threat_reactivity"] = self._clip01(
            0.26 * x["social_stigma_exposure"]
            + 0.22 * x["internal_conflict"]
            + 0.20 * latents["noradrenergic_hyperarousal"]
            + 0.18 * latents["pfc_top_down_weakening"]
            - 0.14 * x["affirming_support"]
        )
        latents["hippocampal_stress_burden"] = self._clip01(
            0.34 * latents["hpa_axis_dysregulation"]
            + 0.24 * x["early_life_stress"]
            + 0.10 * latents["epigenetic_stress_embedding"]
            - 0.08 * x["affirming_support"]
        )
        latents["shame_conflict_dysphoria"] = self._clip01(
            0.30 * latents["amygdala_social_threat_reactivity"]
            + 0.24 * x["internal_conflict"]
            + 0.18 * latents["hpa_axis_dysregulation"]
            + 0.10 * x["social_stigma_exposure"]
            - 0.16 * x["affirming_support"]
        )

        regional_state = {
            "amygdala": self._clip01(
                0.56 * latents["amygdala_social_threat_reactivity"]
                + 0.14 * latents["noradrenergic_hyperarousal"]
                + 0.08 * latents["shame_conflict_dysphoria"]
            ),
            "ofc": self._clip01(
                0.52 * latents["frontostriatal_compulsivity"]
                + 0.14 * latents["dopamine_reward_motivation_bias"]
                + 0.10 * latents["pfc_top_down_weakening"]
            ),
            "acc": self._clip01(
                0.40 * latents["frontostriatal_compulsivity"]
                + 0.22 * latents["shame_conflict_dysphoria"]
                + 0.12 * latents["hpa_axis_dysregulation"]
            ),
            "hippocampus": self._clip01(
                0.56 * latents["hippocampal_stress_burden"]
                + 0.12 * latents["hpa_axis_dysregulation"]
                + 0.08 * latents["shame_conflict_dysphoria"]
            ),
            "vmpfc_proxy": self._clip01(
                0.56 * latents["pfc_top_down_weakening"]
                + 0.16 * latents["amygdala_social_threat_reactivity"]
                + 0.10 * latents["hpa_axis_dysregulation"]
            ),
            "caudate_proxy": self._clip01(
                0.56 * latents["frontostriatal_compulsivity"]
                + 0.18 * latents["dopamine_reward_motivation_bias"]
                + 0.06 * x["obsessive_compulsive_trait_loading"]
            ),
        }

        symptoms = {
            "recurrent_cross_dressing_arousal": self._clip01(
                0.54 * latents["dopamine_reward_motivation_bias"]
                + 0.16 * x["arousal_reinforcement_history"]
                - 0.04 * x["affirming_support"]
            ),
            "compulsive_cross_dressing_behavior": 0.0,
            "shame_distress": self._clip01(
                0.38 * latents["shame_conflict_dysphoria"]
                + 0.18 * regional_state["amygdala"]
                + 0.16 * x["chronic_concealment_stress"]
                - 0.14 * x["affirming_support"]
            ),
            "anxiety_hypervigilance": self._clip01(
                0.34 * regional_state["amygdala"]
                + 0.30 * latents["noradrenergic_hyperarousal"]
                + 0.10 * regional_state["vmpfc_proxy"]
                - 0.12 * x["affirming_support"]
            ),
            "exaggerated_startle": self._clip01(
                0.46 * latents["noradrenergic_hyperarousal"]
                + 0.24 * regional_state["amygdala"]
                - 0.08 * x["affirming_support"]
            ),
            "episodic_dysphoria_irritability": self._clip01(
                0.38 * latents["shame_conflict_dysphoria"]
                + 0.26 * latents["hpa_axis_dysregulation"]
                + 0.08 * regional_state["acc"]
                - 0.10 * x["affirming_support"]
            ),
            "depressive_burden": self._clip01(
                0.28 * latents["shame_conflict_dysphoria"]
                + 0.24 * regional_state["hippocampus"]
                + 0.18 * x["chronic_concealment_stress"]
                + 0.12 * x["internal_conflict"]
                - 0.14 * x["affirming_support"]
            ),
            "functional_impairment": 0.0,
        }
        symptoms["compulsive_cross_dressing_behavior"] = self._clip01(
            0.24 * symptoms["recurrent_cross_dressing_arousal"]
            + 0.26 * regional_state["ofc"]
            + 0.18 * regional_state["caudate_proxy"]
            + 0.14 * regional_state["acc"]
            - 0.08 * x["affirming_support"]
        )
        symptoms["functional_impairment"] = self._clip01(
            0.22 * symptoms["compulsive_cross_dressing_behavior"]
            + 0.20 * symptoms["shame_distress"]
            + 0.16 * symptoms["anxiety_hypervigilance"]
            + 0.18 * symptoms["depressive_burden"]
            + 0.08 * x["chronic_concealment_stress"]
            - 0.12 * x["affirming_support"]
        )

        phenotypes = {
            "reward_compulsivity_profile": self._clip01(
                (
                    latents["dopamine_reward_motivation_bias"]
                    + latents["frontostriatal_compulsivity"]
                    + symptoms["recurrent_cross_dressing_arousal"]
                    + symptoms["compulsive_cross_dressing_behavior"]
                )
                / 4.0
            ),
            "concealment_stress_profile": self._clip01(
                (
                    latents["hpa_axis_dysregulation"]
                    + latents["noradrenergic_hyperarousal"]
                    + symptoms["anxiety_hypervigilance"]
                    + symptoms["exaggerated_startle"]
                )
                / 4.0
            ),
            "dysphoric_internal_conflict_profile": self._clip01(
                (
                    latents["shame_conflict_dysphoria"]
                    + symptoms["shame_distress"]
                    + symptoms["episodic_dysphoria_irritability"]
                    + symptoms["depressive_burden"]
                )
                / 4.0
            ),
            "clinical_td_burden_profile": self._clip01(
                (
                    symptoms["compulsive_cross_dressing_behavior"]
                    + symptoms["shame_distress"]
                    + symptoms["functional_impairment"]
                    + symptoms["depressive_burden"]
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
    model = TransvesticDisorderModel()
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
        print("No circuit connectivity available in this environment.")

    print("\n=== Example amygdala receptor table ===")
    amygdala_receptors = bundle["receptors"].get("amygdala", pd.DataFrame())
    if isinstance(amygdala_receptors, pd.DataFrame) and not amygdala_receptors.empty:
        print(amygdala_receptors.head(10).to_string(index=False))
    else:
        print("No amygdala receptor table available in this environment.")

    print("\n=== Example amygdala gene table ===")
    amygdala_genes = bundle["genes"].get("amygdala", pd.DataFrame())
    if isinstance(amygdala_genes, pd.DataFrame) and not amygdala_genes.empty:
        print(amygdala_genes.head(10).to_string(index=False))
    else:
        print("No amygdala gene-expression table available in this environment.")

    print("\n=== Example OFC connectivity profile ===")
    ofc_conn = bundle["connectivity_profiles"].get("ofc", pd.DataFrame())
    if isinstance(ofc_conn, pd.DataFrame) and not ofc_conn.empty:
        print(ofc_conn.head(10).to_string(index=False))
    else:
        print("No OFC connectivity profile available in this environment.")

    print("\n=== Example simulation ===")
    sim = model.simulate(
        familial_atypical_interest_loading=0.48,
        obsessive_compulsive_trait_loading=0.56,
        arousal_reinforcement_history=0.72,
        early_life_stress=0.34,
        chronic_concealment_stress=0.76,
        internal_conflict=0.68,
        social_stigma_exposure=0.63,
        affirming_support=0.18,
    )
    for section, series in sim.items():
        print(f"\n[{section}]")
        print(series.round(3).to_string())

    # Optional manual region search examples:
    # print(model.suggest_regions("orbitofrontal").head(10).to_string(index=False))
    # print(model.assign_mni_point((-4, 38, -12)).head(10).to_string(index=False))
