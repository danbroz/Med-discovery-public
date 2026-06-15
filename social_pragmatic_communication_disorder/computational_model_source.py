
from __future__ import annotations

"""
Social (Pragmatic) Communication Disorder atlas-grounded siibra scaffold.

This script translates a chapter-level biological discussion of Social
(Pragmatic) Communication Disorder (SCD) into a transparent, research-oriented
mechanistic graph. The chapter frames SCD as a disorder of social communication
systems rather than language in isolation, emphasizing social reward,
stress/anxiety modulation, the "social brain", and large-scale network
integration.

Main biological themes encoded here:
- dopaminergic reward processes supporting social motivation and belonging,
- endocannabinoid modulation of stress, anxiety, and social pleasure,
- polygenic and developmental influences on social communication,
- atypical development of the social-brain network,
- weaker coupling between medial prefrontal and limbic systems,
- reduced integration between temporoparietal mentalizing systems and
  language-related temporal/frontal regions.

Important:
- This is a research scaffold, not a diagnostic or treatment tool.
- Higher simulated values indicate greater dysregulation burden, symptom
  pressure, or functional impairment unless explicitly noted otherwise.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore[assignment]
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_SCD_GENE_PANEL = [
    # Social reward / salience
    "DRD2",
    "DRD1",
    "SLC6A3",
    "COMT",
    # Endocannabinoid regulation
    "CNR1",
    "FAAH",
    "MGLL",
    # Social-affective / serotonergic modulation
    "SLC6A4",
    "HTR1A",
    "HTR2A",
    # Developmental communication / synaptic organization
    "FOXP2",
    "CNTNAP2",
    "SHANK3",
    "NRXN1",
    "BDNF",
    # Social bonding / affiliative learning (theoretical extension)
    "OXTR",
]


class SocialPragmaticCommunicationDisorderModel:
    """
    Atlas-grounded research scaffold for Social (Pragmatic) Communication Disorder.

    The motivating chapter treats SCD as a disturbance of social cognition,
    social motivation, stress regulation, and socio-linguistic integration.
    Accordingly, this scaffold combines atlas-backed social-brain regions with
    explicit latent neurochemical and connectivity processes.

    It is intentionally conservative:
    - named regions in the chapter are anchored directly where practical,
    - systems-level reward circuitry is modeled with a proxy node,
    - neurotransmitters remain latent unless the chapter localizes them.
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
                "siibra is required to use this scaffold. Install it in your Python "
                "environment before running the model."
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

        # Conservative atlas anchors from the chapter:
        # - mPFC, TPJ, STS, ACC, amygdala, insula are named directly.
        # - IFG is added as a language-interface node because the chapter
        #   explicitly discusses reduced TPJ-to-language integration.
        # - ventral striatum is kept as a proxy because reward circuitry is
        #   central to the chapter but may not have a clean Julich match in
        #   every environment.
        self.region_candidates: Dict[str, List[str]] = {
            "mpfc": [
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "Area s32 (sACC) left",
                "Area p32 (pACC) left",
                "medial prefrontal cortex",
                "frontal pole",
            ],
            "tpj": [
                "Area TPJ (STG/SMG) left",
                "TPJ",
                "temporoparietal junction",
                "PGp left",
                "PFm left",
            ],
            "sts_proxy": [
                "Area TE 3 (STG) left",
                "Area TE 2.1 (STG) left",
                "Area TE 2.2 (STG) left",
                "superior temporal sulcus",
                "superior temporal",
            ],
            "acc": [
                "Area p32 (pACC) left",
                "Area 33 (ACC) left",
                "Area s32 (sACC) left",
                "anterior cingulate",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "insula": [
                "Area Id2 (Insula) left",
                "Area Id3 (Insula) left",
                "Area Ig1 (Insula) left",
                "insula",
            ],
            "ifg_language_interface": [
                "Area 45 (IFG) left",
                "Area 44 (IFG) left",
                "inferior frontal gyrus",
                "broca",
            ],
            "ventral_striatum_proxy": [
                "ventral striatum",
                "nucleus accumbens left",
                "BST (Bed Nucleus) left",
            ],
        }

        self.proxy_only_nodes = {"ventral_striatum_proxy"}

        self.region_descriptions: Dict[str, str] = {
            "mpfc": (
                "Medial prefrontal social-cognitive control node representing self-other modeling, "
                "mental state inference, and top-down social evaluation."
            ),
            "tpj": (
                "Temporoparietal junction node representing perspective taking, intention attribution, "
                "and pragmatic social inference."
            ),
            "sts_proxy": (
                "Superior temporal social-perception proxy representing interpretation of dynamic social cues "
                "and their integration into pragmatic meaning."
            ),
            "acc": (
                "Anterior cingulate node representing social conflict monitoring, error processing, and "
                "social-affective control."
            ),
            "amygdala": (
                "Amygdala node representing threat salience, emotional significance, and sensitivity to "
                "negative social evaluation."
            ),
            "insula": (
                "Insula node representing interoceptive social discomfort, affective awareness, and "
                "social-anxiety-linked bodily salience."
            ),
            "ifg_language_interface": (
                "Inferior frontal language-interface node representing controlled pragmatic formulation and "
                "integration of social inference with expressive language."
            ),
            "ventral_striatum_proxy": (
                "Proxy for social reward circuitry involved in belonging, reinforcement, and motivational "
                "engagement with social learning."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "polygenic_social_communication_liability": (
                "Broad inherited vulnerability affecting social cognition, reward responsivity, and "
                "developmental communication trajectories."
            ),
            "repeated_social_failure_exclusion": (
                "Accumulated social misunderstanding, rejection, or exclusion that can diminish reward from "
                "social engagement and strengthen withdrawal."
            ),
            "social_stress_anxiety_burden": (
                "Current stress load and fear of negative evaluation increasing avoidance and dysregulation "
                "during social communication."
            ),
            "pragmatic_language_demand_load": (
                "High real-world demand for context-sensitive, inferential, or nonliteral communication."
            ),
            "social_learning_enrichment": (
                "Protective access to supportive social practice, scaffolding, and repeated successful "
                "social learning opportunities."
            ),
            "communication_support_intervention": (
                "Protective intervention support that improves explicit strategy use, contextual inference, "
                "and compensation."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "dopaminergic_social_reward_blunting": (
                "Reduced or dysregulated reward value of successful social interaction and belonging."
            ),
            "endocannabinoid_stress_buffer_failure": (
                "Weakened buffering of social stress and anxiety, increasing avoidance and negative arousal."
            ),
            "social_brain_network_dysconnectivity": (
                "Atypical large-scale connectivity among medial prefrontal, temporoparietal, temporal, "
                "cingulate, limbic, and insular social-brain nodes."
            ),
            "amygdala_mpfc_regulatory_weakness": (
                "Reduced capacity of medial prefrontal systems to regulate social threat and affective reactivity."
            ),
            "social_threat_bias": (
                "Heightened expectation of negative evaluation and over-weighting of social threat cues."
            ),
            "mental_state_inference_impairment": (
                "Reduced efficiency of inferring intentions, beliefs, and pragmatic meaning from social signals."
            ),
            "pragmatic_language_integration_failure": (
                "Difficulty integrating social inference with contextual language comprehension and production."
            ),
            "reduced_social_learning_drive": (
                "Lower motivation to engage in repeated social-learning episodes needed to build pragmatic skill."
            ),
            "executive_social_context_deficit": (
                "Difficulty flexibly monitoring context, updating conversational rules, and managing social ambiguity."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "impaired_pragmatic_inference": (
                "Difficulty inferring speaker intent, indirect meaning, or social subtext."
            ),
            "poor_context_sensitive_language_use": (
                "Difficulty using language appropriately for listener, setting, or conversational context."
            ),
            "fear_of_negative_evaluation": (
                "Heightened worry about being judged, misunderstood, or embarrassed in social communication."
            ),
            "reduced_social_motivation": (
                "Lower drive to initiate or sustain socially demanding communication."
            ),
            "social_withdrawal": (
                "Avoidance of social situations or reduced participation because interaction feels unrewarding "
                "or threatening."
            ),
            "conversational_breakdown": (
                "Frequent failures in reciprocal conversation, topic management, repair, or inference."
            ),
            "relationship_communication_impairment": (
                "Functional impairment in friendships, classroom/work participation, or broader social integration."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "polygenic_social_communication_liability",
                "target": "social_brain_network_dysconnectivity",
                "relation": "increases developmental vulnerability in large-scale social communication networks",
                "scd_change": "increased",
            },
            {
                "source": "polygenic_social_communication_liability",
                "target": "dopaminergic_social_reward_blunting",
                "relation": "can reduce social reward responsivity and social-learning engagement",
                "scd_change": "increased",
            },
            {
                "source": "polygenic_social_communication_liability",
                "target": "mental_state_inference_impairment",
                "relation": "raises baseline vulnerability in social-cognitive inference systems",
                "scd_change": "increased",
            },
            {
                "source": "repeated_social_failure_exclusion",
                "target": "dopaminergic_social_reward_blunting",
                "relation": "repeated rejection can reduce reward value of social engagement",
                "scd_change": "increased",
            },
            {
                "source": "repeated_social_failure_exclusion",
                "target": "social_threat_bias",
                "relation": "strengthens expectancy of negative social outcomes",
                "scd_change": "increased",
            },
            {
                "source": "repeated_social_failure_exclusion",
                "target": "reduced_social_learning_drive",
                "relation": "limits willingness to re-enter complex social-learning contexts",
                "scd_change": "increased",
            },
            {
                "source": "social_stress_anxiety_burden",
                "target": "endocannabinoid_stress_buffer_failure",
                "relation": "overloads stress-buffering mechanisms linked to anxiety and social avoidance",
                "scd_change": "increased",
            },
            {
                "source": "social_stress_anxiety_burden",
                "target": "social_threat_bias",
                "relation": "amplifies fear of negative evaluation and social vigilance",
                "scd_change": "increased",
            },
            {
                "source": "social_stress_anxiety_burden",
                "target": "amygdala_mpfc_regulatory_weakness",
                "relation": "weakens regulatory balance between limbic and medial prefrontal systems",
                "scd_change": "increased",
            },
            {
                "source": "pragmatic_language_demand_load",
                "target": "pragmatic_language_integration_failure",
                "relation": "raises the burden on socio-linguistic integration mechanisms",
                "scd_change": "increased",
            },
            {
                "source": "pragmatic_language_demand_load",
                "target": "executive_social_context_deficit",
                "relation": "increases pressure on context updating and conversational control",
                "scd_change": "increased",
            },
            {
                "source": "social_learning_enrichment",
                "target": "reduced_social_learning_drive",
                "relation": "supports repeated practice and preserves willingness to engage socially",
                "scd_change": "decreased",
            },
            {
                "source": "social_learning_enrichment",
                "target": "social_brain_network_dysconnectivity",
                "relation": "supports maturation and integration of social-cognitive skills",
                "scd_change": "decreased",
            },
            {
                "source": "communication_support_intervention",
                "target": "pragmatic_language_integration_failure",
                "relation": "improves explicit strategies for contextual and inferential communication",
                "scd_change": "decreased",
            },
            {
                "source": "communication_support_intervention",
                "target": "executive_social_context_deficit",
                "relation": "strengthens compensatory monitoring and repair strategies",
                "scd_change": "decreased",
            },
            {
                "source": "dopaminergic_social_reward_blunting",
                "target": "reduced_social_learning_drive",
                "relation": "reduces motivation to engage in complex social learning",
                "scd_change": "increased",
            },
            {
                "source": "endocannabinoid_stress_buffer_failure",
                "target": "social_threat_bias",
                "relation": "permits stronger stress and anxiety responses to social challenge",
                "scd_change": "increased",
            },
            {
                "source": "social_brain_network_dysconnectivity",
                "target": "mental_state_inference_impairment",
                "relation": "weakens coordinated social-cognitive processing across the social brain",
                "scd_change": "increased",
            },
            {
                "source": "social_brain_network_dysconnectivity",
                "target": "pragmatic_language_integration_failure",
                "relation": "impairs the integration of mentalizing with language processing",
                "scd_change": "increased",
            },
            {
                "source": "social_brain_network_dysconnectivity",
                "target": "executive_social_context_deficit",
                "relation": "reduces smooth coordination of cognitive and social domains",
                "scd_change": "increased",
            },
            {
                "source": "amygdala_mpfc_regulatory_weakness",
                "target": "social_threat_bias",
                "relation": "permits stronger unregulated responses to social threat",
                "scd_change": "increased",
            },
            {
                "source": "mental_state_inference_impairment",
                "target": "mpfc",
                "relation": "increases burden in medial prefrontal social-reasoning circuitry",
                "scd_change": "increased",
            },
            {
                "source": "mental_state_inference_impairment",
                "target": "tpj",
                "relation": "increases burden in temporoparietal perspective-taking circuitry",
                "scd_change": "increased",
            },
            {
                "source": "pragmatic_language_integration_failure",
                "target": "sts_proxy",
                "relation": "increases burden in temporal social-perception and integration circuitry",
                "scd_change": "increased",
            },
            {
                "source": "pragmatic_language_integration_failure",
                "target": "ifg_language_interface",
                "relation": "increases burden in frontal socio-linguistic formulation circuitry",
                "scd_change": "increased",
            },
            {
                "source": "executive_social_context_deficit",
                "target": "acc",
                "relation": "increases conflict-monitoring and context-control burden",
                "scd_change": "increased",
            },
            {
                "source": "social_threat_bias",
                "target": "amygdala",
                "relation": "raises limbic threat salience in social situations",
                "scd_change": "increased",
            },
            {
                "source": "social_threat_bias",
                "target": "insula",
                "relation": "raises interoceptive discomfort and social-anxiety-linked salience",
                "scd_change": "increased",
            },
            {
                "source": "dopaminergic_social_reward_blunting",
                "target": "ventral_striatum_proxy",
                "relation": "increases dysfunction burden in social reward circuitry",
                "scd_change": "increased",
            },
            {
                "source": "mental_state_inference_impairment",
                "target": "impaired_pragmatic_inference",
                "relation": "directly weakens inference of intentions and implied meaning",
                "scd_change": "increased",
            },
            {
                "source": "pragmatic_language_integration_failure",
                "target": "poor_context_sensitive_language_use",
                "relation": "impairs context-appropriate communication",
                "scd_change": "increased",
            },
            {
                "source": "social_threat_bias",
                "target": "fear_of_negative_evaluation",
                "relation": "increases anxious anticipation of judgment",
                "scd_change": "increased",
            },
            {
                "source": "reduced_social_learning_drive",
                "target": "reduced_social_motivation",
                "relation": "lowers the motivation to initiate and sustain social communication",
                "scd_change": "increased",
            },
            {
                "source": "fear_of_negative_evaluation",
                "target": "social_withdrawal",
                "relation": "promotes avoidance of socially demanding situations",
                "scd_change": "increased",
            },
            {
                "source": "poor_context_sensitive_language_use",
                "target": "conversational_breakdown",
                "relation": "increases failures of conversational fit and repair",
                "scd_change": "increased",
            },
            {
                "source": "impaired_pragmatic_inference",
                "target": "conversational_breakdown",
                "relation": "impairs interpretation of conversational subtext and turns",
                "scd_change": "increased",
            },
            {
                "source": "conversational_breakdown",
                "target": "relationship_communication_impairment",
                "relation": "drives broader social and functional impairment",
                "scd_change": "increased",
            },
            {
                "source": "social_withdrawal",
                "target": "relationship_communication_impairment",
                "relation": "reduces opportunities for participation and relationship maintenance",
                "scd_change": "increased",
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
        # Modern siibra examples search from the parcellation itself.
        try:
            matches = self.parcellation.find(query, filter_children=False)
            return list(matches) if matches is not None else []
        except TypeError:
            try:
                matches = self.parcellation.find(query)
                return list(matches) if matches is not None else []
            except Exception:
                pass
        except Exception:
            pass

        # Fallback for older environments.
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
        generic_penalty = 1 if name in {
            "amygdala",
            "insula",
            "anterior cingulate",
            "medial prefrontal cortex",
            "temporoparietal junction",
            "superior temporal sulcus",
            "superior temporal",
            "frontal pole",
            "ventral striatum",
        } else 0
        parent_penalty = 1 if any(k in name for k in {"cortex", "lobe", "gyrus"}) and "area " not in name else 0
        return (left_bonus, right_penalty, generic_penalty, parent_penalty)

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
        volume = getattr(main, "volume", None)
        volume_mm3 = float(volume) if volume is not None else None
        return centroid_xyz, volume_mm3

    def _regional_volume_for_feature_matching(self, region: Any):
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(space=self.space, maptype="labelled")
            except Exception:
                return None

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            mask = self._regional_volume_for_feature_matching(region)
            if mask is not None:
                feats = self._safe_features_any(mask, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()

        for feat in feats:
            try:
                df = feat.data.copy().reset_index()
                if "index" in df.columns and "receptor" not in df.columns:
                    df = df.rename(columns={"index": "receptor"})
                return df.reset_index(drop=True)
            except Exception:
                continue
        return pd.DataFrame()

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            mask = self._regional_volume_for_feature_matching(region)
            if mask is not None:
                feats = self._safe_features_any(mask, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()

        for feat in feats:
            try:
                df = feat.data.copy()
            except Exception:
                continue
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

        return pd.DataFrame()

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (f for f in feats if str(getattr(f, "cohort", "")).upper() == str(self.connectivity_cohort).upper()),
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
        exact = [x for x in labels if self._name_of(x) == region.name]
        if exact:
            return exact[0]
        rn = region.name.lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
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
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return pairwise connectivity values among the scaffold's resolved circuit nodes.
        Missing or unresolved regions are skipped.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        matched_rows: Dict[str, Any] = {}
        matched_cols: Dict[str, Any] = {}

        for key, region in self.region_objects.items():
            row_label = self._match_region_label(list(matrix.index), region)
            col_label = self._match_region_label(list(matrix.columns), region)
            if row_label is not None:
                matched_rows[key] = row_label
            if col_label is not None:
                matched_cols[key] = col_label

        keys = list(self.region_objects.keys())
        for src in keys:
            for dst in keys:
                if src == dst:
                    continue
                row_label = matched_rows.get(src)
                col_label = matched_cols.get(dst)
                if row_label is None or col_label is None:
                    continue
                try:
                    value = matrix.loc[row_label, col_label]
                except Exception:
                    continue
                rows.append(
                    {
                        "source_key": src,
                        "source_region": self.region_objects[src].name,
                        "target_key": dst,
                        "target_region": self.region_objects[dst].name,
                        "value": float(value),
                    }
                )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_SCD_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
        """
        Build the atlas-grounded graph and collect multimodal summaries.
        """
        nodes: List[Dict[str, Any]] = []
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
        self._connectivity_matrix = None

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
            desc = self.region_descriptions.get(key, "Atlas-backed circuit node")

            if region is None:
                proxy_only = key in self.proxy_only_nodes
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()

                if not proxy_only:
                    warnings.warn(f"Could not resolve a region for node '{key}'")

                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region",
                        "description": (
                            f"{desc} Intentional proxy without a stable Julich anchor in this environment."
                            if proxy_only
                            else f"{desc} Unresolved in this siibra environment; keep as an explicit proxy."
                        ),
                        "atlas_region": None,
                        "region_identifier": None,
                        "centroid_mni": None,
                        "volume_mm3": None,
                        "feature_summary": "proxy_only_unresolved" if proxy_only else "unresolved",
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
        polygenic_social_communication_liability: float = 0.55,
        repeated_social_failure_exclusion: float = 0.50,
        social_stress_anxiety_burden: float = 0.55,
        pragmatic_language_demand_load: float = 0.60,
        social_learning_enrichment: float = 0.35,
        communication_support_intervention: float = 0.35,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator for the scaffold.

        Inputs are clipped to 0..1. Protective variables subtract from
        dysregulation. The calculation is acyclic:
            inputs -> latent biology -> regional burden -> symptoms -> phenotypes
        """
        inputs = {
            "polygenic_social_communication_liability": self._clip01(polygenic_social_communication_liability),
            "repeated_social_failure_exclusion": self._clip01(repeated_social_failure_exclusion),
            "social_stress_anxiety_burden": self._clip01(social_stress_anxiety_burden),
            "pragmatic_language_demand_load": self._clip01(pragmatic_language_demand_load),
            "social_learning_enrichment": self._clip01(social_learning_enrichment),
            "communication_support_intervention": self._clip01(communication_support_intervention),
        }

        latents = {
            "dopaminergic_social_reward_blunting": self._clip01(
                0.34 * inputs["polygenic_social_communication_liability"]
                + 0.30 * inputs["repeated_social_failure_exclusion"]
                + 0.12 * inputs["social_stress_anxiety_burden"]
                - 0.18 * inputs["social_learning_enrichment"]
            ),
            "endocannabinoid_stress_buffer_failure": self._clip01(
                0.42 * inputs["social_stress_anxiety_burden"]
                + 0.18 * inputs["repeated_social_failure_exclusion"]
                + 0.10 * inputs["polygenic_social_communication_liability"]
                - 0.22 * inputs["social_learning_enrichment"]
                - 0.08 * inputs["communication_support_intervention"]
            ),
            "social_brain_network_dysconnectivity": self._clip01(
                0.36 * inputs["polygenic_social_communication_liability"]
                + 0.20 * inputs["repeated_social_failure_exclusion"]
                + 0.16 * inputs["social_stress_anxiety_burden"]
                + 0.08 * inputs["pragmatic_language_demand_load"]
                - 0.18 * inputs["social_learning_enrichment"]
                - 0.12 * inputs["communication_support_intervention"]
            ),
        }

        latents["amygdala_mpfc_regulatory_weakness"] = self._clip01(
            0.34 * latents["endocannabinoid_stress_buffer_failure"]
            + 0.26 * inputs["social_stress_anxiety_burden"]
            + 0.18 * latents["social_brain_network_dysconnectivity"]
            + 0.08 * inputs["repeated_social_failure_exclusion"]
            - 0.18 * inputs["communication_support_intervention"]
        )
        latents["social_threat_bias"] = self._clip01(
            0.34 * inputs["repeated_social_failure_exclusion"]
            + 0.28 * inputs["social_stress_anxiety_burden"]
            + 0.22 * latents["amygdala_mpfc_regulatory_weakness"]
            + 0.12 * latents["endocannabinoid_stress_buffer_failure"]
            - 0.14 * inputs["social_learning_enrichment"]
        )
        latents["mental_state_inference_impairment"] = self._clip01(
            0.34 * inputs["polygenic_social_communication_liability"]
            + 0.30 * latents["social_brain_network_dysconnectivity"]
            + 0.12 * inputs["pragmatic_language_demand_load"]
            + 0.10 * inputs["repeated_social_failure_exclusion"]
            - 0.18 * inputs["communication_support_intervention"]
            - 0.10 * inputs["social_learning_enrichment"]
        )
        latents["pragmatic_language_integration_failure"] = self._clip01(
            0.34 * latents["social_brain_network_dysconnectivity"]
            + 0.26 * inputs["pragmatic_language_demand_load"]
            + 0.18 * latents["mental_state_inference_impairment"]
            + 0.08 * inputs["social_stress_anxiety_burden"]
            - 0.20 * inputs["communication_support_intervention"]
            - 0.08 * inputs["social_learning_enrichment"]
        )
        latents["reduced_social_learning_drive"] = self._clip01(
            0.36 * latents["dopaminergic_social_reward_blunting"]
            + 0.24 * inputs["repeated_social_failure_exclusion"]
            + 0.14 * latents["social_threat_bias"]
            + 0.08 * inputs["social_stress_anxiety_burden"]
            - 0.22 * inputs["social_learning_enrichment"]
        )
        latents["executive_social_context_deficit"] = self._clip01(
            0.28 * latents["social_brain_network_dysconnectivity"]
            + 0.24 * inputs["pragmatic_language_demand_load"]
            + 0.18 * latents["pragmatic_language_integration_failure"]
            + 0.12 * latents["social_threat_bias"]
            - 0.20 * inputs["communication_support_intervention"]
        )

        regional_state = {
            "mpfc": self._clip01(
                0.42 * latents["mental_state_inference_impairment"]
                + 0.24 * latents["amygdala_mpfc_regulatory_weakness"]
                + 0.16 * latents["executive_social_context_deficit"]
                + 0.10 * latents["social_brain_network_dysconnectivity"]
                - 0.14 * inputs["communication_support_intervention"]
            ),
            "tpj": self._clip01(
                0.42 * latents["mental_state_inference_impairment"]
                + 0.26 * latents["social_brain_network_dysconnectivity"]
                + 0.18 * latents["pragmatic_language_integration_failure"]
                - 0.10 * inputs["social_learning_enrichment"]
            ),
            "sts_proxy": self._clip01(
                0.38 * latents["pragmatic_language_integration_failure"]
                + 0.24 * latents["social_brain_network_dysconnectivity"]
                + 0.16 * inputs["pragmatic_language_demand_load"]
                + 0.08 * latents["mental_state_inference_impairment"]
                - 0.10 * inputs["communication_support_intervention"]
            ),
            "acc": self._clip01(
                0.38 * latents["executive_social_context_deficit"]
                + 0.28 * latents["social_threat_bias"]
                + 0.14 * inputs["social_stress_anxiety_burden"]
                - 0.12 * inputs["communication_support_intervention"]
            ),
            "amygdala": self._clip01(
                0.50 * latents["social_threat_bias"]
                + 0.22 * latents["amygdala_mpfc_regulatory_weakness"]
                + 0.12 * inputs["repeated_social_failure_exclusion"]
                - 0.10 * inputs["social_learning_enrichment"]
            ),
            "insula": self._clip01(
                0.40 * latents["social_threat_bias"]
                + 0.24 * latents["endocannabinoid_stress_buffer_failure"]
                + 0.12 * inputs["social_stress_anxiety_burden"]
                - 0.10 * inputs["social_learning_enrichment"]
            ),
            "ifg_language_interface": self._clip01(
                0.40 * latents["pragmatic_language_integration_failure"]
                + 0.22 * latents["executive_social_context_deficit"]
                + 0.14 * inputs["pragmatic_language_demand_load"]
                - 0.16 * inputs["communication_support_intervention"]
            ),
            "ventral_striatum_proxy": self._clip01(
                0.46 * latents["dopaminergic_social_reward_blunting"]
                + 0.24 * latents["reduced_social_learning_drive"]
                + 0.12 * inputs["repeated_social_failure_exclusion"]
                - 0.14 * inputs["social_learning_enrichment"]
            ),
        }

        symptoms = {
            "impaired_pragmatic_inference": self._clip01(
                0.36 * latents["mental_state_inference_impairment"]
                + 0.24 * regional_state["tpj"]
                + 0.20 * regional_state["mpfc"]
                + 0.10 * latents["social_brain_network_dysconnectivity"]
                - 0.10 * inputs["communication_support_intervention"]
            ),
            "poor_context_sensitive_language_use": self._clip01(
                0.34 * latents["pragmatic_language_integration_failure"]
                + 0.22 * regional_state["ifg_language_interface"]
                + 0.18 * regional_state["sts_proxy"]
                + 0.12 * latents["executive_social_context_deficit"]
                + 0.08 * inputs["pragmatic_language_demand_load"]
                - 0.12 * inputs["communication_support_intervention"]
            ),
            "fear_of_negative_evaluation": self._clip01(
                0.34 * latents["social_threat_bias"]
                + 0.22 * regional_state["amygdala"]
                + 0.18 * regional_state["insula"]
                + 0.14 * inputs["repeated_social_failure_exclusion"]
                + 0.06 * inputs["social_stress_anxiety_burden"]
            ),
            "reduced_social_motivation": self._clip01(
                0.34 * latents["reduced_social_learning_drive"]
                + 0.26 * latents["dopaminergic_social_reward_blunting"]
                + 0.18 * regional_state["ventral_striatum_proxy"]
                + 0.10 * inputs["repeated_social_failure_exclusion"]
                - 0.12 * inputs["social_learning_enrichment"]
            ),
        }

        symptoms["social_withdrawal"] = self._clip01(
            0.34 * symptoms["fear_of_negative_evaluation"]
            + 0.28 * symptoms["reduced_social_motivation"]
            + 0.14 * latents["endocannabinoid_stress_buffer_failure"]
            + 0.10 * inputs["repeated_social_failure_exclusion"]
            - 0.10 * inputs["social_learning_enrichment"]
        )
        symptoms["conversational_breakdown"] = self._clip01(
            0.34 * symptoms["poor_context_sensitive_language_use"]
            + 0.28 * symptoms["impaired_pragmatic_inference"]
            + 0.16 * latents["executive_social_context_deficit"]
            + 0.10 * inputs["pragmatic_language_demand_load"]
            - 0.08 * inputs["communication_support_intervention"]
        )
        symptoms["relationship_communication_impairment"] = self._clip01(
            0.34 * symptoms["conversational_breakdown"]
            + 0.28 * symptoms["social_withdrawal"]
            + 0.18 * symptoms["fear_of_negative_evaluation"]
            + 0.08 * symptoms["reduced_social_motivation"]
        )

        phenotypes = {
            "scd_core_pragmatic_profile": self._clip01(
                (
                    symptoms["impaired_pragmatic_inference"]
                    + symptoms["poor_context_sensitive_language_use"]
                    + symptoms["conversational_breakdown"]
                    + symptoms["relationship_communication_impairment"]
                ) / 4.0
            ),
            "socially_anxious_withdrawn_profile": self._clip01(
                (
                    symptoms["fear_of_negative_evaluation"]
                    + symptoms["social_withdrawal"]
                    + regional_state["amygdala"]
                    + regional_state["insula"]
                ) / 4.0
            ),
            "low_social_reward_profile": self._clip01(
                (
                    symptoms["reduced_social_motivation"]
                    + latents["dopaminergic_social_reward_blunting"]
                    + latents["reduced_social_learning_drive"]
                    + regional_state["ventral_striatum_proxy"]
                ) / 4.0
            ),
            "socio_language_integration_profile": self._clip01(
                (
                    latents["pragmatic_language_integration_failure"]
                    + regional_state["tpj"]
                    + regional_state["sts_proxy"]
                    + regional_state["ifg_language_interface"]
                ) / 4.0
            ),
        }

        return {
            "inputs": pd.Series(inputs, name="inputs"),
            "latents": pd.Series(latents, name="latents"),
            "regional_state": pd.Series(regional_state, name="regional_state"),
            "symptoms": pd.Series(symptoms, name="symptoms"),
            "phenotypes": pd.Series(phenotypes, name="phenotypes"),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Assign an MNI coordinate to Julich regions using a statistical map.
        """
        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception:
                with siibra.QUIET:
                    self._pmap = self.atlas.get_map(
                        parcellation=self.parcellation,
                        space=self.atlas.get_space(self.assignment_space),
                        maptype="statistical",
                    )

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
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
        """
        Return a regional mask object for a resolved region node.

        Call `.fetch()` on the returned object to obtain the underlying image.
        Returns None for unresolved nodes.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(space=self.space, maptype="labelled")
            except Exception:
                return None


if __name__ == "__main__":
    if siibra is None:
        print(
            "siibra is not installed in this environment. Install siibra, then rerun "
            "this script to build the atlas-grounded Social (Pragmatic) Communication "
            "Disorder scaffold."
        )
        raise SystemExit(0)

    model = SocialPragmaticCommunicationDisorderModel()
    built = model.build(connectivity_rows=10)

    print("\n=== NODES ===")
    print(built["nodes"][["key", "node_type", "label", "atlas_region", "feature_summary"]].to_string(index=False))

    print("\n=== EDGES ===")
    print(built["edges"][["source", "target", "relation", "scd_change"]].to_string(index=False))

    print("\n=== REGION RESOLUTION ===")
    if model.region_objects:
        for key, region in model.region_objects.items():
            print(f"{key}: {region.name}")
    else:
        print("No regions resolved in this environment.")

    for node_key in ["mpfc", "tpj", "amygdala", "insula", "ifg_language_interface"]:
        receptor_df = built["receptors"].get(node_key, pd.DataFrame())
        gene_df = built["genes"].get(node_key, pd.DataFrame())
        conn_df = built["connectivity_profiles"].get(node_key, pd.DataFrame())

        print(f"\n=== FEATURES: {node_key} ===")
        print("Receptors:")
        print(receptor_df.head().to_string(index=False) if not receptor_df.empty else "<none>")
        print("Genes:")
        print(gene_df.head().to_string(index=False) if not gene_df.empty else "<none>")
        print("Connectivity:")
        print(conn_df.head().to_string(index=False) if not conn_df.empty else "<none>")

    circuit_df = built["circuit_connectivity"]
    print("\n=== CIRCUIT CONNECTIVITY ===")
    print(circuit_df.head(12).to_string(index=False) if not circuit_df.empty else "<none>")

    sim = model.simulate(
        polygenic_social_communication_liability=0.62,
        repeated_social_failure_exclusion=0.58,
        social_stress_anxiety_burden=0.64,
        pragmatic_language_demand_load=0.70,
        social_learning_enrichment=0.30,
        communication_support_intervention=0.36,
    )

    print("\n=== SIMULATION: INPUTS ===")
    print(sim["inputs"].round(3).to_string())
    print("\n=== SIMULATION: LATENTS ===")
    print(sim["latents"].round(3).sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: REGIONAL STATE ===")
    print(sim["regional_state"].round(3).sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: SYMPTOMS ===")
    print(sim["symptoms"].round(3).sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: PHENOTYPES ===")
    print(sim["phenotypes"].round(3).sort_values(ascending=False).to_string())

    # Example coordinate assignment for future use:
    # print(model.assign_mni_point((-48, -58, 22)).head())
    # mask = model.region_mask("tpj")
    # if mask is not None:
    #     nii = mask.fetch()
