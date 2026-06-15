from __future__ import annotations

"""
Paranoid Personality Disorder siibra scaffold.

This script translates a chapter on Paranoid Personality Disorder (PPD) into a
transparent, atlas-grounded research scaffold.

The source chapter is mechanistically richer than it is anatomically precise, so
this scaffold is intentionally conservative:

- The chapter directly names medial prefrontal / orbitofrontal cortex,
  amygdala, anterior cingulate cortex, and hippocampus.
- Monoaminergic chemistry, HPA-axis stress biology, and epigenetic programming
  remain latent mechanisms rather than being over-forced into single parcels.
- The medial prefrontal cortex is modeled as a proxy because the exact Julich
  label can vary across siibra versions and subregional choices.
- Electrophysiological claims about P300 and ERN are represented as latent
  marker-like processes rather than as diagnostic outputs.

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
    "SLC6A4",   # serotonergic signaling / affective regulation
    "HTR1A",    # serotonin receptor, anxiety / inhibition proxy
    "HTR2A",    # serotonin receptor, salience / appraisal proxy
    "DRD2",     # dopamine salience and reinforcement signaling
    "COMT",     # prefrontal dopamine regulation
    "SLC6A3",   # dopamine transporter
    "SLC6A2",   # norepinephrine transporter
    "ADRA2A",   # noradrenergic autoreceptor / arousal regulation proxy
    "MAOA",     # monoamine metabolism / aggression-related proxy
    "NR3C1",    # glucocorticoid receptor / stress programming
    "FKBP5",    # stress-sensitivity / HPA regulation
    "CRHR1",    # stress response / CRH signaling
    "BDNF",     # plasticity / adversity sensitivity
    "OXTR",     # interpersonal trust / social salience proxy
]


class ParanoidPersonalityDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Paranoid Personality Disorder.

    Conceptual reading of the chapter:
    - Core symptoms arise from a trait-like bias in threat perception, social
      cognition, and interpersonal trust.
    - Dopamine, serotonin, and norepinephrine are treated as latent contributors
      to suspiciousness, impulsive hostility, and abnormal reward / threat
      appraisal.
    - Early adversity, dysfunctional family environment, intergenerational
      violence exposure, and prenatal stress are modeled as major upstream
      drivers because the chapter emphasizes gene-environment interaction and
      epigenetic programming.
    - Medial prefrontal / orbitofrontal dysfunction, impaired top-down control
      over amygdala reactivity, anterior cingulate hypermonitoring, and
      hippocampal stress sensitization are the main circuit claims.
    - The chapter's ERP discussion is represented through threat-cue salience
      bias (P300-like) and social error hypermonitoring (ERN-like).

    Anatomical caution:
    The chapter does not provide direct disorder-specific imaging maps or a full
    parcel list, so atlas nodes are intentionally modest and can be retuned with
    suggest_regions().
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

        # Directly named or strongly implied chapter anchors.
        # - mpfc_proxy because the chapter specifies medial PFC function but not
        #   a unique cytoarchitectonic parcel.
        # - acc includes pACC / sACC alternatives because the ERP claim is ACC-
        #   centered but not more precise than that.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "mpfc_proxy": [
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "Area 10 left",
                "medial prefrontal cortex",
                "medial frontal cortex",
                "prefrontal cortex",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Area Fo2 (OFC) left",
                "Area Fo1 (OFC) left",
                "orbitofrontal cortex",
                "orbitofrontal",
                "ofc",
            ],
            "acc": [
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "Area 33 (ACC) left",
                "Area s24 (sACC) left",
                "anterior cingulate cortex",
                "anterior cingulate",
                "acc",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "CA2 (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "DG (Hippocampus) left",
                "hippocampus left",
                "hippocampus",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_temperamental_liability": (
                "Heritable temperament liability involving impulsivity, low agreeableness, "
                "aggressive reactivity, and suspicious interpersonal style"
            ),
            "early_life_trauma": (
                "Early abuse or adversity that increases later emotional dysregulation and threat sensitivity"
            ),
            "intergenerational_violence_exposure": (
                "Exposure to violent or abusive family transmission patterns that shape hostile expectations"
            ),
            "prenatal_maternal_stress": (
                "Prenatal stress exposure contributing to fetal stress programming and later vulnerability"
            ),
            "dysfunctional_family_environment": (
                "Shared invalidating or hostile family environment that can reinforce distrust and antagonism"
            ),
            "chronic_interpersonal_stress": (
                "Ongoing rejection, conflict, ambiguity, or social threat that sustains hypervigilance"
            ),
            "interpersonal_safety_support": (
                "Protective exposure to reliable, non-malevolent, stabilizing interpersonal contexts"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "monoaminergic_threat_bias": (
                "Combined dopamine / serotonin / norepinephrine dysregulation biasing salience, arousal, and threat appraisal"
            ),
            "epigenetic_stress_programming": (
                "Stable stress-linked gene-expression programming shaped by trauma and prenatal adversity"
            ),
            "hpa_axis_sensitization": (
                "Stress-system hyperreactivity plausibly related to glucocorticoid receptor programming"
            ),
            "social_trust_calibration_failure": (
                "Failure to calibrate interpersonal trust, with a bias toward expecting exploitation or betrayal"
            ),
            "mentalizing_distortion": (
                "Impaired inference of others' thoughts and intentions, favoring malevolent interpretations"
            ),
            "frontolimbic_regulatory_failure": (
                "Reduced medial / orbitofrontal top-down control over fear, anger, and impulsive reactions"
            ),
            "threat_cue_salience_bias": (
                "Enhanced attention allocation to angry or ambiguous social cues, corresponding to a P300-like bias"
            ),
            "social_error_hypermonitoring": (
                "Excessive monitoring of social mistakes or evaluation, corresponding to an ERN-like bias"
            ),
            "hostile_attribution_bias": (
                "Tendency to interpret ambiguous interpersonal events as attacks, betrayal, or exploitation"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "distrust_suspiciousness": (
                "Pervasive suspiciousness and expectation of malevolent intent"
            ),
            "malevolent_intent_inference": (
                "Interpretation of others' motives as hostile, deceptive, or exploitative"
            ),
            "loyalty_doubts_and_grudges": (
                "Persistent doubts about loyalty and tendency to hold grudges"
            ),
            "anger_hostility_outbursts": (
                "Poorly regulated anger, hostile reactivity, and readiness to counterattack"
            ),
            "social_avoidance_anxiety": (
                "Avoidance and anxiety driven by fear of negative evaluation, betrayal, or social threat"
            ),
            "antagonistic_aggression_risk": (
                "Risk of antagonistic or aggressive interpersonal behavior"
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_temperamental_liability",
                "target": "monoaminergic_threat_bias",
                "relation": "loads inherited monoaminergic vulnerability affecting threat, reward, and impulse regulation",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "genetic_temperamental_liability",
                "target": "social_trust_calibration_failure",
                "relation": "contributes trait-level suspiciousness and antagonistic interpersonal style",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "early_life_trauma",
                "target": "epigenetic_stress_programming",
                "relation": "supports long-lasting stress-linked gene-expression changes",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "prenatal_maternal_stress",
                "target": "epigenetic_stress_programming",
                "relation": "contributes fetal programming of stress reactivity",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "intergenerational_violence_exposure",
                "target": "epigenetic_stress_programming",
                "relation": "adds adversity-linked programming pressure through recurrent abuse environments",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "dysfunctional_family_environment",
                "target": "social_trust_calibration_failure",
                "relation": "teaches hostile expectations and unstable trust calibration",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "intergenerational_violence_exposure",
                "target": "social_trust_calibration_failure",
                "relation": "reinforces expectations of harm, deceit, and betrayal",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "epigenetic_stress_programming",
                "target": "hpa_axis_sensitization",
                "relation": "alters stress responsivity through stable regulatory changes",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "early_life_trauma",
                "target": "hpa_axis_sensitization",
                "relation": "directly increases long-term stress-system reactivity",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "chronic_interpersonal_stress",
                "target": "hpa_axis_sensitization",
                "relation": "maintains chronic fear and anger responsivity",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "genetic_temperamental_liability",
                "target": "frontolimbic_regulatory_failure",
                "relation": "contributes trait vulnerability in emotion regulation and impulsive hostility",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "monoaminergic_threat_bias",
                "target": "frontolimbic_regulatory_failure",
                "relation": "weakens stable top-down regulation of affective salience and impulse control",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "hpa_axis_sensitization",
                "target": "frontolimbic_regulatory_failure",
                "relation": "sustained stress burden undermines prefrontal regulation of fear and anger",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "social_trust_calibration_failure",
                "target": "mentalizing_distortion",
                "relation": "biases inference of others' intentions toward betrayal and exploitation",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "frontolimbic_regulatory_failure",
                "target": "mentalizing_distortion",
                "relation": "reduces accurate social-cognitive interpretation of ambiguous actions",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "monoaminergic_threat_bias",
                "target": "threat_cue_salience_bias",
                "relation": "amplifies attention to threatening or ambiguous social cues",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "hpa_axis_sensitization",
                "target": "threat_cue_salience_bias",
                "relation": "heightens vigilance for threat-laden stimuli",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "chronic_interpersonal_stress",
                "target": "social_error_hypermonitoring",
                "relation": "increases fear of social missteps and negative evaluation",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "hpa_axis_sensitization",
                "target": "social_error_hypermonitoring",
                "relation": "increases anxious error monitoring in social contexts",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "mentalizing_distortion",
                "target": "hostile_attribution_bias",
                "relation": "promotes malevolent interpretations of ambiguous interpersonal acts",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "threat_cue_salience_bias",
                "target": "hostile_attribution_bias",
                "relation": "selectively amplifies evidence consistent with threat and hostility",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "social_trust_calibration_failure",
                "target": "hostile_attribution_bias",
                "relation": "predisposes ambiguous interactions to be read as malevolent",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "interpersonal_safety_support",
                "target": "social_trust_calibration_failure",
                "relation": "buffers hostile default expectations and improves trust calibration",
                "paranoid_pd_change": "decreased",
            },
            {
                "source": "interpersonal_safety_support",
                "target": "hpa_axis_sensitization",
                "relation": "reduces sustained stress responsivity through safer social context",
                "paranoid_pd_change": "decreased",
            },
            {
                "source": "interpersonal_safety_support",
                "target": "frontolimbic_regulatory_failure",
                "relation": "supports better top-down regulation of fear and anger",
                "paranoid_pd_change": "decreased",
            },
            {
                "source": "threat_cue_salience_bias",
                "target": "amygdala",
                "relation": "loads threat-reactive amygdala state",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "frontolimbic_regulatory_failure",
                "target": "mpfc_proxy",
                "relation": "maps impaired social-cognitive regulation onto medial prefrontal control systems",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "mentalizing_distortion",
                "target": "mpfc_proxy",
                "relation": "captures biased theory-of-mind and social inference processing",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "frontolimbic_regulatory_failure",
                "target": "ofc",
                "relation": "maps reduced emotional decision control and impulsive modulation onto OFC state",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "social_error_hypermonitoring",
                "target": "acc",
                "relation": "maps excessive social error monitoring onto anterior cingulate state",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "hpa_axis_sensitization",
                "target": "hippocampus",
                "relation": "maps stress-linked glucocorticoid and mnemonic burden onto hippocampal state",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "distrust_suspiciousness",
                "relation": "heightened amygdala threat reactivity supports chronic suspiciousness",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "hostile_attribution_bias",
                "target": "distrust_suspiciousness",
                "relation": "sustains a pervasive expectation of malevolence",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "mentalizing_distortion",
                "target": "malevolent_intent_inference",
                "relation": "causes others' intentions to be inferred inaccurately as hostile",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "mpfc_proxy",
                "target": "malevolent_intent_inference",
                "relation": "medial prefrontal dysfunction impairs accurate interpretation of social intent",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "distrust_suspiciousness",
                "target": "loyalty_doubts_and_grudges",
                "relation": "persistent suspiciousness fosters loyalty doubts and grudge holding",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "hostile_attribution_bias",
                "target": "loyalty_doubts_and_grudges",
                "relation": "biases memory and interpretation of interpersonal transgressions",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "ofc",
                "target": "anger_hostility_outbursts",
                "relation": "orbitofrontal dysregulation reduces modulation of hostile reactions",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "anger_hostility_outbursts",
                "relation": "fear and anger reactivity promote hostile outbursts",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "acc",
                "target": "social_avoidance_anxiety",
                "relation": "anterior cingulate hypermonitoring promotes anxious social avoidance",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "social_error_hypermonitoring",
                "target": "social_avoidance_anxiety",
                "relation": "fear of negative evaluation increases interpersonal avoidance",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "anger_hostility_outbursts",
                "target": "antagonistic_aggression_risk",
                "relation": "poorly regulated hostility can escalate into antagonistic aggression",
                "paranoid_pd_change": "increased",
            },
            {
                "source": "malevolent_intent_inference",
                "target": "antagonistic_aggression_risk",
                "relation": "perceived attack or betrayal can motivate counteraggression",
                "paranoid_pd_change": "increased",
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
            "amygdala",
            "hippocampus",
            "anterior cingulate cortex",
            "orbitofrontal cortex",
            "prefrontal cortex",
            "medial prefrontal cortex",
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
        genetic_temperamental_liability: float = 0.58,
        early_life_trauma: float = 0.52,
        intergenerational_violence_exposure: float = 0.40,
        prenatal_maternal_stress: float = 0.30,
        dysfunctional_family_environment: float = 0.48,
        chronic_interpersonal_stress: float = 0.54,
        interpersonal_safety_support: float = 0.28,
    ) -> Dict[str, pd.Series]:
        """
        Transparent one-pass normalized simulator.

        Directionality follows the chapter's mechanistic reading:
        adversity and trait liability -> latent biology -> regional state ->
        symptoms -> phenotype summaries.

        All values are clipped to [0, 1].
        """

        inputs = pd.Series(
            {
                "genetic_temperamental_liability": self._clip01(genetic_temperamental_liability),
                "early_life_trauma": self._clip01(early_life_trauma),
                "intergenerational_violence_exposure": self._clip01(intergenerational_violence_exposure),
                "prenatal_maternal_stress": self._clip01(prenatal_maternal_stress),
                "dysfunctional_family_environment": self._clip01(dysfunctional_family_environment),
                "chronic_interpersonal_stress": self._clip01(chronic_interpersonal_stress),
                "interpersonal_safety_support": self._clip01(interpersonal_safety_support),
            },
            name="value",
        )

        latents = pd.Series(dtype=float, name="value")
        latents["monoaminergic_threat_bias"] = self._clip01(
            0.34 * inputs["genetic_temperamental_liability"]
            + 0.18 * inputs["chronic_interpersonal_stress"]
            + 0.16 * inputs["early_life_trauma"]
            + 0.12 * inputs["dysfunctional_family_environment"]
            + 0.08 * inputs["intergenerational_violence_exposure"]
            - 0.14 * inputs["interpersonal_safety_support"]
        )
        latents["epigenetic_stress_programming"] = self._clip01(
            0.34 * inputs["early_life_trauma"]
            + 0.24 * inputs["prenatal_maternal_stress"]
            + 0.20 * inputs["intergenerational_violence_exposure"]
            + 0.12 * inputs["dysfunctional_family_environment"]
            + 0.06 * inputs["chronic_interpersonal_stress"]
            - 0.12 * inputs["interpersonal_safety_support"]
        )
        latents["hpa_axis_sensitization"] = self._clip01(
            0.36 * latents["epigenetic_stress_programming"]
            + 0.24 * inputs["early_life_trauma"]
            + 0.16 * inputs["chronic_interpersonal_stress"]
            + 0.12 * inputs["prenatal_maternal_stress"]
            + 0.08 * inputs["intergenerational_violence_exposure"]
            - 0.16 * inputs["interpersonal_safety_support"]
        )
        latents["social_trust_calibration_failure"] = self._clip01(
            0.28 * inputs["dysfunctional_family_environment"]
            + 0.24 * inputs["intergenerational_violence_exposure"]
            + 0.20 * inputs["genetic_temperamental_liability"]
            + 0.16 * inputs["early_life_trauma"]
            + 0.10 * inputs["chronic_interpersonal_stress"]
            - 0.20 * inputs["interpersonal_safety_support"]
        )
        latents["frontolimbic_regulatory_failure"] = self._clip01(
            0.30 * latents["monoaminergic_threat_bias"]
            + 0.24 * latents["hpa_axis_sensitization"]
            + 0.18 * inputs["genetic_temperamental_liability"]
            + 0.14 * inputs["early_life_trauma"]
            + 0.08 * inputs["chronic_interpersonal_stress"]
            - 0.20 * inputs["interpersonal_safety_support"]
        )
        latents["mentalizing_distortion"] = self._clip01(
            0.34 * latents["frontolimbic_regulatory_failure"]
            + 0.24 * latents["social_trust_calibration_failure"]
            + 0.14 * inputs["genetic_temperamental_liability"]
            + 0.12 * inputs["dysfunctional_family_environment"]
            + 0.10 * inputs["chronic_interpersonal_stress"]
            - 0.14 * inputs["interpersonal_safety_support"]
        )
        latents["threat_cue_salience_bias"] = self._clip01(
            0.42 * latents["monoaminergic_threat_bias"]
            + 0.22 * latents["hpa_axis_sensitization"]
            + 0.16 * inputs["chronic_interpersonal_stress"]
            + 0.12 * latents["frontolimbic_regulatory_failure"]
            - 0.10 * inputs["interpersonal_safety_support"]
        )
        latents["social_error_hypermonitoring"] = self._clip01(
            0.32 * latents["hpa_axis_sensitization"]
            + 0.26 * inputs["chronic_interpersonal_stress"]
            + 0.18 * latents["frontolimbic_regulatory_failure"]
            + 0.12 * latents["mentalizing_distortion"]
            + 0.06 * latents["social_trust_calibration_failure"]
            - 0.12 * inputs["interpersonal_safety_support"]
        )
        latents["hostile_attribution_bias"] = self._clip01(
            0.30 * latents["mentalizing_distortion"]
            + 0.24 * latents["social_trust_calibration_failure"]
            + 0.20 * latents["threat_cue_salience_bias"]
            + 0.12 * latents["frontolimbic_regulatory_failure"]
            + 0.08 * inputs["dysfunctional_family_environment"]
            - 0.12 * inputs["interpersonal_safety_support"]
        )

        regional_state = pd.Series(dtype=float, name="value")
        regional_state["amygdala"] = self._clip01(
            0.40 * latents["threat_cue_salience_bias"]
            + 0.24 * latents["hpa_axis_sensitization"]
            + 0.18 * latents["monoaminergic_threat_bias"]
            + 0.10 * inputs["early_life_trauma"]
            - 0.12 * inputs["interpersonal_safety_support"]
        )
        regional_state["mpfc_proxy"] = self._clip01(
            0.38 * latents["frontolimbic_regulatory_failure"]
            + 0.28 * latents["mentalizing_distortion"]
            + 0.12 * inputs["genetic_temperamental_liability"]
            + 0.10 * inputs["chronic_interpersonal_stress"]
            - 0.18 * inputs["interpersonal_safety_support"]
        )
        regional_state["ofc"] = self._clip01(
            0.42 * latents["frontolimbic_regulatory_failure"]
            + 0.20 * latents["monoaminergic_threat_bias"]
            + 0.14 * inputs["dysfunctional_family_environment"]
            + 0.10 * inputs["chronic_interpersonal_stress"]
            + 0.06 * inputs["genetic_temperamental_liability"]
            - 0.16 * inputs["interpersonal_safety_support"]
        )
        regional_state["acc"] = self._clip01(
            0.44 * latents["social_error_hypermonitoring"]
            + 0.22 * latents["threat_cue_salience_bias"]
            + 0.14 * latents["hpa_axis_sensitization"]
            + 0.10 * inputs["chronic_interpersonal_stress"]
            - 0.10 * inputs["interpersonal_safety_support"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.34 * latents["hpa_axis_sensitization"]
            + 0.26 * latents["epigenetic_stress_programming"]
            + 0.16 * inputs["early_life_trauma"]
            + 0.12 * inputs["prenatal_maternal_stress"]
            + 0.08 * inputs["chronic_interpersonal_stress"]
            - 0.14 * inputs["interpersonal_safety_support"]
        )

        symptoms = pd.Series(dtype=float, name="value")
        symptoms["distrust_suspiciousness"] = self._clip01(
            0.30 * latents["hostile_attribution_bias"]
            + 0.24 * regional_state["amygdala"]
            + 0.20 * latents["social_trust_calibration_failure"]
            + 0.12 * latents["threat_cue_salience_bias"]
            + 0.08 * regional_state["mpfc_proxy"]
            - 0.12 * inputs["interpersonal_safety_support"]
        )
        symptoms["malevolent_intent_inference"] = self._clip01(
            0.36 * latents["mentalizing_distortion"]
            + 0.24 * latents["hostile_attribution_bias"]
            + 0.18 * regional_state["mpfc_proxy"]
            + 0.12 * regional_state["amygdala"]
            - 0.10 * inputs["interpersonal_safety_support"]
        )
        symptoms["loyalty_doubts_and_grudges"] = self._clip01(
            0.34 * symptoms["distrust_suspiciousness"]
            + 0.22 * symptoms["malevolent_intent_inference"]
            + 0.18 * latents["hostile_attribution_bias"]
            + 0.12 * regional_state["hippocampus"]
            + 0.08 * regional_state["acc"]
            - 0.08 * inputs["interpersonal_safety_support"]
        )
        symptoms["anger_hostility_outbursts"] = self._clip01(
            0.34 * regional_state["ofc"]
            + 0.24 * regional_state["amygdala"]
            + 0.18 * latents["frontolimbic_regulatory_failure"]
            + 0.12 * latents["monoaminergic_threat_bias"]
            + 0.08 * latents["hostile_attribution_bias"]
            - 0.12 * inputs["interpersonal_safety_support"]
        )
        symptoms["social_avoidance_anxiety"] = self._clip01(
            0.28 * regional_state["acc"]
            + 0.24 * latents["social_error_hypermonitoring"]
            + 0.18 * regional_state["amygdala"]
            + 0.14 * symptoms["distrust_suspiciousness"]
            + 0.10 * latents["hpa_axis_sensitization"]
            - 0.12 * inputs["interpersonal_safety_support"]
        )
        symptoms["antagonistic_aggression_risk"] = self._clip01(
            0.30 * symptoms["anger_hostility_outbursts"]
            + 0.24 * symptoms["malevolent_intent_inference"]
            + 0.16 * regional_state["ofc"]
            + 0.12 * regional_state["amygdala"]
            + 0.10 * inputs["genetic_temperamental_liability"]
            - 0.08 * inputs["interpersonal_safety_support"]
        )

        phenotypes = pd.Series(dtype=float, name="value")
        phenotypes["threat_hypervigilant_profile"] = self._clip01(
            (
                symptoms["distrust_suspiciousness"]
                + symptoms["social_avoidance_anxiety"]
                + regional_state["amygdala"]
                + regional_state["acc"]
            )
            / 4.0
        )
        phenotypes["hostile_suspicious_profile"] = self._clip01(
            (
                symptoms["malevolent_intent_inference"]
                + symptoms["loyalty_doubts_and_grudges"]
                + symptoms["anger_hostility_outbursts"]
                + latents["hostile_attribution_bias"]
            )
            / 4.0
        )
        phenotypes["trauma_sensitized_paranoid_profile"] = self._clip01(
            (
                latents["hpa_axis_sensitization"]
                + regional_state["hippocampus"]
                + symptoms["distrust_suspiciousness"]
                + latents["social_trust_calibration_failure"]
            )
            / 4.0
        )
        phenotypes["socially_avoidant_paranoid_profile"] = self._clip01(
            (
                symptoms["social_avoidance_anxiety"]
                + symptoms["distrust_suspiciousness"]
                + latents["social_error_hypermonitoring"]
                + regional_state["acc"]
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
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)

    model = ParanoidPersonalityDisorderModel()
    bundle = model.build()

    print("\n=== Nodes ===")
    print(bundle["nodes"].head(30).to_string(index=False))

    print("\n=== Edges ===")
    print(bundle["edges"].to_string(index=False))

    print("\n=== Resolved regions ===")
    if bundle["regions"]:
        for key, region in bundle["regions"].items():
            print(f"- {key}: {getattr(region, 'name', region)}")
    else:
        print("No atlas regions resolved in this environment.")

    print("\n=== Feature availability ===")
    for key in model.region_candidates:
        print(
            f"- {key}: receptors={not model.receptors.get(key, pd.DataFrame()).empty}, "
            f"genes={not model.genes.get(key, pd.DataFrame()).empty}, "
            f"connectivity={not model.connectivity_profiles.get(key, pd.DataFrame()).empty}"
        )

    print("\n=== Example connectivity among resolved circuit nodes ===")
    circuit_df = bundle["circuit_connectivity"]
    if isinstance(circuit_df, pd.DataFrame) and not circuit_df.empty:
        print(circuit_df.head(20).to_string(index=False))
    else:
        print("No circuit connectivity matrix available in this environment.")

    for key in ("amygdala", "ofc", "acc"):
        receptor_df = model.receptors.get(key, pd.DataFrame())
        gene_df = model.genes.get(key, pd.DataFrame())
        conn_df = model.connectivity_profiles.get(key, pd.DataFrame())

        if not receptor_df.empty:
            print(f"\n=== Receptor fingerprint sample: {key} ===")
            print(receptor_df.head(10).to_string(index=False))
        if not gene_df.empty:
            print(f"\n=== Gene expression summary: {key} ===")
            print(gene_df.head(10).to_string(index=False))
        if not conn_df.empty:
            print(f"\n=== Connectivity profile: {key} ===")
            print(conn_df.head(10).to_string(index=False))

    print("\n=== Example simulation ===")
    sim = model.simulate(
        genetic_temperamental_liability=0.64,
        early_life_trauma=0.62,
        intergenerational_violence_exposure=0.45,
        prenatal_maternal_stress=0.30,
        dysfunctional_family_environment=0.58,
        chronic_interpersonal_stress=0.60,
        interpersonal_safety_support=0.22,
    )
    for block_name, series in sim.items():
        print(f"\n[{block_name}]")
        print(series.sort_values(ascending=False).to_string())

    # Optional usage examples in a siibra-enabled environment:
    # print(model.suggest_regions("amygdala").head(10))
    # print(model.assign_mni_point((-6, 42, 18)).head(10))
    # mask = model.region_mask("amygdala")
