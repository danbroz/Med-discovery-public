
from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Psychological Factors Affecting Other Medical Conditions.

This script converts a chapter-level biological summary into a transparent mechanistic
research scaffold. It is intended for hypothesis generation, teaching, and iterative
refinement against atlas-backed evidence. It is not a diagnostic or treatment tool.

Chapter-matched biological themes encoded here:
- Acute and chronic psychological stress engage autonomic and endocrine stress systems.
- Noradrenergic arousal is modeled as a core stress-transduction mechanism, reflecting
  the chapter's emphasis on norepinephrine and locus-coeruleus-linked vigilance systems.
- Stress and medical burden converge on inflammatory cytokine signaling, which can feed
  back on neurotransmission, neuroendocrine function, and synaptic plasticity.
- Early-life adversity is modeled as epigenetic stress embedding that increases later
  frontolimbic vulnerability.
- Depression-like and anxiety-like expressions are treated as emerging from frontolimbic
  dysregulation involving amygdala, anterior cingulate / subgenual cingulate, hippocampal,
  and anterior frontal control systems.
- Post-stroke affective risk is represented through left anterior frontal / basal-ganglia
  proxies, following the lesion literature highlighted in the chapter.
- Psychological burden is allowed to worsen downstream medical-condition outcomes through
  autonomic, inflammatory, pain-affect, and mood-related pathways.

Important modeling note:
This chapter is broad and transdiagnostic. Therefore, some constructs remain latent
biology rather than being forced into a single anatomical parcel. Atlas-backed regions
are used conservatively and proxies are explicit where the chapter is systems-level.

Compatibility note:
Connectivity handling is intentionally defensive. siibra connectivity features can be
compound matrices with duplicate anatomical labels or non-scalar lookups. The helpers
below canonicalize labels and collapse DataFrame / Series lookups into numeric values
so the scaffold remains portable across versions and data configurations.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:  # pragma: no cover - optional dependency in authoring environments
    import siibra  # type: ignore
except Exception:  # pragma: no cover
    siibra = None


DEFAULT_GENE_PANEL = [
    "SLC6A2",   # norepinephrine transporter
    "DBH",      # dopamine beta hydroxylase
    "ADRA2A",   # adrenergic receptor
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor
    "COMT",     # catecholamine metabolism
    "BDNF",     # plasticity / stress-related neural remodeling
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress-response calibration
    "CRHR1",    # stress-axis signaling
    "IL6",      # inflammatory cytokine
    "TNF",      # inflammatory cytokine
    "IL1B",     # inflammatory cytokine
    "NFKB1",    # inflammatory signaling
]


class PsychologicalFactorsAffectingOtherMedicalConditionsModel:
    """
    Mechanistic siibra scaffold for Psychological Factors Affecting Other Medical Conditions.

    Main chapter-derived logic:
    1) stress and medical burden engage autonomic/endocrine stress systems,
    2) stress and illness converge on inflammatory cytokine signaling,
    3) inherited and epigenetically embedded affective vulnerability sensitize frontolimbic circuits,
    4) amygdala / cingulate / hippocampal / prefrontal dysfunction yields depression, anxiety,
       sickness behavior, cognitive inefficiency, and pain amplification,
    5) psychological burden worsens medical outcomes and can be intensified by
       cerebrovascular lesions affecting left anterior frontal and left basal-ganglia systems.

    Regional-state values in this simulator represent dysfunction burden rather than raw
    neural firing. For example, a higher DLPFC regional-state value corresponds to greater
    control-system compromise, not greater healthy activation.

    This is a research scaffold, not a validated disease model.
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
        self.siibra_available = siibra is not None

        self.atlas = None
        self.parcellation = None
        self.space = None
        self._pmap = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

        if self.siibra_available:
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
            except Exception as exc:
                warnings.warn(
                    f"siibra is installed but atlas initialization failed: {exc}. "
                    "Atlas-backed methods will return empty results."
                )
        else:
            warnings.warn(
                "siibra is not installed in this environment. "
                "Atlas-backed methods will return empty results, but simulate() still works."
            )

        self.region_candidates: Dict[str, List[str]] = {
            "left_amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "Amygdala left",
                "amygdala",
            ],
            "left_hippocampus_proxy": [
                "CA1 left",
                "CA left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "left_anterior_frontal_dlpfc_proxy": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "middle frontal gyrus left",
                "dorsolateral prefrontal cortex left",
                "prefrontal cortex left",
            ],
            "left_acc_proxy": [
                "Area p32 left",
                "Area a24pr left",
                "Area p24 left",
                "Area s24 left",
                "anterior cingulate cortex left",
                "cingulate cortex left",
            ],
            "subgenual_acc_proxy": [
                "Area s24 left",
                "Area p24 left",
                "subgenual anterior cingulate cortex left",
                "ventral anterior cingulate cortex left",
                "anterior cingulate cortex left",
            ],
            "left_basal_ganglia_proxy": [
                "nucleus accumbens left",
                "caudate nucleus left",
                "putamen left",
                "striatum left",
                "basal ganglia left",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "left_amygdala": (
                "Amygdala node representing threat reactivity and emotional hyper-responsiveness "
                "observed in stress- and depression-related frontolimbic dysfunction."
            ),
            "left_hippocampus_proxy": (
                "Hippocampal proxy for chronic-stress-linked volume loss, memory burden, and "
                "reduced neuroplastic resilience."
            ),
            "left_anterior_frontal_dlpfc_proxy": (
                "Left anterior frontal / dorsolateral prefrontal proxy for cognitive control, "
                "stress regulation, and the left anterior frontal lesion burden linked to post-stroke depression."
            ),
            "left_acc_proxy": (
                "Anterior cingulate proxy for conflict monitoring, affect regulation, and the "
                "pain-affect overlap emphasized in the chapter."
            ),
            "subgenual_acc_proxy": (
                "Subgenual anterior cingulate proxy for the depression-linked limbic-cingulate "
                "hyperactivity pattern described in functional imaging studies."
            ),
            "left_basal_ganglia_proxy": (
                "Left basal-ganglia proxy for cerebrovascular lesion vulnerability and frontostriatal "
                "affective burden in post-stroke depression."
            ),
        }
        self.proxy_region_keys = {
            "left_hippocampus_proxy",
            "left_anterior_frontal_dlpfc_proxy",
            "left_acc_proxy",
            "subgenual_acc_proxy",
            "left_basal_ganglia_proxy",
        }

        self.input_nodes: Dict[str, str] = {
            "acute_stress_load": (
                "Current acute stressor burden triggering rapid autonomic and endocrine responses."
            ),
            "chronic_stress_load": (
                "Sustained stress exposure that can dysregulate neuroendocrine and immune systems."
            ),
            "medical_illness_burden": (
                "Severity of ongoing medical illness, which can amplify psychological and biological stress."
            ),
            "inflammatory_medical_load": (
                "Inflammatory or cytokine-heavy medical burden, including autoimmune or cytokine-treatment-like effects."
            ),
            "family_history_affective_disorders": (
                "Inherited affective vulnerability increasing the probability of severe mood responses to illness or stress."
            ),
            "early_life_stress": (
                "Early adversity capable of leaving durable epigenetic effects on stress and reward/attachment systems."
            ),
            "cerebrovascular_lesion_burden": (
                "Stroke or focal lesion burden, especially relevant to left anterior frontal and left basal-ganglia risk."
            ),
            "integrated_treatment_support": (
                "Protective support from psychotherapy, pharmacotherapy, and coordinated medical-psychiatric care."
            ),
            "stress_regulation_support": (
                "Protective coping structure or stress-management support that can dampen physiological stress reactivity."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "affective_genetic_vulnerability": (
                "Heritable liability for stronger or more prolonged affective responses to stress and illness."
            ),
            "autonomic_endocrine_stress_activation": (
                "Integrated autonomic and endocrine stress-response activation produced by acute or chronic stress."
            ),
            "noradrenergic_arousal": (
                "Norepinephrine-linked vigilance, arousal, and sympathetic stress signaling."
            ),
            "inflammatory_cytokine_signaling": (
                "Stress- and illness-amplified cytokine signaling that feeds back on brain function and behavior."
            ),
            "epigenetic_stress_embedding": (
                "Longer-term biologic embedding of adversity through altered gene-expression regulation."
            ),
            "hippocampal_neuroplasticity_loss": (
                "Stress- and inflammation-related neuroplastic / structural burden contributing to hippocampal vulnerability."
            ),
            "frontolimbic_emotion_regulation_failure": (
                "Impaired regulation across amygdala, cingulate, hippocampal, and prefrontal control systems."
            ),
            "lesion_sensitized_affective_risk": (
                "Affective vulnerability amplified by left anterior frontal or left basal-ganglia lesion burden."
            ),
            "sickness_behavior_drive": (
                "Inflammation-driven fatigue, anhedonia, withdrawal, and cognitive slowing overlapping with depression."
            ),
            "pain_affect_overlap_processing": (
                "Stress-linked overlap between affective and pain-processing systems."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "depressive_symptoms": "Depressive symptoms including low mood, anhedonia, and reduced motivation.",
            "anxiety_distress": "Anxiety, apprehension, tension, or hypervigilant distress.",
            "sickness_behavior": "Fatigue, anhedonia, social withdrawal, and inflammation-linked malaise.",
            "cognitive_disturbance": "Stress-, depression-, or illness-related cognitive inefficiency and memory burden.",
            "pain_amplification": "Increased pain burden or stress-related amplification of pain perception.",
            "medical_condition_exacerbation": "Worsening of the medical illness through stress, inflammation, and affective burden.",
            "post_stroke_depression_risk": "Elevated risk for post-stroke depression-like outcomes.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "family_history_affective_disorders",
                "target": "affective_genetic_vulnerability",
                "relation": "familial affective loading increases biological vulnerability to illness-related psychological burden",
                "pfamc_change": "increased",
            },
            {
                "source": "acute_stress_load",
                "target": "autonomic_endocrine_stress_activation",
                "relation": "acute stress rapidly activates adaptive autonomic and endocrine stress responses",
                "pfamc_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "autonomic_endocrine_stress_activation",
                "relation": "chronic stress prolongs and dysregulates stress-system activation",
                "pfamc_change": "increased",
            },
            {
                "source": "medical_illness_burden",
                "target": "autonomic_endocrine_stress_activation",
                "relation": "medical illness can itself function as a major stressor",
                "pfamc_change": "increased",
            },
            {
                "source": "acute_stress_load",
                "target": "noradrenergic_arousal",
                "relation": "stress heightens sympathetic arousal and norepinephrine-linked vigilance",
                "pfamc_change": "increased",
            },
            {
                "source": "autonomic_endocrine_stress_activation",
                "target": "noradrenergic_arousal",
                "relation": "stress-system activation drives sustained arousal signaling",
                "pfamc_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "inflammatory_cytokine_signaling",
                "relation": "prolonged stress promotes inflammatory signaling",
                "pfamc_change": "increased",
            },
            {
                "source": "medical_illness_burden",
                "target": "inflammatory_cytokine_signaling",
                "relation": "medical illness contributes inflammatory burden that can signal back to the brain",
                "pfamc_change": "increased",
            },
            {
                "source": "inflammatory_medical_load",
                "target": "inflammatory_cytokine_signaling",
                "relation": "cytokine-heavy illness burden strengthens brain-immune signaling",
                "pfamc_change": "increased",
            },
            {
                "source": "autonomic_endocrine_stress_activation",
                "target": "inflammatory_cytokine_signaling",
                "relation": "stress and inflammation form a self-reinforcing biological cycle",
                "pfamc_change": "increased",
            },
            {
                "source": "early_life_stress",
                "target": "epigenetic_stress_embedding",
                "relation": "early adversity can become biologically embedded through epigenetic change",
                "pfamc_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "epigenetic_stress_embedding",
                "relation": "prolonged stress can produce durable gene-expression changes",
                "pfamc_change": "increased",
            },
            {
                "source": "affective_genetic_vulnerability",
                "target": "frontolimbic_emotion_regulation_failure",
                "relation": "inherited vulnerability increases frontolimbic sensitivity to stress and illness",
                "pfamc_change": "increased",
            },
            {
                "source": "epigenetic_stress_embedding",
                "target": "frontolimbic_emotion_regulation_failure",
                "relation": "embedded stress effects increase later frontolimbic dysregulation",
                "pfamc_change": "increased",
            },
            {
                "source": "inflammatory_cytokine_signaling",
                "target": "frontolimbic_emotion_regulation_failure",
                "relation": "cytokine signaling perturbs neurotransmission, endocrine function, and synaptic plasticity",
                "pfamc_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "hippocampal_neuroplasticity_loss",
                "relation": "chronic stress is associated with hippocampal structural burden",
                "pfamc_change": "increased",
            },
            {
                "source": "epigenetic_stress_embedding",
                "target": "hippocampal_neuroplasticity_loss",
                "relation": "durable stress-related molecular changes can bias later hippocampal vulnerability",
                "pfamc_change": "increased",
            },
            {
                "source": "inflammatory_cytokine_signaling",
                "target": "hippocampal_neuroplasticity_loss",
                "relation": "inflammation contributes to synaptic and neuroplastic burden",
                "pfamc_change": "increased",
            },
            {
                "source": "cerebrovascular_lesion_burden",
                "target": "lesion_sensitized_affective_risk",
                "relation": "stroke lesions can substantially raise affective risk",
                "pfamc_change": "increased",
            },
            {
                "source": "affective_genetic_vulnerability",
                "target": "lesion_sensitized_affective_risk",
                "relation": "preexisting affective vulnerability amplifies lesion-related psychiatric risk",
                "pfamc_change": "increased",
            },
            {
                "source": "inflammatory_cytokine_signaling",
                "target": "sickness_behavior_drive",
                "relation": "cytokine signaling induces fatigue, anhedonia, withdrawal, and cognitive slowing",
                "pfamc_change": "increased",
            },
            {
                "source": "frontolimbic_emotion_regulation_failure",
                "target": "pain_affect_overlap_processing",
                "relation": "emotion-control failure strengthens affective processing of pain and stress",
                "pfamc_change": "increased",
            },
            {
                "source": "noradrenergic_arousal",
                "target": "pain_affect_overlap_processing",
                "relation": "hyperarousal can increase pain-related salience and distress",
                "pfamc_change": "increased",
            },
            {
                "source": "frontolimbic_emotion_regulation_failure",
                "target": "left_amygdala",
                "relation": "frontolimbic dysregulation includes heightened amygdala burden",
                "pfamc_change": "increased",
            },
            {
                "source": "hippocampal_neuroplasticity_loss",
                "target": "left_hippocampus_proxy",
                "relation": "hippocampal neuroplastic burden maps onto hippocampal structural vulnerability",
                "pfamc_change": "increased",
            },
            {
                "source": "frontolimbic_emotion_regulation_failure",
                "target": "left_anterior_frontal_dlpfc_proxy",
                "relation": "emotion-control failure burdens anterior frontal / dorsolateral control systems",
                "pfamc_change": "increased",
            },
            {
                "source": "pain_affect_overlap_processing",
                "target": "left_acc_proxy",
                "relation": "stress and pain overlap engage cingulate systems",
                "pfamc_change": "increased",
            },
            {
                "source": "frontolimbic_emotion_regulation_failure",
                "target": "subgenual_acc_proxy",
                "relation": "depression-linked frontolimbic dysfunction burdens subgenual cingulate systems",
                "pfamc_change": "increased",
            },
            {
                "source": "lesion_sensitized_affective_risk",
                "target": "left_basal_ganglia_proxy",
                "relation": "post-stroke affective risk includes left basal-ganglia lesion vulnerability",
                "pfamc_change": "increased",
            },
            {
                "source": "noradrenergic_arousal",
                "target": "anxiety_distress",
                "relation": "norepinephrine-linked arousal contributes to anxiety and vigilance",
                "pfamc_change": "increased",
            },
            {
                "source": "frontolimbic_emotion_regulation_failure",
                "target": "depressive_symptoms",
                "relation": "frontolimbic regulation failure contributes to depressive symptoms",
                "pfamc_change": "increased",
            },
            {
                "source": "sickness_behavior_drive",
                "target": "sickness_behavior",
                "relation": "inflammatory sickness mechanisms generate fatigue and anhedonia",
                "pfamc_change": "increased",
            },
            {
                "source": "hippocampal_neuroplasticity_loss",
                "target": "cognitive_disturbance",
                "relation": "hippocampal burden contributes to memory and cognitive inefficiency",
                "pfamc_change": "increased",
            },
            {
                "source": "pain_affect_overlap_processing",
                "target": "pain_amplification",
                "relation": "stress-linked overlap between affect and pain increases perceived pain burden",
                "pfamc_change": "increased",
            },
            {
                "source": "inflammatory_cytokine_signaling",
                "target": "medical_condition_exacerbation",
                "relation": "inflammation provides a direct biological route from psychological burden to medical worsening",
                "pfamc_change": "increased",
            },
            {
                "source": "autonomic_endocrine_stress_activation",
                "target": "medical_condition_exacerbation",
                "relation": "prolonged stress physiology worsens physical health outcomes",
                "pfamc_change": "increased",
            },
            {
                "source": "lesion_sensitized_affective_risk",
                "target": "post_stroke_depression_risk",
                "relation": "left frontal and basal-ganglia lesion burden elevates post-stroke depression risk",
                "pfamc_change": "increased",
            },
            {
                "source": "stress_regulation_support",
                "target": "autonomic_endocrine_stress_activation",
                "relation": "stress-management support can reduce physiological stress activation",
                "pfamc_change": "decreased",
            },
            {
                "source": "integrated_treatment_support",
                "target": "frontolimbic_emotion_regulation_failure",
                "relation": "effective pharmacologic or psychotherapeutic support can normalize dysfunctional frontolimbic activity",
                "pfamc_change": "decreased",
            },
            {
                "source": "integrated_treatment_support",
                "target": "inflammatory_cytokine_signaling",
                "relation": "treatment can reduce psychological and inflammatory burden",
                "pfamc_change": "decreased",
            },
            {
                "source": "integrated_treatment_support",
                "target": "depressive_symptoms",
                "relation": "integrated treatment reduces depressive symptom expression",
                "pfamc_change": "decreased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}
        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _preferred_hemisphere(spec: str) -> Optional[str]:
        low = str(spec).lower()
        if "right" in low:
            return "right"
        if "left" in low:
            return "left"
        return None

    def _modality_candidates(self, kind: str) -> List[Any]:
        if not self.siibra_available:
            return []
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
        if not self.siibra_available or concept is None:
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
        if not self.siibra_available or self.atlas is None:
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

    def _region_rank(self, region: Any, prefer_hemisphere: Optional[str] = None) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        if prefer_hemisphere == "right":
            hemi_penalty = 0 if "right" in name else 1
        elif prefer_hemisphere == "left":
            hemi_penalty = 0 if "left" in name else 1
        else:
            hemi_penalty = 0 if "left" in name else 1

        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "prefrontal cortex",
            "cingulate cortex",
            "basal ganglia",
            "striatum",
        } else 0
        cyto_penalty = 0 if "area " in name or "(" in name else 1
        length_penalty = len(name)
        return (hemi_penalty, generic_penalty, cyto_penalty, length_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if not self.siibra_available or self.atlas is None or self.parcellation is None:
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
                prefer = self._preferred_hemisphere(spec)
                return sorted(matches, key=lambda r: self._region_rank(r, prefer))[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        seen = set()
        matches = self._julich_matches(keyword)
        for region in sorted(matches, key=lambda r: self._region_rank(r, None)):
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
                coord = getattr(centroid, "coordinate", centroid)
                centroid_xyz = tuple(float(x) for x in coord[:3])
            except Exception:
                try:
                    centroid_xyz = tuple(float(x) for x in centroid)[:3]
                except Exception:
                    centroid_xyz = None
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
                    .reset_index(drop=True)
                )
            except Exception:
                pass
        return df.reset_index(drop=True)

    def _canonical_label(self, label: Any) -> str:
        return str(self._name_of(label)).strip()

    def _canonicalize_connectivity_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        if matrix is None or matrix.empty:
            return pd.DataFrame()
        out = matrix.copy()

        try:
            out.index = [self._canonical_label(x) for x in out.index]
        except Exception:
            out.index = [str(x) for x in out.index]

        try:
            out.columns = [self._canonical_label(x) for x in out.columns]
        except Exception:
            out.columns = [str(x) for x in out.columns]

        try:
            out = out.apply(pd.to_numeric, errors="coerce")
        except Exception:
            pass

        if pd.Index(out.index).duplicated().any():
            out = out.groupby(level=0).mean(numeric_only=True)

        if pd.Index(out.columns).duplicated().any():
            out = out.T.groupby(level=0).mean(numeric_only=True).T

        return out

    @staticmethod
    def _reduce_connectivity_value(value: Any) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, pd.DataFrame):
            try:
                numeric = value.apply(pd.to_numeric, errors="coerce").to_numpy().ravel()
                numeric = [float(v) for v in numeric if pd.notna(v)]
                if not numeric:
                    return None
                return float(sum(numeric) / len(numeric))
            except Exception:
                return None

        if isinstance(value, pd.Series):
            try:
                ser = pd.to_numeric(value, errors="coerce")
                ser = ser[ser.notna()]
                if ser.empty:
                    return None
                return float(ser.mean())
            except Exception:
                return None

        try:
            return float(value)
        except Exception:
            return None

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix
        if not self.siibra_available or self.parcellation is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), feats[0])

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = self._canonicalize_connectivity_matrix(data)
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            data = compound[0].data.copy()
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = self._canonicalize_connectivity_matrix(data)
                return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._canonical_label(getattr(region, "name", region))
        exact = [x for x in labels if self._canonical_label(x) == region_name]
        if exact:
            return exact[0]

        rn = region_name.lower()
        fuzzy = [x for x in labels if rn in self._canonical_label(x).lower() or self._canonical_label(x).lower() in rn]
        if fuzzy:
            return fuzzy[0]

        rn_tokens = [tok for tok in rn.replace("(", " ").replace(")", " ").replace("-", " ").split() if tok]
        best = None
        best_score = 0
        for label in labels:
            text = self._canonical_label(label).lower()
            score = sum(tok in text for tok in rn_tokens)
            if score > best_score:
                best = label
                best_score = score
        return best if best_score >= 2 else None

    def _series_from_connectivity_lookup(self, raw: Any, axis: str = "index") -> Optional[pd.Series]:
        if raw is None:
            return None
        if isinstance(raw, pd.Series):
            try:
                ser = pd.to_numeric(raw, errors="coerce")
                ser = ser[ser.notna()]
                return ser if not ser.empty else None
            except Exception:
                return None
        if isinstance(raw, pd.DataFrame):
            try:
                numeric = raw.apply(pd.to_numeric, errors="coerce")
                ser = numeric.mean(axis=0 if axis == "index" else 1)
                ser = ser[ser.notna()]
                return ser if not ser.empty else None
            except Exception:
                return None
        return None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if not matrix.empty:
            label = self._match_region_label(list(matrix.index), region)
            axis = "index"
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
                axis = "columns"

            if label is not None:
                try:
                    raw = matrix.loc[label] if axis == "index" else matrix[label]
                    series = self._series_from_connectivity_lookup(raw, axis=axis)
                    if series is not None:
                        df = series.sort_values(ascending=False).reset_index()
                        df.columns = ["connected_region", "value"]
                        df["connected_region"] = df["connected_region"].map(self._canonical_label)
                        df = df[df["connected_region"] != self._canonical_label(getattr(region, "name", region))]
                        return df.head(max_rows).reset_index(drop=True)
                except Exception:
                    pass

        feats = self._safe_features_any(region, self._modality_candidates("connectivity"))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            try:
                data = getattr(feat, "data", None)
                if isinstance(data, pd.DataFrame):
                    out = data.copy().reset_index(drop=False)
                    out.columns = [str(c) for c in out.columns]
                    return out.head(max_rows)
            except Exception:
                continue
        return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return a pairwise connectivity table among resolved regional circuit nodes.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame(columns=["source", "target", "value"])

        labels: Dict[str, Any] = {}
        for key, region in self.region_objects.items():
            label = self._match_region_label(list(matrix.index), region)
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
            if label is not None:
                labels[key] = label

        rows: List[Dict[str, Any]] = []
        keys = list(labels.keys())
        for i, source in enumerate(keys):
            for target in keys[i + 1:]:
                s_label = labels[source]
                t_label = labels[target]
                raw = None

                try:
                    raw = matrix.loc[s_label, t_label]
                except Exception:
                    pass
                if raw is None:
                    try:
                        raw = matrix.loc[t_label, s_label]
                    except Exception:
                        pass
                if raw is None:
                    try:
                        raw = matrix[s_label][t_label]
                    except Exception:
                        raw = None

                value = self._reduce_connectivity_value(raw)
                if value is not None:
                    rows.append({"source": source, "target": target, "value": value})

        if not rows:
            return pd.DataFrame(columns=["source", "target", "value"])
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        """
        Resolve atlas regions when possible and gather receptor / gene / connectivity summaries.
        """
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
                }
            )

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            desc = self.region_node_descriptions.get(key, "Atlas-backed circuit node.")
            if region is None:
                if self.siibra_available:
                    warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region_proxy" if key in self.proxy_region_keys else "region",
                        "description": f"{desc} Unresolved in the current environment.",
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
                    "label": getattr(region, "name", key.replace("_", " ").title()),
                    "node_type": "region_proxy" if key in self.proxy_region_keys else "region",
                    "description": desc,
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

        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": self.circuit_connectivity(),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Probabilistically assign an MNI152 point to Julich regions using a statistical map.
        """
        if not self.siibra_available:
            return pd.DataFrame()

        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception:
                return pd.DataFrame()

        try:
            point = siibra.Point(tuple(float(v) for v in xyz[:3]), space=self.assignment_space)
            with siibra.QUIET:
                assignments = self._pmap.assign(point)
        except Exception:
            return pd.DataFrame()

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        """
        Return a region mask or map representation when available.
        """
        if not self.siibra_available:
            return None
        region = self.region_objects.get(node_key)
        if region is None:
            return None

        try:
            if hasattr(region, "get_regional_mask"):
                mask = region.get_regional_mask(space=self.assignment_space, maptype="labelled")
                try:
                    return mask.fetch()
                except Exception:
                    return mask
        except Exception:
            pass

        try:
            if hasattr(region, "get_regional_map"):
                regional_map = region.get_regional_map(self.assignment_space, "statistical")
                try:
                    return regional_map.fetch()
                except Exception:
                    return regional_map
        except Exception:
            pass

        try:
            pmap = siibra.get_map(
                parcellation=self.parcellation_spec,
                space=self.assignment_space,
                maptype="statistical",
            )
            try:
                return pmap.fetch(region=region)
            except Exception:
                return pmap
        except Exception:
            return None

    def simulate(
        self,
        acute_stress_load: float = 0.45,
        chronic_stress_load: float = 0.55,
        medical_illness_burden: float = 0.50,
        inflammatory_medical_load: float = 0.35,
        family_history_affective_disorders: float = 0.30,
        early_life_stress: float = 0.30,
        cerebrovascular_lesion_burden: float = 0.10,
        integrated_treatment_support: float = 0.30,
        stress_regulation_support: float = 0.30,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Inputs are clipped to [0, 1]. Protective variables reduce downstream burden
        more than they erase upstream vulnerability, consistent with the chapter's
        focus on treatment normalizing function rather than eliminating all biological risk.
        """
        inputs = {
            "acute_stress_load": self._clip01(acute_stress_load),
            "chronic_stress_load": self._clip01(chronic_stress_load),
            "medical_illness_burden": self._clip01(medical_illness_burden),
            "inflammatory_medical_load": self._clip01(inflammatory_medical_load),
            "family_history_affective_disorders": self._clip01(family_history_affective_disorders),
            "early_life_stress": self._clip01(early_life_stress),
            "cerebrovascular_lesion_burden": self._clip01(cerebrovascular_lesion_burden),
            "integrated_treatment_support": self._clip01(integrated_treatment_support),
            "stress_regulation_support": self._clip01(stress_regulation_support),
        }

        support_mean = self._clip01(
            (inputs["integrated_treatment_support"] + inputs["stress_regulation_support"]) / 2.0
        )

        latents: Dict[str, float] = {}
        latents["affective_genetic_vulnerability"] = self._clip01(
            0.80 * inputs["family_history_affective_disorders"]
            + 0.20 * inputs["early_life_stress"]
        )
        latents["epigenetic_stress_embedding"] = self._clip01(
            0.55 * inputs["early_life_stress"]
            + 0.25 * inputs["chronic_stress_load"]
            + 0.20 * inputs["medical_illness_burden"]
        )
        latents["autonomic_endocrine_stress_activation"] = self._clip01(
            0.30 * inputs["acute_stress_load"]
            + 0.30 * inputs["chronic_stress_load"]
            + 0.15 * inputs["medical_illness_burden"]
            + 0.10 * latents["affective_genetic_vulnerability"]
            + 0.15 * latents["epigenetic_stress_embedding"]
            - 0.20 * inputs["stress_regulation_support"]
            - 0.10 * inputs["integrated_treatment_support"]
        )
        latents["noradrenergic_arousal"] = self._clip01(
            0.50 * latents["autonomic_endocrine_stress_activation"]
            + 0.20 * inputs["acute_stress_load"]
            + 0.10 * inputs["chronic_stress_load"]
            + 0.20 * latents["affective_genetic_vulnerability"]
            - 0.15 * inputs["integrated_treatment_support"]
            - 0.10 * inputs["stress_regulation_support"]
        )
        latents["inflammatory_cytokine_signaling"] = self._clip01(
            0.25 * inputs["chronic_stress_load"]
            + 0.20 * inputs["medical_illness_burden"]
            + 0.20 * inputs["inflammatory_medical_load"]
            + 0.20 * latents["autonomic_endocrine_stress_activation"]
            + 0.15 * latents["epigenetic_stress_embedding"]
            - 0.15 * inputs["integrated_treatment_support"]
        )
        latents["hippocampal_neuroplasticity_loss"] = self._clip01(
            0.30 * inputs["chronic_stress_load"]
            + 0.25 * latents["epigenetic_stress_embedding"]
            + 0.20 * latents["inflammatory_cytokine_signaling"]
            + 0.10 * inputs["medical_illness_burden"]
            + 0.15 * latents["affective_genetic_vulnerability"]
            - 0.15 * inputs["integrated_treatment_support"]
        )
        latents["frontolimbic_emotion_regulation_failure"] = self._clip01(
            0.20 * latents["affective_genetic_vulnerability"]
            + 0.15 * latents["epigenetic_stress_embedding"]
            + 0.20 * latents["noradrenergic_arousal"]
            + 0.20 * latents["inflammatory_cytokine_signaling"]
            + 0.15 * inputs["chronic_stress_load"]
            + 0.10 * inputs["medical_illness_burden"]
            - 0.20 * inputs["integrated_treatment_support"]
            - 0.10 * inputs["stress_regulation_support"]
        )
        latents["lesion_sensitized_affective_risk"] = self._clip01(
            0.55 * inputs["cerebrovascular_lesion_burden"]
            + 0.25 * latents["affective_genetic_vulnerability"]
            + 0.20 * latents["inflammatory_cytokine_signaling"]
            - 0.10 * inputs["integrated_treatment_support"]
        )
        latents["sickness_behavior_drive"] = self._clip01(
            0.50 * latents["inflammatory_cytokine_signaling"]
            + 0.20 * inputs["medical_illness_burden"]
            + 0.15 * inputs["chronic_stress_load"]
            + 0.15 * latents["frontolimbic_emotion_regulation_failure"]
            - 0.10 * inputs["integrated_treatment_support"]
        )
        latents["pain_affect_overlap_processing"] = self._clip01(
            0.40 * latents["frontolimbic_emotion_regulation_failure"]
            + 0.30 * latents["noradrenergic_arousal"]
            + 0.20 * latents["inflammatory_cytokine_signaling"]
            + 0.10 * inputs["medical_illness_burden"]
            - 0.15 * inputs["integrated_treatment_support"]
            - 0.10 * inputs["stress_regulation_support"]
        )

        regional_state: Dict[str, float] = {}
        regional_state["left_amygdala"] = self._clip01(
            0.40 * latents["frontolimbic_emotion_regulation_failure"]
            + 0.30 * latents["noradrenergic_arousal"]
            + 0.15 * inputs["acute_stress_load"]
            + 0.15 * latents["affective_genetic_vulnerability"]
            - 0.10 * inputs["integrated_treatment_support"]
        )
        regional_state["left_hippocampus_proxy"] = self._clip01(
            0.45 * latents["hippocampal_neuroplasticity_loss"]
            + 0.20 * inputs["chronic_stress_load"]
            + 0.15 * latents["inflammatory_cytokine_signaling"]
            + 0.20 * latents["epigenetic_stress_embedding"]
            - 0.15 * inputs["integrated_treatment_support"]
        )
        regional_state["left_anterior_frontal_dlpfc_proxy"] = self._clip01(
            0.40 * latents["frontolimbic_emotion_regulation_failure"]
            + 0.20 * latents["lesion_sensitized_affective_risk"]
            + 0.20 * inputs["chronic_stress_load"]
            + 0.20 * latents["pain_affect_overlap_processing"]
            - 0.20 * inputs["integrated_treatment_support"]
            - 0.10 * inputs["stress_regulation_support"]
        )
        regional_state["left_acc_proxy"] = self._clip01(
            0.35 * latents["pain_affect_overlap_processing"]
            + 0.25 * latents["frontolimbic_emotion_regulation_failure"]
            + 0.20 * latents["inflammatory_cytokine_signaling"]
            + 0.20 * latents["autonomic_endocrine_stress_activation"]
            - 0.15 * inputs["integrated_treatment_support"]
        )
        regional_state["subgenual_acc_proxy"] = self._clip01(
            0.40 * latents["frontolimbic_emotion_regulation_failure"]
            + 0.25 * latents["inflammatory_cytokine_signaling"]
            + 0.20 * latents["sickness_behavior_drive"]
            + 0.15 * latents["lesion_sensitized_affective_risk"]
            - 0.15 * inputs["integrated_treatment_support"]
        )
        regional_state["left_basal_ganglia_proxy"] = self._clip01(
            0.45 * latents["lesion_sensitized_affective_risk"]
            + 0.20 * latents["frontolimbic_emotion_regulation_failure"]
            + 0.20 * latents["inflammatory_cytokine_signaling"]
            + 0.15 * latents["sickness_behavior_drive"]
            - 0.10 * inputs["integrated_treatment_support"]
        )

        symptoms: Dict[str, float] = {}
        symptoms["depressive_symptoms"] = self._clip01(
            0.30 * latents["frontolimbic_emotion_regulation_failure"]
            + 0.20 * regional_state["subgenual_acc_proxy"]
            + 0.15 * regional_state["left_hippocampus_proxy"]
            + 0.15 * latents["sickness_behavior_drive"]
            + 0.10 * latents["lesion_sensitized_affective_risk"]
            + 0.10 * inputs["medical_illness_burden"]
            - 0.25 * inputs["integrated_treatment_support"]
            - 0.10 * inputs["stress_regulation_support"]
        )
        symptoms["anxiety_distress"] = self._clip01(
            0.35 * latents["noradrenergic_arousal"]
            + 0.20 * regional_state["left_amygdala"]
            + 0.20 * latents["autonomic_endocrine_stress_activation"]
            + 0.15 * latents["frontolimbic_emotion_regulation_failure"]
            + 0.10 * inputs["acute_stress_load"]
            - 0.20 * inputs["stress_regulation_support"]
            - 0.10 * inputs["integrated_treatment_support"]
        )
        symptoms["sickness_behavior"] = self._clip01(
            0.50 * latents["sickness_behavior_drive"]
            + 0.20 * latents["inflammatory_cytokine_signaling"]
            + 0.15 * regional_state["subgenual_acc_proxy"]
            + 0.15 * inputs["medical_illness_burden"]
            - 0.15 * inputs["integrated_treatment_support"]
        )
        symptoms["cognitive_disturbance"] = self._clip01(
            0.30 * regional_state["left_hippocampus_proxy"]
            + 0.20 * regional_state["left_anterior_frontal_dlpfc_proxy"]
            + 0.20 * latents["frontolimbic_emotion_regulation_failure"]
            + 0.15 * regional_state["left_acc_proxy"]
            + 0.15 * symptoms["depressive_symptoms"]
            - 0.15 * inputs["integrated_treatment_support"]
        )
        symptoms["pain_amplification"] = self._clip01(
            0.35 * latents["pain_affect_overlap_processing"]
            + 0.20 * regional_state["left_acc_proxy"]
            + 0.20 * regional_state["subgenual_acc_proxy"]
            + 0.15 * latents["inflammatory_cytokine_signaling"]
            + 0.10 * symptoms["anxiety_distress"]
            - 0.20 * inputs["integrated_treatment_support"]
            - 0.10 * inputs["stress_regulation_support"]
        )
        symptoms["medical_condition_exacerbation"] = self._clip01(
            0.25 * latents["inflammatory_cytokine_signaling"]
            + 0.20 * latents["autonomic_endocrine_stress_activation"]
            + 0.15 * symptoms["depressive_symptoms"]
            + 0.15 * symptoms["anxiety_distress"]
            + 0.10 * symptoms["sickness_behavior"]
            + 0.15 * inputs["medical_illness_burden"]
            - 0.20 * inputs["integrated_treatment_support"]
            - 0.10 * inputs["stress_regulation_support"]
        )
        symptoms["post_stroke_depression_risk"] = self._clip01(
            0.40 * latents["lesion_sensitized_affective_risk"]
            + 0.20 * regional_state["left_basal_ganglia_proxy"]
            + 0.20 * regional_state["left_anterior_frontal_dlpfc_proxy"]
            + 0.20 * symptoms["depressive_symptoms"]
            - 0.15 * inputs["integrated_treatment_support"]
        )

        phenotypes: Dict[str, float] = {}
        phenotypes["stress_inflammation_cycle_profile"] = self._clip01(
            (
                latents["autonomic_endocrine_stress_activation"]
                + latents["noradrenergic_arousal"]
                + latents["inflammatory_cytokine_signaling"]
                + symptoms["sickness_behavior"]
            ) / 4.0
        )
        phenotypes["depression_medical_interaction_profile"] = self._clip01(
            (
                symptoms["depressive_symptoms"]
                + symptoms["medical_condition_exacerbation"]
                + regional_state["subgenual_acc_proxy"]
                + regional_state["left_hippocampus_proxy"]
            ) / 4.0
        )
        phenotypes["post_stroke_affective_risk_profile"] = self._clip01(
            (
                symptoms["post_stroke_depression_risk"]
                + latents["lesion_sensitized_affective_risk"]
                + regional_state["left_basal_ganglia_proxy"]
                + regional_state["left_anterior_frontal_dlpfc_proxy"]
            ) / 4.0
        )
        phenotypes["pain_stress_overlap_profile"] = self._clip01(
            (
                symptoms["pain_amplification"]
                + latents["pain_affect_overlap_processing"]
                + regional_state["left_acc_proxy"]
                + symptoms["anxiety_distress"]
            ) / 4.0
        )
        phenotypes["overall_psychological_medical_burden"] = self._clip01(
            0.25 * phenotypes["stress_inflammation_cycle_profile"]
            + 0.30 * phenotypes["depression_medical_interaction_profile"]
            + 0.20 * phenotypes["pain_stress_overlap_profile"]
            + 0.15 * symptoms["medical_condition_exacerbation"]
            + 0.10 * symptoms["post_stroke_depression_risk"]
            - 0.10 * support_mean
        )

        return {
            "inputs": pd.Series(inputs, dtype=float),
            "latents": pd.Series(latents, dtype=float),
            "regional_state": pd.Series(regional_state, dtype=float),
            "symptoms": pd.Series(symptoms, dtype=float),
            "phenotypes": pd.Series(phenotypes, dtype=float),
        }


if __name__ == "__main__":
    model = PsychologicalFactorsAffectingOtherMedicalConditionsModel()

    built = model.build()
    print("\n=== Nodes (first 14) ===")
    print(built["nodes"].head(14).to_string(index=False))

    print("\n=== Edge count ===")
    print(len(built["edges"]))

    print("\n=== Resolved regions ===")
    if built["regions"]:
        for key, region in built["regions"].items():
            print(f"- {key}: {getattr(region, 'name', region)}")
    else:
        print("No atlas-backed regions resolved in this environment.")

    print("\n=== Connectivity among resolved circuit nodes ===")
    print(built["circuit_connectivity"].head(10).to_string(index=False))

    print("\n=== Example receptor / gene / connectivity tables ===")
    for key in [
        "left_amygdala",
        "left_anterior_frontal_dlpfc_proxy",
        "subgenual_acc_proxy",
    ]:
        print(f"\nNode: {key}")
        print(f"  receptors rows: {len(built['receptors'].get(key, pd.DataFrame()))}")
        print(f"  genes rows: {len(built['genes'].get(key, pd.DataFrame()))}")
        print(f"  connectivity rows: {len(built['connectivity_profiles'].get(key, pd.DataFrame()))}")

    example = model.simulate(
        acute_stress_load=0.55,
        chronic_stress_load=0.70,
        medical_illness_burden=0.65,
        inflammatory_medical_load=0.50,
        family_history_affective_disorders=0.40,
        early_life_stress=0.35,
        cerebrovascular_lesion_burden=0.15,
        integrated_treatment_support=0.30,
        stress_regulation_support=0.25,
    )

    print("\n=== Latent biology ===")
    print(example["latents"].round(3).to_string())

    print("\n=== Regional state ===")
    print(example["regional_state"].round(3).to_string())

    print("\n=== Symptoms ===")
    print(example["symptoms"].round(3).to_string())

    print("\n=== Phenotypes ===")
    print(example["phenotypes"].round(3).to_string())

    # Example optional anatomical assignment when siibra is available:
    # print(model.assign_mni_point((-24, -10, -18)).head())
