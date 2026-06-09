from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Specific Phobia.

This script converts a chapter-level biological summary into a transparent mechanistic
research scaffold. It is intended for hypothesis generation, teaching, and iterative
refinement against atlas-backed evidence. It is not a diagnostic or treatment tool.

Chapter-matched biological themes encoded here:
- Specific phobia is modeled as an exaggerated fear-circuit disorder rather than merely
  a psychological dislike or ordinary caution.
- Conditioned fear learning, genetic vulnerability, and stress sensitization increase
  amygdala-centered threat bias and hippocampal contextual fear binding.
- Serotonergic dysregulation is treated as a latent vulnerability that weakens top-down
  cortical regulation of fear-related circuits; serotonergic treatment support buffers
  this component.
- CRF-like stress amplification and CCK-linked acute panic susceptibility increase
  autonomic alarm and panic-like somatic reactivity.
- The low-road / high-road cascade is represented with thalamic relay and visual-cortex
  proxies feeding amygdala burden during cue exposure.
- Blood-injection-injury phobia is given a dedicated vasovagal liability pathway because
  the chapter highlights its familial aggregation and fainting-prone autonomic profile.

Important modeling note:
The chapter is stronger at the level of distributed fear circuitry than at the level of
highly specific cytoarchitectonic localization for structures such as thalamus,
hypothalamus, locus coeruleus, and prefrontal control sectors. These are therefore kept
as explicit proxies rather than over-claimed fixed parcels.

The scaffold is compatibility-first and degrades gracefully when:
- siibra is not installed,
- a requested Julich region cannot be resolved,
- receptor, gene-expression, or connectivity features are unavailable.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:  # pragma: no cover - optional dependency in authoring environments
    import siibra  # type: ignore
except Exception:  # pragma: no cover
    siibra = None


DEFAULT_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor
    "HTR2A",    # serotonin receptor
    "TPH2",     # serotonin synthesis
    "MAOA",     # monoamine metabolism
    "COMT",     # catecholamine metabolism
    "DRD2",     # dopamine signaling
    "SLC6A3",   # dopamine transporter
    "CRH",      # corticotropin-releasing hormone
    "CRHR1",    # CRF receptor
    "CCK",      # cholecystokinin
    "CCKBR",    # CCK receptor B
    "BDNF",     # plasticity / fear learning
    "FKBP5",    # stress-response regulation
]


class SpecificPhobiaModel:
    """
    Mechanistic siibra scaffold for Specific Phobia.

    Main chapter-derived logic:
    1) genetic and conditioned-fear vulnerability increase threat-circuit reactivity,
    2) serotonergic dysregulation and stress amplification weaken cortical regulation,
    3) amygdala, hippocampus, insula, ACC, and proxy arousal systems produce fear,
       hyperarousal, context triggering, and avoidance,
    4) blood-injection-injury subtype liability increases vasovagal fainting risk.

    Regional-state values represent disorder-relevant circuit burden rather than raw
    neural firing. Higher values indicate greater dysfunction or pathological recruitment
    in the phobic episode framework.

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

        # Direct anchors are used where the chapter names structures explicitly.
        # Proxies are used where the text is systems-level or Julich labels may vary.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "SF (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "hippocampus": [
                "hippocampus left",
                "subiculum left",
                "CA1 left",
                "CA3 left",
                "dentate gyrus left",
                "hippocampus",
            ],
            "insula": [
                "insula left",
                "anterior insula left",
                "insular cortex left",
            ],
            "acc": [
                "Area p32 left",
                "Area s24 left",
                "anterior cingulate cortex left",
                "cingulate cortex left",
            ],
            "medial_prefrontal_control_proxy": [
                "Area s32 left",
                "Area p32 left",
                "Area 10 left",
                "Area 9 left",
                "medial prefrontal cortex left",
                "prefrontal cortex left",
            ],
            "thalamic_relay_proxy": [
                "thalamus left",
                "mediodorsal thalamus left",
                "pulvinar left",
                "thalamus",
            ],
            "visual_cortex_proxy": [
                "Area hOc1 (V1) left",
                "Area hOc2 (V2) left",
                "visual cortex left",
                "occipital cortex left",
            ],
            "hypothalamus_proxy": [
                "hypothalamus left",
                "hypothalamus",
            ],
            "locus_coeruleus_proxy": [
                "locus coeruleus left",
                "locus coeruleus",
                "pons left",
                "brainstem",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "amygdala": (
                "Amygdala anchor for rapid threat valuation and exaggerated phobic alarm."
            ),
            "hippocampus": (
                "Hippocampal anchor for contextual memory of fear and retrieval of where-and-when threat associations."
            ),
            "insula": (
                "Insular anchor for interoceptive detection of bodily alarm and subjective awareness of fear physiology."
            ),
            "acc": (
                "Anterior cingulate anchor for distress, conflict processing, and the tension between fear-driven escape and external constraints."
            ),
            "medial_prefrontal_control_proxy": (
                "Prefrontal regulatory proxy for insufficient top-down control over amygdala-driven fear responses."
            ),
            "thalamic_relay_proxy": (
                "Thalamic relay proxy for rapid low-road transmission of salient cue information toward fear circuitry."
            ),
            "visual_cortex_proxy": (
                "Visual-cortex proxy for the high-road route that supports more detailed perceptual analysis before amygdala recruitment."
            ),
            "hypothalamus_proxy": (
                "Hypothalamic proxy for autonomic and neuroendocrine mobilization during fight-or-flight states."
            ),
            "locus_coeruleus_proxy": (
                "Locus-coeruleus / brainstem proxy for noradrenergic arousal, vigilance, and sympathetic activation."
            ),
        }
        self.proxy_region_keys = {
            "medial_prefrontal_control_proxy",
            "thalamic_relay_proxy",
            "visual_cortex_proxy",
            "hypothalamus_proxy",
            "locus_coeruleus_proxy",
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Heritable liability to anxiety, fear reactivity, serotonergic dysregulation, and phobic vulnerability."
            ),
            "traumatic_fear_learning_load": (
                "Conditioned fear learning from direct aversive experience linked to a specific object or situation."
            ),
            "chronic_stress_sensitization": (
                "Ongoing stress burden that sensitizes fear, autonomic, and neuroendocrine systems."
            ),
            "phobic_cue_exposure": (
                "Current exposure intensity to the feared cue, object, or situation."
            ),
            "panic_biological_susceptibility": (
                "Inherited or trait-like vulnerability to panic-like physiological surges during fear states."
            ),
            "blood_injection_injury_vasovagal_liability": (
                "Subtype-specific tendency toward vasovagal autonomic reactions and fainting in blood-injection-injury contexts."
            ),
            "safety_learning_support": (
                "Protective safety learning, coping structure, and regulatory support that reduce phobic expression."
            ),
            "serotonergic_treatment_support": (
                "Protective serotonergic treatment support that can reduce fear-circuit dysregulation and hyperarousal."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_dysregulation": (
                "Reduced serotonergic regulation of mood, fear, and top-down control."
            ),
            "crf_stress_amplification": (
                "CRF-linked stress amplification that heightens arousal and anxiety-like responding."
            ),
            "cck_acute_fear_susceptibility": (
                "CCK-linked acute fear and panic susceptibility during phobic challenge."
            ),
            "dopamine_avoidance_reinforcement": (
                "Avoidance learning and reinforcement of escape behavior through relief-driven motivational mechanisms."
            ),
            "contextual_fear_generalization": (
                "Generalization of conditioned fear across contexts and situations linked to memory systems."
            ),
            "prefrontal_fear_regulation_failure": (
                "Failure of cortical regulation to override amygdala-centered alarm responses."
            ),
            "amygdala_threat_bias": (
                "Hyper-reactive threat valuation bias toward phobic stimuli."
            ),
            "autonomic_fight_flight_activation": (
                "Sympathetic fight-or-flight mobilization underlying palpitations, sweating, trembling, and dyspnea."
            ),
            "interoceptive_alarm_gain": (
                "Amplified tracking of bodily alarm signals that magnifies fear experience."
            ),
            "vasovagal_response_tendency": (
                "Diphasic autonomic tendency toward fainting-prone vasovagal reactions in susceptible phobia subtypes."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "overwhelming_fear": "Intense fear or terror in response to the phobic stimulus.",
            "autonomic_hyperarousal": "Palpitations, sweating, trembling, dyspnea, dizziness, or similar fight-or-flight symptoms.",
            "panic_like_somatic_alarm": "Acute panic-like bodily alarm with strong interoceptive distress.",
            "anticipatory_anxiety": "Persistent anticipatory worry about future phobic encounters.",
            "avoidance": "Avoidance or escape from the feared object, situation, or context.",
            "context_bound_triggering": "Triggering of fear by specific learned contexts or reminders.",
            "vasovagal_fainting": "Fainting-prone response especially relevant to blood-injection-injury phobia.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_dysregulation",
                "relation": "heritable vulnerability can bias serotonergic regulation of anxiety and fear",
                "specific_phobia_change": "increased",
            },
            {
                "source": "serotonergic_treatment_support",
                "target": "serotonergic_dysregulation",
                "relation": "serotonergic treatment can reduce fear-circuit dysregulation",
                "specific_phobia_change": "decreased",
            },
            {
                "source": "chronic_stress_sensitization",
                "target": "crf_stress_amplification",
                "relation": "stress load recruits CRF-linked stress systems and amplifies anxiety",
                "specific_phobia_change": "increased",
            },
            {
                "source": "panic_biological_susceptibility",
                "target": "cck_acute_fear_susceptibility",
                "relation": "panic-prone biology increases acute CCK-like fear reactivity",
                "specific_phobia_change": "increased",
            },
            {
                "source": "traumatic_fear_learning_load",
                "target": "contextual_fear_generalization",
                "relation": "traumatic conditioning strengthens context-linked fear memory",
                "specific_phobia_change": "increased",
            },
            {
                "source": "traumatic_fear_learning_load",
                "target": "dopamine_avoidance_reinforcement",
                "relation": "relief from escape reinforces future avoidance learning",
                "specific_phobia_change": "increased",
            },
            {
                "source": "phobic_cue_exposure",
                "target": "amygdala_threat_bias",
                "relation": "current feared cues provoke rapid amygdala-centered threat evaluation",
                "specific_phobia_change": "increased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "prefrontal_fear_regulation_failure",
                "relation": "reduced serotonergic regulation weakens top-down fear control",
                "specific_phobia_change": "increased",
            },
            {
                "source": "safety_learning_support",
                "target": "prefrontal_fear_regulation_failure",
                "relation": "safety learning and coping support improve cortical regulation of fear",
                "specific_phobia_change": "decreased",
            },
            {
                "source": "crf_stress_amplification",
                "target": "autonomic_fight_flight_activation",
                "relation": "stress amplification increases autonomic mobilization",
                "specific_phobia_change": "increased",
            },
            {
                "source": "cck_acute_fear_susceptibility",
                "target": "autonomic_fight_flight_activation",
                "relation": "acute panic susceptibility intensifies bodily alarm",
                "specific_phobia_change": "increased",
            },
            {
                "source": "autonomic_fight_flight_activation",
                "target": "interoceptive_alarm_gain",
                "relation": "intense bodily arousal is registered as alarm by interoceptive systems",
                "specific_phobia_change": "increased",
            },
            {
                "source": "blood_injection_injury_vasovagal_liability",
                "target": "vasovagal_response_tendency",
                "relation": "familial blood-injection-injury liability predisposes to vasovagal reactions",
                "specific_phobia_change": "increased",
            },
            {
                "source": "amygdala_threat_bias",
                "target": "amygdala",
                "relation": "threat bias burdens the amygdala",
                "specific_phobia_change": "increased",
            },
            {
                "source": "contextual_fear_generalization",
                "target": "hippocampus",
                "relation": "contextual fear memory and retrieval burden the hippocampus",
                "specific_phobia_change": "increased",
            },
            {
                "source": "interoceptive_alarm_gain",
                "target": "insula",
                "relation": "bodily alarm monitoring burdens the insula",
                "specific_phobia_change": "increased",
            },
            {
                "source": "interoceptive_alarm_gain",
                "target": "acc",
                "relation": "interoceptive distress and conflict burden the anterior cingulate",
                "specific_phobia_change": "increased",
            },
            {
                "source": "prefrontal_fear_regulation_failure",
                "target": "medial_prefrontal_control_proxy",
                "relation": "fear-regulation failure burdens prefrontal control systems",
                "specific_phobia_change": "increased",
            },
            {
                "source": "phobic_cue_exposure",
                "target": "thalamic_relay_proxy",
                "relation": "rapid low-road cue transmission recruits thalamic relay systems",
                "specific_phobia_change": "increased",
            },
            {
                "source": "phobic_cue_exposure",
                "target": "visual_cortex_proxy",
                "relation": "detailed perceptual processing of the feared cue recruits visual cortex",
                "specific_phobia_change": "increased",
            },
            {
                "source": "autonomic_fight_flight_activation",
                "target": "hypothalamus_proxy",
                "relation": "fight-or-flight activation burdens hypothalamic output systems",
                "specific_phobia_change": "increased",
            },
            {
                "source": "autonomic_fight_flight_activation",
                "target": "locus_coeruleus_proxy",
                "relation": "noradrenergic vigilance and sympathetic arousal burden locus-coeruleus systems",
                "specific_phobia_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "overwhelming_fear",
                "relation": "amygdala alarm drives intense subjective fear",
                "specific_phobia_change": "increased",
            },
            {
                "source": "hypothalamus_proxy",
                "target": "autonomic_hyperarousal",
                "relation": "autonomic output contributes to bodily hyperarousal symptoms",
                "specific_phobia_change": "increased",
            },
            {
                "source": "locus_coeruleus_proxy",
                "target": "autonomic_hyperarousal",
                "relation": "noradrenergic arousal contributes to somatic alarm symptoms",
                "specific_phobia_change": "increased",
            },
            {
                "source": "insula",
                "target": "panic_like_somatic_alarm",
                "relation": "insula magnifies awareness of bodily alarm into panic-like distress",
                "specific_phobia_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "anticipatory_anxiety",
                "relation": "contextual retrieval of threat supports anticipatory anxiety",
                "specific_phobia_change": "increased",
            },
            {
                "source": "dopamine_avoidance_reinforcement",
                "target": "avoidance",
                "relation": "relief-driven reinforcement strengthens avoidance",
                "specific_phobia_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "context_bound_triggering",
                "relation": "fear contexts and reminders trigger phobic responses",
                "specific_phobia_change": "increased",
            },
            {
                "source": "visual_cortex_proxy",
                "target": "context_bound_triggering",
                "relation": "perceptual detail of the feared cue supports trigger recognition",
                "specific_phobia_change": "increased",
            },
            {
                "source": "vasovagal_response_tendency",
                "target": "vasovagal_fainting",
                "relation": "vasovagal liability produces fainting-prone episodes",
                "specific_phobia_change": "increased",
            },
            {
                "source": "safety_learning_support",
                "target": "avoidance",
                "relation": "safety learning can reduce avoidance expression",
                "specific_phobia_change": "decreased",
            },
            {
                "source": "serotonergic_treatment_support",
                "target": "autonomic_hyperarousal",
                "relation": "serotonergic support can reduce hyperarousal and anxiety burden",
                "specific_phobia_change": "decreased",
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
    def _reduce_numeric(value: Any) -> Optional[float]:
        """
        Convert a scalar-like, Series-like, or DataFrame-like connectivity lookup to one float.

        siibra connectivity matrices can occasionally yield duplicated labels after averaging
        compound features, so `.loc[row, col]` may return a scalar, Series, or DataFrame.
        This helper collapses any numeric result to a mean value and returns None when
        no numeric data are available.
        """
        if value is None:
            return None
        try:
            if isinstance(value, pd.DataFrame):
                numeric = value.apply(pd.to_numeric, errors="coerce")
                arr = numeric.to_numpy().ravel()
            elif isinstance(value, pd.Series):
                arr = pd.to_numeric(value, errors="coerce").to_numpy().ravel()
            else:
                try:
                    return float(value)
                except Exception:
                    arr = pd.to_numeric(pd.Series(list(value)), errors="coerce").to_numpy().ravel()
        except Exception:
            return None

        arr = [float(x) for x in arr if pd.notna(x)]
        if not arr:
            return None
        return float(sum(arr) / len(arr))

    def _canonicalize_connectivity_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize connectivity-matrix labels to region names and average duplicates.
        """
        if not isinstance(matrix, pd.DataFrame) or matrix.empty:
            return pd.DataFrame()

        df = matrix.copy()
        try:
            df.index = [self._name_of(x) for x in df.index]
        except Exception:
            df.index = [str(x) for x in df.index]
        try:
            df.columns = [self._name_of(x) for x in df.columns]
        except Exception:
            df.columns = [str(x) for x in df.columns]

        try:
            df = df.apply(pd.to_numeric, errors="coerce")
        except Exception:
            pass

        if not df.index.is_unique:
            df = df.groupby(level=0).mean(numeric_only=True)
        if not df.columns.is_unique:
            df = df.T.groupby(level=0).mean(numeric_only=True).T
        return df

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
            "insula",
            "cingulate cortex",
            "prefrontal cortex",
            "visual cortex",
            "thalamus",
        } else 0
        cyto_penalty = 0 if ("area " in name or "(" in name or "ca" in name or "subiculum" in name) else 1
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
        rows = []
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
            self._connectivity_matrix = self._canonicalize_connectivity_matrix(compound[0].data.copy())
            return self._connectivity_matrix
        except Exception:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = str(getattr(region, "name", region)).strip().lower()
        exact = [x for x in labels if self._name_of(x).strip().lower() == region_name]
        if exact:
            return exact[0]

        fuzzy = [
            x for x in labels
            if region_name in self._name_of(x).lower() or self._name_of(x).lower() in region_name
        ]
        return fuzzy[0] if fuzzy else None

    def _extract_connectivity_series(
        self,
        matrix: pd.DataFrame,
        label: Any,
        axis: str = "index",
    ) -> pd.Series:
        try:
            obj = matrix.loc[label] if axis == "index" else matrix[label]
        except Exception:
            return pd.Series(dtype=float)

        if isinstance(obj, pd.DataFrame):
            try:
                numeric = obj.apply(pd.to_numeric, errors="coerce")
                obj = numeric.mean(axis=0 if axis == "index" else 1)
            except Exception:
                return pd.Series(dtype=float)
        elif isinstance(obj, pd.Series):
            try:
                obj = pd.to_numeric(obj, errors="coerce")
            except Exception:
                return pd.Series(dtype=float)
        else:
            value = self._reduce_numeric(obj)
            if value is None:
                return pd.Series(dtype=float)
            obj = pd.Series({self._name_of(label): value}, dtype=float)

        return obj.dropna().astype(float)

    def _pairwise_connectivity_value(
        self,
        matrix: pd.DataFrame,
        source_label: Any,
        target_label: Any,
    ) -> Optional[float]:
        getters = [
            lambda: matrix.at[source_label, target_label],
            lambda: matrix.at[target_label, source_label],
            lambda: matrix.loc[source_label, target_label],
            lambda: matrix.loc[target_label, source_label],
            lambda: self._extract_connectivity_series(matrix, source_label, axis="index").get(target_label),
            lambda: self._extract_connectivity_series(matrix, source_label, axis="columns").get(target_label),
            lambda: self._extract_connectivity_series(matrix, target_label, axis="index").get(source_label),
            lambda: self._extract_connectivity_series(matrix, target_label, axis="columns").get(source_label),
        ]
        for getter in getters:
            try:
                value = getter()
            except Exception:
                continue
            scalar = self._reduce_numeric(value)
            if scalar is not None:
                return scalar
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
                series = self._extract_connectivity_series(matrix, label, axis=axis)
                if not series.empty:
                    df = series.sort_values(ascending=False).reset_index()
                    df.columns = ["connected_region", "value"]
                    df["connected_region"] = df["connected_region"].map(self._name_of)
                    region_name = str(getattr(region, "name", region))
                    df = df[df["connected_region"] != region_name].head(max_rows)
                    return df.reset_index(drop=True)

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
                value = self._pairwise_connectivity_value(matrix, s_label, t_label)
                if value is not None:
                    rows.append({"source": source, "target": target, "value": float(value)})

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
        genetic_vulnerability: float = 0.45,
        traumatic_fear_learning_load: float = 0.55,
        chronic_stress_sensitization: float = 0.40,
        phobic_cue_exposure: float = 0.70,
        panic_biological_susceptibility: float = 0.35,
        blood_injection_injury_vasovagal_liability: float = 0.10,
        safety_learning_support: float = 0.30,
        serotonergic_treatment_support: float = 0.15,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Inputs are clipped to [0, 1]. Protective variables are modeled as buffers on fear
        regulation and symptom expression rather than erasers of learned phobic circuitry.
        """
        inputs = {
            "genetic_vulnerability": self._clip01(genetic_vulnerability),
            "traumatic_fear_learning_load": self._clip01(traumatic_fear_learning_load),
            "chronic_stress_sensitization": self._clip01(chronic_stress_sensitization),
            "phobic_cue_exposure": self._clip01(phobic_cue_exposure),
            "panic_biological_susceptibility": self._clip01(panic_biological_susceptibility),
            "blood_injection_injury_vasovagal_liability": self._clip01(blood_injection_injury_vasovagal_liability),
            "safety_learning_support": self._clip01(safety_learning_support),
            "serotonergic_treatment_support": self._clip01(serotonergic_treatment_support),
        }

        protect_mean = self._clip01(
            (inputs["safety_learning_support"] + inputs["serotonergic_treatment_support"]) / 2.0
        )

        latents: Dict[str, float] = {}
        latents["serotonergic_dysregulation"] = self._clip01(
            0.40 * inputs["genetic_vulnerability"]
            + 0.25 * inputs["chronic_stress_sensitization"]
            + 0.20 * inputs["panic_biological_susceptibility"]
            + 0.15 * inputs["traumatic_fear_learning_load"]
            - 0.35 * inputs["serotonergic_treatment_support"]
            - 0.15 * inputs["safety_learning_support"]
        )
        latents["crf_stress_amplification"] = self._clip01(
            0.45 * inputs["chronic_stress_sensitization"]
            + 0.25 * inputs["phobic_cue_exposure"]
            + 0.15 * inputs["genetic_vulnerability"]
            + 0.15 * inputs["traumatic_fear_learning_load"]
        )
        latents["cck_acute_fear_susceptibility"] = self._clip01(
            0.55 * inputs["panic_biological_susceptibility"]
            + 0.25 * inputs["phobic_cue_exposure"]
            + 0.20 * inputs["genetic_vulnerability"]
        )
        latents["dopamine_avoidance_reinforcement"] = self._clip01(
            0.40 * inputs["traumatic_fear_learning_load"]
            + 0.25 * inputs["phobic_cue_exposure"]
            + 0.20 * latents["serotonergic_dysregulation"]
            + 0.15 * inputs["genetic_vulnerability"]
            - 0.15 * inputs["safety_learning_support"]
        )
        latents["contextual_fear_generalization"] = self._clip01(
            0.45 * inputs["traumatic_fear_learning_load"]
            + 0.25 * inputs["chronic_stress_sensitization"]
            + 0.15 * inputs["genetic_vulnerability"]
            + 0.15 * inputs["phobic_cue_exposure"]
            - 0.10 * inputs["safety_learning_support"]
        )
        latents["prefrontal_fear_regulation_failure"] = self._clip01(
            0.35 * latents["serotonergic_dysregulation"]
            + 0.25 * latents["crf_stress_amplification"]
            + 0.20 * inputs["phobic_cue_exposure"]
            + 0.20 * inputs["genetic_vulnerability"]
            - 0.25 * inputs["safety_learning_support"]
            - 0.20 * inputs["serotonergic_treatment_support"]
        )
        latents["amygdala_threat_bias"] = self._clip01(
            0.30 * inputs["traumatic_fear_learning_load"]
            + 0.20 * inputs["phobic_cue_exposure"]
            + 0.20 * latents["serotonergic_dysregulation"]
            + 0.15 * latents["crf_stress_amplification"]
            + 0.15 * latents["contextual_fear_generalization"]
        )
        latents["autonomic_fight_flight_activation"] = self._clip01(
            0.35 * latents["amygdala_threat_bias"]
            + 0.30 * latents["crf_stress_amplification"]
            + 0.20 * latents["cck_acute_fear_susceptibility"]
            + 0.15 * latents["prefrontal_fear_regulation_failure"]
            - 0.10 * inputs["serotonergic_treatment_support"]
        )
        latents["interoceptive_alarm_gain"] = self._clip01(
            0.40 * latents["autonomic_fight_flight_activation"]
            + 0.25 * inputs["panic_biological_susceptibility"]
            + 0.20 * latents["serotonergic_dysregulation"]
            + 0.15 * latents["cck_acute_fear_susceptibility"]
        )
        latents["vasovagal_response_tendency"] = self._clip01(
            0.60 * inputs["blood_injection_injury_vasovagal_liability"]
            + 0.20 * inputs["phobic_cue_exposure"]
            + 0.20 * inputs["panic_biological_susceptibility"]
        )

        regional_state: Dict[str, float] = {}
        regional_state["amygdala"] = self._clip01(
            0.45 * latents["amygdala_threat_bias"]
            + 0.20 * latents["autonomic_fight_flight_activation"]
            + 0.15 * inputs["phobic_cue_exposure"]
            + 0.10 * latents["serotonergic_dysregulation"]
            + 0.10 * latents["prefrontal_fear_regulation_failure"]
            - 0.10 * protect_mean
        )
        regional_state["hippocampus"] = self._clip01(
            0.45 * latents["contextual_fear_generalization"]
            + 0.20 * inputs["traumatic_fear_learning_load"]
            + 0.20 * inputs["chronic_stress_sensitization"]
            + 0.15 * latents["amygdala_threat_bias"]
            - 0.10 * inputs["safety_learning_support"]
        )
        regional_state["insula"] = self._clip01(
            0.45 * latents["interoceptive_alarm_gain"]
            + 0.25 * latents["autonomic_fight_flight_activation"]
            + 0.15 * inputs["phobic_cue_exposure"]
            + 0.15 * latents["cck_acute_fear_susceptibility"]
            - 0.10 * inputs["serotonergic_treatment_support"]
        )
        regional_state["acc"] = self._clip01(
            0.35 * latents["prefrontal_fear_regulation_failure"]
            + 0.25 * latents["interoceptive_alarm_gain"]
            + 0.20 * inputs["phobic_cue_exposure"]
            + 0.20 * latents["autonomic_fight_flight_activation"]
            - 0.10 * protect_mean
        )
        regional_state["medial_prefrontal_control_proxy"] = self._clip01(
            0.50 * latents["prefrontal_fear_regulation_failure"]
            + 0.20 * latents["serotonergic_dysregulation"]
            + 0.15 * inputs["chronic_stress_sensitization"]
            + 0.15 * latents["amygdala_threat_bias"]
            - 0.20 * inputs["safety_learning_support"]
            - 0.20 * inputs["serotonergic_treatment_support"]
        )
        regional_state["thalamic_relay_proxy"] = self._clip01(
            0.45 * inputs["phobic_cue_exposure"]
            + 0.30 * regional_state["amygdala"]
            + 0.15 * latents["cck_acute_fear_susceptibility"]
            + 0.10 * latents["autonomic_fight_flight_activation"]
        )
        regional_state["visual_cortex_proxy"] = self._clip01(
            0.40 * inputs["phobic_cue_exposure"]
            + 0.25 * inputs["traumatic_fear_learning_load"]
            + 0.20 * regional_state["amygdala"]
            + 0.15 * regional_state["hippocampus"]
        )
        regional_state["hypothalamus_proxy"] = self._clip01(
            0.50 * latents["autonomic_fight_flight_activation"]
            + 0.25 * regional_state["amygdala"]
            + 0.15 * latents["crf_stress_amplification"]
            + 0.10 * latents["cck_acute_fear_susceptibility"]
        )
        regional_state["locus_coeruleus_proxy"] = self._clip01(
            0.45 * latents["autonomic_fight_flight_activation"]
            + 0.25 * latents["crf_stress_amplification"]
            + 0.20 * latents["cck_acute_fear_susceptibility"]
            + 0.10 * latents["prefrontal_fear_regulation_failure"]
            - 0.10 * inputs["serotonergic_treatment_support"]
        )

        symptoms: Dict[str, float] = {}
        symptoms["overwhelming_fear"] = self._clip01(
            0.35 * regional_state["amygdala"]
            + 0.20 * regional_state["thalamic_relay_proxy"]
            + 0.20 * regional_state["hippocampus"]
            + 0.15 * regional_state["acc"]
            + 0.10 * regional_state["medial_prefrontal_control_proxy"]
            - 0.15 * inputs["safety_learning_support"]
            - 0.15 * inputs["serotonergic_treatment_support"]
        )
        symptoms["autonomic_hyperarousal"] = self._clip01(
            0.30 * regional_state["hypothalamus_proxy"]
            + 0.25 * regional_state["locus_coeruleus_proxy"]
            + 0.25 * regional_state["insula"]
            + 0.20 * latents["autonomic_fight_flight_activation"]
            - 0.10 * inputs["serotonergic_treatment_support"]
        )
        symptoms["panic_like_somatic_alarm"] = self._clip01(
            0.30 * regional_state["insula"]
            + 0.25 * regional_state["locus_coeruleus_proxy"]
            + 0.20 * latents["cck_acute_fear_susceptibility"]
            + 0.15 * regional_state["acc"]
            + 0.10 * inputs["panic_biological_susceptibility"]
            - 0.10 * inputs["serotonergic_treatment_support"]
        )
        symptoms["anticipatory_anxiety"] = self._clip01(
            0.35 * regional_state["hippocampus"]
            + 0.25 * regional_state["amygdala"]
            + 0.20 * regional_state["medial_prefrontal_control_proxy"]
            + 0.20 * latents["contextual_fear_generalization"]
            - 0.10 * inputs["safety_learning_support"]
            - 0.10 * inputs["serotonergic_treatment_support"]
        )
        symptoms["avoidance"] = self._clip01(
            0.35 * latents["dopamine_avoidance_reinforcement"]
            + 0.25 * symptoms["overwhelming_fear"]
            + 0.20 * symptoms["anticipatory_anxiety"]
            + 0.20 * regional_state["hippocampus"]
            - 0.20 * inputs["safety_learning_support"]
        )
        symptoms["context_bound_triggering"] = self._clip01(
            0.40 * regional_state["hippocampus"]
            + 0.25 * regional_state["visual_cortex_proxy"]
            + 0.20 * regional_state["amygdala"]
            + 0.15 * inputs["phobic_cue_exposure"]
            - 0.10 * inputs["safety_learning_support"]
        )
        symptoms["vasovagal_fainting"] = self._clip01(
            0.45 * latents["vasovagal_response_tendency"]
            + 0.20 * latents["autonomic_fight_flight_activation"]
            + 0.15 * regional_state["insula"]
            + 0.10 * inputs["phobic_cue_exposure"]
            + 0.10 * inputs["panic_biological_susceptibility"]
        )

        phenotypes: Dict[str, float] = {}
        phenotypes["acute_phobic_episode_profile"] = self._clip01(
            (
                symptoms["overwhelming_fear"]
                + symptoms["autonomic_hyperarousal"]
                + symptoms["panic_like_somatic_alarm"]
            ) / 3.0
        )
        phenotypes["avoidant_specific_phobia_profile"] = self._clip01(
            (
                symptoms["avoidance"]
                + symptoms["anticipatory_anxiety"]
                + symptoms["context_bound_triggering"]
            ) / 3.0
        )
        phenotypes["blood_injection_injury_profile"] = self._clip01(
            (
                symptoms["vasovagal_fainting"]
                + symptoms["overwhelming_fear"]
                + inputs["blood_injection_injury_vasovagal_liability"]
            ) / 3.0
        )
        phenotypes["panic_somatic_profile"] = self._clip01(
            (
                symptoms["panic_like_somatic_alarm"]
                + symptoms["autonomic_hyperarousal"]
                + regional_state["insula"]
                + regional_state["locus_coeruleus_proxy"]
            ) / 4.0
        )
        phenotypes["fear_control_imbalance"] = self._clip01(
            (
                regional_state["amygdala"]
                + regional_state["acc"]
                + regional_state["medial_prefrontal_control_proxy"]
            ) / 3.0
        )
        phenotypes["overall_specific_phobia_burden"] = self._clip01(
            0.30 * phenotypes["acute_phobic_episode_profile"]
            + 0.30 * phenotypes["avoidant_specific_phobia_profile"]
            + 0.20 * phenotypes["panic_somatic_profile"]
            + 0.10 * phenotypes["fear_control_imbalance"]
            + 0.10 * phenotypes["blood_injection_injury_profile"]
            - 0.10 * protect_mean
        )

        return {
            "inputs": pd.Series(inputs, dtype=float),
            "latents": pd.Series(latents, dtype=float),
            "regional_state": pd.Series(regional_state, dtype=float),
            "symptoms": pd.Series(symptoms, dtype=float),
            "phenotypes": pd.Series(phenotypes, dtype=float),
        }


if __name__ == "__main__":
    model = SpecificPhobiaModel()

    built = model.build()
    print("\n=== Nodes (first 12) ===")
    print(built["nodes"].head(12).to_string(index=False))

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
    for key in ["amygdala", "hippocampus", "insula", "acc"]:
        print(f"\nNode: {key}")
        print(f"  receptors rows: {len(built['receptors'].get(key, pd.DataFrame()))}")
        print(f"  genes rows: {len(built['genes'].get(key, pd.DataFrame()))}")
        print(f"  connectivity rows: {len(built['connectivity_profiles'].get(key, pd.DataFrame()))}")

    example = model.simulate(
        genetic_vulnerability=0.48,
        traumatic_fear_learning_load=0.72,
        chronic_stress_sensitization=0.46,
        phobic_cue_exposure=0.88,
        panic_biological_susceptibility=0.42,
        blood_injection_injury_vasovagal_liability=0.12,
        safety_learning_support=0.28,
        serotonergic_treatment_support=0.18,
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
    # print(model.assign_mni_point((-24, -6, -16)).head())  # amygdala-adjacent example
