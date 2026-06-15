from __future__ import annotations

"""
Pedophilic Disorder siibra scaffold.

This script turns a short Pedophilic Disorder chapter into an atlas-grounded,
mechanistic research scaffold using siibra. It is designed for exploratory
modeling, not diagnosis or treatment.

Chapter logic represented here:
- heterogeneity between a primary developmental form and a secondary acquired
  form linked to brain damage or neurodegeneration,
- shared neurodevelopmental vulnerability with ADHD/conduct/personality-related
  disinhibition and poor self-regulation,
- dopaminergic mesolimbic-prefrontal dysfunction as a possible contributor to
  impulse dyscontrol in a subset of cases,
- serotonergic modulation as an indirect pathway through mood/anxiety burden and
  SSRI-related libido/impulse effects,
- frontotemporal and frontolimbic circuitry involving prefrontal cortex,
  temporal lobe systems, hippocampus, and amygdala.

The scaffold is intentionally conservative. It does not claim a validated
parcel-level disease mechanism. Several nodes are explicit proxies because the
chapter is systems-level rather than cytoarchitectonically specific.

This scaffold concerns a harmful clinical condition. It is for research-style
mechanistic interpretation only and not for justification, operationalization,
or prediction of abusive behavior.
"""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


DEFAULT_GENE_PANEL = [
    "DRD4",    # ADHD / novelty and impulsivity relevance
    "DRD2",    # dopaminergic reward and control circuitry
    "SLC6A3",  # dopamine transporter
    "COMT",    # catecholamine metabolism / executive control
    "MAOA",    # monoamine metabolism / behavioral regulation
    "SLC6A4",  # serotonin transporter
    "HTR2A",   # serotonergic receptor / mood-impulse modulation
    "BDNF",    # neuroplasticity
    "OXTR",    # social cognition relevance
    "AR",      # androgen receptor / sexual differentiation pathway model
    "SRD5A2",  # androgen metabolism / sexual differentiation pathway model
    "NLGN3",   # neurodevelopmental / ASD-related exploratory marker
]


class PedophilicDisorderModel:
    """
    Atlas-grounded research scaffold for Pedophilic Disorder.

    Notes
    -----
    - This is a mechanistic interpretation of a chapter, not a validated disease model.
    - Direct molecular evidence is sparse; the gene panel is exploratory.
    - The chapter frames this disorder as heterogeneous, so the model separates
      developmental-patterning and acquired-disinhibition pathways.
    - Regional state scores reflect dysregulation burden rather than literal fMRI activation.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
        connectivity_subjects_to_average: int = 8,
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
        self.connectivity_subjects_to_average = max(1, int(connectivity_subjects_to_average))

        # Disorder worksheet distilled from the chapter.
        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Exploratory heritable liability mediated through impulsivity, social cognition, and behavioral regulation traits."
            ),
            "early_developmental_liability": (
                "Primary developmental vulnerability for an early-emerging atypical psychosexual pattern."
            ),
            "neurodevelopmental_comorbidity_load": (
                "Burden of ADHD, conduct, personality, and related neurodevelopmental-disinhibitory vulnerability."
            ),
            "mood_anxiety_burden": (
                "Comorbid affective burden relevant to serotonergic modulation and impulse control."
            ),
            "acquired_frontal_temporal_insult": (
                "Brain injury or acquired frontal-temporal dysfunction contributing to a possible secondary form."
            ),
            "neurodegenerative_burden": (
                "Cognitive decline or neurodegenerative change that may unmask late-onset disinhibition."
            ),
            "atypical_sexual_differentiation_load": (
                "Exploratory developmental variation in sexual differentiation pathways used only as a weak modeling hypothesis."
            ),
            "serotonergic_treatment_support": (
                "Indirect serotonergic modulation, such as treatment of comorbid mood/anxiety symptoms, that may reduce libido or improve control."
            ),
            "treatment_support": (
                "Protective treatment engagement, structure, and external support that reduce disinhibition risk."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "developmental_psychosexual_patterning": (
                "Primary developmental organization of atypical psychosexual interest patterns."
            ),
            "neurodevelopmental_disinhibition": (
                "ADHD/externalizing-style impulsivity and poor self-regulation burden."
            ),
            "frontotemporal_disinhibition": (
                "Acquired lesion or neurodegeneration-related breakdown of control, judgment, and sexual regulation."
            ),
            "mesolimbic_dopaminergic_dysregulation": (
                "Reward-motivation and prefrontal-mesolimbic dopaminergic dysregulation linked to impulsive acting on urges."
            ),
            "serotonergic_impulse_modulation_deficit": (
                "Reduced serotonergic support for affect regulation and behavioral restraint."
            ),
            "social_cognitive_regulation_impairment": (
                "Impaired social cognition, normative sexual-interest maintenance, and boundary judgment."
            ),
            "frontolimbic_control_failure": (
                "Insufficient top-down prefrontal regulation over temporal-limbic and motivational systems."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "late_onset_secondary_presentation": (
                "Secondary late-onset presentation linked to acquired brain damage or neurodegenerative change."
            ),
            "pedophilic_interest_pattern": (
                "Persistent atypical child-directed sexual-interest pattern as conceptualized in the chapter."
            ),
            "paraphilic_urge_salience": (
                "Current salience or motivational pull of paraphilic urges."
            ),
            "impaired_impulse_control_over_sexual_urges": (
                "Difficulty suppressing sexual urges once they are present."
            ),
            "social_judgment_impairment": (
                "Poor social and moral judgment relevant to boundary violation risk."
            ),
            "behavioral_disinhibition_risk": (
                "Risk that interest and urges are acted upon under conditions of impaired control and judgment."
            ),
        }

        self.region_candidates: Dict[str, List[str]] = {
            "pfc_control_proxy": [
                "Area 9/46d (DLPFC) left",
                "Area 9/46v (DLPFC) left",
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Area Fp2 (FPole) left",
                "prefrontal cortex",
                "dorsolateral prefrontal",
                "orbitofrontal",
                "frontopolar",
            ],
            "temporal_lobe_proxy": [
                "Area TGd (Temporal pole) left",
                "Area TGv (Temporal pole) left",
                "Area TE1.0 (TE) left",
                "Area TE2.1 (TE) left",
                "temporal pole",
                "temporal",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "DG (Hippocampus) left",
                "hippocampus",
            ],
            "ventral_striatum_proxy": [
                "ventral striatum",
                "nucleus accumbens",
                "BST (Bed Nucleus) left",
                "bed nucleus",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "pfc_control_proxy": (
                "Prefrontal executive-control and judgment proxy capturing systems-level frontal-lobe claims."
            ),
            "temporal_lobe_proxy": (
                "Temporal-lobe proxy for social cognition, sexual-interest maintenance, and acquired frontotemporal change."
            ),
            "amygdala": (
                "Limbic salience and affective-reactivity node within the frontotemporal network."
            ),
            "hippocampus": (
                "Memory-context node for temporal-limbic contribution to sexual-interest organization and regulation."
            ),
            "ventral_striatum_proxy": (
                "Mesolimbic reward-salience proxy for dopaminergic motivation and impulse pressure."
            ),
        }
        self.proxy_regions = {"pfc_control_proxy", "temporal_lobe_proxy", "ventral_striatum_proxy"}

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "early_developmental_liability",
                "target": "developmental_psychosexual_patterning",
                "relation": "supports the primary developmental form with early-emerging atypical psychosexual organization",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "atypical_sexual_differentiation_load",
                "target": "developmental_psychosexual_patterning",
                "relation": "serves as a weak exploratory developmental-pathway hypothesis rather than a direct cause",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "developmental_psychosexual_patterning",
                "relation": "may contribute indirectly through heritable developmental liabilities",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "neurodevelopmental_disinhibition",
                "relation": "loads heritable impulsivity and behavioral regulation vulnerability",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "neurodevelopmental_comorbidity_load",
                "target": "neurodevelopmental_disinhibition",
                "relation": "ADHD/externalizing-style comorbidity raises impulsivity and poor self-regulation",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "mood_anxiety_burden",
                "target": "serotonergic_impulse_modulation_deficit",
                "relation": "affective burden engages serotonergic pathways relevant to libido and control",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "serotonergic_treatment_support",
                "target": "serotonergic_impulse_modulation_deficit",
                "relation": "serotonergic treatment may indirectly reduce urge expression and improve overall restraint",
                "pedophilic_disorder_change": "decreased",
            },
            {
                "source": "acquired_frontal_temporal_insult",
                "target": "frontotemporal_disinhibition",
                "relation": "acquired brain damage can produce a secondary disinhibited presentation",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "neurodegenerative_burden",
                "target": "frontotemporal_disinhibition",
                "relation": "cognitive decline may unmask or create late-onset paraphilic behavior",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "neurodevelopmental_disinhibition",
                "target": "mesolimbic_dopaminergic_dysregulation",
                "relation": "shared ADHD-like control deficits are linked to mesolimbic-prefrontal dopaminergic vulnerability",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "frontotemporal_disinhibition",
                "target": "social_cognitive_regulation_impairment",
                "relation": "frontotemporal breakdown weakens social judgment and maintenance of normative sexual interests",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "developmental_psychosexual_patterning",
                "target": "social_cognitive_regulation_impairment",
                "relation": "developmental patterning can coexist with atypical social-sexual organization",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "neurodevelopmental_disinhibition",
                "target": "social_cognitive_regulation_impairment",
                "relation": "developmental regulation problems can impair boundary and social-cognitive processing",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "neurodevelopmental_disinhibition",
                "target": "frontolimbic_control_failure",
                "relation": "poor self-regulation undermines top-down control over urges",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "frontotemporal_disinhibition",
                "target": "frontolimbic_control_failure",
                "relation": "acquired frontal-temporal dysfunction weakens executive control and sexual regulation",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "serotonergic_impulse_modulation_deficit",
                "target": "frontolimbic_control_failure",
                "relation": "weaker serotonergic affect control reduces restraint over urges",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "treatment_support",
                "target": "frontolimbic_control_failure",
                "relation": "supportive treatment and external structure reduce disinhibition",
                "pedophilic_disorder_change": "decreased",
            },
            {
                "source": "frontolimbic_control_failure",
                "target": "pfc_control_proxy",
                "relation": "control failure is expressed through prefrontal executive and judgment dysfunction",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "social_cognitive_regulation_impairment",
                "target": "temporal_lobe_proxy",
                "relation": "temporal-lobe systems carry social-cognitive and sexual-interest maintenance burden",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "frontolimbic_control_failure",
                "target": "amygdala",
                "relation": "reduced top-down regulation permits stronger limbic salience and affective reactivity",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "social_cognitive_regulation_impairment",
                "target": "hippocampus",
                "relation": "temporal-limbic memory-context systems participate in interest organization and regulation",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "mesolimbic_dopaminergic_dysregulation",
                "target": "ventral_striatum_proxy",
                "relation": "mesolimbic reward-salience circuitry contributes to urge-driven motivation",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "frontotemporal_disinhibition",
                "target": "late_onset_secondary_presentation",
                "relation": "secondary late-onset cases emerge from acquired or degenerative frontal-temporal dysfunction",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "pfc_control_proxy",
                "target": "late_onset_secondary_presentation",
                "relation": "prefrontal disinhibition contributes to behavioral change in secondary cases",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "developmental_psychosexual_patterning",
                "target": "pedophilic_interest_pattern",
                "relation": "primary developmental organization contributes to a stable atypical interest pattern",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "late_onset_secondary_presentation",
                "target": "pedophilic_interest_pattern",
                "relation": "secondary acquired forms can create or unmask abnormal sexual interest patterns",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "pedophilic_interest_pattern",
                "relation": "temporal-limbic memory-context systems are hypothesized to help maintain the interest pattern",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "ventral_striatum_proxy",
                "target": "paraphilic_urge_salience",
                "relation": "reward-salience circuitry increases motivational pull of urges",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "pedophilic_interest_pattern",
                "target": "paraphilic_urge_salience",
                "relation": "stable interest pattern increases salience of related urges",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "paraphilic_urge_salience",
                "relation": "limbic salience and affective tagging may intensify the urgency of impulses",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "pfc_control_proxy",
                "target": "impaired_impulse_control_over_sexual_urges",
                "relation": "prefrontal executive dysfunction weakens resistance once urges are present",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "frontolimbic_control_failure",
                "target": "impaired_impulse_control_over_sexual_urges",
                "relation": "frontolimbic dyscontrol undermines suppression of paraphilic urges",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "serotonergic_treatment_support",
                "target": "impaired_impulse_control_over_sexual_urges",
                "relation": "indirect serotonergic treatment may improve overall impulse control",
                "pedophilic_disorder_change": "decreased",
            },
            {
                "source": "temporal_lobe_proxy",
                "target": "social_judgment_impairment",
                "relation": "temporal-lobe dysfunction weakens social cognition and normative judgment",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "social_cognitive_regulation_impairment",
                "target": "social_judgment_impairment",
                "relation": "impaired social cognition compromises judgment and boundary awareness",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "impaired_impulse_control_over_sexual_urges",
                "target": "behavioral_disinhibition_risk",
                "relation": "poor urge inhibition raises risk of harmful enactment",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "social_judgment_impairment",
                "target": "behavioral_disinhibition_risk",
                "relation": "impaired judgment increases risk of acting despite consequences and norms",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "paraphilic_urge_salience",
                "target": "behavioral_disinhibition_risk",
                "relation": "high urge salience increases pressure toward action in vulnerable individuals",
                "pedophilic_disorder_change": "increased",
            },
            {
                "source": "treatment_support",
                "target": "behavioral_disinhibition_risk",
                "relation": "external support and treatment structure reduce enactment risk",
                "pedophilic_disorder_change": "decreased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame(self.edge_table)
        self._pmap = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _normalize_label_text(text: Any) -> str:
        s = str(text).lower().strip()
        for token in [
            " left",
            " right",
            " (pacc)",
            " (acc)",
            " (ofc)",
            " (dlpfc)",
            " (fpole)",
            " (temporal pole)",
            " (hippocampus)",
            " (amygdala)",
        ]:
            s = s.replace(token, "")
        return " ".join(s.split())

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _modality_candidates(self, kind: str) -> List[Any]:
        candidates: List[Any] = []
        try:
            if kind == "receptor":
                candidates.append(siibra.features.molecular.ReceptorDensityFingerprint)
            elif kind == "gene":
                candidates.append(siibra.features.molecular.GeneExpressions)
            elif kind == "connectivity":
                candidates.append(siibra.features.connectivity.StreamlineCounts)
        except Exception:
            pass

        if kind == "receptor":
            candidates.append("receptor density fingerprint")
        elif kind == "gene":
            candidates.append("gene expressions")
        elif kind == "connectivity":
            candidates.append("StreamlineCounts")
        return candidates

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
            try:
                matches = list(self.parcellation.find(query))
            except Exception:
                matches = []

        out: List[Any] = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower() or self._name_of(self.parcellation) == str(parc_name):
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "temporal lobe",
            "temporal pole",
            "prefrontal cortex",
            "ventral striatum",
            "orbitofrontal cortex",
            "dorsolateral prefrontal cortex",
        } else 0
        proxy_penalty = 1 if "bed nucleus" in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation)
            except Exception:
                pass
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation_spec)
            except Exception:
                pass
            try:
                if hasattr(self.parcellation, "get_region"):
                    return self.parcellation.get_region(spec)
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

    def _point_to_tuple(self, point: Any) -> Optional[Tuple[float, float, float]]:
        if point is None:
            return None
        for attr in ("coordinate", "coordinates", "xyz"):
            value = getattr(point, attr, None)
            if value is not None:
                try:
                    return tuple(float(x) for x in value)  # type: ignore[arg-type]
                except Exception:
                    pass
        try:
            return tuple(float(x) for x in point)  # type: ignore[arg-type]
        except Exception:
            return None

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid_xyz = self._point_to_tuple(getattr(main, "centroid", None))
        try:
            volume_mm3 = float(getattr(main, "volume", float("nan")))
        except Exception:
            volume_mm3 = None
        return centroid_xyz, volume_mm3

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            try:
                df = feat.data.copy()
                if isinstance(df, pd.Series):
                    df = df.to_frame(name="value")
                if not isinstance(df, pd.DataFrame):
                    continue
                df = df.reset_index()
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
                        .reset_index(drop=True)
                    )
                except Exception:
                    pass
            return df.reset_index(drop=True)
        return pd.DataFrame()

    def _first_dataframe(self, value: Any) -> Optional[pd.DataFrame]:
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, pd.Series):
            return value.to_frame()
        return None

    def _mean_connectivity_from_compound(self, compound: Any) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        try:
            direct = self._first_dataframe(getattr(compound, "data", None))
            if direct is not None:
                return direct
        except Exception:
            pass

        try:
            iterable: Iterable[Any] = compound[: self.connectivity_subjects_to_average]
        except Exception:
            iterable = []
            try:
                iterable = [compound[i] for i in range(self.connectivity_subjects_to_average)]
            except Exception:
                iterable = []

        for element in iterable:
            try:
                df = self._first_dataframe(getattr(element, "data", None))
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0]

        try:
            total = frames[0].astype(float).copy()
            for df in frames[1:]:
                total = total.add(df.astype(float), fill_value=0.0)
            return total / float(len(frames))
        except Exception:
            return frames[0]

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort),
            feats[0],
        )
        self._connectivity_matrix = self._mean_connectivity_from_compound(compound)
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._name_of(region)
        region_norm = self._normalize_label_text(region_name)

        for label in labels:
            if self._name_of(label) == region_name:
                return label

        exactish = []
        fuzzy = []
        for label in labels:
            label_name = self._name_of(label)
            label_norm = self._normalize_label_text(label_name)
            if label_norm == region_norm:
                exactish.append(label)
            elif region_norm in label_norm or label_norm in region_norm:
                fuzzy.append(label)

        pool = exactish if exactish else fuzzy
        if not pool:
            return None
        return sorted(pool, key=lambda x: self._region_rank(x))[0]

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        row_label = self._match_region_label(list(matrix.index), region)
        col_label = self._match_region_label(list(matrix.columns), region)
        label = row_label if row_label is not None else col_label
        axis = "index" if row_label is not None else "columns"
        if label is None:
            return pd.DataFrame()

        try:
            series = matrix.loc[label] if axis == "index" else matrix[label]
            if not isinstance(series, pd.Series):
                return pd.DataFrame()
            df = series.dropna().sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != self._name_of(region)].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        pairs: List[Tuple[str, Any]] = []
        for key, region in self.region_objects.items():
            label = self._match_region_label(list(matrix.index), region)
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
            if label is not None:
                pairs.append((key, label))

        if not pairs:
            return pd.DataFrame()

        labels = [label for _, label in pairs]
        keys = [key for key, _ in pairs]
        try:
            sub = matrix.loc[labels, labels].copy()
            sub.index = keys
            sub.columns = keys
            return sub
        except Exception:
            return pd.DataFrame()

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
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
                    "region_role": None,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            description = self.region_node_descriptions.get(key, "Atlas-backed circuit node")
            region_role = "proxy" if key in self.proxy_regions else "atlas_backed"

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
                        "description": f"{description} (unresolved in this environment)",
                        "region_role": region_role,
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
                    "node_type": "region",
                    "description": description,
                    "region_role": region_role,
                    "atlas_region": self._name_of(region),
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
                    "region_role": None,
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
                    "region_role": None,
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
        genetic_vulnerability: float = 0.50,
        early_developmental_liability: float = 0.50,
        neurodevelopmental_comorbidity_load: float = 0.50,
        mood_anxiety_burden: float = 0.50,
        acquired_frontal_temporal_insult: float = 0.20,
        neurodegenerative_burden: float = 0.20,
        atypical_sexual_differentiation_load: float = 0.20,
        serotonergic_treatment_support: float = 0.50,
        treatment_support: float = 0.50,
    ) -> Dict[str, pd.Series]:
        """
        Transparent one-pass simulator on normalized 0..1 inputs.

        Order:
        inputs -> latent biology -> regional state -> symptoms -> phenotype summaries

        These are scaffold scores, not probabilities, diagnoses, or treatment recommendations.
        """
        c = self._clip01

        inputs = pd.Series(
            {
                "genetic_vulnerability": c(genetic_vulnerability),
                "early_developmental_liability": c(early_developmental_liability),
                "neurodevelopmental_comorbidity_load": c(neurodevelopmental_comorbidity_load),
                "mood_anxiety_burden": c(mood_anxiety_burden),
                "acquired_frontal_temporal_insult": c(acquired_frontal_temporal_insult),
                "neurodegenerative_burden": c(neurodegenerative_burden),
                "atypical_sexual_differentiation_load": c(atypical_sexual_differentiation_load),
                "serotonergic_treatment_support": c(serotonergic_treatment_support),
                "treatment_support": c(treatment_support),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["developmental_psychosexual_patterning"] = c(
            0.38 * inputs["early_developmental_liability"]
            + 0.20 * inputs["atypical_sexual_differentiation_load"]
            + 0.18 * inputs["genetic_vulnerability"]
            + 0.12 * inputs["neurodevelopmental_comorbidity_load"]
        )
        latents["neurodevelopmental_disinhibition"] = c(
            0.40 * inputs["neurodevelopmental_comorbidity_load"]
            + 0.24 * inputs["genetic_vulnerability"]
            + 0.14 * inputs["early_developmental_liability"]
            + 0.08 * inputs["mood_anxiety_burden"]
            - 0.10 * inputs["treatment_support"]
        )
        latents["frontotemporal_disinhibition"] = c(
            0.42 * inputs["acquired_frontal_temporal_insult"]
            + 0.34 * inputs["neurodegenerative_burden"]
            + 0.10 * inputs["mood_anxiety_burden"]
            - 0.10 * inputs["treatment_support"]
        )
        latents["mesolimbic_dopaminergic_dysregulation"] = c(
            0.40 * latents["neurodevelopmental_disinhibition"]
            + 0.20 * inputs["genetic_vulnerability"]
            + 0.16 * latents["frontotemporal_disinhibition"]
            + 0.10 * inputs["mood_anxiety_burden"]
            - 0.08 * inputs["treatment_support"]
        )
        latents["serotonergic_impulse_modulation_deficit"] = c(
            0.42 * inputs["mood_anxiety_burden"]
            + 0.18 * latents["neurodevelopmental_disinhibition"]
            + 0.12 * inputs["neurodegenerative_burden"]
            - 0.26 * inputs["serotonergic_treatment_support"]
            - 0.08 * inputs["treatment_support"]
        )
        latents["social_cognitive_regulation_impairment"] = c(
            0.32 * latents["developmental_psychosexual_patterning"]
            + 0.28 * latents["neurodevelopmental_disinhibition"]
            + 0.28 * latents["frontotemporal_disinhibition"]
            + 0.08 * inputs["mood_anxiety_burden"]
            - 0.12 * inputs["treatment_support"]
        )
        latents["frontolimbic_control_failure"] = c(
            0.34 * latents["neurodevelopmental_disinhibition"]
            + 0.28 * latents["frontotemporal_disinhibition"]
            + 0.20 * latents["serotonergic_impulse_modulation_deficit"]
            + 0.12 * latents["mesolimbic_dopaminergic_dysregulation"]
            - 0.20 * inputs["treatment_support"]
            - 0.08 * inputs["serotonergic_treatment_support"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["pfc_control_proxy"] = c(
            0.56 * latents["frontolimbic_control_failure"]
            + 0.18 * latents["frontotemporal_disinhibition"]
            + 0.12 * latents["neurodevelopmental_disinhibition"]
            - 0.18 * inputs["treatment_support"]
            - 0.08 * inputs["serotonergic_treatment_support"]
        )
        regional_state["temporal_lobe_proxy"] = c(
            0.42 * latents["social_cognitive_regulation_impairment"]
            + 0.28 * latents["frontotemporal_disinhibition"]
            + 0.16 * latents["developmental_psychosexual_patterning"]
        )
        regional_state["amygdala"] = c(
            0.40 * latents["frontolimbic_control_failure"]
            + 0.22 * latents["serotonergic_impulse_modulation_deficit"]
            + 0.16 * inputs["mood_anxiety_burden"]
            + 0.10 * latents["social_cognitive_regulation_impairment"]
        )
        regional_state["hippocampus"] = c(
            0.34 * latents["developmental_psychosexual_patterning"]
            + 0.24 * latents["social_cognitive_regulation_impairment"]
            + 0.20 * latents["frontotemporal_disinhibition"]
            + 0.10 * inputs["mood_anxiety_burden"]
        )
        regional_state["ventral_striatum_proxy"] = c(
            0.54 * latents["mesolimbic_dopaminergic_dysregulation"]
            + 0.16 * latents["neurodevelopmental_disinhibition"]
            + 0.10 * latents["frontolimbic_control_failure"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["late_onset_secondary_presentation"] = c(
            0.50 * latents["frontotemporal_disinhibition"]
            + 0.24 * regional_state["pfc_control_proxy"]
            + 0.18 * regional_state["temporal_lobe_proxy"]
            + 0.08 * inputs["neurodegenerative_burden"]
        )
        symptoms["pedophilic_interest_pattern"] = c(
            0.42 * latents["developmental_psychosexual_patterning"]
            + 0.22 * latents["social_cognitive_regulation_impairment"]
            + 0.16 * regional_state["hippocampus"]
            + 0.12 * symptoms["late_onset_secondary_presentation"]
        )
        symptoms["paraphilic_urge_salience"] = c(
            0.34 * symptoms["pedophilic_interest_pattern"]
            + 0.24 * regional_state["ventral_striatum_proxy"]
            + 0.18 * regional_state["amygdala"]
            + 0.12 * latents["mesolimbic_dopaminergic_dysregulation"]
            - 0.10 * inputs["serotonergic_treatment_support"]
        )
        symptoms["social_judgment_impairment"] = c(
            0.34 * latents["social_cognitive_regulation_impairment"]
            + 0.24 * regional_state["temporal_lobe_proxy"]
            + 0.20 * regional_state["pfc_control_proxy"]
            + 0.10 * latents["frontotemporal_disinhibition"]
            - 0.14 * inputs["treatment_support"]
        )
        symptoms["impaired_impulse_control_over_sexual_urges"] = c(
            0.34 * latents["frontolimbic_control_failure"]
            + 0.24 * regional_state["pfc_control_proxy"]
            + 0.18 * regional_state["ventral_striatum_proxy"]
            + 0.10 * latents["serotonergic_impulse_modulation_deficit"]
            + 0.08 * symptoms["paraphilic_urge_salience"]
            - 0.14 * inputs["treatment_support"]
            - 0.10 * inputs["serotonergic_treatment_support"]
        )
        symptoms["behavioral_disinhibition_risk"] = c(
            0.28 * symptoms["impaired_impulse_control_over_sexual_urges"]
            + 0.24 * symptoms["social_judgment_impairment"]
            + 0.20 * symptoms["paraphilic_urge_salience"]
            + 0.16 * symptoms["late_onset_secondary_presentation"]
            - 0.18 * inputs["treatment_support"]
            - 0.08 * inputs["serotonergic_treatment_support"]
        )

        phenotypes = pd.Series(dtype=float, name="phenotypes")
        phenotypes["developmental_primary_profile"] = c(
            self._mean(
                [
                    latents["developmental_psychosexual_patterning"],
                    symptoms["pedophilic_interest_pattern"],
                    symptoms["paraphilic_urge_salience"],
                ]
            )
        )
        phenotypes["acquired_secondary_profile"] = c(
            self._mean(
                [
                    latents["frontotemporal_disinhibition"],
                    symptoms["late_onset_secondary_presentation"],
                    symptoms["social_judgment_impairment"],
                    symptoms["behavioral_disinhibition_risk"],
                ]
            )
        )
        phenotypes["neurodevelopmental_impulsivity_profile"] = c(
            self._mean(
                [
                    latents["neurodevelopmental_disinhibition"],
                    latents["mesolimbic_dopaminergic_dysregulation"],
                    symptoms["impaired_impulse_control_over_sexual_urges"],
                ]
            )
        )
        phenotypes["interest_plus_disinhibition_profile"] = c(
            self._mean(
                [
                    symptoms["pedophilic_interest_pattern"],
                    symptoms["paraphilic_urge_salience"],
                    symptoms["impaired_impulse_control_over_sexual_urges"],
                    symptoms["behavioral_disinhibition_risk"],
                ]
            )
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

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in getattr(assignments, "columns", []):
                return assignments.sort_values(candidate, ascending=False).reset_index(drop=True)
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Optional[Any]:
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        try:
            return region.get_regional_mask(self.space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(self.space)
            except Exception:
                return None


if __name__ == "__main__":
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 180)

    model = PedophilicDisorderModel()
    built = model.build()

    print("\n=== Nodes ===")
    print(
        built["nodes"][
            [
                "key",
                "node_type",
                "region_role",
                "atlas_region",
                "feature_summary",
            ]
        ].to_string(index=False)
    )

    print("\n=== Edges ===")
    print(built["edges"].to_string(index=False))

    print("\n=== Region suggestions for 'temporal' ===")
    print(model.suggest_regions("temporal").head(10).to_string(index=False))

    for key in [
        "pfc_control_proxy",
        "temporal_lobe_proxy",
        "amygdala",
        "hippocampus",
        "ventral_striatum_proxy",
    ]:
        print(f"\n=== Region evidence: {key} ===")
        if key in built["regions"]:
            region = built["regions"][key]
            print("Resolved region:", getattr(region, "name", region))
        else:
            print("Resolved region: None")
        print("Receptors:")
        rec = built["receptors"].get(key, pd.DataFrame())
        print(rec.head(10).to_string(index=False) if not rec.empty else "<no receptor table>")
        print("Genes:")
        gene = built["genes"].get(key, pd.DataFrame())
        print(gene.head(10).to_string(index=False) if not gene.empty else "<no gene table>")
        print("Connectivity profile:")
        conn = built["connectivity_profiles"].get(key, pd.DataFrame())
        print(conn.head(10).to_string(index=False) if not conn.empty else "<no connectivity profile>")

    print("\n=== Circuit connectivity ===")
    circuit = built["circuit_connectivity"]
    print(circuit.to_string() if not circuit.empty else "<no circuit connectivity matrix>")

    sim = model.simulate(
        genetic_vulnerability=0.65,
        early_developmental_liability=0.70,
        neurodevelopmental_comorbidity_load=0.72,
        mood_anxiety_burden=0.55,
        acquired_frontal_temporal_insult=0.15,
        neurodegenerative_burden=0.10,
        atypical_sexual_differentiation_load=0.25,
        serotonergic_treatment_support=0.45,
        treatment_support=0.55,
    )

    print("\n=== Simulation: inputs ===")
    print(sim["inputs"].to_string())
    print("\n=== Simulation: latents ===")
    print(sim["latents"].to_string())
    print("\n=== Simulation: regional_state ===")
    print(sim["regional_state"].to_string())
    print("\n=== Simulation: symptoms ===")
    print(sim["symptoms"].to_string())
    print("\n=== Simulation: phenotypes ===")
    print(sim["phenotypes"].to_string())

    # Example coordinate assignment for interactive use:
    # print(model.assign_mni_point((0, 38, 18)).head())
