from __future__ import annotations

"""
Somatic Symptom Disorder siibra scaffold.

This script translates a chapter-level biological summary of Somatic Symptom
Disorder (SSD) into a transparent, atlas-grounded mechanistic scaffold using
siibra. It is intended for research exploration, education, and further model
refinement. It is not a validated disease model, and it must not be used for
clinical diagnosis, treatment selection, or risk prediction.

Design notes
------------
- The chapter is partly systems-level rather than parcel-precise. Interoceptive
  amplification and monoamine chemistry are therefore kept as latent biology
  nodes instead of being forced into unsupported anatomical precision.
- Higher values in ``regional_state`` reflect disorder-relevant burden,
  dysregulation, or maladaptive recruitment pressure rather than raw neural
  activation.
- The simulator is intentionally acyclic and normalized to 0..1 so the chapter's
  logic remains readable: inputs -> latent biology -> regional state -> symptoms
  -> phenotype summaries.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import siibra
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This scaffold requires siibra-python. Install it before running this script."
    ) from exc


DEFAULT_SSD_GENE_PANEL: List[str] = [
    "SLC6A4",
    "SLC6A2",
    "DRD2",
    "COMT",
    "BDNF",
    "NR3C1",
    "FKBP5",
    "CRHR1",
    "HTR1A",
    "HTR2A",
    "MAOA",
]


class SomaticSymptomDisorderModel:
    """
    Atlas-grounded research scaffold for Somatic Symptom Disorder.

    The model captures chapter-level claims linking genetic and temperamental
    vulnerability, anxiety-depression comorbidity, chronic stress, bodily-signal
    hypervigilance, frontolimbic threat dysregulation, impaired hippocampal
    contextualization, and fronto-striatal inhibition bias to persistent somatic
    distress, illness worry, catastrophic interpretation, fatigue/anhedonia, and
    conversion-like motor or sensory symptoms.
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

        # Conservative atlas anchors for the chapter's named circuits.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "Amygdala left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 left",
                "CA3 left",
                "DG left",
                "hippocampus left",
                "hippocampus",
            ],
            "pfc_control": [
                "Area MFG5 left",
                "MFG5 left",
                "Area MFG4 left",
                "MFG4 left",
                "Area MFG2 left",
                "MFG2 left",
                "Area MFG1 left",
                "MFG1 left",
                "Area SFG4 left",
                "SFG4 left",
                "Area SFS1 left",
                "SFS1 left",
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "Area 9 left",
                "middle frontal gyrus left",
                "dorsolateral prefrontal cortex left",
                "prefrontal cortex left",
                "prefrontal cortex",
                "PFC",
            ],
            "caudate_proxy": [
                "Body of Caudate Nucleus, caudate Nucleus (Basal Ganglia) left",
                "Fundus of Caudate Nucleus, ventral Striatum (Basal Ganglia) left",
                "Caudate Nucleus left",
                "caudate nucleus left",
                "dorsal striatum left",
                "caudate",
                "striatum",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_trait_liability": (
                "Inherited liability for mood and anxiety dysregulation as well as somatoform vulnerability"
            ),
            "neuroticism_anxious_temperament": (
                "Anxious worrying temperament and negative-affect sensitivity that promote bodily hypervigilance"
            ),
            "early_adversity_history": (
                "Earlier adversity that sensitizes later stress response and symptom appraisal"
            ),
            "chronic_stress_load": (
                "Sustained stress burden that maintains physiological arousal and symptom focus"
            ),
            "depression_anxiety_comorbidity": (
                "Concurrent depression and anxiety loading that shares neurobiology with SSD"
            ),
            "bodily_sensation_load": (
                "Intensity or persistence of bodily sensations and functional somatic complaints"
            ),
            "recovery_support": (
                "Protective support, reassurance, and effective care that reduce escalation of symptom focus"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "stress_system_sensitization": (
                "Stress-system reactivity bias shaped by genes, temperament, adversity, and chronic strain"
            ),
            "monoamine_regulatory_dysfunction": (
                "Shared serotonergic and noradrenergic dysregulation associated with anxiety/depression overlap"
            ),
            "dopaminergic_motivational_erosion": (
                "Motivational and energy-system burden contributing to fatigue and reduced interest"
            ),
            "interoceptive_hypervigilance": (
                "Excessive monitoring of bodily sensations and exaggerated attention to minor changes"
            ),
            "frontolimbic_threat_dysregulation": (
                "Amygdala-PFC imbalance that biases threat appraisal and weakens emotional control"
            ),
            "hippocampal_contextualization_failure": (
                "Reduced ability to contextualize bodily sensations and constrain fear generalization"
            ),
            "frontostriatal_action_inhibition_bias": (
                "Top-down inhibitory pressure on action and sensorimotor processing relevant to conversion-like symptoms"
            ),
            "somatic_signal_amplification": (
                "Amplification and maladaptive interpretation of bodily signals into persistent distress"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "persistent_somatic_distress": (
                "Persistent physical symptoms and distressing bodily complaints"
            ),
            "illness_worry": (
                "Excessive worry focused on bodily symptoms and possible disease"
            ),
            "catastrophic_health_interpretation": (
                "Catastrophic thinking and worst-case interpretation of bodily sensations"
            ),
            "fear_of_illness": (
                "Fearfulness and threat expectancy centered on being seriously ill"
            ),
            "fatigue_anhedonia": (
                "Profound fatigue, loss of interest, and motivational depletion"
            ),
            "conversion_like_motor_sensory_symptoms": (
                "Functional motor or sensory deficits without corresponding structural lesion"
            ),
        }

        self.edge_table: List[Dict[str, Any]] = [
            {
                "source": "genetic_trait_liability",
                "target": "stress_system_sensitization",
                "relation": "genetic vulnerability raises stress-reactivity liability",
                "ssd_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "genetic_trait_liability",
                "target": "monoamine_regulatory_dysfunction",
                "relation": "shared inherited mood and anxiety risk loads monoamine regulation",
                "ssd_change": "increased",
                "weight": 0.28,
            },
            {
                "source": "neuroticism_anxious_temperament",
                "target": "interoceptive_hypervigilance",
                "relation": "anxious worrying temperament heightens monitoring of bodily sensations",
                "ssd_change": "increased",
                "weight": 0.42,
            },
            {
                "source": "neuroticism_anxious_temperament",
                "target": "frontolimbic_threat_dysregulation",
                "relation": "negative-affect temperament biases threat appraisal",
                "ssd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "early_adversity_history",
                "target": "stress_system_sensitization",
                "relation": "earlier adversity sensitizes later stress-system responding",
                "ssd_change": "increased",
                "weight": 0.32,
            },
            {
                "source": "chronic_stress_load",
                "target": "stress_system_sensitization",
                "relation": "sustained stress amplifies reactivity and symptom persistence",
                "ssd_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "depression_anxiety_comorbidity",
                "target": "monoamine_regulatory_dysfunction",
                "relation": "anxiety and depression overlap implies shared monoamine dysregulation",
                "ssd_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "depression_anxiety_comorbidity",
                "target": "frontolimbic_threat_dysregulation",
                "relation": "comorbid affective disorders load emotional regulation circuits",
                "ssd_change": "increased",
                "weight": 0.28,
            },
            {
                "source": "bodily_sensation_load",
                "target": "interoceptive_hypervigilance",
                "relation": "more persistent bodily signals promote attention capture",
                "ssd_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "bodily_sensation_load",
                "target": "somatic_signal_amplification",
                "relation": "ongoing physical sensations feed symptom amplification",
                "ssd_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "recovery_support",
                "target": "stress_system_sensitization",
                "relation": "protective support buffers stress escalation",
                "ssd_change": "decreased",
                "weight": -0.18,
            },
            {
                "source": "recovery_support",
                "target": "somatic_signal_amplification",
                "relation": "effective support can reduce escalating symptom focus",
                "ssd_change": "decreased",
                "weight": -0.16,
            },
            {
                "source": "stress_system_sensitization",
                "target": "monoamine_regulatory_dysfunction",
                "relation": "stress burden worsens mood/anxiety-linked transmitter regulation",
                "ssd_change": "increased",
                "weight": 0.22,
            },
            {
                "source": "stress_system_sensitization",
                "target": "frontolimbic_threat_dysregulation",
                "relation": "stress sensitization destabilizes amygdala-PFC balance",
                "ssd_change": "increased",
                "weight": 0.32,
            },
            {
                "source": "stress_system_sensitization",
                "target": "hippocampal_contextualization_failure",
                "relation": "chronic stress impairs context processing and fear constraint",
                "ssd_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "stress_system_sensitization",
                "target": "frontostriatal_action_inhibition_bias",
                "relation": "stress can bias rigid top-down inhibitory responding",
                "ssd_change": "increased",
                "weight": 0.18,
            },
            {
                "source": "monoamine_regulatory_dysfunction",
                "target": "dopaminergic_motivational_erosion",
                "relation": "monoamine disruption contributes to fatigue and reduced motivation",
                "ssd_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "monoamine_regulatory_dysfunction",
                "target": "frontolimbic_threat_dysregulation",
                "relation": "serotonin/norepinephrine imbalance weakens emotional regulation",
                "ssd_change": "increased",
                "weight": 0.26,
            },
            {
                "source": "interoceptive_hypervigilance",
                "target": "somatic_signal_amplification",
                "relation": "excessive bodily monitoring amplifies symptom salience",
                "ssd_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "frontolimbic_threat_dysregulation",
                "target": "somatic_signal_amplification",
                "relation": "threat-biased appraisal intensifies bodily symptom meaning",
                "ssd_change": "increased",
                "weight": 0.26,
            },
            {
                "source": "frontolimbic_threat_dysregulation",
                "target": "amygdala",
                "relation": "maps dysregulated threat appraisal onto limbic salience circuitry",
                "ssd_change": "increased burden",
                "weight": 0.36,
            },
            {
                "source": "monoamine_regulatory_dysfunction",
                "target": "pfc_control",
                "relation": "shared affective dysregulation burdens top-down prefrontal control",
                "ssd_change": "increased dysfunction",
                "weight": 0.30,
            },
            {
                "source": "hippocampal_contextualization_failure",
                "target": "hippocampus",
                "relation": "maps contextualization failure onto hippocampal burden",
                "ssd_change": "increased burden",
                "weight": 0.34,
            },
            {
                "source": "frontostriatal_action_inhibition_bias",
                "target": "caudate_proxy",
                "relation": "fronto-striatal inhibition bias loads caudate-centered action loops",
                "ssd_change": "increased burden",
                "weight": 0.34,
            },
            {
                "source": "frontolimbic_threat_dysregulation",
                "target": "pfc_control",
                "relation": "control regions are burdened when top-down regulation weakens",
                "ssd_change": "increased dysfunction",
                "weight": 0.32,
            },
            {
                "source": "somatic_signal_amplification",
                "target": "persistent_somatic_distress",
                "relation": "amplified bodily signals become persistent symptom burden",
                "ssd_change": "increased",
                "weight": 0.42,
            },
            {
                "source": "interoceptive_hypervigilance",
                "target": "illness_worry",
                "relation": "hypervigilance to sensations drives worry about illness",
                "ssd_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "amygdala",
                "target": "illness_worry",
                "relation": "threat salience circuitry magnifies bodily danger expectation",
                "ssd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "pfc_control",
                "target": "illness_worry",
                "relation": "weakened top-down regulation allows worry to persist",
                "ssd_change": "increased dysfunction",
                "weight": 0.18,
            },
            {
                "source": "hippocampus",
                "target": "catastrophic_health_interpretation",
                "relation": "poor contextualization favors overgeneralized fear from bodily sensations",
                "ssd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "pfc_control",
                "target": "catastrophic_health_interpretation",
                "relation": "executive control failure reduces reappraisal of feared symptom meaning",
                "ssd_change": "increased dysfunction",
                "weight": 0.28,
            },
            {
                "source": "somatic_signal_amplification",
                "target": "catastrophic_health_interpretation",
                "relation": "amplified bodily signals are more likely to be interpreted catastrophically",
                "ssd_change": "increased",
                "weight": 0.22,
            },
            {
                "source": "illness_worry",
                "target": "fear_of_illness",
                "relation": "persistent worry consolidates illness-related fear",
                "ssd_change": "increased",
                "weight": 0.36,
            },
            {
                "source": "catastrophic_health_interpretation",
                "target": "fear_of_illness",
                "relation": "catastrophic appraisals deepen disease expectancy",
                "ssd_change": "increased",
                "weight": 0.32,
            },
            {
                "source": "dopaminergic_motivational_erosion",
                "target": "fatigue_anhedonia",
                "relation": "dopaminergic burden contributes to low energy and diminished interest",
                "ssd_change": "increased",
                "weight": 0.38,
            },
            {
                "source": "persistent_somatic_distress",
                "target": "fatigue_anhedonia",
                "relation": "chronic symptom burden drains motivation and pleasure",
                "ssd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "frontostriatal_action_inhibition_bias",
                "target": "conversion_like_motor_sensory_symptoms",
                "relation": "top-down inhibitory bias can contribute to functional deficits",
                "ssd_change": "increased",
                "weight": 0.32,
            },
            {
                "source": "caudate_proxy",
                "target": "conversion_like_motor_sensory_symptoms",
                "relation": "caudate-centered action planning/inhibition loops support conversion-like presentations",
                "ssd_change": "increased",
                "weight": 0.28,
            },
            {
                "source": "pfc_control",
                "target": "conversion_like_motor_sensory_symptoms",
                "relation": "maladaptive prefrontal inhibition can suppress motor or sensory processing",
                "ssd_change": "increased dysfunction",
                "weight": 0.24,
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

    def _safe_features_any(
        self,
        concept: Any,
        modalities: Sequence[Any],
        **kwargs: Any,
    ) -> List[Any]:
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
        generic_penalty = 1 if name in {"amygdala", "hippocampus", "prefrontal cortex", "caudate"} else 0
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

    def _main_component(
        self,
        region: Any,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
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
        feats = self._safe_features_any(
            region,
            self._modality_candidates("gene"),
            gene=list(genes),
        )
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

        feats = self._safe_features_any(
            self.parcellation,
            self._modality_candidates("connectivity"),
        )
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort),
            feats[0],
        )

        for attr in ("data", "matrix"):
            try:
                candidate = getattr(compound, attr, None)
                if isinstance(candidate, pd.DataFrame):
                    self._connectivity_matrix = candidate.copy()
                    return self._connectivity_matrix
            except Exception:
                pass

        for idx in (0,):
            try:
                element = compound[idx]
                candidate = getattr(element, "data", None)
                if isinstance(candidate, pd.DataFrame):
                    self._connectivity_matrix = candidate.copy()
                    return self._connectivity_matrix
            except Exception:
                continue

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    @staticmethod
    def _normalize_label(text: Any) -> str:
        value = str(text).lower()
        for ch in "()[],;:-_/":
            value = value.replace(ch, " ")
        return " ".join(value.split())

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = region.name
        region_name_l = region_name.lower()
        region_id = str(getattr(region, "identifier", "")).lower()
        normalized_region = self._normalize_label(region_name)
        region_tokens = {
            tok
            for tok in normalized_region.split()
            if tok not in {"left", "right", "area", "region", "part", "of", "the"}
        }

        def as_name(label: Any) -> str:
            return self._name_of(label)

        def as_name_l(label: Any) -> str:
            return as_name(label).lower()

        def normalized(label: Any) -> str:
            return self._normalize_label(as_name(label))

        for label in labels:
            if as_name_l(label) == region_name_l:
                return label

        for label in labels:
            lname = as_name_l(label)
            if region_id and (region_id == lname or region_id in lname):
                return label

        for label in labels:
            nlabel = normalized(label)
            if nlabel == normalized_region:
                return label

        for label in labels:
            lname = as_name_l(label)
            if region_name_l in lname or lname in region_name_l:
                return label

        scored: List[Tuple[int, int, Any]] = []
        for label in labels:
            nlabel = normalized(label)
            label_tokens = {
                tok
                for tok in nlabel.split()
                if tok not in {"left", "right", "area", "region", "part", "of", "the"}
            }
            overlap = len(region_tokens & label_tokens)
            hemi_match = int(
                ("left" in normalized_region and "left" in nlabel)
                or ("right" in normalized_region and "right" in nlabel)
            )
            if overlap >= 2:
                scored.append((overlap, hemi_match, label))

        if scored:
            scored.sort(key=lambda x: (-x[0], -x[1], self._name_of(x[2])))
            return scored[0][2]
        return None

    @staticmethod
    def _to_numeric_mean(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if np.isscalar(value):
                scalar = float(value)
                return scalar if np.isfinite(scalar) else None
        except Exception:
            pass

        try:
            if isinstance(value, pd.DataFrame):
                stacked = value.stack(dropna=True)
                numeric = pd.to_numeric(stacked, errors="coerce").dropna()
            elif isinstance(value, pd.Series):
                numeric = pd.to_numeric(value, errors="coerce").dropna()
            else:
                arr = np.asarray(value)
                numeric = pd.to_numeric(pd.Series(arr.ravel()), errors="coerce").dropna()
            if len(numeric) == 0:
                return None
            out = float(numeric.mean())
            return out if np.isfinite(out) else None
        except Exception:
            return None

    def _matrix_series_from_row(self, matrix: pd.DataFrame, row_label: Any) -> pd.Series:
        obj = matrix.loc[row_label]
        if isinstance(obj, pd.DataFrame):
            obj = obj.apply(pd.to_numeric, errors="coerce").mean(axis=0)
        elif not isinstance(obj, pd.Series):
            obj = pd.Series(obj)
        return pd.to_numeric(obj, errors="coerce").dropna()

    def _matrix_series_from_col(self, matrix: pd.DataFrame, col_label: Any) -> pd.Series:
        obj = matrix[col_label]
        if isinstance(obj, pd.DataFrame):
            obj = obj.apply(pd.to_numeric, errors="coerce").mean(axis=1)
        elif not isinstance(obj, pd.Series):
            obj = pd.Series(obj)
        return pd.to_numeric(obj, errors="coerce").dropna()

    def _lookup_connectivity_value(
        self,
        matrix: pd.DataFrame,
        row_label: Any,
        col_label: Any,
    ) -> Optional[float]:
        try:
            return self._to_numeric_mean(matrix.loc[row_label, col_label])
        except Exception:
            pass
        try:
            row_series = self._matrix_series_from_row(matrix, row_label)
            if col_label in row_series.index:
                return self._to_numeric_mean(row_series.loc[col_label])
        except Exception:
            pass
        try:
            col_series = self._matrix_series_from_col(matrix, col_label)
            if row_label in col_series.index:
                return self._to_numeric_mean(col_series.loc[row_label])
        except Exception:
            pass
        return None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        columns = ["connected_region", "value"]
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame(columns=columns)

        row_label = self._match_region_label(list(matrix.index), region)
        if row_label is not None:
            try:
                series = self._matrix_series_from_row(matrix, row_label)
                df = series.sort_values(ascending=False).reset_index()
                df.columns = columns
                df["connected_region"] = df["connected_region"].map(self._name_of)
                df = df[df["connected_region"] != region.name].head(max_rows)
                return df.reset_index(drop=True)
            except Exception:
                pass

        col_label = self._match_region_label(list(matrix.columns), region)
        if col_label is not None:
            try:
                series = self._matrix_series_from_col(matrix, col_label)
                df = series.sort_values(ascending=False).reset_index()
                df.columns = columns
                df["connected_region"] = df["connected_region"].map(self._name_of)
                df = df[df["connected_region"] != region.name].head(max_rows)
                return df.reset_index(drop=True)
            except Exception:
                pass

        return pd.DataFrame(columns=columns)

    def circuit_connectivity(self) -> pd.DataFrame:
        columns = [
            "source_key",
            "source_region",
            "target_key",
            "target_region",
            "mean_streamline_count",
            "n_observations",
        ]
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame(columns=columns)

        resolved = {k: r for k, r in self.region_objects.items() if r is not None}
        if len(resolved) < 2:
            return pd.DataFrame(columns=columns)

        row_labels = list(matrix.index)
        col_labels = list(matrix.columns)
        matched_rows = {k: self._match_region_label(row_labels, r) for k, r in resolved.items()}
        matched_cols = {k: self._match_region_label(col_labels, r) for k, r in resolved.items()}

        rows: List[Dict[str, Any]] = []
        keys = list(resolved.keys())
        for i, src_key in enumerate(keys):
            for tgt_key in keys[i + 1 :]:
                vals: List[float] = []
                src_row = matched_rows.get(src_key)
                tgt_row = matched_rows.get(tgt_key)
                src_col = matched_cols.get(src_key)
                tgt_col = matched_cols.get(tgt_key)

                candidate_pairs = [
                    (src_row, tgt_col),
                    (tgt_row, src_col),
                    (src_row, tgt_row),
                    (src_col, tgt_col),
                ]
                for row_label, col_label in candidate_pairs:
                    if row_label is None or col_label is None:
                        continue
                    value = self._lookup_connectivity_value(matrix, row_label, col_label)
                    if value is not None:
                        vals.append(value)

                finite_vals = [v for v in vals if np.isfinite(v)]
                if not finite_vals:
                    continue
                rows.append(
                    {
                        "source_key": src_key,
                        "source_region": resolved[src_key].name,
                        "target_key": tgt_key,
                        "target_region": resolved[tgt_key].name,
                        "mean_streamline_count": round(float(np.nanmean(finite_vals)), 6),
                        "n_observations": len(finite_vals),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=columns).sort_values(
            "mean_streamline_count",
            ascending=False,
        ).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_SSD_GENE_PANEL,
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
            if region is None:
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region_proxy" if key.endswith("_proxy") else "region",
                        "description": "Atlas-backed circuit node or proxy (unresolved in this environment)",
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
                    "node_type": "region_proxy" if key.endswith("_proxy") else "region",
                    "description": "Atlas-backed circuit node" if not key.endswith("_proxy") else "Atlas-matched proxy node",
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

    def region_mask(self, node_key: str, maptype: str = "labelled") -> Any:
        region = self.region_objects.get(node_key)
        if region is None:
            raise KeyError(f"No resolved region is available for node '{node_key}'.")
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype=maptype)
        except Exception:
            return region.get_regional_mask(space=self.space, maptype=maptype)

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if self._pmap is None:
            with siibra.QUIET:
                try:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
                except Exception:
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
        return assignments.reset_index(drop=True)

    def simulate(
        self,
        genetic_trait_liability: float = 0.52,
        neuroticism_anxious_temperament: float = 0.64,
        early_adversity_history: float = 0.36,
        chronic_stress_load: float = 0.70,
        depression_anxiety_comorbidity: float = 0.62,
        bodily_sensation_load: float = 0.72,
        recovery_support: float = 0.24,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized one-pass simulation.

        Parameters are normalized to 0..1:
        - higher genetic_trait_liability means greater inherited mood/anxiety/somatic vulnerability
        - higher neuroticism_anxious_temperament means greater bodily hypervigilance bias
        - higher early_adversity_history means more sensitized stress response
        - higher chronic_stress_load means more ongoing stress burden
        - higher depression_anxiety_comorbidity means stronger shared affective loading
        - higher bodily_sensation_load means more persistent bodily or functional symptom input
        - higher recovery_support means more protective buffering and clinical support
        """
        genetic = self._clip01(genetic_trait_liability)
        temperament = self._clip01(neuroticism_anxious_temperament)
        adversity = self._clip01(early_adversity_history)
        stress = self._clip01(chronic_stress_load)
        comorbidity = self._clip01(depression_anxiety_comorbidity)
        bodily = self._clip01(bodily_sensation_load)
        support = self._clip01(recovery_support)

        inputs = pd.Series(
            {
                "genetic_trait_liability": genetic,
                "neuroticism_anxious_temperament": temperament,
                "early_adversity_history": adversity,
                "chronic_stress_load": stress,
                "depression_anxiety_comorbidity": comorbidity,
                "bodily_sensation_load": bodily,
                "recovery_support": support,
            },
            name="inputs",
        )

        stress_system_sensitization = self._clip01(
            0.26 * genetic
            + 0.18 * temperament
            + 0.22 * adversity
            + 0.24 * stress
            + 0.10 * comorbidity
            - 0.16 * support
        )
        monoamine_regulatory_dysfunction = self._clip01(
            0.30 * comorbidity
            + 0.18 * stress
            + 0.16 * genetic
            + 0.18 * stress_system_sensitization
            + 0.08 * temperament
            - 0.14 * support
        )
        dopaminergic_motivational_erosion = self._clip01(
            0.36 * monoamine_regulatory_dysfunction
            + 0.18 * comorbidity
            + 0.16 * stress
            + 0.18 * bodily
            - 0.12 * support
        )
        interoceptive_hypervigilance = self._clip01(
            0.30 * temperament
            + 0.24 * bodily
            + 0.22 * stress_system_sensitization
            + 0.12 * comorbidity
            - 0.12 * support
        )
        frontolimbic_threat_dysregulation = self._clip01(
            0.28 * stress_system_sensitization
            + 0.24 * monoamine_regulatory_dysfunction
            + 0.22 * interoceptive_hypervigilance
            + 0.12 * comorbidity
            - 0.12 * support
        )
        hippocampal_contextualization_failure = self._clip01(
            0.30 * stress_system_sensitization
            + 0.18 * comorbidity
            + 0.14 * adversity
            + 0.16 * monoamine_regulatory_dysfunction
            + 0.08 * interoceptive_hypervigilance
            - 0.10 * support
        )
        frontostriatal_action_inhibition_bias = self._clip01(
            0.28 * stress_system_sensitization
            + 0.22 * frontolimbic_threat_dysregulation
            + 0.20 * monoamine_regulatory_dysfunction
            + 0.18 * bodily
            - 0.10 * support
        )
        somatic_signal_amplification = self._clip01(
            0.30 * interoceptive_hypervigilance
            + 0.22 * frontolimbic_threat_dysregulation
            + 0.18 * monoamine_regulatory_dysfunction
            + 0.18 * bodily
            + 0.08 * hippocampal_contextualization_failure
            - 0.12 * support
        )

        latents = pd.Series(
            {
                "stress_system_sensitization": stress_system_sensitization,
                "monoamine_regulatory_dysfunction": monoamine_regulatory_dysfunction,
                "dopaminergic_motivational_erosion": dopaminergic_motivational_erosion,
                "interoceptive_hypervigilance": interoceptive_hypervigilance,
                "frontolimbic_threat_dysregulation": frontolimbic_threat_dysregulation,
                "hippocampal_contextualization_failure": hippocampal_contextualization_failure,
                "frontostriatal_action_inhibition_bias": frontostriatal_action_inhibition_bias,
                "somatic_signal_amplification": somatic_signal_amplification,
            },
            name="latents",
        )

        amygdala = self._clip01(
            0.48 * frontolimbic_threat_dysregulation
            + 0.22 * stress_system_sensitization
            + 0.16 * interoceptive_hypervigilance
            + 0.10 * temperament
        )
        pfc_control = self._clip01(
            0.42 * frontolimbic_threat_dysregulation
            + 0.22 * monoamine_regulatory_dysfunction
            + 0.18 * stress_system_sensitization
            + 0.10 * comorbidity
            - 0.12 * support
        )
        hippocampus = self._clip01(
            0.48 * hippocampal_contextualization_failure
            + 0.22 * stress_system_sensitization
            + 0.16 * monoamine_regulatory_dysfunction
        )
        caudate_proxy = self._clip01(
            0.46 * frontostriatal_action_inhibition_bias
            + 0.20 * bodily
            + 0.16 * monoamine_regulatory_dysfunction
            + 0.10 * stress_system_sensitization
        )

        regional_state = pd.Series(
            {
                "amygdala": amygdala,
                "pfc_control": pfc_control,
                "hippocampus": hippocampus,
                "caudate_proxy": caudate_proxy,
            },
            name="regional_state",
        )

        persistent_somatic_distress = self._clip01(
            0.38 * somatic_signal_amplification
            + 0.20 * amygdala
            + 0.18 * bodily
            + 0.12 * frontolimbic_threat_dysregulation
            + 0.08 * monoamine_regulatory_dysfunction
        )
        illness_worry = self._clip01(
            0.34 * interoceptive_hypervigilance
            + 0.22 * amygdala
            + 0.22 * pfc_control
            + 0.10 * temperament
        )
        catastrophic_health_interpretation = self._clip01(
            0.34 * illness_worry
            + 0.22 * hippocampus
            + 0.22 * pfc_control
            + 0.14 * somatic_signal_amplification
        )
        fear_of_illness = self._clip01(
            0.38 * illness_worry
            + 0.28 * catastrophic_health_interpretation
            + 0.14 * amygdala
        )
        fatigue_anhedonia = self._clip01(
            0.40 * dopaminergic_motivational_erosion
            + 0.24 * persistent_somatic_distress
            + 0.18 * monoamine_regulatory_dysfunction
            + 0.10 * comorbidity
        )
        conversion_like_motor_sensory_symptoms = self._clip01(
            0.34 * frontostriatal_action_inhibition_bias
            + 0.26 * caudate_proxy
            + 0.20 * pfc_control
            + 0.10 * stress_system_sensitization
        )

        symptoms = pd.Series(
            {
                "persistent_somatic_distress": persistent_somatic_distress,
                "illness_worry": illness_worry,
                "catastrophic_health_interpretation": catastrophic_health_interpretation,
                "fear_of_illness": fear_of_illness,
                "fatigue_anhedonia": fatigue_anhedonia,
                "conversion_like_motor_sensory_symptoms": conversion_like_motor_sensory_symptoms,
            },
            name="symptoms",
        )

        phenotypes = pd.Series(
            {
                "hypervigilant_catastrophizing_profile": self._clip01(
                    np.mean(
                        [
                            interoceptive_hypervigilance,
                            illness_worry,
                            catastrophic_health_interpretation,
                            fear_of_illness,
                        ]
                    )
                ),
                "affective_somatic_profile": self._clip01(
                    np.mean(
                        [
                            monoamine_regulatory_dysfunction,
                            persistent_somatic_distress,
                            fatigue_anhedonia,
                        ]
                    )
                ),
                "conversion_bias_profile": self._clip01(
                    np.mean(
                        [
                            frontostriatal_action_inhibition_bias,
                            caudate_proxy,
                            conversion_like_motor_sensory_symptoms,
                        ]
                    )
                ),
                "stress_somatization_profile": self._clip01(
                    np.mean(
                        [
                            stress_system_sensitization,
                            frontolimbic_threat_dysregulation,
                            somatic_signal_amplification,
                            persistent_somatic_distress,
                        ]
                    )
                ),
            },
            name="phenotypes",
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }


def _print_section(title: str, obj: Any, max_rows: int = 12) -> None:
    print(f"\n=== {title} ===")
    if isinstance(obj, pd.DataFrame):
        if obj.empty:
            print("(empty)")
        else:
            print(obj.head(max_rows).to_string(index=False))
    elif isinstance(obj, pd.Series):
        print(obj.round(4).to_string())
    else:
        print(obj)


if __name__ == "__main__":
    model = SomaticSymptomDisorderModel()

    bundle = model.build()

    _print_section("Nodes", bundle["nodes"], max_rows=40)
    _print_section("Edges", bundle["edges"], max_rows=40)

    resolved_rows = []
    for key, region in bundle["regions"].items():
        resolved_rows.append(
            {
                "node_key": key,
                "resolved": region is not None,
                "region_name": getattr(region, "name", None),
                "identifier": getattr(region, "identifier", None),
            }
        )
    _print_section("Resolved regions", pd.DataFrame(resolved_rows), max_rows=20)

    for probe_key in ("amygdala", "pfc_control", "hippocampus", "caudate_proxy"):
        _print_section(
            f"Receptors: {probe_key}",
            bundle["receptors"].get(probe_key, pd.DataFrame()),
            max_rows=10,
        )
        _print_section(
            f"Genes: {probe_key}",
            bundle["genes"].get(probe_key, pd.DataFrame()),
            max_rows=10,
        )
        _print_section(
            f"Connectivity profile: {probe_key}",
            bundle["connectivity_profiles"].get(probe_key, pd.DataFrame()),
            max_rows=10,
        )

    _print_section("Circuit connectivity", bundle["circuit_connectivity"], max_rows=30)

    sim = model.simulate(
        genetic_trait_liability=0.56,
        neuroticism_anxious_temperament=0.68,
        early_adversity_history=0.34,
        chronic_stress_load=0.72,
        depression_anxiety_comorbidity=0.64,
        bodily_sensation_load=0.78,
        recovery_support=0.24,
    )
    _print_section("Simulation inputs", sim["inputs"])
    _print_section("Latent biology", sim["latents"])
    _print_section("Regional state", sim["regional_state"])
    _print_section("Symptoms", sim["symptoms"])
    _print_section("Phenotypes", sim["phenotypes"])

    # Example coordinate assignment for later local testing:
    # print(model.assign_mni_point((-6, 28, 18)).head(10).to_string(index=False))
