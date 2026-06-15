from __future__ import annotations

"""
Prolonged Grief Disorder siibra scaffold.

This script translates a chapter-level biological summary of Prolonged Grief
Disorder (PGD) into a transparent, atlas-grounded mechanistic scaffold using
siibra. It is intended for research exploration, education, and further model
refinement. It is not a validated disease model, and it must not be used for
clinical diagnosis, treatment selection, or risk prediction.

Design notes
------------
- The source chapter is partly systems-level rather than parcel-precise. As a
  result, stress chemistry, epigenetic embedding, and some regulatory functions
  remain latent biology nodes instead of being forced into false anatomical
  precision.
- Higher values in ``regional_state`` reflect disorder-relevant burden,
  dysregulation, or maladaptive recruitment pressure rather than raw neural
  activation.
- The simulator is intentionally acyclic and normalized to 0..1 so the chapter's
  logic stays readable: inputs -> latent biology -> regional state -> symptoms ->
  phenotype summaries.
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


DEFAULT_PGD_GENE_PANEL: List[str] = [
    "SLC6A4",
    "SLC6A2",
    "HTR1A",
    "BDNF",
    "NR3C1",
    "FKBP5",
    "CRHR1",
    "GAD1",
    "GABRA2",
    "GRIN1",
    "GRIN2B",
    "COMT",
]


class ProlongedGriefDisorderModel:
    """
    Atlas-grounded research scaffold for Prolonged Grief Disorder.

    The model captures chapter-level claims linking major interpersonal loss,
    inherited affective/anxiety liability, trauma-related stress sensitization,
    noradrenergic hyperarousal, GABAergic erosion, NMDA-linked maladaptive grief
    memory fixation, epigenetic embedding of stress responses, and disrupted
    frontolimbic / salience-memory regulation to persistent emotional pain,
    intrusive memories, avoidance, anhedonia, and a chronically "stuck" grief
    state.
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

        # Conservative anchors for emotion regulation, memory, and salience.
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
            "insula": [
                "Area Id1 left",
                "Area Id2 left",
                "Area Ig2 left",
                "insula left",
                "insula",
            ],
            "acc": [
                "Area p24ab left",
                "Area p24pr left",
                "Area a24pr left",
                "anterior cingulate cortex left",
                "ACC",
            ],
            "pfc_control": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "Area 9 left",
                "Area MFG1 left",
                "Area MFG2 left",
                "middle frontal gyrus left",
                "dorsolateral prefrontal cortex left",
                "prefrontal cortex left",
                "prefrontal cortex",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "bereavement_loss_severity": (
                "Severity and destabilizing impact of the interpersonal loss event"
            ),
            "genetic_affective_anxiety_liability": (
                "Familial or polygenic liability for depression, anxiety, and stress reactivity"
            ),
            "early_life_trauma_history": (
                "Earlier adversity that sensitizes later stress-response and emotion-regulation systems"
            ),
            "chronic_stress_load": (
                "Persistent stress burden surrounding or following the loss"
            ),
            "reminder_exposure": (
                "Ongoing exposure to grief reminders and memory triggers that reactivate the loss"
            ),
            "recovery_support": (
                "Protective mourning support, social buffering, and adaptive recovery conditions"
            ),
            "treatment_support": (
                "Protective treatment support that can reduce anxious arousal and depressive burden"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "acute_grief_trauma_load": (
                "Combined loss-related traumatic burden and immediate grief shock"
            ),
            "stress_sensitization": (
                "Inherited and acquired stress-reactivity bias that increases vulnerability after loss"
            ),
            "epigenetic_grief_embedding": (
                "Biological embedding of loss-related stress through durable gene-expression changes"
            ),
            "noradrenergic_hyperarousal": (
                "Elevated arousal, vigilance, and startle pressure associated with norepinephrine"
            ),
            "serotonergic_affective_dysregulation": (
                "Low-mood and emotional-regulation burden related to serotonergic imbalance"
            ),
            "gabaergic_inhibitory_erosion": (
                "Reduced inhibitory buffering that permits persistent anxiety and hyperarousal"
            ),
            "nmda_glutamatergic_memory_fixation": (
                "Stress-linked maladaptive plasticity that entrenches grief memories and emotional traces"
            ),
            "frontolimbic_regulation_failure": (
                "Reduced top-down regulation of limbic distress and grief-related salience"
            ),
            "salience_memory_network_dysregulation": (
                "Persistent coupling of salience, emotional pain, and autobiographical grief memory"
            ),
            "mourning_resolution_failure": (
                "Failure of grief adaptation leading to chronic arrest of mourning"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "persistent_emotional_pain": (
                "Ongoing emotional pain and distress connected to the loss"
            ),
            "intrusive_grief_memories": (
                "Repeated, difficult-to-regulate grief-related memories and reminders"
            ),
            "avoidance": (
                "Avoidance of reminders, situations, or internal states linked to the loss"
            ),
            "hyperarousal": (
                "Hypervigilance, exaggerated startle, tension, and persistent arousal"
            ),
            "anhedonia": (
                "Reduced capacity for pleasure or re-engagement with life after the loss"
            ),
            "somatic_distress": (
                "Fatigue, bodily complaints, and stress-linked physical discomfort"
            ),
            "sleep_disruption": (
                "Insomnia or arousal-linked sleep disruption in bereavement"
            ),
            "persistent_grief_state": (
                "Chronically stuck grief process with impaired natural resolution"
            ),
        }

        self.edge_table: List[Dict[str, Any]] = [
            {
                "source": "bereavement_loss_severity",
                "target": "acute_grief_trauma_load",
                "relation": "major loss acts as a severe stressor and traumatic grief trigger",
                "pgd_change": "increased",
                "weight": 0.46,
            },
            {
                "source": "genetic_affective_anxiety_liability",
                "target": "stress_sensitization",
                "relation": "familial affective and anxiety liability raises vulnerability after loss",
                "pgd_change": "increased",
                "weight": 0.42,
            },
            {
                "source": "early_life_trauma_history",
                "target": "stress_sensitization",
                "relation": "earlier adversity sensitizes stress-response systems",
                "pgd_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "chronic_stress_load",
                "target": "stress_sensitization",
                "relation": "sustained stress amplifies later reactivity and dysregulation",
                "pgd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "recovery_support",
                "target": "acute_grief_trauma_load",
                "relation": "adaptive support can buffer the destabilizing impact of loss",
                "pgd_change": "decreased",
                "weight": -0.16,
            },
            {
                "source": "stress_sensitization",
                "target": "epigenetic_grief_embedding",
                "relation": "stress-sensitive systems are more likely to biologically embed the loss",
                "pgd_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "acute_grief_trauma_load",
                "target": "epigenetic_grief_embedding",
                "relation": "severe loss can induce lasting molecular scars that maintain dysregulation",
                "pgd_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "chronic_stress_load",
                "target": "gabaergic_inhibitory_erosion",
                "relation": "chronic stress disrupts inhibitory buffering and sustains anxiety",
                "pgd_change": "increased",
                "weight": 0.39,
            },
            {
                "source": "acute_grief_trauma_load",
                "target": "noradrenergic_hyperarousal",
                "relation": "loss-related stress increases vigilance and arousal",
                "pgd_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "stress_sensitization",
                "target": "noradrenergic_hyperarousal",
                "relation": "stress-reactive vulnerability heightens PTSD-like hyperarousal",
                "pgd_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "reminder_exposure",
                "target": "noradrenergic_hyperarousal",
                "relation": "grief reminders can reactivate arousal and threat-monitoring",
                "pgd_change": "increased",
                "weight": 0.18,
            },
            {
                "source": "treatment_support",
                "target": "noradrenergic_hyperarousal",
                "relation": "supportive treatment may reduce anxious arousal burden",
                "pgd_change": "decreased",
                "weight": -0.18,
            },
            {
                "source": "genetic_affective_anxiety_liability",
                "target": "serotonergic_affective_dysregulation",
                "relation": "affective liability increases low-mood and emotional dysregulation pressure",
                "pgd_change": "increased",
                "weight": 0.28,
            },
            {
                "source": "acute_grief_trauma_load",
                "target": "serotonergic_affective_dysregulation",
                "relation": "severe loss can deepen depressive-affective burden",
                "pgd_change": "increased",
                "weight": 0.26,
            },
            {
                "source": "epigenetic_grief_embedding",
                "target": "serotonergic_affective_dysregulation",
                "relation": "embedded stress vulnerability sustains mood dysregulation",
                "pgd_change": "increased",
                "weight": 0.22,
            },
            {
                "source": "chronic_stress_load",
                "target": "nmda_glutamatergic_memory_fixation",
                "relation": "stress-linked glutamatergic plasticity can entrench grief memory traces",
                "pgd_change": "increased",
                "weight": 0.26,
            },
            {
                "source": "acute_grief_trauma_load",
                "target": "nmda_glutamatergic_memory_fixation",
                "relation": "intense grief experience loads maladaptive memory consolidation",
                "pgd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "reminder_exposure",
                "target": "nmda_glutamatergic_memory_fixation",
                "relation": "repeated cue reactivation reinforces grief-memory fixation",
                "pgd_change": "increased",
                "weight": 0.22,
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "frontolimbic_regulation_failure",
                "relation": "high arousal weakens effective top-down emotion regulation",
                "pgd_change": "increased",
                "weight": 0.27,
            },
            {
                "source": "gabaergic_inhibitory_erosion",
                "target": "frontolimbic_regulation_failure",
                "relation": "loss of inhibitory control permits persistent distress states",
                "pgd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "serotonergic_affective_dysregulation",
                "target": "frontolimbic_regulation_failure",
                "relation": "depressive-affective dysregulation weakens adaptive emotion regulation",
                "pgd_change": "increased",
                "weight": 0.22,
            },
            {
                "source": "nmda_glutamatergic_memory_fixation",
                "target": "salience_memory_network_dysregulation",
                "relation": "maladaptive plasticity couples grief memory with persistent salience",
                "pgd_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "salience_memory_network_dysregulation",
                "relation": "hyperarousal amplifies the salience of grief reminders",
                "pgd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "frontolimbic_regulation_failure",
                "target": "salience_memory_network_dysregulation",
                "relation": "poor regulation permits emotional pain and memory systems to remain coupled",
                "pgd_change": "increased",
                "weight": 0.22,
            },
            {
                "source": "epigenetic_grief_embedding",
                "target": "mourning_resolution_failure",
                "relation": "lasting molecular changes can maintain a chronic stuck grief state",
                "pgd_change": "increased",
                "weight": 0.31,
            },
            {
                "source": "salience_memory_network_dysregulation",
                "target": "mourning_resolution_failure",
                "relation": "persistent grief salience prevents natural resolution",
                "pgd_change": "increased",
                "weight": 0.29,
            },
            {
                "source": "serotonergic_affective_dysregulation",
                "target": "mourning_resolution_failure",
                "relation": "depressive burden hampers adaptive mourning",
                "pgd_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "recovery_support",
                "target": "mourning_resolution_failure",
                "relation": "supportive recovery context can reduce grief entrenchment",
                "pgd_change": "decreased",
                "weight": -0.18,
            },
            {
                "source": "frontolimbic_regulation_failure",
                "target": "pfc_control",
                "relation": "maps regulatory burden onto prefrontal control systems",
                "pgd_change": "increased",
                "weight": 0.41,
            },
            {
                "source": "salience_memory_network_dysregulation",
                "target": "acc",
                "relation": "maps persistent salience and emotional pain onto cingulate control circuitry",
                "pgd_change": "increased",
                "weight": 0.36,
            },
            {
                "source": "salience_memory_network_dysregulation",
                "target": "insula",
                "relation": "maps salience and somatic distress burden onto insular circuitry",
                "pgd_change": "increased",
                "weight": 0.39,
            },
            {
                "source": "nmda_glutamatergic_memory_fixation",
                "target": "hippocampus",
                "relation": "maps grief-memory fixation onto memory circuitry",
                "pgd_change": "increased",
                "weight": 0.42,
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "amygdala",
                "relation": "maps hyperarousal onto threat and emotional salience circuitry",
                "pgd_change": "increased",
                "weight": 0.39,
            },
            {
                "source": "insula",
                "target": "persistent_emotional_pain",
                "relation": "supports salience-laden emotional and bodily pain experience",
                "pgd_change": "increased",
                "weight": 0.31,
            },
            {
                "source": "hippocampus",
                "target": "intrusive_grief_memories",
                "relation": "memory-system dysregulation promotes repetitive grief recollection",
                "pgd_change": "increased",
                "weight": 0.34,
            },
            {
                "source": "amygdala",
                "target": "avoidance",
                "relation": "threat-weighted emotional salience promotes avoidance of reminders",
                "pgd_change": "increased",
                "weight": 0.27,
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "hyperarousal",
                "relation": "drives hypervigilance and exaggerated startle",
                "pgd_change": "increased",
                "weight": 0.44,
            },
            {
                "source": "serotonergic_affective_dysregulation",
                "target": "anhedonia",
                "relation": "supports depressive withdrawal and reduced pleasure",
                "pgd_change": "increased",
                "weight": 0.33,
            },
            {
                "source": "insula",
                "target": "somatic_distress",
                "relation": "interoceptive salience contributes to bodily grief symptoms",
                "pgd_change": "increased",
                "weight": 0.24,
            },
            {
                "source": "gabaergic_inhibitory_erosion",
                "target": "sleep_disruption",
                "relation": "reduced inhibitory buffering promotes anxiety-linked insomnia",
                "pgd_change": "increased",
                "weight": 0.28,
            },
            {
                "source": "mourning_resolution_failure",
                "target": "persistent_grief_state",
                "relation": "arrested mourning produces a chronic prolonged grief phenotype",
                "pgd_change": "increased",
                "weight": 0.46,
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
    def _coerce_xyz(point_like: Any) -> Optional[Tuple[float, float, float]]:
        if point_like is None:
            return None
        if hasattr(point_like, "coordinate"):
            coords = getattr(point_like, "coordinate")
        elif isinstance(point_like, (list, tuple)):
            coords = point_like
        else:
            try:
                coords = tuple(point_like)
            except Exception:
                return None
        try:
            xyz = tuple(float(v) for v in coords[:3])
        except Exception:
            return None
        return xyz if len(xyz) == 3 else None

    @staticmethod
    def _preferred_hemi(query: str) -> Optional[str]:
        lower = str(query).lower()
        if " right" in lower or lower.endswith("right") or "(right)" in lower:
            return "right"
        if " left" in lower or lower.endswith("left") or "(left)" in lower:
            return "left"
        return None

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

    def _region_rank(self, region: Any, preferred_hemi: Optional[str] = None) -> Tuple[int, int, int]:
        name = self._name_of(region).lower()
        if preferred_hemi == "right":
            hemi_bonus = 0 if "right" in name else 1
            opposite_penalty = 1 if "left" in name else 0
        else:
            hemi_bonus = 0 if "left" in name else 1
            opposite_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {"amygdala", "hippocampus", "prefrontal cortex"} else 0
        return (hemi_bonus, opposite_penalty, generic_penalty)

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
                preferred_hemi = self._preferred_hemi(spec)
                matches = sorted(matches, key=lambda r: self._region_rank(r, preferred_hemi))
                return matches[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        seen = set()
        preferred_hemi = self._preferred_hemi(keyword)
        matches = self._julich_matches(keyword)
        for region in sorted(matches, key=lambda r: self._region_rank(r, preferred_hemi)):
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
        centroid_xyz = self._coerce_xyz(getattr(main, "centroid", None))
        volume_mm3: Optional[float]
        try:
            vol = float(getattr(main, "volume", float("nan")))
            volume_mm3 = vol if np.isfinite(vol) else None
        except Exception:
            volume_mm3 = None
        return centroid_xyz, volume_mm3

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            try:
                df = feat.data.copy().reset_index()
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
            lower_cols = {str(c).lower(): c for c in df.columns}
            required = {"gene", "level", "zscore"}
            if required.issubset(lower_cols):
                gene_col = lower_cols["gene"]
                level_col = lower_cols["level"]
                zscore_col = lower_cols["zscore"]
                grouped = (
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
                return grouped
            return df.reset_index(drop=True)
        return pd.DataFrame()

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), feats[0])

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

    def _lookup_connectivity_value(self, matrix: pd.DataFrame, row_label: Any, col_label: Any) -> Optional[float]:
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
            "mean_streamline_count", ascending=False
        ).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_PGD_GENE_PANEL,
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
                self.region_objects[key] = None
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.upper(),
                        "node_type": "region",
                        "description": "Atlas-backed node (unresolved in this environment)",
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
                    "description": "Atlas-backed circuit node",
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
        bereavement_loss_severity: float = 0.75,
        genetic_affective_anxiety_liability: float = 0.50,
        early_life_trauma_history: float = 0.40,
        chronic_stress_load: float = 0.65,
        reminder_exposure: float = 0.60,
        recovery_support: float = 0.28,
        treatment_support: float = 0.22,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized one-pass simulation.

        Parameters are normalized to 0..1:
        - higher bereavement_loss_severity means more destabilizing interpersonal loss burden
        - higher genetic_affective_anxiety_liability means greater inherited depression/anxiety vulnerability
        - higher early_life_trauma_history means greater pre-existing stress sensitization
        - higher chronic_stress_load means more persistent post-loss stress burden
        - higher reminder_exposure means more grief cue reactivation
        - higher recovery_support means more adaptive social and mourning support
        - higher treatment_support means more protective treatment buffering
        """
        loss = self._clip01(bereavement_loss_severity)
        genetic = self._clip01(genetic_affective_anxiety_liability)
        trauma = self._clip01(early_life_trauma_history)
        stress = self._clip01(chronic_stress_load)
        reminders = self._clip01(reminder_exposure)
        recovery = self._clip01(recovery_support)
        treatment = self._clip01(treatment_support)

        inputs = pd.Series(
            {
                "bereavement_loss_severity": loss,
                "genetic_affective_anxiety_liability": genetic,
                "early_life_trauma_history": trauma,
                "chronic_stress_load": stress,
                "reminder_exposure": reminders,
                "recovery_support": recovery,
                "treatment_support": treatment,
            },
            name="inputs",
        )

        acute_grief_trauma_load = self._clip01(
            0.52 * loss + 0.16 * trauma + 0.10 * stress + 0.08 * reminders - 0.16 * recovery
        )
        stress_sensitization = self._clip01(
            0.34 * genetic + 0.34 * trauma + 0.18 * stress + 0.08 * loss - 0.10 * recovery
        )
        epigenetic_grief_embedding = self._clip01(
            0.36 * stress_sensitization + 0.28 * acute_grief_trauma_load + 0.16 * stress - 0.10 * recovery
        )
        noradrenergic_hyperarousal = self._clip01(
            0.34 * acute_grief_trauma_load
            + 0.28 * stress_sensitization
            + 0.16 * stress
            + 0.10 * reminders
            - 0.12 * treatment
            - 0.08 * recovery
        )
        serotonergic_affective_dysregulation = self._clip01(
            0.30 * genetic
            + 0.24 * acute_grief_trauma_load
            + 0.22 * epigenetic_grief_embedding
            + 0.10 * stress
            - 0.14 * treatment
            - 0.08 * recovery
        )
        gabaergic_inhibitory_erosion = self._clip01(
            0.38 * stress
            + 0.24 * noradrenergic_hyperarousal
            + 0.16 * stress_sensitization
            - 0.12 * treatment
            - 0.08 * recovery
        )
        nmda_glutamatergic_memory_fixation = self._clip01(
            0.30 * epigenetic_grief_embedding
            + 0.24 * acute_grief_trauma_load
            + 0.18 * reminders
            + 0.14 * stress
            - 0.10 * treatment
        )
        frontolimbic_regulation_failure = self._clip01(
            0.28 * noradrenergic_hyperarousal
            + 0.24 * serotonergic_affective_dysregulation
            + 0.20 * gabaergic_inhibitory_erosion
            + 0.12 * epigenetic_grief_embedding
            - 0.14 * recovery
            - 0.08 * treatment
        )
        salience_memory_network_dysregulation = self._clip01(
            0.32 * nmda_glutamatergic_memory_fixation
            + 0.24 * noradrenergic_hyperarousal
            + 0.20 * frontolimbic_regulation_failure
            + 0.12 * reminders
            - 0.12 * recovery
        )
        mourning_resolution_failure = self._clip01(
            0.30 * epigenetic_grief_embedding
            + 0.26 * salience_memory_network_dysregulation
            + 0.18 * serotonergic_affective_dysregulation
            + 0.12 * frontolimbic_regulation_failure
            - 0.14 * recovery
            - 0.08 * treatment
        )

        latents = pd.Series(
            {
                "acute_grief_trauma_load": acute_grief_trauma_load,
                "stress_sensitization": stress_sensitization,
                "epigenetic_grief_embedding": epigenetic_grief_embedding,
                "noradrenergic_hyperarousal": noradrenergic_hyperarousal,
                "serotonergic_affective_dysregulation": serotonergic_affective_dysregulation,
                "gabaergic_inhibitory_erosion": gabaergic_inhibitory_erosion,
                "nmda_glutamatergic_memory_fixation": nmda_glutamatergic_memory_fixation,
                "frontolimbic_regulation_failure": frontolimbic_regulation_failure,
                "salience_memory_network_dysregulation": salience_memory_network_dysregulation,
                "mourning_resolution_failure": mourning_resolution_failure,
            },
            name="latents",
        )

        amygdala = self._clip01(
            0.44 * noradrenergic_hyperarousal
            + 0.20 * frontolimbic_regulation_failure
            + 0.16 * acute_grief_trauma_load
            + 0.08 * reminders
        )
        hippocampus = self._clip01(
            0.44 * nmda_glutamatergic_memory_fixation
            + 0.22 * epigenetic_grief_embedding
            + 0.14 * acute_grief_trauma_load
            + 0.10 * reminders
        )
        insula = self._clip01(
            0.38 * salience_memory_network_dysregulation
            + 0.24 * noradrenergic_hyperarousal
            + 0.18 * serotonergic_affective_dysregulation
            + 0.08 * acute_grief_trauma_load
        )
        acc = self._clip01(
            0.34 * frontolimbic_regulation_failure
            + 0.28 * salience_memory_network_dysregulation
            + 0.18 * serotonergic_affective_dysregulation
            + 0.08 * acute_grief_trauma_load
        )
        pfc_control = self._clip01(
            0.42 * frontolimbic_regulation_failure
            + 0.20 * stress_sensitization
            + 0.18 * serotonergic_affective_dysregulation
            + 0.10 * stress
        )

        regional_state = pd.Series(
            {
                "amygdala": amygdala,
                "hippocampus": hippocampus,
                "insula": insula,
                "acc": acc,
                "pfc_control": pfc_control,
            },
            name="regional_state",
        )

        persistent_emotional_pain = self._clip01(
            0.30 * insula + 0.24 * acc + 0.18 * amygdala + 0.14 * acute_grief_trauma_load
        )
        intrusive_grief_memories = self._clip01(
            0.34 * hippocampus + 0.22 * nmda_glutamatergic_memory_fixation + 0.18 * amygdala + 0.12 * reminders
        )
        avoidance = self._clip01(
            0.28 * amygdala + 0.22 * noradrenergic_hyperarousal + 0.18 * frontolimbic_regulation_failure + 0.12 * intrusive_grief_memories
        )
        hyperarousal = self._clip01(
            0.42 * noradrenergic_hyperarousal + 0.22 * gabaergic_inhibitory_erosion + 0.16 * amygdala + 0.10 * reminders
        )
        anhedonia = self._clip01(
            0.34 * serotonergic_affective_dysregulation + 0.20 * mourning_resolution_failure + 0.16 * pfc_control + 0.10 * acc
        )
        somatic_distress = self._clip01(
            0.30 * noradrenergic_hyperarousal + 0.22 * insula + 0.16 * serotonergic_affective_dysregulation + 0.10 * hyperarousal
        )
        sleep_disruption = self._clip01(
            0.30 * hyperarousal + 0.20 * gabaergic_inhibitory_erosion + 0.16 * persistent_emotional_pain + 0.08 * reminders
        )
        persistent_grief_state = self._clip01(
            0.34 * mourning_resolution_failure + 0.20 * intrusive_grief_memories + 0.18 * avoidance + 0.14 * persistent_emotional_pain
        )

        symptoms = pd.Series(
            {
                "persistent_emotional_pain": persistent_emotional_pain,
                "intrusive_grief_memories": intrusive_grief_memories,
                "avoidance": avoidance,
                "hyperarousal": hyperarousal,
                "anhedonia": anhedonia,
                "somatic_distress": somatic_distress,
                "sleep_disruption": sleep_disruption,
                "persistent_grief_state": persistent_grief_state,
            },
            name="symptoms",
        )

        phenotypes = pd.Series(
            {
                "hyperaroused_grief_profile": self._clip01(
                    np.mean([hyperarousal, avoidance, sleep_disruption, somatic_distress])
                ),
                "memory_fixation_profile": self._clip01(
                    np.mean([intrusive_grief_memories, hippocampus, nmda_glutamatergic_memory_fixation])
                ),
                "depressive_anxious_grief_profile": self._clip01(
                    np.mean([persistent_emotional_pain, anhedonia, hyperarousal, serotonergic_affective_dysregulation])
                ),
                "entrenched_grief_profile": self._clip01(
                    np.mean([persistent_grief_state, mourning_resolution_failure, intrusive_grief_memories, avoidance])
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
    model = ProlongedGriefDisorderModel()

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

    for probe_key in ("amygdala", "hippocampus", "insula", "acc", "pfc_control"):
        _print_section(f"Receptors: {probe_key}", bundle["receptors"].get(probe_key, pd.DataFrame()), max_rows=10)
        _print_section(f"Genes: {probe_key}", bundle["genes"].get(probe_key, pd.DataFrame()), max_rows=10)
        _print_section(
            f"Connectivity profile: {probe_key}",
            bundle["connectivity_profiles"].get(probe_key, pd.DataFrame()),
            max_rows=10,
        )

    _print_section("Circuit connectivity", bundle["circuit_connectivity"], max_rows=30)

    sim = model.simulate(
        bereavement_loss_severity=0.82,
        genetic_affective_anxiety_liability=0.56,
        early_life_trauma_history=0.42,
        chronic_stress_load=0.70,
        reminder_exposure=0.66,
        recovery_support=0.24,
        treatment_support=0.18,
    )
    _print_section("Simulation inputs", sim["inputs"])
    _print_section("Latent biology", sim["latents"])
    _print_section("Regional state", sim["regional_state"])
    _print_section("Symptoms", sim["symptoms"])
    _print_section("Phenotypes", sim["phenotypes"])

    # Example coordinate assignment for later local testing:
    # print(model.assign_mni_point((-24, -6, -18)).head(10).to_string(index=False))
