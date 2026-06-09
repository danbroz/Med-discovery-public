
from __future__ import annotations

"""
Schizoaffective disorder siibra scaffold.

This script translates a chapter-level biological overview of schizoaffective
disorder into a transparent, atlas-grounded research scaffold using siibra
idioms. It is intended as a reusable mechanistic template rather than a
diagnostic or treatment tool.

Important notes
---------------
- This is a research scaffold, not a clinical model.
- The simulator is a faithful interpretation of the supplied chapter, not a
  validated disease simulation.
- The chapter frames schizoaffective disorder as lying on a continuum between
  schizophrenia and bipolar disorder, so the scaffold explicitly models both
  psychosis-related and mood-related pathways.
- Several nodes are modeled as conservative proxies because the chapter names
  circuit families more clearly than exact cytoarchitectonic parcels.
- Dopamine, serotonin, norepinephrine, HPA-axis stress biology, and synaptic /
  neurodevelopmental processes are kept primarily as latent mechanisms rather
  than being forced into single parcels.
- The gene panel is heuristic and designed to cover psychosis-spectrum, mood,
  neurodevelopmental, stress-response, myelination, and synaptic-plasticity
  mechanisms strongly suggested by the chapter.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception as exc:  # pragma: no cover - import guard for portability
    siibra = None
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - trivial branch
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_GENE_PANEL = [
    "DRD2",
    "DRD1",
    "SLC6A3",
    "HTR2A",
    "HTR1A",
    "SLC6A4",
    "SLC6A2",
    "ADRA2A",
    "COMT",
    "CACNA1C",
    "ANK3",
    "BDNF",
    "DISC1",
    "NRG1",
    "RELN",
    "GAD1",
    "GRIN2A",
    "NR3C1",
    "FKBP5",
    "CRHR1",
    "MBP",
]


class SchizoaffectiveDisorderModel:
    """
    Atlas-grounded scaffold for schizoaffective disorder.

    Chapter logic encoded here emphasizes:
    - overlap between schizophrenia-spectrum and bipolar-spectrum biology;
    - dopamine-serotonin interaction relevant to both psychosis and mood;
    - a noradrenergic symptom cluster linked to depressive fatigue, low energy,
      and concentration problems;
    - neurodevelopmental and synaptic-plasticity burden interacting with chronic
      stress and HPA-axis dysregulation;
    - combined frontolimbic mood dysregulation and frontostriatal /
      frontotemporal psychosis circuitry;
    - disrupted connectivity between prefrontal cortex and subcortical systems
      including thalamus and basal ganglia, with hippocampal and temporal
      involvement.
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
                "siibra is required to instantiate SchizoaffectiveDisorderModel. "
                "Install siibra-python in your environment first."
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

        self.region_candidates: Dict[str, List[str]] = {
            "pfc_control": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 9/46v left",
                "prefrontal cortex",
                "middle frontal gyrus",
                "frontal",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "Hippocampus left",
                "CA1 left",
                "Subiculum left",
                "hippocampus",
            ],
            "thalamus": [
                "Thalamus left",
                "thalamus",
                "Pulvinar left",
                "Mediodorsal thalamus left",
            ],
            "basal_ganglia_proxy": [
                "Putamen left",
                "Caudate nucleus left",
                "Nucleus accumbens left",
                "striatum",
                "basal ganglia",
            ],
            "temporal_cortex_proxy": [
                "Area TE 3 left",
                "Area TE 2.1 (STG) left",
                "Area TE 1.0 (HESCHL) left",
                "superior temporal gyrus",
                "temporal",
            ],
        }

        self.region_notes: Dict[str, str] = {
            "pfc_control": (
                "Proxy for the chapter's prefrontal top-down control system over "
                "both emotional and psychotic processes."
            ),
            "amygdala": (
                "Conservative limbic anchor for the chapter's frontolimbic mood "
                "regulation circuit."
            ),
            "hippocampus": (
                "Named structural abnormality and plausible contributor to memory, "
                "stress sensitivity, and circuit instability."
            ),
            "thalamus": (
                "Named structural and functional node in prefrontal-subcortical "
                "disconnection."
            ),
            "basal_ganglia_proxy": (
                "Proxy for the subcortical motivational and psychosis-related "
                "circuitry explicitly named in the chapter."
            ),
            "temporal_cortex_proxy": (
                "Proxy for fronto-temporal circuitry implicated in psychotic and "
                "cognitive symptoms."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Inherited liability spanning schizophrenia-spectrum and "
                "bipolar-spectrum risk."
            ),
            "neurodevelopmental_burden": (
                "Developmental disruption affecting neuronal migration, synapse "
                "formation, and myelination."
            ),
            "chronic_stress_load": (
                "Ongoing stress burden relevant to depression and HPA-axis "
                "dysregulation."
            ),
            "depressive_episode_load": (
                "Current severity of depressive syndromic burden."
            ),
            "manic_mixed_load": (
                "Current severity of manic or mixed-state drive in bipolar-type "
                "presentations."
            ),
            "prodromal_decline": (
                "Pre-episode decline in functioning indicating emerging network "
                "instability."
            ),
            "atypical_antipsychotic_coverage": (
                "Protective D2 / 5-HT2A antagonist coverage expected to reduce "
                "psychosis and partly stabilize mood-linked circuitry."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "dopamine_serotonin_interplay": (
                "Coupled dopaminergic-serotonergic dysregulation linking mood and "
                "psychotic symptoms."
            ),
            "noradrenergic_dysfunction": (
                "Noradrenergic burden contributing to fatigue, poor energy, and "
                "concentration problems."
            ),
            "neurodevelopmental_connectivity_burden": (
                "Developmental disruption of maturation, myelination, and "
                "long-range circuit integration."
            ),
            "synaptic_plasticity_resilience_failure": (
                "Reduced synaptic resilience and impaired neuroplastic adaptation."
            ),
            "hpa_axis_dysregulation": (
                "Stress-linked neuroendocrine dysregulation relevant to mood burden."
            ),
            "frontolimbic_mood_dysregulation": (
                "Disordered prefrontal-limbic regulation of affect and emotional "
                "reactivity."
            ),
            "frontostriatal_psychosis_dysregulation": (
                "Psychosis-linked disruption in prefrontal-subcortical control."
            ),
            "frontotemporal_disconnection": (
                "Disconnection between frontal executive systems and temporal "
                "perceptual/language systems."
            ),
            "thalamocortical_dysconnectivity": (
                "Impaired coordination between thalamus and cortex."
            ),
            "psychotic_mood_coupling": (
                "Cross-coupling in which psychotic and severe mood symptoms amplify "
                "one another."
            ),
            "top_down_control_failure": (
                "Widespread failure of prefrontal control over perception, emotion, "
                "and motivation."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "psychotic_symptoms": (
                "Delusions, hallucinations, and related psychotic symptom burden."
            ),
            "disorganized_thought": (
                "Disorganized thinking and impaired cognitive coherence."
            ),
            "depressive_fatigue_concentration": (
                "Fatigue, loss of energy, and concentration impairment in "
                "depressive states."
            ),
            "manic_affective_activation": (
                "Mood activation and energized affective drive in manic or mixed "
                "states."
            ),
            "emotional_lability": (
                "Rapid or unstable shifts in affective tone."
            ),
            "cognitive_impairment": (
                "Executive and memory-related inefficiency across mood and "
                "psychosis burden."
            ),
            "negative_symptoms_avolition": (
                "Avolition, diminished motivation, and affective flattening."
            ),
            "mixed_psychotic_mood_burden": (
                "Concurrent expression of severe psychotic and mood symptom burden."
            ),
        }

        self.edge_table: List[Dict[str, Any]] = [
            {
                "source": "genetic_vulnerability",
                "target": "neurodevelopmental_connectivity_burden",
                "relation": "confers inherited vulnerability across psychosis and bipolar-spectrum pathways",
                "schizoaffective_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "genetic_vulnerability",
                "target": "dopamine_serotonin_interplay",
                "relation": "biases coupled monoaminergic dysregulation",
                "schizoaffective_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "neurodevelopmental_burden",
                "target": "neurodevelopmental_connectivity_burden",
                "relation": "disrupts maturation, myelination, and synaptic organization",
                "schizoaffective_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "prodromal_decline",
                "target": "neurodevelopmental_connectivity_burden",
                "relation": "signals emerging connectivity failure before full episodes",
                "schizoaffective_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "chronic_stress_load",
                "target": "hpa_axis_dysregulation",
                "relation": "activates neuroendocrine stress burden relevant to depression",
                "schizoaffective_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "chronic_stress_load",
                "target": "synaptic_plasticity_resilience_failure",
                "relation": "reduces resilience and adaptive neuroplastic response",
                "schizoaffective_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "depressive_episode_load",
                "target": "noradrenergic_dysfunction",
                "relation": "aligns with the chapter's noradrenergic symptom cluster",
                "schizoaffective_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "depressive_episode_load",
                "target": "frontolimbic_mood_dysregulation",
                "relation": "drives depressive-affective circuit burden",
                "schizoaffective_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "manic_mixed_load",
                "target": "frontolimbic_mood_dysregulation",
                "relation": "drives bipolar-type affective circuit activation",
                "schizoaffective_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "manic_mixed_load",
                "target": "dopamine_serotonin_interplay",
                "relation": "increases mood-linked monoaminergic instability",
                "schizoaffective_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "synaptic_plasticity_resilience_failure",
                "relation": "chronic stress signaling impairs cellular resilience and adaptive plasticity",
                "schizoaffective_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "neurodevelopmental_connectivity_burden",
                "target": "frontostriatal_psychosis_dysregulation",
                "relation": "impairs prefrontal-subcortical psychosis circuitry",
                "schizoaffective_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "dopamine_serotonin_interplay",
                "target": "frontostriatal_psychosis_dysregulation",
                "relation": "coupled monoaminergic dysregulation intensifies psychosis-linked circuitry",
                "schizoaffective_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "neurodevelopmental_connectivity_burden",
                "target": "frontotemporal_disconnection",
                "relation": "weakens frontal-temporal integration",
                "schizoaffective_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "neurodevelopmental_connectivity_burden",
                "target": "thalamocortical_dysconnectivity",
                "relation": "impairs thalamocortical integration",
                "schizoaffective_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "frontolimbic_mood_dysregulation",
                "target": "psychotic_mood_coupling",
                "relation": "links severe mood dysregulation to psychotic amplification",
                "schizoaffective_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "frontostriatal_psychosis_dysregulation",
                "target": "psychotic_mood_coupling",
                "relation": "allows psychotic symptoms and mood episodes to influence each other",
                "schizoaffective_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "thalamocortical_dysconnectivity",
                "target": "top_down_control_failure",
                "relation": "reduces coordinated prefrontal control over subcortical systems",
                "schizoaffective_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "frontolimbic_mood_dysregulation",
                "target": "top_down_control_failure",
                "relation": "destabilizes emotional regulation from prefrontal systems",
                "schizoaffective_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "frontostriatal_psychosis_dysregulation",
                "target": "top_down_control_failure",
                "relation": "weakens cognitive control over perception and motivation",
                "schizoaffective_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "frontotemporal_disconnection",
                "target": "disorganized_thought",
                "relation": "impairs coherent integration of perceptual and executive systems",
                "schizoaffective_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "frontostriatal_psychosis_dysregulation",
                "target": "psychotic_symptoms",
                "relation": "drives psychotic burden through prefrontal-subcortical dysfunction",
                "schizoaffective_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "psychotic_mood_coupling",
                "target": "psychotic_symptoms",
                "relation": "mood-psychosis coupling intensifies psychotic expression",
                "schizoaffective_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "noradrenergic_dysfunction",
                "target": "depressive_fatigue_concentration",
                "relation": "contributes to fatigue, low energy, and poor concentration",
                "schizoaffective_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "frontolimbic_mood_dysregulation",
                "target": "manic_affective_activation",
                "relation": "contributes to bipolar-type affective activation",
                "schizoaffective_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "frontolimbic_mood_dysregulation",
                "target": "emotional_lability",
                "relation": "produces unstable affect and emotional dysregulation",
                "schizoaffective_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "top_down_control_failure",
                "target": "mixed_psychotic_mood_burden",
                "relation": "permits simultaneous severe disruption of thought and affect",
                "schizoaffective_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "psychotic_mood_coupling",
                "target": "mixed_psychotic_mood_burden",
                "relation": "integrates mood and psychotic severity into a joint phenotype",
                "schizoaffective_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "top_down_control_failure",
                "target": "cognitive_impairment",
                "relation": "reduces executive and motivational coordination",
                "schizoaffective_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "synaptic_plasticity_resilience_failure",
                "target": "cognitive_impairment",
                "relation": "undermines adaptive circuit function and cognitive reserve",
                "schizoaffective_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "top_down_control_failure",
                "target": "negative_symptoms_avolition",
                "relation": "supports avolition and diminished motivation",
                "schizoaffective_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "atypical_antipsychotic_coverage",
                "target": "dopamine_serotonin_interplay",
                "relation": "5-HT2A and D2 antagonism is expected to partially normalize monoaminergic dysregulation",
                "schizoaffective_change": "decreased",
                "weight": -0.30,
            },
            {
                "source": "atypical_antipsychotic_coverage",
                "target": "frontostriatal_psychosis_dysregulation",
                "relation": "reduces psychosis-driving circuit instability",
                "schizoaffective_change": "decreased",
                "weight": -0.30,
            },
            {
                "source": "atypical_antipsychotic_coverage",
                "target": "psychotic_symptoms",
                "relation": "expected to reduce positive and some negative symptom burden",
                "schizoaffective_change": "decreased",
                "weight": -0.30,
            },
            {
                "source": "atypical_antipsychotic_coverage",
                "target": "mixed_psychotic_mood_burden",
                "relation": "may confer partial mood-stabilizing benefit while reducing psychosis",
                "schizoaffective_change": "decreased",
                "weight": -0.20,
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
    def _float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, pd.Series):
                numeric = pd.to_numeric(value, errors="coerce")
                if numeric.notna().any():
                    return float(numeric.mean())
                return None
            return float(value)
        except Exception:
            try:
                return float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0])
            except Exception:
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
            "insula",
            "prefrontal cortex",
            "striatum",
            "basal ganglia",
            "putamen",
            "caudate nucleus",
        } else 0
        proxy_penalty = 1 if "proxy" in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

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
        volume_mm3 = self._float_or_none(getattr(main, "volume", float("nan")))
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

    def _choose_connectivity_feature(self, concept: Any) -> Optional[Any]:
        feats = self._safe_features_any(concept, self._modality_candidates("connectivity"))
        if not feats:
            return None

        cohort_matches = []
        for feat in feats:
            cohort_text = str(getattr(feat, "cohort", ""))
            name_text = str(getattr(feat, "name", ""))
            if self.connectivity_cohort.lower() in cohort_text.lower() or self.connectivity_cohort.lower() in name_text.lower():
                cohort_matches.append(feat)
        return cohort_matches[0] if cohort_matches else feats[0]

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feature = self._choose_connectivity_feature(self.parcellation)
        if feature is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        try:
            data = getattr(feature, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            first = feature[0]
            data = getattr(first, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        exact = [x for x in labels if self._name_of(x) == region.name]
        if exact:
            return exact[0]

        rn = region.name.lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        if fuzzy:
            return fuzzy[0]

        short_rn = (
            rn.replace("area ", "")
            .replace(" left", "")
            .replace(" right", "")
            .replace(" hemisphere", "")
            .replace(" proxy", "")
        )
        fuzzy_short = [
            x
            for x in labels
            if short_rn and short_rn in self._name_of(x).lower().replace("area ", "")
        ]
        return fuzzy_short[0] if fuzzy_short else None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        row_label = self._match_region_label(list(matrix.index), region)
        col_label = self._match_region_label(list(matrix.columns), region)
        series = None

        if row_label is not None:
            try:
                series = matrix.loc[row_label]
            except Exception:
                series = None

        if series is None and col_label is not None:
            try:
                series = matrix[col_label]
            except Exception:
                series = None

        if series is None:
            return pd.DataFrame()

        try:
            if isinstance(series, pd.DataFrame):
                series = series.mean(axis=0)
            numeric = pd.to_numeric(series, errors="coerce")
            numeric = numeric.dropna().sort_values(ascending=False)
            df = numeric.reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def build(self, gene_panel: Sequence[str] = DEFAULT_GENE_PANEL, connectivity_rows: int = 15) -> dict:
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
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
                        "node_type": "region",
                        "description": self.region_notes.get(
                            key,
                            "Atlas-backed node that could not be resolved in this environment.",
                        ),
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
                    "description": self.region_notes.get(key, "Atlas-backed circuit node."),
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

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        index_labels = list(matrix.index)
        column_labels = list(matrix.columns)

        for source_key, source_region in self.region_objects.items():
            src_row = self._match_region_label(index_labels, source_region)
            src_col = self._match_region_label(column_labels, source_region)
            for target_key, target_region in self.region_objects.items():
                if source_key == target_key:
                    continue
                tgt_row = self._match_region_label(index_labels, target_region)
                tgt_col = self._match_region_label(column_labels, target_region)
                value = None

                for lhs, rhs in ((src_row, tgt_col), (src_row, tgt_row), (src_col, tgt_col), (src_col, tgt_row)):
                    if lhs is None or rhs is None:
                        continue
                    try:
                        value = self._float_or_none(matrix.loc[lhs, rhs])
                    except Exception:
                        try:
                            value = self._float_or_none(matrix.loc[rhs, lhs])
                        except Exception:
                            value = None
                    if value is not None:
                        break

                if value is None:
                    continue
                rows.append(
                    {
                        "source_key": source_key,
                        "source_region": source_region.name,
                        "target_key": target_key,
                        "target_region": target_region.name,
                        "value": value,
                    }
                )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def simulate(
        self,
        genetic_vulnerability: float = 0.62,
        neurodevelopmental_burden: float = 0.56,
        chronic_stress_load: float = 0.52,
        depressive_episode_load: float = 0.48,
        manic_mixed_load: float = 0.36,
        prodromal_decline: float = 0.44,
        atypical_antipsychotic_coverage: float = 0.34,
    ) -> Dict[str, pd.Series]:
        """
        Run a one-pass normalized simulation.

        The calculation order is intentionally acyclic and transparent:
        inputs -> latent biology -> regional state -> symptoms -> phenotype summaries
        """

        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "neurodevelopmental_burden": self._clip01(neurodevelopmental_burden),
                "chronic_stress_load": self._clip01(chronic_stress_load),
                "depressive_episode_load": self._clip01(depressive_episode_load),
                "manic_mixed_load": self._clip01(manic_mixed_load),
                "prodromal_decline": self._clip01(prodromal_decline),
                "atypical_antipsychotic_coverage": self._clip01(atypical_antipsychotic_coverage),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["dopamine_serotonin_interplay"] = self._clip01(
            0.25 * inputs["genetic_vulnerability"]
            + 0.20 * inputs["manic_mixed_load"]
            + 0.15 * inputs["depressive_episode_load"]
            + 0.15 * inputs["neurodevelopmental_burden"]
            + 0.10 * inputs["chronic_stress_load"]
            - 0.25 * inputs["atypical_antipsychotic_coverage"]
        )
        latents["noradrenergic_dysfunction"] = self._clip01(
            0.35 * inputs["depressive_episode_load"]
            + 0.25 * inputs["chronic_stress_load"]
            + 0.10 * inputs["genetic_vulnerability"]
            + 0.05 * inputs["prodromal_decline"]
        )
        latents["neurodevelopmental_connectivity_burden"] = self._clip01(
            0.35 * inputs["neurodevelopmental_burden"]
            + 0.25 * inputs["genetic_vulnerability"]
            + 0.15 * inputs["prodromal_decline"]
            + 0.10 * inputs["chronic_stress_load"]
        )
        latents["synaptic_plasticity_resilience_failure"] = self._clip01(
            0.25 * inputs["neurodevelopmental_burden"]
            + 0.25 * inputs["chronic_stress_load"]
            + 0.15 * latents["neurodevelopmental_connectivity_burden"]
            + 0.10 * inputs["depressive_episode_load"]
        )
        latents["hpa_axis_dysregulation"] = self._clip01(
            0.40 * inputs["chronic_stress_load"]
            + 0.20 * inputs["depressive_episode_load"]
            + 0.15 * latents["synaptic_plasticity_resilience_failure"]
            + 0.10 * inputs["genetic_vulnerability"]
        )
        latents["frontolimbic_mood_dysregulation"] = self._clip01(
            0.25 * inputs["depressive_episode_load"]
            + 0.25 * inputs["manic_mixed_load"]
            + 0.20 * latents["hpa_axis_dysregulation"]
            + 0.15 * latents["synaptic_plasticity_resilience_failure"]
            + 0.10 * latents["dopamine_serotonin_interplay"]
            - 0.10 * inputs["atypical_antipsychotic_coverage"]
        )
        latents["frontostriatal_psychosis_dysregulation"] = self._clip01(
            0.25 * latents["dopamine_serotonin_interplay"]
            + 0.25 * latents["neurodevelopmental_connectivity_burden"]
            + 0.15 * inputs["prodromal_decline"]
            + 0.10 * inputs["genetic_vulnerability"]
            - 0.25 * inputs["atypical_antipsychotic_coverage"]
        )
        latents["frontotemporal_disconnection"] = self._clip01(
            0.30 * latents["neurodevelopmental_connectivity_burden"]
            + 0.20 * latents["frontostriatal_psychosis_dysregulation"]
            + 0.15 * inputs["prodromal_decline"]
            + 0.10 * latents["synaptic_plasticity_resilience_failure"]
        )
        latents["thalamocortical_dysconnectivity"] = self._clip01(
            0.30 * latents["neurodevelopmental_connectivity_burden"]
            + 0.20 * latents["frontostriatal_psychosis_dysregulation"]
            + 0.15 * latents["synaptic_plasticity_resilience_failure"]
            + 0.10 * inputs["prodromal_decline"]
        )
        latents["psychotic_mood_coupling"] = self._clip01(
            0.25 * latents["frontolimbic_mood_dysregulation"]
            + 0.25 * latents["frontostriatal_psychosis_dysregulation"]
            + 0.15 * latents["dopamine_serotonin_interplay"]
            + 0.10 * inputs["depressive_episode_load"]
            + 0.10 * inputs["manic_mixed_load"]
            - 0.10 * inputs["atypical_antipsychotic_coverage"]
        )
        latents["top_down_control_failure"] = self._clip01(
            0.25 * latents["frontolimbic_mood_dysregulation"]
            + 0.25 * latents["frontostriatal_psychosis_dysregulation"]
            + 0.20 * latents["thalamocortical_dysconnectivity"]
            + 0.15 * latents["frontotemporal_disconnection"]
            - 0.15 * inputs["atypical_antipsychotic_coverage"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["pfc_control"] = self._clip01(
            0.40 * latents["top_down_control_failure"]
            + 0.20 * latents["frontotemporal_disconnection"]
            + 0.10 * latents["neurodevelopmental_connectivity_burden"]
            - 0.15 * inputs["atypical_antipsychotic_coverage"]
        )
        regional_state["amygdala"] = self._clip01(
            0.35 * latents["frontolimbic_mood_dysregulation"]
            + 0.20 * latents["hpa_axis_dysregulation"]
            + 0.10 * latents["psychotic_mood_coupling"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.25 * latents["hpa_axis_dysregulation"]
            + 0.25 * latents["synaptic_plasticity_resilience_failure"]
            + 0.15 * latents["neurodevelopmental_connectivity_burden"]
            + 0.10 * latents["top_down_control_failure"]
        )
        regional_state["thalamus"] = self._clip01(
            0.35 * latents["thalamocortical_dysconnectivity"]
            + 0.20 * latents["frontostriatal_psychosis_dysregulation"]
            + 0.10 * latents["psychotic_mood_coupling"]
            - 0.10 * inputs["atypical_antipsychotic_coverage"]
        )
        regional_state["basal_ganglia_proxy"] = self._clip01(
            0.35 * latents["frontostriatal_psychosis_dysregulation"]
            + 0.20 * latents["dopamine_serotonin_interplay"]
            + 0.10 * latents["psychotic_mood_coupling"]
            - 0.10 * inputs["atypical_antipsychotic_coverage"]
        )
        regional_state["temporal_cortex_proxy"] = self._clip01(
            0.35 * latents["frontotemporal_disconnection"]
            + 0.20 * latents["thalamocortical_dysconnectivity"]
            + 0.10 * latents["psychotic_mood_coupling"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["psychotic_symptoms"] = self._clip01(
            0.35 * latents["frontostriatal_psychosis_dysregulation"]
            + 0.20 * regional_state["thalamus"]
            + 0.15 * regional_state["basal_ganglia_proxy"]
            + 0.15 * latents["psychotic_mood_coupling"]
            - 0.25 * inputs["atypical_antipsychotic_coverage"]
        )
        symptoms["disorganized_thought"] = self._clip01(
            0.30 * latents["frontotemporal_disconnection"]
            + 0.25 * regional_state["pfc_control"]
            + 0.15 * regional_state["temporal_cortex_proxy"]
            + 0.10 * regional_state["thalamus"]
        )
        symptoms["depressive_fatigue_concentration"] = self._clip01(
            0.35 * latents["noradrenergic_dysfunction"]
            + 0.20 * latents["hpa_axis_dysregulation"]
            + 0.20 * inputs["depressive_episode_load"]
            + 0.10 * regional_state["hippocampus"]
        )
        symptoms["manic_affective_activation"] = self._clip01(
            0.30 * inputs["manic_mixed_load"]
            + 0.25 * latents["frontolimbic_mood_dysregulation"]
            + 0.15 * latents["dopamine_serotonin_interplay"]
            - 0.10 * inputs["atypical_antipsychotic_coverage"]
        )
        symptoms["emotional_lability"] = self._clip01(
            0.30 * latents["frontolimbic_mood_dysregulation"]
            + 0.20 * regional_state["amygdala"]
            + 0.15 * symptoms["manic_affective_activation"]
            + 0.10 * inputs["depressive_episode_load"]
        )
        symptoms["cognitive_impairment"] = self._clip01(
            0.30 * regional_state["pfc_control"]
            + 0.20 * regional_state["hippocampus"]
            + 0.15 * latents["synaptic_plasticity_resilience_failure"]
            + 0.10 * symptoms["depressive_fatigue_concentration"]
            + 0.10 * symptoms["disorganized_thought"]
        )
        symptoms["negative_symptoms_avolition"] = self._clip01(
            0.25 * regional_state["pfc_control"]
            + 0.25 * symptoms["cognitive_impairment"]
            + 0.15 * inputs["depressive_episode_load"]
            + 0.10 * latents["top_down_control_failure"]
            - 0.10 * inputs["atypical_antipsychotic_coverage"]
        )
        symptoms["mixed_psychotic_mood_burden"] = self._clip01(
            0.30 * latents["psychotic_mood_coupling"]
            + 0.20 * symptoms["psychotic_symptoms"]
            + 0.15 * symptoms["emotional_lability"]
            + 0.10 * symptoms["depressive_fatigue_concentration"]
            + 0.10 * symptoms["manic_affective_activation"]
            - 0.10 * inputs["atypical_antipsychotic_coverage"]
        )

        phenotypes = pd.Series(dtype=float, name="phenotypes")
        phenotypes["depressive_type_profile"] = self._clip01(
            0.30 * symptoms["depressive_fatigue_concentration"]
            + 0.20 * symptoms["psychotic_symptoms"]
            + 0.15 * symptoms["negative_symptoms_avolition"]
            + 0.10 * latents["noradrenergic_dysfunction"]
        )
        phenotypes["bipolar_type_profile"] = self._clip01(
            0.30 * symptoms["manic_affective_activation"]
            + 0.20 * symptoms["emotional_lability"]
            + 0.20 * symptoms["psychotic_symptoms"]
            + 0.10 * latents["dopamine_serotonin_interplay"]
        )
        phenotypes["mixed_psychotic_mood_profile"] = self._clip01(
            0.30 * symptoms["mixed_psychotic_mood_burden"]
            + 0.20 * symptoms["psychotic_symptoms"]
            + 0.15 * symptoms["emotional_lability"]
            + 0.10 * latents["psychotic_mood_coupling"]
        )
        phenotypes["cognitive_disorganized_profile"] = self._clip01(
            0.30 * symptoms["cognitive_impairment"]
            + 0.30 * symptoms["disorganized_thought"]
            + 0.15 * regional_state["pfc_control"]
            + 0.10 * regional_state["temporal_cortex_proxy"]
        )
        phenotypes["overall_schizoaffective_severity"] = self._clip01(
            0.20 * symptoms["psychotic_symptoms"]
            + 0.20 * symptoms["mixed_psychotic_mood_burden"]
            + 0.15 * symptoms["cognitive_impairment"]
            + 0.15 * symptoms["emotional_lability"]
            + 0.10 * symptoms["negative_symptoms_avolition"]
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

        point = siibra.Point(tuple(float(x) for x in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        sort_priority = [
            "map value",
            "value",
            "correlation",
            "contains",
            "contained",
            "intersection over union",
        ]
        lower_cols = {str(c).lower(): c for c in assignments.columns}
        for candidate in sort_priority:
            if candidate in lower_cols:
                assignments = assignments.sort_values(lower_cols[candidate], ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        region = self.region_objects.get(node_key)
        if region is None:
            candidates = self.region_candidates.get(node_key)
            if not candidates:
                return None
            region = self._resolve_region(candidates)
            if region is None:
                return None

        try:
            mask = region.get_regional_mask(space=self.assignment_space, maptype="labelled")
            return mask.fetch() if hasattr(mask, "fetch") else mask
        except Exception:
            pass

        try:
            return region.fetch_regional_map(space=self.assignment_space, maptype="labelled")
        except Exception:
            return None




if __name__ == "__main__":
    if siibra is None:
        print(
            "siibra is not installed in this environment. "
            "Install siibra-python to run the atlas-backed parts of this scaffold."
        )
    else:
        model = SchizoaffectiveDisorderModel()
        bundle = model.build(connectivity_rows=10)

        print("\n=== Nodes ===")
        print(
            bundle["nodes"][["key", "node_type", "atlas_region", "feature_summary"]]
            .fillna("")
            .to_string(index=False)
        )

        print("\n=== Edges (first 14) ===")
        print(bundle["edges"].head(14).to_string(index=False))

        print("\n=== Region suggestions for 'temporal' ===")
        print(model.suggest_regions("temporal").head(10).to_string(index=False))

        print("\n=== Example receptor / gene / connectivity tables ===")
        for region_key in (
            "pfc_control",
            "amygdala",
            "hippocampus",
            "thalamus",
            "basal_ganglia_proxy",
            "temporal_cortex_proxy",
        ):
            if region_key in bundle["regions"]:
                print(f"\n[{region_key}] receptor rows: {len(bundle['receptors'][region_key])}")
                print(f"[{region_key}] gene rows: {len(bundle['genes'][region_key])}")
                print(f"[{region_key}] connectivity rows: {len(bundle['connectivity_profiles'][region_key])}")

        print("\n=== Example simulation ===")
        sim = model.simulate(
            genetic_vulnerability=0.63,
            neurodevelopmental_burden=0.58,
            chronic_stress_load=0.55,
            depressive_episode_load=0.52,
            manic_mixed_load=0.38,
            prodromal_decline=0.45,
            atypical_antipsychotic_coverage=0.35,
        )
        for name, series in sim.items():
            print(f"\n-- {name} --")
            print(series.sort_values(ascending=False).to_string())

        # Example coordinate assignment:
        # print(model.assign_mni_point((-20, -8, 10)).head())
