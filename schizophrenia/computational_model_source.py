
from __future__ import annotations

"""
Schizophrenia siibra scaffold.

This script translates a chapter-level biological overview of schizophrenia into
a transparent, atlas-grounded research scaffold using siibra idioms. It is
intended as a reusable starting point for mechanistic exploration, not as a
diagnostic or treatment tool.

Important notes
---------------
- This is a research scaffold, not a clinical model.
- The simulator is a faithful mechanistic interpretation of the supplied
  chapter, not a validated disease simulation.
- The chapter emphasizes distributed circuit dysfunction rather than a single
  lesion, so several nodes are modeled as conservative proxies instead of
  over-precise parcels.
- Dopamine, serotonin, GABA, and glutamate are represented primarily as latent
  biology rather than forced into single anatomical parcels.
- Hallucination-related circuitry is represented conservatively through Heschl's
  gyrus, inferotemporal cortex proxy, thalamus, and basal ganglia proxy.
- The gene panel is heuristic and aims to cover dopamine, serotonin,
  GABA/glutamate balance, neurodevelopment, myelination, and synaptic-plasticity
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
    "COMT",
    "HTR2A",
    "HTR1A",
    "SLC6A4",
    "GAD1",
    "GABRA1",
    "GABRB2",
    "GRIN1",
    "GRIN2A",
    "GRM3",
    "DISC1",
    "NRG1",
    "RELN",
    "DTNBP1",
    "CACNA1C",
    "BDNF",
    "MBP",
    "TCF4",
    "ZNF804A",
]


class SchizophreniaModel:
    """
    Atlas-grounded scaffold for schizophrenia.

    Chapter logic encoded here emphasizes:
    - polygenic and neurodevelopmental vulnerability within a diathesis-stress /
      multiple-hit framework;
    - dopamine dysregulation interacting with serotonergic, GABAergic, and
      glutamatergic disturbance rather than acting alone;
    - hippocampal GABA loss, cortical disinhibition, and working-memory /
      oscillatory disruption;
    - thalamocortical, frontotemporal, and broader large-scale circuit
      dysconnectivity;
    - auditory hallucination circuitry involving Heschl's gyrus and visual
      perceptual disturbance involving inferotemporal-thalamic-basal ganglia
      pathways;
    - distributed frontal, cingulate, thalamic, temporal, and cerebellar
      involvement in psychosis, cognition, and social dysfunction.
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
                "siibra is required to instantiate SchizophreniaModel. "
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
            "acc": [
                "Area p24pr left",
                "Area p24 left",
                "Area p32pr left",
                "anterior cingulate",
                "cingulate",
            ],
            "thalamus": [
                "Thalamus left",
                "thalamus",
                "Pulvinar left",
                "Mediodorsal thalamus left",
            ],
            "hippocampus": [
                "Hippocampus left",
                "CA1 left",
                "Subiculum left",
                "hippocampus",
            ],
            "heschls_gyrus": [
                "Area TE 1.0 (HESCHL) left",
                "Area TE 1.2 (HESCHL) left",
                "Heschl",
                "Heschl's gyrus",
                "transverse temporal",
            ],
            "inferotemporal_cortex_proxy": [
                "Area TE 3 left",
                "Area TE 2.1 (STG) left",
                "fusiform gyrus",
                "inferotemporal",
                "temporal inferior",
            ],
            "basal_ganglia_proxy": [
                "Putamen left",
                "Caudate nucleus left",
                "Nucleus accumbens left",
                "striatum",
                "basal ganglia",
            ],
            "cerebellum_proxy": [
                "cerebellum left",
                "cerebellum",
            ],
        }

        self.region_notes: Dict[str, str] = {
            "pfc_control": (
                "Proxy for the orbital, dorsolateral, and medial frontal systems "
                "named by the chapter and implicated in top-down control."
            ),
            "acc": (
                "Anterior cingulate region explicitly named in the chapter's PET "
                "network abnormalities."
            ),
            "thalamus": (
                "Named thalamic involvement in large-scale dysconnectivity and in "
                "hallucination-related perceptual circuits."
            ),
            "hippocampus": (
                "Anchors the chapter's hypothesis of hippocampal GABAergic loss "
                "and downstream dopaminergic dysregulation."
            ),
            "heschls_gyrus": (
                "Named auditory processing substrate for spontaneous activity "
                "associated with auditory hallucinations."
            ),
            "inferotemporal_cortex_proxy": (
                "Proxy for the inferotemporal visual-recognition system named in "
                "the chapter's visual-hallucination circuit."
            ),
            "basal_ganglia_proxy": (
                "Proxy for basal ganglia components of dopaminergic and perceptual "
                "circuit disturbance."
            ),
            "cerebellum_proxy": (
                "Proxy for the cerebellar abnormalities noted in PET studies; may "
                "remain unresolved in Julich-only environments."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Polygenic liability and familial susceptibility contributing to "
                "schizophrenia risk."
            ),
            "neurodevelopmental_burden": (
                "Early developmental disruption affecting maturation, synapse "
                "formation, and myelination."
            ),
            "environmental_stress_load": (
                "Stress-related and other non-genetic pressures that interact with "
                "underlying vulnerability."
            ),
            "multiple_hit_burden": (
                "Accumulated non-genetic hits in a diathesis-stress / multiple-hit "
                "framework."
            ),
            "prodromal_decline": (
                "Preclinical functional decline suggesting emerging network "
                "instability before full illness expression."
            ),
            "atypical_antipsychotic_coverage": (
                "Protective D2 / 5-HT2A antagonism expected to reduce positive "
                "symptoms and partly rebalance regional dysregulation."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonin_5ht2a_dysregulation": (
                "Serotonergic dysregulation, especially 5-HT2A-linked modulation of "
                "perception and cortical/striatal dopamine release."
            ),
            "gaba_interneuron_deficit": (
                "Reduced inhibitory interneuron function, especially hippocampal / "
                "cortical GABAergic deficits."
            ),
            "glutamate_gaba_imbalance": (
                "Excitatory-inhibitory imbalance emerging from GABA loss and "
                "glutamatergic dysregulation."
            ),
            "hippocampal_disinhibition": (
                "Hippocampal overactivity and disinhibition that may amplify "
                "downstream dopaminergic abnormalities."
            ),
            "dopamine_salience_dysregulation": (
                "Aberrant dopaminergic salience assignment relevant to psychosis "
                "and perceptual misattribution."
            ),
            "cortical_hyperexcitability": (
                "Cortical disinhibition and hyperexcitability contributing to "
                "psychotic and cognitive symptoms."
            ),
            "working_memory_oscillation_failure": (
                "Breakdown of inhibitory synchronization needed for working memory "
                "and cognitive control."
            ),
            "large_scale_circuit_disruption": (
                "Distributed dysfunction across frontal, cingulate, thalamic, "
                "temporal, and cerebellar networks."
            ),
            "thalamocortical_dysconnectivity": (
                "Impaired coordination between thalamic relay systems and cortex."
            ),
            "frontotemporal_dysconnectivity": (
                "Dysconnectivity between frontal control systems and temporal "
                "perceptual / language systems."
            ),
            "salience_misattribution": (
                "Misassignment of internally generated activity as externally "
                "meaningful or threatening."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "positive_psychotic_symptoms": (
                "Delusions, hallucinations, and related positive psychotic burden."
            ),
            "auditory_hallucinations": (
                "Misattributed internal auditory activity experienced as external "
                "voices."
            ),
            "visual_hallucination_vulnerability": (
                "Propensity toward altered visual perception through basal "
                "ganglia-thalamic-inferotemporal disruption."
            ),
            "disorganized_thought": (
                "Thought disorder and disorganization from large-scale circuit "
                "breakdown."
            ),
            "cognitive_impairment": (
                "Deficits in working memory, attention, and executive processing."
            ),
            "negative_symptoms": (
                "Blunted affect, avolition, and related negative symptom burden."
            ),
            "social_function_decline": (
                "Decline in social and psychosocial functioning downstream of "
                "cognitive and psychotic burden."
            ),
        }

        self.edge_table: List[Dict[str, Any]] = [
            {
                "source": "genetic_vulnerability",
                "target": "large_scale_circuit_disruption",
                "relation": "confers broad inherited vulnerability to distributed circuit dysfunction",
                "schizophrenia_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "genetic_vulnerability",
                "target": "dopamine_salience_dysregulation",
                "relation": "biases dopamine-related salience mechanisms",
                "schizophrenia_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "genetic_vulnerability",
                "target": "gaba_interneuron_deficit",
                "relation": "raises vulnerability to inhibitory interneuron dysfunction",
                "schizophrenia_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "neurodevelopmental_burden",
                "target": "large_scale_circuit_disruption",
                "relation": "disrupts maturation of frontal, temporal, and thalamic systems",
                "schizophrenia_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "multiple_hit_burden",
                "target": "large_scale_circuit_disruption",
                "relation": "adds cumulative non-genetic insults in the multiple-hit framework",
                "schizophrenia_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "environmental_stress_load",
                "target": "serotonin_5ht2a_dysregulation",
                "relation": "can amplify perceptual and affective serotonergic dysregulation",
                "schizophrenia_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "prodromal_decline",
                "target": "frontotemporal_dysconnectivity",
                "relation": "signals emerging disconnection before overt psychosis",
                "schizophrenia_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "gaba_interneuron_deficit",
                "target": "glutamate_gaba_imbalance",
                "relation": "weakens inhibition and destabilizes excitatory-inhibitory balance",
                "schizophrenia_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "gaba_interneuron_deficit",
                "target": "hippocampal_disinhibition",
                "relation": "loss of hippocampal GABAergic control may increase hippocampal drive",
                "schizophrenia_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "glutamate_gaba_imbalance",
                "target": "cortical_hyperexcitability",
                "relation": "produces disinhibited cortical firing and instability",
                "schizophrenia_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "gaba_interneuron_deficit",
                "target": "working_memory_oscillation_failure",
                "relation": "disrupts inhibitory synchronization needed for working memory",
                "schizophrenia_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "hippocampal_disinhibition",
                "target": "dopamine_salience_dysregulation",
                "relation": "may elevate downstream dopaminergic drive",
                "schizophrenia_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "serotonin_5ht2a_dysregulation",
                "target": "dopamine_salience_dysregulation",
                "relation": "modulates regional dopamine release in cortex and striatum",
                "schizophrenia_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "large_scale_circuit_disruption",
                "target": "thalamocortical_dysconnectivity",
                "relation": "breaks coordination between thalamus and cortex",
                "schizophrenia_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "large_scale_circuit_disruption",
                "target": "frontotemporal_dysconnectivity",
                "relation": "compromises frontal-temporal integration",
                "schizophrenia_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "cortical_hyperexcitability",
                "target": "frontotemporal_dysconnectivity",
                "relation": "destabilizes distributed perceptual and cognitive networks",
                "schizophrenia_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "thalamocortical_dysconnectivity",
                "target": "salience_misattribution",
                "relation": "distorts relay and filtering of internally generated signals",
                "schizophrenia_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "dopamine_salience_dysregulation",
                "target": "salience_misattribution",
                "relation": "drives aberrant assignment of importance to internal events",
                "schizophrenia_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "cortical_hyperexcitability",
                "target": "salience_misattribution",
                "relation": "increases spontaneous internally generated activity available for misattribution",
                "schizophrenia_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "salience_misattribution",
                "target": "positive_psychotic_symptoms",
                "relation": "contributes directly to delusions and hallucination burden",
                "schizophrenia_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "frontotemporal_dysconnectivity",
                "target": "auditory_hallucinations",
                "relation": "destabilizes auditory-language circuits",
                "schizophrenia_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "thalamocortical_dysconnectivity",
                "target": "auditory_hallucinations",
                "relation": "disrupts perceptual gating in auditory relay systems",
                "schizophrenia_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "salience_misattribution",
                "target": "auditory_hallucinations",
                "relation": "supports external attribution of internal auditory activity",
                "schizophrenia_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "dopamine_salience_dysregulation",
                "target": "visual_hallucination_vulnerability",
                "relation": "contributes to basal-ganglia-linked visual distortions",
                "schizophrenia_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "thalamocortical_dysconnectivity",
                "target": "visual_hallucination_vulnerability",
                "relation": "perturbs thalamic contributions to visual perception",
                "schizophrenia_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "frontotemporal_dysconnectivity",
                "target": "disorganized_thought",
                "relation": "impairs coherent integration of language and executive control",
                "schizophrenia_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "thalamocortical_dysconnectivity",
                "target": "disorganized_thought",
                "relation": "reduces coordinated information flow needed for organized thought",
                "schizophrenia_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "working_memory_oscillation_failure",
                "target": "cognitive_impairment",
                "relation": "drives working-memory and attentional impairment",
                "schizophrenia_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "frontotemporal_dysconnectivity",
                "target": "cognitive_impairment",
                "relation": "compromises executive-linguistic integration",
                "schizophrenia_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "large_scale_circuit_disruption",
                "target": "negative_symptoms",
                "relation": "supports broader affective and motivational flattening",
                "schizophrenia_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "cognitive_impairment",
                "target": "social_function_decline",
                "relation": "reduces real-world functioning and psychosocial adaptation",
                "schizophrenia_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "negative_symptoms",
                "target": "social_function_decline",
                "relation": "worsens motivation and engagement with others",
                "schizophrenia_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "positive_psychotic_symptoms",
                "target": "social_function_decline",
                "relation": "contributes to broader psychosocial impairment",
                "schizophrenia_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "atypical_antipsychotic_coverage",
                "target": "serotonin_5ht2a_dysregulation",
                "relation": "5-HT2A antagonism is expected to partially normalize serotonergic overactivity",
                "schizophrenia_change": "decreased",
                "weight": -0.30,
            },
            {
                "source": "atypical_antipsychotic_coverage",
                "target": "dopamine_salience_dysregulation",
                "relation": "D2 antagonism is expected to reduce positive-symptom-driving dopamine dysregulation",
                "schizophrenia_change": "decreased",
                "weight": -0.35,
            },
            {
                "source": "atypical_antipsychotic_coverage",
                "target": "positive_psychotic_symptoms",
                "relation": "expected to directly reduce psychotic burden",
                "schizophrenia_change": "decreased",
                "weight": -0.30,
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
        genetic_vulnerability: float = 0.65,
        neurodevelopmental_burden: float = 0.60,
        environmental_stress_load: float = 0.45,
        multiple_hit_burden: float = 0.50,
        prodromal_decline: float = 0.55,
        atypical_antipsychotic_coverage: float = 0.35,
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
                "environmental_stress_load": self._clip01(environmental_stress_load),
                "multiple_hit_burden": self._clip01(multiple_hit_burden),
                "prodromal_decline": self._clip01(prodromal_decline),
                "atypical_antipsychotic_coverage": self._clip01(atypical_antipsychotic_coverage),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["serotonin_5ht2a_dysregulation"] = self._clip01(
            0.30 * inputs["genetic_vulnerability"]
            + 0.20 * inputs["environmental_stress_load"]
            + 0.15 * inputs["multiple_hit_burden"]
            + 0.10 * inputs["prodromal_decline"]
            - 0.25 * inputs["atypical_antipsychotic_coverage"]
        )
        latents["gaba_interneuron_deficit"] = self._clip01(
            0.30 * inputs["genetic_vulnerability"]
            + 0.30 * inputs["neurodevelopmental_burden"]
            + 0.15 * inputs["prodromal_decline"]
            + 0.10 * inputs["multiple_hit_burden"]
        )
        latents["glutamate_gaba_imbalance"] = self._clip01(
            0.35 * latents["gaba_interneuron_deficit"]
            + 0.20 * inputs["neurodevelopmental_burden"]
            + 0.10 * inputs["environmental_stress_load"]
            + 0.10 * inputs["multiple_hit_burden"]
        )
        latents["hippocampal_disinhibition"] = self._clip01(
            0.35 * latents["gaba_interneuron_deficit"]
            + 0.20 * latents["glutamate_gaba_imbalance"]
            + 0.15 * inputs["prodromal_decline"]
            + 0.10 * inputs["neurodevelopmental_burden"]
        )
        latents["dopamine_salience_dysregulation"] = self._clip01(
            0.25 * inputs["genetic_vulnerability"]
            + 0.20 * latents["hippocampal_disinhibition"]
            + 0.20 * latents["serotonin_5ht2a_dysregulation"]
            + 0.10 * inputs["multiple_hit_burden"]
            - 0.30 * inputs["atypical_antipsychotic_coverage"]
        )
        latents["cortical_hyperexcitability"] = self._clip01(
            0.35 * latents["glutamate_gaba_imbalance"]
            + 0.25 * latents["gaba_interneuron_deficit"]
            + 0.10 * latents["serotonin_5ht2a_dysregulation"]
            + 0.05 * inputs["environmental_stress_load"]
        )
        latents["working_memory_oscillation_failure"] = self._clip01(
            0.35 * latents["gaba_interneuron_deficit"]
            + 0.25 * latents["glutamate_gaba_imbalance"]
            + 0.15 * inputs["prodromal_decline"]
            + 0.10 * inputs["neurodevelopmental_burden"]
        )
        latents["large_scale_circuit_disruption"] = self._clip01(
            0.35 * inputs["neurodevelopmental_burden"]
            + 0.25 * inputs["genetic_vulnerability"]
            + 0.15 * inputs["multiple_hit_burden"]
            + 0.10 * inputs["prodromal_decline"]
        )
        latents["thalamocortical_dysconnectivity"] = self._clip01(
            0.30 * latents["large_scale_circuit_disruption"]
            + 0.20 * latents["dopamine_salience_dysregulation"]
            + 0.15 * inputs["neurodevelopmental_burden"]
            + 0.10 * inputs["multiple_hit_burden"]
            - 0.10 * inputs["atypical_antipsychotic_coverage"]
        )
        latents["frontotemporal_dysconnectivity"] = self._clip01(
            0.30 * latents["large_scale_circuit_disruption"]
            + 0.20 * latents["cortical_hyperexcitability"]
            + 0.15 * inputs["prodromal_decline"]
            + 0.15 * inputs["neurodevelopmental_burden"]
        )
        latents["salience_misattribution"] = self._clip01(
            0.35 * latents["dopamine_salience_dysregulation"]
            + 0.25 * latents["thalamocortical_dysconnectivity"]
            + 0.15 * latents["cortical_hyperexcitability"]
            - 0.20 * inputs["atypical_antipsychotic_coverage"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["pfc_control"] = self._clip01(
            0.35 * latents["frontotemporal_dysconnectivity"]
            + 0.25 * latents["working_memory_oscillation_failure"]
            + 0.15 * latents["thalamocortical_dysconnectivity"]
            + 0.10 * latents["large_scale_circuit_disruption"]
            - 0.10 * inputs["atypical_antipsychotic_coverage"]
        )
        regional_state["acc"] = self._clip01(
            0.30 * latents["salience_misattribution"]
            + 0.20 * latents["thalamocortical_dysconnectivity"]
            + 0.20 * regional_state["pfc_control"]
            + 0.10 * latents["cortical_hyperexcitability"]
        )
        regional_state["thalamus"] = self._clip01(
            0.35 * latents["thalamocortical_dysconnectivity"]
            + 0.20 * latents["dopamine_salience_dysregulation"]
            + 0.15 * latents["large_scale_circuit_disruption"]
            - 0.10 * inputs["atypical_antipsychotic_coverage"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.40 * latents["hippocampal_disinhibition"]
            + 0.20 * latents["gaba_interneuron_deficit"]
            + 0.15 * latents["cortical_hyperexcitability"]
        )
        regional_state["heschls_gyrus"] = self._clip01(
            0.30 * latents["frontotemporal_dysconnectivity"]
            + 0.25 * latents["thalamocortical_dysconnectivity"]
            + 0.15 * latents["salience_misattribution"]
            + 0.10 * latents["cortical_hyperexcitability"]
        )
        regional_state["inferotemporal_cortex_proxy"] = self._clip01(
            0.30 * latents["frontotemporal_dysconnectivity"]
            + 0.25 * latents["thalamocortical_dysconnectivity"]
            + 0.15 * latents["dopamine_salience_dysregulation"]
            + 0.10 * latents["salience_misattribution"]
        )
        regional_state["basal_ganglia_proxy"] = self._clip01(
            0.40 * latents["dopamine_salience_dysregulation"]
            + 0.20 * latents["salience_misattribution"]
            + 0.10 * latents["thalamocortical_dysconnectivity"]
            - 0.10 * inputs["atypical_antipsychotic_coverage"]
        )
        regional_state["cerebellum_proxy"] = self._clip01(
            0.30 * latents["large_scale_circuit_disruption"]
            + 0.20 * latents["thalamocortical_dysconnectivity"]
            + 0.20 * regional_state["pfc_control"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["positive_psychotic_symptoms"] = self._clip01(
            0.35 * latents["salience_misattribution"]
            + 0.25 * latents["dopamine_salience_dysregulation"]
            + 0.15 * regional_state["acc"]
            + 0.10 * regional_state["thalamus"]
            - 0.25 * inputs["atypical_antipsychotic_coverage"]
        )
        symptoms["auditory_hallucinations"] = self._clip01(
            0.35 * regional_state["heschls_gyrus"]
            + 0.20 * symptoms["positive_psychotic_symptoms"]
            + 0.15 * regional_state["thalamus"]
            + 0.10 * latents["salience_misattribution"]
            - 0.15 * inputs["atypical_antipsychotic_coverage"]
        )
        symptoms["visual_hallucination_vulnerability"] = self._clip01(
            0.30 * regional_state["inferotemporal_cortex_proxy"]
            + 0.20 * regional_state["basal_ganglia_proxy"]
            + 0.20 * regional_state["thalamus"]
            + 0.10 * symptoms["positive_psychotic_symptoms"]
            - 0.10 * inputs["atypical_antipsychotic_coverage"]
        )
        symptoms["disorganized_thought"] = self._clip01(
            0.30 * regional_state["pfc_control"]
            + 0.25 * latents["frontotemporal_dysconnectivity"]
            + 0.20 * regional_state["acc"]
            + 0.10 * regional_state["thalamus"]
        )
        symptoms["cognitive_impairment"] = self._clip01(
            0.35 * latents["working_memory_oscillation_failure"]
            + 0.20 * regional_state["pfc_control"]
            + 0.15 * regional_state["hippocampus"]
            + 0.10 * regional_state["cerebellum_proxy"]
        )
        symptoms["negative_symptoms"] = self._clip01(
            0.30 * regional_state["pfc_control"]
            + 0.25 * latents["large_scale_circuit_disruption"]
            + 0.15 * symptoms["cognitive_impairment"]
            + 0.05 * inputs["prodromal_decline"]
            - 0.10 * inputs["atypical_antipsychotic_coverage"]
        )
        symptoms["social_function_decline"] = self._clip01(
            0.30 * symptoms["negative_symptoms"]
            + 0.25 * symptoms["cognitive_impairment"]
            + 0.15 * symptoms["positive_psychotic_symptoms"]
            + 0.10 * symptoms["disorganized_thought"]
        )

        phenotypes = pd.Series(dtype=float, name="phenotypes")
        phenotypes["paranoid_hallucinatory_profile"] = self._clip01(
            0.30 * symptoms["positive_psychotic_symptoms"]
            + 0.30 * symptoms["auditory_hallucinations"]
            + 0.15 * latents["salience_misattribution"]
            + 0.10 * regional_state["thalamus"]
        )
        phenotypes["disorganized_cognitive_profile"] = self._clip01(
            0.30 * symptoms["disorganized_thought"]
            + 0.30 * symptoms["cognitive_impairment"]
            + 0.15 * regional_state["pfc_control"]
            + 0.10 * latents["frontotemporal_dysconnectivity"]
        )
        phenotypes["negative_symptom_profile"] = self._clip01(
            0.35 * symptoms["negative_symptoms"]
            + 0.25 * symptoms["social_function_decline"]
            + 0.15 * symptoms["cognitive_impairment"]
            + 0.10 * regional_state["pfc_control"]
        )
        phenotypes["multimodal_perceptual_psychosis_profile"] = self._clip01(
            0.25 * symptoms["auditory_hallucinations"]
            + 0.25 * symptoms["visual_hallucination_vulnerability"]
            + 0.20 * symptoms["positive_psychotic_symptoms"]
            + 0.10 * regional_state["basal_ganglia_proxy"]
        )
        phenotypes["overall_schizophrenia_severity"] = self._clip01(
            0.20 * symptoms["positive_psychotic_symptoms"]
            + 0.20 * symptoms["cognitive_impairment"]
            + 0.15 * symptoms["negative_symptoms"]
            + 0.15 * symptoms["disorganized_thought"]
            + 0.10 * symptoms["social_function_decline"]
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
        model = SchizophreniaModel()
        bundle = model.build(connectivity_rows=10)

        print("\n=== Nodes ===")
        print(
            bundle["nodes"][["key", "node_type", "atlas_region", "feature_summary"]]
            .fillna("")
            .to_string(index=False)
        )

        print("\n=== Edges (first 14) ===")
        print(bundle["edges"].head(14).to_string(index=False))

        print("\n=== Region suggestions for 'thalamus' ===")
        print(model.suggest_regions("thalamus").head(10).to_string(index=False))

        print("\n=== Example receptor / gene / connectivity tables ===")
        for region_key in (
            "pfc_control",
            "acc",
            "thalamus",
            "hippocampus",
            "heschls_gyrus",
            "basal_ganglia_proxy",
        ):
            if region_key in bundle["regions"]:
                print(f"\n[{region_key}] receptor rows: {len(bundle['receptors'][region_key])}")
                print(f"[{region_key}] gene rows: {len(bundle['genes'][region_key])}")
                print(f"[{region_key}] connectivity rows: {len(bundle['connectivity_profiles'][region_key])}")

        print("\n=== Example simulation ===")
        sim = model.simulate(
            genetic_vulnerability=0.68,
            neurodevelopmental_burden=0.62,
            environmental_stress_load=0.47,
            multiple_hit_burden=0.54,
            prodromal_decline=0.58,
            atypical_antipsychotic_coverage=0.36,
        )
        for name, series in sim.items():
            print(f"\n-- {name} --")
            print(series.sort_values(ascending=False).to_string())

        # Example coordinate assignment:
        # print(model.assign_mni_point((-48, -18, 8)).head())
