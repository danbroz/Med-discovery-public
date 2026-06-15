from __future__ import annotations

"""
Functional Neurological Symptom Disorder (Conversion Disorder) siibra scaffold.

This script translates a Functional Neurological Symptom Disorder (FNSD) chapter
into a transparent, atlas-grounded research scaffold using siibra. It is a
mechanistic interpretation of chapter logic, not a validated disease model,
diagnostic tool, or treatment recommender.

The source chapter frames FNSD as a disorder of brain function rather than brain
structure, with altered top-down regulation from prefrontal and limbic systems
impacting motor and sensorimotor control. To stay faithful to the chapter while
remaining conservative, this scaffold anchors only regions that are explicitly
named or strongly implied by the text, and it keeps neurotransmitter and stress
biology as latent processes unless localization is justified.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except Exception as exc:  # pragma: no cover - depends on user environment
    siibra = None
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on user environment
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_FNSD_GENE_PANEL = [
    "SLC6A4",  # serotonin transporter
    "HTR1A",   # serotonergic receptor signaling
    "TPH2",    # serotonin synthesis
    "SLC6A2",  # norepinephrine transporter
    "COMT",    # catecholamine metabolism
    "BDNF",    # stress-sensitive plasticity
    "FKBP5",   # stress-axis sensitivity
    "NR3C1",   # glucocorticoid receptor
    "GAD1",    # GABA synthesis
    "GABRA2",  # GABA-A receptor subunit
    "SLC1A2",  # glutamate transport
    "GRIN2B",  # glutamatergic signaling
]


class FunctionalNeurologicalSymptomDisorderModel:
    """
    Atlas-grounded research scaffold for Functional Neurological Symptom Disorder.

    The model follows a one-pass transparent order:
        inputs -> latent biology -> regional dysregulation burden -> symptoms -> phenotypes

    Important caveats
    -----------------
    - Higher `regional_state` values indicate modeled *dysregulation burden* rather
      than healthy activation. For motor regions, a high burden often corresponds to
      impaired recruitment or excessive inhibition, whereas for frontal/limbic nodes
      it often corresponds to hypercontrol or hyperreactivity.
    - This is a chapter-faithful mechanistic scaffold for research and teaching. It
      should not be interpreted as a validated causal model of FNSD.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        if siibra is None:  # pragma: no cover - environment dependent
            raise ImportError(
                "siibra is required for this scaffold. Install siibra-python in the "
                "target environment before instantiating the model."
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
            "primary_motor_cortex": [
                "Area 4p (PreCG) left",
                "Area 4a (PreCG) left",
                "precentral gyrus left",
                "primary motor cortex",
                "Area 4p",
                "Area 4a",
            ],
            "supplementary_motor_area": [
                "Area 6mp (SMA, mesial SFG) left",
                "Area 6ma (preSMA, mesial SFG) left",
                "supplementary motor area left",
                "preSMA left",
                "SMA",
            ],
            "dlpfc": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "MFG1 left",
                "MFG2 left",
                "middle frontal gyrus left",
                "dorsolateral prefrontal cortex",
            ],
            "vlpfc": [
                "Area 45 (IFG) left",
                "Area 44 (IFG) left",
                "inferior frontal gyrus left",
                "ventrolateral prefrontal cortex",
                "IFG",
            ],
            "acc": [
                "Area p32 (pACC) left",
                "Area 33 (ACC) left",
                "Area s32 (sACC) left",
                "anterior cingulate cortex left",
                "cingulate cortex",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "IF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "primary_motor_cortex": "Motor execution node; burden here reflects hypoactivation or inhibitory suppression rather than tissue damage.",
            "supplementary_motor_area": "Motor preparation and initiation node implicated in impaired voluntary movement planning.",
            "dlpfc": "Executive-control region used as a dorsolateral prefrontal proxy for excessive top-down regulation.",
            "vlpfc": "Inferior frontal / ventrolateral inhibitory-control node implicated in nonconscious motor suppression.",
            "acc": "Anterior cingulate control-conflict and emotional-regulatory node often recruited during functional symptoms.",
            "amygdala": "Limbic salience region linking emotional arousal to symptom expression.",
        }

        self.input_nodes: Dict[str, str] = {
            "acute_stress_arousal": "Current stress, emotional conflict, or high arousal that can precipitate symptom expression without being required for diagnosis.",
            "chronic_stress_load": "Sustained stress burden contributing to broader functional dysregulation.",
            "comorbid_mood_anxiety": "Mood and anxiety burden used by the chapter to motivate monoaminergic and inhibitory dysregulation.",
            "genetic_diathesis": "Shared heritable liability overlapping with mood, anxiety, and related neurotic-spectrum disorders.",
            "neurotic_trait_liability": "Stable disposition toward high emotional reactivity and somatic amplification.",
            "neuroimmune_load": "Stress-linked inflammatory or neuroimmune burden included as a mechanistic vulnerability axis.",
            "treatment_support": "Protective psychiatric, rehabilitative, and functional retraining support.",
        }

        self.latent_nodes: Dict[str, str] = {
            "stress_response_sensitization": "Stress-system sensitization increasing arousal and symptom susceptibility.",
            "monoaminergic_dysregulation": "Serotonergic and noradrenergic dysregulation affecting mood, arousal, and cognitive control.",
            "gaba_glutamate_imbalance": "Excitatory-inhibitory imbalance affecting motor regulation, anxiety, and seizure-like phenomena.",
            "neuroimmune_stress_diathesis": "Shared stress-inflammatory susceptibility interacting with psychiatric vulnerability.",
            "altered_top_down_regulation": "Excessive or maladaptive frontal control over sensorimotor systems.",
            "limbic_motor_coupling": "Abnormal emotional-salience influence on motor and bodily control networks.",
            "top_down_motor_inhibition": "Active but nonconsciously willed suppression of motor output.",
            "aberrant_sensorimotor_integration": "Disordered integration of affective, sensory, and motor signals producing functional symptoms.",
        }

        self.symptom_nodes: Dict[str, str] = {
            "functional_weakness_paralysis": "Modeled burden for functional weakness or paralysis presentations.",
            "abnormal_movement_inhibition": "Modeled burden for abnormal movement, gait disturbance, or motor arrest/inhibition.",
            "functional_sensory_disturbance": "Modeled burden for sensory symptoms not explained by structural neurological lesions.",
            "seizure_like_phenomena": "Modeled burden for functional seizure-like or nonepileptic seizure-like events.",
            "somatic_distress_arousal": "Autonomic and affective bodily distress accompanying symptom episodes.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "acute_stress_arousal",
                "target": "stress_response_sensitization",
                "relation": "acute stress can precipitate high-arousal functional states",
                "fnsd_change": "increased",
            },
            {
                "source": "acute_stress_arousal",
                "target": "limbic_motor_coupling",
                "relation": "emotional arousal strengthens limbic influence on motor output",
                "fnsd_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "stress_response_sensitization",
                "relation": "ongoing stress sustains functional dysregulation risk",
                "fnsd_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "neuroimmune_stress_diathesis",
                "relation": "prolonged stress can reinforce neuroimmune vulnerability",
                "fnsd_change": "increased",
            },
            {
                "source": "comorbid_mood_anxiety",
                "target": "monoaminergic_dysregulation",
                "relation": "mood and anxiety comorbidity implicate serotonergic and noradrenergic dysregulation",
                "fnsd_change": "increased",
            },
            {
                "source": "comorbid_mood_anxiety",
                "target": "gaba_glutamate_imbalance",
                "relation": "anxious and agitated states can reflect altered inhibitory-excitatory balance",
                "fnsd_change": "increased",
            },
            {
                "source": "comorbid_mood_anxiety",
                "target": "altered_top_down_regulation",
                "relation": "comorbid affective burden can bias frontal control networks",
                "fnsd_change": "increased",
            },
            {
                "source": "genetic_diathesis",
                "target": "monoaminergic_dysregulation",
                "relation": "shared heritable liability can bias monoaminergic regulation",
                "fnsd_change": "increased",
            },
            {
                "source": "genetic_diathesis",
                "target": "neuroimmune_stress_diathesis",
                "relation": "genetic vulnerability may contribute to stress-reactive and neuroimmune susceptibility",
                "fnsd_change": "increased",
            },
            {
                "source": "neurotic_trait_liability",
                "target": "stress_response_sensitization",
                "relation": "high trait reactivity increases susceptibility to stress-linked symptom generation",
                "fnsd_change": "increased",
            },
            {
                "source": "neurotic_trait_liability",
                "target": "limbic_motor_coupling",
                "relation": "neurotic disposition amplifies affective capture of bodily experience",
                "fnsd_change": "increased",
            },
            {
                "source": "neuroimmune_load",
                "target": "neuroimmune_stress_diathesis",
                "relation": "neuroimmune burden contributes to broad vulnerability of functional circuits",
                "fnsd_change": "increased",
            },
            {
                "source": "treatment_support",
                "target": "stress_response_sensitization",
                "relation": "supportive care and functional retraining can reduce runaway stress responsivity",
                "fnsd_change": "decreased",
            },
            {
                "source": "treatment_support",
                "target": "altered_top_down_regulation",
                "relation": "treatment can reduce maladaptive top-down control and improve functional regulation",
                "fnsd_change": "decreased",
            },
            {
                "source": "treatment_support",
                "target": "top_down_motor_inhibition",
                "relation": "rehabilitative support can reduce nonconscious motor suppression",
                "fnsd_change": "decreased",
            },
            {
                "source": "neuroimmune_stress_diathesis",
                "target": "monoaminergic_dysregulation",
                "relation": "stress-inflammatory burden can worsen monoaminergic function",
                "fnsd_change": "increased",
            },
            {
                "source": "stress_response_sensitization",
                "target": "amygdala",
                "relation": "stress sensitization heightens limbic salience and alarm signaling",
                "fnsd_change": "increased",
            },
            {
                "source": "monoaminergic_dysregulation",
                "target": "dlpfc",
                "relation": "monoaminergic dysregulation perturbs frontal executive control systems",
                "fnsd_change": "increased",
            },
            {
                "source": "monoaminergic_dysregulation",
                "target": "acc",
                "relation": "monoaminergic dysregulation perturbs arousal and control monitoring systems",
                "fnsd_change": "increased",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "top_down_motor_inhibition",
                "relation": "excitatory-inhibitory imbalance can favor maladaptive motor suppression",
                "fnsd_change": "increased",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "aberrant_sensorimotor_integration",
                "relation": "excitatory-inhibitory dysbalance can distort sensory and motor signal integration",
                "fnsd_change": "increased",
            },
            {
                "source": "altered_top_down_regulation",
                "target": "dlpfc",
                "relation": "functional motor paradigms show excessive engagement of frontal control systems",
                "fnsd_change": "increased",
            },
            {
                "source": "altered_top_down_regulation",
                "target": "vlpfc",
                "relation": "ventrolateral inhibitory-control systems contribute to nonconscious motor suppression",
                "fnsd_change": "increased",
            },
            {
                "source": "altered_top_down_regulation",
                "target": "acc",
                "relation": "control-conflict and emotional-monitoring regions become over-engaged",
                "fnsd_change": "increased",
            },
            {
                "source": "limbic_motor_coupling",
                "target": "amygdala",
                "relation": "limbic salience systems drive symptom-related bodily threat tagging",
                "fnsd_change": "increased",
            },
            {
                "source": "limbic_motor_coupling",
                "target": "top_down_motor_inhibition",
                "relation": "emotional networks can translate distress into suppressed movement output",
                "fnsd_change": "increased",
            },
            {
                "source": "top_down_motor_inhibition",
                "target": "primary_motor_cortex",
                "relation": "top-down suppression manifests as motor cortex under-recruitment",
                "fnsd_change": "increased",
            },
            {
                "source": "top_down_motor_inhibition",
                "target": "supplementary_motor_area",
                "relation": "top-down suppression disrupts motor preparation and initiation",
                "fnsd_change": "increased",
            },
            {
                "source": "aberrant_sensorimotor_integration",
                "target": "functional_sensory_disturbance",
                "relation": "misintegrated sensory-motor signals produce functional sensory symptoms",
                "fnsd_change": "increased",
            },
            {
                "source": "aberrant_sensorimotor_integration",
                "target": "seizure_like_phenomena",
                "relation": "widespread network instability can contribute to seizure-like events",
                "fnsd_change": "increased",
            },
            {
                "source": "primary_motor_cortex",
                "target": "functional_weakness_paralysis",
                "relation": "motor cortex under-recruitment aligns with functional weakness or paralysis",
                "fnsd_change": "increased",
            },
            {
                "source": "primary_motor_cortex",
                "target": "abnormal_movement_inhibition",
                "relation": "motor execution disruption contributes to broader functional motor symptoms",
                "fnsd_change": "increased",
            },
            {
                "source": "supplementary_motor_area",
                "target": "functional_weakness_paralysis",
                "relation": "disrupted motor planning and initiation worsen weakness presentations",
                "fnsd_change": "increased",
            },
            {
                "source": "supplementary_motor_area",
                "target": "abnormal_movement_inhibition",
                "relation": "SMA dysregulation contributes to abnormal movement and initiation failure",
                "fnsd_change": "increased",
            },
            {
                "source": "vlpfc",
                "target": "abnormal_movement_inhibition",
                "relation": "excess inhibitory frontal control can reinforce nonvolitional movement suppression",
                "fnsd_change": "increased",
            },
            {
                "source": "acc",
                "target": "somatic_distress_arousal",
                "relation": "ACC burden can amplify distress monitoring and bodily conflict signals",
                "fnsd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "somatic_distress_arousal",
                "relation": "limbic hyperreactivity amplifies bodily alarm and symptom salience",
                "fnsd_change": "increased",
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
            tabular = getattr(siibra.features, "tabular", None)
            if tabular is not None:
                if kind == "receptor" and hasattr(tabular, "ReceptorDensityFingerprint"):
                    cands.append(getattr(tabular, "ReceptorDensityFingerprint"))
                elif kind == "gene" and hasattr(tabular, "GeneExpressions"):
                    cands.append(getattr(tabular, "GeneExpressions"))
        except Exception:
            pass

        try:
            molecular = getattr(siibra.features, "molecular", None)
            if molecular is not None:
                if kind == "receptor" and hasattr(molecular, "ReceptorDensityFingerprint"):
                    cands.append(getattr(molecular, "ReceptorDensityFingerprint"))
                elif kind == "gene" and hasattr(molecular, "GeneExpressions"):
                    cands.append(getattr(molecular, "GeneExpressions"))
        except Exception:
            pass

        try:
            connectivity = getattr(siibra.features, "connectivity", None)
            if kind == "connectivity" and connectivity is not None and hasattr(connectivity, "StreamlineCounts"):
                cands.append(getattr(connectivity, "StreamlineCounts"))
        except Exception:
            pass

        if kind == "receptor":
            cands.extend([
                "receptor density fingerprint",
                "ReceptorDensityFingerprint",
            ])
        elif kind == "gene":
            cands.extend([
                "gene expressions",
                "GeneExpressions",
            ])
        elif kind == "connectivity":
            cands.extend([
                "StreamlineCounts",
                "streamline counts",
            ])
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "primary motor cortex",
            "supplementary motor area",
            "dorsolateral prefrontal cortex",
            "ventrolateral prefrontal cortex",
            "anterior cingulate cortex",
            "cingulate cortex",
            "amygdala",
            "precentral gyrus",
            "middle frontal gyrus",
            "inferior frontal gyrus",
        } else 0
        gapmap_penalty = 1 if "gapmap" in name else 0
        return (left_bonus, right_penalty, generic_penalty, gapmap_penalty)

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
                return self.parcellation.get_region(spec)
            except Exception:
                pass
            try:
                return siibra.get_region(self.parcellation_spec, spec)
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
        for space_arg in (self.space, self.space_spec):
            try:
                props = region.spatial_props(space=space_arg)
            except Exception:
                continue
            if props is None:
                continue
            if isinstance(props, dict):
                return list(props.values())
            if isinstance(props, (list, tuple)):
                return list(props)
            if hasattr(props, "components"):
                return list(getattr(props, "components", []))
            return [props]
        return []

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz = None
        if centroid is not None:
            try:
                centroid_xyz = tuple(float(x) for x in centroid)
            except Exception:
                try:
                    centroid_xyz = tuple(float(x) for x in centroid.coordinate)
                except Exception:
                    centroid_xyz = None
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
                    return df.reset_index(drop=True)
            return df.reset_index(drop=True)
        return pd.DataFrame()

    @staticmethod
    def _cohort_of(feature: Any) -> str:
        return str(getattr(feature, "cohort", "")).upper().replace(" ", "")

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        wanted = str(self.connectivity_cohort).upper().replace(" ", "")
        compound = next((f for f in feats if self._cohort_of(f) == wanted), feats[0])

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            first = compound[0]
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
        fuzzy = [
            x for x in labels
            if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn
        ]
        if fuzzy:
            return fuzzy[0]

        region_key = next((k for k, v in self.region_objects.items() if v == region), "")
        for candidate in self.region_candidates.get(region_key, []):
            candidate_lower = candidate.lower()
            fuzzy = [x for x in labels if candidate_lower in self._name_of(x).lower()]
            if fuzzy:
                return fuzzy[0]
        return None

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
        """Return pairwise structural connectivity among resolved circuit nodes."""
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame(columns=["source", "target", "value"])

        rows: List[Dict[str, Any]] = []
        for src_key, src_region in self.region_objects.items():
            src_label = self._match_region_label(list(matrix.index), src_region)
            axis = "index"
            if src_label is None:
                src_label = self._match_region_label(list(matrix.columns), src_region)
                axis = "columns"
            if src_label is None:
                continue

            try:
                src_series = matrix.loc[src_label] if axis == "index" else matrix[src_label]
            except Exception:
                continue

            for dst_key, dst_region in self.region_objects.items():
                if src_key == dst_key:
                    continue
                dst_label = self._match_region_label(list(src_series.index), dst_region)
                if dst_label is None:
                    continue
                try:
                    value = float(src_series.loc[dst_label])
                except Exception:
                    continue
                rows.append(
                    {
                        "source": src_key,
                        "target": dst_key,
                        "value": value,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["source", "target", "value"])
        return (
            pd.DataFrame(rows)
            .sort_values(["source", "value"], ascending=[True, False])
            .reset_index(drop=True)
        )

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_FNSD_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}

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
                        "description": self.region_descriptions.get(key, "Atlas-backed region or proxy node."),
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
                    "description": self.region_descriptions.get(key, "Atlas-backed region or proxy node."),
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
        circuit_conn = self.circuit_connectivity()
        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": circuit_conn,
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """Probabilistically assign an MNI coordinate to Julich regions."""
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
                        space=self.assignment_space,
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

    def region_mask(self, node_key: str) -> Any:
        """Return a best-effort statistical regional map or mask for a resolved node."""
        region = self.region_objects.get(node_key)
        if region is None:
            return None

        for space_arg in (self.assignment_space, self.space, self.space_spec):
            try:
                return region.get_regional_map(space_arg, "statistical")
            except Exception:
                pass
            try:
                return region.get_regional_map(space=space_arg, maptype="statistical")
            except Exception:
                pass
            try:
                return region.get_regional_mask(space_arg)
            except Exception:
                pass
        return None

    def simulate(
        self,
        acute_stress_arousal: float = 0.65,
        chronic_stress_load: float = 0.60,
        comorbid_mood_anxiety: float = 0.55,
        genetic_diathesis: float = 0.35,
        neurotic_trait_liability: float = 0.50,
        neuroimmune_load: float = 0.30,
        treatment_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized FNSD simulation.

        Parameters are 0..1 normalized knobs. Higher symptom values indicate greater
        modeled burden.
        """
        inputs = pd.Series(
            {
                "acute_stress_arousal": self._clip01(acute_stress_arousal),
                "chronic_stress_load": self._clip01(chronic_stress_load),
                "comorbid_mood_anxiety": self._clip01(comorbid_mood_anxiety),
                "genetic_diathesis": self._clip01(genetic_diathesis),
                "neurotic_trait_liability": self._clip01(neurotic_trait_liability),
                "neuroimmune_load": self._clip01(neuroimmune_load),
                "treatment_support": self._clip01(treatment_support),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["stress_response_sensitization"] = self._clip01(
            0.35 * inputs["acute_stress_arousal"]
            + 0.23 * inputs["chronic_stress_load"]
            + 0.16 * inputs["neurotic_trait_liability"]
            + 0.12 * inputs["genetic_diathesis"]
            + 0.08 * inputs["neuroimmune_load"]
            - 0.20 * inputs["treatment_support"]
        )
        latents["monoaminergic_dysregulation"] = self._clip01(
            0.30 * inputs["comorbid_mood_anxiety"]
            + 0.20 * inputs["chronic_stress_load"]
            + 0.16 * inputs["genetic_diathesis"]
            + 0.14 * latents["stress_response_sensitization"]
            + 0.10 * inputs["neuroimmune_load"]
            - 0.12 * inputs["treatment_support"]
        )
        latents["gaba_glutamate_imbalance"] = self._clip01(
            0.24 * inputs["comorbid_mood_anxiety"]
            + 0.20 * inputs["acute_stress_arousal"]
            + 0.18 * latents["monoaminergic_dysregulation"]
            + 0.12 * inputs["neuroimmune_load"]
            + 0.10 * inputs["genetic_diathesis"]
            - 0.10 * inputs["treatment_support"]
        )
        latents["neuroimmune_stress_diathesis"] = self._clip01(
            0.40 * inputs["neuroimmune_load"]
            + 0.20 * inputs["chronic_stress_load"]
            + 0.16 * inputs["genetic_diathesis"]
            + 0.10 * inputs["neurotic_trait_liability"]
            - 0.10 * inputs["treatment_support"]
        )
        latents["altered_top_down_regulation"] = self._clip01(
            0.24 * latents["monoaminergic_dysregulation"]
            + 0.22 * latents["gaba_glutamate_imbalance"]
            + 0.18 * latents["stress_response_sensitization"]
            + 0.16 * inputs["comorbid_mood_anxiety"]
            + 0.12 * inputs["neurotic_trait_liability"]
            - 0.18 * inputs["treatment_support"]
        )
        latents["limbic_motor_coupling"] = self._clip01(
            0.30 * inputs["acute_stress_arousal"]
            + 0.20 * latents["stress_response_sensitization"]
            + 0.18 * inputs["comorbid_mood_anxiety"]
            + 0.16 * inputs["neurotic_trait_liability"]
            + 0.12 * latents["neuroimmune_stress_diathesis"]
            - 0.12 * inputs["treatment_support"]
        )
        latents["top_down_motor_inhibition"] = self._clip01(
            0.34 * latents["altered_top_down_regulation"]
            + 0.26 * latents["limbic_motor_coupling"]
            + 0.16 * latents["gaba_glutamate_imbalance"]
            + 0.10 * inputs["acute_stress_arousal"]
            - 0.14 * inputs["treatment_support"]
        )
        latents["aberrant_sensorimotor_integration"] = self._clip01(
            0.28 * latents["top_down_motor_inhibition"]
            + 0.24 * latents["gaba_glutamate_imbalance"]
            + 0.16 * latents["monoaminergic_dysregulation"]
            + 0.14 * latents["limbic_motor_coupling"]
            + 0.10 * latents["neuroimmune_stress_diathesis"]
            - 0.10 * inputs["treatment_support"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["amygdala"] = self._clip01(
            0.48 * latents["limbic_motor_coupling"]
            + 0.18 * latents["stress_response_sensitization"]
            + 0.12 * inputs["acute_stress_arousal"]
            - 0.14 * inputs["treatment_support"]
        )
        regional_state["acc"] = self._clip01(
            0.40 * latents["altered_top_down_regulation"]
            + 0.20 * latents["limbic_motor_coupling"]
            + 0.12 * inputs["comorbid_mood_anxiety"]
            - 0.14 * inputs["treatment_support"]
        )
        regional_state["dlpfc"] = self._clip01(
            0.42 * latents["altered_top_down_regulation"]
            + 0.18 * latents["monoaminergic_dysregulation"]
            + 0.10 * inputs["chronic_stress_load"]
            - 0.18 * inputs["treatment_support"]
        )
        regional_state["vlpfc"] = self._clip01(
            0.44 * latents["top_down_motor_inhibition"]
            + 0.18 * latents["altered_top_down_regulation"]
            + 0.10 * inputs["acute_stress_arousal"]
            - 0.16 * inputs["treatment_support"]
        )
        regional_state["supplementary_motor_area"] = self._clip01(
            0.46 * latents["top_down_motor_inhibition"]
            + 0.18 * latents["aberrant_sensorimotor_integration"]
            + 0.12 * latents["altered_top_down_regulation"]
            - 0.12 * inputs["treatment_support"]
        )
        regional_state["primary_motor_cortex"] = self._clip01(
            0.42 * latents["top_down_motor_inhibition"]
            + 0.22 * latents["aberrant_sensorimotor_integration"]
            + 0.12 * regional_state["supplementary_motor_area"]
            - 0.10 * inputs["treatment_support"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["functional_weakness_paralysis"] = self._clip01(
            0.40 * regional_state["primary_motor_cortex"]
            + 0.24 * regional_state["supplementary_motor_area"]
            + 0.16 * regional_state["vlpfc"]
            + 0.08 * regional_state["amygdala"]
        )
        symptoms["abnormal_movement_inhibition"] = self._clip01(
            0.32 * regional_state["supplementary_motor_area"]
            + 0.28 * regional_state["primary_motor_cortex"]
            + 0.16 * regional_state["acc"]
            + 0.12 * latents["aberrant_sensorimotor_integration"]
        )
        symptoms["functional_sensory_disturbance"] = self._clip01(
            0.42 * latents["aberrant_sensorimotor_integration"]
            + 0.18 * regional_state["acc"]
            + 0.14 * latents["monoaminergic_dysregulation"]
            + 0.10 * latents["stress_response_sensitization"]
        )
        symptoms["seizure_like_phenomena"] = self._clip01(
            0.28 * latents["aberrant_sensorimotor_integration"]
            + 0.22 * latents["gaba_glutamate_imbalance"]
            + 0.18 * latents["limbic_motor_coupling"]
            + 0.08 * inputs["acute_stress_arousal"]
        )
        symptoms["somatic_distress_arousal"] = self._clip01(
            0.34 * regional_state["amygdala"]
            + 0.24 * regional_state["acc"]
            + 0.20 * latents["stress_response_sensitization"]
            + 0.10 * latents["monoaminergic_dysregulation"]
        )

        phenotypes = pd.Series(dtype=float, name="phenotypes")
        phenotypes["motor_conversion_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        symptoms["functional_weakness_paralysis"],
                        symptoms["abnormal_movement_inhibition"],
                        regional_state["primary_motor_cortex"],
                        regional_state["supplementary_motor_area"],
                    ]
                ).mean()
            )
        )
        phenotypes["sensory_conversion_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        symptoms["functional_sensory_disturbance"],
                        latents["aberrant_sensorimotor_integration"],
                        symptoms["somatic_distress_arousal"],
                    ]
                ).mean()
            )
        )
        phenotypes["functional_seizure_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        symptoms["seizure_like_phenomena"],
                        latents["gaba_glutamate_imbalance"],
                        latents["limbic_motor_coupling"],
                    ]
                ).mean()
            )
        )
        phenotypes["frontolimbic_override_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        regional_state["amygdala"],
                        regional_state["acc"],
                        regional_state["dlpfc"],
                        regional_state["vlpfc"],
                        latents["top_down_motor_inhibition"],
                    ]
                ).mean()
            )
        )
        phenotypes["affective_somatic_profile"] = self._clip01(
            float(
                pd.Series(
                    [
                        latents["monoaminergic_dysregulation"],
                        latents["stress_response_sensitization"],
                        symptoms["somatic_distress_arousal"],
                    ]
                ).mean()
            )
        )

        return {
            "inputs": inputs,
            "latents": latents.round(4),
            "regional_state": regional_state.round(4),
            "symptoms": symptoms.round(4),
            "phenotypes": phenotypes.round(4),
        }


if __name__ == "__main__":  # pragma: no cover - example usage
    try:
        model = FunctionalNeurologicalSymptomDisorderModel()
        scaffold = model.build()

        print("\n=== Nodes ===")
        print(scaffold["nodes"][["key", "node_type", "label", "atlas_region", "feature_summary"]].to_string(index=False))

        print("\n=== Edges ===")
        print(scaffold["edges"][["source", "target", "relation", "fnsd_change"]].to_string(index=False))

        print("\n=== Resolved regions ===")
        for key, region in scaffold["regions"].items():
            print(f"- {key}: {region.name}")

        print("\n=== Example receptor table: primary_motor_cortex ===")
        motor_receptors = scaffold["receptors"].get("primary_motor_cortex", pd.DataFrame())
        if motor_receptors.empty:
            print("No receptor fingerprint available for primary_motor_cortex in this environment.")
        else:
            print(motor_receptors.head().to_string(index=False))

        print("\n=== Example gene table: acc ===")
        acc_genes = scaffold["genes"].get("acc", pd.DataFrame())
        if acc_genes.empty:
            print("No gene-expression table available for acc in this environment.")
        else:
            print(acc_genes.head().to_string(index=False))

        print("\n=== Example connectivity profile: supplementary_motor_area ===")
        sma_conn = scaffold["connectivity_profiles"].get("supplementary_motor_area", pd.DataFrame())
        if sma_conn.empty:
            print("No connectivity profile available for supplementary_motor_area in this environment.")
        else:
            print(sma_conn.head(10).to_string(index=False))

        print("\n=== Example circuit connectivity ===")
        if scaffold["circuit_connectivity"].empty:
            print("No pairwise circuit connectivity values available in this environment.")
        else:
            print(scaffold["circuit_connectivity"].head(20).to_string(index=False))

        sim = model.simulate(
            acute_stress_arousal=0.75,
            chronic_stress_load=0.65,
            comorbid_mood_anxiety=0.60,
            genetic_diathesis=0.35,
            neurotic_trait_liability=0.55,
            neuroimmune_load=0.30,
            treatment_support=0.25,
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

        # Example optional coordinate assignment:
        # assignments = model.assign_mni_point((-38, -18, 56))
        # print(assignments.head())

    except ImportError as exc:
        print("This script requires siibra-python to run the atlas-backed portions of the scaffold.")
        print(exc)
