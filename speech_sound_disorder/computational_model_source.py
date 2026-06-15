from __future__ import annotations

"""
Speech Sound Disorder siibra scaffold.

This research scaffold translates a chapter-level summary of Speech Sound
Disorder (SSD) into an atlas-grounded, transparent mechanistic model using
siibra where available.

It is intended for exploratory modeling, reproducible atlas lookup, and region
retuning. It is not a diagnostic or treatment tool.
"""

import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore

    SIIBRA_AVAILABLE = True
except Exception:  # pragma: no cover - depends on runtime
    siibra = None  # type: ignore
    SIIBRA_AVAILABLE = False


DEFAULT_GENE_PANEL = [
    "FOXP2",
    "FOXP1",
    "CNTNAP2",
    "ROBO2",
    "ROBO1",
    "DCDC2",
    "KIAA0319",
    "ATP2C2",
    "CMIP",
    "BDNF",
]


class SpeechSoundDisorderModel:
    """
    Atlas-grounded scaffold for Speech Sound Disorder.

    Chapter logic emphasized here:
    - multifactorial familial and polygenic liability for speech/language
      impairment,
    - rare but mechanistically informative FOXP2-linked verbal dyspraxia style
      disruption of speech-network development,
    - ROBO2 / CNTNAP2-style contributions to language wiring and phonological
      endophenotypes,
    - congenital cortical maldevelopment, callosal anomalies, cerebellar
      disruption, and perinatal white-matter injury as structural routes to
      speech and language impairment,
    - temporal auditory-language dysfunction and verbal auditory agnosia /
      "word deafness" as evidence for temporal-lobe contribution,
    - developmental plasticity that can partially reorganize language after
      early lesions, but often incompletely.

    Because the chapter mixes developmental SSD with evidence from congenital
    malformations and acquired pediatric lesions, several atlas anchors are
    conservative proxies rather than claims of a single precise parcel.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.disorder_name = "Speech Sound Disorder"
        self.domain_key = "ssd"
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec

        self.atlas = None
        self.parcellation = None
        self.space = None

        if SIIBRA_AVAILABLE:
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
            except Exception as exc:  # pragma: no cover - runtime dependent
                warnings.warn(
                    "siibra atlas initialization failed; atlas-backed features "
                    f"will be limited: {exc}"
                )
        else:
            warnings.warn(
                "siibra is not installed in this environment; atlas-backed region "
                "resolution, feature queries, and coordinate assignments will return "
                "empty results. The mechanistic simulator remains usable."
            )

        self.region_candidates: Dict[str, List[str]] = {
            "temporal_auditory_language_cortex": [
                "Area TE 3 (STG) left",
                "Area TE 2.1 (STG) left",
                "Area TE 1.2 (HESCHL) left",
                "Area TE 1.0 (HESCHL) left",
                "superior temporal gyrus",
                "temporal lobe",
                "wernicke",
            ],
            "frontal_speech_motor_proxy": [
                "Area 44 left",
                "Area 45 left",
                "Area 6 left",
                "inferior frontal gyrus",
                "premotor cortex",
                "broca",
            ],
            "corpus_callosum_proxy": [
                "corpus callosum",
                "callosal white matter",
                "white matter",
            ],
            "cerebellum_proxy": [
                "cerebellum left",
                "cerebellum",
                "lobule VI left",
                "crus I left",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_speech_language_liability": (
                "Familial and polygenic liability affecting speech, language, and learning development."
            ),
            "congenital_cortical_maldevelopment_load": (
                "Burden of congenital forebrain or cortical developmental anomalies that disrupt language architecture."
            ),
            "corpus_callosum_anomaly_load": (
                "Burden of callosal abnormality impairing interhemispheric communication relevant to language."
            ),
            "cerebellar_developmental_anomaly_load": (
                "Burden of cerebellar developmental disruption affecting timing, coordination, and learning."
            ),
            "acquired_childhood_brain_lesion_load": (
                "Burden of pediatric brain injury from stroke, trauma, infection, or other focal lesions."
            ),
            "perinatal_white_matter_injury_load": (
                "Burden of perinatal white-matter injury such as periventricular leukomalacia."
            ),
            "shared_neurodevelopmental_comorbidity_load": (
                "Shared developmental burden associated with executive, attentional, or broader language vulnerabilities."
            ),
            "developmental_intervention_support": (
                "Protective effect of early intervention, therapy, enriched learning context, and compensatory support."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "developmental_plasticity_compensation": (
                "Partially adaptive reorganization of language function after early injury or developmental disruption."
            ),
            "transcriptional_speech_network_disruption": (
                "Speech-language network vulnerability linked to developmental gene regulation, exemplified by FOXP2 pathways."
            ),
            "axon_guidance_language_wiring_disruption": (
                "Disrupted long-range wiring and phonological-network formation linked to axon-guidance and language genes."
            ),
            "white_matter_disconnection": (
                "Disconnection across developing speech-language pathways due to injury or maldevelopment."
            ),
            "frontostriatal_executive_modulation_weakness": (
                "Reduced executive modulation of speech and language processing in shared neurodevelopmental burden states."
            ),
            "auditory_phonological_encoding_weakness": (
                "Weak auditory discrimination and phonological encoding of speech sounds."
            ),
            "speech_motor_planning_instability": (
                "Imprecise planning and sequencing of speech movements, including dyspraxia-like burden."
            ),
            "interhemispheric_language_integration_failure": (
                "Poor integration across hemispheres for language and speech-sound processing."
            ),
            "cerebellar_timing_sensorimotor_learning_deficit": (
                "Impaired timing and sensorimotor calibration of speech-related learning."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "phonological_encoding_errors": (
                "Difficulty representing or selecting speech sounds accurately."
            ),
            "articulation_inaccuracy": (
                "Imprecise production of speech sounds."
            ),
            "inconsistent_speech_output": (
                "Variable production across repetitions or contexts."
            ),
            "reduced_speech_intelligibility": (
                "Speech that is difficult for listeners to understand."
            ),
            "receptive_auditory_language_overlap": (
                "Overlap with auditory-language comprehension problems when temporal auditory systems are affected."
            ),
            "broader_communication_impairment": (
                "Overall impact on communication effectiveness and message transmission."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_speech_language_liability",
                "target": "transcriptional_speech_network_disruption",
                "relation": "raises developmental vulnerability of speech-language transcriptional programs",
                "ssd_change": "increased",
            },
            {
                "source": "genetic_speech_language_liability",
                "target": "axon_guidance_language_wiring_disruption",
                "relation": "raises vulnerability of language-network wiring and phonological endophenotypes",
                "ssd_change": "increased",
            },
            {
                "source": "congenital_cortical_maldevelopment_load",
                "target": "white_matter_disconnection",
                "relation": "disrupts the developmental architecture needed for coordinated language networks",
                "ssd_change": "increased",
            },
            {
                "source": "corpus_callosum_anomaly_load",
                "target": "interhemispheric_language_integration_failure",
                "relation": "impairs interhemispheric communication relevant to language and complex cognition",
                "ssd_change": "increased",
            },
            {
                "source": "cerebellar_developmental_anomaly_load",
                "target": "cerebellar_timing_sensorimotor_learning_deficit",
                "relation": "impairs timing, coordination, and learning support for speech production",
                "ssd_change": "increased",
            },
            {
                "source": "acquired_childhood_brain_lesion_load",
                "target": "white_matter_disconnection",
                "relation": "can disconnect speech-language pathways after pediatric injury",
                "ssd_change": "increased",
            },
            {
                "source": "acquired_childhood_brain_lesion_load",
                "target": "auditory_phonological_encoding_weakness",
                "relation": "can damage temporal auditory-language systems and weaken speech-sound encoding",
                "ssd_change": "increased",
            },
            {
                "source": "perinatal_white_matter_injury_load",
                "target": "white_matter_disconnection",
                "relation": "damages developing white-matter pathways needed for coordinated language function",
                "ssd_change": "increased",
            },
            {
                "source": "shared_neurodevelopmental_comorbidity_load",
                "target": "frontostriatal_executive_modulation_weakness",
                "relation": "adds executive and attentional inefficiency that can worsen speech-sound control",
                "ssd_change": "increased",
            },
            {
                "source": "developmental_intervention_support",
                "target": "developmental_plasticity_compensation",
                "relation": "supports compensatory learning and partial functional reorganization",
                "ssd_change": "increased",
            },
            {
                "source": "transcriptional_speech_network_disruption",
                "target": "speech_motor_planning_instability",
                "relation": "can produce dyspraxia-like planning and sequencing burden",
                "ssd_change": "increased",
            },
            {
                "source": "axon_guidance_language_wiring_disruption",
                "target": "auditory_phonological_encoding_weakness",
                "relation": "weakens the developmental wiring needed for robust phonological representations",
                "ssd_change": "increased",
            },
            {
                "source": "axon_guidance_language_wiring_disruption",
                "target": "interhemispheric_language_integration_failure",
                "relation": "burdens long-range integration across language-relevant networks",
                "ssd_change": "increased",
            },
            {
                "source": "white_matter_disconnection",
                "target": "corpus_callosum_proxy",
                "relation": "is represented through disrupted callosal and white-matter integration pathways",
                "ssd_change": "increased",
            },
            {
                "source": "auditory_phonological_encoding_weakness",
                "target": "temporal_auditory_language_cortex",
                "relation": "is represented through temporal auditory-language dysfunction",
                "ssd_change": "increased",
            },
            {
                "source": "speech_motor_planning_instability",
                "target": "frontal_speech_motor_proxy",
                "relation": "is represented through frontal speech-motor planning circuitry",
                "ssd_change": "increased",
            },
            {
                "source": "cerebellar_timing_sensorimotor_learning_deficit",
                "target": "cerebellum_proxy",
                "relation": "is represented through cerebellar support for timing and learning",
                "ssd_change": "increased",
            },
            {
                "source": "temporal_auditory_language_cortex",
                "target": "phonological_encoding_errors",
                "relation": "temporal auditory-language dysfunction contributes to speech-sound encoding errors",
                "ssd_change": "increased",
            },
            {
                "source": "frontal_speech_motor_proxy",
                "target": "articulation_inaccuracy",
                "relation": "frontal speech-motor burden contributes to inaccurate speech production",
                "ssd_change": "increased",
            },
            {
                "source": "cerebellum_proxy",
                "target": "inconsistent_speech_output",
                "relation": "cerebellar timing and learning burden can increase inconsistency",
                "ssd_change": "increased",
            },
            {
                "source": "interhemispheric_language_integration_failure",
                "target": "reduced_speech_intelligibility",
                "relation": "poor integration across language systems degrades stable intelligible output",
                "ssd_change": "increased",
            },
            {
                "source": "auditory_phonological_encoding_weakness",
                "target": "receptive_auditory_language_overlap",
                "relation": "can extend to auditory-language comprehension overlap when temporal systems are affected",
                "ssd_change": "increased",
            },
            {
                "source": "phonological_encoding_errors",
                "target": "reduced_speech_intelligibility",
                "relation": "reduces listener comprehension of speech output",
                "ssd_change": "increased",
            },
            {
                "source": "articulation_inaccuracy",
                "target": "reduced_speech_intelligibility",
                "relation": "reduces clarity of spoken output",
                "ssd_change": "increased",
            },
            {
                "source": "inconsistent_speech_output",
                "target": "reduced_speech_intelligibility",
                "relation": "increases variability and listener difficulty",
                "ssd_change": "increased",
            },
            {
                "source": "reduced_speech_intelligibility",
                "target": "broader_communication_impairment",
                "relation": "interferes with effective communication of messages",
                "ssd_change": "increased",
            },
            {
                "source": "receptive_auditory_language_overlap",
                "target": "broader_communication_impairment",
                "relation": "adds comprehension-level burden to overall communication difficulty",
                "ssd_change": "increased",
            },
            {
                "source": "developmental_plasticity_compensation",
                "target": "broader_communication_impairment",
                "relation": "partially offsets developmental speech-language burden",
                "ssd_change": "decreased",
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

    def _quiet_context(self):
        return siibra.QUIET if SIIBRA_AVAILABLE else nullcontext()

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    def _modality_candidates(self, kind: str) -> List[Any]:
        cands: List[Any] = []
        if SIIBRA_AVAILABLE:
            try:
                if kind == "receptor":
                    cands.append(siibra.features.molecular.ReceptorDensityFingerprint)
                elif kind == "gene":
                    cands.append(siibra.features.molecular.GeneExpressions)
                elif kind == "connectivity":
                    cands.append(siibra.features.connectivity.StreamlineCounts)
            except Exception:
                pass

            # Compatibility fallback if the installation exposes these under a
            # different submodule.
            try:
                if kind == "receptor" and hasattr(siibra.features, "tabular"):
                    cands.append(siibra.features.tabular.ReceptorDensityFingerprint)
                elif kind == "gene" and hasattr(siibra.features, "tabular"):
                    cands.append(siibra.features.tabular.GeneExpressions)
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
        if not SIIBRA_AVAILABLE or concept is None:
            return []
        for modality in modalities:
            try:
                with self._quiet_context():
                    feats = siibra.features.get(concept, modality, **kwargs)
                if feats:
                    return list(feats)
            except Exception:
                continue
        return []

    def _julich_matches(self, query: str) -> List[Any]:
        if self.atlas is None:
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "superior temporal gyrus",
            "inferior frontal gyrus",
            "cerebellum",
            "corpus callosum",
        } else 0
        return (left_bonus, right_penalty, generic_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if self.atlas is None or self.parcellation is None:
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
        try:
            df = feats[0].data.copy().reset_index()
            if "index" in df.columns and "receptor" not in df.columns:
                df = df.rename(columns={"index": "receptor"})
            return df.reset_index(drop=True)
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
                .reset_index(drop=True)
            )
        return df.reset_index(drop=True)

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix
        if self.parcellation is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort),
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
        exact = [x for x in labels if self._name_of(x) == getattr(region, "name", None)]
        if exact:
            return exact[0]
        region_name = getattr(region, "name", "").lower()
        fuzzy = [
            x
            for x in labels
            if region_name and (region_name in self._name_of(x).lower() or self._name_of(x).lower() in region_name)
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
            df = df[df["connected_region"] != getattr(region, "name", None)].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _pairwise_connectivity_value(self, matrix: pd.DataFrame, region_a: Any, region_b: Any) -> Optional[float]:
        row_a = self._match_region_label(list(matrix.index), region_a)
        row_b = self._match_region_label(list(matrix.index), region_b)
        col_a = self._match_region_label(list(matrix.columns), region_a)
        col_b = self._match_region_label(list(matrix.columns), region_b)

        values: List[float] = []
        for row_label, col_label in ((row_a, col_b), (row_b, col_a)):
            if row_label is None or col_label is None:
                continue
            try:
                value = matrix.loc[row_label, col_label]
                if pd.notna(value):
                    values.append(float(value))
            except Exception:
                continue
        if not values:
            return None
        return round(sum(values) / len(values), 6)

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or len(self.region_objects) < 2:
            return pd.DataFrame()
        rows: List[Dict[str, Any]] = []
        keys = list(self.region_objects.keys())
        for i, src_key in enumerate(keys):
            for tgt_key in keys[i + 1 :]:
                src_region = self.region_objects[src_key]
                tgt_region = self.region_objects[tgt_key]
                value = self._pairwise_connectivity_value(matrix, src_region, tgt_region)
                if value is None:
                    continue
                rows.append(
                    {
                        "source_key": src_key,
                        "source_region": getattr(src_region, "name", src_key),
                        "target_key": tgt_key,
                        "target_region": getattr(tgt_region, "name", tgt_key),
                        "value": value,
                    }
                )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
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
                warnings.warn(f"Could not resolve a Julich region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region",
                        "description": "Atlas-backed node or conservative proxy unresolved in this environment.",
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
                    "label": getattr(region, "name", key),
                    "node_type": "region",
                    "description": "Atlas-backed circuit node or proxy anchor.",
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
        circuit_df = self.circuit_connectivity()

        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": circuit_df,
        }

    def simulate(
        self,
        genetic_speech_language_liability: float = 0.45,
        congenital_cortical_maldevelopment_load: float = 0.20,
        corpus_callosum_anomaly_load: float = 0.15,
        cerebellar_developmental_anomaly_load: float = 0.15,
        acquired_childhood_brain_lesion_load: float = 0.10,
        perinatal_white_matter_injury_load: float = 0.15,
        shared_neurodevelopmental_comorbidity_load: float = 0.20,
        developmental_intervention_support: float = 0.35,
    ) -> Dict[str, pd.Series]:
        """
        Simulate normalized SSD burden from chapter-derived mechanisms.

        The simulator is intentionally transparent and acyclic:
        inputs -> latent biology -> regional state -> symptoms -> phenotypes.
        """
        inputs = pd.Series(
            {
                "genetic_speech_language_liability": self._clip01(genetic_speech_language_liability),
                "congenital_cortical_maldevelopment_load": self._clip01(congenital_cortical_maldevelopment_load),
                "corpus_callosum_anomaly_load": self._clip01(corpus_callosum_anomaly_load),
                "cerebellar_developmental_anomaly_load": self._clip01(cerebellar_developmental_anomaly_load),
                "acquired_childhood_brain_lesion_load": self._clip01(acquired_childhood_brain_lesion_load),
                "perinatal_white_matter_injury_load": self._clip01(perinatal_white_matter_injury_load),
                "shared_neurodevelopmental_comorbidity_load": self._clip01(shared_neurodevelopmental_comorbidity_load),
                "developmental_intervention_support": self._clip01(developmental_intervention_support),
            },
            dtype=float,
        )

        developmental_plasticity_compensation = self._clip01(
            0.42 * inputs["developmental_intervention_support"]
            + 0.12 * inputs["acquired_childhood_brain_lesion_load"]
            + 0.12 * inputs["congenital_cortical_maldevelopment_load"]
            + 0.10 * inputs["perinatal_white_matter_injury_load"]
            - 0.10 * inputs["corpus_callosum_anomaly_load"]
        )

        transcriptional_speech_network_disruption = self._clip01(
            0.42 * inputs["genetic_speech_language_liability"]
            + 0.18 * inputs["congenital_cortical_maldevelopment_load"]
            + 0.10 * inputs["shared_neurodevelopmental_comorbidity_load"]
            - 0.10 * developmental_plasticity_compensation
        )

        axon_guidance_language_wiring_disruption = self._clip01(
            0.36 * inputs["genetic_speech_language_liability"]
            + 0.22 * inputs["corpus_callosum_anomaly_load"]
            + 0.14 * inputs["perinatal_white_matter_injury_load"]
            - 0.10 * developmental_plasticity_compensation
        )

        white_matter_disconnection = self._clip01(
            0.34 * inputs["perinatal_white_matter_injury_load"]
            + 0.22 * inputs["acquired_childhood_brain_lesion_load"]
            + 0.20 * inputs["corpus_callosum_anomaly_load"]
            + 0.18 * inputs["congenital_cortical_maldevelopment_load"]
            - 0.12 * developmental_plasticity_compensation
        )

        frontostriatal_executive_modulation_weakness = self._clip01(
            0.34 * inputs["shared_neurodevelopmental_comorbidity_load"]
            + 0.16 * inputs["genetic_speech_language_liability"]
            - 0.12 * developmental_plasticity_compensation
        )

        cerebellar_timing_sensorimotor_learning_deficit = self._clip01(
            0.52 * inputs["cerebellar_developmental_anomaly_load"]
            + 0.16 * inputs["congenital_cortical_maldevelopment_load"]
            + 0.10 * inputs["acquired_childhood_brain_lesion_load"]
            - 0.08 * developmental_plasticity_compensation
        )

        auditory_phonological_encoding_weakness = self._clip01(
            0.34 * axon_guidance_language_wiring_disruption
            + 0.24 * white_matter_disconnection
            + 0.16 * inputs["acquired_childhood_brain_lesion_load"]
            + 0.10 * inputs["congenital_cortical_maldevelopment_load"]
            + 0.08 * inputs["shared_neurodevelopmental_comorbidity_load"]
            - 0.10 * developmental_plasticity_compensation
        )

        interhemispheric_language_integration_failure = self._clip01(
            0.44 * axon_guidance_language_wiring_disruption
            + 0.32 * white_matter_disconnection
            + 0.18 * inputs["corpus_callosum_anomaly_load"]
            - 0.10 * developmental_plasticity_compensation
        )

        speech_motor_planning_instability = self._clip01(
            0.34 * transcriptional_speech_network_disruption
            + 0.24 * cerebellar_timing_sensorimotor_learning_deficit
            + 0.14 * frontostriatal_executive_modulation_weakness
            + 0.14 * inputs["congenital_cortical_maldevelopment_load"]
            + 0.08 * inputs["acquired_childhood_brain_lesion_load"]
            - 0.10 * developmental_plasticity_compensation
        )

        latents = pd.Series(
            {
                "developmental_plasticity_compensation": developmental_plasticity_compensation,
                "transcriptional_speech_network_disruption": transcriptional_speech_network_disruption,
                "axon_guidance_language_wiring_disruption": axon_guidance_language_wiring_disruption,
                "white_matter_disconnection": white_matter_disconnection,
                "frontostriatal_executive_modulation_weakness": frontostriatal_executive_modulation_weakness,
                "auditory_phonological_encoding_weakness": auditory_phonological_encoding_weakness,
                "speech_motor_planning_instability": speech_motor_planning_instability,
                "interhemispheric_language_integration_failure": interhemispheric_language_integration_failure,
                "cerebellar_timing_sensorimotor_learning_deficit": cerebellar_timing_sensorimotor_learning_deficit,
            },
            dtype=float,
        )

        temporal_auditory_language_cortex = self._clip01(
            0.36 * auditory_phonological_encoding_weakness
            + 0.20 * white_matter_disconnection
            + 0.16 * inputs["acquired_childhood_brain_lesion_load"]
            + 0.10 * inputs["congenital_cortical_maldevelopment_load"]
            + 0.08 * interhemispheric_language_integration_failure
            - 0.12 * developmental_plasticity_compensation
        )

        frontal_speech_motor_proxy = self._clip01(
            0.34 * speech_motor_planning_instability
            + 0.18 * frontostriatal_executive_modulation_weakness
            + 0.14 * inputs["congenital_cortical_maldevelopment_load"]
            + 0.12 * inputs["acquired_childhood_brain_lesion_load"]
            + 0.08 * white_matter_disconnection
            - 0.10 * developmental_plasticity_compensation
        )

        corpus_callosum_proxy = self._clip01(
            0.42 * interhemispheric_language_integration_failure
            + 0.28 * white_matter_disconnection
            + 0.18 * inputs["corpus_callosum_anomaly_load"]
            + 0.08 * inputs["perinatal_white_matter_injury_load"]
        )

        cerebellum_proxy = self._clip01(
            0.48 * cerebellar_timing_sensorimotor_learning_deficit
            + 0.20 * speech_motor_planning_instability
            + 0.14 * inputs["cerebellar_developmental_anomaly_load"]
            + 0.08 * inputs["acquired_childhood_brain_lesion_load"]
            - 0.08 * developmental_plasticity_compensation
        )

        regional_state = pd.Series(
            {
                "temporal_auditory_language_cortex": temporal_auditory_language_cortex,
                "frontal_speech_motor_proxy": frontal_speech_motor_proxy,
                "corpus_callosum_proxy": corpus_callosum_proxy,
                "cerebellum_proxy": cerebellum_proxy,
            },
            dtype=float,
        )

        phonological_encoding_errors = self._clip01(
            0.42 * auditory_phonological_encoding_weakness
            + 0.16 * temporal_auditory_language_cortex
            + 0.15 * interhemispheric_language_integration_failure
            + 0.08 * frontostriatal_executive_modulation_weakness
            - 0.12 * developmental_plasticity_compensation
        )

        articulation_inaccuracy = self._clip01(
            0.35 * speech_motor_planning_instability
            + 0.18 * frontal_speech_motor_proxy
            + 0.18 * cerebellar_timing_sensorimotor_learning_deficit
            + 0.10 * cerebellum_proxy
            + 0.08 * white_matter_disconnection
            - 0.10 * developmental_plasticity_compensation
        )

        inconsistent_speech_output = self._clip01(
            0.34 * speech_motor_planning_instability
            + 0.22 * cerebellar_timing_sensorimotor_learning_deficit
            + 0.12 * white_matter_disconnection
            + 0.10 * inputs["acquired_childhood_brain_lesion_load"]
            - 0.10 * developmental_plasticity_compensation
        )

        reduced_speech_intelligibility = self._clip01(
            0.28 * phonological_encoding_errors
            + 0.32 * articulation_inaccuracy
            + 0.20 * inconsistent_speech_output
            + 0.10 * interhemispheric_language_integration_failure
            - 0.08 * developmental_plasticity_compensation
        )

        receptive_auditory_language_overlap = self._clip01(
            0.34 * auditory_phonological_encoding_weakness
            + 0.20 * temporal_auditory_language_cortex
            + 0.18 * inputs["acquired_childhood_brain_lesion_load"]
            + 0.12 * white_matter_disconnection
            - 0.10 * developmental_plasticity_compensation
        )

        broader_communication_impairment = self._clip01(
            0.34 * reduced_speech_intelligibility
            + 0.24 * receptive_auditory_language_overlap
            + 0.12 * frontostriatal_executive_modulation_weakness
            + 0.08 * interhemispheric_language_integration_failure
            - 0.10 * developmental_plasticity_compensation
        )

        symptoms = pd.Series(
            {
                "phonological_encoding_errors": phonological_encoding_errors,
                "articulation_inaccuracy": articulation_inaccuracy,
                "inconsistent_speech_output": inconsistent_speech_output,
                "reduced_speech_intelligibility": reduced_speech_intelligibility,
                "receptive_auditory_language_overlap": receptive_auditory_language_overlap,
                "broader_communication_impairment": broader_communication_impairment,
            },
            dtype=float,
        )

        developmental_phonological_profile = self._clip01(
            0.34 * phonological_encoding_errors
            + 0.28 * articulation_inaccuracy
            + 0.28 * reduced_speech_intelligibility
            - 0.10 * developmental_plasticity_compensation
        )

        verbal_dyspraxia_like_profile = self._clip01(
            0.38 * speech_motor_planning_instability
            + 0.28 * inconsistent_speech_output
            + 0.24 * articulation_inaccuracy
            - 0.08 * developmental_plasticity_compensation
        )

        disconnection_language_profile = self._clip01(
            0.30 * interhemispheric_language_integration_failure
            + 0.26 * white_matter_disconnection
            + 0.20 * reduced_speech_intelligibility
            + 0.14 * receptive_auditory_language_overlap
        )

        acquired_temporal_language_profile = self._clip01(
            0.26 * inputs["acquired_childhood_brain_lesion_load"]
            + 0.24 * temporal_auditory_language_cortex
            + 0.22 * receptive_auditory_language_overlap
            + 0.18 * broader_communication_impairment
            - 0.08 * developmental_plasticity_compensation
        )

        phenotypes = pd.Series(
            {
                "developmental_phonological_profile": developmental_phonological_profile,
                "verbal_dyspraxia_like_profile": verbal_dyspraxia_like_profile,
                "disconnection_language_profile": disconnection_language_profile,
                "acquired_temporal_language_profile": acquired_temporal_language_profile,
            },
            dtype=float,
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if not SIIBRA_AVAILABLE:
            return pd.DataFrame()
        if self._pmap is None:
            try:
                with self._quiet_context():
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception:
                try:
                    with self._quiet_context():
                        self._pmap = self.atlas.get_map(
                            parcellation=self.parcellation,
                            space=self.assignment_space,
                            maptype="statistical",
                        )
                except Exception:
                    return pd.DataFrame()

        try:
            point = siibra.Point(tuple(float(x) for x in xyz), space=self.assignment_space)
            with self._quiet_context():
                assignments = self._pmap.assign(point)
        except Exception:
            return pd.DataFrame()

        if not isinstance(assignments, pd.DataFrame):
            try:
                assignments = pd.DataFrame(assignments)
            except Exception:
                return pd.DataFrame()

        for candidate in (
            "map value",
            "correlation",
            "intersection over union",
            "intersection_over_union",
            "probability",
        ):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        region = self.region_objects.get(node_key)
        if region is None or not SIIBRA_AVAILABLE:
            return None
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(space=self.assignment_space)
            except Exception:
                return None


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 12)

    model = SpeechSoundDisorderModel()
    bundle = model.build()

    print("\n=== Nodes (first 24) ===")
    print(bundle["nodes"].head(24).to_string(index=False))

    print("\n=== Edges ===")
    print(bundle["edges"].to_string(index=False))

    print("\n=== Resolved regions ===")
    if bundle["regions"]:
        for key, region in bundle["regions"].items():
            print(f"- {key}: {getattr(region, 'name', key)}")
    else:
        print("No atlas-backed regions resolved in this runtime.")

    print("\n=== Example multimodal tables ===")
    for region_key in list(bundle["regions"].keys())[:2]:
        print(f"\nRegion: {region_key}")
        print("Receptors:")
        print(bundle["receptors"].get(region_key, pd.DataFrame()).head().to_string(index=False))
        print("Genes:")
        print(bundle["genes"].get(region_key, pd.DataFrame()).head().to_string(index=False))
        print("Connectivity profile:")
        print(bundle["connectivity_profiles"].get(region_key, pd.DataFrame()).head().to_string(index=False))

    print("\n=== Circuit connectivity ===")
    print(bundle["circuit_connectivity"].head(10).to_string(index=False))

    sim = model.simulate(
        genetic_speech_language_liability=0.55,
        congenital_cortical_maldevelopment_load=0.18,
        corpus_callosum_anomaly_load=0.22,
        cerebellar_developmental_anomaly_load=0.20,
        acquired_childhood_brain_lesion_load=0.10,
        perinatal_white_matter_injury_load=0.18,
        shared_neurodevelopmental_comorbidity_load=0.28,
        developmental_intervention_support=0.40,
    )

    print("\n=== Simulation ===")
    for name, series in sim.items():
        print(f"\n{name.upper()}")
        print(series.to_string())

    if SIIBRA_AVAILABLE:
        print("\n=== Example region suggestions for 'speech' ===")
        print(model.suggest_regions("speech", limit=10).to_string(index=False))

        # Example coordinate assignment:
        # print(model.assign_mni_point((-56, -32, 8)).head().to_string(index=False))
