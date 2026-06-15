
from __future__ import annotations

"""
Language Disorder siibra scaffold.

This script converts a biologically focused chapter on Language Disorder into an
atlas-grounded siibra research scaffold. It is designed for transparent
mechanistic exploration, not diagnosis, prognosis, or treatment selection.

Modeling emphasis from the chapter:
- intrinsic central nervous system dysfunction affecting language acquisition,
- dopaminergic and fronto-striatal contributions to attention, sequencing, and speech fluency,
- synaptic-plasticity constraints on phonology, syntax, and semantics,
- procedural-memory impairment for grammar and sequence learning,
- declarative-memory support for lexical compensation,
- FOXP2-related speech motor planning burden,
- left temporo-parietal phonological vulnerability,
- fronto-temporal / fronto-parietal connectivity atypicality,
- pragmatic language burden when social-cognitive networks are affected.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


DEFAULT_GENE_PANEL = [
    "FOXP2",
    "CYFIP1",
    "NIPA1",
    "NIPA2",
    "TUBGCP5",
    "BDNF",
    "DRD2",
    "SLC6A3",
    "COMT",
]


class LanguageDisorderModel:
    """
    Atlas-grounded scaffold for Language Disorder.

    The scaffold translates a chapter-level biological narrative into:
    - input nodes (genetic, developmental, attentional, and environmental loads),
    - latent biology nodes (plasticity, procedural learning, connectivity, and
      compensation mechanisms),
    - atlas-backed regions or clearly labeled proxies,
    - symptom nodes and phenotype summaries.

    Important:
    This is a research scaffold for inspection and hypothesis generation. It is
    not a validated disease model and must not be used as a clinical tool.
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

        # Conservative atlas anchors and proxies derived from the chapter.
        self.region_candidates: Dict[str, List[str]] = {
            "inferior_frontal_gyrus": [
                "Area 44 left",
                "Area 45 left",
                "ifg 44 left",
                "ifg 45 left",
                "ifg 44",
                "broca",
                "inferior frontal gyrus",
            ],
            "temporoparietal_phonology_proxy": [
                "Area PFm left",
                "Area PF left",
                "Area PGa left",
                "Area PGp left",
                "supramarginal left",
                "angular left",
                "inferior parietal left",
                "temporo-parietal",
            ],
            "pfc_control_proxy": [
                "Area 9/46d left",
                "Area 46 left",
                "Area 9 left",
                "dorsolateral prefrontal",
                "middle frontal left",
                "prefrontal cortex",
            ],
            "basal_ganglia_proxy": [
                "caudate left",
                "putamen left",
                "striatum left",
                "basal ganglia",
                "striatum",
            ],
            "hippocampus": [
                "CA1 left",
                "Subiculum left",
                "DG left",
                "hippocampus left",
                "hippocampus",
            ],
            "right_inferior_frontal_compensation_proxy": [
                "Area 44 right",
                "Area 45 right",
                "ifg 44 right",
                "ifg 45 right",
                "inferior frontal gyrus right",
                "right inferior frontal",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "inferior_frontal_gyrus": "Broca-like inferior frontal region explicitly tied to language production and compensatory activation.",
            "temporoparietal_phonology_proxy": "Proxy for left temporo-parietal / posterior phonological circuitry implicated in dyslexia-like and phonological deficits.",
            "pfc_control_proxy": "Proxy for executive-control and hypofrontal prefrontal circuitry affecting attention, working memory, and language access.",
            "basal_ganglia_proxy": "Proxy for fronto-striatal sequencing and motor-programming circuits relevant to speech and grammar learning.",
            "hippocampus": "Medial temporal declarative-memory region supporting lexical learning and compensation.",
            "right_inferior_frontal_compensation_proxy": "Proxy for right-hemisphere / inferior frontal compensatory recruitment mentioned in the chapter.",
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_language_liability": "Polygenic liability for developmental language problems and overlapping language-learning disorders.",
            "foxp2_pathway_disruption": "Rare-gene burden affecting speech motor planning, sequencing, and corticostriatal development.",
            "chromatin_transcription_burden": "Broad transcriptional / chromatin-regulation burden as in syndromic microdeletion or microduplication states.",
            "dopaminergic_frontostriatal_dysregulation": "Dopamine-related disruption of sequencing, executive control, speech fluency, and motor learning.",
            "executive_attention_burden": "Attention, inhibition, and working-memory burden limiting access to language learning and intervention.",
            "neurodevelopmental_cns_burden": "Intrinsic central nervous system developmental dysfunction underlying persistent language difficulty.",
            "left_lateralized_language_network_vulnerability": "Dominant-hemisphere posterior language vulnerability affecting phonological processing.",
            "social_pragmatic_network_burden": "Burden on distributed circuits supporting social-cognitive and pragmatic language use.",
            "environmental_language_exacerbation": "Non-primary under-support or deprivation that can worsen severity without being the core cause.",
            "stimulant_attention_support": "Protective attentional benefit from stimulant treatment or similar support that improves access to intervention.",
            "speech_language_therapy_support": "Protective speech-language intervention and structured practice support.",
            "enriched_language_environment": "Protective educational and language-rich environment improving learning opportunities.",
            "declarative_compensation_support": "Protective reliance on relatively preserved lexical / declarative memory resources.",
        }

        self.latent_nodes: Dict[str, str] = {
            "synaptic_plasticity_constraint": "Reduced efficiency of plasticity mechanisms needed to consolidate phonology, syntax, and semantics.",
            "frontostriatal_sequencing_dysfunction": "Fronto-striatal sequencing and motor-learning burden affecting rule learning and fluent output.",
            "procedural_memory_grammar_impairment": "Impaired procedural learning of grammatical rules and sequential regularities.",
            "phonological_representation_instability": "Weak or noisy phonological representations undermining sound-based language processing.",
            "speech_motor_planning_deficit": "Impaired planning and sequencing of speech movements, including apraxic features.",
            "prefrontal_executive_access_constraint": "Executive bottleneck limiting access to comprehension, expression, and intervention gains.",
            "frontotemporal_connectivity_atypicality": "Atypical fronto-temporal or fronto-parietal coordination during language processing.",
            "pragmatic_social_language_dysconnectivity": "Distributed network burden affecting pragmatic and context-sensitive language use.",
            "declarative_lexical_compensation": "Compensatory recruitment of lexical / declarative memory to offset procedural-language weakness.",
        }

        self.symptom_nodes: Dict[str, str] = {
            "expressive_language_impairment": "Difficulty producing age-expected spoken or written language output.",
            "receptive_language_impairment": "Difficulty understanding spoken or written language input.",
            "phonological_processing_difficulty": "Difficulty analyzing and manipulating speech sounds and phonological structure.",
            "grammatical_rule_learning_difficulty": "Difficulty learning and automating syntax and rule-based language patterns.",
            "speech_motor_planning_difficulty": "Difficulty programming fluent and accurate speech movements.",
            "pragmatic_language_difficulty": "Difficulty using language appropriately in social and contextual settings.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_language_liability",
                "target": "synaptic_plasticity_constraint",
                "relation": "loads inherited risk affecting neural development and language-learning efficiency",
                "domain": "genetics",
                "language_disorder_change": "increased",
            },
            {
                "source": "foxp2_pathway_disruption",
                "target": "speech_motor_planning_deficit",
                "relation": "raises speech motor planning burden and apraxia-like vulnerability",
                "domain": "genetics",
                "language_disorder_change": "increased",
            },
            {
                "source": "foxp2_pathway_disruption",
                "target": "frontostriatal_sequencing_dysfunction",
                "relation": "disrupts corticostriatal development and sequencing for speech",
                "domain": "genetics",
                "language_disorder_change": "increased",
            },
            {
                "source": "chromatin_transcription_burden",
                "target": "synaptic_plasticity_constraint",
                "relation": "broadly disrupts downstream targets needed for normal brain development",
                "domain": "genetics",
                "language_disorder_change": "increased",
            },
            {
                "source": "chromatin_transcription_burden",
                "target": "frontotemporal_connectivity_atypicality",
                "relation": "contributes to distributed developmental dysconnectivity",
                "domain": "genetics",
                "language_disorder_change": "increased",
            },
            {
                "source": "dopaminergic_frontostriatal_dysregulation",
                "target": "frontostriatal_sequencing_dysfunction",
                "relation": "impairs sequencing, motor learning, and speech fluency",
                "domain": "neurotransmitter",
                "language_disorder_change": "increased",
            },
            {
                "source": "dopaminergic_frontostriatal_dysregulation",
                "target": "prefrontal_executive_access_constraint",
                "relation": "reduces executive control and attentional access to language tasks",
                "domain": "neurotransmitter",
                "language_disorder_change": "increased",
            },
            {
                "source": "executive_attention_burden",
                "target": "prefrontal_executive_access_constraint",
                "relation": "adds working-memory, inhibition, and sustained-attention burden",
                "domain": "executive",
                "language_disorder_change": "increased",
            },
            {
                "source": "neurodevelopmental_cns_burden",
                "target": "frontotemporal_connectivity_atypicality",
                "relation": "reflects intrinsic CNS dysfunction underlying persistent language problems",
                "domain": "development",
                "language_disorder_change": "increased",
            },
            {
                "source": "left_lateralized_language_network_vulnerability",
                "target": "phonological_representation_instability",
                "relation": "undermines dominant left posterior phonological processing networks",
                "domain": "circuit",
                "language_disorder_change": "increased",
            },
            {
                "source": "social_pragmatic_network_burden",
                "target": "pragmatic_social_language_dysconnectivity",
                "relation": "disrupts pragmatic language and social-cognitive language use",
                "domain": "circuit",
                "language_disorder_change": "increased",
            },
            {
                "source": "environmental_language_exacerbation",
                "target": "phonological_representation_instability",
                "relation": "worsens expression of the disorder without serving as the primary cause",
                "domain": "environment",
                "language_disorder_change": "increased",
            },
            {
                "source": "stimulant_attention_support",
                "target": "prefrontal_executive_access_constraint",
                "relation": "improves attention and access to educational or therapeutic input",
                "domain": "protective",
                "language_disorder_change": "decreased",
            },
            {
                "source": "speech_language_therapy_support",
                "target": "phonological_representation_instability",
                "relation": "supports rehearsal and strengthening of phonological and expressive language skills",
                "domain": "protective",
                "language_disorder_change": "decreased",
            },
            {
                "source": "speech_language_therapy_support",
                "target": "speech_motor_planning_deficit",
                "relation": "partially offsets speech-motor programming burden through practice",
                "domain": "protective",
                "language_disorder_change": "decreased",
            },
            {
                "source": "enriched_language_environment",
                "target": "synaptic_plasticity_constraint",
                "relation": "improves language-learning opportunities and support for consolidation",
                "domain": "protective",
                "language_disorder_change": "decreased",
            },
            {
                "source": "declarative_compensation_support",
                "target": "declarative_lexical_compensation",
                "relation": "supports lexical compensation when procedural learning is weak",
                "domain": "protective",
                "language_disorder_change": "decreased",
            },
            {
                "source": "frontostriatal_sequencing_dysfunction",
                "target": "basal_ganglia_proxy",
                "relation": "loads striatal sequencing and motor-programming circuitry",
                "domain": "circuit",
                "language_disorder_change": "increased",
            },
            {
                "source": "speech_motor_planning_deficit",
                "target": "inferior_frontal_gyrus",
                "relation": "raises burden in inferior frontal speech-production circuitry",
                "domain": "circuit",
                "language_disorder_change": "increased",
            },
            {
                "source": "prefrontal_executive_access_constraint",
                "target": "pfc_control_proxy",
                "relation": "raises executive-control burden in prefrontal language-access circuitry",
                "domain": "circuit",
                "language_disorder_change": "increased",
            },
            {
                "source": "phonological_representation_instability",
                "target": "temporoparietal_phonology_proxy",
                "relation": "loads left temporo-parietal phonological circuitry",
                "domain": "circuit",
                "language_disorder_change": "increased",
            },
            {
                "source": "declarative_lexical_compensation",
                "target": "hippocampus",
                "relation": "recruits declarative-memory systems for lexical compensation",
                "domain": "circuit",
                "language_disorder_change": "increased",
            },
            {
                "source": "declarative_lexical_compensation",
                "target": "right_inferior_frontal_compensation_proxy",
                "relation": "supports compensatory right-hemisphere / inferior frontal recruitment",
                "domain": "circuit",
                "language_disorder_change": "increased",
            },
            {
                "source": "procedural_memory_grammar_impairment",
                "target": "grammatical_rule_learning_difficulty",
                "relation": "directly impairs rule learning and syntax automatization",
                "domain": "symptom",
                "language_disorder_change": "increased",
            },
            {
                "source": "phonological_representation_instability",
                "target": "phonological_processing_difficulty",
                "relation": "disrupts sound-based parsing and manipulation",
                "domain": "symptom",
                "language_disorder_change": "increased",
            },
            {
                "source": "speech_motor_planning_deficit",
                "target": "speech_motor_planning_difficulty",
                "relation": "impairs fluent and accurate programming of speech output",
                "domain": "symptom",
                "language_disorder_change": "increased",
            },
            {
                "source": "prefrontal_executive_access_constraint",
                "target": "expressive_language_impairment",
                "relation": "limits access to planning, retrieval, and task engagement for output",
                "domain": "symptom",
                "language_disorder_change": "increased",
            },
            {
                "source": "frontotemporal_connectivity_atypicality",
                "target": "receptive_language_impairment",
                "relation": "weakens coordinated processing of incoming language information",
                "domain": "symptom",
                "language_disorder_change": "increased",
            },
            {
                "source": "pragmatic_social_language_dysconnectivity",
                "target": "pragmatic_language_difficulty",
                "relation": "reduces context-sensitive and socially appropriate language use",
                "domain": "symptom",
                "language_disorder_change": "increased",
            },
            {
                "source": "declarative_lexical_compensation",
                "target": "expressive_language_impairment",
                "relation": "partially compensates expressive burden through lexical memory support",
                "domain": "symptom",
                "language_disorder_change": "decreased",
            },
            {
                "source": "declarative_lexical_compensation",
                "target": "receptive_language_impairment",
                "relation": "partially compensates receptive burden through preserved lexical knowledge",
                "domain": "symptom",
                "language_disorder_change": "decreased",
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

    def _parcellation_specs(self) -> List[str]:
        specs = [
            self.parcellation_spec,
            getattr(self.parcellation, "name", None),
            getattr(self.parcellation, "key", None),
        ]
        if "julich" in self.parcellation_spec.lower():
            specs.extend(["julich 3.0.3", "julich 2.9", "julich"])
        return [s for s in specs if isinstance(s, str) and s]

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
            cands.extend(["receptor density fingerprint", "ReceptorDensityFingerprint"])
        elif kind == "gene":
            cands.extend(["gene expressions", "GeneExpressions"])
        elif kind == "connectivity":
            cands.extend(["StreamlineCounts", "streamline counts"])
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
        out: List[Any] = []

        try:
            if hasattr(self.parcellation, "find"):
                found = self.parcellation.find(query)
                if found:
                    out.extend(list(found))
        except Exception:
            pass

        try:
            if hasattr(self.atlas, "find_regions"):
                found = self.atlas.find_regions(
                    query,
                    all_versions=False,
                    filter_children=False,
                    find_topmost=False,
                )
                if found:
                    out.extend(list(found))
        except Exception:
            pass

        dedup: List[Any] = []
        seen: set[str] = set()
        for region in out:
            parcellation_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" not in str(parcellation_name).lower() and out:
                continue
            ident = getattr(region, "identifier", None) or self._name_of(region)
            if str(ident) in seen:
                continue
            seen.add(str(ident))
            dedup.append(region)
        return dedup

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {"hippocampus", "striatum", "prefrontal cortex", "inferior frontal gyrus"} else 0
        cyto_bonus_penalty = 0 if "area " in name or "ca1" in name or "subiculum" in name else 1
        return (left_bonus, right_penalty, generic_penalty, cyto_bonus_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation_spec)
            except Exception:
                pass
            try:
                return self.parcellation.get_region(spec)
            except Exception:
                pass
            try:
                if hasattr(self.parcellation, "find"):
                    matches = list(self.parcellation.find(spec))
                    if matches:
                        matches = sorted(matches, key=self._region_rank)
                        return matches[0]
            except Exception:
                pass
            matches = self._julich_matches(spec)
            if matches:
                matches = sorted(matches, key=self._region_rank)
                return matches[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
            row = (
                self._name_of(region),
                str(getattr(region, "identifier", "") or ""),
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
        try:
            centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        except Exception:
            centroid_xyz = None
        volume_mm3 = getattr(main, "volume", None)
        try:
            volume_value = float(volume_mm3) if volume_mm3 is not None else None
        except Exception:
            volume_value = None
        return centroid_xyz, volume_value

    @staticmethod
    def _to_dataframe(data: Any) -> pd.DataFrame:
        if data is None:
            return pd.DataFrame()
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, pd.Series):
            name = data.name if data.name is not None else "value"
            df = data.to_frame(name=name).reset_index()
            if len(df.columns) == 2:
                df.columns = ["index", name]
            return df
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        try:
            df = self._to_dataframe(getattr(feats[0], "data", None)).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        if "index" in df.columns and "receptor" not in df.columns:
            df = df.rename(columns={"index": "receptor"})
        return df

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()
        df = self._to_dataframe(getattr(feats[0], "data", None)).reset_index(drop=True)
        if df.empty:
            return df
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
        return df

    def _pick_connectivity_feature(self, features: Sequence[Any]) -> Optional[Any]:
        if not features:
            return None
        preferred = [
            f
            for f in features
            if str(getattr(f, "cohort", "") or "").lower() == self.connectivity_cohort.lower()
        ]
        return preferred[0] if preferred else features[0]

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        feature = self._pick_connectivity_feature(feats)
        if feature is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        data = getattr(feature, "data", None)
        df = self._to_dataframe(data)
        if isinstance(data, pd.DataFrame) and not data.empty:
            self._connectivity_matrix = data.copy()
            return self._connectivity_matrix
        if not df.empty and df.shape[0] > 1 and df.shape[1] > 1:
            self._connectivity_matrix = df.copy()
            return self._connectivity_matrix

        try:
            subfeature = feature[0]
            subdata = getattr(subfeature, "data", None)
            if isinstance(subdata, pd.DataFrame):
                self._connectivity_matrix = subdata.copy()
                return self._connectivity_matrix
            subdf = self._to_dataframe(subdata)
            self._connectivity_matrix = subdf.copy()
            return self._connectivity_matrix
        except Exception:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._name_of(region)
        exact = [x for x in labels if self._name_of(x) == region_name]
        if exact:
            return exact[0]
        rn = region_name.lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        return fuzzy[0] if fuzzy else None

    def _connectivity_profile_from_matrix(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        idx_match = self._match_region_label(list(matrix.index), region)
        col_match = self._match_region_label(list(matrix.columns), region)
        try:
            if idx_match is not None:
                series = matrix.loc[idx_match]
            elif col_match is not None:
                series = matrix[col_match]
            else:
                return pd.DataFrame()
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _connectivity_profile_from_region_feature(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("connectivity"))
        feature = self._pick_connectivity_feature(feats)
        if feature is None:
            return pd.DataFrame()
        data = getattr(feature, "data", None)
        if isinstance(data, pd.Series):
            df = data.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            return df.head(max_rows).reset_index(drop=True)
        if isinstance(data, pd.DataFrame):
            idx_match = self._match_region_label(list(data.index), region)
            col_match = self._match_region_label(list(data.columns), region)
            try:
                if idx_match is not None:
                    series = data.loc[idx_match]
                elif col_match is not None:
                    series = data[col_match]
                else:
                    return pd.DataFrame()
                df = series.sort_values(ascending=False).reset_index()
                df.columns = ["connected_region", "value"]
                df["connected_region"] = df["connected_region"].map(self._name_of)
                df = df[df["connected_region"] != region.name].head(max_rows)
                return df.reset_index(drop=True)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        df = self._connectivity_profile_from_matrix(region, max_rows=max_rows)
        if not df.empty:
            return df
        return self._connectivity_profile_from_region_feature(region, max_rows=max_rows)

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return long-form pairwise connectivity between resolved circuit nodes.

        The output is empty when the runtime lacks compatible connectivity
        features. This is normal and not an error.
        """
        rows: List[Dict[str, Any]] = []
        matrix = self._get_connectivity_matrix()

        if not matrix.empty and self.region_objects:
            idx_labels = list(matrix.index)
            col_labels = list(matrix.columns)
            mapped_labels = {
                key: (
                    self._match_region_label(idx_labels, region),
                    self._match_region_label(col_labels, region),
                )
                for key, region in self.region_objects.items()
            }
            for src_key, src_region in self.region_objects.items():
                idx_label, col_label = mapped_labels.get(src_key, (None, None))
                if idx_label is None and col_label is None:
                    continue
                for tgt_key, tgt_region in self.region_objects.items():
                    if src_key == tgt_key:
                        continue
                    tgt_idx_label, tgt_col_label = mapped_labels.get(tgt_key, (None, None))
                    value = None
                    try:
                        if idx_label is not None and tgt_col_label is not None:
                            value = matrix.loc[idx_label, tgt_col_label]
                        elif col_label is not None and tgt_idx_label is not None:
                            value = matrix.loc[tgt_idx_label, col_label]
                        elif idx_label is not None and tgt_idx_label is not None:
                            value = matrix.loc[idx_label, tgt_idx_label]
                        elif col_label is not None and tgt_col_label is not None:
                            value = matrix.loc[col_label, tgt_col_label]
                    except Exception:
                        value = None
                    try:
                        value = float(value) if value is not None else None
                    except Exception:
                        value = None
                    if value is None:
                        continue
                    rows.append(
                        {
                            "source_key": src_key,
                            "source_region": src_region.name,
                            "target_key": tgt_key,
                            "target_region": tgt_region.name,
                            "value": value,
                        }
                    )
            if rows:
                return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

        for src_key, profile in self.connectivity_profiles.items():
            if profile.empty:
                continue
            for tgt_key, tgt_region in self.region_objects.items():
                if src_key == tgt_key:
                    continue
                try:
                    mask = profile["connected_region"].astype(str).str.lower().eq(tgt_region.name.lower())
                    if not mask.any():
                        mask = profile["connected_region"].astype(str).str.lower().str.contains(
                            tgt_region.name.lower(),
                            regex=False,
                        )
                    if not mask.any():
                        continue
                    value = float(profile.loc[mask, "value"].iloc[0])
                except Exception:
                    continue
                rows.append(
                    {
                        "source_key": src_key,
                        "source_region": self.region_objects[src_key].name,
                        "target_key": tgt_key,
                        "target_region": tgt_region.name,
                        "value": value,
                    }
                )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(self, gene_panel: Sequence[str] = DEFAULT_GENE_PANEL, connectivity_rows: int = 15) -> dict:
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
                    "is_proxy": False,
                }
            )

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            is_proxy = key.endswith("_proxy")
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
                        "description": self.region_node_descriptions.get(key, "Atlas-backed circuit node"),
                        "atlas_region": None,
                        "region_identifier": None,
                        "centroid_mni": None,
                        "volume_mm3": None,
                        "feature_summary": "unresolved",
                        "is_proxy": is_proxy,
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
                    "description": self.region_node_descriptions.get(key, "Atlas-backed circuit node"),
                    "atlas_region": region.name,
                    "region_identifier": getattr(region, "identifier", None),
                    "centroid_mni": centroid_mni,
                    "volume_mm3": volume_mm3,
                    "feature_summary": (
                        f"receptors={'yes' if not receptor_df.empty else 'no'}; "
                        f"genes={'yes' if not gene_df.empty else 'no'}; "
                        f"connectivity={'yes' if not conn_df.empty else 'no'}"
                    ),
                    "is_proxy": is_proxy,
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
                    "is_proxy": False,
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
                    "is_proxy": False,
                }
            )

        self.nodes_df = pd.DataFrame(nodes)
        self.edges_df = pd.DataFrame(self.edge_table)
        circuit_connectivity = self.circuit_connectivity()
        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": circuit_connectivity,
        }

    def _statistical_map(self):
        if self._pmap is not None:
            return self._pmap
        last_exc: Optional[Exception] = None
        for spec in self._parcellation_specs():
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
                return self._pmap
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError("Could not obtain a statistical parcellation map.") from last_exc

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        pmap = self._statistical_map()
        point = siibra.Point(tuple(xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = pmap.assign(point)
        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str, fetch: bool = False):
        """
        Return a region mask/volume for an atlas-backed node.

        Parameters
        ----------
        node_key:
            Region node key, for example "inferior_frontal_gyrus" or
            "temporoparietal_phonology_proxy".
        fetch:
            If True, fetch and return the concrete image object when possible.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        mask = None
        try:
            mask = region.get_regional_mask(self.space, maptype="labelled")
        except Exception:
            try:
                mask = region.get_regional_map(self.space, maptype="statistical")
            except Exception:
                return None
        if fetch:
            try:
                return mask.fetch()
            except Exception:
                return mask
        return mask

    def simulate(
        self,
        *,
        genetic_language_liability: float = 0.0,
        foxp2_pathway_disruption: float = 0.0,
        chromatin_transcription_burden: float = 0.0,
        dopaminergic_frontostriatal_dysregulation: float = 0.0,
        executive_attention_burden: float = 0.0,
        neurodevelopmental_cns_burden: float = 0.0,
        left_lateralized_language_network_vulnerability: float = 0.0,
        social_pragmatic_network_burden: float = 0.0,
        environmental_language_exacerbation: float = 0.0,
        stimulant_attention_support: float = 0.0,
        speech_language_therapy_support: float = 0.0,
        enriched_language_environment: float = 0.0,
        declarative_compensation_support: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Calculation order:
        inputs -> latent biology -> regional state -> symptoms -> phenotype summaries

        Values are clipped to [0, 1]. Protective terms subtract from burden,
        while declarative compensation is modeled as a positive compensatory
        latent that can offset selected downstream symptoms.
        """
        inputs = pd.Series(
            {
                "genetic_language_liability": self._clip01(genetic_language_liability),
                "foxp2_pathway_disruption": self._clip01(foxp2_pathway_disruption),
                "chromatin_transcription_burden": self._clip01(chromatin_transcription_burden),
                "dopaminergic_frontostriatal_dysregulation": self._clip01(dopaminergic_frontostriatal_dysregulation),
                "executive_attention_burden": self._clip01(executive_attention_burden),
                "neurodevelopmental_cns_burden": self._clip01(neurodevelopmental_cns_burden),
                "left_lateralized_language_network_vulnerability": self._clip01(left_lateralized_language_network_vulnerability),
                "social_pragmatic_network_burden": self._clip01(social_pragmatic_network_burden),
                "environmental_language_exacerbation": self._clip01(environmental_language_exacerbation),
                "stimulant_attention_support": self._clip01(stimulant_attention_support),
                "speech_language_therapy_support": self._clip01(speech_language_therapy_support),
                "enriched_language_environment": self._clip01(enriched_language_environment),
                "declarative_compensation_support": self._clip01(declarative_compensation_support),
            },
            dtype=float,
        )

        latents = pd.Series(dtype=float)
        latents["synaptic_plasticity_constraint"] = self._clip01(
            0.24 * inputs["genetic_language_liability"]
            + 0.24 * inputs["chromatin_transcription_burden"]
            + 0.20 * inputs["neurodevelopmental_cns_burden"]
            + 0.12 * inputs["environmental_language_exacerbation"]
            + 0.08 * inputs["executive_attention_burden"]
            - 0.18 * inputs["speech_language_therapy_support"]
            - 0.14 * inputs["enriched_language_environment"]
        )
        latents["frontostriatal_sequencing_dysfunction"] = self._clip01(
            0.28 * inputs["foxp2_pathway_disruption"]
            + 0.22 * inputs["dopaminergic_frontostriatal_dysregulation"]
            + 0.20 * inputs["executive_attention_burden"]
            + 0.16 * inputs["neurodevelopmental_cns_burden"]
            + 0.10 * inputs["genetic_language_liability"]
            - 0.18 * inputs["stimulant_attention_support"]
            - 0.10 * inputs["speech_language_therapy_support"]
        )
        latents["procedural_memory_grammar_impairment"] = self._clip01(
            0.38 * latents["frontostriatal_sequencing_dysfunction"]
            + 0.22 * latents["synaptic_plasticity_constraint"]
            + 0.14 * inputs["left_lateralized_language_network_vulnerability"]
            + 0.10 * inputs["environmental_language_exacerbation"]
            - 0.16 * inputs["speech_language_therapy_support"]
            - 0.06 * inputs["declarative_compensation_support"]
        )
        latents["phonological_representation_instability"] = self._clip01(
            0.32 * inputs["left_lateralized_language_network_vulnerability"]
            + 0.22 * latents["synaptic_plasticity_constraint"]
            + 0.14 * inputs["executive_attention_burden"]
            + 0.12 * inputs["environmental_language_exacerbation"]
            + 0.10 * inputs["neurodevelopmental_cns_burden"]
            - 0.18 * inputs["speech_language_therapy_support"]
            - 0.12 * inputs["enriched_language_environment"]
        )
        latents["speech_motor_planning_deficit"] = self._clip01(
            0.42 * inputs["foxp2_pathway_disruption"]
            + 0.24 * latents["frontostriatal_sequencing_dysfunction"]
            + 0.14 * inputs["dopaminergic_frontostriatal_dysregulation"]
            + 0.10 * inputs["neurodevelopmental_cns_burden"]
            - 0.18 * inputs["speech_language_therapy_support"]
        )
        latents["prefrontal_executive_access_constraint"] = self._clip01(
            0.36 * inputs["executive_attention_burden"]
            + 0.22 * inputs["dopaminergic_frontostriatal_dysregulation"]
            + 0.18 * inputs["neurodevelopmental_cns_burden"]
            + 0.10 * inputs["environmental_language_exacerbation"]
            - 0.24 * inputs["stimulant_attention_support"]
            - 0.10 * inputs["enriched_language_environment"]
        )
        latents["frontotemporal_connectivity_atypicality"] = self._clip01(
            0.24 * inputs["neurodevelopmental_cns_burden"]
            + 0.22 * inputs["left_lateralized_language_network_vulnerability"]
            + 0.18 * latents["synaptic_plasticity_constraint"]
            + 0.14 * latents["prefrontal_executive_access_constraint"]
            + 0.12 * inputs["chromatin_transcription_burden"]
            - 0.14 * inputs["speech_language_therapy_support"]
            - 0.10 * inputs["enriched_language_environment"]
        )
        latents["pragmatic_social_language_dysconnectivity"] = self._clip01(
            0.34 * inputs["social_pragmatic_network_burden"]
            + 0.20 * latents["frontotemporal_connectivity_atypicality"]
            + 0.16 * latents["prefrontal_executive_access_constraint"]
            + 0.12 * inputs["neurodevelopmental_cns_burden"]
            - 0.12 * inputs["speech_language_therapy_support"]
            - 0.08 * inputs["enriched_language_environment"]
        )
        latents["declarative_lexical_compensation"] = self._clip01(
            0.44 * inputs["declarative_compensation_support"]
            + 0.24 * inputs["enriched_language_environment"]
            + 0.16 * inputs["speech_language_therapy_support"]
            + 0.10 * (1.0 - latents["procedural_memory_grammar_impairment"])
        )

        regional_state = pd.Series(dtype=float)
        regional_state["inferior_frontal_gyrus"] = self._clip01(
            0.48 * latents["speech_motor_planning_deficit"]
            + 0.30 * latents["procedural_memory_grammar_impairment"]
            + 0.12 * latents["prefrontal_executive_access_constraint"]
            + 0.10 * latents["frontotemporal_connectivity_atypicality"]
        )
        regional_state["temporoparietal_phonology_proxy"] = self._clip01(
            0.52 * latents["phonological_representation_instability"]
            + 0.26 * latents["frontotemporal_connectivity_atypicality"]
            + 0.12 * inputs["left_lateralized_language_network_vulnerability"]
        )
        regional_state["pfc_control_proxy"] = self._clip01(
            0.54 * latents["prefrontal_executive_access_constraint"]
            + 0.22 * latents["frontotemporal_connectivity_atypicality"]
            + 0.12 * inputs["executive_attention_burden"]
        )
        regional_state["basal_ganglia_proxy"] = self._clip01(
            0.46 * latents["frontostriatal_sequencing_dysfunction"]
            + 0.26 * inputs["dopaminergic_frontostriatal_dysregulation"]
            + 0.14 * latents["speech_motor_planning_deficit"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.46 * latents["declarative_lexical_compensation"]
            + 0.20 * latents["frontotemporal_connectivity_atypicality"]
            + 0.12 * latents["synaptic_plasticity_constraint"]
            + 0.10 * inputs["enriched_language_environment"]
        )
        regional_state["right_inferior_frontal_compensation_proxy"] = self._clip01(
            0.48 * latents["declarative_lexical_compensation"]
            + 0.18 * inputs["speech_language_therapy_support"]
            + 0.12 * latents["phonological_representation_instability"]
            + 0.10 * latents["prefrontal_executive_access_constraint"]
        )

        symptoms = pd.Series(dtype=float)
        symptoms["expressive_language_impairment"] = self._clip01(
            0.30 * latents["procedural_memory_grammar_impairment"]
            + 0.22 * latents["speech_motor_planning_deficit"]
            + 0.18 * latents["prefrontal_executive_access_constraint"]
            + 0.16 * regional_state["inferior_frontal_gyrus"]
            + 0.08 * regional_state["basal_ganglia_proxy"]
            - 0.12 * latents["declarative_lexical_compensation"]
        )
        symptoms["receptive_language_impairment"] = self._clip01(
            0.30 * latents["phonological_representation_instability"]
            + 0.22 * latents["frontotemporal_connectivity_atypicality"]
            + 0.16 * regional_state["temporoparietal_phonology_proxy"]
            + 0.10 * latents["synaptic_plasticity_constraint"]
            + 0.10 * inputs["left_lateralized_language_network_vulnerability"]
            - 0.12 * latents["declarative_lexical_compensation"]
        )
        symptoms["phonological_processing_difficulty"] = self._clip01(
            0.38 * latents["phonological_representation_instability"]
            + 0.20 * regional_state["temporoparietal_phonology_proxy"]
            + 0.16 * latents["synaptic_plasticity_constraint"]
            + 0.10 * latents["prefrontal_executive_access_constraint"]
            - 0.10 * inputs["speech_language_therapy_support"]
            - 0.08 * inputs["enriched_language_environment"]
        )
        symptoms["grammatical_rule_learning_difficulty"] = self._clip01(
            0.42 * latents["procedural_memory_grammar_impairment"]
            + 0.20 * latents["frontostriatal_sequencing_dysfunction"]
            + 0.16 * regional_state["inferior_frontal_gyrus"]
            + 0.12 * regional_state["basal_ganglia_proxy"]
            - 0.10 * inputs["speech_language_therapy_support"]
            - 0.06 * latents["declarative_lexical_compensation"]
        )
        symptoms["speech_motor_planning_difficulty"] = self._clip01(
            0.48 * latents["speech_motor_planning_deficit"]
            + 0.20 * regional_state["inferior_frontal_gyrus"]
            + 0.16 * regional_state["basal_ganglia_proxy"]
            + 0.08 * inputs["dopaminergic_frontostriatal_dysregulation"]
            - 0.14 * inputs["speech_language_therapy_support"]
        )
        symptoms["pragmatic_language_difficulty"] = self._clip01(
            0.36 * latents["pragmatic_social_language_dysconnectivity"]
            + 0.20 * latents["frontotemporal_connectivity_atypicality"]
            + 0.16 * latents["prefrontal_executive_access_constraint"]
            + 0.12 * regional_state["pfc_control_proxy"]
            - 0.08 * inputs["speech_language_therapy_support"]
            - 0.08 * inputs["enriched_language_environment"]
        )

        phenotypes = pd.Series(dtype=float)
        phenotypes["overall_language_disorder_severity"] = self._clip01(
            0.22 * symptoms["expressive_language_impairment"]
            + 0.22 * symptoms["receptive_language_impairment"]
            + 0.18 * symptoms["phonological_processing_difficulty"]
            + 0.16 * symptoms["grammatical_rule_learning_difficulty"]
            + 0.12 * symptoms["speech_motor_planning_difficulty"]
            + 0.10 * symptoms["pragmatic_language_difficulty"]
        )
        phenotypes["procedural_language_profile"] = self._clip01(
            0.36 * symptoms["grammatical_rule_learning_difficulty"]
            + 0.24 * latents["frontostriatal_sequencing_dysfunction"]
            + 0.20 * regional_state["basal_ganglia_proxy"]
            + 0.20 * symptoms["expressive_language_impairment"]
        )
        phenotypes["speech_apraxia_profile"] = self._clip01(
            0.42 * symptoms["speech_motor_planning_difficulty"]
            + 0.24 * inputs["foxp2_pathway_disruption"]
            + 0.18 * regional_state["inferior_frontal_gyrus"]
            + 0.16 * regional_state["basal_ganglia_proxy"]
        )
        phenotypes["phonological_dyslexia_overlap"] = self._clip01(
            0.36 * symptoms["phonological_processing_difficulty"]
            + 0.24 * symptoms["receptive_language_impairment"]
            + 0.20 * regional_state["temporoparietal_phonology_proxy"]
            + 0.12 * regional_state["right_inferior_frontal_compensation_proxy"]
            + 0.08 * latents["phonological_representation_instability"]
        )
        phenotypes["executive_attention_language_profile"] = self._clip01(
            0.30 * latents["prefrontal_executive_access_constraint"]
            + 0.24 * symptoms["expressive_language_impairment"]
            + 0.22 * symptoms["pragmatic_language_difficulty"]
            + 0.14 * regional_state["pfc_control_proxy"]
            + 0.10 * inputs["executive_attention_burden"]
        )
        phenotypes["declarative_compensation_profile"] = self._clip01(
            0.40 * latents["declarative_lexical_compensation"]
            + 0.24 * regional_state["hippocampus"]
            + 0.18 * regional_state["right_inferior_frontal_compensation_proxy"]
            + 0.10 * inputs["declarative_compensation_support"]
            + 0.08 * inputs["enriched_language_environment"]
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }


if __name__ == "__main__":
    model = LanguageDisorderModel()
    scaffold = model.build()

    print("\n=== NODE SUMMARY ===")
    print(
        scaffold["nodes"][
            [
                "key",
                "node_type",
                "atlas_region",
                "feature_summary",
                "is_proxy",
            ]
        ].to_string(index=False)
    )

    print("\n=== EDGE SUMMARY ===")
    print(
        scaffold["edges"][
            ["source", "target", "relation", "language_disorder_change"]
        ].head(20).to_string(index=False)
    )

    print("\n=== REGION FEATURE SNAPSHOT ===")
    for key in ("inferior_frontal_gyrus", "temporoparietal_phonology_proxy", "hippocampus"):
        print(f"\n[{key}]")
        region_rows = scaffold["nodes"].query("key == @key")[["atlas_region", "centroid_mni", "feature_summary"]]
        print(region_rows.to_string(index=False))
        if not scaffold["receptors"].get(key, pd.DataFrame()).empty:
            print("receptors:")
            print(scaffold["receptors"][key].head().to_string(index=False))
        if not scaffold["genes"].get(key, pd.DataFrame()).empty:
            print("genes:")
            print(scaffold["genes"][key].head().to_string(index=False))
        if not scaffold["connectivity_profiles"].get(key, pd.DataFrame()).empty:
            print("connectivity:")
            print(scaffold["connectivity_profiles"][key].head().to_string(index=False))

    print("\n=== CIRCUIT CONNECTIVITY ===")
    if scaffold["circuit_connectivity"].empty:
        print("No pairwise circuit connectivity available in this runtime.")
    else:
        print(scaffold["circuit_connectivity"].head(15).to_string(index=False))

    example = model.simulate(
        genetic_language_liability=0.55,
        foxp2_pathway_disruption=0.25,
        chromatin_transcription_burden=0.20,
        dopaminergic_frontostriatal_dysregulation=0.45,
        executive_attention_burden=0.60,
        neurodevelopmental_cns_burden=0.55,
        left_lateralized_language_network_vulnerability=0.65,
        social_pragmatic_network_burden=0.35,
        environmental_language_exacerbation=0.20,
        stimulant_attention_support=0.35,
        speech_language_therapy_support=0.45,
        enriched_language_environment=0.40,
        declarative_compensation_support=0.50,
    )

    print("\n=== SIMULATED SYMPTOMS ===")
    print(example["symptoms"].sort_values(ascending=False).to_string())

    print("\n=== SIMULATED PHENOTYPES ===")
    print(example["phenotypes"].sort_values(ascending=False).to_string())

    # Optional manual testing examples:
    # assignments = model.assign_mni_point((-52.0, 14.0, 18.0))
    # print(assignments.head().to_string(index=False))
    # mask_img = model.region_mask("inferior_frontal_gyrus", fetch=True)
