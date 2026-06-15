from __future__ import annotations

import warnings
from itertools import islice
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore
    _SIIBRA_IMPORT_ERROR = exc
else:
    _SIIBRA_IMPORT_ERROR = None


IDD_GENE_PANEL = [
    "FMR1",
    "MECP2",
    "TCF4",
    "ARID1B",
    "SYNGAP1",
    "SHANK3",
    "DYRK1A",
    "UBE3A",
    "EHMT1",
    "SCN2A",
    "PTEN",
    "SLC2A1",
]


class IntellectualDevelopmentalDisorderModel:
    """
    Atlas-grounded siibra scaffold for Intellectual Developmental Disorder (IDD).

    This script is a research scaffold, not a diagnostic, prognostic, or treatment tool.
    It translates a chapter-level description of IDD into a transparent mechanistic model
    with conservative atlas anchoring. The source chapter emphasizes that IDD is a
    heterogeneous neurodevelopmental condition affecting both "cold" cognition
    (reasoning, memory, processing) and "hot" cognition (emotion regulation, social
    perception), with broad genetic influence and strong gene-environment interaction.

    Design choices in this scaffold:
    - Chemistry is kept largely latent because the chapter does not localize specific
      neurotransmitter systems to precise parcels.
    - Prefrontal and temporoparietal systems are handled conservatively with labeled
      proxies, while hippocampus and amygdala are treated as explicit regional anchors.
    - The gene panel is illustrative rather than exhaustive, reflecting common
      neurodevelopmental / synaptic-development genes relevant to IDD heterogeneity.

    Higher regional-state values produced by ``simulate()`` indicate greater dysfunction
    burden in that node, not greater healthy activation.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
        connectivity_average_n: int = 5,
    ) -> None:
        if siibra is None:  # pragma: no cover - environment dependent
            raise ImportError(
                "siibra is required to use IntellectualDevelopmentalDisorderModel. "
                "Install siibra-python in your runtime before building the scaffold."
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
        self.connectivity_average_n = max(1, int(connectivity_average_n))

        self.region_candidates: Dict[str, List[str]] = {
            "pfc_control": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area IFJ left",
                "9/46 left",
                "dorsolateral prefrontal",
                "prefrontal cortex",
            ],
            "hippocampus": [
                "CA1 left",
                "Subiculum left",
                "CA2 left",
                "CA3 left",
                "DG left",
                "hippocampus left",
                "hippocampus",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "medial_pfc_proxy": [
                "Area p32 (pACC) left",
                "Area s24 (pACC) left",
                "Area 10 left",
                "p32 left",
                "medial prefrontal",
            ],
            "tpj_proxy": [
                "Area TPJ (STG/SMG) left",
                "TPJ left",
                "Area PGp left",
                "Area PGa left",
                "temporoparietal junction",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "pfc_control": (
                "Proxy for frontal/prefrontal executive systems implicated in planning, "
                "cognitive flexibility, and inhibition"
            ),
            "hippocampus": (
                "Medial temporal memory circuitry linked to encoding and retrieval burden"
            ),
            "amygdala": (
                "Affective salience and social-emotional response node within the social brain"
            ),
            "medial_pfc_proxy": (
                "Proxy for medial prefrontal social-cognitive and self-regulatory cortex"
            ),
            "tpj_proxy": (
                "Proxy for temporoparietal junction processes involved in social inference and theory of mind"
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_liability": (
                "Broad heritable neurodevelopmental risk load reflecting the chapter's heterogeneous genetic causes"
            ),
            "home_environment_quality": (
                "Protective quality of the home and developmental environment"
            ),
            "educational_support": (
                "Protective access to education, scaffolding, and learning opportunity"
            ),
            "nutrition_health_support": (
                "Protective nutritional and health support relevant to development"
            ),
            "social_support": (
                "Protective social support buffering social-cognitive and emotional burden"
            ),
            "stress_burden": (
                "Stress load that can worsen executive, emotional, and attentional inefficiency"
            ),
            "comorbid_mood_anxiety_load": (
                "Comorbid emotional burden that can amplify dysregulation and adaptation difficulty"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "distributed_neurodevelopmental_dysregulation": (
                "Broad developmental CNS burden affecting information intake, processing, storage, and response"
            ),
            "global_cognitive_processing_constraint": (
                "Cold-cognition burden on reasoning, information processing speed, and general intellectual function"
            ),
            "executive_control_inefficiency": (
                "Planning, inhibition, and cognitive-flexibility inefficiency linked to frontal systems"
            ),
            "memory_encoding_retrieval_constraint": (
                "Memory burden associated with temporal-lobe and hippocampal circuitry"
            ),
            "emotion_regulation_dysfunction": (
                "Hot-cognition dysregulation affecting affective control and emotionally guided behavior"
            ),
            "social_cognitive_impairment": (
                "Burden in emotion perception, empathy, and theory-of-mind processes"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "reasoning_learning_difficulty": (
                "Reduced general intellectual performance, reasoning, and learning efficiency"
            ),
            "executive_dysfunction": (
                "Difficulties with planning, inhibition, and flexible goal-directed behavior"
            ),
            "memory_difficulty": (
                "Memory encoding or retrieval difficulty"
            ),
            "emotion_regulation_difficulty": (
                "Difficulty modulating emotion and affective arousal"
            ),
            "social_perception_difficulty": (
                "Difficulty perceiving, interpreting, and responding to social information"
            ),
            "decision_making_problem_solving_difficulty": (
                "Difficulty with problem solving and decisions when cognition and emotion interact"
            ),
            "affective_lability_irritability": (
                "Irritability or affective lability associated with poor emotional regulation"
            ),
            "inappropriate_social_response": (
                "Behaviorally miscalibrated social response arising from social-cognitive and emotional burden"
            ),
            "adaptive_functioning_difficulty": (
                "Broad downstream difficulty in everyday adaptive functioning"
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_liability",
                "target": "distributed_neurodevelopmental_dysregulation",
                "relation": "increases broad developmental CNS burden",
                "idd_change": "increased",
            },
            {
                "source": "home_environment_quality",
                "target": "distributed_neurodevelopmental_dysregulation",
                "relation": "buffers developmental burden through environmental quality",
                "idd_change": "decreased",
            },
            {
                "source": "educational_support",
                "target": "distributed_neurodevelopmental_dysregulation",
                "relation": "buffers developmental burden through learning support",
                "idd_change": "decreased",
            },
            {
                "source": "nutrition_health_support",
                "target": "distributed_neurodevelopmental_dysregulation",
                "relation": "buffers developmental burden through health and nutrition support",
                "idd_change": "decreased",
            },
            {
                "source": "social_support",
                "target": "social_cognitive_impairment",
                "relation": "buffers downstream social-cognitive burden",
                "idd_change": "decreased",
            },
            {
                "source": "stress_burden",
                "target": "emotion_regulation_dysfunction",
                "relation": "increases emotional dysregulation and arousal burden",
                "idd_change": "increased",
            },
            {
                "source": "comorbid_mood_anxiety_load",
                "target": "emotion_regulation_dysfunction",
                "relation": "amplifies emotional self-regulation difficulty",
                "idd_change": "increased",
            },
            {
                "source": "distributed_neurodevelopmental_dysregulation",
                "target": "global_cognitive_processing_constraint",
                "relation": "reduces efficient receiving, processing, storing, and responding to information",
                "idd_change": "increased",
            },
            {
                "source": "distributed_neurodevelopmental_dysregulation",
                "target": "executive_control_inefficiency",
                "relation": "drives frontal-system inefficiency",
                "idd_change": "increased",
            },
            {
                "source": "distributed_neurodevelopmental_dysregulation",
                "target": "memory_encoding_retrieval_constraint",
                "relation": "drives temporal-limbic memory burden",
                "idd_change": "increased",
            },
            {
                "source": "distributed_neurodevelopmental_dysregulation",
                "target": "social_cognitive_impairment",
                "relation": "increases burden on distributed social-brain functions",
                "idd_change": "increased",
            },
            {
                "source": "emotion_regulation_dysfunction",
                "target": "social_cognitive_impairment",
                "relation": "degrades social interpretation when affective processing is unstable",
                "idd_change": "increased",
            },
            {
                "source": "executive_control_inefficiency",
                "target": "pfc_control",
                "relation": "maps to dysfunction burden in frontal executive-control systems",
                "idd_change": "increased",
            },
            {
                "source": "memory_encoding_retrieval_constraint",
                "target": "hippocampus",
                "relation": "maps to dysfunction burden in hippocampal memory circuitry",
                "idd_change": "increased",
            },
            {
                "source": "emotion_regulation_dysfunction",
                "target": "amygdala",
                "relation": "maps to dysfunction burden in affective salience circuitry",
                "idd_change": "increased",
            },
            {
                "source": "social_cognitive_impairment",
                "target": "medial_pfc_proxy",
                "relation": "maps to dysfunction burden in medial prefrontal social-cognitive systems",
                "idd_change": "increased",
            },
            {
                "source": "social_cognitive_impairment",
                "target": "tpj_proxy",
                "relation": "maps to dysfunction burden in temporoparietal social-inference systems",
                "idd_change": "increased",
            },
            {
                "source": "global_cognitive_processing_constraint",
                "target": "reasoning_learning_difficulty",
                "relation": "reduces general reasoning and learning efficiency",
                "idd_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "executive_dysfunction",
                "relation": "increases planning, inhibition, and flexibility problems",
                "idd_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "memory_difficulty",
                "relation": "increases encoding and retrieval problems",
                "idd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "emotion_regulation_difficulty",
                "relation": "increases affective arousal and regulation problems",
                "idd_change": "increased",
            },
            {
                "source": "medial_pfc_proxy",
                "target": "social_perception_difficulty",
                "relation": "increases social interpretation difficulty",
                "idd_change": "increased",
            },
            {
                "source": "tpj_proxy",
                "target": "social_perception_difficulty",
                "relation": "increases theory-of-mind and perspective-taking difficulty",
                "idd_change": "increased",
            },
            {
                "source": "emotion_regulation_dysfunction",
                "target": "decision_making_problem_solving_difficulty",
                "relation": "disrupts higher-order cognition when emotion and cognition interact",
                "idd_change": "increased",
            },
            {
                "source": "emotion_regulation_difficulty",
                "target": "affective_lability_irritability",
                "relation": "produces irritability and affective lability",
                "idd_change": "increased",
            },
            {
                "source": "social_perception_difficulty",
                "target": "inappropriate_social_response",
                "relation": "increases miscalibrated social responding",
                "idd_change": "increased",
            },
            {
                "source": "reasoning_learning_difficulty",
                "target": "adaptive_functioning_difficulty",
                "relation": "impairs broad daily adaptation",
                "idd_change": "increased",
            },
            {
                "source": "executive_dysfunction",
                "target": "adaptive_functioning_difficulty",
                "relation": "reduces organized goal-directed adaptation",
                "idd_change": "increased",
            },
            {
                "source": "memory_difficulty",
                "target": "adaptive_functioning_difficulty",
                "relation": "reduces effective daily learning and recall",
                "idd_change": "increased",
            },
            {
                "source": "emotion_regulation_difficulty",
                "target": "adaptive_functioning_difficulty",
                "relation": "reduces daily emotional and behavioral stability",
                "idd_change": "increased",
            },
            {
                "source": "inappropriate_social_response",
                "target": "adaptive_functioning_difficulty",
                "relation": "worsens social adaptation demands",
                "idd_change": "increased",
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
            matches.extend(
                list(
                    self.atlas.find_regions(
                        query,
                        all_versions=False,
                        filter_children=False,
                        find_topmost=False,
                    )
                )
            )
        except Exception:
            pass
        try:
            if hasattr(self.parcellation, "find"):
                found = self.parcellation.find(query, filter_children=False)
                if found:
                    matches.extend(list(found))
        except Exception:
            pass

        out: List[Any] = []
        seen = set()
        for region in matches:
            key = (self._name_of(region), getattr(region, "identifier", None))
            if key in seen:
                continue
            seen.add(key)
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower() or self.parcellation.name in str(parc_name):
                out.append(region)
                continue
            if not parc_name:
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_terms = {"amygdala", "hippocampus", "prefrontal cortex", "medial prefrontal"}
        generic_penalty = 1 if name in generic_terms else 0
        parent_penalty = 1 if name.count("(") == 0 and "area" not in name else 0
        return (left_bonus, right_penalty, generic_penalty, parent_penalty)

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

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz = None
        if centroid is not None:
            raw = getattr(centroid, "coordinate", centroid)
            try:
                centroid_xyz = tuple(float(x) for x in raw)
            except Exception:
                centroid_xyz = None
        volume_mm3 = getattr(main, "volume", None)
        try:
            volume_mm3 = float(volume_mm3) if volume_mm3 is not None else None
        except Exception:
            volume_mm3 = None
        return centroid_xyz, volume_mm3

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        feat = feats[0]
        try:
            df = feat.data.copy()
        except Exception:
            try:
                df = pd.DataFrame(feat.data)
            except Exception:
                return pd.DataFrame()
        df = df.reset_index()
        if "index" in df.columns and "receptor" not in df.columns:
            df = df.rename(columns={"index": "receptor"})
        return df

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()
        feat = feats[0]
        try:
            df = feat.data.copy()
        except Exception:
            try:
                df = pd.DataFrame(feat.data)
            except Exception:
                return pd.DataFrame()
        if df.empty:
            return df

        lower_cols = {c.lower(): c for c in df.columns}
        gene_col = next((lower_cols[c] for c in ("gene", "gene_symbol", "symbol") if c in lower_cols), None)
        level_col = next((lower_cols[c] for c in ("level", "expression", "value") if c in lower_cols), None)
        zscore_col = next((lower_cols[c] for c in ("zscore", "z_score", "z-score") if c in lower_cols), None)

        if gene_col and level_col:
            agg: Dict[str, Tuple[str, str]] = {
                "level_mean": (level_col, "mean"),
                "level_std": (level_col, "std"),
                "probe_count": (level_col, "count"),
            }
            if zscore_col:
                agg.update(
                    {
                        "zscore_mean": (zscore_col, "mean"),
                        "zscore_std": (zscore_col, "std"),
                    }
                )
            return (
                df.groupby(gene_col, dropna=False)
                .agg(**agg)
                .reset_index()
                .rename(columns={gene_col: "gene"})
                .sort_values("gene")
                .reset_index(drop=True)
            )
        return df.reset_index(drop=True)

    @staticmethod
    def _average_aligned_matrices(matrices: Sequence[pd.DataFrame]) -> pd.DataFrame:
        if not matrices:
            return pd.DataFrame()
        labels = []
        seen = set()
        for mat in matrices:
            for label in mat.index:
                key = str(label)
                if key not in seen:
                    labels.append(label)
                    seen.add(key)
        out = pd.DataFrame(index=labels, columns=labels, dtype=float)
        counts = pd.DataFrame(0.0, index=labels, columns=labels)
        for mat in matrices:
            work = mat.copy()
            work = work.apply(pd.to_numeric, errors="coerce")
            work = work.reindex(index=labels, columns=labels)
            mask = work.notna().astype(float)
            out = out.add(work.fillna(0.0), fill_value=0.0)
            counts = counts.add(mask, fill_value=0.0)
        counts = counts.replace(0.0, pd.NA)
        return out.divide(counts).dropna(axis=0, how="all").dropna(axis=1, how="all")

    def _iter_feature_elements(self, feature: Any, limit: int) -> Iterable[Any]:
        try:
            iterator = iter(feature)
        except Exception:
            return []
        return list(islice(iterator, limit))

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), feats[0])

        matrices: List[pd.DataFrame] = []
        for elem in self._iter_feature_elements(compound, self.connectivity_average_n):
            data = getattr(elem, "data", None)
            if isinstance(data, pd.DataFrame) and not data.empty:
                matrices.append(data.copy())

        if matrices:
            self._connectivity_matrix = self._average_aligned_matrices(matrices)
            return self._connectivity_matrix

        data = getattr(compound, "data", None)
        if isinstance(data, pd.DataFrame):
            self._connectivity_matrix = data.copy()
            return self._connectivity_matrix

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
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        return fuzzy[0] if fuzzy else None

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
            series = pd.to_numeric(series, errors="coerce").dropna().sort_values(ascending=False)
            df = series.reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        matched: Dict[str, Any] = {}
        all_labels = list(dict.fromkeys(list(matrix.index) + list(matrix.columns)))
        for key, region in self.region_objects.items():
            label = self._match_region_label(all_labels, region)
            if label is not None:
                matched[key] = label

        if not matched:
            return pd.DataFrame()

        rows: List[pd.Series] = []
        for src_key, src_label in matched.items():
            row: Dict[str, float] = {}
            for dst_key, dst_label in matched.items():
                value = pd.NA
                try:
                    value = matrix.loc[src_label, dst_label]
                except Exception:
                    try:
                        value = matrix[dst_label].loc[src_label]
                    except Exception:
                        value = pd.NA
                row[dst_key] = value
            rows.append(pd.Series(row, name=src_key))
        return pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")

    def build(self, gene_panel: Sequence[str] = IDD_GENE_PANEL, connectivity_rows: int = 15) -> dict:
        nodes = []
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
                        "description": self.region_node_descriptions.get(
                            key, "Atlas-backed region or proxy node (unresolved in this environment)"
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
            "metadata": {
                "model_name": self.__class__.__name__,
                "gene_panel": list(gene_panel),
                "connectivity_cohort": self.connectivity_cohort,
                "space": self.space_spec,
                "parcellation": self.parcellation_spec,
                "notes": (
                    "Research scaffold only. Molecular detail is intentionally conservative "
                    "because the chapter is systems-level and genetically heterogeneous."
                ),
            },
        }

    def simulate(
        self,
        genetic_liability: float = 0.5,
        home_environment_quality: float = 0.65,
        educational_support: float = 0.65,
        nutrition_health_support: float = 0.65,
        social_support: float = 0.65,
        stress_burden: float = 0.35,
        comorbid_mood_anxiety_load: float = 0.2,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized one-pass simulation.

        All inputs are expected on a 0..1 scale. Higher support values are protective,
        whereas higher genetic, stress, and comorbidity values increase burden.
        Returned regional-state values represent dysfunction burden in the anchored system.
        """

        inputs = {
            "genetic_liability": self._clip01(genetic_liability),
            "home_environment_quality": self._clip01(home_environment_quality),
            "educational_support": self._clip01(educational_support),
            "nutrition_health_support": self._clip01(nutrition_health_support),
            "social_support": self._clip01(social_support),
            "stress_burden": self._clip01(stress_burden),
            "comorbid_mood_anxiety_load": self._clip01(comorbid_mood_anxiety_load),
        }

        support_mean = (
            inputs["home_environment_quality"]
            + inputs["educational_support"]
            + inputs["nutrition_health_support"]
            + inputs["social_support"]
        ) / 4.0

        latents = {
            "distributed_neurodevelopmental_dysregulation": self._clip01(
                0.48 * inputs["genetic_liability"]
                + 0.12 * inputs["stress_burden"]
                + 0.14 * (1.0 - inputs["home_environment_quality"])
                + 0.10 * (1.0 - inputs["educational_support"])
                + 0.08 * (1.0 - inputs["nutrition_health_support"])
                + 0.08 * (1.0 - inputs["social_support"])
            ),
            "global_cognitive_processing_constraint": 0.0,
            "executive_control_inefficiency": 0.0,
            "memory_encoding_retrieval_constraint": 0.0,
            "emotion_regulation_dysfunction": 0.0,
            "social_cognitive_impairment": 0.0,
        }

        latents["global_cognitive_processing_constraint"] = self._clip01(
            0.60 * latents["distributed_neurodevelopmental_dysregulation"]
            + 0.12 * inputs["comorbid_mood_anxiety_load"]
            + 0.08 * inputs["stress_burden"]
            - 0.12 * support_mean
        )

        latents["executive_control_inefficiency"] = self._clip01(
            0.44 * latents["distributed_neurodevelopmental_dysregulation"]
            + 0.22 * inputs["stress_burden"]
            + 0.12 * inputs["comorbid_mood_anxiety_load"]
            - 0.15 * inputs["educational_support"]
            - 0.07 * inputs["home_environment_quality"]
        )

        latents["memory_encoding_retrieval_constraint"] = self._clip01(
            0.46 * latents["distributed_neurodevelopmental_dysregulation"]
            + 0.16 * inputs["stress_burden"]
            + 0.12 * (1.0 - inputs["nutrition_health_support"])
            - 0.14 * inputs["educational_support"]
        )

        latents["emotion_regulation_dysfunction"] = self._clip01(
            0.28 * latents["distributed_neurodevelopmental_dysregulation"]
            + 0.30 * inputs["stress_burden"]
            + 0.24 * inputs["comorbid_mood_anxiety_load"]
            - 0.12 * inputs["social_support"]
            - 0.10 * inputs["home_environment_quality"]
        )

        latents["social_cognitive_impairment"] = self._clip01(
            0.34 * latents["distributed_neurodevelopmental_dysregulation"]
            + 0.22 * latents["emotion_regulation_dysfunction"]
            + 0.10 * inputs["stress_burden"]
            - 0.18 * inputs["social_support"]
            - 0.08 * inputs["educational_support"]
        )

        regional_state = {
            "pfc_control": self._clip01(
                0.56 * latents["executive_control_inefficiency"]
                + 0.16 * latents["emotion_regulation_dysfunction"]
                + 0.12 * inputs["stress_burden"]
                - 0.14 * inputs["educational_support"]
                - 0.08 * inputs["home_environment_quality"]
            ),
            "hippocampus": self._clip01(
                0.60 * latents["memory_encoding_retrieval_constraint"]
                + 0.14 * inputs["stress_burden"]
                + 0.08 * latents["distributed_neurodevelopmental_dysregulation"]
                - 0.12 * inputs["nutrition_health_support"]
            ),
            "amygdala": self._clip01(
                0.52 * latents["emotion_regulation_dysfunction"]
                + 0.18 * inputs["stress_burden"]
                + 0.10 * latents["social_cognitive_impairment"]
                - 0.10 * inputs["social_support"]
            ),
            "medial_pfc_proxy": self._clip01(
                0.40 * latents["social_cognitive_impairment"]
                + 0.18 * latents["emotion_regulation_dysfunction"]
                + 0.12 * latents["executive_control_inefficiency"]
                - 0.14 * inputs["social_support"]
                - 0.08 * inputs["educational_support"]
            ),
            "tpj_proxy": self._clip01(
                0.44 * latents["social_cognitive_impairment"]
                + 0.12 * latents["global_cognitive_processing_constraint"]
                + 0.10 * inputs["stress_burden"]
                - 0.14 * inputs["social_support"]
            ),
        }

        symptoms = {
            "reasoning_learning_difficulty": self._clip01(
                0.58 * latents["global_cognitive_processing_constraint"]
                + 0.16 * latents["memory_encoding_retrieval_constraint"]
                + 0.12 * regional_state["pfc_control"]
                - 0.14 * inputs["educational_support"]
            ),
            "executive_dysfunction": self._clip01(
                0.56 * latents["executive_control_inefficiency"]
                + 0.24 * regional_state["pfc_control"]
                + 0.08 * inputs["stress_burden"]
            ),
            "memory_difficulty": self._clip01(
                0.60 * latents["memory_encoding_retrieval_constraint"]
                + 0.24 * regional_state["hippocampus"]
                + 0.06 * inputs["stress_burden"]
            ),
            "emotion_regulation_difficulty": self._clip01(
                0.56 * latents["emotion_regulation_dysfunction"]
                + 0.24 * regional_state["amygdala"]
                + 0.08 * regional_state["medial_pfc_proxy"]
                - 0.12 * inputs["social_support"]
            ),
            "social_perception_difficulty": self._clip01(
                0.42 * latents["social_cognitive_impairment"]
                + 0.18 * regional_state["tpj_proxy"]
                + 0.16 * regional_state["medial_pfc_proxy"]
                + 0.08 * regional_state["amygdala"]
                - 0.12 * inputs["social_support"]
            ),
            "decision_making_problem_solving_difficulty": self._clip01(
                0.34 * latents["executive_control_inefficiency"]
                + 0.24 * latents["emotion_regulation_dysfunction"]
                + 0.18 * regional_state["pfc_control"]
                + 0.10 * latents["memory_encoding_retrieval_constraint"]
            ),
            "affective_lability_irritability": self._clip01(
                0.42 * latents["emotion_regulation_dysfunction"]
                + 0.20 * regional_state["amygdala"]
                + 0.12 * inputs["stress_burden"]
                + 0.10 * inputs["comorbid_mood_anxiety_load"]
                - 0.10 * inputs["social_support"]
            ),
            "inappropriate_social_response": self._clip01(
                0.36 * latents["social_cognitive_impairment"]
                + 0.24 * latents["emotion_regulation_dysfunction"]
                + 0.14 * regional_state["tpj_proxy"]
                + 0.10 * regional_state["amygdala"]
                - 0.10 * inputs["social_support"]
            ),
            "adaptive_functioning_difficulty": 0.0,
        }

        symptoms["adaptive_functioning_difficulty"] = self._clip01(
            0.24 * symptoms["reasoning_learning_difficulty"]
            + 0.18 * symptoms["executive_dysfunction"]
            + 0.12 * symptoms["memory_difficulty"]
            + 0.16 * symptoms["emotion_regulation_difficulty"]
            + 0.14 * symptoms["social_perception_difficulty"]
            + 0.16 * symptoms["inappropriate_social_response"]
        )

        phenotypes = {
            "cold_cognition_burden": self._clip01(
                (
                    symptoms["reasoning_learning_difficulty"]
                    + symptoms["executive_dysfunction"]
                    + symptoms["memory_difficulty"]
                )
                / 3.0
            ),
            "hot_cognition_burden": self._clip01(
                (
                    symptoms["emotion_regulation_difficulty"]
                    + symptoms["affective_lability_irritability"]
                    + symptoms["decision_making_problem_solving_difficulty"]
                )
                / 3.0
            ),
            "social_cognition_burden": self._clip01(
                (
                    symptoms["social_perception_difficulty"]
                    + symptoms["inappropriate_social_response"]
                    + symptoms["emotion_regulation_difficulty"]
                )
                / 3.0
            ),
            "adaptive_support_needs": self._clip01(
                (
                    symptoms["adaptive_functioning_difficulty"]
                    + symptoms["reasoning_learning_difficulty"]
                    + symptoms["social_perception_difficulty"]
                )
                / 3.0
            ),
            "comorbidity_vulnerability": self._clip01(
                0.42 * symptoms["emotion_regulation_difficulty"]
                + 0.20 * inputs["stress_burden"]
                + 0.22 * inputs["comorbid_mood_anxiety_load"]
                - 0.12 * inputs["social_support"]
            ),
        }

        return {
            "inputs": pd.Series(inputs, name="value"),
            "latents": pd.Series(latents, name="value"),
            "regional_state": pd.Series(regional_state, name="value"),
            "symptoms": pd.Series(symptoms, name="value"),
            "phenotypes": pd.Series(phenotypes, name="value"),
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

        out = assignments.copy()
        if "region" in out.columns:
            out["region"] = out["region"].map(self._name_of)
        for candidate in (
            "map value",
            "correlation",
            "intersection over union",
            "contained",
            "contains",
        ):
            if candidate in out.columns:
                out = out.sort_values(candidate, ascending=False)
                break
        return out.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Any:
        region = self.region_objects.get(node_key)
        if region is None and node_key in self.region_candidates:
            region = self._resolve_region(self.region_candidates[node_key])
        if region is None:
            return None
        try:
            with siibra.QUIET:
                return region.get_regional_mask(self.space, maptype="labelled")
        except Exception:
            return None


if __name__ == "__main__":
    if siibra is None:  # pragma: no cover - environment dependent
        print(
            "siibra is not installed in this runtime. Install siibra-python to build and query the scaffold."
        )
        raise SystemExit(0)

    model = IntellectualDevelopmentalDisorderModel()
    built = model.build(connectivity_rows=10)

    print("\n=== NODES ===")
    print(
        built["nodes"][
            [
                "key",
                "node_type",
                "atlas_region",
                "region_identifier",
                "feature_summary",
            ]
        ].to_string(index=False)
    )

    print("\n=== EDGES ===")
    print(built["edges"].to_string(index=False))

    print("\n=== CIRCUIT CONNECTIVITY ===")
    circuit_df = built["circuit_connectivity"]
    if circuit_df.empty:
        print("No circuit connectivity matrix available in this environment.")
    else:
        print(circuit_df.round(3).to_string())

    for region_key in ("pfc_control", "hippocampus", "amygdala"):
        receptor_df = built["receptors"].get(region_key, pd.DataFrame())
        gene_df = built["genes"].get(region_key, pd.DataFrame())
        conn_df = built["connectivity_profiles"].get(region_key, pd.DataFrame())

        if not receptor_df.empty:
            print(f"\n=== RECEPTOR TABLE: {region_key} ===")
            print(receptor_df.head(10).to_string(index=False))
        else:
            print(f"\n=== RECEPTOR TABLE: {region_key} ===")
            print("No receptor fingerprint available.")

        if not gene_df.empty:
            print(f"\n=== GENE TABLE: {region_key} ===")
            print(gene_df.head(10).to_string(index=False))
        else:
            print(f"\n=== GENE TABLE: {region_key} ===")
            print("No gene-expression summary available.")

        if not conn_df.empty:
            print(f"\n=== CONNECTIVITY PROFILE: {region_key} ===")
            print(conn_df.to_string(index=False))
        else:
            print(f"\n=== CONNECTIVITY PROFILE: {region_key} ===")
            print("No connectivity profile available.")

    print("\n=== SIMULATION EXAMPLE ===")
    simulated = model.simulate(
        genetic_liability=0.75,
        home_environment_quality=0.45,
        educational_support=0.50,
        nutrition_health_support=0.60,
        social_support=0.48,
        stress_burden=0.62,
        comorbid_mood_anxiety_load=0.55,
    )
    for name, series in simulated.items():
        print(f"\n--- {name.upper()} ---")
        print(series.sort_values(ascending=False).round(3).to_string())

    # Example coordinate assignment in MNI152:
    # print(model.assign_mni_point((-4, 34, 26)).head())
