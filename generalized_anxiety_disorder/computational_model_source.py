from __future__ import annotations

"""
Generalized anxiety disorder (GAD) siibra scaffold.

This script turns a short chapter on the biological basis of generalized
anxiety disorder into an atlas-grounded, multimodal research scaffold.
It is intentionally conservative and only models mechanisms clearly named
or strongly implied in the source chapter:

- familial / genetic vulnerability,
- GABAergic inhibitory deficit,
- stress-response / neuroendocrine dysregulation,
- weakened medial / orbitofrontal prefrontal control,
- impaired PFC-amygdala coupling,
- excessive and difficult-to-control worry.

The resulting model is a transparent research scaffold, not a validated
biophysical disease model and not a clinical tool.
"""

import inspect
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except Exception as exc:  # pragma: no cover - dependency may be absent locally
    siibra = None  # type: ignore[assignment]
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised only where siibra is installed
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_GAD_GENE_PANEL = [
    "GAD1",
    "GAD2",
    "GABRA1",
    "GABRA2",
    "SLC6A4",
    "BDNF",
    "FKBP5",
    "CRHR1",
    "NR3C1",
    "COMT",
]


class GeneralizedAnxietyDisorderModel:
    """
    Atlas-grounded GAD scaffold derived from a brief chapter summary.

    Notes
    -----
    - High values in the simulator represent greater dysregulation burden,
      except for explicitly protective inputs such as `treatment_support`
      and `benzodiazepine_like_gaba_support`.
    - Region nodes are anatomical anchors or clearly labeled proxies.
      The medial prefrontal control node is left as a proxy because the
      source chapter names the medial prefrontal cortex at a systems level
      without committing to one cytoarchitectonic parcel.
    - Empty receptor, gene, or connectivity tables are acceptable and do
      not indicate a script failure.
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
                "siibra is required to instantiate GeneralizedAnxietyDisorderModel. "
                "Install siibra and try again."
            ) from _SIIBRA_IMPORT_ERROR

        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

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

        # Conservative chapter extraction: a small set of clearly described
        # regions plus one explicit proxy for medial PFC control.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "medial_pfc_proxy": [
                "anterior cingulate cortex left",
                "medial prefrontal cortex left",
                "Area p32 left",
                "Area s32 left",
                "Area 24 left",
                "Area 33 left",
                "medial prefrontal",
                "anterior cingulate",
            ],
            "orbitofrontal_cortex": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Area Fo2 (OFC) left",
                "orbitofrontal cortex left",
                "orbitofrontal",
                "Fo4",
                "Fo3",
                "Fo2",
            ],
        }
        self.region_node_notes: Dict[str, str] = {
            "amygdala": "Threat and fear-generating limbic node emphasized by the chapter.",
            "medial_pfc_proxy": (
                "Proxy for medial prefrontal top-down control over limbic threat signaling."
            ),
            "orbitofrontal_cortex": (
                "Orbitofrontal appraisal / regulatory node named explicitly in the chapter."
            ),
        }
        self.region_is_proxy: Dict[str, bool] = {
            "amygdala": False,
            "medial_pfc_proxy": True,
            "orbitofrontal_cortex": False,
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Familial and heritable liability shaping vulnerability to anxiety circuitry dysregulation."
            ),
            "environmental_trigger_load": (
                "Environmental and life-experience triggers acting on biological vulnerability."
            ),
            "chronic_uncontrollable_stress": (
                "Persistent stress burden that can weaken prefrontal control and amplify worry."
            ),
            "treatment_support": (
                "Protective support from treatment, structure, and regulation strategies."
            ),
            "benzodiazepine_like_gaba_support": (
                "Short-term GABA-enhancing pharmacologic support used as a mechanistic proxy."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "gabaergic_deficit": (
                "Reduced inhibitory GABAergic tone producing disinhibition of anxiety-generating circuits."
            ),
            "stress_response_dysregulation": (
                "Stress-response / neuroendocrine dysregulation that sustains anxious arousal."
            ),
            "pfc_amygdala_disconnectivity": (
                "Weakened functional coupling between prefrontal regulatory systems and the amygdala."
            ),
            "pfc_top_down_control_failure": (
                "Impaired executive control over threat appraisal and worry regulation."
            ),
            "amygdala_threat_bias": (
                "Excessive limbic threat signaling that promotes fear and worry."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "excessive_worry": "Excessive worry as a core downstream expression of the circuit.",
            "difficulty_controlling_worry": (
                "Difficulty controlling worry, matching the chapter's core clinical feature."
            ),
            "persistent_anxiety": "Sustained anxiety and tension arising from chronic circuit dysregulation.",
            "worry_stress_cycle_pressure": (
                "Self-reinforcing pressure whereby worry intensifies stress and stress further weakens regulation."
            ),
        }

        # Directional edge table extracted from the chapter. Cycles are allowed
        # in the conceptual graph even though the simulator remains acyclic.
        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "gabaergic_deficit",
                "relation": "contributes to inherited vulnerability affecting inhibitory tone",
                "gad_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "pfc_amygdala_disconnectivity",
                "relation": "contributes to biologic vulnerability in frontolimbic circuitry",
                "gad_change": "increased",
            },
            {
                "source": "environmental_trigger_load",
                "target": "stress_response_dysregulation",
                "relation": "environmental triggers act on biologic vulnerability and amplify stress signaling",
                "gad_change": "increased",
            },
            {
                "source": "chronic_uncontrollable_stress",
                "target": "stress_response_dysregulation",
                "relation": "persistent stress worsens neuroendocrine and stress-response dysregulation",
                "gad_change": "increased",
            },
            {
                "source": "chronic_uncontrollable_stress",
                "target": "pfc_top_down_control_failure",
                "relation": "uncontrollable stress weakens prefrontal control over worry",
                "gad_change": "increased",
            },
            {
                "source": "treatment_support",
                "target": "stress_response_dysregulation",
                "relation": "supportive treatment can reduce stress-system burden",
                "gad_change": "decreased",
            },
            {
                "source": "benzodiazepine_like_gaba_support",
                "target": "gabaergic_deficit",
                "relation": "GABA-A enhancement partially offsets deficient inhibitory tone",
                "gad_change": "decreased",
            },
            {
                "source": "gabaergic_deficit",
                "target": "amygdala_threat_bias",
                "relation": "reduced inhibition disinhibits fear and worry-generating circuits",
                "gad_change": "increased",
            },
            {
                "source": "stress_response_dysregulation",
                "target": "pfc_amygdala_disconnectivity",
                "relation": "stress burden weakens frontolimbic communication",
                "gad_change": "increased",
            },
            {
                "source": "pfc_amygdala_disconnectivity",
                "target": "pfc_top_down_control_failure",
                "relation": "poor frontolimbic coupling undermines executive regulation",
                "gad_change": "increased",
            },
            {
                "source": "pfc_top_down_control_failure",
                "target": "medial_pfc_proxy",
                "relation": "top-down control failure is expressed as medial PFC dysregulation",
                "gad_change": "increased",
            },
            {
                "source": "pfc_top_down_control_failure",
                "target": "orbitofrontal_cortex",
                "relation": "top-down control failure is expressed as orbitofrontal regulatory dysfunction",
                "gad_change": "increased",
            },
            {
                "source": "gabaergic_deficit",
                "target": "amygdala",
                "relation": "loss of inhibitory braking increases limbic excitability",
                "gad_change": "increased",
            },
            {
                "source": "amygdala_threat_bias",
                "target": "amygdala",
                "relation": "threat bias manifests as amygdala hyperreactivity",
                "gad_change": "increased",
            },
            {
                "source": "pfc_amygdala_disconnectivity",
                "target": "amygdala",
                "relation": "weaker prefrontal damping leaves amygdala signals less constrained",
                "gad_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "excessive_worry",
                "relation": "hyperreactive fear circuitry promotes excessive worry",
                "gad_change": "increased",
            },
            {
                "source": "medial_pfc_proxy",
                "target": "difficulty_controlling_worry",
                "relation": "medial prefrontal dysregulation impairs control of worry",
                "gad_change": "increased",
            },
            {
                "source": "orbitofrontal_cortex",
                "target": "excessive_worry",
                "relation": "orbitofrontal dysregulation impairs realistic threat appraisal",
                "gad_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "persistent_anxiety",
                "relation": "limbic hyperreactivity sustains anxious distress",
                "gad_change": "increased",
            },
            {
                "source": "stress_response_dysregulation",
                "target": "persistent_anxiety",
                "relation": "stress-system dysregulation supports chronic anxiety",
                "gad_change": "increased",
            },
            {
                "source": "excessive_worry",
                "target": "worry_stress_cycle_pressure",
                "relation": "worry feeds the stress-worsening cycle",
                "gad_change": "increased",
            },
            {
                "source": "chronic_uncontrollable_stress",
                "target": "worry_stress_cycle_pressure",
                "relation": "stress directly strengthens the worry-stress cycle",
                "gad_change": "increased",
            },
            {
                "source": "worry_stress_cycle_pressure",
                "target": "chronic_uncontrollable_stress",
                "relation": "vicious cycle: worry further intensifies experienced stress burden",
                "gad_change": "increased",
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
    def _normalize_string(text: Any) -> str:
        return str(text).strip().lower()

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return round(sum(float(v) for v in values) / len(values), 4)

    def _modality_candidates(self, kind: str) -> List[Any]:
        cands: List[Any] = []
        try:  # pragma: no cover - depends on siibra version
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

    def _try_find_regions(self, query: str) -> List[Any]:
        # Compatibility helper across atlas / parcellation APIs.
        finders = [
            getattr(self.atlas, "find_regions", None),
            getattr(self.parcellation, "find", None),
        ]
        for finder in finders:
            if finder is None:
                continue
            attempts = [
                {"query": query, "all_versions": False, "filter_children": False, "find_topmost": False},
                {"regionspec": query, "filter_children": False, "find_topmost": False},
                {"regionspec": query},
                {"query": query},
                {},
            ]
            for params in attempts:
                try:
                    if params:
                        found = finder(**params)
                    else:
                        found = finder(query)
                    if found:
                        return list(found)
                except Exception:
                    continue
        return []

    def _julich_matches(self, query: str) -> List[Any]:
        matches = self._try_find_regions(query)
        out: List[Any] = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if not parc_name or "julich" in str(parc_name).lower():
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._normalize_string(self._name_of(region))
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {"amygdala", "orbitofrontal cortex", "prefrontal cortex"} else 0
        proxy_penalty = 1 if "cingulate" in name and "area" not in name else 0
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

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        volume_mm3 = getattr(main, "volume", None)
        try:
            volume_mm3 = float(volume_mm3) if volume_mm3 is not None else None
        except Exception:
            volume_mm3 = None
        return centroid_xyz, volume_mm3

    @staticmethod
    def _dataframe_from_feature_data(data: Any) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, pd.Series):
            return data.reset_index(name="value")
        if isinstance(data, dict):
            return pd.DataFrame(data)
        return pd.DataFrame()

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        for feat in feats:
            df = self._dataframe_from_feature_data(getattr(feat, "data", None))
            if df.empty:
                continue
            df = df.reset_index(drop=True)
            if "index" in df.columns and "receptor" not in df.columns:
                df = df.rename(columns={"index": "receptor"})
            if df.columns.tolist() == ["index", "value"]:
                df = df.rename(columns={"index": "receptor"})
            return df
        return pd.DataFrame()

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        query_attempts = [
            {"gene": list(genes)},
            {"genes": list(genes)},
            {},
        ]
        feats: List[Any] = []
        for qkwargs in query_attempts:
            feats = self._safe_features_any(region, self._modality_candidates("gene"), **qkwargs)
            if feats:
                break
        for feat in feats:
            df = self._dataframe_from_feature_data(getattr(feat, "data", None))
            if df.empty:
                continue
            lower_cols = {str(c).lower(): c for c in df.columns}
            if "gene" in lower_cols:
                gene_col = lower_cols["gene"]
                df = df[df[gene_col].astype(str).isin(list(genes))].copy()
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
        return pd.DataFrame()

    def _extract_matrix_from_connectivity_feature(self, feature: Any) -> pd.DataFrame:
        data = getattr(feature, "data", None)
        if isinstance(data, pd.DataFrame):
            return data.copy()
        try:
            first = feature[0]
            data = getattr(first, "data", None)
            if isinstance(data, pd.DataFrame):
                return data.copy()
        except Exception:
            pass
        try:
            for elem in feature:
                data = getattr(elem, "data", None)
                if isinstance(data, pd.DataFrame):
                    return data.copy()
        except Exception:
            pass
        return pd.DataFrame()

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix
        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        selected = None
        for feat in feats:
            if getattr(feat, "cohort", None) == self.connectivity_cohort:
                selected = feat
                break
        if selected is None:
            selected = feats[0]

        self._connectivity_matrix = self._extract_matrix_from_connectivity_feature(selected)
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._normalize_string(region.name)
        region_id = self._normalize_string(getattr(region, "identifier", ""))

        for label in labels:
            if label is region:
                return label

        exact = [label for label in labels if self._normalize_string(self._name_of(label)) == region_name]
        if exact:
            return exact[0]

        if region_id:
            exact_id = [
                label
                for label in labels
                if self._normalize_string(getattr(label, "identifier", "")) == region_id
            ]
            if exact_id:
                return exact_id[0]

        fuzzy = []
        for label in labels:
            label_name = self._normalize_string(self._name_of(label))
            if region_name in label_name or label_name in region_name:
                fuzzy.append(label)
        if fuzzy:
            return fuzzy[0]
        return None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        row_label = self._match_region_label(list(matrix.index), region)
        if row_label is not None:
            try:
                series = matrix.loc[row_label]
            except Exception:
                series = None
        else:
            series = None

        if series is None:
            col_label = self._match_region_label(list(matrix.columns), region)
            if col_label is None:
                return pd.DataFrame()
            try:
                series = matrix[col_label]
            except Exception:
                return pd.DataFrame()

        try:
            df = series.sort_values(ascending=False).reset_index()
        except Exception:
            return pd.DataFrame()
        df.columns = ["connected_region", "value"]
        df["connected_region"] = df["connected_region"].map(self._name_of)
        df = df[df["connected_region"] != region.name]
        df = df[pd.notnull(df["value"])]
        return df.head(max_rows).reset_index(drop=True)

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        label_map: Dict[str, Any] = {}
        for key, region in self.region_objects.items():
            label = self._match_region_label(list(matrix.index), region)
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
            if label is not None:
                label_map[key] = label

        if len(label_map) < 2:
            return pd.DataFrame()

        ordered_keys = list(label_map.keys())
        ordered_labels = [label_map[k] for k in ordered_keys]
        try:
            sub = matrix.loc[ordered_labels, ordered_labels].copy()
        except Exception:
            try:
                sub = matrix.reindex(index=ordered_labels, columns=ordered_labels).copy()
            except Exception:
                return pd.DataFrame()
        sub.index = ordered_keys
        sub.columns = ordered_keys
        return sub

    def _get_map(self, maptype: str = "labelled", space_spec: Optional[str] = None) -> Any:
        target_space = space_spec or self.space_spec
        try:
            with siibra.QUIET:
                return siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=target_space,
                    maptype=maptype,
                )
        except Exception:
            with siibra.QUIET:
                return self.atlas.get_map(
                    space=self.space if target_space == self.space_spec else target_space,
                    parcellation=self.parcellation,
                    maptype=maptype,
                )

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GAD_GENE_PANEL,
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
                    "is_proxy": False,
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
                        "description": self.region_node_notes.get(
                            key, "Atlas-backed or proxy circuit node unresolved in this environment."
                        ),
                        "atlas_region": None,
                        "region_identifier": None,
                        "centroid_mni": None,
                        "volume_mm3": None,
                        "feature_summary": "unresolved",
                        "is_proxy": self.region_is_proxy.get(key, False),
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
                    "description": self.region_node_notes.get(key, "Atlas-backed circuit node."),
                    "atlas_region": region.name,
                    "region_identifier": getattr(region, "identifier", None),
                    "centroid_mni": centroid_mni,
                    "volume_mm3": volume_mm3,
                    "feature_summary": (
                        f"receptors={'yes' if not receptor_df.empty else 'no'}; "
                        f"genes={'yes' if not gene_df.empty else 'no'}; "
                        f"connectivity={'yes' if not conn_df.empty else 'no'}"
                    ),
                    "is_proxy": self.region_is_proxy.get(key, False),
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
        genetic_vulnerability: float = 0.5,
        environmental_trigger_load: float = 0.4,
        chronic_uncontrollable_stress: float = 0.5,
        treatment_support: float = 0.3,
        benzodiazepine_like_gaba_support: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized mechanistic simulation.

        All arguments are clipped to [0, 1]. Larger values represent greater
        burden except for the protective support variables.
        """

        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "environmental_trigger_load": self._clip01(environmental_trigger_load),
                "chronic_uncontrollable_stress": self._clip01(chronic_uncontrollable_stress),
                "treatment_support": self._clip01(treatment_support),
                "benzodiazepine_like_gaba_support": self._clip01(
                    benzodiazepine_like_gaba_support
                ),
            },
            dtype=float,
        )

        latents = pd.Series(dtype=float)
        latents["gabaergic_deficit"] = self._clip01(
            0.5 * inputs["genetic_vulnerability"]
            + 0.2 * inputs["environmental_trigger_load"]
            + 0.2 * inputs["chronic_uncontrollable_stress"]
            - 0.35 * inputs["benzodiazepine_like_gaba_support"]
            - 0.1 * inputs["treatment_support"]
        )
        latents["stress_response_dysregulation"] = self._clip01(
            0.5 * inputs["chronic_uncontrollable_stress"]
            + 0.2 * inputs["environmental_trigger_load"]
            + 0.15 * inputs["genetic_vulnerability"]
            - 0.25 * inputs["treatment_support"]
        )
        latents["pfc_amygdala_disconnectivity"] = self._clip01(
            0.3 * inputs["genetic_vulnerability"]
            + 0.35 * latents["stress_response_dysregulation"]
            + 0.2 * inputs["chronic_uncontrollable_stress"]
            + 0.1 * inputs["environmental_trigger_load"]
            - 0.2 * inputs["treatment_support"]
        )
        latents["pfc_top_down_control_failure"] = self._clip01(
            0.4 * inputs["chronic_uncontrollable_stress"]
            + 0.3 * latents["pfc_amygdala_disconnectivity"]
            + 0.15 * latents["stress_response_dysregulation"]
            + 0.05 * inputs["genetic_vulnerability"]
            - 0.25 * inputs["treatment_support"]
        )
        latents["amygdala_threat_bias"] = self._clip01(
            0.35 * latents["gabaergic_deficit"]
            + 0.25 * latents["stress_response_dysregulation"]
            + 0.25 * latents["pfc_top_down_control_failure"]
            + 0.15 * latents["pfc_amygdala_disconnectivity"]
        )

        regional_state = pd.Series(dtype=float)
        regional_state["amygdala"] = self._clip01(
            0.7 * latents["amygdala_threat_bias"]
            + 0.2 * latents["stress_response_dysregulation"]
            + 0.1 * latents["gabaergic_deficit"]
        )
        regional_state["medial_pfc_proxy"] = self._clip01(
            0.55 * latents["pfc_top_down_control_failure"]
            + 0.25 * latents["pfc_amygdala_disconnectivity"]
            + 0.2 * latents["stress_response_dysregulation"]
        )
        regional_state["orbitofrontal_cortex"] = self._clip01(
            0.45 * latents["pfc_top_down_control_failure"]
            + 0.35 * latents["pfc_amygdala_disconnectivity"]
            + 0.2 * latents["amygdala_threat_bias"]
        )

        symptoms = pd.Series(dtype=float)
        symptoms["excessive_worry"] = self._clip01(
            0.45 * regional_state["amygdala"]
            + 0.25 * regional_state["orbitofrontal_cortex"]
            + 0.2 * latents["stress_response_dysregulation"]
            + 0.1 * inputs["genetic_vulnerability"]
        )
        symptoms["difficulty_controlling_worry"] = self._clip01(
            0.5 * regional_state["medial_pfc_proxy"]
            + 0.2 * regional_state["orbitofrontal_cortex"]
            + 0.2 * latents["pfc_amygdala_disconnectivity"]
            + 0.1 * regional_state["amygdala"]
        )
        symptoms["persistent_anxiety"] = self._clip01(
            0.35 * regional_state["amygdala"]
            + 0.25 * symptoms["excessive_worry"]
            + 0.2 * latents["stress_response_dysregulation"]
            + 0.2 * latents["gabaergic_deficit"]
        )
        symptoms["worry_stress_cycle_pressure"] = self._clip01(
            0.45 * inputs["chronic_uncontrollable_stress"]
            + 0.3 * symptoms["excessive_worry"]
            + 0.25 * symptoms["difficulty_controlling_worry"]
        )

        phenotypes = pd.Series(
            {
                "gad_core_profile": self._clip01(
                    self._mean(
                        [
                            symptoms["excessive_worry"],
                            symptoms["difficulty_controlling_worry"],
                            symptoms["persistent_anxiety"],
                        ]
                    )
                ),
                "stress_amplified_profile": self._clip01(
                    self._mean(
                        [
                            symptoms["worry_stress_cycle_pressure"],
                            symptoms["persistent_anxiety"],
                            latents["stress_response_dysregulation"],
                        ]
                    )
                ),
                "frontolimbic_dysregulation_profile": self._clip01(
                    self._mean(
                        [
                            regional_state["amygdala"],
                            regional_state["medial_pfc_proxy"],
                            regional_state["orbitofrontal_cortex"],
                        ]
                    )
                ),
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
        if len(tuple(xyz)) != 3:
            raise ValueError("xyz must be a 3-element sequence in MNI152 space.")
        if self._pmap is None:
            with siibra.QUIET:
                self._pmap = self._get_map(maptype="statistical", space_spec=self.assignment_space)

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(
        self,
        node_key: str,
        maptype: str = "labelled",
        threshold: float = 0.2,
        fetch: bool = True,
    ) -> Any:
        region = self.region_objects.get(node_key)
        if region is None:
            raise KeyError(f"Region node '{node_key}' is unresolved or build() has not been run.")

        # Newer siibra versions expose get_regional_mask(), older ones expose
        # fetch_regional_map(). We try both patterns.
        if hasattr(region, "get_regional_mask"):
            try:
                mask = region.get_regional_mask(self.space, maptype=maptype)
                return mask.fetch() if fetch and hasattr(mask, "fetch") else mask
            except Exception:
                pass

        if hasattr(region, "fetch_regional_map"):
            try:
                mask = region.fetch_regional_map(self.space, maptype=maptype, threshold=threshold)
            except TypeError:
                try:
                    mask = region.fetch_regional_map(self.space, maptype=maptype)
                except Exception as exc:
                    raise RuntimeError(f"Could not fetch mask for node '{node_key}'.") from exc
            return mask.fetch() if fetch and hasattr(mask, "fetch") else mask

        raise RuntimeError(
            f"Region '{region.name}' does not expose a compatible regional mask method in this siibra version."
        )


if __name__ == "__main__":
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 140)

    if siibra is None:  # pragma: no cover - environment dependent
        print(
            "siibra is not installed in this environment, so the atlas queries cannot run here. "
            "The script itself is ready to use once siibra is available."
        )
        raise SystemExit(0)

    model = GeneralizedAnxietyDisorderModel()
    built = model.build()

    print("\n=== Nodes ===")
    print(built["nodes"].head(20).to_string(index=False))

    print("\n=== Edges ===")
    print(built["edges"].to_string(index=False))

    print("\n=== Resolved regions ===")
    for key, region in built["regions"].items():
        print(f"- {key}: {region.name}")

    print("\n=== Receptor fingerprints (head) ===")
    for key, df in built["receptors"].items():
        print(f"\n[{key}]")
        print(df.head().to_string(index=False) if not df.empty else "<no receptor fingerprint>")

    print("\n=== Gene summaries (head) ===")
    for key, df in built["genes"].items():
        print(f"\n[{key}]")
        print(df.head().to_string(index=False) if not df.empty else "<no gene summary>")

    print("\n=== Connectivity profiles (head) ===")
    for key, df in built["connectivity_profiles"].items():
        print(f"\n[{key}]")
        print(df.head().to_string(index=False) if not df.empty else "<no connectivity profile>")

    print("\n=== Circuit connectivity ===")
    circuit_df = built["circuit_connectivity"]
    print(circuit_df.to_string() if not circuit_df.empty else "<no pairwise circuit matrix>")

    sim = model.simulate(
        genetic_vulnerability=0.75,
        environmental_trigger_load=0.55,
        chronic_uncontrollable_stress=0.8,
        treatment_support=0.3,
        benzodiazepine_like_gaba_support=0.15,
    )
    print("\n=== Simulation: inputs ===")
    print(sim["inputs"].to_string())
    print("\n=== Simulation: latents ===")
    print(sim["latents"].to_string())
    print("\n=== Simulation: regional state ===")
    print(sim["regional_state"].to_string())
    print("\n=== Simulation: symptoms ===")
    print(sim["symptoms"].to_string())
    print("\n=== Simulation: phenotypes ===")
    print(sim["phenotypes"].to_string())

    # Optional examples once siibra is available:
    # print(model.suggest_regions("amygdala").head())
    # print(model.assign_mni_point((-6, 24, -10)).head())
    # mask_img = model.region_mask("amygdala")
