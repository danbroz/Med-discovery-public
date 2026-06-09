from __future__ import annotations

"""
Restless legs syndrome (RLS, Willis-Ekbom disease) siibra scaffold.

This script translates a short chapter on the biology of restless legs
syndrome into an atlas-grounded, multimodal research scaffold.

The model is intentionally conservative and only encodes mechanisms clearly
named or strongly implied in the source chapter:

- central dopaminergic dysregulation with reduced striatal D2 binding,
- brain iron homeostasis failure centered on the substantia nigra and putamen,
- polygenic / developmental vulnerability with BTBD9-style iron and movement
  links plus broader axonal-guidance burden,
- modulatory contributions from opioid and glutamate systems,
- contributions from both central and peripheral sensorimotor systems,
- disruption of basal ganglia and cortico-striatal-thalamocortical loops,
- downstream urge to move, sensorimotor restlessness, periodic limb
  movements, and sleep-related motor disturbance.

This is a mechanistic research scaffold, not a validated disease model and
not a tool for diagnosis or treatment.
"""

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


DEFAULT_RLS_GENE_PANEL = [
    "BTBD9",
    "MEIS1",
    "MAP2K5",
    "PTPRD",
    "TH",
    "SLC6A3",
    "DRD2",
    "TF",
    "FTH1",
    "FTL",
]


class RestlessLegsSyndromeModel:
    """
    Atlas-grounded scaffold for restless legs syndrome.

    Notes
    -----
    - Larger simulator values represent greater dysregulation or symptom
      burden.
    - Several anatomical nodes are explicit proxies because the chapter names
      systems or nuclei at a level that may not cleanly map to one Julich
      parcel in every siibra environment.
    - Empty receptor, gene, or connectivity tables are acceptable and do not
      imply a script failure.
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
                "siibra is required to instantiate RestlessLegsSyndromeModel. "
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

        # Conservative region choices: explicitly named structures plus clear
        # proxy nodes for chapter-level loop descriptions.
        self.region_candidates: Dict[str, List[str]] = {
            "substantia_nigra_proxy": [
                "Substantia nigra pars compacta left",
                "substantia nigra pars compacta",
                "Substantia nigra left",
                "substantia nigra left",
                "Substantia nigra",
                "substantia nigra",
            ],
            "putamen": [
                "Putamen left",
                "putamen left",
                "Putamen",
                "putamen",
            ],
            "caudate": [
                "Caudate nucleus left",
                "caudate nucleus left",
                "Caudate left",
                "caudate left",
                "Caudate",
                "caudate",
                "striatum left",
                "striatum",
            ],
            "thalamus_proxy": [
                "Thalamus left",
                "thalamus left",
                "Thalamus",
                "thalamus",
            ],
            "sensorimotor_cortex_proxy": [
                "Area 4a left",
                "Area 4p left",
                "Area 3b left",
                "Area 1 left",
                "precentral gyrus left",
                "postcentral gyrus left",
                "primary motor cortex left",
                "sensorimotor cortex left",
                "sensorimotor cortex",
            ],
        }
        self.region_node_notes: Dict[str, str] = {
            "substantia_nigra_proxy": (
                "Proxy for the substantia nigra, the chapter's main iron-deficit and dopaminergic anchor."
            ),
            "putamen": (
                "Putamen node explicitly named as a basal-ganglia site of reduced iron burden."
            ),
            "caudate": (
                "Caudate / striatal node explicitly named as part of the basal-ganglia sensorimotor circuit."
            ),
            "thalamus_proxy": (
                "Proxy for the thalamic relay within cortico-striatal-thalamocortical loops."
            ),
            "sensorimotor_cortex_proxy": (
                "Proxy for the cortical arm of the cortico-striatal-thalamocortical loop implied by the chapter."
            ),
        }
        self.region_is_proxy: Dict[str, bool] = {
            "substantia_nigra_proxy": True,
            "putamen": False,
            "caudate": False,
            "thalamus_proxy": True,
            "sensorimotor_cortex_proxy": True,
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Polygenic risk influencing iron homeostasis, dopamine signaling, limb development, and axonal guidance."
            ),
            "iron_deficiency_state": (
                "Iron-deficiency burden or low-ferritin state acting as a principal biological driver."
            ),
            "peripheral_sensorimotor_burden": (
                "Peripheral nervous system contribution to the sensorimotor discomfort and movement drive."
            ),
            "dopaminergic_treatment_support": (
                "Protective symptomatic support from dopaminergic treatment."
            ),
            "iron_repletion_support": (
                "Protective support from restoring iron availability and homeostasis."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "brain_iron_homeostasis_failure": (
                "Localized CNS iron dysregulation centered on the substantia nigra and putamen."
            ),
            "developmental_axonal_guidance_vulnerability": (
                "Developmental and axonal-guidance burden linked to the sensorimotor architecture of the disorder."
            ),
            "nigrostriatal_dopamine_dysregulation": (
                "Impaired dopaminergic modulation and synthesis within nigrostriatal pathways."
            ),
            "striatal_d2_receptor_dysregulation": (
                "Post-synaptic striatal D2 receptor dysfunction suggested by PET findings."
            ),
            "opioid_glutamate_modulatory_burden": (
                "Additional neurotransmitter-system involvement beyond dopamine and iron."
            ),
            "direct_indirect_pathway_imbalance": (
                "Imbalance between the pro-kinetic direct and anti-kinetic indirect basal-ganglia pathways."
            ),
            "cstc_loop_dysregulation": (
                "Disrupted cortico-striatal-thalamocortical integration underlying motor and sensory control failure."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "sensorimotor_restlessness": (
                "Aversive internal sensorimotor restlessness or discomfort."
            ),
            "urge_to_move": (
                "The core urge to move the limbs in response to abnormal sensorimotor signaling."
            ),
            "periodic_limb_movements": (
                "Involuntary periodic limb movements as a motor manifestation of loop dysregulation."
            ),
            "sleep_related_motor_disturbance": (
                "Nocturnal movement burden and sleep disruption linked to RLS motor activity."
            ),
        }

        # Conceptual graph extracted from the chapter. The simulator below
        # remains acyclic even though the conceptual graph is richer.
        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "developmental_axonal_guidance_vulnerability",
                "relation": "genetic risk includes variants tied to fetal limb development and axonal guidance",
                "rls_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "brain_iron_homeostasis_failure",
                "relation": "genetic burden may impair proteins regulating iron uptake, storage, and transport",
                "rls_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "nigrostriatal_dopamine_dysregulation",
                "relation": "genetic burden may influence dopamine synthesis and receptor function",
                "rls_change": "increased",
            },
            {
                "source": "iron_deficiency_state",
                "target": "brain_iron_homeostasis_failure",
                "relation": "iron-deficiency states anchor the modern biological model of RLS",
                "rls_change": "increased",
            },
            {
                "source": "iron_repletion_support",
                "target": "brain_iron_homeostasis_failure",
                "relation": "improved iron availability can reduce iron-homeostasis burden",
                "rls_change": "decreased",
            },
            {
                "source": "dopaminergic_treatment_support",
                "target": "nigrostriatal_dopamine_dysregulation",
                "relation": "dopaminergic therapy can compensate for pathway dysfunction",
                "rls_change": "decreased",
            },
            {
                "source": "peripheral_sensorimotor_burden",
                "target": "cstc_loop_dysregulation",
                "relation": "peripheral contributions add to the distributed sensorimotor burden",
                "rls_change": "increased",
            },
            {
                "source": "peripheral_sensorimotor_burden",
                "target": "sensorimotor_restlessness",
                "relation": "peripheral sensory burden can directly intensify restlessness",
                "rls_change": "increased",
            },
            {
                "source": "brain_iron_homeostasis_failure",
                "target": "substantia_nigra_proxy",
                "relation": "MRI and neuropathology localize reduced brain iron to the substantia nigra",
                "rls_change": "increased",
            },
            {
                "source": "brain_iron_homeostasis_failure",
                "target": "putamen",
                "relation": "MRI studies localize reduced brain iron to the putamen",
                "rls_change": "increased",
            },
            {
                "source": "brain_iron_homeostasis_failure",
                "target": "nigrostriatal_dopamine_dysregulation",
                "relation": "localized iron deficit can impair tyrosine-hydroxylase-dependent dopamine production",
                "rls_change": "increased",
            },
            {
                "source": "brain_iron_homeostasis_failure",
                "target": "striatal_d2_receptor_dysregulation",
                "relation": "iron-dopamine coupling failure can worsen post-synaptic striatal receptor dysfunction",
                "rls_change": "increased",
            },
            {
                "source": "developmental_axonal_guidance_vulnerability",
                "target": "cstc_loop_dysregulation",
                "relation": "developmental circuitry vulnerability can bias sensorimotor loop organization",
                "rls_change": "increased",
            },
            {
                "source": "nigrostriatal_dopamine_dysregulation",
                "target": "substantia_nigra_proxy",
                "relation": "dopaminergic dysfunction is expressed through the substantia-nigra-centered pathway",
                "rls_change": "increased",
            },
            {
                "source": "nigrostriatal_dopamine_dysregulation",
                "target": "direct_indirect_pathway_imbalance",
                "relation": "insufficient dopaminergic modulation disrupts the direct-indirect pathway balance",
                "rls_change": "increased",
            },
            {
                "source": "striatal_d2_receptor_dysregulation",
                "target": "putamen",
                "relation": "reduced striatal D2 binding anchors post-synaptic dysfunction in the putamen",
                "rls_change": "increased",
            },
            {
                "source": "striatal_d2_receptor_dysregulation",
                "target": "caudate",
                "relation": "reduced striatal D2 binding anchors post-synaptic dysfunction in the caudate",
                "rls_change": "increased",
            },
            {
                "source": "striatal_d2_receptor_dysregulation",
                "target": "direct_indirect_pathway_imbalance",
                "relation": "D2-system dysregulation disturbs indirect-pathway modulation",
                "rls_change": "increased",
            },
            {
                "source": "opioid_glutamate_modulatory_burden",
                "target": "cstc_loop_dysregulation",
                "relation": "non-dopaminergic transmitter burden further destabilizes the sensorimotor network",
                "rls_change": "increased",
            },
            {
                "source": "direct_indirect_pathway_imbalance",
                "target": "thalamus_proxy",
                "relation": "abnormal basal-ganglia output propagates through the thalamic relay",
                "rls_change": "increased",
            },
            {
                "source": "direct_indirect_pathway_imbalance",
                "target": "cstc_loop_dysregulation",
                "relation": "basal-ganglia pathway imbalance destabilizes the broader cortico-striatal-thalamocortical loop",
                "rls_change": "increased",
            },
            {
                "source": "cstc_loop_dysregulation",
                "target": "sensorimotor_cortex_proxy",
                "relation": "loop disruption burdens the cortical motor-planning and sensorimotor arm",
                "rls_change": "increased",
            },
            {
                "source": "putamen",
                "target": "sensorimotor_restlessness",
                "relation": "putaminal dysfunction contributes to abnormal sensorimotor experience",
                "rls_change": "increased",
            },
            {
                "source": "substantia_nigra_proxy",
                "target": "sensorimotor_restlessness",
                "relation": "nigrostriatal dysfunction contributes to sensorimotor discomfort and drive",
                "rls_change": "increased",
            },
            {
                "source": "sensorimotor_cortex_proxy",
                "target": "urge_to_move",
                "relation": "cortical motor-planning burden contributes to the urge to move",
                "rls_change": "increased",
            },
            {
                "source": "direct_indirect_pathway_imbalance",
                "target": "urge_to_move",
                "relation": "pathway imbalance can produce inappropriate motor output signals and the urge to move",
                "rls_change": "increased",
            },
            {
                "source": "cstc_loop_dysregulation",
                "target": "urge_to_move",
                "relation": "sensorimotor loop disruption drives the core urge-to-move symptom",
                "rls_change": "increased",
            },
            {
                "source": "cstc_loop_dysregulation",
                "target": "periodic_limb_movements",
                "relation": "sensorimotor loop disruption contributes to involuntary periodic limb movements",
                "rls_change": "increased",
            },
            {
                "source": "urge_to_move",
                "target": "periodic_limb_movements",
                "relation": "stronger movement drive increases motor manifestation burden",
                "rls_change": "increased",
            },
            {
                "source": "periodic_limb_movements",
                "target": "sleep_related_motor_disturbance",
                "relation": "periodic nocturnal movements disrupt sleep and produce nighttime motor burden",
                "rls_change": "increased",
            },
            {
                "source": "urge_to_move",
                "target": "sleep_related_motor_disturbance",
                "relation": "persistent movement urge increases nocturnal symptom burden",
                "rls_change": "increased",
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
        finders = [
            getattr(self.atlas, "find_regions", None),
            getattr(self.parcellation, "find", None),
        ]
        for finder in finders:
            if finder is None:
                continue
            attempts = [
                {
                    "query": query,
                    "all_versions": False,
                    "filter_children": False,
                    "find_topmost": False,
                },
                {
                    "regionspec": query,
                    "filter_children": False,
                    "find_topmost": False,
                },
                {"regionspec": query},
                {"query": query},
                {},
            ]
            for params in attempts:
                try:
                    found = finder(**params) if params else finder(query)
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
        generic_penalty = 1 if name in {"putamen", "caudate", "thalamus", "sensorimotor cortex"} else 0
        lobar_penalty = 1 if "cortex" in name and "area" not in name else 0
        return (left_bonus, right_penalty, generic_penalty, lobar_penalty)

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
            try:
                fallback_space = (
                    self.atlas.get_space(target_space)
                    if hasattr(self.atlas, "get_space")
                    else self.atlas.spaces.get(target_space)
                )
            except Exception:
                fallback_space = self.space if target_space == self.space_spec else target_space
            with siibra.QUIET:
                return self.atlas.get_map(
                    space=fallback_space,
                    parcellation=self.parcellation,
                    maptype=maptype,
                )

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_RLS_GENE_PANEL,
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
        iron_deficiency_state: float = 0.6,
        peripheral_sensorimotor_burden: float = 0.4,
        dopaminergic_treatment_support: float = 0.3,
        iron_repletion_support: float = 0.3,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized mechanistic simulation.

        All arguments are clipped to [0, 1]. Larger values represent greater
        risk or burden except for explicitly protective support variables.
        """

        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "iron_deficiency_state": self._clip01(iron_deficiency_state),
                "peripheral_sensorimotor_burden": self._clip01(peripheral_sensorimotor_burden),
                "dopaminergic_treatment_support": self._clip01(dopaminergic_treatment_support),
                "iron_repletion_support": self._clip01(iron_repletion_support),
            },
            dtype=float,
        )

        latents = pd.Series(dtype=float)
        latents["brain_iron_homeostasis_failure"] = self._clip01(
            0.55 * inputs["iron_deficiency_state"]
            + 0.25 * inputs["genetic_vulnerability"]
            + 0.1 * inputs["peripheral_sensorimotor_burden"]
            - 0.35 * inputs["iron_repletion_support"]
        )
        latents["developmental_axonal_guidance_vulnerability"] = self._clip01(
            0.5 * inputs["genetic_vulnerability"]
            + 0.2 * inputs["iron_deficiency_state"]
            + 0.2 * inputs["peripheral_sensorimotor_burden"]
            - 0.1 * inputs["iron_repletion_support"]
        )
        latents["nigrostriatal_dopamine_dysregulation"] = self._clip01(
            0.4 * latents["brain_iron_homeostasis_failure"]
            + 0.2 * inputs["genetic_vulnerability"]
            + 0.15 * latents["developmental_axonal_guidance_vulnerability"]
            + 0.15 * inputs["iron_deficiency_state"]
            + 0.1 * inputs["peripheral_sensorimotor_burden"]
            - 0.35 * inputs["dopaminergic_treatment_support"]
            - 0.15 * inputs["iron_repletion_support"]
        )
        latents["striatal_d2_receptor_dysregulation"] = self._clip01(
            0.35 * latents["nigrostriatal_dopamine_dysregulation"]
            + 0.3 * latents["brain_iron_homeostasis_failure"]
            + 0.2 * inputs["genetic_vulnerability"]
            + 0.1 * latents["developmental_axonal_guidance_vulnerability"]
            - 0.15 * inputs["dopaminergic_treatment_support"]
        )
        latents["opioid_glutamate_modulatory_burden"] = self._clip01(
            0.3 * inputs["peripheral_sensorimotor_burden"]
            + 0.25 * latents["nigrostriatal_dopamine_dysregulation"]
            + 0.2 * latents["brain_iron_homeostasis_failure"]
            + 0.15 * inputs["genetic_vulnerability"]
            + 0.1 * inputs["iron_deficiency_state"]
        )
        latents["direct_indirect_pathway_imbalance"] = self._clip01(
            0.45 * latents["nigrostriatal_dopamine_dysregulation"]
            + 0.35 * latents["striatal_d2_receptor_dysregulation"]
            + 0.1 * latents["developmental_axonal_guidance_vulnerability"]
            + 0.1 * latents["opioid_glutamate_modulatory_burden"]
            - 0.15 * inputs["dopaminergic_treatment_support"]
        )
        latents["cstc_loop_dysregulation"] = self._clip01(
            0.35 * latents["direct_indirect_pathway_imbalance"]
            + 0.2 * latents["striatal_d2_receptor_dysregulation"]
            + 0.15 * latents["developmental_axonal_guidance_vulnerability"]
            + 0.15 * inputs["peripheral_sensorimotor_burden"]
            + 0.15 * latents["opioid_glutamate_modulatory_burden"]
        )

        regional_state = pd.Series(dtype=float)
        regional_state["substantia_nigra_proxy"] = self._clip01(
            0.6 * latents["brain_iron_homeostasis_failure"]
            + 0.4 * latents["nigrostriatal_dopamine_dysregulation"]
        )
        regional_state["putamen"] = self._clip01(
            0.4 * latents["brain_iron_homeostasis_failure"]
            + 0.35 * latents["striatal_d2_receptor_dysregulation"]
            + 0.25 * latents["cstc_loop_dysregulation"]
        )
        regional_state["caudate"] = self._clip01(
            0.45 * latents["striatal_d2_receptor_dysregulation"]
            + 0.35 * latents["cstc_loop_dysregulation"]
            + 0.2 * latents["developmental_axonal_guidance_vulnerability"]
        )
        regional_state["thalamus_proxy"] = self._clip01(
            0.55 * latents["cstc_loop_dysregulation"]
            + 0.35 * latents["direct_indirect_pathway_imbalance"]
            + 0.1 * latents["opioid_glutamate_modulatory_burden"]
        )
        regional_state["sensorimotor_cortex_proxy"] = self._clip01(
            0.45 * latents["cstc_loop_dysregulation"]
            + 0.25 * regional_state["thalamus_proxy"]
            + 0.15 * regional_state["putamen"]
            + 0.15 * regional_state["caudate"]
        )

        symptoms = pd.Series(dtype=float)
        symptoms["sensorimotor_restlessness"] = self._clip01(
            0.3 * regional_state["putamen"]
            + 0.2 * regional_state["substantia_nigra_proxy"]
            + 0.2 * inputs["peripheral_sensorimotor_burden"]
            + 0.15 * latents["opioid_glutamate_modulatory_burden"]
            + 0.15 * regional_state["sensorimotor_cortex_proxy"]
        )
        symptoms["urge_to_move"] = self._clip01(
            0.3 * latents["direct_indirect_pathway_imbalance"]
            + 0.25 * latents["cstc_loop_dysregulation"]
            + 0.2 * symptoms["sensorimotor_restlessness"]
            + 0.15 * regional_state["sensorimotor_cortex_proxy"]
            + 0.1 * regional_state["thalamus_proxy"]
        )
        symptoms["periodic_limb_movements"] = self._clip01(
            0.35 * latents["direct_indirect_pathway_imbalance"]
            + 0.25 * latents["cstc_loop_dysregulation"]
            + 0.15 * regional_state["thalamus_proxy"]
            + 0.15 * symptoms["urge_to_move"]
            + 0.1 * latents["developmental_axonal_guidance_vulnerability"]
        )
        symptoms["sleep_related_motor_disturbance"] = self._clip01(
            0.45 * symptoms["periodic_limb_movements"]
            + 0.25 * symptoms["urge_to_move"]
            + 0.15 * symptoms["sensorimotor_restlessness"]
            + 0.15 * regional_state["thalamus_proxy"]
        )

        phenotypes = pd.Series(
            {
                "rls_core_sensorimotor_profile": self._clip01(
                    self._mean(
                        [
                            symptoms["sensorimotor_restlessness"],
                            symptoms["urge_to_move"],
                            symptoms["periodic_limb_movements"],
                        ]
                    )
                ),
                "iron_dopamine_profile": self._clip01(
                    self._mean(
                        [
                            latents["brain_iron_homeostasis_failure"],
                            latents["nigrostriatal_dopamine_dysregulation"],
                            regional_state["substantia_nigra_proxy"],
                        ]
                    )
                ),
                "sleep_movement_profile": self._clip01(
                    self._mean(
                        [
                            symptoms["periodic_limb_movements"],
                            symptoms["sleep_related_motor_disturbance"],
                            regional_state["thalamus_proxy"],
                        ]
                    )
                ),
                "developmental_circuit_profile": self._clip01(
                    self._mean(
                        [
                            latents["developmental_axonal_guidance_vulnerability"],
                            latents["cstc_loop_dysregulation"],
                            regional_state["sensorimotor_cortex_proxy"],
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

    model = RestlessLegsSyndromeModel()
    built = model.build()

    print("\n=== Nodes ===")
    print(built["nodes"].head(25).to_string(index=False))

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
        genetic_vulnerability=0.7,
        iron_deficiency_state=0.8,
        peripheral_sensorimotor_burden=0.55,
        dopaminergic_treatment_support=0.35,
        iron_repletion_support=0.25,
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
    # print(model.suggest_regions("substantia nigra").head())
    # print(model.suggest_regions("putamen").head())
    # print(model.assign_mni_point((-12, -16, -10)).head())
    # mask_img = model.region_mask("putamen")
