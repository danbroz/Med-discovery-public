from __future__ import annotations

"""
Major or mild neurocognitive disorder due to Parkinson's disease siibra scaffold.

This script translates a short chapter on the biological basis of cognitive and
neuropsychiatric complications in Parkinson's disease into an atlas-grounded,
multimodal research scaffold.

The model is intentionally conservative and only encodes mechanisms clearly
named or strongly implied in the source chapter:

- foundational dopaminergic neuron loss in the substantia nigra pars compacta,
- alpha-synuclein / Lewy body pathology with topographic spread,
- multisystem neurotransmitter involvement beyond dopamine, especially
  noradrenergic, cholinergic, and serotonergic pathways,
- degeneration extending from brainstem structures to limbic and cortical
  territories,
- frontal, temporal, parietal, and hippocampal atrophy burden,
- altered connectivity within basal ganglia-thalamo-cortical, default mode,
  frontoparietal attention, and fronto-limbic systems,
- downstream executive, attentional, mnemonic, and neuropsychiatric burden.

This is a mechanistic research scaffold, not a validated disease model and not
an instrument for diagnosis or treatment.
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


DEFAULT_PD_NCD_GENE_PANEL = [
    "SNCA",
    "LRRK2",
    "TH",
    "SLC6A3",
    "DBH",
    "SLC6A2",
    "CHAT",
    "ACHE",
    "TPH2",
    "SLC6A4",
]


class ParkinsonsDiseaseNeurocognitiveDisorderModel:
    """
    Atlas-grounded scaffold for major or mild neurocognitive disorder due to PD.

    Notes
    -----
    - Larger simulator values represent greater pathology or dysfunction burden.
    - Several anatomical nodes are explicit proxies because the chapter names
      systems or lobar territories rather than one precise cytoarchitectonic
      parcel.
    - Empty receptor, gene, or connectivity tables are acceptable and should
      not be interpreted as a script error.
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
                "siibra is required to instantiate "
                "ParkinsonsDiseaseNeurocognitiveDisorderModel. Install siibra and try again."
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

        # Conservative region choices: only chapter-named regions or explicit
        # proxies for named systems / lobar territories.
        self.region_candidates: Dict[str, List[str]] = {
            "substantia_nigra_pars_compacta_proxy": [
                "Substantia nigra pars compacta left",
                "substantia nigra pars compacta",
                "Substantia nigra left",
                "substantia nigra",
            ],
            "locus_coeruleus_proxy": [
                "Locus coeruleus left",
                "locus coeruleus left",
                "Locus coeruleus",
                "locus coeruleus",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "CA2 (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "frontal_association_cortex_proxy": [
                "Area 9 left",
                "Area 46 left",
                "Area 45 left",
                "middle frontal gyrus left",
                "superior frontal gyrus left",
                "frontal cortex",
                "frontal",
            ],
            "temporal_association_cortex_proxy": [
                "Area TE 3 left",
                "Area TE 2.1 left",
                "Area TE 1.0 left",
                "middle temporal gyrus left",
                "superior temporal gyrus left",
                "temporal cortex",
                "temporal",
            ],
            "parietal_association_cortex_proxy": [
                "Area PGp left",
                "Area PGa left",
                "Area PFm left",
                "inferior parietal lobule left",
                "superior parietal lobule left",
                "parietal cortex",
                "parietal",
            ],
            "basal_ganglia_proxy": [
                "caudate nucleus left",
                "putamen left",
                "globus pallidus left",
                "basal ganglia left",
                "basal ganglia",
            ],
        }
        self.region_node_notes: Dict[str, str] = {
            "substantia_nigra_pars_compacta_proxy": (
                "Proxy for the substantia nigra pars compacta, the chapter's foundational PD pathology node."
            ),
            "locus_coeruleus_proxy": (
                "Proxy for the locus coeruleus, the chapter's key noradrenergic brainstem node."
            ),
            "hippocampus": "Hippocampal atrophy anchor named explicitly in the chapter.",
            "frontal_association_cortex_proxy": (
                "Proxy for frontal cortical atrophy burden and executive-network involvement."
            ),
            "temporal_association_cortex_proxy": (
                "Proxy for temporal cortical atrophy burden linked to memory and network dysfunction."
            ),
            "parietal_association_cortex_proxy": (
                "Proxy for parietal cortical atrophy burden and frontoparietal attentional dysfunction."
            ),
            "basal_ganglia_proxy": (
                "Proxy for the basal ganglia side of the chapter's basal ganglia-thalamo-cortical loops."
            ),
        }
        self.region_is_proxy: Dict[str, bool] = {
            "substantia_nigra_pars_compacta_proxy": True,
            "locus_coeruleus_proxy": True,
            "hippocampus": False,
            "frontal_association_cortex_proxy": True,
            "temporal_association_cortex_proxy": True,
            "parietal_association_cortex_proxy": True,
            "basal_ganglia_proxy": True,
        }

        self.input_nodes: Dict[str, str] = {
            "alpha_synuclein_lewy_pathology": (
                "Accumulation of alpha-synuclein into Lewy bodies and Lewy neurites with phenotype-linked progression."
            ),
            "genetic_vulnerability": (
                "Familial or monogenic liability, especially involving SNCA and LRRK2-related risk."
            ),
            "dopaminergic_neuron_loss": (
                "Foundational loss of dopaminergic neurons in the substantia nigra pars compacta."
            ),
            "brainstem_to_cortex_spread": (
                "Topographic neurodegenerative spread from brainstem to limbic and cortical systems."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "widespread_neurodegenerative_spread": (
                "Distributed pathology extending beyond classic motor circuits into limbic and cortical systems."
            ),
            "noradrenergic_depletion": (
                "Loss of locus coeruleus norepinephrine signaling affecting arousal, mood, and executive function."
            ),
            "cholinergic_dysfunction": (
                "Non-dopaminergic cholinergic dysfunction contributing to cognitive decline."
            ),
            "serotonergic_dysfunction": (
                "Serotonergic pathway dysfunction contributing to neuropsychiatric burden."
            ),
            "cortical_atrophy_burden": (
                "Frontal, temporal, parietal, and hippocampal atrophy burden associated with cognitive decline."
            ),
            "basal_ganglia_thalamocortical_dysconnectivity": (
                "Disrupted basal ganglia-thalamo-cortical loop communication."
            ),
            "default_mode_network_dysconnectivity": (
                "Resting-state disorganization within default mode network-related systems."
            ),
            "frontoparietal_attention_dysconnectivity": (
                "Disrupted frontoparietal attentional control circuitry."
            ),
            "frontolimbic_dysconnectivity": (
                "Altered fronto-limbic connectivity linked to depression, anxiety, and apathy."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "executive_dysfunction": (
                "Executive impairment associated with frontal and frontoparietal dysfunction."
            ),
            "attention_arousal_instability": (
                "Instability of attention and arousal linked to noradrenergic and attentional-network failure."
            ),
            "memory_impairment": (
                "Mnemonic impairment associated with hippocampal and temporal-system burden."
            ),
            "neuropsychiatric_burden": (
                "Depression, anxiety, and apathy burden linked to fronto-limbic and monoaminergic changes."
            ),
            "postural_instability_related_burden": (
                "Postural instability burden associated with basal-ganglia and noradrenergic dysfunction."
            ),
            "global_cognitive_decline": (
                "Integrated cognitive decline across executive, attentional, and mnemonic domains."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "alpha_synuclein_lewy_pathology",
                "relation": "familial or monogenic risk can bias core synuclein-related pathology",
                "pd_ncd_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "dopaminergic_neuron_loss",
                "relation": "genetic liability contributes to PD-related neuronal vulnerability",
                "pd_ncd_change": "increased",
            },
            {
                "source": "alpha_synuclein_lewy_pathology",
                "target": "widespread_neurodegenerative_spread",
                "relation": "Lewy pathology progression tracks the evolving clinical phenotype",
                "pd_ncd_change": "increased",
            },
            {
                "source": "brainstem_to_cortex_spread",
                "target": "widespread_neurodegenerative_spread",
                "relation": "pathology extends anatomically from brainstem to limbic and cortical systems",
                "pd_ncd_change": "increased",
            },
            {
                "source": "widespread_neurodegenerative_spread",
                "target": "noradrenergic_depletion",
                "relation": "distributed degeneration affects the locus coeruleus norepinephrine system",
                "pd_ncd_change": "increased",
            },
            {
                "source": "widespread_neurodegenerative_spread",
                "target": "cholinergic_dysfunction",
                "relation": "distributed degeneration recruits cholinergic systems beyond dopamine",
                "pd_ncd_change": "increased",
            },
            {
                "source": "widespread_neurodegenerative_spread",
                "target": "serotonergic_dysfunction",
                "relation": "distributed degeneration recruits serotonergic systems beyond dopamine",
                "pd_ncd_change": "increased",
            },
            {
                "source": "dopaminergic_neuron_loss",
                "target": "substantia_nigra_pars_compacta_proxy",
                "relation": "foundational SNc pathology is expressed as substantia nigra dysfunction",
                "pd_ncd_change": "increased",
            },
            {
                "source": "noradrenergic_depletion",
                "target": "locus_coeruleus_proxy",
                "relation": "noradrenergic depletion is expressed in the locus coeruleus proxy node",
                "pd_ncd_change": "increased",
            },
            {
                "source": "alpha_synuclein_lewy_pathology",
                "target": "cortical_atrophy_burden",
                "relation": "progressive pathology contributes to cortical and hippocampal degeneration",
                "pd_ncd_change": "increased",
            },
            {
                "source": "widespread_neurodegenerative_spread",
                "target": "cortical_atrophy_burden",
                "relation": "disease spread drives frontal, temporal, parietal, and hippocampal burden",
                "pd_ncd_change": "increased",
            },
            {
                "source": "dopaminergic_neuron_loss",
                "target": "basal_ganglia_thalamocortical_dysconnectivity",
                "relation": "dopaminergic loss disrupts basal ganglia-thalamo-cortical circuitry",
                "pd_ncd_change": "increased",
            },
            {
                "source": "widespread_neurodegenerative_spread",
                "target": "default_mode_network_dysconnectivity",
                "relation": "distributed pathology yields widespread resting-state disorganization",
                "pd_ncd_change": "increased",
            },
            {
                "source": "widespread_neurodegenerative_spread",
                "target": "frontoparietal_attention_dysconnectivity",
                "relation": "distributed pathology disrupts attention networks",
                "pd_ncd_change": "increased",
            },
            {
                "source": "widespread_neurodegenerative_spread",
                "target": "frontolimbic_dysconnectivity",
                "relation": "distributed pathology disrupts fronto-limbic communication",
                "pd_ncd_change": "increased",
            },
            {
                "source": "locus_coeruleus_proxy",
                "target": "attention_arousal_instability",
                "relation": "LC dysfunction undermines attention and arousal regulation",
                "pd_ncd_change": "increased",
            },
            {
                "source": "locus_coeruleus_proxy",
                "target": "neuropsychiatric_burden",
                "relation": "LC degeneration is linked to depression, anxiety, and apathy",
                "pd_ncd_change": "increased",
            },
            {
                "source": "locus_coeruleus_proxy",
                "target": "postural_instability_related_burden",
                "relation": "noradrenergic deficits contribute to postural instability refractory to dopamine therapy",
                "pd_ncd_change": "increased",
            },
            {
                "source": "frontal_association_cortex_proxy",
                "target": "executive_dysfunction",
                "relation": "frontal cortical burden worsens executive function",
                "pd_ncd_change": "increased",
            },
            {
                "source": "parietal_association_cortex_proxy",
                "target": "attention_arousal_instability",
                "relation": "parietal attentional-system burden contributes to attentional instability",
                "pd_ncd_change": "increased",
            },
            {
                "source": "temporal_association_cortex_proxy",
                "target": "memory_impairment",
                "relation": "temporal cortical burden contributes to mnemonic dysfunction",
                "pd_ncd_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "memory_impairment",
                "relation": "hippocampal atrophy contributes to memory impairment",
                "pd_ncd_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "executive_dysfunction",
                "relation": "disrupted subcortical loops impair executive control",
                "pd_ncd_change": "increased",
            },
            {
                "source": "frontoparietal_attention_dysconnectivity",
                "target": "executive_dysfunction",
                "relation": "frontoparietal disruption is associated with executive dysfunction",
                "pd_ncd_change": "increased",
            },
            {
                "source": "frontolimbic_dysconnectivity",
                "target": "neuropsychiatric_burden",
                "relation": "fronto-limbic dysconnectivity contributes to depression, anxiety, and apathy",
                "pd_ncd_change": "increased",
            },
            {
                "source": "default_mode_network_dysconnectivity",
                "target": "global_cognitive_decline",
                "relation": "default mode disorganization contributes to broad cognitive dysfunction",
                "pd_ncd_change": "increased",
            },
            {
                "source": "executive_dysfunction",
                "target": "global_cognitive_decline",
                "relation": "executive impairment contributes to global neurocognitive burden",
                "pd_ncd_change": "increased",
            },
            {
                "source": "attention_arousal_instability",
                "target": "global_cognitive_decline",
                "relation": "attentional instability contributes to global neurocognitive burden",
                "pd_ncd_change": "increased",
            },
            {
                "source": "memory_impairment",
                "target": "global_cognitive_decline",
                "relation": "memory impairment contributes to global neurocognitive burden",
                "pd_ncd_change": "increased",
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
        generic_penalty = 1 if name in {
            "substantia nigra",
            "locus coeruleus",
            "hippocampus",
            "frontal cortex",
            "temporal cortex",
            "parietal cortex",
            "basal ganglia",
        } else 0
        lobar_penalty = 1 if any(k in name for k in ["frontal", "temporal", "parietal"]) and "area" not in name else 0
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
                df = df[df[gene_col].astype(str).str.upper().isin([g.upper() for g in genes])].copy()
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
        gene_panel: Sequence[str] = DEFAULT_PD_NCD_GENE_PANEL,
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
        alpha_synuclein_lewy_pathology: float = 0.6,
        genetic_vulnerability: float = 0.35,
        dopaminergic_neuron_loss: float = 0.55,
        brainstem_to_cortex_spread: float = 0.5,
    ) -> Dict[str, pd.Series]:
        """
        Run a transparent normalized simulation of chapter logic.

        All arguments are clipped to [0, 1]. Higher values represent greater
        pathology burden.
        """

        inputs = pd.Series(
            {
                "alpha_synuclein_lewy_pathology": self._clip01(alpha_synuclein_lewy_pathology),
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "dopaminergic_neuron_loss": self._clip01(dopaminergic_neuron_loss),
                "brainstem_to_cortex_spread": self._clip01(brainstem_to_cortex_spread),
            },
            dtype=float,
        )

        latents = pd.Series(dtype=float)
        latents["widespread_neurodegenerative_spread"] = self._clip01(
            0.45 * inputs["alpha_synuclein_lewy_pathology"]
            + 0.25 * inputs["brainstem_to_cortex_spread"]
            + 0.15 * inputs["genetic_vulnerability"]
            + 0.15 * inputs["dopaminergic_neuron_loss"]
        )
        latents["noradrenergic_depletion"] = self._clip01(
            0.45 * latents["widespread_neurodegenerative_spread"]
            + 0.25 * inputs["brainstem_to_cortex_spread"]
            + 0.15 * inputs["alpha_synuclein_lewy_pathology"]
            + 0.15 * inputs["genetic_vulnerability"]
        )
        latents["cholinergic_dysfunction"] = self._clip01(
            0.45 * latents["widespread_neurodegenerative_spread"]
            + 0.25 * inputs["alpha_synuclein_lewy_pathology"]
            + 0.15 * inputs["brainstem_to_cortex_spread"]
            + 0.15 * inputs["genetic_vulnerability"]
        )
        latents["serotonergic_dysfunction"] = self._clip01(
            0.4 * latents["widespread_neurodegenerative_spread"]
            + 0.25 * inputs["alpha_synuclein_lewy_pathology"]
            + 0.2 * inputs["brainstem_to_cortex_spread"]
            + 0.15 * inputs["genetic_vulnerability"]
        )
        latents["cortical_atrophy_burden"] = self._clip01(
            0.35 * latents["widespread_neurodegenerative_spread"]
            + 0.25 * inputs["alpha_synuclein_lewy_pathology"]
            + 0.15 * latents["cholinergic_dysfunction"]
            + 0.15 * latents["noradrenergic_depletion"]
            + 0.1 * latents["serotonergic_dysfunction"]
        )
        latents["basal_ganglia_thalamocortical_dysconnectivity"] = self._clip01(
            0.4 * inputs["dopaminergic_neuron_loss"]
            + 0.25 * latents["widespread_neurodegenerative_spread"]
            + 0.2 * inputs["alpha_synuclein_lewy_pathology"]
            + 0.15 * latents["noradrenergic_depletion"]
        )
        latents["default_mode_network_dysconnectivity"] = self._clip01(
            0.35 * latents["cortical_atrophy_burden"]
            + 0.25 * latents["cholinergic_dysfunction"]
            + 0.2 * latents["widespread_neurodegenerative_spread"]
            + 0.2 * latents["serotonergic_dysfunction"]
        )
        latents["frontoparietal_attention_dysconnectivity"] = self._clip01(
            0.35 * latents["cortical_atrophy_burden"]
            + 0.25 * latents["noradrenergic_depletion"]
            + 0.2 * latents["cholinergic_dysfunction"]
            + 0.2 * latents["widespread_neurodegenerative_spread"]
        )
        latents["frontolimbic_dysconnectivity"] = self._clip01(
            0.35 * latents["serotonergic_dysfunction"]
            + 0.25 * latents["noradrenergic_depletion"]
            + 0.2 * latents["cortical_atrophy_burden"]
            + 0.2 * latents["widespread_neurodegenerative_spread"]
        )

        regional_state = pd.Series(dtype=float)
        regional_state["substantia_nigra_pars_compacta_proxy"] = self._clip01(
            0.75 * inputs["dopaminergic_neuron_loss"]
            + 0.25 * inputs["alpha_synuclein_lewy_pathology"]
        )
        regional_state["locus_coeruleus_proxy"] = self._clip01(
            0.7 * latents["noradrenergic_depletion"]
            + 0.3 * latents["widespread_neurodegenerative_spread"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.5 * latents["cortical_atrophy_burden"]
            + 0.25 * latents["default_mode_network_dysconnectivity"]
            + 0.25 * latents["widespread_neurodegenerative_spread"]
        )
        regional_state["frontal_association_cortex_proxy"] = self._clip01(
            0.45 * latents["cortical_atrophy_burden"]
            + 0.3 * latents["frontoparietal_attention_dysconnectivity"]
            + 0.25 * latents["frontolimbic_dysconnectivity"]
        )
        regional_state["temporal_association_cortex_proxy"] = self._clip01(
            0.45 * latents["cortical_atrophy_burden"]
            + 0.3 * latents["default_mode_network_dysconnectivity"]
            + 0.25 * latents["frontolimbic_dysconnectivity"]
        )
        regional_state["parietal_association_cortex_proxy"] = self._clip01(
            0.45 * latents["cortical_atrophy_burden"]
            + 0.35 * latents["frontoparietal_attention_dysconnectivity"]
            + 0.2 * latents["default_mode_network_dysconnectivity"]
        )
        regional_state["basal_ganglia_proxy"] = self._clip01(
            0.55 * latents["basal_ganglia_thalamocortical_dysconnectivity"]
            + 0.3 * inputs["dopaminergic_neuron_loss"]
            + 0.15 * inputs["alpha_synuclein_lewy_pathology"]
        )

        symptoms = pd.Series(dtype=float)
        symptoms["executive_dysfunction"] = self._clip01(
            0.4 * regional_state["frontal_association_cortex_proxy"]
            + 0.25 * latents["frontoparietal_attention_dysconnectivity"]
            + 0.2 * regional_state["locus_coeruleus_proxy"]
            + 0.15 * regional_state["basal_ganglia_proxy"]
        )
        symptoms["attention_arousal_instability"] = self._clip01(
            0.4 * regional_state["locus_coeruleus_proxy"]
            + 0.3 * latents["frontoparietal_attention_dysconnectivity"]
            + 0.15 * latents["cholinergic_dysfunction"]
            + 0.15 * regional_state["parietal_association_cortex_proxy"]
        )
        symptoms["memory_impairment"] = self._clip01(
            0.4 * regional_state["hippocampus"]
            + 0.25 * regional_state["temporal_association_cortex_proxy"]
            + 0.2 * latents["default_mode_network_dysconnectivity"]
            + 0.15 * latents["cortical_atrophy_burden"]
        )
        symptoms["neuropsychiatric_burden"] = self._clip01(
            0.35 * latents["frontolimbic_dysconnectivity"]
            + 0.3 * regional_state["locus_coeruleus_proxy"]
            + 0.2 * latents["serotonergic_dysfunction"]
            + 0.15 * regional_state["temporal_association_cortex_proxy"]
        )
        symptoms["postural_instability_related_burden"] = self._clip01(
            0.4 * regional_state["basal_ganglia_proxy"]
            + 0.3 * regional_state["locus_coeruleus_proxy"]
            + 0.3 * inputs["dopaminergic_neuron_loss"]
        )
        symptoms["global_cognitive_decline"] = self._clip01(
            0.3 * symptoms["executive_dysfunction"]
            + 0.25 * symptoms["attention_arousal_instability"]
            + 0.25 * symptoms["memory_impairment"]
            + 0.2 * latents["cortical_atrophy_burden"]
        )

        phenotypes = pd.Series(
            {
                "mild_pd_neurocognitive_profile": self._clip01(
                    self._mean(
                        [
                            symptoms["executive_dysfunction"],
                            symptoms["attention_arousal_instability"],
                            symptoms["memory_impairment"],
                        ]
                    )
                ),
                "major_pd_neurocognitive_profile": self._clip01(
                    self._mean(
                        [
                            symptoms["global_cognitive_decline"],
                            symptoms["memory_impairment"],
                            latents["cortical_atrophy_burden"],
                            regional_state["frontal_association_cortex_proxy"],
                        ]
                    )
                ),
                "frontolimbic_neuropsychiatric_profile": self._clip01(
                    self._mean(
                        [
                            symptoms["neuropsychiatric_burden"],
                            regional_state["locus_coeruleus_proxy"],
                            latents["frontolimbic_dysconnectivity"],
                        ]
                    )
                ),
                "brainstem_cortical_spread_profile": self._clip01(
                    self._mean(
                        [
                            latents["widespread_neurodegenerative_spread"],
                            latents["cortical_atrophy_burden"],
                            symptoms["global_cognitive_decline"],
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
    pd.set_option("display.width", 160)

    if siibra is None:  # pragma: no cover - environment dependent
        print(
            "siibra is not installed in this environment, so the atlas queries cannot run here. "
            "The script itself is ready to use once siibra is available."
        )
        raise SystemExit(0)

    model = ParkinsonsDiseaseNeurocognitiveDisorderModel()
    built = model.build()

    print("\n=== Nodes ===")
    print(built["nodes"].head(24).to_string(index=False))

    print("\n=== Edges ===")
    print(built["edges"].to_string(index=False))

    print("\n=== Resolved regions ===")
    for key, region in built["regions"].items():
        print(f"- {key}: {region.name}")

    print("\n=== Circuit connectivity ===")
    circuit = built["circuit_connectivity"]
    print(circuit.to_string() if not circuit.empty else "<no circuit connectivity matrix>")

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

    sim = model.simulate(
        alpha_synuclein_lewy_pathology=0.75,
        genetic_vulnerability=0.45,
        dopaminergic_neuron_loss=0.7,
        brainstem_to_cortex_spread=0.65,
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

    # Example coordinate assignment once siibra is installed:
    # print(model.assign_mni_point((-30, -20, 8)).head())

    # For retrieving microarray data, siibra connects to the web API of
    # the Allen Brain Atlas (© 2015 Allen Institute for Brain Science),
    # available from https://brain-map.org/api/index.html. Any use of the
    # microarray data needs to be in accordance with their terms of use,
    # as specified at https://alleninstitute.org/legal/terms-use/.
