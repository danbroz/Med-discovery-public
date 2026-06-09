from __future__ import annotations

"""
Oppositional defiant disorder (ODD) siibra scaffold.

This script turns a brief chapter on the biological basis of oppositional
 defiant disorder into an atlas-grounded, multimodal research scaffold.
The chapter itself emphasizes that ODD neurobiology remains incompletely
characterized and is often inferred from related externalizing and mood/
anxiety conditions. Accordingly, this scaffold stays conservative and only
models mechanisms that are either directly named or strongly implied:

- heritable / familial externalizing vulnerability,
- early life stress and trauma exposure,
- HPA-axis stress reactivity dysregulation,
- GABAergic disinhibition affecting behavioral control,
- broad monoaminergic dysregulation across dopamine, serotonin, and
  norepinephrine systems,
- disrupted amygdala-prefrontal regulation,
- amygdala threat / anger hyperreactivity with inadequate prefrontal
  inhibition,
- downstream angry-irritable mood, emotional lability, impulsive
  disinhibition, defiance, and reactive aggression.

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


DEFAULT_ODD_GENE_PANEL = [
    "DRD4",
    "DRD2",
    "SLC6A3",
    "COMT",
    "MAOA",
    "SLC6A4",
    "TPH2",
    "DBH",
    "GAD1",
    "GABRA2",
    "FKBP5",
    "NR3C1",
]


class OppositionalDefiantDisorderModel:
    """
    Atlas-grounded scaffold for oppositional defiant disorder.

    Notes
    -----
    - Larger simulator values represent greater dysregulation or symptom
      burden.
    - The source chapter presents much of ODD biology as inferential and
      shared with related externalizing conditions, so several nodes are
      intentionally broad latent biology constructs.
    - `prefrontal_control_proxy` is a proxy because the chapter names the
      prefrontal cortex at a systems level rather than committing to one
      cytoarchitectonic parcel.
    - `hippocampus` is included conservatively as a limbic context anchor
      because the chapter explicitly invokes hippocampal GABA-related
      observations and broader limbic-system involvement from related
      disorders.
    - Empty receptor, gene, or connectivity tables are acceptable and do not
      imply a script error.
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
                "siibra is required to instantiate OppositionalDefiantDisorderModel. "
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

        # Conservative region choices: only chapter-named regions or explicit
        # proxies for chapter-level systems descriptions.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "prefrontal_control_proxy": [
                "Area 46 left",
                "Area 9 left",
                "Area 45 left",
                "Area 44 left",
                "middle frontal gyrus left",
                "inferior frontal gyrus left",
                "prefrontal cortex left",
                "prefrontal cortex",
                "frontal cortex",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "CA2 (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
        }
        self.region_node_notes: Dict[str, str] = {
            "amygdala": (
                "Threat, fear, and anger-processing limbic node explicitly emphasized by the chapter."
            ),
            "prefrontal_control_proxy": (
                "Proxy for the chapter's general prefrontal control system that normally inhibits defiant or aggressive responses."
            ),
            "hippocampus": (
                "Conservative limbic context anchor derived from the chapter's hippocampal GABA discussion and broader limbic-system framing."
            ),
        }
        self.region_is_proxy: Dict[str, bool] = {
            "amygdala": False,
            "prefrontal_control_proxy": True,
            "hippocampus": False,
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Heritable liability for disruptive behavior, aggression, and related externalizing psychopathology."
            ),
            "early_life_stress_trauma": (
                "Early life stress, abuse, or trauma that can sensitise stress systems and disrupt emotion-regulation circuitry."
            ),
            "environmental_adversity_load": (
                "Broader adverse environmental influences interacting with biologic vulnerability."
            ),
            "shared_externalizing_liability": (
                "Transdiagnostic liability shared with ADHD and conduct-disorder spectrum problems."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "hpa_axis_dysregulation": (
                "Stress-response dysregulation producing heightened stress reactivity and irritability."
            ),
            "gabaergic_disinhibition": (
                "Reduced GABA-mediated inhibitory control contributing to failure to suppress inappropriate responses."
            ),
            "monoaminergic_dysregulation": (
                "Broad dopamine, serotonin, and norepinephrine dysregulation linked to impulsivity, emotional lability, and aggression."
            ),
            "amygdala_pfc_disconnectivity": (
                "Weakened functional coupling between the amygdala and prefrontal regulatory systems."
            ),
            "prefrontal_inhibitory_control_failure": (
                "Breakdown of top-down behavioral inhibition and emotion regulation."
            ),
            "threat_anger_bias": (
                "Heightened threat, anger, and hostile-reactive bias in emotional processing."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "angry_irritable_mood": (
                "Persistent angry or irritable mood downstream of frontolimbic dysregulation."
            ),
            "emotional_lability": (
                "Rapid shifts in affective state linked to monoaminergic and limbic instability."
            ),
            "impulsive_disinhibition": (
                "Difficulty suppressing inappropriate or provocative responses."
            ),
            "defiant_behavior": (
                "Argumentative, oppositional, or noncompliant behavior."
            ),
            "reactive_aggression": (
                "Reactive aggressive responding under stress, threat, or anger load."
            ),
        }

        # Conceptual edge table derived from chapter claims. The simulator
        # remains acyclic even though the conceptual graph is richer.
        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "monoaminergic_dysregulation",
                "relation": "shared heritable liability likely influences the monoaminergic systems implicated by related disorders",
                "odd_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "gabaergic_disinhibition",
                "relation": "biologic liability may contribute to inhibitory-control vulnerability",
                "odd_change": "increased",
            },
            {
                "source": "shared_externalizing_liability",
                "target": "monoaminergic_dysregulation",
                "relation": "ODD shares externalizing-spectrum risk with ADHD and conduct-disorder pathways",
                "odd_change": "increased",
            },
            {
                "source": "shared_externalizing_liability",
                "target": "prefrontal_inhibitory_control_failure",
                "relation": "shared externalizing vulnerability burdens inhibitory control systems",
                "odd_change": "increased",
            },
            {
                "source": "early_life_stress_trauma",
                "target": "hpa_axis_dysregulation",
                "relation": "early stress and trauma can produce long-term dysregulation of the HPA axis",
                "odd_change": "increased",
            },
            {
                "source": "early_life_stress_trauma",
                "target": "amygdala_pfc_disconnectivity",
                "relation": "childhood trauma can disrupt amygdala-prefrontal connectivity",
                "odd_change": "increased",
            },
            {
                "source": "environmental_adversity_load",
                "target": "hpa_axis_dysregulation",
                "relation": "adverse environmental pressures amplify stress-system burden",
                "odd_change": "increased",
            },
            {
                "source": "environmental_adversity_load",
                "target": "threat_anger_bias",
                "relation": "adversity can increase threat sensitivity and angry reactivity",
                "odd_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "amygdala_pfc_disconnectivity",
                "relation": "heightened stress reactivity weakens emotion-regulation circuitry",
                "odd_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "threat_anger_bias",
                "relation": "stress-system dysregulation contributes to irritable and aggressive phenotype expression",
                "odd_change": "increased",
            },
            {
                "source": "gabaergic_disinhibition",
                "target": "prefrontal_inhibitory_control_failure",
                "relation": "reduced GABAergic braking undermines suppression of inappropriate defiant or aggressive responses",
                "odd_change": "increased",
            },
            {
                "source": "gabaergic_disinhibition",
                "target": "hippocampus",
                "relation": "chapter-linked hippocampal GABA changes provide an indirect limbic context anchor",
                "odd_change": "increased",
            },
            {
                "source": "monoaminergic_dysregulation",
                "target": "prefrontal_inhibitory_control_failure",
                "relation": "dopamine, serotonin, and norepinephrine dysregulation contribute to impulsivity and disinhibition",
                "odd_change": "increased",
            },
            {
                "source": "monoaminergic_dysregulation",
                "target": "threat_anger_bias",
                "relation": "monoaminergic dysregulation contributes to emotional lability and aggression",
                "odd_change": "increased",
            },
            {
                "source": "amygdala_pfc_disconnectivity",
                "target": "prefrontal_inhibitory_control_failure",
                "relation": "weakened frontolimbic coupling reduces top-down regulation",
                "odd_change": "increased",
            },
            {
                "source": "threat_anger_bias",
                "target": "amygdala",
                "relation": "threat and anger bias manifests as amygdala hyperreactivity",
                "odd_change": "increased",
            },
            {
                "source": "amygdala_pfc_disconnectivity",
                "target": "amygdala",
                "relation": "insufficient prefrontal constraint leaves amygdala reactivity less controlled",
                "odd_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_control_failure",
                "target": "prefrontal_control_proxy",
                "relation": "control failure is expressed as dysregulated prefrontal inhibitory control",
                "odd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "angry_irritable_mood",
                "relation": "heightened threat and anger processing promotes irritable mood",
                "odd_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "emotional_lability",
                "relation": "limbic-context burden can contribute to affective instability",
                "odd_change": "increased",
            },
            {
                "source": "prefrontal_control_proxy",
                "target": "impulsive_disinhibition",
                "relation": "prefrontal dysregulation impairs behavioral inhibition",
                "odd_change": "increased",
            },
            {
                "source": "monoaminergic_dysregulation",
                "target": "emotional_lability",
                "relation": "monoaminergic imbalance contributes to emotional lability",
                "odd_change": "increased",
            },
            {
                "source": "monoaminergic_dysregulation",
                "target": "reactive_aggression",
                "relation": "monoaminergic imbalance contributes to aggression propensity",
                "odd_change": "increased",
            },
            {
                "source": "angry_irritable_mood",
                "target": "reactive_aggression",
                "relation": "irritable mood increases likelihood of reactive aggressive responses",
                "odd_change": "increased",
            },
            {
                "source": "impulsive_disinhibition",
                "target": "defiant_behavior",
                "relation": "disinhibition increases oppositional and noncompliant responding",
                "odd_change": "increased",
            },
            {
                "source": "impulsive_disinhibition",
                "target": "reactive_aggression",
                "relation": "disinhibited responding facilitates reactive aggression",
                "odd_change": "increased",
            },
            {
                "source": "angry_irritable_mood",
                "target": "defiant_behavior",
                "relation": "irritability increases oppositional and confrontational behavior",
                "odd_change": "increased",
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
        generic_penalty = 1 if name in {"amygdala", "hippocampus", "prefrontal cortex", "frontal cortex"} else 0
        lobar_penalty = 1 if "frontal" in name and "area" not in name else 0
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
        gene_panel: Sequence[str] = DEFAULT_ODD_GENE_PANEL,
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
        early_life_stress_trauma: float = 0.4,
        environmental_adversity_load: float = 0.4,
        shared_externalizing_liability: float = 0.5,
    ) -> Dict[str, pd.Series]:
        """
        Run a simple normalized mechanistic simulation.

        All arguments are clipped to [0, 1]. Larger values represent greater
        risk or burden.
        """

        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "early_life_stress_trauma": self._clip01(early_life_stress_trauma),
                "environmental_adversity_load": self._clip01(environmental_adversity_load),
                "shared_externalizing_liability": self._clip01(shared_externalizing_liability),
            },
            dtype=float,
        )

        latents = pd.Series(dtype=float)
        latents["hpa_axis_dysregulation"] = self._clip01(
            0.45 * inputs["early_life_stress_trauma"]
            + 0.35 * inputs["environmental_adversity_load"]
            + 0.1 * inputs["genetic_vulnerability"]
            + 0.1 * inputs["shared_externalizing_liability"]
        )
        latents["gabaergic_disinhibition"] = self._clip01(
            0.3 * inputs["genetic_vulnerability"]
            + 0.25 * inputs["shared_externalizing_liability"]
            + 0.2 * inputs["early_life_stress_trauma"]
            + 0.15 * latents["hpa_axis_dysregulation"]
            + 0.1 * inputs["environmental_adversity_load"]
        )
        latents["monoaminergic_dysregulation"] = self._clip01(
            0.35 * inputs["genetic_vulnerability"]
            + 0.3 * inputs["shared_externalizing_liability"]
            + 0.15 * inputs["environmental_adversity_load"]
            + 0.1 * inputs["early_life_stress_trauma"]
            + 0.1 * latents["hpa_axis_dysregulation"]
        )
        latents["amygdala_pfc_disconnectivity"] = self._clip01(
            0.4 * inputs["early_life_stress_trauma"]
            + 0.25 * latents["hpa_axis_dysregulation"]
            + 0.2 * inputs["environmental_adversity_load"]
            + 0.15 * latents["monoaminergic_dysregulation"]
        )
        latents["prefrontal_inhibitory_control_failure"] = self._clip01(
            0.35 * inputs["shared_externalizing_liability"]
            + 0.25 * latents["monoaminergic_dysregulation"]
            + 0.2 * latents["gabaergic_disinhibition"]
            + 0.2 * latents["amygdala_pfc_disconnectivity"]
        )
        latents["threat_anger_bias"] = self._clip01(
            0.35 * latents["hpa_axis_dysregulation"]
            + 0.25 * latents["amygdala_pfc_disconnectivity"]
            + 0.2 * latents["monoaminergic_dysregulation"]
            + 0.2 * latents["gabaergic_disinhibition"]
        )

        regional_state = pd.Series(dtype=float)
        regional_state["amygdala"] = self._clip01(
            0.65 * latents["threat_anger_bias"]
            + 0.25 * latents["amygdala_pfc_disconnectivity"]
            + 0.1 * latents["monoaminergic_dysregulation"]
        )
        regional_state["prefrontal_control_proxy"] = self._clip01(
            0.55 * latents["prefrontal_inhibitory_control_failure"]
            + 0.25 * latents["amygdala_pfc_disconnectivity"]
            + 0.2 * latents["monoaminergic_dysregulation"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.45 * latents["hpa_axis_dysregulation"]
            + 0.35 * inputs["early_life_stress_trauma"]
            + 0.2 * latents["gabaergic_disinhibition"]
        )

        symptoms = pd.Series(dtype=float)
        symptoms["angry_irritable_mood"] = self._clip01(
            0.45 * regional_state["amygdala"]
            + 0.25 * latents["threat_anger_bias"]
            + 0.15 * latents["monoaminergic_dysregulation"]
            + 0.15 * regional_state["hippocampus"]
        )
        symptoms["emotional_lability"] = self._clip01(
            0.35 * latents["monoaminergic_dysregulation"]
            + 0.25 * regional_state["hippocampus"]
            + 0.2 * regional_state["amygdala"]
            + 0.2 * latents["hpa_axis_dysregulation"]
        )
        symptoms["impulsive_disinhibition"] = self._clip01(
            0.45 * regional_state["prefrontal_control_proxy"]
            + 0.25 * latents["gabaergic_disinhibition"]
            + 0.2 * latents["monoaminergic_dysregulation"]
            + 0.1 * inputs["shared_externalizing_liability"]
        )
        symptoms["defiant_behavior"] = self._clip01(
            0.4 * symptoms["impulsive_disinhibition"]
            + 0.3 * symptoms["angry_irritable_mood"]
            + 0.2 * regional_state["prefrontal_control_proxy"]
            + 0.1 * inputs["environmental_adversity_load"]
        )
        symptoms["reactive_aggression"] = self._clip01(
            0.35 * symptoms["angry_irritable_mood"]
            + 0.25 * symptoms["impulsive_disinhibition"]
            + 0.2 * regional_state["amygdala"]
            + 0.1 * latents["monoaminergic_dysregulation"]
            + 0.1 * latents["hpa_axis_dysregulation"]
        )

        phenotypes = pd.Series(
            {
                "odd_core_profile": self._clip01(
                    self._mean(
                        [
                            symptoms["angry_irritable_mood"],
                            symptoms["defiant_behavior"],
                            symptoms["reactive_aggression"],
                        ]
                    )
                ),
                "irritable_emotional_dysregulation_profile": self._clip01(
                    self._mean(
                        [
                            symptoms["angry_irritable_mood"],
                            symptoms["emotional_lability"],
                            latents["threat_anger_bias"],
                        ]
                    )
                ),
                "externalizing_disinhibition_profile": self._clip01(
                    self._mean(
                        [
                            symptoms["impulsive_disinhibition"],
                            symptoms["defiant_behavior"],
                            regional_state["prefrontal_control_proxy"],
                        ]
                    )
                ),
                "trauma_stress_sensitized_profile": self._clip01(
                    self._mean(
                        [
                            latents["hpa_axis_dysregulation"],
                            latents["amygdala_pfc_disconnectivity"],
                            symptoms["angry_irritable_mood"],
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

    model = OppositionalDefiantDisorderModel()
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
        genetic_vulnerability=0.7,
        early_life_stress_trauma=0.8,
        environmental_adversity_load=0.6,
        shared_externalizing_liability=0.75,
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
    # print(model.suggest_regions("prefrontal").head())
    # print(model.assign_mni_point((-24, 0, -18)).head())
    # mask_img = model.region_mask("amygdala")
