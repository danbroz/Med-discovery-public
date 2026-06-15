from __future__ import annotations

"""
Gender Dysphoria siibra scaffold.

This script turns a biologically focused chapter on Gender Dysphoria into a
transparent, atlas-grounded research scaffold. It is intentionally conservative:

- The source chapter emphasizes genetic and hormonal mechanisms but provides no
  direct neuroimaging or post-mortem region list.
- Therefore, most biology remains latent rather than being over-forced into
  specific parcels.
- Region nodes are cautious proxies that can be retuned with suggest_regions().

This is a research scaffold for mechanistic exploration only. It is not a
validated disease model, not a diagnostic tool, and not a treatment guide.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception:  # pragma: no cover - import guard for portable execution
    siibra = None  # type: ignore


DEFAULT_GENE_PANEL = [
    "AR",        # androgen receptor
    "ESR1",      # estrogen receptor alpha
    "ESR2",      # estrogen receptor beta
    "CYP19A1",   # aromatase
    "SRD5A2",    # steroid 5-alpha-reductase 2
    "CYP17A1",   # steroidogenesis enzyme
    "HSD17B3",   # testosterone synthesis pathway
    "NR3C1",     # glucocorticoid receptor
    "BDNF",      # neurodevelopment / plasticity
    "OXTR",      # social neurobiology proxy
    "AVPR1A",    # social / sex-typed behavior proxy
    "SRY",       # sex-determining region Y
]


class GenderDysphoriaModel:
    """
    Atlas-grounded mechanistic scaffold for the chapter's Gender Dysphoria model.

    Conceptual reading of the chapter:
    - Gender identity is treated as a deeply rooted, developmentally established
      trait with strong biological influence.
    - The chapter frames Gender Dysphoria as a divergence between brain-based
      gender identity development and bodily sex development.
    - Prenatal hormone exposure and genes governing steroid synthesis,
      metabolism, and receptor signaling are the main upstream drivers.
    - Later circulating hormones are modeled as activational amplifiers because
      the chapter explicitly notes that hormones modulate established circuits
      and secondary sex characteristics later in life.

    Because the chapter is anatomically sparse, the scaffold uses a minimal set
    of conservative region proxies:
    - hypothalamus_proxy: neuroendocrine sexual-differentiation proxy
    - amygdala_proxy: socio-affective / sex-differentiated behavior proxy

    The second proxy is intentionally optional and should be retuned if the user
    has more specific imaging evidence than the chapter provides.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        self.atlas = None
        self.parcellation = None
        self.space = None

        if siibra is None:
            warnings.warn(
                "siibra is not installed in this environment. Atlas lookups, "
                "masks, feature queries, and coordinate assignments will return "
                "empty results, but build() and simulate() remain usable."
            )
        else:
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
            except Exception as exc:  # pragma: no cover - environment dependent
                warnings.warn(
                    f"Could not initialize siibra atlas resources: {exc}. "
                    "The scaffold will continue with unresolved atlas nodes."
                )
                self.atlas = None
                self.parcellation = None
                self.space = None

        # Conservative region mapping: the chapter names no specific parcels.
        # These are explicit proxies for users to retune later.
        self.region_candidates: Dict[str, List[str]] = {
            "hypothalamus_proxy": [
                "Hypothalamus left",
                "Hypothalamus",
                "hypothalamus",
            ],
            "amygdala_proxy": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_steroid_signaling_variation": (
                "Variation in genes controlling steroid synthesis, metabolism, or "
                "receptor signaling that could shift sexual differentiation of the brain"
            ),
            "prenatal_hormone_signal_divergence": (
                "Prenatal hormonal milieu leading brain sexual differentiation to diverge "
                "from genital and somatic differentiation"
            ),
            "activational_hormone_load": (
                "Later-life circulating hormone effects that modulate established circuits "
                "and intensify secondary sex characteristic salience"
            ),
            "body_sex_characteristic_misalignment": (
                "Current mismatch between bodily / secondary sex characteristics and the "
                "experienced gender identity"
            ),
            "gender_affirming_hormone_support": (
                "Hormonal alignment support that can shift secondary sex characteristics "
                "toward the experienced gender"
            ),
            "gender_affirming_support": (
                "Broader affirming clinical and social support that buffers psychosocial burden"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "steroid_signaling_divergence": (
                "Net divergence in steroid synthesis, metabolism, and receptor signaling"
            ),
            "brain_body_sex_differentiation_mismatch": (
                "Developmental divergence between brain-based gender identity programming "
                "and bodily sex differentiation"
            ),
            "gender_identity_encoding_divergence": (
                "Brain organization becoming more congruent with experienced gender than "
                "assigned sex at birth"
            ),
            "activational_hormone_amplification": (
                "Later hormonal modulation that amplifies circuit salience and bodily cues"
            ),
            "secondary_sex_alignment": (
                "Protective alignment of secondary sex characteristics with experienced gender"
            ),
            "body_identity_incongruence": (
                "Lived mismatch between body characteristics and experienced gender identity"
            ),
            "sex_typed_behavioral_expression_bias": (
                "Behavioral preferences and expression aligning with experienced gender"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "gender_dysphoria_distress": (
                "Distress arising from incongruence between experienced gender and body"
            ),
            "secondary_sex_characteristic_distress": (
                "Distress focused on secondary sex characteristics that intensify incongruence"
            ),
            "body_related_distress": (
                "Body-focused distress linked to mismatch between identity and physical traits"
            ),
            "psychosocial_adjustment_difficulty": (
                "Broader psychosocial burden that may improve when bodily traits are aligned"
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_steroid_signaling_variation",
                "target": "steroid_signaling_divergence",
                "relation": "alters steroid synthesis, metabolism, and receptor sensitivity",
                "gd_change": "increased",
            },
            {
                "source": "prenatal_hormone_signal_divergence",
                "target": "steroid_signaling_divergence",
                "relation": "captures atypical prenatal androgen / estrogen exposure effects",
                "gd_change": "increased",
            },
            {
                "source": "steroid_signaling_divergence",
                "target": "brain_body_sex_differentiation_mismatch",
                "relation": "decouples brain sexual differentiation from genital and somatic development",
                "gd_change": "increased",
            },
            {
                "source": "prenatal_hormone_signal_divergence",
                "target": "brain_body_sex_differentiation_mismatch",
                "relation": "directly shifts developmental trajectory of brain sexual differentiation",
                "gd_change": "increased",
            },
            {
                "source": "brain_body_sex_differentiation_mismatch",
                "target": "gender_identity_encoding_divergence",
                "relation": "organizes identity-related brain development toward experienced gender",
                "gd_change": "increased",
            },
            {
                "source": "gender_identity_encoding_divergence",
                "target": "sex_typed_behavioral_expression_bias",
                "relation": "biases early play and preference patterns toward experienced gender",
                "gd_change": "increased",
            },
            {
                "source": "activational_hormone_load",
                "target": "activational_hormone_amplification",
                "relation": "modulates established neural circuits later in life",
                "gd_change": "increased",
            },
            {
                "source": "gender_affirming_hormone_support",
                "target": "secondary_sex_alignment",
                "relation": "promotes bodily alignment with experienced gender via secondary sex traits",
                "gd_change": "decreased_distress",
            },
            {
                "source": "gender_identity_encoding_divergence",
                "target": "body_identity_incongruence",
                "relation": "creates conflict when body characteristics do not match experienced gender",
                "gd_change": "increased",
            },
            {
                "source": "body_sex_characteristic_misalignment",
                "target": "body_identity_incongruence",
                "relation": "loads the lived mismatch between body and experienced gender",
                "gd_change": "increased",
            },
            {
                "source": "activational_hormone_amplification",
                "target": "body_identity_incongruence",
                "relation": "amplifies salience of secondary sex characteristics",
                "gd_change": "increased",
            },
            {
                "source": "secondary_sex_alignment",
                "target": "body_identity_incongruence",
                "relation": "reduces bodily mismatch with experienced gender",
                "gd_change": "decreased",
            },
            {
                "source": "body_identity_incongruence",
                "target": "gender_dysphoria_distress",
                "relation": "drives core dysphoric distress",
                "gd_change": "increased",
            },
            {
                "source": "body_identity_incongruence",
                "target": "body_related_distress",
                "relation": "focuses distress on bodily characteristics",
                "gd_change": "increased",
            },
            {
                "source": "activational_hormone_amplification",
                "target": "secondary_sex_characteristic_distress",
                "relation": "intensifies distress through later hormonal effects on secondary sex traits",
                "gd_change": "increased",
            },
            {
                "source": "gender_dysphoria_distress",
                "target": "psychosocial_adjustment_difficulty",
                "relation": "contributes to psychosocial burden",
                "gd_change": "increased",
            },
            {
                "source": "secondary_sex_characteristic_distress",
                "target": "psychosocial_adjustment_difficulty",
                "relation": "adds body-related burden to adjustment difficulties",
                "gd_change": "increased",
            },
            {
                "source": "gender_affirming_support",
                "target": "psychosocial_adjustment_difficulty",
                "relation": "buffers psychosocial burden",
                "gd_change": "decreased",
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
    def _identifier_of(obj: Any) -> Optional[str]:
        return getattr(obj, "identifier", getattr(obj, "id", None))

    def _modality_candidates(self, kind: str) -> List[Any]:
        if siibra is None:
            return []

        cands: List[Any] = []
        try:
            if kind == "receptor":
                if hasattr(siibra.features, "molecular") and hasattr(
                    siibra.features.molecular, "ReceptorDensityFingerprint"
                ):
                    cands.append(siibra.features.molecular.ReceptorDensityFingerprint)
                if hasattr(siibra.features, "tabular") and hasattr(
                    siibra.features.tabular, "ReceptorDensityFingerprint"
                ):
                    cands.append(siibra.features.tabular.ReceptorDensityFingerprint)
            elif kind == "gene":
                if hasattr(siibra.features, "molecular") and hasattr(
                    siibra.features.molecular, "GeneExpressions"
                ):
                    cands.append(siibra.features.molecular.GeneExpressions)
                if hasattr(siibra.features, "tabular") and hasattr(
                    siibra.features.tabular, "GeneExpressions"
                ):
                    cands.append(siibra.features.tabular.GeneExpressions)
            elif kind == "connectivity":
                if hasattr(siibra.features, "connectivity") and hasattr(
                    siibra.features.connectivity, "StreamlineCounts"
                ):
                    cands.append(siibra.features.connectivity.StreamlineCounts)
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
            ])
        return cands

    def _safe_features_any(self, concept: Any, modalities: Sequence[Any], **kwargs: Any) -> List[Any]:
        if siibra is None or concept is None:
            return []
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
        if self.atlas is None:
            return []
        matches: List[Any] = []

        try:
            matches = list(
                self.atlas.find_regions(
                    query,
                    all_versions=False,
                    filter_children=False,
                    find_topmost=False,
                )
            )
        except Exception:
            matches = []

        if not matches and self.parcellation is not None:
            try:
                matches = list(self.parcellation.find(query, filter_children=False, find_topmost=False))
            except Exception:
                matches = []

        out = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower() or self.parcellation is None:
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {"amygdala", "hypothalamus"} else 0
        proxy_penalty = 1 if "area " not in name and "(" not in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if self.atlas is None and self.parcellation is None:
            return None

        for spec in candidates:
            if self.atlas is not None:
                try:
                    return self.atlas.get_region(spec, parcellation=self.parcellation)
                except Exception:
                    pass
            if self.parcellation is not None:
                try:
                    return self.parcellation.get_region(spec)
                except Exception:
                    pass
            matches = self._julich_matches(spec)
            if matches:
                return sorted(matches, key=self._region_rank)[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows = []
        seen = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
            row = (
                self._name_of(region),
                self._identifier_of(region),
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
        if region is None or self.space is None:
            return []
        try:
            props = region.spatial_props(space=self.space)
        except Exception:
            return []
        if props is None:
            return []
        if hasattr(props, "components"):
            return list(getattr(props, "components", []))
        if isinstance(props, dict):
            return list(props.values())
        if isinstance(props, (list, tuple)):
            return list(props)
        return [props]

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz: Optional[Tuple[float, float, float]] = None
        if centroid is not None:
            try:
                centroid_xyz = tuple(float(x) for x in tuple(centroid))  # type: ignore[arg-type]
            except Exception:
                try:
                    centroid_xyz = tuple(float(x) for x in centroid.coordinate)  # type: ignore[attr-defined]
                except Exception:
                    centroid_xyz = None
        volume_mm3 = getattr(main, "volume", None)
        return centroid_xyz, (float(volume_mm3) if volume_mm3 is not None else None)

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

        for feat in feats:
            try:
                df = feat.data.copy().reset_index(drop=True)
            except Exception:
                continue
            lower_cols = {str(c).lower(): c for c in df.columns}
            if {"gene", "level", "zscore"}.issubset(lower_cols):
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
        return pd.DataFrame()

    def _select_connectivity_compound(self) -> Optional[Any]:
        if self.parcellation is None:
            return None
        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            return None

        preferred = [f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort]
        return preferred[0] if preferred else feats[0]

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        compound = self._select_connectivity_compound()
        if compound is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            if len(compound) > 0:
                self._connectivity_matrix = compound[0].data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        if region is None:
            return None
        region_names = {
            self._name_of(region).lower(),
            str(getattr(region, "name", "")).lower(),
        }

        exact = [x for x in labels if self._name_of(x).lower() in region_names]
        if exact:
            return exact[0]

        rn = self._name_of(region).lower()
        fuzzy = [
            x
            for x in labels
            if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn
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
            df = df[df["connected_region"] != self._name_of(region)].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self, node_keys: Optional[Sequence[str]] = None) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame(columns=["source", "target", "value"])

        keys = list(node_keys) if node_keys is not None else list(self.region_objects.keys())
        rows: List[Dict[str, Any]] = []

        for src_key in keys:
            src_region = self.region_objects.get(src_key)
            if src_region is None:
                continue
            src_label = self._match_region_label(list(matrix.index), src_region)
            if src_label is None:
                src_label = self._match_region_label(list(matrix.columns), src_region)
            if src_label is None:
                continue

            for dst_key in keys:
                if src_key == dst_key:
                    continue
                dst_region = self.region_objects.get(dst_key)
                if dst_region is None:
                    continue
                dst_label = self._match_region_label(list(matrix.columns), dst_region)
                if dst_label is None:
                    dst_label = self._match_region_label(list(matrix.index), dst_region)
                if dst_label is None:
                    continue

                value = None
                try:
                    value = matrix.loc[src_label, dst_label]
                except Exception:
                    try:
                        value = matrix.loc[dst_label, src_label]
                    except Exception:
                        value = None
                if value is None:
                    continue
                try:
                    rows.append(
                        {
                            "source": src_key,
                            "target": dst_key,
                            "value": float(value),
                        }
                    )
                except Exception:
                    continue

        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values(["source", "value"], ascending=[True, False]).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
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
                }
            )

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            if region is None:
                warnings.warn(
                    f"Could not resolve a region for node '{key}'. This is acceptable for a proxy scaffold."
                )
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " "),
                        "node_type": "region_proxy",
                        "description": "Proxy circuit node unresolved in this siibra environment",
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
                    "label": self._name_of(region),
                    "node_type": "region_proxy",
                    "description": "Atlas-backed proxy node for a chapter-implied circuit",
                    "atlas_region": self._name_of(region),
                    "region_identifier": self._identifier_of(region),
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

    def simulate(
        self,
        genetic_steroid_signaling_variation: float = 0.55,
        prenatal_hormone_signal_divergence: float = 0.60,
        activational_hormone_load: float = 0.50,
        body_sex_characteristic_misalignment: float = 0.65,
        gender_affirming_hormone_support: float = 0.20,
        gender_affirming_support: float = 0.30,
    ) -> Dict[str, pd.Series]:
        """
        Transparent one-pass normalized simulator.

        The simulation reflects chapter directionality:
        inputs -> latent developmental biology -> proxy regional state ->
        symptoms -> phenotype summaries

        All values are clipped to [0, 1].
        """

        inputs = pd.Series(
            {
                "genetic_steroid_signaling_variation": self._clip01(genetic_steroid_signaling_variation),
                "prenatal_hormone_signal_divergence": self._clip01(prenatal_hormone_signal_divergence),
                "activational_hormone_load": self._clip01(activational_hormone_load),
                "body_sex_characteristic_misalignment": self._clip01(body_sex_characteristic_misalignment),
                "gender_affirming_hormone_support": self._clip01(gender_affirming_hormone_support),
                "gender_affirming_support": self._clip01(gender_affirming_support),
            },
            name="value",
        )

        latents = pd.Series(dtype=float, name="value")
        latents["steroid_signaling_divergence"] = self._clip01(
            0.55 * inputs["genetic_steroid_signaling_variation"]
            + 0.45 * inputs["prenatal_hormone_signal_divergence"]
        )
        latents["brain_body_sex_differentiation_mismatch"] = self._clip01(
            0.60 * latents["steroid_signaling_divergence"]
            + 0.40 * inputs["prenatal_hormone_signal_divergence"]
        )
        latents["gender_identity_encoding_divergence"] = self._clip01(
            0.75 * latents["brain_body_sex_differentiation_mismatch"]
            + 0.25 * latents["steroid_signaling_divergence"]
        )
        latents["activational_hormone_amplification"] = self._clip01(
            0.70 * inputs["activational_hormone_load"]
            + 0.20 * inputs["body_sex_characteristic_misalignment"]
            - 0.10 * inputs["gender_affirming_hormone_support"]
        )
        latents["secondary_sex_alignment"] = self._clip01(
            0.75 * inputs["gender_affirming_hormone_support"]
            + 0.25 * inputs["gender_affirming_support"]
        )
        latents["body_identity_incongruence"] = self._clip01(
            0.45 * latents["gender_identity_encoding_divergence"]
            + 0.25 * latents["activational_hormone_amplification"]
            + 0.30 * inputs["body_sex_characteristic_misalignment"]
            - 0.35 * latents["secondary_sex_alignment"]
            - 0.10 * inputs["gender_affirming_support"]
        )
        latents["sex_typed_behavioral_expression_bias"] = self._clip01(
            0.70 * latents["gender_identity_encoding_divergence"]
            + 0.20 * inputs["prenatal_hormone_signal_divergence"]
            - 0.05 * inputs["gender_affirming_support"]
        )

        regional_state = pd.Series(dtype=float, name="value")
        regional_state["hypothalamus_proxy"] = self._clip01(
            0.60 * latents["steroid_signaling_divergence"]
            + 0.40 * latents["activational_hormone_amplification"]
        )
        regional_state["amygdala_proxy"] = self._clip01(
            0.55 * latents["body_identity_incongruence"]
            + 0.25 * latents["activational_hormone_amplification"]
            + 0.20 * latents["sex_typed_behavioral_expression_bias"]
        )

        symptoms = pd.Series(dtype=float, name="value")
        symptoms["gender_dysphoria_distress"] = self._clip01(
            0.65 * latents["body_identity_incongruence"]
            + 0.20 * latents["activational_hormone_amplification"]
            - 0.15 * inputs["gender_affirming_support"]
        )
        symptoms["secondary_sex_characteristic_distress"] = self._clip01(
            0.55 * latents["activational_hormone_amplification"]
            + 0.35 * inputs["body_sex_characteristic_misalignment"]
            - 0.35 * inputs["gender_affirming_hormone_support"]
        )
        symptoms["body_related_distress"] = self._clip01(
            0.60 * latents["body_identity_incongruence"]
            + 0.25 * symptoms["secondary_sex_characteristic_distress"]
            - 0.15 * latents["secondary_sex_alignment"]
        )
        symptoms["psychosocial_adjustment_difficulty"] = self._clip01(
            0.45 * symptoms["gender_dysphoria_distress"]
            + 0.25 * symptoms["secondary_sex_characteristic_distress"]
            + 0.10 * symptoms["body_related_distress"]
            - 0.25 * inputs["gender_affirming_support"]
            - 0.15 * inputs["gender_affirming_hormone_support"]
        )

        phenotypes = pd.Series(dtype=float, name="value")
        phenotypes["developmental_mismatch_profile"] = self._clip01(
            (
                latents["brain_body_sex_differentiation_mismatch"]
                + latents["gender_identity_encoding_divergence"]
                + latents["body_identity_incongruence"]
            )
            / 3.0
        )
        phenotypes["hormone_responsive_dysphoria_profile"] = self._clip01(
            (
                latents["activational_hormone_amplification"]
                + symptoms["secondary_sex_characteristic_distress"]
                + symptoms["gender_dysphoria_distress"]
            )
            / 3.0
        )
        phenotypes["early_identity_consistency_signal"] = self._clip01(
            (
                latents["gender_identity_encoding_divergence"]
                + latents["sex_typed_behavioral_expression_bias"]
            )
            / 2.0
        )
        phenotypes["support_buffered_adjustment"] = self._clip01(
            (
                latents["secondary_sex_alignment"]
                + inputs["gender_affirming_support"]
                + (1.0 - symptoms["psychosocial_adjustment_difficulty"])
            )
            / 3.0
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if siibra is None or self.parcellation is None:
            warnings.warn("siibra/parcellation unavailable; returning empty assignment table.")
            return pd.DataFrame()

        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception as exc:
                warnings.warn(f"Could not load probabilistic map: {exc}")
                return pd.DataFrame()

        try:
            point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
            with siibra.QUIET:
                assignments = self._pmap.assign(point)
        except Exception as exc:
            warnings.warn(f"Coordinate assignment failed: {exc}")
            return pd.DataFrame()

        if not isinstance(assignments, pd.DataFrame):
            try:
                assignments = pd.DataFrame(assignments)
            except Exception:
                return pd.DataFrame()

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                return assignments.sort_values(candidate, ascending=False).reset_index(drop=True)
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Any:
        region = self.region_objects.get(node_key)
        if region is None or self.space is None:
            return None

        for method_name in ("get_regional_mask", "fetch_regional_map"):
            method = getattr(region, method_name, None)
            if callable(method):
                try:
                    return method(self.space, maptype="labelled")
                except TypeError:
                    try:
                        return method(space=self.space, maptype="labelled")
                    except Exception:
                        continue
                except Exception:
                    continue
        return None


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    model = GenderDysphoriaModel()
    bundle = model.build()

    print("\n=== Nodes ===")
    print(bundle["nodes"].head(25).to_string(index=False))

    print("\n=== Edges ===")
    print(bundle["edges"].to_string(index=False))

    print("\n=== Resolved regions ===")
    if bundle["regions"]:
        for k, region in bundle["regions"].items():
            print(f"- {k}: {getattr(region, 'name', region)}")
    else:
        print("No atlas regions resolved in this environment.")

    print("\n=== Feature availability ===")
    for k in model.region_candidates:
        print(
            f"- {k}: receptors={not model.receptors.get(k, pd.DataFrame()).empty}, "
            f"genes={not model.genes.get(k, pd.DataFrame()).empty}, "
            f"connectivity={not model.connectivity_profiles.get(k, pd.DataFrame()).empty}"
        )

    print("\n=== Example simulation ===")
    sim = model.simulate(
        genetic_steroid_signaling_variation=0.60,
        prenatal_hormone_signal_divergence=0.70,
        activational_hormone_load=0.55,
        body_sex_characteristic_misalignment=0.80,
        gender_affirming_hormone_support=0.25,
        gender_affirming_support=0.40,
    )
    for name, series in sim.items():
        print(f"\n{name.upper()}")
        print(series.sort_index().to_string())

    print("\nTip: use model.suggest_regions('amygdala') or model.suggest_regions('hypothalamus') to retune proxies.")
    print("Tip: use model.assign_mni_point((x, y, z)) when siibra is installed and online atlas resources are available.")
