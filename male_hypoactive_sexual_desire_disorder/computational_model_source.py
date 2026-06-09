from __future__ import annotations

"""
Male Hypoactive Sexual Desire Disorder siibra scaffold.

This script translates a chapter-level summary of male hypoactive sexual desire
into a transparent, atlas-grounded research scaffold. It is designed to be
readable, portable across evolving siibra APIs, and tolerant of missing
multimodal data.

Important notes
---------------
- This is a research scaffold, not a diagnostic or treatment tool.
- The simulator is a mechanistic interpretation of the supplied chapter, not a
  validated disease model.
- Endocrine and monoamine mechanisms are modeled primarily as latent biology,
  because the chapter emphasizes systems-level regulation rather than highly
  localized neuroanatomy.
- The broad frontal / prefrontal contribution is represented using a
  conservative `pfc_control` proxy. Temporal-lobe contributions remain mostly
  latent because the chapter does not specify a single temporal parcel.
- The gene panel is intentionally heuristic: it captures androgen, prolactin,
  dopamine, serotonin, norepinephrine, neuroplasticity, and stress-axis
  mechanisms mentioned or strongly implied by the chapter, not validated
  disorder-specific biomarkers.
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
    "AR",
    "ESR1",
    "ESR2",
    "CYP19A1",
    "SRD5A2",
    "SHBG",
    "PRL",
    "PRLR",
    "DRD2",
    "SLC6A3",
    "TH",
    "SLC6A4",
    "HTR1A",
    "HTR2A",
    "SLC6A2",
    "COMT",
    "MAOA",
    "BDNF",
    "NR3C1",
    "FKBP5",
]


class MaleHypoactiveSexualDesireDisorderModel:
    """
    Atlas-grounded male hypoactive sexual desire disorder scaffold.

    Chapter logic encoded here emphasizes:
    - endocrine regulation, especially testosterone and prolactin,
    - serotonergic inhibition of libido and sexual reward,
    - dopaminergic facilitation of motivation and reward,
    - broader catecholaminergic / motivational reduction,
    - frontolimbic dysregulation linking mood, stress, and desire,
    - enduring HPA-axis effects of early life stress,
    - broad prefrontal and limbic contributions to motivation and top-down
      regulation of sexual behavior.
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
                "siibra is required to instantiate "
                "MaleHypoactiveSexualDesireDisorderModel. "
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
                "frontal pole",
                "frontal",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "Hippocampus left",
                "hippocampus left",
                "hippocampus",
            ],
        }

        self.region_notes: Dict[str, str] = {
            "pfc_control": (
                "Proxy for the chapter's broad prefrontal / frontal control and "
                "motivational regulation network."
            ),
            "amygdala": (
                "Limbic affective-salience anchor used because the chapter links "
                "mood circuitry to sexual desire."
            ),
            "hippocampus": (
                "Memory and mood-regulation anchor named in the chapter's "
                "fronto-limbic discussion."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "low_testosterone": "Reduced androgenic drive that can diminish male libido.",
            "hyperprolactinemia": "Elevated prolactin burden that can suppress sexual desire.",
            "serotonergic_medication_burden": (
                "Predominantly serotonergic medication load, especially SSRI-like libido suppression."
            ),
            "dopamine_blockade_burden": (
                "D2-blocking or otherwise anti-dopaminergic burden that reduces reward drive."
            ),
            "depression_anxiety_burden": (
                "Comorbid mood or anxiety burden linked to low desire and anhedonia."
            ),
            "early_life_stress": (
                "Stress history that can leave enduring epigenetic / HPA-axis effects."
            ),
            "vascular_neurological_burden": (
                "Broader vascular or neurological illness burden that can reduce sexual motivation."
            ),
            "treatment_optimization": (
                "Protective support from medication review, endocrine correction, and targeted treatment."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "androgen_deficiency": "Reduced androgenic support for sexual motivation and libido.",
            "prolactin_excess": "Hyperprolactinemic suppression of sexual drive and reward regulation.",
            "serotonergic_inhibition": (
                "Serotonin-mediated suppression of libido, orgasmic drive, and sexual reward."
            ),
            "mesolimbic_dopamine_deficit": (
                "Reduced dopamine-facilitated wanting, reward salience, and motivational drive."
            ),
            "reward_drive_reduction": (
                "Integrated dopamine/norepinephrine-and-endocrine reduction in incentive motivation."
            ),
            "hpa_axis_dysregulation": (
                "Stress-system dysregulation linking early adversity to later mood and libido effects."
            ),
            "frontolimbic_motivational_dysregulation": (
                "Disrupted interaction of prefrontal control with limbic and mood-regulation circuits."
            ),
            "anhedonia_apathy_state": (
                "Generalized reduction in pleasure, initiative, and motivational engagement."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "low_sexual_desire": "Core reduction in libido and sexual interest.",
            "sexual_anhedonia": "Reduced pleasure or reward from sexual activity.",
            "apathy_avolition": "Motivational flattening that can generalize to sexual desire.",
            "desire_avoidance": (
                "Stress- or negative-affect-linked avoidance of sexual activity and intimacy."
            ),
            "psychotropic_nonadherence_risk": (
                "Risk of medication non-adherence driven by sexual side-effect burden."
            ),
        }

        self.edge_table: List[Dict[str, Any]] = [
            {
                "source": "low_testosterone",
                "target": "androgen_deficiency",
                "relation": "reduces endocrine support for libido",
                "mhsdd_change": "increased",
                "weight": 0.55,
            },
            {
                "source": "hyperprolactinemia",
                "target": "prolactin_excess",
                "relation": "raises prolactin-mediated suppression of desire",
                "mhsdd_change": "increased",
                "weight": 0.55,
            },
            {
                "source": "serotonergic_medication_burden",
                "target": "serotonergic_inhibition",
                "relation": "drives serotonin-linked inhibition of sexual function",
                "mhsdd_change": "increased",
                "weight": 0.55,
            },
            {
                "source": "dopamine_blockade_burden",
                "target": "mesolimbic_dopamine_deficit",
                "relation": "reduces dopamine-facilitated wanting and reward",
                "mhsdd_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "dopamine_blockade_burden",
                "target": "prolactin_excess",
                "relation": "promotes tuberoinfundibular hyperprolactinemia",
                "mhsdd_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "depression_anxiety_burden",
                "target": "frontolimbic_motivational_dysregulation",
                "relation": "links mood pathology to desire loss",
                "mhsdd_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "depression_anxiety_burden",
                "target": "anhedonia_apathy_state",
                "relation": "adds avolition and anhedonic pressure",
                "mhsdd_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "early_life_stress",
                "target": "hpa_axis_dysregulation",
                "relation": "creates enduring stress-axis dysregulation",
                "mhsdd_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "frontolimbic_motivational_dysregulation",
                "relation": "amplifies mood-linked circuit burden",
                "mhsdd_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "androgen_deficiency",
                "target": "reward_drive_reduction",
                "relation": "lowers endocrine support for sexual motivation",
                "mhsdd_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "prolactin_excess",
                "target": "mesolimbic_dopamine_deficit",
                "relation": "suppresses dopamine-facilitated sexual wanting",
                "mhsdd_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "serotonergic_inhibition",
                "target": "reward_drive_reduction",
                "relation": "suppresses sexual reward and desire",
                "mhsdd_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "mesolimbic_dopamine_deficit",
                "target": "reward_drive_reduction",
                "relation": "reduces incentive salience and wanting",
                "mhsdd_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "frontolimbic_motivational_dysregulation",
                "target": "pfc_control",
                "relation": "degrades top-down motivational regulation",
                "mhsdd_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "hippocampus",
                "relation": "adds stress-linked burden to memory and mood circuitry",
                "mhsdd_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "frontolimbic_motivational_dysregulation",
                "target": "amygdala",
                "relation": "increases affective-salience disturbance in libido regulation",
                "mhsdd_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "reward_drive_reduction",
                "target": "low_sexual_desire",
                "relation": "directly lowers libido and sexual initiative",
                "mhsdd_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "serotonergic_inhibition",
                "target": "sexual_anhedonia",
                "relation": "reduces sexual reward and pleasure",
                "mhsdd_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "anhedonia_apathy_state",
                "target": "apathy_avolition",
                "relation": "generalizes motivational loss beyond sexuality",
                "mhsdd_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "amygdala",
                "target": "desire_avoidance",
                "relation": "couples negative affect to sexual avoidance",
                "mhsdd_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "serotonergic_medication_burden",
                "target": "psychotropic_nonadherence_risk",
                "relation": "sexual side effects increase non-adherence pressure",
                "mhsdd_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "treatment_optimization",
                "target": "androgen_deficiency",
                "relation": "can improve endocrine contributors to libido loss",
                "mhsdd_change": "decreased",
                "weight": -0.20,
            },
            {
                "source": "treatment_optimization",
                "target": "prolactin_excess",
                "relation": "can reduce medication- or endocrine-driven prolactin burden",
                "mhsdd_change": "decreased",
                "weight": -0.25,
            },
            {
                "source": "treatment_optimization",
                "target": "serotonergic_inhibition",
                "relation": "can reduce libido-suppressing medication burden",
                "mhsdd_change": "decreased",
                "weight": -0.20,
            },
            {
                "source": "treatment_optimization",
                "target": "frontolimbic_motivational_dysregulation",
                "relation": "can improve mood and motivational circuitry burden",
                "mhsdd_change": "decreased",
                "weight": -0.15,
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
        generic_penalty = 1 if name in {"prefrontal cortex", "amygdala", "hippocampus"} else 0
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
        try:
            centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
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
        low_testosterone: float = 0.55,
        hyperprolactinemia: float = 0.35,
        serotonergic_medication_burden: float = 0.45,
        dopamine_blockade_burden: float = 0.20,
        depression_anxiety_burden: float = 0.50,
        early_life_stress: float = 0.35,
        vascular_neurological_burden: float = 0.20,
        treatment_optimization: float = 0.35,
    ) -> Dict[str, pd.Series]:
        """
        Run a one-pass normalized simulation.

        The calculation order is intentionally acyclic and transparent:
        inputs -> latent biology -> regional burden -> symptoms -> phenotype summaries
        """

        inputs = pd.Series(
            {
                "low_testosterone": self._clip01(low_testosterone),
                "hyperprolactinemia": self._clip01(hyperprolactinemia),
                "serotonergic_medication_burden": self._clip01(serotonergic_medication_burden),
                "dopamine_blockade_burden": self._clip01(dopamine_blockade_burden),
                "depression_anxiety_burden": self._clip01(depression_anxiety_burden),
                "early_life_stress": self._clip01(early_life_stress),
                "vascular_neurological_burden": self._clip01(vascular_neurological_burden),
                "treatment_optimization": self._clip01(treatment_optimization),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["androgen_deficiency"] = self._clip01(
            0.55 * inputs["low_testosterone"]
            + 0.10 * inputs["vascular_neurological_burden"]
            + 0.05 * inputs["depression_anxiety_burden"]
            - 0.20 * inputs["treatment_optimization"]
        )
        latents["prolactin_excess"] = self._clip01(
            0.55 * inputs["hyperprolactinemia"]
            + 0.25 * inputs["dopamine_blockade_burden"]
            + 0.05 * inputs["depression_anxiety_burden"]
            - 0.25 * inputs["treatment_optimization"]
        )
        latents["serotonergic_inhibition"] = self._clip01(
            0.55 * inputs["serotonergic_medication_burden"]
            + 0.15 * inputs["depression_anxiety_burden"]
            + 0.10 * inputs["vascular_neurological_burden"]
            - 0.20 * inputs["treatment_optimization"]
        )
        latents["hpa_axis_dysregulation"] = self._clip01(
            0.40 * inputs["early_life_stress"]
            + 0.25 * inputs["depression_anxiety_burden"]
            + 0.10 * inputs["vascular_neurological_burden"]
            - 0.15 * inputs["treatment_optimization"]
        )
        latents["mesolimbic_dopamine_deficit"] = self._clip01(
            0.35 * inputs["dopamine_blockade_burden"]
            + 0.25 * latents["prolactin_excess"]
            + 0.20 * latents["androgen_deficiency"]
            + 0.10 * inputs["depression_anxiety_burden"]
            + 0.10 * latents["serotonergic_inhibition"]
            - 0.20 * inputs["treatment_optimization"]
        )
        latents["reward_drive_reduction"] = self._clip01(
            0.35 * latents["mesolimbic_dopamine_deficit"]
            + 0.25 * latents["androgen_deficiency"]
            + 0.15 * latents["serotonergic_inhibition"]
            + 0.10 * latents["prolactin_excess"]
            + 0.10 * inputs["depression_anxiety_burden"]
            + 0.10 * inputs["vascular_neurological_burden"]
            - 0.15 * inputs["treatment_optimization"]
        )
        latents["frontolimbic_motivational_dysregulation"] = self._clip01(
            0.30 * inputs["depression_anxiety_burden"]
            + 0.25 * latents["hpa_axis_dysregulation"]
            + 0.15 * inputs["vascular_neurological_burden"]
            + 0.15 * latents["serotonergic_inhibition"]
            + 0.10 * latents["mesolimbic_dopamine_deficit"]
            - 0.15 * inputs["treatment_optimization"]
        )
        latents["anhedonia_apathy_state"] = self._clip01(
            0.25 * latents["serotonergic_inhibition"]
            + 0.25 * latents["mesolimbic_dopamine_deficit"]
            + 0.20 * latents["frontolimbic_motivational_dysregulation"]
            + 0.15 * inputs["depression_anxiety_burden"]
            + 0.10 * latents["prolactin_excess"]
            - 0.15 * inputs["treatment_optimization"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["pfc_control"] = self._clip01(
            0.35 * latents["frontolimbic_motivational_dysregulation"]
            + 0.20 * latents["anhedonia_apathy_state"]
            + 0.15 * latents["serotonergic_inhibition"]
            + 0.10 * inputs["depression_anxiety_burden"]
            + 0.10 * inputs["vascular_neurological_burden"]
            + 0.10 * latents["hpa_axis_dysregulation"]
            - 0.20 * inputs["treatment_optimization"]
        )
        regional_state["amygdala"] = self._clip01(
            0.25 * inputs["depression_anxiety_burden"]
            + 0.25 * latents["hpa_axis_dysregulation"]
            + 0.20 * latents["serotonergic_inhibition"]
            + 0.10 * latents["frontolimbic_motivational_dysregulation"]
            + 0.10 * inputs["vascular_neurological_burden"]
            - 0.15 * inputs["treatment_optimization"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.25 * latents["hpa_axis_dysregulation"]
            + 0.20 * inputs["depression_anxiety_burden"]
            + 0.15 * inputs["vascular_neurological_burden"]
            + 0.15 * latents["frontolimbic_motivational_dysregulation"]
            + 0.10 * latents["anhedonia_apathy_state"]
            - 0.15 * inputs["treatment_optimization"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["low_sexual_desire"] = self._clip01(
            0.30 * latents["reward_drive_reduction"]
            + 0.20 * latents["androgen_deficiency"]
            + 0.15 * latents["prolactin_excess"]
            + 0.15 * latents["serotonergic_inhibition"]
            + 0.10 * regional_state["pfc_control"]
            + 0.10 * latents["anhedonia_apathy_state"]
            - 0.20 * inputs["treatment_optimization"]
        )
        symptoms["sexual_anhedonia"] = self._clip01(
            0.35 * latents["serotonergic_inhibition"]
            + 0.25 * latents["anhedonia_apathy_state"]
            + 0.20 * latents["reward_drive_reduction"]
            + 0.10 * latents["mesolimbic_dopamine_deficit"]
            + 0.10 * regional_state["amygdala"]
            - 0.15 * inputs["treatment_optimization"]
        )
        symptoms["apathy_avolition"] = self._clip01(
            0.35 * latents["anhedonia_apathy_state"]
            + 0.25 * latents["frontolimbic_motivational_dysregulation"]
            + 0.15 * regional_state["pfc_control"]
            + 0.10 * regional_state["hippocampus"]
            + 0.10 * latents["reward_drive_reduction"]
            - 0.15 * inputs["treatment_optimization"]
        )
        symptoms["desire_avoidance"] = self._clip01(
            0.30 * regional_state["amygdala"]
            + 0.20 * latents["hpa_axis_dysregulation"]
            + 0.20 * inputs["depression_anxiety_burden"]
            + 0.15 * latents["anhedonia_apathy_state"]
            + 0.10 * latents["serotonergic_inhibition"]
            - 0.10 * inputs["treatment_optimization"]
        )
        symptoms["psychotropic_nonadherence_risk"] = self._clip01(
            0.35 * inputs["serotonergic_medication_burden"]
            + 0.20 * inputs["dopamine_blockade_burden"]
            + 0.20 * symptoms["sexual_anhedonia"]
            + 0.15 * symptoms["low_sexual_desire"]
            + 0.10 * symptoms["apathy_avolition"]
            - 0.15 * inputs["treatment_optimization"]
        )

        phenotypes = pd.Series(dtype=float, name="phenotypes")
        phenotypes["endocrine_dominant_low_desire"] = self._clip01(
            0.30 * latents["androgen_deficiency"]
            + 0.25 * latents["prolactin_excess"]
            + 0.25 * latents["reward_drive_reduction"]
            + 0.20 * symptoms["low_sexual_desire"]
        )
        phenotypes["serotonergic_medication_induced_profile"] = self._clip01(
            0.25 * inputs["serotonergic_medication_burden"]
            + 0.25 * latents["serotonergic_inhibition"]
            + 0.20 * symptoms["sexual_anhedonia"]
            + 0.15 * symptoms["low_sexual_desire"]
            + 0.15 * symptoms["psychotropic_nonadherence_risk"]
        )
        phenotypes["dopamine_blockade_hyperprolactinemia_profile"] = self._clip01(
            0.25 * inputs["dopamine_blockade_burden"]
            + 0.25 * latents["prolactin_excess"]
            + 0.25 * latents["mesolimbic_dopamine_deficit"]
            + 0.25 * symptoms["low_sexual_desire"]
        )
        phenotypes["stress_depression_linked_low_desire"] = self._clip01(
            0.15 * inputs["early_life_stress"]
            + 0.20 * inputs["depression_anxiety_burden"]
            + 0.20 * latents["hpa_axis_dysregulation"]
            + 0.20 * latents["frontolimbic_motivational_dysregulation"]
            + 0.25 * symptoms["desire_avoidance"]
        )
        phenotypes["frontolimbic_apathy_profile"] = self._clip01(
            0.20 * regional_state["pfc_control"]
            + 0.15 * regional_state["hippocampus"]
            + 0.25 * latents["anhedonia_apathy_state"]
            + 0.20 * symptoms["apathy_avolition"]
            + 0.20 * latents["frontolimbic_motivational_dysregulation"]
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
        model = MaleHypoactiveSexualDesireDisorderModel()
        bundle = model.build(connectivity_rows=10)

        print("\n=== Nodes ===")
        print(
            bundle["nodes"][["key", "node_type", "atlas_region", "feature_summary"]]
            .fillna("")
            .to_string(index=False)
        )

        print("\n=== Edges (first 12) ===")
        print(bundle["edges"].head(12).to_string(index=False))

        print("\n=== Region suggestions for 'prefrontal' ===")
        print(model.suggest_regions("prefrontal").head(10).to_string(index=False))

        print("\n=== Example receptor / gene / connectivity tables ===")
        for region_key in ("pfc_control", "amygdala", "hippocampus"):
            if region_key in bundle["regions"]:
                print(f"\n[{region_key}] receptor rows: {len(bundle['receptors'][region_key])}")
                print(f"[{region_key}] gene rows: {len(bundle['genes'][region_key])}")
                print(f"[{region_key}] connectivity rows: {len(bundle['connectivity_profiles'][region_key])}")

        print("\n=== Example simulation ===")
        sim = model.simulate(
            low_testosterone=0.65,
            hyperprolactinemia=0.35,
            serotonergic_medication_burden=0.55,
            dopamine_blockade_burden=0.20,
            depression_anxiety_burden=0.50,
            early_life_stress=0.30,
            vascular_neurological_burden=0.20,
            treatment_optimization=0.35,
        )
        for name, series in sim.items():
            print(f"\n{name.upper()}")
            print(series.sort_values(ascending=False).to_string())

        print("\n=== Example MNI assignment ===")
        try:
            print(model.assign_mni_point((-24, 20, 32)).head(10).to_string(index=False))
        except Exception as exc:
            print(f"MNI assignment example skipped: {exc}")

        # Example region-mask retrieval:
        # pfc_mask = model.region_mask("pfc_control")
        # if pfc_mask is not None:
        #     print(type(pfc_mask))
