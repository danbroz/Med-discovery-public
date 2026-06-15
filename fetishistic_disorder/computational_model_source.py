
from __future__ import annotations

"""
FetishisticDisorderModel
========================

Atlas-grounded research scaffold for Fetishistic Disorder.

This script translates a chapter-level biological summary of Fetishistic
Disorder into a simple, transparent mechanistic model that can be explored
with siibra when atlas data are available. It is intended for research
scaffolding and hypothesis tracing, not diagnosis or treatment.

Chapter-derived themes emphasized here:
- Mesolimbic dopamine is kept as a distributed reward-process signal rather
  than forced into a single parcel.
- Prefrontal control is represented conservatively using orbitofrontal cortex
  plus a dorsolateral-prefrontal proxy, because exact atlas labels vary.
- Ventral tegmental area and nucleus accumbens / ventral striatum are explicit
  proxies, which is appropriate because the chapter is systems-level and Julich
  coverage can differ by siibra version.
- Trauma-linked epigenetic effects are modeled as latent biology rather than as
  a falsely localized lesion.

The scaffold degrades gracefully when siibra is unavailable: the simulator still
runs, while atlas-backed feature retrieval methods return empty tables.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - environment dependent
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_GENE_PANEL = [
    # Dopamine / reward salience
    "DRD2",
    "DRD3",
    "DRD4",
    "SLC6A3",
    "COMT",
    "MAOA",
    "OPRM1",
    # Serotonin / impulse control
    "SLC6A4",
    "HTR1A",
    "HTR2A",
    "TPH2",
    # Plasticity and social-attachment biology
    "BDNF",
    "OXTR",
    # Stress and epigenetic embedding
    "NR3C1",
    "FKBP5",
    "CRHR1",
]


class FetishisticDisorderModel:
    """
    Research scaffold for Fetishistic Disorder.

    The graph is a chapter-faithful interpretation of fetishistic pathology as
    an interaction among conditioned reward learning, mesolimbic incentive
    salience, serotonergic impulse-control burden, trauma-linked biological
    embedding, and weakened top-down prefrontal restraint.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.siibra_available = siibra is not None
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        self.atlas = None
        self.parcellation = None
        self.space = None
        self._pmap = None
        self._labelmap = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

        if self.siibra_available:
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
            except Exception as exc:
                warnings.warn(
                    "siibra is installed but atlas resources could not be initialized; "
                    f"atlas-backed methods will degrade gracefully. Detail: {exc}"
                )
                self.atlas = None
                self.parcellation = None
                self.space = None
        else:
            warnings.warn(
                "siibra is not installed in this environment. Atlas-backed build(), "
                "feature lookup, and anatomical assignment will return partial outputs, "
                "but simulate() remains usable."
            )

        # Conservative region mapping from the chapter.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "orbitofrontal_cortex": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Area Fo5 (OFC) left",
                "orbitofrontal cortex",
                "Fo4",
                "Fo3",
            ],
            "dorsolateral_prefrontal_proxy": [
                "Area 8v2 (MFG) left",
                "Area 8v1 (MFG) left",
                "Area 8d2 (SFG) left",
                "Area 8d1 (SFG) left",
                "middle frontal gyrus",
                "dorsolateral prefrontal cortex",
                "prefrontal cortex",
            ],
            "ventral_striatum_proxy": [
                "nucleus accumbens left",
                "ventral striatum left",
                "accumbens",
                "ventral striatum",
                "striatum",
            ],
            "vta_proxy": [
                "ventral tegmental area left",
                "ventral tegmental area",
                "VTA",
                "midbrain tegmentum",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_reward_control_vulnerability": (
                "Inherited liability affecting sex-drive intensity, impulsivity, personality, "
                "and reward-pathway responsiveness"
            ),
            "early_life_trauma": (
                "Childhood abuse, neglect, or early stress exposure that may become biologically embedded"
            ),
            "reward_pairing_history": (
                "Repetitive pairing of a specific cue or object with sexual arousal and orgasm"
            ),
            "fetish_cue_exposure": (
                "Current availability or salience of the conditioned cue or object"
            ),
            "impulsivity_trait_load": (
                "Trait-level impulsivity or compulsivity burden relevant to control failure"
            ),
            "sex_drive_intensity": (
                "Baseline libido / reward-seeking drive intensity contributing to motivational pressure"
            ),
            "stress_mood_burden": (
                "Mood and stress burden that can erode control and amplify compulsive behavior"
            ),
            "ssri_support": (
                "Protective serotonergic treatment support, modeled from the chapter's SSRI discussion"
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "heritable_reward_control_liability": (
                "Shared inherited vulnerability affecting reward sensitivity and inhibitory control"
            ),
            "epigenetic_reward_attachment_shift": (
                "Trauma-linked biological embedding affecting stress, attachment, and reward circuits"
            ),
            "conditioned_object_reward_association": (
                "Learned high-value cue association created by repeated cue–arousal–orgasm pairing"
            ),
            "dopamine_incentive_salience": (
                "Mesolimbic 'wanting' signal that intensifies motivational pull toward the fetish cue"
            ),
            "serotonergic_impulse_control_deficit": (
                "Reduced serotonergic support for impulse control, mood regulation, and compulsivity restraint"
            ),
            "prefrontal_inhibitory_control_failure": (
                "Weak top-down control over subcortical reward and affective impulses"
            ),
            "compulsive_urge_loop": (
                "Self-reinforcing loop linking conditioned reward, craving-like urge, and repetitive behavior"
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "intrusive_fetish_fantasies": (
                "Repetitive and intrusive fetish-focused sexual thoughts or fantasies"
            ),
            "cue_triggered_sexual_urge": (
                "Strong urge elicited by the fetish cue as a conditioned reward signal"
            ),
            "impaired_control_over_behavior": (
                "Persistent difficulty controlling repetitive fetish-driven sexual behavior"
            ),
            "compulsive_fetish_behavior": (
                "Recurrent enactment driven by reward pull and weakened inhibition"
            ),
            "guilt_or_distress": (
                "Distress, shame, or conflict surrounding the behavior or its consequences"
            ),
            "partner_relationship_difficulty": (
                "Interpersonal strain, intimacy disruption, or partner conflict related to the behavior"
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_reward_control_vulnerability",
                "target": "heritable_reward_control_liability",
                "relation": "provides inherited vulnerability affecting reward sensitivity and control",
                "fetishistic_change": "increased",
            },
            {
                "source": "impulsivity_trait_load",
                "target": "heritable_reward_control_liability",
                "relation": "indexes trait-level control vulnerability",
                "fetishistic_change": "increased",
            },
            {
                "source": "sex_drive_intensity",
                "target": "heritable_reward_control_liability",
                "relation": "captures inherited variation in sexual motivation",
                "fetishistic_change": "increased",
            },
            {
                "source": "early_life_trauma",
                "target": "epigenetic_reward_attachment_shift",
                "relation": "may become biologically embedded in stress and reward systems",
                "fetishistic_change": "increased",
            },
            {
                "source": "stress_mood_burden",
                "target": "epigenetic_reward_attachment_shift",
                "relation": "can reinforce stress-linked circuit burden",
                "fetishistic_change": "increased",
            },
            {
                "source": "reward_pairing_history",
                "target": "conditioned_object_reward_association",
                "relation": "creates strong cue–reward learning",
                "fetishistic_change": "increased",
            },
            {
                "source": "fetish_cue_exposure",
                "target": "conditioned_object_reward_association",
                "relation": "keeps conditioned cue value active",
                "fetishistic_change": "increased",
            },
            {
                "source": "conditioned_object_reward_association",
                "target": "dopamine_incentive_salience",
                "relation": "transforms the cue into a powerful motivational trigger",
                "fetishistic_change": "increased",
            },
            {
                "source": "sex_drive_intensity",
                "target": "dopamine_incentive_salience",
                "relation": "amplifies reward-seeking motivation",
                "fetishistic_change": "increased",
            },
            {
                "source": "heritable_reward_control_liability",
                "target": "dopamine_incentive_salience",
                "relation": "biases reward circuitry toward stronger cue-driven 'wanting'",
                "fetishistic_change": "increased",
            },
            {
                "source": "impulsivity_trait_load",
                "target": "serotonergic_impulse_control_deficit",
                "relation": "matches low-control / compulsive traits discussed in the chapter",
                "fetishistic_change": "increased",
            },
            {
                "source": "stress_mood_burden",
                "target": "serotonergic_impulse_control_deficit",
                "relation": "reduces mood-regulated inhibitory stability",
                "fetishistic_change": "increased",
            },
            {
                "source": "ssri_support",
                "target": "serotonergic_impulse_control_deficit",
                "relation": "counteracts compulsive sexual behavior through serotonergic support",
                "fetishistic_change": "decreased",
            },
            {
                "source": "serotonergic_impulse_control_deficit",
                "target": "prefrontal_inhibitory_control_failure",
                "relation": "weakens top-down restraint over reward-driven impulses",
                "fetishistic_change": "increased",
            },
            {
                "source": "dopamine_incentive_salience",
                "target": "prefrontal_inhibitory_control_failure",
                "relation": "overloads cortical control with high incentive pressure",
                "fetishistic_change": "increased",
            },
            {
                "source": "epigenetic_reward_attachment_shift",
                "target": "amygdala",
                "relation": "raises affective salience and maladaptive attachment-related cue tagging",
                "fetishistic_change": "increased",
            },
            {
                "source": "conditioned_object_reward_association",
                "target": "amygdala",
                "relation": "strengthens affective cue associations",
                "fetishistic_change": "increased",
            },
            {
                "source": "dopamine_incentive_salience",
                "target": "vta_proxy",
                "relation": "matches elevated mesolimbic reward-drive state",
                "fetishistic_change": "increased",
            },
            {
                "source": "conditioned_object_reward_association",
                "target": "ventral_striatum_proxy",
                "relation": "consolidates learned reward value of the fetish cue",
                "fetishistic_change": "increased",
            },
            {
                "source": "dopamine_incentive_salience",
                "target": "ventral_striatum_proxy",
                "relation": "boosts motivational pull toward reward pursuit",
                "fetishistic_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_control_failure",
                "target": "orbitofrontal_cortex",
                "relation": "reflects impaired value-based restraint and decision control",
                "fetishistic_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_control_failure",
                "target": "dorsolateral_prefrontal_proxy",
                "relation": "reflects weakened executive control and behavioral inhibition",
                "fetishistic_change": "increased",
            },
            {
                "source": "dopamine_incentive_salience",
                "target": "compulsive_urge_loop",
                "relation": "intensifies repetitive reward-seeking",
                "fetishistic_change": "increased",
            },
            {
                "source": "conditioned_object_reward_association",
                "target": "compulsive_urge_loop",
                "relation": "locks the urge to specific cues",
                "fetishistic_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_control_failure",
                "target": "compulsive_urge_loop",
                "relation": "allows reward-driven repetition to escape restraint",
                "fetishistic_change": "increased",
            },
            {
                "source": "conditioned_object_reward_association",
                "target": "intrusive_fetish_fantasies",
                "relation": "produces recurrent cue-linked thought content",
                "fetishistic_change": "increased",
            },
            {
                "source": "ventral_striatum_proxy",
                "target": "cue_triggered_sexual_urge",
                "relation": "translates conditioned reward value into urge",
                "fetishistic_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "cue_triggered_sexual_urge",
                "relation": "adds affective salience to fetish cue exposure",
                "fetishistic_change": "increased",
            },
            {
                "source": "prefrontal_inhibitory_control_failure",
                "target": "impaired_control_over_behavior",
                "relation": "drives persistent control failure",
                "fetishistic_change": "increased",
            },
            {
                "source": "compulsive_urge_loop",
                "target": "impaired_control_over_behavior",
                "relation": "makes urges harder to regulate",
                "fetishistic_change": "increased",
            },
            {
                "source": "cue_triggered_sexual_urge",
                "target": "compulsive_fetish_behavior",
                "relation": "pushes enactment of the reward-seeking behavior",
                "fetishistic_change": "increased",
            },
            {
                "source": "impaired_control_over_behavior",
                "target": "compulsive_fetish_behavior",
                "relation": "prevents effective inhibition once urges arise",
                "fetishistic_change": "increased",
            },
            {
                "source": "compulsive_fetish_behavior",
                "target": "guilt_or_distress",
                "relation": "creates distress or shame about consequences",
                "fetishistic_change": "increased",
            },
            {
                "source": "compulsive_fetish_behavior",
                "target": "partner_relationship_difficulty",
                "relation": "can disrupt intimacy and relationships",
                "fetishistic_change": "increased",
            },
            {
                "source": "guilt_or_distress",
                "target": "partner_relationship_difficulty",
                "relation": "adds relational strain and conflict",
                "fetishistic_change": "increased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    def _modality_candidates(self, kind: str) -> List[Any]:
        if not self.siibra_available:
            return []

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
        if not self.siibra_available or concept is None:
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
        if not self.siibra_available or self.atlas is None:
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
            if "julich" in str(parc_name).lower() or self.parcellation is None:
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        proxy_penalty = 1 if "proxy" in name else 0
        generic_penalty = 1 if name in {"amygdala", "prefrontal cortex", "striatum"} else 0
        return (left_bonus, right_penalty, proxy_penalty, generic_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if not self.siibra_available or self.atlas is None:
            return None

        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation)
            except Exception:
                pass
            try:
                if self.parcellation is not None and hasattr(self.parcellation, "get_region"):
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
        centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        volume_mm3 = float(getattr(main, "volume", float("nan")))
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

        compound = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), feats[0])

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            first = compound[0]
            if hasattr(first, "data") and isinstance(first.data, pd.DataFrame):
                self._connectivity_matrix = first.data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._name_of(region)
        exact = [x for x in labels if self._name_of(x) == region_name]
        if exact:
            return exact[0]

        rn = region_name.lower()
        fuzzy = [
            x
            for x in labels
            if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn
        ]
        if fuzzy:
            return fuzzy[0]

        compact_rn = rn.replace(" left", "").replace(" right", "")
        for x in labels:
            xl = self._name_of(x).lower().replace(" left", "").replace(" right", "")
            if compact_rn == xl or compact_rn in xl or xl in compact_rn:
                return x
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
            df = pd.DataFrame(series).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = (
                df.dropna(subset=["value"])
                .sort_values("value", ascending=False)
                .query("connected_region != @region.name", engine="python")
                .head(max_rows)
                .reset_index(drop=True)
            )
            return df
        except Exception:
            return pd.DataFrame()

    def _lookup_connectivity_value(
        self,
        matrix: pd.DataFrame,
        src_region: Any,
        dst_region: Any,
    ) -> Tuple[float, Optional[str], Optional[str]]:
        if matrix.empty:
            return float("nan"), None, None

        src_row = self._match_region_label(list(matrix.index), src_region)
        src_col = self._match_region_label(list(matrix.columns), src_region)
        dst_row = self._match_region_label(list(matrix.index), dst_region)
        dst_col = self._match_region_label(list(matrix.columns), dst_region)

        if src_row is not None and dst_col is not None:
            try:
                value = float(pd.to_numeric(pd.Series([matrix.loc[src_row, dst_col]]), errors="coerce").iloc[0])
                return value, self._name_of(src_row), self._name_of(dst_col)
            except Exception:
                pass

        if dst_row is not None and src_col is not None:
            try:
                value = float(pd.to_numeric(pd.Series([matrix.loc[dst_row, src_col]]), errors="coerce").iloc[0])
                return value, self._name_of(src_col), self._name_of(dst_row)
            except Exception:
                pass

        return float("nan"), None, None

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        rows: List[Dict[str, Any]] = []

        for src_key, src_region in self.region_objects.items():
            for dst_key, dst_region in self.region_objects.items():
                if src_key == dst_key:
                    continue
                value, src_label, dst_label = self._lookup_connectivity_value(matrix, src_region, dst_region)
                rows.append(
                    {
                        "source_key": src_key,
                        "target_key": dst_key,
                        "source_region": self._name_of(src_region),
                        "target_region": self._name_of(dst_region),
                        "matrix_source_label": src_label,
                        "matrix_target_label": dst_label,
                        "value": value,
                    }
                )

        return pd.DataFrame(rows)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
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
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region_proxy" if key.endswith("_proxy") else "region",
                        "description": "Atlas-backed circuit node unresolved in this environment; proxy retained",
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
                    "node_type": "region_proxy" if key.endswith("_proxy") else "region",
                    "description": "Atlas-backed circuit node",
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

    def simulate(
        self,
        genetic_reward_control_vulnerability: float = 0.55,
        early_life_trauma: float = 0.35,
        reward_pairing_history: float = 0.75,
        fetish_cue_exposure: float = 0.65,
        impulsivity_trait_load: float = 0.45,
        sex_drive_intensity: float = 0.60,
        stress_mood_burden: float = 0.40,
        ssri_support: float = 0.20,
    ) -> Dict[str, pd.Series]:
        """
        Run a transparent one-pass normalized simulation.

        Inputs are clipped to [0, 1]. Higher values indicate greater burden,
        except ssri_support, which is protective.
        """

        inputs = pd.Series(
            {
                "genetic_reward_control_vulnerability": self._clip01(genetic_reward_control_vulnerability),
                "early_life_trauma": self._clip01(early_life_trauma),
                "reward_pairing_history": self._clip01(reward_pairing_history),
                "fetish_cue_exposure": self._clip01(fetish_cue_exposure),
                "impulsivity_trait_load": self._clip01(impulsivity_trait_load),
                "sex_drive_intensity": self._clip01(sex_drive_intensity),
                "stress_mood_burden": self._clip01(stress_mood_burden),
                "ssri_support": self._clip01(ssri_support),
            }
        )

        latents = pd.Series(
            {
                "heritable_reward_control_liability": self._clip01(
                    0.35 * inputs["genetic_reward_control_vulnerability"]
                    + 0.25 * inputs["impulsivity_trait_load"]
                    + 0.20 * inputs["sex_drive_intensity"]
                    + 0.10 * inputs["stress_mood_burden"]
                    + 0.05 * inputs["fetish_cue_exposure"]
                ),
                "epigenetic_reward_attachment_shift": self._clip01(
                    0.55 * inputs["early_life_trauma"]
                    + 0.20 * inputs["stress_mood_burden"]
                    + 0.10 * inputs["genetic_reward_control_vulnerability"]
                    + 0.05 * inputs["reward_pairing_history"]
                ),
            }
        )

        latents["conditioned_object_reward_association"] = self._clip01(
            0.50 * inputs["reward_pairing_history"]
            + 0.20 * inputs["fetish_cue_exposure"]
            + 0.10 * inputs["sex_drive_intensity"]
            + 0.10 * latents["epigenetic_reward_attachment_shift"]
            + 0.05 * latents["heritable_reward_control_liability"]
        )

        latents["dopamine_incentive_salience"] = self._clip01(
            0.35 * latents["conditioned_object_reward_association"]
            + 0.25 * inputs["sex_drive_intensity"]
            + 0.15 * inputs["fetish_cue_exposure"]
            + 0.10 * latents["heritable_reward_control_liability"]
            + 0.05 * latents["epigenetic_reward_attachment_shift"]
            - 0.05 * inputs["ssri_support"]
        )

        latents["serotonergic_impulse_control_deficit"] = self._clip01(
            0.30 * inputs["impulsivity_trait_load"]
            + 0.25 * inputs["stress_mood_burden"]
            + 0.15 * inputs["early_life_trauma"]
            + 0.10 * inputs["genetic_reward_control_vulnerability"]
            + 0.05 * inputs["fetish_cue_exposure"]
            - 0.35 * inputs["ssri_support"]
        )

        latents["prefrontal_inhibitory_control_failure"] = self._clip01(
            0.40 * latents["serotonergic_impulse_control_deficit"]
            + 0.25 * latents["dopamine_incentive_salience"]
            + 0.15 * inputs["stress_mood_burden"]
            + 0.10 * latents["heritable_reward_control_liability"]
            + 0.05 * inputs["fetish_cue_exposure"]
            - 0.10 * inputs["ssri_support"]
        )

        latents["compulsive_urge_loop"] = self._clip01(
            0.30 * latents["dopamine_incentive_salience"]
            + 0.25 * latents["conditioned_object_reward_association"]
            + 0.20 * latents["prefrontal_inhibitory_control_failure"]
            + 0.10 * latents["epigenetic_reward_attachment_shift"]
            + 0.05 * inputs["stress_mood_burden"]
        )

        regional_state = pd.Series(
            {
                "vta_proxy": self._clip01(
                    0.75 * latents["dopamine_incentive_salience"]
                    + 0.15 * inputs["fetish_cue_exposure"]
                    + 0.05 * inputs["sex_drive_intensity"]
                ),
                "ventral_striatum_proxy": self._clip01(
                    0.40 * latents["dopamine_incentive_salience"]
                    + 0.40 * latents["conditioned_object_reward_association"]
                    + 0.10 * inputs["fetish_cue_exposure"]
                    + 0.05 * latents["compulsive_urge_loop"]
                ),
                "amygdala": self._clip01(
                    0.35 * latents["epigenetic_reward_attachment_shift"]
                    + 0.30 * latents["conditioned_object_reward_association"]
                    + 0.20 * inputs["stress_mood_burden"]
                    + 0.10 * latents["dopamine_incentive_salience"]
                ),
                "orbitofrontal_cortex": self._clip01(
                    0.45 * latents["prefrontal_inhibitory_control_failure"]
                    + 0.20 * latents["compulsive_urge_loop"]
                    + 0.15 * latents["dopamine_incentive_salience"]
                    + 0.05 * inputs["stress_mood_burden"]
                    - 0.05 * inputs["ssri_support"]
                ),
                "dorsolateral_prefrontal_proxy": self._clip01(
                    0.50 * latents["prefrontal_inhibitory_control_failure"]
                    + 0.20 * latents["serotonergic_impulse_control_deficit"]
                    + 0.15 * inputs["stress_mood_burden"]
                    + 0.05 * latents["compulsive_urge_loop"]
                    - 0.05 * inputs["ssri_support"]
                ),
            }
        )

        symptoms = pd.Series(
            {
                "intrusive_fetish_fantasies": self._clip01(
                    0.35 * latents["conditioned_object_reward_association"]
                    + 0.25 * latents["dopamine_incentive_salience"]
                    + 0.15 * regional_state["amygdala"]
                    + 0.10 * latents["compulsive_urge_loop"]
                    + 0.05 * inputs["fetish_cue_exposure"]
                ),
                "cue_triggered_sexual_urge": self._clip01(
                    0.35 * regional_state["ventral_striatum_proxy"]
                    + 0.25 * regional_state["vta_proxy"]
                    + 0.20 * regional_state["amygdala"]
                    + 0.10 * latents["conditioned_object_reward_association"]
                    + 0.05 * inputs["sex_drive_intensity"]
                ),
                "impaired_control_over_behavior": self._clip01(
                    0.40 * latents["prefrontal_inhibitory_control_failure"]
                    + 0.20 * regional_state["dorsolateral_prefrontal_proxy"]
                    + 0.15 * regional_state["orbitofrontal_cortex"]
                    + 0.10 * latents["compulsive_urge_loop"]
                    + 0.05 * inputs["stress_mood_burden"]
                ),
            }
        )

        symptoms["compulsive_fetish_behavior"] = self._clip01(
            0.30 * symptoms["cue_triggered_sexual_urge"]
            + 0.25 * symptoms["impaired_control_over_behavior"]
            + 0.20 * latents["compulsive_urge_loop"]
            + 0.10 * symptoms["intrusive_fetish_fantasies"]
            + 0.05 * regional_state["ventral_striatum_proxy"]
        )

        symptoms["guilt_or_distress"] = self._clip01(
            0.30 * symptoms["compulsive_fetish_behavior"]
            + 0.25 * inputs["stress_mood_burden"]
            + 0.15 * symptoms["impaired_control_over_behavior"]
            + 0.10 * symptoms["intrusive_fetish_fantasies"]
        )

        symptoms["partner_relationship_difficulty"] = self._clip01(
            0.35 * symptoms["compulsive_fetish_behavior"]
            + 0.25 * symptoms["guilt_or_distress"]
            + 0.15 * symptoms["impaired_control_over_behavior"]
            + 0.10 * symptoms["intrusive_fetish_fantasies"]
        )

        phenotypes = pd.Series(
            {
                "conditioned_arousal_profile": self._clip01(
                    (
                        symptoms["cue_triggered_sexual_urge"]
                        + symptoms["intrusive_fetish_fantasies"]
                        + regional_state["ventral_striatum_proxy"]
                        + regional_state["amygdala"]
                    )
                    / 4.0
                ),
                "compulsive_loss_of_control_profile": self._clip01(
                    (
                        symptoms["compulsive_fetish_behavior"]
                        + symptoms["impaired_control_over_behavior"]
                        + latents["prefrontal_inhibitory_control_failure"]
                        + latents["compulsive_urge_loop"]
                    )
                    / 4.0
                ),
                "distress_impairment_profile": self._clip01(
                    (
                        symptoms["guilt_or_distress"]
                        + symptoms["partner_relationship_difficulty"]
                        + inputs["stress_mood_burden"]
                    )
                    / 3.0
                ),
                "reward_over_control_imbalance": self._clip01(
                    (
                        latents["dopamine_incentive_salience"]
                        + latents["conditioned_object_reward_association"]
                        + regional_state["ventral_striatum_proxy"]
                        + regional_state["vta_proxy"]
                    )
                    / 4.0
                    - 0.25 * (1.0 - latents["prefrontal_inhibitory_control_failure"])
                    - 0.15 * inputs["ssri_support"]
                ),
            }
        )

        return {
            "inputs": inputs,
            "latents": latents.sort_index(),
            "regional_state": regional_state.sort_index(),
            "symptoms": symptoms.sort_index(),
            "phenotypes": phenotypes.sort_index(),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Assign an MNI152 coordinate to the most likely parcellation regions.

        Returns a dataframe sorted by the most informative scoring column that is
        available in the current siibra environment.
        """
        if not self.siibra_available or self.atlas is None or self.parcellation is None:
            return pd.DataFrame(
                [{
                    "region": None,
                    "detail": "siibra atlas resources unavailable in this environment",
                    "xyz": tuple(xyz),
                }]
            )

        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception:
                try:
                    with siibra.QUIET:
                        self._pmap = self.atlas.get_map(
                            parcellation=self.parcellation,
                            space=self.atlas.get_space(self.assignment_space)
                            if hasattr(self.atlas, "get_space")
                            else self.assignment_space,
                            maptype="statistical",
                        )
                except Exception as exc:
                    return pd.DataFrame(
                        [{
                            "region": None,
                            "detail": f"could not create statistical map: {exc}",
                            "xyz": tuple(xyz),
                        }]
                    )

        try:
            point = siibra.Point(tuple(xyz), space=self.assignment_space)
            with siibra.QUIET:
                assignments = self._pmap.assign(point)
        except Exception as exc:
            return pd.DataFrame(
                [{
                    "region": None,
                    "detail": f"assignment failed: {exc}",
                    "xyz": tuple(xyz),
                }]
            )

        for candidate in ("map value", "correlation", "intersection over union", "value"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Any:
        """
        Return a region-specific map or mask when possible.

        Depending on siibra version, this may be a fetched statistical map,
        a labelled map fragment, or None if no suitable accessor exists.
        """
        region = self.region_objects.get(node_key)
        if region is None or not self.siibra_available:
            return None

        try:
            if self._pmap is None:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            return self._pmap.fetch(region)
        except Exception:
            pass

        try:
            if self._labelmap is None:
                with siibra.QUIET:
                    self._labelmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="labelled",
                    )
            return self._labelmap.fetch(region)
        except Exception:
            pass

        for attr in ("get_regional_map", "fetch"):
            try:
                fn = getattr(region, attr)
            except Exception:
                continue
            try:
                return fn(self.space) if attr == "get_regional_map" else fn()
            except Exception:
                continue
        return None


if __name__ == "__main__":
    model = FetishisticDisorderModel()
    build = model.build()

    print("\n=== Nodes (first 12) ===")
    print(build["nodes"].head(12).to_string(index=False))

    print("\n=== Edges (first 12) ===")
    print(build["edges"].head(12).to_string(index=False))

    print("\n=== Resolved region keys ===")
    print(list(build["regions"].keys()))

    sample_key = "orbitofrontal_cortex"
    print(f"\n=== Receptor table: {sample_key} ===")
    print(build["receptors"].get(sample_key, pd.DataFrame()).head().to_string(index=False))

    print(f"\n=== Gene table: {sample_key} ===")
    print(build["genes"].get(sample_key, pd.DataFrame()).head().to_string(index=False))

    print(f"\n=== Connectivity profile: {sample_key} ===")
    print(build["connectivity_profiles"].get(sample_key, pd.DataFrame()).head().to_string(index=False))

    print("\n=== Circuit connectivity ===")
    print(build["circuit_connectivity"].head(12).to_string(index=False))

    sim = model.simulate(
        genetic_reward_control_vulnerability=0.60,
        early_life_trauma=0.40,
        reward_pairing_history=0.85,
        fetish_cue_exposure=0.75,
        impulsivity_trait_load=0.50,
        sex_drive_intensity=0.65,
        stress_mood_burden=0.45,
        ssri_support=0.20,
    )

    print("\n=== Latents ===")
    print(sim["latents"].sort_values(ascending=False).to_string())

    print("\n=== Symptoms ===")
    print(sim["symptoms"].sort_values(ascending=False).to_string())

    print("\n=== Phenotypes ===")
    print(sim["phenotypes"].sort_values(ascending=False).to_string())

    # Optional coordinate assignment example when siibra is available:
    # print(model.assign_mni_point((-14, 34, -12)).head())
