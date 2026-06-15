from __future__ import annotations

"""
Kleptomania siibra scaffold.

This script turns a short kleptomania chapter into an atlas-grounded,
mechanistic research scaffold using siibra. It is designed for exploratory
modeling, not diagnosis or treatment.

Chapter logic represented here:
- impaired top-down prefrontal control,
- stress-linked noradrenergic arousal and tension before theft,
- amygdala and insula involvement in negative affect and urge pressure,
- reward/salience involvement via a ventral-striatal proxy,
- speculative developmental/genetic contributions to impulsivity and control.

The scaffold is intentionally conservative. Neurochemistry such as
norepinephrine and dopamine is modeled as latent biology rather than forced into
specific parcels. Reward circuitry is represented with a clearly labeled proxy
node because the chapter is systems-level rather than parcel-specific.
"""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


DEFAULT_GENE_PANEL = [
    "SLC6A2",  # norepinephrine transporter
    "DBH",     # dopamine beta-hydroxylase / NE synthesis
    "ADRA2A",  # alpha-2 adrenergic receptor
    "COMT",    # catecholamine metabolism / control-related phenotypes
    "DRD2",    # dopamine D2 receptor
    "SLC6A4",  # serotonin transporter
    "MAOA",    # monoamine metabolism / impulsivity relevance
    "BDNF",    # neuroplasticity
    "CRHR1",   # stress-axis signaling
    "OPRM1",   # reward / relief-related opioid signaling
]


class KleptomaniaModel:
    """
    Atlas-grounded research scaffold for Kleptomania.

    Notes
    -----
    - This is a mechanistic interpretation of a chapter, not a validated disease model.
    - Direct molecular evidence for kleptomania is sparse, so the gene panel is exploratory.
    - Some nodes are explicit proxies when the chapter names a system but not a precise parcel.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
        connectivity_subjects_to_average: int = 8,
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
        self.connectivity_subjects_to_average = max(1, int(connectivity_subjects_to_average))

        # Disorder worksheet distilled from the chapter.
        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Exploratory polygenic liability for impulsivity, reward sensitivity, and stress reactivity."
            ),
            "acute_stress_trigger": (
                "Immediate stressor or arousal trigger preceding an urge episode."
            ),
            "chronic_stress_load": (
                "Sustained stress burden that weakens control and sensitizes arousal systems."
            ),
            "novelty_seeking_trait": (
                "Temperamental novelty and reward seeking linked to compulsive approach behavior."
            ),
            "emotional_reactivity_trait": (
                "High emotional reactivity / harm-avoidant arousal sensitivity that amplifies tension."
            ),
            "poor_self_control_trait": (
                "Temperamental weakness of self-regulation and delay tolerance."
            ),
            "family_emotional_neglect": (
                "Parental indifference or lack of love as a developmental risk load for regulation circuits."
            ),
            "recovery_support": (
                "Protective structure, treatment engagement, and supportive relationships."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "trait_impulsivity_liability": (
                "Inherited/developmental impulsivity and preference for immediate over delayed outcomes."
            ),
            "noradrenergic_arousal": (
                "Hyper-noradrenergic stress arousal state that raises tension before the act."
            ),
            "pfc_subcortical_disconnect": (
                "Reduced integrity of prefrontal communication with limbic and striatal systems."
            ),
            "frontostriatal_control_failure": (
                "Weak top-down control over urges and immediate reward seeking."
            ),
            "frontolimbic_dysregulation": (
                "Maladaptive coupling of control failure with fear, tension, and interoceptive salience."
            ),
            "dopaminergic_reward_drive": (
                "Reward/salience pull and anticipated gratification or relief from the act."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "tension_preceding_theft": (
                "Escalating tension, anxiety, or restlessness before stealing."
            ),
            "urge_to_steal": (
                "Compelling desire to steal despite anticipated consequences."
            ),
            "impaired_impulse_resistance": (
                "Failure to inhibit the act when the urge is present."
            ),
            "gratification_relief_after_theft": (
                "Post-act pleasure or rapid relief after theft."
            ),
            "repetitive_stealing_cycle": (
                "Self-reinforcing repetition of the tension-relief loop."
            ),
        }

        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "insula": [
                "Area Ia1 (Insula) left",
                "Area Ia2 (Insula) left",
                "Area Id7 (Insula) left",
                "insula",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo7 (OFC) left",
                "Area Fo5 (OFC) left",
                "Area Fo6 (OFC) left",
                "orbitofrontal",
                "OFC",
            ],
            "acc": [
                "Area p24ab (pACC) left",
                "Area p24c (pACC) left",
                "Area p32 (pACC) left",
                "Area 33 (ACC) left",
                "anterior cingulate",
                "p24",
            ],
            "ventral_striatum_proxy": [
                "ventral striatum",
                "BST (Bed Nucleus) left",
                "bed nucleus",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "amygdala": (
                "Threat/arousal node for tension, anxiety, and negative affect before theft."
            ),
            "insula": (
                "Interoceptive urge and aversive-body-state node for rising tension."
            ),
            "ofc": (
                "Orbitofrontal consequence-evaluation and urge-suppression node."
            ),
            "acc": (
                "Anterior cingulate control/conflict node for action monitoring and inhibition."
            ),
            "ventral_striatum_proxy": (
                "Reward/salience proxy for anticipated gratification and relief after theft."
            ),
        }
        self.proxy_regions = {"ventral_striatum_proxy"}

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "trait_impulsivity_liability",
                "relation": "contributes speculative heritable liability for impulsivity and poor control",
                "kleptomania_change": "increased",
            },
            {
                "source": "novelty_seeking_trait",
                "target": "trait_impulsivity_liability",
                "relation": "increases approach bias toward immediate reward and novelty",
                "kleptomania_change": "increased",
            },
            {
                "source": "poor_self_control_trait",
                "target": "trait_impulsivity_liability",
                "relation": "directly weakens inhibition of prepotent urges",
                "kleptomania_change": "increased",
            },
            {
                "source": "acute_stress_trigger",
                "target": "noradrenergic_arousal",
                "relation": "acute stress escalates fight-or-flight arousal before the act",
                "kleptomania_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "noradrenergic_arousal",
                "relation": "sustained stress sensitizes hyper-arousal and vigilance",
                "kleptomania_change": "increased",
            },
            {
                "source": "emotional_reactivity_trait",
                "target": "noradrenergic_arousal",
                "relation": "high affective reactivity intensifies anxious arousal",
                "kleptomania_change": "increased",
            },
            {
                "source": "family_emotional_neglect",
                "target": "pfc_subcortical_disconnect",
                "relation": "developmental adversity may shape regulation circuitry and communication",
                "kleptomania_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "pfc_subcortical_disconnect",
                "relation": "chronic stress may contribute to structural or functional disconnectivity",
                "kleptomania_change": "increased",
            },
            {
                "source": "noradrenergic_arousal",
                "target": "frontostriatal_control_failure",
                "relation": "stress-linked norepinephrine impairs prefrontal impulse control",
                "kleptomania_change": "increased",
            },
            {
                "source": "trait_impulsivity_liability",
                "target": "frontostriatal_control_failure",
                "relation": "trait impulsivity reduces resistance to the urge",
                "kleptomania_change": "increased",
            },
            {
                "source": "pfc_subcortical_disconnect",
                "target": "frontostriatal_control_failure",
                "relation": "weaker PFC-subcortical communication undermines behavioral inhibition",
                "kleptomania_change": "increased",
            },
            {
                "source": "noradrenergic_arousal",
                "target": "frontolimbic_dysregulation",
                "relation": "hyper-arousal couples control loss to stronger limbic reactivity",
                "kleptomania_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "frontolimbic_dysregulation",
                "relation": "control failure permits emotion-driven responding",
                "kleptomania_change": "increased",
            },
            {
                "source": "novelty_seeking_trait",
                "target": "dopaminergic_reward_drive",
                "relation": "novelty/reward seeking strengthens anticipated gratification",
                "kleptomania_change": "increased",
            },
            {
                "source": "trait_impulsivity_liability",
                "target": "dopaminergic_reward_drive",
                "relation": "impulsivity amplifies immediate reward salience",
                "kleptomania_change": "increased",
            },
            {
                "source": "noradrenergic_arousal",
                "target": "amygdala",
                "relation": "stress arousal potentiates amygdala-driven fear and negative affect",
                "kleptomania_change": "increased",
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "insula",
                "relation": "maladaptive regulation amplifies interoceptive urge and tension signaling",
                "kleptomania_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "ofc",
                "relation": "control failure is expressed in orbitofrontal dysfunction",
                "kleptomania_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "acc",
                "relation": "control failure is expressed in anterior cingulate conflict-monitoring dysfunction",
                "kleptomania_change": "increased",
            },
            {
                "source": "dopaminergic_reward_drive",
                "target": "ventral_striatum_proxy",
                "relation": "reward/salience circuitry proxies anticipated gratification and relief",
                "kleptomania_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "tension_preceding_theft",
                "relation": "negative-affect and threat reactivity intensify pre-theft tension",
                "kleptomania_change": "increased",
            },
            {
                "source": "insula",
                "target": "tension_preceding_theft",
                "relation": "interoceptive arousal contributes to rising bodily tension and restlessness",
                "kleptomania_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "impaired_impulse_resistance",
                "relation": "weak top-down control reduces resistance to stealing",
                "kleptomania_change": "increased",
            },
            {
                "source": "ventral_striatum_proxy",
                "target": "urge_to_steal",
                "relation": "reward salience strengthens the urge to act",
                "kleptomania_change": "increased",
            },
            {
                "source": "tension_preceding_theft",
                "target": "urge_to_steal",
                "relation": "rising aversive tension pressures behavioral discharge",
                "kleptomania_change": "increased",
            },
            {
                "source": "dopaminergic_reward_drive",
                "target": "gratification_relief_after_theft",
                "relation": "reward circuitry contributes to gratification after the act",
                "kleptomania_change": "increased",
            },
            {
                "source": "tension_preceding_theft",
                "target": "gratification_relief_after_theft",
                "relation": "stealing discharges the aversive tension state, creating relief",
                "kleptomania_change": "increased",
            },
            {
                "source": "gratification_relief_after_theft",
                "target": "repetitive_stealing_cycle",
                "relation": "post-act reward and relief reinforce future repetition",
                "kleptomania_change": "increased",
            },
            {
                "source": "recovery_support",
                "target": "noradrenergic_arousal",
                "relation": "support and structure buffer stress-linked hyper-arousal",
                "kleptomania_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "frontostriatal_control_failure",
                "relation": "supportive treatment and structure strengthen control capacity",
                "kleptomania_change": "decreased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame(self.edge_table)
        self._pmap = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _normalize_label_text(text: Any) -> str:
        s = str(text).lower().strip()
        for token in [
            " left",
            " right",
            " (pacc)",
            " (sacc)",
            " (acc)",
            " (ofc)",
            " (insula)",
            " (amygdala)",
        ]:
            s = s.replace(token, "")
        return " ".join(s.split())

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _modality_candidates(self, kind: str) -> List[Any]:
        candidates: List[Any] = []
        try:
            if kind == "receptor":
                candidates.append(siibra.features.molecular.ReceptorDensityFingerprint)
            elif kind == "gene":
                candidates.append(siibra.features.molecular.GeneExpressions)
            elif kind == "connectivity":
                candidates.append(siibra.features.connectivity.StreamlineCounts)
        except Exception:
            pass

        if kind == "receptor":
            candidates.append("receptor density fingerprint")
        elif kind == "gene":
            candidates.append("gene expressions")
        elif kind == "connectivity":
            candidates.append("StreamlineCounts")
        return candidates

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
            matches = list(
                self.atlas.find_regions(
                    query,
                    all_versions=False,
                    filter_children=False,
                    find_topmost=False,
                )
            )
        except Exception:
            try:
                matches = list(self.parcellation.find(query))
            except Exception:
                matches = []

        out: List[Any] = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower() or self._name_of(self.parcellation) == str(parc_name):
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "insula",
            "ventral striatum",
            "anterior cingulate",
            "orbitofrontal cortex",
            "cingulate gyrus, frontal part",
        } else 0
        proxy_penalty = 1 if "bed nucleus" in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation)
            except Exception:
                pass
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation_spec)
            except Exception:
                pass
            try:
                if hasattr(self.parcellation, "get_region"):
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

    def _point_to_tuple(self, point: Any) -> Optional[Tuple[float, float, float]]:
        if point is None:
            return None
        for attr in ("coordinate", "coordinates", "xyz"):
            value = getattr(point, attr, None)
            if value is not None:
                try:
                    return tuple(float(x) for x in value)  # type: ignore[arg-type]
                except Exception:
                    pass
        try:
            return tuple(float(x) for x in point)  # type: ignore[arg-type]
        except Exception:
            return None

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid_xyz = self._point_to_tuple(getattr(main, "centroid", None))
        try:
            volume_mm3 = float(getattr(main, "volume", float("nan")))
        except Exception:
            volume_mm3 = None
        return centroid_xyz, volume_mm3

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            try:
                df = feat.data.copy()
                if isinstance(df, pd.Series):
                    df = df.to_frame(name="value")
                if not isinstance(df, pd.DataFrame):
                    continue
                df = df.reset_index()
                if "index" in df.columns and "receptor" not in df.columns:
                    df = df.rename(columns={"index": "receptor"})
                return df
            except Exception:
                continue
        return pd.DataFrame()

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            try:
                df = feat.data.copy()
            except Exception:
                continue
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            lower_cols = {str(c).lower(): c for c in df.columns}
            required = {"gene", "level", "zscore"}
            if required.issubset(lower_cols):
                gene_col = lower_cols["gene"]
                level_col = lower_cols["level"]
                zscore_col = lower_cols["zscore"]
                try:
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
                except Exception:
                    pass
            return df.reset_index(drop=True)
        return pd.DataFrame()

    def _first_dataframe(self, value: Any) -> Optional[pd.DataFrame]:
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if isinstance(value, pd.Series):
            return value.to_frame()
        return None

    def _mean_connectivity_from_compound(self, compound: Any) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        try:
            direct = self._first_dataframe(getattr(compound, "data", None))
            if direct is not None:
                return direct
        except Exception:
            pass

        try:
            iterable: Iterable[Any] = compound[: self.connectivity_subjects_to_average]
        except Exception:
            iterable = []
            try:
                iterable = [compound[i] for i in range(self.connectivity_subjects_to_average)]
            except Exception:
                iterable = []

        for element in iterable:
            try:
                df = self._first_dataframe(getattr(element, "data", None))
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception:
                continue

        if not frames:
            return pd.DataFrame()
        if len(frames) == 1:
            return frames[0]

        try:
            total = frames[0].astype(float).copy()
            for df in frames[1:]:
                total = total.add(df.astype(float), fill_value=0.0)
            return total / float(len(frames))
        except Exception:
            return frames[0]

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort),
            feats[0],
        )
        self._connectivity_matrix = self._mean_connectivity_from_compound(compound)
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._name_of(region)
        region_norm = self._normalize_label_text(region_name)

        for label in labels:
            if self._name_of(label) == region_name:
                return label

        exactish = []
        fuzzy = []
        for label in labels:
            label_name = self._name_of(label)
            label_norm = self._normalize_label_text(label_name)
            if label_norm == region_norm:
                exactish.append(label)
            elif region_norm in label_norm or label_norm in region_norm:
                fuzzy.append(label)

        pool = exactish if exactish else fuzzy
        if not pool:
            return None
        return sorted(pool, key=lambda x: self._region_rank(x))[0]

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        row_label = self._match_region_label(list(matrix.index), region)
        col_label = self._match_region_label(list(matrix.columns), region)
        label = row_label if row_label is not None else col_label
        axis = "index" if row_label is not None else "columns"
        if label is None:
            return pd.DataFrame()

        try:
            series = matrix.loc[label] if axis == "index" else matrix[label]
            if not isinstance(series, pd.Series):
                return pd.DataFrame()
            df = series.dropna().sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != self._name_of(region)].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        pairs: List[Tuple[str, Any]] = []
        for key, region in self.region_objects.items():
            label = self._match_region_label(list(matrix.index), region)
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
            if label is not None:
                pairs.append((key, label))

        if not pairs:
            return pd.DataFrame()

        labels = [label for _, label in pairs]
        keys = [key for key, _ in pairs]
        try:
            sub = matrix.loc[labels, labels].copy()
            sub.index = keys
            sub.columns = keys
            return sub
        except Exception:
            return pd.DataFrame()

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
        self._connectivity_matrix = None

        for key, desc in self.input_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "input",
                    "description": desc,
                    "region_role": None,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            description = self.region_node_descriptions.get(key, "Atlas-backed circuit node")
            region_role = "proxy" if key in self.proxy_regions else "atlas_backed"

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
                        "description": f"{description} (unresolved in this environment)",
                        "region_role": region_role,
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
                    "node_type": "region",
                    "description": description,
                    "region_role": region_role,
                    "atlas_region": self._name_of(region),
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
                    "region_role": None,
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
                    "region_role": None,
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
        genetic_vulnerability: float = 0.50,
        acute_stress_trigger: float = 0.50,
        chronic_stress_load: float = 0.50,
        novelty_seeking_trait: float = 0.50,
        emotional_reactivity_trait: float = 0.50,
        poor_self_control_trait: float = 0.50,
        family_emotional_neglect: float = 0.50,
        recovery_support: float = 0.50,
    ) -> Dict[str, pd.Series]:
        """
        Transparent one-pass simulator on normalized 0..1 inputs.

        The order is:
        inputs -> latent biology -> regional state -> symptoms -> phenotypes

        These are not probabilities, diagnoses, or treatment predictions.
        They are interpretable scaffold scores reflecting chapter logic.
        """
        c = self._clip01

        inputs = pd.Series(
            {
                "genetic_vulnerability": c(genetic_vulnerability),
                "acute_stress_trigger": c(acute_stress_trigger),
                "chronic_stress_load": c(chronic_stress_load),
                "novelty_seeking_trait": c(novelty_seeking_trait),
                "emotional_reactivity_trait": c(emotional_reactivity_trait),
                "poor_self_control_trait": c(poor_self_control_trait),
                "family_emotional_neglect": c(family_emotional_neglect),
                "recovery_support": c(recovery_support),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["trait_impulsivity_liability"] = c(
            0.32 * inputs["genetic_vulnerability"]
            + 0.22 * inputs["novelty_seeking_trait"]
            + 0.31 * inputs["poor_self_control_trait"]
            + 0.15 * inputs["family_emotional_neglect"]
            - 0.10 * inputs["recovery_support"]
        )
        latents["noradrenergic_arousal"] = c(
            0.38 * inputs["acute_stress_trigger"]
            + 0.28 * inputs["chronic_stress_load"]
            + 0.24 * inputs["emotional_reactivity_trait"]
            - 0.18 * inputs["recovery_support"]
        )
        latents["pfc_subcortical_disconnect"] = c(
            0.30 * inputs["chronic_stress_load"]
            + 0.22 * inputs["family_emotional_neglect"]
            + 0.20 * inputs["genetic_vulnerability"]
            + 0.12 * inputs["poor_self_control_trait"]
            - 0.12 * inputs["recovery_support"]
        )
        latents["frontostriatal_control_failure"] = c(
            0.34 * latents["trait_impulsivity_liability"]
            + 0.30 * latents["noradrenergic_arousal"]
            + 0.22 * latents["pfc_subcortical_disconnect"]
            + 0.10 * inputs["chronic_stress_load"]
            - 0.22 * inputs["recovery_support"]
        )
        latents["frontolimbic_dysregulation"] = c(
            0.34 * latents["frontostriatal_control_failure"]
            + 0.28 * latents["noradrenergic_arousal"]
            + 0.20 * inputs["emotional_reactivity_trait"]
            + 0.12 * inputs["family_emotional_neglect"]
            - 0.18 * inputs["recovery_support"]
        )
        latents["dopaminergic_reward_drive"] = c(
            0.38 * inputs["novelty_seeking_trait"]
            + 0.26 * latents["trait_impulsivity_liability"]
            + 0.16 * inputs["genetic_vulnerability"]
            + 0.08 * inputs["acute_stress_trigger"]
            - 0.10 * inputs["recovery_support"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["amygdala"] = c(
            0.52 * latents["noradrenergic_arousal"]
            + 0.24 * latents["frontolimbic_dysregulation"]
            + 0.16 * inputs["emotional_reactivity_trait"]
        )
        regional_state["insula"] = c(
            0.42 * latents["frontolimbic_dysregulation"]
            + 0.26 * latents["noradrenergic_arousal"]
            + 0.14 * inputs["acute_stress_trigger"]
            + 0.08 * inputs["emotional_reactivity_trait"]
        )
        regional_state["ofc"] = c(
            0.56 * latents["frontostriatal_control_failure"]
            + 0.20 * latents["pfc_subcortical_disconnect"]
            + 0.10 * inputs["chronic_stress_load"]
            - 0.18 * inputs["recovery_support"]
        )
        regional_state["acc"] = c(
            0.46 * latents["frontostriatal_control_failure"]
            + 0.24 * latents["frontolimbic_dysregulation"]
            + 0.10 * inputs["chronic_stress_load"]
            - 0.16 * inputs["recovery_support"]
        )
        regional_state["ventral_striatum_proxy"] = c(
            0.54 * latents["dopaminergic_reward_drive"]
            + 0.18 * inputs["novelty_seeking_trait"]
            + 0.12 * latents["trait_impulsivity_liability"]
            + 0.08 * inputs["acute_stress_trigger"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["tension_preceding_theft"] = c(
            0.50 * latents["noradrenergic_arousal"]
            + 0.24 * regional_state["amygdala"]
            + 0.20 * regional_state["insula"]
            - 0.10 * inputs["recovery_support"]
        )
        symptoms["urge_to_steal"] = c(
            0.34 * symptoms["tension_preceding_theft"]
            + 0.24 * regional_state["ventral_striatum_proxy"]
            + 0.18 * regional_state["insula"]
            + 0.16 * latents["frontostriatal_control_failure"]
            + 0.06 * inputs["novelty_seeking_trait"]
        )
        symptoms["impaired_impulse_resistance"] = c(
            0.36 * latents["frontostriatal_control_failure"]
            + 0.20 * regional_state["ofc"]
            + 0.16 * regional_state["acc"]
            + 0.16 * latents["trait_impulsivity_liability"]
            - 0.12 * inputs["recovery_support"]
        )
        symptoms["gratification_relief_after_theft"] = c(
            0.42 * regional_state["ventral_striatum_proxy"]
            + 0.24 * latents["dopaminergic_reward_drive"]
            + 0.24 * symptoms["tension_preceding_theft"]
        )
        symptoms["repetitive_stealing_cycle"] = c(
            0.28 * symptoms["urge_to_steal"]
            + 0.24 * symptoms["impaired_impulse_resistance"]
            + 0.22 * symptoms["gratification_relief_after_theft"]
            + 0.16 * symptoms["tension_preceding_theft"]
            + 0.08 * latents["frontolimbic_dysregulation"]
        )

        phenotypes = pd.Series(dtype=float, name="phenotypes")
        phenotypes["stress_triggered_episode_profile"] = c(
            self._mean(
                [
                    symptoms["tension_preceding_theft"],
                    symptoms["urge_to_steal"],
                    symptoms["impaired_impulse_resistance"],
                ]
            )
        )
        phenotypes["tension_release_cycle_profile"] = c(
            self._mean(
                [
                    symptoms["tension_preceding_theft"],
                    symptoms["gratification_relief_after_theft"],
                    symptoms["repetitive_stealing_cycle"],
                ]
            )
        )
        phenotypes["trait_impulsivity_profile"] = c(
            self._mean(
                [
                    latents["trait_impulsivity_liability"],
                    latents["frontostriatal_control_failure"],
                    symptoms["impaired_impulse_resistance"],
                ]
            )
        )
        phenotypes["reward_discharge_profile"] = c(
            self._mean(
                [
                    latents["dopaminergic_reward_drive"],
                    regional_state["ventral_striatum_proxy"],
                    symptoms["gratification_relief_after_theft"],
                ]
            )
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

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in getattr(assignments, "columns", []):
                return assignments.sort_values(candidate, ascending=False).reset_index(drop=True)
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Optional[Any]:
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        try:
            return region.get_regional_mask(self.space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(self.space)
            except Exception:
                return None


if __name__ == "__main__":
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 160)

    model = KleptomaniaModel()
    built = model.build()

    print("\n=== Nodes ===")
    print(
        built["nodes"][
            [
                "key",
                "node_type",
                "region_role",
                "atlas_region",
                "feature_summary",
            ]
        ].to_string(index=False)
    )

    print("\n=== Edges ===")
    print(built["edges"].to_string(index=False))

    print("\n=== Region suggestions for 'insula' ===")
    print(model.suggest_regions("insula").head(10).to_string(index=False))

    for key in ["amygdala", "insula", "ofc", "acc", "ventral_striatum_proxy"]:
        print(f"\n=== Region evidence: {key} ===")
        if key in built["regions"]:
            region = built["regions"][key]
            print("Resolved region:", getattr(region, "name", region))
        else:
            print("Resolved region: None")
        print("Receptors:")
        rec = built["receptors"].get(key, pd.DataFrame())
        print(rec.head(10).to_string(index=False) if not rec.empty else "<no receptor table>")
        print("Genes:")
        gene = built["genes"].get(key, pd.DataFrame())
        print(gene.head(10).to_string(index=False) if not gene.empty else "<no gene table>")
        print("Connectivity profile:")
        conn = built["connectivity_profiles"].get(key, pd.DataFrame())
        print(conn.head(10).to_string(index=False) if not conn.empty else "<no connectivity profile>")

    print("\n=== Circuit connectivity among resolved nodes ===")
    cmat = built["circuit_connectivity"]
    print(cmat.to_string() if not cmat.empty else "<no circuit connectivity matrix>")

    simulation = model.simulate(
        genetic_vulnerability=0.70,
        acute_stress_trigger=0.85,
        chronic_stress_load=0.60,
        novelty_seeking_trait=0.72,
        emotional_reactivity_trait=0.78,
        poor_self_control_trait=0.74,
        family_emotional_neglect=0.48,
        recovery_support=0.20,
    )

    print("\n=== Simulated inputs ===")
    print(simulation["inputs"].to_string())
    print("\n=== Simulated latents ===")
    print(simulation["latents"].to_string())
    print("\n=== Simulated regional state ===")
    print(simulation["regional_state"].to_string())
    print("\n=== Simulated symptoms ===")
    print(simulation["symptoms"].to_string())
    print("\n=== Simulated phenotypes ===")
    print(simulation["phenotypes"].to_string())

    # Optional coordinate assignment example:
    # assignments = model.assign_mni_point((-6.0, 24.0, -12.0))
    # print(assignments.head(10).to_string(index=False))
