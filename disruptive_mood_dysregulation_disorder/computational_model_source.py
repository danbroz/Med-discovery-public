from __future__ import annotations

"""
Disruptive Mood Dysregulation Disorder (DMDD) atlas-grounded siibra scaffold.

This script turns a chapter-level biological summary of DMDD into a transparent,
research-oriented mechanistic graph. The source chapter explicitly notes that
DMDD-specific neurobiology is still sparse, so this scaffold encodes an
inferential model drawn from the chapter's links to irritability, ADHD,
externalizing dyscontrol, anxiety, and depressive liability.

Important:
- This is a research scaffold, not a diagnostic or treatment tool.
- Higher simulated values indicate greater dysregulation burden or symptom
  pressure unless otherwise noted.
- Several nodes are deliberately modeled as atlas-backed proxies because the
  chapter is systems-level rather than parcel-specific.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except Exception as exc:  # pragma: no cover - import guard for environments without siibra
    siibra = None  # type: ignore[assignment]
    _SIIBRA_IMPORT_ERROR = exc
else:  # pragma: no cover - import success path is environment-specific
    _SIIBRA_IMPORT_ERROR = None


DEFAULT_DMDD_GENE_PANEL = [
    "SLC6A3",   # dopamine transporter
    "DRD4",     # dopamine receptor D4
    "DRD2",     # dopamine receptor D2
    "SLC6A2",   # norepinephrine transporter
    "COMT",     # catecholamine metabolism
    "MAOA",     # monoamine degradation / impulsive aggression literature
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor 1A
    "TPH2",     # serotonin synthesis
    "BDNF",     # neuroplasticity / stress sensitivity
    "CACNA1C",  # mood dysregulation / affective liability
]


class DisruptiveMoodDysregulationDisorderModel:
    """
    Atlas-grounded research scaffold for DMDD.

    The chapter motivating this scaffold does not claim a settled DMDD-specific
    biological mechanism. Instead, it frames DMDD as a severe irritability and
    emotion-regulation syndrome whose likely biology overlaps with:

    - catecholaminergic control systems implicated in ADHD,
    - serotonergic mood-regulation systems implicated in depression/anxiety,
    - fronto-striatal executive control networks,
    - amygdala-prefrontal emotion-regulation circuitry,
    - stress-sensitized developmental dysregulation.

    This class therefore encodes a conservative, interpretable mechanistic model
    using atlas-backed regions where feasible and clearly marked proxies where
    the chapter remains systems-level.
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
                "siibra is required to use this scaffold. Install it in your Python "
                "environment before running the model."
            ) from _SIIBRA_IMPORT_ERROR

        # Compatibility-first atlas initialization.
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

        # Region nodes are conservative. Only the amygdala is directly named with
        # a reasonably specific atlas target. PFC, striatal, and cerebellar nodes
        # remain explicit proxies because the chapter is not parcel-specific.
        self.region_candidates: Dict[str, List[str]] = {
            "pfc_control": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 9 left",
                "Area 45 left",
                "prefrontal cortex",
            ],
            "striatum_proxy": [
                "nucleus accumbens left",
                "caudate left",
                "putamen left",
                "striatum",
                "basal ganglia",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "cerebellum_proxy": [
                "cerebellum left",
                "cerebellum",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "pfc_control": (
                "Proxy for prefrontal executive-control regions supporting response inhibition, "
                "attention regulation, and top-down emotion control."
            ),
            "striatum_proxy": (
                "Proxy for basal ganglia / striatal circuitry involved in action selection, "
                "reward-frustration processing, and behavioral gating."
            ),
            "amygdala": (
                "Amygdala node for threat/frustration reactivity and amplified anger responses."
            ),
            "cerebellum_proxy": (
                "Proxy for cerebellar contributions to timing, prediction, and broader regulatory load."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Inherited liability spanning irritability, ADHD-like dyscontrol, depression, and anxiety risk."
            ),
            "early_life_stress": (
                "Early adversity or trauma burden that may weaken amygdala-prefrontal regulation."
            ),
            "adhd_externalizing_load": (
                "Developmental impulsivity, hyperactivity, and disruptive-behavior burden overlapping with DMDD."
            ),
            "mood_anxiety_liability": (
                "Internalizing liability contributing to persistent irritability and later depressive/anxious outcomes."
            ),
            "acute_stress_arousal": (
                "Current stress-related arousal that raises reactivity and reduces regulatory capacity."
            ),
            "catecholamine_support": (
                "Protective support from stimulant/nonstimulant catecholaminergic treatment or equivalent support."
            ),
            "psychosocial_regulation_support": (
                "Protective structure, psychotherapy, family support, and emotion-regulation scaffolding."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "catecholaminergic_dysregulation": (
                "Dopamine/norepinephrine dysregulation weakening executive control and behavioral inhibition."
            ),
            "serotonergic_mood_regulation_deficit": (
                "Serotonergic regulation deficit plausibly contributing to baseline irritability and mood lability."
            ),
            "amygdala_pfc_disconnect": (
                "Disrupted functional coupling between prefrontal control systems and limbic emotional generators."
            ),
            "frontostriatal_control_failure": (
                "Weak top-down control across prefrontal and striatal systems, impairing inhibition and frustration tolerance."
            ),
            "threat_frustration_reactivity": (
                "Heightened anger/threat/frustration reactivity that amplifies rapid escalation into outbursts."
            ),
            "chronic_irritable_tone": (
                "Persistently elevated irritable baseline state between discrete outbursts."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "persistent_irritable_mood": "Chronic angry or irritable mood present across contexts.",
            "temper_outbursts": "Severe recurrent verbal or behavioral outbursts disproportionate to provocation.",
            "frustration_intolerance": "Low threshold for frustration and rapid escalation under blocked goals.",
            "behavioral_disinhibition": "Poor impulse control and difficulty inhibiting emotionally driven responses.",
            "depressive_anxious_trajectory": (
                "Downstream risk phenotype linking chronic childhood irritability to later internalizing disorders."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "catecholaminergic_dysregulation",
                "relation": "raises inherited vulnerability in dopamine/norepinephrine control systems",
                "dmdd_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_mood_regulation_deficit",
                "relation": "raises inherited vulnerability in mood-regulation systems",
                "dmdd_change": "increased",
            },
            {
                "source": "early_life_stress",
                "target": "amygdala_pfc_disconnect",
                "relation": "can weaken functional coupling between amygdala and prefrontal regulation",
                "dmdd_change": "increased",
            },
            {
                "source": "adhd_externalizing_load",
                "target": "catecholaminergic_dysregulation",
                "relation": "tracks with ADHD-like catecholamine dysfunction and impulsive dyscontrol",
                "dmdd_change": "increased",
            },
            {
                "source": "adhd_externalizing_load",
                "target": "frontostriatal_control_failure",
                "relation": "loads executive-control circuits involved in inhibition and action selection",
                "dmdd_change": "increased",
            },
            {
                "source": "mood_anxiety_liability",
                "target": "serotonergic_mood_regulation_deficit",
                "relation": "adds mood-lability and internalizing pressure",
                "dmdd_change": "increased",
            },
            {
                "source": "acute_stress_arousal",
                "target": "threat_frustration_reactivity",
                "relation": "acutely heightens reactivity and anger escalation",
                "dmdd_change": "increased",
            },
            {
                "source": "catecholamine_support",
                "target": "catecholaminergic_dysregulation",
                "relation": "reduces catecholamine-related control failure",
                "dmdd_change": "decreased",
            },
            {
                "source": "psychosocial_regulation_support",
                "target": "amygdala_pfc_disconnect",
                "relation": "supports more effective emotion regulation and buffering",
                "dmdd_change": "decreased",
            },
            {
                "source": "catecholaminergic_dysregulation",
                "target": "frontostriatal_control_failure",
                "relation": "impairs inhibition, attention control, and action regulation",
                "dmdd_change": "increased",
            },
            {
                "source": "serotonergic_mood_regulation_deficit",
                "target": "chronic_irritable_tone",
                "relation": "supports persistent baseline irritability and mood instability",
                "dmdd_change": "increased",
            },
            {
                "source": "amygdala_pfc_disconnect",
                "target": "threat_frustration_reactivity",
                "relation": "weakens top-down damping of emotional responses",
                "dmdd_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "pfc_control",
                "relation": "increases dysfunction burden in executive-control regions",
                "dmdd_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "striatum_proxy",
                "relation": "increases dyscontrol burden in fronto-striatal action-selection circuitry",
                "dmdd_change": "increased",
            },
            {
                "source": "amygdala_pfc_disconnect",
                "target": "amygdala",
                "relation": "permits greater unmodulated limbic reactivity",
                "dmdd_change": "increased",
            },
            {
                "source": "threat_frustration_reactivity",
                "target": "amygdala",
                "relation": "amplifies limbic reactivity during anger/frustration states",
                "dmdd_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "cerebellum_proxy",
                "relation": "increases broader regulatory burden on distributed timing and coordination systems",
                "dmdd_change": "increased",
            },
            {
                "source": "chronic_irritable_tone",
                "target": "persistent_irritable_mood",
                "relation": "drives the chronic irritable baseline",
                "dmdd_change": "increased",
            },
            {
                "source": "threat_frustration_reactivity",
                "target": "temper_outbursts",
                "relation": "precipitates explosive anger responses",
                "dmdd_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "temper_outbursts",
                "relation": "reduces inhibitory control over escalating affect",
                "dmdd_change": "increased",
            },
            {
                "source": "threat_frustration_reactivity",
                "target": "frustration_intolerance",
                "relation": "lowers tolerance for blocked goals and minor provocations",
                "dmdd_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "behavioral_disinhibition",
                "relation": "impairs inhibitory control and action gating",
                "dmdd_change": "increased",
            },
            {
                "source": "serotonergic_mood_regulation_deficit",
                "target": "depressive_anxious_trajectory",
                "relation": "links persistent irritability to later internalizing outcomes",
                "dmdd_change": "increased",
            },
            {
                "source": "chronic_irritable_tone",
                "target": "depressive_anxious_trajectory",
                "relation": "extends chronic irritability into later depressive/anxious risk",
                "dmdd_change": "increased",
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
        try:
            matches = self.atlas.find_regions(
                query,
                all_versions=False,
                filter_children=False,
                find_topmost=False,
            )
        except Exception:
            return []

        out: List[Any] = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower():
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {"amygdala", "hippocampus", "prefrontal cortex", "cerebellum"} else 0
        proxy_penalty = 1 if "proxy" in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                region = self.atlas.get_region(spec, parcellation=self.parcellation)
                if region is not None:
                    return region
            except Exception:
                pass
            try:
                region = self.parcellation.get_region(spec)
                if region is not None:
                    return region
            except Exception:
                pass
            matches = self._julich_matches(spec)
            if matches:
                matches = sorted(matches, key=self._region_rank)
                return matches[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
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
        centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        volume = getattr(main, "volume", None)
        volume_mm3 = float(volume) if volume is not None else None
        return centroid_xyz, volume_mm3

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            try:
                df = feat.data.copy().reset_index()
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

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats and self.region_objects:
            first_region = next(iter(self.region_objects.values()))
            feats = self._safe_features_any(first_region, self._modality_candidates("connectivity"))

        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (f for f in feats if str(getattr(f, "cohort", "")).lower() == self.connectivity_cohort.lower()),
            feats[0],
        )

        candidates: List[Any] = [compound]
        try:
            candidates.extend(list(compound))
        except Exception:
            pass

        for candidate in candidates:
            try:
                data = getattr(candidate, "data", None)
                if isinstance(data, pd.DataFrame):
                    self._connectivity_matrix = data.copy()
                    return self._connectivity_matrix
            except Exception:
                continue

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        if not labels:
            return None

        region_name = region.name.lower()
        label_pairs = [(lab, self._name_of(lab).lower()) for lab in labels]

        for original, label_name in label_pairs:
            if label_name == region_name:
                return original

        for original, label_name in label_pairs:
            if region_name in label_name or label_name in region_name:
                return original

        tokens = [tok for tok in region_name.replace("(", " ").replace(")", " ").replace(",", " ").split() if tok]
        ranked = []
        for original, label_name in label_pairs:
            overlap = sum(1 for tok in tokens if tok in label_name)
            ranked.append((overlap, original))
        ranked.sort(key=lambda x: x[0], reverse=True)
        if ranked and ranked[0][0] > 0:
            return ranked[0][1]
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
            if not isinstance(series, pd.Series):
                return pd.DataFrame()
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return pairwise connectivity values among the scaffold's resolved circuit nodes.

        The output is a tidy DataFrame. Missing or unresolved regions are skipped.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        matched_rows: Dict[str, Any] = {}
        matched_cols: Dict[str, Any] = {}

        for key, region in self.region_objects.items():
            row_label = self._match_region_label(list(matrix.index), region)
            col_label = self._match_region_label(list(matrix.columns), region)
            if row_label is not None:
                matched_rows[key] = row_label
            if col_label is not None:
                matched_cols[key] = col_label

        keys = list(self.region_objects.keys())
        for src in keys:
            for dst in keys:
                if src == dst:
                    continue
                row_label = matched_rows.get(src)
                col_label = matched_cols.get(dst)
                if row_label is None or col_label is None:
                    continue
                try:
                    value = matrix.loc[row_label, col_label]
                except Exception:
                    continue
                rows.append(
                    {
                        "source_key": src,
                        "source_region": self.region_objects[src].name,
                        "target_key": dst,
                        "target_region": self.region_objects[dst].name,
                        "value": float(value),
                    }
                )

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_DMDD_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
        """
        Build the atlas-grounded graph and collect multimodal summaries.
        """
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
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            desc = self.region_descriptions.get(key, "Atlas-backed circuit node")

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
                        "description": f"{desc} Unresolved in this siibra environment; keep as an explicit proxy.",
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
                    "description": desc,
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
        genetic_vulnerability: float = 0.55,
        early_life_stress: float = 0.35,
        adhd_externalizing_load: float = 0.60,
        mood_anxiety_liability: float = 0.45,
        acute_stress_arousal: float = 0.50,
        catecholamine_support: float = 0.35,
        psychosocial_regulation_support: float = 0.30,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator for the scaffold.

        Parameters are 0..1. Protective variables subtract from dysregulation.
        The computation is acyclic:
            inputs -> latent biology -> regional burden -> symptoms -> phenotype summaries
        """
        inputs = {
            "genetic_vulnerability": self._clip01(genetic_vulnerability),
            "early_life_stress": self._clip01(early_life_stress),
            "adhd_externalizing_load": self._clip01(adhd_externalizing_load),
            "mood_anxiety_liability": self._clip01(mood_anxiety_liability),
            "acute_stress_arousal": self._clip01(acute_stress_arousal),
            "catecholamine_support": self._clip01(catecholamine_support),
            "psychosocial_regulation_support": self._clip01(psychosocial_regulation_support),
        }

        latents = {
            "catecholaminergic_dysregulation": self._clip01(
                0.42 * inputs["genetic_vulnerability"]
                + 0.36 * inputs["adhd_externalizing_load"]
                + 0.12 * inputs["acute_stress_arousal"]
                - 0.28 * inputs["catecholamine_support"]
            ),
            "serotonergic_mood_regulation_deficit": self._clip01(
                0.33 * inputs["genetic_vulnerability"]
                + 0.28 * inputs["mood_anxiety_liability"]
                + 0.18 * inputs["early_life_stress"]
                + 0.14 * inputs["acute_stress_arousal"]
                - 0.18 * inputs["psychosocial_regulation_support"]
            ),
        }
        latents["amygdala_pfc_disconnect"] = self._clip01(
            0.40 * inputs["early_life_stress"]
            + 0.22 * inputs["acute_stress_arousal"]
            + 0.20 * inputs["mood_anxiety_liability"]
            + 0.10 * inputs["genetic_vulnerability"]
            - 0.24 * inputs["psychosocial_regulation_support"]
        )
        latents["frontostriatal_control_failure"] = self._clip01(
            0.43 * latents["catecholaminergic_dysregulation"]
            + 0.22 * inputs["adhd_externalizing_load"]
            + 0.15 * inputs["acute_stress_arousal"]
            + 0.12 * latents["amygdala_pfc_disconnect"]
            - 0.16 * inputs["catecholamine_support"]
            - 0.10 * inputs["psychosocial_regulation_support"]
        )
        latents["threat_frustration_reactivity"] = self._clip01(
            0.44 * latents["amygdala_pfc_disconnect"]
            + 0.24 * latents["serotonergic_mood_regulation_deficit"]
            + 0.20 * inputs["acute_stress_arousal"]
            + 0.10 * inputs["mood_anxiety_liability"]
        )
        latents["chronic_irritable_tone"] = self._clip01(
            0.36 * latents["serotonergic_mood_regulation_deficit"]
            + 0.30 * latents["threat_frustration_reactivity"]
            + 0.18 * latents["catecholaminergic_dysregulation"]
            + 0.08 * inputs["genetic_vulnerability"]
            - 0.16 * inputs["psychosocial_regulation_support"]
        )

        regional_state = {
            # Regional values represent dysregulation burden rather than healthy function.
            "pfc_control": self._clip01(
                0.56 * latents["frontostriatal_control_failure"]
                + 0.28 * latents["amygdala_pfc_disconnect"]
                + 0.08 * inputs["acute_stress_arousal"]
                - 0.16 * inputs["catecholamine_support"]
                - 0.12 * inputs["psychosocial_regulation_support"]
            ),
            "striatum_proxy": self._clip01(
                0.46 * latents["catecholaminergic_dysregulation"]
                + 0.30 * latents["frontostriatal_control_failure"]
                + 0.12 * inputs["acute_stress_arousal"]
                - 0.18 * inputs["catecholamine_support"]
            ),
            "amygdala": self._clip01(
                0.54 * latents["threat_frustration_reactivity"]
                + 0.24 * latents["amygdala_pfc_disconnect"]
                + 0.12 * inputs["acute_stress_arousal"]
                - 0.10 * inputs["psychosocial_regulation_support"]
            ),
            "cerebellum_proxy": self._clip01(
                0.32 * latents["frontostriatal_control_failure"]
                + 0.24 * latents["chronic_irritable_tone"]
                + 0.18 * inputs["acute_stress_arousal"]
                - 0.10 * inputs["psychosocial_regulation_support"]
            ),
        }

        symptoms = {
            "persistent_irritable_mood": self._clip01(
                0.48 * latents["chronic_irritable_tone"]
                + 0.22 * regional_state["amygdala"]
                + 0.14 * latents["serotonergic_mood_regulation_deficit"]
                + 0.08 * inputs["acute_stress_arousal"]
            ),
            "temper_outbursts": self._clip01(
                0.36 * latents["frontostriatal_control_failure"]
                + 0.34 * latents["threat_frustration_reactivity"]
                + 0.16 * regional_state["striatum_proxy"]
                + 0.08 * inputs["acute_stress_arousal"]
            ),
            "frustration_intolerance": self._clip01(
                0.36 * latents["threat_frustration_reactivity"]
                + 0.24 * latents["frontostriatal_control_failure"]
                + 0.20 * latents["chronic_irritable_tone"]
                + 0.10 * inputs["mood_anxiety_liability"]
            ),
            "behavioral_disinhibition": self._clip01(
                0.42 * latents["frontostriatal_control_failure"]
                + 0.24 * latents["catecholaminergic_dysregulation"]
                + 0.18 * regional_state["striatum_proxy"]
                - 0.10 * inputs["catecholamine_support"]
            ),
            "depressive_anxious_trajectory": self._clip01(
                0.34 * latents["serotonergic_mood_regulation_deficit"]
                + 0.26 * latents["chronic_irritable_tone"]
                + 0.18 * latents["amygdala_pfc_disconnect"]
                + 0.14 * inputs["mood_anxiety_liability"]
            ),
        }

        phenotypes = {
            "dmdd_core_profile": self._clip01(
                (symptoms["persistent_irritable_mood"] + symptoms["temper_outbursts"] + symptoms["frustration_intolerance"]) / 3.0
            ),
            "externalizing_irritability_profile": self._clip01(
                (symptoms["temper_outbursts"] + symptoms["behavioral_disinhibition"] + latents["frontostriatal_control_failure"]) / 3.0
            ),
            "stress_sensitized_profile": self._clip01(
                (latents["threat_frustration_reactivity"] + latents["amygdala_pfc_disconnect"] + regional_state["amygdala"] + inputs["acute_stress_arousal"]) / 4.0
            ),
            "depressive_irritability_profile": self._clip01(
                (symptoms["depressive_anxious_trajectory"] + latents["chronic_irritable_tone"] + latents["serotonergic_mood_regulation_deficit"]) / 3.0
            ),
        }

        return {
            "inputs": pd.Series(inputs, name="inputs"),
            "latents": pd.Series(latents, name="latents"),
            "regional_state": pd.Series(regional_state, name="regional_state"),
            "symptoms": pd.Series(symptoms, name="symptoms"),
            "phenotypes": pd.Series(phenotypes, name="phenotypes"),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Assign an MNI coordinate to Julich regions using a statistical map.
        """
        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
            except Exception:
                with siibra.QUIET:
                    self._pmap = self.atlas.get_map(
                        parcellation=self.parcellation,
                        space=self.atlas.get_space(self.assignment_space),
                        maptype="statistical",
                    )

        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break

        if "region" in assignments.columns:
            assignments = assignments.copy()
            assignments["region"] = assignments["region"].map(self._name_of)
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        """
        Return a regional mask object for a resolved region node.

        Call `.fetch()` on the returned object to obtain the image volume.
        Returns None for unresolved nodes.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(space=self.space, maptype="labelled")
            except Exception:
                return None


if __name__ == "__main__":
    # Example usage. This block requires a working siibra installation.
    if siibra is None:
        print("siibra is not installed in this environment. Install siibra, then rerun this script to build the atlas-grounded DMDD scaffold.")
        raise SystemExit(0)

    model = DisruptiveMoodDysregulationDisorderModel()
    built = model.build(connectivity_rows=10)

    print("\n=== NODES ===")
    print(built["nodes"][["key", "node_type", "label", "atlas_region", "feature_summary"]].to_string(index=False))

    print("\n=== EDGES ===")
    print(built["edges"][["source", "target", "relation", "dmdd_change"]].to_string(index=False))

    print("\n=== REGION RESOLUTION ===")
    if model.region_objects:
        for key, region in model.region_objects.items():
            print(f"{key}: {region.name}")
    else:
        print("No regions resolved in this environment.")

    for node_key in ["amygdala", "pfc_control", "striatum_proxy"]:
        receptor_df = built["receptors"].get(node_key, pd.DataFrame())
        gene_df = built["genes"].get(node_key, pd.DataFrame())
        conn_df = built["connectivity_profiles"].get(node_key, pd.DataFrame())

        print(f"\n=== FEATURES: {node_key} ===")
        print("Receptors:")
        print(receptor_df.head().to_string(index=False) if not receptor_df.empty else "<none>")
        print("Genes:")
        print(gene_df.head().to_string(index=False) if not gene_df.empty else "<none>")
        print("Connectivity:")
        print(conn_df.head().to_string(index=False) if not conn_df.empty else "<none>")

    circuit_df = built["circuit_connectivity"]
    print("\n=== CIRCUIT CONNECTIVITY ===")
    print(circuit_df.head(12).to_string(index=False) if not circuit_df.empty else "<none>")

    sim = model.simulate(
        genetic_vulnerability=0.70,
        early_life_stress=0.45,
        adhd_externalizing_load=0.75,
        mood_anxiety_liability=0.50,
        acute_stress_arousal=0.60,
        catecholamine_support=0.40,
        psychosocial_regulation_support=0.35,
    )

    print("\n=== SIMULATION: INPUTS ===")
    print(sim["inputs"].round(3).to_string())
    print("\n=== SIMULATION: LATENTS ===")
    print(sim["latents"].round(3).sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: REGIONAL STATE ===")
    print(sim["regional_state"].round(3).sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: SYMPTOMS ===")
    print(sim["symptoms"].round(3).sort_values(ascending=False).to_string())
    print("\n=== SIMULATION: PHENOTYPES ===")
    print(sim["phenotypes"].round(3).sort_values(ascending=False).to_string())

    # Example coordinate assignment for future use:
    # print(model.assign_mni_point((-24, -4, -18)).head())
    # mask = model.region_mask("amygdala")
    # if mask is not None:
    #     nii = mask.fetch()
