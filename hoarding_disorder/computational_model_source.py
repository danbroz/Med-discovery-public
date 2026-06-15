from __future__ import annotations

"""Atlas-grounded siibra scaffold for Hoarding Disorder.

This script turns a short mechanistic chapter on Hoarding Disorder (HD) into an
interpretable research scaffold. It is designed for exploratory modeling and
multimodal atlas profiling, not for diagnosis, prognosis, or treatment.

Conceptual interpretation of the chapter
----------------------------------------
The source text emphasizes five core ideas:
1. dopaminergic strain on executive-control systems and fronto-striatal loops,
2. glutamatergic contributions to maladaptive learning and object attachment,
3. heritable / neurodevelopmental vulnerability,
4. stress-sensitive plasticity, including a BDNF-linked vulnerability frame,
5. weakened communication between prefrontal control regions and limbic emotion
   systems, especially DLPFC/ACC interactions with the amygdala.

The scaffold therefore models a one-pass causal chain:
inputs -> latent biology -> regional dysregulation -> symptoms -> phenotypes

Some regions are represented conservatively with proxies when the chapter is
systems-level or when atlas labels vary across siibra / Julich versions.
"""

import itertools
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


DEFAULT_GENE_PANEL = [
    "BDNF",
    "COMT",
    "DRD2",
    "DRD1",
    "SLC6A3",
    "GRIN2B",
    "SLC1A2",
    "SLC17A7",
    "SNAP25",
    "DLG4",
]


class HoardingDisorderModel:
    """Mechanistic siibra scaffold for Hoarding Disorder.

    Notes
    -----
    - This is a research scaffold built from chapter-level claims.
    - Higher values in the simulator generally represent greater pathophysiologic
      burden or symptom severity.
    - Region nodes are atlas-grounded where possible; some use explicitly named
      proxies to avoid overclaiming cytoarchitectonic precision.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        assignment_space: str = "mni152",
        connectivity_cohort: str = "HCP",
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
        self.assignment_space = assignment_space
        self.connectivity_cohort = connectivity_cohort

        # Region roles are written separately from region labels so unresolved
        # proxies are still interpretable in the build output.
        self.region_node_descriptions: Dict[str, str] = {
            "dlpfc": (
                "Executive-control anchor for planning, decision-making, and "
                "cognitive flexibility; resolved conservatively to a DLPFC-like Julich area."
            ),
            "acc": (
                "Anterior cingulate control / conflict-monitoring anchor for "
                "emotion regulation and top-down control."
            ),
            "amygdala": (
                "Limbic threat/anxiety anchor used for discard-related distress "
                "and emotional salience."
            ),
            "hippocampus": (
                "Associative-memory anchor included conservatively for the chapter's "
                "maladaptive learning / object-meaning theme."
            ),
            "ventral_striatum_proxy": (
                "Proxy for basal-ganglia / fronto-striatal gating burden when the "
                "chapter speaks at the level of basal ganglia or frontal-striatal circuitry."
            ),
        }

        self.region_candidates: Dict[str, List[str]] = {
            "dlpfc": [
                # Newer Julich / prefrontal labels may differ by version.
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "Area 9 left",
                "MFG1 left",
                "MFG2 left",
                "SFG4 left",
                "middle frontal gyrus",
                "dorsolateral prefrontal cortex",
            ],
            "acc": [
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "Area 33 (ACC) left",
                "Area p24 left",
                "anterior cingulate cortex",
                "cingulate",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "DG (Hippocampus) left",
                "HC-Subiculum (Hippocampus) left",
                "hippocampus",
            ],
            "ventral_striatum_proxy": [
                "ventral striatum",
                "BST (Bed Nucleus) left",
                "striatum",
                "basal ganglia",
                "caudate",
                "putamen",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Heritable liability for altered neurodevelopment, plasticity, and control systems."
            ),
            "stressful_life_events": (
                "Stress exposure that may precipitate or amplify hoarding-related biology."
            ),
            "neurodevelopmental_liability": (
                "Developmental vulnerability linked to early symptom emergence and related conditions."
            ),
            "adhd_comorbidity_load": (
                "Attention/executive-control burden representing the chapter's ADHD overlap signal."
            ),
            "recovery_support": (
                "Protective support factor representing treatment structure, compensation, and external organization."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "stress_sensitive_neuroplastic_vulnerability": (
                "BDNF-linked stress/plasticity liability that can bias emotional regulation and circuit adaptation."
            ),
            "dopaminergic_control_dysregulation": (
                "Dopaminergic disturbance affecting planning, decision-making, attention, and cognitive flexibility."
            ),
            "glutamatergic_association_bias": (
                "Glutamatergic imbalance supporting maladaptive learning and rigid object-value associations."
            ),
            "frontostriatal_control_failure": (
                "Failure of frontal-striatal control loops underlying executive dysfunction and compulsive actions."
            ),
            "frontolimbic_disconnection": (
                "Weak coupling between prefrontal control systems and limbic emotion systems."
            ),
            "object_safety_identity_binding": (
                "Maladaptive learned attachment in which possessions become linked to safety, identity, or relief."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "executive_dysfunction": (
                "Deficits in planning, organization, decision-making, and cognitive control."
            ),
            "cognitive_inflexibility": (
                "Difficulty shifting rules, priorities, and responses during sorting or discarding."
            ),
            "attention_disorganization": (
                "Distractibility / disorganized attention that worsens sorting and follow-through."
            ),
            "distress_when_discarding": (
                "Anxiety or marked emotional discomfort when possessions must be discarded."
            ),
            "indecisiveness": (
                "Difficulty deciding what to keep, discard, or organize."
            ),
            "compulsive_saving": (
                "Persistent over-saving driven by attachment, anxiety, and poor control."
            ),
            "clutter_accumulation": (
                "Downstream environmental burden reflecting persistent retention and impaired organization."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "stress_sensitive_neuroplastic_vulnerability",
                "relation": "raises stress-sensitive plasticity liability",
                "hd_change": "increased",
            },
            {
                "source": "stressful_life_events",
                "target": "stress_sensitive_neuroplastic_vulnerability",
                "relation": "can alter stress adaptation and BDNF-linked plasticity",
                "hd_change": "increased",
            },
            {
                "source": "neurodevelopmental_liability",
                "target": "dopaminergic_control_dysregulation",
                "relation": "biases attention and executive-control development",
                "hd_change": "increased",
            },
            {
                "source": "adhd_comorbidity_load",
                "target": "dopaminergic_control_dysregulation",
                "relation": "marks shared dopaminergic vulnerability with executive-control consequences",
                "hd_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "glutamatergic_association_bias",
                "relation": "contributes to maladaptive learning and plasticity bias",
                "hd_change": "increased",
            },
            {
                "source": "dopaminergic_control_dysregulation",
                "target": "frontostriatal_control_failure",
                "relation": "weakens planning, decision-making, and cognitive flexibility",
                "hd_change": "increased",
            },
            {
                "source": "glutamatergic_association_bias",
                "target": "frontostriatal_control_failure",
                "relation": "adds rigid learning pressure to control loops",
                "hd_change": "increased",
            },
            {
                "source": "stress_sensitive_neuroplastic_vulnerability",
                "target": "frontolimbic_disconnection",
                "relation": "sensitizes emotional-control circuitry under stress",
                "hd_change": "increased",
            },
            {
                "source": "glutamatergic_association_bias",
                "target": "object_safety_identity_binding",
                "relation": "strengthens maladaptive possession-safety and possession-identity learning",
                "hd_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "dlpfc",
                "relation": "reduces effective DLPFC control efficiency",
                "hd_change": "decreased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "acc",
                "relation": "reduces cingulate control over emotional conflict",
                "hd_change": "decreased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "amygdala",
                "relation": "permits greater limbic threat / anxiety reactivity",
                "hd_change": "increased",
            },
            {
                "source": "object_safety_identity_binding",
                "target": "hippocampus",
                "relation": "reinforces object-memory and contextual meaning binding",
                "hd_change": "increased",
            },
            {
                "source": "dopaminergic_control_dysregulation",
                "target": "ventral_striatum_proxy",
                "relation": "biases striatal gating and salience assignment",
                "hd_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "executive_dysfunction",
                "relation": "drives broad executive impairment",
                "hd_change": "increased",
            },
            {
                "source": "glutamatergic_association_bias",
                "target": "cognitive_inflexibility",
                "relation": "supports rigid learned rules and poor set shifting",
                "hd_change": "increased",
            },
            {
                "source": "dopaminergic_control_dysregulation",
                "target": "attention_disorganization",
                "relation": "reduces attention stability and follow-through",
                "hd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "distress_when_discarding",
                "relation": "amplifies anxiety associated with discarding possessions",
                "hd_change": "increased",
            },
            {
                "source": "executive_dysfunction",
                "target": "indecisiveness",
                "relation": "undermines efficient keep/discard decisions",
                "hd_change": "increased",
            },
            {
                "source": "distress_when_discarding",
                "target": "compulsive_saving",
                "relation": "discourages discarding and promotes retention",
                "hd_change": "increased",
            },
            {
                "source": "object_safety_identity_binding",
                "target": "compulsive_saving",
                "relation": "makes possessions feel too meaningful or protective to discard",
                "hd_change": "increased",
            },
            {
                "source": "compulsive_saving",
                "target": "clutter_accumulation",
                "relation": "drives downstream clutter burden",
                "hd_change": "increased",
            },
            {
                "source": "recovery_support",
                "target": "frontostriatal_control_failure",
                "relation": "partially compensates executive-control burden",
                "hd_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "distress_when_discarding",
                "relation": "buffers anxiety during discarding",
                "hd_change": "decreased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame(self.edge_table)
        self._pmap: Any = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None
        self._connectivity_feature: Any = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _name_key(obj: Any) -> str:
        text = str(getattr(obj, "name", obj)).lower().replace("-", " ")
        return " ".join(text.split())

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
        name = self._name_key(region)
        left_penalty = 0 if " left" in f" {name}" else 1
        right_penalty = 1 if " right" in f" {name}" else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "striatum",
            "basal ganglia",
            "anterior cingulate cortex",
            "cingulate",
            "middle frontal gyrus",
            "dorsolateral prefrontal cortex",
        } else 0
        specificity_penalty = 0 if ("area " in name or "(" in name) else 1
        return (left_penalty, right_penalty, generic_penalty, specificity_penalty)

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
        volume_mm3 = float(getattr(main, "volume", float("nan")))
        return centroid_xyz, volume_mm3

    def _region_volume(self, region: Any):
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(space=self.assignment_space)
            except Exception:
                return None

    def _feature_df(self, feature: Any) -> pd.DataFrame:
        try:
            data = getattr(feature, "data", None)
            if isinstance(data, pd.DataFrame):
                return data.copy()
            if data is not None:
                return pd.DataFrame(data)
        except Exception:
            pass
        return pd.DataFrame()

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            volume = self._region_volume(region)
            if volume is not None:
                feats = self._safe_features_any(volume, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        df = self._feature_df(feats[0]).reset_index(drop=False)
        if df.empty:
            return df
        lower_cols = {c.lower(): c for c in df.columns}
        if "index" in df.columns and "receptor" not in lower_cols:
            df = df.rename(columns={"index": "receptor"})
        return df.reset_index(drop=True)

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()
        df = self._feature_df(feats[0])
        if df.empty:
            return df

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

    def _get_connectivity_feature(self) -> Any:
        if self._connectivity_feature is not None:
            return self._connectivity_feature
        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_feature = None
            return None
        preferred = next((f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort), feats[0])
        self._connectivity_feature = preferred
        return preferred

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feature = self._get_connectivity_feature()
        if feature is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        # Compound features usually expose individual matrices through elements.
        df = self._feature_df(feature)
        if not df.empty:
            self._connectivity_matrix = df
            return self._connectivity_matrix

        element: Any = None
        try:
            element = feature[0]
        except Exception:
            element = None

        if element is None:
            try:
                first_index = feature.indices[0]
                element = feature.get_element(first_index)
            except Exception:
                element = None

        if element is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        df = self._feature_df(element)
        if df.empty:
            raw = getattr(element, "data", None)
            regions = getattr(element, "regions", None)
            try:
                if raw is not None and regions is not None:
                    region_list = list(regions)
                    df = pd.DataFrame(raw, index=region_list, columns=region_list)
            except Exception:
                df = pd.DataFrame()

        self._connectivity_matrix = df if isinstance(df, pd.DataFrame) else pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._name_key(region)

        exact = [x for x in labels if self._name_key(x) == region_name]
        if exact:
            return exact[0]

        fuzzy = [
            x for x in labels
            if region_name in self._name_key(x) or self._name_key(x) in region_name
        ]
        if fuzzy:
            return fuzzy[0]

        # Token overlap fallback
        region_tokens = set(region_name.split())
        scored: List[Tuple[int, Any]] = []
        for x in labels:
            name = self._name_key(x)
            tokens = set(name.split())
            overlap = len(region_tokens & tokens)
            if overlap:
                scored.append((overlap, x))
        if scored:
            scored.sort(key=lambda pair: pair[0], reverse=True)
            return scored[0][1]
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
            series = pd.Series(series)
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name]
            df = df[df["value"].notna()]
            if "value" in df.columns:
                try:
                    df = df[df["value"] > 0]
                except Exception:
                    pass
            return df.head(max_rows).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame(columns=["source", "target", "value", "cohort"])

        rows: List[Dict[str, Any]] = []
        for src_key, dst_key in itertools.combinations(self.region_objects.keys(), 2):
            src = self.region_objects[src_key]
            dst = self.region_objects[dst_key]
            src_label = self._match_region_label(list(matrix.index), src) or self._match_region_label(list(matrix.columns), src)
            dst_label = self._match_region_label(list(matrix.columns), dst) or self._match_region_label(list(matrix.index), dst)
            if src_label is None or dst_label is None:
                continue

            value: Optional[float] = None
            try:
                value = float(matrix.loc[src_label, dst_label])
            except Exception:
                try:
                    value = float(matrix.loc[dst_label, src_label])
                except Exception:
                    value = None
            if value is None:
                continue
            rows.append(
                {
                    "source": src_key,
                    "target": dst_key,
                    "value": value,
                    "cohort": getattr(self._get_connectivity_feature(), "cohort", None),
                }
            )
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True) if rows else pd.DataFrame(columns=["source", "target", "value", "cohort"])

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
                    "is_proxy": False,
                    "description": desc,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}

        for key, candidates in self.region_candidates.items():
            is_proxy = key.endswith("_proxy")
            desc = self.region_node_descriptions.get(key, "Atlas-backed circuit node")
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
                        "is_proxy": is_proxy,
                        "description": f"{desc} (unresolved in this siibra environment)",
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
                    "is_proxy": is_proxy,
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
                    "is_proxy": False,
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
                    "is_proxy": False,
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
        stressful_life_events: float = 0.45,
        neurodevelopmental_liability: float = 0.45,
        adhd_comorbidity_load: float = 0.35,
        recovery_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """Run a simple normalized HD simulator.

        All values are clipped to [0, 1]. Higher values indicate greater burden.
        Recovery support subtracts from several pathological pathways.
        """

        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "stressful_life_events": self._clip01(stressful_life_events),
                "neurodevelopmental_liability": self._clip01(neurodevelopmental_liability),
                "adhd_comorbidity_load": self._clip01(adhd_comorbidity_load),
                "recovery_support": self._clip01(recovery_support),
            },
            name="inputs",
        )

        latents = pd.Series(dtype=float, name="latents")
        latents["stress_sensitive_neuroplastic_vulnerability"] = self._clip01(
            0.45 * inputs["genetic_vulnerability"]
            + 0.40 * inputs["stressful_life_events"]
            + 0.20 * inputs["neurodevelopmental_liability"]
            - 0.25 * inputs["recovery_support"]
        )
        latents["dopaminergic_control_dysregulation"] = self._clip01(
            0.35 * inputs["genetic_vulnerability"]
            + 0.30 * inputs["adhd_comorbidity_load"]
            + 0.20 * inputs["neurodevelopmental_liability"]
            + 0.15 * inputs["stressful_life_events"]
            - 0.20 * inputs["recovery_support"]
        )
        latents["glutamatergic_association_bias"] = self._clip01(
            0.30 * inputs["genetic_vulnerability"]
            + 0.25 * inputs["stressful_life_events"]
            + 0.25 * inputs["neurodevelopmental_liability"]
            + 0.15 * latents["dopaminergic_control_dysregulation"]
            - 0.15 * inputs["recovery_support"]
        )
        latents["frontostriatal_control_failure"] = self._clip01(
            0.45 * latents["dopaminergic_control_dysregulation"]
            + 0.20 * latents["glutamatergic_association_bias"]
            + 0.20 * inputs["neurodevelopmental_liability"]
            + 0.10 * latents["stress_sensitive_neuroplastic_vulnerability"]
            - 0.25 * inputs["recovery_support"]
        )
        latents["frontolimbic_disconnection"] = self._clip01(
            0.40 * latents["stress_sensitive_neuroplastic_vulnerability"]
            + 0.30 * latents["frontostriatal_control_failure"]
            + 0.15 * latents["glutamatergic_association_bias"]
            - 0.20 * inputs["recovery_support"]
        )
        latents["object_safety_identity_binding"] = self._clip01(
            0.40 * latents["glutamatergic_association_bias"]
            + 0.25 * latents["stress_sensitive_neuroplastic_vulnerability"]
            + 0.15 * latents["frontolimbic_disconnection"]
            + 0.10 * inputs["genetic_vulnerability"]
            - 0.10 * inputs["recovery_support"]
        )

        regional_state = pd.Series(dtype=float, name="regional_state")
        regional_state["dlpfc"] = self._clip01(
            0.55 * latents["frontostriatal_control_failure"]
            + 0.20 * latents["dopaminergic_control_dysregulation"]
            + 0.10 * latents["stress_sensitive_neuroplastic_vulnerability"]
            - 0.25 * inputs["recovery_support"]
        )
        regional_state["acc"] = self._clip01(
            0.45 * latents["frontolimbic_disconnection"]
            + 0.25 * latents["frontostriatal_control_failure"]
            + 0.10 * latents["stress_sensitive_neuroplastic_vulnerability"]
            - 0.20 * inputs["recovery_support"]
        )
        regional_state["amygdala"] = self._clip01(
            0.50 * latents["frontolimbic_disconnection"]
            + 0.30 * latents["stress_sensitive_neuroplastic_vulnerability"]
            + 0.15 * latents["object_safety_identity_binding"]
            - 0.20 * inputs["recovery_support"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.45 * latents["object_safety_identity_binding"]
            + 0.20 * latents["glutamatergic_association_bias"]
            + 0.15 * latents["stress_sensitive_neuroplastic_vulnerability"]
            - 0.10 * inputs["recovery_support"]
        )
        regional_state["ventral_striatum_proxy"] = self._clip01(
            0.45 * latents["dopaminergic_control_dysregulation"]
            + 0.30 * latents["frontostriatal_control_failure"]
            + 0.15 * latents["object_safety_identity_binding"]
            - 0.15 * inputs["recovery_support"]
        )

        symptoms = pd.Series(dtype=float, name="symptoms")
        symptoms["executive_dysfunction"] = self._clip01(
            0.45 * latents["frontostriatal_control_failure"]
            + 0.25 * regional_state["dlpfc"]
            + 0.15 * regional_state["acc"]
            + 0.10 * inputs["adhd_comorbidity_load"]
            - 0.20 * inputs["recovery_support"]
        )
        symptoms["cognitive_inflexibility"] = self._clip01(
            0.35 * latents["frontostriatal_control_failure"]
            + 0.25 * latents["glutamatergic_association_bias"]
            + 0.15 * regional_state["dlpfc"]
            + 0.10 * latents["stress_sensitive_neuroplastic_vulnerability"]
            - 0.15 * inputs["recovery_support"]
        )
        symptoms["attention_disorganization"] = self._clip01(
            0.40 * latents["dopaminergic_control_dysregulation"]
            + 0.25 * symptoms["executive_dysfunction"]
            + 0.15 * regional_state["ventral_striatum_proxy"]
            + 0.10 * inputs["adhd_comorbidity_load"]
            - 0.15 * inputs["recovery_support"]
        )
        symptoms["distress_when_discarding"] = self._clip01(
            0.35 * latents["object_safety_identity_binding"]
            + 0.30 * regional_state["amygdala"]
            + 0.20 * latents["frontolimbic_disconnection"]
            + 0.10 * regional_state["acc"]
            - 0.15 * inputs["recovery_support"]
        )
        symptoms["indecisiveness"] = self._clip01(
            0.30 * symptoms["executive_dysfunction"]
            + 0.25 * symptoms["cognitive_inflexibility"]
            + 0.20 * symptoms["distress_when_discarding"]
            + 0.15 * regional_state["dlpfc"]
            - 0.10 * inputs["recovery_support"]
        )
        symptoms["compulsive_saving"] = self._clip01(
            0.30 * symptoms["distress_when_discarding"]
            + 0.25 * latents["object_safety_identity_binding"]
            + 0.20 * symptoms["attention_disorganization"]
            + 0.15 * regional_state["ventral_striatum_proxy"]
            - 0.10 * inputs["recovery_support"]
        )
        symptoms["clutter_accumulation"] = self._clip01(
            0.35 * symptoms["compulsive_saving"]
            + 0.25 * symptoms["indecisiveness"]
            + 0.20 * symptoms["executive_dysfunction"]
            + 0.10 * symptoms["distress_when_discarding"]
            - 0.10 * inputs["recovery_support"]
        )

        phenotypes = pd.Series(
            {
                "control_deficit_profile": self._clip01(
                    (
                        symptoms["executive_dysfunction"]
                        + symptoms["cognitive_inflexibility"]
                        + symptoms["attention_disorganization"]
                        + symptoms["indecisiveness"]
                    ) / 4.0
                ),
                "discard_anxiety_profile": self._clip01(
                    (
                        symptoms["distress_when_discarding"]
                        + latents["object_safety_identity_binding"]
                        + regional_state["amygdala"]
                    ) / 3.0
                ),
                "clutter_burden_profile": self._clip01(
                    (
                        symptoms["compulsive_saving"]
                        + symptoms["clutter_accumulation"]
                        + symptoms["indecisiveness"]
                    ) / 3.0
                ),
                "network_disconnection_profile": self._clip01(
                    (
                        latents["frontostriatal_control_failure"]
                        + latents["frontolimbic_disconnection"]
                        + regional_state["dlpfc"]
                        + regional_state["acc"]
                    ) / 4.0
                ),
                "overall_hoarding_burden": self._clip01(
                    (
                        symptoms["clutter_accumulation"]
                        + symptoms["compulsive_saving"]
                        + symptoms["distress_when_discarding"]
                        + symptoms["executive_dysfunction"]
                    ) / 4.0
                ),
            },
            name="phenotypes",
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

        point = siibra.Point(tuple(xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        region = self.region_objects.get(node_key)
        if region is None:
            region = self._resolve_region(self.region_candidates.get(node_key, []))
        if region is None:
            return None
        try:
            return region.get_regional_mask(space=self.assignment_space, maptype="labelled")
        except Exception:
            try:
                return region.get_regional_mask(space=self.assignment_space)
            except Exception:
                return None


if __name__ == "__main__":
    model = HoardingDisorderModel()
    built = model.build(connectivity_rows=10)

    print("\n=== NODES ===")
    node_cols = [
        "key",
        "node_type",
        "is_proxy",
        "atlas_region",
        "feature_summary",
    ]
    print(built["nodes"][node_cols].to_string(index=False))

    print("\n=== EDGES ===")
    edge_cols = ["source", "target", "relation", "hd_change"]
    print(built["edges"][edge_cols].to_string(index=False))

    print("\n=== CIRCUIT CONNECTIVITY (resolved region pairs) ===")
    circuit_df = built["circuit_connectivity"]
    if circuit_df.empty:
        print("No circuit connectivity matrix could be resolved in this environment.")
    else:
        print(circuit_df.head(10).to_string(index=False))

    for key in ["amygdala", "dlpfc", "acc", "ventral_striatum_proxy"]:
        print(f"\n=== {key.upper()} GENE SUMMARY ===")
        gene_df = built["genes"].get(key, pd.DataFrame())
        print(gene_df.head(10).to_string(index=False) if not gene_df.empty else "No gene data available.")

        print(f"\n=== {key.upper()} CONNECTIVITY PROFILE ===")
        conn_df = built["connectivity_profiles"].get(key, pd.DataFrame())
        print(conn_df.head(10).to_string(index=False) if not conn_df.empty else "No connectivity profile available.")

    sim = model.simulate(
        genetic_vulnerability=0.70,
        stressful_life_events=0.65,
        neurodevelopmental_liability=0.55,
        adhd_comorbidity_load=0.45,
        recovery_support=0.25,
    )

    print("\n=== SIMULATION OUTPUT ===")
    for name, series in sim.items():
        print(f"\n[{name}]")
        print(series.sort_values(ascending=False).to_string())

    print("\n=== OPTIONAL USAGE ===")
    print("# model.assign_mni_point((-6, 28, 24)).head()")
    print("# mask = model.region_mask('amygdala')")
