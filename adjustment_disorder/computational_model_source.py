from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Adjustment-Disorder-oriented panel:
# - stress/HPA axis
# - noradrenergic and dopaminergic regulation
# - monoamine metabolism / transport
# - inflammatory signalling
DEFAULT_GENE_PANEL = [
    "nr3c1",   # glucocorticoid receptor
    "fkbp5",   # stress responsivity / HPA-axis regulation
    "crhr1",   # CRH signalling
    "slc6a2",  # norepinephrine transporter
    "dbh",     # dopamine beta hydroxylase
    "adra2a",  # alpha-2A adrenergic receptor
    "drd2",    # dopamine receptor D2
    "slc6a3",  # dopamine transporter
    "comt",    # catecholamine metabolism
    "maoa",    # monoamine oxidase A
    "slc6a4",  # serotonin transporter
    "il1b",    # inflammation
    "il6",     # inflammation
    "tnf",     # inflammation
]


class AdjustmentDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Adjustment Disorder (AD) using siibra.

    This is NOT a diagnostic model.
    It is a structured network with:
      - atlas-backed region nodes for amygdala / hippocampus / ACC / PFC proxies
      - latent biology nodes for diathesis-stress, HPA-axis, epigenetics,
        inflammation, norepinephrine, and dopamine
      - symptom nodes for anxious arousal, depressed mood, anhedonia,
        somatic burden, executive dyscontrol, and impaired adaptation
      - empirical siibra feature pulls where available
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
    ) -> None:
        self.atlas = siibra.atlases.get(atlas_spec)
        self.parcellation = self.atlas.get_parcellation(parcellation_spec)
        self.space = self.atlas.get_space(space_spec)

        # Broad PFC is described in the chapter, but Julich is more specific.
        # So this uses frontopolar areas as a practical PFC proxy.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "SF (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "DG (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "hippocampus",
            ],
            "pacc": [
                "Area p32 (pACC) left",
                "Area p32 (pACC)",
                "p32",
                "pACC",
            ],
            "sacc": [
                "Area s32 (sACC) left",
                "Area s32 (sACC)",
                "s32",
                "sACC",
            ],
            "acc": [
                "Area 33 (ACC) left",
                "Area 33 (ACC)",
                "33 (ACC)",
                "ACC",
            ],
            "pfc": [
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "Area Fp2 (FPole)",
                "Area Fp1 (FPole)",
                "Fp2",
                "Fp1",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "life_stressor": "External stressor such as loss, conflict, illness, or major life disruption",
            "biological_diathesis": "Pre-existing polygenic, temperamental, and mood/anxiety vulnerability",
            "early_life_programming": "Early experience shaping long-term stress responsivity",
            "coping_resources": "Protective coping flexibility, support, and adaptive capacity",
        }

        self.latent_nodes: Dict[str, str] = {
            "hpa_axis": "Stress-hormone and glucocorticoid regulation",
            "epigenetics": "Experience-dependent calibration of stress responsivity",
            "inflammation": "Stress-linked pro-inflammatory signalling",
            "noradrenaline": "Sympathetic arousal / vigilance tone",
            "dopamine": "Reward / motivation / salience regulation",
        }

        self.symptom_nodes: Dict[str, str] = {
            "anxious_arousal": "Heightened worry, vigilance, and stress arousal",
            "depressed_mood": "Low mood after stress exposure",
            "anhedonia": "Reduced reward sensitivity / loss of interest",
            "somatic_distress": "Fatigue, bodily distress, and stress-linked physical symptoms",
            "impaired_adaptation": "Failure to adapt to the stressor",
            "executive_dyscontrol": "Impaired planning, flexibility, and problem-solving",
            "emotional_overreactivity": "Poor regulation of emotional responses to stress",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "life_stressor",
                "target": "amygdala",
                "relation": "increases threat appraisal and emotional salience",
                "ad_change": "increased",
            },
            {
                "source": "life_stressor",
                "target": "hpa_axis",
                "relation": "loads acute stress-response systems",
                "ad_change": "increased",
            },
            {
                "source": "life_stressor",
                "target": "noradrenaline",
                "relation": "raises sympathetic arousal and vigilance",
                "ad_change": "increased",
            },
            {
                "source": "life_stressor",
                "target": "inflammation",
                "relation": "can trigger stress-linked inflammatory signalling",
                "ad_change": "increased",
            },
            {
                "source": "biological_diathesis",
                "target": "hpa_axis",
                "relation": "raises vulnerability of stress regulation",
                "ad_change": "increased susceptibility",
            },
            {
                "source": "biological_diathesis",
                "target": "noradrenaline",
                "relation": "raises trait arousal susceptibility",
                "ad_change": "increased susceptibility",
            },
            {
                "source": "biological_diathesis",
                "target": "dopamine",
                "relation": "raises reward/motivation vulnerability",
                "ad_change": "increased susceptibility",
            },
            {
                "source": "early_life_programming",
                "target": "epigenetics",
                "relation": "programs long-term stress reactivity",
                "ad_change": "increased susceptibility",
            },
            {
                "source": "early_life_programming",
                "target": "hpa_axis",
                "relation": "calibrates later stress-response set-point",
                "ad_change": "increased susceptibility",
            },
            {
                "source": "coping_resources",
                "target": "pacc",
                "relation": "supports regulatory recovery and adaptation",
                "ad_change": "protective",
            },
            {
                "source": "coping_resources",
                "target": "impaired_adaptation",
                "relation": "buffers the failure of adaptation",
                "ad_change": "protective",
            },
            {
                "source": "epigenetics",
                "target": "hpa_axis",
                "relation": "biases glucocorticoid responsivity",
                "ad_change": "dysregulated",
            },
            {
                "source": "epigenetics",
                "target": "pfc",
                "relation": "can alter stress-sensitive plasticity",
                "ad_change": "reduced resilience",
            },
            {
                "source": "inflammation",
                "target": "depressed_mood",
                "relation": "promotes low mood and sickness behaviour",
                "ad_change": "increased",
            },
            {
                "source": "inflammation",
                "target": "somatic_distress",
                "relation": "promotes fatigue and bodily symptom burden",
                "ad_change": "increased",
            },
            {
                "source": "hpa_axis",
                "target": "hippocampus",
                "relation": "impairs contextual appraisal and adaptive learning",
                "ad_change": "reduced function",
            },
            {
                "source": "hpa_axis",
                "target": "pfc",
                "relation": "weakens executive control under stress",
                "ad_change": "reduced function",
            },
            {
                "source": "hpa_axis",
                "target": "pacc",
                "relation": "weakens top-down regulation of emotion",
                "ad_change": "reduced function",
            },
            {
                "source": "noradrenaline",
                "target": "anxious_arousal",
                "relation": "increases vigilance, anxiety, and physiological arousal",
                "ad_change": "increased",
            },
            {
                "source": "noradrenaline",
                "target": "somatic_distress",
                "relation": "amplifies autonomic and physical symptom burden",
                "ad_change": "increased",
            },
            {
                "source": "dopamine",
                "target": "anhedonia",
                "relation": "reduces reward sensitivity and pleasure",
                "ad_change": "increased",
            },
            {
                "source": "dopamine",
                "target": "impaired_adaptation",
                "relation": "reduces motivation for flexible goal-directed coping",
                "ad_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "anxious_arousal",
                "relation": "drives threat-reactive affective output",
                "ad_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "emotional_overreactivity",
                "relation": "amplifies emotional salience and reactivity",
                "ad_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "impaired_adaptation",
                "relation": "reduced contextualization impairs adaptive stress appraisal",
                "ad_change": "reduced contextualization",
            },
            {
                "source": "pacc",
                "target": "amygdala",
                "relation": "normally exerts top-down regulation",
                "ad_change": "reduced inhibition",
            },
            {
                "source": "sacc",
                "target": "amygdala",
                "relation": "normally supports emotional regulation",
                "ad_change": "reduced inhibition",
            },
            {
                "source": "acc",
                "target": "executive_dyscontrol",
                "relation": "reduced conflict monitoring and control",
                "ad_change": "increased",
            },
            {
                "source": "acc",
                "target": "emotional_overreactivity",
                "relation": "reduced regulation of stress-linked emotion",
                "ad_change": "increased",
            },
            {
                "source": "pfc",
                "target": "executive_dyscontrol",
                "relation": "reduced planning, flexibility, and problem-solving",
                "ad_change": "increased",
            },
            {
                "source": "pfc",
                "target": "amygdala",
                "relation": "normally provides cognitive control over emotional salience",
                "ad_change": "reduced inhibition",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity: Dict[str, pd.DataFrame] = {}
        self.nodes_df: pd.DataFrame = pd.DataFrame()
        self.edges_df: pd.DataFrame = pd.DataFrame()
        self._pmap = None

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation)
            except Exception:
                continue
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        """
        Helper for tuning region vocabulary against the installed atlas.
        Useful if you want to swap the broad PFC proxy for another Julich region.
        """
        try:
            matches = self.atlas.find_regions(keyword, filter_children=False)
        except Exception:
            return pd.DataFrame(columns=["name", "identifier", "parcellation"])

        rows = []
        seen = set()
        for r in matches:
            name = getattr(r, "name", str(r))
            identifier = getattr(r, "identifier", None)
            parcellation = getattr(getattr(r, "parcellation", None), "name", None)
            key = (name, identifier, parcellation)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "name": name,
                    "identifier": identifier,
                    "parcellation": parcellation,
                }
            )

        return pd.DataFrame(rows).head(limit)

    def _main_component(
        self, region: Any
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        try:
            props = region.spatial_props(space=self.space)
            if not props:
                return None, None
            main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
            centroid = tuple(main.centroid)
            volume_mm3 = float(getattr(main, "volume", float("nan")))
            return centroid, volume_mm3
        except Exception:
            return None, None

    def _safe_features(self, concept: Any, modality: Any, **kwargs: Any) -> List[Any]:
        try:
            with siibra.QUIET:
                feats = siibra.features.get(concept, modality, **kwargs)
            return list(feats) if feats else []
        except Exception:
            return []

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features(region, "receptor density fingerprint")
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
        feats = self._safe_features(region, "gene expressions", gene=list(genes))
        if not feats:
            return pd.DataFrame()

        try:
            df = feats[0].data.copy()
            if {"gene", "level", "zscore"}.issubset(df.columns):
                summary = (
                    df.groupby("gene", dropna=False)
                    .agg(
                        level_mean=("level", "mean"),
                        level_std=("level", "std"),
                        probe_count=("level", "count"),
                        zscore_mean=("zscore", "mean"),
                        zscore_std=("zscore", "std"),
                    )
                    .reset_index()
                    .sort_values("gene")
                )
                return summary
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _connectivity_table(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        feats = self._safe_features(region, "StreamlineCounts")
        if not feats:
            return pd.DataFrame()

        conn = next((f for f in feats if getattr(f, "cohort", None) == "HCP"), feats[0])

        # First try a profile-style API if available.
        try:
            if hasattr(conn, "get_profile"):
                profile = conn.get_profile(region, max_rows=max_rows)
                df = getattr(profile, "data", profile)
                if isinstance(df, pd.DataFrame):
                    df = df.copy().reset_index()
                    first_col = df.columns[0]
                    df = df.rename(columns={first_col: "connected_region"})
                    df["connected_region"] = df["connected_region"].map(
                        lambda r: getattr(r, "name", str(r))
                    )
                    return df.head(max_rows)
        except Exception:
            pass

        # Fallback: pull the first matrix element and extract the row/column manually.
        try:
            element = conn[0] if hasattr(conn, "__getitem__") else conn
            matrix = getattr(element, "data", None)
            if not isinstance(matrix, pd.DataFrame):
                return pd.DataFrame()

            def _match(labels: Sequence[Any]) -> Optional[Any]:
                exact = [x for x in labels if getattr(x, "name", str(x)) == region.name]
                if exact:
                    return exact[0]

                rname = region.name.lower()
                fuzzy = [
                    x
                    for x in labels
                    if rname in getattr(x, "name", str(x)).lower()
                    or getattr(x, "name", str(x)).lower() in rname
                ]
                return fuzzy[0] if fuzzy else None

            label = _match(list(matrix.index))
            axis = "index"
            if label is None:
                label = _match(list(matrix.columns))
                axis = "columns"
            if label is None:
                return pd.DataFrame()

            series = matrix.loc[label] if axis == "index" else matrix[label]
            df = (
                series.sort_values(ascending=False)
                .reset_index()
                .rename(columns={"index": "connected_region", 0: "value"})
            )
            df["connected_region"] = df["connected_region"].map(
                lambda r: getattr(r, "name", str(r))
            )
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

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
                warnings.warn(f"Could not resolve a Julich region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.upper(),
                        "node_type": "region",
                        "description": "Atlas-backed node (unresolved in this environment)",
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
            conn_df = self._connectivity_table(region, connectivity_rows)

            self.receptors[key] = receptor_df
            self.genes[key] = gene_df
            self.connectivity[key] = conn_df

            feature_summary = (
                f"receptors={'yes' if not receptor_df.empty else 'no'}; "
                f"genes={'yes' if not gene_df.empty else 'no'}; "
                f"connectivity={'yes' if not conn_df.empty else 'no'}"
            )

            nodes.append(
                {
                    "key": key,
                    "label": region.name,
                    "node_type": "region",
                    "description": "Atlas-backed Adjustment Disorder circuit node",
                    "atlas_region": region.name,
                    "region_identifier": getattr(region, "identifier", None),
                    "centroid_mni": centroid_mni,
                    "volume_mm3": volume_mm3,
                    "feature_summary": feature_summary,
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
            "connectivity": self.connectivity,
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate back to probabilistic Julich regions.
        Example:
            model.assign_mni_point((-8, 34, 20)).head(10)
        """
        if self._pmap is None:
            with siibra.QUIET:
                self._pmap = self.atlas.get_map(
                    space=self.atlas.get_space("mni152"),
                    parcellation=self.parcellation,
                    maptype="statistical",
                )

        point = siibra.Point(tuple(xyz), space="mni152")
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        sort_key = None
        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                sort_key = candidate
                break

        if sort_key is not None:
            assignments = assignments.sort_values(sort_key, ascending=False)
        return assignments

    def region_mask(self, node_key: str):
        """
        Fetch a binary regional mask for a resolved region node.
        """
        region = self.region_objects[node_key]
        return region.get_regional_mask(self.space, maptype="labelled")


if __name__ == "__main__":
    model = AdjustmentDisorderModel()
    bundle = model.build()

    print("\n=== NODES ===")
    print(
        bundle["nodes"][
            ["key", "node_type", "atlas_region", "centroid_mni", "feature_summary"]
        ].to_string(index=False)
    )

    print("\n=== EDGES ===")
    print(
        bundle["edges"][["source", "target", "relation", "ad_change"]].to_string(index=False)
    )

    for key in ["amygdala", "hippocampus", "pacc", "pfc"]:
        print(f"\n=== {key.upper()} : receptor fingerprint ===")
        if not bundle["receptors"][key].empty:
            print(bundle["receptors"][key].head(10).to_string(index=False))
        else:
            print("No receptor fingerprint available for this node.")

        print(f"\n=== {key.upper()} : gene panel summary ===")
        if not bundle["genes"][key].empty:
            print(bundle["genes"][key].to_string(index=False))
        else:
            print("No gene-expression summary available for this node.")

        print(f"\n=== {key.upper()} : top structural connectivity ===")
        if not bundle["connectivity"][key].empty:
            print(bundle["connectivity"][key].head(10).to_string(index=False))
        else:
            print("No connectivity profile available for this node.")

    # Example coordinate assignment:
    # print(model.assign_mni_point((-8, 34, 20)).head(10))

    # Example helper for refining region vocabulary:
    # print(model.suggest_regions("prefrontal").head(20).to_string(index=False))
