from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Stress / trauma-related panel inferred from your excerpt.
# siibra docs show gene strings can be passed directly to
# siibra.features.get(region, "gene expressions", gene=[...])
DEFAULT_GENE_PANEL = [
    "nr3c1",   # glucocorticoid receptor
    "fkbp5",   # HPA-axis stress regulation
    "slc6a4",  # serotonin transporter
    "maoa",    # monoamine oxidase A
    "comt",    # catecholamine metabolism
    "drd2",    # dopamine receptor D2
    "htr2a",   # serotonin receptor
    "il1b",    # inflammation
    "il6",     # inflammation
    "tnf",     # inflammation
]


class AcuteStressDisorderModel:
    """
    Atlas-grounded, mechanistic ASD scaffold using siibra.

    This is NOT a diagnostic model.
    It is a structured knowledge graph + feature puller built on:
      - atlas/parcellation/space objects
      - region lookup
      - spatial properties in MNI space
      - receptor fingerprints
      - gene-expression summaries
      - structural connectivity profiles
      - optional MNI coordinate assignment
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
    ) -> None:
        self.atlas = siibra.atlases.get(atlas_spec)
        self.parcellation = self.atlas.parcellations.get(parcellation_spec)
        self.space = self.atlas.spaces.get(space_spec)

        # Compact left-sided representatives to keep the graph small.
        # You can duplicate these with right-sided homologues for a bilateral model.
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
                "hippocampus",
            ],
            "pacc": [
                "Area p32 (pACC) left",
                "Area p32 (pACC)",
                "p32",
            ],
            "sacc": [
                "Area s32 (sACC) left",
                "Area s32 (sACC)",
                "s32",
            ],
            "acc": [
                "Area 33 (ACC) left",
                "Area 33 (ACC)",
                "33 (ACC)",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Fo4",
                "Fo3",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "trauma_exposure": "Severity / immediacy of traumatic event exposure",
            "biological_vulnerability": "Predisposition from genetics, prior trauma, comorbidity",
            "psychosocial_support": "Protective environment / support after trauma",
        }

        self.latent_nodes: Dict[str, str] = {
            "hpa_axis": "Stress-hormone regulation / glucocorticoid signalling",
            "epigenetics": "Trauma-linked epigenetic regulation / DNA methylation burden",
            "inflammation": "Pro-inflammatory cytokine signalling",
            "dopamine": "Mesolimbic dopamine dysregulation",
            "serotonin": "Serotonergic dysregulation",
            "noradrenaline": "Noradrenergic hyperarousal tone",
        }

        self.symptom_nodes: Dict[str, str] = {
            "hyperarousal": "Persistent alarm / autonomic activation",
            "reexperiencing": "Flashbacks and intrusive traumatic memory",
            "dissociation": "Depersonalization / derealization / dissociative amnesia",
            "negative_mood": "Anhedonia / withdrawal / mood disturbance",
            "cognitive_control": "Poor concentration / indecisiveness / emotion regulation failure",
            "psychotic_like": "Rare severe psychotic-like symptoms under acute stress",
        }

        # Qualitative edge list from your excerpt.
        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "trauma_exposure",
                "target": "amygdala",
                "relation": "activates threat/alarm circuitry",
                "asd_change": "increased",
            },
            {
                "source": "trauma_exposure",
                "target": "inflammation",
                "relation": "triggers pro-inflammatory response",
                "asd_change": "increased",
            },
            {
                "source": "trauma_exposure",
                "target": "epigenetics",
                "relation": "induces stress-responsive regulation changes",
                "asd_change": "increased",
            },
            {
                "source": "biological_vulnerability",
                "target": "hpa_axis",
                "relation": "raises susceptibility of stress regulation",
                "asd_change": "increased susceptibility",
            },
            {
                "source": "biological_vulnerability",
                "target": "amygdala",
                "relation": "raises sensitivity to salient threat cues",
                "asd_change": "increased susceptibility",
            },
            {
                "source": "psychosocial_support",
                "target": "epigenetics",
                "relation": "can normalize stress-responsive regulation",
                "asd_change": "protective",
            },
            {
                "source": "psychosocial_support",
                "target": "pacc",
                "relation": "supports regulatory recovery",
                "asd_change": "protective",
            },
            {
                "source": "epigenetics",
                "target": "hpa_axis",
                "relation": "biases glucocorticoid signalling",
                "asd_change": "dysregulated",
            },
            {
                "source": "hpa_axis",
                "target": "hippocampus",
                "relation": "impairs contextual memory regulation",
                "asd_change": "reduced function",
            },
            {
                "source": "hpa_axis",
                "target": "pacc",
                "relation": "weakens top-down control",
                "asd_change": "reduced function",
            },
            {
                "source": "inflammation",
                "target": "negative_mood",
                "relation": "promotes sickness behaviour / low mood",
                "asd_change": "increased",
            },
            {
                "source": "inflammation",
                "target": "cognitive_control",
                "relation": "burdens executive function",
                "asd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "hyperarousal",
                "relation": "drives fear and alarm output",
                "asd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "reexperiencing",
                "relation": "amplifies salience of traumatic cues",
                "asd_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "reexperiencing",
                "relation": "normally contextualizes traumatic memory",
                "asd_change": "reduced contextualization",
            },
            {
                "source": "pacc",
                "target": "amygdala",
                "relation": "normally exerts top-down inhibition",
                "asd_change": "reduced inhibition",
            },
            {
                "source": "sacc",
                "target": "amygdala",
                "relation": "normally supports emotional regulation",
                "asd_change": "reduced inhibition",
            },
            {
                "source": "acc",
                "target": "cognitive_control",
                "relation": "normally supports executive control",
                "asd_change": "reduced function",
            },
            {
                "source": "ofc",
                "target": "amygdala",
                "relation": "normally supports extinction / valuation",
                "asd_change": "reduced inhibition",
            },
            {
                "source": "dopamine",
                "target": "negative_mood",
                "relation": "reward / salience dysregulation contributes to anhedonia",
                "asd_change": "increased dysregulation",
            },
            {
                "source": "dopamine",
                "target": "psychotic_like",
                "relation": "aberrant salience may support severe altered perception",
                "asd_change": "possible increase",
            },
            {
                "source": "serotonin",
                "target": "negative_mood",
                "relation": "modulates mood regulation",
                "asd_change": "dysregulated",
            },
            {
                "source": "noradrenaline",
                "target": "hyperarousal",
                "relation": "supports arousal / vigilance",
                "asd_change": "increased",
            },
            {
                "source": "sacc",
                "target": "dissociation",
                "relation": "fronto-limbic dysregulation may contribute to dissociative states",
                "asd_change": "increased",
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
                # documented pattern: atlas.get_region(spec, parcellation="julich")
                return self.atlas.get_region(spec, parcellation="julich")
            except Exception:
                continue
        return None

    def _main_component(
        self, region: Any
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        try:
            props = region.spatial_props(space=self.space)
            if not props:
                return None, None
            main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
            centroid = tuple(main.centroid)  # Point is iterable in siibra
            volume_mm3 = float(getattr(main, "volume", float("nan")))
            return centroid, volume_mm3
        except Exception:
            return None, None

    def _safe_features(self, region: Any, modality: Any, **kwargs: Any) -> List[Any]:
        try:
            with siibra.QUIET:
                feats = siibra.features.get(region, modality, **kwargs)
            return list(feats) if feats else []
        except Exception:
            return []

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features(region, "receptor density fingerprint")
        if not feats:
            return pd.DataFrame()

        df = feats[0].data.copy().reset_index()
        if "index" in df.columns and "receptor" not in df.columns:
            df = df.rename(columns={"index": "receptor"})
        return df

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features(region, "gene expressions", gene=list(genes))
        if not feats:
            return pd.DataFrame()

        df = feats[0].data.copy()
        needed = {"gene", "level", "zscore"}
        if not needed.issubset(df.columns):
            return df

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

    def _connectivity_table(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        feats = self._safe_features(region, "StreamlineCounts")
        if not feats:
            return pd.DataFrame()

        # Prefer HCP if available, otherwise take first returned compound feature.
        conn = next((f for f in feats if getattr(f, "cohort", None) == "HCP"), feats[0])

        try:
            profile = conn.get_profile(region, max_rows=max_rows)
            df = getattr(profile, "data", profile)
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            df = df.copy().reset_index()

            first_col = df.columns[0]
            df = df.rename(columns={first_col: "connected_region"})
            df["connected_region"] = df["connected_region"].map(
                lambda r: getattr(r, "name", str(r))
            )
            return df
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
                        "description": "Atlas-backed region node (unresolved in this environment)",
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
                    "description": "Atlas-backed ASD circuit node",
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
        Map a literature MNI coordinate back to Julich probabilistic regions.
        Example:
            model.assign_mni_point((-20, -4, -16)).head(10)
        """
        if self._pmap is None:
            with siibra.QUIET:
                self._pmap = siibra.get_map(
                    parcellation="julich",
                    space="mni152",
                    maptype="statistical",
                )

        point = siibra.Point(tuple(xyz), space="mni152")
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        sort_key = "map value" if "map value" in assignments.columns else (
            "correlation" if "correlation" in assignments.columns else None
        )
        if sort_key is not None:
            assignments = assignments.sort_values(sort_key, ascending=False)
        return assignments

    def region_mask(self, node_key: str):
        """
        Fetch a binary regional mask for one resolved region node.
        Useful for nilearn plotting.
        """
        region = self.region_objects[node_key]
        return region.get_regional_mask(self.space, maptype="labelled")


if __name__ == "__main__":
    model = AcuteStressDisorderModel()
    bundle = model.build()

    print("\n=== NODES ===")
    print(
        bundle["nodes"][
            ["key", "node_type", "atlas_region", "centroid_mni", "feature_summary"]
        ].to_string(index=False)
    )

    print("\n=== EDGES ===")
    print(
        bundle["edges"][["source", "target", "relation", "asd_change"]].to_string(index=False)
    )

    # Show a few empirical summaries where available
    for key in ["amygdala", "hippocampus", "pacc"]:
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

    # Example for mapping an MNI coordinate from a paper into the model's atlas:
    # print(model.assign_mni_point((-20, -4, -16)).head(10))
