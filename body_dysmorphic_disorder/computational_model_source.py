from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# BDD-oriented panel:
# - serotonin / OCRD overlap
# - dopamine / salience and habit reinforcement
# - stress / epigenetic susceptibility
# - visual/body-representation plasticity
#
# This is a hypothesis-driven panel because the chapter is strongest on
# circuit logic and less explicit about a fixed gene list.
DEFAULT_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR2A",    # serotonin receptor
    "HTR1A",    # serotonin receptor
    "DRD2",     # dopamine receptor
    "COMT",     # catecholamine metabolism
    "BDNF",     # plasticity
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
    "SLC1A1",   # glutamate transport / OCD-related candidate
    "DLGAP3",   # SAPAP3 / compulsivity-related candidate
    "GRIN2B",   # NMDA receptor subunit
    "GABRA2",   # inhibitory regulation
]


class BodyDysmorphicDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Body Dysmorphic Disorder (BDD).

    What it does:
      1) Resolves BDD-relevant anatomy to atlas regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates obsessional appearance focus, compulsive rituals,
         local-over-global visual bias, and body-schema distortion.

    This is a research scaffold, not a clinical diagnostic tool.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.atlas = siibra.atlases.get(atlas_spec)
        self.parcellation_spec = parcellation_spec
        self.parcellation = self.atlas.get_parcellation(parcellation_spec)
        self.space_spec = space_spec
        self.space = self.atlas.get_space(space_spec)
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        # Functional-network chapter concepts mapped to Julich-friendly search terms.
        # Several are proxies because BDD is described at a network level rather than
        # as exact cytoarchitectonic parcels.
        self.region_candidates: Dict[str, List[str]] = {
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "orbitofrontal",
                "Fo4",
                "Fo3",
            ],
            "acc": [
                "Area 33 (ACC) left",
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "ACC",
                "anterior cingulate",
            ],
            "striatum_proxy": [
                "caudate",
                "putamen",
                "striatum",
                "basal ganglia",
            ],
            "visual_cortex_proxy": [
                "Area hOc3v left",
                "Area hOc2 left",
                "Area hOc1 left",
                "visual cortex",
                "occipital",
            ],
            "posterior_parietal_proxy": [
                "Area PGp (IPL) left",
                "Area PGa (IPL) left",
                "Area PFt (IPL) left",
                "Area 7A (SPL) left",
                "parietal",
            ],
            "premotor_proxy": [
                "Area 6d1 (PreCG) left",
                "Area 6d2 (PreCG) left",
                "Area 6v1 (PreCG) left",
                "premotor",
                "precentral",
            ],
            "self_midline_proxy": [
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "medial prefrontal",
                "pACC",
                "sACC",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": "Familial/polygenic vulnerability to BDD-like traits and OCRD-related psychopathology",
            "environmental_psychological_load": "Environmental and psychological contributors that shape symptom expression",
            "appearance_related_stress": "Stress and self-focused appearance pressure amplifying preoccupation",
            "right_hemisphere_body_image_insult": "Focal body-image/body-schema disturbance analogue from right-hemisphere injury",
            "treatment_support": "Protective treatment, ritual reduction, and cognitive/perceptual support",
        }

        self.latent_nodes: Dict[str, str] = {
            "ocrd_compulsivity": "Obsessive-compulsive-spectrum compulsive habit pressure",
            "frontostriatal_dysregulation": "OFC-ACC-striatal loop dysfunction in inhibition and habit control",
            "behavioral_inhibition_failure": "Failure to suppress compulsive checking or grooming rituals",
            "visual_detail_bias": "Bias toward local/detail processing over global/holistic processing",
            "global_processing_failure": "Failure to represent overall normal appearance coherently",
            "body_schema_distortion": "Distortion in dynamic representation of the body in space",
            "self_referential_distortion": "Distorted cognitive/affective body-image representation",
            "negative_affect_trait": "Neuroticism-like negative affect and self-consciousness vulnerability",
            "gene_environment_sensitization": "Epigenetic or developmental sensitization of trait vulnerability",
        }

        self.symptom_nodes: Dict[str, str] = {
            "appearance_preoccupation": "Obsessive concern with imagined or slight appearance flaws",
            "perceptual_distortion": "Distorted visual-perceptual experience of appearance",
            "body_image_distortion": "Distorted cognitive-affective body representation",
            "local_feature_overfocus": "Excessive focus on fine detail instead of whole appearance",
            "compulsive_checking": "Mirror checking and repetitive appearance monitoring",
            "grooming_picking": "Skin picking, grooming, or related ritualized acts",
            "avoidance_distress": "Distress-driven avoidance of scrutiny or exposure",
            "shame_inadequacy": "Shame and defectiveness linked to perceived flaws",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "ocrd_compulsivity",
                "relation": "raises liability for obsessive-compulsive-spectrum symptom expression",
                "bdd_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "negative_affect_trait",
                "relation": "raises liability for neurotic, self-conscious affective style",
                "bdd_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "gene_environment_sensitization",
                "relation": "creates vulnerability for environmental amplification of symptoms",
                "bdd_change": "increased susceptibility",
            },
            {
                "source": "environmental_psychological_load",
                "target": "gene_environment_sensitization",
                "relation": "amplifies stable symptom vulnerability",
                "bdd_change": "increased",
            },
            {
                "source": "environmental_psychological_load",
                "target": "negative_affect_trait",
                "relation": "increases self-conscious distress and vulnerability",
                "bdd_change": "increased",
            },
            {
                "source": "appearance_related_stress",
                "target": "visual_detail_bias",
                "relation": "focuses attention on minor local imperfections",
                "bdd_change": "increased",
            },
            {
                "source": "appearance_related_stress",
                "target": "self_referential_distortion",
                "relation": "loads negative self-referential body image processing",
                "bdd_change": "increased",
            },
            {
                "source": "right_hemisphere_body_image_insult",
                "target": "body_schema_distortion",
                "relation": "provides a body-image/body-schema lesion analogue",
                "bdd_change": "increased",
            },
            {
                "source": "treatment_support",
                "target": "ocrd_compulsivity",
                "relation": "buffers obsessive-compulsive ritual pressure",
                "bdd_change": "protective",
            },
            {
                "source": "treatment_support",
                "target": "appearance_preoccupation",
                "relation": "buffers obsessional appearance focus",
                "bdd_change": "protective",
            },
            {
                "source": "treatment_support",
                "target": "compulsive_checking",
                "relation": "reduces ritualized checking behavior",
                "bdd_change": "protective",
            },
            {
                "source": "gene_environment_sensitization",
                "target": "negative_affect_trait",
                "relation": "stabilizes self-conscious negative-affect vulnerability",
                "bdd_change": "increased",
            },
            {
                "source": "ocrd_compulsivity",
                "target": "frontostriatal_dysregulation",
                "relation": "loads cortico-striatal-thalamo-cortical compulsive circuitry",
                "bdd_change": "increased",
            },
            {
                "source": "ocrd_compulsivity",
                "target": "compulsive_checking",
                "relation": "increases ritualized appearance-checking behavior",
                "bdd_change": "increased",
            },
            {
                "source": "ocrd_compulsivity",
                "target": "grooming_picking",
                "relation": "increases ritualized picking/grooming behavior",
                "bdd_change": "increased",
            },
            {
                "source": "frontostriatal_dysregulation",
                "target": "ofc",
                "relation": "burdens orbitofrontal habit/valuation circuitry",
                "bdd_change": "increased dysregulation",
            },
            {
                "source": "frontostriatal_dysregulation",
                "target": "acc",
                "relation": "burdens conflict/error monitoring circuitry",
                "bdd_change": "increased dysregulation",
            },
            {
                "source": "frontostriatal_dysregulation",
                "target": "striatum_proxy",
                "relation": "burdens caudate-putamen habit/inhibition circuitry",
                "bdd_change": "increased dysregulation",
            },
            {
                "source": "frontostriatal_dysregulation",
                "target": "behavioral_inhibition_failure",
                "relation": "weakens suppression of ritualistic behavior",
                "bdd_change": "increased",
            },
            {
                "source": "behavioral_inhibition_failure",
                "target": "compulsive_checking",
                "relation": "allows compulsive checking to become entrenched",
                "bdd_change": "increased",
            },
            {
                "source": "behavioral_inhibition_failure",
                "target": "grooming_picking",
                "relation": "allows picking/grooming rituals to become entrenched",
                "bdd_change": "increased",
            },
            {
                "source": "visual_detail_bias",
                "target": "visual_cortex_proxy",
                "relation": "loads local visual processing over global appearance processing",
                "bdd_change": "increased bias",
            },
            {
                "source": "visual_detail_bias",
                "target": "local_feature_overfocus",
                "relation": "amplifies fixation on minor details or asymmetries",
                "bdd_change": "increased",
            },
            {
                "source": "global_processing_failure",
                "target": "perceptual_distortion",
                "relation": "reduces coherent whole-appearance perception",
                "bdd_change": "increased",
            },
            {
                "source": "body_schema_distortion",
                "target": "posterior_parietal_proxy",
                "relation": "loads posterior parietal body-schema construction",
                "bdd_change": "increased dysregulation",
            },
            {
                "source": "body_schema_distortion",
                "target": "premotor_proxy",
                "relation": "loads premotor body-schema representation",
                "bdd_change": "increased dysregulation",
            },
            {
                "source": "body_schema_distortion",
                "target": "body_image_distortion",
                "relation": "destabilizes perceived body representation",
                "bdd_change": "increased",
            },
            {
                "source": "self_referential_distortion",
                "target": "self_midline_proxy",
                "relation": "loads cortical-midline self-referential body-image processing",
                "bdd_change": "increased dysregulation",
            },
            {
                "source": "self_referential_distortion",
                "target": "appearance_preoccupation",
                "relation": "stabilizes obsessional self-focused concern",
                "bdd_change": "increased",
            },
            {
                "source": "negative_affect_trait",
                "target": "appearance_preoccupation",
                "relation": "amplifies self-conscious distress around perceived flaws",
                "bdd_change": "increased",
            },
            {
                "source": "negative_affect_trait",
                "target": "shame_inadequacy",
                "relation": "amplifies shame and perceived defectiveness",
                "bdd_change": "increased",
            },
            {
                "source": "visual_cortex_proxy",
                "target": "local_feature_overfocus",
                "relation": "supports detail-biased visual parsing",
                "bdd_change": "increased",
            },
            {
                "source": "posterior_parietal_proxy",
                "target": "body_image_distortion",
                "relation": "distorted parietal body representation contributes to body-image disturbance",
                "bdd_change": "increased",
            },
            {
                "source": "premotor_proxy",
                "target": "body_image_distortion",
                "relation": "distorted premotor body schema contributes to body-image disturbance",
                "bdd_change": "increased",
            },
            {
                "source": "self_midline_proxy",
                "target": "appearance_preoccupation",
                "relation": "supports self-focused appearance appraisal",
                "bdd_change": "increased",
            },
            {
                "source": "appearance_preoccupation",
                "target": "compulsive_checking",
                "relation": "drives repeated appearance monitoring",
                "bdd_change": "increased",
            },
            {
                "source": "appearance_preoccupation",
                "target": "avoidance_distress",
                "relation": "drives distress and avoidance around social exposure",
                "bdd_change": "increased",
            },
            {
                "source": "body_image_distortion",
                "target": "perceptual_distortion",
                "relation": "contributes to misperception of overall appearance",
                "bdd_change": "increased",
            },
            {
                "source": "perceptual_distortion",
                "target": "shame_inadequacy",
                "relation": "strengthens conviction of defectiveness",
                "bdd_change": "increased",
            },
            {
                "source": "shame_inadequacy",
                "target": "avoidance_distress",
                "relation": "promotes avoidance of scrutiny and evaluation",
                "bdd_change": "increased",
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "visual cortex",
            "striatum",
            "basal ganglia",
            "thalamus",
            "anterior cingulate",
            "prefrontal cortex",
            "orbitofrontal cortex",
        } else 0
        return (left_bonus, right_penalty, generic_penalty)

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
        """
        Helper for refining proxy searches against the atlas.
        Useful for tuning visual/parietal/premotor/striatal candidates.
        """
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

    def _main_component(
        self,
        region: Any,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None

        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)

        centroid = getattr(main, "centroid", None)
        if centroid is not None:
            try:
                centroid_xyz = tuple(float(x) for x in centroid)
            except Exception:
                centroid_xyz = None
        else:
            centroid_xyz = None

        try:
            volume_mm3 = float(getattr(main, "volume", float("nan")))
        except Exception:
            volume_mm3 = None

        return centroid_xyz, volume_mm3

    def _safe_features(self, concept: Any, modality: Any, **kwargs: Any) -> List[Any]:
        try:
            with siibra.QUIET:
                feats = siibra.features.get(concept, modality, **kwargs)
            return list(feats) if feats else []
        except Exception:
            return []

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features(
            region,
            siibra.features.molecular.ReceptorDensityFingerprint,
        )
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
        feats = self._safe_features(
            region,
            siibra.features.molecular.GeneExpressions,
            gene=list(genes),
        )
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

            summary = (
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
            )
            return summary

        return df.reset_index(drop=True)

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features(
            self.parcellation,
            siibra.features.connectivity.StreamlineCounts,
        )
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort),
            feats[0],
        )

        try:
            data = getattr(compound, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            self._connectivity_matrix = compound[0].data.copy()
        except Exception:
            self._connectivity_matrix = pd.DataFrame()

        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        exact = [x for x in labels if self._name_of(x) == region.name]
        if exact:
            return exact[0]

        rn = region.name.lower()
        fuzzy = [
            x
            for x in labels
            if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn
        ]
        return fuzzy[0] if fuzzy else None

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
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(
        self,
        node_keys: Sequence[str] = (
            "ofc",
            "acc",
            "striatum_proxy",
            "visual_cortex_proxy",
            "posterior_parietal_proxy",
            "premotor_proxy",
            "self_midline_proxy",
        ),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the BDD circuit.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        labels = []
        names = []

        for key in node_keys:
            region = self.region_objects.get(key)
            if region is None:
                continue

            match = self._match_region_label(list(matrix.index), region)
            if match is None:
                continue

            labels.append(match)
            names.append(region.name)

        if not labels:
            return pd.DataFrame()

        try:
            sub = matrix.loc[labels, labels].copy()
            sub.index = names
            sub.columns = names
            return sub
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
                self.connectivity_profiles[key] = pd.DataFrame()
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
            conn_df = self._connectivity_profile(region, max_rows=connectivity_rows)

            self.receptors[key] = receptor_df
            self.genes[key] = gene_df
            self.connectivity_profiles[key] = conn_df

            nodes.append(
                {
                    "key": key,
                    "label": region.name,
                    "node_type": "region",
                    "description": "Atlas-backed BDD circuit node",
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
        genetic_vulnerability: float,
        environmental_psychological_load: float,
        appearance_related_stress: float,
        right_hemisphere_body_image_insult: float,
        treatment_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while treatment_support is protective.
        """
        g = self._clip01(genetic_vulnerability)
        e = self._clip01(environmental_psychological_load)
        a = self._clip01(appearance_related_stress)
        r = self._clip01(right_hemisphere_body_image_insult)
        t = self._clip01(treatment_support)

        # Latent biology
        gene_environment_sensitization = self._clip01(
            0.30 * g + 0.30 * e + 0.15 * a - 0.15 * t
        )
        negative_affect_trait = self._clip01(
            0.25 * g + 0.25 * e + 0.20 * gene_environment_sensitization - 0.10 * t
        )
        ocrd_compulsivity = self._clip01(
            0.30 * g + 0.25 * gene_environment_sensitization + 0.10 * e - 0.15 * t
        )
        frontostriatal_dysregulation = self._clip01(
            0.40 * ocrd_compulsivity + 0.20 * negative_affect_trait - 0.10 * t
        )
        behavioral_inhibition_failure = self._clip01(
            0.40 * frontostriatal_dysregulation + 0.15 * negative_affect_trait - 0.10 * t
        )
        visual_detail_bias = self._clip01(
            0.30 * g + 0.30 * a + 0.15 * e - 0.05 * t
        )
        global_processing_failure = self._clip01(
            0.45 * visual_detail_bias + 0.15 * gene_environment_sensitization
        )
        body_schema_distortion = self._clip01(
            0.30 * global_processing_failure + 0.20 * a + 0.20 * r
        )
        self_referential_distortion = self._clip01(
            0.30 * negative_affect_trait
            + 0.25 * body_schema_distortion
            + 0.15 * a
            - 0.10 * t
        )

        # Regional state proxies
        ofc = self._clip01(
            0.40 * frontostriatal_dysregulation + 0.15 * behavioral_inhibition_failure - 0.10 * t
        )
        acc = self._clip01(
            0.35 * frontostriatal_dysregulation + 0.25 * self_referential_distortion - 0.10 * t
        )
        striatum_proxy = self._clip01(
            0.40 * frontostriatal_dysregulation + 0.15 * ocrd_compulsivity
        )
        visual_cortex_proxy = self._clip01(
            0.45 * visual_detail_bias + 0.25 * global_processing_failure
        )
        posterior_parietal_proxy = self._clip01(
            0.45 * body_schema_distortion + 0.20 * r
        )
        premotor_proxy = self._clip01(
            0.35 * body_schema_distortion + 0.20 * r
        )
        self_midline_proxy = self._clip01(
            0.45 * self_referential_distortion + 0.20 * negative_affect_trait - 0.10 * t
        )

        # Symptoms / behavior
        local_feature_overfocus = self._clip01(
            0.45 * visual_cortex_proxy + 0.25 * visual_detail_bias
        )
        body_image_distortion = self._clip01(
            0.30 * posterior_parietal_proxy
            + 0.20 * premotor_proxy
            + 0.25 * self_midline_proxy
            + 0.10 * body_schema_distortion
        )
        perceptual_distortion = self._clip01(
            0.35 * local_feature_overfocus
            + 0.35 * body_image_distortion
            + 0.15 * global_processing_failure
        )
        appearance_preoccupation = self._clip01(
            0.35 * self_midline_proxy
            + 0.25 * self_referential_distortion
            + 0.15 * negative_affect_trait
            + 0.10 * perceptual_distortion
            - 0.10 * t
        )
        compulsive_checking = self._clip01(
            0.35 * ocrd_compulsivity
            + 0.25 * frontostriatal_dysregulation
            + 0.20 * appearance_preoccupation
            - 0.15 * t
        )
        grooming_picking = self._clip01(
            0.35 * behavioral_inhibition_failure
            + 0.25 * compulsive_checking
            + 0.15 * frontostriatal_dysregulation
            - 0.10 * t
        )
        shame_inadequacy = self._clip01(
            0.35 * appearance_preoccupation
            + 0.30 * perceptual_distortion
            + 0.15 * negative_affect_trait
        )
        avoidance_distress = self._clip01(
            0.30 * shame_inadequacy
            + 0.25 * appearance_preoccupation
            + 0.20 * perceptual_distortion
            - 0.15 * t
        )

        return {
            "inputs": pd.Series(
                {
                    "genetic_vulnerability": g,
                    "environmental_psychological_load": e,
                    "appearance_related_stress": a,
                    "right_hemisphere_body_image_insult": r,
                    "treatment_support": t,
                }
            ),
            "latents": pd.Series(
                {
                    "frontostriatal_dysregulation": frontostriatal_dysregulation,
                    "self_referential_distortion": self_referential_distortion,
                    "body_schema_distortion": body_schema_distortion,
                    "visual_detail_bias": visual_detail_bias,
                    "ocrd_compulsivity": ocrd_compulsivity,
                    "behavioral_inhibition_failure": behavioral_inhibition_failure,
                    "negative_affect_trait": negative_affect_trait,
                    "gene_environment_sensitization": gene_environment_sensitization,
                    "global_processing_failure": global_processing_failure,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "ofc": ofc,
                    "acc": acc,
                    "striatum_proxy": striatum_proxy,
                    "visual_cortex_proxy": visual_cortex_proxy,
                    "posterior_parietal_proxy": posterior_parietal_proxy,
                    "premotor_proxy": premotor_proxy,
                    "self_midline_proxy": self_midline_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "appearance_preoccupation": appearance_preoccupation,
                    "perceptual_distortion": perceptual_distortion,
                    "body_image_distortion": body_image_distortion,
                    "compulsive_checking": compulsive_checking,
                    "grooming_picking": grooming_picking,
                    "local_feature_overfocus": local_feature_overfocus,
                    "shame_inadequacy": shame_inadequacy,
                    "avoidance_distress": avoidance_distress,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "ocrd_compulsive_ritual_profile": self._clip01(
                        0.45 * compulsive_checking
                        + 0.30 * grooming_picking
                        + 0.15 * ocrd_compulsivity
                    ),
                    "visual_distortion_profile": self._clip01(
                        0.45 * perceptual_distortion
                        + 0.25 * local_feature_overfocus
                        + 0.20 * body_image_distortion
                    ),
                    "body_schema_profile": self._clip01(
                        0.40 * body_image_distortion
                        + 0.30 * body_schema_distortion
                        + 0.20 * posterior_parietal_proxy
                    ),
                    "shame_avoidant_profile": self._clip01(
                        0.45 * avoidance_distress
                        + 0.25 * shame_inadequacy
                        + 0.20 * appearance_preoccupation
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-30, -70, -6)).head(10)
        """
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
        return assignments

    def region_mask(self, node_key: str):
        """
        Return a siibra regional mask object for a resolved node.
        Use .fetch() to obtain the NIfTI image.
        """
        region = self.region_objects[node_key]
        return region.get_regional_mask(self.space, maptype="labelled")


if __name__ == "__main__":
    model = BodyDysmorphicDisorderModel()

    # Build atlas-backed graph + evidence tables
    bundle = model.build()

    print("\n=== NODES ===")
    print(
        bundle["nodes"][
            ["key", "node_type", "atlas_region", "centroid_mni", "feature_summary"]
        ].to_string(index=False)
    )

    print("\n=== EDGES ===")
    print(
        bundle["edges"][["source", "target", "relation", "bdd_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in [
        "ofc",
        "acc",
        "striatum_proxy",
        "visual_cortex_proxy",
        "posterior_parietal_proxy",
        "premotor_proxy",
        "self_midline_proxy",
    ]:
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
        if not bundle["connectivity_profiles"][key].empty:
            print(bundle["connectivity_profiles"][key].head(10).to_string(index=False))
        else:
            print("No connectivity profile available for this node.")

    # Example simulation
    sim = model.simulate(
        genetic_vulnerability=0.75,
        environmental_psychological_load=0.60,
        appearance_related_stress=0.85,
        right_hemisphere_body_image_insult=0.20,
        treatment_support=0.15,
    )

    print("\n=== LATENT BIOLOGY ===")
    print(sim["latents"].to_string())

    print("\n=== REGIONAL STATE ===")
    print(sim["regional_state"].to_string())

    print("\n=== SYMPTOMS ===")
    print(sim["symptoms"].to_string())

    print("\n=== PHENOTYPES ===")
    print(sim["phenotypes"].to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((-30, -70, -6)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("visual cortex").to_string(index=False))
    # print(model.suggest_regions("parietal").to_string(index=False))
    # print(model.suggest_regions("caudate").to_string(index=False))
