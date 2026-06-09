from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Anorexia Nervosa-oriented panel:
# - reward / dopamine
# - serotonin / anxiety-depression overlap
# - stress / HPA-axis
# - puberty / sex-hormone sensitivity
# - appetite / weight-regulation biology
DEFAULT_GENE_PANEL = [
    "drd2",
    "slc6a3",
    "comt",
    "maoa",
    "slc6a4",
    "htr1a",
    "nr3c1",
    "fkbp5",
    "crhr1",
    "bdnf",
    "esr1",
    "esr2",
    "lep",
    "lepr",
    "mc4r",
]


class AnorexiaNervosaModel:
    """
    Atlas-grounded mechanistic scaffold for Anorexia Nervosa (AN) using siibra.

    What it does:
      1) Resolves AN-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates interoception, overcontrol, reward inversion, and starvation maintenance.

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
        self.parcellation = self.atlas.parcellations.get(parcellation_spec)
        self.space_spec = space_spec
        self.space = self.atlas.spaces.get(space_spec)
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        # Left-sided representative nodes keep the graph compact.
        # dLPFC / vmPFC are functional concepts, so these are Julich proxies.
        self.region_candidates: Dict[str, List[str]] = {
            "insula": [
                "Area Id2 (Insula) left",
                "Area Id3 (Insula) left",
                "Area Id4 (Insula) left",
                "insula",
            ],
            "acc": [
                "Area 33 (ACC) left",
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "ACC",
            ],
            "dlpfc": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "prefrontal",
            ],
            "vmpfc": [
                "Area s32 (sACC) left",
                "Area p32 (pACC) left",
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Area Fp2 (FPole) left",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": "Polygenic susceptibility involving body weight, appetite, metabolism, and personality",
            "early_life_stress": "Early stress or adversity shaping long-term behavioral and physiological responses",
            "pubertal_transition": "Pubertal neuroendocrine activation and female developmental sensitivity",
            "thinness_pressure": "Sociocultural pressure toward thinness",
            "nutritional_restoration": "Protective nutritional rehabilitation and recovery support",
        }

        self.latent_nodes: Dict[str, str] = {
            "appetite_metabolic_susceptibility": "Inherited vulnerability in appetite / body-weight regulation",
            "pubertal_hormone_sensitivity": "Sex-hormone-linked developmental sensitivity",
            "mesolimbic_dopamine": "Reward / motivation circuitry relevant to food and restriction",
            "restriction_reinforcement": "Inversion of reward so restriction becomes reinforcing",
            "interoceptive_distortion": "Distorted hunger, satiety, and internal-body perception",
            "thinness_overvaluation": "Excessively high value assigned to thinness",
            "self_discrepancy": "Persistent mismatch between actual and idealized body state",
            "hpa_axis": "Stress-hormone dysregulation",
            "malnutrition_stress": "Brain dysfunction perpetuated by starvation and physiological stress",
        }

        self.symptom_nodes: Dict[str, str] = {
            "restrictive_eating": "Persistent restrictive eating behavior",
            "hunger_satiety_misperception": "Altered perception of hunger and fullness",
            "fear_weight_gain": "Persistent fear of weight gain",
            "body_image_discrepancy": "Pathological discrepancy between actual and ideal body",
            "anxiety_depression": "High anxiety / depressive affect / negative emotionality",
            "compulsive_thinness_pursuit": "Compulsive pursuit of thinness over homeostatic need",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "appetite_metabolic_susceptibility",
                "relation": "creates polygenic vulnerability in body-weight and appetite regulation",
                "an_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "mesolimbic_dopamine",
                "relation": "biases reward and motivational processing",
                "an_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "thinness_overvaluation",
                "relation": "may contribute to stable vulnerability traits",
                "an_change": "increased susceptibility",
            },
            {
                "source": "early_life_stress",
                "target": "hpa_axis",
                "relation": "programs long-term stress responsivity",
                "an_change": "increased",
            },
            {
                "source": "early_life_stress",
                "target": "interoceptive_distortion",
                "relation": "alters long-term physiological and behavioral regulation",
                "an_change": "increased",
            },
            {
                "source": "pubertal_transition",
                "target": "pubertal_hormone_sensitivity",
                "relation": "activates sex-hormone-related neurodevelopmental processes",
                "an_change": "increased",
            },
            {
                "source": "thinness_pressure",
                "target": "thinness_overvaluation",
                "relation": "raises the subjective value of thinness",
                "an_change": "increased",
            },
            {
                "source": "thinness_pressure",
                "target": "self_discrepancy",
                "relation": "amplifies discrepancy between actual and ideal body state",
                "an_change": "increased",
            },
            {
                "source": "nutritional_restoration",
                "target": "malnutrition_stress",
                "relation": "reduces starvation-driven brain dysfunction",
                "an_change": "protective",
            },
            {
                "source": "nutritional_restoration",
                "target": "hpa_axis",
                "relation": "buffers starvation-linked stress physiology",
                "an_change": "protective",
            },
            {
                "source": "pubertal_hormone_sensitivity",
                "target": "thinness_overvaluation",
                "relation": "creates a biological-developmental vulnerability around body change",
                "an_change": "increased",
            },
            {
                "source": "mesolimbic_dopamine",
                "target": "restriction_reinforcement",
                "relation": "supports reward-circuit inversion in favor of restriction",
                "an_change": "increased",
            },
            {
                "source": "thinness_overvaluation",
                "target": "restriction_reinforcement",
                "relation": "makes restriction behavior more rewarding than eating",
                "an_change": "increased",
            },
            {
                "source": "interoceptive_distortion",
                "target": "insula",
                "relation": "loads distorted internal-body-state processing",
                "an_change": "increased",
            },
            {
                "source": "self_discrepancy",
                "target": "acc",
                "relation": "drives conflict monitoring and error signaling around body state",
                "an_change": "increased",
            },
            {
                "source": "thinness_overvaluation",
                "target": "dlpfc",
                "relation": "biases self-control toward overcontrolled intake restriction",
                "an_change": "increased control bias",
            },
            {
                "source": "thinness_overvaluation",
                "target": "vmpfc",
                "relation": "biases value assignment toward thinness over homeostasis",
                "an_change": "increased valuation bias",
            },
            {
                "source": "hpa_axis",
                "target": "amygdala",
                "relation": "amplifies stress-linked emotional dysregulation",
                "an_change": "increased",
            },
            {
                "source": "malnutrition_stress",
                "target": "hpa_axis",
                "relation": "perpetuates stress physiology",
                "an_change": "increased",
            },
            {
                "source": "malnutrition_stress",
                "target": "interoceptive_distortion",
                "relation": "distorts hunger and satiety cues further",
                "an_change": "increased",
            },
            {
                "source": "restriction_reinforcement",
                "target": "restrictive_eating",
                "relation": "makes restriction behavior self-reinforcing",
                "an_change": "increased",
            },
            {
                "source": "acc",
                "target": "body_image_discrepancy",
                "relation": "supports persistent discrepancy monitoring",
                "an_change": "increased",
            },
            {
                "source": "insula",
                "target": "hunger_satiety_misperception",
                "relation": "malfunction distorts hunger and fullness perception",
                "an_change": "increased",
            },
            {
                "source": "dlpfc",
                "target": "restrictive_eating",
                "relation": "supports rigid top-down self-control over intake",
                "an_change": "increased",
            },
            {
                "source": "vmpfc",
                "target": "restrictive_eating",
                "relation": "assigns higher value to thinness than to feeding",
                "an_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "anxiety_depression",
                "relation": "supports negative affect and emotional dysregulation",
                "an_change": "increased",
            },
            {
                "source": "anxiety_depression",
                "target": "restrictive_eating",
                "relation": "restriction may be used as maladaptive affect regulation",
                "an_change": "increased",
            },
            {
                "source": "body_image_discrepancy",
                "target": "fear_weight_gain",
                "relation": "drives fear of weight gain and body change",
                "an_change": "increased",
            },
            {
                "source": "fear_weight_gain",
                "target": "restrictive_eating",
                "relation": "strengthens food restriction",
                "an_change": "increased",
            },
            {
                "source": "restrictive_eating",
                "target": "malnutrition_stress",
                "relation": "drives physiological stress and starvation maintenance",
                "an_change": "increased",
            },
            {
                "source": "restrictive_eating",
                "target": "compulsive_thinness_pursuit",
                "relation": "stabilizes the pathological pursuit of thinness",
                "an_change": "increased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()

        self._pmap = None
        self._connectivity_feature = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

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
            try:
                matches = self.atlas.find_regions(
                    spec,
                    all_versions=False,
                    filter_children=False,
                    find_topmost=False,
                )
                for region in matches:
                    parc_name = getattr(getattr(region, "parcellation", None), "name", "")
                    if "julich" in str(parc_name).lower():
                        return region
            except Exception:
                pass
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        """
        Helper for tuning region proxies against the atlas.
        Useful for refining dLPFC/vmPFC candidates.
        """
        rows: List[Dict[str, Any]] = []
        seen = set()

        try:
            matches = self.atlas.find_regions(
                keyword,
                all_versions=False,
                filter_children=True,
                find_topmost=False,
            )
        except Exception:
            return pd.DataFrame(columns=["name", "identifier", "parcellation"])

        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" not in str(parc_name).lower():
                continue

            row = (
                self._name_of(region),
                getattr(region, "identifier", None),
                parc_name,
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
                features = siibra.features.get(concept, modality, **kwargs)
            return list(features) if features else []
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

    def _get_connectivity_feature(self):
        if self._connectivity_feature is not None:
            return self._connectivity_feature

        feats = self._safe_features(self.parcellation, "StreamlineCounts")
        if not feats:
            return None

        self._connectivity_feature = next(
            (f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort),
            feats[0],
        )
        return self._connectivity_feature

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        cf = self._get_connectivity_feature()
        if cf is None:
            return pd.DataFrame()

        try:
            profile = cf.get_profile(region, max_rows=max_rows)
            df = getattr(profile, "data", profile)
            if not isinstance(df, pd.DataFrame):
                return pd.DataFrame()
            df = df.copy().reset_index()
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "connected_region"})
            df["connected_region"] = df["connected_region"].map(self._name_of)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        cf = self._get_connectivity_feature()
        if cf is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        try:
            # Use the first subject-level matrix as a representative matrix.
            self._connectivity_matrix = cf[0].data.copy()
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

    def circuit_connectivity(
        self,
        node_keys: Sequence[str] = ("insula", "acc", "dlpfc", "vmpfc", "amygdala"),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the AN circuit.
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
                    "description": "Atlas-backed AN circuit node",
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
        early_life_stress: float,
        pubertal_transition: float,
        thinness_pressure: float,
        nutritional_restoration: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        except nutritional_restoration which is protective.
        """
        g = self._clip01(genetic_vulnerability)
        e = self._clip01(early_life_stress)
        p = self._clip01(pubertal_transition)
        t = self._clip01(thinness_pressure)
        n = self._clip01(nutritional_restoration)

        # initial latent biology
        appetite_metabolic_susceptibility = self._clip01(0.35 * g + 0.10 * e)
        pubertal_hormone_sensitivity = self._clip01(0.35 * p + 0.10 * g)
        thinness_overvaluation = self._clip01(
            0.35 * t + 0.20 * pubertal_hormone_sensitivity + 0.15 * g
        )
        self_discrepancy = self._clip01(0.30 * t + 0.25 * thinness_overvaluation)
        hpa_axis_pre = self._clip01(0.30 * e + 0.10 * t + 0.10 * g - 0.10 * n)
        mesolimbic_dopamine = self._clip01(
            0.25 * g + 0.20 * thinness_overvaluation + 0.10 * hpa_axis_pre
        )
        restriction_reinforcement = self._clip01(
            0.35 * thinness_overvaluation + 0.25 * mesolimbic_dopamine + 0.10 * e
        )
        interoceptive_distortion_pre = self._clip01(
            0.25 * e + 0.20 * g + 0.10 * hpa_axis_pre
        )

        # regional state proxies
        amygdala = self._clip01(0.30 * hpa_axis_pre + 0.20 * e + 0.10 * t)
        insula = self._clip01(0.40 * interoceptive_distortion_pre + 0.10 * hpa_axis_pre)
        acc = self._clip01(0.35 * self_discrepancy + 0.10 * thinness_overvaluation)
        dlpfc = self._clip01(0.35 * thinness_overvaluation + 0.20 * self_discrepancy - 0.05 * n)
        vmpfc = self._clip01(0.35 * thinness_overvaluation + 0.20 * restriction_reinforcement - 0.05 * n)

        # first-pass symptoms
        body_image_discrepancy = self._clip01(
            0.30 * acc + 0.25 * self_discrepancy + 0.10 * insula
        )
        fear_weight_gain = self._clip01(
            0.35 * body_image_discrepancy + 0.20 * amygdala + 0.10 * vmpfc
        )
        anxiety_depression = self._clip01(
            0.35 * amygdala + 0.20 * hpa_axis_pre + 0.10 * acc
        )
        restrictive_eating = self._clip01(
            0.25 * restriction_reinforcement
            + 0.25 * thinness_overvaluation
            + 0.20 * fear_weight_gain
            + 0.15 * dlpfc
            + 0.10 * anxiety_depression
            - 0.25 * n
        )

        # starvation feedback loop
        malnutrition_stress = self._clip01(
            0.40 * restrictive_eating + 0.15 * hpa_axis_pre - 0.35 * n
        )
        hpa_axis = self._clip01(hpa_axis_pre + 0.20 * malnutrition_stress)
        interoceptive_distortion = self._clip01(
            interoceptive_distortion_pre + 0.25 * malnutrition_stress - 0.15 * n
        )
        insula_final = self._clip01(0.45 * interoceptive_distortion + 0.10 * hpa_axis)

        # final symptoms / maintenance
        hunger_satiety_misperception = self._clip01(
            0.45 * insula_final + 0.15 * malnutrition_stress
        )
        compulsive_thinness_pursuit = self._clip01(
            0.35 * restrictive_eating
            + 0.25 * thinness_overvaluation
            + 0.20 * restriction_reinforcement
            + 0.10 * fear_weight_gain
        )

        return {
            "inputs": pd.Series(
                {
                    "genetic_vulnerability": g,
                    "early_life_stress": e,
                    "pubertal_transition": p,
                    "thinness_pressure": t,
                    "nutritional_restoration": n,
                }
            ),
            "latents": pd.Series(
                {
                    "thinness_overvaluation": thinness_overvaluation,
                    "restriction_reinforcement": restriction_reinforcement,
                    "interoceptive_distortion": interoceptive_distortion,
                    "hpa_axis": hpa_axis,
                    "self_discrepancy": self_discrepancy,
                    "mesolimbic_dopamine": mesolimbic_dopamine,
                    "malnutrition_stress": malnutrition_stress,
                    "pubertal_hormone_sensitivity": pubertal_hormone_sensitivity,
                    "appetite_metabolic_susceptibility": appetite_metabolic_susceptibility,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "insula": insula_final,
                    "amygdala": amygdala,
                    "acc": acc,
                    "dlpfc": dlpfc,
                    "vmpfc": vmpfc,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "restrictive_eating": restrictive_eating,
                    "compulsive_thinness_pursuit": compulsive_thinness_pursuit,
                    "fear_weight_gain": fear_weight_gain,
                    "body_image_discrepancy": body_image_discrepancy,
                    "hunger_satiety_misperception": hunger_satiety_misperception,
                    "anxiety_depression": anxiety_depression,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "reward_inverted_restrictive_profile": self._clip01(
                        0.45 * restriction_reinforcement
                        + 0.30 * restrictive_eating
                        + 0.15 * thinness_overvaluation
                    ),
                    "anxious_overcontrolled_profile": self._clip01(
                        0.40 * anxiety_depression
                        + 0.25 * fear_weight_gain
                        + 0.20 * restrictive_eating
                        + 0.10 * dlpfc
                    ),
                    "starvation_maintenance_profile": self._clip01(
                        0.40 * restrictive_eating
                        + 0.25 * malnutrition_stress
                        + 0.15 * interoceptive_distortion
                        + 0.10 * appetite_metabolic_susceptibility
                        - 0.20 * n
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-34, 18, 4)).head(10)
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
    model = AnorexiaNervosaModel()

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
        bundle["edges"][["source", "target", "relation", "an_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["insula", "acc", "dlpfc", "vmpfc", "amygdala"]:
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
        genetic_vulnerability=0.70,
        early_life_stress=0.50,
        pubertal_transition=0.85,
        thinness_pressure=0.90,
        nutritional_restoration=0.20,
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
    # print(model.assign_mni_point((-34, 18, 4)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("insula").to_string(index=False))
    # print(model.suggest_regions("cingulate").to_string(index=False))
    # print(model.suggest_regions("frontal").to_string(index=False))
