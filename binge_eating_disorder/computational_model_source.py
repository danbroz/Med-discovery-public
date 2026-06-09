from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Binge-Eating Disorder-oriented panel:
# - dopamine / reward wanting
# - opioid / hedonic liking
# - endocannabinoid appetite enhancement
# - serotonin / satiety / impulse control
# - stress-response and epigenetic sensitization
DEFAULT_GENE_PANEL = [
    "DRD2",    # dopamine receptor
    "SLC6A3",  # dopamine transporter
    "COMT",    # catecholamine metabolism
    "OPRM1",   # mu-opioid receptor
    "CNR1",    # cannabinoid receptor 1
    "FAAH",    # endocannabinoid degradation
    "SLC6A4",  # serotonin transporter
    "HTR2C",   # serotonin receptor relevant to satiety/control
    "NR3C1",   # glucocorticoid receptor
    "FKBP5",   # stress responsivity
    "CRHR1",   # CRH signaling
    "LEP",     # leptin
    "LEPR",    # leptin receptor
    "MC4R",    # appetite regulation
    "BDNF",    # plasticity
]


class BingeEatingDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Binge-Eating Disorder (BED).

    What it does:
      1) Resolves chapter-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates craving, loss of control, cue-reactive bingeing,
         and stress-triggered binge eating.

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

        # BED chapter-consistent reward/control/emotion circuit.
        # NAc is implemented as a ventral-striatal proxy because exact naming
        # can vary across atlas versions.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "nac_proxy": [
                "Nucleus accumbens left",
                "Accumbens left",
                "accumbens",
                "ventral striatum",
            ],
            "pfc_control": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "prefrontal",
            ],
            "acc": [
                "Area 33 (ACC) left",
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "ACC",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "orbitofrontal",
                "Fo4",
                "Fo3",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": "Polygenic liability affecting reward, satiety, stress response, and impulse control",
            "early_life_stress": "Early stress exposure that can amplify later emotional eating vulnerability",
            "palatable_food_cues": "Environmental cues linked to highly palatable food",
            "negative_affect_stress": "Negative mood and emotional distress that trigger eating",
            "maternal_metabolic_programming": "Developmental programming from maternal high-fat/high-sugar exposure",
            "recovery_support": "Protective treatment, structure, and recovery support",
        }

        self.latent_nodes: Dict[str, str] = {
            "gene_environment_sensitization": "Epigenetic amplification of binge-eating vulnerability",
            "hpa_axis_dysregulation": "Stress-response dysregulation promoting emotional eating",
            "dopamine_wanting": "Incentive salience / craving for palatable food",
            "opioid_liking": "Hedonic reinforcement of palatable intake",
            "endocannabinoid_appetite_drive": "Appetite-enhancing reward amplification",
            "serotonin_satiety_impulse_dysregulation": "Reduced satiety signaling and impaired impulse control",
            "satiety_failure": "Impaired termination of intake despite rising food consumption",
            "food_addiction_like_process": "Tolerance/withdrawal-like and compulsive reward dynamics",
            "emotional_eating_drive": "Use of eating to modulate negative affect",
            "executive_control_network_weakness": "Reduced recruitment of deliberate inhibitory control",
            "amygdala_nac_coupling": "Stronger emotional-reward coupling to food cues or stress",
        }

        self.symptom_nodes: Dict[str, str] = {
            "loss_of_control": "Subjective loss of control over eating",
            "craving": "Strong desire or urge for palatable food",
            "binge_eating": "Consumption of large quantities with diminished control",
            "tolerance_escalation": "Need for greater intake to achieve the same effect",
            "withdrawal_like_distress": "Irritability/anxiety-like distress when restricting intake",
            "cue_triggered_bingeing": "Bingeing precipitated by food cues",
            "stress_triggered_bingeing": "Bingeing precipitated by negative emotion or stress",
            "impulsive_eating": "Fast, poorly inhibited eating behavior",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "gene_environment_sensitization",
                "relation": "creates heritable vulnerability shaped by the environment",
                "bed_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "dopamine_wanting",
                "relation": "raises vulnerability to reward-driven food motivation",
                "bed_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "serotonin_satiety_impulse_dysregulation",
                "relation": "raises vulnerability to satiety and impulse-control dysregulation",
                "bed_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "satiety_failure",
                "relation": "raises vulnerability to impaired meal termination",
                "bed_change": "increased susceptibility",
            },
            {
                "source": "early_life_stress",
                "target": "gene_environment_sensitization",
                "relation": "induces lasting molecular changes that amplify later risk",
                "bed_change": "increased",
            },
            {
                "source": "early_life_stress",
                "target": "hpa_axis_dysregulation",
                "relation": "programs altered stress responsivity",
                "bed_change": "increased",
            },
            {
                "source": "negative_affect_stress",
                "target": "emotional_eating_drive",
                "relation": "pushes eating toward affect regulation",
                "bed_change": "increased",
            },
            {
                "source": "palatable_food_cues",
                "target": "dopamine_wanting",
                "relation": "loads incentive salience and craving",
                "bed_change": "increased",
            },
            {
                "source": "palatable_food_cues",
                "target": "amygdala_nac_coupling",
                "relation": "strengthens emotional-reward cue reactivity",
                "bed_change": "increased",
            },
            {
                "source": "maternal_metabolic_programming",
                "target": "gene_environment_sensitization",
                "relation": "can program later metabolic and appetite vulnerability",
                "bed_change": "increased",
            },
            {
                "source": "maternal_metabolic_programming",
                "target": "satiety_failure",
                "relation": "can bias appetite-regulating systems toward dysregulation",
                "bed_change": "increased",
            },
            {
                "source": "maternal_metabolic_programming",
                "target": "endocannabinoid_appetite_drive",
                "relation": "can bias reward-appetite systems toward heightened responsivity",
                "bed_change": "increased",
            },
            {
                "source": "recovery_support",
                "target": "executive_control_network_weakness",
                "relation": "supports recruitment of deliberate control",
                "bed_change": "protective",
            },
            {
                "source": "recovery_support",
                "target": "binge_eating",
                "relation": "buffers expression of loss-of-control eating",
                "bed_change": "protective",
            },
            {
                "source": "gene_environment_sensitization",
                "target": "hpa_axis_dysregulation",
                "relation": "stabilizes stress-linked eating vulnerability",
                "bed_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "emotional_eating_drive",
                "relation": "links stress physiology to eating for relief",
                "bed_change": "increased",
            },
            {
                "source": "dopamine_wanting",
                "target": "nac_proxy",
                "relation": "loads ventral-striatal reward motivation",
                "bed_change": "increased reward drive",
            },
            {
                "source": "dopamine_wanting",
                "target": "craving",
                "relation": "drives incentive salience and wanting",
                "bed_change": "increased",
            },
            {
                "source": "opioid_liking",
                "target": "food_addiction_like_process",
                "relation": "supports hedonic reinforcement of binge eating",
                "bed_change": "increased",
            },
            {
                "source": "endocannabinoid_appetite_drive",
                "target": "craving",
                "relation": "enhances appetite and reward value of food",
                "bed_change": "increased",
            },
            {
                "source": "endocannabinoid_appetite_drive",
                "target": "satiety_failure",
                "relation": "promotes sustained drive to keep eating",
                "bed_change": "increased",
            },
            {
                "source": "serotonin_satiety_impulse_dysregulation",
                "target": "satiety_failure",
                "relation": "weakens satiety and behavioral inhibition",
                "bed_change": "increased",
            },
            {
                "source": "serotonin_satiety_impulse_dysregulation",
                "target": "executive_control_network_weakness",
                "relation": "impairs inhibitory control over eating responses",
                "bed_change": "increased",
            },
            {
                "source": "satiety_failure",
                "target": "loss_of_control",
                "relation": "weakens the ability to stop eating once started",
                "bed_change": "increased",
            },
            {
                "source": "food_addiction_like_process",
                "target": "tolerance_escalation",
                "relation": "supports needing more food for the same effect",
                "bed_change": "increased",
            },
            {
                "source": "food_addiction_like_process",
                "target": "withdrawal_like_distress",
                "relation": "supports anxiety/irritability-like responses when intake is restricted",
                "bed_change": "increased",
            },
            {
                "source": "food_addiction_like_process",
                "target": "craving",
                "relation": "stabilizes compulsive appetitive drive",
                "bed_change": "increased",
            },
            {
                "source": "emotional_eating_drive",
                "target": "amygdala",
                "relation": "loads emotional-processing circuitry during distress",
                "bed_change": "increased dysregulation",
            },
            {
                "source": "emotional_eating_drive",
                "target": "stress_triggered_bingeing",
                "relation": "directly promotes binge eating under negative affect",
                "bed_change": "increased",
            },
            {
                "source": "amygdala_nac_coupling",
                "target": "amygdala",
                "relation": "strengthens food-cue and stress-linked emotional salience",
                "bed_change": "increased dysregulation",
            },
            {
                "source": "amygdala_nac_coupling",
                "target": "nac_proxy",
                "relation": "strengthens emotional-reward coupling to food",
                "bed_change": "increased dysregulation",
            },
            {
                "source": "amygdala_nac_coupling",
                "target": "cue_triggered_bingeing",
                "relation": "biases behavior toward emotionally amplified cue-driven eating",
                "bed_change": "increased",
            },
            {
                "source": "executive_control_network_weakness",
                "target": "pfc_control",
                "relation": "reflects weaker executive-control recruitment",
                "bed_change": "reduced function",
            },
            {
                "source": "executive_control_network_weakness",
                "target": "acc",
                "relation": "reflects weaker monitoring and inhibition of eating responses",
                "bed_change": "reduced function",
            },
            {
                "source": "executive_control_network_weakness",
                "target": "loss_of_control",
                "relation": "reduces deliberate control over eating",
                "bed_change": "increased",
            },
            {
                "source": "executive_control_network_weakness",
                "target": "impulsive_eating",
                "relation": "biases behavior toward rapid, poorly inhibited intake",
                "bed_change": "increased",
            },
            {
                "source": "ofc",
                "target": "craving",
                "relation": "reward valuation bias can amplify desire for palatable food",
                "bed_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "stress_triggered_bingeing",
                "relation": "emotional salience amplifies distress-driven eating",
                "bed_change": "increased",
            },
            {
                "source": "nac_proxy",
                "target": "craving",
                "relation": "supports reward-linked wanting",
                "bed_change": "increased",
            },
            {
                "source": "loss_of_control",
                "target": "binge_eating",
                "relation": "directly promotes binge episodes",
                "bed_change": "increased",
            },
            {
                "source": "craving",
                "target": "binge_eating",
                "relation": "promotes compulsive approach to palatable food",
                "bed_change": "increased",
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
        """
        Prefer left-sided, more specific names over generic/right-sided parents.
        Lower tuple is better.
        """
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "anterior cingulate",
            "prefrontal cortex",
            "orbitofrontal cortex",
            "ventral striatum",
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
        Useful for tuning NAc or frontal-control candidates.
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
        node_keys: Sequence[str] = ("amygdala", "nac_proxy", "pfc_control", "acc", "ofc"),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the BED circuit.
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
                    "description": "Atlas-backed BED circuit node",
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
        palatable_food_cues: float,
        negative_affect_stress: float,
        maternal_metabolic_programming: float,
        recovery_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while recovery_support is protective.
        """
        g = self._clip01(genetic_vulnerability)
        e = self._clip01(early_life_stress)
        c = self._clip01(palatable_food_cues)
        s = self._clip01(negative_affect_stress)
        m = self._clip01(maternal_metabolic_programming)
        r = self._clip01(recovery_support)

        # Latent biology
        gene_environment_sensitization = self._clip01(
            0.30 * g + 0.25 * e + 0.20 * m - 0.10 * r
        )
        hpa_axis_dysregulation = self._clip01(
            0.30 * e + 0.25 * s + 0.20 * gene_environment_sensitization - 0.10 * r
        )
        dopamine_wanting = self._clip01(
            0.25 * g + 0.30 * c + 0.10 * s - 0.10 * r
        )
        opioid_liking = self._clip01(
            0.30 * c + 0.15 * dopamine_wanting
        )
        endocannabinoid_appetite_drive = self._clip01(
            0.20 * g + 0.25 * c + 0.15 * m + 0.10 * s
        )
        serotonin_satiety_impulse_dysregulation = self._clip01(
            0.30 * g + 0.15 * e + 0.15 * s - 0.10 * r
        )
        satiety_failure = self._clip01(
            0.30 * serotonin_satiety_impulse_dysregulation
            + 0.25 * endocannabinoid_appetite_drive
            + 0.15 * m
        )
        emotional_eating_drive = self._clip01(
            0.35 * s + 0.25 * hpa_axis_dysregulation + 0.15 * gene_environment_sensitization - 0.10 * r
        )
        food_addiction_like_process = self._clip01(
            0.25 * dopamine_wanting
            + 0.20 * opioid_liking
            + 0.20 * c
            + 0.15 * endocannabinoid_appetite_drive
            + 0.10 * satiety_failure
        )
        executive_control_network_weakness = self._clip01(
            0.25 * e
            + 0.25 * s
            + 0.20 * food_addiction_like_process
            + 0.10 * gene_environment_sensitization
            - 0.30 * r
        )
        amygdala_nac_coupling = self._clip01(
            0.25 * c + 0.25 * emotional_eating_drive + 0.20 * food_addiction_like_process + 0.10 * s
        )

        # Regional state proxies
        amygdala = self._clip01(
            0.35 * emotional_eating_drive + 0.20 * hpa_axis_dysregulation + 0.15 * amygdala_nac_coupling
        )
        nac_proxy = self._clip01(
            0.35 * dopamine_wanting + 0.25 * food_addiction_like_process + 0.20 * amygdala_nac_coupling
        )
        pfc_control = self._clip01(
            0.45 * executive_control_network_weakness + 0.10 * s - 0.15 * r
        )
        acc = self._clip01(
            0.35 * executive_control_network_weakness + 0.15 * amygdala - 0.10 * r
        )
        ofc = self._clip01(
            0.30 * dopamine_wanting
            + 0.25 * food_addiction_like_process
            + 0.20 * c
            + 0.10 * endocannabinoid_appetite_drive
            - 0.10 * r
        )

        # Symptoms / behavior
        craving = self._clip01(
            0.30 * nac_proxy
            + 0.20 * dopamine_wanting
            + 0.15 * c
            + 0.15 * food_addiction_like_process
            + 0.10 * ofc
        )
        loss_of_control = self._clip01(
            0.30 * executive_control_network_weakness
            + 0.25 * craving
            + 0.20 * satiety_failure
            + 0.10 * emotional_eating_drive
            - 0.10 * r
        )
        impulsive_eating = self._clip01(
            0.35 * loss_of_control + 0.20 * pfc_control + 0.15 * acc - 0.10 * r
        )
        cue_triggered_bingeing = self._clip01(
            0.35 * craving + 0.25 * amygdala_nac_coupling + 0.15 * c
        )
        stress_triggered_bingeing = self._clip01(
            0.35 * emotional_eating_drive + 0.20 * amygdala + 0.15 * hpa_axis_dysregulation
        )
        withdrawal_like_distress = self._clip01(
            0.30 * food_addiction_like_process
            + 0.20 * craving
            + 0.20 * hpa_axis_dysregulation
            + 0.10 * s
        )
        tolerance_escalation = self._clip01(
            0.35 * food_addiction_like_process + 0.25 * craving + 0.15 * dopamine_wanting
        )
        binge_eating = self._clip01(
            0.25 * loss_of_control
            + 0.20 * craving
            + 0.20 * cue_triggered_bingeing
            + 0.20 * stress_triggered_bingeing
            + 0.10 * satiety_failure
            - 0.20 * r
        )

        return {
            "inputs": pd.Series(
                {
                    "genetic_vulnerability": g,
                    "early_life_stress": e,
                    "palatable_food_cues": c,
                    "negative_affect_stress": s,
                    "maternal_metabolic_programming": m,
                    "recovery_support": r,
                }
            ),
            "latents": pd.Series(
                {
                    "food_addiction_like_process": food_addiction_like_process,
                    "executive_control_network_weakness": executive_control_network_weakness,
                    "amygdala_nac_coupling": amygdala_nac_coupling,
                    "craving_drive_dopamine": dopamine_wanting,
                    "satiety_failure": satiety_failure,
                    "emotional_eating_drive": emotional_eating_drive,
                    "hpa_axis_dysregulation": hpa_axis_dysregulation,
                    "gene_environment_sensitization": gene_environment_sensitization,
                    "opioid_liking": opioid_liking,
                    "endocannabinoid_appetite_drive": endocannabinoid_appetite_drive,
                    "serotonin_satiety_impulse_dysregulation": serotonin_satiety_impulse_dysregulation,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "amygdala": amygdala,
                    "nac_proxy": nac_proxy,
                    "pfc_control": pfc_control,
                    "acc": acc,
                    "ofc": ofc,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "binge_eating": binge_eating,
                    "loss_of_control": loss_of_control,
                    "craving": craving,
                    "cue_triggered_bingeing": cue_triggered_bingeing,
                    "stress_triggered_bingeing": stress_triggered_bingeing,
                    "impulsive_eating": impulsive_eating,
                    "tolerance_escalation": tolerance_escalation,
                    "withdrawal_like_distress": withdrawal_like_distress,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "food_addiction_like_profile": self._clip01(
                        0.45 * food_addiction_like_process
                        + 0.25 * tolerance_escalation
                        + 0.20 * withdrawal_like_distress
                    ),
                    "emotional_binge_profile": self._clip01(
                        0.45 * stress_triggered_bingeing
                        + 0.25 * emotional_eating_drive
                        + 0.20 * binge_eating
                    ),
                    "cue_reactive_binge_profile": self._clip01(
                        0.45 * cue_triggered_bingeing
                        + 0.25 * craving
                        + 0.20 * binge_eating
                    ),
                    "control_deficit_binge_profile": self._clip01(
                        0.40 * loss_of_control
                        + 0.30 * impulsive_eating
                        + 0.20 * binge_eating
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-10, 10, -8)).head(10)
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
    model = BingeEatingDisorderModel()

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
        bundle["edges"][["source", "target", "relation", "bed_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["amygdala", "nac_proxy", "pfc_control", "acc", "ofc"]:
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
        early_life_stress=0.55,
        palatable_food_cues=0.90,
        negative_affect_stress=0.75,
        maternal_metabolic_programming=0.45,
        recovery_support=0.20,
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
    # print(model.assign_mni_point((-10, 10, -8)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("accumbens").to_string(index=False))
    # print(model.suggest_regions("orbitofrontal").to_string(index=False))
    # print(model.suggest_regions("cingulate").to_string(index=False))
