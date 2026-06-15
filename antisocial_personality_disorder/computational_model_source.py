from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# ASPD-oriented panel:
# - serotonin / impulse-aggression
# - dopamine / reward-seeking
# - HPA-axis stress regulation
# - inflammation
# - hormonal modulation / developmental sensitivity
DEFAULT_GENE_PANEL = [
    "SLC6A4",  # serotonin transporter
    "HTR1A",   # serotonin receptor
    "HTR2A",   # serotonin receptor
    "TPH2",    # serotonin synthesis
    "MAOA",    # monoamine metabolism / aggression relevance
    "DRD2",    # dopamine receptor
    "SLC6A3",  # dopamine transporter
    "COMT",    # catecholamine metabolism
    "NR3C1",   # glucocorticoid receptor
    "FKBP5",   # stress responsivity
    "CRHR1",   # CRH signalling
    "IL1B",    # inflammation
    "IL6",     # inflammation
    "TNF",     # inflammation
    "AR",      # androgen receptor
]


class AntisocialPersonalityDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Antisocial Personality Disorder (ASPD).

    What it does:
      1) Resolves ASPD-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates developmental risk, impulsive aggression, callousness,
         and persistent antisocial behavior.

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

        # The chapter mainly implies a prefrontal-limbic model.
        # dLPFC / vmPFC / OFC are represented with Julich-compatible proxies.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
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
                "prefrontal",
            ],
            "vmpfc": [
                "Area s32 (sACC) left",
                "Area p32 (pACC) left",
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Fo4",
                "Fo3",
                "orbitofrontal",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": "Polygenic liability for impulsivity, negative emotionality, and persistent antisocial traits",
            "childhood_adversity": "Abuse, neglect, family dysfunction, and social stress",
            "developmental_insult": "Disrupted maturation from early insults, prenatal exposures, or acquired injuries",
            "adolescent_hormonal_sensitivity": "Hormonal modulation of aggression and risk-taking during development",
            "alcohol_disinhibition": "Aggression- and judgment-disinhibiting effects of alcohol misuse",
            "protective_environment": "Protective caregiving, social buffering, and prosocial developmental supports",
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_deficit": "Reduced inhibitory serotonergic regulation of aggression and impulse control",
            "dopaminergic_reward_bias": "Reward system bias toward immediate gratification and risk-taking",
            "hpa_axis_dysregulation": "Stress-response dysregulation shaped by early adversity",
            "low_grade_inflammation": "Inflammatory burden that may alter neurotransmitter function",
            "prefrontal_limbic_maturation_disruption": "Impaired development/integrity of control-emotion circuits",
            "punishment_insensitivity": "Reduced behavioral impact of punishment or negative consequences",
            "empathy_moral_circuit_failure": "Blunted emotional-social processing supporting callous behavior",
        }

        self.symptom_nodes: Dict[str, str] = {
            "impulsive_aggression": "Poorly controlled aggressive responding",
            "poor_emotion_regulation": "Weak regulation of anger, frustration, and threat responses",
            "reward_seeking": "Immediate-gratification bias and sensation/risk seeking",
            "empathy_deficit": "Blunted empathic and prosocial processing",
            "callous_remorselessness": "Lack of remorse and exploitative interpersonal stance",
            "persistent_rule_violation": "Stable antisocial / exploitative / norm-violating behavior",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_deficit",
                "relation": "raises vulnerability for poor inhibitory control",
                "aspd_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "dopaminergic_reward_bias",
                "relation": "raises vulnerability for immediate reward seeking",
                "aspd_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "prefrontal_limbic_maturation_disruption",
                "relation": "raises vulnerability of developmental circuit formation",
                "aspd_change": "increased susceptibility",
            },
            {
                "source": "childhood_adversity",
                "target": "hpa_axis_dysregulation",
                "relation": "programs stress-response dysregulation",
                "aspd_change": "increased",
            },
            {
                "source": "childhood_adversity",
                "target": "low_grade_inflammation",
                "relation": "raises inflammatory burden linked to aggressive development",
                "aspd_change": "increased",
            },
            {
                "source": "childhood_adversity",
                "target": "prefrontal_limbic_maturation_disruption",
                "relation": "alters developmental pathways of control-emotion circuitry",
                "aspd_change": "increased",
            },
            {
                "source": "developmental_insult",
                "target": "prefrontal_limbic_maturation_disruption",
                "relation": "directly burdens neural maturation and integrity",
                "aspd_change": "increased",
            },
            {
                "source": "adolescent_hormonal_sensitivity",
                "target": "dopaminergic_reward_bias",
                "relation": "can amplify risk-taking and reward sensitivity",
                "aspd_change": "increased",
            },
            {
                "source": "adolescent_hormonal_sensitivity",
                "target": "impulsive_aggression",
                "relation": "can amplify aggressive and risk-taking tendencies",
                "aspd_change": "increased",
            },
            {
                "source": "alcohol_disinhibition",
                "target": "impulsive_aggression",
                "relation": "disinhibits aggression and impairs judgment",
                "aspd_change": "increased",
            },
            {
                "source": "alcohol_disinhibition",
                "target": "persistent_rule_violation",
                "relation": "weakens behavioral restraint and judgment",
                "aspd_change": "increased",
            },
            {
                "source": "protective_environment",
                "target": "hpa_axis_dysregulation",
                "relation": "buffers stress-related dysregulation",
                "aspd_change": "protective",
            },
            {
                "source": "protective_environment",
                "target": "prefrontal_limbic_maturation_disruption",
                "relation": "supports healthier circuit development",
                "aspd_change": "protective",
            },
            {
                "source": "protective_environment",
                "target": "persistent_rule_violation",
                "relation": "buffers expression of genetic and developmental risk",
                "aspd_change": "protective",
            },
            {
                "source": "low_grade_inflammation",
                "target": "serotonergic_deficit",
                "relation": "can impair neurotransmitter function and inhibitory control",
                "aspd_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "amygdala",
                "relation": "loads threat-reactive and affective processing",
                "aspd_change": "increased dysregulation",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "vmpfc",
                "relation": "weakens top-down emotional regulation under stress",
                "aspd_change": "reduced function",
            },
            {
                "source": "prefrontal_limbic_maturation_disruption",
                "target": "dlpfc",
                "relation": "impairs executive control and response inhibition",
                "aspd_change": "reduced function",
            },
            {
                "source": "prefrontal_limbic_maturation_disruption",
                "target": "vmpfc",
                "relation": "impairs social-emotional valuation and regulation",
                "aspd_change": "reduced function",
            },
            {
                "source": "prefrontal_limbic_maturation_disruption",
                "target": "ofc",
                "relation": "impairs outcome valuation and punishment learning",
                "aspd_change": "reduced function",
            },
            {
                "source": "prefrontal_limbic_maturation_disruption",
                "target": "acc",
                "relation": "impairs monitoring and conflict regulation",
                "aspd_change": "reduced function",
            },
            {
                "source": "serotonergic_deficit",
                "target": "impulsive_aggression",
                "relation": "lowers the threshold for aggressive responses",
                "aspd_change": "increased",
            },
            {
                "source": "serotonergic_deficit",
                "target": "poor_emotion_regulation",
                "relation": "reduces inhibitory affect regulation",
                "aspd_change": "increased",
            },
            {
                "source": "dopaminergic_reward_bias",
                "target": "reward_seeking",
                "relation": "biases behavior toward immediate reward",
                "aspd_change": "increased",
            },
            {
                "source": "dopaminergic_reward_bias",
                "target": "punishment_insensitivity",
                "relation": "shifts behavior away from future-cost weighting",
                "aspd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "poor_emotion_regulation",
                "relation": "amplifies reactive emotional responding",
                "aspd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "empathy_moral_circuit_failure",
                "relation": "abnormal limbic processing weakens social-emotional resonance",
                "aspd_change": "increased",
            },
            {
                "source": "dlpfc",
                "target": "persistent_rule_violation",
                "relation": "weakened executive control reduces behavioral restraint",
                "aspd_change": "increased",
            },
            {
                "source": "acc",
                "target": "poor_emotion_regulation",
                "relation": "weakened monitoring impairs behavioral adjustment",
                "aspd_change": "increased",
            },
            {
                "source": "vmpfc",
                "target": "empathy_moral_circuit_failure",
                "relation": "reduced social-emotional integration impairs empathy and remorse",
                "aspd_change": "increased",
            },
            {
                "source": "ofc",
                "target": "punishment_insensitivity",
                "relation": "reduced valuation of negative consequences impairs learning from punishment",
                "aspd_change": "increased",
            },
            {
                "source": "empathy_moral_circuit_failure",
                "target": "empathy_deficit",
                "relation": "drives blunted empathic responding",
                "aspd_change": "increased",
            },
            {
                "source": "empathy_moral_circuit_failure",
                "target": "callous_remorselessness",
                "relation": "drives remorseless and exploitative behavior",
                "aspd_change": "increased",
            },
            {
                "source": "punishment_insensitivity",
                "target": "persistent_rule_violation",
                "relation": "reduces behavioral correction after negative outcomes",
                "aspd_change": "increased",
            },
            {
                "source": "reward_seeking",
                "target": "persistent_rule_violation",
                "relation": "promotes risky, exploitative, and immediate-gratification behavior",
                "aspd_change": "increased",
            },
            {
                "source": "impulsive_aggression",
                "target": "persistent_rule_violation",
                "relation": "promotes unstable and harmful behavior patterns",
                "aspd_change": "increased",
            },
            {
                "source": "poor_emotion_regulation",
                "target": "persistent_rule_violation",
                "relation": "reduces capacity to inhibit destructive responses",
                "aspd_change": "increased",
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

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation_spec)
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
        Useful for refining dLPFC / vmPFC / OFC candidates.
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
            df = (
                series.sort_values(ascending=False)
                .reset_index()
                .rename(columns={"index": "connected_region", 0: "value"})
            )
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(
        self,
        node_keys: Sequence[str] = ("amygdala", "acc", "dlpfc", "vmpfc", "ofc"),
    ) -> pd.DataFrame:
        """
        Extract a structural-connectivity submatrix for the ASPD circuit.
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
                    "description": "Atlas-backed ASPD circuit node",
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
        childhood_adversity: float,
        developmental_insult: float,
        adolescent_hormonal_sensitivity: float,
        alcohol_disinhibition: float,
        protective_environment: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        except protective_environment which is protective.
        """
        g = self._clip01(genetic_vulnerability)
        c = self._clip01(childhood_adversity)
        d = self._clip01(developmental_insult)
        h = self._clip01(adolescent_hormonal_sensitivity)
        a = self._clip01(alcohol_disinhibition)
        p = self._clip01(protective_environment)

        # Latent developmental biology
        hpa_axis_dysregulation = self._clip01(0.35 * c + 0.15 * d + 0.10 * g - 0.20 * p)
        low_grade_inflammation = self._clip01(0.25 * c + 0.15 * hpa_axis_dysregulation)
        serotonergic_deficit = self._clip01(
            0.30 * g + 0.20 * low_grade_inflammation + 0.10 * c + 0.05 * a
        )
        dopaminergic_reward_bias = self._clip01(
            0.30 * g + 0.20 * h + 0.10 * c + 0.10 * a
        )
        prefrontal_limbic_maturation_disruption = self._clip01(
            0.30 * c + 0.25 * d + 0.15 * g + 0.10 * hpa_axis_dysregulation - 0.20 * p
        )
        punishment_insensitivity = self._clip01(
            0.35 * dopaminergic_reward_bias
            + 0.25 * prefrontal_limbic_maturation_disruption
            + 0.10 * a
        )
        empathy_moral_circuit_failure = self._clip01(
            0.30 * prefrontal_limbic_maturation_disruption
            + 0.20 * serotonergic_deficit
            + 0.10 * hpa_axis_dysregulation
        )

        # Regional state proxies
        amygdala = self._clip01(
            0.35 * hpa_axis_dysregulation
            + 0.20 * prefrontal_limbic_maturation_disruption
            + 0.10 * c
        )
        acc = self._clip01(
            0.30 * prefrontal_limbic_maturation_disruption
            + 0.15 * serotonergic_deficit
            - 0.10 * p
        )
        dlpfc = self._clip01(
            0.35 * prefrontal_limbic_maturation_disruption
            + 0.15 * dopaminergic_reward_bias
            + 0.10 * a
            - 0.15 * p
        )
        vmpfc = self._clip01(
            0.35 * prefrontal_limbic_maturation_disruption
            + 0.15 * hpa_axis_dysregulation
            + 0.10 * amygdala
            - 0.15 * p
        )
        ofc = self._clip01(
            0.30 * prefrontal_limbic_maturation_disruption
            + 0.20 * dopaminergic_reward_bias
            + 0.10 * a
            - 0.10 * p
        )

        # Symptoms / behaviors
        poor_emotion_regulation = self._clip01(
            0.30 * serotonergic_deficit
            + 0.25 * amygdala
            + 0.15 * acc
            + 0.10 * hpa_axis_dysregulation
        )
        impulsive_aggression = self._clip01(
            0.35 * serotonergic_deficit
            + 0.20 * poor_emotion_regulation
            + 0.15 * a
            + 0.10 * h
        )
        reward_seeking = self._clip01(
            0.45 * dopaminergic_reward_bias
            + 0.20 * punishment_insensitivity
            + 0.10 * ofc
        )
        empathy_deficit = self._clip01(
            0.35 * empathy_moral_circuit_failure
            + 0.20 * vmpfc
            + 0.15 * amygdala
        )
        callous_remorselessness = self._clip01(
            0.35 * empathy_deficit
            + 0.20 * punishment_insensitivity
            + 0.15 * vmpfc
            + 0.10 * ofc
        )
        persistent_rule_violation = self._clip01(
            0.25 * impulsive_aggression
            + 0.20 * reward_seeking
            + 0.20 * callous_remorselessness
            + 0.15 * dlpfc
            + 0.10 * punishment_insensitivity
            - 0.20 * p
        )

        return {
            "inputs": pd.Series(
                {
                    "genetic_vulnerability": g,
                    "childhood_adversity": c,
                    "developmental_insult": d,
                    "adolescent_hormonal_sensitivity": h,
                    "alcohol_disinhibition": a,
                    "protective_environment": p,
                }
            ),
            "latents": pd.Series(
                {
                    "prefrontal_limbic_maturation_disruption": prefrontal_limbic_maturation_disruption,
                    "dopaminergic_reward_bias": dopaminergic_reward_bias,
                    "serotonergic_deficit": serotonergic_deficit,
                    "punishment_insensitivity": punishment_insensitivity,
                    "empathy_moral_circuit_failure": empathy_moral_circuit_failure,
                    "hpa_axis_dysregulation": hpa_axis_dysregulation,
                    "low_grade_inflammation": low_grade_inflammation,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "amygdala": amygdala,
                    "acc": acc,
                    "dlpfc": dlpfc,
                    "vmpfc": vmpfc,
                    "ofc": ofc,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "persistent_rule_violation": persistent_rule_violation,
                    "impulsive_aggression": impulsive_aggression,
                    "reward_seeking": reward_seeking,
                    "poor_emotion_regulation": poor_emotion_regulation,
                    "empathy_deficit": empathy_deficit,
                    "callous_remorselessness": callous_remorselessness,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "impulsive_aggressive_profile": self._clip01(
                        0.45 * impulsive_aggression
                        + 0.25 * poor_emotion_regulation
                        + 0.15 * a
                    ),
                    "callous_unemotional_profile": self._clip01(
                        0.45 * callous_remorselessness
                        + 0.30 * empathy_deficit
                        + 0.10 * punishment_insensitivity
                    ),
                    "reward_dominant_antisocial_profile": self._clip01(
                        0.45 * reward_seeking
                        + 0.25 * punishment_insensitivity
                        + 0.20 * persistent_rule_violation
                    ),
                    "persistent_aspd_profile": self._clip01(
                        0.40 * persistent_rule_violation
                        + 0.20 * callous_remorselessness
                        + 0.15 * impulsive_aggression
                        + 0.10 * reward_seeking
                        - 0.15 * p
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-24, 20, -10)).head(10)
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
    model = AntisocialPersonalityDisorderModel()

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
        bundle["edges"][["source", "target", "relation", "aspd_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["amygdala", "acc", "dlpfc", "vmpfc", "ofc"]:
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
        childhood_adversity=0.85,
        developmental_insult=0.45,
        adolescent_hormonal_sensitivity=0.60,
        alcohol_disinhibition=0.50,
        protective_environment=0.15,
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
    # print(model.assign_mni_point((-24, 20, -10)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("cingulate").to_string(index=False))
    # print(model.suggest_regions("orbitofrontal").to_string(index=False))
    # print(model.suggest_regions("frontal").to_string(index=False))
