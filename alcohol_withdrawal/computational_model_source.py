from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Alcohol Withdrawal Syndrome-oriented panel:
# - GABA-A downregulation
# - NMDA / glutamatergic upregulation
# - receptor plasticity / kindling
# - stress-system recruitment
DEFAULT_GENE_PANEL = [
    "GABRA2",
    "GABRB2",
    "GABRG1",
    "GAD1",
    "GRIN1",
    "GRIN2A",
    "GRIN2B",
    "SLC1A2",
    "SLC1A3",
    "CACNA1C",
    "KCNJ6",
    "CRH",
    "CRHR1",
    "BDNF",
    "COMT",
]


class AlcoholWithdrawalModel:
    """
    Atlas-grounded mechanistic scaffold for Alcohol Withdrawal Syndrome (AWS) using siibra.

    What it does:
      1) Resolves AWS-relevant anatomy to atlas regions when possible.
      2) Pulls receptor, gene-expression, and connectivity evidence from siibra.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates hyperexcitability, kindling, seizure risk, and complicated withdrawal.

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

        # Concrete atlas-backed proxies for the chapter's whole-brain AWS network.
        # Brainstem/autonomic storm is modeled below as a latent system node because
        # the chapter describes it functionally rather than as one specific parcel.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "Ce (central nucleus) left",
                "CM (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "DG (Hippocampus) left",
                "hippocampus",
            ],
            "acc": [
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "Area 33 (ACC) left",
                "ACC",
            ],
            "pfc": [
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "prefrontal",
            ],
            "cerebellum_proxy": [
                "Fastigial Nucleus (Cerebellum) left",
                "Ventral Dentate Nucleus (Cerebellum) left",
                "Interposed Nucleus (Cerebellum) left",
                "cerebellum",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "chronic_alcohol_exposure": "Duration and heaviness of alcohol exposure before cessation",
            "abrupt_cessation": "Sudden removal or major reduction of alcohol intake",
            "withdrawal_history": "Prior withdrawal episodes that may sensitize the system",
            "genetic_liability": "Heritable susceptibility to dependence and severe withdrawal",
            "medical_stabilization": "Protective medical management and physiologic stabilization",
        }

        self.latent_nodes: Dict[str, str] = {
            "gaba_a_downregulation": "Compensatory loss of inhibitory GABA-A receptor function",
            "nmda_upregulation": "Compensatory increase of excitatory glutamatergic drive",
            "receptor_plasticity": "Widespread receptor adaptation from chronic alcohol exposure",
            "glutamate_surge": "Excess excitation after alcohol removal",
            "stress_activation": "Chronic stress-system recruitment during withdrawal",
            "kindling": "Progressive sensitization across repeated withdrawal episodes",
            "global_hyperexcitability": "Whole-brain excitatory dominance",
            "excitotoxicity": "Glutamate-driven neuronal injury burden",
            "brainstem_autonomic_disinhibition": "Loss of autonomic restraint / hyperadrenergic state",
            "sleep_architecture_disruption": "Profound withdrawal-related sleep disturbance",
        }

        self.symptom_nodes: Dict[str, str] = {
            "anxiety_agitation": "Anxiety, irritability, and psychomotor agitation",
            "tremor": "Tremulousness and motor instability",
            "autonomic_storm": "Hyperadrenergic state with severe autonomic activation",
            "emotional_lability": "Rapid affective instability during withdrawal",
            "cognitive_impairment": "Clouding, executive burden, and acute cognitive dysfunction",
            "sleep_disturbance": "Major disruption of normal sleep architecture",
            "seizures": "Generalized tonic-clonic seizure risk under severe hyperexcitability",
        }

        self.complication_nodes: Dict[str, str] = {
            "delirium_tremens_risk": "Risk of severe complicated withdrawal / delirium tremens",
            "grey_matter_loss_risk": "Risk of structural brain injury with severe withdrawal burden",
            "post_seizure_hypoperfusion_risk": "Risk of transient hypoperfusion after withdrawal seizures",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "chronic_alcohol_exposure",
                "target": "gaba_a_downregulation",
                "relation": "drives compensatory reduction in inhibitory tone",
                "aws_change": "increased",
            },
            {
                "source": "chronic_alcohol_exposure",
                "target": "nmda_upregulation",
                "relation": "drives compensatory excitatory receptor adaptation",
                "aws_change": "increased",
            },
            {
                "source": "chronic_alcohol_exposure",
                "target": "receptor_plasticity",
                "relation": "induces widespread neuroadaptation",
                "aws_change": "increased",
            },
            {
                "source": "chronic_alcohol_exposure",
                "target": "stress_activation",
                "relation": "recruits withdrawal-related stress systems",
                "aws_change": "increased",
            },
            {
                "source": "abrupt_cessation",
                "target": "glutamate_surge",
                "relation": "unmasks the upregulated excitatory system",
                "aws_change": "increased",
            },
            {
                "source": "abrupt_cessation",
                "target": "global_hyperexcitability",
                "relation": "abruptly removes alcohol's inhibitory buffering",
                "aws_change": "increased",
            },
            {
                "source": "withdrawal_history",
                "target": "kindling",
                "relation": "sensitizes the system across episodes",
                "aws_change": "increased",
            },
            {
                "source": "genetic_liability",
                "target": "kindling",
                "relation": "raises susceptibility to severe withdrawal progression",
                "aws_change": "increased susceptibility",
            },
            {
                "source": "genetic_liability",
                "target": "receptor_plasticity",
                "relation": "modulates dependence and withdrawal liability",
                "aws_change": "increased susceptibility",
            },
            {
                "source": "medical_stabilization",
                "target": "global_hyperexcitability",
                "relation": "buffers severe excitatory rebound",
                "aws_change": "protective",
            },
            {
                "source": "medical_stabilization",
                "target": "seizures",
                "relation": "reduces seizure expression",
                "aws_change": "protective",
            },
            {
                "source": "medical_stabilization",
                "target": "autonomic_storm",
                "relation": "reduces physiologic escalation",
                "aws_change": "protective",
            },
            {
                "source": "gaba_a_downregulation",
                "target": "global_hyperexcitability",
                "relation": "reduces inhibitory restraint",
                "aws_change": "increased",
            },
            {
                "source": "nmda_upregulation",
                "target": "glutamate_surge",
                "relation": "amplifies post-cessation excitation",
                "aws_change": "increased",
            },
            {
                "source": "receptor_plasticity",
                "target": "global_hyperexcitability",
                "relation": "widens instability across neurotransmitter systems",
                "aws_change": "increased",
            },
            {
                "source": "glutamate_surge",
                "target": "global_hyperexcitability",
                "relation": "creates excitatory dominance",
                "aws_change": "increased",
            },
            {
                "source": "glutamate_surge",
                "target": "excitotoxicity",
                "relation": "increases risk of neuronal injury",
                "aws_change": "increased",
            },
            {
                "source": "kindling",
                "target": "global_hyperexcitability",
                "relation": "progressively lowers tolerance for withdrawal stress",
                "aws_change": "increased",
            },
            {
                "source": "kindling",
                "target": "seizures",
                "relation": "raises seizure propensity",
                "aws_change": "increased",
            },
            {
                "source": "stress_activation",
                "target": "amygdala",
                "relation": "amplifies limbic threat and distress processing",
                "aws_change": "increased",
            },
            {
                "source": "stress_activation",
                "target": "brainstem_autonomic_disinhibition",
                "relation": "amplifies autonomic activation",
                "aws_change": "increased",
            },
            {
                "source": "global_hyperexcitability",
                "target": "pfc",
                "relation": "disrupts executive cortical function",
                "aws_change": "increased dysfunction",
            },
            {
                "source": "global_hyperexcitability",
                "target": "acc",
                "relation": "disrupts monitoring and regulation",
                "aws_change": "increased dysfunction",
            },
            {
                "source": "global_hyperexcitability",
                "target": "hippocampus",
                "relation": "burdens memory-related circuitry",
                "aws_change": "increased dysfunction",
            },
            {
                "source": "global_hyperexcitability",
                "target": "cerebellum_proxy",
                "relation": "burdens motor coordination circuitry",
                "aws_change": "increased dysfunction",
            },
            {
                "source": "global_hyperexcitability",
                "target": "brainstem_autonomic_disinhibition",
                "relation": "destabilizes autonomic control",
                "aws_change": "increased",
            },
            {
                "source": "global_hyperexcitability",
                "target": "sleep_architecture_disruption",
                "relation": "disrupts normal sleep patterning",
                "aws_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "anxiety_agitation",
                "relation": "amplifies fear and distress",
                "aws_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "emotional_lability",
                "relation": "destabilizes affective regulation",
                "aws_change": "increased",
            },
            {
                "source": "pfc",
                "target": "cognitive_impairment",
                "relation": "reduces executive control",
                "aws_change": "increased",
            },
            {
                "source": "acc",
                "target": "emotional_lability",
                "relation": "reduces regulatory monitoring",
                "aws_change": "increased",
            },
            {
                "source": "acc",
                "target": "cognitive_impairment",
                "relation": "burdens conflict monitoring and control",
                "aws_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "cognitive_impairment",
                "relation": "burdens memory-related function",
                "aws_change": "increased",
            },
            {
                "source": "cerebellum_proxy",
                "target": "tremor",
                "relation": "supports motor instability expression",
                "aws_change": "increased",
            },
            {
                "source": "brainstem_autonomic_disinhibition",
                "target": "autonomic_storm",
                "relation": "drives severe autonomic activation",
                "aws_change": "increased",
            },
            {
                "source": "sleep_architecture_disruption",
                "target": "sleep_disturbance",
                "relation": "drives severe sleep disruption",
                "aws_change": "increased",
            },
            {
                "source": "global_hyperexcitability",
                "target": "seizures",
                "relation": "raises generalized seizure risk",
                "aws_change": "increased",
            },
            {
                "source": "global_hyperexcitability",
                "target": "tremor",
                "relation": "raises motor excitability",
                "aws_change": "increased",
            },
            {
                "source": "global_hyperexcitability",
                "target": "anxiety_agitation",
                "relation": "raises subjective and behavioral arousal",
                "aws_change": "increased",
            },
            {
                "source": "excitotoxicity",
                "target": "grey_matter_loss_risk",
                "relation": "raises structural injury burden",
                "aws_change": "increased",
            },
            {
                "source": "excitotoxicity",
                "target": "cognitive_impairment",
                "relation": "burdens cortical and limbic function",
                "aws_change": "increased",
            },
            {
                "source": "seizures",
                "target": "post_seizure_hypoperfusion_risk",
                "relation": "raises transient perfusion disturbance risk",
                "aws_change": "increased",
            },
            {
                "source": "seizures",
                "target": "delirium_tremens_risk",
                "relation": "marks severe complicated withdrawal",
                "aws_change": "increased",
            },
            {
                "source": "autonomic_storm",
                "target": "delirium_tremens_risk",
                "relation": "marks severe physiologic escalation",
                "aws_change": "increased",
            },
            {
                "source": "cognitive_impairment",
                "target": "delirium_tremens_risk",
                "relation": "marks severe cerebral dysfunction",
                "aws_change": "increased",
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
                continue
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        """
        Search the current atlas for candidate regions matching a keyword.
        Helpful for swapping proxies, e.g. for PFC or cerebellar regions.
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
        node_keys: Sequence[str] = (
            "amygdala",
            "hippocampus",
            "acc",
            "pfc",
            "cerebellum_proxy",
        ),
    ) -> pd.DataFrame:
        """
        Extract a structural-connectivity submatrix for the AWS circuit.
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
                    "description": "Atlas-backed AWS circuit node",
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

        for key, desc in self.complication_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "complication",
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
        chronic_alcohol_exposure: float,
        abrupt_cessation: float,
        withdrawal_history: float,
        genetic_liability: float,
        medical_stabilization: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean more dysregulation / worse symptom burden,
        except medical_stabilization which is protective.
        """
        c = self._clip01(chronic_alcohol_exposure)
        a = self._clip01(abrupt_cessation)
        w = self._clip01(withdrawal_history)
        g = self._clip01(genetic_liability)
        m = self._clip01(medical_stabilization)

        # latent biology
        gaba_a_downregulation = self._clip01(0.40 * c + 0.15 * g)
        nmda_upregulation = self._clip01(0.35 * c + 0.15 * w + 0.10 * g)
        receptor_plasticity = self._clip01(0.30 * c + 0.20 * w + 0.10 * g)
        glutamate_surge = self._clip01(
            0.40 * a + 0.30 * nmda_upregulation + 0.10 * receptor_plasticity
        )
        stress_activation = self._clip01(0.25 * c + 0.25 * a + 0.20 * w)
        kindling = self._clip01(0.35 * w + 0.20 * nmda_upregulation + 0.10 * g)
        global_hyperexcitability = self._clip01(
            0.25 * glutamate_surge
            + 0.25 * gaba_a_downregulation
            + 0.20 * kindling
            + 0.10 * stress_activation
            + 0.10 * receptor_plasticity
            - 0.25 * m
        )
        excitotoxicity = self._clip01(
            0.30 * glutamate_surge + 0.25 * global_hyperexcitability + 0.10 * kindling
        )
        brainstem_autonomic_disinhibition = self._clip01(
            0.30 * global_hyperexcitability
            + 0.25 * stress_activation
            + 0.10 * a
            - 0.15 * m
        )
        sleep_architecture_disruption = self._clip01(
            0.25 * global_hyperexcitability
            + 0.20 * stress_activation
            + 0.10 * kindling
        )

        # region dysfunction proxies
        amygdala = self._clip01(
            0.35 * stress_activation + 0.20 * global_hyperexcitability
        )
        hippocampus = self._clip01(
            0.25 * excitotoxicity + 0.15 * global_hyperexcitability
        )
        pfc = self._clip01(
            0.25 * global_hyperexcitability
            + 0.15 * excitotoxicity
            + 0.10 * sleep_architecture_disruption
            - 0.10 * m
        )
        acc = self._clip01(
            0.20 * pfc + 0.20 * amygdala + 0.10 * global_hyperexcitability
        )
        cerebellum_proxy = self._clip01(
            0.25 * global_hyperexcitability + 0.20 * excitotoxicity
        )

        # symptoms
        anxiety_agitation = self._clip01(
            0.30 * amygdala
            + 0.25 * global_hyperexcitability
            + 0.20 * stress_activation
            + 0.10 * brainstem_autonomic_disinhibition
            - 0.10 * m
        )
        tremor = self._clip01(
            0.35 * global_hyperexcitability
            + 0.25 * cerebellum_proxy
            + 0.20 * brainstem_autonomic_disinhibition
            - 0.10 * m
        )
        autonomic_storm = self._clip01(
            0.45 * brainstem_autonomic_disinhibition
            + 0.20 * stress_activation
            + 0.15 * global_hyperexcitability
            - 0.15 * m
        )
        emotional_lability = self._clip01(
            0.30 * amygdala + 0.20 * acc + 0.15 * sleep_architecture_disruption
        )
        cognitive_impairment = self._clip01(
            0.30 * pfc
            + 0.20 * hippocampus
            + 0.15 * sleep_architecture_disruption
            + 0.10 * excitotoxicity
        )
        sleep_disturbance = self._clip01(
            0.45 * sleep_architecture_disruption
            + 0.20 * stress_activation
            + 0.10 * autonomic_storm
        )
        seizures = self._clip01(
            0.40 * global_hyperexcitability
            + 0.25 * kindling
            + 0.20 * glutamate_surge
            - 0.20 * m
        )

        # complications
        delirium_tremens_risk = self._clip01(
            0.25 * autonomic_storm
            + 0.20 * seizures
            + 0.20 * cognitive_impairment
            + 0.10 * sleep_disturbance
            - 0.10 * m
        )
        grey_matter_loss_risk = self._clip01(
            0.45 * excitotoxicity + 0.20 * seizures + 0.10 * c
        )
        post_seizure_hypoperfusion_risk = self._clip01(
            0.50 * seizures + 0.20 * global_hyperexcitability
        )

        return {
            "inputs": pd.Series(
                {
                    "chronic_alcohol_exposure": c,
                    "abrupt_cessation": a,
                    "withdrawal_history": w,
                    "genetic_liability": g,
                    "medical_stabilization": m,
                }
            ),
            "latents": pd.Series(
                {
                    "global_hyperexcitability": global_hyperexcitability,
                    "gaba_a_downregulation": gaba_a_downregulation,
                    "nmda_upregulation": nmda_upregulation,
                    "receptor_plasticity": receptor_plasticity,
                    "glutamate_surge": glutamate_surge,
                    "stress_activation": stress_activation,
                    "kindling": kindling,
                    "excitotoxicity": excitotoxicity,
                    "brainstem_autonomic_disinhibition": brainstem_autonomic_disinhibition,
                    "sleep_architecture_disruption": sleep_architecture_disruption,
                }
            ).sort_values(ascending=False),
            "regions": pd.Series(
                {
                    "amygdala": amygdala,
                    "hippocampus": hippocampus,
                    "pfc": pfc,
                    "acc": acc,
                    "cerebellum_proxy": cerebellum_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "anxiety_agitation": anxiety_agitation,
                    "tremor": tremor,
                    "autonomic_storm": autonomic_storm,
                    "emotional_lability": emotional_lability,
                    "cognitive_impairment": cognitive_impairment,
                    "sleep_disturbance": sleep_disturbance,
                    "seizures": seizures,
                }
            ).sort_values(ascending=False),
            "complications": pd.Series(
                {
                    "delirium_tremens_risk": delirium_tremens_risk,
                    "grey_matter_loss_risk": grey_matter_loss_risk,
                    "post_seizure_hypoperfusion_risk": post_seizure_hypoperfusion_risk,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "hyperexcitable_withdrawal": self._clip01(
                        0.40 * global_hyperexcitability
                        + 0.25 * anxiety_agitation
                        + 0.20 * tremor
                    ),
                    "seizure_prone_withdrawal": self._clip01(
                        0.45 * seizures + 0.30 * kindling + 0.15 * glutamate_surge
                    ),
                    "autonomic_dominant_withdrawal": self._clip01(
                        0.50 * autonomic_storm + 0.25 * anxiety_agitation + 0.15 * tremor
                    ),
                    "delirium_tremens_profile": self._clip01(
                        0.45 * delirium_tremens_risk
                        + 0.25 * autonomic_storm
                        + 0.20 * cognitive_impairment
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-8, -18, 10)).head(10)
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
        Use .fetch() on the returned object to obtain the NIfTI image.
        """
        region = self.region_objects[node_key]
        return region.get_regional_mask(self.space, maptype="labelled")


if __name__ == "__main__":
    model = AlcoholWithdrawalModel()

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
        bundle["edges"][["source", "target", "relation", "aws_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["amygdala", "hippocampus", "pfc", "acc", "cerebellum_proxy"]:
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
        chronic_alcohol_exposure=0.90,
        abrupt_cessation=0.95,
        withdrawal_history=0.70,
        genetic_liability=0.50,
        medical_stabilization=0.20,
    )

    print("\n=== LATENT BIOLOGY ===")
    print(sim["latents"].to_string())

    print("\n=== REGIONS ===")
    print(sim["regions"].to_string())

    print("\n=== SYMPTOMS ===")
    print(sim["symptoms"].to_string())

    print("\n=== COMPLICATIONS ===")
    print(sim["complications"].to_string())

    print("\n=== PHENOTYPES ===")
    print(sim["phenotypes"].to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((-8, -18, 10)).head(10))

    # Example region search helpers:
    # print(model.suggest_regions("cerebellum").to_string(index=False))
    # print(model.suggest_regions("cingulate").to_string(index=False))
