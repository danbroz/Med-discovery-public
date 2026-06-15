from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Cannabis Withdrawal-oriented panel:
# - endocannabinoid rebound
# - dopamine / anhedonia / craving
# - stress / CRF / glucocorticoid signaling
# - GABA / glutamate balance
# - neuroplasticity and cognitive control
DEFAULT_GENE_PANEL = [
    "CNR1",    # CB1 receptor
    "FAAH",    # endocannabinoid degradation
    "MGLL",    # 2-AG metabolism
    "DRD2",    # dopamine receptor
    "SLC6A3",  # dopamine transporter
    "COMT",    # dopamine metabolism
    "SLC6A4",  # serotonin transporter
    "SLC6A2",  # norepinephrine transporter
    "DBH",     # norepinephrine synthesis step
    "GABRA2",  # GABA-A receptor
    "GABRB2",  # GABA-A receptor
    "GAD1",    # GABA synthesis
    "GRIN2B",  # NMDA receptor subunit
    "SLC1A1",  # glutamate transport
    "NR3C1",   # glucocorticoid receptor
    "FKBP5",   # stress responsivity
    "CRHR1",   # CRF signaling
    "BDNF",    # neuroplasticity
]


class CannabisWithdrawalModel:
    """
    Atlas-grounded mechanistic scaffold for Cannabis Withdrawal Syndrome (CWS).

    What it does:
      1) Resolves withdrawal-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates withdrawal negative affect, sleep disruption, anhedonia,
         cognitive burden, and relapse pressure.

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

        # Chapter-consistent withdrawal circuit:
        # hippocampus, amygdala, PFC, ventral striatum.
        self.region_candidates: Dict[str, List[str]] = {
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "DG (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "hippocampus",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "pfc_control": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "prefrontal",
                "frontal",
            ],
            "ventral_striatum_proxy": [
                "nucleus accumbens",
                "accumbens",
                "ventral striatum",
                "striatum",
                "basal ganglia",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "chronic_thc_exposure": "Chronic heavy THC exposure producing dependence",
            "cessation_intensity": "Abruptness and intensity of cessation",
            "genetic_sud_vulnerability": "Heritable vulnerability influencing withdrawal severity",
            "withdrawal_stress": "Stress burden during abstinence and withdrawal",
            "recovery_support": "Protective recovery environment and homeostatic support",
        }

        self.latent_nodes: Dict[str, str] = {
            "ecs_rebound_deficit": "Endocannabinoid-system rebound dysregulation after chronic THC cessation",
            "hpa_crf_hyperactivation": "Withdrawal-linked stress-system hyperactivation",
            "neuroplastic_scarring": "Lasting synaptic and gene-expression changes from dependence and withdrawal",
            "hippocampal_structural_vulnerability": "Trait or medication-induced hippocampal vulnerability",
            "amygdala_structural_vulnerability": "Trait or medication-induced amygdala vulnerability",
            "ventral_striatal_hypodopaminergia": "Reduced ventral-striatal dopamine tone during withdrawal",
            "prefrontal_hypoactivity": "Reduced top-down executive and regulatory function",
            "limbic_hyperreactivity": "Bottom-up emotional-system overactivity",
            "cue_craving_relapse_circuit": "Strengthened relapse-promoting seeking circuitry",
            "homeostatic_restoration": "Re-equilibration after acute withdrawal",
        }

        self.symptom_nodes: Dict[str, str] = {
            "withdrawal_negative_affect": "Negative emotional state during withdrawal",
            "anxiety_irritability": "Stress, anxiety, and irritability",
            "sleep_disturbance": "Sleep and arousal dysregulation",
            "anhedonia_amotivation": "Anhedonia, amotivation, and low drive",
            "concentration_memory_impairment": "Poor concentration, indecisiveness, and memory burden",
            "impulse_control_difficulty": "Impaired control and decision-making",
            "craving_relapse_pressure": "Craving and relapse pressure",
            "relapse_vulnerability": "Enduring vulnerability to return to use",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "chronic_thc_exposure",
                "target": "ecs_rebound_deficit",
                "relation": "chronic THC exposure creates rebound dysregulation after cessation",
                "cws_change": "increased",
            },
            {
                "source": "chronic_thc_exposure",
                "target": "neuroplastic_scarring",
                "relation": "chronic exposure induces long-term plastic changes",
                "cws_change": "increased",
            },
            {
                "source": "chronic_thc_exposure",
                "target": "hippocampal_structural_vulnerability",
                "relation": "heavy long-term use can burden hippocampal resilience",
                "cws_change": "increased",
            },
            {
                "source": "chronic_thc_exposure",
                "target": "amygdala_structural_vulnerability",
                "relation": "heavy long-term use can burden amygdala resilience",
                "cws_change": "increased",
            },
            {
                "source": "cessation_intensity",
                "target": "ecs_rebound_deficit",
                "relation": "more abrupt cessation intensifies withdrawal-state dysregulation",
                "cws_change": "increased",
            },
            {
                "source": "cessation_intensity",
                "target": "hpa_crf_hyperactivation",
                "relation": "abrupt cessation intensifies stress-system rebound",
                "cws_change": "increased",
            },
            {
                "source": "genetic_sud_vulnerability",
                "target": "ecs_rebound_deficit",
                "relation": "genetic vulnerability may modulate withdrawal severity",
                "cws_change": "increased susceptibility",
            },
            {
                "source": "genetic_sud_vulnerability",
                "target": "ventral_striatal_hypodopaminergia",
                "relation": "genetic vulnerability may worsen reward-system downshift",
                "cws_change": "increased susceptibility",
            },
            {
                "source": "withdrawal_stress",
                "target": "hpa_crf_hyperactivation",
                "relation": "stress amplifies CRF/glucocorticoid burden during withdrawal",
                "cws_change": "increased",
            },
            {
                "source": "withdrawal_stress",
                "target": "neuroplastic_scarring",
                "relation": "repeated stress can deepen long-term withdrawal-related plasticity",
                "cws_change": "increased",
            },
            {
                "source": "recovery_support",
                "target": "homeostatic_restoration",
                "relation": "supports restoration of equilibrium after acute withdrawal",
                "cws_change": "protective",
            },
            {
                "source": "recovery_support",
                "target": "withdrawal_negative_affect",
                "relation": "buffers negative affect during abstinence",
                "cws_change": "protective",
            },
            {
                "source": "ecs_rebound_deficit",
                "target": "hpa_crf_hyperactivation",
                "relation": "withdrawal-state ECS deficit destabilizes stress regulation",
                "cws_change": "increased",
            },
            {
                "source": "ecs_rebound_deficit",
                "target": "ventral_striatal_hypodopaminergia",
                "relation": "withdrawal reduces reward-related dopaminergic tone",
                "cws_change": "increased",
            },
            {
                "source": "ecs_rebound_deficit",
                "target": "sleep_disturbance",
                "relation": "withdrawal-state dysregulation perturbs sleep and arousal",
                "cws_change": "increased",
            },
            {
                "source": "hpa_crf_hyperactivation",
                "target": "limbic_hyperreactivity",
                "relation": "stress-system rebound amplifies limbic overactivity",
                "cws_change": "increased",
            },
            {
                "source": "hpa_crf_hyperactivation",
                "target": "prefrontal_hypoactivity",
                "relation": "stress-system rebound weakens top-down control",
                "cws_change": "increased",
            },
            {
                "source": "hpa_crf_hyperactivation",
                "target": "withdrawal_negative_affect",
                "relation": "supports dysphoria and aversive emotional state",
                "cws_change": "increased",
            },
            {
                "source": "neuroplastic_scarring",
                "target": "cue_craving_relapse_circuit",
                "relation": "strengthens medication-seeking and cue-triggered relapse pathways",
                "cws_change": "increased",
            },
            {
                "source": "neuroplastic_scarring",
                "target": "prefrontal_hypoactivity",
                "relation": "weakens executive-control systems over time",
                "cws_change": "increased",
            },
            {
                "source": "hippocampal_structural_vulnerability",
                "target": "hippocampus",
                "relation": "burdens hippocampal regulation of memory and stress",
                "cws_change": "increased dysregulation",
            },
            {
                "source": "amygdala_structural_vulnerability",
                "target": "amygdala",
                "relation": "burdens fear/anxiety processing systems",
                "cws_change": "increased dysregulation",
            },
            {
                "source": "ventral_striatal_hypodopaminergia",
                "target": "ventral_striatum_proxy",
                "relation": "loads reward circuitry with hypodopaminergic withdrawal state",
                "cws_change": "reduced reward signaling",
            },
            {
                "source": "ventral_striatal_hypodopaminergia",
                "target": "anhedonia_amotivation",
                "relation": "reduced ventral-striatal signaling produces anhedonia and low drive",
                "cws_change": "increased",
            },
            {
                "source": "ventral_striatal_hypodopaminergia",
                "target": "craving_relapse_pressure",
                "relation": "reward deficit increases relapse pressure",
                "cws_change": "increased",
            },
            {
                "source": "prefrontal_hypoactivity",
                "target": "pfc_control",
                "relation": "reduces top-down cognitive control",
                "cws_change": "reduced function",
            },
            {
                "source": "prefrontal_hypoactivity",
                "target": "concentration_memory_impairment",
                "relation": "contributes to poor concentration and indecisiveness",
                "cws_change": "increased",
            },
            {
                "source": "prefrontal_hypoactivity",
                "target": "impulse_control_difficulty",
                "relation": "weakens top-down inhibitory control",
                "cws_change": "increased",
            },
            {
                "source": "limbic_hyperreactivity",
                "target": "amygdala",
                "relation": "amplifies emotional and anxiety circuitry",
                "cws_change": "increased dysregulation",
            },
            {
                "source": "limbic_hyperreactivity",
                "target": "anxiety_irritability",
                "relation": "supports anxiety, irritability, and stress reactivity",
                "cws_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "concentration_memory_impairment",
                "relation": "hippocampal burden contributes to memory and concentration deficits",
                "cws_change": "increased",
            },
            {
                "source": "cue_craving_relapse_circuit",
                "target": "craving_relapse_pressure",
                "relation": "strengthened seeking circuitry drives relapse risk",
                "cws_change": "increased",
            },
            {
                "source": "withdrawal_negative_affect",
                "target": "craving_relapse_pressure",
                "relation": "negative affect promotes craving and return to use",
                "cws_change": "increased",
            },
            {
                "source": "craving_relapse_pressure",
                "target": "relapse_vulnerability",
                "relation": "craving pressure raises risk of relapse",
                "cws_change": "increased",
            },
            {
                "source": "impulse_control_difficulty",
                "target": "relapse_vulnerability",
                "relation": "poor control raises relapse risk",
                "cws_change": "increased",
            },
            {
                "source": "homeostatic_restoration",
                "target": "withdrawal_negative_affect",
                "relation": "re-equilibration reduces withdrawal distress",
                "cws_change": "protective",
            },
            {
                "source": "homeostatic_restoration",
                "target": "sleep_disturbance",
                "relation": "re-equilibration restores sleep and arousal stability",
                "cws_change": "protective",
            },
            {
                "source": "homeostatic_restoration",
                "target": "relapse_vulnerability",
                "relation": "recovery of homeostasis lowers relapse pressure",
                "cws_change": "protective",
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "hippocampus",
            "amygdala",
            "prefrontal cortex",
            "striatum",
            "ventral striatum",
            "basal ganglia",
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
        Useful for tuning ventral-striatal candidates.
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

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

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

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
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
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
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

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
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
            "hippocampus",
            "amygdala",
            "pfc_control",
            "ventral_striatum_proxy",
        ),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the CWS circuit.
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
                    "description": "Atlas-backed cannabis-withdrawal circuit node",
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
        chronic_thc_exposure: float,
        cessation_intensity: float,
        genetic_sud_vulnerability: float,
        withdrawal_stress: float,
        recovery_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger withdrawal dysregulation / symptom burden,
        while recovery_support is protective.
        """
        c = self._clip01(chronic_thc_exposure)
        x = self._clip01(cessation_intensity)
        g = self._clip01(genetic_sud_vulnerability)
        w = self._clip01(withdrawal_stress)
        r = self._clip01(recovery_support)

        # Latent biology
        ecs_rebound_deficit = self._clip01(
            0.45 * c + 0.35 * x + 0.10 * g - 0.20 * r
        )
        hpa_crf_hyperactivation = self._clip01(
            0.35 * w + 0.25 * ecs_rebound_deficit + 0.10 * x - 0.15 * r
        )
        neuroplastic_scarring = self._clip01(
            0.35 * c + 0.20 * w + 0.10 * g - 0.10 * r
        )
        hippocampal_structural_vulnerability = self._clip01(
            0.35 * c + 0.20 * neuroplastic_scarring
        )
        amygdala_structural_vulnerability = self._clip01(
            0.30 * c + 0.20 * neuroplastic_scarring
        )
        ventral_striatal_hypodopaminergia = self._clip01(
            0.35 * ecs_rebound_deficit + 0.20 * g + 0.15 * x
        )
        prefrontal_hypoactivity = self._clip01(
            0.30 * neuroplastic_scarring
            + 0.20 * hpa_crf_hyperactivation
            + 0.15 * ecs_rebound_deficit
            - 0.20 * r
        )
        limbic_hyperreactivity = self._clip01(
            0.30 * hpa_crf_hyperactivation
            + 0.25 * amygdala_structural_vulnerability
            + 0.15 * hippocampal_structural_vulnerability
            - 0.10 * r
        )
        cue_craving_relapse_circuit = self._clip01(
            0.35 * ventral_striatal_hypodopaminergia
            + 0.20 * prefrontal_hypoactivity
            + 0.15 * neuroplastic_scarring
        )
        homeostatic_restoration = self._clip01(
            0.45 * r - 0.20 * hpa_crf_hyperactivation - 0.10 * ecs_rebound_deficit
        )

        # Regional state proxies
        hippocampus = self._clip01(
            0.45 * hippocampal_structural_vulnerability + 0.20 * hpa_crf_hyperactivation
        )
        amygdala = self._clip01(
            0.45 * limbic_hyperreactivity + 0.10 * hpa_crf_hyperactivation
        )
        pfc_control = self._clip01(
            0.45 * prefrontal_hypoactivity + 0.15 * limbic_hyperreactivity - 0.10 * r
        )
        ventral_striatum_proxy = self._clip01(
            0.45 * ventral_striatal_hypodopaminergia + 0.20 * cue_craving_relapse_circuit
        )

        # Symptoms
        withdrawal_negative_affect = self._clip01(
            0.35 * hpa_crf_hyperactivation
            + 0.25 * limbic_hyperreactivity
            + 0.10 * ventral_striatal_hypodopaminergia
            - 0.20 * homeostatic_restoration
        )
        anxiety_irritability = self._clip01(
            0.35 * limbic_hyperreactivity
            + 0.25 * hpa_crf_hyperactivation
            + 0.15 * amygdala
            - 0.15 * homeostatic_restoration
        )
        sleep_disturbance = self._clip01(
            0.35 * hpa_crf_hyperactivation
            + 0.25 * ecs_rebound_deficit
            + 0.10 * w
            - 0.20 * homeostatic_restoration
        )
        anhedonia_amotivation = self._clip01(
            0.40 * ventral_striatal_hypodopaminergia
            + 0.20 * prefrontal_hypoactivity
            + 0.10 * withdrawal_negative_affect
        )
        concentration_memory_impairment = self._clip01(
            0.35 * hippocampus
            + 0.30 * pfc_control
            + 0.15 * hpa_crf_hyperactivation
            - 0.10 * homeostatic_restoration
        )
        impulse_control_difficulty = self._clip01(
            0.35 * pfc_control
            + 0.20 * cue_craving_relapse_circuit
            + 0.10 * limbic_hyperreactivity
            - 0.10 * homeostatic_restoration
        )
        craving_relapse_pressure = self._clip01(
            0.35 * ventral_striatal_hypodopaminergia
            + 0.30 * cue_craving_relapse_circuit
            + 0.15 * withdrawal_negative_affect
            - 0.10 * homeostatic_restoration
        )
        relapse_vulnerability = self._clip01(
            0.35 * craving_relapse_pressure
            + 0.25 * impulse_control_difficulty
            + 0.20 * anxiety_irritability
            + 0.10 * withdrawal_negative_affect
            - 0.20 * homeostatic_restoration
        )

        return {
            "inputs": pd.Series(
                {
                    "chronic_thc_exposure": c,
                    "cessation_intensity": x,
                    "genetic_sud_vulnerability": g,
                    "withdrawal_stress": w,
                    "recovery_support": r,
                }
            ),
            "latents": pd.Series(
                {
                    "ecs_rebound_deficit": ecs_rebound_deficit,
                    "hpa_crf_hyperactivation": hpa_crf_hyperactivation,
                    "neuroplastic_scarring": neuroplastic_scarring,
                    "hippocampal_structural_vulnerability": hippocampal_structural_vulnerability,
                    "amygdala_structural_vulnerability": amygdala_structural_vulnerability,
                    "ventral_striatal_hypodopaminergia": ventral_striatal_hypodopaminergia,
                    "prefrontal_hypoactivity": prefrontal_hypoactivity,
                    "limbic_hyperreactivity": limbic_hyperreactivity,
                    "cue_craving_relapse_circuit": cue_craving_relapse_circuit,
                    "homeostatic_restoration": homeostatic_restoration,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "hippocampus": hippocampus,
                    "amygdala": amygdala,
                    "pfc_control": pfc_control,
                    "ventral_striatum_proxy": ventral_striatum_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "relapse_vulnerability": relapse_vulnerability,
                    "craving_relapse_pressure": craving_relapse_pressure,
                    "withdrawal_negative_affect": withdrawal_negative_affect,
                    "anxiety_irritability": anxiety_irritability,
                    "sleep_disturbance": sleep_disturbance,
                    "anhedonia_amotivation": anhedonia_amotivation,
                    "concentration_memory_impairment": concentration_memory_impairment,
                    "impulse_control_difficulty": impulse_control_difficulty,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "anxious_withdrawal_profile": self._clip01(
                        0.45 * anxiety_irritability
                        + 0.25 * withdrawal_negative_affect
                        + 0.15 * sleep_disturbance
                    ),
                    "cognitive_withdrawal_profile": self._clip01(
                        0.45 * concentration_memory_impairment
                        + 0.25 * prefrontal_hypoactivity
                        + 0.15 * hippocampal_structural_vulnerability
                    ),
                    "anhedonic_relapse_profile": self._clip01(
                        0.40 * anhedonia_amotivation
                        + 0.30 * craving_relapse_pressure
                        + 0.20 * ventral_striatal_hypodopaminergia
                    ),
                    "post_acute_vulnerability_profile": self._clip01(
                        0.35 * relapse_vulnerability
                        + 0.25 * neuroplastic_scarring
                        + 0.20 * withdrawal_negative_affect
                        - 0.15 * homeostatic_restoration
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-22, -12, -14)).head(10)
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
        return region.get_regional_mask(self.space)


if __name__ == "__main__":
    model = CannabisWithdrawalModel()

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
        bundle["edges"][["source", "target", "relation", "cws_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["hippocampus", "amygdala", "pfc_control", "ventral_striatum_proxy"]:
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
        chronic_thc_exposure=0.85,
        cessation_intensity=0.90,
        genetic_sud_vulnerability=0.60,
        withdrawal_stress=0.75,
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
    # print(model.assign_mni_point((-22, -12, -14)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("accumbens").to_string(index=False))
