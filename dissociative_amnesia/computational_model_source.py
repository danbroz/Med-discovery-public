from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Dissociative Amnesia-oriented panel:
# - stress / glucocorticoid signaling
# - monoaminergic modulation
# - cholinergic memory-state regulation
# - GABA / glutamate balance
# - neuroplasticity and autobiographical memory systems
DEFAULT_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor
    "TPH2",     # serotonin synthesis
    "SLC6A2",   # norepinephrine transporter
    "DBH",      # norepinephrine synthesis step
    "DRD2",     # dopamine receptor
    "SLC6A3",   # dopamine transporter
    "COMT",     # catecholamine metabolism
    "CHRM2",    # muscarinic receptor
    "CHAT",     # acetylcholine synthesis
    "GABRA2",   # GABA-A receptor
    "GABRB2",   # GABA-A receptor
    "GAD1",     # GABA synthesis
    "GRIN2B",   # NMDA receptor subunit
    "SLC1A1",   # glutamate transport
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
    "CRHR1",    # CRF signaling
    "BDNF",     # plasticity
]


class DissociativeAmnesiaModel:
    """
    Atlas-grounded mechanistic scaffold for Dissociative Amnesia.

    What it does:
      1) Resolves DA-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates autobiographical memory blockade, preserved new learning,
         dissociative detachment, motivational withdrawal, and recovery potential.

    This is a research scaffold, not a clinical diagnostic tool.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "mni152",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.atlas = siibra.atlases.get(atlas_spec)

        try:
            self.parcellation = self.atlas.get_parcellation(parcellation_spec)
        except Exception:
            self.parcellation = siibra.parcellations.get(parcellation_spec)

        try:
            self.space = self.atlas.get_space(space_spec)
        except Exception:
            self.space = siibra.spaces.get(space_spec)

        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        # Chapter-consistent fronto-limbic autobiographical memory network.
        self.region_candidates: Dict[str, List[str]] = {
            "pfc_control": [
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "prefrontal",
                "frontal pole",
            ],
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
            "thalamus_proxy": [
                "thalamus",
                "anterior thalamus",
                "mediodorsal thalamus",
            ],
            "basal_ganglia_proxy": [
                "caudate",
                "putamen",
                "accumbens",
                "striatum",
                "basal ganglia",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "traumatic_stressor_intensity": "Severity of traumatic or overwhelming psychological stressor",
            "chronic_stress_load": "Persisting stress burden that prolongs biological dysregulation",
            "genetic_internalizing_diathesis": "Heritable vulnerability in affective, anxiety, or psychosis-linked stress sensitivity",
            "dissociative_state_dependency": "State-dependent barrier to autobiographical recall",
            "retrieval_support": "Protective stabilization and retrieval support such as safe context and state modulation",
        }

        self.latent_nodes: Dict[str, str] = {
            "hpa_axis_hyperactivation": "Stress-axis overactivation under trauma",
            "glucocorticoid_gene_expression_shift": "Stress-hormone-driven genomic effects on memory circuits",
            "hippocampal_contextual_disruption": "Stress-linked disruption of contextual autobiographical memory processing",
            "prefrontal_retrieval_inhibition": "Top-down blockade or inhibition of autobiographical retrieval",
            "frontolimbic_disconnection": "Functional disconnection across prefrontal-limbic memory circuitry",
            "autobiographical_access_block": "Failure to bring autobiographical memory into conscious narrative access",
            "state_dependent_memory_compartmentalization": "State-dependent sequestration of autobiographical material",
            "dopaminergic_motivational_suppression": "Reduced spontaneity, drive, and initiation linked to frontal-subcortical systems",
            "cholinergic_memory_state_instability": "Instability in memory-state gating and conscious access",
        }

        self.symptom_nodes: Dict[str, str] = {
            "retrograde_autobiographical_amnesia": "Loss of access to important autobiographical memory",
            "identity_narrative_disruption": "Break in coherent self-narrative continuity",
            "reduced_autonoetic_consciousness": "Reduced ability to consciously re-experience the personal past",
            "dissociative_detachment": "Detachment from autobiographical material or self-state",
            "motivational_withdrawal": "Avolitional or reduced-spontaneity state",
            "recall_recovery_potential": "Potential reversibility of inaccessible autobiographical memories",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "traumatic_stressor_intensity",
                "target": "hpa_axis_hyperactivation",
                "relation": "severe stress activates the HPA axis and trauma physiology",
                "da_change": "increased",
            },
            {
                "source": "traumatic_stressor_intensity",
                "target": "dissociative_state_dependency",
                "relation": "acute trauma can precipitate state-dependent dissociative barriers",
                "da_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "glucocorticoid_gene_expression_shift",
                "relation": "chronic stress sustains hormone-driven gene expression changes",
                "da_change": "increased",
            },
            {
                "source": "chronic_stress_load",
                "target": "hippocampal_contextual_disruption",
                "relation": "chronic stress burdens hippocampal contextual memory systems",
                "da_change": "increased",
            },
            {
                "source": "genetic_internalizing_diathesis",
                "target": "hpa_axis_hyperactivation",
                "relation": "genetic liability raises sensitivity of stress-response systems",
                "da_change": "increased susceptibility",
            },
            {
                "source": "genetic_internalizing_diathesis",
                "target": "dopaminergic_motivational_suppression",
                "relation": "genetic liability may worsen motivational withdrawal under stress",
                "da_change": "increased susceptibility",
            },
            {
                "source": "dissociative_state_dependency",
                "target": "state_dependent_memory_compartmentalization",
                "relation": "state-dependent processing segregates autobiographical access",
                "da_change": "increased",
            },
            {
                "source": "retrieval_support",
                "target": "autobiographical_access_block",
                "relation": "supportive retrieval conditions reduce autobiographical inaccessibility",
                "da_change": "protective",
            },
            {
                "source": "retrieval_support",
                "target": "recall_recovery_potential",
                "relation": "safe retrieval support increases recovery potential",
                "da_change": "protective",
            },
            {
                "source": "retrieval_support",
                "target": "prefrontal_retrieval_inhibition",
                "relation": "supportive state modulation can reduce inhibitory blockade",
                "da_change": "protective",
            },
            {
                "source": "hpa_axis_hyperactivation",
                "target": "glucocorticoid_gene_expression_shift",
                "relation": "glucocorticoids alter transcription in stress-sensitive brain regions",
                "da_change": "increased",
            },
            {
                "source": "hpa_axis_hyperactivation",
                "target": "hippocampal_contextual_disruption",
                "relation": "stress hormones compromise contextual memory integration",
                "da_change": "increased",
            },
            {
                "source": "hpa_axis_hyperactivation",
                "target": "prefrontal_retrieval_inhibition",
                "relation": "stress weakens flexible prefrontal retrieval control",
                "da_change": "increased",
            },
            {
                "source": "glucocorticoid_gene_expression_shift",
                "target": "hippocampus",
                "relation": "gene-expression changes burden hippocampal plasticity",
                "da_change": "increased dysregulation",
            },
            {
                "source": "glucocorticoid_gene_expression_shift",
                "target": "pfc_control",
                "relation": "gene-expression changes burden prefrontal regulation",
                "da_change": "increased dysregulation",
            },
            {
                "source": "glucocorticoid_gene_expression_shift",
                "target": "frontolimbic_disconnection",
                "relation": "longer-term stress effects weaken coordinated memory network function",
                "da_change": "increased",
            },
            {
                "source": "hippocampal_contextual_disruption",
                "target": "hippocampus",
                "relation": "disrupts contextual and autobiographical memory integration",
                "da_change": "increased dysregulation",
            },
            {
                "source": "hippocampal_contextual_disruption",
                "target": "autobiographical_access_block",
                "relation": "contextual memory disruption blocks personal-memory retrieval",
                "da_change": "increased",
            },
            {
                "source": "prefrontal_retrieval_inhibition",
                "target": "pfc_control",
                "relation": "increases inhibitory or maladaptive top-down retrieval control",
                "da_change": "increased dysregulation",
            },
            {
                "source": "prefrontal_retrieval_inhibition",
                "target": "autobiographical_access_block",
                "relation": "prefrontal blockade prevents conscious autobiographical retrieval",
                "da_change": "increased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "amygdala",
                "relation": "disconnects emotional salience from integrated autobiographical access",
                "da_change": "increased dysregulation",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "thalamus_proxy",
                "relation": "disrupts relay and integration within memory/self networks",
                "da_change": "increased dysregulation",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "autobiographical_access_block",
                "relation": "network disconnection prevents coherent personal-memory retrieval",
                "da_change": "increased",
            },
            {
                "source": "state_dependent_memory_compartmentalization",
                "target": "dissociative_detachment",
                "relation": "state-dependent sequestration promotes detachment from the personal past",
                "da_change": "increased",
            },
            {
                "source": "state_dependent_memory_compartmentalization",
                "target": "identity_narrative_disruption",
                "relation": "compartmentalization destabilizes coherent self-narrative continuity",
                "da_change": "increased",
            },
            {
                "source": "dopaminergic_motivational_suppression",
                "target": "basal_ganglia_proxy",
                "relation": "frontal-subcortical motivational systems become underdriven",
                "da_change": "increased dysregulation",
            },
            {
                "source": "dopaminergic_motivational_suppression",
                "target": "motivational_withdrawal",
                "relation": "reduced drive contributes to avolition and low spontaneity",
                "da_change": "increased",
            },
            {
                "source": "cholinergic_memory_state_instability",
                "target": "autobiographical_access_block",
                "relation": "unstable memory-state gating can impair conscious autobiographical access",
                "da_change": "increased",
            },
            {
                "source": "autobiographical_access_block",
                "target": "retrograde_autobiographical_amnesia",
                "relation": "loss of access presents as retrograde autobiographical amnesia",
                "da_change": "increased",
            },
            {
                "source": "autobiographical_access_block",
                "target": "reduced_autonoetic_consciousness",
                "relation": "blocks conscious re-experiencing of the personal past",
                "da_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "dissociative_detachment",
                "relation": "limbic stress burden can intensify detachment states",
                "da_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "identity_narrative_disruption",
                "relation": "relay dysfunction can destabilize integrated self-representation",
                "da_change": "increased",
            },
            {
                "source": "recall_recovery_potential",
                "target": "retrograde_autobiographical_amnesia",
                "relation": "recovery potential can reduce the persistence of the amnestic blockade",
                "da_change": "protective",
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
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

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
            "thalamus",
            "basal ganglia",
            "striatum",
        } else 0
        return (left_bonus, right_penalty, generic_penalty)

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
        node_keys: Sequence[str] = ("pfc_control", "hippocampus", "amygdala", "thalamus_proxy", "basal_ganglia_proxy"),
    ) -> pd.DataFrame:
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
                warnings.warn(f"Could not resolve a region for node '{key}' in this environment")
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
                    "description": "Atlas-backed dissociative-amnesia circuit node",
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
        traumatic_stressor_intensity: float,
        chronic_stress_load: float,
        genetic_internalizing_diathesis: float,
        dissociative_state_dependency: float,
        retrieval_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while retrieval_support is protective.
        """
        t = self._clip01(traumatic_stressor_intensity)
        c = self._clip01(chronic_stress_load)
        g = self._clip01(genetic_internalizing_diathesis)
        d = self._clip01(dissociative_state_dependency)
        r = self._clip01(retrieval_support)

        # Latent biology
        hpa_axis_hyperactivation = self._clip01(
            0.40 * t + 0.20 * c + 0.10 * g - 0.10 * r
        )
        glucocorticoid_gene_expression_shift = self._clip01(
            0.35 * hpa_axis_hyperactivation + 0.25 * c - 0.10 * r
        )
        hippocampal_contextual_disruption = self._clip01(
            0.35 * hpa_axis_hyperactivation
            + 0.25 * glucocorticoid_gene_expression_shift
            + 0.10 * t
        )
        prefrontal_retrieval_inhibition = self._clip01(
            0.30 * hpa_axis_hyperactivation
            + 0.25 * d
            + 0.20 * glucocorticoid_gene_expression_shift
            - 0.15 * r
        )
        dopaminergic_motivational_suppression = self._clip01(
            0.25 * t + 0.20 * hpa_axis_hyperactivation + 0.15 * d - 0.10 * r
        )
        cholinergic_memory_state_instability = self._clip01(
            0.25 * d + 0.15 * t
        )
        frontolimbic_disconnection = self._clip01(
            0.30 * prefrontal_retrieval_inhibition
            + 0.25 * hippocampal_contextual_disruption
            + 0.10 * d
            - 0.15 * r
        )
        autobiographical_access_block = self._clip01(
            0.35 * frontolimbic_disconnection
            + 0.25 * prefrontal_retrieval_inhibition
            + 0.20 * d
            + 0.10 * cholinergic_memory_state_instability
            - 0.20 * r
        )
        state_dependent_memory_compartmentalization = self._clip01(
            0.35 * d + 0.20 * autobiographical_access_block - 0.15 * r
        )

        # Regional state proxies
        pfc_control = self._clip01(
            0.45 * prefrontal_retrieval_inhibition
            + 0.20 * frontolimbic_disconnection
            - 0.10 * r
        )
        hippocampus = self._clip01(
            0.45 * hippocampal_contextual_disruption
            + 0.20 * glucocorticoid_gene_expression_shift
        )
        amygdala = self._clip01(
            0.35 * frontolimbic_disconnection
            + 0.15 * hpa_axis_hyperactivation
        )
        thalamus_proxy = self._clip01(
            0.25 * frontolimbic_disconnection
            + 0.20 * state_dependent_memory_compartmentalization
        )
        basal_ganglia_proxy = self._clip01(
            0.35 * dopaminergic_motivational_suppression
            + 0.10 * frontolimbic_disconnection
        )

        # Symptoms / behavior
        retrograde_autobiographical_amnesia = self._clip01(
            0.45 * autobiographical_access_block
            + 0.20 * hippocampus
            + 0.15 * pfc_control
            - 0.15 * r
        )
        identity_narrative_disruption = self._clip01(
            0.35 * autobiographical_access_block
            + 0.20 * state_dependent_memory_compartmentalization
            + 0.10 * amygdala
            - 0.10 * r
        )
        reduced_autonoetic_consciousness = self._clip01(
            0.35 * autobiographical_access_block
            + 0.20 * pfc_control
            + 0.15 * hippocampus
            - 0.10 * r
        )
        dissociative_detachment = self._clip01(
            0.30 * state_dependent_memory_compartmentalization
            + 0.20 * amygdala
            + 0.15 * hpa_axis_hyperactivation
            - 0.10 * r
        )
        motivational_withdrawal = self._clip01(
            0.40 * dopaminergic_motivational_suppression
            + 0.20 * basal_ganglia_proxy
            + 0.10 * pfc_control
            - 0.10 * r
        )
        recall_recovery_potential = self._clip01(
            0.50 * r + 0.20 * (1.0 - autobiographical_access_block) + 0.10 * (1.0 - state_dependent_memory_compartmentalization)
        )

        preserved_new_learning = self._clip01(
            0.80 - 0.15 * hippocampal_contextual_disruption - 0.10 * hpa_axis_hyperactivation + 0.10 * r
        )

        return {
            "inputs": pd.Series(
                {
                    "traumatic_stressor_intensity": t,
                    "chronic_stress_load": c,
                    "genetic_internalizing_diathesis": g,
                    "dissociative_state_dependency": d,
                    "retrieval_support": r,
                }
            ),
            "latents": pd.Series(
                {
                    "autobiographical_access_block": autobiographical_access_block,
                    "frontolimbic_disconnection": frontolimbic_disconnection,
                    "prefrontal_retrieval_inhibition": prefrontal_retrieval_inhibition,
                    "hippocampal_contextual_disruption": hippocampal_contextual_disruption,
                    "state_dependent_memory_compartmentalization": state_dependent_memory_compartmentalization,
                    "hpa_axis_hyperactivation": hpa_axis_hyperactivation,
                    "glucocorticoid_gene_expression_shift": glucocorticoid_gene_expression_shift,
                    "dopaminergic_motivational_suppression": dopaminergic_motivational_suppression,
                    "cholinergic_memory_state_instability": cholinergic_memory_state_instability,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "pfc_control": pfc_control,
                    "hippocampus": hippocampus,
                    "amygdala": amygdala,
                    "thalamus_proxy": thalamus_proxy,
                    "basal_ganglia_proxy": basal_ganglia_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "retrograde_autobiographical_amnesia": retrograde_autobiographical_amnesia,
                    "identity_narrative_disruption": identity_narrative_disruption,
                    "reduced_autonoetic_consciousness": reduced_autonoetic_consciousness,
                    "dissociative_detachment": dissociative_detachment,
                    "motivational_withdrawal": motivational_withdrawal,
                    "recall_recovery_potential": recall_recovery_potential,
                }
            ).sort_values(ascending=False),
            "retained_functions": pd.Series(
                {
                    "preserved_new_learning": preserved_new_learning,
                }
            ),
            "phenotypes": pd.Series(
                {
                    "trauma_locked_memory_block_profile": self._clip01(
                        0.45 * retrograde_autobiographical_amnesia
                        + 0.25 * autobiographical_access_block
                        + 0.20 * state_dependent_memory_compartmentalization
                    ),
                    "identity_discontinuity_profile": self._clip01(
                        0.45 * identity_narrative_disruption
                        + 0.25 * reduced_autonoetic_consciousness
                        + 0.20 * dissociative_detachment
                    ),
                    "motivational_shutdown_profile": self._clip01(
                        0.45 * motivational_withdrawal
                        + 0.25 * dopaminergic_motivational_suppression
                        + 0.15 * pfc_control
                    ),
                    "reversible_retrieval_barrier_profile": self._clip01(
                        0.35 * retrograde_autobiographical_amnesia
                        + 0.30 * recall_recovery_potential
                        + 0.20 * state_dependent_memory_compartmentalization
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-24, -12, -18)).head(10)
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
    model = DissociativeAmnesiaModel()

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
        bundle["edges"][["source", "target", "relation", "da_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["pfc_control", "hippocampus", "amygdala", "thalamus_proxy", "basal_ganglia_proxy"]:
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
        traumatic_stressor_intensity=0.85,
        chronic_stress_load=0.65,
        genetic_internalizing_diathesis=0.55,
        dissociative_state_dependency=0.80,
        retrieval_support=0.25,
    )

    print("\n=== LATENT BIOLOGY ===")
    print(sim["latents"].to_string())

    print("\n=== REGIONAL STATE ===")
    print(sim["regional_state"].to_string())

    print("\n=== SYMPTOMS ===")
    print(sim["symptoms"].to_string())

    print("\n=== RETAINED FUNCTIONS ===")
    print(sim["retained_functions"].to_string())

    print("\n=== PHENOTYPES ===")
    print(sim["phenotypes"].to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((-24, -12, -18)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("thalamus").to_string(index=False))
    # print(model.suggest_regions("frontal pole").to_string(index=False))
