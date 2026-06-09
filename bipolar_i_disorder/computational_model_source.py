from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Bipolar I-oriented panel:
# - monoamines
# - stress/HPA-axis
# - neurotrophic support
# - circadian rhythm biology
# - inhibitory/excitatory balance
#
# This panel is hypothesis-driven, because the chapter mainly specifies
# systems and circuits rather than a fixed set of genes.
DEFAULT_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor
    "SLC6A2",   # norepinephrine transporter
    "DBH",      # dopamine beta hydroxylase
    "DRD2",     # dopamine receptor
    "COMT",     # catecholamine metabolism
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
    "CRHR1",    # CRH signaling
    "BDNF",     # neurotrophic support
    "CLOCK",    # circadian rhythm
    "ARNTL",    # BMAL1 / circadian rhythm
    "PER3",     # circadian rhythm
    "GABRA2",   # GABA-A receptor subunit
    "GRIN2B",   # NMDA receptor subunit
    "IL6",      # inflammatory tone
    "TNF",      # inflammatory tone
]


class BipolarIDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Bipolar I Disorder.

    What it does:
      1) Resolves chapter-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates mania, bipolar depression, mixed-state pressure,
         impulsivity/disinhibition, and stress-circadian mood instability.

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

        # The chapter emphasizes a distributed prefrontal-limbic-striatal-thalamic circuit.
        # Circadian rhythm is modeled as a latent system, not a parcel.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "DG (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "hippocampus",
            ],
            "acc": [
                "Area 33 (ACC) left",
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "ACC",
            ],
            "pfc_control": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "prefrontal",
            ],
            "striatum_proxy": [
                "caudate",
                "putamen",
                "accumbens",
                "striatum",
                "basal ganglia",
            ],
            "thalamus_proxy": [
                "thalamus",
                "anterior thalamus",
                "mediodorsal thalamus",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_liability": "Heritable liability for Bipolar I Disorder",
            "psychosocial_stress": "Stressful life events that precipitate episodes",
            "circadian_disruption": "Sleep-wake and circadian instability",
            "inflammatory_burden": "Inflammatory or immune burden contributing to mood dysregulation",
            "recovery_support": "Protective treatment, sleep regularity, and recovery structure",
        }

        self.latent_nodes: Dict[str, str] = {
            "hpa_axis_dysregulation": "Sensitized stress-response system across mood states",
            "autonomic_imbalance": "Sympathetic/parasympathetic dysregulation linked to mood state",
            "monoamine_dysregulation": "Serotonin, norepinephrine, and dopamine instability",
            "circadian_rhythm_instability": "Oscillatory rhythm instability linked to episode switching",
            "reward_circuit_bias": "Arousal/reward activation bias linked to manic states",
            "inflammatory_signaling": "Immune-inflammatory contribution to affective burden",
            "bdnf_reduction": "Reduced neurotrophic support in stress-sensitive regions",
            "hypofrontality": "Reduced frontal control / executive recruitment",
            "cortico_limbic_striatal_thalamic_instability": "Distributed mood-circuit dysfunction",
        }

        self.symptom_nodes: Dict[str, str] = {
            "mania_activation": "Manic activation, elevated arousal, and expansive drive",
            "bipolar_depression": "Depressive burden with apathy/cognitive slowing",
            "mood_lability": "Rapid shifts in affective intensity",
            "impulsivity_disinhibition": "Impaired restraint and risky behavior",
            "psychomotor_shift": "Agitation or retardation across mood states",
            "cognitive_impairment": "Executive and memory-related cognitive burden",
            "mixed_state_pressure": "Concurrent activation and depressive distress / mixed instability",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_liability",
                "target": "monoamine_dysregulation",
                "relation": "raises vulnerability of core transmitter regulation",
                "bipolar_i_change": "increased susceptibility",
            },
            {
                "source": "genetic_liability",
                "target": "circadian_rhythm_instability",
                "relation": "raises vulnerability of sleep-wake and rhythm control",
                "bipolar_i_change": "increased susceptibility",
            },
            {
                "source": "genetic_liability",
                "target": "hpa_axis_dysregulation",
                "relation": "raises vulnerability of stress-response regulation",
                "bipolar_i_change": "increased susceptibility",
            },
            {
                "source": "psychosocial_stress",
                "target": "hpa_axis_dysregulation",
                "relation": "precipitates and amplifies episode-related stress biology",
                "bipolar_i_change": "increased",
            },
            {
                "source": "psychosocial_stress",
                "target": "monoamine_dysregulation",
                "relation": "interacts with stress hormones to destabilize mood chemistry",
                "bipolar_i_change": "increased",
            },
            {
                "source": "circadian_disruption",
                "target": "circadian_rhythm_instability",
                "relation": "destabilizes internal timing and episode regulation",
                "bipolar_i_change": "increased",
            },
            {
                "source": "circadian_disruption",
                "target": "mania_activation",
                "relation": "biases toward activation and episode escalation",
                "bipolar_i_change": "increased",
            },
            {
                "source": "inflammatory_burden",
                "target": "inflammatory_signaling",
                "relation": "raises inflammatory pressure on mood systems",
                "bipolar_i_change": "increased",
            },
            {
                "source": "recovery_support",
                "target": "circadian_rhythm_instability",
                "relation": "buffers sleep-wake instability",
                "bipolar_i_change": "protective",
            },
            {
                "source": "recovery_support",
                "target": "hpa_axis_dysregulation",
                "relation": "buffers stress sensitization",
                "bipolar_i_change": "protective",
            },
            {
                "source": "recovery_support",
                "target": "mania_activation",
                "relation": "reduces escalation into severe activation",
                "bipolar_i_change": "protective",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "autonomic_imbalance",
                "relation": "amplifies sympathetic/parasympathetic dysregulation",
                "bipolar_i_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "monoamine_dysregulation",
                "relation": "destabilizes mood transmitter systems bidirectionally",
                "bipolar_i_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "bdnf_reduction",
                "relation": "reduces trophic support under repeated stress load",
                "bipolar_i_change": "increased",
            },
            {
                "source": "circadian_rhythm_instability",
                "target": "reward_circuit_bias",
                "relation": "destabilizes arousal and reward-state regulation",
                "bipolar_i_change": "increased",
            },
            {
                "source": "inflammatory_signaling",
                "target": "bdnf_reduction",
                "relation": "suppresses neurotrophic support in stress-sensitive structures",
                "bipolar_i_change": "increased",
            },
            {
                "source": "inflammatory_signaling",
                "target": "bipolar_depression",
                "relation": "increases depressive burden",
                "bipolar_i_change": "increased",
            },
            {
                "source": "monoamine_dysregulation",
                "target": "reward_circuit_bias",
                "relation": "destabilizes motivational and arousal systems",
                "bipolar_i_change": "increased",
            },
            {
                "source": "monoamine_dysregulation",
                "target": "mania_activation",
                "relation": "supports elevated activation and mood expansion",
                "bipolar_i_change": "increased",
            },
            {
                "source": "monoamine_dysregulation",
                "target": "bipolar_depression",
                "relation": "supports depressive burden and low drive",
                "bipolar_i_change": "increased",
            },
            {
                "source": "reward_circuit_bias",
                "target": "striatum_proxy",
                "relation": "loads subcortical motivational circuitry",
                "bipolar_i_change": "increased activation bias",
            },
            {
                "source": "reward_circuit_bias",
                "target": "mania_activation",
                "relation": "supports incentive-driven manic activation",
                "bipolar_i_change": "increased",
            },
            {
                "source": "bdnf_reduction",
                "target": "hippocampus",
                "relation": "reduces trophic support in limbic memory circuitry",
                "bipolar_i_change": "reduced support",
            },
            {
                "source": "hypofrontality",
                "target": "pfc_control",
                "relation": "reduces executive control and top-down regulation",
                "bipolar_i_change": "reduced function",
            },
            {
                "source": "hypofrontality",
                "target": "cognitive_impairment",
                "relation": "contributes to executive and motivational impairment",
                "bipolar_i_change": "increased",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "amygdala",
                "relation": "destabilizes limbic emotional reactivity",
                "bipolar_i_change": "increased dysregulation",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "hippocampus",
                "relation": "destabilizes stress-sensitive memory/context circuitry",
                "bipolar_i_change": "increased dysregulation",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "acc",
                "relation": "reduces monitoring and emotion-control stability",
                "bipolar_i_change": "reduced function",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "pfc_control",
                "relation": "weakens distributed executive regulation",
                "bipolar_i_change": "reduced function",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "striatum_proxy",
                "relation": "destabilizes motivational and psychomotor circuitry",
                "bipolar_i_change": "increased dysregulation",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "thalamus_proxy",
                "relation": "weakens relay/integration across mood networks",
                "bipolar_i_change": "increased dysregulation",
            },
            {
                "source": "pfc_control",
                "target": "amygdala",
                "relation": "normally provides top-down control",
                "bipolar_i_change": "reduced inhibition",
            },
            {
                "source": "amygdala",
                "target": "mood_lability",
                "relation": "amplifies emotional intensity and instability",
                "bipolar_i_change": "increased",
            },
            {
                "source": "acc",
                "target": "mixed_state_pressure",
                "relation": "weakened monitoring destabilizes mixed mood regulation",
                "bipolar_i_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "impulsivity_disinhibition",
                "relation": "reduced frontal control impairs restraint",
                "bipolar_i_change": "increased",
            },
            {
                "source": "striatum_proxy",
                "target": "psychomotor_shift",
                "relation": "biases psychomotor activation or slowing",
                "bipolar_i_change": "increased",
            },
            {
                "source": "striatum_proxy",
                "target": "mania_activation",
                "relation": "supports activation and drive",
                "bipolar_i_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "cognitive_impairment",
                "relation": "relay dysfunction weakens coordinated regulation",
                "bipolar_i_change": "increased",
            },
            {
                "source": "autonomic_imbalance",
                "target": "mania_activation",
                "relation": "supports sympathetic hyperactivation in manic states",
                "bipolar_i_change": "increased",
            },
            {
                "source": "autonomic_imbalance",
                "target": "mood_lability",
                "relation": "increases physiologic-affective instability",
                "bipolar_i_change": "increased",
            },
            {
                "source": "bipolar_depression",
                "target": "mixed_state_pressure",
                "relation": "depressive burden can coexist with activation pressure",
                "bipolar_i_change": "increased",
            },
            {
                "source": "mania_activation",
                "target": "mixed_state_pressure",
                "relation": "manic activation can coexist with dysphoric burden",
                "bipolar_i_change": "increased",
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "thalamus",
            "striatum",
            "prefrontal cortex",
            "anterior cingulate",
        } else 0
        return (left_bonus, right_penalty, generic_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation_spec)
            except Exception:
                pass

            try:
                matches = self.parcellation.find(spec, filter_children=False)
                if matches:
                    matches = sorted(matches, key=self._region_rank)
                    return matches[0]
            except Exception:
                pass

        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        """
        Helper for tuning region proxies against the atlas.
        Useful for refining striatal or thalamic candidates.
        """
        try:
            matches = self.parcellation.find(keyword, filter_children=False)
        except Exception:
            return pd.DataFrame(columns=["name", "identifier"])

        rows = []
        seen = set()
        for region in sorted(matches, key=self._region_rank):
            row = (self._name_of(region), getattr(region, "identifier", None))
            if row in seen:
                continue
            seen.add(row)
            rows.append(
                {
                    "name": row[0],
                    "identifier": row[1],
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
            "amygdala",
            "hippocampus",
            "acc",
            "pfc_control",
            "striatum_proxy",
            "thalamus_proxy",
        ),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the Bipolar I circuit.
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
                    "description": "Atlas-backed Bipolar I circuit node",
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
        genetic_liability: float,
        psychosocial_stress: float,
        circadian_disruption: float,
        inflammatory_burden: float,
        recovery_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while recovery_support is protective.
        """
        g = self._clip01(genetic_liability)
        s = self._clip01(psychosocial_stress)
        c = self._clip01(circadian_disruption)
        i = self._clip01(inflammatory_burden)
        r = self._clip01(recovery_support)

        # Latent biology
        circadian_rhythm_instability = self._clip01(
            0.40 * c + 0.20 * g - 0.20 * r
        )
        hpa_axis_dysregulation = self._clip01(
            0.35 * s + 0.20 * g + 0.15 * circadian_rhythm_instability - 0.20 * r
        )
        inflammatory_signaling = self._clip01(
            0.55 * i + 0.10 * s - 0.15 * r
        )
        bdnf_reduction = self._clip01(
            0.35 * inflammatory_signaling + 0.25 * hpa_axis_dysregulation
        )
        monoamine_dysregulation = self._clip01(
            0.30 * g
            + 0.20 * hpa_axis_dysregulation
            + 0.20 * circadian_rhythm_instability
            + 0.10 * inflammatory_signaling
            - 0.10 * r
        )
        autonomic_imbalance = self._clip01(
            0.35 * hpa_axis_dysregulation + 0.30 * circadian_rhythm_instability - 0.15 * r
        )
        reward_circuit_bias = self._clip01(
            0.30 * monoamine_dysregulation
            + 0.25 * circadian_rhythm_instability
            + 0.10 * g
            - 0.10 * r
        )
        hypofrontality = self._clip01(
            0.35 * monoamine_dysregulation
            + 0.25 * bdnf_reduction
            + 0.10 * s
            - 0.20 * r
        )
        cortico_limbic_striatal_thalamic_instability = self._clip01(
            0.20 * monoamine_dysregulation
            + 0.20 * hypofrontality
            + 0.15 * circadian_rhythm_instability
            + 0.15 * inflammatory_signaling
            + 0.15 * hpa_axis_dysregulation
            - 0.20 * r
        )

        # Regional state proxies
        amygdala = self._clip01(
            0.35 * cortico_limbic_striatal_thalamic_instability + 0.20 * hpa_axis_dysregulation
        )
        hippocampus = self._clip01(
            0.35 * bdnf_reduction + 0.20 * hpa_axis_dysregulation
        )
        acc = self._clip01(
            0.30 * cortico_limbic_striatal_thalamic_instability + 0.20 * hypofrontality
        )
        pfc_control = self._clip01(
            0.40 * hypofrontality + 0.20 * cortico_limbic_striatal_thalamic_instability - 0.15 * r
        )
        striatum_proxy = self._clip01(
            0.35 * reward_circuit_bias + 0.20 * cortico_limbic_striatal_thalamic_instability
        )
        thalamus_proxy = self._clip01(
            0.30 * cortico_limbic_striatal_thalamic_instability + 0.10 * circadian_rhythm_instability
        )

        # Symptoms / behavior
        mania_activation = self._clip01(
            0.30 * reward_circuit_bias
            + 0.25 * autonomic_imbalance
            + 0.20 * circadian_rhythm_instability
            + 0.10 * striatum_proxy
            - 0.15 * r
        )
        bipolar_depression = self._clip01(
            0.30 * bdnf_reduction
            + 0.25 * monoamine_dysregulation
            + 0.15 * hippocampus
            + 0.10 * inflammatory_signaling
            - 0.10 * r
        )
        mood_lability = self._clip01(
            0.30 * amygdala + 0.20 * autonomic_imbalance + 0.20 * acc + 0.10 * mania_activation
        )
        impulsivity_disinhibition = self._clip01(
            0.35 * pfc_control + 0.20 * mania_activation + 0.15 * striatum_proxy - 0.10 * r
        )
        psychomotor_shift = self._clip01(
            0.30 * striatum_proxy + 0.20 * thalamus_proxy + 0.15 * mania_activation + 0.10 * bipolar_depression
        )
        cognitive_impairment = self._clip01(
            0.30 * pfc_control + 0.20 * hippocampus + 0.15 * acc + 0.10 * hypofrontality
        )
        mixed_state_pressure = self._clip01(
            0.30 * mania_activation
            + 0.25 * bipolar_depression
            + 0.20 * mood_lability
            + 0.10 * autonomic_imbalance
        )

        return {
            "inputs": pd.Series(
                {
                    "genetic_liability": g,
                    "psychosocial_stress": s,
                    "circadian_disruption": c,
                    "inflammatory_burden": i,
                    "recovery_support": r,
                }
            ),
            "latents": pd.Series(
                {
                    "cortico_limbic_striatal_thalamic_instability": cortico_limbic_striatal_thalamic_instability,
                    "monoamine_dysregulation": monoamine_dysregulation,
                    "hpa_axis_dysregulation": hpa_axis_dysregulation,
                    "circadian_rhythm_instability": circadian_rhythm_instability,
                    "reward_circuit_bias": reward_circuit_bias,
                    "autonomic_imbalance": autonomic_imbalance,
                    "inflammatory_signaling": inflammatory_signaling,
                    "bdnf_reduction": bdnf_reduction,
                    "hypofrontality": hypofrontality,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "amygdala": amygdala,
                    "hippocampus": hippocampus,
                    "acc": acc,
                    "pfc_control": pfc_control,
                    "striatum_proxy": striatum_proxy,
                    "thalamus_proxy": thalamus_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "mania_activation": mania_activation,
                    "bipolar_depression": bipolar_depression,
                    "mixed_state_pressure": mixed_state_pressure,
                    "mood_lability": mood_lability,
                    "impulsivity_disinhibition": impulsivity_disinhibition,
                    "psychomotor_shift": psychomotor_shift,
                    "cognitive_impairment": cognitive_impairment,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "mania_dominant_profile": self._clip01(
                        0.45 * mania_activation + 0.25 * impulsivity_disinhibition + 0.15 * psychomotor_shift
                    ),
                    "depression_dominant_profile": self._clip01(
                        0.45 * bipolar_depression + 0.25 * cognitive_impairment + 0.15 * bdnf_reduction
                    ),
                    "mixed_state_profile": self._clip01(
                        0.50 * mixed_state_pressure + 0.20 * mood_lability + 0.15 * autonomic_imbalance
                    ),
                    "stress_circadian_profile": self._clip01(
                        0.35 * hpa_axis_dysregulation
                        + 0.30 * circadian_rhythm_instability
                        + 0.15 * mood_lability
                        + 0.10 * mania_activation
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-16, 18, -12)).head(10)
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
    model = BipolarIDisorderModel()

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
        bundle["edges"][["source", "target", "relation", "bipolar_i_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["amygdala", "hippocampus", "acc", "pfc_control", "striatum_proxy", "thalamus_proxy"]:
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
        genetic_liability=0.80,
        psychosocial_stress=0.70,
        circadian_disruption=0.85,
        inflammatory_burden=0.40,
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
    # print(model.assign_mni_point((-16, 18, -12)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("thalamus").to_string(index=False))
    # print(model.suggest_regions("striatum").to_string(index=False))
