from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Compulsive and Related Disorder-oriented panel:
# - serotonin / obsessional burden / treatment response
# - dopamine / salience and habit reinforcement
# - glutamate / habit plasticity / corticostriatal learning
# - GABA / inhibitory control
# - stress / HPA-CRF sensitization
DEFAULT_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor
    "HTR2A",    # serotonin receptor
    "TPH2",     # serotonin synthesis
    "DRD2",     # dopamine receptor
    "SLC6A3",   # dopamine transporter
    "COMT",     # catecholamine metabolism
    "GABRA2",   # GABA-A receptor
    "GABRB2",   # GABA-A receptor
    "GAD1",     # GABA synthesis
    "GRIN2B",   # NMDA receptor subunit
    "SLC1A1",   # glutamate transport / OCD-related candidate
    "DLGAP3",   # compulsivity-related synaptic scaffold
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
    "CRHR1",    # CRF signaling
    "BDNF",     # plasticity
]


class CompulsiveAndRelatedDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for a generic Compulsive and Related Disorder.

    What it does:
      1) Resolves chapter-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates obsessional burden, compulsive rituals, stress-triggered worsening,
         top-down control failure, and habit-dominant behavior.

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

        # Chapter-consistent CSTC network with frontal lesion-model anchors.
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
            "pfc_control": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "prefrontal",
                "dorsolateral prefrontal",
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
            "genetic_vulnerability": "Heritable liability for compulsive-spectrum symptoms",
            "early_onset_loading": "Early-onset/familial loading associated with stronger heritability",
            "chronic_stress": "Stress burden that lowers the threshold for obsessive-compulsive phenomena",
            "frontal_or_cstc_insult": "Frontal/CSTC lesion-model burden such as TBI or frontotemporal pathology",
            "serotonergic_treatment_support": "Protective serotonergic treatment and control support",
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_dysregulation": "Obsessive-compulsive symptom burden linked to serotonin dysfunction",
            "glutamate_habit_plasticity": "Glutamate-dependent strengthening of pathological habits and beliefs",
            "hpa_crf_stress_sensitization": "Stress-system sensitization lowering compulsive thresholds",
            "stress_habit_shift": "Shift from goal-directed to habitual responding under stress",
            "cstc_loop_hyperactivity": "Cortico-striato-thalamo-cortical circuit overactivity or dysregulation",
            "top_down_control_failure": "Failure of frontal inhibition over intrusive urges and repetitive behavior",
            "error_signal_overdrive": "Persistent error/conflict signaling linked to obsessions",
            "habit_dominance": "Stimulus-response habit dominance over flexible control",
            "compulsive_urge_amplification": "Amplified urge state preceding compulsive acts",
        }

        self.symptom_nodes: Dict[str, str] = {
            "intrusive_obsessional_thoughts": "Intrusive, repetitive, difficult-to-dismiss thoughts or urges",
            "compulsive_rituals": "Repetitive ritualized actions or mental acts",
            "urge_resistance_failure": "Difficulty resisting intrusive urges or ritual completion",
            "stress_triggered_exacerbation": "Stress-related worsening of intrusive thoughts and compulsions",
            "cognitive_rigidity": "Reduced cognitive flexibility and habitual responding",
            "distress_impairment": "Functional impairment and distress from compulsive symptoms",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_dysregulation",
                "relation": "raises vulnerability for obsessional-serotonergic dysregulation",
                "compulsive_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "glutamate_habit_plasticity",
                "relation": "raises vulnerability in glutamate-dependent habit systems",
                "compulsive_change": "increased susceptibility",
            },
            {
                "source": "early_onset_loading",
                "target": "genetic_vulnerability",
                "relation": "marks stronger familial/genetic loading",
                "compulsive_change": "increased",
            },
            {
                "source": "chronic_stress",
                "target": "hpa_crf_stress_sensitization",
                "relation": "stress sensitizes compulsive circuitry",
                "compulsive_change": "increased",
            },
            {
                "source": "chronic_stress",
                "target": "stress_triggered_exacerbation",
                "relation": "stress lowers the threshold for obsessive thoughts and compulsive urges",
                "compulsive_change": "increased",
            },
            {
                "source": "frontal_or_cstc_insult",
                "target": "top_down_control_failure",
                "relation": "lesion-model burden weakens frontal inhibitory control",
                "compulsive_change": "increased",
            },
            {
                "source": "frontal_or_cstc_insult",
                "target": "cstc_loop_hyperactivity",
                "relation": "frontal or subcortical injury can dysregulate CSTC loops",
                "compulsive_change": "increased",
            },
            {
                "source": "serotonergic_treatment_support",
                "target": "serotonergic_dysregulation",
                "relation": "buffers obsessional-serotonergic burden",
                "compulsive_change": "protective",
            },
            {
                "source": "serotonergic_treatment_support",
                "target": "intrusive_obsessional_thoughts",
                "relation": "can reduce intrusive thought burden",
                "compulsive_change": "protective",
            },
            {
                "source": "serotonergic_treatment_support",
                "target": "compulsive_rituals",
                "relation": "can reduce compulsive ritual pressure",
                "compulsive_change": "protective",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "intrusive_obsessional_thoughts",
                "relation": "supports persistent obsessional burden",
                "compulsive_change": "increased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "compulsive_urge_amplification",
                "relation": "amplifies compulsive urge states",
                "compulsive_change": "increased",
            },
            {
                "source": "glutamate_habit_plasticity",
                "target": "stress_habit_shift",
                "relation": "glutamate-dependent plasticity promotes shift toward habits",
                "compulsive_change": "increased",
            },
            {
                "source": "glutamate_habit_plasticity",
                "target": "cstc_loop_hyperactivity",
                "relation": "strengthens pathological corticostriatal habit circuits",
                "compulsive_change": "increased",
            },
            {
                "source": "hpa_crf_stress_sensitization",
                "target": "stress_habit_shift",
                "relation": "stress biases behavior from goal-directed to habitual control",
                "compulsive_change": "increased",
            },
            {
                "source": "hpa_crf_stress_sensitization",
                "target": "compulsive_urge_amplification",
                "relation": "stress-related neurochemical shifts amplify compulsive urges",
                "compulsive_change": "increased",
            },
            {
                "source": "stress_habit_shift",
                "target": "habit_dominance",
                "relation": "habitual responding becomes dominant over flexible control",
                "compulsive_change": "increased",
            },
            {
                "source": "cstc_loop_hyperactivity",
                "target": "ofc",
                "relation": "loads orbitofrontal valuation and checking circuitry",
                "compulsive_change": "increased dysregulation",
            },
            {
                "source": "cstc_loop_hyperactivity",
                "target": "acc",
                "relation": "loads anterior cingulate error/conflict circuitry",
                "compulsive_change": "increased dysregulation",
            },
            {
                "source": "cstc_loop_hyperactivity",
                "target": "striatum_proxy",
                "relation": "loads striatal habit and action-selection circuitry",
                "compulsive_change": "increased dysregulation",
            },
            {
                "source": "cstc_loop_hyperactivity",
                "target": "thalamus_proxy",
                "relation": "loads thalamic relay/re-entry loops in compulsivity",
                "compulsive_change": "increased dysregulation",
            },
            {
                "source": "cstc_loop_hyperactivity",
                "target": "error_signal_overdrive",
                "relation": "dysregulated CSTC looping can sustain persistent alarm/error signals",
                "compulsive_change": "increased",
            },
            {
                "source": "top_down_control_failure",
                "target": "pfc_control",
                "relation": "reduces frontal regulation over intrusive urges",
                "compulsive_change": "reduced function",
            },
            {
                "source": "top_down_control_failure",
                "target": "urge_resistance_failure",
                "relation": "weakens ability to resist obsessions and rituals",
                "compulsive_change": "increased",
            },
            {
                "source": "top_down_control_failure",
                "target": "cognitive_rigidity",
                "relation": "reduces flexible switching away from repetitive control loops",
                "compulsive_change": "increased",
            },
            {
                "source": "error_signal_overdrive",
                "target": "intrusive_obsessional_thoughts",
                "relation": "persistent error/conflict signals reinforce obsessional thoughts",
                "compulsive_change": "increased",
            },
            {
                "source": "error_signal_overdrive",
                "target": "distress_impairment",
                "relation": "persistent inner alarm increases distress and dysfunction",
                "compulsive_change": "increased",
            },
            {
                "source": "habit_dominance",
                "target": "compulsive_rituals",
                "relation": "habitual responding promotes repetitive ritual performance",
                "compulsive_change": "increased",
            },
            {
                "source": "habit_dominance",
                "target": "cognitive_rigidity",
                "relation": "stimulus-response dominance reduces flexible cognition",
                "compulsive_change": "increased",
            },
            {
                "source": "compulsive_urge_amplification",
                "target": "urge_resistance_failure",
                "relation": "amplified urges are harder to inhibit",
                "compulsive_change": "increased",
            },
            {
                "source": "compulsive_urge_amplification",
                "target": "compulsive_rituals",
                "relation": "amplified urge states promote ritual completion",
                "compulsive_change": "increased",
            },
            {
                "source": "ofc",
                "target": "intrusive_obsessional_thoughts",
                "relation": "altered orbitofrontal valuation/checking processes intensify obsessional content",
                "compulsive_change": "increased",
            },
            {
                "source": "acc",
                "target": "error_signal_overdrive",
                "relation": "anterior cingulate dysfunction promotes persistent conflict/error monitoring",
                "compulsive_change": "increased",
            },
            {
                "source": "striatum_proxy",
                "target": "compulsive_rituals",
                "relation": "striatal habit circuitry supports repetitive actions",
                "compulsive_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "intrusive_obsessional_thoughts",
                "relation": "thalamic relay dysfunction can sustain repetitive looped processing",
                "compulsive_change": "increased",
            },
            {
                "source": "stress_triggered_exacerbation",
                "target": "intrusive_obsessional_thoughts",
                "relation": "stress worsens intrusive thoughts and urges",
                "compulsive_change": "increased",
            },
            {
                "source": "stress_triggered_exacerbation",
                "target": "compulsive_rituals",
                "relation": "stress worsens repetitive rituals",
                "compulsive_change": "increased",
            },
            {
                "source": "intrusive_obsessional_thoughts",
                "target": "distress_impairment",
                "relation": "intrusions increase distress and functional impairment",
                "compulsive_change": "increased",
            },
            {
                "source": "compulsive_rituals",
                "target": "distress_impairment",
                "relation": "time-consuming compulsions increase impairment",
                "compulsive_change": "increased",
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
            "orbitofrontal cortex",
            "anterior cingulate",
            "prefrontal cortex",
            "striatum",
            "basal ganglia",
            "thalamus",
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
        Useful for tuning striatal or thalamic candidates.
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
        node_keys: Sequence[str] = ("ofc", "acc", "pfc_control", "striatum_proxy", "thalamus_proxy"),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the CSTC circuit.
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
                    "description": "Atlas-backed compulsive-spectrum circuit node",
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
        early_onset_loading: float,
        chronic_stress: float,
        frontal_or_cstc_insult: float,
        serotonergic_treatment_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while serotonergic_treatment_support is protective.
        """
        g = self._clip01(genetic_vulnerability)
        e = self._clip01(early_onset_loading)
        s = self._clip01(chronic_stress)
        f = self._clip01(frontal_or_cstc_insult)
        t = self._clip01(serotonergic_treatment_support)

        # Latent biology
        serotonergic_dysregulation = self._clip01(
            0.30 * g + 0.20 * e + 0.15 * s - 0.30 * t
        )
        glutamate_habit_plasticity = self._clip01(
            0.25 * g + 0.25 * s + 0.10 * f - 0.10 * t
        )
        hpa_crf_stress_sensitization = self._clip01(
            0.45 * s + 0.10 * e - 0.10 * t
        )
        stress_habit_shift = self._clip01(
            0.35 * hpa_crf_stress_sensitization + 0.25 * glutamate_habit_plasticity
        )
        cstc_loop_hyperactivity = self._clip01(
            0.30 * glutamate_habit_plasticity
            + 0.20 * stress_habit_shift
            + 0.20 * f
            + 0.10 * g
            - 0.10 * t
        )
        top_down_control_failure = self._clip01(
            0.35 * f + 0.25 * cstc_loop_hyperactivity + 0.10 * s - 0.20 * t
        )
        error_signal_overdrive = self._clip01(
            0.35 * cstc_loop_hyperactivity + 0.20 * serotonergic_dysregulation
        )
        habit_dominance = self._clip01(
            0.40 * stress_habit_shift + 0.25 * cstc_loop_hyperactivity
        )
        compulsive_urge_amplification = self._clip01(
            0.35 * serotonergic_dysregulation
            + 0.20 * hpa_crf_stress_sensitization
            + 0.15 * error_signal_overdrive
            - 0.10 * t
        )

        # Regional state proxies
        ofc = self._clip01(
            0.35 * cstc_loop_hyperactivity + 0.20 * top_down_control_failure - 0.10 * t
        )
        acc = self._clip01(
            0.35 * cstc_loop_hyperactivity + 0.25 * error_signal_overdrive - 0.10 * t
        )
        pfc_control = self._clip01(
            0.40 * top_down_control_failure + 0.15 * f - 0.15 * t
        )
        striatum_proxy = self._clip01(
            0.40 * habit_dominance + 0.20 * cstc_loop_hyperactivity
        )
        thalamus_proxy = self._clip01(
            0.30 * cstc_loop_hyperactivity + 0.15 * error_signal_overdrive
        )

        # Symptoms
        intrusive_obsessional_thoughts = self._clip01(
            0.30 * serotonergic_dysregulation
            + 0.25 * error_signal_overdrive
            + 0.15 * ofc
            + 0.10 * thalamus_proxy
            - 0.15 * t
        )
        compulsive_rituals = self._clip01(
            0.30 * habit_dominance
            + 0.25 * compulsive_urge_amplification
            + 0.15 * striatum_proxy
            - 0.15 * t
        )
        urge_resistance_failure = self._clip01(
            0.35 * top_down_control_failure
            + 0.20 * compulsive_urge_amplification
            + 0.15 * pfc_control
            - 0.15 * t
        )
        stress_triggered_exacerbation = self._clip01(
            0.35 * hpa_crf_stress_sensitization + 0.25 * chronic_stress + 0.15 * compulsive_urge_amplification
        )
        cognitive_rigidity = self._clip01(
            0.30 * habit_dominance
            + 0.25 * top_down_control_failure
            + 0.15 * pfc_control
            - 0.10 * t
        )
        distress_impairment = self._clip01(
            0.30 * intrusive_obsessional_thoughts
            + 0.25 * compulsive_rituals
            + 0.20 * error_signal_overdrive
            - 0.10 * t
        )

        return {
            "inputs": pd.Series(
                {
                    "genetic_vulnerability": g,
                    "early_onset_loading": e,
                    "chronic_stress": s,
                    "frontal_or_cstc_insult": f,
                    "serotonergic_treatment_support": t,
                }
            ),
            "latents": pd.Series(
                {
                    "cstc_loop_hyperactivity": cstc_loop_hyperactivity,
                    "top_down_control_failure": top_down_control_failure,
                    "error_signal_overdrive": error_signal_overdrive,
                    "habit_dominance": habit_dominance,
                    "compulsive_urge_amplification": compulsive_urge_amplification,
                    "stress_habit_shift": stress_habit_shift,
                    "hpa_crf_stress_sensitization": hpa_crf_stress_sensitization,
                    "glutamate_habit_plasticity": glutamate_habit_plasticity,
                    "serotonergic_dysregulation": serotonergic_dysregulation,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "ofc": ofc,
                    "acc": acc,
                    "pfc_control": pfc_control,
                    "striatum_proxy": striatum_proxy,
                    "thalamus_proxy": thalamus_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "intrusive_obsessional_thoughts": intrusive_obsessional_thoughts,
                    "compulsive_rituals": compulsive_rituals,
                    "urge_resistance_failure": urge_resistance_failure,
                    "stress_triggered_exacerbation": stress_triggered_exacerbation,
                    "cognitive_rigidity": cognitive_rigidity,
                    "distress_impairment": distress_impairment,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "obsessional_serotonergic_profile": self._clip01(
                        0.45 * intrusive_obsessional_thoughts
                        + 0.25 * serotonergic_dysregulation
                        + 0.15 * error_signal_overdrive
                    ),
                    "habit_dominant_compulsive_profile": self._clip01(
                        0.45 * compulsive_rituals
                        + 0.25 * habit_dominance
                        + 0.15 * glutamate_habit_plasticity
                    ),
                    "stress_sensitive_compulsive_profile": self._clip01(
                        0.45 * stress_triggered_exacerbation
                        + 0.25 * hpa_crf_stress_sensitization
                        + 0.15 * compulsive_urge_amplification
                    ),
                    "lesion_disinhibition_profile": self._clip01(
                        0.40 * frontal_or_cstc_insult
                        + 0.30 * top_down_control_failure
                        + 0.20 * compulsive_rituals
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-12, 34, -10)).head(10)
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
    model = CompulsiveAndRelatedDisorderModel()

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
        bundle["edges"][
            ["source", "target", "relation", "compulsive_change"]
        ].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["ofc", "acc", "pfc_control", "striatum_proxy", "thalamus_proxy"]:
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
        early_onset_loading=0.60,
        chronic_stress=0.80,
        frontal_or_cstc_insult=0.35,
        serotonergic_treatment_support=0.20,
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
    # print(model.assign_mni_point((-12, 34, -10)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("thalamus").to_string(index=False))
    # print(model.suggest_regions("caudate").to_string(index=False))
