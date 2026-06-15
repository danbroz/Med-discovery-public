
from __future__ import annotations

"""
Non–Rapid Eye Movement Sleep Arousal Disorders siibra scaffold.

This script converts a biologically focused chapter on NREM sleep arousal
disorders into an atlas-grounded siibra research scaffold. It is designed for
transparent mechanistic exploration, not diagnosis, prognosis, or treatment
selection.

Modeling emphasis from the chapter:
- disorders of arousal from sleep rather than primary defects of sleep state,
- partial and incomplete awakening from slow-wave sleep,
- functional state dissociation with simultaneous local wake-like and sleep-like
  activity across different brain systems,
- probable GABAergic stabilization effects suggested by clonazepam response,
- serotonergic and noradrenergic modulation of sleep-wake boundary stability,
- possible dopaminergic modulation inferred from antipsychotic-triggered
  sleepwalking,
- activation of motor and thalamocortical circuits while frontoparietal and
  prefrontal awareness systems remain sleep-like,
- limbic and partially awakened sensory activity contributing to fearful or
  hallucinatory mentation,
- strong heritable predisposition interacting with environmental and
  physiological triggers.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Exploratory pathway-oriented panel. The chapter notes that direct disorder
# genes remain uncertain, so this list emphasizes GABAergic and monoaminergic
# mechanisms suggested by pharmacological response and sleep-wake modulation.
DEFAULT_GENE_PANEL = [
    "GABRA1",
    "GABRA2",
    "GABRG2",
    "SLC6A4",
    "HTR2A",
    "SLC6A2",
    "ADRA2A",
    "DRD2",
    "DBH",
]


class NonRapidEyeMovementSleepArousalDisordersModel:
    """
    Atlas-grounded scaffold for NREM sleep arousal disorders.

    The scaffold translates a chapter-level biological narrative into:
    - input nodes reflecting heritable risk, slow-wave instability, trigger
      burden, and destabilizing medication effects,
    - latent biology nodes capturing incomplete awakening, state dissociation,
      local motor release, awareness suppression, and amnestic failure,
    - atlas-backed regions or clearly labeled proxies,
    - symptom nodes and phenotype summaries.

    Important:
    This is a research scaffold for inspection and hypothesis generation. It is
    not a validated disease model and must not be used as a clinical tool.
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
        self.parcellation = (
            self.atlas.get_parcellation(parcellation_spec)
            if hasattr(self.atlas, "get_parcellation")
            else self.atlas.parcellations.get(parcellation_spec)
        )
        self.space_spec = space_spec
        self.space = (
            self.atlas.get_space(space_spec)
            if hasattr(self.atlas, "get_space")
            else self.atlas.spaces.get(space_spec)
        )
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        # Conservative atlas anchors and proxies derived from the chapter.
        self.region_candidates: Dict[str, List[str]] = {
            "motor_cortex": [
                "Area 4a left",
                "Area 4p left",
                "Area 4 left",
                "primary motor cortex left",
                "precentral left",
                "motor cortex",
            ],
            "thalamus_proxy": [
                "Thalamus left",
                "mediodorsal thalamus left",
                "pulvinar left",
                "thalamus",
            ],
            "basal_ganglia_locomotor_proxy": [
                "putamen left",
                "caudate left",
                "striatum left",
                "basal ganglia",
                "striatum",
            ],
            "pfc_control_proxy": [
                "Area 9/46d left",
                "Area 46 left",
                "Area 9 left",
                "dorsolateral prefrontal",
                "prefrontal cortex",
            ],
            "parietal_awareness_proxy": [
                "Area 7A left",
                "Area 7P left",
                "Area PFm left",
                "Area PF left",
                "superior parietal left",
                "inferior parietal left",
                "parietal cortex",
            ],
            "limbic_alarm_proxy": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "Amygdala left",
                "amygdala",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "motor_cortex": "Primary motor cortex anchor for wake-like motor output during parasomnia episodes.",
            "thalamus_proxy": "Proxy for thalamocortical activation involved in partial arousal and motor release.",
            "basal_ganglia_locomotor_proxy": "Proxy for subcortical locomotor and motor-patterning systems supporting ambulatory behavior.",
            "pfc_control_proxy": "Proxy for prefrontal executive and conscious-control systems that remain relatively sleep-like.",
            "parietal_awareness_proxy": "Proxy for frontoparietal awareness and associative circuitry needed for full conscious access and orienting.",
            "limbic_alarm_proxy": "Amygdalar / limbic proxy for fearful affect and emotionally charged mentation during partial arousal.",
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": "Heritable predisposition increasing susceptibility to NREM arousal parasomnias.",
            "sleep_fragmentation_burden": "Frequent sleep disruption that increases the chance of partial arousals from slow-wave sleep.",
            "slow_wave_sleep_pressure": "High slow-wave sleep load or pressure that can favor incomplete awakening from deep sleep.",
            "environmental_physiological_trigger_load": "Environmental or physiological precipitants that interact with predisposition to provoke episodes.",
            "monoaminergic_modulator_shift": "Serotonergic and noradrenergic tone shifts that destabilize sleep-wake boundaries.",
            "antidepressant_boundary_destabilization": "Destabilizing antidepressant effect in susceptible individuals; some agents can instead be therapeutic.",
            "dopamine_blockade_burden": "Possible destabilizing dopaminergic blockade burden inferred from antipsychotic-triggered sleepwalking reports.",
            "benzodiazepine_gaba_support": "Protective GABA-A mediated stabilization that can suppress incomplete arousals and raise arousal threshold.",
        }

        self.latent_nodes: Dict[str, str] = {
            "slow_wave_arousal_instability": "Instability of slow-wave sleep and arousal control that increases partial-awakening risk.",
            "monoaminergic_boundary_instability": "Serotonergic and noradrenergic dysmodulation of sleep-wake transition stability.",
            "incomplete_awakening_vulnerability": "Tendency to awaken incompletely rather than transitioning cleanly to wakefulness.",
            "state_dissociation": "Coexistence of wake-like and sleep-like local brain states during the same episode.",
            "motor_system_release": "Motor-circuit activation with preserved capacity for movement despite impaired conscious awareness.",
            "frontoparietal_awareness_suppression": "Persistence of sleep-like activity in awareness, executive-control, and associative networks.",
            "limbic_sensory_intrusion": "Partial activation of limbic and sensory systems generating fearful or hallucinatory mental content.",
            "amnestic_encoding_failure": "Failure of normal memory encoding during the episode, producing later amnesia.",
        }

        self.symptom_nodes: Dict[str, str] = {
            "confusional_arousal": "Disoriented and confused awakening with impaired responsiveness.",
            "complex_motor_behavior": "Complex motor or ambulatory behavior occurring during partial arousal.",
            "impaired_awareness": "Limited conscious awareness or reflective control during the event.",
            "episode_amnesia": "Reduced recall or complete amnesia for the episode.",
            "impaired_responsiveness": "Poor responsiveness to external cues or attempts to reorient the individual.",
            "fear_or_hallucinatory_experience": "Feelings of doom, fearful affect, or hallucinatory imagery during the event.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "slow_wave_arousal_instability",
                "relation": "predisposes the system to unstable arousal from deep sleep",
                "domain": "genetics",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "sleep_fragmentation_burden",
                "target": "slow_wave_arousal_instability",
                "relation": "raises the number of disruptive arousals from slow-wave sleep",
                "domain": "sleep",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "slow_wave_sleep_pressure",
                "target": "slow_wave_arousal_instability",
                "relation": "loads the deep-sleep state from which episodes commonly emerge",
                "domain": "sleep",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "environmental_physiological_trigger_load",
                "target": "slow_wave_arousal_instability",
                "relation": "interacts with predisposition to precipitate partial arousals",
                "domain": "triggers",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "monoaminergic_modulator_shift",
                "target": "monoaminergic_boundary_instability",
                "relation": "destabilizes boundaries between sleep and wakefulness",
                "domain": "monoamine",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "antidepressant_boundary_destabilization",
                "target": "monoaminergic_boundary_instability",
                "relation": "can increase fragmentation and partial arousal risk in susceptible individuals",
                "domain": "pharmacology",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "dopamine_blockade_burden",
                "target": "state_dissociation",
                "relation": "may contribute to abnormal arousal regulation and parasomnia expression",
                "domain": "pharmacology",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "benzodiazepine_gaba_support",
                "target": "slow_wave_arousal_instability",
                "relation": "suppresses destabilizing deep-sleep arousals and raises arousal threshold",
                "domain": "protective",
                "nrem_arousal_change": "decreased",
            },
            {
                "source": "benzodiazepine_gaba_support",
                "target": "incomplete_awakening_vulnerability",
                "relation": "reduces the chance of incomplete awakening",
                "domain": "protective",
                "nrem_arousal_change": "decreased",
            },
            {
                "source": "slow_wave_arousal_instability",
                "target": "incomplete_awakening_vulnerability",
                "relation": "increases the probability of partial awakening rather than full awakening",
                "domain": "sleep_wake_transition",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "monoaminergic_boundary_instability",
                "target": "incomplete_awakening_vulnerability",
                "relation": "promotes unstable transitions between deep sleep and wakefulness",
                "domain": "sleep_wake_transition",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "incomplete_awakening_vulnerability",
                "target": "state_dissociation",
                "relation": "produces simultaneous local wake-like and sleep-like brain states",
                "domain": "state_dissociation",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "monoaminergic_boundary_instability",
                "target": "state_dissociation",
                "relation": "further destabilizes the integrity of wake-sleep boundaries",
                "domain": "state_dissociation",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "state_dissociation",
                "target": "motor_system_release",
                "relation": "permits wake-like motor output while awareness remains impaired",
                "domain": "circuit",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "state_dissociation",
                "target": "frontoparietal_awareness_suppression",
                "relation": "keeps associative and executive networks in a sleep-like state",
                "domain": "circuit",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "state_dissociation",
                "target": "limbic_sensory_intrusion",
                "relation": "permits partial limbic and sensory activation without full top-down regulation",
                "domain": "circuit",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "frontoparietal_awareness_suppression",
                "target": "amnestic_encoding_failure",
                "relation": "prevents normal awareness-linked encoding of the episode",
                "domain": "memory",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "motor_system_release",
                "target": "motor_cortex",
                "relation": "activates motor output circuitry during the episode",
                "domain": "region",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "state_dissociation",
                "target": "thalamus_proxy",
                "relation": "loads thalamocortical arousal circuitry during partial awakening",
                "domain": "region",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "motor_system_release",
                "target": "basal_ganglia_locomotor_proxy",
                "relation": "recruits locomotor and motor-patterning subcortical systems",
                "domain": "region",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "frontoparietal_awareness_suppression",
                "target": "pfc_control_proxy",
                "relation": "keeps prefrontal control systems offline during the episode",
                "domain": "region",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "frontoparietal_awareness_suppression",
                "target": "parietal_awareness_proxy",
                "relation": "suppresses conscious orienting and associative awareness circuitry",
                "domain": "region",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "limbic_sensory_intrusion",
                "target": "limbic_alarm_proxy",
                "relation": "engages limbic alarm circuitry linked to fearful experience",
                "domain": "region",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "incomplete_awakening_vulnerability",
                "target": "confusional_arousal",
                "relation": "drives confused and disoriented partial awakening",
                "domain": "symptom",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "motor_system_release",
                "target": "complex_motor_behavior",
                "relation": "enables walking and other complex actions without full awareness",
                "domain": "symptom",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "frontoparietal_awareness_suppression",
                "target": "impaired_awareness",
                "relation": "reduces conscious awareness and reflective control",
                "domain": "symptom",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "amnestic_encoding_failure",
                "target": "episode_amnesia",
                "relation": "leads to absent or poor recall of the event",
                "domain": "symptom",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "frontoparietal_awareness_suppression",
                "target": "impaired_responsiveness",
                "relation": "reduces effective response to external attempts to reorient the sleeper",
                "domain": "symptom",
                "nrem_arousal_change": "increased",
            },
            {
                "source": "limbic_sensory_intrusion",
                "target": "fear_or_hallucinatory_experience",
                "relation": "produces doom-laden or hallucinatory mental content",
                "domain": "symptom",
                "nrem_arousal_change": "increased",
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

    def _parcellation_specs(self) -> List[str]:
        specs = [
            self.parcellation_spec,
            getattr(self.parcellation, "name", None),
            getattr(self.parcellation, "key", None),
        ]
        if "julich" in self.parcellation_spec.lower():
            specs.extend(["julich 3.1", "julich 3.0.3", "julich 2.9", "julich"])
        return [s for s in specs if isinstance(s, str) and s]

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
            cands.extend(["receptor density fingerprint", "ReceptorDensityFingerprint"])
        elif kind == "gene":
            cands.extend(["gene expressions", "GeneExpressions"])
        elif kind == "connectivity":
            cands.extend(["StreamlineCounts", "streamline counts"])
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
        out: List[Any] = []

        try:
            if hasattr(self.parcellation, "find"):
                found = self.parcellation.find(query)
                if found:
                    out.extend(list(found))
        except Exception:
            pass

        try:
            if hasattr(self.atlas, "find_regions"):
                found = self.atlas.find_regions(
                    query,
                    all_versions=False,
                    filter_children=False,
                    find_topmost=False,
                )
                if found:
                    out.extend(list(found))
        except Exception:
            pass

        dedup: List[Any] = []
        seen: set[str] = set()
        for region in out:
            parcellation_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" not in str(parcellation_name).lower() and out:
                continue
            ident = getattr(region, "identifier", None) or self._name_of(region)
            if str(ident) in seen:
                continue
            seen.add(str(ident))
            dedup.append(region)
        return dedup

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "thalamus",
            "basal ganglia",
            "prefrontal cortex",
            "parietal cortex",
            "amygdala",
            "motor cortex",
        } else 0
        cyto_bonus_penalty = 0 if (
            "area " in name
            or "amygdala" in name
            or "thalamus" in name
        ) else 1
        return (left_bonus, right_penalty, generic_penalty, cyto_bonus_penalty)

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
                if hasattr(self.parcellation, "find"):
                    matches = list(self.parcellation.find(spec))
                    if matches:
                        matches = sorted(matches, key=self._region_rank)
                        return matches[0]
            except Exception:
                pass
            matches = self._julich_matches(spec)
            if matches:
                matches = sorted(matches, key=self._region_rank)
                return matches[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
            row = (
                self._name_of(region),
                str(getattr(region, "identifier", "") or ""),
                getattr(getattr(region, "parcellation", None), "name", ""),
            )
            if row in seen:
                continue
            seen.add(row)
            rows.append({"name": row[0], "identifier": row[1], "parcellation": row[2]})
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
        try:
            centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        except Exception:
            centroid_xyz = None
        volume_mm3 = getattr(main, "volume", None)
        try:
            volume_value = float(volume_mm3) if volume_mm3 is not None else None
        except Exception:
            volume_value = None
        return centroid_xyz, volume_value

    @staticmethod
    def _to_dataframe(data: Any) -> pd.DataFrame:
        if data is None:
            return pd.DataFrame()
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, pd.Series):
            name = data.name if data.name is not None else "value"
            df = data.to_frame(name=name).reset_index()
            if len(df.columns) == 2:
                df.columns = ["index", name]
            return df
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        try:
            df = self._to_dataframe(getattr(feats[0], "data", None)).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        if "index" in df.columns and "receptor" not in df.columns:
            df = df.rename(columns={"index": "receptor"})
        return df

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()
        df = self._to_dataframe(getattr(feats[0], "data", None)).reset_index(drop=True)
        if df.empty:
            return df
        lower_cols = {str(c).lower(): c for c in df.columns}
        required = {"gene", "level", "zscore"}
        if required.issubset(lower_cols):
            gene_col = lower_cols["gene"]
            level_col = lower_cols["level"]
            zscore_col = lower_cols["zscore"]
            return (
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
                .reset_index(drop=True)
            )
        return df

    def _pick_connectivity_feature(self, features: Sequence[Any]) -> Optional[Any]:
        if not features:
            return None
        preferred = [
            f
            for f in features
            if str(getattr(f, "cohort", "") or "").lower() == self.connectivity_cohort.lower()
        ]
        return preferred[0] if preferred else features[0]

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        feature = self._pick_connectivity_feature(feats)
        if feature is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        data = getattr(feature, "data", None)
        df = self._to_dataframe(data)
        if isinstance(data, pd.DataFrame) and not data.empty:
            self._connectivity_matrix = data.copy()
            return self._connectivity_matrix
        if not df.empty and df.shape[0] > 1 and df.shape[1] > 1:
            self._connectivity_matrix = df.copy()
            return self._connectivity_matrix

        try:
            subfeature = feature[0]
            subdata = getattr(subfeature, "data", None)
            if isinstance(subdata, pd.DataFrame):
                self._connectivity_matrix = subdata.copy()
                return self._connectivity_matrix
            subdf = self._to_dataframe(subdata)
            self._connectivity_matrix = subdf.copy()
            return self._connectivity_matrix
        except Exception:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._name_of(region)
        exact = [x for x in labels if self._name_of(x) == region_name]
        if exact:
            return exact[0]
        rn = region_name.lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        return fuzzy[0] if fuzzy else None

    def _connectivity_profile_from_matrix(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        idx_match = self._match_region_label(list(matrix.index), region)
        col_match = self._match_region_label(list(matrix.columns), region)
        try:
            if idx_match is not None:
                series = matrix.loc[idx_match]
            elif col_match is not None:
                series = matrix[col_match]
            else:
                return pd.DataFrame()
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _connectivity_profile_from_region_feature(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("connectivity"))
        feature = self._pick_connectivity_feature(feats)
        if feature is None:
            return pd.DataFrame()
        data = getattr(feature, "data", None)
        if isinstance(data, pd.Series):
            df = data.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            return df.head(max_rows).reset_index(drop=True)
        if isinstance(data, pd.DataFrame):
            idx_match = self._match_region_label(list(data.index), region)
            col_match = self._match_region_label(list(data.columns), region)
            try:
                if idx_match is not None:
                    series = data.loc[idx_match]
                elif col_match is not None:
                    series = data[col_match]
                else:
                    return pd.DataFrame()
                df = series.sort_values(ascending=False).reset_index()
                df.columns = ["connected_region", "value"]
                df["connected_region"] = df["connected_region"].map(self._name_of)
                df = df[df["connected_region"] != region.name].head(max_rows)
                return df.reset_index(drop=True)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        df = self._connectivity_profile_from_matrix(region, max_rows=max_rows)
        if not df.empty:
            return df
        return self._connectivity_profile_from_region_feature(region, max_rows=max_rows)

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return long-form pairwise connectivity between resolved circuit nodes.

        The output is empty when the runtime lacks compatible connectivity
        features. This is normal and not an error.
        """
        rows: List[Dict[str, Any]] = []
        matrix = self._get_connectivity_matrix()

        if not matrix.empty and self.region_objects:
            idx_labels = list(matrix.index)
            col_labels = list(matrix.columns)
            mapped_labels = {
                key: (
                    self._match_region_label(idx_labels, region),
                    self._match_region_label(col_labels, region),
                )
                for key, region in self.region_objects.items()
            }
            for src_key, src_region in self.region_objects.items():
                idx_label, col_label = mapped_labels.get(src_key, (None, None))
                if idx_label is None and col_label is None:
                    continue
                for tgt_key, tgt_region in self.region_objects.items():
                    if src_key == tgt_key:
                        continue
                    tgt_idx_label, tgt_col_label = mapped_labels.get(tgt_key, (None, None))
                    value = None
                    try:
                        if idx_label is not None and tgt_col_label is not None:
                            value = matrix.loc[idx_label, tgt_col_label]
                        elif col_label is not None and tgt_idx_label is not None:
                            value = matrix.loc[tgt_idx_label, col_label]
                        elif idx_label is not None and tgt_idx_label is not None:
                            value = matrix.loc[idx_label, tgt_idx_label]
                        elif col_label is not None and tgt_col_label is not None:
                            value = matrix.loc[col_label, tgt_col_label]
                    except Exception:
                        value = None
                    try:
                        value = float(value) if value is not None else None
                    except Exception:
                        value = None
                    if value is None:
                        continue
                    rows.append(
                        {
                            "source_key": src_key,
                            "source_region": src_region.name,
                            "target_key": tgt_key,
                            "target_region": tgt_region.name,
                            "value": value,
                        }
                    )
            if rows:
                return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

        for src_key, profile in self.connectivity_profiles.items():
            if profile.empty:
                continue
            for tgt_key, tgt_region in self.region_objects.items():
                if src_key == tgt_key:
                    continue
                try:
                    mask = profile["connected_region"].astype(str).str.lower().eq(tgt_region.name.lower())
                    if not mask.any():
                        mask = profile["connected_region"].astype(str).str.lower().str.contains(
                            tgt_region.name.lower(),
                            regex=False,
                        )
                    if not mask.any():
                        continue
                    value = float(profile.loc[mask, "value"].iloc[0])
                except Exception:
                    continue
                rows.append(
                    {
                        "source_key": src_key,
                        "source_region": self.region_objects[src_key].name,
                        "target_key": tgt_key,
                        "target_region": tgt_region.name,
                        "value": value,
                    }
                )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(self, gene_panel: Sequence[str] = DEFAULT_GENE_PANEL, connectivity_rows: int = 15) -> dict:
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
                    "is_proxy": False,
                }
            )

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            is_proxy = key.endswith("_proxy")
            if region is None:
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region",
                        "description": self.region_node_descriptions.get(key, "Atlas-backed circuit node"),
                        "atlas_region": None,
                        "region_identifier": None,
                        "centroid_mni": None,
                        "volume_mm3": None,
                        "feature_summary": "unresolved",
                        "is_proxy": is_proxy,
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
                    "description": self.region_node_descriptions.get(key, "Atlas-backed circuit node"),
                    "atlas_region": region.name,
                    "region_identifier": getattr(region, "identifier", None),
                    "centroid_mni": centroid_mni,
                    "volume_mm3": volume_mm3,
                    "feature_summary": (
                        f"receptors={'yes' if not receptor_df.empty else 'no'}; "
                        f"genes={'yes' if not gene_df.empty else 'no'}; "
                        f"connectivity={'yes' if not conn_df.empty else 'no'}"
                    ),
                    "is_proxy": is_proxy,
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
                    "is_proxy": False,
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
                    "is_proxy": False,
                }
            )

        self.nodes_df = pd.DataFrame(nodes)
        self.edges_df = pd.DataFrame(self.edge_table)
        circuit_connectivity = self.circuit_connectivity()
        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": circuit_connectivity,
        }

    def _statistical_map(self):
        if self._pmap is not None:
            return self._pmap
        last_exc: Optional[Exception] = None
        for spec in self._parcellation_specs():
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
                return self._pmap
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError("Could not obtain a statistical parcellation map.") from last_exc

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        pmap = self._statistical_map()
        point = siibra.Point(tuple(xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = pmap.assign(point)
        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str, fetch: bool = False):
        """
        Return a region mask/volume for an atlas-backed node.

        Parameters
        ----------
        node_key:
            Region node key, for example "motor_cortex" or "pfc_control_proxy".
        fetch:
            If True, fetch and return the concrete image object when possible.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        mask = None
        try:
            mask = region.get_regional_mask(self.space, maptype="labelled")
        except Exception:
            try:
                mask = region.get_regional_map(self.space, maptype="statistical")
            except Exception:
                return None
        if fetch:
            try:
                return mask.fetch()
            except Exception:
                return mask
        return mask

    def simulate(
        self,
        *,
        genetic_vulnerability: float = 0.0,
        sleep_fragmentation_burden: float = 0.0,
        slow_wave_sleep_pressure: float = 0.0,
        environmental_physiological_trigger_load: float = 0.0,
        monoaminergic_modulator_shift: float = 0.0,
        antidepressant_boundary_destabilization: float = 0.0,
        dopamine_blockade_burden: float = 0.0,
        benzodiazepine_gaba_support: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Calculation order:
        inputs -> latent biology -> regional state -> symptoms -> phenotype summaries

        Values are clipped to [0, 1]. Protective terms subtract. The simulator
        is intentionally acyclic and interpretable rather than exhaustive.
        """
        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "sleep_fragmentation_burden": self._clip01(sleep_fragmentation_burden),
                "slow_wave_sleep_pressure": self._clip01(slow_wave_sleep_pressure),
                "environmental_physiological_trigger_load": self._clip01(environmental_physiological_trigger_load),
                "monoaminergic_modulator_shift": self._clip01(monoaminergic_modulator_shift),
                "antidepressant_boundary_destabilization": self._clip01(antidepressant_boundary_destabilization),
                "dopamine_blockade_burden": self._clip01(dopamine_blockade_burden),
                "benzodiazepine_gaba_support": self._clip01(benzodiazepine_gaba_support),
            },
            dtype=float,
        )

        latents = pd.Series(dtype=float)
        latents["slow_wave_arousal_instability"] = self._clip01(
            0.26 * inputs["genetic_vulnerability"]
            + 0.24 * inputs["sleep_fragmentation_burden"]
            + 0.20 * inputs["slow_wave_sleep_pressure"]
            + 0.18 * inputs["environmental_physiological_trigger_load"]
            + 0.10 * inputs["monoaminergic_modulator_shift"]
            - 0.28 * inputs["benzodiazepine_gaba_support"]
        )
        latents["monoaminergic_boundary_instability"] = self._clip01(
            0.44 * inputs["monoaminergic_modulator_shift"]
            + 0.28 * inputs["antidepressant_boundary_destabilization"]
            + 0.10 * inputs["sleep_fragmentation_burden"]
            + 0.08 * inputs["environmental_physiological_trigger_load"]
            - 0.08 * inputs["benzodiazepine_gaba_support"]
        )
        latents["incomplete_awakening_vulnerability"] = self._clip01(
            0.36 * latents["slow_wave_arousal_instability"]
            + 0.20 * latents["monoaminergic_boundary_instability"]
            + 0.16 * inputs["slow_wave_sleep_pressure"]
            + 0.12 * inputs["sleep_fragmentation_burden"]
            + 0.10 * inputs["environmental_physiological_trigger_load"]
            - 0.20 * inputs["benzodiazepine_gaba_support"]
        )
        latents["state_dissociation"] = self._clip01(
            0.38 * latents["incomplete_awakening_vulnerability"]
            + 0.24 * latents["slow_wave_arousal_instability"]
            + 0.18 * latents["monoaminergic_boundary_instability"]
            + 0.10 * inputs["dopamine_blockade_burden"]
            - 0.20 * inputs["benzodiazepine_gaba_support"]
        )
        latents["motor_system_release"] = self._clip01(
            0.42 * latents["state_dissociation"]
            + 0.24 * latents["incomplete_awakening_vulnerability"]
            + 0.12 * inputs["dopamine_blockade_burden"]
            + 0.10 * latents["slow_wave_arousal_instability"]
            - 0.10 * inputs["benzodiazepine_gaba_support"]
        )
        latents["frontoparietal_awareness_suppression"] = self._clip01(
            0.40 * latents["state_dissociation"]
            + 0.24 * inputs["slow_wave_sleep_pressure"]
            + 0.16 * latents["slow_wave_arousal_instability"]
            + 0.08 * inputs["sleep_fragmentation_burden"]
            - 0.12 * inputs["benzodiazepine_gaba_support"]
        )
        latents["limbic_sensory_intrusion"] = self._clip01(
            0.32 * latents["state_dissociation"]
            + 0.26 * latents["monoaminergic_boundary_instability"]
            + 0.16 * inputs["environmental_physiological_trigger_load"]
            + 0.10 * inputs["sleep_fragmentation_burden"]
            + 0.08 * inputs["dopamine_blockade_burden"]
        )
        latents["amnestic_encoding_failure"] = self._clip01(
            0.44 * latents["frontoparietal_awareness_suppression"]
            + 0.24 * latents["state_dissociation"]
            + 0.14 * latents["incomplete_awakening_vulnerability"]
            - 0.08 * inputs["benzodiazepine_gaba_support"]
        )

        regional_state = pd.Series(dtype=float)
        regional_state["motor_cortex"] = self._clip01(
            0.54 * latents["motor_system_release"]
            + 0.22 * latents["incomplete_awakening_vulnerability"]
            + 0.10 * latents["state_dissociation"]
        )
        regional_state["thalamus_proxy"] = self._clip01(
            0.42 * latents["state_dissociation"]
            + 0.24 * latents["slow_wave_arousal_instability"]
            + 0.16 * latents["incomplete_awakening_vulnerability"]
            + 0.08 * latents["monoaminergic_boundary_instability"]
        )
        regional_state["basal_ganglia_locomotor_proxy"] = self._clip01(
            0.42 * latents["motor_system_release"]
            + 0.26 * inputs["dopamine_blockade_burden"]
            + 0.16 * latents["state_dissociation"]
        )
        regional_state["pfc_control_proxy"] = self._clip01(
            0.54 * latents["frontoparietal_awareness_suppression"]
            + 0.18 * latents["amnestic_encoding_failure"]
            + 0.10 * latents["state_dissociation"]
        )
        regional_state["parietal_awareness_proxy"] = self._clip01(
            0.50 * latents["frontoparietal_awareness_suppression"]
            + 0.22 * latents["amnestic_encoding_failure"]
            + 0.12 * latents["state_dissociation"]
        )
        regional_state["limbic_alarm_proxy"] = self._clip01(
            0.50 * latents["limbic_sensory_intrusion"]
            + 0.22 * latents["state_dissociation"]
            + 0.10 * latents["monoaminergic_boundary_instability"]
        )

        symptoms = pd.Series(dtype=float)
        symptoms["confusional_arousal"] = self._clip01(
            0.34 * latents["incomplete_awakening_vulnerability"]
            + 0.24 * latents["frontoparietal_awareness_suppression"]
            + 0.12 * regional_state["parietal_awareness_proxy"]
            + 0.10 * regional_state["thalamus_proxy"]
            - 0.10 * inputs["benzodiazepine_gaba_support"]
        )
        symptoms["complex_motor_behavior"] = self._clip01(
            0.38 * latents["motor_system_release"]
            + 0.22 * regional_state["motor_cortex"]
            + 0.16 * regional_state["basal_ganglia_locomotor_proxy"]
            + 0.10 * regional_state["thalamus_proxy"]
        )
        symptoms["impaired_awareness"] = self._clip01(
            0.40 * latents["frontoparietal_awareness_suppression"]
            + 0.22 * latents["state_dissociation"]
            + 0.14 * regional_state["pfc_control_proxy"]
            + 0.10 * regional_state["parietal_awareness_proxy"]
        )
        symptoms["episode_amnesia"] = self._clip01(
            0.44 * latents["amnestic_encoding_failure"]
            + 0.18 * symptoms["impaired_awareness"]
            + 0.14 * latents["frontoparietal_awareness_suppression"]
            + 0.08 * regional_state["pfc_control_proxy"]
        )
        symptoms["impaired_responsiveness"] = self._clip01(
            0.32 * latents["frontoparietal_awareness_suppression"]
            + 0.20 * symptoms["confusional_arousal"]
            + 0.18 * regional_state["thalamus_proxy"]
            + 0.10 * regional_state["pfc_control_proxy"]
        )
        symptoms["fear_or_hallucinatory_experience"] = self._clip01(
            0.38 * latents["limbic_sensory_intrusion"]
            + 0.20 * regional_state["limbic_alarm_proxy"]
            + 0.16 * latents["state_dissociation"]
            + 0.10 * symptoms["impaired_awareness"]
        )

        phenotypes = pd.Series(dtype=float)
        phenotypes["overall_nrem_arousal_disorder_severity"] = self._clip01(
            0.20 * symptoms["confusional_arousal"]
            + 0.20 * symptoms["complex_motor_behavior"]
            + 0.18 * symptoms["impaired_awareness"]
            + 0.18 * symptoms["episode_amnesia"]
            + 0.12 * symptoms["impaired_responsiveness"]
            + 0.12 * symptoms["fear_or_hallucinatory_experience"]
        )
        phenotypes["sleepwalking_ambulatory_profile"] = self._clip01(
            0.34 * symptoms["complex_motor_behavior"]
            + 0.24 * symptoms["episode_amnesia"]
            + 0.20 * symptoms["impaired_awareness"]
            + 0.12 * regional_state["motor_cortex"]
            + 0.10 * regional_state["basal_ganglia_locomotor_proxy"]
        )
        phenotypes["confusional_arousal_profile"] = self._clip01(
            0.36 * symptoms["confusional_arousal"]
            + 0.26 * symptoms["impaired_responsiveness"]
            + 0.20 * symptoms["impaired_awareness"]
            + 0.10 * regional_state["parietal_awareness_proxy"]
            + 0.08 * regional_state["thalamus_proxy"]
        )
        phenotypes["fearful_limbic_parasomnia_profile"] = self._clip01(
            0.34 * symptoms["fear_or_hallucinatory_experience"]
            + 0.22 * symptoms["impaired_awareness"]
            + 0.18 * symptoms["confusional_arousal"]
            + 0.16 * regional_state["limbic_alarm_proxy"]
            + 0.10 * latents["limbic_sensory_intrusion"]
        )
        phenotypes["state_dissociation_profile"] = self._clip01(
            0.34 * latents["state_dissociation"]
            + 0.24 * latents["frontoparietal_awareness_suppression"]
            + 0.20 * latents["motor_system_release"]
            + 0.12 * latents["amnestic_encoding_failure"]
            + 0.10 * regional_state["thalamus_proxy"]
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }


if __name__ == "__main__":
    model = NonRapidEyeMovementSleepArousalDisordersModel()
    scaffold = model.build()

    print("\n=== NODE SUMMARY ===")
    print(
        scaffold["nodes"][
            [
                "key",
                "node_type",
                "atlas_region",
                "feature_summary",
                "is_proxy",
            ]
        ].to_string(index=False)
    )

    print("\n=== EDGE SUMMARY ===")
    print(
        scaffold["edges"][
            ["source", "target", "relation", "nrem_arousal_change"]
        ].head(20).to_string(index=False)
    )

    print("\n=== REGION FEATURE SNAPSHOT ===")
    for key in ("motor_cortex", "pfc_control_proxy", "limbic_alarm_proxy"):
        print(f"\n[{key}]")
        region_rows = scaffold["nodes"].query("key == @key")[["atlas_region", "centroid_mni", "feature_summary"]]
        print(region_rows.to_string(index=False))
        if not scaffold["receptors"].get(key, pd.DataFrame()).empty:
            print("receptors:")
            print(scaffold["receptors"][key].head().to_string(index=False))
        if not scaffold["genes"].get(key, pd.DataFrame()).empty:
            print("genes:")
            print(scaffold["genes"][key].head().to_string(index=False))
        if not scaffold["connectivity_profiles"].get(key, pd.DataFrame()).empty:
            print("connectivity:")
            print(scaffold["connectivity_profiles"][key].head().to_string(index=False))

    print("\n=== CIRCUIT CONNECTIVITY ===")
    if scaffold["circuit_connectivity"].empty:
        print("No pairwise circuit connectivity available in this runtime.")
    else:
        print(scaffold["circuit_connectivity"].head(15).to_string(index=False))

    example = model.simulate(
        genetic_vulnerability=0.60,
        sleep_fragmentation_burden=0.70,
        slow_wave_sleep_pressure=0.65,
        environmental_physiological_trigger_load=0.45,
        monoaminergic_modulator_shift=0.35,
        antidepressant_boundary_destabilization=0.20,
        dopamine_blockade_burden=0.15,
        benzodiazepine_gaba_support=0.35,
    )

    print("\n=== SIMULATED SYMPTOMS ===")
    print(example["symptoms"].sort_values(ascending=False).to_string())

    print("\n=== SIMULATED PHENOTYPES ===")
    print(example["phenotypes"].sort_values(ascending=False).to_string())

    # Optional manual testing examples:
    # assignments = model.assign_mni_point((-38.0, -22.0, 58.0))
    # print(assignments.head().to_string(index=False))
    # mask_img = model.region_mask("motor_cortex", fetch=True)
