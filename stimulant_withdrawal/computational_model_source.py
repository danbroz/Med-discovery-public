from __future__ import annotations

"""
Stimulant Withdrawal siibra scaffold.

This script turns a chapter-level biological summary of stimulant withdrawal
into an atlas-grounded mechanistic scaffold using siibra. It is intended for
research exploration only. It is not a diagnostic, prognostic, or treatment
system.

Conservative design choices:
- named structures from the chapter are anchored when possible
  (amygdala, hippocampus, orbitofrontal cortex, anterior cingulate,
  insula, and broader prefrontal control systems),
- reward circuitry is represented with a clearly labeled
  ``ventral_striatum_proxy`` because the chapter is systems-level there,
- transmitter biology such as dopamine, serotonin, and norepinephrine is kept
  primarily as latent biology unless the chapter localizes it,
- the simulator follows one transparent direction:
  inputs -> latent biology -> regional state -> symptoms -> phenotype summaries.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import siibra
except ImportError as exc:  # pragma: no cover - import guard for portability
    raise ImportError(
        "This scaffold requires the 'siibra' package. Install siibra in your "
        "Python environment before running this script."
    ) from exc


STIMULANT_WITHDRAWAL_GENE_PANEL = [
    # Dopamine / reward and salience
    "DRD2",
    "DRD3",
    "DRD4",
    "SLC6A3",
    "TH",
    "COMT",
    # Serotonin / mood and cognitive symptoms
    "SLC6A4",
    "HTR1A",
    "HTR2A",
    "TPH2",
    # Norepinephrine / arousal and stress
    "SLC6A2",
    "ADRA2A",
    "DBH",
    # Stress, plasticity, and epigenetic vulnerability
    "NR3C1",
    "FKBP5",
    "CRHR1",
    "BDNF",
    "CREB1",
]

DEFAULT_GENE_PANEL = STIMULANT_WITHDRAWAL_GENE_PANEL


class StimulantWithdrawalModel:
    """
    Atlas-grounded research scaffold for stimulant withdrawal.

    Higher values in the simulation generally indicate greater dysregulation or
    symptom burden, except the protective ``recovery_support`` input where
    higher values indicate stronger stabilizing support.
    """

    disorder_name = "Stimulant Withdrawal"
    abbreviation = "stimulant_withdrawal"

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

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": (
                "Inherited liability affecting addiction transition, monoamine "
                "signaling, stress reactivity, and withdrawal severity."
            ),
            "chronic_stimulant_exposure": (
                "Cumulative stimulant exposure contributing to compulsive use, "
                "allostatic burden, and neuroplastic adaptation."
            ),
            "acute_abstinence_shift": (
                "Recent cessation or sharp reduction that unmasks the withdrawal state."
            ),
            "serotonergic_neurotoxicity_load": (
                "Cumulative serotonin-heavy toxicity burden relevant to mood and "
                "cognitive problems, especially in MDMA-weighted exposure patterns."
            ),
            "stress_load": (
                "Concurrent stress burden that amplifies negative affect, cue reactivity, "
                "and relapse risk."
            ),
            "medication_cue_exposure": (
                "Exposure to medication-associated cues that reactivate reward-memory circuitry."
            ),
            "recovery_support": (
                "Protective treatment structure, abstinence support, and restorative recovery context."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "dopaminergic_reward_crash": (
                "Withdrawal-related reward deficit and blunted incentive tone following stimulant cessation."
            ),
            "serotonergic_noradrenergic_depletion": (
                "Reduced serotonin and norepinephrine signaling contributing to depressive, anxious, and fatigue symptoms."
            ),
            "stress_epigenetic_sensitization": (
                "Stress- and medication-linked enduring biological sensitization consistent with epigenetic scarring."
            ),
            "amygdala_threat_reactivity": (
                "Limbic hyper-reactivity that biases abstinent states toward dysphoria, anxiety, and defensiveness."
            ),
            "hippocampal_craving_memory_burden": (
                "Memory-linked cue associations that sustain craving during abstinence."
            ),
            "cue_reactivity_relapse_circuit": (
                "Cue-triggered reactivation of reward and memory circuitry that drives medication seeking."
            ),
            "frontolimbic_control_failure": (
                "Reduced executive control and top-down inhibition during withdrawal and abstinence."
            ),
            "allostatic_negative_affect_state": (
                "Withdrawal-specific negative emotional state consistent with hyperkatifeia and negative reinforcement."
            ),
            "neuroplastic_addiction_entrenchment": (
                "Long-lasting circuit adaptation that stabilizes compulsive use and slows recovery."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "depressed_mood": "Depressed mood emerging during stimulant withdrawal.",
            "anhedonia": "Reduced capacity to experience reward or pleasure.",
            "fatigue_psychomotor_slowing": (
                "Low energy, slowed drive, and psychomotor fatigue."
            ),
            "sleep_disturbance": (
                "Sleep disruption or poor restorative sleep during withdrawal."
            ),
            "anxiety_irritability": (
                "Anxiety, irritability, and dysphoric tension during abstinence."
            ),
            "craving": "Urges to resume stimulant use driven by cues and negative affect.",
            "executive_dysfunction": (
                "Poor cognitive control, decision-making, and inhibitory regulation."
            ),
            "relapse_vulnerability": (
                "High vulnerability to renewed stimulant use under stress, cues, and negative affect."
            ),
        }

        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 left",
                "CA3 left",
                "DG left",
                "Subiculum left",
                "hippocampus left",
                "hippocampus",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Fo4",
                "Fo3",
                "orbitofrontal",
            ],
            "acc": [
                "Area p24pr left",
                "Area p24 left",
                "Area 24dv (ACC) left",
                "Area 24d left",
                "anterior cingulate",
                "cingulate",
            ],
            "insula": [
                "Area Id1 left",
                "Area Id2 left",
                "Area Ig2 left",
                "insula left",
                "insula",
            ],
            "pfc_control": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 9/46v left",
                "dorsolateral prefrontal",
                "prefrontal cortex",
            ],
            "ventral_striatum_proxy": [
                "nucleus accumbens left",
                "accumbens",
                "ventral striatum",
                "striatum",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "amygdala": "Atlas-backed limbic reactivity node for dysphoria, cue salience, and threat sensitivity.",
            "hippocampus": "Atlas-backed memory-context node for cue associations and relapse-linked learning.",
            "ofc": "Atlas-backed orbitofrontal decision and valuation node named in the chapter's structural findings.",
            "acc": "Atlas-backed anterior cingulate control-conflict node named in the chapter's structural findings.",
            "insula": "Atlas-backed interoceptive and affective burden node named in the chapter's structural findings.",
            "pfc_control": "Atlas-backed or proxy prefrontal executive-control node for top-down inhibition.",
            "ventral_striatum_proxy": "Proxy reward-salience node representing withdrawal-related incentive and reward deficit circuitry.",
        }

        self.edge_table: List[Dict[str, Any]] = [
            {
                "source": "genetic_vulnerability",
                "target": "stress_epigenetic_sensitization",
                "relation": "increases susceptibility to lasting stress- and medication-linked biological sensitization",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.20,
            },
            {
                "source": "genetic_vulnerability",
                "target": "dopaminergic_reward_crash",
                "relation": "modulates severity of reward-system collapse during withdrawal",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.18,
            },
            {
                "source": "chronic_stimulant_exposure",
                "target": "dopaminergic_reward_crash",
                "relation": "drives post-use reward deficit and anhedonic withdrawal tone",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "chronic_stimulant_exposure",
                "target": "serotonergic_noradrenergic_depletion",
                "relation": "contributes to monoamine depletion underlying depressive and fatigue symptoms",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "chronic_stimulant_exposure",
                "target": "neuroplastic_addiction_entrenchment",
                "relation": "induces lasting molecular and circuit-level adaptation",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.42,
            },
            {
                "source": "acute_abstinence_shift",
                "target": "dopaminergic_reward_crash",
                "relation": "unmasks withdrawal-related reward collapse after cessation",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "acute_abstinence_shift",
                "target": "allostatic_negative_affect_state",
                "relation": "pushes the post-use state into negative reinforcement mode",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.28,
            },
            {
                "source": "serotonergic_neurotoxicity_load",
                "target": "serotonergic_noradrenergic_depletion",
                "relation": "intensifies serotonin-linked mood and cognitive burden",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "stress_load",
                "target": "stress_epigenetic_sensitization",
                "relation": "converges with medication exposure to create durable vulnerability",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "stress_load",
                "target": "allostatic_negative_affect_state",
                "relation": "amplifies withdrawal dysphoria and negative reinforcement pressure",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "medication_cue_exposure",
                "target": "hippocampal_craving_memory_burden",
                "relation": "reactivates medication-use memories during abstinence",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.30,
            },
            {
                "source": "medication_cue_exposure",
                "target": "cue_reactivity_relapse_circuit",
                "relation": "triggers cue-reactive relapse pressure",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "recovery_support",
                "target": "frontolimbic_control_failure",
                "relation": "buffers executive-control failure during abstinence",
                "stimulant_withdrawal_change": "decreased",
                "weight": -0.25,
            },
            {
                "source": "recovery_support",
                "target": "allostatic_negative_affect_state",
                "relation": "partly counters withdrawal dysphoria",
                "stimulant_withdrawal_change": "decreased",
                "weight": -0.20,
            },
            {
                "source": "stress_epigenetic_sensitization",
                "target": "amygdala_threat_reactivity",
                "relation": "sensitizes limbic circuits involved in negative affect and stress reactivity",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "stress_epigenetic_sensitization",
                "target": "frontolimbic_control_failure",
                "relation": "weakens cortical regulation under chronic medication-stress burden",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.25,
            },
            {
                "source": "dopaminergic_reward_crash",
                "target": "ventral_striatum_proxy",
                "relation": "maps to reward-circuit deficit during withdrawal",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.60,
            },
            {
                "source": "dopaminergic_reward_crash",
                "target": "anhedonia",
                "relation": "reduces reward responsiveness and pleasure",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.55,
            },
            {
                "source": "serotonergic_noradrenergic_depletion",
                "target": "depressed_mood",
                "relation": "contributes to depressive overlap during withdrawal",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "serotonergic_noradrenergic_depletion",
                "target": "fatigue_psychomotor_slowing",
                "relation": "contributes to fatigue and psychomotor slowing",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.35,
            },
            {
                "source": "amygdala_threat_reactivity",
                "target": "amygdala",
                "relation": "maps to limbic hyper-reactivity during abstinence",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.65,
            },
            {
                "source": "hippocampal_craving_memory_burden",
                "target": "hippocampus",
                "relation": "maps to memory-circuit reactivation by medication cues",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.65,
            },
            {
                "source": "cue_reactivity_relapse_circuit",
                "target": "craving",
                "relation": "drives cue-induced urge to resume use",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.55,
            },
            {
                "source": "frontolimbic_control_failure",
                "target": "pfc_control",
                "relation": "maps to prefrontal hypoactivation and impaired top-down control",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.60,
            },
            {
                "source": "frontolimbic_control_failure",
                "target": "ofc",
                "relation": "maps to orbitofrontal burden consistent with structural findings",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "frontolimbic_control_failure",
                "target": "acc",
                "relation": "maps to anterior cingulate burden consistent with structural findings",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "allostatic_negative_affect_state",
                "target": "insula",
                "relation": "maps to interoceptive and dysphoric burden during withdrawal",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.45,
            },
            {
                "source": "allostatic_negative_affect_state",
                "target": "anxiety_irritability",
                "relation": "translates dysphoria into anxious and irritable affect",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.40,
            },
            {
                "source": "pfc_control",
                "target": "executive_dysfunction",
                "relation": "permits impaired planning, control, and inhibition",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.50,
            },
            {
                "source": "craving",
                "target": "relapse_vulnerability",
                "relation": "raises the probability of relapse under abstinence pressure",
                "stimulant_withdrawal_change": "increased",
                "weight": 0.45,
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame(self.edge_table)
        self._pmap: Any = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _clean_region_name(name: str) -> str:
        return (
            str(name)
            .lower()
            .replace("area ", "")
            .replace(" (gapmap)", "")
            .replace(" left", "")
            .replace(" right", "")
            .replace(" cortex", "")
            .replace("region", "")
            .strip()
        )

    @staticmethod
    def _scalar_from_entry(value: Any) -> float:
        if value is None:
            return float("nan")
        if isinstance(value, (int, float, np.integer, np.floating)):
            try:
                return float(value)
            except Exception:
                return float("nan")
        if isinstance(value, pd.Series):
            arr = pd.to_numeric(value, errors="coerce").to_numpy(dtype=float)
            return float(np.nanmean(arr)) if arr.size else float("nan")
        if isinstance(value, pd.DataFrame):
            arr = value.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            return float(np.nanmean(arr)) if arr.size else float("nan")
        try:
            return float(value)
        except Exception:
            return float("nan")

    def _modality_candidates(self, kind: str) -> List[Any]:
        candidates: List[Any] = []
        try:
            if kind == "receptor":
                candidates.append(siibra.features.molecular.ReceptorDensityFingerprint)
            elif kind == "gene":
                candidates.append(siibra.features.molecular.GeneExpressions)
            elif kind == "connectivity":
                candidates.append(siibra.features.connectivity.StreamlineCounts)
        except Exception:
            pass

        if kind == "receptor":
            candidates.extend([
                "receptor density fingerprint",
                "receptor density profile",
            ])
        elif kind == "gene":
            candidates.append("gene expressions")
        elif kind == "connectivity":
            candidates.append("StreamlineCounts")
        return candidates

    def _safe_features_any(
        self,
        concept: Any,
        modalities: Sequence[Any],
        **kwargs: Any,
    ) -> List[Any]:
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
        attempts: List[Tuple[Any, str, bool]] = [
            (self.parcellation, "find", True),
            (self.parcellation, "find_regions", True),
            (self.atlas, "find_regions", False),
        ]
        matches: List[Any] = []
        from_julich_source = False
        for obj, method_name, is_julich_source in attempts:
            method = getattr(obj, method_name, None)
            if method is None:
                continue
            try:
                result = method(
                    query,
                    all_versions=False,
                    filter_children=False,
                    find_topmost=False,
                )
            except TypeError:
                try:
                    result = method(query, filter_children=False, find_topmost=False)
                except TypeError:
                    try:
                        result = method(query)
                    except Exception:
                        continue
            except Exception:
                continue
            if result:
                matches = list(result)
                from_julich_source = is_julich_source
                break

        out: List[Any] = []
        seen = set()
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if not from_julich_source and "julich" not in str(parc_name).lower():
                continue
            identifier = getattr(region, "identifier", None)
            key = identifier or self._name_of(region)
            if key in seen:
                continue
            seen.add(key)
            out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "insula",
            "orbitofrontal cortex",
            "anterior cingulate cortex",
            "prefrontal cortex",
            "striatum",
        } else 0
        gapmap_penalty = 1 if "gapmap" in name else 0
        return (left_bonus, right_penalty, generic_penalty, gapmap_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            for getter in (
                lambda s: self.atlas.get_region(s, parcellation=self.parcellation),
                lambda s: self.parcellation.get_region(s),
                lambda s: siibra.get_region(self.parcellation_spec, s),
            ):
                try:
                    return getter(spec)
                except Exception:
                    pass
            matches = self._julich_matches(spec)
            if matches:
                return sorted(matches, key=self._region_rank)[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        seen = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
            record = (
                self._name_of(region),
                getattr(region, "identifier", None),
                getattr(getattr(region, "parcellation", None), "name", ""),
            )
            if record in seen:
                continue
            seen.add(record)
            rows.append(
                {
                    "name": record[0],
                    "identifier": record[1],
                    "parcellation": record[2],
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
        components = getattr(props, "components", None)
        if components is not None:
            return list(components)
        return [props]

    def _main_component(
        self,
        region: Any,
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda item: float(getattr(item, "volume", 0.0) or 0.0))
        centroid = getattr(main, "centroid", None)
        centroid_xyz: Optional[Tuple[float, float, float]] = None
        if centroid is not None:
            try:
                centroid_xyz = tuple(float(v) for v in centroid)
            except Exception:
                coordinate = getattr(centroid, "coordinate", None)
                if coordinate is not None:
                    centroid_xyz = tuple(float(v) for v in coordinate)
        volume_mm3: Optional[float]
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
        except Exception:
            return pd.DataFrame()

        lower_cols = {str(c).lower(): c for c in df.columns}
        if "index" in lower_cols and "receptor" not in lower_cols:
            df = df.rename(columns={lower_cols["index"]: "receptor"})
        elif df.columns.tolist():
            first_col = df.columns[0]
            if str(first_col).lower() not in {"receptor", "name"}:
                df = df.rename(columns={first_col: "receptor"})
        return df.reset_index(drop=True)

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(
            region,
            self._modality_candidates("gene"),
            gene=list(genes),
        )
        if not feats:
            return pd.DataFrame()
        try:
            df = feats[0].data.copy()
        except Exception:
            return pd.DataFrame()

        lower_cols = {str(c).lower(): c for c in df.columns}
        gene_col = lower_cols.get("gene")
        level_col = lower_cols.get("level") or lower_cols.get("expression level")
        zscore_col = lower_cols.get("zscore") or lower_cols.get("z-score")
        if gene_col and level_col:
            agg_spec: Dict[str, Tuple[str, str]] = {
                "level_mean": (level_col, "mean"),
                "level_std": (level_col, "std"),
                "probe_count": (level_col, "count"),
            }
            if zscore_col:
                agg_spec["zscore_mean"] = (zscore_col, "mean")
                agg_spec["zscore_std"] = (zscore_col, "std")
            grouped = (
                df.groupby(gene_col, dropna=False)
                .agg(**agg_spec)
                .reset_index()
                .rename(columns={gene_col: "gene"})
            )
            grouped["gene"] = grouped["gene"].astype(str).str.upper()
            return grouped.sort_values("gene").reset_index(drop=True)

        return df.reset_index(drop=True)

    def _normalize_connectivity_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        if matrix.empty:
            return matrix

        out = matrix.copy()
        out.index = [self._name_of(idx) for idx in out.index]
        out.columns = [self._name_of(col) for col in out.columns]

        if out.index.has_duplicates:
            out = out.groupby(level=0).mean(numeric_only=False)
        if out.columns.has_duplicates:
            out = out.T.groupby(level=0).mean(numeric_only=False).T

        out = out.applymap(self._scalar_from_entry)
        out = out.apply(pd.to_numeric, errors="coerce")
        return out

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(
            self.parcellation,
            self._modality_candidates("connectivity"),
        )
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        chosen = next(
            (
                f
                for f in feats
                if self.connectivity_cohort.lower() in str(getattr(f, "cohort", "")).lower()
            ),
            feats[0],
        )

        candidate_data = getattr(chosen, "data", None)
        if isinstance(candidate_data, pd.DataFrame):
            self._connectivity_matrix = self._normalize_connectivity_matrix(candidate_data)
            return self._connectivity_matrix

        if hasattr(chosen, "__getitem__"):
            try:
                element = chosen[0]
                candidate_data = getattr(element, "data", None)
                if isinstance(candidate_data, pd.DataFrame):
                    self._connectivity_matrix = self._normalize_connectivity_matrix(candidate_data)
                    return self._connectivity_matrix
            except Exception:
                pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[str]:
        labels_str = [self._name_of(lbl) for lbl in labels]
        region_name = self._name_of(region)
        region_name_l = region_name.lower()
        cleaned_region = self._clean_region_name(region_name)

        exact = [item for item in labels_str if item == region_name]
        if exact:
            return exact[0]

        exact_lower = [item for item in labels_str if item.lower() == region_name_l]
        if exact_lower:
            return exact_lower[0]

        fuzzy = [
            item
            for item in labels_str
            if region_name_l in item.lower() or item.lower() in region_name_l
        ]
        if fuzzy:
            return fuzzy[0]

        loose = [
            item
            for item in labels_str
            if cleaned_region
            and (
                cleaned_region in self._clean_region_name(item)
                or self._clean_region_name(item) in cleaned_region
            )
        ]
        return loose[0] if loose else None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        row_label = self._match_region_label(list(matrix.index), region)
        col_label = self._match_region_label(list(matrix.columns), region)

        series: Optional[pd.Series] = None
        if row_label is not None and row_label in matrix.index:
            selected = matrix.loc[row_label]
            if isinstance(selected, pd.DataFrame):
                series = selected.mean(axis=0, numeric_only=True)
            else:
                series = pd.to_numeric(selected, errors="coerce")
        if series is None and col_label is not None and col_label in matrix.columns:
            selected = matrix[col_label]
            if isinstance(selected, pd.DataFrame):
                series = selected.mean(axis=1, numeric_only=True)
            else:
                series = pd.to_numeric(selected, errors="coerce")
        if series is None:
            return pd.DataFrame()

        try:
            df = (
                series.sort_values(ascending=False)
                .reset_index()
                .rename(columns={"index": "connected_region", 0: "value"})
            )
        except Exception:
            return pd.DataFrame()

        if df.shape[1] >= 2:
            df.columns = ["connected_region", "value"]
        df["connected_region"] = df["connected_region"].map(self._name_of)
        df = df[df["connected_region"] != region.name]
        df = df.dropna(subset=["value"])
        return df.head(max_rows).reset_index(drop=True)

    def circuit_connectivity(self) -> pd.DataFrame:
        """Return a pairwise connectivity table among resolved circuit nodes."""
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        region_items = list(self.region_objects.items())
        for src_key, src_region in region_items:
            for dst_key, dst_region in region_items:
                if src_key == dst_key:
                    continue
                src_label = self._match_region_label(list(matrix.index), src_region)
                dst_label = self._match_region_label(list(matrix.columns), dst_region)
                if src_label is None or dst_label is None:
                    continue
                if src_label not in matrix.index or dst_label not in matrix.columns:
                    continue
                try:
                    value = self._scalar_from_entry(matrix.loc[src_label, dst_label])
                except Exception:
                    continue
                if np.isnan(value):
                    continue
                rows.append(
                    {
                        "source_key": src_key,
                        "source_region": src_region.name,
                        "target_key": dst_key,
                        "target_region": dst_region.name,
                        "value": float(value),
                    }
                )
        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows)
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        out = out.dropna(subset=["value"])
        return out.sort_values(["source_key", "value"], ascending=[True, False]).reset_index(
            drop=True
        )

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = []
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
        self._connectivity_matrix = None

        for key, description in self.input_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "input",
                    "description": description,
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
                warnings.warn(
                    f"Could not resolve a Julich region for '{key}'. Keeping it as a proxy node."
                )
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region_proxy",
                        "description": self.region_descriptions.get(
                            key,
                            "Proxy region node unresolved in this siibra environment.",
                        ),
                        "atlas_region": None,
                        "region_identifier": None,
                        "centroid_mni": None,
                        "volume_mm3": None,
                        "feature_summary": "unresolved proxy",
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
                    "description": self.region_descriptions.get(key, "Atlas-backed circuit node."),
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

        for key, description in self.latent_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "latent_biology",
                    "description": description,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, description in self.symptom_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "symptom",
                    "description": description,
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

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if self._pmap is None:
            with siibra.QUIET:
                self._pmap = siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=self.assignment_space,
                    maptype="statistical",
                )
        point = siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)
        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(
        self,
        node_key: str,
        space: Optional[str] = None,
        maptype: str = "labelled",
        fetch: bool = True,
    ) -> Any:
        region = self.region_objects.get(node_key)
        if region is None:
            candidates = self.region_candidates.get(node_key)
            if candidates:
                region = self._resolve_region(candidates)
                if region is not None:
                    self.region_objects[node_key] = region
        if region is None:
            return None

        target_space = space or self.assignment_space

        try:
            mask = region.get_regional_mask(space=target_space, maptype=maptype)
            if fetch and hasattr(mask, "fetch"):
                return mask.fetch()
            return mask
        except Exception:
            pass

        try:
            return region.fetch_regional_map(space=target_space, maptype=maptype)
        except Exception:
            pass

        try:
            mask = region.get_regional_map(target_space, maptype)
            if fetch and hasattr(mask, "fetch"):
                return mask.fetch()
            return mask
        except Exception:
            return None

    def simulate(
        self,
        genetic_vulnerability: float = 0.45,
        chronic_stimulant_exposure: float = 0.70,
        acute_abstinence_shift: float = 0.65,
        serotonergic_neurotoxicity_load: float = 0.40,
        stress_load: float = 0.55,
        medication_cue_exposure: float = 0.50,
        recovery_support: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """Run a transparent normalized simulation for the stimulant-withdrawal scaffold."""
        inputs = pd.Series(
            {
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "chronic_stimulant_exposure": self._clip01(chronic_stimulant_exposure),
                "acute_abstinence_shift": self._clip01(acute_abstinence_shift),
                "serotonergic_neurotoxicity_load": self._clip01(serotonergic_neurotoxicity_load),
                "stress_load": self._clip01(stress_load),
                "medication_cue_exposure": self._clip01(medication_cue_exposure),
                "recovery_support": self._clip01(recovery_support),
            },
            name="inputs",
        )

        # Inputs -> latent biology
        dopaminergic_reward_crash = self._clip01(
            0.38 * inputs["chronic_stimulant_exposure"]
            + 0.28 * inputs["acute_abstinence_shift"]
            + 0.15 * inputs["genetic_vulnerability"]
            + 0.08 * inputs["stress_load"]
            - 0.12 * inputs["recovery_support"]
        )

        serotonergic_noradrenergic_depletion = self._clip01(
            0.28 * inputs["chronic_stimulant_exposure"]
            + 0.38 * inputs["serotonergic_neurotoxicity_load"]
            + 0.12 * inputs["acute_abstinence_shift"]
            + 0.08 * inputs["stress_load"]
            - 0.10 * inputs["recovery_support"]
        )

        stress_epigenetic_sensitization = self._clip01(
            0.30 * inputs["stress_load"]
            + 0.24 * inputs["chronic_stimulant_exposure"]
            + 0.18 * inputs["genetic_vulnerability"]
            + 0.12 * inputs["serotonergic_neurotoxicity_load"]
            - 0.10 * inputs["recovery_support"]
        )

        amygdala_threat_reactivity = self._clip01(
            0.34 * stress_epigenetic_sensitization
            + 0.22 * inputs["stress_load"]
            + 0.16 * serotonergic_noradrenergic_depletion
            + 0.12 * dopaminergic_reward_crash
            - 0.10 * inputs["recovery_support"]
        )

        hippocampal_craving_memory_burden = self._clip01(
            0.28 * inputs["medication_cue_exposure"]
            + 0.26 * stress_epigenetic_sensitization
            + 0.18 * inputs["chronic_stimulant_exposure"]
            + 0.16 * amygdala_threat_reactivity
        )

        cue_reactivity_relapse_circuit = self._clip01(
            0.35 * inputs["medication_cue_exposure"]
            + 0.24 * hippocampal_craving_memory_burden
            + 0.18 * amygdala_threat_reactivity
            + 0.14 * dopaminergic_reward_crash
            - 0.10 * inputs["recovery_support"]
        )

        frontolimbic_control_failure = self._clip01(
            0.26 * stress_epigenetic_sensitization
            + 0.24 * amygdala_threat_reactivity
            + 0.18 * serotonergic_noradrenergic_depletion
            + 0.14 * dopaminergic_reward_crash
            + 0.10 * hippocampal_craving_memory_burden
            - 0.25 * inputs["recovery_support"]
        )

        allostatic_negative_affect_state = self._clip01(
            0.34 * dopaminergic_reward_crash
            + 0.28 * serotonergic_noradrenergic_depletion
            + 0.16 * amygdala_threat_reactivity
            + 0.12 * inputs["acute_abstinence_shift"]
            + 0.08 * inputs["stress_load"]
            - 0.12 * inputs["recovery_support"]
        )

        neuroplastic_addiction_entrenchment = self._clip01(
            0.34 * inputs["chronic_stimulant_exposure"]
            + 0.24 * stress_epigenetic_sensitization
            + 0.18 * cue_reactivity_relapse_circuit
            + 0.14 * hippocampal_craving_memory_burden
            - 0.08 * inputs["recovery_support"]
        )

        latents = pd.Series(
            {
                "dopaminergic_reward_crash": dopaminergic_reward_crash,
                "serotonergic_noradrenergic_depletion": serotonergic_noradrenergic_depletion,
                "stress_epigenetic_sensitization": stress_epigenetic_sensitization,
                "amygdala_threat_reactivity": amygdala_threat_reactivity,
                "hippocampal_craving_memory_burden": hippocampal_craving_memory_burden,
                "cue_reactivity_relapse_circuit": cue_reactivity_relapse_circuit,
                "frontolimbic_control_failure": frontolimbic_control_failure,
                "allostatic_negative_affect_state": allostatic_negative_affect_state,
                "neuroplastic_addiction_entrenchment": neuroplastic_addiction_entrenchment,
            },
            name="latents",
        )

        # Latent biology -> regional state
        ventral_striatum_proxy = self._clip01(
            0.64 * dopaminergic_reward_crash
            + 0.16 * cue_reactivity_relapse_circuit
            + 0.10 * allostatic_negative_affect_state
        )

        amygdala = self._clip01(
            0.66 * amygdala_threat_reactivity
            + 0.14 * cue_reactivity_relapse_circuit
            + 0.10 * allostatic_negative_affect_state
        )

        hippocampus = self._clip01(
            0.62 * hippocampal_craving_memory_burden
            + 0.14 * stress_epigenetic_sensitization
            + 0.10 * inputs["medication_cue_exposure"]
        )

        ofc = self._clip01(
            0.54 * frontolimbic_control_failure
            + 0.18 * dopaminergic_reward_crash
            + 0.10 * cue_reactivity_relapse_circuit
        )

        acc = self._clip01(
            0.56 * frontolimbic_control_failure
            + 0.18 * allostatic_negative_affect_state
            + 0.10 * stress_epigenetic_sensitization
        )

        insula = self._clip01(
            0.48 * allostatic_negative_affect_state
            + 0.20 * cue_reactivity_relapse_circuit
            + 0.14 * serotonergic_noradrenergic_depletion
        )

        pfc_control = self._clip01(
            0.64 * frontolimbic_control_failure
            + 0.14 * allostatic_negative_affect_state
            + 0.10 * neuroplastic_addiction_entrenchment
        )

        regional_state = pd.Series(
            {
                "ventral_striatum_proxy": ventral_striatum_proxy,
                "amygdala": amygdala,
                "hippocampus": hippocampus,
                "ofc": ofc,
                "acc": acc,
                "insula": insula,
                "pfc_control": pfc_control,
            },
            name="regional_state",
        )

        # Regional state -> symptoms
        depressed_mood = self._clip01(
            0.40 * allostatic_negative_affect_state
            + 0.26 * serotonergic_noradrenergic_depletion
            + 0.10 * inputs["stress_load"]
            + 0.08 * insula
        )

        anhedonia = self._clip01(
            0.50 * dopaminergic_reward_crash
            + 0.22 * allostatic_negative_affect_state
            + 0.12 * ventral_striatum_proxy
        )

        fatigue_psychomotor_slowing = self._clip01(
            0.34 * serotonergic_noradrenergic_depletion
            + 0.24 * dopaminergic_reward_crash
            + 0.16 * inputs["acute_abstinence_shift"]
            + 0.08 * depressed_mood
        )

        sleep_disturbance = self._clip01(
            0.28 * allostatic_negative_affect_state
            + 0.20 * serotonergic_noradrenergic_depletion
            + 0.18 * inputs["stress_load"]
            + 0.10 * insula
        )

        anxiety_irritability = self._clip01(
            0.34 * amygdala
            + 0.24 * allostatic_negative_affect_state
            + 0.16 * inputs["stress_load"]
            + 0.08 * cue_reactivity_relapse_circuit
        )

        craving = self._clip01(
            0.38 * cue_reactivity_relapse_circuit
            + 0.18 * hippocampus
            + 0.18 * ventral_striatum_proxy
            + 0.12 * inputs["medication_cue_exposure"]
            + 0.08 * allostatic_negative_affect_state
        )

        executive_dysfunction = self._clip01(
            0.34 * pfc_control
            + 0.18 * ofc
            + 0.16 * acc
            + 0.10 * allostatic_negative_affect_state
        )

        relapse_vulnerability = self._clip01(
            0.34 * craving
            + 0.22 * allostatic_negative_affect_state
            + 0.18 * executive_dysfunction
            + 0.10 * inputs["stress_load"]
            + 0.10 * neuroplastic_addiction_entrenchment
            - 0.12 * inputs["recovery_support"]
        )

        symptoms = pd.Series(
            {
                "depressed_mood": depressed_mood,
                "anhedonia": anhedonia,
                "fatigue_psychomotor_slowing": fatigue_psychomotor_slowing,
                "sleep_disturbance": sleep_disturbance,
                "anxiety_irritability": anxiety_irritability,
                "craving": craving,
                "executive_dysfunction": executive_dysfunction,
                "relapse_vulnerability": relapse_vulnerability,
            },
            name="symptoms",
        )

        phenotypes = pd.Series(
            {
                "depressive_withdrawal_profile": self._clip01(
                    (
                        depressed_mood
                        + anhedonia
                        + fatigue_psychomotor_slowing
                        + sleep_disturbance
                    )
                    / 4.0
                ),
                "cue_reactive_relapse_profile": self._clip01(
                    (craving + relapse_vulnerability + cue_reactivity_relapse_circuit) / 3.0
                ),
                "executive_control_impairment_profile": self._clip01(
                    (executive_dysfunction + pfc_control + ofc + acc) / 4.0
                ),
                "hyperkatifeia_negative_affect_profile": self._clip01(
                    (allostatic_negative_affect_state + anxiety_irritability + depressed_mood) / 3.0
                ),
            },
            name="phenotypes",
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }


if __name__ == "__main__":
    model = StimulantWithdrawalModel()
    bundle = model.build(connectivity_rows=10)

    pd.set_option("display.width", 150)
    pd.set_option("display.max_columns", 12)

    print("\n=== NODES ===")
    print(
        bundle["nodes"][
            [
                "key",
                "node_type",
                "atlas_region",
                "centroid_mni",
                "feature_summary",
            ]
        ].to_string(index=False)
    )

    print("\n=== EDGES ===")
    print(
        bundle["edges"][["source", "target", "relation", "weight"]].to_string(index=False)
    )

    for key in ["amygdala", "hippocampus", "ofc", "pfc_control"]:
        print(f"\n=== REGION SUMMARY: {key} ===")
        region = bundle["regions"].get(key)
        print(f"Resolved region: {getattr(region, 'name', 'unresolved proxy')}")
        receptor_df = bundle["receptors"].get(key, pd.DataFrame())
        gene_df = bundle["genes"].get(key, pd.DataFrame())
        conn_df = bundle["connectivity_profiles"].get(key, pd.DataFrame())
        print("Receptors:")
        print(receptor_df.head(8).to_string(index=False) if not receptor_df.empty else "  <none>")
        print("Genes:")
        print(gene_df.head(8).to_string(index=False) if not gene_df.empty else "  <none>")
        print("Connectivity:")
        print(conn_df.head(8).to_string(index=False) if not conn_df.empty else "  <none>")

    circuit_df = bundle["circuit_connectivity"]
    print("\n=== CIRCUIT CONNECTIVITY ===")
    print(circuit_df.head(20).to_string(index=False) if not circuit_df.empty else "<none>")

    sim = model.simulate(
        genetic_vulnerability=0.50,
        chronic_stimulant_exposure=0.80,
        acute_abstinence_shift=0.70,
        serotonergic_neurotoxicity_load=0.45,
        stress_load=0.60,
        medication_cue_exposure=0.55,
        recovery_support=0.25,
    )

    for section, series in sim.items():
        print(f"\n=== {section.upper()} ===")
        print(series.to_string())

    # Optional interactive checks:
    # print(model.suggest_regions("cingulate").head(10))
    # print(model.assign_mni_point((20, -6, -14)).head(10))
    # mask = model.region_mask("amygdala")
    # print(type(mask))
