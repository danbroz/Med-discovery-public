from __future__ import annotations

"""
Opioid Withdrawal atlas-grounded siibra scaffold.

This script translates a short neurobiological chapter on opioid withdrawal
into a transparent research scaffold. It is intended for mechanistic
exploration and atlas-backed feature querying, not for diagnosis, treatment
selection, or prediction for any individual.

Core chapter logic encoded here:
- chronic opioid exposure induces homeostatic recalibration across opioid and
  non-opioid transmitter systems,
- abrupt opioid removal unmasks these adaptations and produces widespread
  neuronal hyperexcitability,
- glutamatergic / NMDA upregulation is a major withdrawal driver,
- norepinephrine, dopamine, and serotonin dysregulation contribute to arousal,
  reward disruption, and executive burden,
- maladaptive synaptic plasticity can strengthen medication- and withdrawal-linked
  memories that support craving and relapse,
- hippocampal and prefrontal dysfunction contribute to cognitive deficits,
  while amygdala and locus-coeruleus-like arousal systems plausibly amplify
  distress during withdrawal.
"""

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise ImportError(
        "This scaffold requires the 'siibra' package. Install siibra-python and rerun."
    ) from exc


OPIOID_WITHDRAWAL_GENE_PANEL = [
    "OPRM1",
    "OPRK1",
    "OPRD1",
    "PENK",
    "PDYN",
    "POMC",
    "GRIN1",
    "GRIN2B",
    "SLC1A2",
    "SLC6A2",
    "DBH",
    "TH",
    "DRD2",
    "SLC6A3",
    "SLC6A4",
    "COMT",
    "BDNF",
    "NR3C1",
    "FKBP5",
]


class OpioidWithdrawalModel:
    """
    Atlas-grounded research scaffold for opioid withdrawal.

    Notes
    -----
    - The simulator is intentionally simple, normalized, and acyclic.
    - Region mappings are conservative. Broad or uncertain systems are modeled
      as explicit proxies rather than forced into overly precise parcels.
    - Some chapter circuitry claims are framed as plausible hypotheses rather
      than established imaging findings; these are preserved as cautious proxy
      nodes in the model.
    - Multimodal feature lookup degrades gracefully when receptor, gene, or
      connectivity data are missing.
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

        self.region_candidates: Dict[str, List[str]] = {
            "hippocampus": [
                "CA1 left",
                "Subiculum left",
                "DG left",
                "hippocampus left",
                "hippocampus",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "Amygdala left",
                "amygdala",
            ],
            "pfc_control": [
                "Area 46 left",
                "Area 9/46d left",
                "Area 9/46v left",
                "dorsolateral prefrontal",
                "prefrontal cortex",
            ],
            "nucleus_accumbens_proxy": [
                "nucleus accumbens",
                "accumbens",
                "ventral striatum",
            ],
            "locus_coeruleus_proxy": [
                "locus coeruleus",
                "locus coeruleus left",
                "LC",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "hippocampus": "Memory-related node capturing withdrawal-linked mnemonic and contextual burden.",
            "amygdala": "Limbic salience and affective-arousal node plausibly amplified during withdrawal.",
            "pfc_control": "Atlas-resolved proxy for broad prefrontal executive control and decision-making burden.",
            "nucleus_accumbens_proxy": "Reward-deficit proxy for ventral striatal / nucleus accumbens involvement in craving and relapse.",
            "locus_coeruleus_proxy": "Arousal-system proxy for noradrenergic withdrawal activation and hypervigilant distress.",
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": "Inherited liability affecting addiction vulnerability, transmitter balance, stress reactivity, and impulsivity.",
            "chronic_opioid_exposure": "Cumulative opioid exposure driving chronic homeostatic adaptations.",
            "abrupt_opioid_removal": "Acute opioid discontinuation that unmasks compensatory adaptations and precipitates withdrawal.",
            "stress_reactivity_load": "Background stress-reactivity burden that can intensify withdrawal-related arousal and relapse risk.",
            "prior_withdrawal_cycles": "Repeated dependence and withdrawal episodes that may strengthen sensitization and maladaptive learning.",
            "withdrawal_treatment_support": "Protective treatment support that reduces acute withdrawal burden and stabilizes physiology.",
            "recovery_support": "Protective psychosocial and recovery structure buffering relapse pressure and chronic dysregulation.",
        }

        self.latent_nodes: Dict[str, str] = {
            "homeostatic_recalibration": "Chronic CNS adaptation across opioid and non-opioid systems after prolonged exposure.",
            "glutamatergic_nmda_upregulation": "Compensatory glutamatergic / NMDA overactivity contributing to withdrawal hyperexcitability.",
            "noradrenergic_hyperarousal": "Heightened arousal-state pressure linked to norepinephrine and locus-coeruleus-like activation.",
            "monoamine_reward_control_dysregulation": "Dopamine and serotonin disruption affecting reward, mood regulation, and executive control.",
            "maladaptive_synaptic_plasticity": "Withdrawal-linked plasticity that strengthens distress, cue memory, and relapse vulnerability.",
            "frontolimbic_cognitive_dysregulation": "Disruption of hippocampal-prefrontal-limbic coordination underlying concentration, memory, and decision deficits.",
        }

        self.symptom_nodes: Dict[str, str] = {
            "withdrawal_hyperexcitability": "Core withdrawal burden rooted in widespread neuronal overactivation.",
            "cognitive_impairment": "Difficulty with concentration, memory, and decision-making during acute or protracted withdrawal.",
            "affective_arousal_distress": "Emotion-arousal disruption across amygdala, stress, and monoamine systems.",
            "craving_relapse_vulnerability": "Heightened craving and relapse pressure linked to maladaptive learning and reward deficit.",
            "protracted_withdrawal_burden": "Persistent downstream burden spanning cognition, arousal, and relapse liability.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "chronic_opioid_exposure",
                "target": "homeostatic_recalibration",
                "relation": "drives chronic compensatory adaptation across opioid and non-opioid systems",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "chronic_opioid_exposure",
                "target": "glutamatergic_nmda_upregulation",
                "relation": "upregulates NMDA-mediated glutamatergic tone under chronic inhibitory opioid pressure",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "chronic_opioid_exposure",
                "target": "monoamine_reward_control_dysregulation",
                "relation": "alters dopamine and serotonin systems involved in reward and control",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "homeostatic_recalibration",
                "relation": "modulates susceptibility to dependence-related adaptation",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "monoamine_reward_control_dysregulation",
                "relation": "raises baseline vulnerability in transmitter systems and stress reactivity",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "stress_reactivity_load",
                "target": "noradrenergic_hyperarousal",
                "relation": "amplifies arousal burden and stress-linked withdrawal intensity",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "prior_withdrawal_cycles",
                "target": "maladaptive_synaptic_plasticity",
                "relation": "strengthens withdrawal-linked learning and sensitization over repeated episodes",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "prior_withdrawal_cycles",
                "target": "glutamatergic_nmda_upregulation",
                "relation": "supports kindling-like amplification of excitatory withdrawal mechanisms",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "abrupt_opioid_removal",
                "target": "glutamatergic_nmda_upregulation",
                "relation": "unmasks compensatory glutamatergic overactivity during withdrawal onset",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "abrupt_opioid_removal",
                "target": "noradrenergic_hyperarousal",
                "relation": "triggers acute arousal escalation as inhibitory opioid tone is lost",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "withdrawal_treatment_support",
                "target": "glutamatergic_nmda_upregulation",
                "relation": "buffers acute withdrawal physiology and excitatory overactivation",
                "opioid_withdrawal_change": "decreased",
            },
            {
                "source": "withdrawal_treatment_support",
                "target": "noradrenergic_hyperarousal",
                "relation": "reduces autonomic and arousal burden during withdrawal",
                "opioid_withdrawal_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "maladaptive_synaptic_plasticity",
                "relation": "reduces reinforcement of medication- and distress-linked learning loops",
                "opioid_withdrawal_change": "decreased",
            },
            {
                "source": "recovery_support",
                "target": "craving_relapse_vulnerability",
                "relation": "buffers relapse pressure and promotes recovery structure",
                "opioid_withdrawal_change": "decreased",
            },
            {
                "source": "homeostatic_recalibration",
                "target": "glutamatergic_nmda_upregulation",
                "relation": "supports excitatory rebound after chronic opioid adaptation",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "glutamatergic_nmda_upregulation",
                "target": "maladaptive_synaptic_plasticity",
                "relation": "disrupts LTP/LTD balance and stamps in medication- and withdrawal-associated memories",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "glutamatergic_nmda_upregulation",
                "target": "withdrawal_hyperexcitability",
                "relation": "drives widespread neuronal hyperexcitability",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "withdrawal_hyperexcitability",
                "relation": "amplifies arousal and physiological withdrawal intensity",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "monoamine_reward_control_dysregulation",
                "target": "frontolimbic_cognitive_dysregulation",
                "relation": "disrupts reward, mood, and executive-regulation systems",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "maladaptive_synaptic_plasticity",
                "target": "hippocampus",
                "relation": "burdens contextual and withdrawal-linked memory processing",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "frontolimbic_cognitive_dysregulation",
                "target": "pfc_control",
                "relation": "reduces executive control and decision stability",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "frontolimbic_cognitive_dysregulation",
                "target": "amygdala",
                "relation": "increases emotional-arousal dysregulation in limbic circuitry",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "monoamine_reward_control_dysregulation",
                "target": "nucleus_accumbens_proxy",
                "relation": "contributes to reward deficit and craving pressure in ventral striatal circuitry",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "locus_coeruleus_proxy",
                "relation": "maps arousal dysregulation to a locus-coeruleus-like proxy",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "cognitive_impairment",
                "relation": "hippocampal dysfunction contributes to memory and concentration problems",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "cognitive_impairment",
                "relation": "prefrontal dysfunction contributes to executive and decision difficulties",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "affective_arousal_distress",
                "relation": "heightens negative arousal and emotional salience during withdrawal",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "locus_coeruleus_proxy",
                "target": "affective_arousal_distress",
                "relation": "increases noradrenergic arousal burden",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "nucleus_accumbens_proxy",
                "target": "craving_relapse_vulnerability",
                "relation": "reward deficit and cue salience support craving and relapse pressure",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "maladaptive_synaptic_plasticity",
                "target": "craving_relapse_vulnerability",
                "relation": "strengthens learned associations between withdrawal distress and medication-seeking",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "withdrawal_hyperexcitability",
                "target": "protracted_withdrawal_burden",
                "relation": "acute excitatory burden can feed ongoing withdrawal difficulty",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "cognitive_impairment",
                "target": "protracted_withdrawal_burden",
                "relation": "executive and memory deficits contribute to persistent functional burden",
                "opioid_withdrawal_change": "increased",
            },
            {
                "source": "craving_relapse_vulnerability",
                "target": "protracted_withdrawal_burden",
                "relation": "relapse pressure sustains the chronic burden of withdrawal and recovery instability",
                "opioid_withdrawal_change": "increased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}

        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()
        self._pmap: Any = None
        self._connectivity_feature: Any = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _mean(values: Iterable[float]) -> float:
        vals = [float(v) for v in values]
        return 0.0 if not vals else sum(vals) / len(vals)

    @staticmethod
    def _region_identifier(region: Any) -> Optional[str]:
        return getattr(region, "identifier", getattr(region, "id", None))

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
            candidates.append("receptor density fingerprint")
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
                    features = siibra.features.get(concept, modality, **kwargs)
                if features:
                    return list(features)
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {"hippocampus", "amygdala", "prefrontal cortex"} else 0
        parent_penalty = 1 if any(token in name for token in ["lobe", "cortex"]) and "area" not in name else 0
        return (left_bonus, right_penalty, generic_penalty, parent_penalty)

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
                return sorted(matches, key=self._region_rank)[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        seen = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
            row = (
                self._name_of(region),
                self._region_identifier(region),
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
        self, region: Any
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_raw = getattr(centroid, "coordinate", centroid)
        centroid_xyz = None
        if centroid_raw is not None:
            try:
                centroid_xyz = tuple(float(x) for x in centroid_raw)
            except Exception:
                centroid_xyz = None
        volume_mm3 = getattr(main, "volume", None)
        volume = float(volume_mm3) if volume_mm3 is not None else None
        return centroid_xyz, volume

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        features = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not features:
            return pd.DataFrame()
        try:
            data = features[0].data.copy()
            if isinstance(data, pd.Series):
                df = data.reset_index()
                if len(df.columns) == 2:
                    df.columns = ["receptor", "value"]
                return df
            if isinstance(data, pd.DataFrame):
                df = data.reset_index()
                if "index" in df.columns and "receptor" not in df.columns:
                    df = df.rename(columns={"index": "receptor"})
                return df
        except Exception:
            pass
        return pd.DataFrame()

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        features = self._safe_features_any(
            region,
            self._modality_candidates("gene"),
            gene=list(genes),
        )
        if not features:
            return pd.DataFrame()
        try:
            df = features[0].data.copy()
        except Exception:
            return pd.DataFrame()
        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()

        lower_cols = {str(c).lower(): c for c in df.columns}
        required = {"gene", "level", "zscore"}
        if required.issubset(lower_cols):
            gene_col = lower_cols["gene"]
            level_col = lower_cols["level"]
            zscore_col = lower_cols["zscore"]
            out = (
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
            return out
        return df.reset_index(drop=True)

    def _pick_connectivity_feature(self, features: Sequence[Any]) -> Optional[Any]:
        if not features:
            return None
        wanted = str(self.connectivity_cohort).lower()
        for feat in features:
            cohort = str(getattr(feat, "cohort", "")).lower()
            if cohort == wanted:
                return feat
        return features[0]

    @staticmethod
    def _coerce_scalar(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            pass
        if isinstance(value, pd.Series):
            numeric = pd.to_numeric(value, errors="coerce").dropna()
            return float(numeric.mean()) if not numeric.empty else None
        if isinstance(value, pd.DataFrame):
            numeric = value.apply(pd.to_numeric, errors="coerce").stack().dropna()
            return float(numeric.mean()) if not numeric.empty else None
        if isinstance(value, (list, tuple)):
            numeric = [OpioidWithdrawalModel._coerce_scalar(v) for v in value]
            numeric = [v for v in numeric if v is not None]
            return float(sum(numeric) / len(numeric)) if numeric else None
        return None

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        features = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not features and self.region_objects:
            first_region = next(iter(self.region_objects.values()))
            features = self._safe_features_any(first_region, self._modality_candidates("connectivity"))

        feature = self._pick_connectivity_feature(features)
        self._connectivity_feature = feature
        if feature is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        try:
            data = getattr(feature, "data", None)
            if isinstance(data, pd.DataFrame):
                self._connectivity_matrix = data.copy()
                return self._connectivity_matrix
        except Exception:
            pass

        try:
            for element in feature:
                data = getattr(element, "data", None)
                if isinstance(data, pd.DataFrame):
                    self._connectivity_matrix = data.copy()
                    return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        target = self._name_of(region)
        exact = [label for label in labels if self._name_of(label) == target]
        if exact:
            return exact[0]
        target_lower = target.lower()
        fuzzy = [
            label
            for label in labels
            if target_lower in self._name_of(label).lower()
            or self._name_of(label).lower() in target_lower
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
            if not isinstance(series, pd.Series):
                return pd.DataFrame()
            numeric = pd.to_numeric(series, errors="coerce").dropna().sort_values(ascending=False)
            if numeric.empty:
                return pd.DataFrame()
            df = numeric.reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        labels_index = list(matrix.index)
        labels_columns = list(matrix.columns)
        keys = list(self.region_objects.keys())
        rows: List[Dict[str, Any]] = []

        for i, source_key in enumerate(keys):
            source_region = self.region_objects[source_key]
            source_idx = self._match_region_label(labels_index, source_region)
            source_col = self._match_region_label(labels_columns, source_region)
            for target_key in keys[i + 1 :]:
                target_region = self.region_objects[target_key]
                target_idx = self._match_region_label(labels_index, target_region)
                target_col = self._match_region_label(labels_columns, target_region)

                value = None
                try:
                    if source_idx is not None and target_col is not None:
                        value = self._coerce_scalar(matrix.loc[source_idx, target_col])
                except Exception:
                    value = None
                if value is None:
                    try:
                        if target_idx is not None and source_col is not None:
                            value = self._coerce_scalar(matrix.loc[target_idx, source_col])
                    except Exception:
                        value = None

                rows.append(
                    {
                        "source_key": source_key,
                        "source_region": source_region.name,
                        "target_key": target_key,
                        "target_region": target_region.name,
                        "value": value,
                        "cohort": self.connectivity_cohort,
                        "modality": "StreamlineCounts",
                    }
                )

        df = pd.DataFrame(rows)
        if not df.empty and "value" in df.columns:
            df = df.sort_values("value", ascending=False, na_position="last").reset_index(drop=True)
        return df

    def build(
        self,
        gene_panel: Sequence[str] = OPIOID_WITHDRAWAL_GENE_PANEL,
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
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region",
                        "description": self.region_node_descriptions.get(
                            key,
                            "Atlas-backed region node unresolved in this environment.",
                        ),
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
            connectivity_df = self._connectivity_profile(region, max_rows=connectivity_rows)
            self.receptors[key] = receptor_df
            self.genes[key] = gene_df
            self.connectivity_profiles[key] = connectivity_df

            nodes.append(
                {
                    "key": key,
                    "label": region.name,
                    "node_type": "region",
                    "description": self.region_node_descriptions.get(key, "Atlas-backed circuit node."),
                    "atlas_region": region.name,
                    "region_identifier": self._region_identifier(region),
                    "centroid_mni": centroid_mni,
                    "volume_mm3": volume_mm3,
                    "feature_summary": (
                        f"receptors={'yes' if not receptor_df.empty else 'no'}; "
                        f"genes={'yes' if not gene_df.empty else 'no'}; "
                        f"connectivity={'yes' if not connectivity_df.empty else 'no'}"
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
        circuit_df = self.circuit_connectivity()
        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": circuit_df,
            "metadata": {
                "disorder": "Opioid Withdrawal",
                "gene_panel": list(gene_panel),
                "atlas": getattr(self.atlas, "name", str(self.atlas)),
                "parcellation": getattr(self.parcellation, "name", str(self.parcellation)),
                "space": getattr(self.space, "name", str(self.space)),
                "assignment_space": self.assignment_space,
                "connectivity_cohort": self.connectivity_cohort,
                "notes": "Research scaffold only; not a clinical tool.",
            },
        }

    def simulate(
        self,
        genetic_vulnerability: float = 0.35,
        chronic_opioid_exposure: float = 0.75,
        abrupt_opioid_removal: float = 0.65,
        stress_reactivity_load: float = 0.35,
        prior_withdrawal_cycles: float = 0.30,
        withdrawal_treatment_support: float = 0.20,
        recovery_support: float = 0.20,
    ) -> Dict[str, pd.Series]:
        """
        Run a transparent normalized opioid withdrawal simulation.

        All inputs are clipped to [0, 1]. Higher values generally represent
        more burden, except the two protective support variables.
        """

        gv = self._clip01(genetic_vulnerability)
        coe = self._clip01(chronic_opioid_exposure)
        aor = self._clip01(abrupt_opioid_removal)
        srl = self._clip01(stress_reactivity_load)
        pwc = self._clip01(prior_withdrawal_cycles)
        wts = self._clip01(withdrawal_treatment_support)
        rs = self._clip01(recovery_support)

        inputs = pd.Series(
            {
                "genetic_vulnerability": gv,
                "chronic_opioid_exposure": coe,
                "abrupt_opioid_removal": aor,
                "stress_reactivity_load": srl,
                "prior_withdrawal_cycles": pwc,
                "withdrawal_treatment_support": wts,
                "recovery_support": rs,
            },
            name="inputs",
        )

        homeostatic_recalibration = self._clip01(
            0.45 * coe + 0.20 * gv + 0.15 * pwc + 0.10 * srl - 0.10 * wts - 0.10 * rs
        )
        glutamatergic_nmda_upregulation = self._clip01(
            0.30 * coe
            + 0.25 * aor
            + 0.20 * homeostatic_recalibration
            + 0.15 * pwc
            - 0.20 * wts
            - 0.05 * rs
        )
        noradrenergic_hyperarousal = self._clip01(
            0.30 * aor
            + 0.25 * srl
            + 0.20 * glutamatergic_nmda_upregulation
            + 0.10 * pwc
            + 0.05 * gv
            - 0.20 * wts
            - 0.05 * rs
        )
        monoamine_reward_control_dysregulation = self._clip01(
            0.25 * coe
            + 0.20 * aor
            + 0.20 * gv
            + 0.15 * srl
            + 0.10 * homeostatic_recalibration
            - 0.10 * wts
            - 0.10 * rs
        )
        maladaptive_synaptic_plasticity = self._clip01(
            0.30 * pwc
            + 0.25 * glutamatergic_nmda_upregulation
            + 0.20 * coe
            + 0.10 * monoamine_reward_control_dysregulation
            + 0.05 * srl
            - 0.15 * rs
        )
        frontolimbic_cognitive_dysregulation = self._clip01(
            0.30 * glutamatergic_nmda_upregulation
            + 0.25 * monoamine_reward_control_dysregulation
            + 0.20 * noradrenergic_hyperarousal
            + 0.10 * maladaptive_synaptic_plasticity
            + 0.05 * aor
            - 0.10 * wts
            - 0.10 * rs
        )

        latents = pd.Series(
            {
                "homeostatic_recalibration": homeostatic_recalibration,
                "glutamatergic_nmda_upregulation": glutamatergic_nmda_upregulation,
                "noradrenergic_hyperarousal": noradrenergic_hyperarousal,
                "monoamine_reward_control_dysregulation": monoamine_reward_control_dysregulation,
                "maladaptive_synaptic_plasticity": maladaptive_synaptic_plasticity,
                "frontolimbic_cognitive_dysregulation": frontolimbic_cognitive_dysregulation,
            },
            name="latents",
        )

        hippocampal_memory_disruption = self._clip01(
            0.35 * maladaptive_synaptic_plasticity
            + 0.30 * glutamatergic_nmda_upregulation
            + 0.15 * srl
            + 0.10 * homeostatic_recalibration
            + 0.10 * aor
            - 0.10 * rs
        )
        amygdala_hyperreactivity = self._clip01(
            0.35 * noradrenergic_hyperarousal
            + 0.25 * monoamine_reward_control_dysregulation
            + 0.20 * aor
            + 0.10 * srl
            + 0.10 * frontolimbic_cognitive_dysregulation
            - 0.10 * wts
        )
        pfc_control_failure = self._clip01(
            0.35 * frontolimbic_cognitive_dysregulation
            + 0.25 * monoamine_reward_control_dysregulation
            + 0.15 * glutamatergic_nmda_upregulation
            + 0.10 * noradrenergic_hyperarousal
            + 0.05 * aor
            - 0.15 * wts
            - 0.10 * rs
        )
        nucleus_accumbens_reward_deficit = self._clip01(
            0.30 * homeostatic_recalibration
            + 0.25 * monoamine_reward_control_dysregulation
            + 0.20 * aor
            + 0.15 * maladaptive_synaptic_plasticity
            - 0.10 * wts
            - 0.10 * rs
        )
        locus_coeruleus_activation = self._clip01(
            0.45 * noradrenergic_hyperarousal
            + 0.25 * aor
            + 0.20 * glutamatergic_nmda_upregulation
            + 0.10 * srl
            - 0.15 * wts
        )

        regional_state = pd.Series(
            {
                "hippocampal_memory_disruption": hippocampal_memory_disruption,
                "amygdala_hyperreactivity": amygdala_hyperreactivity,
                "pfc_control_failure": pfc_control_failure,
                "nucleus_accumbens_reward_deficit": nucleus_accumbens_reward_deficit,
                "locus_coeruleus_activation": locus_coeruleus_activation,
            },
            name="regional_state",
        )

        withdrawal_hyperexcitability = self._clip01(
            0.40 * glutamatergic_nmda_upregulation
            + 0.25 * locus_coeruleus_activation
            + 0.20 * aor
            + 0.15 * noradrenergic_hyperarousal
        )
        cognitive_impairment = self._clip01(
            0.35 * hippocampal_memory_disruption
            + 0.35 * pfc_control_failure
            + 0.15 * noradrenergic_hyperarousal
            + 0.15 * withdrawal_hyperexcitability
        )
        affective_arousal_distress = self._clip01(
            0.35 * amygdala_hyperreactivity
            + 0.30 * locus_coeruleus_activation
            + 0.20 * nucleus_accumbens_reward_deficit
            + 0.15 * withdrawal_hyperexcitability
        )
        craving_relapse_vulnerability = self._clip01(
            0.35 * maladaptive_synaptic_plasticity
            + 0.25 * nucleus_accumbens_reward_deficit
            + 0.20 * affective_arousal_distress
            + 0.10 * withdrawal_hyperexcitability
            + 0.10 * pfc_control_failure
            - 0.15 * rs
        )
        protracted_withdrawal_burden = self._clip01(
            0.30 * withdrawal_hyperexcitability
            + 0.25 * cognitive_impairment
            + 0.25 * craving_relapse_vulnerability
            + 0.20 * affective_arousal_distress
        )

        symptoms = pd.Series(
            {
                "withdrawal_hyperexcitability": withdrawal_hyperexcitability,
                "cognitive_impairment": cognitive_impairment,
                "affective_arousal_distress": affective_arousal_distress,
                "craving_relapse_vulnerability": craving_relapse_vulnerability,
                "protracted_withdrawal_burden": protracted_withdrawal_burden,
            },
            name="symptoms",
        )

        phenotypes = pd.Series(
            {
                "acute_hyperexcitable_withdrawal_profile": self._clip01(
                    self._mean(
                        [
                            withdrawal_hyperexcitability,
                            locus_coeruleus_activation,
                            glutamatergic_nmda_upregulation,
                        ]
                    )
                ),
                "memory_relapse_profile": self._clip01(
                    self._mean(
                        [
                            maladaptive_synaptic_plasticity,
                            hippocampal_memory_disruption,
                            craving_relapse_vulnerability,
                        ]
                    )
                ),
                "cognitive_control_disruption_profile": self._clip01(
                    self._mean(
                        [
                            pfc_control_failure,
                            cognitive_impairment,
                            frontolimbic_cognitive_dysregulation,
                        ]
                    )
                ),
                "stress_amplified_arousal_profile": self._clip01(
                    self._mean(
                        [
                            noradrenergic_hyperarousal,
                            affective_arousal_distress,
                            amygdala_hyperreactivity,
                        ]
                    )
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

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        if self._pmap is None:
            try:
                with siibra.QUIET:
                    self._pmap = self.atlas.get_map(
                        space=self.assignment_space,
                        parcellation=self.parcellation,
                        maptype="statistical",
                    )
            except Exception:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )

        point = siibra.Point(tuple(float(x) for x in xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in (
            "map value",
            "value",
            "correlation",
            "intersection over union",
            "contains",
            "contained",
        ):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(
        self,
        node_key: str,
        maptype: str = "labelled",
        threshold: float = 0.05,
        fetch: bool = True,
    ) -> Any:
        region = self.region_objects.get(node_key)
        if region is None and node_key in self.region_candidates:
            region = self._resolve_region(self.region_candidates[node_key])
            if region is not None:
                self.region_objects[node_key] = region
        if region is None:
            warnings.warn(f"Unknown or unresolved region node: {node_key}")
            return None

        errors: List[Exception] = []

        if hasattr(region, "get_regional_mask"):
            try:
                mask_obj = region.get_regional_mask(space=self.assignment_space, maptype=maptype)
                return mask_obj.fetch() if fetch and hasattr(mask_obj, "fetch") else mask_obj
            except Exception as exc:
                errors.append(exc)

        if hasattr(region, "get_regional_map"):
            try:
                map_obj = region.get_regional_map(self.assignment_space, maptype)
                return map_obj.fetch() if fetch and hasattr(map_obj, "fetch") else map_obj
            except Exception as exc:
                errors.append(exc)

        if hasattr(region, "fetch_regional_map"):
            try:
                return region.fetch_regional_map(
                    space=self.assignment_space,
                    maptype=maptype,
                    threshold=threshold,
                )
            except Exception as exc:
                errors.append(exc)

        if errors:
            warnings.warn(f"Could not fetch mask for '{node_key}': {errors[-1]}")
        return None


if __name__ == "__main__":
    model = OpioidWithdrawalModel()
    bundle = model.build(connectivity_rows=10)

    print("\n=== Nodes ===")
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

    print("\n=== Edge sample ===")
    print(bundle["edges"].head(12).to_string(index=False))

    print("\n=== Resolved regions ===")
    for key, region in bundle["regions"].items():
        print(f"- {key}: {region.name}")

    print("\n=== Example multimodal summaries ===")
    for node_key in ["hippocampus", "amygdala", "pfc_control"]:
        receptor_df = bundle["receptors"].get(node_key, pd.DataFrame())
        gene_df = bundle["genes"].get(node_key, pd.DataFrame())
        conn_df = bundle["connectivity_profiles"].get(node_key, pd.DataFrame())
        print(
            f"\n[{node_key}] receptors rows={len(receptor_df)}, "
            f"genes rows={len(gene_df)}, connectivity rows={len(conn_df)}"
        )
        if not gene_df.empty:
            print(gene_df.head(5).to_string(index=False))
        if not conn_df.empty:
            print(conn_df.head(5).to_string(index=False))

    print("\n=== Circuit connectivity ===")
    circuit_df = bundle["circuit_connectivity"]
    if circuit_df.empty:
        print("No pairwise circuit connectivity matrix available in this environment.")
    else:
        print(circuit_df.to_string(index=False))

    print("\n=== Example simulation ===")
    sim = model.simulate(
        genetic_vulnerability=0.55,
        chronic_opioid_exposure=0.90,
        abrupt_opioid_removal=0.85,
        stress_reactivity_load=0.50,
        prior_withdrawal_cycles=0.65,
        withdrawal_treatment_support=0.30,
        recovery_support=0.15,
    )
    for name, series in sim.items():
        print(f"\n{name.upper()}")
        print(series.to_string())

    # Optional helpers for interactive use:
    # print(model.suggest_regions("hippocampus").head(10).to_string(index=False))
    # print(model.assign_mni_point((-28.0, -18.0, -14.0)).head(10).to_string(index=False))
    # mask_img = model.region_mask("amygdala")
