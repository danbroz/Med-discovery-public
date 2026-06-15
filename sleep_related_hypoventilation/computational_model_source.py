from __future__ import annotations

"""
Sleep-Related Hypoventilation siibra scaffold.

This script converts a biologically focused chapter on Sleep-Related
Hypoventilation (SRH) into an atlas-grounded siibra research scaffold. It is
intended for transparent mechanistic exploration, not diagnosis, prognosis, or
clinical decision-making.

Modeling emphasis from the chapter:
- central respiratory control depends on brainstem rhythmogenic and
  chemosensory circuits, especially for CO2-responsive ventilatory drive,
- the congenital central hypoventilation syndrome (CCHS) subtype is strongly
  linked to PHOX2B-related autonomic respiratory circuit development,
- obesity-related SRH is shaped by leptin dysregulation and inflammatory
  cytokine effects on central respiratory drive,
- sleep removes part of wakefulness-related drive, exposing state-dependent
  ventilatory control failure,
- chronic hypoxia, hypercapnia, and sleep fragmentation may contribute to
  hippocampal/temporal vulnerability, cognitive morbidity, and brain-aging
  pressure.

Important genetics note:
PHOX2B is the clearest validated genetic contributor in the chapter, especially
for CCHS. Other genes in the default panel are included as mechanistic or
modulatory candidates relevant to autonomic development, catecholaminergic
signaling, leptin pathways, hypoxic response, and inflammatory burden. They are
not validated SRH risk genes to the same degree as PHOX2B.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


DEFAULT_GENE_PANEL = [
    "PHOX2B",
    "DBH",
    "RET",
    "LEP",
    "LEPR",
    "HIF1A",
    "EPAS1",
    "IL6",
    "TNF",
    "BDNF",
]


class SleepRelatedHypoventilationModel:
    """
    Atlas-grounded scaffold for Sleep-Related Hypoventilation.

    The scaffold translates a chapter-level biological narrative into:
    - input nodes capturing congenital autonomic risk, obesity-related load,
      inflammatory burden, pulmonary/mechanical burden, and state-dependent
      sleep vulnerability,
    - latent biology nodes capturing chemosensitivity failure, reduced
      ventilatory drive, leptin resistance, inflammatory suppression of drive,
      and chronic neural burden,
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
        # Brainstem respiratory nuclei are modeled as a proxy because the chapter
        # is systems-level and exact cytoarchitectonic labels may vary by build.
        self.region_candidates: Dict[str, List[str]] = {
            "brainstem_respiratory_proxy": [
                "medulla oblongata left",
                "pons left",
                "mesencephalon left",
                "brainstem left",
                "brainstem",
                "medulla oblongata",
                "pons",
            ],
            "hypothalamus_proxy": [
                "hypothalamus left",
                "hypothalamus",
                "anterior hypothalamus",
                "posterior hypothalamus",
            ],
            "nucleus_accumbens": [
                "Nucleus accumbens left",
                "Acb left",
                "accumbens left",
                "ventral striatum left",
                "ventral striatum",
                "striatum left",
            ],
            "hippocampus": [
                "CA1 left",
                "Subiculum left",
                "DG left",
                "hippocampus left",
                "hippocampus",
            ],
            "temporal_lobe_proxy": [
                "Area TE 2.1 (STG) left",
                "superior temporal gyrus left",
                "middle temporal gyrus left",
                "temporal pole left",
                "temporal left",
                "temporal lobe",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "brainstem_respiratory_proxy": "Proxy for medullary and pontine respiratory rhythmogenic and chemosensory integration circuits.",
            "hypothalamus_proxy": "Proxy for homeostatic and hormonal modulation of respiratory drive, including leptin-linked effects.",
            "nucleus_accumbens": "Exploratory dopaminergic state-modulation node reflecting the chapter's limited discussion of dopamine-related sleep control.",
            "hippocampus": "Memory-system node vulnerable to chronic hypoxia, hypercapnia, and sleep-related neural burden.",
            "temporal_lobe_proxy": "Proxy for broader temporal lobe gray-matter vulnerability discussed in relation to chronic sleep-disordered breathing burden.",
        }

        self.input_nodes: Dict[str, str] = {
            "phox2b_autonomic_defect": "Congenital autonomic developmental liability relevant to CCHS and impaired automatic breathing control.",
            "obesity_leptin_dysregulation": "Obesity-related load that may reduce ventilatory drive through leptin resistance and altered respiratory control.",
            "inflammatory_cytokine_load": "Inflammatory burden, including cytokine effects that may suppress or distort central respiratory drive.",
            "pulmonary_mechanical_burden": "Mechanical or pulmonary burden that increases nocturnal ventilatory challenge.",
            "sleep_state_dependence": "Degree to which ventilation fails when wakefulness support is withdrawn during sleep.",
            "chronic_gas_exchange_burden": "Accumulated hypoxic and hypercapnic burden over time.",
            "dopaminergic_modulatory_instability": "Exploratory state-dependent dopaminergic instability affecting arousal and respiratory control boundaries.",
            "wake_drive_reserve": "Residual wakefulness-related respiratory support and arousal reserve that can partially buffer sleep-related failure.",
        }

        self.latent_nodes: Dict[str, str] = {
            "autonomic_breathing_circuit_dysdevelopment": "Developmental failure of autonomic respiratory circuitry, most salient in PHOX2B-related CCHS.",
            "co2_chemosensitivity_failure": "Blunted ventilatory response to hypercapnia, a cardinal abnormality in many forms of SRH.",
            "hypoxic_response_blunting": "Reduced responsiveness to hypoxia relative to metabolic demand.",
            "leptin_central_resistance": "Reduced effectiveness of leptin-related respiratory stimulation in obesity-linked SRH.",
            "inflammatory_drive_suppression": "Cytokine-linked suppression or distortion of central respiratory drive.",
            "sleep_transition_drive_loss": "Loss of wakefulness-related ventilatory support during sleep transitions and stable sleep.",
            "central_ventilatory_drive_reduction": "Net reduction in automatic ventilatory output from convergent central mechanisms.",
            "nocturnal_gas_exchange_failure": "Failure to maintain adequate nocturnal ventilation, producing sleep-related gas derangement.",
            "cerebral_autoregulation_stress": "Stress on cerebral perfusion and autoregulation from chronic gas-exchange disturbance.",
            "chronic_hypoxic_hypercapnic_neural_burden": "Accumulated neural burden from repeated hypoxia, hypercapnia, and sleep disruption.",
            "brain_aging_pressure": "Longer-term pressure toward gray-matter loss, white-matter damage, and neurodegenerative vulnerability.",
        }

        self.symptom_nodes: Dict[str, str] = {
            "blunted_ventilatory_response": "Weak ventilatory response to carbon dioxide and related chemical drive signals.",
            "sleep_related_hypoventilation": "Core sleep-related alveolar hypoventilation phenotype.",
            "nocturnal_hypercapnia": "Sleep-period CO2 retention.",
            "nocturnal_hypoxemia": "Sleep-period oxygen desaturation.",
            "sleep_fragmentation": "Arousals and disrupted sleep architecture secondary to unstable respiration.",
            "autonomic_respiratory_failure": "Failure of automatic respiratory control, especially during sleep.",
            "neurocognitive_morbidity": "Cognitive and behavioral burden linked to chronic sleep-disordered breathing and brain effects.",
            "brain_aging_vulnerability": "Risk of structural brain aging and neurodegenerative burden related to chronic hypoxic stress.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "phox2b_autonomic_defect",
                "target": "autonomic_breathing_circuit_dysdevelopment",
                "relation": "disrupts development of autonomic breathing and chemoreceptor circuitry",
                "domain": "genetic",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "obesity_leptin_dysregulation",
                "target": "leptin_central_resistance",
                "relation": "reduces effective hormonal stimulation of ventilatory drive",
                "domain": "metabolic",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "inflammatory_cytokine_load",
                "target": "inflammatory_drive_suppression",
                "relation": "adds cytokine-mediated suppression and distortion of respiratory control",
                "domain": "inflammatory",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "sleep_state_dependence",
                "target": "sleep_transition_drive_loss",
                "relation": "exposes state-dependent loss of wakefulness support for ventilation",
                "domain": "state",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "wake_drive_reserve",
                "target": "sleep_transition_drive_loss",
                "relation": "buffers loss of respiratory drive during sleep",
                "domain": "protective",
                "sleep_related_hypoventilation_change": "decreased",
            },
            {
                "source": "wake_drive_reserve",
                "target": "co2_chemosensitivity_failure",
                "relation": "partially compensates for reduced automatic ventilatory drive",
                "domain": "protective",
                "sleep_related_hypoventilation_change": "decreased",
            },
            {
                "source": "autonomic_breathing_circuit_dysdevelopment",
                "target": "co2_chemosensitivity_failure",
                "relation": "produces marked reduction in CO2 responsiveness",
                "domain": "autonomic",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "autonomic_breathing_circuit_dysdevelopment",
                "target": "central_ventilatory_drive_reduction",
                "relation": "reduces automatic ventilatory output during sleep",
                "domain": "autonomic",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "co2_chemosensitivity_failure",
                "target": "central_ventilatory_drive_reduction",
                "relation": "weakens ventilatory compensation to hypercapnia",
                "domain": "chemosensory",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "co2_chemosensitivity_failure",
                "target": "blunted_ventilatory_response",
                "relation": "manifests clinically as poor chemical-drive responsiveness",
                "domain": "symptom",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "hypoxic_response_blunting",
                "target": "blunted_ventilatory_response",
                "relation": "further reduces appropriate ventilatory response to gas derangement",
                "domain": "symptom",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "obesity_leptin_dysregulation",
                "target": "hypoxic_response_blunting",
                "relation": "adds obesity-linked weakening of ventilatory responsiveness",
                "domain": "metabolic",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "pulmonary_mechanical_burden",
                "target": "hypoxic_response_blunting",
                "relation": "raises the ventilatory challenge and worsens gas-handling reserve",
                "domain": "mechanical",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "leptin_central_resistance",
                "target": "central_ventilatory_drive_reduction",
                "relation": "reduces homeostatic support of respiratory drive",
                "domain": "metabolic",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "inflammatory_drive_suppression",
                "target": "central_ventilatory_drive_reduction",
                "relation": "suppresses central drive through inflammatory signaling",
                "domain": "inflammatory",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "sleep_transition_drive_loss",
                "target": "central_ventilatory_drive_reduction",
                "relation": "reduces state-dependent ventilatory support",
                "domain": "state",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "central_ventilatory_drive_reduction",
                "target": "nocturnal_gas_exchange_failure",
                "relation": "produces inadequate nocturnal ventilation",
                "domain": "respiratory",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "pulmonary_mechanical_burden",
                "target": "nocturnal_gas_exchange_failure",
                "relation": "adds mechanical load to nocturnal ventilation failure",
                "domain": "mechanical",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "nocturnal_gas_exchange_failure",
                "target": "sleep_related_hypoventilation",
                "relation": "directly manifests as sleep-related hypoventilation",
                "domain": "symptom",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "nocturnal_gas_exchange_failure",
                "target": "nocturnal_hypercapnia",
                "relation": "produces sleep-period carbon dioxide retention",
                "domain": "symptom",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "nocturnal_gas_exchange_failure",
                "target": "nocturnal_hypoxemia",
                "relation": "produces sleep-period oxygen desaturation",
                "domain": "symptom",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "autonomic_breathing_circuit_dysdevelopment",
                "target": "autonomic_respiratory_failure",
                "relation": "impairs automatic respiratory control during sleep",
                "domain": "symptom",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "nocturnal_hypoxemia",
                "target": "sleep_fragmentation",
                "relation": "destabilizes sleep continuity and architecture",
                "domain": "sleep",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "nocturnal_hypercapnia",
                "target": "sleep_fragmentation",
                "relation": "adds arousal pressure and disrupted sleep",
                "domain": "sleep",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "chronic_gas_exchange_burden",
                "target": "cerebral_autoregulation_stress",
                "relation": "stresses cerebral perfusion and autoregulation over time",
                "domain": "vascular",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "nocturnal_gas_exchange_failure",
                "target": "cerebral_autoregulation_stress",
                "relation": "repeated nocturnal derangement loads cerebral homeostasis",
                "domain": "vascular",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "cerebral_autoregulation_stress",
                "target": "chronic_hypoxic_hypercapnic_neural_burden",
                "relation": "contributes to structural and functional neural injury burden",
                "domain": "neural",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "sleep_fragmentation",
                "target": "chronic_hypoxic_hypercapnic_neural_burden",
                "relation": "adds chronic sleep-disruption burden to neural morbidity",
                "domain": "neural",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "chronic_hypoxic_hypercapnic_neural_burden",
                "target": "brain_aging_pressure",
                "relation": "increases long-term risk of gray-matter loss and brain aging",
                "domain": "neurodegeneration",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "brain_aging_pressure",
                "target": "brain_aging_vulnerability",
                "relation": "raises structural brain-aging vulnerability",
                "domain": "symptom",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "chronic_hypoxic_hypercapnic_neural_burden",
                "target": "neurocognitive_morbidity",
                "relation": "contributes to cognitive and behavioral morbidity",
                "domain": "symptom",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "co2_chemosensitivity_failure",
                "target": "brainstem_respiratory_proxy",
                "relation": "loads respiratory brainstem circuitry central to chemosensory drive",
                "domain": "circuit",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "central_ventilatory_drive_reduction",
                "target": "brainstem_respiratory_proxy",
                "relation": "burdens the automatic respiratory-control network",
                "domain": "circuit",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "leptin_central_resistance",
                "target": "hypothalamus_proxy",
                "relation": "loads hormonal-homeostatic respiratory modulation circuitry",
                "domain": "circuit",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "sleep_transition_drive_loss",
                "target": "hypothalamus_proxy",
                "relation": "loads sleep-state and homeostatic regulation circuitry",
                "domain": "circuit",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "dopaminergic_modulatory_instability",
                "target": "nucleus_accumbens",
                "relation": "loads exploratory dopaminergic arousal-modulation circuitry",
                "domain": "circuit",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "chronic_hypoxic_hypercapnic_neural_burden",
                "target": "hippocampus",
                "relation": "burdens hippocampal systems vulnerable to chronic hypoxic stress",
                "domain": "circuit",
                "sleep_related_hypoventilation_change": "increased",
            },
            {
                "source": "brain_aging_pressure",
                "target": "temporal_lobe_proxy",
                "relation": "loads temporal-lobe structures discussed as vulnerable to atrophy",
                "domain": "circuit",
                "sleep_related_hypoventilation_change": "increased",
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
            specs.extend(["julich 3.0.3", "julich 2.9", "julich"])
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
        generic_penalty = 1 if name in {"hippocampus", "hypothalamus", "brainstem", "temporal lobe"} else 0
        cyto_bonus_penalty = 0 if (
            "area " in name or "ca1" in name or "subiculum" in name or "dg" in name or "accumbens" in name
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
            Region node key, for example "hippocampus" or
            "brainstem_respiratory_proxy".
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
        phox2b_autonomic_defect: float = 0.0,
        obesity_leptin_dysregulation: float = 0.0,
        inflammatory_cytokine_load: float = 0.0,
        pulmonary_mechanical_burden: float = 0.0,
        sleep_state_dependence: float = 0.0,
        chronic_gas_exchange_burden: float = 0.0,
        dopaminergic_modulatory_instability: float = 0.0,
        wake_drive_reserve: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Calculation order:
        inputs -> latent biology -> regional state -> symptoms -> phenotype summaries

        Values are clipped to [0, 1]. Protective terms subtract where
        appropriate. The formulas are intentionally simple and interpretable,
        reflecting a chapter-faithful one-pass approximation rather than a
        validated model.
        """
        inputs = pd.Series(
            {
                "phox2b_autonomic_defect": self._clip01(phox2b_autonomic_defect),
                "obesity_leptin_dysregulation": self._clip01(obesity_leptin_dysregulation),
                "inflammatory_cytokine_load": self._clip01(inflammatory_cytokine_load),
                "pulmonary_mechanical_burden": self._clip01(pulmonary_mechanical_burden),
                "sleep_state_dependence": self._clip01(sleep_state_dependence),
                "chronic_gas_exchange_burden": self._clip01(chronic_gas_exchange_burden),
                "dopaminergic_modulatory_instability": self._clip01(dopaminergic_modulatory_instability),
                "wake_drive_reserve": self._clip01(wake_drive_reserve),
            },
            dtype=float,
        )

        latents = pd.Series(dtype=float)
        latents["autonomic_breathing_circuit_dysdevelopment"] = self._clip01(
            0.70 * inputs["phox2b_autonomic_defect"]
            + 0.08 * inputs["sleep_state_dependence"]
            + 0.06 * inputs["chronic_gas_exchange_burden"]
        )
        latents["co2_chemosensitivity_failure"] = self._clip01(
            0.34 * latents["autonomic_breathing_circuit_dysdevelopment"]
            + 0.20 * inputs["obesity_leptin_dysregulation"]
            + 0.12 * inputs["inflammatory_cytokine_load"]
            + 0.12 * inputs["sleep_state_dependence"]
            - 0.22 * inputs["wake_drive_reserve"]
        )
        latents["hypoxic_response_blunting"] = self._clip01(
            0.28 * latents["co2_chemosensitivity_failure"]
            + 0.18 * inputs["obesity_leptin_dysregulation"]
            + 0.18 * inputs["pulmonary_mechanical_burden"]
            + 0.10 * inputs["inflammatory_cytokine_load"]
            - 0.10 * inputs["wake_drive_reserve"]
        )
        latents["leptin_central_resistance"] = self._clip01(
            0.56 * inputs["obesity_leptin_dysregulation"]
            + 0.14 * inputs["inflammatory_cytokine_load"]
            + 0.06 * inputs["chronic_gas_exchange_burden"]
        )
        latents["inflammatory_drive_suppression"] = self._clip01(
            0.46 * inputs["inflammatory_cytokine_load"]
            + 0.16 * inputs["obesity_leptin_dysregulation"]
            + 0.08 * inputs["chronic_gas_exchange_burden"]
        )
        latents["sleep_transition_drive_loss"] = self._clip01(
            0.52 * inputs["sleep_state_dependence"]
            + 0.14 * latents["co2_chemosensitivity_failure"]
            + 0.08 * inputs["dopaminergic_modulatory_instability"]
            - 0.26 * inputs["wake_drive_reserve"]
        )
        latents["central_ventilatory_drive_reduction"] = self._clip01(
            0.18 * latents["autonomic_breathing_circuit_dysdevelopment"]
            + 0.18 * latents["co2_chemosensitivity_failure"]
            + 0.10 * latents["hypoxic_response_blunting"]
            + 0.14 * latents["leptin_central_resistance"]
            + 0.12 * latents["inflammatory_drive_suppression"]
            + 0.16 * latents["sleep_transition_drive_loss"]
            + 0.08 * inputs["pulmonary_mechanical_burden"]
            - 0.08 * inputs["wake_drive_reserve"]
        )
        latents["nocturnal_gas_exchange_failure"] = self._clip01(
            0.34 * latents["central_ventilatory_drive_reduction"]
            + 0.22 * inputs["pulmonary_mechanical_burden"]
            + 0.16 * latents["sleep_transition_drive_loss"]
            + 0.12 * latents["hypoxic_response_blunting"]
            + 0.10 * inputs["chronic_gas_exchange_burden"]
            - 0.10 * inputs["wake_drive_reserve"]
        )
        latents["cerebral_autoregulation_stress"] = self._clip01(
            0.34 * inputs["chronic_gas_exchange_burden"]
            + 0.26 * latents["nocturnal_gas_exchange_failure"]
            + 0.14 * latents["hypoxic_response_blunting"]
            + 0.08 * inputs["inflammatory_cytokine_load"]
        )
        latents["chronic_hypoxic_hypercapnic_neural_burden"] = self._clip01(
            0.30 * inputs["chronic_gas_exchange_burden"]
            + 0.22 * latents["nocturnal_gas_exchange_failure"]
            + 0.18 * latents["cerebral_autoregulation_stress"]
            + 0.10 * latents["sleep_transition_drive_loss"]
        )
        latents["brain_aging_pressure"] = self._clip01(
            0.32 * latents["chronic_hypoxic_hypercapnic_neural_burden"]
            + 0.20 * latents["cerebral_autoregulation_stress"]
            + 0.10 * inputs["obesity_leptin_dysregulation"]
            + 0.08 * inputs["inflammatory_cytokine_load"]
        )

        regional_state = pd.Series(dtype=float)
        regional_state["brainstem_respiratory_proxy"] = self._clip01(
            0.42 * latents["autonomic_breathing_circuit_dysdevelopment"]
            + 0.28 * latents["co2_chemosensitivity_failure"]
            + 0.18 * latents["central_ventilatory_drive_reduction"]
        )
        regional_state["hypothalamus_proxy"] = self._clip01(
            0.40 * latents["leptin_central_resistance"]
            + 0.24 * latents["sleep_transition_drive_loss"]
            + 0.16 * latents["inflammatory_drive_suppression"]
            + 0.08 * inputs["obesity_leptin_dysregulation"]
        )
        regional_state["nucleus_accumbens"] = self._clip01(
            0.34 * inputs["dopaminergic_modulatory_instability"]
            + 0.16 * latents["sleep_transition_drive_loss"]
            + 0.08 * inputs["sleep_state_dependence"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.38 * latents["chronic_hypoxic_hypercapnic_neural_burden"]
            + 0.24 * latents["cerebral_autoregulation_stress"]
            + 0.18 * latents["brain_aging_pressure"]
        )
        regional_state["temporal_lobe_proxy"] = self._clip01(
            0.30 * latents["brain_aging_pressure"]
            + 0.24 * latents["chronic_hypoxic_hypercapnic_neural_burden"]
            + 0.12 * inputs["chronic_gas_exchange_burden"]
        )

        symptoms = pd.Series(dtype=float)
        symptoms["blunted_ventilatory_response"] = self._clip01(
            0.40 * latents["co2_chemosensitivity_failure"]
            + 0.20 * latents["hypoxic_response_blunting"]
            + 0.18 * latents["central_ventilatory_drive_reduction"]
        )
        symptoms["sleep_related_hypoventilation"] = self._clip01(
            0.38 * latents["central_ventilatory_drive_reduction"]
            + 0.28 * latents["nocturnal_gas_exchange_failure"]
            + 0.14 * latents["sleep_transition_drive_loss"]
            + 0.08 * inputs["pulmonary_mechanical_burden"]
        )
        symptoms["nocturnal_hypercapnia"] = self._clip01(
            0.44 * latents["nocturnal_gas_exchange_failure"]
            + 0.18 * symptoms["sleep_related_hypoventilation"]
            + 0.10 * inputs["chronic_gas_exchange_burden"]
        )
        symptoms["nocturnal_hypoxemia"] = self._clip01(
            0.34 * latents["nocturnal_gas_exchange_failure"]
            + 0.22 * latents["hypoxic_response_blunting"]
            + 0.12 * inputs["pulmonary_mechanical_burden"]
            + 0.10 * inputs["chronic_gas_exchange_burden"]
        )
        symptoms["sleep_fragmentation"] = self._clip01(
            0.24 * latents["sleep_transition_drive_loss"]
            + 0.22 * symptoms["nocturnal_hypoxemia"]
            + 0.18 * symptoms["nocturnal_hypercapnia"]
            + 0.10 * inputs["dopaminergic_modulatory_instability"]
        )
        symptoms["autonomic_respiratory_failure"] = self._clip01(
            0.42 * latents["autonomic_breathing_circuit_dysdevelopment"]
            + 0.24 * latents["central_ventilatory_drive_reduction"]
            + 0.16 * latents["co2_chemosensitivity_failure"]
        )
        symptoms["neurocognitive_morbidity"] = self._clip01(
            0.28 * regional_state["hippocampus"]
            + 0.18 * symptoms["sleep_fragmentation"]
            + 0.16 * symptoms["nocturnal_hypoxemia"]
            + 0.14 * latents["brain_aging_pressure"]
        )
        symptoms["brain_aging_vulnerability"] = self._clip01(
            0.34 * latents["brain_aging_pressure"]
            + 0.22 * latents["chronic_hypoxic_hypercapnic_neural_burden"]
            + 0.12 * regional_state["temporal_lobe_proxy"]
        )

        phenotypes = pd.Series(dtype=float)
        phenotypes["cchs_autonomic_failure_profile"] = self._clip01(
            0.24 * inputs["phox2b_autonomic_defect"]
            + 0.22 * latents["autonomic_breathing_circuit_dysdevelopment"]
            + 0.18 * latents["co2_chemosensitivity_failure"]
            + 0.18 * symptoms["autonomic_respiratory_failure"]
            + 0.14 * symptoms["sleep_related_hypoventilation"]
        )
        phenotypes["obesity_hypoventilation_profile"] = self._clip01(
            0.24 * inputs["obesity_leptin_dysregulation"]
            + 0.20 * latents["leptin_central_resistance"]
            + 0.16 * latents["inflammatory_drive_suppression"]
            + 0.18 * symptoms["sleep_related_hypoventilation"]
            + 0.14 * symptoms["nocturnal_hypercapnia"]
        )
        phenotypes["state_dependent_control_failure_profile"] = self._clip01(
            0.24 * latents["sleep_transition_drive_loss"]
            + 0.22 * latents["central_ventilatory_drive_reduction"]
            + 0.18 * symptoms["blunted_ventilatory_response"]
            + 0.18 * symptoms["sleep_related_hypoventilation"]
            + 0.10 * inputs["sleep_state_dependence"]
        )
        phenotypes["chronic_brain_consequence_profile"] = self._clip01(
            0.24 * latents["chronic_hypoxic_hypercapnic_neural_burden"]
            + 0.20 * latents["brain_aging_pressure"]
            + 0.18 * symptoms["neurocognitive_morbidity"]
            + 0.18 * symptoms["brain_aging_vulnerability"]
            + 0.10 * regional_state["hippocampus"]
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }


if __name__ == "__main__":
    model = SleepRelatedHypoventilationModel()
    built = model.build()

    print("\n=== Nodes (head) ===")
    print(built["nodes"].head(12).to_string(index=False))

    print("\n=== Edge table (head) ===")
    print(built["edges"].head(12).to_string(index=False))

    print("\n=== Resolved regions ===")
    if model.region_objects:
        for key, region in model.region_objects.items():
            print(f"- {key}: {region.name}")
    else:
        print("No regions resolved in this runtime.")

    print("\n=== Example receptor availability ===")
    for key in model.region_candidates:
        df = built["receptors"].get(key, pd.DataFrame())
        print(f"- {key}: {0 if df is None else len(df)} rows")

    print("\n=== Example gene summaries ===")
    for key in model.region_candidates:
        df = built["genes"].get(key, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"\n{key}")
            print(df.head(5).to_string(index=False))
        else:
            print(f"- {key}: no gene rows returned")

    print("\n=== Circuit connectivity ===")
    circuit_df = built["circuit_connectivity"]
    if isinstance(circuit_df, pd.DataFrame) and not circuit_df.empty:
        print(circuit_df.head(15).to_string(index=False))
    else:
        print("No circuit connectivity rows available in this runtime.")

    print("\n=== Simulation example: obesity-linked SRH with chronic burden ===")
    sim = model.simulate(
        phox2b_autonomic_defect=0.10,
        obesity_leptin_dysregulation=0.85,
        inflammatory_cytokine_load=0.70,
        pulmonary_mechanical_burden=0.60,
        sleep_state_dependence=0.75,
        chronic_gas_exchange_burden=0.65,
        dopaminergic_modulatory_instability=0.20,
        wake_drive_reserve=0.25,
    )
    for section_name, section in sim.items():
        print(f"\n[{section_name}]")
        print(section.sort_values(ascending=False).to_string())

    # Optional coordinate assignment example:
    # print(model.assign_mni_point((0, -24, -20)).head())
