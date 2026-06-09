from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Central Sleep Apnea-oriented panel:
# - serotonin / respiratory motor output
# - GABA / glycine-like inhibitory tone
# - glutamatergic plasticity / AMPA-LTP-related stability
# - stress / autonomic regulation
DEFAULT_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor
    "TPH2",     # serotonin synthesis
    "GABRA2",   # GABA-A receptor
    "GABRB2",   # GABA-A receptor
    "GAD1",     # GABA synthesis
    "GLRA1",    # glycine receptor alpha-1
    "GLRB",     # glycine receptor beta
    "GRIN2B",   # NMDA receptor subunit
    "GRIA1",    # AMPA receptor subunit
    "GRIA2",    # AMPA receptor subunit
    "SLC1A1",   # glutamate transport
    "BDNF",     # plasticity
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
    "CRHR1",    # CRH signaling
]


class CentralSleepApneaModel:
    """
    Atlas-grounded mechanistic scaffold for Central Sleep Apnea (CSA).

    What it does:
      1) Resolves CSA-relevant anatomy to atlas regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates respiratory instability, central apneas, sleep fragmentation,
         autonomic/interoceptive dysregulation, and daytime sleepiness.

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
        self.space = self.atlas.spaces.get(space_spec)
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        # Brainstem/hypothalamus labels may not resolve cleanly in every siibra setup,
        # so these are broad proxies with fallbacks.
        self.region_candidates: Dict[str, List[str]] = {
            "brainstem_raphe_proxy": [
                "raphe",
                "raphe nuclei",
                "medulla",
                "pons",
                "brainstem",
            ],
            "hypothalamus_proxy": [
                "hypothalamus",
                "suprachiasmatic nucleus",
                "SCN",
            ],
            "insula": [
                "Area Id2 (Insula) left",
                "Area Id3 (Insula) left",
                "Area Id4 (Insula) left",
                "insula",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "sleep_state_serotonin_drop": "Sleep-related reduction in raphe serotonergic firing",
            "gaba_glycine_inhibitory_load": "Excess inhibitory tone or medication-related respiratory suppression",
            "circadian_hypothalamic_disruption": "SCN/hypothalamic circadian mismatch affecting respiratory regulation",
            "white_matter_disconnectivity": "Reduced integrity of pathways linking brainstem, chemosensory, and suprapontine systems",
            "genetic_channel_plasticity_vulnerability": "Genetic vulnerability in channels, receptors, or plasticity mechanisms",
            "homeostatic_support": "Protective compensatory and treatment-related stabilization",
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_respiratory_underdrive": "Reduced serotonergic excitatory support to respiratory output during sleep",
            "inhibitory_brainstem_suppression": "Excess inhibitory pressure on respiratory pattern generation",
            "respiratory_motoneuron_underdrive": "Reduced downstream drive to respiratory motor output",
            "chemosensory_response_deficit": "Blunted response to chemical respiratory stimuli",
            "apneic_threshold_instability": "Drive oscillates near the apneic threshold during sleep",
            "circadian_hypothalamic_mismatch": "Circadian-state mismatch destabilizing respiratory control",
            "synaptic_plasticity_failure": "Impaired adaptive plasticity in respiratory control circuits",
            "network_disconnectivity": "Disconnection within broader respiratory/autonomic control pathways",
            "autonomic_interoceptive_dysregulation": "Disrupted autonomic and interoceptive regulation",
        }

        self.symptom_nodes: Dict[str, str] = {
            "central_apnea_events": "Sleep-related cessation of respiratory effort",
            "unstable_ventilatory_rhythm": "Rhythmically unstable breathing during sleep",
            "chemosensory_blunting": "Reduced effective chemosensory responsiveness",
            "arousal_fragmentation": "Repeated arousals and sleep fragmentation",
            "autonomic_instability": "Autonomic instability linked to respiratory dysregulation",
            "daytime_sleepiness": "Daytime consequences of unstable sleep and breathing",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "sleep_state_serotonin_drop",
                "target": "serotonergic_respiratory_underdrive",
                "relation": "reduces excitatory serotonergic respiratory support during sleep",
                "csa_change": "increased",
            },
            {
                "source": "sleep_state_serotonin_drop",
                "target": "respiratory_motoneuron_underdrive",
                "relation": "can lower respiratory motor output below threshold in vulnerable individuals",
                "csa_change": "increased",
            },
            {
                "source": "gaba_glycine_inhibitory_load",
                "target": "inhibitory_brainstem_suppression",
                "relation": "increases inhibitory suppression of respiratory pattern generation",
                "csa_change": "increased",
            },
            {
                "source": "circadian_hypothalamic_disruption",
                "target": "circadian_hypothalamic_mismatch",
                "relation": "disrupts state-appropriate respiratory timing and homeostasis",
                "csa_change": "increased",
            },
            {
                "source": "white_matter_disconnectivity",
                "target": "network_disconnectivity",
                "relation": "reduces coherent communication across the respiratory control network",
                "csa_change": "increased",
            },
            {
                "source": "genetic_channel_plasticity_vulnerability",
                "target": "synaptic_plasticity_failure",
                "relation": "raises vulnerability in excitability and adaptive plasticity mechanisms",
                "csa_change": "increased susceptibility",
            },
            {
                "source": "genetic_channel_plasticity_vulnerability",
                "target": "chemosensory_response_deficit",
                "relation": "may impair chemical-respiratory response thresholds",
                "csa_change": "increased susceptibility",
            },
            {
                "source": "homeostatic_support",
                "target": "apneic_threshold_instability",
                "relation": "buffers instability around the apneic threshold",
                "csa_change": "protective",
            },
            {
                "source": "homeostatic_support",
                "target": "arousal_fragmentation",
                "relation": "buffers repeated respiratory-related arousals",
                "csa_change": "protective",
            },
            {
                "source": "homeostatic_support",
                "target": "daytime_sleepiness",
                "relation": "reduces daytime burden from sleep disruption",
                "csa_change": "protective",
            },
            {
                "source": "serotonergic_respiratory_underdrive",
                "target": "brainstem_raphe_proxy",
                "relation": "reflects impaired serotonergic modulation of brainstem respiratory control",
                "csa_change": "increased dysregulation",
            },
            {
                "source": "serotonergic_respiratory_underdrive",
                "target": "respiratory_motoneuron_underdrive",
                "relation": "reduces drive to respiratory motor output",
                "csa_change": "increased",
            },
            {
                "source": "inhibitory_brainstem_suppression",
                "target": "brainstem_raphe_proxy",
                "relation": "suppresses rhythmic respiratory output in brainstem circuitry",
                "csa_change": "increased dysregulation",
            },
            {
                "source": "inhibitory_brainstem_suppression",
                "target": "respiratory_motoneuron_underdrive",
                "relation": "suppresses downstream respiratory motor tone",
                "csa_change": "increased",
            },
            {
                "source": "circadian_hypothalamic_mismatch",
                "target": "hypothalamus_proxy",
                "relation": "burdens hypothalamic circadian and homeostatic regulation",
                "csa_change": "increased dysregulation",
            },
            {
                "source": "circadian_hypothalamic_mismatch",
                "target": "unstable_ventilatory_rhythm",
                "relation": "promotes mismatch between sleep state and respiratory drive",
                "csa_change": "increased",
            },
            {
                "source": "synaptic_plasticity_failure",
                "target": "chemosensory_response_deficit",
                "relation": "impairs adaptive respiratory plasticity",
                "csa_change": "increased",
            },
            {
                "source": "synaptic_plasticity_failure",
                "target": "apneic_threshold_instability",
                "relation": "reduces stable adaptation to metabolic demand",
                "csa_change": "increased",
            },
            {
                "source": "network_disconnectivity",
                "target": "brainstem_raphe_proxy",
                "relation": "weakens integration with respiratory centers",
                "csa_change": "increased dysregulation",
            },
            {
                "source": "network_disconnectivity",
                "target": "hypothalamus_proxy",
                "relation": "weakens circadian-autonomic coordination",
                "csa_change": "increased dysregulation",
            },
            {
                "source": "network_disconnectivity",
                "target": "insula",
                "relation": "weakens interoceptive/autonomic regulation pathways",
                "csa_change": "increased dysregulation",
            },
            {
                "source": "network_disconnectivity",
                "target": "amygdala",
                "relation": "weakens stable modulation of emotional-autonomic responses",
                "csa_change": "increased dysregulation",
            },
            {
                "source": "network_disconnectivity",
                "target": "autonomic_interoceptive_dysregulation",
                "relation": "disconnectivity destabilizes autonomic/interoceptive regulation",
                "csa_change": "increased",
            },
            {
                "source": "autonomic_interoceptive_dysregulation",
                "target": "insula",
                "relation": "loads insular interoceptive-autonomic processing",
                "csa_change": "increased dysregulation",
            },
            {
                "source": "autonomic_interoceptive_dysregulation",
                "target": "amygdala",
                "relation": "loads autonomic-affective modulation of respiration",
                "csa_change": "increased dysregulation",
            },
            {
                "source": "autonomic_interoceptive_dysregulation",
                "target": "autonomic_instability",
                "relation": "promotes autonomic irregularity around respiratory events",
                "csa_change": "increased",
            },
            {
                "source": "chemosensory_response_deficit",
                "target": "central_apnea_events",
                "relation": "poor chemical responsiveness promotes apnea events",
                "csa_change": "increased",
            },
            {
                "source": "chemosensory_response_deficit",
                "target": "chemosensory_blunting",
                "relation": "produces blunted effective chemosensory reactivity",
                "csa_change": "increased",
            },
            {
                "source": "respiratory_motoneuron_underdrive",
                "target": "central_apnea_events",
                "relation": "reduces respiratory effort during sleep",
                "csa_change": "increased",
            },
            {
                "source": "apneic_threshold_instability",
                "target": "central_apnea_events",
                "relation": "oscillation near threshold promotes recurrent apneas",
                "csa_change": "increased",
            },
            {
                "source": "apneic_threshold_instability",
                "target": "unstable_ventilatory_rhythm",
                "relation": "drives rhythmic instability during sleep",
                "csa_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "arousal_fragmentation",
                "relation": "autonomic-affective respiratory instability can worsen sleep continuity",
                "csa_change": "increased",
            },
            {
                "source": "central_apnea_events",
                "target": "arousal_fragmentation",
                "relation": "recurrent apneas fragment sleep",
                "csa_change": "increased",
            },
            {
                "source": "arousal_fragmentation",
                "target": "daytime_sleepiness",
                "relation": "fragmented sleep produces daytime burden",
                "csa_change": "increased",
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
            "amygdala",
            "insula",
            "hypothalamus",
            "brainstem",
            "medulla",
            "pons",
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
        """
        Helper for refining proxy searches against the atlas.
        Useful for tuning hypothalamic or brainstem proxy choices.
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
            "brainstem_raphe_proxy",
            "hypothalamus_proxy",
            "insula",
            "amygdala",
        ),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the CSA circuit.
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
                    "description": "Atlas-backed CSA circuit node",
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
        sleep_state_serotonin_drop: float,
        gaba_glycine_inhibitory_load: float,
        circadian_hypothalamic_disruption: float,
        white_matter_disconnectivity: float,
        genetic_channel_plasticity_vulnerability: float,
        homeostatic_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while homeostatic_support is protective.
        """
        s = self._clip01(sleep_state_serotonin_drop)
        g = self._clip01(gaba_glycine_inhibitory_load)
        c = self._clip01(circadian_hypothalamic_disruption)
        w = self._clip01(white_matter_disconnectivity)
        v = self._clip01(genetic_channel_plasticity_vulnerability)
        h = self._clip01(homeostatic_support)

        # Latent biology
        serotonergic_respiratory_underdrive = self._clip01(
            0.45 * s + 0.15 * c - 0.15 * h
        )
        inhibitory_brainstem_suppression = self._clip01(
            0.50 * g + 0.10 * s - 0.15 * h
        )
        synaptic_plasticity_failure = self._clip01(
            0.35 * v + 0.20 * c + 0.10 * w - 0.10 * h
        )
        network_disconnectivity = self._clip01(
            0.50 * w + 0.20 * synaptic_plasticity_failure
        )
        circadian_hypothalamic_mismatch = self._clip01(
            0.45 * c + 0.15 * v - 0.15 * h
        )
        chemosensory_response_deficit = self._clip01(
            0.30 * serotonergic_respiratory_underdrive
            + 0.25 * synaptic_plasticity_failure
            + 0.15 * network_disconnectivity
            - 0.10 * h
        )
        respiratory_motoneuron_underdrive = self._clip01(
            0.35 * serotonergic_respiratory_underdrive
            + 0.35 * inhibitory_brainstem_suppression
            + 0.10 * circadian_hypothalamic_mismatch
            - 0.10 * h
        )
        apneic_threshold_instability = self._clip01(
            0.35 * chemosensory_response_deficit
            + 0.30 * respiratory_motoneuron_underdrive
            + 0.15 * circadian_hypothalamic_mismatch
            - 0.15 * h
        )
        autonomic_interoceptive_dysregulation = self._clip01(
            0.30 * network_disconnectivity
            + 0.20 * circadian_hypothalamic_mismatch
            + 0.10 * s
            - 0.10 * h
        )

        # Regional state proxies
        brainstem_raphe_proxy = self._clip01(
            0.40 * serotonergic_respiratory_underdrive
            + 0.35 * inhibitory_brainstem_suppression
            + 0.20 * network_disconnectivity
            - 0.10 * h
        )
        hypothalamus_proxy = self._clip01(
            0.45 * circadian_hypothalamic_mismatch
            + 0.20 * network_disconnectivity
            - 0.10 * h
        )
        insula = self._clip01(
            0.45 * autonomic_interoceptive_dysregulation
            + 0.20 * network_disconnectivity
        )
        amygdala = self._clip01(
            0.35 * autonomic_interoceptive_dysregulation
            + 0.15 * circadian_hypothalamic_mismatch
        )

        # Symptoms
        central_apnea_events = self._clip01(
            0.40 * apneic_threshold_instability
            + 0.25 * respiratory_motoneuron_underdrive
            + 0.15 * chemosensory_response_deficit
            - 0.10 * h
        )
        unstable_ventilatory_rhythm = self._clip01(
            0.35 * apneic_threshold_instability
            + 0.25 * circadian_hypothalamic_mismatch
            + 0.20 * network_disconnectivity
            - 0.10 * h
        )
        chemosensory_blunting = self._clip01(
            0.50 * chemosensory_response_deficit + 0.20 * insula
        )
        autonomic_instability = self._clip01(
            0.35 * amygdala + 0.25 * insula + 0.15 * network_disconnectivity
        )
        arousal_fragmentation = self._clip01(
            0.40 * central_apnea_events
            + 0.20 * autonomic_instability
            + 0.10 * unstable_ventilatory_rhythm
            - 0.15 * h
        )
        daytime_sleepiness = self._clip01(
            0.45 * arousal_fragmentation + 0.15 * unstable_ventilatory_rhythm
        )

        return {
            "inputs": pd.Series(
                {
                    "sleep_state_serotonin_drop": s,
                    "gaba_glycine_inhibitory_load": g,
                    "circadian_hypothalamic_disruption": c,
                    "white_matter_disconnectivity": w,
                    "genetic_channel_plasticity_vulnerability": v,
                    "homeostatic_support": h,
                }
            ),
            "latents": pd.Series(
                {
                    "apneic_threshold_instability": apneic_threshold_instability,
                    "respiratory_motoneuron_underdrive": respiratory_motoneuron_underdrive,
                    "chemosensory_response_deficit": chemosensory_response_deficit,
                    "serotonergic_respiratory_underdrive": serotonergic_respiratory_underdrive,
                    "inhibitory_brainstem_suppression": inhibitory_brainstem_suppression,
                    "circadian_hypothalamic_mismatch": circadian_hypothalamic_mismatch,
                    "network_disconnectivity": network_disconnectivity,
                    "synaptic_plasticity_failure": synaptic_plasticity_failure,
                    "autonomic_interoceptive_dysregulation": autonomic_interoceptive_dysregulation,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "brainstem_raphe_proxy": brainstem_raphe_proxy,
                    "hypothalamus_proxy": hypothalamus_proxy,
                    "insula": insula,
                    "amygdala": amygdala,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "central_apnea_events": central_apnea_events,
                    "unstable_ventilatory_rhythm": unstable_ventilatory_rhythm,
                    "arousal_fragmentation": arousal_fragmentation,
                    "daytime_sleepiness": daytime_sleepiness,
                    "chemosensory_blunting": chemosensory_blunting,
                    "autonomic_instability": autonomic_instability,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "serotonergic_underdrive_profile": self._clip01(
                        0.45 * serotonergic_respiratory_underdrive
                        + 0.25 * respiratory_motoneuron_underdrive
                        + 0.15 * central_apnea_events
                    ),
                    "inhibitory_oversuppression_profile": self._clip01(
                        0.45 * inhibitory_brainstem_suppression
                        + 0.25 * central_apnea_events
                        + 0.15 * arousal_fragmentation
                    ),
                    "circadian_hypothalamic_profile": self._clip01(
                        0.45 * circadian_hypothalamic_mismatch
                        + 0.25 * unstable_ventilatory_rhythm
                        + 0.15 * daytime_sleepiness
                    ),
                    "disconnectivity_profile": self._clip01(
                        0.45 * network_disconnectivity
                        + 0.25 * chemosensory_blunting
                        + 0.15 * autonomic_instability
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-34, -12, 4)).head(10)
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
    model = CentralSleepApneaModel()

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
        bundle["edges"][["source", "target", "relation", "csa_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["brainstem_raphe_proxy", "hypothalamus_proxy", "insula", "amygdala"]:
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
        sleep_state_serotonin_drop=0.80,
        gaba_glycine_inhibitory_load=0.55,
        circadian_hypothalamic_disruption=0.70,
        white_matter_disconnectivity=0.50,
        genetic_channel_plasticity_vulnerability=0.45,
        homeostatic_support=0.20,
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
    # print(model.assign_mni_point((-34, -12, 4)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("hypothalamus").to_string(index=False))
    # print(model.suggest_regions("brainstem").to_string(index=False))
