from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Circadian Rhythm Sleep-Wake Disorders-oriented panel:
# - core clock genes
# - monoaminergic arousal systems
# - plasticity / stress-response modulation
DEFAULT_GENE_PANEL = [
    "CLOCK",
    "ARNTL",   # BMAL1
    "PER1",
    "PER2",
    "PER3",
    "CRY1",
    "CRY2",
    "NR1D1",   # REV-ERBα
    "RORA",
    "SLC6A4",  # serotonin transporter
    "TPH2",    # serotonin synthesis
    "SLC6A2",  # norepinephrine transporter
    "DBH",     # norepinephrine synthesis
    "DRD2",    # dopamine receptor
    "COMT",    # dopamine metabolism
    "BDNF",    # plasticity
    "NR3C1",   # glucocorticoid receptor
    "FKBP5",   # stress responsivity
]


class CircadianRhythmSleepWakeDisordersModel:
    """
    Atlas-grounded mechanistic scaffold for Circadian Rhythm Sleep-Wake Disorders (CRSWDs).

    What it does:
      1) Resolves chapter-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates clock drift, entrainment failure, sleep-phase misalignment,
         daytime sleepiness, mood instability, and metabolic/autonomic burden.

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

        # The chapter is explicit about SCN/hypothalamus and monoaminergic nuclei,
        # but direct human neuroimaging is described as limited. Therefore several
        # nodes are intentionally "proxy" anchors.
        self.region_candidates: Dict[str, List[str]] = {
            "hypothalamus_proxy": [
                "hypothalamus",
                "suprachiasmatic nucleus",
                "SCN",
            ],
            "brainstem_arousal_proxy": [
                "locus coeruleus",
                "dorsal raphe",
                "raphe",
                "pons",
                "medulla",
                "brainstem",
            ],
            "pfc_cognition_proxy": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "prefrontal",
            ],
            "amygdala_emotion_proxy": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "clock_gene_vulnerability": "Inherited vulnerability in core clock machinery",
            "light_dark_zeitgeber_disruption": "Abnormal light timing or weak environmental entrainment cues",
            "shift_work_social_jetlag": "Behavioral misalignment between internal time and social schedule",
            "psychiatric_circadian_vulnerability": "Mood- and cognition-linked vulnerability to circadian disruption",
            "chronotherapy_support": "Protective resynchronizing inputs such as light and schedule stabilization",
        }

        self.latent_nodes: Dict[str, str] = {
            "clock_gene_feedback_instability": "Instability in transcription-translation clock feedback loops",
            "phase_response_abnormality": "Abnormal timing response to light and zeitgebers",
            "light_entrainment_failure": "Failure to properly entrain the internal clock to the 24-hour day",
            "monoaminergic_phase_desynchrony": "Circadian desynchronization of serotonin, norepinephrine, and dopamine systems",
            "sleep_state_transition_instability": "Instability in transitions between wake, NREM, and REM-related states",
            "peripheral_clock_desynchrony": "Mismatch between the SCN and peripheral tissue clocks",
            "circadian_network_misalignment": "Global misalignment of timekeeping across brain networks",
            "mood_cognition_network_instability": "Mood- and cognition-related consequences of temporal neural disorganization",
            "autonomic_metabolic_misalignment": "Autonomic and metabolic burden from circadian misalignment",
        }

        self.symptom_nodes: Dict[str, str] = {
            "sleep_phase_misalignment": "Misalignment of sleep timing with the desired or required schedule",
            "insomnia_at_desired_time": "Difficulty sleeping at socially or biologically intended times",
            "daytime_sleepiness": "Excessive sleepiness or low alertness when wakefulness is required",
            "mood_instability": "Mood lability or circadian-linked affective dysregulation",
            "cognitive_inefficiency": "Reduced attention, executive efficiency, and stable cognitive performance",
            "autonomic_dysregulation": "Autonomic instability arising from circadian misalignment",
            "metabolic_cardiovascular_strain": "Downstream metabolic/cardiovascular burden from persistent desynchrony",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "clock_gene_vulnerability",
                "target": "clock_gene_feedback_instability",
                "relation": "raises vulnerability in core molecular clock loops",
                "crswd_change": "increased susceptibility",
            },
            {
                "source": "clock_gene_vulnerability",
                "target": "phase_response_abnormality",
                "relation": "can alter intrinsic period length or light sensitivity",
                "crswd_change": "increased susceptibility",
            },
            {
                "source": "light_dark_zeitgeber_disruption",
                "target": "light_entrainment_failure",
                "relation": "reduces effective alignment of the internal clock to the environment",
                "crswd_change": "increased",
            },
            {
                "source": "shift_work_social_jetlag",
                "target": "peripheral_clock_desynchrony",
                "relation": "mismatches central and peripheral clocks",
                "crswd_change": "increased",
            },
            {
                "source": "shift_work_social_jetlag",
                "target": "circadian_network_misalignment",
                "relation": "creates persistent temporal mismatch across brain systems",
                "crswd_change": "increased",
            },
            {
                "source": "psychiatric_circadian_vulnerability",
                "target": "monoaminergic_phase_desynchrony",
                "relation": "amplifies sensitivity of mood/arousal systems to circadian disruption",
                "crswd_change": "increased",
            },
            {
                "source": "psychiatric_circadian_vulnerability",
                "target": "mood_cognition_network_instability",
                "relation": "amplifies downstream mood and cognition effects of misalignment",
                "crswd_change": "increased",
            },
            {
                "source": "chronotherapy_support",
                "target": "light_entrainment_failure",
                "relation": "improves entrainment to the external day",
                "crswd_change": "protective",
            },
            {
                "source": "chronotherapy_support",
                "target": "circadian_network_misalignment",
                "relation": "supports network resynchronization",
                "crswd_change": "protective",
            },
            {
                "source": "chronotherapy_support",
                "target": "sleep_phase_misalignment",
                "relation": "reduces phase misalignment burden",
                "crswd_change": "protective",
            },
            {
                "source": "clock_gene_feedback_instability",
                "target": "hypothalamus_proxy",
                "relation": "loads SCN/hypothalamic timekeeping systems",
                "crswd_change": "increased dysregulation",
            },
            {
                "source": "clock_gene_feedback_instability",
                "target": "circadian_network_misalignment",
                "relation": "destabilizes temporal organization across the circadian system",
                "crswd_change": "increased",
            },
            {
                "source": "phase_response_abnormality",
                "target": "light_entrainment_failure",
                "relation": "abnormal phase shifting undermines entrainment",
                "crswd_change": "increased",
            },
            {
                "source": "phase_response_abnormality",
                "target": "sleep_phase_misalignment",
                "relation": "shifts or destabilizes sleep phase relative to clock time",
                "crswd_change": "increased",
            },
            {
                "source": "light_entrainment_failure",
                "target": "sleep_phase_misalignment",
                "relation": "failure of entrainment drives phase misalignment",
                "crswd_change": "increased",
            },
            {
                "source": "circadian_network_misalignment",
                "target": "monoaminergic_phase_desynchrony",
                "relation": "misaligns arousal-promoting neurotransmitter rhythms",
                "crswd_change": "increased",
            },
            {
                "source": "circadian_network_misalignment",
                "target": "mood_cognition_network_instability",
                "relation": "temporal disorganization destabilizes mood and cognition networks",
                "crswd_change": "increased",
            },
            {
                "source": "monoaminergic_phase_desynchrony",
                "target": "brainstem_arousal_proxy",
                "relation": "burdens raphe/locus-coeruleus-like arousal systems",
                "crswd_change": "increased dysregulation",
            },
            {
                "source": "monoaminergic_phase_desynchrony",
                "target": "sleep_state_transition_instability",
                "relation": "destabilizes wake/NREM/REM transition chemistry",
                "crswd_change": "increased",
            },
            {
                "source": "monoaminergic_phase_desynchrony",
                "target": "mood_instability",
                "relation": "links circadian disruption to dysregulated mood",
                "crswd_change": "increased",
            },
            {
                "source": "monoaminergic_phase_desynchrony",
                "target": "insomnia_at_desired_time",
                "relation": "misaligned arousal systems can disturb intended sleep timing",
                "crswd_change": "increased",
            },
            {
                "source": "sleep_state_transition_instability",
                "target": "insomnia_at_desired_time",
                "relation": "unstable sleep-state transitions undermine consolidated sleep",
                "crswd_change": "increased",
            },
            {
                "source": "sleep_state_transition_instability",
                "target": "daytime_sleepiness",
                "relation": "unstable sleep architecture increases daytime sleepiness",
                "crswd_change": "increased",
            },
            {
                "source": "peripheral_clock_desynchrony",
                "target": "autonomic_metabolic_misalignment",
                "relation": "desynchronizes systemic physiology from the SCN",
                "crswd_change": "increased",
            },
            {
                "source": "peripheral_clock_desynchrony",
                "target": "metabolic_cardiovascular_strain",
                "relation": "persistent central-peripheral mismatch increases long-term strain",
                "crswd_change": "increased",
            },
            {
                "source": "mood_cognition_network_instability",
                "target": "pfc_cognition_proxy",
                "relation": "burdens suprapontine cognition-control networks",
                "crswd_change": "increased dysregulation",
            },
            {
                "source": "mood_cognition_network_instability",
                "target": "amygdala_emotion_proxy",
                "relation": "burdens emotion-linked circadian sensitivity",
                "crswd_change": "increased dysregulation",
            },
            {
                "source": "mood_cognition_network_instability",
                "target": "mood_instability",
                "relation": "drives circadian-linked affective instability",
                "crswd_change": "increased",
            },
            {
                "source": "mood_cognition_network_instability",
                "target": "cognitive_inefficiency",
                "relation": "drives unstable cognition and executive inefficiency",
                "crswd_change": "increased",
            },
            {
                "source": "autonomic_metabolic_misalignment",
                "target": "autonomic_dysregulation",
                "relation": "produces autonomic consequences of circadian desynchrony",
                "crswd_change": "increased",
            },
            {
                "source": "autonomic_metabolic_misalignment",
                "target": "metabolic_cardiovascular_strain",
                "relation": "supports downstream metabolic and cardiovascular burden",
                "crswd_change": "increased",
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
        """
        Best-effort string label for region-like or index-like objects.
        """
        if obj is None:
            return ""

        if isinstance(obj, str):
            return obj

        name = getattr(obj, "name", None)
        if isinstance(name, str) and name:
            return name

        label = getattr(obj, "label", None)
        if isinstance(label, str) and label:
            return label

        identifier = getattr(obj, "identifier", None)
        if isinstance(identifier, str) and identifier:
            return identifier

        if isinstance(obj, tuple):
            parts = [str(x) for x in obj if x is not None and str(x) != ""]
            return " | ".join(parts)

        return str(obj)

    @staticmethod
    def _clip01(value: Any) -> float:
        """
        Clamp a numeric input to the closed interval [0, 1].
        Non-numeric or missing values fall back to 0.0.
        """
        try:
            x = float(value)
        except Exception:
            return 0.0

        if pd.isna(x):
            return 0.0

        return max(0.0, min(1.0, x))

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

        out = []
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
            "hypothalamus",
            "brainstem",
            "medulla",
            "pons",
            "amygdala",
            "prefrontal cortex",
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
            "hypothalamus_proxy",
            "brainstem_arousal_proxy",
            "pfc_cognition_proxy",
            "amygdala_emotion_proxy",
        ),
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
                    "description": "Atlas-backed CRSWD circuit node",
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
        clock_gene_vulnerability: float,
        light_dark_zeitgeber_disruption: float,
        shift_work_social_jetlag: float,
        psychiatric_circadian_vulnerability: float,
        chronotherapy_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while chronotherapy_support is protective.
        """
        g = self._clip01(clock_gene_vulnerability)
        l = self._clip01(light_dark_zeitgeber_disruption)
        s = self._clip01(shift_work_social_jetlag)
        p = self._clip01(psychiatric_circadian_vulnerability)
        t = self._clip01(chronotherapy_support)

        # Latent biology
        clock_gene_feedback_instability = self._clip01(
            0.45 * g - 0.10 * t
        )
        phase_response_abnormality = self._clip01(
            0.35 * g + 0.20 * l - 0.15 * t
        )
        light_entrainment_failure = self._clip01(
            0.40 * l + 0.25 * phase_response_abnormality + 0.10 * clock_gene_feedback_instability - 0.20 * t
        )
        peripheral_clock_desynchrony = self._clip01(
            0.40 * s + 0.20 * light_entrainment_failure + 0.10 * g - 0.10 * t
        )
        circadian_network_misalignment = self._clip01(
            0.30 * clock_gene_feedback_instability
            + 0.25 * light_entrainment_failure
            + 0.20 * s
            - 0.20 * t
        )
        monoaminergic_phase_desynchrony = self._clip01(
            0.35 * circadian_network_misalignment
            + 0.20 * p
            + 0.10 * l
            - 0.15 * t
        )
        sleep_state_transition_instability = self._clip01(
            0.35 * monoaminergic_phase_desynchrony
            + 0.20 * circadian_network_misalignment
            + 0.10 * l
            - 0.15 * t
        )
        mood_cognition_network_instability = self._clip01(
            0.35 * circadian_network_misalignment
            + 0.25 * monoaminergic_phase_desynchrony
            + 0.15 * p
            - 0.15 * t
        )
        autonomic_metabolic_misalignment = self._clip01(
            0.35 * peripheral_clock_desynchrony
            + 0.20 * circadian_network_misalignment
            + 0.10 * s
            - 0.10 * t
        )

        # Regional state proxies
        hypothalamus_proxy = self._clip01(
            0.45 * clock_gene_feedback_instability
            + 0.25 * circadian_network_misalignment
            - 0.15 * t
        )
        brainstem_arousal_proxy = self._clip01(
            0.45 * monoaminergic_phase_desynchrony
            + 0.20 * sleep_state_transition_instability
            - 0.10 * t
        )
        pfc_cognition_proxy = self._clip01(
            0.45 * mood_cognition_network_instability
            + 0.15 * circadian_network_misalignment
            - 0.15 * t
        )
        amygdala_emotion_proxy = self._clip01(
            0.40 * mood_cognition_network_instability
            + 0.15 * monoaminergic_phase_desynchrony
            - 0.10 * t
        )

        # Symptoms
        sleep_phase_misalignment = self._clip01(
            0.35 * light_entrainment_failure
            + 0.30 * phase_response_abnormality
            + 0.15 * hypothalamus_proxy
            - 0.20 * t
        )
        insomnia_at_desired_time = self._clip01(
            0.30 * sleep_phase_misalignment
            + 0.25 * sleep_state_transition_instability
            + 0.15 * brainstem_arousal_proxy
            - 0.15 * t
        )
        daytime_sleepiness = self._clip01(
            0.35 * insomnia_at_desired_time
            + 0.25 * sleep_state_transition_instability
            + 0.15 * circadian_network_misalignment
            - 0.10 * t
        )
        mood_instability = self._clip01(
            0.35 * mood_cognition_network_instability
            + 0.20 * amygdala_emotion_proxy
            + 0.10 * monoaminergic_phase_desynchrony
            - 0.10 * t
        )
        cognitive_inefficiency = self._clip01(
            0.35 * pfc_cognition_proxy
            + 0.20 * mood_cognition_network_instability
            + 0.15 * daytime_sleepiness
            - 0.10 * t
        )
        autonomic_dysregulation = self._clip01(
            0.35 * autonomic_metabolic_misalignment
            + 0.20 * hypothalamus_proxy
            + 0.10 * brainstem_arousal_proxy
            - 0.10 * t
        )
        metabolic_cardiovascular_strain = self._clip01(
            0.45 * peripheral_clock_desynchrony
            + 0.25 * autonomic_metabolic_misalignment
            + 0.10 * s
            - 0.10 * t
        )

        return {
            "inputs": pd.Series(
                {
                    "clock_gene_vulnerability": g,
                    "light_dark_zeitgeber_disruption": l,
                    "shift_work_social_jetlag": s,
                    "psychiatric_circadian_vulnerability": p,
                    "chronotherapy_support": t,
                }
            ),
            "latents": pd.Series(
                {
                    "circadian_network_misalignment": circadian_network_misalignment,
                    "monoaminergic_phase_desynchrony": monoaminergic_phase_desynchrony,
                    "light_entrainment_failure": light_entrainment_failure,
                    "peripheral_clock_desynchrony": peripheral_clock_desynchrony,
                    "mood_cognition_network_instability": mood_cognition_network_instability,
                    "autonomic_metabolic_misalignment": autonomic_metabolic_misalignment,
                    "sleep_state_transition_instability": sleep_state_transition_instability,
                    "phase_response_abnormality": phase_response_abnormality,
                    "clock_gene_feedback_instability": clock_gene_feedback_instability,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "hypothalamus_proxy": hypothalamus_proxy,
                    "brainstem_arousal_proxy": brainstem_arousal_proxy,
                    "pfc_cognition_proxy": pfc_cognition_proxy,
                    "amygdala_emotion_proxy": amygdala_emotion_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "sleep_phase_misalignment": sleep_phase_misalignment,
                    "insomnia_at_desired_time": insomnia_at_desired_time,
                    "daytime_sleepiness": daytime_sleepiness,
                    "mood_instability": mood_instability,
                    "cognitive_inefficiency": cognitive_inefficiency,
                    "autonomic_dysregulation": autonomic_dysregulation,
                    "metabolic_cardiovascular_strain": metabolic_cardiovascular_strain,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "phase_delay_advance_profile": self._clip01(
                        0.45 * sleep_phase_misalignment
                        + 0.25 * light_entrainment_failure
                        + 0.20 * phase_response_abnormality
                    ),
                    "shift_work_misalignment_profile": self._clip01(
                        0.45 * peripheral_clock_desynchrony
                        + 0.25 * daytime_sleepiness
                        + 0.20 * metabolic_cardiovascular_strain
                    ),
                    "mood_linked_circadian_profile": self._clip01(
                        0.45 * mood_instability
                        + 0.25 * mood_cognition_network_instability
                        + 0.20 * monoaminergic_phase_desynchrony
                    ),
                    "non_entrained_clock_profile": self._clip01(
                        0.45 * light_entrainment_failure
                        + 0.25 * circadian_network_misalignment
                        + 0.20 * insomnia_at_desired_time
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-8, -2, -8)).head(10)
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
    model = CircadianRhythmSleepWakeDisordersModel()

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
        bundle["edges"][["source", "target", "relation", "crswd_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in [
        "hypothalamus_proxy",
        "brainstem_arousal_proxy",
        "pfc_cognition_proxy",
        "amygdala_emotion_proxy",
    ]:
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
        clock_gene_vulnerability=0.70,
        light_dark_zeitgeber_disruption=0.80,
        shift_work_social_jetlag=0.75,
        psychiatric_circadian_vulnerability=0.60,
        chronotherapy_support=0.20,
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
    # print(model.assign_mni_point((-8, -2, -8)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("hypothalamus").to_string(index=False))
    # print(model.suggest_regions("raphe").to_string(index=False))
