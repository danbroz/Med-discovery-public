from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Stuttering-oriented panel:
# - dopamine / basal-ganglia timing and gating
# - norepinephrine / attentional modulation
# - GABA / glutamate timing balance
# - synaptic release / plasticity
# - known stuttering-linked trafficking genes
DEFAULT_GENE_PANEL = [
    "DRD2",    # dopamine receptor
    "SLC6A3",  # dopamine transporter
    "COMT",    # dopamine metabolism
    "SLC6A2",  # norepinephrine transporter
    "DBH",     # norepinephrine synthesis step
    "GABRA2",  # GABA-A receptor
    "GABRB2",  # GABA-A receptor
    "GAD1",    # GABA synthesis
    "GRIN2B",  # NMDA receptor subunit
    "SLC1A1",  # glutamate transport
    "SV2A",    # synaptic vesicle release target
    "BDNF",    # plasticity
    "GNPTAB",  # stuttering-linked trafficking gene
    "GNPTG",   # stuttering-linked trafficking gene
    "NAGPA",   # stuttering-linked trafficking gene
]


class ChildhoodOnsetFluencyDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Childhood-Onset Fluency Disorder (Stuttering).

    What it does:
      1) Resolves stuttering-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates speech-motor timing breakdown, initiation blocks,
         auditory-motor mismatch, and fluency variability.

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

        # Chapter-consistent speech network:
        # left IFG/Broca proxy, bilateral STG, speech-motor/premotor proxy,
        # basal-ganglia timing proxy, optional corpus callosum proxy.
        self.region_candidates: Dict[str, List[str]] = {
            "ifg_broca_left": [
                "Area 44 (IFG) left",
                "Area 45 (IFG) left",
                "broca left",
                "inferior frontal gyrus left",
                "ifg left",
            ],
            "stg_left": [
                "Area TE 2.1 (STG) left",
                "Area TE 1.1 (STG) left",
                "Area TE 1.0 (HESCHL) left",
                "STG left",
                "superior temporal gyrus left",
            ],
            "stg_right": [
                "Area TE 2.1 (STG) right",
                "Area TE 1.1 (STG) right",
                "Area TE 1.0 (HESCHL) right",
                "STG right",
                "superior temporal gyrus right",
            ],
            "premotor_speech_left": [
                "Area 6v1 (PreCG) left",
                "Area 6v2 (PreCG) left",
                "Area 6d1 (PreCG) left",
                "premotor left",
                "precentral left",
            ],
            "basal_ganglia_proxy": [
                "caudate",
                "putamen",
                "accumbens",
                "striatum",
                "basal ganglia",
            ],
            "corpus_callosum_proxy": [
                "corpus callosum",
                "callosal",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": "Polygenic and rare-variant liability for stuttering",
            "essence_comorbidity_load": "ESSENCE-style neurodevelopmental overlap such as ADHD/anxiety burden",
            "dopaminergic_perturbation": "Endogenous or medication-linked dopaminergic dysregulation",
            "excitatory_inhibitory_instability": "GABA/glutamate timing instability or neuronal hyperexcitability",
            "developmental_white_matter_disconnection": "Reduced tract integrity in arcuate/SLF/callosal pathways",
            "fluency_support": "Protective support, treatment, and compensatory stabilization",
        }

        self.latent_nodes: Dict[str, str] = {
            "polygenic_speech_motor_liability": "Inherited vulnerability in speech-motor control systems",
            "dopaminergic_timing_dysregulation": "Basal-ganglia timing dysregulation linked to dopamine",
            "basal_ganglia_gating_instability": "Impaired initiation and sequencing of speech motor programs",
            "synaptic_release_instability": "Imprecise neurotransmitter release and synaptic timing",
            "excitatory_inhibitory_timing_imbalance": "Unstable E/I balance disrupting temporal precision",
            "speech_network_disconnectivity": "Disconnection across frontal-temporal speech networks",
            "auditory_motor_monitoring_mismatch": "Impaired coordination between speech output and auditory monitoring",
            "interhemispheric_timing_inefficiency": "Reduced cross-hemispheric timing coordination",
            "attention_impulse_control_modulation": "Attention and executive control influence on fluency output",
            "speech_motor_timing_breakdown": "Failure of precise temporal coordination in speech production",
        }

        self.symptom_nodes: Dict[str, str] = {
            "repetitions_blocks_prolongations": "Core disfluencies of repetitions, blocks, and prolongations",
            "speech_initiation_failure": "Difficulty initiating a fluent motor program",
            "auditory_feedback_instability": "Instability in monitoring self-produced speech",
            "fluency_variability": "Marked variability across conditions and states",
            "secondary_tension_behaviors": "Tension and secondary compensatory motor behaviors",
            "attention_related_fluency_modulation": "Fluency changes with attentional/executive modulation",
            "psychosocial_impact": "Functional and emotional burden of persistent disfluency",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "polygenic_speech_motor_liability",
                "relation": "raises inherited vulnerability for speech disfluency",
                "stuttering_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "dopaminergic_timing_dysregulation",
                "relation": "raises vulnerability of basal-ganglia timing systems",
                "stuttering_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "synaptic_release_instability",
                "relation": "raises vulnerability in synaptic release and trafficking mechanisms",
                "stuttering_change": "increased susceptibility",
            },
            {
                "source": "essence_comorbidity_load",
                "target": "attention_impulse_control_modulation",
                "relation": "loads shared neurodevelopmental executive-control influences",
                "stuttering_change": "increased",
            },
            {
                "source": "essence_comorbidity_load",
                "target": "speech_network_disconnectivity",
                "relation": "shared neurodevelopmental load can amplify network inefficiency",
                "stuttering_change": "increased",
            },
            {
                "source": "dopaminergic_perturbation",
                "target": "dopaminergic_timing_dysregulation",
                "relation": "destabilizes dopamine-sensitive speech-timing systems",
                "stuttering_change": "increased",
            },
            {
                "source": "dopaminergic_perturbation",
                "target": "attention_impulse_control_modulation",
                "relation": "can alter attentional control over fluency",
                "stuttering_change": "increased",
            },
            {
                "source": "excitatory_inhibitory_instability",
                "target": "excitatory_inhibitory_timing_imbalance",
                "relation": "destabilizes temporal precision of neural firing",
                "stuttering_change": "increased",
            },
            {
                "source": "excitatory_inhibitory_instability",
                "target": "synaptic_release_instability",
                "relation": "can destabilize synaptic timing and release dynamics",
                "stuttering_change": "increased",
            },
            {
                "source": "developmental_white_matter_disconnection",
                "target": "speech_network_disconnectivity",
                "relation": "reduces integrity of frontal-temporal speech pathways",
                "stuttering_change": "increased",
            },
            {
                "source": "developmental_white_matter_disconnection",
                "target": "interhemispheric_timing_inefficiency",
                "relation": "reduces interhemispheric timing efficiency",
                "stuttering_change": "increased",
            },
            {
                "source": "fluency_support",
                "target": "speech_motor_timing_breakdown",
                "relation": "buffers unstable motor timing",
                "stuttering_change": "protective",
            },
            {
                "source": "fluency_support",
                "target": "repetitions_blocks_prolongations",
                "relation": "reduces overt disfluency severity",
                "stuttering_change": "protective",
            },
            {
                "source": "polygenic_speech_motor_liability",
                "target": "speech_motor_timing_breakdown",
                "relation": "loads vulnerability of speech-timing control",
                "stuttering_change": "increased",
            },
            {
                "source": "dopaminergic_timing_dysregulation",
                "target": "basal_ganglia_gating_instability",
                "relation": "destabilizes sequencing and initiation in basal-ganglia loops",
                "stuttering_change": "increased",
            },
            {
                "source": "dopaminergic_timing_dysregulation",
                "target": "basal_ganglia_proxy",
                "relation": "loads basal-ganglia timing circuitry",
                "stuttering_change": "increased dysregulation",
            },
            {
                "source": "basal_ganglia_gating_instability",
                "target": "speech_initiation_failure",
                "relation": "impairs fluent initiation of speech motor programs",
                "stuttering_change": "increased",
            },
            {
                "source": "basal_ganglia_gating_instability",
                "target": "repetitions_blocks_prolongations",
                "relation": "disrupts smooth sequencing of speech motor output",
                "stuttering_change": "increased",
            },
            {
                "source": "synaptic_release_instability",
                "target": "speech_motor_timing_breakdown",
                "relation": "reduces precision of temporal coordination",
                "stuttering_change": "increased",
            },
            {
                "source": "excitatory_inhibitory_timing_imbalance",
                "target": "speech_motor_timing_breakdown",
                "relation": "disrupts fine timing needed for fluent speech",
                "stuttering_change": "increased",
            },
            {
                "source": "speech_network_disconnectivity",
                "target": "ifg_broca_left",
                "relation": "burdens frontal speech-planning regions",
                "stuttering_change": "increased dysregulation",
            },
            {
                "source": "speech_network_disconnectivity",
                "target": "stg_left",
                "relation": "burdens left auditory-language monitoring regions",
                "stuttering_change": "increased dysregulation",
            },
            {
                "source": "speech_network_disconnectivity",
                "target": "stg_right",
                "relation": "burdens bilateral auditory timing and monitoring systems",
                "stuttering_change": "increased dysregulation",
            },
            {
                "source": "speech_network_disconnectivity",
                "target": "corpus_callosum_proxy",
                "relation": "can burden interhemispheric communication when this proxy resolves",
                "stuttering_change": "increased dysregulation",
            },
            {
                "source": "speech_network_disconnectivity",
                "target": "auditory_motor_monitoring_mismatch",
                "relation": "weakens coordination between auditory and motor speech nodes",
                "stuttering_change": "increased",
            },
            {
                "source": "interhemispheric_timing_inefficiency",
                "target": "corpus_callosum_proxy",
                "relation": "reflects inefficient interhemispheric timing coordination",
                "stuttering_change": "increased dysregulation",
            },
            {
                "source": "interhemispheric_timing_inefficiency",
                "target": "fluency_variability",
                "relation": "makes output less stable across contexts and loads",
                "stuttering_change": "increased",
            },
            {
                "source": "auditory_motor_monitoring_mismatch",
                "target": "auditory_feedback_instability",
                "relation": "disrupts monitoring of self-produced speech",
                "stuttering_change": "increased",
            },
            {
                "source": "auditory_motor_monitoring_mismatch",
                "target": "stg_left",
                "relation": "loads auditory-monitoring circuitry in the left temporal lobe",
                "stuttering_change": "increased dysregulation",
            },
            {
                "source": "auditory_motor_monitoring_mismatch",
                "target": "stg_right",
                "relation": "loads bilateral speech-sound monitoring circuitry",
                "stuttering_change": "increased dysregulation",
            },
            {
                "source": "speech_motor_timing_breakdown",
                "target": "ifg_broca_left",
                "relation": "burdens left inferior frontal articulatory planning systems",
                "stuttering_change": "increased dysregulation",
            },
            {
                "source": "speech_motor_timing_breakdown",
                "target": "premotor_speech_left",
                "relation": "burdens left speech motor timing and coordination systems",
                "stuttering_change": "increased dysregulation",
            },
            {
                "source": "speech_motor_timing_breakdown",
                "target": "repetitions_blocks_prolongations",
                "relation": "directly produces overt timing failures in speech",
                "stuttering_change": "increased",
            },
            {
                "source": "ifg_broca_left",
                "target": "speech_initiation_failure",
                "relation": "left frontal speech-planning dysfunction contributes to initiation difficulty",
                "stuttering_change": "increased",
            },
            {
                "source": "premotor_speech_left",
                "target": "repetitions_blocks_prolongations",
                "relation": "motor planning/timing instability contributes to disfluencies",
                "stuttering_change": "increased",
            },
            {
                "source": "attention_impulse_control_modulation",
                "target": "fluency_variability",
                "relation": "executive and attentional control can modulate fluency severity",
                "stuttering_change": "increased variability",
            },
            {
                "source": "attention_impulse_control_modulation",
                "target": "attention_related_fluency_modulation",
                "relation": "captures executive-control effects on fluency",
                "stuttering_change": "increased",
            },
            {
                "source": "fluency_variability",
                "target": "secondary_tension_behaviors",
                "relation": "instability promotes compensatory effort and tension",
                "stuttering_change": "increased",
            },
            {
                "source": "repetitions_blocks_prolongations",
                "target": "psychosocial_impact",
                "relation": "persistent disfluency increases functional and emotional burden",
                "stuttering_change": "increased",
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
            "inferior frontal gyrus",
            "superior temporal gyrus",
            "precentral gyrus",
            "basal ganglia",
            "striatum",
            "corpus callosum",
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
        Useful for tuning IFG/STG/callosal candidates.
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
            "ifg_broca_left",
            "stg_left",
            "stg_right",
            "premotor_speech_left",
            "basal_ganglia_proxy",
            "corpus_callosum_proxy",
        ),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the stuttering circuit.
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
                    "description": "Atlas-backed stuttering circuit node",
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
        essence_comorbidity_load: float,
        dopaminergic_perturbation: float,
        excitatory_inhibitory_instability: float,
        developmental_white_matter_disconnection: float,
        fluency_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while fluency_support is protective.
        """
        g = self._clip01(genetic_vulnerability)
        e = self._clip01(essence_comorbidity_load)
        d = self._clip01(dopaminergic_perturbation)
        x = self._clip01(excitatory_inhibitory_instability)
        w = self._clip01(developmental_white_matter_disconnection)
        f = self._clip01(fluency_support)

        # Latent biology
        polygenic_speech_motor_liability = self._clip01(
            0.40 * g + 0.15 * e
        )
        dopaminergic_timing_dysregulation = self._clip01(
            0.35 * d + 0.20 * g + 0.10 * e - 0.10 * f
        )
        synaptic_release_instability = self._clip01(
            0.30 * x + 0.20 * g - 0.10 * f
        )
        excitatory_inhibitory_timing_imbalance = self._clip01(
            0.40 * x + 0.15 * g + 0.10 * d - 0.15 * f
        )
        speech_network_disconnectivity = self._clip01(
            0.40 * w + 0.15 * e + 0.10 * g - 0.10 * f
        )
        interhemispheric_timing_inefficiency = self._clip01(
            0.35 * w + 0.20 * speech_network_disconnectivity
        )
        basal_ganglia_gating_instability = self._clip01(
            0.35 * dopaminergic_timing_dysregulation
            + 0.20 * excitatory_inhibitory_timing_imbalance
            + 0.10 * g
            - 0.10 * f
        )
        auditory_motor_monitoring_mismatch = self._clip01(
            0.35 * speech_network_disconnectivity
            + 0.20 * interhemispheric_timing_inefficiency
            + 0.10 * x
            - 0.10 * f
        )
        attention_impulse_control_modulation = self._clip01(
            0.35 * e + 0.20 * d - 0.10 * f
        )
        speech_motor_timing_breakdown = self._clip01(
            0.30 * polygenic_speech_motor_liability
            + 0.25 * basal_ganglia_gating_instability
            + 0.20 * synaptic_release_instability
            + 0.15 * excitatory_inhibitory_timing_imbalance
            + 0.10 * speech_network_disconnectivity
            - 0.20 * f
        )

        # Regional state proxies
        ifg_broca_left = self._clip01(
            0.40 * speech_motor_timing_breakdown
            + 0.20 * auditory_motor_monitoring_mismatch
            - 0.10 * f
        )
        stg_left = self._clip01(
            0.40 * auditory_motor_monitoring_mismatch
            + 0.20 * speech_network_disconnectivity
            - 0.10 * f
        )
        stg_right = self._clip01(
            0.35 * auditory_motor_monitoring_mismatch
            + 0.25 * interhemispheric_timing_inefficiency
            - 0.10 * f
        )
        premotor_speech_left = self._clip01(
            0.40 * speech_motor_timing_breakdown
            + 0.15 * excitatory_inhibitory_timing_imbalance
            - 0.10 * f
        )
        basal_ganglia_proxy = self._clip01(
            0.45 * basal_ganglia_gating_instability
            + 0.15 * dopaminergic_timing_dysregulation
            - 0.10 * f
        )
        corpus_callosum_proxy = self._clip01(
            0.45 * interhemispheric_timing_inefficiency
            + 0.20 * speech_network_disconnectivity
            - 0.10 * f
        )

        # Symptoms
        speech_initiation_failure = self._clip01(
            0.35 * basal_ganglia_gating_instability
            + 0.25 * ifg_broca_left
            + 0.15 * premotor_speech_left
            - 0.10 * f
        )
        repetitions_blocks_prolongations = self._clip01(
            0.35 * speech_motor_timing_breakdown
            + 0.20 * basal_ganglia_proxy
            + 0.15 * ifg_broca_left
            + 0.10 * premotor_speech_left
            - 0.15 * f
        )
        auditory_feedback_instability = self._clip01(
            0.35 * auditory_motor_monitoring_mismatch
            + 0.20 * stg_left
            + 0.15 * stg_right
            - 0.10 * f
        )
        fluency_variability = self._clip01(
            0.30 * repetitions_blocks_prolongations
            + 0.20 * attention_impulse_control_modulation
            + 0.20 * auditory_feedback_instability
            + 0.10 * interhemispheric_timing_inefficiency
            - 0.10 * f
        )
        secondary_tension_behaviors = self._clip01(
            0.35 * fluency_variability
            + 0.20 * speech_initiation_failure
            - 0.10 * f
        )
        attention_related_fluency_modulation = self._clip01(
            0.45 * attention_impulse_control_modulation
            + 0.15 * dopaminergic_timing_dysregulation
        )
        psychosocial_impact = self._clip01(
            0.40 * repetitions_blocks_prolongations
            + 0.20 * fluency_variability
            + 0.15 * secondary_tension_behaviors
            - 0.10 * f
        )

        return {
            "inputs": pd.Series(
                {
                    "genetic_vulnerability": g,
                    "essence_comorbidity_load": e,
                    "dopaminergic_perturbation": d,
                    "excitatory_inhibitory_instability": x,
                    "developmental_white_matter_disconnection": w,
                    "fluency_support": f,
                }
            ),
            "latents": pd.Series(
                {
                    "speech_motor_timing_breakdown": speech_motor_timing_breakdown,
                    "basal_ganglia_gating_instability": basal_ganglia_gating_instability,
                    "auditory_motor_monitoring_mismatch": auditory_motor_monitoring_mismatch,
                    "speech_network_disconnectivity": speech_network_disconnectivity,
                    "dopaminergic_timing_dysregulation": dopaminergic_timing_dysregulation,
                    "attention_impulse_control_modulation": attention_impulse_control_modulation,
                    "interhemispheric_timing_inefficiency": interhemispheric_timing_inefficiency,
                    "excitatory_inhibitory_timing_imbalance": excitatory_inhibitory_timing_imbalance,
                    "synaptic_release_instability": synaptic_release_instability,
                    "polygenic_speech_motor_liability": polygenic_speech_motor_liability,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "ifg_broca_left": ifg_broca_left,
                    "stg_left": stg_left,
                    "stg_right": stg_right,
                    "premotor_speech_left": premotor_speech_left,
                    "basal_ganglia_proxy": basal_ganglia_proxy,
                    "corpus_callosum_proxy": corpus_callosum_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "repetitions_blocks_prolongations": repetitions_blocks_prolongations,
                    "speech_initiation_failure": speech_initiation_failure,
                    "auditory_feedback_instability": auditory_feedback_instability,
                    "fluency_variability": fluency_variability,
                    "secondary_tension_behaviors": secondary_tension_behaviors,
                    "attention_related_fluency_modulation": attention_related_fluency_modulation,
                    "psychosocial_impact": psychosocial_impact,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "basal_ganglia_timing_profile": self._clip01(
                        0.45 * basal_ganglia_gating_instability
                        + 0.25 * speech_initiation_failure
                        + 0.20 * repetitions_blocks_prolongations
                    ),
                    "auditory_motor_mismatch_profile": self._clip01(
                        0.45 * auditory_feedback_instability
                        + 0.30 * auditory_motor_monitoring_mismatch
                        + 0.15 * fluency_variability
                    ),
                    "disconnection_profile": self._clip01(
                        0.40 * speech_network_disconnectivity
                        + 0.25 * interhemispheric_timing_inefficiency
                        + 0.20 * fluency_variability
                    ),
                    "attention_modulated_profile": self._clip01(
                        0.45 * attention_related_fluency_modulation
                        + 0.25 * fluency_variability
                        + 0.15 * dopaminergic_timing_dysregulation
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-50, 10, 18)).head(10)
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
    model = ChildhoodOnsetFluencyDisorderModel()

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
        bundle["edges"][["source", "target", "relation", "stuttering_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in [
        "ifg_broca_left",
        "stg_left",
        "stg_right",
        "premotor_speech_left",
        "basal_ganglia_proxy",
        "corpus_callosum_proxy",
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
        genetic_vulnerability=0.75,
        essence_comorbidity_load=0.55,
        dopaminergic_perturbation=0.70,
        excitatory_inhibitory_instability=0.45,
        developmental_white_matter_disconnection=0.65,
        fluency_support=0.20,
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
    # print(model.assign_mni_point((-50, 10, 18)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("ifg").to_string(index=False))
    # print(model.suggest_regions("stg").to_string(index=False))
    # print(model.suggest_regions("corpus callosum").to_string(index=False))
