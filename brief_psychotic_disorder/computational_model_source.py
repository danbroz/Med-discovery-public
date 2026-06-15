from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Brief Psychotic Disorder-oriented panel:
# - dopamine / salience / psychosis
# - serotonin / norepinephrine stress-linked modulation
# - GABA / glutamate balance
# - stress responsivity / recovery biology
DEFAULT_GENE_PANEL = [
    "DRD2",    # dopamine receptor
    "SLC6A3",  # dopamine transporter
    "COMT",    # catecholamine metabolism
    "SLC6A4",  # serotonin transporter
    "SLC6A2",  # norepinephrine transporter
    "DBH",     # dopamine beta hydroxylase
    "MAOA",    # monoamine metabolism
    "GABRA2",  # GABA-A receptor
    "GABRB2",  # GABA-A receptor
    "GAD1",    # GABA synthesis
    "GRIN2B",  # NMDA receptor subunit
    "SLC1A1",  # glutamate transport
    "NR3C1",   # glucocorticoid receptor
    "FKBP5",   # stress responsivity
    "CRHR1",   # CRH signaling
    "BDNF",    # neuroplasticity / recovery
]


class BriefPsychoticDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Brief Psychotic Disorder.

    What it does:
      1) Resolves chapter-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates acute stress psychosis, disorganization, and recovery.

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

        # I avoid the abbreviation "BPD" here to prevent confusion with
        # Borderline Personality Disorder. This is specifically Brief Psychotic Disorder.
        self.region_candidates: Dict[str, List[str]] = {
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
            "pfc_control": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "prefrontal",
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
                "basal ganglia",
                "striatum",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "acute_stressor": "Severe overwhelming stressor precipitating acute decompensation",
            "genetic_psychosis_vulnerability": "Heritable vulnerability for psychosis-spectrum symptoms",
            "mood_instability_loading": "Heritable or familial vulnerability for affect-laden psychosis",
            "sleep_arousal_disruption": "Acute sleep/arousal destabilization amplifying psychosis risk",
            "resilience_support": "Protective recovery, containment, and restoration of homeostasis",
        }

        self.latent_nodes: Dict[str, str] = {
            "hpa_axis_surge": "Acute stress-hormone surge under overwhelming stress",
            "monoamine_storm": "Stress-linked dopamine, norepinephrine, and serotonin dysregulation",
            "dopamine_salience_overload": "Aberrant salience and psychosis-linked dopaminergic surge",
            "gaba_inhibitory_failure": "Transient breakdown of inhibitory cortical control",
            "glutamate_hyperexcitability": "Stress-linked excitatory overdrive",
            "hippocampal_disinhibition": "Loss of GABAergic restraint in hippocampal circuitry",
            "cortical_disorganization": "Disorganized cortical information processing",
            "thalamic_filter_failure": "Reduced sensory/thalamic gating and filtering",
            "basal_ganglia_integration_failure": "Disrupted integration of motor, cognitive, and emotional signals",
            "homeostatic_recovery": "Restoration of stress and circuit equilibrium after the acute episode",
            "resilience_factors": "Latent recovery-promoting biological resilience",
        }

        self.symptom_nodes: Dict[str, str] = {
            "hallucination_delusion": "Transient psychotic perception and belief disturbance",
            "disorganized_thought_speech": "Disorganized thought and speech under cortical dysregulation",
            "agitation": "Acute agitation and behavioral dyscontrol",
            "affective_lability": "Rapid mood shifts and affect-laden psychosis",
            "catatonic_disorganized_behavior": "Motor or behavioral disorganization / catatonia-like state",
            "full_remission": "Return toward baseline with restoration of homeostasis",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "acute_stressor",
                "target": "hpa_axis_surge",
                "relation": "triggers acute stress-system overload",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "acute_stressor",
                "target": "sleep_arousal_disruption",
                "relation": "destabilizes sleep, arousal, and cognitive control",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "genetic_psychosis_vulnerability",
                "target": "dopamine_salience_overload",
                "relation": "raises vulnerability to psychotic salience dysregulation",
                "brief_psychosis_change": "increased susceptibility",
            },
            {
                "source": "genetic_psychosis_vulnerability",
                "target": "gaba_inhibitory_failure",
                "relation": "raises vulnerability to inhibitory breakdown under stress",
                "brief_psychosis_change": "increased susceptibility",
            },
            {
                "source": "mood_instability_loading",
                "target": "monoamine_storm",
                "relation": "raises vulnerability to affect-laden transmitter instability",
                "brief_psychosis_change": "increased susceptibility",
            },
            {
                "source": "mood_instability_loading",
                "target": "affective_lability",
                "relation": "supports polymorphic affective presentation",
                "brief_psychosis_change": "increased susceptibility",
            },
            {
                "source": "resilience_support",
                "target": "homeostatic_recovery",
                "relation": "supports restoration of system equilibrium",
                "brief_psychosis_change": "protective",
            },
            {
                "source": "resilience_support",
                "target": "agitation",
                "relation": "buffers escalation of acute agitation",
                "brief_psychosis_change": "protective",
            },
            {
                "source": "resilience_support",
                "target": "full_remission",
                "relation": "supports rapid return toward baseline",
                "brief_psychosis_change": "protective",
            },
            {
                "source": "hpa_axis_surge",
                "target": "monoamine_storm",
                "relation": "cortisol perturbs dopamine, norepinephrine, and serotonin regulation",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "hpa_axis_surge",
                "target": "glutamate_hyperexcitability",
                "relation": "stress increases excitatory destabilization",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "hpa_axis_surge",
                "target": "hippocampus",
                "relation": "acute stress burdens hippocampal processing",
                "brief_psychosis_change": "increased dysregulation",
            },
            {
                "source": "hpa_axis_surge",
                "target": "pfc_control",
                "relation": "acute stress weakens cognitive control systems",
                "brief_psychosis_change": "reduced function",
            },
            {
                "source": "monoamine_storm",
                "target": "dopamine_salience_overload",
                "relation": "shifts transmitter balance toward psychotic salience dysregulation",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "monoamine_storm",
                "target": "affective_lability",
                "relation": "supports rapid affective instability",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "gaba_inhibitory_failure",
                "target": "hippocampal_disinhibition",
                "relation": "reduces inhibitory restraint in hippocampal circuitry",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "gaba_inhibitory_failure",
                "target": "cortical_disorganization",
                "relation": "permits disorganized neuronal firing",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "glutamate_hyperexcitability",
                "target": "cortical_disorganization",
                "relation": "amplifies hyperexcitable cortical processing",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "hippocampal_disinhibition",
                "target": "hippocampus",
                "relation": "destabilizes stress-memory processing",
                "brief_psychosis_change": "increased dysregulation",
            },
            {
                "source": "hippocampal_disinhibition",
                "target": "dopamine_salience_overload",
                "relation": "can disinhibit downstream dopaminergic psychosis pathways",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "cortical_disorganization",
                "target": "disorganized_thought_speech",
                "relation": "drives thought and language disorganization",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "dopamine_salience_overload",
                "target": "hallucination_delusion",
                "relation": "supports aberrant salience and psychotic symptom formation",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "thalamic_filter_failure",
                "target": "hallucination_delusion",
                "relation": "weakens sensory filtering and perceptual organization",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "thalamic_filter_failure",
                "target": "thalamus_proxy",
                "relation": "destabilizes thalamic relay and gating functions",
                "brief_psychosis_change": "increased dysregulation",
            },
            {
                "source": "basal_ganglia_integration_failure",
                "target": "basal_ganglia_proxy",
                "relation": "disrupts motor-cognitive-emotional integration",
                "brief_psychosis_change": "increased dysregulation",
            },
            {
                "source": "basal_ganglia_integration_failure",
                "target": "catatonic_disorganized_behavior",
                "relation": "can promote motor and behavioral abnormalities",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "dopamine_salience_overload",
                "target": "basal_ganglia_integration_failure",
                "relation": "dopamine surge destabilizes subcortical integration",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "sleep_arousal_disruption",
                "target": "cortical_disorganization",
                "relation": "weakens coherent cognitive control under acute stress",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "sleep_arousal_disruption",
                "target": "agitation",
                "relation": "amplifies psychomotor dyscontrol",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "agitation",
                "relation": "amplifies acute emotional salience and alarm output",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "affective_lability",
                "relation": "stress-memory dysregulation amplifies unstable affective tone",
                "brief_psychosis_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "cortical_disorganization",
                "relation": "normally stabilizes thought and behavioral control",
                "brief_psychosis_change": "reduced inhibition",
            },
            {
                "source": "pfc_control",
                "target": "hallucination_delusion",
                "relation": "normally constrains psychotic interpretation",
                "brief_psychosis_change": "reduced inhibition",
            },
            {
                "source": "resilience_factors",
                "target": "homeostatic_recovery",
                "relation": "support restoration after acute dysregulation",
                "brief_psychosis_change": "protective",
            },
            {
                "source": "homeostatic_recovery",
                "target": "full_remission",
                "relation": "restores stable cognition and affect after the acute storm",
                "brief_psychosis_change": "increased remission",
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

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        """
        Helper for refining proxy searches against the atlas.
        Useful for tuning thalamic / basal ganglia candidates.
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "hippocampus",
            "amygdala",
            "thalamus",
            "basal ganglia",
            "striatum",
            "prefrontal cortex",
        } else 0
        return (left_bonus, right_penalty, generic_penalty)

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

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

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
        feats = self._safe_features(region, "receptor density fingerprint")
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
        feats = self._safe_features(region, "gene expressions", gene=list(genes))
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

        feats = self._safe_features(self.parcellation, "StreamlineCounts")
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
            "hippocampus",
            "amygdala",
            "pfc_control",
            "thalamus_proxy",
            "basal_ganglia_proxy",
        ),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the brief psychosis circuit.
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
                    "description": "Atlas-backed brief psychosis circuit node",
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
        acute_stressor: float,
        genetic_psychosis_vulnerability: float,
        mood_instability_loading: float,
        sleep_arousal_disruption: float,
        resilience_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while resilience_support is protective.
        """
        s = self._clip01(acute_stressor)
        g = self._clip01(genetic_psychosis_vulnerability)
        m = self._clip01(mood_instability_loading)
        a = self._clip01(sleep_arousal_disruption)
        r = self._clip01(resilience_support)

        # Latent biology
        resilience_factors = self._clip01(0.55 * r)
        hpa_axis_surge = self._clip01(0.50 * s + 0.15 * a - 0.15 * r)
        monoamine_storm = self._clip01(
            0.30 * hpa_axis_surge + 0.20 * m + 0.15 * s - 0.10 * r
        )
        gaba_inhibitory_failure = self._clip01(
            0.25 * s + 0.25 * g + 0.20 * a - 0.15 * r
        )
        glutamate_hyperexcitability = self._clip01(
            0.35 * hpa_axis_surge + 0.25 * gaba_inhibitory_failure + 0.10 * a
        )
        hippocampal_disinhibition = self._clip01(
            0.40 * gaba_inhibitory_failure + 0.25 * glutamate_hyperexcitability + 0.10 * s
        )
        dopamine_salience_overload = self._clip01(
            0.30 * monoamine_storm + 0.25 * hippocampal_disinhibition + 0.15 * g
        )
        cortical_disorganization = self._clip01(
            0.35 * glutamate_hyperexcitability
            + 0.25 * gaba_inhibitory_failure
            + 0.15 * a
            - 0.10 * r
        )
        thalamic_filter_failure = self._clip01(
            0.30 * cortical_disorganization + 0.20 * dopamine_salience_overload
        )
        basal_ganglia_integration_failure = self._clip01(
            0.30 * dopamine_salience_overload + 0.20 * cortical_disorganization
        )
        homeostatic_recovery = self._clip01(
            0.40 * resilience_factors + 0.20 * r - 0.15 * hpa_axis_surge
        )

        # Regional state proxies
        hippocampus = self._clip01(
            0.40 * hippocampal_disinhibition + 0.20 * hpa_axis_surge
        )
        amygdala = self._clip01(
            0.30 * hpa_axis_surge + 0.20 * monoamine_storm + 0.10 * s
        )
        pfc_control = self._clip01(
            0.35 * cortical_disorganization + 0.20 * hpa_axis_surge + 0.10 * a - 0.15 * r
        )
        thalamus_proxy = self._clip01(
            0.40 * thalamic_filter_failure + 0.15 * cortical_disorganization
        )
        basal_ganglia_proxy = self._clip01(
            0.40 * basal_ganglia_integration_failure + 0.15 * dopamine_salience_overload
        )

        # Symptoms
        hallucination_delusion = self._clip01(
            0.35 * dopamine_salience_overload
            + 0.25 * thalamic_filter_failure
            + 0.10 * cortical_disorganization
            - 0.10 * homeostatic_recovery
        )
        disorganized_thought_speech = self._clip01(
            0.45 * cortical_disorganization + 0.15 * pfc_control - 0.10 * homeostatic_recovery
        )
        agitation = self._clip01(
            0.30 * amygdala
            + 0.25 * monoamine_storm
            + 0.20 * a
            - 0.10 * homeostatic_recovery
        )
        affective_lability = self._clip01(
            0.35 * monoamine_storm + 0.25 * amygdala + 0.15 * m
        )
        catatonic_disorganized_behavior = self._clip01(
            0.35 * basal_ganglia_integration_failure
            + 0.20 * cortical_disorganization
            + 0.10 * thalamic_filter_failure
            - 0.10 * homeostatic_recovery
        )
        full_remission = self._clip01(
            0.55 * homeostatic_recovery + 0.20 * resilience_factors
        )

        return {
            "inputs": pd.Series(
                {
                    "acute_stressor": s,
                    "genetic_psychosis_vulnerability": g,
                    "mood_instability_loading": m,
                    "sleep_arousal_disruption": a,
                    "resilience_support": r,
                }
            ),
            "latents": pd.Series(
                {
                    "hpa_axis_surge": hpa_axis_surge,
                    "monoamine_storm": monoamine_storm,
                    "dopamine_salience_overload": dopamine_salience_overload,
                    "gaba_inhibitory_failure": gaba_inhibitory_failure,
                    "glutamate_hyperexcitability": glutamate_hyperexcitability,
                    "hippocampal_disinhibition": hippocampal_disinhibition,
                    "cortical_disorganization": cortical_disorganization,
                    "thalamic_filter_failure": thalamic_filter_failure,
                    "basal_ganglia_integration_failure": basal_ganglia_integration_failure,
                    "homeostatic_recovery": homeostatic_recovery,
                    "resilience_factors": resilience_factors,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "hippocampus": hippocampus,
                    "amygdala": amygdala,
                    "pfc_control": pfc_control,
                    "thalamus_proxy": thalamus_proxy,
                    "basal_ganglia_proxy": basal_ganglia_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "hallucination_delusion": hallucination_delusion,
                    "disorganized_thought_speech": disorganized_thought_speech,
                    "agitation": agitation,
                    "affective_lability": affective_lability,
                    "catatonic_disorganized_behavior": catatonic_disorganized_behavior,
                    "full_remission": full_remission,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "stress_triggered_psychosis_profile": self._clip01(
                        0.40 * hpa_axis_surge
                        + 0.25 * hallucination_delusion
                        + 0.15 * disorganized_thought_speech
                    ),
                    "affective_polymorphic_profile": self._clip01(
                        0.40 * affective_lability
                        + 0.30 * hallucination_delusion
                        + 0.15 * mood_instability_loading
                    ),
                    "disorganized_agitated_profile": self._clip01(
                        0.40 * agitation
                        + 0.30 * disorganized_thought_speech
                        + 0.20 * catatonic_disorganized_behavior
                    ),
                    "rapid_recovery_profile": self._clip01(
                        0.55 * full_remission + 0.25 * homeostatic_recovery
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-18, -10, -12)).head(10)
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
    model = BriefPsychoticDisorderModel()

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
        bundle["edges"][["source", "target", "relation", "brief_psychosis_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["hippocampus", "amygdala", "pfc_control", "thalamus_proxy", "basal_ganglia_proxy"]:
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
        acute_stressor=0.90,
        genetic_psychosis_vulnerability=0.55,
        mood_instability_loading=0.45,
        sleep_arousal_disruption=0.70,
        resilience_support=0.35,
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
    # print(model.assign_mni_point((-18, -10, -12)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("thalamus").to_string(index=False))
    # print(model.suggest_regions("caudate").to_string(index=False))
