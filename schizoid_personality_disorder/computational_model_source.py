from __future__ import annotations

"""
Schizoid Personality Disorder siibra scaffold.

This script turns a chapter-level mechanistic summary of Schizoid Personality
Disorder (SPD) into an atlas-grounded research scaffold built around siibra
idioms. It is meant for hypothesis generation, teaching, and reproducible
exploration of atlas-backed regional features. It is not a clinical diagnostic
or treatment tool.

Conceptual choices:
- The chapter provides very limited SPD-specific molecular evidence. The model
  therefore makes explicit, conservative inferences from schizophrenia-spectrum
  and negative-symptom literature instead of pretending there are validated
  SPD-specific biomarkers.
- The core mechanistic hypothesis is that shared schizophrenia-spectrum
  liability, interacting with early adversity or emotionally impoverished
  environments, can yield a stable phenotype of social withdrawal, emotional
  detachment, and avolition without requiring overt psychosis.
- Because the chapter explicitly highlights NMDA-related glutamatergic
  hypofunction, downstream dopamine/GABA dysregulation, prefrontal hypoactivity,
  amygdala-centered social-emotional dysregulation, and thalamocortical
  dysconnectivity, those mechanisms form the latent backbone of the simulator.
- Direct anchors are used where the chapter is reasonably specific (amygdala).
  Conservative proxies are used where the chapter remains systems-level
  (prefrontal cortex, thalamus).
- The gene panel is hypothesis-driven rather than disorder-validated. It is
  organized around glutamatergic/NMDA signaling, inhibitory interneuron
  regulation, dopamine/serotonin modulation, synaptic plasticity, and broad
  schizophrenia-spectrum risk biology.

The scaffold degrades gracefully when siibra or particular multimodal features
are unavailable. Atlas-backed nodes remain visible in the graph, but unresolved
feature tables are returned empty instead of raising an error.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
except Exception:  # pragma: no cover - optional dependency in some environments
    siibra = None


DEFAULT_GENE_PANEL = [
    # Glutamatergic / NMDA signaling
    "GRIN1",
    "GRIN2A",
    "GRIN2B",
    "GRM3",
    "SLC1A2",
    # Inhibitory interneuron regulation
    "GAD1",
    "GABRA1",
    "GABRB2",
    # Dopamine and salience / motivation
    "DRD2",
    "DRD3",
    "COMT",
    "SLC6A3",
    # Serotonergic social-affective modulation
    "HTR2A",
    "SLC6A4",
    # Synaptic plasticity / spectrum vulnerability
    "BDNF",
    "DISC1",
    "CACNA1C",
    "NRG1",
]


class SchizoidPersonalityDisorderModel:
    """
    Atlas-grounded research scaffold for Schizoid Personality Disorder.

    The simulator keeps a transparent one-pass causal flow:
        inputs -> latent biology -> regional burden -> symptoms -> phenotypes

    Because direct SPD biological evidence is sparse, several latent variables
    are intentionally labeled as spectrum-informed hypotheses rather than
    disorder-specific facts.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        assignment_space: str = "mni152",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.siibra = siibra
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = assignment_space
        self.connectivity_cohort = connectivity_cohort

        self.atlas = None
        self.parcellation = None
        self.space = None
        self._pmap = None
        self._connectivity_matrix: Optional[pd.DataFrame] = None

        if self.siibra is None:
            warnings.warn(
                "siibra is not installed in this environment. Atlas-backed build "
                "steps will degrade gracefully, returning unresolved region nodes "
                "and empty feature tables until siibra is available."
            )
        else:
            try:
                self.atlas = self.siibra.atlases.get(atlas_spec)
                self.parcellation = (
                    self.atlas.get_parcellation(parcellation_spec)
                    if hasattr(self.atlas, "get_parcellation")
                    else self.atlas.parcellations.get(parcellation_spec)
                )
                self.space = (
                    self.atlas.get_space(space_spec)
                    if hasattr(self.atlas, "get_space")
                    else self.atlas.spaces.get(space_spec)
                )
            except Exception as exc:  # pragma: no cover - depends on local siibra data
                warnings.warn(
                    f"Could not initialize atlas/parcellation resources: {exc!r}. "
                    "Atlas-backed methods may return empty outputs."
                )

        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala left",
                "amygdala",
            ],
            "prefrontal_cortex_proxy": [
                "Area 46 left",
                "Area 9 left",
                "Area 45 (IFG) left",
                "Area 10 left",
                "prefrontal cortex",
            ],
            "thalamus_proxy": [
                "thalamus left",
                "thalamus",
                "mediodorsal thalamus",
                "metathalamus",
            ],
        }

        self.proxy_region_nodes = {
            "prefrontal_cortex_proxy",
            "thalamus_proxy",
        }

        self.region_descriptions: Dict[str, str] = {
            "amygdala": (
                "Direct atlas anchor for the amygdala-centered social-emotional "
                "processing network highlighted in the chapter."
            ),
            "prefrontal_cortex_proxy": (
                "Conservative proxy for broad prefrontal control and motivation "
                "territory described as hypoactive in schizoid functioning."
            ),
            "thalamus_proxy": (
                "Conservative proxy for thalamocortical relay circuitry described "
                "as dysconnected in schizophrenia-spectrum models relevant to SPD."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "schizophrenia_spectrum_genetic_liability": (
                "Shared familial and polygenic liability across schizophrenia-spectrum conditions."
            ),
            "neurodevelopmental_vulnerability": (
                "Broad neurodevelopmental vulnerability shaping circuit maturation."
            ),
            "childhood_adversity": (
                "Early adversity, neglect, or trauma that can leave lasting effects on social-emotional development."
            ),
            "emotional_nonresponsive_environment": (
                "Emotionally non-responsive caregiving or relational context that may reinforce detachment."
            ),
            "social_impoverishment": (
                "Socially impoverished developmental environment limiting social learning."
            ),
            "supportive_relational_context": (
                "Protective corrective relationships and social scaffolding that may buffer detachment."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "shared_spectrum_vulnerability": (
                "Integrated schizophrenia-spectrum liability arising from genetic and neurodevelopmental factors."
            ),
            "developmental_detachment_adaptation": (
                "Stable adaptation toward emotional distance and self-protective social withdrawal."
            ),
            "nmda_interneuron_hypofunction": (
                "Subtle chronic NMDA-related glutamatergic hypofunction, especially affecting inhibitory interneurons."
            ),
            "gaba_interneuron_dysregulation": (
                "Downstream inhibitory dysregulation linked to impaired interneuron function."
            ),
            "monoamine_social_reward_blunting": (
                "Attenuated dopamine/serotonin signaling contributing to social reward reduction and affective flattening."
            ),
            "synaptic_connectivity_disruption": (
                "Reduced synaptic plasticity and connectivity supporting inefficient network communication."
            ),
            "thalamic_sensory_hyperconnectivity": (
                "Relative overcoupling of thalamic signaling with sensory systems."
            ),
            "thalamic_prefrontal_hypoconnectivity": (
                "Reduced thalamic coupling with prefrontal control systems."
            ),
            "thalamocortical_dysconnectivity": (
                "Net disruption of thalamocortical communication relevant to negative-spectrum phenomena."
            ),
            "prefrontal_hypoactivity": (
                "Reduced prefrontal engagement in motivation, planning, and top-down social regulation."
            ),
            "amygdala_social_emotional_blunting": (
                "Blunted or dysregulated amygdala-centered social-emotional processing."
            ),
            "social_cognition_underengagement": (
                "Under-recruitment of systems needed for social learning and interpersonal salience."
            ),
            "motivational_negative_symptom_load": (
                "Integrated burden of avolition, anhedonia, and diminished drive."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "social_withdrawal": (
                "Marked tendency to disengage from social relationships and activity."
            ),
            "preference_for_solitude": (
                "Stable preference for being alone over affiliative engagement."
            ),
            "emotional_detachment": (
                "Interpersonal distance and reduced emotional reciprocity."
            ),
            "avolition": (
                "Reduced motivation and initiative resembling negative symptoms."
            ),
            "flattened_affect": (
                "Restricted or blunted emotional expression."
            ),
            "anhedonia": (
                "Reduced capacity to derive reward or pleasure, especially socially."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "schizophrenia_spectrum_genetic_liability",
                "target": "shared_spectrum_vulnerability",
                "relation": "shared familial liability contributes to schizophrenia-spectrum vulnerability",
                "schizoid_change": "increased",
            },
            {
                "source": "neurodevelopmental_vulnerability",
                "target": "shared_spectrum_vulnerability",
                "relation": "neurodevelopmental factors shape spectrum-level risk architecture",
                "schizoid_change": "increased",
            },
            {
                "source": "childhood_adversity",
                "target": "developmental_detachment_adaptation",
                "relation": "early adversity can promote defensive withdrawal and emotional distance",
                "schizoid_change": "increased",
            },
            {
                "source": "emotional_nonresponsive_environment",
                "target": "developmental_detachment_adaptation",
                "relation": "emotionally barren environments can reinforce schizoid relating",
                "schizoid_change": "increased",
            },
            {
                "source": "social_impoverishment",
                "target": "developmental_detachment_adaptation",
                "relation": "reduced social learning context shapes enduring interpersonal disengagement",
                "schizoid_change": "increased",
            },
            {
                "source": "shared_spectrum_vulnerability",
                "target": "developmental_detachment_adaptation",
                "relation": "spectrum liability interacts with environmental context to shape stable schizoid adaptation",
                "schizoid_change": "increased",
            },
            {
                "source": "supportive_relational_context",
                "target": "developmental_detachment_adaptation",
                "relation": "corrective relationships may buffer entrenched detachment",
                "schizoid_change": "decreased",
            },
            {
                "source": "shared_spectrum_vulnerability",
                "target": "nmda_interneuron_hypofunction",
                "relation": "spectrum vulnerability plausibly includes subtle chronic NMDA hypofunction",
                "schizoid_change": "increased",
            },
            {
                "source": "developmental_detachment_adaptation",
                "target": "nmda_interneuron_hypofunction",
                "relation": "developmental adaptation may stabilize inefficient glutamatergic social-learning circuitry",
                "schizoid_change": "increased",
            },
            {
                "source": "nmda_interneuron_hypofunction",
                "target": "gaba_interneuron_dysregulation",
                "relation": "NMDA dysfunction in inhibitory interneurons perturbs inhibitory regulation",
                "schizoid_change": "increased",
            },
            {
                "source": "nmda_interneuron_hypofunction",
                "target": "monoamine_social_reward_blunting",
                "relation": "glutamatergic dysfunction can secondarily dysregulate dopamine and serotonin systems",
                "schizoid_change": "increased",
            },
            {
                "source": "nmda_interneuron_hypofunction",
                "target": "synaptic_connectivity_disruption",
                "relation": "chronic glutamatergic inefficiency impairs synaptic plasticity and connectivity",
                "schizoid_change": "increased",
            },
            {
                "source": "gaba_interneuron_dysregulation",
                "target": "synaptic_connectivity_disruption",
                "relation": "inhibitory dysregulation further destabilizes coordinated network communication",
                "schizoid_change": "increased",
            },
            {
                "source": "shared_spectrum_vulnerability",
                "target": "synaptic_connectivity_disruption",
                "relation": "heritable spectrum risk contributes to distributed connectivity inefficiency",
                "schizoid_change": "increased",
            },
            {
                "source": "synaptic_connectivity_disruption",
                "target": "thalamic_sensory_hyperconnectivity",
                "relation": "distributed dysconnectivity can yield thalamic overcoupling with sensory systems",
                "schizoid_change": "increased",
            },
            {
                "source": "synaptic_connectivity_disruption",
                "target": "thalamic_prefrontal_hypoconnectivity",
                "relation": "connectivity disruption weakens thalamic coupling with prefrontal systems",
                "schizoid_change": "increased",
            },
            {
                "source": "thalamic_sensory_hyperconnectivity",
                "target": "thalamocortical_dysconnectivity",
                "relation": "sensory overcoupling contributes to broader thalamocortical imbalance",
                "schizoid_change": "increased",
            },
            {
                "source": "thalamic_prefrontal_hypoconnectivity",
                "target": "thalamocortical_dysconnectivity",
                "relation": "weakened prefrontal-thalamic coupling is a core dysconnectivity pattern",
                "schizoid_change": "increased",
            },
            {
                "source": "thalamocortical_dysconnectivity",
                "target": "thalamus_proxy",
                "relation": "distributed thalamocortical dysfunction burdens thalamic relay systems",
                "schizoid_change": "increased",
            },
            {
                "source": "thalamic_prefrontal_hypoconnectivity",
                "target": "prefrontal_hypoactivity",
                "relation": "reduced thalamic support contributes to prefrontal under-engagement",
                "schizoid_change": "increased",
            },
            {
                "source": "monoamine_social_reward_blunting",
                "target": "prefrontal_hypoactivity",
                "relation": "monoaminergic attenuation reduces motivational and executive activation",
                "schizoid_change": "increased",
            },
            {
                "source": "developmental_detachment_adaptation",
                "target": "prefrontal_hypoactivity",
                "relation": "rehearsed withdrawal can reduce recruitment of social-control systems",
                "schizoid_change": "increased",
            },
            {
                "source": "monoamine_social_reward_blunting",
                "target": "amygdala_social_emotional_blunting",
                "relation": "attenuated monoaminergic modulation can flatten social-emotional responsiveness",
                "schizoid_change": "increased",
            },
            {
                "source": "developmental_detachment_adaptation",
                "target": "amygdala_social_emotional_blunting",
                "relation": "enduring detachment adaptation shapes amygdala-centered social-emotional processing",
                "schizoid_change": "increased",
            },
            {
                "source": "synaptic_connectivity_disruption",
                "target": "amygdala_social_emotional_blunting",
                "relation": "reduced connectivity can blunt coordinated affective processing",
                "schizoid_change": "increased",
            },
            {
                "source": "amygdala_social_emotional_blunting",
                "target": "amygdala",
                "relation": "social-emotional blunting burdens amygdala-centered circuitry",
                "schizoid_change": "increased",
            },
            {
                "source": "prefrontal_hypoactivity",
                "target": "prefrontal_cortex_proxy",
                "relation": "prefrontal underactivity manifests in broad PFC control territory",
                "schizoid_change": "increased",
            },
            {
                "source": "prefrontal_hypoactivity",
                "target": "social_cognition_underengagement",
                "relation": "reduced control-system engagement weakens social cognitive participation",
                "schizoid_change": "increased",
            },
            {
                "source": "amygdala_social_emotional_blunting",
                "target": "social_cognition_underengagement",
                "relation": "blunted emotional salience diminishes social learning and reciprocity",
                "schizoid_change": "increased",
            },
            {
                "source": "developmental_detachment_adaptation",
                "target": "social_cognition_underengagement",
                "relation": "early adaptation toward distance narrows social-cognitive engagement",
                "schizoid_change": "increased",
            },
            {
                "source": "monoamine_social_reward_blunting",
                "target": "motivational_negative_symptom_load",
                "relation": "attenuated reward signaling contributes to apathy and anhedonia",
                "schizoid_change": "increased",
            },
            {
                "source": "prefrontal_hypoactivity",
                "target": "motivational_negative_symptom_load",
                "relation": "prefrontal underactivity weakens initiative and goal-directed engagement",
                "schizoid_change": "increased",
            },
            {
                "source": "thalamocortical_dysconnectivity",
                "target": "motivational_negative_symptom_load",
                "relation": "distributed dysconnectivity supports negative-spectrum functional burden",
                "schizoid_change": "increased",
            },
            {
                "source": "social_cognition_underengagement",
                "target": "social_withdrawal",
                "relation": "reduced social cognitive engagement fosters withdrawal",
                "schizoid_change": "increased",
            },
            {
                "source": "motivational_negative_symptom_load",
                "target": "social_withdrawal",
                "relation": "negative-symptom burden reduces participation in relationships",
                "schizoid_change": "increased",
            },
            {
                "source": "developmental_detachment_adaptation",
                "target": "preference_for_solitude",
                "relation": "stable adaptation toward distance reinforces solitude-seeking",
                "schizoid_change": "increased",
            },
            {
                "source": "social_cognition_underengagement",
                "target": "preference_for_solitude",
                "relation": "low interpersonal salience favors solitary behavior",
                "schizoid_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "emotional_detachment",
                "relation": "blunted amygdala-centered processing contributes to emotional distance",
                "schizoid_change": "increased",
            },
            {
                "source": "social_cognition_underengagement",
                "target": "emotional_detachment",
                "relation": "reduced interpersonal salience weakens reciprocity and warmth",
                "schizoid_change": "increased",
            },
            {
                "source": "motivational_negative_symptom_load",
                "target": "avolition",
                "relation": "integrated negative-symptom burden produces reduced drive",
                "schizoid_change": "increased",
            },
            {
                "source": "prefrontal_cortex_proxy",
                "target": "avolition",
                "relation": "prefrontal hypoactivity weakens initiation and planning",
                "schizoid_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "flattened_affect",
                "relation": "blunted social-emotional processing restricts expressive range",
                "schizoid_change": "increased",
            },
            {
                "source": "monoamine_social_reward_blunting",
                "target": "flattened_affect",
                "relation": "monoaminergic attenuation can contribute to blunted affect",
                "schizoid_change": "increased",
            },
            {
                "source": "monoamine_social_reward_blunting",
                "target": "anhedonia",
                "relation": "reduced dopamine/serotonin signaling lowers pleasure and reward response",
                "schizoid_change": "increased",
            },
            {
                "source": "motivational_negative_symptom_load",
                "target": "anhedonia",
                "relation": "negative-symptom burden deepens low-pleasure states",
                "schizoid_change": "increased",
            },
            {
                "source": "supportive_relational_context",
                "target": "social_cognition_underengagement",
                "relation": "supportive relationships may increase interpersonal engagement",
                "schizoid_change": "decreased",
            },
            {
                "source": "supportive_relational_context",
                "target": "motivational_negative_symptom_load",
                "relation": "social scaffolding may partially counter apathy and disengagement",
                "schizoid_change": "decreased",
            },
        ]

        self.region_objects: Dict[str, Any] = {}
        self.receptors: Dict[str, pd.DataFrame] = {}
        self.genes: Dict[str, pd.DataFrame] = {}
        self.connectivity_profiles: Dict[str, pd.DataFrame] = {}
        self.region_resolution: Dict[str, Dict[str, Any]] = {}
        self.nodes_df = pd.DataFrame()
        self.edges_df = pd.DataFrame()

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    def _modality_candidates(self, kind: str) -> List[Any]:
        if self.siibra is None:
            return []
        cands: List[Any] = []
        try:
            if kind == "receptor":
                cands.append(self.siibra.features.molecular.ReceptorDensityFingerprint)
            elif kind == "gene":
                cands.append(self.siibra.features.molecular.GeneExpressions)
            elif kind == "connectivity":
                cands.append(self.siibra.features.connectivity.StreamlineCounts)
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
        if self.siibra is None:
            return []
        for modality in modalities:
            try:
                with self.siibra.QUIET:
                    feats = self.siibra.features.get(concept, modality, **kwargs)
                if feats:
                    return list(feats)
            except Exception:
                continue
        return []

    def _julich_matches(self, query: str) -> List[Any]:
        if self.atlas is None:
            return []
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "prefrontal cortex",
            "thalamus",
        } else 0
        proxy_penalty = 1 if "proxy" in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        if self.atlas is None or self.parcellation is None:
            return None

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
        rows = []
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
            rows.append({"name": row[0], "identifier": row[1], "parcellation": row[2]})
            if len(rows) >= limit:
                break
        return pd.DataFrame(rows)

    def _spatial_props_list(self, region: Any) -> List[Any]:
        if region is None or self.space is None:
            return []
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
        centroid_xyz = None
        if centroid is not None:
            try:
                centroid_xyz = tuple(float(x) for x in centroid)
            except Exception:
                centroid_xyz = None
        volume = getattr(main, "volume", None)
        try:
            volume_mm3 = float(volume) if volume is not None else None
        except Exception:
            volume_mm3 = None
        return centroid_xyz, volume_mm3

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            data = getattr(feat, "data", None)
            if isinstance(data, pd.DataFrame):
                df = data.copy().reset_index()
                if "index" in df.columns and "receptor" not in df.columns:
                    df = df.rename(columns={"index": "receptor"})
                return df
            try:
                df = pd.DataFrame(data).reset_index()
                if not df.empty:
                    if "index" in df.columns and "receptor" not in df.columns:
                        df = df.rename(columns={"index": "receptor"})
                    return df
            except Exception:
                continue
        return pd.DataFrame()

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()
        for feat in feats:
            try:
                df = feat.data.copy()
            except Exception:
                continue
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue

            lower_cols = {str(c).lower(): c for c in df.columns}
            required = {"gene", "level", "zscore"}
            if required.issubset(lower_cols):
                gene_col = lower_cols["gene"]
                level_col = lower_cols["level"]
                zscore_col = lower_cols["zscore"]
                try:
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
                    )
                except Exception:
                    return df.reset_index(drop=True)

            return df.reset_index(drop=True)
        return pd.DataFrame()

    def _extract_connectivity_data(self, feature: Any) -> pd.DataFrame:
        data = getattr(feature, "data", None)
        if isinstance(data, pd.DataFrame):
            return data.copy()
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix
        if self.parcellation is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        chosen = None
        for feat in feats:
            if getattr(feat, "cohort", None) == self.connectivity_cohort:
                chosen = feat
                break
        if chosen is None:
            chosen = feats[0]

        df = self._extract_connectivity_data(chosen)
        if df.empty:
            try:
                df = self._extract_connectivity_data(chosen[0])
            except Exception:
                df = pd.DataFrame()

        self._connectivity_matrix = df
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        if region is None:
            return None
        exact = [x for x in labels if self._name_of(x) == region.name]
        if exact:
            return exact[0]
        rn = region.name.lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        if fuzzy:
            return fuzzy[0]

        region_tokens = {tok for tok in rn.replace("(", " ").replace(")", " ").split() if len(tok) > 2}
        best = None
        best_score = 0
        for label in labels:
            label_name = self._name_of(label).lower()
            label_tokens = {tok for tok in label_name.replace("(", " ").replace(")", " ").split() if len(tok) > 2}
            score = len(region_tokens & label_tokens)
            if score > best_score:
                best = label
                best_score = score
        return best if best_score >= 2 else None

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty or region is None:
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
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            series = pd.Series(series).sort_values(ascending=False)
            df = series.reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name]
            return df.head(max_rows).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return a submatrix among resolved disorder-relevant regional nodes.

        Rows and columns are renamed to the model's stable node keys so the
        result remains readable even when atlas labels differ across versions.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        selected_index = []
        node_keys = []
        for node_key, region in self.region_objects.items():
            label = self._match_region_label(list(matrix.index), region)
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
            if label is None:
                continue
            selected_index.append(label)
            node_keys.append(node_key)

        if not selected_index:
            return pd.DataFrame()

        try:
            sub = matrix.loc[selected_index, selected_index].copy()
        except Exception:
            return pd.DataFrame()

        rename_map = {label: key for label, key in zip(selected_index, node_keys)}
        sub.index = [rename_map.get(idx, self._name_of(idx)) for idx in sub.index]
        sub.columns = [rename_map.get(col, self._name_of(col)) for col in sub.columns]
        return sub

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        nodes = []
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
        self.region_resolution = {}

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
            description = self.region_descriptions.get(key, "Atlas-backed proxy node")
            if region is None:
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                self.region_resolution[key] = {
                    "node_key": key,
                    "resolved": False,
                    "requested_candidates": list(candidates),
                    "resolved_name": None,
                    "identifier": None,
                }
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region_proxy" if key in self.proxy_region_nodes else "region",
                        "description": f"{description} Unresolved in this environment.",
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
            self.region_resolution[key] = {
                "node_key": key,
                "resolved": True,
                "requested_candidates": list(candidates),
                "resolved_name": region.name,
                "identifier": getattr(region, "identifier", None),
            }
            nodes.append(
                {
                    "key": key,
                    "label": region.name,
                    "node_type": "region_proxy" if key in self.proxy_region_nodes else "region",
                    "description": description,
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
        region_resolution_df = pd.DataFrame(self.region_resolution.values())
        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "region_resolution": region_resolution_df,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": self.circuit_connectivity(),
        }

    def simulate(
        self,
        schizophrenia_spectrum_genetic_liability: float = 0.55,
        neurodevelopmental_vulnerability: float = 0.5,
        childhood_adversity: float = 0.45,
        emotional_nonresponsive_environment: float = 0.65,
        social_impoverishment: float = 0.55,
        supportive_relational_context: float = 0.25,
    ) -> Dict[str, pd.Series]:
        """
        Transparent one-pass simulator for a schizoid-spectrum phenotype.

        Inputs are normalized to [0, 1]. The output expresses relative burden,
        not probability of diagnosis.
        """
        inputs = {
            "schizophrenia_spectrum_genetic_liability": schizophrenia_spectrum_genetic_liability,
            "neurodevelopmental_vulnerability": neurodevelopmental_vulnerability,
            "childhood_adversity": childhood_adversity,
            "emotional_nonresponsive_environment": emotional_nonresponsive_environment,
            "social_impoverishment": social_impoverishment,
            "supportive_relational_context": supportive_relational_context,
        }
        x = {k: self._clip01(v) for k, v in inputs.items()}

        latents: Dict[str, float] = {}
        latents["shared_spectrum_vulnerability"] = self._clip01(
            0.44 * x["schizophrenia_spectrum_genetic_liability"]
            + 0.36 * x["neurodevelopmental_vulnerability"]
            + 0.06 * x["childhood_adversity"]
            - 0.04 * x["supportive_relational_context"]
        )
        latents["developmental_detachment_adaptation"] = self._clip01(
            0.28 * x["emotional_nonresponsive_environment"]
            + 0.22 * x["social_impoverishment"]
            + 0.18 * x["childhood_adversity"]
            + 0.16 * latents["shared_spectrum_vulnerability"]
            - 0.16 * x["supportive_relational_context"]
        )
        latents["nmda_interneuron_hypofunction"] = self._clip01(
            0.42 * latents["shared_spectrum_vulnerability"]
            + 0.22 * latents["developmental_detachment_adaptation"]
            + 0.06 * x["neurodevelopmental_vulnerability"]
            - 0.06 * x["supportive_relational_context"]
        )
        latents["gaba_interneuron_dysregulation"] = self._clip01(
            0.46 * latents["nmda_interneuron_hypofunction"]
            + 0.14 * latents["shared_spectrum_vulnerability"]
            - 0.04 * x["supportive_relational_context"]
        )
        latents["monoamine_social_reward_blunting"] = self._clip01(
            0.34 * latents["nmda_interneuron_hypofunction"]
            + 0.24 * latents["shared_spectrum_vulnerability"]
            + 0.18 * latents["developmental_detachment_adaptation"]
            - 0.08 * x["supportive_relational_context"]
        )
        latents["synaptic_connectivity_disruption"] = self._clip01(
            0.30 * latents["nmda_interneuron_hypofunction"]
            + 0.22 * latents["gaba_interneuron_dysregulation"]
            + 0.18 * latents["shared_spectrum_vulnerability"]
            + 0.12 * x["neurodevelopmental_vulnerability"]
            - 0.06 * x["supportive_relational_context"]
        )
        latents["thalamic_sensory_hyperconnectivity"] = self._clip01(
            0.38 * latents["synaptic_connectivity_disruption"]
            + 0.14 * latents["nmda_interneuron_hypofunction"]
            + 0.06 * x["neurodevelopmental_vulnerability"]
        )
        latents["thalamic_prefrontal_hypoconnectivity"] = self._clip01(
            0.42 * latents["synaptic_connectivity_disruption"]
            + 0.18 * latents["shared_spectrum_vulnerability"]
            + 0.10 * latents["developmental_detachment_adaptation"]
            - 0.06 * x["supportive_relational_context"]
        )
        latents["thalamocortical_dysconnectivity"] = self._clip01(
            0.34 * latents["thalamic_sensory_hyperconnectivity"]
            + 0.42 * latents["thalamic_prefrontal_hypoconnectivity"]
            + 0.10 * latents["synaptic_connectivity_disruption"]
        )
        latents["prefrontal_hypoactivity"] = self._clip01(
            0.36 * latents["thalamic_prefrontal_hypoconnectivity"]
            + 0.26 * latents["monoamine_social_reward_blunting"]
            + 0.18 * latents["developmental_detachment_adaptation"]
            + 0.10 * latents["synaptic_connectivity_disruption"]
            - 0.12 * x["supportive_relational_context"]
        )
        latents["amygdala_social_emotional_blunting"] = self._clip01(
            0.34 * latents["developmental_detachment_adaptation"]
            + 0.28 * latents["monoamine_social_reward_blunting"]
            + 0.16 * latents["synaptic_connectivity_disruption"]
            - 0.08 * x["supportive_relational_context"]
        )
        latents["social_cognition_underengagement"] = self._clip01(
            0.30 * latents["amygdala_social_emotional_blunting"]
            + 0.28 * latents["prefrontal_hypoactivity"]
            + 0.22 * latents["developmental_detachment_adaptation"]
            + 0.08 * latents["thalamocortical_dysconnectivity"]
            - 0.12 * x["supportive_relational_context"]
        )
        latents["motivational_negative_symptom_load"] = self._clip01(
            0.34 * latents["monoamine_social_reward_blunting"]
            + 0.26 * latents["prefrontal_hypoactivity"]
            + 0.20 * latents["thalamocortical_dysconnectivity"]
            + 0.10 * latents["social_cognition_underengagement"]
            - 0.10 * x["supportive_relational_context"]
        )

        regional_state = {
            "amygdala": self._clip01(
                0.52 * latents["amygdala_social_emotional_blunting"]
                + 0.16 * latents["social_cognition_underengagement"]
                + 0.10 * latents["developmental_detachment_adaptation"]
            ),
            "prefrontal_cortex_proxy": self._clip01(
                0.54 * latents["prefrontal_hypoactivity"]
                + 0.20 * latents["thalamocortical_dysconnectivity"]
                + 0.12 * latents["motivational_negative_symptom_load"]
            ),
            "thalamus_proxy": self._clip01(
                0.58 * latents["thalamocortical_dysconnectivity"]
                + 0.16 * latents["shared_spectrum_vulnerability"]
                + 0.08 * latents["synaptic_connectivity_disruption"]
            ),
        }

        symptoms = {
            "social_withdrawal": self._clip01(
                0.30 * latents["social_cognition_underengagement"]
                + 0.22 * latents["motivational_negative_symptom_load"]
                + 0.16 * regional_state["prefrontal_cortex_proxy"]
                + 0.12 * regional_state["amygdala"]
                - 0.12 * x["supportive_relational_context"]
            ),
            "preference_for_solitude": 0.0,
            "emotional_detachment": self._clip01(
                0.34 * latents["amygdala_social_emotional_blunting"]
                + 0.22 * latents["developmental_detachment_adaptation"]
                + 0.18 * latents["social_cognition_underengagement"]
                - 0.10 * x["supportive_relational_context"]
            ),
            "avolition": self._clip01(
                0.42 * latents["motivational_negative_symptom_load"]
                + 0.24 * regional_state["prefrontal_cortex_proxy"]
                + 0.10 * latents["monoamine_social_reward_blunting"]
                - 0.10 * x["supportive_relational_context"]
            ),
            "flattened_affect": self._clip01(
                0.34 * regional_state["amygdala"]
                + 0.24 * latents["monoamine_social_reward_blunting"]
                + 0.14 * regional_state["prefrontal_cortex_proxy"]
            ),
            "anhedonia": self._clip01(
                0.46 * latents["monoamine_social_reward_blunting"]
                + 0.22 * latents["motivational_negative_symptom_load"]
                - 0.08 * x["supportive_relational_context"]
            ),
        }
        symptoms["preference_for_solitude"] = self._clip01(
            0.36 * latents["developmental_detachment_adaptation"]
            + 0.26 * symptoms["social_withdrawal"]
            + 0.16 * latents["social_cognition_underengagement"]
            - 0.10 * x["supportive_relational_context"]
        )

        phenotypes = {
            "core_schizoid_profile": self._clip01(
                (
                    symptoms["social_withdrawal"]
                    + symptoms["preference_for_solitude"]
                    + symptoms["emotional_detachment"]
                    + symptoms["avolition"]
                )
                / 4.0
            ),
            "negative_symptom_overlap_profile": self._clip01(
                (
                    symptoms["avolition"]
                    + symptoms["anhedonia"]
                    + symptoms["flattened_affect"]
                    + latents["motivational_negative_symptom_load"]
                )
                / 4.0
            ),
            "developmental_detachment_profile": self._clip01(
                (
                    latents["developmental_detachment_adaptation"]
                    + symptoms["emotional_detachment"]
                    + symptoms["preference_for_solitude"]
                    + latents["social_cognition_underengagement"]
                )
                / 4.0
            ),
            "thalamocortical_disconnection_profile": self._clip01(
                (
                    latents["thalamocortical_dysconnectivity"]
                    + regional_state["thalamus_proxy"]
                    + regional_state["prefrontal_cortex_proxy"]
                    + symptoms["social_withdrawal"]
                )
                / 4.0
            ),
        }

        return {
            "inputs": pd.Series(x, name="value"),
            "latents": pd.Series(latents, name="value"),
            "regional_state": pd.Series(regional_state, name="value"),
            "symptoms": pd.Series(symptoms, name="value"),
            "phenotypes": pd.Series(phenotypes, name="value"),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Probabilistically assign an MNI coordinate to Julich regions.

        Returns an empty dataframe if siibra or map resources are unavailable.
        """
        if self.siibra is None or self.parcellation is None:
            warnings.warn("siibra resources are unavailable; returning empty assignment.")
            return pd.DataFrame()

        try:
            if self._pmap is None:
                with self.siibra.QUIET:
                    self._pmap = self.siibra.get_map(
                        parcellation=self.parcellation_spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )

            point = self.siibra.Point(tuple(float(v) for v in xyz), space=self.assignment_space)
            with self.siibra.QUIET:
                assignments = self._pmap.assign(point)

            if not isinstance(assignments, pd.DataFrame):
                assignments = pd.DataFrame(assignments)

            for candidate in ("map value", "correlation", "intersection over union"):
                if candidate in assignments.columns:
                    assignments = assignments.sort_values(candidate, ascending=False)
                    break
            return assignments.reset_index(drop=True)
        except Exception as exc:
            warnings.warn(f"Coordinate assignment failed: {exc!r}")
            return pd.DataFrame()

    def region_mask(self, node_key: str) -> Any:
        """
        Fetch a regional mask for a resolved node.

        Returns a Nifti image if available, otherwise None.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            warnings.warn(f"No resolved region available for '{node_key}'.")
            return None
        try:
            mask = region.get_regional_mask(space=self.assignment_space, maptype="labelled")
            return mask.fetch() if hasattr(mask, "fetch") else mask
        except Exception as exc:
            warnings.warn(f"Could not fetch region mask for '{node_key}': {exc!r}")
            return None


if __name__ == "__main__":
    model = SchizoidPersonalityDisorderModel()
    bundle = model.build()

    print("\n=== Nodes (head) ===")
    print(bundle["nodes"].head(12).to_string(index=False))

    print("\n=== Edges (head) ===")
    print(bundle["edges"].head(12).to_string(index=False))

    print("\n=== Region resolution ===")
    rr = bundle["region_resolution"]
    if isinstance(rr, pd.DataFrame) and not rr.empty:
        print(rr.to_string(index=False))
    else:
        print("No region resolution information available.")

    print("\n=== Example connectivity subgraph ===")
    cc = bundle["circuit_connectivity"]
    if isinstance(cc, pd.DataFrame) and not cc.empty:
        print(cc.to_string())
    else:
        print("No circuit connectivity available in this environment.")

    print("\n=== Example amygdala gene table ===")
    amygdala_genes = bundle["genes"].get("amygdala", pd.DataFrame())
    if isinstance(amygdala_genes, pd.DataFrame) and not amygdala_genes.empty:
        print(amygdala_genes.head(10).to_string(index=False))
    else:
        print("No amygdala gene-expression table available in this environment.")

    print("\n=== Example simulation ===")
    sim = model.simulate(
        schizophrenia_spectrum_genetic_liability=0.58,
        neurodevelopmental_vulnerability=0.52,
        childhood_adversity=0.40,
        emotional_nonresponsive_environment=0.72,
        social_impoverishment=0.60,
        supportive_relational_context=0.22,
    )
    for name, series in sim.items():
        print(f"\n-- {name} --")
        print(series.sort_values(ascending=False).to_string())

    print("\n# Optional: model.assign_mni_point((0, -20, 8))")
    print("# Optional: model.region_mask('amygdala')")
