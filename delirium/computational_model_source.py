
from __future__ import annotations

"""
Delirium siibra scaffold.

This script turns a delirium chapter into a conservative, atlas-grounded
mechanistic scaffold using siibra where available. It is intended for
research prototyping and transparent hypothesis exploration only. It is not a
diagnostic, prognostic, or treatment tool.

Core modeling choices from the chapter:
- Delirium is modeled as an acute failure of distributed brain integration.
- The main latent biology emphasizes cholinergic deficiency, dopaminergic
  dysregulation, neuroinflammatory activation, HPA-axis stress burden,
  noradrenergic hyperarousal, GABA/glutamate instability, and large-scale
  network disintegration.
- Region anchors are deliberately conservative. When the chapter names a broad
  system rather than a precise parcel, proxy nodes are used.
- The simulator is intentionally simple and acyclic:
  inputs -> latent biology -> regional dysfunction burden -> symptoms -> phenotypes

The script degrades gracefully:
- If siibra is not installed, or if a feature/modality is unavailable, the
  atlas-backed parts stay empty instead of crashing.
- The simulator still runs even without atlas data.

Source chapter themes encoded here include:
- acute illness / surgery / ICU stress,
- inflammation and metabolic disruption,
- intoxication and especially withdrawal-related hyperexcitability,
- stress-hormone effects on hippocampal memory systems,
- low brain resilience from age, baseline impairment, and genetic liability,
- diffuse cortical slowing and functional disconnection underlying symptoms.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra  # type: ignore
    _SIIBRA_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    siibra = None  # type: ignore
    _SIIBRA_IMPORT_ERROR = exc


DEFAULT_GENE_PANEL = [
    # Cholinergic signaling / attention
    "CHAT",
    "SLC5A7",
    "CHRM1",
    "CHRM2",
    "CHRNA4",
    # Dopamine / salience
    "DRD2",
    "SLC6A3",
    "COMT",
    # Serotonin / arousal modulation
    "HTR1A",
    "HTR2A",
    "SLC6A4",
    # Noradrenergic stress signaling
    "SLC6A2",
    "DBH",
    # GABA / glutamate balance
    "GAD1",
    "GABRA1",
    "GRIN2A",
    "SLC1A2",
    # Stress, plasticity, inflammation, resilience
    "NR3C1",
    "FKBP5",
    "BDNF",
    "IL6",
    "TNF",
    "APOE",
]


class DeliriumModel:
    """
    Atlas-grounded mechanistic scaffold for delirium.

    Notes
    -----
    - This is a research scaffold, not a validated disease model.
    - "Regional state" in `simulate()` quantifies dysfunction burden in each
      region/proxy node, not healthy activation.
    - Proxies are used where the chapter stays systems-level or where a stable
      Julich label may vary across environments.
    """

    def __init__(
        self,
        atlas_spec: str = "human",
        parcellation_spec: str = "julich",
        space_spec: str = "icbm 2009c asym",
        connectivity_cohort: str = "HCP",
    ) -> None:
        self.atlas_spec = atlas_spec
        self.parcellation_spec = parcellation_spec
        self.space_spec = space_spec
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        self.atlas = None
        self.parcellation = None
        self.space = None
        self._atlas_available = False

        if siibra is None:
            warnings.warn(
                f"siibra could not be imported ({_SIIBRA_IMPORT_ERROR}). "
                "Atlas-backed methods will remain available only as graceful stubs; "
                "the simulator still works."
            )
        else:
            try:
                self.atlas = siibra.atlases.get(atlas_spec)
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
                self._atlas_available = (
                    self.atlas is not None and self.parcellation is not None and self.space is not None
                )
            except Exception as exc:
                warnings.warn(
                    f"Could not initialize siibra atlas resources: {exc}. "
                    "Atlas-backed helpers will degrade gracefully."
                )

        # Region anchors are conservative and left-hemisphere first by default.
        self.region_candidates: Dict[str, List[str]] = {
            "pfc_control": [
                "Area 9/46d left",
                "Area 9/46v left",
                "Area 46 left",
                "Area 45 (IFG) left",
                "prefrontal cortex",
            ],
            "parietal_attention": [
                "Area 7A (SPL) left",
                "Area 7P (SPL) left",
                "Area PGa (IPL) left",
                "Area PFm (IPL) left",
                "parietal cortex",
            ],
            "thalamus_proxy": [
                "thalamus left",
                "thalamus",
            ],
            "basal_ganglia_proxy": [
                "caudate left",
                "putamen left",
                "striatum left",
                "basal ganglia",
            ],
            "hippocampus": [
                "CA1 left",
                "CA2 left",
                "CA3 left",
                "DG left",
                "hippocampus left",
                "hippocampus",
            ],
            "locus_coeruleus_proxy": [
                "locus coeruleus left",
                "locus coeruleus",
                "brainstem",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "pfc_control": (
                "Prefrontal executive-control proxy for top-down attentional and "
                "integrative regulation named broadly in the chapter."
            ),
            "parietal_attention": (
                "Parietal attention-system proxy for orienting and sustaining "
                "attention, consistent with the chapter's attentional deficit emphasis."
            ),
            "thalamus_proxy": (
                "Thalamic gating proxy for fluctuating consciousness and "
                "thalamo-cortical information transfer."
            ),
            "basal_ganglia_proxy": (
                "Basal ganglia / striatal proxy for subcortical salience-motor loops "
                "and dopamine-sensitive dysregulation."
            ),
            "hippocampus": (
                "Hippocampal memory anchor reflecting stress vulnerability and "
                "episodic disorganization discussed in the chapter."
            ),
            "locus_coeruleus_proxy": (
                "Brainstem noradrenergic proxy reflecting chapter emphasis on "
                "stress-linked locus coeruleus hyperarousal."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "acute_medical_stress": (
                "Acute physiological burden from illness, surgery, ICU exposure, or other "
                "major medical stressors."
            ),
            "infection_inflammation": (
                "Peripheral or systemic inflammatory load capable of propagating into "
                "neuroinflammatory signaling."
            ),
            "metabolic_disruption": (
                "Metabolic, hypoxic, toxic, or homeostatic disturbance impairing brain function."
            ),
            "substance_intoxication": (
                "Substance-related intoxication contributing to acute neurotransmitter imbalance."
            ),
            "substance_withdrawal": (
                "Withdrawal-related hyperexcitability, especially relevant for alcohol or "
                "benzodiazepine withdrawal states."
            ),
            "sleep_wake_disruption": (
                "Sleep loss and circadian fragmentation that worsen cognition and stress burden."
            ),
            "psychological_stress": (
                "Fear, anxiety, and hospitalization-related stress that can amplify delirium risk."
            ),
            "advanced_age": (
                "Age-related reduction in reserve and resilience."
            ),
            "baseline_cognitive_impairment": (
                "Dementia or pre-existing cognitive impairment that lowers brain resilience."
            ),
            "genetic_vulnerability": (
                "Host-level molecular liability affecting transmitter systems, inflammation, "
                "and stress response efficiency."
            ),
            "physiologic_restoration": (
                "Protective correction of the underlying medical insult and restoration of systemic stability."
            ),
            "sleep_restoration": (
                "Protective restoration of restorative sleep and circadian regularity."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "low_brain_resilience": (
                "Trait-like vulnerability that lowers the threshold for decompensation into delirium."
            ),
            "cerebral_metabolic_failure": (
                "Acute failure of neuronal energy/homeostatic support producing diffuse brain dysfunction."
            ),
            "neuroinflammatory_activation": (
                "Inflammation-linked neuroimmune signaling contributing to acute cognitive failure."
            ),
            "hpa_axis_dysregulation": (
                "Stress-system and glucocorticoid dysregulation burdening memory and plasticity."
            ),
            "cholinergic_deficiency": (
                "Failure of cholinergic attentional support, a classic mechanistic theme in delirium."
            ),
            "dopaminergic_dysregulation": (
                "Excess or mistuned dopamine signaling increasing salience distortion and agitation risk."
            ),
            "noradrenergic_hyperarousal": (
                "Stress-driven arousal amplification, especially relevant for hyperactive delirium."
            ),
            "gaba_glutamate_instability": (
                "Excitation-inhibition imbalance, strongly relevant during sedative or alcohol withdrawal."
            ),
            "network_disintegration": (
                "Large-scale cortical-subcortical disconnection underlying fragmented cognition."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "attentional_impairment": (
                "Reduced ability to focus, sustain, direct, and shift attention."
            ),
            "arousal_fluctuation": (
                "Waxing and waning alertness or consciousness over hours to days."
            ),
            "disorientation": (
                "Impaired situational and contextual orientation."
            ),
            "memory_impairment": (
                "Difficulty storing and recalling information during the delirious state."
            ),
            "perceptual_disturbance": (
                "Misinterpretation of stimuli or frank perceptual distortion."
            ),
            "agitation": (
                "Hyperarousal, restlessness, or behavioral agitation."
            ),
            "psychomotor_slowing": (
                "Reduced processing speed and slowed psychomotor output consistent with diffuse cortical slowing."
            ),
            "sleep_wake_cycle_disruption": (
                "Nocturnal worsening and circadian instability."
            ),
        }

        # Directional chapter-derived claims.
        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "acute_medical_stress",
                "target": "cerebral_metabolic_failure",
                "relation": "acute illness or surgery strains cerebral energy balance",
                "delirium_change": "increased",
            },
            {
                "source": "infection_inflammation",
                "target": "neuroinflammatory_activation",
                "relation": "systemic inflammation amplifies neuroimmune signaling",
                "delirium_change": "increased",
            },
            {
                "source": "metabolic_disruption",
                "target": "cerebral_metabolic_failure",
                "relation": "metabolic disturbance impairs neuronal function",
                "delirium_change": "increased",
            },
            {
                "source": "substance_intoxication",
                "target": "dopaminergic_dysregulation",
                "relation": "intoxication perturbs salience and transmitter balance",
                "delirium_change": "increased",
            },
            {
                "source": "substance_withdrawal",
                "target": "gaba_glutamate_instability",
                "relation": "withdrawal produces central hyperexcitability",
                "delirium_change": "increased",
            },
            {
                "source": "substance_withdrawal",
                "target": "noradrenergic_hyperarousal",
                "relation": "withdrawal increases autonomic and arousal drive",
                "delirium_change": "increased",
            },
            {
                "source": "sleep_wake_disruption",
                "target": "hpa_axis_dysregulation",
                "relation": "sleep loss magnifies stress-hormone burden",
                "delirium_change": "increased",
            },
            {
                "source": "psychological_stress",
                "target": "hpa_axis_dysregulation",
                "relation": "anxiety and fear related to hospitalization activate stress pathways",
                "delirium_change": "increased",
            },
            {
                "source": "advanced_age",
                "target": "low_brain_resilience",
                "relation": "age lowers reserve and resilience",
                "delirium_change": "increased",
            },
            {
                "source": "baseline_cognitive_impairment",
                "target": "low_brain_resilience",
                "relation": "pre-existing impairment lowers reserve",
                "delirium_change": "increased",
            },
            {
                "source": "genetic_vulnerability",
                "target": "low_brain_resilience",
                "relation": "host molecular liability shapes vulnerability to decompensation",
                "delirium_change": "increased",
            },
            {
                "source": "low_brain_resilience",
                "target": "cholinergic_deficiency",
                "relation": "lower reserve permits cholinergic attentional collapse with smaller insults",
                "delirium_change": "increased",
            },
            {
                "source": "cerebral_metabolic_failure",
                "target": "cholinergic_deficiency",
                "relation": "metabolic stress impairs integrative neurotransmission",
                "delirium_change": "increased",
            },
            {
                "source": "cholinergic_deficiency",
                "target": "dopaminergic_dysregulation",
                "relation": "loss of cholinergic balance favors dopaminergic misregulation",
                "delirium_change": "increased",
            },
            {
                "source": "acute_medical_stress",
                "target": "hpa_axis_dysregulation",
                "relation": "physiological stress activates the HPA axis and cortisol release",
                "delirium_change": "increased",
            },
            {
                "source": "neuroinflammatory_activation",
                "target": "network_disintegration",
                "relation": "inflammation destabilizes distributed brain integration",
                "delirium_change": "increased",
            },
            {
                "source": "cholinergic_deficiency",
                "target": "network_disintegration",
                "relation": "attention-supporting cholinergic systems fail",
                "delirium_change": "increased",
            },
            {
                "source": "gaba_glutamate_instability",
                "target": "network_disintegration",
                "relation": "excitation-inhibition imbalance fragments coherent processing",
                "delirium_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "hippocampus",
                "relation": "cortisol burden stresses hippocampal memory circuitry",
                "delirium_change": "stressed",
            },
            {
                "source": "network_disintegration",
                "target": "pfc_control",
                "relation": "diffuse cerebral dysfunction impairs executive control",
                "delirium_change": "dysregulated",
            },
            {
                "source": "network_disintegration",
                "target": "parietal_attention",
                "relation": "functional disconnection weakens attentional network integration",
                "delirium_change": "dysregulated",
            },
            {
                "source": "cerebral_metabolic_failure",
                "target": "thalamus_proxy",
                "relation": "metabolic insult compromises thalamocortical gating",
                "delirium_change": "dysregulated",
            },
            {
                "source": "dopaminergic_dysregulation",
                "target": "basal_ganglia_proxy",
                "relation": "dopamine imbalance perturbs subcortical salience-motor loops",
                "delirium_change": "dysregulated",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "locus_coeruleus_proxy",
                "relation": "stress-linked locus coeruleus activation drives hyperarousal",
                "delirium_change": "hyperactive",
            },
            {
                "source": "pfc_control",
                "target": "attentional_impairment",
                "relation": "executive-attentional failure reduces focus and set shifting",
                "delirium_change": "increased",
            },
            {
                "source": "parietal_attention",
                "target": "attentional_impairment",
                "relation": "posterior attentional dysfunction impairs orienting and sustaining attention",
                "delirium_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "arousal_fluctuation",
                "relation": "thalamic gating instability contributes to fluctuating consciousness",
                "delirium_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "memory_impairment",
                "relation": "hippocampal stress undermines encoding and recall",
                "delirium_change": "increased",
            },
            {
                "source": "attentional_impairment",
                "target": "disorientation",
                "relation": "loss of sustained attention destabilizes coherent awareness of context",
                "delirium_change": "increased",
            },
            {
                "source": "dopaminergic_dysregulation",
                "target": "perceptual_disturbance",
                "relation": "aberrant salience may amplify perceptual misinterpretation",
                "delirium_change": "increased",
            },
            {
                "source": "locus_coeruleus_proxy",
                "target": "agitation",
                "relation": "noradrenergic overdrive contributes to hyperactive agitation",
                "delirium_change": "increased",
            },
            {
                "source": "network_disintegration",
                "target": "psychomotor_slowing",
                "relation": "diffuse cortical slowing manifests as reduced processing speed",
                "delirium_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "sleep_wake_cycle_disruption",
                "relation": "stress hormones destabilize restorative rhythms",
                "delirium_change": "increased",
            },
            {
                "source": "sleep_wake_cycle_disruption",
                "target": "arousal_fluctuation",
                "relation": "circadian fragmentation worsens waxing and waning alertness",
                "delirium_change": "increased",
            },
            {
                "source": "physiologic_restoration",
                "target": "cerebral_metabolic_failure",
                "relation": "stabilizing the underlying medical insult reduces acute brain failure",
                "delirium_change": "decreased",
            },
            {
                "source": "sleep_restoration",
                "target": "hpa_axis_dysregulation",
                "relation": "restorative sleep reduces stress-system overactivation",
                "delirium_change": "decreased",
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
        if siibra is None:
            return []

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
        if siibra is None or concept is None:
            return []
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
        """
        Lower tuples rank better.
        Prefer left hemisphere, specific labels, and Julich labels over generic names.
        """
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {"amygdala", "hippocampus", "prefrontal cortex", "thalamus"} else 0
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
            rows.append(
                {"name": row[0], "identifier": row[1], "parcellation": row[2]}
            )
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

    def _main_component(
        self, region: Any
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
        props = self._spatial_props_list(region)
        if not props:
            return None, None
        main = max(props, key=lambda p: getattr(p, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        volume_mm3 = float(getattr(main, "volume", float("nan")))
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
        if not genes:
            return pd.DataFrame()

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

        lower_cols = {c.lower(): c for c in df.columns}
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
        return df.reset_index(drop=True)

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix
        if self.parcellation is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        feats = self._safe_features_any(
            self.parcellation,
            self._modality_candidates("connectivity"),
        )
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
        if region is None:
            return None

        # Exact object identity
        for x in labels:
            if x is region:
                return x

        # Exact name
        exact = [x for x in labels if self._name_of(x) == self._name_of(region)]
        if exact:
            return exact[0]

        rn = self._name_of(region).lower()
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
            df = df[df["connected_region"] != self._name_of(region)].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return a within-model connectivity submatrix for resolved region nodes,
        using fuzzy matching against the selected connectivity matrix.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        selected_labels: Dict[str, Any] = {}
        for node_key, region in self.region_objects.items():
            row_label = self._match_region_label(list(matrix.index), region)
            col_label = self._match_region_label(list(matrix.columns), region)
            if row_label is not None and col_label is not None:
                if row_label in matrix.index and col_label in matrix.columns:
                    selected_labels[node_key] = row_label

        if not selected_labels:
            return pd.DataFrame()

        common = [lbl for lbl in selected_labels.values() if lbl in matrix.index and lbl in matrix.columns]
        if not common:
            return pd.DataFrame()

        try:
            sub = matrix.loc[common, common].copy()
        except Exception:
            return pd.DataFrame()

        inverse = {v: k for k, v in selected_labels.items()}
        sub.index = [inverse.get(x, self._name_of(x)) for x in sub.index]
        sub.columns = [inverse.get(x, self._name_of(x)) for x in sub.columns]
        return sub

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
            desc = self.region_node_descriptions.get(key, "Atlas-backed circuit node")
            if region is None:
                if self._atlas_available:
                    warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.upper(),
                        "node_type": "region",
                        "description": f"{desc} (unresolved in this environment)",
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
                    "label": self._name_of(region),
                    "node_type": "region",
                    "description": desc,
                    "atlas_region": self._name_of(region),
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

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Assign an MNI152 coordinate to Julich regions using a statistical map.

        Returns an empty dataframe if siibra or a statistical map is unavailable.
        """
        if siibra is None or self.parcellation is None:
            return pd.DataFrame()

        if self._pmap is None:
            try:
                with siibra.QUIET:
                    try:
                        self._pmap = siibra.get_map(
                            parcellation=self.parcellation_spec,
                            space=self.assignment_space,
                            maptype="statistical",
                        )
                    except Exception:
                        self._pmap = self.atlas.get_map(
                            parcellation=self.parcellation,
                            space=self.assignment_space,
                            maptype="statistical",
                        )
            except Exception:
                return pd.DataFrame()

        try:
            point = siibra.Point(tuple(xyz), space=self.assignment_space)
            with siibra.QUIET:
                assignments = self._pmap.assign(point)
        except Exception:
            return pd.DataFrame()

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str, maptype: str = "labelled") -> Any:
        """
        Return a regional mask/volume object for a resolved region node, or None.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        try:
            if hasattr(region, "get_regional_mask"):
                return region.get_regional_mask(self.assignment_space, maptype=maptype)
        except Exception:
            pass
        try:
            if hasattr(region, "fetch_regional_map"):
                return region.fetch_regional_map(self.assignment_space, maptype=maptype)
        except Exception:
            pass
        return None

    def simulate(
        self,
        acute_medical_stress: float = 0.70,
        infection_inflammation: float = 0.55,
        metabolic_disruption: float = 0.45,
        substance_intoxication: float = 0.05,
        substance_withdrawal: float = 0.00,
        sleep_wake_disruption: float = 0.65,
        psychological_stress: float = 0.40,
        advanced_age: float = 0.80,
        baseline_cognitive_impairment: float = 0.50,
        genetic_vulnerability: float = 0.30,
        physiologic_restoration: float = 0.20,
        sleep_restoration: float = 0.10,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        Values are clipped to [0, 1]. Higher regional-state values mean greater
        dysfunction burden of that region/proxy in the current simulated state.
        """
        inputs = pd.Series(
            {
                "acute_medical_stress": self._clip01(acute_medical_stress),
                "infection_inflammation": self._clip01(infection_inflammation),
                "metabolic_disruption": self._clip01(metabolic_disruption),
                "substance_intoxication": self._clip01(substance_intoxication),
                "substance_withdrawal": self._clip01(substance_withdrawal),
                "sleep_wake_disruption": self._clip01(sleep_wake_disruption),
                "psychological_stress": self._clip01(psychological_stress),
                "advanced_age": self._clip01(advanced_age),
                "baseline_cognitive_impairment": self._clip01(baseline_cognitive_impairment),
                "genetic_vulnerability": self._clip01(genetic_vulnerability),
                "physiologic_restoration": self._clip01(physiologic_restoration),
                "sleep_restoration": self._clip01(sleep_restoration),
            },
            name="value",
        )

        # Inputs -> latent biology
        low_brain_resilience = self._clip01(
            0.40 * inputs["advanced_age"]
            + 0.40 * inputs["baseline_cognitive_impairment"]
            + 0.20 * inputs["genetic_vulnerability"]
        )

        cerebral_metabolic_failure = self._clip01(
            0.35 * inputs["acute_medical_stress"]
            + 0.35 * inputs["metabolic_disruption"]
            + 0.15 * inputs["infection_inflammation"]
            + 0.10 * inputs["sleep_wake_disruption"]
            - 0.25 * inputs["physiologic_restoration"]
        )

        neuroinflammatory_activation = self._clip01(
            0.55 * inputs["infection_inflammation"]
            + 0.20 * inputs["acute_medical_stress"]
            + 0.10 * inputs["psychological_stress"]
            + 0.10 * low_brain_resilience
            - 0.20 * inputs["physiologic_restoration"]
        )

        hpa_axis_dysregulation = self._clip01(
            0.40 * inputs["acute_medical_stress"]
            + 0.30 * inputs["psychological_stress"]
            + 0.20 * inputs["sleep_wake_disruption"]
            + 0.10 * low_brain_resilience
            - 0.20 * inputs["physiologic_restoration"]
            - 0.15 * inputs["sleep_restoration"]
        )

        cholinergic_deficiency = self._clip01(
            0.30 * cerebral_metabolic_failure
            + 0.25 * low_brain_resilience
            + 0.15 * inputs["infection_inflammation"]
            + 0.15 * inputs["substance_intoxication"]
            + 0.10 * inputs["sleep_wake_disruption"]
            - 0.20 * inputs["physiologic_restoration"]
        )

        dopaminergic_dysregulation = self._clip01(
            0.35 * inputs["substance_intoxication"]
            + 0.20 * inputs["substance_withdrawal"]
            + 0.20 * cholinergic_deficiency
            + 0.15 * inputs["acute_medical_stress"]
            + 0.10 * inputs["sleep_wake_disruption"]
        )

        noradrenergic_hyperarousal = self._clip01(
            0.35 * inputs["psychological_stress"]
            + 0.25 * inputs["acute_medical_stress"]
            + 0.25 * inputs["substance_withdrawal"]
            + 0.10 * inputs["sleep_wake_disruption"]
            - 0.20 * inputs["sleep_restoration"]
        )

        gaba_glutamate_instability = self._clip01(
            0.45 * inputs["substance_withdrawal"]
            + 0.25 * inputs["substance_intoxication"]
            + 0.15 * inputs["metabolic_disruption"]
            + 0.10 * inputs["sleep_wake_disruption"]
            - 0.15 * inputs["physiologic_restoration"]
        )

        network_disintegration = self._clip01(
            0.22 * cholinergic_deficiency
            + 0.18 * cerebral_metabolic_failure
            + 0.18 * neuroinflammatory_activation
            + 0.14 * hpa_axis_dysregulation
            + 0.10 * dopaminergic_dysregulation
            + 0.08 * gaba_glutamate_instability
            + 0.10 * low_brain_resilience
            - 0.20 * inputs["physiologic_restoration"]
        )

        latents = pd.Series(
            {
                "low_brain_resilience": low_brain_resilience,
                "cerebral_metabolic_failure": cerebral_metabolic_failure,
                "neuroinflammatory_activation": neuroinflammatory_activation,
                "hpa_axis_dysregulation": hpa_axis_dysregulation,
                "cholinergic_deficiency": cholinergic_deficiency,
                "dopaminergic_dysregulation": dopaminergic_dysregulation,
                "noradrenergic_hyperarousal": noradrenergic_hyperarousal,
                "gaba_glutamate_instability": gaba_glutamate_instability,
                "network_disintegration": network_disintegration,
            },
            name="value",
        )

        # Latent biology -> regional dysfunction burden
        regional_state = pd.Series(
            {
                "pfc_control": self._clip01(
                    0.35 * network_disintegration
                    + 0.20 * hpa_axis_dysregulation
                    + 0.15 * dopaminergic_dysregulation
                    + 0.10 * low_brain_resilience
                    - 0.15 * inputs["physiologic_restoration"]
                    - 0.05 * inputs["sleep_restoration"]
                ),
                "parietal_attention": self._clip01(
                    0.35 * network_disintegration
                    + 0.25 * cholinergic_deficiency
                    + 0.15 * cerebral_metabolic_failure
                    + 0.10 * neuroinflammatory_activation
                    - 0.10 * inputs["physiologic_restoration"]
                ),
                "thalamus_proxy": self._clip01(
                    0.35 * cerebral_metabolic_failure
                    + 0.25 * neuroinflammatory_activation
                    + 0.15 * gaba_glutamate_instability
                    + 0.10 * network_disintegration
                    - 0.10 * inputs["physiologic_restoration"]
                ),
                "basal_ganglia_proxy": self._clip01(
                    0.35 * dopaminergic_dysregulation
                    + 0.20 * gaba_glutamate_instability
                    + 0.15 * network_disintegration
                    + 0.10 * cerebral_metabolic_failure
                ),
                "hippocampus": self._clip01(
                    0.35 * hpa_axis_dysregulation
                    + 0.20 * neuroinflammatory_activation
                    + 0.20 * cerebral_metabolic_failure
                    + 0.15 * inputs["sleep_wake_disruption"]
                    - 0.10 * inputs["sleep_restoration"]
                ),
                "locus_coeruleus_proxy": self._clip01(
                    0.45 * noradrenergic_hyperarousal
                    + 0.20 * inputs["substance_withdrawal"]
                    + 0.10 * hpa_axis_dysregulation
                    + 0.10 * network_disintegration
                    - 0.10 * inputs["sleep_restoration"]
                ),
            },
            name="value",
        )

        # Regional dysfunction -> symptoms
        attentional_impairment = self._clip01(
            0.35 * regional_state["parietal_attention"]
            + 0.25 * regional_state["pfc_control"]
            + 0.20 * cholinergic_deficiency
            + 0.10 * network_disintegration
            + 0.05 * cerebral_metabolic_failure
        )

        arousal_fluctuation = self._clip01(
            0.25 * regional_state["thalamus_proxy"]
            + 0.20 * regional_state["locus_coeruleus_proxy"]
            + 0.20 * inputs["sleep_wake_disruption"]
            + 0.15 * gaba_glutamate_instability
            + 0.10 * network_disintegration
            - 0.10 * inputs["sleep_restoration"]
        )

        disorientation = self._clip01(
            0.40 * attentional_impairment
            + 0.25 * regional_state["hippocampus"]
            + 0.15 * arousal_fluctuation
            + 0.10 * regional_state["pfc_control"]
        )

        memory_impairment = self._clip01(
            0.45 * regional_state["hippocampus"]
            + 0.25 * cholinergic_deficiency
            + 0.15 * hpa_axis_dysregulation
            + 0.10 * attentional_impairment
        )

        perceptual_disturbance = self._clip01(
            0.30 * dopaminergic_dysregulation
            + 0.20 * arousal_fluctuation
            + 0.20 * network_disintegration
            + 0.15 * regional_state["thalamus_proxy"]
            + 0.10 * inputs["substance_withdrawal"]
        )

        agitation = self._clip01(
            0.35 * regional_state["locus_coeruleus_proxy"]
            + 0.20 * dopaminergic_dysregulation
            + 0.15 * inputs["substance_withdrawal"]
            + 0.10 * arousal_fluctuation
            - 0.10 * inputs["sleep_restoration"]
        )

        psychomotor_slowing = self._clip01(
            0.30 * cerebral_metabolic_failure
            + 0.20 * network_disintegration
            + 0.20 * cholinergic_deficiency
            + 0.15 * regional_state["pfc_control"]
            - 0.15 * regional_state["locus_coeruleus_proxy"]
        )

        sleep_wake_cycle_disruption = self._clip01(
            0.35 * inputs["sleep_wake_disruption"]
            + 0.25 * hpa_axis_dysregulation
            + 0.20 * arousal_fluctuation
            + 0.10 * regional_state["locus_coeruleus_proxy"]
            - 0.20 * inputs["sleep_restoration"]
        )

        symptoms = pd.Series(
            {
                "attentional_impairment": attentional_impairment,
                "arousal_fluctuation": arousal_fluctuation,
                "disorientation": disorientation,
                "memory_impairment": memory_impairment,
                "perceptual_disturbance": perceptual_disturbance,
                "agitation": agitation,
                "psychomotor_slowing": psychomotor_slowing,
                "sleep_wake_cycle_disruption": sleep_wake_cycle_disruption,
            },
            name="value",
        )

        # Symptom bundles / summary phenotypes
        hyperactive_delirium_profile = self._clip01(
            0.35 * agitation
            + 0.25 * perceptual_disturbance
            + 0.25 * arousal_fluctuation
            + 0.10 * noradrenergic_hyperarousal
        )

        hypoactive_delirium_profile = self._clip01(
            0.35 * psychomotor_slowing
            + 0.25 * attentional_impairment
            + 0.20 * disorientation
            + 0.10 * cerebral_metabolic_failure
        )

        mixed_delirium_profile = self._clip01(
            0.45 * hyperactive_delirium_profile
            + 0.35 * hypoactive_delirium_profile
            + 0.10 * arousal_fluctuation
        )

        withdrawal_delirium_profile = self._clip01(
            0.35 * inputs["substance_withdrawal"]
            + 0.25 * gaba_glutamate_instability
            + 0.20 * agitation
            + 0.15 * perceptual_disturbance
        )

        global_delirium_burden = self._clip01(
            0.25 * attentional_impairment
            + 0.20 * disorientation
            + 0.15 * memory_impairment
            + 0.15 * arousal_fluctuation
            + 0.15 * perceptual_disturbance
            + 0.10 * psychomotor_slowing
        )

        phenotypes = pd.Series(
            {
                "hyperactive_delirium_profile": hyperactive_delirium_profile,
                "hypoactive_delirium_profile": hypoactive_delirium_profile,
                "mixed_delirium_profile": mixed_delirium_profile,
                "withdrawal_delirium_profile": withdrawal_delirium_profile,
                "global_delirium_burden": global_delirium_burden,
            },
            name="value",
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }


if __name__ == "__main__":
    model = DeliriumModel()

    print("\n=== Building atlas-backed scaffold ===")
    scaffold = model.build(connectivity_rows=10)

    node_cols = ["key", "node_type", "atlas_region", "feature_summary"]
    print("\nNodes:")
    print(scaffold["nodes"][node_cols].to_string(index=False))

    print("\nEdges (first 18):")
    print(scaffold["edges"].head(18).to_string(index=False))

    if "hippocampus" in scaffold["receptors"] and not scaffold["receptors"]["hippocampus"].empty:
        print("\nHippocampus receptor fingerprint:")
        print(scaffold["receptors"]["hippocampus"].head().to_string(index=False))
    else:
        print("\nNo hippocampal receptor table available in this environment.")

    if "pfc_control" in scaffold["genes"] and not scaffold["genes"]["pfc_control"].empty:
        print("\nPFC-control gene summary:")
        print(scaffold["genes"]["pfc_control"].head().to_string(index=False))
    else:
        print("\nNo PFC-control gene table available in this environment.")

    if (
        "parietal_attention" in scaffold["connectivity_profiles"]
        and not scaffold["connectivity_profiles"]["parietal_attention"].empty
    ):
        print("\nParietal attention connectivity profile:")
        print(scaffold["connectivity_profiles"]["parietal_attention"].head().to_string(index=False))
    else:
        print("\nNo parietal attention connectivity profile available in this environment.")

    if not scaffold["circuit_connectivity"].empty:
        print("\nWithin-model circuit connectivity:")
        print(scaffold["circuit_connectivity"].round(3).to_string())
    else:
        print("\nNo within-model circuit connectivity matrix available in this environment.")

    print("\n=== Simulation example: vulnerable older adult with medical stress ===")
    sim = model.simulate(
        acute_medical_stress=0.85,
        infection_inflammation=0.65,
        metabolic_disruption=0.55,
        substance_intoxication=0.05,
        substance_withdrawal=0.00,
        sleep_wake_disruption=0.75,
        psychological_stress=0.45,
        advanced_age=0.90,
        baseline_cognitive_impairment=0.70,
        genetic_vulnerability=0.35,
        physiologic_restoration=0.20,
        sleep_restoration=0.10,
    )
    for name, series in sim.items():
        print(f"\n{name}:")
        print(series.sort_values(ascending=False).to_string())

    # Example coordinate assignment:
    # print(model.assign_mni_point((-24, -12, -20)).head())

    # Example regional mask retrieval:
    # mask = model.region_mask("hippocampus")
    # if mask is not None:
    #     print(mask)
