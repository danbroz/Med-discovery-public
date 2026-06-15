from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Medical-condition-related anxiety panel:
# - serotonin / norepinephrine / dopamine / GABA
# - stress / HPA-axis
# - inflammation / TBI-specific signaling
# - pain sensitization and synaptic plasticity
DEFAULT_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor
    "HTR2A",    # serotonin receptor
    "TPH2",     # serotonin synthesis
    "SLC6A2",   # norepinephrine transporter
    "ADRA2A",   # adrenergic receptor
    "DBH",      # dopamine beta hydroxylase
    "DRD2",     # dopamine receptor
    "SLC6A3",   # dopamine transporter
    "GABRA2",   # GABA-A receptor subunit
    "GABRB2",   # GABA-A receptor subunit
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
    "CRHR1",    # CRH signaling
    "IL1B",     # inflammation
    "IL6",      # inflammation
    "TNF",      # inflammation
    "EIF2AK3",  # PERK
    "TMEM173",  # STING
    "BDNF",     # synaptic plasticity
]


class AnxietyDueToMedicalConditionModel:
    """
    Atlas-grounded mechanistic scaffold for Anxiety Disorder Due to Another Medical Condition.

    What it does:
      1) Resolves chapter-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates multiple medical-driver pathways into secondary anxiety.

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

        # The chapter's core circuit is limbic-cortical: amygdala, hippocampus, PFC, ACC.
        # Left frontal/basal-ganglia stroke effects are represented with a left-PFC node
        # and a basal-ganglia proxy node. If the basal-ganglia proxy does not resolve in a
        # given atlas version, the rest of the model still works.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "DG (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "hippocampus",
            ],
            "acc": [
                "Area 33 (ACC) left",
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "ACC",
            ],
            "left_pfc": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "prefrontal",
            ],
            "basal_ganglia_proxy": [
                "Nucleus accumbens left",
                "Caudate nucleus left",
                "Putamen left",
                "basal ganglia",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "focal_cns_lesion": "Stroke or focal lesion burden affecting aminergic pathways and left frontal/subcortical systems",
            "traumatic_brain_injury": "TBI burden including focal frontal/temporal injury and diffuse axonal injury",
            "endocrine_catecholamine_excess": "Hyperthyroid / pheochromocytoma-like hyperadrenergic state",
            "chronic_pain_condition": "Pain-driven central sensitization and persistent bodily threat signaling",
            "early_medical_trauma": "Severe early illness/medical trauma shaping later stress vulnerability",
            "medical_stabilization": "Protective medical treatment and systemic stabilization",
        }

        self.latent_nodes: Dict[str, str] = {
            "amine_pathway_disruption": "Damage or dysfunction of serotonin/norepinephrine/dopamine pathways",
            "serotonergic_disruption": "Secondary serotonergic dysfunction",
            "noradrenergic_disruption": "Secondary noradrenergic dysfunction",
            "dopaminergic_disruption": "Secondary dopaminergic dysfunction",
            "gaba_destabilization": "Secondary inhibitory instability in anxiety circuitry",
            "catecholamine_hyperarousal": "Peripheral and central hyperadrenergic arousal state",
            "perk_sting_neuroinflammation": "TBI-linked PERK-STING inflammatory signaling",
            "neuroinflammation": "Cytokine-driven neuroimmune burden",
            "epigenetic_scarring": "Illness/stress-linked stable gene-expression change",
            "hpa_axis_dysregulation": "Stress-axis dysregulation",
            "central_sensitization": "Pain-amplification and persistent bodily threat encoding",
            "diffuse_axonal_disconnection": "TBI-linked network disconnection",
            "fear_network_instability": "Medical-disorder-driven instability of amygdala-hippocampus-PFC-ACC circuitry",
        }

        self.symptom_nodes: Dict[str, str] = {
            "autonomic_hyperarousal": "Palpitations, adrenergic arousal, and physiologic anxiety",
            "panic_like_symptoms": "Acute surges of fear and bodily alarm",
            "somatic_anxiety": "Anxiety dominated by bodily distress and internal threat sensations",
            "anxious_distress": "Persistent subjective anxiety and worry burden",
            "depression_anxiety_overlap": "Mixed anxious-depressive burden from shared aminergic dysfunction",
            "contextual_threat_bias": "Biased threat processing and poor contextual safety learning",
            "cognitive_affective_dysregulation": "Impaired regulation, attention, and affective control",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "focal_cns_lesion",
                "target": "amine_pathway_disruption",
                "relation": "disrupts aminergic pathways linked to mood and anxiety",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "focal_cns_lesion",
                "target": "left_pfc",
                "relation": "burdens left frontal cortical function",
                "medical_anxiety_change": "increased dysfunction",
            },
            {
                "source": "focal_cns_lesion",
                "target": "basal_ganglia_proxy",
                "relation": "burdens subcortical affective circuitry",
                "medical_anxiety_change": "increased dysfunction",
            },
            {
                "source": "traumatic_brain_injury",
                "target": "diffuse_axonal_disconnection",
                "relation": "disrupts large-scale communication across networks",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "traumatic_brain_injury",
                "target": "perk_sting_neuroinflammation",
                "relation": "activates inflammatory signaling after brain injury",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "traumatic_brain_injury",
                "target": "left_pfc",
                "relation": "burdens frontal-lobe function after contusion/disconnection",
                "medical_anxiety_change": "increased dysfunction",
            },
            {
                "source": "endocrine_catecholamine_excess",
                "target": "catecholamine_hyperarousal",
                "relation": "raises hyperadrenergic bodily and mental arousal",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "chronic_pain_condition",
                "target": "central_sensitization",
                "relation": "amplifies persistent bodily threat processing",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "chronic_pain_condition",
                "target": "epigenetic_scarring",
                "relation": "can induce stable molecular changes sustaining sensitization",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "early_medical_trauma",
                "target": "epigenetic_scarring",
                "relation": "creates long-term stress-response molecular vulnerability",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "early_medical_trauma",
                "target": "hpa_axis_dysregulation",
                "relation": "programs long-term stress-response vulnerability",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "medical_stabilization",
                "target": "catecholamine_hyperarousal",
                "relation": "buffers systemic hyperarousal",
                "medical_anxiety_change": "protective",
            },
            {
                "source": "medical_stabilization",
                "target": "neuroinflammation",
                "relation": "buffers downstream inflammatory burden",
                "medical_anxiety_change": "protective",
            },
            {
                "source": "medical_stabilization",
                "target": "fear_network_instability",
                "relation": "supports restoration of physiologic and neural stability",
                "medical_anxiety_change": "protective",
            },
            {
                "source": "amine_pathway_disruption",
                "target": "serotonergic_disruption",
                "relation": "reduces serotonergic regulation",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "amine_pathway_disruption",
                "target": "noradrenergic_disruption",
                "relation": "reduces/deranges noradrenergic signaling",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "amine_pathway_disruption",
                "target": "dopaminergic_disruption",
                "relation": "reduces/deranges dopaminergic signaling",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "amine_pathway_disruption",
                "target": "fear_network_instability",
                "relation": "destabilizes limbic-cortical threat processing",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "perk_sting_neuroinflammation",
                "target": "neuroinflammation",
                "relation": "amplifies cytokine and injury-linked inflammatory burden",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "neuroinflammation",
                "target": "epigenetic_scarring",
                "relation": "alters gene expression and synaptic/neurogenic processes",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "neuroinflammation",
                "target": "fear_network_instability",
                "relation": "destabilizes synaptic function in anxiety circuitry",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "epigenetic_scarring",
                "target": "hpa_axis_dysregulation",
                "relation": "sustains long-term stress-response vulnerability",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "central_sensitization",
                "target": "somatic_anxiety",
                "relation": "converts bodily threat processing into chronic anxiety symptoms",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "diffuse_axonal_disconnection",
                "target": "fear_network_instability",
                "relation": "weakens coordinated control across distributed circuits",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "diffuse_axonal_disconnection",
                "target": "left_pfc",
                "relation": "impairs frontal executive regulation",
                "medical_anxiety_change": "increased dysfunction",
            },
            {
                "source": "diffuse_axonal_disconnection",
                "target": "acc",
                "relation": "impairs monitoring and control integration",
                "medical_anxiety_change": "increased dysfunction",
            },
            {
                "source": "catecholamine_hyperarousal",
                "target": "autonomic_hyperarousal",
                "relation": "drives adrenergic symptoms and physiologic anxiety",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "catecholamine_hyperarousal",
                "target": "panic_like_symptoms",
                "relation": "promotes acute surges of bodily alarm",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "amygdala",
                "relation": "amplifies threat-reactive emotional processing",
                "medical_anxiety_change": "increased dysregulation",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "hippocampus",
                "relation": "burdens stress-sensitive contextual processing",
                "medical_anxiety_change": "reduced function",
            },
            {
                "source": "hpa_axis_dysregulation",
                "target": "left_pfc",
                "relation": "weakens top-down regulatory control",
                "medical_anxiety_change": "reduced function",
            },
            {
                "source": "fear_network_instability",
                "target": "amygdala",
                "relation": "loads threat detection and fear generation",
                "medical_anxiety_change": "increased dysregulation",
            },
            {
                "source": "fear_network_instability",
                "target": "hippocampus",
                "relation": "impairs contextual safety learning",
                "medical_anxiety_change": "reduced function",
            },
            {
                "source": "fear_network_instability",
                "target": "acc",
                "relation": "burdens conflict monitoring and emotional appraisal",
                "medical_anxiety_change": "reduced function",
            },
            {
                "source": "fear_network_instability",
                "target": "left_pfc",
                "relation": "burdens executive control over anxiety responses",
                "medical_anxiety_change": "reduced function",
            },
            {
                "source": "left_pfc",
                "target": "amygdala",
                "relation": "normally exerts top-down inhibition",
                "medical_anxiety_change": "reduced inhibition",
            },
            {
                "source": "amygdala",
                "target": "anxious_distress",
                "relation": "amplifies subjective threat and anxious affect",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "contextual_threat_bias",
                "relation": "biases perception toward danger",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "contextual_threat_bias",
                "relation": "reduced contextualization impairs safety discrimination",
                "medical_anxiety_change": "reduced contextualization",
            },
            {
                "source": "acc",
                "target": "cognitive_affective_dysregulation",
                "relation": "impaired monitoring weakens adaptive regulation",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "left_pfc",
                "target": "cognitive_affective_dysregulation",
                "relation": "impaired executive control weakens affect regulation",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "serotonergic_disruption",
                "target": "depression_anxiety_overlap",
                "relation": "contributes to mixed anxious-depressive burden",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "noradrenergic_disruption",
                "target": "depression_anxiety_overlap",
                "relation": "contributes to arousal/mood dysregulation",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "dopaminergic_disruption",
                "target": "depression_anxiety_overlap",
                "relation": "contributes to motivation and affective burden",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "autonomic_hyperarousal",
                "target": "panic_like_symptoms",
                "relation": "escalates physiologic alarm into panic-like episodes",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "autonomic_hyperarousal",
                "target": "somatic_anxiety",
                "relation": "amplifies bodily anxiety symptoms",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "panic_like_symptoms",
                "target": "anxious_distress",
                "relation": "intensifies ongoing anxiety",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "cognitive_affective_dysregulation",
                "target": "anxious_distress",
                "relation": "sustains difficulty controlling anxiety",
                "medical_anxiety_change": "increased",
            },
            {
                "source": "contextual_threat_bias",
                "target": "anxious_distress",
                "relation": "biases situations toward perceived danger",
                "medical_anxiety_change": "increased",
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
            try:
                matches = self.atlas.find_regions(
                    spec,
                    all_versions=False,
                    filter_children=False,
                    find_topmost=False,
                )
                for region in matches:
                    parc_name = getattr(getattr(region, "parcellation", None), "name", "")
                    if "julich" in str(parc_name).lower():
                        return region
            except Exception:
                pass
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        """
        Helper for tuning region proxies against the atlas.
        Useful for refining left-frontal or basal-ganglia candidates.
        """
        rows: List[Dict[str, Any]] = []
        seen = set()

        try:
            matches = self.atlas.find_regions(
                keyword,
                all_versions=False,
                filter_children=True,
                find_topmost=False,
            )
        except Exception:
            return pd.DataFrame(columns=["name", "identifier", "parcellation"])

        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" not in str(parc_name).lower():
                continue

            row = (
                self._name_of(region),
                getattr(region, "identifier", None),
                parc_name,
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

    def _safe_features(self, concept: Any, modality: Any, **kwargs: Any) -> List[Any]:
        try:
            with siibra.QUIET:
                feats = siibra.features.get(concept, modality, **kwargs)
            return list(feats) if feats else []
        except Exception:
            return []

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features(
            region,
            siibra.features.molecular.ReceptorDensityFingerprint,
        )
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
        feats = self._safe_features(
            region,
            siibra.features.molecular.GeneExpressions,
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

        feats = self._safe_features(
            self.parcellation,
            siibra.features.connectivity.StreamlineCounts,
        )
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (f for f in feats if getattr(f, "cohort", None) == self.connectivity_cohort),
            feats[0],
        )

        # Documented pattern: compound feature elements expose DataFrame matrices.
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
            "amygdala",
            "hippocampus",
            "acc",
            "left_pfc",
            "basal_ganglia_proxy",
        ),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the circuit.
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
                    "description": "Atlas-backed medical-anxiety circuit node",
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
        focal_cns_lesion: float,
        traumatic_brain_injury: float,
        endocrine_catecholamine_excess: float,
        chronic_pain_condition: float,
        early_medical_trauma: float,
        medical_stabilization: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        except medical_stabilization which is protective.
        """
        s = self._clip01(focal_cns_lesion)
        t = self._clip01(traumatic_brain_injury)
        e = self._clip01(endocrine_catecholamine_excess)
        p = self._clip01(chronic_pain_condition)
        m = self._clip01(early_medical_trauma)
        z = self._clip01(medical_stabilization)

        # Medical-driver latents
        amine_pathway_disruption = self._clip01(0.40 * s + 0.20 * t)
        perk_sting_neuroinflammation = self._clip01(0.45 * t + 0.10 * s)
        neuroinflammation = self._clip01(
            0.35 * perk_sting_neuroinflammation
            + 0.20 * p
            + 0.10 * m
            - 0.15 * z
        )
        epigenetic_scarring = self._clip01(
            0.35 * m + 0.25 * p + 0.10 * neuroinflammation
        )
        hpa_axis_dysregulation = self._clip01(
            0.35 * epigenetic_scarring + 0.20 * m + 0.10 * neuroinflammation - 0.10 * z
        )
        central_sensitization = self._clip01(0.55 * p + 0.15 * epigenetic_scarring)
        catecholamine_hyperarousal = self._clip01(
            0.60 * e + 0.15 * hpa_axis_dysregulation - 0.20 * z
        )
        diffuse_axonal_disconnection = self._clip01(
            0.45 * t + 0.20 * neuroinflammation + 0.10 * s
        )

        serotonergic_disruption = self._clip01(
            0.40 * amine_pathway_disruption + 0.15 * neuroinflammation
        )
        noradrenergic_disruption = self._clip01(
            0.35 * amine_pathway_disruption + 0.15 * catecholamine_hyperarousal
        )
        dopaminergic_disruption = self._clip01(
            0.30 * amine_pathway_disruption + 0.10 * neuroinflammation
        )
        gaba_destabilization = self._clip01(
            0.25 * t + 0.15 * neuroinflammation + 0.10 * amine_pathway_disruption
        )

        fear_network_instability = self._clip01(
            0.20 * amine_pathway_disruption
            + 0.20 * neuroinflammation
            + 0.20 * hpa_axis_dysregulation
            + 0.15 * diffuse_axonal_disconnection
            + 0.10 * catecholamine_hyperarousal
            + 0.05 * gaba_destabilization
            - 0.20 * z
        )

        # Regional state proxies
        amygdala = self._clip01(
            0.35 * fear_network_instability + 0.20 * hpa_axis_dysregulation
        )
        hippocampus = self._clip01(
            0.25 * hpa_axis_dysregulation
            + 0.20 * neuroinflammation
            + 0.10 * epigenetic_scarring
        )
        acc = self._clip01(
            0.25 * fear_network_instability
            + 0.15 * diffuse_axonal_disconnection
            + 0.10 * amine_pathway_disruption
        )
        left_pfc = self._clip01(
            0.30 * amine_pathway_disruption
            + 0.25 * diffuse_axonal_disconnection
            + 0.15 * hpa_axis_dysregulation
            + 0.10 * neuroinflammation
            - 0.15 * z
        )
        basal_ganglia_proxy = self._clip01(
            0.40 * amine_pathway_disruption + 0.15 * s
        )

        # Symptoms
        autonomic_hyperarousal = self._clip01(
            0.50 * catecholamine_hyperarousal
            + 0.15 * amygdala
            + 0.10 * noradrenergic_disruption
            - 0.15 * z
        )
        panic_like_symptoms = self._clip01(
            0.40 * autonomic_hyperarousal
            + 0.20 * catecholamine_hyperarousal
            + 0.10 * amygdala
        )
        somatic_anxiety = self._clip01(
            0.40 * central_sensitization
            + 0.30 * autonomic_hyperarousal
            + 0.10 * neuroinflammation
        )
        contextual_threat_bias = self._clip01(
            0.35 * amygdala + 0.20 * hippocampus + 0.10 * left_pfc
        )
        cognitive_affective_dysregulation = self._clip01(
            0.35 * left_pfc + 0.20 * acc + 0.10 * diffuse_axonal_disconnection
        )
        depression_anxiety_overlap = self._clip01(
            0.25 * serotonergic_disruption
            + 0.25 * noradrenergic_disruption
            + 0.15 * dopaminergic_disruption
            + 0.10 * left_pfc
            + 0.10 * hippocampus
            + 0.10 * neuroinflammation
        )
        anxious_distress = self._clip01(
            0.30 * amygdala
            + 0.20 * autonomic_hyperarousal
            + 0.15 * contextual_threat_bias
            + 0.15 * cognitive_affective_dysregulation
            + 0.10 * panic_like_symptoms
        )

        return {
            "inputs": pd.Series(
                {
                    "focal_cns_lesion": s,
                    "traumatic_brain_injury": t,
                    "endocrine_catecholamine_excess": e,
                    "chronic_pain_condition": p,
                    "early_medical_trauma": m,
                    "medical_stabilization": z,
                }
            ),
            "latents": pd.Series(
                {
                    "fear_network_instability": fear_network_instability,
                    "amine_pathway_disruption": amine_pathway_disruption,
                    "catecholamine_hyperarousal": catecholamine_hyperarousal,
                    "neuroinflammation": neuroinflammation,
                    "diffuse_axonal_disconnection": diffuse_axonal_disconnection,
                    "central_sensitization": central_sensitization,
                    "hpa_axis_dysregulation": hpa_axis_dysregulation,
                    "epigenetic_scarring": epigenetic_scarring,
                    "perk_sting_neuroinflammation": perk_sting_neuroinflammation,
                    "serotonergic_disruption": serotonergic_disruption,
                    "noradrenergic_disruption": noradrenergic_disruption,
                    "dopaminergic_disruption": dopaminergic_disruption,
                    "gaba_destabilization": gaba_destabilization,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "left_pfc": left_pfc,
                    "amygdala": amygdala,
                    "acc": acc,
                    "hippocampus": hippocampus,
                    "basal_ganglia_proxy": basal_ganglia_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "anxious_distress": anxious_distress,
                    "autonomic_hyperarousal": autonomic_hyperarousal,
                    "somatic_anxiety": somatic_anxiety,
                    "panic_like_symptoms": panic_like_symptoms,
                    "depression_anxiety_overlap": depression_anxiety_overlap,
                    "contextual_threat_bias": contextual_threat_bias,
                    "cognitive_affective_dysregulation": cognitive_affective_dysregulation,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "post_stroke_aminergic_profile": self._clip01(
                        0.45 * s + 0.30 * amine_pathway_disruption + 0.15 * depression_anxiety_overlap
                    ),
                    "tbi_inflammatory_anxiety_profile": self._clip01(
                        0.35 * t + 0.30 * neuroinflammation + 0.20 * diffuse_axonal_disconnection
                    ),
                    "endocrine_hyperadrenergic_profile": self._clip01(
                        0.45 * e + 0.30 * autonomic_hyperarousal + 0.15 * panic_like_symptoms
                    ),
                    "pain_sensitized_anxiety_profile": self._clip01(
                        0.40 * p + 0.30 * central_sensitization + 0.20 * somatic_anxiety
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-24, 12, -10)).head(10)
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
    model = AnxietyDueToMedicalConditionModel()

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
        bundle["edges"][
            ["source", "target", "relation", "medical_anxiety_change"]
        ].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["amygdala", "hippocampus", "acc", "left_pfc", "basal_ganglia_proxy"]:
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
        focal_cns_lesion=0.70,
        traumatic_brain_injury=0.55,
        endocrine_catecholamine_excess=0.30,
        chronic_pain_condition=0.60,
        early_medical_trauma=0.40,
        medical_stabilization=0.20,
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
    # print(model.assign_mni_point((-24, 12, -10)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("basal ganglia").to_string(index=False))
    # print(model.suggest_regions("frontal").to_string(index=False))
    # print(model.suggest_regions("cingulate").to_string(index=False))
