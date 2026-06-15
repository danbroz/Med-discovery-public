from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Bipolar / secondary mood-disorder-oriented panel:
# - monoamines
# - glutamate / GABA balance
# - endocrine / thyroid / cortisol stress biology
# - inflammation / trophic support
DEFAULT_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor
    "SLC6A2",   # norepinephrine transporter
    "DBH",      # dopamine beta hydroxylase / catecholamine synthesis step
    "DRD2",     # dopamine receptor
    "COMT",     # catecholamine metabolism
    "GABRA2",   # GABA-A receptor subunit
    "GABRB2",   # GABA-A receptor subunit
    "GRIN2B",   # NMDA receptor subunit
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
    "CRHR1",    # CRH signaling
    "BDNF",     # neurotrophic support / plasticity
    "THRA",     # thyroid hormone receptor alpha
    "THRB",     # thyroid hormone receptor beta
    "HSD11B1",  # glucocorticoid regulation
    "IL6",      # inflammation
    "TNF",      # inflammation
]


class BipolarDueToMedicalConditionModel:
    """
    Atlas-grounded mechanistic scaffold for Bipolar and Related Disorder
    Due to Another Medical Condition.

    What it does:
      1) Resolves chapter-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates lesion-, endocrine-, inflammatory-, epilepsy-, and
         TBI-driven mood dysregulation.

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

        # The chapter names a broad mood circuit:
        # PFC / ACC / amygdala / hippocampus / basal ganglia / thalamus,
        # and highlights OFC / vmPFC vulnerability in TBI.
        # Some subcortical nodes are proxies because exact Julich coverage
        # can vary by version.
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
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "orbitofrontal",
                "Fo4",
                "Fo3",
            ],
            "vmpfc_proxy": [
                "Area s32 (sACC) left",
                "Area p32 (pACC) left",
                "sACC",
                "pACC",
                "medial prefrontal",
            ],
            "basal_ganglia_proxy": [
                "caudate",
                "putamen",
                "accumbens",
                "basal ganglia",
            ],
            "thalamus_proxy": [
                "thalamus",
                "anterior thalamus",
                "mediodorsal thalamus",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "focal_cns_lesion": "Stroke/tumor/MS-like focal lesion burden affecting mood-regulatory pathways",
            "traumatic_brain_injury": "TBI burden including focal frontal/temporal injury and diffuse axonal injury",
            "thyroid_dysregulation": "Hypo- or hyperthyroid disturbance of central neurotransmitter systems",
            "cortisol_excess": "Cushing-like or steroid-induced endocrine mood destabilization",
            "epilepsy_burden": "Seizure-related excitatory/inhibitory instability and ictal/post-ictal mood shifts",
            "inflammatory_medical_burden": "Autoimmune/infectious/chronic illness inflammatory burden",
            "genetic_affective_liability": "Inherited liability for bipolarity, HPA dysregulation, or inflammatory sensitivity",
            "medical_correction": "Protective treatment of the underlying medical driver",
        }

        self.latent_nodes: Dict[str, str] = {
            "monoamine_disruption": "Serotonin/norepinephrine/dopamine pathway disturbance",
            "thyroid_neurochemical_shift": "Thyroid-linked mood destabilization through aminergic/catecholaminergic effects",
            "catecholamine_sensitivity": "Hyperadrenergic/catecholaminergic sensitivity contributing to mania-like activation",
            "cortisol_glutamate_pressure": "Hypercortisolemic pressure on monoamine/glutamate systems",
            "glutamate_gaba_instability": "Excitatory/inhibitory imbalance in limbic and seizure-related circuits",
            "diffuse_axonal_disconnection": "Network disconnection after TBI",
            "neuroinflammatory_stress": "Inflammatory brain burden linked to chronic illness",
            "bdnf_suppression": "Reduced neurotrophic support contributing to depressive burden",
            "hpa_axis_vulnerability": "Stress-axis vulnerability interacting with medical illness",
            "cortico_limbic_striatal_thalamic_instability": "Distributed mood-circuit dysfunction",
        }

        self.symptom_nodes: Dict[str, str] = {
            "mania_activation": "Manic/hypomanic activation and elevated activation state",
            "depressive_burden": "Depressive episode burden",
            "irritability_lability": "Emotional lability, irritability, and rapid affective shifts",
            "disinhibition": "Impulsivity, reduced restraint, and behavior disinhibition",
            "mixed_mood_instability": "Concurrent activation and dysphoria-like mixed instability",
            "seizure_linked_affective_shift": "Acute mood change around ictal/post-ictal periods",
            "cognitive_affective_dysregulation": "Poor regulation of mood in relation to internal and external cues",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "focal_cns_lesion",
                "target": "monoamine_disruption",
                "relation": "damages aminergic pathways implicated in mood regulation",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "focal_cns_lesion",
                "target": "left_pfc",
                "relation": "burdens left frontal regulation linked to post-lesion mood syndrome",
                "secondary_bipolar_change": "increased dysfunction",
            },
            {
                "source": "focal_cns_lesion",
                "target": "basal_ganglia_proxy",
                "relation": "burdens striatal circuitry implicated in secondary mood states",
                "secondary_bipolar_change": "increased dysfunction",
            },
            {
                "source": "traumatic_brain_injury",
                "target": "diffuse_axonal_disconnection",
                "relation": "severs white matter communication across distributed mood networks",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "traumatic_brain_injury",
                "target": "ofc",
                "relation": "burdens orbitofrontal circuitry linked to disinhibition and lability",
                "secondary_bipolar_change": "increased dysfunction",
            },
            {
                "source": "traumatic_brain_injury",
                "target": "vmpfc_proxy",
                "relation": "burdens ventromedial regulatory control over mood",
                "secondary_bipolar_change": "increased dysfunction",
            },
            {
                "source": "thyroid_dysregulation",
                "target": "thyroid_neurochemical_shift",
                "relation": "alters neurotransmitter turnover and catecholamine responsiveness",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "thyroid_dysregulation",
                "target": "catecholamine_sensitivity",
                "relation": "can create hyperadrenergic activation in thyrotoxic states",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "cortisol_excess",
                "target": "cortisol_glutamate_pressure",
                "relation": "pushes monoamine and glutamate systems toward mood destabilization",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "cortisol_excess",
                "target": "hpa_axis_vulnerability",
                "relation": "amplifies endocrine stress-system dysregulation",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "epilepsy_burden",
                "target": "glutamate_gaba_instability",
                "relation": "destabilizes excitatory/inhibitory balance in limbic circuits",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "epilepsy_burden",
                "target": "seizure_linked_affective_shift",
                "relation": "permits acute ictal and post-ictal mood changes",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "inflammatory_medical_burden",
                "target": "neuroinflammatory_stress",
                "relation": "raises inflammatory pressure on mood-regulatory systems",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "genetic_affective_liability",
                "target": "hpa_axis_vulnerability",
                "relation": "raises inherited vulnerability to stress-linked mood dysregulation",
                "secondary_bipolar_change": "increased susceptibility",
            },
            {
                "source": "genetic_affective_liability",
                "target": "monoamine_disruption",
                "relation": "lowers threshold for developing mood symptoms when the brain is stressed or injured",
                "secondary_bipolar_change": "increased susceptibility",
            },
            {
                "source": "medical_correction",
                "target": "thyroid_neurochemical_shift",
                "relation": "can normalize endocrine-driven mood symptoms",
                "secondary_bipolar_change": "protective",
            },
            {
                "source": "medical_correction",
                "target": "cortisol_glutamate_pressure",
                "relation": "can reduce endocrine neurochemical destabilization",
                "secondary_bipolar_change": "protective",
            },
            {
                "source": "medical_correction",
                "target": "cortico_limbic_striatal_thalamic_instability",
                "relation": "reduces downstream circuit dysregulation",
                "secondary_bipolar_change": "protective",
            },
            {
                "source": "thyroid_neurochemical_shift",
                "target": "monoamine_disruption",
                "relation": "shifts serotonergic and noradrenergic function",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "catecholamine_sensitivity",
                "target": "mania_activation",
                "relation": "drives hyperactivation, irritability, and mania-like states",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "cortisol_glutamate_pressure",
                "target": "glutamate_gaba_instability",
                "relation": "pushes excitation/inhibition balance toward instability",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "neuroinflammatory_stress",
                "target": "bdnf_suppression",
                "relation": "reduces neurotrophic support in stress-sensitive limbic regions",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "neuroinflammatory_stress",
                "target": "hpa_axis_vulnerability",
                "relation": "amplifies stress susceptibility in chronic illness",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "bdnf_suppression",
                "target": "hippocampus",
                "relation": "burdens stress-sensitive hippocampal plasticity",
                "secondary_bipolar_change": "reduced support",
            },
            {
                "source": "monoamine_disruption",
                "target": "depressive_burden",
                "relation": "supports depressive episodes under medical burden",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "glutamate_gaba_instability",
                "target": "mixed_mood_instability",
                "relation": "destabilizes limbic mood tone and seizure-linked affect",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "diffuse_axonal_disconnection",
                "target": "cortico_limbic_striatal_thalamic_instability",
                "relation": "weakens coordination across distributed mood circuits",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "hpa_axis_vulnerability",
                "target": "bdnf_suppression",
                "relation": "reduces trophic support under sustained stress",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "amygdala",
                "relation": "burdens limbic salience and affective reactivity",
                "secondary_bipolar_change": "increased dysregulation",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "acc",
                "relation": "burdens conflict monitoring and mood regulation",
                "secondary_bipolar_change": "reduced function",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "left_pfc",
                "relation": "burdens executive regulation of mood",
                "secondary_bipolar_change": "reduced function",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "basal_ganglia_proxy",
                "relation": "burdens subcortical action-selection and motivational circuitry",
                "secondary_bipolar_change": "reduced function",
            },
            {
                "source": "cortico_limbic_striatal_thalamic_instability",
                "target": "thalamus_proxy",
                "relation": "burdens thalamocortical mood-circuit integration",
                "secondary_bipolar_change": "reduced function",
            },
            {
                "source": "left_pfc",
                "target": "cognitive_affective_dysregulation",
                "relation": "impaired frontal control weakens mood regulation",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "ofc",
                "target": "disinhibition",
                "relation": "impaired orbitofrontal control promotes disinhibited affect and behavior",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "vmpfc_proxy",
                "target": "irritability_lability",
                "relation": "reduced ventromedial control promotes emotional lability",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "acc",
                "target": "mixed_mood_instability",
                "relation": "reduced regulatory monitoring destabilizes mixed states",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "irritability_lability",
                "relation": "increases reactive emotional intensity",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "depressive_burden",
                "relation": "stress-sensitive hippocampal dysfunction contributes to depressive burden",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "basal_ganglia_proxy",
                "target": "mania_activation",
                "relation": "striatal dysfunction can bias motivational and activation states",
                "secondary_bipolar_change": "increased",
            },
            {
                "source": "thalamus_proxy",
                "target": "cognitive_affective_dysregulation",
                "relation": "thalamocortical disruption weakens coordinated mood regulation",
                "secondary_bipolar_change": "increased",
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "anterior cingulate",
            "prefrontal cortex",
            "thalamus",
            "basal ganglia",
        } else 0
        return (left_bonus, right_penalty, generic_penalty)

    def _resolve_region(self, candidates: Sequence[str]) -> Optional[Any]:
        for spec in candidates:
            try:
                return self.atlas.get_region(spec, parcellation=self.parcellation_spec)
            except Exception:
                pass

            try:
                matches = self.parcellation.find(spec, filter_children=False)
                if matches:
                    matches = sorted(matches, key=self._region_rank)
                    return matches[0]
            except Exception:
                pass

        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        """
        Helper for tuning proxy searches against the atlas.
        Useful for refining basal-ganglia or thalamic candidates.
        """
        try:
            matches = self.parcellation.find(keyword, filter_children=False)
        except Exception:
            return pd.DataFrame(columns=["name", "identifier"])

        rows = []
        seen = set()
        for region in sorted(matches, key=self._region_rank):
            row = (self._name_of(region), getattr(region, "identifier", None))
            if row in seen:
                continue
            seen.add(row)
            rows.append(
                {
                    "name": row[0],
                    "identifier": row[1],
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
            "amygdala",
            "hippocampus",
            "acc",
            "left_pfc",
            "ofc",
            "vmpfc_proxy",
            "basal_ganglia_proxy",
            "thalamus_proxy",
        ),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the
        secondary bipolar mood circuit.
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
                    "description": "Atlas-backed secondary bipolar circuit node",
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
        thyroid_dysregulation: float,
        cortisol_excess: float,
        epilepsy_burden: float,
        inflammatory_medical_burden: float,
        genetic_affective_liability: float,
        medical_correction: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while medical_correction is protective.
        """
        l = self._clip01(focal_cns_lesion)
        t = self._clip01(traumatic_brain_injury)
        y = self._clip01(thyroid_dysregulation)
        c = self._clip01(cortisol_excess)
        e = self._clip01(epilepsy_burden)
        i = self._clip01(inflammatory_medical_burden)
        g = self._clip01(genetic_affective_liability)
        m = self._clip01(medical_correction)

        # Latent biology
        monoamine_disruption = self._clip01(
            0.35 * l + 0.20 * y + 0.10 * t + 0.10 * c + 0.10 * g
        )
        thyroid_neurochemical_shift = self._clip01(
            0.55 * y + 0.10 * g - 0.20 * m
        )
        catecholamine_sensitivity = self._clip01(
            0.40 * y + 0.25 * c - 0.15 * m
        )
        cortisol_glutamate_pressure = self._clip01(
            0.45 * c + 0.15 * g - 0.15 * m
        )
        glutamate_gaba_instability = self._clip01(
            0.45 * e + 0.20 * t + 0.15 * cortisol_glutamate_pressure
        )
        diffuse_axonal_disconnection = self._clip01(
            0.50 * t + 0.10 * l - 0.10 * m
        )
        neuroinflammatory_stress = self._clip01(
            0.45 * i + 0.15 * t + 0.10 * e - 0.10 * m
        )
        hpa_axis_vulnerability = self._clip01(
            0.30 * g + 0.25 * c + 0.15 * neuroinflammatory_stress - 0.10 * m
        )
        bdnf_suppression = self._clip01(
            0.35 * neuroinflammatory_stress + 0.25 * hpa_axis_vulnerability
        )
        cortico_limbic_striatal_thalamic_instability = self._clip01(
            0.20 * monoamine_disruption
            + 0.20 * diffuse_axonal_disconnection
            + 0.15 * glutamate_gaba_instability
            + 0.15 * thyroid_neurochemical_shift
            + 0.10 * bdnf_suppression
            + 0.10 * neuroinflammatory_stress
            - 0.20 * m
        )

        # Regional state proxies
        amygdala = self._clip01(
            0.30 * cortico_limbic_striatal_thalamic_instability + 0.15 * hpa_axis_vulnerability
        )
        hippocampus = self._clip01(
            0.35 * bdnf_suppression + 0.20 * hpa_axis_vulnerability + 0.10 * neuroinflammatory_stress
        )
        acc = self._clip01(
            0.30 * cortico_limbic_striatal_thalamic_instability + 0.10 * diffuse_axonal_disconnection
        )
        left_pfc = self._clip01(
            0.30 * monoamine_disruption
            + 0.25 * diffuse_axonal_disconnection
            + 0.10 * l
            - 0.15 * m
        )
        ofc = self._clip01(
            0.35 * t + 0.20 * diffuse_axonal_disconnection + 0.10 * catecholamine_sensitivity - 0.15 * m
        )
        vmpfc_proxy = self._clip01(
            0.30 * t + 0.20 * hpa_axis_vulnerability + 0.10 * amygdala - 0.15 * m
        )
        basal_ganglia_proxy = self._clip01(
            0.30 * l + 0.20 * monoamine_disruption + 0.10 * e
        )
        thalamus_proxy = self._clip01(
            0.25 * cortico_limbic_striatal_thalamic_instability + 0.15 * e
        )

        # Symptoms
        mania_activation = self._clip01(
            0.30 * catecholamine_sensitivity
            + 0.20 * thyroid_neurochemical_shift
            + 0.15 * basal_ganglia_proxy
            + 0.10 * ofc
            - 0.10 * m
        )
        depressive_burden = self._clip01(
            0.30 * monoamine_disruption
            + 0.25 * bdnf_suppression
            + 0.15 * hippocampus
            + 0.10 * neuroinflammatory_stress
            - 0.10 * m
        )
        irritability_lability = self._clip01(
            0.25 * amygdala + 0.25 * vmpfc_proxy + 0.15 * mania_activation + 0.10 * catecholamine_sensitivity
        )
        disinhibition = self._clip01(
            0.35 * ofc + 0.20 * vmpfc_proxy + 0.15 * mania_activation - 0.10 * m
        )
        mixed_mood_instability = self._clip01(
            0.25 * glutamate_gaba_instability
            + 0.20 * irritability_lability
            + 0.20 * mania_activation
            + 0.15 * depressive_burden
            + 0.10 * acc
        )
        seizure_linked_affective_shift = self._clip01(
            0.45 * e + 0.25 * glutamate_gaba_instability + 0.10 * thalamus_proxy
        )
        cognitive_affective_dysregulation = self._clip01(
            0.30 * left_pfc
            + 0.20 * acc
            + 0.15 * diffuse_axonal_disconnection
            + 0.10 * thalamus_proxy
            - 0.10 * m
        )

        return {
            "inputs": pd.Series(
                {
                    "focal_cns_lesion": l,
                    "traumatic_brain_injury": t,
                    "thyroid_dysregulation": y,
                    "cortisol_excess": c,
                    "epilepsy_burden": e,
                    "inflammatory_medical_burden": i,
                    "genetic_affective_liability": g,
                    "medical_correction": m,
                }
            ),
            "latents": pd.Series(
                {
                    "cortico_limbic_striatal_thalamic_instability": cortico_limbic_striatal_thalamic_instability,
                    "monoamine_disruption": monoamine_disruption,
                    "thyroid_neurochemical_shift": thyroid_neurochemical_shift,
                    "catecholamine_sensitivity": catecholamine_sensitivity,
                    "glutamate_gaba_instability": glutamate_gaba_instability,
                    "diffuse_axonal_disconnection": diffuse_axonal_disconnection,
                    "neuroinflammatory_stress": neuroinflammatory_stress,
                    "bdnf_suppression": bdnf_suppression,
                    "hpa_axis_vulnerability": hpa_axis_vulnerability,
                    "cortisol_glutamate_pressure": cortisol_glutamate_pressure,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "amygdala": amygdala,
                    "hippocampus": hippocampus,
                    "acc": acc,
                    "left_pfc": left_pfc,
                    "ofc": ofc,
                    "vmpfc_proxy": vmpfc_proxy,
                    "basal_ganglia_proxy": basal_ganglia_proxy,
                    "thalamus_proxy": thalamus_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "mania_activation": mania_activation,
                    "depressive_burden": depressive_burden,
                    "irritability_lability": irritability_lability,
                    "disinhibition": disinhibition,
                    "mixed_mood_instability": mixed_mood_instability,
                    "seizure_linked_affective_shift": seizure_linked_affective_shift,
                    "cognitive_affective_dysregulation": cognitive_affective_dysregulation,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "stroke_frontostriatal_mania_profile": self._clip01(
                        0.40 * l + 0.25 * mania_activation + 0.20 * basal_ganglia_proxy
                    ),
                    "thyroid_endocrine_mood_profile": self._clip01(
                        0.45 * y + 0.30 * thyroid_neurochemical_shift + 0.15 * mania_activation
                    ),
                    "steroid_cortisol_mixed_profile": self._clip01(
                        0.40 * c + 0.25 * mixed_mood_instability + 0.20 * hpa_axis_vulnerability
                    ),
                    "tbi_disinhibited_mood_profile": self._clip01(
                        0.40 * t + 0.25 * disinhibition + 0.20 * diffuse_axonal_disconnection
                    ),
                    "epilepsy_affective_instability_profile": self._clip01(
                        0.45 * e + 0.25 * seizure_linked_affective_shift + 0.20 * glutamate_gaba_instability
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-18, 20, -10)).head(10)
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
    model = BipolarDueToMedicalConditionModel()

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
            ["source", "target", "relation", "secondary_bipolar_change"]
        ].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in [
        "amygdala",
        "hippocampus",
        "acc",
        "left_pfc",
        "ofc",
        "vmpfc_proxy",
        "basal_ganglia_proxy",
        "thalamus_proxy",
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
        focal_cns_lesion=0.60,
        traumatic_brain_injury=0.55,
        thyroid_dysregulation=0.70,
        cortisol_excess=0.45,
        epilepsy_burden=0.35,
        inflammatory_medical_burden=0.40,
        genetic_affective_liability=0.50,
        medical_correction=0.20,
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
    # print(model.assign_mni_point((-18, 20, -10)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("thalamus").to_string(index=False))
    # print(model.suggest_regions("caudate").to_string(index=False))
    # print(model.suggest_regions("orbitofrontal").to_string(index=False))
