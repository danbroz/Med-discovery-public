from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Cyclothymic Disorder-oriented panel:
# - bipolar-spectrum genetic liability
# - GABA / glutamate balance
# - catecholamine modulation
# - neurotrophic / oxidative stress biology
DEFAULT_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor
    "HTR2A",    # serotonin receptor
    "TPH2",     # serotonin synthesis
    "SLC6A2",   # norepinephrine transporter
    "DBH",      # norepinephrine synthesis step
    "DRD2",     # dopamine receptor
    "SLC6A3",   # dopamine transporter
    "COMT",     # catecholamine metabolism
    "GABRA2",   # GABA-A receptor
    "GABRB2",   # GABA-A receptor
    "GAD1",     # GABA synthesis
    "GRIN2B",   # NMDA receptor subunit
    "SLC1A1",   # glutamate transport
    "BDNF",     # neurotrophic support / plasticity
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
]


class CyclothymicDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Cyclothymic Disorder.

    What it does:
      1) Resolves chapter-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates subthreshold depressive/hypomanic shifts, affective lability,
         irritability, and unstable cognitive control.

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

        # Chapter-consistent mood-regulation network
        self.region_candidates: Dict[str, List[str]] = {
            "dlpfc": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "dorsolateral prefrontal",
                "middle frontal",
                "superior frontal",
            ],
            "acc": [
                "Area 33 (ACC) left",
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "ACC",
                "anterior cingulate",
            ],
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
            "thalamus_proxy": [
                "thalamus",
                "anterior thalamus",
                "mediodorsal thalamus",
            ],
            "basal_ganglia_proxy": [
                "caudate",
                "putamen",
                "accumbens",
                "striatum",
                "basal ganglia",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "bipolar_family_loading": "Family-history and bipolar-spectrum genetic loading",
            "adolescent_onset_loading": "Developmental sensitivity of adolescence/early adulthood",
            "psychosocial_stress_trigger": "Stress burden that can destabilize mood states",
            "gaba_glutamate_instability": "Excitatory-inhibitory instability contributing to mood switching",
            "oxidative_plasticity_stress": "Oxidative/neurotrophic strain contributing to vulnerability",
            "mood_stabilizing_support": "Protective stabilization and treatment support",
        }

        self.latent_nodes: Dict[str, str] = {
            "bipolar_spectrum_genetic_liability": "Inherited bipolar-spectrum liability expressed as softer affective instability",
            "developmental_onset_sensitivity": "Developmental sensitivity of onset window",
            "gaba_glutamate_balance_failure": "Unstable inhibitory/excitatory balance across mood states",
            "oxidative_neurotrophic_strain": "Reduced neurotrophic support and oxidative stress burden",
            "cortico_limbic_dysregulation": "Impaired coordination between cortical control and limbic emotion systems",
            "prefrontal_top_down_failure": "Weak top-down cognitive control over emotional shifts",
            "limbic_mood_hyperreactivity": "Emotion-generating systems biasing mood state intensity",
            "thalamo_striatal_mood_gating_dysregulation": "Subcortical gating instability in mood/motor/cognitive loops",
            "mood_state_switching_instability": "Liability to repeated oscillation between low and activated states",
        }

        self.symptom_nodes: Dict[str, str] = {
            "subthreshold_depressive_shifts": "Low-energy, low-mood, subthreshold depressive periods",
            "hypomanic_activation_shifts": "Activated, agitated, or elevated subthreshold hypomanic periods",
            "affective_lability": "Rapid or unstable mood fluctuation",
            "irritability_agitation": "Irritable or agitated activation states",
            "cognitive_control_fluctuation": "Variable attention, planning, and regulation across states",
            "chronic_bipolar_spectrum_instability": "Persistent cyclothymic instability across time",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "bipolar_family_loading",
                "target": "bipolar_spectrum_genetic_liability",
                "relation": "raises inherited bipolar-spectrum liability",
                "cyclothymia_change": "increased susceptibility",
            },
            {
                "source": "bipolar_family_loading",
                "target": "developmental_onset_sensitivity",
                "relation": "family loading is associated with early and stronger expression",
                "cyclothymia_change": "increased susceptibility",
            },
            {
                "source": "adolescent_onset_loading",
                "target": "developmental_onset_sensitivity",
                "relation": "loads a sensitive developmental period for expression",
                "cyclothymia_change": "increased",
            },
            {
                "source": "psychosocial_stress_trigger",
                "target": "mood_state_switching_instability",
                "relation": "stress can destabilize mood switching thresholds",
                "cyclothymia_change": "increased",
            },
            {
                "source": "psychosocial_stress_trigger",
                "target": "oxidative_neurotrophic_strain",
                "relation": "stress amplifies neurotrophic and oxidative burden",
                "cyclothymia_change": "increased",
            },
            {
                "source": "gaba_glutamate_instability",
                "target": "gaba_glutamate_balance_failure",
                "relation": "directly destabilizes inhibitory-excitatory balance",
                "cyclothymia_change": "increased",
            },
            {
                "source": "oxidative_plasticity_stress",
                "target": "oxidative_neurotrophic_strain",
                "relation": "raises cellular vulnerability relevant to mood instability",
                "cyclothymia_change": "increased",
            },
            {
                "source": "mood_stabilizing_support",
                "target": "mood_state_switching_instability",
                "relation": "buffers oscillation between low and activated states",
                "cyclothymia_change": "protective",
            },
            {
                "source": "mood_stabilizing_support",
                "target": "prefrontal_top_down_failure",
                "relation": "supports better top-down regulation",
                "cyclothymia_change": "protective",
            },
            {
                "source": "bipolar_spectrum_genetic_liability",
                "target": "mood_state_switching_instability",
                "relation": "shared bipolar liability supports unstable mood cycling",
                "cyclothymia_change": "increased",
            },
            {
                "source": "bipolar_spectrum_genetic_liability",
                "target": "thalamo_striatal_mood_gating_dysregulation",
                "relation": "shared subcortical gating liability may destabilize mood networks",
                "cyclothymia_change": "increased",
            },
            {
                "source": "developmental_onset_sensitivity",
                "target": "cortico_limbic_dysregulation",
                "relation": "early developmental expression can destabilize maturing mood circuits",
                "cyclothymia_change": "increased",
            },
            {
                "source": "gaba_glutamate_balance_failure",
                "target": "mood_state_switching_instability",
                "relation": "unstable E/I balance supports oscillation between low and activated states",
                "cyclothymia_change": "increased",
            },
            {
                "source": "gaba_glutamate_balance_failure",
                "target": "irritability_agitation",
                "relation": "activation-side imbalance can promote agitation and irritability",
                "cyclothymia_change": "increased",
            },
            {
                "source": "oxidative_neurotrophic_strain",
                "target": "hippocampus",
                "relation": "reduced neurotrophic support burdens hippocampal resilience",
                "cyclothymia_change": "increased dysregulation",
            },
            {
                "source": "oxidative_neurotrophic_strain",
                "target": "cortico_limbic_dysregulation",
                "relation": "cellular stress weakens coordinated mood regulation",
                "cyclothymia_change": "increased",
            },
            {
                "source": "cortico_limbic_dysregulation",
                "target": "dlpfc",
                "relation": "burdens cognitive-control regions",
                "cyclothymia_change": "increased dysregulation",
            },
            {
                "source": "cortico_limbic_dysregulation",
                "target": "acc",
                "relation": "burdens affective monitoring and conflict regulation",
                "cyclothymia_change": "increased dysregulation",
            },
            {
                "source": "cortico_limbic_dysregulation",
                "target": "amygdala",
                "relation": "burdens emotional-response generation circuitry",
                "cyclothymia_change": "increased dysregulation",
            },
            {
                "source": "cortico_limbic_dysregulation",
                "target": "hippocampus",
                "relation": "burdens memory-context and stress-sensitive limbic circuitry",
                "cyclothymia_change": "increased dysregulation",
            },
            {
                "source": "cortico_limbic_dysregulation",
                "target": "prefrontal_top_down_failure",
                "relation": "weakens regulatory control over emotion-generating systems",
                "cyclothymia_change": "increased",
            },
            {
                "source": "prefrontal_top_down_failure",
                "target": "dlpfc",
                "relation": "reflects reduced dorsolateral control of cognition and affect",
                "cyclothymia_change": "reduced function",
            },
            {
                "source": "prefrontal_top_down_failure",
                "target": "cognitive_control_fluctuation",
                "relation": "reduces stable executive function across mood states",
                "cyclothymia_change": "increased",
            },
            {
                "source": "limbic_mood_hyperreactivity",
                "target": "amygdala",
                "relation": "amplifies limbic emotional output",
                "cyclothymia_change": "increased dysregulation",
            },
            {
                "source": "limbic_mood_hyperreactivity",
                "target": "affective_lability",
                "relation": "heightens rapid mood fluctuation",
                "cyclothymia_change": "increased",
            },
            {
                "source": "thalamo_striatal_mood_gating_dysregulation",
                "target": "thalamus_proxy",
                "relation": "burdens relay and gating functions in mood circuits",
                "cyclothymia_change": "increased dysregulation",
            },
            {
                "source": "thalamo_striatal_mood_gating_dysregulation",
                "target": "basal_ganglia_proxy",
                "relation": "burdens subcortical gating of mood, drive, and psychomotor tone",
                "cyclothymia_change": "increased dysregulation",
            },
            {
                "source": "thalamo_striatal_mood_gating_dysregulation",
                "target": "hypomanic_activation_shifts",
                "relation": "subcortical gating instability can bias activated states",
                "cyclothymia_change": "increased",
            },
            {
                "source": "mood_state_switching_instability",
                "target": "subthreshold_depressive_shifts",
                "relation": "supports repeated depressive-direction mood fluctuations",
                "cyclothymia_change": "increased",
            },
            {
                "source": "mood_state_switching_instability",
                "target": "hypomanic_activation_shifts",
                "relation": "supports repeated activation-direction mood fluctuations",
                "cyclothymia_change": "increased",
            },
            {
                "source": "mood_state_switching_instability",
                "target": "affective_lability",
                "relation": "produces unstable oscillation across affective states",
                "cyclothymia_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "irritability_agitation",
                "relation": "limbic overactivity promotes irritability and agitation",
                "cyclothymia_change": "increased",
            },
            {
                "source": "acc",
                "target": "affective_lability",
                "relation": "ACC dysregulation destabilizes affective monitoring",
                "cyclothymia_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "subthreshold_depressive_shifts",
                "relation": "stress-sensitive hippocampal burden can bias low mood states",
                "cyclothymia_change": "increased",
            },
            {
                "source": "dlpfc",
                "target": "cognitive_control_fluctuation",
                "relation": "dlPFC burden impairs stable cognitive regulation across states",
                "cyclothymia_change": "increased",
            },
            {
                "source": "subthreshold_depressive_shifts",
                "target": "chronic_bipolar_spectrum_instability",
                "relation": "persistent depressive-direction shifts stabilize spectrum burden",
                "cyclothymia_change": "increased",
            },
            {
                "source": "hypomanic_activation_shifts",
                "target": "chronic_bipolar_spectrum_instability",
                "relation": "persistent activation-direction shifts stabilize spectrum burden",
                "cyclothymia_change": "increased",
            },
            {
                "source": "affective_lability",
                "target": "chronic_bipolar_spectrum_instability",
                "relation": "rapid unstable affect contributes to chronic cycling phenotype",
                "cyclothymia_change": "increased",
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
            "hippocampus",
            "anterior cingulate",
            "prefrontal cortex",
            "thalamus",
            "striatum",
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

    def _main_component(self, region: Any) -> Tuple[Optional[Tuple[float, float, float]], Optional[float]]:
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
        node_keys: Sequence[str] = ("dlpfc", "acc", "hippocampus", "amygdala", "thalamus_proxy", "basal_ganglia_proxy"),
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
                    "description": "Atlas-backed cyclothymia circuit node",
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
        bipolar_family_loading: float,
        adolescent_onset_loading: float,
        psychosocial_stress_trigger: float,
        gaba_glutamate_instability: float,
        oxidative_plasticity_stress: float,
        mood_stabilizing_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while mood_stabilizing_support is protective.
        """
        b = self._clip01(bipolar_family_loading)
        a = self._clip01(adolescent_onset_loading)
        s = self._clip01(psychosocial_stress_trigger)
        g = self._clip01(gaba_glutamate_instability)
        o = self._clip01(oxidative_plasticity_stress)
        m = self._clip01(mood_stabilizing_support)

        # Latent biology
        bipolar_spectrum_genetic_liability = self._clip01(
            0.50 * b + 0.15 * a
        )
        developmental_onset_sensitivity = self._clip01(
            0.40 * a + 0.20 * bipolar_spectrum_genetic_liability
        )
        gaba_glutamate_balance_failure = self._clip01(
            0.50 * g + 0.10 * developmental_onset_sensitivity - 0.10 * m
        )
        oxidative_neurotrophic_strain = self._clip01(
            0.45 * o + 0.20 * s + 0.10 * bipolar_spectrum_genetic_liability - 0.10 * m
        )
        cortico_limbic_dysregulation = self._clip01(
            0.25 * developmental_onset_sensitivity
            + 0.25 * oxidative_neurotrophic_strain
            + 0.15 * gaba_glutamate_balance_failure
            + 0.10 * s
            - 0.15 * m
        )
        prefrontal_top_down_failure = self._clip01(
            0.35 * cortico_limbic_dysregulation
            + 0.20 * s
            - 0.20 * m
        )
        limbic_mood_hyperreactivity = self._clip01(
            0.35 * cortico_limbic_dysregulation
            + 0.20 * s
            + 0.10 * bipolar_spectrum_genetic_liability
            - 0.10 * m
        )
        thalamo_striatal_mood_gating_dysregulation = self._clip01(
            0.30 * bipolar_spectrum_genetic_liability
            + 0.20 * gaba_glutamate_balance_failure
            + 0.15 * cortico_limbic_dysregulation
            - 0.10 * m
        )
        mood_state_switching_instability = self._clip01(
            0.25 * bipolar_spectrum_genetic_liability
            + 0.25 * gaba_glutamate_balance_failure
            + 0.20 * limbic_mood_hyperreactivity
            + 0.10 * psychosocial_stress_trigger
            - 0.20 * m
        )

        # Regional state proxies
        dlpfc = self._clip01(
            0.45 * prefrontal_top_down_failure
            + 0.15 * cortico_limbic_dysregulation
            - 0.10 * m
        )
        acc = self._clip01(
            0.35 * cortico_limbic_dysregulation
            + 0.20 * mood_state_switching_instability
            - 0.10 * m
        )
        hippocampus = self._clip01(
            0.35 * oxidative_neurotrophic_strain
            + 0.20 * cortico_limbic_dysregulation
        )
        amygdala = self._clip01(
            0.45 * limbic_mood_hyperreactivity
            + 0.10 * mood_state_switching_instability
            - 0.05 * m
        )
        thalamus_proxy = self._clip01(
            0.45 * thalamo_striatal_mood_gating_dysregulation
            + 0.10 * mood_state_switching_instability
        )
        basal_ganglia_proxy = self._clip01(
            0.45 * thalamo_striatal_mood_gating_dysregulation
            + 0.15 * mood_state_switching_instability
        )

        # Symptoms
        subthreshold_depressive_shifts = self._clip01(
            0.35 * hippocampus
            + 0.20 * dlpfc
            + 0.20 * mood_state_switching_instability
            + 0.10 * oxidative_neurotrophic_strain
            - 0.10 * m
        )
        hypomanic_activation_shifts = self._clip01(
            0.30 * basal_ganglia_proxy
            + 0.25 * thalamus_proxy
            + 0.20 * mood_state_switching_instability
            - 0.10 * m
        )
        affective_lability = self._clip01(
            0.35 * amygdala
            + 0.25 * mood_state_switching_instability
            + 0.15 * acc
            - 0.10 * m
        )
        irritability_agitation = self._clip01(
            0.35 * amygdala
            + 0.25 * hypomanic_activation_shifts
            + 0.10 * gaba_glutamate_balance_failure
            - 0.10 * m
        )
        cognitive_control_fluctuation = self._clip01(
            0.35 * dlpfc
            + 0.20 * subthreshold_depressive_shifts
            + 0.15 * affective_lability
            - 0.10 * m
        )
        chronic_bipolar_spectrum_instability = self._clip01(
            0.30 * subthreshold_depressive_shifts
            + 0.30 * hypomanic_activation_shifts
            + 0.20 * affective_lability
            - 0.10 * m
        )

        return {
            "inputs": pd.Series(
                {
                    "bipolar_family_loading": b,
                    "adolescent_onset_loading": a,
                    "psychosocial_stress_trigger": s,
                    "gaba_glutamate_instability": g,
                    "oxidative_plasticity_stress": o,
                    "mood_stabilizing_support": m,
                }
            ),
            "latents": pd.Series(
                {
                    "mood_state_switching_instability": mood_state_switching_instability,
                    "cortico_limbic_dysregulation": cortico_limbic_dysregulation,
                    "thalamo_striatal_mood_gating_dysregulation": thalamo_striatal_mood_gating_dysregulation,
                    "prefrontal_top_down_failure": prefrontal_top_down_failure,
                    "limbic_mood_hyperreactivity": limbic_mood_hyperreactivity,
                    "oxidative_neurotrophic_strain": oxidative_neurotrophic_strain,
                    "gaba_glutamate_balance_failure": gaba_glutamate_balance_failure,
                    "developmental_onset_sensitivity": developmental_onset_sensitivity,
                    "bipolar_spectrum_genetic_liability": bipolar_spectrum_genetic_liability,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "dlpfc": dlpfc,
                    "acc": acc,
                    "hippocampus": hippocampus,
                    "amygdala": amygdala,
                    "thalamus_proxy": thalamus_proxy,
                    "basal_ganglia_proxy": basal_ganglia_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "chronic_bipolar_spectrum_instability": chronic_bipolar_spectrum_instability,
                    "affective_lability": affective_lability,
                    "hypomanic_activation_shifts": hypomanic_activation_shifts,
                    "subthreshold_depressive_shifts": subthreshold_depressive_shifts,
                    "irritability_agitation": irritability_agitation,
                    "cognitive_control_fluctuation": cognitive_control_fluctuation,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "soft_bipolar_instability_profile": self._clip01(
                        0.45 * chronic_bipolar_spectrum_instability
                        + 0.25 * bipolar_spectrum_genetic_liability
                        + 0.15 * mood_state_switching_instability
                    ),
                    "depressive_bias_profile": self._clip01(
                        0.45 * subthreshold_depressive_shifts
                        + 0.25 * hippocampus
                        + 0.15 * oxidative_neurotrophic_strain
                    ),
                    "activation_bias_profile": self._clip01(
                        0.45 * hypomanic_activation_shifts
                        + 0.25 * thalamo_striatal_mood_gating_dysregulation
                        + 0.15 * irritability_agitation
                    ),
                    "stress_sensitive_cyclothymia_profile": self._clip01(
                        0.40 * psychosocial_stress_trigger
                        + 0.30 * affective_lability
                        + 0.20 * cortico_limbic_dysregulation
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-22, 24, -10)).head(10)
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
    model = CyclothymicDisorderModel()

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
        bundle["edges"][["source", "target", "relation", "cyclothymia_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["dlpfc", "acc", "hippocampus", "amygdala", "thalamus_proxy", "basal_ganglia_proxy"]:
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
        bipolar_family_loading=0.80,
        adolescent_onset_loading=0.70,
        psychosocial_stress_trigger=0.65,
        gaba_glutamate_instability=0.55,
        oxidative_plasticity_stress=0.45,
        mood_stabilizing_support=0.20,
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
    # print(model.assign_mni_point((-22, 24, -10)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("thalamus").to_string(index=False))
    # print(model.suggest_regions("caudate").to_string(index=False))
