from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Catatonia-oriented panel:
# - GABA / inhibitory failure
# - glutamate / NMDA / excitotoxicity
# - dopamine / motor gating
# - monoaminergic modulation
# - stress / neuroplastic vulnerability
DEFAULT_GENE_PANEL = [
    "GABRA2",   # GABA-A receptor
    "GABRB2",   # GABA-A receptor
    "GAD1",     # GABA synthesis
    "GRIN1",    # NMDA receptor subunit
    "GRIN2B",   # NMDA receptor subunit
    "SLC1A1",   # glutamate transport
    "DRD2",     # dopamine receptor
    "SLC6A3",   # dopamine transporter
    "COMT",     # catecholamine metabolism
    "SLC6A4",   # serotonin transporter
    "SLC6A2",   # norepinephrine transporter
    "DBH",      # norepinephrine synthesis step
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
    "CRHR1",    # CRH signaling
    "BDNF",     # neuroplasticity
]


class CatatonicDisorderDueToMedicalConditionModel:
    """
    Atlas-grounded mechanistic scaffold for Catatonic Disorder Due to Another Medical Condition.

    What it does:
      1) Resolves chapter-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates psychomotor shutdown, catatonic agitation, rigidity/posturing,
         autonomic instability, and recovery pressure.

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

        # Chapter-consistent fronto-limbic / thalamo-basal-ganglia circuit.
        self.region_candidates: Dict[str, List[str]] = {
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "orbitofrontal",
                "Fo4",
                "Fo3",
            ],
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
            "basal_ganglia_proxy": [
                "caudate",
                "putamen",
                "accumbens",
                "striatum",
                "basal ganglia",
            ],
            "thalamus_proxy": [
                "thalamus",
                "anterior thalamus",
                "mediodorsal thalamus",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "neurological_insult": "Neurological insult such as encephalitis, stroke, TBI, or tumor",
            "metabolic_toxic_burden": "Metabolic, toxic, hepatic, calcium, or medication-related physiological burden",
            "autoimmune_nmda_process": "Autoimmune/NMDA-mediated catatonic trigger",
            "genetic_hyperexcitability": "Inherited vulnerability for neuronal hyperexcitability",
            "mood_psychosis_diathesis": "Latent mood/psychosis-spectrum vulnerability unmasked by illness",
            "gaba_nmda_treatment_support": "Protective GABA-supportive / NMDA-modulating treatment and stabilization",
        }

        self.latent_nodes: Dict[str, str] = {
            "gabaergic_failure": "Failure of inhibitory GABAergic control",
            "nmda_disruption": "NMDA-receptor dysfunction or autoimmune blockade",
            "glutamate_excitotoxicity": "Glutamate-driven hyperexcitability or excitotoxic stress",
            "dopaminergic_motor_gating_failure": "Dopamine-linked disruption of motor/cognitive/emotional integration",
            "monoaminergic_modulatory_shift": "Serotonergic/noradrenergic modulatory shift",
            "neuronal_hyperexcitability": "Global hyperexcitable brain state",
            "hypofrontality": "Reduced orbital/dorsolateral/medial frontal function",
            "thalamo_basal_ganglia_relay_failure": "Breakdown of relay/gating across motor and volitional circuits",
            "frontolimbic_motor_circuit_breakdown": "Failure of coordinated frontal-limbic-subcortical control",
            "homeostatic_restoration": "Recovery of inhibitory-excitatory balance and circuit stability",
        }

        self.symptom_nodes: Dict[str, str] = {
            "psychomotor_shutdown": "Stupor, immobility, and profound psychomotor inhibition",
            "mutism_volitional_block": "Mutism and failure of goal-directed initiation",
            "posturing_rigidity": "Rigidity, posturing, or maintenance of abnormal position",
            "catatonic_agitation": "Episodes of agitated or excited catatonia",
            "affective_withdrawal": "Affective and behavioral withdrawal",
            "autonomic_instability": "Risk of malignant/autonomic dysregulation",
            "catatonic_recovery": "Return of psychomotor and behavioral control",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "neurological_insult",
                "target": "glutamate_excitotoxicity",
                "relation": "acute brain injury can produce glutamate-mediated hyperexcitability",
                "catatonia_change": "increased",
            },
            {
                "source": "neurological_insult",
                "target": "frontolimbic_motor_circuit_breakdown",
                "relation": "damages or functionally disrupts core psychomotor regulation circuits",
                "catatonia_change": "increased",
            },
            {
                "source": "neurological_insult",
                "target": "thalamo_basal_ganglia_relay_failure",
                "relation": "disrupts relay systems for motor initiation and volition",
                "catatonia_change": "increased",
            },
            {
                "source": "metabolic_toxic_burden",
                "target": "gabaergic_failure",
                "relation": "physiological and toxic states can destabilize inhibitory tone",
                "catatonia_change": "increased",
            },
            {
                "source": "metabolic_toxic_burden",
                "target": "monoaminergic_modulatory_shift",
                "relation": "medical illness can perturb modulatory transmitter systems",
                "catatonia_change": "increased",
            },
            {
                "source": "metabolic_toxic_burden",
                "target": "neuronal_hyperexcitability",
                "relation": "metabolic and toxic burden can push the CNS toward instability",
                "catatonia_change": "increased",
            },
            {
                "source": "autoimmune_nmda_process",
                "target": "nmda_disruption",
                "relation": "autoimmune NMDA pathology directly disrupts glutamatergic signaling",
                "catatonia_change": "increased",
            },
            {
                "source": "autoimmune_nmda_process",
                "target": "glutamate_excitotoxicity",
                "relation": "NMDA-related dysfunction destabilizes excitatory control",
                "catatonia_change": "increased",
            },
            {
                "source": "genetic_hyperexcitability",
                "target": "neuronal_hyperexcitability",
                "relation": "genetic variants can lower the threshold for catatonic decompensation",
                "catatonia_change": "increased susceptibility",
            },
            {
                "source": "genetic_hyperexcitability",
                "target": "gabaergic_failure",
                "relation": "genetic risk may weaken inhibitory resilience under challenge",
                "catatonia_change": "increased susceptibility",
            },
            {
                "source": "mood_psychosis_diathesis",
                "target": "hypofrontality",
                "relation": "psychosis/mood-spectrum vulnerability may predispose frontal control failure",
                "catatonia_change": "increased susceptibility",
            },
            {
                "source": "mood_psychosis_diathesis",
                "target": "frontolimbic_motor_circuit_breakdown",
                "relation": "latent vulnerability may be unmasked by severe medical stress",
                "catatonia_change": "increased susceptibility",
            },
            {
                "source": "gaba_nmda_treatment_support",
                "target": "gabaergic_failure",
                "relation": "supports restoration of inhibitory control",
                "catatonia_change": "protective",
            },
            {
                "source": "gaba_nmda_treatment_support",
                "target": "nmda_disruption",
                "relation": "buffers glutamatergic/NMDA dysregulation",
                "catatonia_change": "protective",
            },
            {
                "source": "gaba_nmda_treatment_support",
                "target": "homeostatic_restoration",
                "relation": "supports recovery of circuit stability",
                "catatonia_change": "protective",
            },
            {
                "source": "nmda_disruption",
                "target": "glutamate_excitotoxicity",
                "relation": "NMDA dysfunction can drive excitatory instability",
                "catatonia_change": "increased",
            },
            {
                "source": "gabaergic_failure",
                "target": "neuronal_hyperexcitability",
                "relation": "loss of inhibition promotes hyperexcitable network states",
                "catatonia_change": "increased",
            },
            {
                "source": "glutamate_excitotoxicity",
                "target": "neuronal_hyperexcitability",
                "relation": "excess excitatory pressure destabilizes network equilibrium",
                "catatonia_change": "increased",
            },
            {
                "source": "monoaminergic_modulatory_shift",
                "target": "dopaminergic_motor_gating_failure",
                "relation": "modulatory imbalance destabilizes motor and behavioral gating",
                "catatonia_change": "increased",
            },
            {
                "source": "neuronal_hyperexcitability",
                "target": "frontolimbic_motor_circuit_breakdown",
                "relation": "hyperexcitable networks fail to coordinate psychomotor control",
                "catatonia_change": "increased",
            },
            {
                "source": "neuronal_hyperexcitability",
                "target": "thalamo_basal_ganglia_relay_failure",
                "relation": "relay and gating systems become unstable under hyperexcitability",
                "catatonia_change": "increased",
            },
            {
                "source": "hypofrontality",
                "target": "ofc",
                "relation": "reduces orbital regulation of behavior and social-motor appropriateness",
                "catatonia_change": "reduced function",
            },
            {
                "source": "hypofrontality",
                "target": "dlpfc",
                "relation": "reduces executive control and volitional initiation",
                "catatonia_change": "reduced function",
            },
            {
                "source": "hypofrontality",
                "target": "acc",
                "relation": "reduces motivational and conflict-monitoring control",
                "catatonia_change": "reduced function",
            },
            {
                "source": "frontolimbic_motor_circuit_breakdown",
                "target": "amygdala",
                "relation": "destabilizes affective salience and behavioral freezing/agitation responses",
                "catatonia_change": "increased dysregulation",
            },
            {
                "source": "frontolimbic_motor_circuit_breakdown",
                "target": "hippocampus",
                "relation": "destabilizes contextual and stress-memory regulation",
                "catatonia_change": "increased dysregulation",
            },
            {
                "source": "frontolimbic_motor_circuit_breakdown",
                "target": "psychomotor_shutdown",
                "relation": "failure of coordinated circuit control promotes stuporous catatonia",
                "catatonia_change": "increased",
            },
            {
                "source": "frontolimbic_motor_circuit_breakdown",
                "target": "catatonic_agitation",
                "relation": "dysregulated circuits can also produce excited/agitated catatonia",
                "catatonia_change": "increased",
            },
            {
                "source": "thalamo_basal_ganglia_relay_failure",
                "target": "basal_ganglia_proxy",
                "relation": "burdens subcortical motor/cognitive integration",
                "catatonia_change": "increased dysregulation",
            },
            {
                "source": "thalamo_basal_ganglia_relay_failure",
                "target": "thalamus_proxy",
                "relation": "burdens sensory-motor relay and gating",
                "catatonia_change": "increased dysregulation",
            },
            {
                "source": "thalamo_basal_ganglia_relay_failure",
                "target": "mutism_volitional_block",
                "relation": "relay failure impairs goal-directed initiation and speech output",
                "catatonia_change": "increased",
            },
            {
                "source": "thalamo_basal_ganglia_relay_failure",
                "target": "posturing_rigidity",
                "relation": "motor-loop failure promotes rigidity and posturing",
                "catatonia_change": "increased",
            },
            {
                "source": "dopaminergic_motor_gating_failure",
                "target": "catatonic_agitation",
                "relation": "dopaminergic gating instability can promote behavioral dyscontrol",
                "catatonia_change": "increased",
            },
            {
                "source": "dopaminergic_motor_gating_failure",
                "target": "posturing_rigidity",
                "relation": "disrupted subcortical gating may contribute to motor abnormalities",
                "catatonia_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "affective_withdrawal",
                "relation": "limbic disruption contributes to emotional withdrawal and behavioral arrest",
                "catatonia_change": "increased",
            },
            {
                "source": "acc",
                "target": "mutism_volitional_block",
                "relation": "reduced ACC drive can impair initiation and motivated behavior",
                "catatonia_change": "increased",
            },
            {
                "source": "ofc",
                "target": "catatonic_agitation",
                "relation": "reduced orbitofrontal regulation can destabilize excited states",
                "catatonia_change": "increased",
            },
            {
                "source": "dlpfc",
                "target": "psychomotor_shutdown",
                "relation": "reduced executive initiation contributes to stupor-like states",
                "catatonia_change": "increased",
            },
            {
                "source": "neuronal_hyperexcitability",
                "target": "autonomic_instability",
                "relation": "global instability can escalate toward malignant/autonomic burden",
                "catatonia_change": "increased",
            },
            {
                "source": "homeostatic_restoration",
                "target": "catatonic_recovery",
                "relation": "restoration of balance supports reversal of psychomotor syndrome",
                "catatonia_change": "increased recovery",
            },
            {
                "source": "homeostatic_restoration",
                "target": "psychomotor_shutdown",
                "relation": "restoration reduces stupor and immobility",
                "catatonia_change": "protective",
            },
            {
                "source": "homeostatic_restoration",
                "target": "catatonic_agitation",
                "relation": "restoration reduces excited/agitated catatonia",
                "catatonia_change": "protective",
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "hippocampus",
            "amygdala",
            "orbitofrontal cortex",
            "prefrontal cortex",
            "anterior cingulate",
            "thalamus",
            "striatum",
            "basal ganglia",
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

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        """
        Helper for refining proxy searches against the atlas.
        Useful for tuning thalamic or striatal candidates.
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
            "ofc",
            "dlpfc",
            "acc",
            "hippocampus",
            "amygdala",
            "basal_ganglia_proxy",
            "thalamus_proxy",
        ),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the catatonia circuit.
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
                    "description": "Atlas-backed catatonia circuit node",
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
        neurological_insult: float,
        metabolic_toxic_burden: float,
        autoimmune_nmda_process: float,
        genetic_hyperexcitability: float,
        mood_psychosis_diathesis: float,
        gaba_nmda_treatment_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while gaba_nmda_treatment_support is protective.
        """
        n = self._clip01(neurological_insult)
        m = self._clip01(metabolic_toxic_burden)
        a = self._clip01(autoimmune_nmda_process)
        g = self._clip01(genetic_hyperexcitability)
        p = self._clip01(mood_psychosis_diathesis)
        t = self._clip01(gaba_nmda_treatment_support)

        # Latent biology
        gabaergic_failure = self._clip01(
            0.30 * m + 0.20 * g + 0.10 * n - 0.25 * t
        )
        nmda_disruption = self._clip01(
            0.50 * a + 0.15 * n - 0.20 * t
        )
        glutamate_excitotoxicity = self._clip01(
            0.30 * n + 0.25 * nmda_disruption + 0.15 * m - 0.10 * t
        )
        neuronal_hyperexcitability = self._clip01(
            0.30 * glutamate_excitotoxicity
            + 0.25 * gabaergic_failure
            + 0.20 * g
            + 0.10 * m
            - 0.15 * t
        )
        monoaminergic_modulatory_shift = self._clip01(
            0.25 * m + 0.20 * p + 0.10 * n
        )
        dopaminergic_motor_gating_failure = self._clip01(
            0.35 * monoaminergic_modulatory_shift
            + 0.20 * neuronal_hyperexcitability
            + 0.10 * p
            - 0.10 * t
        )
        hypofrontality = self._clip01(
            0.25 * n + 0.20 * p + 0.15 * neuronal_hyperexcitability - 0.20 * t
        )
        thalamo_basal_ganglia_relay_failure = self._clip01(
            0.25 * n + 0.25 * dopaminergic_motor_gating_failure + 0.15 * neuronal_hyperexcitability - 0.10 * t
        )
        frontolimbic_motor_circuit_breakdown = self._clip01(
            0.25 * n
            + 0.20 * neuronal_hyperexcitability
            + 0.20 * hypofrontality
            + 0.10 * p
            - 0.20 * t
        )
        homeostatic_restoration = self._clip01(
            0.55 * t - 0.20 * neuronal_hyperexcitability - 0.10 * nmda_disruption
        )

        # Regional state proxies
        ofc = self._clip01(
            0.40 * hypofrontality + 0.10 * frontolimbic_motor_circuit_breakdown - 0.10 * t
        )
        dlpfc = self._clip01(
            0.40 * hypofrontality + 0.10 * frontolimbic_motor_circuit_breakdown - 0.10 * t
        )
        acc = self._clip01(
            0.30 * hypofrontality + 0.25 * glutamate_excitotoxicity + 0.10 * frontolimbic_motor_circuit_breakdown - 0.10 * t
        )
        hippocampus = self._clip01(
            0.35 * frontolimbic_motor_circuit_breakdown + 0.20 * glutamate_excitotoxicity
        )
        amygdala = self._clip01(
            0.35 * frontolimbic_motor_circuit_breakdown + 0.10 * monoaminergic_modulatory_shift
        )
        basal_ganglia_proxy = self._clip01(
            0.40 * thalamo_basal_ganglia_relay_failure + 0.15 * dopaminergic_motor_gating_failure
        )
        thalamus_proxy = self._clip01(
            0.40 * thalamo_basal_ganglia_relay_failure + 0.15 * neuronal_hyperexcitability
        )

        # Symptoms
        psychomotor_shutdown = self._clip01(
            0.30 * frontolimbic_motor_circuit_breakdown
            + 0.25 * dlpfc
            + 0.15 * thalamus_proxy
            - 0.20 * homeostatic_restoration
        )
        mutism_volitional_block = self._clip01(
            0.30 * thalamo_basal_ganglia_relay_failure
            + 0.20 * acc
            + 0.15 * psychomotor_shutdown
            - 0.15 * homeostatic_restoration
        )
        posturing_rigidity = self._clip01(
            0.35 * basal_ganglia_proxy
            + 0.20 * thalamus_proxy
            + 0.10 * dopaminergic_motor_gating_failure
            - 0.15 * homeostatic_restoration
        )
        catatonic_agitation = self._clip01(
            0.25 * dopaminergic_motor_gating_failure
            + 0.25 * ofc
            + 0.15 * amygdala
            + 0.10 * neuronal_hyperexcitability
            - 0.15 * homeostatic_restoration
        )
        affective_withdrawal = self._clip01(
            0.30 * amygdala
            + 0.20 * psychomotor_shutdown
            + 0.10 * hippocampus
            - 0.10 * homeostatic_restoration
        )
        autonomic_instability = self._clip01(
            0.35 * neuronal_hyperexcitability
            + 0.20 * catatonic_agitation
            + 0.10 * m
            - 0.15 * homeostatic_restoration
        )
        catatonic_recovery = self._clip01(
            0.55 * homeostatic_restoration + 0.20 * t
        )

        return {
            "inputs": pd.Series(
                {
                    "neurological_insult": n,
                    "metabolic_toxic_burden": m,
                    "autoimmune_nmda_process": a,
                    "genetic_hyperexcitability": g,
                    "mood_psychosis_diathesis": p,
                    "gaba_nmda_treatment_support": t,
                }
            ),
            "latents": pd.Series(
                {
                    "neuronal_hyperexcitability": neuronal_hyperexcitability,
                    "frontolimbic_motor_circuit_breakdown": frontolimbic_motor_circuit_breakdown,
                    "thalamo_basal_ganglia_relay_failure": thalamo_basal_ganglia_relay_failure,
                    "glutamate_excitotoxicity": glutamate_excitotoxicity,
                    "gabaergic_failure": gabaergic_failure,
                    "dopaminergic_motor_gating_failure": dopaminergic_motor_gating_failure,
                    "hypofrontality": hypofrontality,
                    "nmda_disruption": nmda_disruption,
                    "monoaminergic_modulatory_shift": monoaminergic_modulatory_shift,
                    "homeostatic_restoration": homeostatic_restoration,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "ofc": ofc,
                    "dlpfc": dlpfc,
                    "acc": acc,
                    "hippocampus": hippocampus,
                    "amygdala": amygdala,
                    "basal_ganglia_proxy": basal_ganglia_proxy,
                    "thalamus_proxy": thalamus_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "psychomotor_shutdown": psychomotor_shutdown,
                    "mutism_volitional_block": mutism_volitional_block,
                    "posturing_rigidity": posturing_rigidity,
                    "catatonic_agitation": catatonic_agitation,
                    "affective_withdrawal": affective_withdrawal,
                    "autonomic_instability": autonomic_instability,
                    "catatonic_recovery": catatonic_recovery,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "stuporous_catatonia_profile": self._clip01(
                        0.45 * psychomotor_shutdown
                        + 0.25 * mutism_volitional_block
                        + 0.15 * affective_withdrawal
                    ),
                    "agitated_catatonia_profile": self._clip01(
                        0.45 * catatonic_agitation
                        + 0.25 * autonomic_instability
                        + 0.15 * dopaminergic_motor_gating_failure
                    ),
                    "nmda_autoimmune_profile": self._clip01(
                        0.45 * nmda_disruption
                        + 0.25 * glutamate_excitotoxicity
                        + 0.20 * psychomotor_shutdown
                    ),
                    "malignant_risk_profile": self._clip01(
                        0.40 * autonomic_instability
                        + 0.25 * neuronal_hyperexcitability
                        + 0.20 * catatonic_agitation
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-20, -10, -12)).head(10)
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
    model = CatatonicDisorderDueToMedicalConditionModel()

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
        bundle["edges"][["source", "target", "relation", "catatonia_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["ofc", "dlpfc", "acc", "hippocampus", "amygdala", "basal_ganglia_proxy", "thalamus_proxy"]:
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
        neurological_insult=0.65,
        metabolic_toxic_burden=0.55,
        autoimmune_nmda_process=0.40,
        genetic_hyperexcitability=0.50,
        mood_psychosis_diathesis=0.45,
        gaba_nmda_treatment_support=0.20,
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
    # print(model.assign_mni_point((-20, -10, -12)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("thalamus").to_string(index=False))
    # print(model.suggest_regions("caudate").to_string(index=False))
