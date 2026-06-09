from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Conduct Disorder-oriented panel:
# - serotonin / aggression / impulse control
# - dopamine / reward sensitivity / punishment insensitivity
# - stress / HPA-axis and adversity embedding
# - inhibitory/excitatory balance
# - plasticity / inflammation-adjacent stress biology
DEFAULT_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor
    "HTR2A",    # serotonin receptor
    "TPH2",     # serotonin synthesis
    "DRD2",     # dopamine receptor
    "SLC6A3",   # dopamine transporter
    "COMT",     # catecholamine metabolism
    "DBH",      # norepinephrine synthesis step
    "GABRA2",   # GABA-A receptor
    "GABRB2",   # GABA-A receptor
    "GAD1",     # GABA synthesis
    "GRIN2B",   # NMDA receptor subunit
    "SLC1A1",   # glutamate transport
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
    "CRHR1",    # CRF signaling
    "BDNF",     # plasticity
]


class ConductDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Conduct Disorder.

    What it does:
      1) Resolves chapter-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates aggression, cruelty/property destruction, rule violation,
         callous-unemotional burden, and executive-control failure.

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

        # Chapter-consistent emotional-control / empathy / reward-control network.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "insula": [
                "Area Id2 (Insula) left",
                "Area Id3 (Insula) left",
                "Area Id4 (Insula) left",
                "insula",
            ],
            "ofc": [
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "orbitofrontal",
                "Fo4",
                "Fo3",
            ],
            "acc": [
                "Area 33 (ACC) left",
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "ACC",
                "anterior cingulate",
            ],
            "pfc_control": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
                "prefrontal",
                "frontal",
            ],
            "striatum_proxy": [
                "nucleus accumbens",
                "accumbens",
                "caudate",
                "putamen",
                "striatum",
                "basal ganglia",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": "Heritable liability for impulsive, aggressive, and callous traits",
            "early_adversity_trauma": "Childhood abuse, neglect, family dysfunction, and adversity burden",
            "callous_unemotional_loading": "Severity of callous-unemotional trait expression",
            "substance_disinhibition_load": "Alcohol/medication-related disinhibition worsening aggression and judgment",
            "reward_seeking_pressure": "Strong reward-seeking and immediate-gratification pressure",
            "prosocial_support": "Protective caregiving, regulation, and prosocial buffering",
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_disinhibition": "Reduced serotonergic inhibitory control over aggression and impulse",
            "dopaminergic_reward_bias": "Reward sensitivity with diminished punishment sensitivity",
            "hpa_axis_trauma_sensitization": "Stress-system sensitization from early adversity",
            "epigenetic_embedding": "Lasting adversity-linked molecular embedding of behavior risk",
            "fearlessness_threat_underreactivity": "Reduced fear/distress responsiveness",
            "empathy_distress_cue_deficit": "Impaired recognition of distress cues and reduced compassion",
            "executive_control_failure": "Weak top-down behavioral and emotional control",
            "frontolimbic_disconnection": "Weak connectivity between regulatory and emotional centers",
            "punishment_insensitivity": "Poor learning from negative consequences",
            "reward_antisocial_drive": "Immediate-reward drive dominating prosocial restraint",
        }

        self.symptom_nodes: Dict[str, str] = {
            "impulsive_aggression": "Impulsive physical or verbal aggression",
            "cruelty_property_destruction": "Cruelty, destruction of property, and severe externalizing acts",
            "rule_violation_deceit": "Persistent rule-breaking, deceit, and norm violation",
            "low_guilt_empathy": "Low guilt, low empathy, and weak concern for others",
            "fearlessness": "Low fearfulness and reduced sensitivity to threat/distress",
            "persistent_antisocial_pattern": "Stable, pervasive conduct-problem pattern",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_disinhibition",
                "relation": "raises vulnerability for poor serotonergic impulse control",
                "conduct_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "dopaminergic_reward_bias",
                "relation": "raises vulnerability for reward-seeking and immediate gratification",
                "conduct_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "hpa_axis_trauma_sensitization",
                "relation": "raises vulnerability for stress-related dysregulation",
                "conduct_change": "increased susceptibility",
            },
            {
                "source": "early_adversity_trauma",
                "target": "hpa_axis_trauma_sensitization",
                "relation": "early adversity sensitizes stress-response systems",
                "conduct_change": "increased",
            },
            {
                "source": "early_adversity_trauma",
                "target": "epigenetic_embedding",
                "relation": "adverse experience can become biologically embedded",
                "conduct_change": "increased",
            },
            {
                "source": "early_adversity_trauma",
                "target": "frontolimbic_disconnection",
                "relation": "trauma can distort maturation of emotional-control circuits",
                "conduct_change": "increased",
            },
            {
                "source": "callous_unemotional_loading",
                "target": "fearlessness_threat_underreactivity",
                "relation": "supports low fear and reduced distress-cue responsiveness",
                "conduct_change": "increased",
            },
            {
                "source": "callous_unemotional_loading",
                "target": "empathy_distress_cue_deficit",
                "relation": "supports reduced guilt, empathy, and concern for others",
                "conduct_change": "increased",
            },
            {
                "source": "substance_disinhibition_load",
                "target": "serotonergic_disinhibition",
                "relation": "substances can worsen disinhibition and aggression",
                "conduct_change": "increased",
            },
            {
                "source": "substance_disinhibition_load",
                "target": "executive_control_failure",
                "relation": "substances impair judgment and self-control",
                "conduct_change": "increased",
            },
            {
                "source": "reward_seeking_pressure",
                "target": "dopaminergic_reward_bias",
                "relation": "loads immediate reward and salience systems",
                "conduct_change": "increased",
            },
            {
                "source": "reward_seeking_pressure",
                "target": "reward_antisocial_drive",
                "relation": "biases action selection toward immediate payoff",
                "conduct_change": "increased",
            },
            {
                "source": "prosocial_support",
                "target": "executive_control_failure",
                "relation": "buffers top-down control failure",
                "conduct_change": "protective",
            },
            {
                "source": "prosocial_support",
                "target": "persistent_antisocial_pattern",
                "relation": "buffers persistence and severity of conduct symptoms",
                "conduct_change": "protective",
            },
            {
                "source": "hpa_axis_trauma_sensitization",
                "target": "serotonergic_disinhibition",
                "relation": "stress-linked dysregulation lowers aggression threshold",
                "conduct_change": "increased",
            },
            {
                "source": "hpa_axis_trauma_sensitization",
                "target": "frontolimbic_disconnection",
                "relation": "stress and trauma weaken regulatory-emotional coordination",
                "conduct_change": "increased",
            },
            {
                "source": "epigenetic_embedding",
                "target": "executive_control_failure",
                "relation": "long-lasting biological changes stabilize maladaptive behavioral tendencies",
                "conduct_change": "increased",
            },
            {
                "source": "epigenetic_embedding",
                "target": "reward_antisocial_drive",
                "relation": "stable molecular traces may bias future responses toward antisocial patterns",
                "conduct_change": "increased",
            },
            {
                "source": "serotonergic_disinhibition",
                "target": "impulsive_aggression",
                "relation": "reduced serotonin lowers threshold for aggressive responses",
                "conduct_change": "increased",
            },
            {
                "source": "dopaminergic_reward_bias",
                "target": "punishment_insensitivity",
                "relation": "high reward sensitivity can diminish learning from punishment",
                "conduct_change": "increased",
            },
            {
                "source": "dopaminergic_reward_bias",
                "target": "reward_antisocial_drive",
                "relation": "reward bias promotes immediate-gain antisocial behavior",
                "conduct_change": "increased",
            },
            {
                "source": "fearlessness_threat_underreactivity",
                "target": "amygdala",
                "relation": "maps onto amygdala-linked low fear/distress processing",
                "conduct_change": "increased dysregulation",
            },
            {
                "source": "fearlessness_threat_underreactivity",
                "target": "fearlessness",
                "relation": "supports low fearfulness and weak distress responsiveness",
                "conduct_change": "increased",
            },
            {
                "source": "empathy_distress_cue_deficit",
                "target": "insula",
                "relation": "loads interoceptive-empathic processing systems",
                "conduct_change": "increased dysregulation",
            },
            {
                "source": "empathy_distress_cue_deficit",
                "target": "amygdala",
                "relation": "weak distress-cue processing burdens limbic empathy systems",
                "conduct_change": "increased dysregulation",
            },
            {
                "source": "empathy_distress_cue_deficit",
                "target": "low_guilt_empathy",
                "relation": "reduced distress-cue processing weakens guilt and empathy",
                "conduct_change": "increased",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "ofc",
                "relation": "weakens orbitofrontal regulation of social behavior and restraint",
                "conduct_change": "increased dysregulation",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "acc",
                "relation": "weakens conflict/error monitoring and regulatory control",
                "conduct_change": "increased dysregulation",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "pfc_control",
                "relation": "weakens prefrontal executive control over behavior",
                "conduct_change": "increased dysregulation",
            },
            {
                "source": "frontolimbic_disconnection",
                "target": "executive_control_failure",
                "relation": "reduced regulatory-emotional connectivity weakens control",
                "conduct_change": "increased",
            },
            {
                "source": "executive_control_failure",
                "target": "pfc_control",
                "relation": "reflects impaired prefrontal top-down control",
                "conduct_change": "reduced function",
            },
            {
                "source": "executive_control_failure",
                "target": "impulsive_aggression",
                "relation": "poor control promotes impulsive aggression",
                "conduct_change": "increased",
            },
            {
                "source": "executive_control_failure",
                "target": "rule_violation_deceit",
                "relation": "poor control promotes rule-breaking and deceptive behavior",
                "conduct_change": "increased",
            },
            {
                "source": "punishment_insensitivity",
                "target": "striatum_proxy",
                "relation": "loads reward-punishment learning circuitry",
                "conduct_change": "increased dysregulation",
            },
            {
                "source": "punishment_insensitivity",
                "target": "rule_violation_deceit",
                "relation": "weak learning from consequences promotes repeated violations",
                "conduct_change": "increased",
            },
            {
                "source": "reward_antisocial_drive",
                "target": "striatum_proxy",
                "relation": "loads reward and approach circuitry",
                "conduct_change": "increased dysregulation",
            },
            {
                "source": "reward_antisocial_drive",
                "target": "cruelty_property_destruction",
                "relation": "reward-seeking can dominate prosocial restraint",
                "conduct_change": "increased",
            },
            {
                "source": "ofc",
                "target": "rule_violation_deceit",
                "relation": "orbitofrontal dysfunction weakens socially appropriate decision-making",
                "conduct_change": "increased",
            },
            {
                "source": "acc",
                "target": "impulsive_aggression",
                "relation": "weakened conflict monitoring increases dysregulated aggression",
                "conduct_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "fearlessness",
                "relation": "amygdala dysfunction supports fearlessness and poor distress recognition",
                "conduct_change": "increased",
            },
            {
                "source": "insula",
                "target": "low_guilt_empathy",
                "relation": "insula dysfunction weakens empathic/interoceptive response to others",
                "conduct_change": "increased",
            },
            {
                "source": "impulsive_aggression",
                "target": "persistent_antisocial_pattern",
                "relation": "aggressive behavior contributes to persistent conduct problems",
                "conduct_change": "increased",
            },
            {
                "source": "cruelty_property_destruction",
                "target": "persistent_antisocial_pattern",
                "relation": "severe destructive behavior strengthens antisocial patterning",
                "conduct_change": "increased",
            },
            {
                "source": "rule_violation_deceit",
                "target": "persistent_antisocial_pattern",
                "relation": "persistent rule-breaking and deceit stabilize disorder expression",
                "conduct_change": "increased",
            },
            {
                "source": "low_guilt_empathy",
                "target": "persistent_antisocial_pattern",
                "relation": "low guilt and empathy reduce internal brakes on antisocial behavior",
                "conduct_change": "increased",
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
            "orbitofrontal cortex",
            "anterior cingulate",
            "prefrontal cortex",
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
        """
        Helper for refining proxy searches against the atlas.
        Useful for tuning insular or striatal candidates.
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
        node_keys: Sequence[str] = ("amygdala", "insula", "ofc", "acc", "pfc_control", "striatum_proxy"),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the conduct-disorder circuit.
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
                    "description": "Atlas-backed conduct-disorder circuit node",
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
        early_adversity_trauma: float,
        callous_unemotional_loading: float,
        substance_disinhibition_load: float,
        reward_seeking_pressure: float,
        prosocial_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while prosocial_support is protective.
        """
        g = self._clip01(genetic_vulnerability)
        e = self._clip01(early_adversity_trauma)
        c = self._clip01(callous_unemotional_loading)
        s = self._clip01(substance_disinhibition_load)
        r = self._clip01(reward_seeking_pressure)
        p = self._clip01(prosocial_support)

        # Latent biology
        serotonergic_disinhibition = self._clip01(
            0.30 * g + 0.20 * e + 0.20 * s - 0.20 * p
        )
        dopaminergic_reward_bias = self._clip01(
            0.30 * g + 0.25 * r + 0.10 * s - 0.10 * p
        )
        hpa_axis_trauma_sensitization = self._clip01(
            0.40 * e + 0.15 * g - 0.15 * p
        )
        epigenetic_embedding = self._clip01(
            0.40 * e + 0.20 * hpa_axis_trauma_sensitization
        )
        fearlessness_threat_underreactivity = self._clip01(
            0.40 * c + 0.20 * dopaminergic_reward_bias
        )
        empathy_distress_cue_deficit = self._clip01(
            0.35 * c + 0.25 * fearlessness_threat_underreactivity + 0.10 * e
        )
        frontolimbic_disconnection = self._clip01(
            0.30 * e + 0.20 * epigenetic_embedding + 0.15 * hpa_axis_trauma_sensitization - 0.15 * p
        )
        executive_control_failure = self._clip01(
            0.30 * serotonergic_disinhibition
            + 0.25 * frontolimbic_disconnection
            + 0.15 * s
            - 0.25 * p
        )
        punishment_insensitivity = self._clip01(
            0.35 * dopaminergic_reward_bias + 0.25 * fearlessness_threat_underreactivity
        )
        reward_antisocial_drive = self._clip01(
            0.35 * dopaminergic_reward_bias
            + 0.25 * reward_seeking_pressure
            + 0.15 * punishment_insensitivity
            - 0.10 * p
        )

        # Regional state proxies
        amygdala = self._clip01(
            0.45 * fearlessness_threat_underreactivity
            + 0.20 * empathy_distress_cue_deficit
            + 0.10 * e
        )
        insula = self._clip01(
            0.45 * empathy_distress_cue_deficit
            + 0.10 * frontolimbic_disconnection
        )
        ofc = self._clip01(
            0.35 * executive_control_failure
            + 0.20 * reward_antisocial_drive
            - 0.10 * p
        )
        acc = self._clip01(
            0.35 * frontolimbic_disconnection
            + 0.20 * serotonergic_disinhibition
            - 0.10 * p
        )
        pfc_control = self._clip01(
            0.45 * executive_control_failure
            + 0.15 * hpa_axis_trauma_sensitization
            - 0.15 * p
        )
        striatum_proxy = self._clip01(
            0.45 * reward_antisocial_drive
            + 0.25 * punishment_insensitivity
        )

        # Symptoms
        impulsive_aggression = self._clip01(
            0.35 * serotonergic_disinhibition
            + 0.25 * executive_control_failure
            + 0.15 * substance_disinhibition_load
            + 0.10 * reward_antisocial_drive
            - 0.10 * p
        )
        low_guilt_empathy = self._clip01(
            0.40 * empathy_distress_cue_deficit
            + 0.25 * fearlessness_threat_underreactivity
            + 0.10 * amygdala
        )
        fearlessness = self._clip01(
            0.45 * fearlessness_threat_underreactivity + 0.20 * amygdala
        )
        cruelty_property_destruction = self._clip01(
            0.30 * low_guilt_empathy
            + 0.25 * impulsive_aggression
            + 0.20 * reward_antisocial_drive
            - 0.10 * p
        )
        rule_violation_deceit = self._clip01(
            0.30 * punishment_insensitivity
            + 0.25 * executive_control_failure
            + 0.20 * reward_antisocial_drive
            - 0.10 * p
        )
        persistent_antisocial_pattern = self._clip01(
            0.25 * impulsive_aggression
            + 0.25 * rule_violation_deceit
            + 0.20 * cruelty_property_destruction
            + 0.15 * low_guilt_empathy
            - 0.15 * p
        )

        return {
            "inputs": pd.Series(
                {
                    "genetic_vulnerability": g,
                    "early_adversity_trauma": e,
                    "callous_unemotional_loading": c,
                    "substance_disinhibition_load": s,
                    "reward_seeking_pressure": r,
                    "prosocial_support": p,
                }
            ),
            "latents": pd.Series(
                {
                    "reward_antisocial_drive": reward_antisocial_drive,
                    "executive_control_failure": executive_control_failure,
                    "empathy_distress_cue_deficit": empathy_distress_cue_deficit,
                    "fearlessness_threat_underreactivity": fearlessness_threat_underreactivity,
                    "dopaminergic_reward_bias": dopaminergic_reward_bias,
                    "punishment_insensitivity": punishment_insensitivity,
                    "serotonergic_disinhibition": serotonergic_disinhibition,
                    "frontolimbic_disconnection": frontolimbic_disconnection,
                    "hpa_axis_trauma_sensitization": hpa_axis_trauma_sensitization,
                    "epigenetic_embedding": epigenetic_embedding,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "amygdala": amygdala,
                    "insula": insula,
                    "ofc": ofc,
                    "acc": acc,
                    "pfc_control": pfc_control,
                    "striatum_proxy": striatum_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "persistent_antisocial_pattern": persistent_antisocial_pattern,
                    "rule_violation_deceit": rule_violation_deceit,
                    "cruelty_property_destruction": cruelty_property_destruction,
                    "impulsive_aggression": impulsive_aggression,
                    "low_guilt_empathy": low_guilt_empathy,
                    "fearlessness": fearlessness,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "callous_unemotional_profile": self._clip01(
                        0.45 * low_guilt_empathy
                        + 0.25 * fearlessness
                        + 0.20 * empathy_distress_cue_deficit
                    ),
                    "impulsive_aggressive_profile": self._clip01(
                        0.45 * impulsive_aggression
                        + 0.25 * serotonergic_disinhibition
                        + 0.20 * executive_control_failure
                    ),
                    "reward_driven_rule_breaking_profile": self._clip01(
                        0.45 * rule_violation_deceit
                        + 0.25 * reward_antisocial_drive
                        + 0.20 * punishment_insensitivity
                    ),
                    "trauma_embedded_conduct_profile": self._clip01(
                        0.40 * hpa_axis_trauma_sensitization
                        + 0.30 * epigenetic_embedding
                        + 0.20 * persistent_antisocial_pattern
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-20, -6, -16)).head(10)
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
    model = ConductDisorderModel()

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
            ["source", "target", "relation", "conduct_change"]
        ].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["amygdala", "insula", "ofc", "acc", "pfc_control", "striatum_proxy"]:
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
        genetic_vulnerability=0.70,
        early_adversity_trauma=0.80,
        callous_unemotional_loading=0.75,
        substance_disinhibition_load=0.45,
        reward_seeking_pressure=0.70,
        prosocial_support=0.20,
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
    # print(model.assign_mni_point((-20, -6, -16)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("insula").to_string(index=False))
    # print(model.suggest_regions("caudate").to_string(index=False))
