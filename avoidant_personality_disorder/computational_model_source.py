from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Avoidant Personality Disorder-oriented panel:
# - serotonin / anxiety / rejection sensitivity
# - dopamine / social reward / anhedonia
# - stress and epigenetic susceptibility
DEFAULT_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor
    "HTR2A",    # serotonin receptor
    "TPH2",     # serotonin synthesis
    "MAOA",     # monoamine metabolism
    "DRD2",     # dopamine receptor
    "SLC6A3",   # dopamine transporter
    "COMT",     # catecholamine metabolism
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
    "CRHR1",    # CRH signalling
    "BDNF",     # plasticity
    "OXTR",     # social-affiliative signalling
    "IL6",      # inflammation / stress biology
    "TNF",
]


class AvoidantPersonalityDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Avoidant Personality Disorder (AvPD).

    What it does:
      1) Resolves chapter-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates rejection sensitivity, social anhedonia, rumination,
         and avoidance dynamics.

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

        # The chapter explicitly emphasizes a hyper-reactive amygdala and
        # hypo-functional medial/orbital PFC. ACC is added as a pragmatic
        # cortical-midline proxy for self-referential monitoring/rumination.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "mpfc": [
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "pACC",
                "sACC",
                "medial prefrontal",
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
                "ACC",
                "anterior cingulate",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": "Polygenic liability affecting anxiety, temperament, and social reward processing",
            "parental_rejection": "Early rejection / criticism / low social safety in attachment relationships",
            "peer_humiliation": "Bullying, ridicule, humiliation, or repeated social defeat",
            "supportive_environment": "Protective relational safety and validating support",
            "ssri_modulation": "Protective serotonergic treatment effect",
        }

        self.latent_nodes: Dict[str, str] = {
            "serotonergic_dysregulation": "Serotonergic instability in anxiety, mood, and rejection sensitivity",
            "dopaminergic_social_reward_deficit": "Reduced reward value of social interaction",
            "epigenetic_sensitization": "Gene-environment amplification of anxiety and avoidance vulnerability",
            "amygdala_threat_bias": "Hyper-reactive social threat detection",
            "pfc_regulatory_failure": "Weak top-down control over fear, rumination, and withdrawal impulses",
            "rejection_sensitivity": "High social-pain and criticism sensitivity",
            "social_anhedonia": "Low pleasure or motivational pull from social contact",
            "negative_self_rumination": "Persistent self-critical replay of social failure and inadequacy",
        }

        self.symptom_nodes: Dict[str, str] = {
            "social_inhibition": "Persistent restraint and inhibition in interpersonal situations",
            "avoidance": "Avoidance of social contact and scrutiny",
            "feelings_inadequacy": "Stable self-view of inferiority / inadequacy",
            "negative_evaluation_fear": "Fear of criticism, rejection, or embarrassment",
            "anxious_distress": "Chronic tension, anxiety, and anticipatory worry",
            "depressive_distress": "Sadness, stress, unhappiness, and social demoralization",
            "anger_irritability": "Intense anger or irritability when criticized or misunderstood",
            "solitude_preference": "Default retreat to solitary, lower-risk contexts",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "serotonergic_dysregulation",
                "relation": "raises vulnerability for anxious and socially painful affective reactivity",
                "avpd_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "dopaminergic_social_reward_deficit",
                "relation": "raises vulnerability for low social reward and apathy",
                "avpd_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "epigenetic_sensitization",
                "relation": "creates vulnerability for strong environmental calibration of trait avoidance",
                "avpd_change": "increased susceptibility",
            },
            {
                "source": "parental_rejection",
                "target": "epigenetic_sensitization",
                "relation": "amplifies long-term vulnerability through developmental stress embedding",
                "avpd_change": "increased",
            },
            {
                "source": "parental_rejection",
                "target": "rejection_sensitivity",
                "relation": "conditions anticipation of criticism and disapproval",
                "avpd_change": "increased",
            },
            {
                "source": "peer_humiliation",
                "target": "rejection_sensitivity",
                "relation": "intensifies social pain and fear of scrutiny",
                "avpd_change": "increased",
            },
            {
                "source": "peer_humiliation",
                "target": "negative_self_rumination",
                "relation": "creates self-critical replay of social failure",
                "avpd_change": "increased",
            },
            {
                "source": "supportive_environment",
                "target": "rejection_sensitivity",
                "relation": "buffers expectation of criticism and rejection",
                "avpd_change": "protective",
            },
            {
                "source": "supportive_environment",
                "target": "avoidance",
                "relation": "supports approach and corrective social experience",
                "avpd_change": "protective",
            },
            {
                "source": "ssri_modulation",
                "target": "serotonergic_dysregulation",
                "relation": "reduces serotonergic instability",
                "avpd_change": "protective",
            },
            {
                "source": "ssri_modulation",
                "target": "anxious_distress",
                "relation": "reduces social-anxiety-like symptom burden",
                "avpd_change": "protective",
            },
            {
                "source": "epigenetic_sensitization",
                "target": "serotonergic_dysregulation",
                "relation": "stabilizes high-anxiety trait expression",
                "avpd_change": "increased",
            },
            {
                "source": "epigenetic_sensitization",
                "target": "amygdala_threat_bias",
                "relation": "biases emotional circuitry toward social threat",
                "avpd_change": "increased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "amygdala_threat_bias",
                "relation": "weakens modulation of fear and rejection signals",
                "avpd_change": "increased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "anxious_distress",
                "relation": "supports pervasive anxiety, tension, and worry",
                "avpd_change": "increased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "depressive_distress",
                "relation": "supports sadness, stress, and low mood",
                "avpd_change": "increased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "rejection_sensitivity",
                "relation": "heightens social pain and sensitivity to criticism",
                "avpd_change": "increased",
            },
            {
                "source": "dopaminergic_social_reward_deficit",
                "target": "social_anhedonia",
                "relation": "reduces pleasure and motivational pull from social encounters",
                "avpd_change": "increased",
            },
            {
                "source": "dopaminergic_social_reward_deficit",
                "target": "solitude_preference",
                "relation": "fails to reinforce social approach behavior",
                "avpd_change": "increased",
            },
            {
                "source": "social_anhedonia",
                "target": "avoidance",
                "relation": "makes social approach less rewarding and less likely",
                "avpd_change": "increased",
            },
            {
                "source": "rejection_sensitivity",
                "target": "negative_evaluation_fear",
                "relation": "drives fear of criticism, rejection, and embarrassment",
                "avpd_change": "increased",
            },
            {
                "source": "rejection_sensitivity",
                "target": "amygdala",
                "relation": "loads social threat detection and alarm responses",
                "avpd_change": "increased dysregulation",
            },
            {
                "source": "amygdala_threat_bias",
                "target": "amygdala",
                "relation": "supports hyper-reactive social threat processing",
                "avpd_change": "increased dysregulation",
            },
            {
                "source": "pfc_regulatory_failure",
                "target": "mpfc",
                "relation": "reflects weak top-down regulation of emotional responses",
                "avpd_change": "reduced function",
            },
            {
                "source": "pfc_regulatory_failure",
                "target": "ofc",
                "relation": "reflects weak orbitofrontal evaluation/regulation of social threat",
                "avpd_change": "reduced function",
            },
            {
                "source": "pfc_regulatory_failure",
                "target": "acc",
                "relation": "reflects weak disengagement from self-referential negative loops",
                "avpd_change": "reduced function",
            },
            {
                "source": "mpfc",
                "target": "amygdala",
                "relation": "normally exerts top-down inhibition",
                "avpd_change": "reduced inhibition",
            },
            {
                "source": "ofc",
                "target": "negative_evaluation_fear",
                "relation": "reduced regulatory valuation sustains exaggerated social threat appraisal",
                "avpd_change": "increased",
            },
            {
                "source": "acc",
                "target": "negative_self_rumination",
                "relation": "supports repetitive self-referential error/failure monitoring",
                "avpd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "anxious_distress",
                "relation": "amplifies social fear and distress",
                "avpd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "social_inhibition",
                "relation": "promotes freezing, restraint, and withdrawal in social situations",
                "avpd_change": "increased",
            },
            {
                "source": "negative_self_rumination",
                "target": "feelings_inadequacy",
                "relation": "stabilizes inferiority and inadequacy beliefs",
                "avpd_change": "increased",
            },
            {
                "source": "negative_self_rumination",
                "target": "depressive_distress",
                "relation": "sustains demoralization and sadness",
                "avpd_change": "increased",
            },
            {
                "source": "negative_self_rumination",
                "target": "anger_irritability",
                "relation": "can amplify frustrated, misunderstood, or criticized states",
                "avpd_change": "increased",
            },
            {
                "source": "negative_evaluation_fear",
                "target": "social_inhibition",
                "relation": "promotes guarded behavior and inhibition",
                "avpd_change": "increased",
            },
            {
                "source": "negative_evaluation_fear",
                "target": "avoidance",
                "relation": "promotes escape from scrutiny and embarrassment risk",
                "avpd_change": "increased",
            },
            {
                "source": "feelings_inadequacy",
                "target": "avoidance",
                "relation": "supports withdrawal from social comparison and evaluation",
                "avpd_change": "increased",
            },
            {
                "source": "solitude_preference",
                "target": "avoidance",
                "relation": "reinforces low-risk solitary behavioral defaults",
                "avpd_change": "increased",
            },
            {
                "source": "anxious_distress",
                "target": "avoidance",
                "relation": "sustains avoidance as an anxiolytic short-term strategy",
                "avpd_change": "increased",
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
        """
        Prefer left-sided, more specific names over generic/right-sided parents.
        Lower tuple is better.
        """
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "anterior cingulate",
            "prefrontal cortex",
            "orbitofrontal cortex",
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
        Useful for tuning medial/orbital PFC candidates.
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
            # Representative subject-level matrix
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
        node_keys: Sequence[str] = ("amygdala", "mpfc", "ofc", "acc"),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the AvPD circuit.
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
                    "description": "Atlas-backed AvPD circuit node",
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
        parental_rejection: float,
        peer_humiliation: float,
        supportive_environment: float,
        ssri_modulation: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while supportive_environment and ssri_modulation are protective.
        """
        g = self._clip01(genetic_vulnerability)
        r = self._clip01(parental_rejection)
        h = self._clip01(peer_humiliation)
        s = self._clip01(supportive_environment)
        m = self._clip01(ssri_modulation)

        # Latent biology
        epigenetic_sensitization = self._clip01(
            0.30 * g + 0.30 * r + 0.25 * h - 0.15 * s
        )
        serotonergic_dysregulation = self._clip01(
            0.35 * g + 0.25 * epigenetic_sensitization + 0.10 * h - 0.25 * m
        )
        dopaminergic_social_reward_deficit = self._clip01(
            0.35 * g + 0.15 * r + 0.10 * h - 0.10 * s
        )
        rejection_sensitivity = self._clip01(
            0.30 * serotonergic_dysregulation
            + 0.25 * r
            + 0.20 * h
            + 0.10 * epigenetic_sensitization
            - 0.15 * s
        )
        social_anhedonia = self._clip01(
            0.40 * dopaminergic_social_reward_deficit
            + 0.15 * rejection_sensitivity
            - 0.10 * s
        )
        amygdala_threat_bias = self._clip01(
            0.35 * serotonergic_dysregulation
            + 0.25 * rejection_sensitivity
            + 0.10 * h
            - 0.10 * m
        )
        pfc_regulatory_failure = self._clip01(
            0.30 * epigenetic_sensitization
            + 0.20 * amygdala_threat_bias
            + 0.15 * rejection_sensitivity
            - 0.20 * s
            - 0.10 * m
        )
        negative_self_rumination = self._clip01(
            0.30 * rejection_sensitivity
            + 0.30 * pfc_regulatory_failure
            + 0.15 * h
            - 0.15 * s
        )

        # Regional state proxies
        amygdala = self._clip01(0.45 * amygdala_threat_bias + 0.15 * rejection_sensitivity)
        mpfc = self._clip01(0.45 * pfc_regulatory_failure + 0.10 * negative_self_rumination)
        ofc = self._clip01(0.40 * pfc_regulatory_failure + 0.10 * dopaminergic_social_reward_deficit)
        acc = self._clip01(0.35 * negative_self_rumination + 0.20 * pfc_regulatory_failure)

        # Symptoms / traits
        anxious_distress = self._clip01(
            0.35 * serotonergic_dysregulation
            + 0.25 * amygdala
            + 0.15 * rejection_sensitivity
            - 0.15 * m
        )
        depressive_distress = self._clip01(
            0.30 * serotonergic_dysregulation
            + 0.25 * negative_self_rumination
            + 0.15 * social_anhedonia
            - 0.10 * m
        )
        negative_evaluation_fear = self._clip01(
            0.40 * rejection_sensitivity + 0.25 * amygdala + 0.10 * ofc
        )
        feelings_inadequacy = self._clip01(
            0.35 * negative_self_rumination
            + 0.25 * negative_evaluation_fear
            + 0.15 * depressive_distress
        )
        social_inhibition = self._clip01(
            0.35 * negative_evaluation_fear
            + 0.20 * anxious_distress
            + 0.15 * amygdala
        )
        solitude_preference = self._clip01(
            0.45 * social_anhedonia + 0.15 * negative_evaluation_fear
        )
        anger_irritability = self._clip01(
            0.35 * negative_self_rumination
            + 0.20 * anxious_distress
            + 0.10 * amygdala
        )
        avoidance = self._clip01(
            0.30 * negative_evaluation_fear
            + 0.20 * social_inhibition
            + 0.20 * solitude_preference
            + 0.15 * feelings_inadequacy
            + 0.10 * anxious_distress
            - 0.20 * s
        )

        return {
            "inputs": pd.Series(
                {
                    "genetic_vulnerability": g,
                    "parental_rejection": r,
                    "peer_humiliation": h,
                    "supportive_environment": s,
                    "ssri_modulation": m,
                }
            ),
            "latents": pd.Series(
                {
                    "rejection_sensitivity": rejection_sensitivity,
                    "pfc_regulatory_failure": pfc_regulatory_failure,
                    "negative_self_rumination": negative_self_rumination,
                    "amygdala_threat_bias": amygdala_threat_bias,
                    "serotonergic_dysregulation": serotonergic_dysregulation,
                    "social_anhedonia": social_anhedonia,
                    "dopaminergic_social_reward_deficit": dopaminergic_social_reward_deficit,
                    "epigenetic_sensitization": epigenetic_sensitization,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "amygdala": amygdala,
                    "mpfc": mpfc,
                    "ofc": ofc,
                    "acc": acc,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "avoidance": avoidance,
                    "negative_evaluation_fear": negative_evaluation_fear,
                    "social_inhibition": social_inhibition,
                    "feelings_inadequacy": feelings_inadequacy,
                    "anxious_distress": anxious_distress,
                    "depressive_distress": depressive_distress,
                    "solitude_preference": solitude_preference,
                    "anger_irritability": anger_irritability,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "rejection_sensitive_avoidant_profile": self._clip01(
                        0.40 * negative_evaluation_fear
                        + 0.30 * avoidance
                        + 0.20 * anxious_distress
                    ),
                    "social_anhedonic_avoidant_profile": self._clip01(
                        0.45 * solitude_preference
                        + 0.30 * social_anhedonia
                        + 0.20 * avoidance
                    ),
                    "ruminative_inadequacy_profile": self._clip01(
                        0.40 * feelings_inadequacy
                        + 0.30 * negative_self_rumination
                        + 0.20 * depressive_distress
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-24, 24, -12)).head(10)
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
    model = AvoidantPersonalityDisorderModel()

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
        bundle["edges"][["source", "target", "relation", "avpd_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["amygdala", "mpfc", "ofc", "acc"]:
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
        parental_rejection=0.65,
        peer_humiliation=0.80,
        supportive_environment=0.20,
        ssri_modulation=0.15,
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
    # print(model.assign_mni_point((-24, 24, -12)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("orbitofrontal").to_string(index=False))
    # print(model.suggest_regions("cingulate").to_string(index=False))
