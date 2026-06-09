from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Borderline Personality Disorder-oriented panel:
# - serotonin / impulse-affect regulation
# - dopamine / impulsive drive / novelty seeking
# - norepinephrine / abandonment-linked hyperarousal
# - GABA / glutamate balance
# - stress responsivity / trauma sensitization
DEFAULT_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR1A",    # serotonin receptor
    "HTR2A",    # serotonin receptor
    "TPH2",     # serotonin synthesis
    "DRD2",     # dopamine receptor
    "SLC6A3",   # dopamine transporter
    "COMT",     # catecholamine metabolism
    "DBH",      # dopamine beta hydroxylase / norepinephrine synthesis step
    "MAOA",     # monoamine metabolism
    "GABRA2",   # GABA-A receptor
    "GABRB2",   # GABA-A receptor
    "GRIN2B",   # NMDA receptor subunit
    "SLC1A1",   # glutamatergic/OCD-spectrum relevance
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
    "CRHR1",    # CRH signaling
    "BDNF",     # plasticity / trophic support
]


class BorderlinePersonalityDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Borderline Personality Disorder (BPD).

    What it does:
      1) Resolves BPD-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates affective instability, impulsivity, abandonment panic,
         self-harm risk, emptiness, and interpersonal instability.

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

        # Chapter-consistent fronto-limbic circuit with a caudate proxy.
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
            "caudate_proxy": [
                "caudate",
                "caudate nucleus",
                "striatum",
                "basal ganglia",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_liability": "Heritable liability for unstable temperament and BPD risk",
            "developmental_trauma": "Childhood trauma during critical periods of neurodevelopment",
            "abandonment_cues": "Attachment threat and abandonment-triggering interpersonal cues",
            "chronic_invalidating_stress": "Repeated invalidation, relational chaos, and stress exposure",
            "treatment_support": "Protective treatment and stabilizing recovery support",
        }

        self.latent_nodes: Dict[str, str] = {
            "unstable_temperament_profile": "Trait profile combining high harm avoidance, high novelty seeking, and low reward dependence",
            "gene_environment_sensitization": "Trauma-amplified biological vulnerability",
            "serotonergic_dysregulation": "Poor affective and impulse regulation through serotonin dysregulation",
            "dopaminergic_impulsive_drive": "Impulsive/exploratory drive and unstable salience assignment",
            "noradrenergic_hyperarousal": "Threat-linked arousal and rapid stress reactivity",
            "gaba_glutamate_imbalance": "Excitatory/inhibitory imbalance contributing to hyperexcitability",
            "acc_glutamate_pressure": "ACC-centered glutamatergic stress burden",
            "frontolimbic_dysregulation": "Weak prefrontal inhibition of hyper-reactive limbic systems",
            "negative_self_other_schema": "Unstable self/other representations linked to identity and relationship disturbance",
        }

        self.symptom_nodes: Dict[str, str] = {
            "affective_instability": "Rapid, intense emotional lability",
            "impulsivity": "Poor inhibitory control and rash behavior",
            "anger_reactivity": "Inappropriate or intense anger and irritability",
            "abandonment_panic": "Frantic responses to perceived abandonment",
            "self_harm_suicidality": "Self-damaging behavior or suicidal crises under dysregulation",
            "identity_disturbance": "Unstable self-image and self-experience",
            "chronic_emptiness": "Persistent inner emptiness and detachment",
            "interpersonal_instability": "Unstable relationships shaped by fear, anger, and self-distortion",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_liability",
                "target": "unstable_temperament_profile",
                "relation": "loads heritable temperament vulnerabilities",
                "bpd_change": "increased susceptibility",
            },
            {
                "source": "genetic_liability",
                "target": "serotonergic_dysregulation",
                "relation": "raises liability for affective and impulse-control instability",
                "bpd_change": "increased susceptibility",
            },
            {
                "source": "genetic_liability",
                "target": "dopaminergic_impulsive_drive",
                "relation": "raises liability for novelty seeking and impulsivity",
                "bpd_change": "increased susceptibility",
            },
            {
                "source": "genetic_liability",
                "target": "gene_environment_sensitization",
                "relation": "creates vulnerability for trauma-amplified expression",
                "bpd_change": "increased susceptibility",
            },
            {
                "source": "developmental_trauma",
                "target": "gene_environment_sensitization",
                "relation": "amplifies biological risk through developmental stress",
                "bpd_change": "increased",
            },
            {
                "source": "developmental_trauma",
                "target": "gaba_glutamate_imbalance",
                "relation": "pushes excitation/inhibition balance toward hyperexcitability",
                "bpd_change": "increased",
            },
            {
                "source": "developmental_trauma",
                "target": "acc_glutamate_pressure",
                "relation": "increases stress-linked glutamatergic burden in ACC-related regulation systems",
                "bpd_change": "increased",
            },
            {
                "source": "developmental_trauma",
                "target": "frontolimbic_dysregulation",
                "relation": "impairs maturation of prefrontal-limbic control",
                "bpd_change": "increased",
            },
            {
                "source": "abandonment_cues",
                "target": "noradrenergic_hyperarousal",
                "relation": "triggers rapid threat-linked arousal",
                "bpd_change": "increased",
            },
            {
                "source": "abandonment_cues",
                "target": "abandonment_panic",
                "relation": "directly provokes frantic attachment panic",
                "bpd_change": "increased",
            },
            {
                "source": "chronic_invalidating_stress",
                "target": "gene_environment_sensitization",
                "relation": "stabilizes biological vulnerability under repeated stress",
                "bpd_change": "increased",
            },
            {
                "source": "chronic_invalidating_stress",
                "target": "negative_self_other_schema",
                "relation": "shapes unstable, negative self-other representations",
                "bpd_change": "increased",
            },
            {
                "source": "treatment_support",
                "target": "gaba_glutamate_imbalance",
                "relation": "buffers excitatory/inhibitory dysregulation",
                "bpd_change": "protective",
            },
            {
                "source": "treatment_support",
                "target": "frontolimbic_dysregulation",
                "relation": "supports regulatory stabilization",
                "bpd_change": "protective",
            },
            {
                "source": "treatment_support",
                "target": "self_harm_suicidality",
                "relation": "reduces crisis expression under dysregulation",
                "bpd_change": "protective",
            },
            {
                "source": "unstable_temperament_profile",
                "target": "dopaminergic_impulsive_drive",
                "relation": "supports novelty seeking and unstable salience",
                "bpd_change": "increased",
            },
            {
                "source": "unstable_temperament_profile",
                "target": "noradrenergic_hyperarousal",
                "relation": "supports anxious, quick-tempered stress reactivity",
                "bpd_change": "increased",
            },
            {
                "source": "gene_environment_sensitization",
                "target": "serotonergic_dysregulation",
                "relation": "stabilizes affective vulnerability",
                "bpd_change": "increased",
            },
            {
                "source": "gene_environment_sensitization",
                "target": "frontolimbic_dysregulation",
                "relation": "stabilizes trauma-linked regulatory dysfunction",
                "bpd_change": "increased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "affective_instability",
                "relation": "weakens modulation of intense shifting affect",
                "bpd_change": "increased",
            },
            {
                "source": "serotonergic_dysregulation",
                "target": "impulsivity",
                "relation": "weakens inhibitory control over action",
                "bpd_change": "increased",
            },
            {
                "source": "dopaminergic_impulsive_drive",
                "target": "impulsivity",
                "relation": "supports rash and exploratory behavior",
                "bpd_change": "increased",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "anger_reactivity",
                "relation": "amplifies hot, stress-linked anger states",
                "bpd_change": "increased",
            },
            {
                "source": "noradrenergic_hyperarousal",
                "target": "abandonment_panic",
                "relation": "amplifies urgent threat-linked attachment responses",
                "bpd_change": "increased",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "impulsivity",
                "relation": "hyperexcitability weakens emotional and behavioral braking",
                "bpd_change": "increased",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "anger_reactivity",
                "relation": "hyperexcitability amplifies irritability and explosiveness",
                "bpd_change": "increased",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "acc_glutamate_pressure",
                "relation": "can concentrate stress-linked excitation in ACC-centered regulation systems",
                "bpd_change": "increased",
            },
            {
                "source": "acc_glutamate_pressure",
                "target": "acc",
                "relation": "burdens ACC emotional/cognitive control function",
                "bpd_change": "increased dysregulation",
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "amygdala",
                "relation": "permits limbic hyper-reactivity",
                "bpd_change": "increased dysregulation",
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "hippocampus",
                "relation": "burdens context and emotional-memory systems",
                "bpd_change": "increased dysregulation",
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "ofc",
                "relation": "reduces orbitofrontal impulse and social-behavior regulation",
                "bpd_change": "reduced function",
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "dlpfc",
                "relation": "reduces executive control and working self-regulation",
                "bpd_change": "reduced function",
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "acc",
                "relation": "reduces conflict monitoring and emotion regulation",
                "bpd_change": "reduced function",
            },
            {
                "source": "frontolimbic_dysregulation",
                "target": "caudate_proxy",
                "relation": "burdens subcortical action-selection and inhibitory control systems",
                "bpd_change": "reduced function",
            },
            {
                "source": "ofc",
                "target": "amygdala",
                "relation": "normally provides top-down inhibition",
                "bpd_change": "reduced inhibition",
            },
            {
                "source": "dlpfc",
                "target": "impulsivity",
                "relation": "reduced executive control weakens restraint",
                "bpd_change": "increased",
            },
            {
                "source": "acc",
                "target": "affective_instability",
                "relation": "reduced ACC regulation destabilizes mood intensity",
                "bpd_change": "increased",
            },
            {
                "source": "caudate_proxy",
                "target": "impulsivity",
                "relation": "subcortical regulatory burden weakens action control",
                "bpd_change": "increased",
            },
            {
                "source": "negative_self_other_schema",
                "target": "identity_disturbance",
                "relation": "destabilizes self-representation",
                "bpd_change": "increased",
            },
            {
                "source": "negative_self_other_schema",
                "target": "chronic_emptiness",
                "relation": "promotes fragmented inner continuity and emptiness",
                "bpd_change": "increased",
            },
            {
                "source": "negative_self_other_schema",
                "target": "interpersonal_instability",
                "relation": "destabilizes expectations and interpretations of others",
                "bpd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "affective_instability",
                "relation": "amplifies intense poorly controlled emotions",
                "bpd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "anger_reactivity",
                "relation": "amplifies threat-linked anger reactivity",
                "bpd_change": "increased",
            },
            {
                "source": "abandonment_panic",
                "target": "interpersonal_instability",
                "relation": "drives frantic, unstable relational responses",
                "bpd_change": "increased",
            },
            {
                "source": "abandonment_panic",
                "target": "self_harm_suicidality",
                "relation": "can precipitate crisis behavior under attachment threat",
                "bpd_change": "increased",
            },
            {
                "source": "affective_instability",
                "target": "self_harm_suicidality",
                "relation": "intense rapidly shifting affect increases crisis risk",
                "bpd_change": "increased",
            },
            {
                "source": "anger_reactivity",
                "target": "interpersonal_instability",
                "relation": "destabilizes relationships through explosive affective reactions",
                "bpd_change": "increased",
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

    def _region_rank(self, region: Any) -> Tuple[int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "hippocampus",
            "anterior cingulate",
            "orbitofrontal cortex",
            "prefrontal cortex",
            "basal ganglia",
            "striatum",
            "caudate",
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

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

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
        Useful for tuning caudate / frontal candidates.
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

    def _safe_features_any(self, concept: Any, modalities: Sequence[Any], **kwargs: Any) -> List[Any]:
        for modality in modalities:
            feats = self._safe_features(concept, modality, **kwargs)
            if feats:
                return feats
        return []

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(
            region,
            [
                getattr(getattr(siibra, "features", object()), "molecular", object()).ReceptorDensityFingerprint
                if hasattr(getattr(siibra, "features", object()), "molecular")
                and hasattr(getattr(siibra.features.molecular, "ReceptorDensityFingerprint", object()), "__class__")
                else "receptor density fingerprint",
                "receptor density fingerprint",
            ],
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
        gene_modalities = ["gene expressions"]
        try:
            gene_modalities.insert(0, siibra.features.molecular.GeneExpressions)
        except Exception:
            pass

        feats = self._safe_features_any(region, gene_modalities, gene=list(genes))
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

        conn_modalities = ["StreamlineCounts"]
        try:
            conn_modalities.insert(0, siibra.features.connectivity.StreamlineCounts)
        except Exception:
            pass

        feats = self._safe_features_any(self.parcellation, conn_modalities)
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
        node_keys: Sequence[str] = ("amygdala", "hippocampus", "ofc", "dlpfc", "acc", "caudate_proxy"),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the BPD circuit.
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
                    "description": "Atlas-backed BPD circuit node",
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
        genetic_liability: float,
        developmental_trauma: float,
        abandonment_cues: float,
        chronic_invalidating_stress: float,
        treatment_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while treatment_support is protective.
        """
        g = self._clip01(genetic_liability)
        d = self._clip01(developmental_trauma)
        a = self._clip01(abandonment_cues)
        s = self._clip01(chronic_invalidating_stress)
        t = self._clip01(treatment_support)

        # Latent biology
        unstable_temperament_profile = self._clip01(
            0.35 * g + 0.10 * s
        )
        gene_environment_sensitization = self._clip01(
            0.30 * g + 0.30 * d + 0.20 * s - 0.15 * t
        )
        serotonergic_dysregulation = self._clip01(
            0.30 * g + 0.25 * gene_environment_sensitization + 0.10 * s - 0.10 * t
        )
        dopaminergic_impulsive_drive = self._clip01(
            0.30 * unstable_temperament_profile + 0.20 * g + 0.10 * a
        )
        noradrenergic_hyperarousal = self._clip01(
            0.30 * unstable_temperament_profile + 0.25 * a + 0.10 * d
        )
        gaba_glutamate_imbalance = self._clip01(
            0.30 * d + 0.20 * gene_environment_sensitization + 0.15 * s - 0.15 * t
        )
        acc_glutamate_pressure = self._clip01(
            0.35 * gaba_glutamate_imbalance + 0.20 * d + 0.10 * s - 0.10 * t
        )
        negative_self_other_schema = self._clip01(
            0.30 * s + 0.20 * gene_environment_sensitization + 0.20 * a - 0.15 * t
        )
        frontolimbic_dysregulation = self._clip01(
            0.30 * gene_environment_sensitization
            + 0.20 * gaba_glutamate_imbalance
            + 0.20 * noradrenergic_hyperarousal
            + 0.10 * serotonergic_dysregulation
            - 0.20 * t
        )

        # Regional state proxies
        amygdala = self._clip01(
            0.35 * frontolimbic_dysregulation + 0.25 * noradrenergic_hyperarousal
        )
        hippocampus = self._clip01(
            0.25 * frontolimbic_dysregulation + 0.20 * negative_self_other_schema
        )
        ofc = self._clip01(
            0.35 * frontolimbic_dysregulation + 0.15 * dopaminergic_impulsive_drive - 0.15 * t
        )
        dlpfc = self._clip01(
            0.35 * frontolimbic_dysregulation + 0.15 * negative_self_other_schema - 0.15 * t
        )
        acc = self._clip01(
            0.30 * frontolimbic_dysregulation + 0.25 * acc_glutamate_pressure - 0.15 * t
        )
        caudate_proxy = self._clip01(
            0.25 * frontolimbic_dysregulation + 0.20 * dopaminergic_impulsive_drive
        )

        # Symptoms / behavior
        affective_instability = self._clip01(
            0.30 * serotonergic_dysregulation
            + 0.25 * amygdala
            + 0.20 * acc
            + 0.10 * noradrenergic_hyperarousal
        )
        impulsivity = self._clip01(
            0.30 * dopaminergic_impulsive_drive
            + 0.25 * gaba_glutamate_imbalance
            + 0.15 * ofc
            + 0.10 * dlpfc
            + 0.10 * caudate_proxy
            - 0.10 * t
        )
        anger_reactivity = self._clip01(
            0.35 * amygdala
            + 0.25 * noradrenergic_hyperarousal
            + 0.15 * serotonergic_dysregulation
        )
        abandonment_panic = self._clip01(
            0.35 * a
            + 0.25 * noradrenergic_hyperarousal
            + 0.20 * amygdala
            + 0.10 * negative_self_other_schema
        )
        identity_disturbance = self._clip01(
            0.40 * negative_self_other_schema
            + 0.20 * dlpfc
            + 0.10 * hippocampus
        )
        chronic_emptiness = self._clip01(
            0.35 * negative_self_other_schema
            + 0.25 * identity_disturbance
            + 0.10 * serotonergic_dysregulation
        )
        interpersonal_instability = self._clip01(
            0.30 * abandonment_panic
            + 0.20 * anger_reactivity
            + 0.20 * identity_disturbance
            + 0.10 * affective_instability
        )
        self_harm_suicidality = self._clip01(
            0.30 * affective_instability
            + 0.25 * abandonment_panic
            + 0.20 * impulsivity
            + 0.10 * chronic_emptiness
            - 0.20 * t
        )

        return {
            "inputs": pd.Series(
                {
                    "genetic_liability": g,
                    "developmental_trauma": d,
                    "abandonment_cues": a,
                    "chronic_invalidating_stress": s,
                    "treatment_support": t,
                }
            ),
            "latents": pd.Series(
                {
                    "frontolimbic_dysregulation": frontolimbic_dysregulation,
                    "negative_self_other_schema": negative_self_other_schema,
                    "gene_environment_sensitization": gene_environment_sensitization,
                    "acc_glutamate_pressure": acc_glutamate_pressure,
                    "gaba_glutamate_imbalance": gaba_glutamate_imbalance,
                    "serotonergic_dysregulation": serotonergic_dysregulation,
                    "noradrenergic_hyperarousal": noradrenergic_hyperarousal,
                    "dopaminergic_impulsive_drive": dopaminergic_impulsive_drive,
                    "unstable_temperament_profile": unstable_temperament_profile,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "amygdala": amygdala,
                    "hippocampus": hippocampus,
                    "ofc": ofc,
                    "dlpfc": dlpfc,
                    "acc": acc,
                    "caudate_proxy": caudate_proxy,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "self_harm_suicidality": self_harm_suicidality,
                    "affective_instability": affective_instability,
                    "interpersonal_instability": interpersonal_instability,
                    "abandonment_panic": abandonment_panic,
                    "anger_reactivity": anger_reactivity,
                    "impulsivity": impulsivity,
                    "identity_disturbance": identity_disturbance,
                    "chronic_emptiness": chronic_emptiness,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "affective_instability_profile": self._clip01(
                        0.45 * affective_instability
                        + 0.25 * anger_reactivity
                        + 0.15 * abandonment_panic
                    ),
                    "impulsive_self_damaging_profile": self._clip01(
                        0.45 * self_harm_suicidality
                        + 0.25 * impulsivity
                        + 0.15 * affective_instability
                    ),
                    "abandonment_interpersonal_profile": self._clip01(
                        0.45 * abandonment_panic
                        + 0.30 * interpersonal_instability
                        + 0.15 * identity_disturbance
                    ),
                    "emptiness_identity_profile": self._clip01(
                        0.45 * chronic_emptiness
                        + 0.30 * identity_disturbance
                        + 0.15 * negative_self_other_schema
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
    model = BorderlinePersonalityDisorderModel()

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
        bundle["edges"][["source", "target", "relation", "bpd_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["amygdala", "hippocampus", "ofc", "dlpfc", "acc", "caudate_proxy"]:
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
        genetic_liability=0.75,
        developmental_trauma=0.85,
        abandonment_cues=0.80,
        chronic_invalidating_stress=0.70,
        treatment_support=0.20,
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
    # print(model.suggest_regions("caudate").to_string(index=False))
    # print(model.suggest_regions("orbitofrontal").to_string(index=False))
    # print(model.suggest_regions("cingulate").to_string(index=False))
