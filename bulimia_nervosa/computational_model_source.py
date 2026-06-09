from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Bulimia Nervosa-oriented panel:
# - serotonin / satiety / impulse control
# - dopamine / cue-driven wanting
# - norepinephrine / arousal and stress
# - opioid / hedonic liking
# - GABA / glutamate balance
# - stress and interoceptive regulation
DEFAULT_GENE_PANEL = [
    "SLC6A4",   # serotonin transporter
    "HTR2C",    # serotonin receptor relevant to satiety/control
    "HTR1A",    # serotonin receptor
    "TPH2",     # serotonin synthesis
    "SLC6A2",   # norepinephrine transporter
    "DBH",      # dopamine beta hydroxylase / NE synthesis
    "DRD2",     # dopamine receptor
    "SLC6A3",   # dopamine transporter
    "COMT",     # catecholamine metabolism
    "OPRM1",    # mu-opioid receptor
    "GABRA2",   # GABA-A receptor
    "GABRB2",   # GABA-A receptor
    "GRIN2B",   # NMDA receptor subunit
    "SLC1A1",   # glutamate transport
    "NR3C1",    # glucocorticoid receptor
    "FKBP5",    # stress responsivity
    "CRHR1",    # CRH signaling
    "BDNF",     # plasticity
]


class BulimiaNervosaModel:
    """
    Atlas-grounded mechanistic scaffold for Bulimia Nervosa (BN).

    What it does:
      1) Resolves BN-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates binge urge, loss of control, purging, cue reactivity,
         and stress-linked binge-purge cycling.

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

        # BN chapter-consistent reward / control / interoception circuit.
        # Ventral striatum is implemented as an accumbens proxy because exact
        # naming can vary across atlas versions.
        self.region_candidates: Dict[str, List[str]] = {
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
            "insula": [
                "Area Id2 (Insula) left",
                "Area Id3 (Insula) left",
                "Area Id4 (Insula) left",
                "insula",
            ],
            "dlpfc": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "dorsolateral prefrontal",
                "middle frontal",
                "superior frontal",
            ],
            "ventral_striatum_proxy": [
                "nucleus accumbens",
                "accumbens",
                "ventral striatum",
                "striatum",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_vulnerability": "Polygenic liability affecting reward, satiety, mood, and impulse-control systems",
            "dietary_restriction": "Restriction-driven hunger and physiological deprivation",
            "interpersonal_stress": "Interpersonal stressors and psychosocial conflict",
            "negative_mood": "Anxiety, depression, guilt, and dysphoric affect",
            "palatable_food_cues": "Highly salient food cues and reward triggers",
            "recovery_support": "Protective treatment, structure, and recovery support",
        }

        self.latent_nodes: Dict[str, str] = {
            "gene_environment_sensitization": "Gene-environment amplification of BN vulnerability",
            "serotonin_satiety_impulse_dysregulation": "Reduced satiety and impaired impulse control",
            "dopamine_wanting": "Motivational salience and craving for food",
            "noradrenergic_arousal": "Stress-linked arousal and activation",
            "opioid_liking": "Hedonic reinforcement of palatable intake",
            "gaba_glutamate_imbalance": "Cortical balance disruption affecting control and relapse",
            "interoceptive_distortion": "Disturbed sensing of hunger, fullness, and visceral state",
            "frontostriatal_control_failure": "Impaired engagement of prefrontal control circuits",
            "amygdala_reward_coupling": "Bottom-up reward/emotion amplification to food cues or distress",
            "purge_negative_reinforcement": "Temporary relief from guilt and discomfort that perpetuates the cycle",
        }

        self.symptom_nodes: Dict[str, str] = {
            "craving": "Urge and motivational pull toward palatable food",
            "loss_of_control": "Subjective inability to stop or regulate intake",
            "binge_eating": "Consumption of large amounts of food with diminished control",
            "post_binge_distress": "Guilt, discomfort, and dysphoric aftermath",
            "purging": "Compensatory purging behavior",
            "cue_triggered_bingeing": "Bingeing precipitated by food cues",
            "stress_triggered_bingeing": "Bingeing precipitated by stress or negative affect",
            "body_state_misperception": "Disturbed interpretation of hunger, satiety, and bodily signals",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_vulnerability",
                "target": "gene_environment_sensitization",
                "relation": "creates latent vulnerability that can be amplified by stress and context",
                "bn_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "serotonin_satiety_impulse_dysregulation",
                "relation": "raises vulnerability for satiety and impulse-control dysregulation",
                "bn_change": "increased susceptibility",
            },
            {
                "source": "genetic_vulnerability",
                "target": "dopamine_wanting",
                "relation": "raises vulnerability for reward-driven food wanting",
                "bn_change": "increased susceptibility",
            },
            {
                "source": "dietary_restriction",
                "target": "serotonin_satiety_impulse_dysregulation",
                "relation": "restriction destabilizes satiety and control signaling",
                "bn_change": "increased",
            },
            {
                "source": "dietary_restriction",
                "target": "craving",
                "relation": "restriction-driven hunger increases binge urge",
                "bn_change": "increased",
            },
            {
                "source": "dietary_restriction",
                "target": "binge_eating",
                "relation": "restriction increases vulnerability to rebound binge episodes",
                "bn_change": "increased",
            },
            {
                "source": "interpersonal_stress",
                "target": "noradrenergic_arousal",
                "relation": "increases arousal and stress-linked activation",
                "bn_change": "increased",
            },
            {
                "source": "interpersonal_stress",
                "target": "stress_triggered_bingeing",
                "relation": "can precipitate bingeing under psychosocial load",
                "bn_change": "increased",
            },
            {
                "source": "negative_mood",
                "target": "stress_triggered_bingeing",
                "relation": "promotes bingeing as maladaptive affect regulation",
                "bn_change": "increased",
            },
            {
                "source": "negative_mood",
                "target": "post_binge_distress",
                "relation": "amplifies guilt and dysphoria after bingeing",
                "bn_change": "increased",
            },
            {
                "source": "palatable_food_cues",
                "target": "dopamine_wanting",
                "relation": "amplifies incentive salience of food",
                "bn_change": "increased",
            },
            {
                "source": "palatable_food_cues",
                "target": "cue_triggered_bingeing",
                "relation": "can directly trigger binge episodes",
                "bn_change": "increased",
            },
            {
                "source": "palatable_food_cues",
                "target": "amygdala_reward_coupling",
                "relation": "strengthens bottom-up reward and emotional salience",
                "bn_change": "increased",
            },
            {
                "source": "recovery_support",
                "target": "frontostriatal_control_failure",
                "relation": "supports recruitment of self-regulatory control",
                "bn_change": "protective",
            },
            {
                "source": "recovery_support",
                "target": "purging",
                "relation": "reduces expression of the binge-purge cycle",
                "bn_change": "protective",
            },
            {
                "source": "gene_environment_sensitization",
                "target": "serotonin_satiety_impulse_dysregulation",
                "relation": "stabilizes satiety and mood-control vulnerability",
                "bn_change": "increased",
            },
            {
                "source": "gene_environment_sensitization",
                "target": "noradrenergic_arousal",
                "relation": "stabilizes stress-reactive vulnerability",
                "bn_change": "increased",
            },
            {
                "source": "dopamine_wanting",
                "target": "ventral_striatum_proxy",
                "relation": "loads ventral-striatal incentive salience circuitry",
                "bn_change": "increased activation bias",
            },
            {
                "source": "dopamine_wanting",
                "target": "craving",
                "relation": "drives food wanting and salience",
                "bn_change": "increased",
            },
            {
                "source": "noradrenergic_arousal",
                "target": "stress_triggered_bingeing",
                "relation": "amplifies stress-linked binge pressure",
                "bn_change": "increased",
            },
            {
                "source": "opioid_liking",
                "target": "binge_eating",
                "relation": "hedonic reinforcement supports binge continuation",
                "bn_change": "increased",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "frontostriatal_control_failure",
                "relation": "destabilizes cortical balance and control circuits",
                "bn_change": "increased",
            },
            {
                "source": "gaba_glutamate_imbalance",
                "target": "loss_of_control",
                "relation": "supports failure to inhibit prepotent binge urges",
                "bn_change": "increased",
            },
            {
                "source": "serotonin_satiety_impulse_dysregulation",
                "target": "interoceptive_distortion",
                "relation": "weakens normal satiety and internal-state regulation",
                "bn_change": "increased",
            },
            {
                "source": "interoceptive_distortion",
                "target": "insula",
                "relation": "loads insular interoceptive processing",
                "bn_change": "increased dysregulation",
            },
            {
                "source": "interoceptive_distortion",
                "target": "body_state_misperception",
                "relation": "distorts hunger, fullness, and visceral discomfort signals",
                "bn_change": "increased",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "dlpfc",
                "relation": "reflects weak top-down recruitment of control systems",
                "bn_change": "reduced function",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "ofc",
                "relation": "reflects altered reward valuation and decision control",
                "bn_change": "reduced function",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "acc",
                "relation": "reflects impaired error/conflict monitoring during binge-purge behavior",
                "bn_change": "reduced function",
            },
            {
                "source": "frontostriatal_control_failure",
                "target": "loss_of_control",
                "relation": "reduces deliberate inhibition of binge urges",
                "bn_change": "increased",
            },
            {
                "source": "amygdala_reward_coupling",
                "target": "amygdala",
                "relation": "amplifies emotional salience of food and distress",
                "bn_change": "increased dysregulation",
            },
            {
                "source": "amygdala_reward_coupling",
                "target": "ventral_striatum_proxy",
                "relation": "amplifies reward-linked cue reactivity",
                "bn_change": "increased dysregulation",
            },
            {
                "source": "amygdala_reward_coupling",
                "target": "cue_triggered_bingeing",
                "relation": "biases behavior toward impulsive cue-driven intake",
                "bn_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "stress_triggered_bingeing",
                "relation": "amplifies negative-affect-driven eating",
                "bn_change": "increased",
            },
            {
                "source": "ventral_striatum_proxy",
                "target": "craving",
                "relation": "supports reward-linked wanting",
                "bn_change": "increased",
            },
            {
                "source": "ofc",
                "target": "craving",
                "relation": "altered OFC valuation can amplify desire for food",
                "bn_change": "increased",
            },
            {
                "source": "acc",
                "target": "post_binge_distress",
                "relation": "error/conflict burden contributes to distress after bingeing",
                "bn_change": "increased",
            },
            {
                "source": "craving",
                "target": "binge_eating",
                "relation": "drives approach and intake of palatable food",
                "bn_change": "increased",
            },
            {
                "source": "loss_of_control",
                "target": "binge_eating",
                "relation": "directly supports binge episodes",
                "bn_change": "increased",
            },
            {
                "source": "binge_eating",
                "target": "post_binge_distress",
                "relation": "generates discomfort and guilt after the episode",
                "bn_change": "increased",
            },
            {
                "source": "post_binge_distress",
                "target": "purging",
                "relation": "promotes purging to reduce distress and discomfort",
                "bn_change": "increased",
            },
            {
                "source": "purging",
                "target": "purge_negative_reinforcement",
                "relation": "temporarily relieves discomfort and guilt",
                "bn_change": "increased",
            },
            {
                "source": "purge_negative_reinforcement",
                "target": "dietary_restriction",
                "relation": "helps perpetuate the binge-purge cycle through renewed restriction pressure",
                "bn_change": "increased cycle pressure",
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
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {
            "amygdala",
            "insula",
            "striatum",
            "ventral striatum",
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
        Useful for tuning insula or ventral-striatal candidates.
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
        node_keys: Sequence[str] = ("ofc", "acc", "insula", "dlpfc", "ventral_striatum_proxy", "amygdala"),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the BN circuit.
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
                    "description": "Atlas-backed BN circuit node",
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
        dietary_restriction: float,
        interpersonal_stress: float,
        negative_mood: float,
        palatable_food_cues: float,
        recovery_support: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger dysregulation / symptom burden,
        while recovery_support is protective.
        """
        g = self._clip01(genetic_vulnerability)
        d = self._clip01(dietary_restriction)
        i = self._clip01(interpersonal_stress)
        n = self._clip01(negative_mood)
        c = self._clip01(palatable_food_cues)
        r = self._clip01(recovery_support)

        # Latent biology
        gene_environment_sensitization = self._clip01(
            0.30 * g + 0.20 * i + 0.20 * n - 0.10 * r
        )
        serotonin_satiety_impulse_dysregulation = self._clip01(
            0.25 * g + 0.25 * d + 0.15 * n + 0.10 * gene_environment_sensitization - 0.15 * r
        )
        noradrenergic_arousal = self._clip01(
            0.30 * i + 0.25 * n + 0.10 * gene_environment_sensitization - 0.10 * r
        )
        dopamine_wanting = self._clip01(
            0.25 * g + 0.30 * c + 0.15 * d + 0.10 * n - 0.10 * r
        )
        opioid_liking = self._clip01(
            0.30 * c + 0.20 * dopamine_wanting
        )
        gaba_glutamate_imbalance = self._clip01(
            0.20 * n + 0.20 * i + 0.20 * serotonin_satiety_impulse_dysregulation - 0.10 * r
        )
        interoceptive_distortion = self._clip01(
            0.35 * serotonin_satiety_impulse_dysregulation + 0.20 * d + 0.10 * n
        )
        frontostriatal_control_failure = self._clip01(
            0.30 * gaba_glutamate_imbalance
            + 0.25 * noradrenergic_arousal
            + 0.15 * dopamine_wanting
            - 0.25 * r
        )
        amygdala_reward_coupling = self._clip01(
            0.30 * c + 0.25 * n + 0.20 * noradrenergic_arousal + 0.10 * i
        )

        # Regional state proxies
        ofc = self._clip01(
            0.35 * frontostriatal_control_failure + 0.20 * dopamine_wanting + 0.10 * c - 0.10 * r
        )
        acc = self._clip01(
            0.35 * frontostriatal_control_failure + 0.25 * n + 0.10 * i - 0.10 * r
        )
        insula = self._clip01(
            0.45 * interoceptive_distortion + 0.15 * d + 0.10 * n
        )
        dlpfc = self._clip01(
            0.45 * frontostriatal_control_failure + 0.10 * n - 0.15 * r
        )
        ventral_striatum_proxy = self._clip01(
            0.40 * dopamine_wanting + 0.20 * amygdala_reward_coupling + 0.15 * opioid_liking
        )
        amygdala = self._clip01(
            0.35 * amygdala_reward_coupling + 0.25 * noradrenergic_arousal
        )

        # Symptoms / behavior
        craving = self._clip01(
            0.30 * dopamine_wanting
            + 0.20 * ventral_striatum_proxy
            + 0.20 * c
            + 0.10 * opioid_liking
            + 0.10 * ofc
        )
        body_state_misperception = self._clip01(
            0.45 * insula + 0.20 * interoceptive_distortion
        )
        loss_of_control = self._clip01(
            0.30 * frontostriatal_control_failure
            + 0.25 * craving
            + 0.20 * serotonin_satiety_impulse_dysregulation
            + 0.10 * n
            - 0.10 * r
        )
        cue_triggered_bingeing = self._clip01(
            0.35 * craving + 0.25 * amygdala_reward_coupling + 0.15 * c
        )
        stress_triggered_bingeing = self._clip01(
            0.35 * n + 0.25 * noradrenergic_arousal + 0.15 * amygdala + 0.10 * i
        )
        binge_eating = self._clip01(
            0.25 * loss_of_control
            + 0.20 * cue_triggered_bingeing
            + 0.20 * stress_triggered_bingeing
            + 0.15 * d
            + 0.10 * body_state_misperception
            - 0.15 * r
        )
        post_binge_distress = self._clip01(
            0.35 * binge_eating + 0.25 * acc + 0.20 * n
        )
        purging = self._clip01(
            0.35 * post_binge_distress + 0.25 * loss_of_control + 0.15 * binge_eating - 0.15 * r
        )
        purge_negative_reinforcement = self._clip01(
            0.35 * purging + 0.20 * post_binge_distress
        )

        return {
            "inputs": pd.Series(
                {
                    "genetic_vulnerability": g,
                    "dietary_restriction": d,
                    "interpersonal_stress": i,
                    "negative_mood": n,
                    "palatable_food_cues": c,
                    "recovery_support": r,
                }
            ),
            "latents": pd.Series(
                {
                    "frontostriatal_control_failure": frontostriatal_control_failure,
                    "amygdala_reward_coupling": amygdala_reward_coupling,
                    "dopamine_wanting": dopamine_wanting,
                    "serotonin_satiety_impulse_dysregulation": serotonin_satiety_impulse_dysregulation,
                    "interoceptive_distortion": interoceptive_distortion,
                    "noradrenergic_arousal": noradrenergic_arousal,
                    "gaba_glutamate_imbalance": gaba_glutamate_imbalance,
                    "gene_environment_sensitization": gene_environment_sensitization,
                    "opioid_liking": opioid_liking,
                    "purge_negative_reinforcement": purge_negative_reinforcement,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "ofc": ofc,
                    "acc": acc,
                    "insula": insula,
                    "dlpfc": dlpfc,
                    "ventral_striatum_proxy": ventral_striatum_proxy,
                    "amygdala": amygdala,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "binge_eating": binge_eating,
                    "purging": purging,
                    "loss_of_control": loss_of_control,
                    "craving": craving,
                    "post_binge_distress": post_binge_distress,
                    "cue_triggered_bingeing": cue_triggered_bingeing,
                    "stress_triggered_bingeing": stress_triggered_bingeing,
                    "body_state_misperception": body_state_misperception,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "restriction_triggered_bn_profile": self._clip01(
                        0.40 * dietary_restriction
                        + 0.30 * binge_eating
                        + 0.20 * loss_of_control
                    ),
                    "cue_reactive_bn_profile": self._clip01(
                        0.45 * cue_triggered_bingeing
                        + 0.30 * craving
                        + 0.15 * binge_eating
                    ),
                    "affect_driven_bn_profile": self._clip01(
                        0.45 * stress_triggered_bingeing
                        + 0.25 * post_binge_distress
                        + 0.20 * purging
                    ),
                    "binge_purge_cycle_profile": self._clip01(
                        0.35 * binge_eating
                        + 0.30 * purging
                        + 0.20 * purge_negative_reinforcement
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-10, 12, -8)).head(10)
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
    model = BulimiaNervosaModel()

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
        bundle["edges"][["source", "target", "relation", "bn_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["ofc", "acc", "insula", "dlpfc", "ventral_striatum_proxy", "amygdala"]:
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
        dietary_restriction=0.80,
        interpersonal_stress=0.60,
        negative_mood=0.75,
        palatable_food_cues=0.85,
        recovery_support=0.20,
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
    # print(model.assign_mni_point((-10, 12, -8)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("insula").to_string(index=False))
    # print(model.suggest_regions("accumbens").to_string(index=False))
