from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Autism Spectrum Disorder-oriented panel:
# - synaptogenesis / pruning
# - excitatory-inhibitory balance
# - social-brain development
# - dopamine / reward-regulation of repetitive behavior
#
# These genes are hypothesis-driven proxies for the chapter's themes.
DEFAULT_GENE_PANEL = [
    "SHANK3",
    "NRXN1",
    "NLGN3",
    "NLGN4X",
    "CNTNAP2",
    "CHD8",
    "SCN2A",
    "RELN",
    "GABRB3",
    "SLC6A1",
    "GRIN2B",
    "DRD2",
    "SLC6A4",
    "OXTR",
    "BDNF",
]


class AutismSpectrumDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Autism Spectrum Disorder (ASD).

    What it does:
      1) Resolves chapter-relevant anatomy to Julich regions when possible.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Simulates social-brain variation, sensory load, routine regulation,
         and social-communication differences.

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

        # Functional chapter concepts mapped to Julich-friendly search terms.
        # Fusiform and STS are intentionally proxies rather than exact functional ROIs.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "amygdala",
            ],
            "fusiform_face_proxy": [
                "Area FG2 (FusG) left",
                "Area FG1 (FusG) left",
                "fusiform",
                "FG2",
                "FG1",
            ],
            "sts_proxy": [
                "superior temporal sulcus",
                "STS",
                "superior temporal",
                "TE3",
                "TE 3",
            ],
            "mpfc": [
                "Area p32 (pACC) left",
                "Area s32 (sACC) left",
                "pACC",
                "sACC",
                "anterior cingulate",
                "medial prefrontal",
            ],
            "dlpfc": [
                "Area 8v1 (MFG) left",
                "Area 8v2 (MFG) left",
                "Area 8d1 (SFG) left",
                "middle frontal",
                "superior frontal",
                "dorsolateral prefrontal",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_neurodevelopmental_variation": "Inborn neurodevelopmental variation shaping autistic development",
            "psychosocial_context_quality": "Supportive psychosocial context that can buffer expression of difficulty",
            "sensory_social_demand": "Environmental sensory and social complexity load",
            "predictability_support": "Predictable routines and structured support",
            "pharmacologic_modulation": "Supportive neurochemical modulation of arousal or repetitive behavior",
        }

        self.latent_nodes: Dict[str, str] = {
            "synaptogenesis_pruning_variation": "Variation in synapse formation and refinement",
            "excitatory_inhibitory_imbalance": "Atypical balance of excitatory and inhibitory signaling",
            "social_motivation_difference": "Altered foundational motivation for social engagement",
            "dopamine_reward_regulation": "Reward-circuit contribution to routines and repetitive behavior",
            "social_brain_connectivity": "Altered co-activation / interdependence among social-brain nodes",
            "sensory_hyperreactivity": "Heightened sensory and arousal responsivity",
            "theory_of_mind_burden": "Difficulty with mentalizing / attributing mental states",
            "executive_inflexibility": "Reduced cognitive flexibility and preference for predictability",
            "routine_self_regulation": "Use of routines/stimming to regulate internal arousal",
        }

        self.symptom_nodes: Dict[str, str] = {
            "social_communication_difference": "Differences in interpreting and responding to social information",
            "face_processing_difference": "Reduced social salience / difference in face processing",
            "dynamic_social_cue_difficulty": "Difficulty interpreting gaze, gesture, and biological motion",
            "mentalizing_difficulty": "Difficulty inferring the mental states of self and others",
            "repetitive_restricted_behavior": "Restricted interests, routines, or stereotyped movements",
            "cognitive_inflexibility": "Reduced flexibility and preference for sameness",
            "social_anxiety_arousal": "Heightened anxiety/arousal in socially demanding contexts",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_neurodevelopmental_variation",
                "target": "synaptogenesis_pruning_variation",
                "relation": "biases the formation and refinement of neural connections",
                "autism_change": "increased susceptibility",
            },
            {
                "source": "genetic_neurodevelopmental_variation",
                "target": "excitatory_inhibitory_imbalance",
                "relation": "raises vulnerability for atypical synaptic signaling balance",
                "autism_change": "increased susceptibility",
            },
            {
                "source": "genetic_neurodevelopmental_variation",
                "target": "social_motivation_difference",
                "relation": "biases early social-engagement trajectory",
                "autism_change": "increased susceptibility",
            },
            {
                "source": "psychosocial_context_quality",
                "target": "social_brain_connectivity",
                "relation": "can support better functional coordination within social systems",
                "autism_change": "protective",
            },
            {
                "source": "psychosocial_context_quality",
                "target": "social_communication_difference",
                "relation": "can buffer practical impact of social-processing differences",
                "autism_change": "protective",
            },
            {
                "source": "sensory_social_demand",
                "target": "sensory_hyperreactivity",
                "relation": "loads sensory and internal-state regulation",
                "autism_change": "increased",
            },
            {
                "source": "sensory_social_demand",
                "target": "social_anxiety_arousal",
                "relation": "raises arousal in complex social environments",
                "autism_change": "increased",
            },
            {
                "source": "predictability_support",
                "target": "executive_inflexibility",
                "relation": "reduces burden from unpredictability and transition demand",
                "autism_change": "protective",
            },
            {
                "source": "predictability_support",
                "target": "repetitive_restricted_behavior",
                "relation": "can reduce need for rigid self-generated regulation",
                "autism_change": "protective",
            },
            {
                "source": "pharmacologic_modulation",
                "target": "dopamine_reward_regulation",
                "relation": "can modulate reward-linked repetitive behavior",
                "autism_change": "protective",
            },
            {
                "source": "pharmacologic_modulation",
                "target": "sensory_hyperreactivity",
                "relation": "can reduce arousal-linked behavioral amplification",
                "autism_change": "protective",
            },
            {
                "source": "synaptogenesis_pruning_variation",
                "target": "social_brain_connectivity",
                "relation": "alters long-range coordination among social-brain nodes",
                "autism_change": "altered connectivity",
            },
            {
                "source": "excitatory_inhibitory_imbalance",
                "target": "sensory_hyperreactivity",
                "relation": "supports hypersensitivity and arousal dysregulation",
                "autism_change": "increased",
            },
            {
                "source": "social_motivation_difference",
                "target": "fusiform_face_proxy",
                "relation": "reduces spontaneous prioritization of faces as salient input",
                "autism_change": "reduced salience",
            },
            {
                "source": "social_motivation_difference",
                "target": "social_communication_difference",
                "relation": "changes developmental reinforcement of social learning",
                "autism_change": "increased difference",
            },
            {
                "source": "dopamine_reward_regulation",
                "target": "routine_self_regulation",
                "relation": "supports reward-linked repetition and routine stabilization",
                "autism_change": "increased",
            },
            {
                "source": "dopamine_reward_regulation",
                "target": "repetitive_restricted_behavior",
                "relation": "can reinforce routines and stereotyped behaviors",
                "autism_change": "increased",
            },
            {
                "source": "social_brain_connectivity",
                "target": "amygdala",
                "relation": "alters coordination of social-emotional salience processing",
                "autism_change": "altered function",
            },
            {
                "source": "social_brain_connectivity",
                "target": "fusiform_face_proxy",
                "relation": "alters integration of face-related social input",
                "autism_change": "altered function",
            },
            {
                "source": "social_brain_connectivity",
                "target": "sts_proxy",
                "relation": "alters integration of dynamic social cues",
                "autism_change": "altered function",
            },
            {
                "source": "social_brain_connectivity",
                "target": "mpfc",
                "relation": "alters social inference / mentalizing coordination",
                "autism_change": "altered function",
            },
            {
                "source": "social_brain_connectivity",
                "target": "dlpfc",
                "relation": "alters cognitive control and flexible adaptation",
                "autism_change": "altered function",
            },
            {
                "source": "sensory_hyperreactivity",
                "target": "amygdala",
                "relation": "amplifies arousal to socially or sensorily salient input",
                "autism_change": "increased dysregulation",
            },
            {
                "source": "sensory_hyperreactivity",
                "target": "repetitive_restricted_behavior",
                "relation": "increases reliance on repetitive behaviors for regulation",
                "autism_change": "increased",
            },
            {
                "source": "sensory_hyperreactivity",
                "target": "social_anxiety_arousal",
                "relation": "raises internal arousal in demanding environments",
                "autism_change": "increased",
            },
            {
                "source": "theory_of_mind_burden",
                "target": "mpfc",
                "relation": "loads mentalizing-related medial prefrontal processing",
                "autism_change": "increased burden",
            },
            {
                "source": "theory_of_mind_burden",
                "target": "mentalizing_difficulty",
                "relation": "increases difficulty attributing mental states",
                "autism_change": "increased",
            },
            {
                "source": "executive_inflexibility",
                "target": "dlpfc",
                "relation": "loads cognitive flexibility and planning systems",
                "autism_change": "increased burden",
            },
            {
                "source": "executive_inflexibility",
                "target": "cognitive_inflexibility",
                "relation": "increases difficulty shifting strategies and routines",
                "autism_change": "increased",
            },
            {
                "source": "routine_self_regulation",
                "target": "repetitive_restricted_behavior",
                "relation": "supports routines/stimming as internal-state regulation",
                "autism_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "social_anxiety_arousal",
                "relation": "alters salience and anxiety responses to social cues",
                "autism_change": "increased",
            },
            {
                "source": "fusiform_face_proxy",
                "target": "face_processing_difference",
                "relation": "alters face salience and recognition processing",
                "autism_change": "increased difference",
            },
            {
                "source": "sts_proxy",
                "target": "dynamic_social_cue_difficulty",
                "relation": "alters interpretation of gaze, gesture, and biological motion",
                "autism_change": "increased",
            },
            {
                "source": "mpfc",
                "target": "mentalizing_difficulty",
                "relation": "alters theory-of-mind-related processing",
                "autism_change": "increased",
            },
            {
                "source": "dlpfc",
                "target": "cognitive_inflexibility",
                "relation": "alters planning and flexible response selection",
                "autism_change": "increased",
            },
            {
                "source": "face_processing_difference",
                "target": "social_communication_difference",
                "relation": "changes access to a major source of social information",
                "autism_change": "increased difference",
            },
            {
                "source": "dynamic_social_cue_difficulty",
                "target": "social_communication_difference",
                "relation": "reduces interpretation of dynamic social signals",
                "autism_change": "increased difference",
            },
            {
                "source": "mentalizing_difficulty",
                "target": "social_communication_difference",
                "relation": "reduces inference about others' intentions and states",
                "autism_change": "increased difference",
            },
            {
                "source": "social_anxiety_arousal",
                "target": "repetitive_restricted_behavior",
                "relation": "increases reliance on routines or stimming for regulation",
                "autism_change": "increased",
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
        for r in matches:
            parc_name = getattr(getattr(r, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower():
                out.append(r)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int]:
        """
        Prefer left-sided, more specific names over generic/right-sided parents.
        Lower tuple is better.
        """
        name = self._name_of(region).lower()
        right_penalty = 1 if "right" in name else 0
        left_bonus = 0 if "left" in name else 1
        generic_penalty = 1 if name in {
            "amygdala",
            "insula",
            "superior temporal",
            "prefrontal cortex",
            "frontal lobe",
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
        Helper for tuning proxy searches against the atlas.
        Useful for refining fusiform/STS/mPFC candidates.
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
            # Representative subject-level matrix, matching the connectivity-matrix examples.
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
            "fusiform_face_proxy",
            "sts_proxy",
            "mpfc",
            "dlpfc",
        ),
    ) -> pd.DataFrame:
        """
        Extract a representative structural-connectivity submatrix for the autism social-brain circuit.
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
                    "description": "Atlas-backed autism social-brain node",
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
        genetic_neurodevelopmental_variation: float,
        psychosocial_context_quality: float,
        sensory_social_demand: float,
        predictability_support: float,
        pharmacologic_modulation: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.

        Higher values mean stronger difference / burden,
        while psychosocial_context_quality, predictability_support,
        and pharmacologic_modulation are protective when high.
        """
        g = self._clip01(genetic_neurodevelopmental_variation)
        p = self._clip01(psychosocial_context_quality)
        s = self._clip01(sensory_social_demand)
        r = self._clip01(predictability_support)
        m = self._clip01(pharmacologic_modulation)

        # Latent developmental biology
        synaptogenesis_pruning_variation = self._clip01(0.45 * g - 0.10 * p)
        excitatory_inhibitory_imbalance = self._clip01(0.35 * g + 0.20 * s - 0.10 * m)
        social_motivation_difference = self._clip01(0.30 * g + 0.10 * s - 0.10 * p)
        social_brain_connectivity = self._clip01(
            0.35 * synaptogenesis_pruning_variation
            + 0.20 * social_motivation_difference
            + 0.10 * s
            - 0.25 * p
        )
        sensory_hyperreactivity = self._clip01(
            0.35 * excitatory_inhibitory_imbalance
            + 0.25 * s
            - 0.10 * r
            - 0.10 * m
        )
        dopamine_reward_regulation = self._clip01(
            0.25 * g + 0.20 * s + 0.15 * social_motivation_difference - 0.15 * m
        )
        theory_of_mind_burden = self._clip01(
            0.35 * social_brain_connectivity + 0.15 * social_motivation_difference - 0.10 * p
        )
        executive_inflexibility = self._clip01(
            0.25 * social_brain_connectivity + 0.20 * sensory_hyperreactivity - 0.20 * r
        )
        routine_self_regulation = self._clip01(
            0.35 * sensory_hyperreactivity
            + 0.20 * dopamine_reward_regulation
            - 0.15 * r
            - 0.10 * m
        )

        # Regional state proxies
        amygdala = self._clip01(
            0.30 * sensory_hyperreactivity + 0.25 * social_brain_connectivity + 0.10 * s
        )
        fusiform_face_proxy = self._clip01(
            0.35 * social_brain_connectivity + 0.20 * social_motivation_difference
        )
        sts_proxy = self._clip01(
            0.35 * social_brain_connectivity + 0.20 * sensory_hyperreactivity + 0.10 * s
        )
        mpfc = self._clip01(
            0.35 * theory_of_mind_burden + 0.20 * social_brain_connectivity - 0.10 * p
        )
        dlpfc = self._clip01(
            0.35 * executive_inflexibility + 0.15 * social_brain_connectivity - 0.10 * r
        )

        # Symptoms / functional expressions
        face_processing_difference = self._clip01(
            0.45 * fusiform_face_proxy + 0.15 * social_motivation_difference
        )
        dynamic_social_cue_difficulty = self._clip01(
            0.45 * sts_proxy + 0.15 * sensory_hyperreactivity
        )
        mentalizing_difficulty = self._clip01(
            0.45 * mpfc + 0.25 * theory_of_mind_burden
        )
        social_anxiety_arousal = self._clip01(
            0.35 * amygdala + 0.25 * sensory_hyperreactivity + 0.10 * s
        )
        cognitive_inflexibility = self._clip01(
            0.45 * dlpfc + 0.25 * executive_inflexibility
        )
        repetitive_restricted_behavior = self._clip01(
            0.30 * routine_self_regulation
            + 0.25 * cognitive_inflexibility
            + 0.15 * social_anxiety_arousal
            + 0.15 * sensory_hyperreactivity
            - 0.20 * r
            - 0.10 * m
        )
        social_communication_difference = self._clip01(
            0.25 * face_processing_difference
            + 0.25 * dynamic_social_cue_difficulty
            + 0.25 * mentalizing_difficulty
            + 0.10 * social_motivation_difference
            + 0.10 * social_anxiety_arousal
        )

        return {
            "inputs": pd.Series(
                {
                    "genetic_neurodevelopmental_variation": g,
                    "psychosocial_context_quality": p,
                    "sensory_social_demand": s,
                    "predictability_support": r,
                    "pharmacologic_modulation": m,
                }
            ),
            "latents": pd.Series(
                {
                    "social_brain_connectivity": social_brain_connectivity,
                    "sensory_hyperreactivity": sensory_hyperreactivity,
                    "theory_of_mind_burden": theory_of_mind_burden,
                    "routine_self_regulation": routine_self_regulation,
                    "executive_inflexibility": executive_inflexibility,
                    "dopamine_reward_regulation": dopamine_reward_regulation,
                    "synaptogenesis_pruning_variation": synaptogenesis_pruning_variation,
                    "excitatory_inhibitory_imbalance": excitatory_inhibitory_imbalance,
                    "social_motivation_difference": social_motivation_difference,
                }
            ).sort_values(ascending=False),
            "regional_state": pd.Series(
                {
                    "amygdala": amygdala,
                    "fusiform_face_proxy": fusiform_face_proxy,
                    "sts_proxy": sts_proxy,
                    "mpfc": mpfc,
                    "dlpfc": dlpfc,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "social_communication_difference": social_communication_difference,
                    "repetitive_restricted_behavior": repetitive_restricted_behavior,
                    "face_processing_difference": face_processing_difference,
                    "dynamic_social_cue_difficulty": dynamic_social_cue_difficulty,
                    "mentalizing_difficulty": mentalizing_difficulty,
                    "cognitive_inflexibility": cognitive_inflexibility,
                    "social_anxiety_arousal": social_anxiety_arousal,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "social_brain_difference_profile": self._clip01(
                        0.45 * social_communication_difference
                        + 0.25 * mentalizing_difficulty
                        + 0.15 * face_processing_difference
                    ),
                    "sensory_routine_regulation_profile": self._clip01(
                        0.40 * repetitive_restricted_behavior
                        + 0.30 * sensory_hyperreactivity
                        + 0.20 * cognitive_inflexibility
                    ),
                    "anxious_social_arousal_profile": self._clip01(
                        0.45 * social_anxiety_arousal
                        + 0.25 * amygdala
                        + 0.20 * social_communication_difference
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.

        Example:
            model.assign_mni_point((-38, -52, -18)).head(10)
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
    model = AutismSpectrumDisorderModel()

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
        bundle["edges"][["source", "target", "relation", "autism_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit connectivity submatrix available.")

    for key in ["amygdala", "fusiform_face_proxy", "sts_proxy", "mpfc", "dlpfc"]:
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
        genetic_neurodevelopmental_variation=0.80,
        psychosocial_context_quality=0.35,
        sensory_social_demand=0.75,
        predictability_support=0.25,
        pharmacologic_modulation=0.15,
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
    # print(model.assign_mni_point((-38, -52, -18)).head(10))

    # Example proxy tuning:
    # print(model.suggest_regions("fusiform").to_string(index=False))
    # print(model.suggest_regions("superior temporal").to_string(index=False))
    # print(model.suggest_regions("cingulate").to_string(index=False))
