from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


# Hypothesis-driven gene panel for agoraphobia:
# - HPA-axis / stress feedback
# - serotonergic / noradrenergic / GABAergic fear regulation
# - monoamine metabolism
DEFAULT_GENE_PANEL = [
    "NR3C1",   # glucocorticoid receptor
    "FKBP5",   # stress responsivity / HPA-axis tuning
    "CRHR1",   # CRH signalling
    "SLC6A4",  # serotonin transporter
    "HTR1A",   # serotonin receptor
    "HTR2A",   # serotonin receptor
    "SLC6A2",  # norepinephrine transporter
    "ADRA2A",  # alpha-2A adrenergic receptor
    "COMT",    # catecholamine metabolism
    "MAOA",    # monoamine oxidase A
    "GABRA2",  # GABA-A receptor subunit
    "GABRB2",  # GABA-A receptor subunit
]


class AgoraphobiaModel:
    """
    Atlas-grounded mechanistic scaffold for Agoraphobia using siibra.

    What it does:
      1) Resolves agoraphobia-relevant circuit nodes to Julich regions.
      2) Pulls receptor, gene-expression, and structural-connectivity evidence.
      3) Builds a node/edge graph from the chapter text.
      4) Provides a simple simulator for panic-linked avoidance dynamics.

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
        self.parcellation = self.atlas.parcellations.get(parcellation_spec)
        self.space = siibra.spaces.get(space_spec)
        self.connectivity_cohort = connectivity_cohort

        # Practical Julich anchors for the chapter's fear network.
        # vmPFC is represented by sACC/pACC with OFC/frontopolar fallbacks,
        # because Julich uses cytoarchitectonic parcels instead of the
        # broader functional label "vmPFC".
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "SF (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "DG (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "hippocampus",
            ],
            "insula": [
                "Area Id2 (Insula) left",
                "Area Id3 (Insula) left",
                "Area Id4 (Insula) left",
                "insula",
            ],
            "dacc": [
                "Area 33 (ACC) left",
                "Area p32 (pACC) left",
                "ACC",
            ],
            "vmpfc": [
                "Area s32 (sACC) left",
                "Area p32 (pACC) left",
                "Area Fo4 (OFC) left",
                "Area Fo3 (OFC) left",
                "Area Fp2 (FPole) left",
                "Area Fp1 (FPole) left",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "panic_history": "Past panic attacks and panic conditioning",
            "temperamental_vulnerability": "Behavioral inhibition, neuroticism, and anxiety sensitivity",
            "early_life_adversity": "Early stress, insecure attachment, or parental anxiety exposure",
            "feared_environment_cues": "Crowds, open spaces, transport, or escape-difficult contexts",
            "treatment_engagement": "Protective engagement with pharmacotherapy and psychotherapy",
        }

        self.latent_nodes: Dict[str, str] = {
            "epigenetics": "Experience-dependent molecular calibration of stress responsivity",
            "hpa_axis": "Stress-hormone regulation and glucocorticoid feedback",
            "autonomic_reactivity": "Sympathetic arousal and bodily alarm responsivity",
            "interoceptive_salience": "Heightened monitoring of bodily sensations",
            "threat_prediction": "Expectation that contexts or sensations signal danger",
            "avoidance_learning": "Reinforced escape and safety-seeking behavior",
            "neurotransmitter_dysregulation": "Broad anxiety-circuit transmitter imbalance",
        }

        self.symptom_nodes: Dict[str, str] = {
            "panic_surges": "Acute panic-like autonomic episodes",
            "anticipatory_anxiety": "Persistent expectation of panic or loss of control",
            "escape_fear": "Fear of situations where escape or help may be difficult",
            "agoraphobic_avoidance": "Avoidance of feared public situations",
            "home_confinement": "Severe behavioral restriction to home or safe zones",
            "social_isolation": "Loneliness and reduced participation in the outside world",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "panic_history",
                "target": "interoceptive_salience",
                "relation": "makes bodily sensations more fear-relevant",
                "ago_change": "increased",
            },
            {
                "source": "panic_history",
                "target": "avoidance_learning",
                "relation": "reinforces escape and safety behaviors",
                "ago_change": "increased",
            },
            {
                "source": "temperamental_vulnerability",
                "target": "autonomic_reactivity",
                "relation": "raises baseline nervous-system reactivity",
                "ago_change": "increased susceptibility",
            },
            {
                "source": "temperamental_vulnerability",
                "target": "interoceptive_salience",
                "relation": "raises fear of anxiety sensations",
                "ago_change": "increased susceptibility",
            },
            {
                "source": "temperamental_vulnerability",
                "target": "threat_prediction",
                "relation": "biases ambiguous situations toward danger expectancy",
                "ago_change": "increased susceptibility",
            },
            {
                "source": "early_life_adversity",
                "target": "epigenetics",
                "relation": "can leave long-lasting stress-related molecular changes",
                "ago_change": "increased",
            },
            {
                "source": "epigenetics",
                "target": "hpa_axis",
                "relation": "biases stress-response feedback and responsivity",
                "ago_change": "dysregulated",
            },
            {
                "source": "feared_environment_cues",
                "target": "amygdala",
                "relation": "triggers fear-network activation",
                "ago_change": "increased",
            },
            {
                "source": "feared_environment_cues",
                "target": "insula",
                "relation": "amplifies bodily-state monitoring in feared contexts",
                "ago_change": "increased",
            },
            {
                "source": "feared_environment_cues",
                "target": "dacc",
                "relation": "increases anticipation and threat monitoring",
                "ago_change": "increased",
            },
            {
                "source": "treatment_engagement",
                "target": "vmpfc",
                "relation": "supports regulatory control and relearning",
                "ago_change": "protective",
            },
            {
                "source": "treatment_engagement",
                "target": "avoidance_learning",
                "relation": "helps weaken maladaptive avoidance loops",
                "ago_change": "protective",
            },
            {
                "source": "neurotransmitter_dysregulation",
                "target": "autonomic_reactivity",
                "relation": "destabilizes anxiety and panic circuitry",
                "ago_change": "increased",
            },
            {
                "source": "hpa_axis",
                "target": "hippocampus",
                "relation": "impairs contextual safety learning",
                "ago_change": "reduced function",
            },
            {
                "source": "hpa_axis",
                "target": "vmpfc",
                "relation": "weakens top-down emotional regulation",
                "ago_change": "reduced function",
            },
            {
                "source": "autonomic_reactivity",
                "target": "panic_surges",
                "relation": "drives bodily panic symptoms",
                "ago_change": "increased",
            },
            {
                "source": "interoceptive_salience",
                "target": "insula",
                "relation": "loads interoceptive fear processing",
                "ago_change": "increased",
            },
            {
                "source": "threat_prediction",
                "target": "dacc",
                "relation": "supports anticipatory threat monitoring",
                "ago_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "panic_surges",
                "relation": "amplifies alarm output",
                "ago_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "agoraphobic_avoidance",
                "relation": "promotes fear-driven behavioral withdrawal",
                "ago_change": "increased",
            },
            {
                "source": "insula",
                "target": "panic_surges",
                "relation": "amplifies perceived bodily catastrophe",
                "ago_change": "increased",
            },
            {
                "source": "insula",
                "target": "escape_fear",
                "relation": "links bodily sensations to dangerous contexts",
                "ago_change": "increased",
            },
            {
                "source": "dacc",
                "target": "anticipatory_anxiety",
                "relation": "supports hypervigilant anticipation of threat",
                "ago_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "escape_fear",
                "relation": "normally helps distinguish safe from unsafe contexts",
                "ago_change": "reduced contextualization",
            },
            {
                "source": "vmpfc",
                "target": "amygdala",
                "relation": "normally exerts top-down inhibition",
                "ago_change": "reduced inhibition",
            },
            {
                "source": "vmpfc",
                "target": "avoidance_learning",
                "relation": "normally supports extinction and flexible relearning",
                "ago_change": "reduced extinction",
            },
            {
                "source": "panic_surges",
                "target": "anticipatory_anxiety",
                "relation": "make future attacks more expected",
                "ago_change": "increased",
            },
            {
                "source": "panic_surges",
                "target": "agoraphobic_avoidance",
                "relation": "drive avoidance of panic-linked situations",
                "ago_change": "increased",
            },
            {
                "source": "escape_fear",
                "target": "agoraphobic_avoidance",
                "relation": "promotes avoidance of escape-difficult contexts",
                "ago_change": "increased",
            },
            {
                "source": "agoraphobic_avoidance",
                "target": "home_confinement",
                "relation": "shrinks behavioral range toward safe zones",
                "ago_change": "increased",
            },
            {
                "source": "home_confinement",
                "target": "social_isolation",
                "relation": "limits participation and contact with others",
                "ago_change": "increased",
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
                continue
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        """
        Helper for tuning region vocabulary against the atlas.
        Useful if you want to refine the insula or vmPFC proxy.
        """
        rows: List[Dict[str, Any]] = []
        seen = set()
        try:
            matches = self.atlas.find_regions(
                keyword,
                all_versions=False,
                filter_children=True,
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
                features = siibra.features.get(concept, modality, **kwargs)
            return list(features) if features else []
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
            data = compound.data
            self._connectivity_matrix = (
                data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame()
            )
        except Exception:
            self._connectivity_matrix = pd.DataFrame()

        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        exact = [x for x in labels if self._name_of(x) == region.name]
        if exact:
            return exact[0]

        region_name = region.name.lower()
        fuzzy = [
            x for x in labels
            if region_name in self._name_of(x).lower()
            or self._name_of(x).lower() in region_name
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
            df = (
                series.sort_values(ascending=False)
                .reset_index()
                .rename(columns={"index": "connected_region", 0: "value"})
            )
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
            "insula",
            "dacc",
            "vmpfc",
        ),
    ) -> pd.DataFrame:
        """
        Extract the structural connectivity submatrix for the agoraphobia circuit.
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
                    "description": "Atlas-backed agoraphobia fear-network node",
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
        panic_history: float,
        temperamental_vulnerability: float,
        early_life_adversity: float,
        feared_environment_cues: float,
        treatment_engagement: float,
    ) -> Dict[str, pd.Series]:
        """
        Simple normalized 0..1 simulator.
        Higher values mean more dysregulation / worse symptom burden,
        except treatment_engagement which is protective.
        """
        p = self._clip01(panic_history)
        t = self._clip01(temperamental_vulnerability)
        e = self._clip01(early_life_adversity)
        c = self._clip01(feared_environment_cues)
        r = self._clip01(treatment_engagement)

        epigenetics = self._clip01(0.50 * e + 0.10 * t - 0.10 * r)
        hpa_axis = self._clip01(0.35 * e + 0.25 * t + 0.20 * epigenetics - 0.10 * r)
        neurotransmitter_dysregulation = self._clip01(
            0.25 * p + 0.20 * hpa_axis + 0.15 * t - 0.20 * r
        )
        autonomic_reactivity = self._clip01(
            0.35 * p
            + 0.25 * t
            + 0.20 * neurotransmitter_dysregulation
            + 0.10 * hpa_axis
            - 0.15 * r
        )
        interoceptive_salience = self._clip01(
            0.35 * p + 0.35 * t + 0.20 * autonomic_reactivity - 0.10 * r
        )
        threat_prediction = self._clip01(
            0.35 * c + 0.25 * t + 0.15 * p + 0.10 * autonomic_reactivity
        )

        amygdala = self._clip01(
            0.35 * c + 0.25 * autonomic_reactivity + 0.20 * threat_prediction
        )
        insula = self._clip01(
            0.40 * interoceptive_salience + 0.25 * autonomic_reactivity + 0.15 * c
        )
        dacc = self._clip01(
            0.35 * threat_prediction + 0.25 * autonomic_reactivity + 0.15 * c
        )
        hippocampus = self._clip01(0.30 * hpa_axis + 0.20 * epigenetics + 0.10 * p)
        vmpfc = self._clip01(
            0.35 * hpa_axis + 0.20 * epigenetics + 0.15 * amygdala - 0.25 * r
        )

        panic_surges = self._clip01(
            0.35 * autonomic_reactivity + 0.25 * insula + 0.20 * amygdala + 0.10 * dacc
        )
        anticipatory_anxiety = self._clip01(
            0.30 * dacc + 0.25 * amygdala + 0.20 * panic_surges + 0.15 * threat_prediction
        )
        escape_fear = self._clip01(
            0.35 * insula + 0.20 * panic_surges + 0.20 * anticipatory_anxiety + 0.15 * hippocampus
        )
        avoidance_learning = self._clip01(
            0.30 * panic_surges
            + 0.25 * escape_fear
            + 0.20 * amygdala
            + 0.15 * interoceptive_salience
            + 0.10 * vmpfc
            - 0.25 * r
        )
        agoraphobic_avoidance = self._clip01(
            0.35 * escape_fear
            + 0.25 * panic_surges
            + 0.20 * anticipatory_anxiety
            + 0.15 * avoidance_learning
            - 0.20 * r
        )
        home_confinement = self._clip01(0.55 * agoraphobic_avoidance + 0.20 * anticipatory_anxiety)
        social_isolation = self._clip01(0.55 * home_confinement + 0.20 * agoraphobic_avoidance)

        return {
            "inputs": pd.Series(
                {
                    "panic_history": p,
                    "temperamental_vulnerability": t,
                    "early_life_adversity": e,
                    "feared_environment_cues": c,
                    "treatment_engagement": r,
                }
            ),
            "latents": pd.Series(
                {
                    "epigenetics": epigenetics,
                    "hpa_axis": hpa_axis,
                    "neurotransmitter_dysregulation": neurotransmitter_dysregulation,
                    "autonomic_reactivity": autonomic_reactivity,
                    "interoceptive_salience": interoceptive_salience,
                    "threat_prediction": threat_prediction,
                    "avoidance_learning": avoidance_learning,
                }
            ).sort_values(ascending=False),
            "region_dysfunction": pd.Series(
                {
                    "amygdala": amygdala,
                    "insula": insula,
                    "dacc": dacc,
                    "vmpfc": vmpfc,
                    "hippocampus": hippocampus,
                }
            ).sort_values(ascending=False),
            "symptoms": pd.Series(
                {
                    "agoraphobic_avoidance": agoraphobic_avoidance,
                    "anticipatory_anxiety": anticipatory_anxiety,
                    "panic_surges": panic_surges,
                    "escape_fear": escape_fear,
                    "home_confinement": home_confinement,
                    "social_isolation": social_isolation,
                }
            ).sort_values(ascending=False),
            "phenotypes": pd.Series(
                {
                    "panic_linked_agoraphobia": self._clip01(
                        0.45 * panic_surges + 0.30 * escape_fear + 0.25 * agoraphobic_avoidance
                    ),
                    "anticipatory_avoidant_agoraphobia": self._clip01(
                        0.40 * anticipatory_anxiety
                        + 0.35 * agoraphobic_avoidance
                        + 0.25 * home_confinement
                    ),
                    "isolation_severity": self._clip01(
                        0.55 * home_confinement + 0.45 * social_isolation
                    ),
                }
            ).sort_values(ascending=False),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Map an MNI coordinate into probabilistic Julich assignments.
        Example:
            model.assign_mni_point((-30, 18, -12)).head(10)
        """
        if self._pmap is None:
            with siibra.QUIET:
                self._pmap = siibra.get_map(
                    parcellation=self.parcellation,
                    space="mni152",
                    maptype="statistical",
                )

        point = siibra.Point(tuple(xyz), space="mni152")
        with siibra.QUIET:
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments

    def region_mask(self, node_key: str):
        """
        Fetch a binary mask for a resolved region node in self.space.
        """
        region = self.region_objects[node_key]
        return region.fetch_regional_map(self.space, maptype="labelled")


if __name__ == "__main__":
    model = AgoraphobiaModel()
    bundle = model.build()

    print("\n=== NODES ===")
    print(
        bundle["nodes"][
            ["key", "node_type", "atlas_region", "centroid_mni", "feature_summary"]
        ].to_string(index=False)
    )

    print("\n=== EDGES ===")
    print(
        bundle["edges"][["source", "target", "relation", "ago_change"]].to_string(index=False)
    )

    print("\n=== CIRCUIT CONNECTIVITY SUBMATRIX ===")
    if not bundle["circuit_connectivity"].empty:
        print(bundle["circuit_connectivity"].to_string())
    else:
        print("No circuit submatrix available.")

    for key in ["amygdala", "insula", "dacc", "vmpfc"]:
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

    sim = model.simulate(
        panic_history=0.80,
        temperamental_vulnerability=0.70,
        early_life_adversity=0.45,
        feared_environment_cues=0.85,
        treatment_engagement=0.20,
    )

    print("\n=== LATENT BIOLOGY ===")
    print(sim["latents"].to_string())

    print("\n=== REGION DYSFUNCTION ===")
    print(sim["region_dysfunction"].to_string())

    print("\n=== SYMPTOMS ===")
    print(sim["symptoms"].to_string())

    print("\n=== PHENOTYPES ===")
    print(sim["phenotypes"].to_string())

    # Example literature coordinate assignment:
    # print(model.assign_mni_point((-30, 18, -12)).head(10))

    # Example for tuning insula / vmPFC proxies:
    # print(model.suggest_regions("insula").to_string(index=False))
    # print(model.suggest_regions("cingulate").to_string(index=False))
