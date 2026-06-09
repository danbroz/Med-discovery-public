
from __future__ import annotations

"""
Delusional Disorder siibra scaffold
==================================

Research scaffold that translates a narrative chapter on Delusional Disorder into
an atlas-grounded mechanistic model using siibra.

This script is intentionally conservative:
- it only atlas-anchors regions that are named or strongly implied by the chapter,
- it keeps dopamine / serotonin / polygenic liability as latent biology unless the
  chapter localizes them clearly,
- it uses proxy nodes when the chapter is systems-level,
- it tolerates missing receptor, gene, and connectivity features without crashing.

This is a research scaffold, not a diagnostic or treatment tool.
"""

import warnings
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import siibra
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This scaffold requires the 'siibra' package. "
        "Install it in your environment before running the script."
    ) from exc


DEFAULT_GENE_PANEL = [
    "DRD2",
    "SLC6A3",
    "COMT",
    "HTR2A",
    "SLC6A4",
    "BDNF",
    "CACNA1C",
    "GRIN2A",
    "ZNF804A",
]


class DelusionalDisorderModel:
    """
    Atlas-grounded mechanistic scaffold for Delusional Disorder.

    Chapter translation choices
    ---------------------------
    Inputs
        - psychosis-spectrum polygenic liability
        - affective polygenic liability
        - late-life structural vulnerability
        - focal neurological insult burden
        - stimulant / dopamine-excess load
        - current affective burden
        - D2 antagonist treatment
        - serotonergic adjuvant treatment

    Latent biology
        - dopaminergic aberrant salience
        - serotonergic affective modulation shift
        - fronto-striatal-limbic dysconnectivity
        - hippocampal context instability
        - limbic threat tagging
        - belief evaluation failure
        - body representation distortion

    Regions / proxies
        - pfc_belief_evaluation_proxy
        - temporal_association_proxy
        - hippocampus
        - amygdala
        - striatal_salience_proxy
        - parietal_body_representation_proxy

    Symptoms
        - fixed_false_belief
        - paranoid_interpretation
        - belief_rigidity
        - depressive_distress
        - somatic_delusional_intensity

    Notes
    -----
    The chapter names frontal, temporal, hippocampal, amygdalar, striatal, and
    parietal contributions, but not one definitive cytoarchitectonic parcel for
    all of them. Therefore several nodes are explicit proxies with ranked Julich
    candidates and safe unresolved fallbacks.
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
        self.parcellation = (
            self.atlas.get_parcellation(parcellation_spec)
            if hasattr(self.atlas, "get_parcellation")
            else self.atlas.parcellations.get(parcellation_spec)
        )
        self.space_spec = space_spec
        self.space = (
            self.atlas.get_space(space_spec)
            if hasattr(self.atlas, "get_space")
            else self.atlas.spaces.get(space_spec)
        )
        self.assignment_space = "mni152"
        self.connectivity_cohort = connectivity_cohort

        self.region_candidates: Dict[str, List[str]] = {
            "pfc_belief_evaluation_proxy": [
                "Area 45 (IFG) left",
                "Area 44 (IFG) left",
                "Area Fo3 (OFC) left",
                "Area Fo2 (OFC) left",
                "prefrontal cortex",
            ],
            "temporal_association_proxy": [
                "Area TE 3 (STG) left",
                "Area TE 2.2 (STG) left",
                "Area TE 2.1 (STG) left",
                "Area TPJ (STG/SMG) left",
                "superior temporal gyrus",
            ],
            "hippocampus": [
                "CA1 (Hippocampus) left",
                "CA3 (Hippocampus) left",
                "DG (Hippocampus) left",
                "HC-Subiculum (Hippocampus) left",
                "hippocampus",
            ],
            "amygdala": [
                "LB (Amygdala) left",
                "SF (Amygdala) left",
                "CM (Amygdala) left",
                "amygdala",
            ],
            "striatal_salience_proxy": [
                "nucleus accumbens",
                "caudate",
                "putamen",
                "striatum",
                "basal ganglia",
            ],
            "parietal_body_representation_proxy": [
                "Area PFt (IPL) left",
                "Area PF (IPL) left",
                "Area PGp (IPL) left",
                "Area 7A (SPL) left",
                "inferior parietal",
            ],
        }

        self.region_descriptions: Dict[str, str] = {
            "pfc_belief_evaluation_proxy": (
                "Conservative frontal-control proxy for the chapter's impaired belief "
                "evaluation and prefrontal dysfunction claims."
            ),
            "temporal_association_proxy": (
                "Proxy for temporal-lobe contributions to context integration and "
                "interpretive misattribution."
            ),
            "hippocampus": (
                "Atlas-backed memory-context node reflecting hippocampal vulnerability, "
                "especially in late-life and memory-impaired delusional presentations."
            ),
            "amygdala": (
                "Atlas-backed limbic salience / affective tagging node relevant to "
                "paranoid and emotionally colored delusional beliefs."
            ),
            "striatal_salience_proxy": (
                "Proxy for striatal dopaminergic salience generation implied by D2 "
                "antagonist response and stimulant-linked paranoia."
            ),
            "parietal_body_representation_proxy": (
                "Proxy for body-representation circuitry relevant to somatic delusions."
            ),
        }

        self.input_nodes: Dict[str, str] = {
            "psychosis_polygenic_load": (
                "Polygenic schizophrenia-spectrum liability that may predispose to "
                "psychosis-related phenotypes."
            ),
            "affective_polygenic_load": (
                "Heritable affective liability capturing the chapter's overlap with "
                "bipolar and depressive disorders."
            ),
            "late_life_structural_vulnerability": (
                "Age-associated or late-onset structural burden that can increase "
                "psychosis vulnerability in later life."
            ),
            "focal_neurological_insult": (
                "Burden from stroke, tumor, traumatic brain injury, hydrocephalus, or "
                "other lesion-level neurological disruption."
            ),
            "stimulant_dopamine_load": (
                "Dopamine-excess pressure, including stimulant-associated paranoid states."
            ),
            "affective_state_load": (
                "Current depressive or affective burden that can color delusional content "
                "and clinical distress."
            ),
            "d2_antagonist_treatment": (
                "Protective D2 receptor blockade from antipsychotic treatment."
            ),
            "serotonergic_adjuvant_treatment": (
                "Protective serotonergic support, such as atypical antipsychotic serotonin "
                "actions or SSRI augmentation."
            ),
        }

        self.latent_nodes: Dict[str, str] = {
            "dopaminergic_aberrant_salience": (
                "Excessive attribution of salience to internal or external cues via "
                "dopaminergic mechanisms."
            ),
            "serotonergic_affective_modulation_shift": (
                "Serotonergic modulation of mood, cognition, and affective coloring of "
                "belief content."
            ),
            "fronto_striatal_limbic_dysconnectivity": (
                "Disordered connectivity across frontal evaluative systems, striatal "
                "salience machinery, and limbic tagging circuits."
            ),
            "hippocampal_context_instability": (
                "Context-memory instability that can weaken coherent reality checking and "
                "episodic anchoring."
            ),
            "limbic_threat_tagging": (
                "Affective and threat-biased tagging of ambiguous events, relevant to "
                "paranoid interpretation."
            ),
            "belief_evaluation_failure": (
                "Reduced capacity to revise implausible beliefs once salience signals and "
                "context errors accumulate."
            ),
            "body_representation_distortion": (
                "Distorted interoceptive or body-schema processing that can support "
                "somatic delusional themes."
            ),
        }

        self.symptom_nodes: Dict[str, str] = {
            "fixed_false_belief": (
                "Stable delusional conviction as the core psychopathology."
            ),
            "paranoid_interpretation": (
                "Persecutory or suspicious interpretation of ambiguous events."
            ),
            "belief_rigidity": (
                "Resistance to belief revision once the delusional system is established."
            ),
            "depressive_distress": (
                "Mood-related distress or depressive symptoms comorbid with delusional "
                "states."
            ),
            "somatic_delusional_intensity": (
                "Intensity of body-focused false beliefs or somatic preoccupation."
            ),
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "psychosis_polygenic_load",
                "target": "dopaminergic_aberrant_salience",
                "relation": "raises vulnerability to psychosis-like salience misassignment",
                "dd_change": "increased",
            },
            {
                "source": "psychosis_polygenic_load",
                "target": "fronto_striatal_limbic_dysconnectivity",
                "relation": "adds distributed circuit liability across schizophrenia-spectrum systems",
                "dd_change": "increased",
            },
            {
                "source": "affective_polygenic_load",
                "target": "serotonergic_affective_modulation_shift",
                "relation": "supports mood-linked serotonergic dysregulation",
                "dd_change": "increased",
            },
            {
                "source": "late_life_structural_vulnerability",
                "target": "hippocampal_context_instability",
                "relation": "increases memory-context fragility in later life",
                "dd_change": "increased",
            },
            {
                "source": "late_life_structural_vulnerability",
                "target": "fronto_striatal_limbic_dysconnectivity",
                "relation": "adds structural burden to distributed psychosis-relevant circuits",
                "dd_change": "increased",
            },
            {
                "source": "focal_neurological_insult",
                "target": "fronto_striatal_limbic_dysconnectivity",
                "relation": "lesions can precipitate delusional syndromes through focal or network disruption",
                "dd_change": "increased",
            },
            {
                "source": "focal_neurological_insult",
                "target": "body_representation_distortion",
                "relation": "neurological damage can distort body representation and somatic belief content",
                "dd_change": "increased",
            },
            {
                "source": "stimulant_dopamine_load",
                "target": "dopaminergic_aberrant_salience",
                "relation": "dopamine excess promotes paranoid delusional ideation",
                "dd_change": "increased",
            },
            {
                "source": "affective_state_load",
                "target": "serotonergic_affective_modulation_shift",
                "relation": "current mood burden amplifies serotonergic-affective dysregulation",
                "dd_change": "increased",
            },
            {
                "source": "d2_antagonist_treatment",
                "target": "dopaminergic_aberrant_salience",
                "relation": "D2 blockade dampens psychosis-linked salience overload",
                "dd_change": "decreased",
            },
            {
                "source": "serotonergic_adjuvant_treatment",
                "target": "serotonergic_affective_modulation_shift",
                "relation": "serotonergic support reduces affective and cognitive coloring",
                "dd_change": "decreased",
            },
            {
                "source": "dopaminergic_aberrant_salience",
                "target": "striatal_salience_proxy",
                "relation": "drives salience-related striatal burden",
                "dd_change": "increased",
            },
            {
                "source": "fronto_striatal_limbic_dysconnectivity",
                "target": "pfc_belief_evaluation_proxy",
                "relation": "weakens frontal belief evaluation and cognitive control",
                "dd_change": "increased",
            },
            {
                "source": "fronto_striatal_limbic_dysconnectivity",
                "target": "temporal_association_proxy",
                "relation": "disrupts temporal-lobe integration of context and interpretation",
                "dd_change": "increased",
            },
            {
                "source": "hippocampal_context_instability",
                "target": "hippocampus",
                "relation": "loads the memory-context node with instability",
                "dd_change": "increased",
            },
            {
                "source": "limbic_threat_tagging",
                "target": "amygdala",
                "relation": "amplifies limbic affective tagging",
                "dd_change": "increased",
            },
            {
                "source": "body_representation_distortion",
                "target": "parietal_body_representation_proxy",
                "relation": "loads body-representation circuitry",
                "dd_change": "increased",
            },
            {
                "source": "serotonergic_affective_modulation_shift",
                "target": "limbic_threat_tagging",
                "relation": "colors salience with affective threat and emotional bias",
                "dd_change": "increased",
            },
            {
                "source": "dopaminergic_aberrant_salience",
                "target": "belief_evaluation_failure",
                "relation": "repeated salience errors destabilize flexible belief testing",
                "dd_change": "increased",
            },
            {
                "source": "hippocampal_context_instability",
                "target": "belief_evaluation_failure",
                "relation": "context-memory impairment weakens plausibility checks",
                "dd_change": "increased",
            },
            {
                "source": "belief_evaluation_failure",
                "target": "fixed_false_belief",
                "relation": "failure to revise implausible beliefs supports delusional conviction",
                "dd_change": "increased",
            },
            {
                "source": "striatal_salience_proxy",
                "target": "fixed_false_belief",
                "relation": "aberrant salience pushes neutral events into delusional meaning",
                "dd_change": "increased",
            },
            {
                "source": "amygdala",
                "target": "paranoid_interpretation",
                "relation": "affective threat tagging biases interpretation toward persecution",
                "dd_change": "increased",
            },
            {
                "source": "temporal_association_proxy",
                "target": "paranoid_interpretation",
                "relation": "misintegration of social or contextual cues promotes suspicious meaning-making",
                "dd_change": "increased",
            },
            {
                "source": "pfc_belief_evaluation_proxy",
                "target": "belief_rigidity",
                "relation": "poor evaluative flexibility hardens delusional beliefs",
                "dd_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "belief_rigidity",
                "relation": "memory-context disturbance helps sustain a self-consistent delusional system",
                "dd_change": "increased",
            },
            {
                "source": "serotonergic_affective_modulation_shift",
                "target": "depressive_distress",
                "relation": "mood dysregulation increases depressive comorbidity",
                "dd_change": "increased",
            },
            {
                "source": "parietal_body_representation_proxy",
                "target": "somatic_delusional_intensity",
                "relation": "body-schema distortion increases somatic delusional content",
                "dd_change": "increased",
            },
            {
                "source": "fixed_false_belief",
                "target": "belief_rigidity",
                "relation": "once formed, fixed beliefs become harder to update",
                "dd_change": "increased",
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
    def _quiet_context():
        return getattr(siibra, "QUIET", nullcontext())

    @staticmethod
    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 4)))

    @staticmethod
    def _name_of(obj: Any) -> str:
        return getattr(obj, "name", str(obj))

    @staticmethod
    def _norm(text: Any) -> str:
        return (
            str(text)
            .lower()
            .replace("area ", "")
            .replace("(ifg)", "")
            .replace("(ofc)", "")
            .replace("(stg)", "")
            .replace("(smg)", "")
            .replace("(ipl)", "")
            .replace("(spl)", "")
            .replace("(hippocampus)", "")
            .replace("(amygdala)", "")
            .replace("left", "l")
            .replace("right", "r")
            .replace("  ", " ")
            .strip()
        )

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
        if concept is None:
            return []
        for modality in modalities:
            try:
                with self._quiet_context():
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

        out = []
        for region in matches:
            parc_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" in str(parc_name).lower():
                out.append(region)
        return out

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {"amygdala", "hippocampus", "prefrontal cortex", "striatum"} else 0
        proxy_penalty = 1 if "component" in name else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

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
        rows = []
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
            rows.append({"name": row[0], "identifier": row[1], "parcellation": row[2]})
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
        centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        volume_mm3 = float(getattr(main, "volume", float("nan")))
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
            return (
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
        return df.reset_index(drop=True)

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats and self.region_objects:
            exemplar = next(iter(self.region_objects.values()))
            feats = self._safe_features_any(exemplar, self._modality_candidates("connectivity"))

        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next(
            (
                f
                for f in feats
                if self.connectivity_cohort.lower() in str(getattr(f, "cohort", "")).lower()
                or self.connectivity_cohort.lower() in str(getattr(f, "name", "")).lower()
            ),
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
            if isinstance(self._connectivity_matrix, pd.DataFrame):
                return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        if region is None:
            return None

        exact = [x for x in labels if self._name_of(x) == getattr(region, "name", None)]
        if exact:
            return exact[0]

        target = self._norm(getattr(region, "name", region))
        fuzzy = [x for x in labels if target in self._norm(self._name_of(x)) or self._norm(self._name_of(x)) in target]
        if fuzzy:
            return fuzzy[0]

        # Fall back to matching using abbreviated candidate terms.
        name = getattr(region, "name", "").lower()
        tokens = [t for t in name.replace("(", " ").replace(")", " ").replace(",", " ").split() if len(t) > 2]
        for label in labels:
            lname = self._name_of(label).lower()
            if any(tok in lname for tok in tokens[:3]):
                return label
        return None

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
            if not isinstance(series, pd.Series):
                return pd.DataFrame()
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Build a small square connectivity table among resolved circuit nodes.

        Rows and columns are node keys (not region labels), making it easier to use
        directly in downstream modeling code.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame()

        row_labels = list(matrix.index)
        col_labels = list(matrix.columns)

        matched_rows = {
            key: self._match_region_label(row_labels, region)
            for key, region in self.region_objects.items()
        }
        matched_cols = {
            key: self._match_region_label(col_labels, region)
            for key, region in self.region_objects.items()
        }

        usable = [k for k in self.region_objects if matched_rows.get(k) is not None and matched_cols.get(k) is not None]
        if not usable:
            return pd.DataFrame()

        out = pd.DataFrame(index=usable, columns=usable, dtype=float)
        for src in usable:
            for dst in usable:
                try:
                    out.loc[src, dst] = float(matrix.loc[matched_rows[src], matched_cols[dst]])
                except Exception:
                    out.loc[src, dst] = float("nan")
        return out

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> dict:
        """
        Resolve atlas regions, collect multimodal features, and assemble node / edge tables.
        """
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}
        self._connectivity_matrix = None

        nodes = []

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
            region_desc = self.region_descriptions.get(key, "Atlas-backed circuit node")
            if region is None:
                warnings.warn(f"Could not resolve a region for node '{key}'")
                self.receptors[key] = pd.DataFrame()
                self.genes[key] = pd.DataFrame()
                self.connectivity_profiles[key] = pd.DataFrame()
                nodes.append(
                    {
                        "key": key,
                        "label": key.replace("_", " ").title(),
                        "node_type": "region",
                        "description": f"{region_desc} (unresolved in this environment)",
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
                    "description": region_desc,
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
        psychosis_polygenic_load: float = 0.0,
        affective_polygenic_load: float = 0.0,
        late_life_structural_vulnerability: float = 0.0,
        focal_neurological_insult: float = 0.0,
        stimulant_dopamine_load: float = 0.0,
        affective_state_load: float = 0.0,
        d2_antagonist_treatment: float = 0.0,
        serotonergic_adjuvant_treatment: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        Transparent one-pass simulator using normalized 0..1 inputs.

        Interpretation
        --------------
        Higher values in `regional_state` indicate *burden / dysregulation* on the
        corresponding region node, not healthy function.
        """
        inputs = {
            "psychosis_polygenic_load": self._clip01(psychosis_polygenic_load),
            "affective_polygenic_load": self._clip01(affective_polygenic_load),
            "late_life_structural_vulnerability": self._clip01(late_life_structural_vulnerability),
            "focal_neurological_insult": self._clip01(focal_neurological_insult),
            "stimulant_dopamine_load": self._clip01(stimulant_dopamine_load),
            "affective_state_load": self._clip01(affective_state_load),
            "d2_antagonist_treatment": self._clip01(d2_antagonist_treatment),
            "serotonergic_adjuvant_treatment": self._clip01(serotonergic_adjuvant_treatment),
        }

        latents = {}
        latents["dopaminergic_aberrant_salience"] = self._clip01(
            0.35 * inputs["psychosis_polygenic_load"]
            + 0.35 * inputs["stimulant_dopamine_load"]
            + 0.15 * inputs["late_life_structural_vulnerability"]
            + 0.15 * inputs["focal_neurological_insult"]
            + 0.10 * inputs["affective_state_load"]
            - 0.35 * inputs["d2_antagonist_treatment"]
            - 0.08 * inputs["serotonergic_adjuvant_treatment"]
        )

        latents["serotonergic_affective_modulation_shift"] = self._clip01(
            0.35 * inputs["affective_polygenic_load"]
            + 0.30 * inputs["affective_state_load"]
            + 0.10 * inputs["psychosis_polygenic_load"]
            + 0.10 * inputs["late_life_structural_vulnerability"]
            - 0.30 * inputs["serotonergic_adjuvant_treatment"]
            - 0.05 * inputs["d2_antagonist_treatment"]
        )

        latents["fronto_striatal_limbic_dysconnectivity"] = self._clip01(
            0.30 * inputs["psychosis_polygenic_load"]
            + 0.25 * inputs["late_life_structural_vulnerability"]
            + 0.20 * inputs["focal_neurological_insult"]
            + 0.15 * latents["dopaminergic_aberrant_salience"]
            + 0.10 * latents["serotonergic_affective_modulation_shift"]
            - 0.15 * inputs["d2_antagonist_treatment"]
        )

        latents["hippocampal_context_instability"] = self._clip01(
            0.35 * inputs["late_life_structural_vulnerability"]
            + 0.30 * inputs["focal_neurological_insult"]
            + 0.15 * inputs["psychosis_polygenic_load"]
            + 0.10 * latents["serotonergic_affective_modulation_shift"]
            - 0.10 * inputs["serotonergic_adjuvant_treatment"]
        )

        latents["limbic_threat_tagging"] = self._clip01(
            0.35 * latents["dopaminergic_aberrant_salience"]
            + 0.25 * latents["serotonergic_affective_modulation_shift"]
            + 0.15 * inputs["affective_state_load"]
            + 0.15 * latents["fronto_striatal_limbic_dysconnectivity"]
            - 0.10 * inputs["d2_antagonist_treatment"]
        )

        latents["belief_evaluation_failure"] = self._clip01(
            0.45 * latents["fronto_striatal_limbic_dysconnectivity"]
            + 0.25 * latents["hippocampal_context_instability"]
            + 0.15 * latents["dopaminergic_aberrant_salience"]
            + 0.10 * latents["serotonergic_affective_modulation_shift"]
            - 0.20 * inputs["d2_antagonist_treatment"]
            - 0.05 * inputs["serotonergic_adjuvant_treatment"]
        )

        latents["body_representation_distortion"] = self._clip01(
            0.40 * inputs["focal_neurological_insult"]
            + 0.20 * latents["fronto_striatal_limbic_dysconnectivity"]
            + 0.15 * latents["hippocampal_context_instability"]
            + 0.10 * latents["dopaminergic_aberrant_salience"]
            + 0.10 * inputs["late_life_structural_vulnerability"]
        )

        regional_state = {}
        regional_state["pfc_belief_evaluation_proxy"] = self._clip01(
            0.55 * latents["belief_evaluation_failure"]
            + 0.20 * latents["fronto_striatal_limbic_dysconnectivity"]
            + 0.10 * latents["serotonergic_affective_modulation_shift"]
            - 0.20 * inputs["d2_antagonist_treatment"]
            - 0.05 * inputs["serotonergic_adjuvant_treatment"]
        )
        regional_state["temporal_association_proxy"] = self._clip01(
            0.35 * latents["fronto_striatal_limbic_dysconnectivity"]
            + 0.30 * latents["hippocampal_context_instability"]
            + 0.15 * latents["dopaminergic_aberrant_salience"]
            + 0.10 * latents["belief_evaluation_failure"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.65 * latents["hippocampal_context_instability"]
            + 0.10 * latents["fronto_striatal_limbic_dysconnectivity"]
        )
        regional_state["amygdala"] = self._clip01(
            0.55 * latents["limbic_threat_tagging"]
            + 0.10 * latents["dopaminergic_aberrant_salience"]
            + 0.05 * inputs["affective_state_load"]
        )
        regional_state["striatal_salience_proxy"] = self._clip01(
            0.60 * latents["dopaminergic_aberrant_salience"]
            + 0.15 * latents["fronto_striatal_limbic_dysconnectivity"]
            + 0.05 * inputs["focal_neurological_insult"]
            - 0.25 * inputs["d2_antagonist_treatment"]
        )
        regional_state["parietal_body_representation_proxy"] = self._clip01(
            0.60 * latents["body_representation_distortion"]
            + 0.10 * latents["hippocampal_context_instability"]
            + 0.10 * inputs["focal_neurological_insult"]
        )

        symptoms = {}
        symptoms["fixed_false_belief"] = self._clip01(
            0.30 * regional_state["striatal_salience_proxy"]
            + 0.30 * regional_state["pfc_belief_evaluation_proxy"]
            + 0.20 * regional_state["temporal_association_proxy"]
            + 0.15 * regional_state["hippocampus"]
            + 0.05 * regional_state["amygdala"]
        )
        symptoms["paranoid_interpretation"] = self._clip01(
            0.30 * regional_state["amygdala"]
            + 0.25 * regional_state["striatal_salience_proxy"]
            + 0.20 * regional_state["temporal_association_proxy"]
            + 0.15 * symptoms["fixed_false_belief"]
            + 0.05 * inputs["affective_state_load"]
        )
        symptoms["belief_rigidity"] = self._clip01(
            0.35 * regional_state["pfc_belief_evaluation_proxy"]
            + 0.25 * regional_state["hippocampus"]
            + 0.20 * symptoms["fixed_false_belief"]
            + 0.10 * regional_state["temporal_association_proxy"]
        )
        symptoms["depressive_distress"] = self._clip01(
            0.45 * latents["serotonergic_affective_modulation_shift"]
            + 0.20 * regional_state["amygdala"]
            + 0.20 * inputs["affective_state_load"]
            + 0.10 * symptoms["belief_rigidity"]
        )
        symptoms["somatic_delusional_intensity"] = self._clip01(
            0.40 * regional_state["parietal_body_representation_proxy"]
            + 0.20 * regional_state["hippocampus"]
            + 0.20 * symptoms["fixed_false_belief"]
            + 0.10 * inputs["focal_neurological_insult"]
            + 0.05 * symptoms["belief_rigidity"]
        )

        global_disorganization = self._clip01(
            0.25 * inputs["focal_neurological_insult"]
            + 0.20 * latents["serotonergic_affective_modulation_shift"]
            + 0.20 * latents["fronto_striatal_limbic_dysconnectivity"]
            + 0.10 * inputs["late_life_structural_vulnerability"]
        )

        phenotypes = {}
        phenotypes["systematized_delusion_profile"] = self._clip01(
            0.40 * symptoms["fixed_false_belief"]
            + 0.30 * symptoms["belief_rigidity"]
            + 0.15 * symptoms["paranoid_interpretation"]
        )
        phenotypes["persecutory_profile"] = self._clip01(
            0.45 * symptoms["paranoid_interpretation"]
            + 0.25 * symptoms["fixed_false_belief"]
            + 0.15 * regional_state["amygdala"]
            + 0.05 * latents["dopaminergic_aberrant_salience"]
        )
        phenotypes["somatic_profile"] = self._clip01(
            0.55 * symptoms["somatic_delusional_intensity"]
            + 0.15 * symptoms["fixed_false_belief"]
            + 0.10 * regional_state["parietal_body_representation_proxy"]
            + 0.05 * inputs["focal_neurological_insult"]
        )
        phenotypes["affective_overlap_profile"] = self._clip01(
            0.45 * symptoms["depressive_distress"]
            + 0.20 * symptoms["fixed_false_belief"]
            + 0.15 * latents["serotonergic_affective_modulation_shift"]
            + 0.10 * regional_state["amygdala"]
        )
        phenotypes["late_onset_vulnerability_profile"] = self._clip01(
            0.35 * inputs["late_life_structural_vulnerability"]
            + 0.25 * inputs["focal_neurological_insult"]
            + 0.20 * regional_state["hippocampus"]
            + 0.10 * regional_state["pfc_belief_evaluation_proxy"]
        )
        phenotypes["circumscribed_delusion_profile"] = self._clip01(
            0.35 * symptoms["fixed_false_belief"]
            + 0.20 * symptoms["belief_rigidity"]
            + 0.10 * symptoms["paranoid_interpretation"]
            - 0.25 * global_disorganization
        )

        return {
            "inputs": pd.Series(inputs, name="input"),
            "latents": pd.Series(latents, name="latent_biology"),
            "regional_state": pd.Series(regional_state, name="regional_burden"),
            "symptoms": pd.Series(symptoms, name="symptom"),
            "phenotypes": pd.Series(phenotypes, name="phenotype"),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        """
        Probabilistically assign an MNI152 coordinate to Julich regions.
        """
        if self._pmap is None:
            with self._quiet_context():
                self._pmap = siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=self.assignment_space,
                    maptype="statistical",
                )

        point = siibra.Point(tuple(float(x) for x in xyz), space=self.assignment_space)
        with self._quiet_context():
            assignments = self._pmap.assign(point)

        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str):
        """
        Return a regional mask or map-like object for a resolved region node.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None

        attempts = [
            {"space": self.assignment_space, "maptype": "labelled"},
            {"space": self.assignment_space, "maptype": "statistical"},
            {"space": self.assignment_space},
        ]
        for kwargs in attempts:
            try:
                return region.get_regional_mask(**kwargs)
            except Exception:
                continue
        for args in [
            (self.assignment_space, "statistical"),
            (self.assignment_space,),
        ]:
            try:
                return region.get_regional_map(*args)
            except Exception:
                continue
        return None


if __name__ == "__main__":
    model = DelusionalDisorderModel()

    print("Building Delusional Disorder scaffold...")
    scaffold = model.build(connectivity_rows=10)

    print("\nNodes:")
    node_cols = ["key", "node_type", "atlas_region", "feature_summary"]
    print(scaffold["nodes"][node_cols].to_string(index=False))

    print("\nEdges:")
    edge_cols = ["source", "target", "relation", "dd_change"]
    print(scaffold["edges"][edge_cols].to_string(index=False))

    circuit_df = scaffold["circuit_connectivity"]
    print("\nCircuit connectivity among resolved region nodes:")
    if circuit_df.empty:
        print("No circuit connectivity matrix available in this environment.")
    else:
        print(circuit_df.round(4).to_string())

    for key in ["amygdala", "hippocampus", "pfc_belief_evaluation_proxy"]:
        receptor_df = scaffold["receptors"].get(key, pd.DataFrame())
        if not receptor_df.empty:
            print(f"\nExample receptor fingerprint for {key}:")
            print(receptor_df.head(10).to_string(index=False))
            break

    for key in ["amygdala", "hippocampus", "pfc_belief_evaluation_proxy"]:
        gene_df = scaffold["genes"].get(key, pd.DataFrame())
        if not gene_df.empty:
            print(f"\nExample gene summary for {key}:")
            print(gene_df.head(10).to_string(index=False))
            break

    for key in ["amygdala", "hippocampus", "pfc_belief_evaluation_proxy"]:
        conn_df = scaffold["connectivity_profiles"].get(key, pd.DataFrame())
        if not conn_df.empty:
            print(f"\nExample connectivity profile for {key}:")
            print(conn_df.head(10).to_string(index=False))
            break

    simulation = model.simulate(
        psychosis_polygenic_load=0.65,
        affective_polygenic_load=0.40,
        late_life_structural_vulnerability=0.35,
        focal_neurological_insult=0.10,
        stimulant_dopamine_load=0.20,
        affective_state_load=0.45,
        d2_antagonist_treatment=0.55,
        serotonergic_adjuvant_treatment=0.30,
    )

    print("\nSimulation example:")
    for section, series in simulation.items():
        print(f"\n[{section}]")
        print(series.round(4).to_string())

    # Optional coordinate assignment examples:
    # print(model.assign_mni_point((-24, -8, -18)).head())
    # print(model.assign_mni_point((-32, -22, 10)).head())
