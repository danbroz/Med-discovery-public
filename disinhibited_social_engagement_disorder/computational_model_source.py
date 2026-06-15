from __future__ import annotations

"""
Atlas-grounded siibra scaffold for Disinhibited Social Engagement Disorder (DSED).

This script turns a chapter-level biological summary into a transparent research
scaffold. It is not a diagnostic or treatment tool. The model encodes chapter
logic as an interpretable graph:

    inputs -> latent biology -> regional state -> symptoms -> phenotype summaries

The chapter motivating this scaffold emphasizes:
- early social deprivation, institutional care, neglect, and family dysfunction,
- disruption of attachment reward chemistry (oxytocin / vasopressin / dopamine /
  endogenous opioids),
- chronic stress and probable biological embedding / epigenetic scarring,
- dysregulated social fear calibration involving the amygdala, hippocampus, and
  broadly described prefrontal control systems,
- downstream indiscriminate social approach and reduced wariness of strangers.

Because the source chapter names the prefrontal cortex only at a systems level,
this scaffold keeps that circuit as a conservative proxy node (``pfc_control``)
that can resolve to a Julich candidate when available but is explicitly treated
as an approximation.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


DEFAULT_GENE_PANEL = [
    "OXTR",    # oxytocin receptor
    "AVPR1A",  # vasopressin receptor
    "DRD2",    # dopamine D2 receptor signaling
    "SLC6A4",  # serotonin transporter / stress regulation context
    "BDNF",    # plasticity and developmental adaptation
    "NR3C1",   # glucocorticoid receptor
    "FKBP5",   # stress-response regulation
    "OPRM1",   # mu-opioid receptor / social comfort
    "CRHR1",   # stress signaling
]


class DisinhibitedSocialEngagementDisorderModel:
    """
    Research scaffold for DSED based on a chapter describing attachment
    disruption, social fear calibration failure, chronic stress, and persistent
    indiscriminate approach behavior after early deprivation.

    The model is deliberately conservative:
    - only chapter-named or strongly implied regions are atlas-anchored,
    - broad systems (for example, generic prefrontal control) are represented as
      proxy nodes,
    - neurochemistry remains in latent biology unless the chapter clearly localizes it.
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

        # Conservative region anchors. The chapter explicitly names amygdala,
        # hippocampus, and prefrontal cortex. The PFC remains a proxy because no
        # specific subfield is justified by the source text.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "SF (Amygdala) left",
                "CM (Amygdala) left",
                "Amygdala left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 left",
                "Subiculum left",
                "CA3 left",
                "hippocampus left",
                "hippocampus",
            ],
            "pfc_control": [
                "prefrontal cortex",
                "Area 46 left",
                "Area 9/46d left",
                "Area Fp2 left",
                "frontopolar cortex",
            ],
        }

        self.input_nodes: Dict[str, str] = {
            "early_social_deprivation": "Institutional care, profound neglect, or absence of stable contingent caregiving during early development.",
            "attachment_disruption": "Lack of a consistent secure-base relationship and repeated caregiver turnover or separation.",
            "chronic_social_stress": "Persistent stress related to social loss, insecurity, and unbuffered distress.",
            "family_dysfunction": "Risky family environment marked by conflict, poor responsiveness, parental psychopathology, or substance misuse.",
            "neglect_abuse_load": "Severity of neglect, abuse, and developmental deprivation.",
            "intergenerational_vulnerability": "Inherited liability and parental behavioral risk that can shape the caregiving environment and stress sensitivity.",
            "caregiving_stability": "Protective continuity, responsiveness, and predictable caregiving.",
            "enriched_recovery_environment": "Quality of foster, adoptive, or rehabilitative placement that provides social and cognitive enrichment.",
            "developmental_plasticity": "Capacity for catch-up following intervention, including earlier placement and broader developmental flexibility.",
            "genetic_resilience": "Protective biological resilience limiting stress embedding and improving recovery.",
        }

        self.latent_nodes: Dict[str, str] = {
            "oxytocin_vasopressin_signaling_disruption": "Impaired attachment-related social affiliation signaling due to deprivation of stable caregiving.",
            "dopaminergic_social_reward_blunting": "Weakening of selective social reward learning for caregiver proximity and familiarity.",
            "endogenous_opioid_attachment_dysregulation": "Altered attachment comfort / separation-distress chemistry linked to unstable bonds and repeated social loss.",
            "attachment_reward_underdevelopment": "Failure to develop a strongly rewarding, selective attachment preference for a trusted caregiver.",
            "hpa_axis_stress_sensitization": "Chronic stress activation and poor buffering under early adversity.",
            "epigenetic_embedding": "Persistent biological embedding of adversity, plausibly altering stress and social-behavior gene regulation.",
            "social_fear_calibration_failure": "Failure to learn appropriate discrimination between safe familiar adults and unfamiliar or potentially risky adults.",
            "frontolimbic_regulatory_weakening": "Reduced top-down regulation across prefrontal-limbic circuits under chronic adversity.",
            "hippocampal_context_memory_disruption": "Impaired contextual memory for caregiver-specific safety and social context under stress burden.",
            "white_matter_social_brain_immaturity": "Delayed integration of distributed social and executive circuitry due to deprivation and low stimulation.",
        }

        self.symptom_nodes: Dict[str, str] = {
            "reduced_wariness_of_strangers": "Weak or absent typical social caution toward unfamiliar adults.",
            "indiscriminate_social_approach": "Overly familiar approach behavior toward unfamiliar adults without developmentally expected restraint.",
            "nonselective_attachment_behavior": "Low selectivity in seeking proximity or comfort across adults rather than prioritizing a primary attachment figure.",
            "shallow_social_bonding": "Reduced depth or exclusivity of attachment despite social approach behavior.",
            "social_boundary_impairment": "Difficulty respecting interpersonal boundaries and context-appropriate social distance.",
            "persistence_after_placement": "Residual symptoms despite later improvement in caregiving context, consistent with partial catch-up.",
        }

        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "early_social_deprivation",
                "target": "oxytocin_vasopressin_signaling_disruption",
                "relation": "reduces the normal affiliative input needed for selective attachment signaling",
                "dsed_change": "increased disruption",
            },
            {
                "source": "early_social_deprivation",
                "target": "dopaminergic_social_reward_blunting",
                "relation": "weakens socially rewarding learning around a stable caregiver",
                "dsed_change": "increased blunting",
            },
            {
                "source": "attachment_disruption",
                "target": "endogenous_opioid_attachment_dysregulation",
                "relation": "repeated separation and unstable bonds dysregulate social comfort and separation-distress chemistry",
                "dsed_change": "increased dysregulation",
            },
            {
                "source": "family_dysfunction",
                "target": "chronic_social_stress",
                "relation": "creates an unbuffered risky family environment",
                "dsed_change": "increased",
            },
            {
                "source": "neglect_abuse_load",
                "target": "hpa_axis_stress_sensitization",
                "relation": "sustained adversity loads the stress-response system",
                "dsed_change": "increased",
            },
            {
                "source": "intergenerational_vulnerability",
                "target": "family_dysfunction",
                "relation": "parental psychopathology and substance-related risk can propagate a neglectful environment",
                "dsed_change": "increased",
            },
            {
                "source": "intergenerational_vulnerability",
                "target": "epigenetic_embedding",
                "relation": "genetic liability can interact with adversity to amplify persistent biological embedding",
                "dsed_change": "increased vulnerability",
            },
            {
                "source": "oxytocin_vasopressin_signaling_disruption",
                "target": "attachment_reward_underdevelopment",
                "relation": "disrupted affiliation chemistry undermines selective bond formation",
                "dsed_change": "increased",
            },
            {
                "source": "dopaminergic_social_reward_blunting",
                "target": "attachment_reward_underdevelopment",
                "relation": "reduced social reward weakens reinforcement of caregiver preference",
                "dsed_change": "increased",
            },
            {
                "source": "endogenous_opioid_attachment_dysregulation",
                "target": "attachment_reward_underdevelopment",
                "relation": "reduced attachment comfort and abnormal separation distress impede deep selective bonding",
                "dsed_change": "increased",
            },
            {
                "source": "chronic_social_stress",
                "target": "hpa_axis_stress_sensitization",
                "relation": "unbuffered adversity chronically activates stress regulation systems",
                "dsed_change": "increased",
            },
            {
                "source": "neglect_abuse_load",
                "target": "epigenetic_embedding",
                "relation": "severe adversity plausibly induces lasting changes in stress and social-behavior gene regulation",
                "dsed_change": "increased",
            },
            {
                "source": "hpa_axis_stress_sensitization",
                "target": "frontolimbic_regulatory_weakening",
                "relation": "stress burden weakens regulatory development across prefrontal-limbic systems",
                "dsed_change": "increased",
            },
            {
                "source": "hpa_axis_stress_sensitization",
                "target": "hippocampal_context_memory_disruption",
                "relation": "stress-sensitive memory systems are burdened by chronic adversity",
                "dsed_change": "increased",
            },
            {
                "source": "epigenetic_embedding",
                "target": "frontolimbic_regulatory_weakening",
                "relation": "biological embedding can help symptoms persist even after the environment improves",
                "dsed_change": "increased persistence",
            },
            {
                "source": "early_social_deprivation",
                "target": "white_matter_social_brain_immaturity",
                "relation": "low social and cognitive stimulation slows distributed circuit integration",
                "dsed_change": "increased",
            },
            {
                "source": "attachment_reward_underdevelopment",
                "target": "social_fear_calibration_failure",
                "relation": "without a rewarding selective bond, safe-versus-unsafe social learning is less well calibrated",
                "dsed_change": "increased",
            },
            {
                "source": "attachment_disruption",
                "target": "social_fear_calibration_failure",
                "relation": "absence of a secure base impairs caregiver-guided stranger wariness learning",
                "dsed_change": "increased",
            },
            {
                "source": "social_fear_calibration_failure",
                "target": "amygdala",
                "relation": "impairs amygdala-based social threat discrimination",
                "dsed_change": "dysregulated",
            },
            {
                "source": "frontolimbic_regulatory_weakening",
                "target": "pfc_control",
                "relation": "weakens prefrontal regulation of social approach and boundary setting",
                "dsed_change": "dysregulated",
            },
            {
                "source": "hippocampal_context_memory_disruption",
                "target": "hippocampus",
                "relation": "reduces contextual memory support for familiar-versus-unfamiliar social cues",
                "dsed_change": "dysregulated",
            },
            {
                "source": "amygdala",
                "target": "reduced_wariness_of_strangers",
                "relation": "impaired social threat discrimination reduces stranger caution",
                "dsed_change": "increased",
            },
            {
                "source": "pfc_control",
                "target": "social_boundary_impairment",
                "relation": "weaker top-down regulation reduces context-appropriate social restraint",
                "dsed_change": "increased",
            },
            {
                "source": "hippocampus",
                "target": "nonselective_attachment_behavior",
                "relation": "weaker contextual encoding blunts selective reliance on familiar caregivers",
                "dsed_change": "increased",
            },
            {
                "source": "reduced_wariness_of_strangers",
                "target": "indiscriminate_social_approach",
                "relation": "low stranger caution promotes overfamiliar approach",
                "dsed_change": "increased",
            },
            {
                "source": "attachment_reward_underdevelopment",
                "target": "nonselective_attachment_behavior",
                "relation": "poor selective reward learning promotes indiscriminate seeking of social contact",
                "dsed_change": "increased",
            },
            {
                "source": "endogenous_opioid_attachment_dysregulation",
                "target": "shallow_social_bonding",
                "relation": "abnormal attachment comfort chemistry reduces depth of selective emotional bonds",
                "dsed_change": "increased",
            },
            {
                "source": "epigenetic_embedding",
                "target": "persistence_after_placement",
                "relation": "persistent biological embedding can leave residual difficulties after environmental improvement",
                "dsed_change": "increased",
            },
            {
                "source": "caregiving_stability",
                "target": "social_fear_calibration_failure",
                "relation": "predictable caregiving supports safe-versus-unsafe social learning",
                "dsed_change": "decreased",
            },
            {
                "source": "enriched_recovery_environment",
                "target": "frontolimbic_regulatory_weakening",
                "relation": "better caregiving and stimulation support developmental catch-up",
                "dsed_change": "decreased",
            },
            {
                "source": "developmental_plasticity",
                "target": "persistence_after_placement",
                "relation": "greater plasticity enables stronger recovery after adversity",
                "dsed_change": "decreased",
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

    def _modality_candidates(self, kind: str) -> List[Any]:
        candidates: List[Any] = []
        try:
            if kind == "receptor":
                candidates.append(siibra.features.molecular.ReceptorDensityFingerprint)
            elif kind == "gene":
                candidates.append(siibra.features.molecular.GeneExpressions)
            elif kind == "connectivity":
                candidates.append(siibra.features.connectivity.StreamlineCounts)
        except Exception:
            pass

        if kind == "receptor":
            candidates.append("receptor density fingerprint")
        elif kind == "gene":
            candidates.append("gene expressions")
        elif kind == "connectivity":
            candidates.append("StreamlineCounts")
        return candidates

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

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        proxy_penalty = 1 if any(term in name for term in ["cortex", "brain", "region"]) else 0
        generic_penalty = 1 if name in {"amygdala", "hippocampus", "prefrontal cortex"} else 0
        return (left_bonus, right_penalty, proxy_penalty, generic_penalty)

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
        rows: List[Dict[str, Any]] = []
        seen = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
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
        main = max(props, key=lambda item: getattr(item, "volume", 0.0) or 0.0)
        centroid = getattr(main, "centroid", None)
        centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        volume_mm3 = getattr(main, "volume", None)
        volume_value = float(volume_mm3) if volume_mm3 is not None else None
        return centroid_xyz, volume_value

    def _to_dataframe(self, obj: Any) -> pd.DataFrame:
        if isinstance(obj, pd.DataFrame):
            return obj.copy()
        if obj is None:
            return pd.DataFrame()
        if hasattr(obj, "to_dataframe"):
            try:
                return obj.to_dataframe().copy()
            except Exception:
                pass
        if hasattr(obj, "data"):
            data = getattr(obj, "data")
            if isinstance(data, pd.DataFrame):
                return data.copy()
            if hasattr(data, "to_dataframe"):
                try:
                    return data.to_dataframe().copy()
                except Exception:
                    pass
            try:
                return pd.DataFrame(data)
            except Exception:
                return pd.DataFrame()
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame()

    def _numeric_scalar(self, value: Any) -> Optional[float]:
        """
        Convert connectivity lookups to a single numeric value.

        In some siibra / pandas combinations, duplicated row or column labels can
        make a lookup return a Series or DataFrame rather than a scalar. We
        conservatively collapse such selections by averaging the numeric values.
        """
        if value is None:
            return None
        if isinstance(value, pd.DataFrame):
            if value.empty:
                return None
            flattened = pd.to_numeric(pd.Series(value.to_numpy().ravel()), errors="coerce").dropna()
            return float(flattened.mean()) if not flattened.empty else None
        if isinstance(value, pd.Series):
            if value.empty:
                return None
            flattened = pd.to_numeric(value, errors="coerce").dropna()
            return float(flattened.mean()) if not flattened.empty else None
        try:
            return float(value)
        except Exception:
            try:
                flattened = pd.to_numeric(pd.Series(list(value)), errors="coerce").dropna()
                return float(flattened.mean()) if not flattened.empty else None
            except Exception:
                return None

    def _normalize_connectivity_matrix(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize connectivity matrices to string labels and unique axes.

        siibra connectivity matrices commonly use region objects for row/column
        labels. Depending on version details, duplicate labels can appear, which
        makes downstream ``.loc`` selections return Series/DataFrames. This
        helper converts axes to stable region names, coerces values to numeric,
        and averages duplicate rows/columns.
        """
        if not isinstance(matrix, pd.DataFrame) or matrix.empty:
            return pd.DataFrame()

        df = matrix.copy()
        try:
            df.index = [self._name_of(label) for label in df.index]
        except Exception:
            pass
        try:
            df.columns = [self._name_of(label) for label in df.columns]
        except Exception:
            pass

        try:
            mapper = getattr(df, "map", None)
            df = mapper(self._numeric_scalar) if callable(mapper) else df.applymap(self._numeric_scalar)
        except Exception:
            pass
        try:
            df = df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
        except Exception:
            return pd.DataFrame()

        try:
            if not df.index.is_unique:
                df = df.groupby(level=0, sort=False).mean()
        except Exception:
            pass
        try:
            if not df.columns.is_unique:
                df = df.T.groupby(level=0, sort=False).mean().T
        except Exception:
            pass
        return df

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        df = self._to_dataframe(feats[0])
        if df.empty:
            return df
        df = df.reset_index()
        if "index" in df.columns and "receptor" not in df.columns:
            df = df.rename(columns={"index": "receptor"})
        return df

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()

        frames: List[pd.DataFrame] = []
        for feat in feats:
            df = self._to_dataframe(feat)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        lower_cols = {str(col).lower(): col for col in df.columns}
        required = {"gene", "level", "zscore"}
        if required.issubset(lower_cols):
            gene_col = lower_cols["gene"]
            level_col = lower_cols["level"]
            zscore_col = lower_cols["zscore"]
            grouped = (
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
                .reset_index(drop=True)
            )
            return grouped
        return df.reset_index(drop=True)

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        if not feats:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        compound = next((feat for feat in feats if getattr(feat, "cohort", None) == self.connectivity_cohort), feats[0])

        def _accept(candidate: Any) -> Optional[pd.DataFrame]:
            if isinstance(candidate, pd.DataFrame):
                df = self._normalize_connectivity_matrix(candidate)
                return df if not df.empty else None
            return None

        # Try the most direct access patterns first.
        for candidate in (compound, getattr(compound, "data", None)):
            accepted = _accept(candidate)
            if accepted is not None:
                self._connectivity_matrix = accepted
                return self._connectivity_matrix

        try:
            accepted = _accept(compound[0].data)
            if accepted is not None:
                self._connectivity_matrix = accepted
                return self._connectivity_matrix
        except Exception:
            pass

        # Some siibra versions store matrices per subject or cohort entry.
        try:
            for item in compound:
                accepted = _accept(getattr(item, "data", None))
                if accepted is not None:
                    self._connectivity_matrix = accepted
                    return self._connectivity_matrix
        except Exception:
            pass

        self._connectivity_matrix = pd.DataFrame()
        return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        exact = [label for label in labels if self._name_of(label) == region.name]
        if exact:
            return exact[0]

        region_name = region.name.lower()
        fuzzy = [
            label
            for label in labels
            if region_name in self._name_of(label).lower() or self._name_of(label).lower() in region_name
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
            if isinstance(series, pd.DataFrame):
                series = series.mean(axis=0 if axis == "index" else 1, numeric_only=True)
            if not isinstance(series, pd.Series):
                series = pd.Series(series)
            series = pd.to_numeric(series, errors="coerce").dropna()
            if series.empty:
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
        Pairwise connectivity among the scaffold's resolved circuit nodes.
        Returns an edge-style table rather than a wide matrix for easy inspection.
        """
        matrix = self._get_connectivity_matrix()
        if matrix.empty or not self.region_objects:
            return pd.DataFrame(columns=["source", "target", "value"])

        labels: Dict[str, Any] = {}
        for key, region in self.region_objects.items():
            label = self._match_region_label(list(matrix.index), region)
            if label is None:
                label = self._match_region_label(list(matrix.columns), region)
            if label is not None:
                labels[key] = label

        rows: List[Dict[str, Any]] = []
        for source_key, source_label in labels.items():
            for target_key, target_label in labels.items():
                if source_key == target_key:
                    continue
                value = None
                try:
                    value = matrix.loc[source_label, target_label]
                except Exception:
                    try:
                        value = matrix[target_label].loc[source_label]
                    except Exception:
                        value = None
                value_scalar = self._numeric_scalar(value)
                if value_scalar is None:
                    continue
                rows.append(
                    {
                        "source": source_key,
                        "target": target_key,
                        "value": value_scalar,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["source", "target", "value"])
        return pd.DataFrame(rows).sort_values(["source", "value"], ascending=[True, False]).reset_index(drop=True)

    def build(
        self,
        gene_panel: Sequence[str] = DEFAULT_GENE_PANEL,
        connectivity_rows: int = 15,
    ) -> Dict[str, Any]:
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}

        nodes: List[Dict[str, Any]] = []

        for key, description in self.input_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "input",
                    "description": description,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            region_description = (
                "Proxy region for generic prefrontal control described in the chapter."
                if key == "pfc_control"
                else "Atlas-backed circuit node named in or strongly implied by the chapter."
            )

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
                        "description": f"{region_description} Unresolved in this siibra environment.",
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
                    "description": region_description,
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

        for key, description in self.latent_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "latent_biology",
                    "description": description,
                    "atlas_region": None,
                    "region_identifier": None,
                    "centroid_mni": None,
                    "volume_mm3": None,
                    "feature_summary": None,
                }
            )

        for key, description in self.symptom_nodes.items():
            nodes.append(
                {
                    "key": key,
                    "label": key.replace("_", " ").title(),
                    "node_type": "symptom",
                    "description": description,
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
        *,
        early_social_deprivation: float = 0.8,
        attachment_disruption: float = 0.8,
        chronic_social_stress: float = 0.7,
        family_dysfunction: float = 0.6,
        neglect_abuse_load: float = 0.7,
        intergenerational_vulnerability: float = 0.5,
        caregiving_stability: float = 0.2,
        enriched_recovery_environment: float = 0.3,
        developmental_plasticity: float = 0.5,
        genetic_resilience: float = 0.4,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        All inputs are clipped to [0, 1]. Larger values mean more of that factor,
        except the protective variables, which reduce downstream burden.
        """
        inputs = {
            "early_social_deprivation": self._clip01(early_social_deprivation),
            "attachment_disruption": self._clip01(attachment_disruption),
            "chronic_social_stress": self._clip01(chronic_social_stress),
            "family_dysfunction": self._clip01(family_dysfunction),
            "neglect_abuse_load": self._clip01(neglect_abuse_load),
            "intergenerational_vulnerability": self._clip01(intergenerational_vulnerability),
            "caregiving_stability": self._clip01(caregiving_stability),
            "enriched_recovery_environment": self._clip01(enriched_recovery_environment),
            "developmental_plasticity": self._clip01(developmental_plasticity),
            "genetic_resilience": self._clip01(genetic_resilience),
        }
        protective_buffer = self._clip01(
            0.4 * inputs["caregiving_stability"]
            + 0.35 * inputs["enriched_recovery_environment"]
            + 0.15 * inputs["developmental_plasticity"]
            + 0.10 * inputs["genetic_resilience"]
        )

        latents = {
            "oxytocin_vasopressin_signaling_disruption": self._clip01(
                0.42 * inputs["early_social_deprivation"]
                + 0.16 * inputs["attachment_disruption"]
                + 0.12 * inputs["family_dysfunction"]
                + 0.08 * inputs["intergenerational_vulnerability"]
                - 0.26 * inputs["caregiving_stability"]
                - 0.12 * inputs["enriched_recovery_environment"]
            ),
            "dopaminergic_social_reward_blunting": self._clip01(
                0.38 * inputs["early_social_deprivation"]
                + 0.24 * inputs["attachment_disruption"]
                + 0.18 * inputs["chronic_social_stress"]
                - 0.20 * inputs["caregiving_stability"]
                - 0.12 * inputs["enriched_recovery_environment"]
            ),
            "endogenous_opioid_attachment_dysregulation": self._clip01(
                0.30 * inputs["attachment_disruption"]
                + 0.24 * inputs["chronic_social_stress"]
                + 0.16 * inputs["family_dysfunction"]
                + 0.10 * inputs["intergenerational_vulnerability"]
                - 0.18 * inputs["caregiving_stability"]
                - 0.10 * inputs["enriched_recovery_environment"]
            ),
            "hpa_axis_stress_sensitization": self._clip01(
                0.34 * inputs["chronic_social_stress"]
                + 0.26 * inputs["neglect_abuse_load"]
                + 0.18 * inputs["family_dysfunction"]
                + 0.08 * inputs["attachment_disruption"]
                - 0.18 * inputs["enriched_recovery_environment"]
                - 0.10 * inputs["genetic_resilience"]
            ),
        }

        latents["attachment_reward_underdevelopment"] = self._clip01(
            0.34 * latents["oxytocin_vasopressin_signaling_disruption"]
            + 0.30 * latents["dopaminergic_social_reward_blunting"]
            + 0.24 * latents["endogenous_opioid_attachment_dysregulation"]
            + 0.08 * inputs["early_social_deprivation"]
            - 0.16 * inputs["caregiving_stability"]
            - 0.08 * inputs["enriched_recovery_environment"]
        )

        latents["epigenetic_embedding"] = self._clip01(
            0.24 * inputs["neglect_abuse_load"]
            + 0.22 * latents["hpa_axis_stress_sensitization"]
            + 0.16 * inputs["family_dysfunction"]
            + 0.14 * inputs["intergenerational_vulnerability"]
            + 0.10 * inputs["chronic_social_stress"]
            - 0.14 * inputs["developmental_plasticity"]
            - 0.10 * inputs["enriched_recovery_environment"]
        )

        latents["social_fear_calibration_failure"] = self._clip01(
            0.34 * inputs["attachment_disruption"]
            + 0.24 * inputs["early_social_deprivation"]
            + 0.20 * latents["attachment_reward_underdevelopment"]
            + 0.12 * latents["hpa_axis_stress_sensitization"]
            - 0.20 * inputs["caregiving_stability"]
            - 0.08 * inputs["enriched_recovery_environment"]
        )

        latents["frontolimbic_regulatory_weakening"] = self._clip01(
            0.30 * latents["hpa_axis_stress_sensitization"]
            + 0.22 * latents["epigenetic_embedding"]
            + 0.18 * inputs["early_social_deprivation"]
            + 0.12 * inputs["family_dysfunction"]
            + 0.08 * inputs["neglect_abuse_load"]
            - 0.20 * inputs["enriched_recovery_environment"]
            - 0.10 * inputs["developmental_plasticity"]
        )

        latents["hippocampal_context_memory_disruption"] = self._clip01(
            0.34 * latents["hpa_axis_stress_sensitization"]
            + 0.22 * inputs["neglect_abuse_load"]
            + 0.16 * latents["epigenetic_embedding"]
            + 0.14 * inputs["early_social_deprivation"]
            - 0.16 * inputs["enriched_recovery_environment"]
            - 0.10 * inputs["developmental_plasticity"]
        )

        latents["white_matter_social_brain_immaturity"] = self._clip01(
            0.32 * inputs["early_social_deprivation"]
            + 0.18 * inputs["neglect_abuse_load"]
            + 0.18 * inputs["chronic_social_stress"]
            + 0.14 * latents["frontolimbic_regulatory_weakening"]
            - 0.18 * inputs["enriched_recovery_environment"]
            - 0.12 * inputs["developmental_plasticity"]
        )

        regional_state = {
            # Interpreted as dysfunction burden rather than simple activation.
            "amygdala": self._clip01(
                0.46 * latents["social_fear_calibration_failure"]
                + 0.18 * latents["hpa_axis_stress_sensitization"]
                + 0.16 * latents["attachment_reward_underdevelopment"]
                - 0.10 * inputs["caregiving_stability"]
            ),
            "pfc_control": self._clip01(
                0.44 * latents["frontolimbic_regulatory_weakening"]
                + 0.20 * latents["white_matter_social_brain_immaturity"]
                + 0.16 * latents["hpa_axis_stress_sensitization"]
                - 0.16 * inputs["enriched_recovery_environment"]
                - 0.10 * inputs["developmental_plasticity"]
            ),
            "hippocampus": self._clip01(
                0.48 * latents["hippocampal_context_memory_disruption"]
                + 0.18 * latents["hpa_axis_stress_sensitization"]
                + 0.12 * latents["white_matter_social_brain_immaturity"]
                - 0.14 * inputs["enriched_recovery_environment"]
                - 0.08 * inputs["genetic_resilience"]
            ),
        }

        symptoms = {
            "reduced_wariness_of_strangers": self._clip01(
                0.42 * latents["social_fear_calibration_failure"]
                + 0.20 * regional_state["amygdala"]
                + 0.12 * latents["endogenous_opioid_attachment_dysregulation"]
                + 0.10 * latents["attachment_reward_underdevelopment"]
                - 0.10 * inputs["caregiving_stability"]
            ),
            "nonselective_attachment_behavior": self._clip01(
                0.34 * latents["attachment_reward_underdevelopment"]
                + 0.24 * latents["dopaminergic_social_reward_blunting"]
                + 0.16 * regional_state["hippocampus"]
                + 0.12 * latents["endogenous_opioid_attachment_dysregulation"]
                - 0.10 * inputs["caregiving_stability"]
            ),
            "shallow_social_bonding": self._clip01(
                0.40 * latents["attachment_reward_underdevelopment"]
                + 0.22 * latents["endogenous_opioid_attachment_dysregulation"]
                + 0.14 * regional_state["hippocampus"]
                + 0.10 * latents["oxytocin_vasopressin_signaling_disruption"]
                - 0.12 * inputs["enriched_recovery_environment"]
            ),
        }

        symptoms["indiscriminate_social_approach"] = self._clip01(
            0.42 * symptoms["reduced_wariness_of_strangers"]
            + 0.22 * symptoms["nonselective_attachment_behavior"]
            + 0.16 * regional_state["pfc_control"]
            + 0.10 * latents["white_matter_social_brain_immaturity"]
            - 0.10 * inputs["enriched_recovery_environment"]
        )

        symptoms["social_boundary_impairment"] = self._clip01(
            0.30 * symptoms["indiscriminate_social_approach"]
            + 0.28 * regional_state["pfc_control"]
            + 0.20 * symptoms["reduced_wariness_of_strangers"]
            + 0.10 * latents["white_matter_social_brain_immaturity"]
            - 0.10 * inputs["enriched_recovery_environment"]
        )

        symptoms["persistence_after_placement"] = self._clip01(
            0.34 * latents["epigenetic_embedding"]
            + 0.20 * latents["white_matter_social_brain_immaturity"]
            + 0.18 * latents["attachment_reward_underdevelopment"]
            + 0.14 * regional_state["hippocampus"]
            - 0.14 * inputs["developmental_plasticity"]
            - 0.12 * inputs["enriched_recovery_environment"]
        )

        phenotypes = {
            "dsed_core_profile": self._clip01(
                (
                    symptoms["reduced_wariness_of_strangers"]
                    + symptoms["indiscriminate_social_approach"]
                    + symptoms["nonselective_attachment_behavior"]
                )
                / 3.0
            ),
            "attachment_depth_impairment": self._clip01(
                (symptoms["shallow_social_bonding"] + symptoms["nonselective_attachment_behavior"]) / 2.0
            ),
            "persistent_social_risk_profile": self._clip01(
                (
                    symptoms["persistence_after_placement"]
                    + symptoms["social_boundary_impairment"]
                    + regional_state["pfc_control"]
                )
                / 3.0
            ),
            "recovery_buffer": self._clip01(protective_buffer),
        }

        return {
            "inputs": pd.Series(inputs).sort_index(),
            "latents": pd.Series(latents).sort_index(),
            "regional_state": pd.Series(regional_state).sort_index(),
            "symptoms": pd.Series(symptoms).sort_index(),
            "phenotypes": pd.Series(phenotypes).sort_index(),
        }

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
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
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str) -> Any:
        """
        Return a region-specific map or mask object when available.
        The exact returned type depends on the siibra version.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None

        for method_name in ("get_regional_map", "fetch", "build_mask"):
            method = getattr(region, method_name, None)
            if callable(method):
                try:
                    with siibra.QUIET:
                        if method_name == "get_regional_map":
                            return method(space=self.space)
                        return method()
                except Exception:
                    continue

        try:
            with siibra.QUIET:
                pmap = siibra.get_map(
                    parcellation=self.parcellation_spec,
                    space=self.space_spec,
                    maptype="labelled",
                )
            return pmap.fetch(region)
        except Exception:
            return None


if __name__ == "__main__":
    model = DisinhibitedSocialEngagementDisorderModel()
    bundle = model.build()

    print("\n=== Nodes ===")
    print(bundle["nodes"][["key", "node_type", "atlas_region", "feature_summary"]].to_string(index=False))

    print("\n=== Edges ===")
    print(bundle["edges"][["source", "target", "relation", "dsed_change"]].head(15).to_string(index=False))

    print("\n=== Region suggestions for 'amygdala' ===")
    print(model.suggest_regions("amygdala").head(10).to_string(index=False))

    print("\n=== Pairwise circuit connectivity ===")
    circuit_df = bundle["circuit_connectivity"]
    if circuit_df.empty:
        print("No connectivity matrix could be resolved in this environment.")
    else:
        print(circuit_df.to_string(index=False))

    print("\n=== Example receptor table: amygdala ===")
    receptor_df = bundle["receptors"].get("amygdala", pd.DataFrame())
    print(receptor_df.head(10).to_string(index=False) if not receptor_df.empty else "No receptor data available.")

    print("\n=== Example gene table: hippocampus ===")
    gene_df = bundle["genes"].get("hippocampus", pd.DataFrame())
    print(gene_df.head(10).to_string(index=False) if not gene_df.empty else "No gene-expression data available.")

    print("\n=== Example connectivity profile: pfc_control ===")
    conn_df = bundle["connectivity_profiles"].get("pfc_control", pd.DataFrame())
    print(conn_df.head(10).to_string(index=False) if not conn_df.empty else "No connectivity profile available.")

    print("\n=== Simulation example ===")
    result = model.simulate(
        early_social_deprivation=0.90,
        attachment_disruption=0.85,
        chronic_social_stress=0.80,
        family_dysfunction=0.65,
        neglect_abuse_load=0.75,
        intergenerational_vulnerability=0.55,
        caregiving_stability=0.20,
        enriched_recovery_environment=0.35,
        developmental_plasticity=0.55,
        genetic_resilience=0.45,
    )
    for name, series in result.items():
        print(f"\n{name.upper()}")
        print(series.sort_values(ascending=False).to_string())

    # Example coordinate assignment usage:
    # print(model.assign_mni_point((-24, -4, -18)).head())
