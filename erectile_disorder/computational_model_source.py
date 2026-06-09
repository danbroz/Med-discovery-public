from __future__ import annotations

"""
Erectile Disorder siibra scaffold.

This script converts a biologically focused chapter on Erectile Disorder into an
atlas-grounded siibra research scaffold. It is designed for transparent
mechanistic exploration, not diagnosis, treatment selection, or clinical risk
stratification.

Modeling emphasis from the chapter:
- neurovascular and hemodynamic vulnerability,
- central and peripheral neurogenic burden,
- autonomic balance between erection-facilitating and detumescence pathways,
- endocrine modulation of desire and erectile capacity,
- limbic, frontal, thalamic, and basal-ganglia contributions,
- medication-related serotonergic/dopaminergic interference,
- vascular-depressive and white-matter/subcortical burden.
"""

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import siibra


DEFAULT_GENE_PANEL = [
    "NOS1",
    "NOS3",
    "PDE5A",
    "AR",
    "PRLR",
    "DRD2",
    "SLC6A4",
    "COMT",
    "BDNF",
    "TH",
    "VEGFA",
    "ACE",
]


class ErectileDisorderModel:
    """
    Atlas-grounded scaffold for Erectile Disorder.

    The scaffold translates a chapter-level biological narrative into:
    - input nodes (risk loads, lesion burden, supports),
    - latent biology nodes (autonomic, endocrine, vascular, motivational,
      and disconnection mechanisms),
    - atlas-backed regions or clearly labeled proxies, and
    - symptom / phenotype summaries.

    Important:
    This is a research scaffold for inspection and hypothesis generation. It is
    not a validated disease model and should not be used as a clinical tool.
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

        # Conservative atlas anchors and proxies derived from the chapter.
        self.region_candidates: Dict[str, List[str]] = {
            "amygdala": [
                "LB (Amygdala) left",
                "CM (Amygdala) left",
                "SF (Amygdala) left",
                "Amygdala left",
                "amygdala",
            ],
            "hippocampus": [
                "CA1 left",
                "Subiculum left",
                "DG left",
                "hippocampus left",
                "hippocampus",
            ],
            "frontal_lobe_proxy": [
                "Area Fp1 left",
                "Area Fp2 left",
                "Area Fo4 left",
                "Area Fo3 left",
                "frontal pole",
                "orbitofrontal",
                "frontopolar",
            ],
            "temporal_lobe_proxy": [
                "Area TGd left",
                "Area TGv left",
                "Area TE 1.0 left",
                "Area TE 1.1 left",
                "Area TE 1.2 left",
                "temporal pole",
                "inferior temporal",
            ],
            "basal_ganglia_proxy": [
                "caudate left",
                "putamen left",
                "nucleus accumbens left",
                "striatum",
                "basal ganglia",
            ],
            "thalamus_proxy": [
                "mediodorsal thalamus left",
                "anterior thalamic nucleus left",
                "thalamus left",
                "thalamus",
            ],
        }

        self.region_node_descriptions: Dict[str, str] = {
            "amygdala": "Limbic salience/arousal structure explicitly implicated by the chapter.",
            "hippocampus": "Limbic memory-context structure explicitly implicated by the chapter.",
            "frontal_lobe_proxy": "Proxy for chapter-level frontal lobe lesion/control burden in neurogenic ED.",
            "temporal_lobe_proxy": "Proxy for broader temporal-lobe dysfunction beyond named limbic structures.",
            "basal_ganglia_proxy": "Proxy for subcortical motivational/gating circuitry implicated after stroke and degeneration.",
            "thalamus_proxy": "Proxy for thalamic relay/disconnection burden implicated in cerebrovascular ED pathways.",
        }

        self.input_nodes: Dict[str, str] = {
            "genetic_liability": "Inherited liability loading cardiovascular disease, diabetes, hypertension, depression, and related ED risk.",
            "vascular_risk_load": "Combined cardiometabolic and endothelial burden reducing penile perfusion and vascular integrity.",
            "endocrine_dysregulation": "Hormonal burden such as hypogonadism or hyperprolactinemia reducing libido and erectile capacity.",
            "medication_burden": "Medications affecting dopamine, serotonin, autonomic tone, or hormonal balance.",
            "central_neurological_insult": "Stroke, TBI, tumor, Parkinsonian, or other CNS lesion burden affecting sexual circuitry.",
            "peripheral_neuropathy": "Spinal cord, diabetic, or peripheral neurogenic signaling disruption.",
            "depression_load": "Depressive burden linked to reduced desire, motivation, autonomic efficiency, and vascular overlap.",
            "seizure_limbic_instability": "Temporal-lobe/limbic epileptiform instability affecting arousal and potency.",
            "vascular_health_support": "Protective control of cardiometabolic risk and vascular function.",
            "endocrine_treatment_support": "Protective correction of hormonal deficits or prolactin excess.",
            "medication_review_support": "Protective reduction of medication-induced sexual side effects through medication review.",
            "rehabilitation_support": "Protective neurorehabilitation or adaptive support for neurological injury.",
            "mood_treatment_support": "Protective treatment that lowers depression burden and motivational suppression.",
        }

        self.latent_nodes: Dict[str, str] = {
            "neurovascular_cavernosal_impairment": "Downstream vascular-hemodynamic impairment of erection generation and maintenance.",
            "autonomic_erection_drive_failure": "Reduced erection-facilitating parasympathetic/NANC efficacy or impaired autonomic integration.",
            "sympathetic_detumescence_bias": "Relative dominance of detumescence-promoting sympathetic tone over erection-facilitating balance.",
            "dopaminergic_motivation_deficit": "Reduced central sexual motivation and incentive salience.",
            "serotonergic_inhibitory_pressure": "Serotonergic or medication-linked inhibition of arousal and erection.",
            "endocrine_libido_deficit": "Hormone-linked reduction in sexual desire and erectile readiness.",
            "frontolimbic_arousal_disruption": "Breakdown in frontal-limbic integration of sexual cues, drive, and behavioral control.",
            "limbic_temporal_disruption": "Temporal-limbic dysfunction affecting emotional, mnemonic, and seizure-related sexual processing.",
            "subcortical_vascular_disconnection": "Stroke/white-matter/subcortical burden disrupting relay and network integration.",
            "peripheral_neurogenic_signal_loss": "Loss of peripheral afferent/efferent signaling relevant to erection.",
        }

        self.symptom_nodes: Dict[str, str] = {
            "diminished_sexual_desire": "Reduced libido or subjective sexual interest.",
            "impaired_erection_initiation": "Difficulty initiating erection when sexually stimulated.",
            "impaired_erection_maintenance": "Difficulty maintaining erection once initiated.",
            "neurogenic_ed_expression": "Predominantly neurogenic ED pattern linked to central/peripheral lesion burden.",
            "vascular_depressive_ed_expression": "Mixed vascular-subcortical-depressive ED phenotype.",
        }

        # Directional edges extracted from the chapter narrative.
        self.edge_table: List[Dict[str, str]] = [
            {
                "source": "genetic_liability",
                "target": "vascular_risk_load",
                "relation": "loads cardiovascular, hypertensive, and diabetic risk pathways that raise ED vulnerability",
                "domain": "risk",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "genetic_liability",
                "target": "depression_load",
                "relation": "loads inherited vulnerability to depressive illness that can secondarily increase sexual dysfunction",
                "domain": "risk",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "vascular_risk_load",
                "target": "neurovascular_cavernosal_impairment",
                "relation": "reduces endothelial and vascular support for erection",
                "domain": "vascular",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "endocrine_dysregulation",
                "target": "endocrine_libido_deficit",
                "relation": "lowers libido and erectile readiness through hormonal insufficiency or prolactin excess",
                "domain": "endocrine",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "medication_burden",
                "target": "serotonergic_inhibitory_pressure",
                "relation": "raises centrally mediated inhibition of arousal and erection",
                "domain": "pharmacology",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "medication_burden",
                "target": "autonomic_erection_drive_failure",
                "relation": "impairs autonomic efficiency through antihypertensive, antipsychotic, or antidepressant effects",
                "domain": "pharmacology",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "central_neurological_insult",
                "target": "frontolimbic_arousal_disruption",
                "relation": "disrupts cortical-subcortical processing of sexual stimuli",
                "domain": "neurology",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "central_neurological_insult",
                "target": "subcortical_vascular_disconnection",
                "relation": "adds stroke-like or lesion-related relay/disconnection burden",
                "domain": "neurology",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "peripheral_neuropathy",
                "target": "peripheral_neurogenic_signal_loss",
                "relation": "reduces peripheral neural transmission required for erection",
                "domain": "neurology",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "depression_load",
                "target": "dopaminergic_motivation_deficit",
                "relation": "reduces sexual motivation and incentive salience",
                "domain": "mood",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "depression_load",
                "target": "frontolimbic_arousal_disruption",
                "relation": "weakens mood-linked drive and regulation of sexual behavior",
                "domain": "mood",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "seizure_limbic_instability",
                "target": "limbic_temporal_disruption",
                "relation": "abnormal electrical activity in temporal-limbic structures disrupts normal sexual behavior",
                "domain": "epilepsy",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "vascular_health_support",
                "target": "neurovascular_cavernosal_impairment",
                "relation": "improves vascular reserve and partially offsets neurovascular ED load",
                "domain": "protective",
                "erectile_disorder_change": "decreased",
            },
            {
                "source": "endocrine_treatment_support",
                "target": "endocrine_libido_deficit",
                "relation": "corrects hormone-linked reduction in libido and erectile readiness",
                "domain": "protective",
                "erectile_disorder_change": "decreased",
            },
            {
                "source": "medication_review_support",
                "target": "serotonergic_inhibitory_pressure",
                "relation": "reduces medication-induced inhibitory burden",
                "domain": "protective",
                "erectile_disorder_change": "decreased",
            },
            {
                "source": "rehabilitation_support",
                "target": "peripheral_neurogenic_signal_loss",
                "relation": "partially offsets neurological disability burden",
                "domain": "protective",
                "erectile_disorder_change": "decreased",
            },
            {
                "source": "mood_treatment_support",
                "target": "depression_load",
                "relation": "reduces depressive burden that otherwise worsens sexual dysfunction",
                "domain": "protective",
                "erectile_disorder_change": "decreased",
            },
            {
                "source": "frontolimbic_arousal_disruption",
                "target": "amygdala",
                "relation": "dysregulates limbic salience and emotional arousal processing",
                "domain": "circuit",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "limbic_temporal_disruption",
                "target": "hippocampus",
                "relation": "alters mnemonic and contextual processing of sexual stimuli",
                "domain": "circuit",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "frontolimbic_arousal_disruption",
                "target": "frontal_lobe_proxy",
                "relation": "weakens frontal integration and control relevant to sexual response",
                "domain": "circuit",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "limbic_temporal_disruption",
                "target": "temporal_lobe_proxy",
                "relation": "adds broader temporal-lobe dysfunction beyond limbic nuclei",
                "domain": "circuit",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "subcortical_vascular_disconnection",
                "target": "basal_ganglia_proxy",
                "relation": "adds subcortical gating and motivational circuitry burden",
                "domain": "circuit",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "subcortical_vascular_disconnection",
                "target": "thalamus_proxy",
                "relation": "adds thalamic relay/disconnection burden relevant to sexual network integration",
                "domain": "circuit",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "endocrine_libido_deficit",
                "target": "diminished_sexual_desire",
                "relation": "lowers baseline libido and erotic responsiveness",
                "domain": "symptom",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "dopaminergic_motivation_deficit",
                "target": "diminished_sexual_desire",
                "relation": "reduces motivational drive for sexual behavior",
                "domain": "symptom",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "autonomic_erection_drive_failure",
                "target": "impaired_erection_initiation",
                "relation": "directly impairs erection triggering",
                "domain": "symptom",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "peripheral_neurogenic_signal_loss",
                "target": "impaired_erection_initiation",
                "relation": "weakens afferent-efferent signaling needed to initiate erection",
                "domain": "symptom",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "neurovascular_cavernosal_impairment",
                "target": "impaired_erection_maintenance",
                "relation": "impairs erection rigidity and maintenance",
                "domain": "symptom",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "sympathetic_detumescence_bias",
                "target": "impaired_erection_maintenance",
                "relation": "promotes detumescence relative to erection maintenance",
                "domain": "symptom",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "subcortical_vascular_disconnection",
                "target": "vascular_depressive_ed_expression",
                "relation": "contributes to mixed vascular and mood-linked ED expression",
                "domain": "symptom",
                "erectile_disorder_change": "increased",
            },
            {
                "source": "peripheral_neurogenic_signal_loss",
                "target": "neurogenic_ed_expression",
                "relation": "drives neurogenic erectile dysfunction expression",
                "domain": "symptom",
                "erectile_disorder_change": "increased",
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

    def _parcellation_specs(self) -> List[str]:
        specs = [
            self.parcellation_spec,
            getattr(self.parcellation, "name", None),
            getattr(self.parcellation, "key", None),
        ]
        if "julich" in self.parcellation_spec.lower():
            specs.extend(["julich 3.0.3", "julich 2.9", "julich"])
        return [s for s in specs if isinstance(s, str) and s]

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
            cands.extend(["receptor density fingerprint", "ReceptorDensityFingerprint"])
        elif kind == "gene":
            cands.extend(["gene expressions", "GeneExpressions"])
        elif kind == "connectivity":
            cands.extend(["StreamlineCounts", "streamline counts"])
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
        out: List[Any] = []

        # Preferred path: search within the chosen parcellation.
        try:
            if hasattr(self.parcellation, "find"):
                found = self.parcellation.find(query)
                if found:
                    out.extend(list(found))
        except Exception:
            pass

        # Atlas-level fallback.
        try:
            if hasattr(self.atlas, "find_regions"):
                found = self.atlas.find_regions(
                    query,
                    all_versions=False,
                    filter_children=False,
                    find_topmost=False,
                )
                if found:
                    out.extend(list(found))
        except Exception:
            pass

        dedup: List[Any] = []
        seen: set[str] = set()
        for region in out:
            parcellation_name = getattr(getattr(region, "parcellation", None), "name", "")
            if "julich" not in str(parcellation_name).lower() and out:
                # Keep only Julich hits when mixed sources are present.
                continue
            ident = getattr(region, "identifier", None) or self._name_of(region)
            if ident in seen:
                continue
            seen.add(str(ident))
            dedup.append(region)
        return dedup

    def _region_rank(self, region: Any) -> Tuple[int, int, int, int]:
        name = self._name_of(region).lower()
        left_bonus = 0 if "left" in name else 1
        right_penalty = 1 if "right" in name else 0
        generic_penalty = 1 if name in {"amygdala", "hippocampus", "thalamus", "striatum"} else 0
        proxy_penalty = 1 if "area" not in name and all(k not in name for k in ("amygdala", "hippocampus", "thalamus")) else 0
        return (left_bonus, right_penalty, generic_penalty, proxy_penalty)

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
            try:
                if hasattr(self.parcellation, "find"):
                    matches = list(self.parcellation.find(spec))
                    if matches:
                        matches = sorted(matches, key=self._region_rank)
                        return matches[0]
            except Exception:
                pass
            matches = self._julich_matches(spec)
            if matches:
                matches = sorted(matches, key=self._region_rank)
                return matches[0]
        return None

    def suggest_regions(self, keyword: str, limit: int = 25) -> pd.DataFrame:
        rows = []
        seen: set[Tuple[str, str, str]] = set()
        for region in sorted(self._julich_matches(keyword), key=self._region_rank):
            row = (
                self._name_of(region),
                str(getattr(region, "identifier", "") or ""),
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
        try:
            centroid_xyz = tuple(float(x) for x in centroid) if centroid is not None else None
        except Exception:
            centroid_xyz = None
        volume_mm3 = getattr(main, "volume", None)
        try:
            volume_value = float(volume_mm3) if volume_mm3 is not None else None
        except Exception:
            volume_value = None
        return centroid_xyz, volume_value

    @staticmethod
    def _to_dataframe(data: Any) -> pd.DataFrame:
        if data is None:
            return pd.DataFrame()
        if isinstance(data, pd.DataFrame):
            return data.copy()
        if isinstance(data, pd.Series):
            name = data.name if data.name is not None else "value"
            df = data.to_frame(name=name).reset_index()
            if len(df.columns) == 2:
                df.columns = ["index", name]
            return df
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()

    def _receptor_table(self, region: Any) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("receptor"))
        if not feats:
            return pd.DataFrame()
        try:
            df = self._to_dataframe(getattr(feats[0], "data", None)).reset_index(drop=True)
        except Exception:
            return pd.DataFrame()
        if df.empty:
            return df
        if "index" in df.columns and "receptor" not in df.columns:
            df = df.rename(columns={"index": "receptor"})
        return df

    def _gene_table(self, region: Any, genes: Sequence[str]) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("gene"), gene=list(genes))
        if not feats:
            return pd.DataFrame()
        df = self._to_dataframe(getattr(feats[0], "data", None)).reset_index(drop=True)
        if df.empty:
            return df
        lower_cols = {str(c).lower(): c for c in df.columns}
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
                .reset_index(drop=True)
            )
        return df

    def _pick_connectivity_feature(self, features: Sequence[Any]) -> Optional[Any]:
        if not features:
            return None
        preferred = [
            f
            for f in features
            if str(getattr(f, "cohort", "") or "").lower() == self.connectivity_cohort.lower()
        ]
        return preferred[0] if preferred else features[0]

    def _get_connectivity_matrix(self) -> pd.DataFrame:
        if self._connectivity_matrix is not None:
            return self._connectivity_matrix

        feats = self._safe_features_any(self.parcellation, self._modality_candidates("connectivity"))
        feature = self._pick_connectivity_feature(feats)
        if feature is None:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

        data = getattr(feature, "data", None)
        df = self._to_dataframe(data)
        if isinstance(data, pd.DataFrame) and not data.empty:
            self._connectivity_matrix = data.copy()
            return self._connectivity_matrix
        if not df.empty and df.shape[0] > 1 and df.shape[1] > 1:
            self._connectivity_matrix = df.copy()
            return self._connectivity_matrix

        # Fallback for compound-feature style access.
        try:
            subfeature = feature[0]
            subdata = getattr(subfeature, "data", None)
            if isinstance(subdata, pd.DataFrame):
                self._connectivity_matrix = subdata.copy()
                return self._connectivity_matrix
            subdf = self._to_dataframe(subdata)
            self._connectivity_matrix = subdf.copy()
            return self._connectivity_matrix
        except Exception:
            self._connectivity_matrix = pd.DataFrame()
            return self._connectivity_matrix

    def _match_region_label(self, labels: Sequence[Any], region: Any) -> Optional[Any]:
        region_name = self._name_of(region)
        exact = [x for x in labels if self._name_of(x) == region_name]
        if exact:
            return exact[0]
        rn = region_name.lower()
        fuzzy = [x for x in labels if rn in self._name_of(x).lower() or self._name_of(x).lower() in rn]
        return fuzzy[0] if fuzzy else None

    def _connectivity_profile_from_matrix(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        matrix = self._get_connectivity_matrix()
        if matrix.empty:
            return pd.DataFrame()

        idx_match = self._match_region_label(list(matrix.index), region)
        col_match = self._match_region_label(list(matrix.columns), region)
        try:
            if idx_match is not None:
                series = matrix.loc[idx_match]
            elif col_match is not None:
                series = matrix[col_match]
            else:
                return pd.DataFrame()
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            df = series.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            df = df[df["connected_region"] != region.name].head(max_rows)
            return df.reset_index(drop=True)
        except Exception:
            return pd.DataFrame()

    def _connectivity_profile_from_region_feature(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        feats = self._safe_features_any(region, self._modality_candidates("connectivity"))
        feature = self._pick_connectivity_feature(feats)
        if feature is None:
            return pd.DataFrame()
        data = getattr(feature, "data", None)
        if isinstance(data, pd.Series):
            df = data.sort_values(ascending=False).reset_index()
            df.columns = ["connected_region", "value"]
            df["connected_region"] = df["connected_region"].map(self._name_of)
            return df.head(max_rows).reset_index(drop=True)
        if isinstance(data, pd.DataFrame):
            idx_match = self._match_region_label(list(data.index), region)
            col_match = self._match_region_label(list(data.columns), region)
            try:
                if idx_match is not None:
                    series = data.loc[idx_match]
                elif col_match is not None:
                    series = data[col_match]
                else:
                    return pd.DataFrame()
                df = series.sort_values(ascending=False).reset_index()
                df.columns = ["connected_region", "value"]
                df["connected_region"] = df["connected_region"].map(self._name_of)
                df = df[df["connected_region"] != region.name].head(max_rows)
                return df.reset_index(drop=True)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def _connectivity_profile(self, region: Any, max_rows: int = 15) -> pd.DataFrame:
        df = self._connectivity_profile_from_matrix(region, max_rows=max_rows)
        if not df.empty:
            return df
        return self._connectivity_profile_from_region_feature(region, max_rows=max_rows)

    def circuit_connectivity(self) -> pd.DataFrame:
        """
        Return long-form pairwise connectivity between resolved circuit nodes.

        The output is empty when the environment lacks compatible connectivity
        features. This is normal and should not be treated as an error.
        """
        rows: List[Dict[str, Any]] = []
        matrix = self._get_connectivity_matrix()

        if not matrix.empty and self.region_objects:
            idx_labels = list(matrix.index)
            col_labels = list(matrix.columns)
            mapped_labels = {
                key: (
                    self._match_region_label(idx_labels, region),
                    self._match_region_label(col_labels, region),
                )
                for key, region in self.region_objects.items()
            }
            for src_key, src_region in self.region_objects.items():
                idx_label, col_label = mapped_labels.get(src_key, (None, None))
                if idx_label is None and col_label is None:
                    continue
                for tgt_key, tgt_region in self.region_objects.items():
                    if src_key == tgt_key:
                        continue
                    tgt_idx_label, tgt_col_label = mapped_labels.get(tgt_key, (None, None))
                    value = None
                    try:
                        if idx_label is not None and tgt_col_label is not None:
                            value = matrix.loc[idx_label, tgt_col_label]
                        elif col_label is not None and tgt_idx_label is not None:
                            value = matrix.loc[tgt_idx_label, col_label]
                        elif idx_label is not None and tgt_idx_label is not None:
                            value = matrix.loc[idx_label, tgt_idx_label]
                        elif col_label is not None and tgt_col_label is not None:
                            value = matrix.loc[col_label, tgt_col_label]
                    except Exception:
                        value = None
                    try:
                        value = float(value) if value is not None else None
                    except Exception:
                        value = None
                    if value is None:
                        continue
                    rows.append(
                        {
                            "source_key": src_key,
                            "source_region": src_region.name,
                            "target_key": tgt_key,
                            "target_region": tgt_region.name,
                            "value": value,
                        }
                    )
            if rows:
                return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

        # Fallback: reconstruct partial long-form connectivity from regional profiles.
        for src_key, profile in self.connectivity_profiles.items():
            if profile.empty:
                continue
            for tgt_key, tgt_region in self.region_objects.items():
                if src_key == tgt_key:
                    continue
                try:
                    mask = profile["connected_region"].astype(str).str.lower().eq(tgt_region.name.lower())
                    if not mask.any():
                        mask = profile["connected_region"].astype(str).str.lower().str.contains(
                            tgt_region.name.lower(),
                            regex=False,
                        )
                    if not mask.any():
                        continue
                    value = float(profile.loc[mask, "value"].iloc[0])
                except Exception:
                    continue
                rows.append(
                    {
                        "source_key": src_key,
                        "source_region": self.region_objects[src_key].name,
                        "target_key": tgt_key,
                        "target_region": tgt_region.name,
                        "value": value,
                    }
                )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)

    def build(self, gene_panel: Sequence[str] = DEFAULT_GENE_PANEL, connectivity_rows: int = 15) -> dict:
        nodes: List[Dict[str, Any]] = []
        self.region_objects = {}
        self.receptors = {}
        self.genes = {}
        self.connectivity_profiles = {}

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
                    "is_proxy": False,
                }
            )

        for key, candidates in self.region_candidates.items():
            region = self._resolve_region(candidates)
            is_proxy = key.endswith("_proxy")
            description = self.region_node_descriptions.get(key, "Atlas-backed circuit node")

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
                        "description": f"{description} (unresolved in this environment)",
                        "atlas_region": None,
                        "region_identifier": None,
                        "centroid_mni": None,
                        "volume_mm3": None,
                        "feature_summary": "unresolved",
                        "is_proxy": is_proxy,
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
                    "description": description,
                    "atlas_region": region.name,
                    "region_identifier": getattr(region, "identifier", None),
                    "centroid_mni": centroid_mni,
                    "volume_mm3": volume_mm3,
                    "feature_summary": (
                        f"receptors={'yes' if not receptor_df.empty else 'no'}; "
                        f"genes={'yes' if not gene_df.empty else 'no'}; "
                        f"connectivity={'yes' if not conn_df.empty else 'no'}"
                    ),
                    "is_proxy": is_proxy,
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
                    "is_proxy": False,
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
                    "is_proxy": False,
                }
            )

        self.nodes_df = pd.DataFrame(nodes)
        self.edges_df = pd.DataFrame(self.edge_table)
        circuit_connectivity = self.circuit_connectivity()
        return {
            "nodes": self.nodes_df,
            "edges": self.edges_df,
            "regions": self.region_objects,
            "receptors": self.receptors,
            "genes": self.genes,
            "connectivity_profiles": self.connectivity_profiles,
            "circuit_connectivity": circuit_connectivity,
        }

    def _statistical_map(self):
        if self._pmap is not None:
            return self._pmap
        last_exc: Optional[Exception] = None
        for spec in self._parcellation_specs():
            try:
                with siibra.QUIET:
                    self._pmap = siibra.get_map(
                        parcellation=spec,
                        space=self.assignment_space,
                        maptype="statistical",
                    )
                return self._pmap
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError("Could not obtain a statistical parcellation map.") from last_exc

    def assign_mni_point(self, xyz: Sequence[float]) -> pd.DataFrame:
        pmap = self._statistical_map()
        point = siibra.Point(tuple(xyz), space=self.assignment_space)
        with siibra.QUIET:
            assignments = pmap.assign(point)
        for candidate in ("map value", "correlation", "intersection over union"):
            if candidate in assignments.columns:
                assignments = assignments.sort_values(candidate, ascending=False)
                break
        return assignments.reset_index(drop=True)

    def region_mask(self, node_key: str, fetch: bool = False):
        """
        Return a region mask/volume for an atlas-backed node.

        Parameters
        ----------
        node_key:
            Region node key, for example "amygdala" or "frontal_lobe_proxy".
        fetch:
            If True, fetch and return the concrete image object when possible.
        """
        region = self.region_objects.get(node_key)
        if region is None:
            return None
        mask = None
        try:
            mask = region.get_regional_mask(self.space, maptype="labelled")
        except Exception:
            try:
                mask = region.get_regional_map(self.space, maptype="statistical")
            except Exception:
                return None
        if fetch:
            try:
                return mask.fetch()
            except Exception:
                return mask
        return mask

    def simulate(
        self,
        *,
        genetic_liability: float = 0.0,
        vascular_risk_load: float = 0.0,
        endocrine_dysregulation: float = 0.0,
        medication_burden: float = 0.0,
        central_neurological_insult: float = 0.0,
        peripheral_neuropathy: float = 0.0,
        depression_load: float = 0.0,
        seizure_limbic_instability: float = 0.0,
        vascular_health_support: float = 0.0,
        endocrine_treatment_support: float = 0.0,
        medication_review_support: float = 0.0,
        rehabilitation_support: float = 0.0,
        mood_treatment_support: float = 0.0,
    ) -> Dict[str, pd.Series]:
        """
        Transparent normalized simulator.

        The calculation order is intentionally acyclic and readable:
        inputs -> latent biology -> regional state -> symptoms -> phenotype summaries

        Every value is clipped into [0, 1]. Protective terms subtract from burden.
        """

        inputs = pd.Series(
            {
                "genetic_liability": self._clip01(genetic_liability),
                "vascular_risk_load": self._clip01(vascular_risk_load),
                "endocrine_dysregulation": self._clip01(endocrine_dysregulation),
                "medication_burden": self._clip01(medication_burden),
                "central_neurological_insult": self._clip01(central_neurological_insult),
                "peripheral_neuropathy": self._clip01(peripheral_neuropathy),
                "depression_load": self._clip01(depression_load),
                "seizure_limbic_instability": self._clip01(seizure_limbic_instability),
                "vascular_health_support": self._clip01(vascular_health_support),
                "endocrine_treatment_support": self._clip01(endocrine_treatment_support),
                "medication_review_support": self._clip01(medication_review_support),
                "rehabilitation_support": self._clip01(rehabilitation_support),
                "mood_treatment_support": self._clip01(mood_treatment_support),
            },
            dtype=float,
        )

        latents = pd.Series(dtype=float)
        latents["neurovascular_cavernosal_impairment"] = self._clip01(
            0.42 * inputs["vascular_risk_load"]
            + 0.16 * inputs["genetic_liability"]
            + 0.10 * inputs["central_neurological_insult"]
            + 0.08 * inputs["depression_load"]
            - 0.24 * inputs["vascular_health_support"]
            - 0.08 * inputs["rehabilitation_support"]
        )
        latents["autonomic_erection_drive_failure"] = self._clip01(
            0.28 * inputs["central_neurological_insult"]
            + 0.24 * inputs["peripheral_neuropathy"]
            + 0.18 * inputs["medication_burden"]
            + 0.12 * inputs["depression_load"]
            + 0.10 * inputs["seizure_limbic_instability"]
            - 0.18 * inputs["rehabilitation_support"]
            - 0.08 * inputs["medication_review_support"]
        )
        latents["sympathetic_detumescence_bias"] = self._clip01(
            0.22 * inputs["depression_load"]
            + 0.18 * inputs["medication_burden"]
            + 0.16 * inputs["central_neurological_insult"]
            + 0.10 * inputs["vascular_risk_load"]
            - 0.12 * inputs["mood_treatment_support"]
            - 0.10 * inputs["medication_review_support"]
        )
        latents["dopaminergic_motivation_deficit"] = self._clip01(
            0.32 * inputs["depression_load"]
            + 0.20 * inputs["medication_burden"]
            + 0.12 * inputs["central_neurological_insult"]
            + 0.10 * inputs["genetic_liability"]
            - 0.18 * inputs["mood_treatment_support"]
        )
        latents["serotonergic_inhibitory_pressure"] = self._clip01(
            0.42 * inputs["medication_burden"]
            + 0.14 * inputs["depression_load"]
            + 0.08 * inputs["central_neurological_insult"]
            - 0.22 * inputs["medication_review_support"]
            - 0.06 * inputs["mood_treatment_support"]
        )
        latents["endocrine_libido_deficit"] = self._clip01(
            0.52 * inputs["endocrine_dysregulation"]
            + 0.10 * inputs["vascular_risk_load"]
            + 0.08 * inputs["genetic_liability"]
            - 0.30 * inputs["endocrine_treatment_support"]
        )
        latents["frontolimbic_arousal_disruption"] = self._clip01(
            0.24 * inputs["central_neurological_insult"]
            + 0.22 * inputs["depression_load"]
            + 0.16 * inputs["vascular_risk_load"]
            + 0.14 * latents["serotonergic_inhibitory_pressure"]
            + 0.10 * latents["dopaminergic_motivation_deficit"]
            + 0.10 * inputs["seizure_limbic_instability"]
            - 0.18 * inputs["mood_treatment_support"]
        )
        latents["limbic_temporal_disruption"] = self._clip01(
            0.36 * inputs["seizure_limbic_instability"]
            + 0.18 * inputs["central_neurological_insult"]
            + 0.14 * inputs["depression_load"]
            + 0.12 * inputs["vascular_risk_load"]
            - 0.12 * inputs["rehabilitation_support"]
        )
        latents["subcortical_vascular_disconnection"] = self._clip01(
            0.38 * inputs["vascular_risk_load"]
            + 0.24 * inputs["central_neurological_insult"]
            + 0.12 * inputs["genetic_liability"]
            - 0.24 * inputs["vascular_health_support"]
        )
        latents["peripheral_neurogenic_signal_loss"] = self._clip01(
            0.48 * inputs["peripheral_neuropathy"]
            + 0.18 * inputs["central_neurological_insult"]
            + 0.10 * inputs["vascular_risk_load"]
            - 0.18 * inputs["rehabilitation_support"]
        )

        regional_state = pd.Series(dtype=float)
        regional_state["amygdala"] = self._clip01(
            0.45 * latents["frontolimbic_arousal_disruption"]
            + 0.35 * latents["limbic_temporal_disruption"]
            + 0.10 * inputs["depression_load"]
        )
        regional_state["hippocampus"] = self._clip01(
            0.42 * latents["limbic_temporal_disruption"]
            + 0.30 * latents["frontolimbic_arousal_disruption"]
            + 0.12 * latents["subcortical_vascular_disconnection"]
        )
        regional_state["frontal_lobe_proxy"] = self._clip01(
            0.52 * latents["frontolimbic_arousal_disruption"]
            + 0.30 * latents["subcortical_vascular_disconnection"]
            + 0.12 * latents["serotonergic_inhibitory_pressure"]
        )
        regional_state["temporal_lobe_proxy"] = self._clip01(
            0.50 * latents["limbic_temporal_disruption"]
            + 0.22 * inputs["central_neurological_insult"]
            + 0.12 * regional_state["hippocampus"]
        )
        regional_state["basal_ganglia_proxy"] = self._clip01(
            0.44 * latents["subcortical_vascular_disconnection"]
            + 0.28 * latents["dopaminergic_motivation_deficit"]
            + 0.14 * inputs["central_neurological_insult"]
        )
        regional_state["thalamus_proxy"] = self._clip01(
            0.50 * latents["subcortical_vascular_disconnection"]
            + 0.24 * inputs["central_neurological_insult"]
            + 0.16 * latents["autonomic_erection_drive_failure"]
        )

        symptoms = pd.Series(dtype=float)
        symptoms["diminished_sexual_desire"] = self._clip01(
            0.40 * latents["endocrine_libido_deficit"]
            + 0.28 * latents["dopaminergic_motivation_deficit"]
            + 0.12 * regional_state["amygdala"]
            + 0.10 * regional_state["hippocampus"]
            + 0.10 * latents["serotonergic_inhibitory_pressure"]
        )
        symptoms["impaired_erection_initiation"] = self._clip01(
            0.34 * latents["autonomic_erection_drive_failure"]
            + 0.24 * latents["peripheral_neurogenic_signal_loss"]
            + 0.18 * latents["frontolimbic_arousal_disruption"]
            + 0.12 * latents["endocrine_libido_deficit"]
            + 0.10 * latents["serotonergic_inhibitory_pressure"]
        )
        symptoms["impaired_erection_maintenance"] = self._clip01(
            0.36 * latents["neurovascular_cavernosal_impairment"]
            + 0.22 * latents["sympathetic_detumescence_bias"]
            + 0.18 * latents["autonomic_erection_drive_failure"]
            + 0.12 * latents["subcortical_vascular_disconnection"]
            + 0.08 * latents["peripheral_neurogenic_signal_loss"]
        )
        symptoms["neurogenic_ed_expression"] = self._clip01(
            0.32 * inputs["central_neurological_insult"]
            + 0.26 * latents["peripheral_neurogenic_signal_loss"]
            + 0.18 * regional_state["thalamus_proxy"]
            + 0.14 * regional_state["basal_ganglia_proxy"]
            + 0.10 * regional_state["temporal_lobe_proxy"]
        )
        symptoms["vascular_depressive_ed_expression"] = self._clip01(
            0.34 * latents["neurovascular_cavernosal_impairment"]
            + 0.20 * latents["subcortical_vascular_disconnection"]
            + 0.18 * symptoms["diminished_sexual_desire"]
            + 0.14 * inputs["depression_load"]
            + 0.14 * latents["serotonergic_inhibitory_pressure"]
        )

        phenotypes = pd.Series(dtype=float)
        phenotypes["overall_ed_severity"] = self._clip01(
            0.30 * symptoms["impaired_erection_initiation"]
            + 0.30 * symptoms["impaired_erection_maintenance"]
            + 0.20 * symptoms["diminished_sexual_desire"]
            + 0.10 * symptoms["neurogenic_ed_expression"]
            + 0.10 * symptoms["vascular_depressive_ed_expression"]
        )
        phenotypes["neurogenic_profile"] = self._clip01(
            0.40 * symptoms["neurogenic_ed_expression"]
            + 0.25 * latents["autonomic_erection_drive_failure"]
            + 0.20 * latents["peripheral_neurogenic_signal_loss"]
            + 0.15 * inputs["central_neurological_insult"]
        )
        phenotypes["vascular_endocrine_profile"] = self._clip01(
            0.36 * latents["neurovascular_cavernosal_impairment"]
            + 0.28 * latents["endocrine_libido_deficit"]
            + 0.16 * symptoms["impaired_erection_maintenance"]
            + 0.12 * inputs["vascular_risk_load"]
            + 0.08 * inputs["endocrine_dysregulation"]
        )
        phenotypes["limbic_medication_profile"] = self._clip01(
            0.30 * latents["limbic_temporal_disruption"]
            + 0.28 * latents["serotonergic_inhibitory_pressure"]
            + 0.20 * latents["dopaminergic_motivation_deficit"]
            + 0.12 * inputs["medication_burden"]
            + 0.10 * symptoms["diminished_sexual_desire"]
        )
        phenotypes["stroke_subcortical_profile"] = self._clip01(
            0.42 * latents["subcortical_vascular_disconnection"]
            + 0.24 * inputs["central_neurological_insult"]
            + 0.18 * regional_state["thalamus_proxy"]
            + 0.16 * symptoms["vascular_depressive_ed_expression"]
        )

        return {
            "inputs": inputs,
            "latents": latents,
            "regional_state": regional_state,
            "symptoms": symptoms,
            "phenotypes": phenotypes,
        }


if __name__ == "__main__":
    model = ErectileDisorderModel()
    scaffold = model.build()

    print("\n=== NODE SUMMARY ===")
    print(
        scaffold["nodes"][
            [
                "key",
                "node_type",
                "atlas_region",
                "feature_summary",
                "is_proxy",
            ]
        ].to_string(index=False)
    )

    print("\n=== EDGE SUMMARY ===")
    print(scaffold["edges"][["source", "target", "relation", "erectile_disorder_change"]].head(20).to_string(index=False))

    print("\n=== REGION FEATURE SNAPSHOT ===")
    for key in ("amygdala", "hippocampus", "frontal_lobe_proxy"):
        print(f"\n[{key}]")
        region_rows = scaffold["nodes"].query("key == @key")[["atlas_region", "centroid_mni", "feature_summary"]]
        print(region_rows.to_string(index=False))
        if not scaffold["receptors"].get(key, pd.DataFrame()).empty:
            print("receptors:")
            print(scaffold["receptors"][key].head().to_string(index=False))
        if not scaffold["genes"].get(key, pd.DataFrame()).empty:
            print("genes:")
            print(scaffold["genes"][key].head().to_string(index=False))
        if not scaffold["connectivity_profiles"].get(key, pd.DataFrame()).empty:
            print("connectivity:")
            print(scaffold["connectivity_profiles"][key].head().to_string(index=False))

    print("\n=== CIRCUIT CONNECTIVITY ===")
    if scaffold["circuit_connectivity"].empty:
        print("No pairwise circuit connectivity available in this runtime.")
    else:
        print(scaffold["circuit_connectivity"].head(15).to_string(index=False))

    example = model.simulate(
        genetic_liability=0.45,
        vascular_risk_load=0.70,
        endocrine_dysregulation=0.50,
        medication_burden=0.40,
        central_neurological_insult=0.35,
        peripheral_neuropathy=0.30,
        depression_load=0.55,
        seizure_limbic_instability=0.10,
        vascular_health_support=0.20,
        endocrine_treatment_support=0.15,
        medication_review_support=0.10,
        rehabilitation_support=0.20,
        mood_treatment_support=0.20,
    )

    print("\n=== SIMULATED SYMPTOMS ===")
    print(example["symptoms"].sort_values(ascending=False).to_string())

    print("\n=== SIMULATED PHENOTYPES ===")
    print(example["phenotypes"].sort_values(ascending=False).to_string())

    # Optional manual testing examples:
    # assignments = model.assign_mni_point((-8.5, -82.4, 2.0))
    # print(assignments.head().to_string(index=False))
    # mask_img = model.region_mask("amygdala", fetch=True)
