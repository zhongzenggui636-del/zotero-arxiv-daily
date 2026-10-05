from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import requests
from loguru import logger

from .base import BaseRetriever, register_retriever
from ..protocol import Paper


OPENALEX_API = "https://api.openalex.org/works"
REQUEST_TIMEOUT = 30


# ---------------------------------------------------------
# 顶会 / 顶刊白名单
# ---------------------------------------------------------

VENUE_RULES = [
    # ======================
    # Computer Vision
    # ======================
    (
        "顶会",
        "CVPR",
        [
            "computer vision and pattern recognition",
            "cvpr",
        ],
    ),
    (
        "顶会",
        "ICCV",
        [
            "international conference on computer vision",
            "iccv",
        ],
    ),
    (
        "顶会",
        "ECCV",
        [
            "european conference on computer vision",
            "eccv",
        ],
    ),

    # ======================
    # NLP
    # ======================
    (
        "顶会",
        "ACL",
        [
            "annual meeting of the association for computational linguistics",
        ],
    ),
    (
        "顶会",
        "EMNLP",
        [
            "empirical methods in natural language processing",
            "emnlp",
        ],
    ),
    (
        "顶会",
        "NAACL",
        [
            "north american chapter of the association for computational linguistics",
            "naacl",
        ],
    ),

    # ======================
    # Multimedia
    # ======================
    (
        "顶会",
        "ACM MM",
        [
            "acm international conference on multimedia",
            "acm multimedia",
        ],
    ),

    # ======================
    # AI
    # ======================
    (
        "顶会",
        "AAAI",
        [
            "aaai conference on artificial intelligence",
        ],
    ),
    (
        "顶会",
        "IJCAI",
        [
            "international joint conference on artificial intelligence",
            "ijcai",
        ],
    ),

    # ======================
    # Machine Learning
    # ======================
    (
        "顶会",
        "NeurIPS",
        [
            "advances in neural information processing systems",
            "neural information processing systems",
            "neurips",
        ],
    ),
    (
        "顶会",
        "ICML",
        [
            "international conference on machine learning",
            "icml",
        ],
    ),
    (
        "顶会",
        "ICLR",
        [
            "international conference on learning representations",
            "iclr",
        ],
    ),

    # ======================
    # Journals
    # ======================
    (
        "顶刊",
        "IEEE TAFFC",
        [
            "ieee transactions on affective computing",
        ],
    ),
    (
        "顶刊",
        "IEEE TMM",
        [
            "ieee transactions on multimedia",
        ],
    ),
    (
        "顶刊",
        "IEEE TPAMI",
        [
            "ieee transactions on pattern analysis and machine intelligence",
        ],
    ),
    (
        "顶刊",
        "IEEE TIP",
        [
            "ieee transactions on image processing",
        ],
    ),
    (
        "顶刊",
        "IJCV",
        [
            "international journal of computer vision",
        ],
    ),
    (
        "顶刊",
        "Information Fusion",
        [
            "information fusion",
        ],
    ),
    (
        "顶刊",
        "Pattern Recognition",
        [
            "pattern recognition",
        ],
    ),
]


DEFAULT_KEYWORDS = [
    "emotion",
    "emotional",
    "affective",
    "affect",
    "valence arousal",
    "continuous emotion",
    "emotion understanding",
    "emotion generation",
    "multimodal emotion",
    "emotion reasoning",
]


# ---------------------------------------------------------
# 工具函数
# ---------------------------------------------------------

def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """
    OpenAlex 的 abstract 是 inverted index：
    {
        "emotion": [0, 5],
        "recognition": [1],
        ...
    }

    将其恢复成普通文本。
    """
    if not inverted_index:
        return ""

    positions: list[tuple[int, str]] = []

    for word, indexes in inverted_index.items():
        for index in indexes:
            positions.append((index, word))

    positions.sort(key=lambda x: x[0])

    return " ".join(word for _, word in positions)


def _get_venue_name(work: dict[str, Any]) -> str:
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}

    return str(source.get("display_name") or "").strip()


def _match_venue(venue_name: str) -> tuple[str, str] | None:
    """
    返回：
        ("顶会", "ACL")
    或：
        ("顶刊", "IEEE TAFFC")
    """
    venue_lower = venue_name.lower()

    for venue_type, short_name, aliases in VENUE_RULES:
        for alias in aliases:
            if alias.lower() in venue_lower:
                return venue_type, short_name

    return None


def _topic_tags(title: str, abstract: str) -> list[str]:
    """
    给论文加科研方向标签。
    """
    text = f"{title} {abstract}".lower()
    tags: list[str] = []

    # 连续情感
    continuous_terms = [
        "valence",
        "arousal",
        "dominance",
        "vad",
        "continuous emotion",
        "continuous affect",
        "dimensional emotion",
        "affective manifold",
        "emotion trajectory",
    ]

    if any(term in text for term in continuous_terms):
        tags.append("连续情感")

    # 离散情感
    discrete_terms = [
        "discrete emotion",
        "emotion category",
        "emotion classification",
        "categorical emotion",
        "happy",
        "sadness",
        "anger",
        "fear",
        "disgust",
        "surprise",
    ]

    if any(term in text for term in discrete_terms):
        tags.append("离散情感")

    # 情感理解
    understanding_terms = [
        "emotion understanding",
        "emotion recognition",
        "emotion reasoning",
        "emotion perception",
        "affect recognition",
        "emotional understanding",
        "emotion classification",
    ]

    if any(term in text for term in understanding_terms):
        tags.append("理解")

    # 情感生成
    generation_terms = [
        "emotion generation",
        "emotional generation",
        "affective generation",
        "emotion-conditioned",
        "emotion guided generation",
        "emotion-guided",
        "text-to-image",
        "image generation",
        "diffusion",
    ]

    if any(term in text for term in generation_terms):
        tags.append("生成")

    # 多模态
    multimodal_terms = [
        "multimodal",
        "multi-modal",
        "vision-language",
        "vision language",
        "audio-visual",
        "audio visual",
    ]

    if any(term in text for term in multimodal_terms):
        tags.append("多模态")

    # MLLM
    mllm_terms = [
        "mllm",
        "multimodal large language model",
        "large vision-language model",
        "large vision language model",
    ]

    if any(term in text for term in mllm_terms):
        tags.append("MLLM")

    # Diffusion
    if "diffusion" in text:
        tags.append("Diffusion")

    return tags


def _looks_emotion_related(title: str, abstract: str) -> bool:
    """
    第二层保险：
    防止关键词搜索偶尔带进明显无关论文。
    """
    text = f"{title} {abstract}".lower()

    emotion_terms = [
        "emotion",
        "emotional",
        "affective",
        "affect recognition",
        "valence",
        "arousal",
        "dominance",
        "sentiment",
    ]

    return any(term in text for term in emotion_terms)


# ---------------------------------------------------------
# Retriever
# ---------------------------------------------------------

@register_retriever("top_venue")
class TopVenueRetriever(BaseRetriever):

    def __init__(self, config):
        super().__init__(config)

        self.lookback_days = int(
            self.retriever_config.get("lookback_days", 3)
        )

        self.per_query = int(
            self.retriever_config.get("per_query", 50)
        )

        configured_keywords = self.retriever_config.get(
            "keywords",
            None,
        )

        if configured_keywords:
            self.keywords = list(configured_keywords)
        else:
            self.keywords = DEFAULT_KEYWORDS

    def _retrieve_raw_papers(self) -> list[dict[str, Any]]:
        """
        从 OpenAlex 查询近期论文，
        然后只保留顶会 / 顶刊 + 情感相关论文。
        """

        today = date.today()
        start_date = today - timedelta(days=self.lookback_days)

        logger.info(
            f"Searching top-venue papers from "
            f"{start_date.isoformat()} to {today.isoformat()}"
        )

        seen_ids: set[str] = set()
        selected: list[dict[str, Any]] = []

        for keyword in self.keywords:

            logger.info(f"Searching OpenAlex keyword: {keyword}")

            params = {
                "search": keyword,
                "filter": (
                    f"from_publication_date:{start_date.isoformat()},"
                    f"to_publication_date:{today.isoformat()},"
                    f"has_abstract:true"
                ),
                "per-page": self.per_query,
            }

            try:
                response = requests.get(
                    OPENALEX_API,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                )

                response.raise_for_status()

                data = response.json()

            except Exception as exc:
                logger.warning(
                    f"OpenAlex request failed for keyword "
                    f"{keyword}: {exc}"
                )
                continue

            works = data.get("results", [])

            for work in works:

                work_id = str(work.get("id") or "")

                if not work_id:
                    continue

                if work_id in seen_ids:
                    continue

                venue_name = _get_venue_name(work)

                venue_match = _match_venue(venue_name)

                if venue_match is None:
                    continue

                title = str(
                    work.get("title")
                    or work.get("display_name")
                    or ""
                ).strip()

                abstract = _reconstruct_abstract(
                    work.get("abstract_inverted_index")
                )

                if not _looks_emotion_related(title, abstract):
                    continue

                seen_ids.add(work_id)

                venue_type, short_name = venue_match

                work["_zotero_arxiv_daily_venue_type"] = venue_type
                work["_zotero_arxiv_daily_venue_short"] = short_name
                work["_zotero_arxiv_daily_venue_full"] = venue_name

                selected.append(work)

        logger.info(
            f"Found {len(selected)} emotion-related "
            f"top venue/journal papers"
        )

        if self.config.executor.debug:
            selected = selected[:10]

        return selected

    def convert_to_paper(
        self,
        raw_paper: dict[str, Any],
    ) -> Paper | None:

        title = str(
            raw_paper.get("title")
            or raw_paper.get("display_name")
            or ""
        ).strip()

        abstract = _reconstruct_abstract(
            raw_paper.get("abstract_inverted_index")
        )

        if not title:
            return None

        venue_type = raw_paper.get(
            "_zotero_arxiv_daily_venue_type",
            "正式发表",
        )

        venue_short = raw_paper.get(
            "_zotero_arxiv_daily_venue_short",
            "Unknown",
        )

        publication_year = raw_paper.get("publication_year")

        venue_label = venue_short

        if publication_year:
            venue_label += f" {publication_year}"

        tags = _topic_tags(title, abstract)

        prefix_parts = [
            f"[{venue_type} · {venue_label}]"
        ]

        prefix_parts.extend(
            f"[{tag}]"
            for tag in tags
        )

        tagged_title = (
            " ".join(prefix_parts)
            + " "
            + title
        )

        # -------------------------
        # Authors
        # -------------------------

        authors: list[str] = []

        for authorship in raw_paper.get("authorships", []):
            author = authorship.get("author") or {}

            name = str(
                author.get("display_name") or ""
            ).strip()

            if name:
                authors.append(name)

        if not authors:
            authors = ["Unknown"]

        # -------------------------
        # URL / PDF
        # -------------------------

        primary_location = (
            raw_paper.get("primary_location")
            or {}
        )

        best_oa_location = (
            raw_paper.get("best_oa_location")
            or {}
        )

        doi = raw_paper.get("doi")

        url = (
            primary_location.get("landing_page_url")
            or doi
            or raw_paper.get("id")
            or ""
        )

        pdf_url = (
            best_oa_location.get("pdf_url")
            or primary_location.get("pdf_url")
            or url
        )

        return Paper(
            source="top_venue",
            title=tagged_title,
            authors=authors,
            abstract=abstract,
            url=url,
            pdf_url=pdf_url,
            # 第一版故意不抓全文：
            # 省 API token，也减少超时风险
            full_text=None,
        )
