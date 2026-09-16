"""Conservative query rewriting for policy concepts and missing aspects."""

from __future__ import annotations

from dataclasses import dataclass

from .models import RetrievalQuery


@dataclass(frozen=True)
class RewriteCandidate:
    text: str
    reason: str


class QueryRewriter:
    def rewrite(self, query: RetrievalQuery, missing_aspects: list[str] | None = None) -> list[RewriteCandidate]:
        normalized = query.text.replace(" ", "")
        terms: list[str] = []
        if any(word in normalized for word in ("杯", "耳机", "鞋", "键盘", "商品")):
            terms.append("商品售后政策")
        if query.metadata_filter.get("product_type") == "custom" or "定制" in normalized:
            terms.extend(["定制商品", "例外规则"])
        else:
            terms.append("普通商品")
        if query.metadata_filter.get("reason") == "quality_issue" or any(
            word in normalized for word in ("坏", "质量", "损坏", "故障")
        ):
            terms.extend(["质量问题", "三十天"])
        elif any(word in normalized for word in ("无理由", "不想要", "不合适", "能退") ):
            terms.extend(["无理由退货", "签收", "七天"])
        if missing_aspects:
            terms.extend(missing_aspects)
        rewritten = " ".join(dict.fromkeys([query.text] + terms))
        if rewritten == query.text:
            return []
        return [
            RewriteCandidate(
                text=rewritten,
                reason="补充商品类型、原因和政策时限等检索维度",
            )
        ]

    def as_query(
        self, query: RetrievalQuery, missing_aspects: list[str] | None = None
    ) -> RetrievalQuery:
        candidates = self.rewrite(query, missing_aspects)
        return RetrievalQuery(
            text=candidates[0].text if candidates else query.text,
            as_of=query.as_of,
            metadata_filter=dict(query.metadata_filter),
            required_tags=query.required_tags,
            pass_number=query.pass_number + 1,
        )
