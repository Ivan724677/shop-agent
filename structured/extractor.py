"""确定性高精度提取器（阶段三感知层的正则分支）。

与 LLM 语义解析器并行工作，专门负责低歧义、高风险信号的提取：
精确订单号/商品号、整句精确匹配的确认/取消词、商品别名、
否定范围（"I001 不退"）、咨询意图、相对订单指代。

产出的 DeterministicSignals 置信度恒为 1.0，在 reducer 中优先于 LLM 候选。
"""

from __future__ import annotations

import re

from repositories import InMemoryStore

from .models import DeterministicSignals, Intent


ORDER_PATTERN = re.compile(r"\bO\d{5,}\b", flags=re.IGNORECASE)
ITEM_PATTERN = re.compile(r"\bI\d{3,}\b", flags=re.IGNORECASE)


class DeterministicExtractor:
    """Extract IDs and safety-critical explicit language with high precision."""

    DEFAULT_ALIASES = {
        "蓝牙耳机": ["蓝牙耳机", "耳机"],
        "运动鞋": ["运动鞋", "鞋子", "鞋"],
        "定制马克杯": ["定制马克杯", "马克杯", "杯子", "杯"],
        "机械键盘": ["机械键盘", "键盘"],
        "企业定制礼盒": ["企业定制礼盒", "定制礼盒", "礼盒"],
    }

    def __init__(self, store: InMemoryStore | None = None) -> None:
        self.store = store or InMemoryStore()
        known_names = {product.name for product in self.store.products.values()}
        self.aliases = {
            name: self.DEFAULT_ALIASES.get(name, [name]) for name in known_names
        }

    def extract(self, text: str) -> DeterministicSignals:
        normalized = re.sub(r"\s+", "", text)
        order_match = ORDER_PATTERN.search(text)
        all_item_ids = sorted({match.upper() for match in ITEM_PATTERN.findall(text)})
        excluded_item_ids = [
            item_id
            for item_id in all_item_ids
            if re.search(
                rf"(?:{re.escape(item_id)}(?:不退|不要退|别退|不处理|保留)|"
                rf"(?:不退|不要退|别退|不处理|保留){re.escape(item_id)})",
                normalized,
                flags=re.IGNORECASE,
            )
        ]
        item_ids = sorted(set(all_item_ids) - set(excluded_item_ids))
        confirmation = self._confirmation(normalized)
        explicit_intent = self._intent(normalized, confirmation)
        mentions, excluded = self._product_mentions(normalized)
        reason = (
            "quality_issue"
            if any(word in normalized for word in ("质量问题", "坏了", "损坏", "故障", "破损"))
            else None
        )
        consultation_only = any(
            phrase in normalized
            for phrase in ("只是问问", "先问问", "先看看", "能不能退", "可以退吗", "多少钱")
        ) and confirmation != "confirm"
        return DeterministicSignals(
            order_id=order_match.group(0).upper() if order_match else None,
            item_ids=item_ids,
            excluded_item_ids=excluded_item_ids,
            product_mentions=mentions,
            excluded_product_mentions=excluded,
            explicit_intent=explicit_intent,
            reason=reason,
            consultation_only=consultation_only,
            confirmation=confirmation,
            relative_order_reference=any(
                phrase in normalized for phrase in ("上一单", "最近一单", "刚才那单", "那个订单")
            ),
        )

    @staticmethod
    def _confirmation(normalized: str) -> str:
        if re.fullmatch(r"(?:确认|确定|同意|执行|确认退货|确认退款|可以执行)", normalized):
            return "confirm"
        if re.fullmatch(r"(?:取消|算了|不退了|别处理了|不要了)", normalized):
            return "reject"
        return "none"

    @staticmethod
    def _intent(normalized: str, confirmation: str) -> Intent:
        if confirmation != "none":
            return Intent.KEEP_CURRENT
        if any(word in normalized for word in ("人工", "客服人员", "投诉", "转接")):
            return Intent.HUMAN_HANDOFF
        if any(word in normalized for word in ("退货", "退款", "退掉", "申请售后", "能退", "可以退")):
            return Intent.REQUEST_RETURN
        if "退" in normalized and not any(word in normalized for word in ("退回物流", "物流退回")):
            return Intent.REQUEST_RETURN
        if any(word in normalized for word in ("物流", "到哪", "包裹", "快递", "运到")):
            return Intent.QUERY_LOGISTICS
        if any(
            word in normalized
            for word in (
                "查订单",
                "查询订单",
                "订单状态",
                "买了什么",
                "有哪些商品",
                "查一下",
                "查询",
            )
        ):
            return Intent.QUERY_ORDER
        return Intent.UNKNOWN

    def _product_mentions(self, normalized: str) -> tuple[list[str], list[str]]:
        mentions: list[str] = []
        excluded: list[str] = []
        for canonical_name, aliases in self.aliases.items():
            matched_alias = next((alias for alias in aliases if alias in normalized), None)
            if not matched_alias:
                continue
            mentions.append(canonical_name)
            negative = re.search(
                rf"(?:{re.escape(matched_alias)}(?:不退|不要退|别退|不处理|保留)|"
                rf"(?:不退|不要退|别退|不处理|保留){re.escape(matched_alias)})",
                normalized,
            )
            if negative:
                excluded.append(canonical_name)
        return sorted(set(mentions) - set(excluded)), sorted(set(excluded))
