# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Per-model capabilities for the Kiro ``additionalModelRequestFields`` field.

Kiro validates that field against a per-model JSON Schema and answers HTTP 400
for anything the schema does not allow, which fails the whole request. The
schema differs by model: which effort levels exist, whether ``thinking`` and
``max_tokens`` are accepted at all, and what the max_tokens bounds are.

Kiro CLI reads that schema from ``ListAvailableModels`` (per-model key
``additionalModelRequestFieldsSchema``) and only offers what it declares. This
module does the same: :func:`update_from_models` feeds the live schemas in, and
the lookups below answer from them. A static table measured against the live
API backs the lookups until the fetch happens, and covers the case where the
gateway talks to a chat host that does not serve the metadata endpoint.

An unknown model resolves to ``None`` so callers keep the gateway's
pass-through behaviour: send effort and let Kiro decide, rather than assume a
model is incapable because this table has not heard of it.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from loguru import logger


# ==================================================================================================
# Capability record
# ==================================================================================================

@dataclass(frozen=True)
class ModelCapability:
    """
    What one Kiro model accepts inside additionalModelRequestFields.

    Attributes:
        effort_path: Schema key holding the effort property ("output_config" for
            Claude models, "reasoning" for the rest), or None when the model
            takes no additional fields at all
        effort_levels: Effort values the model's enum allows
        default_effort: Effort Kiro applies when the field is omitted
        thinking_types: Allowed thinking.type values (empty when the model has
            no thinking property; sending one then is a 400)
        display_values: Allowed thinking.display values
        max_tokens_bounds: Inclusive (minimum, maximum) for max_tokens, or None
            when the model has no max_tokens property

    Examples:
        >>> cap = ModelCapability(effort_path="output_config", effort_levels=("low", "high"))
        >>> cap.accepts_additional_fields
        True
        >>> ModelCapability(effort_path=None).accepts_additional_fields
        False
    """
    effort_path: Optional[str]
    effort_levels: Tuple[str, ...] = ()
    default_effort: Optional[str] = None
    thinking_types: Tuple[str, ...] = ()
    display_values: Tuple[str, ...] = ()
    max_tokens_bounds: Optional[Tuple[int, int]] = None

    @property
    def accepts_additional_fields(self) -> bool:
        """True when at least one property may be sent for this model."""
        return bool(self.effort_path or self.thinking_types or self.max_tokens_bounds)


# ==================================================================================================
# Static table (measured against the live Kiro API, 2026-09-06)
# ==================================================================================================

# Claude models with the five-level effort enum. max_tokens tops out at 128000.
_CLAUDE_5_LEVEL = ModelCapability(
    effort_path="output_config",
    effort_levels=("low", "medium", "high", "xhigh", "max"),
    default_effort="high",
    thinking_types=("adaptive", "disabled"),
    display_values=("summarized", "omitted"),
    max_tokens_bounds=(1024, 128000),
)

# claude-opus-4.7 shares the schema but Kiro defaults it to xhigh, not high.
_CLAUDE_5_LEVEL_XHIGH_DEFAULT = ModelCapability(
    effort_path="output_config",
    effort_levels=("low", "medium", "high", "xhigh", "max"),
    default_effort="xhigh",
    thinking_types=("adaptive", "disabled"),
    display_values=("summarized", "omitted"),
    max_tokens_bounds=(1024, 128000),
)

# The 4.6 generation has no xhigh level and a lower max_tokens ceiling.
# Verified: claude-sonnet-4.6 + xhigh answers 400 naming enum
# ["low", "medium", "high", "max"].
_CLAUDE_4_LEVEL = ModelCapability(
    effort_path="output_config",
    effort_levels=("low", "medium", "high", "max"),
    default_effort="high",
    thinking_types=("adaptive", "disabled"),
    display_values=("summarized", "omitted"),
    max_tokens_bounds=(1024, 64000),
)

# Non-Claude reasoning models: effort only, under the "reasoning" key, and the
# enum adds "none" to switch reasoning off. They have no thinking property, so
# sending one answers 400 "property 'thinking' is not defined in the schema".
_REASONING_ONLY = ModelCapability(
    effort_path="reasoning",
    effort_levels=("none", "low", "medium", "high", "xhigh", "max"),
    default_effort="high",
    thinking_types=(),
    display_values=(),
    max_tokens_bounds=None,
)

# Models Kiro publishes without a schema and which answer 400
# "additionalModelRequestFields is not supported for this model".
_NO_ADDITIONAL_FIELDS = ModelCapability(effort_path=None)

_STATIC_CAPABILITIES: Dict[str, ModelCapability] = {
    # "auto" is the one model Kiro publishes no schema for while still accepting
    # the field: it routes to the Claude models above, and effort=xhigh,
    # effort=max, thinking.type=disabled and max_tokens=64000 were all measured
    # to answer 200. Absent metadata is therefore not treated as a refusal.
    "auto": _CLAUDE_5_LEVEL,
    "claude-opus-5": _CLAUDE_5_LEVEL,
    "claude-sonnet-5": _CLAUDE_5_LEVEL,
    "claude-opus-4.8": _CLAUDE_5_LEVEL,
    "claude-opus-4.7": _CLAUDE_5_LEVEL_XHIGH_DEFAULT,
    "claude-opus-4.6": _CLAUDE_4_LEVEL,
    "claude-sonnet-4.6": _CLAUDE_4_LEVEL,
    "gpt-5.6-sol": _REASONING_ONLY,
    "gpt-5.6-terra": _REASONING_ONLY,
    "gpt-5.6-luna": _REASONING_ONLY,
    "claude-opus-4.5": _NO_ADDITIONAL_FIELDS,
    "claude-sonnet-4.5": _NO_ADDITIONAL_FIELDS,
    "claude-sonnet-4": _NO_ADDITIONAL_FIELDS,
    "claude-haiku-4.5": _NO_ADDITIONAL_FIELDS,
    "deepseek-3.2": _NO_ADDITIONAL_FIELDS,
    "minimax-m2.1": _NO_ADDITIONAL_FIELDS,
    "minimax-m2.5": _NO_ADDITIONAL_FIELDS,
    "glm-5": _NO_ADDITIONAL_FIELDS,
    "qwen3-coder-next": _NO_ADDITIONAL_FIELDS,
}

# Capabilities learned from ListAvailableModels. Takes precedence over the
# static table, which only exists to cover the window before the first fetch
# and hosts that do not serve the metadata endpoint.
_runtime_capabilities: Dict[str, ModelCapability] = {}


# ==================================================================================================
# Schema parsing
# ==================================================================================================

def _enum_of(node: Any) -> Tuple[str, ...]:
    """Return a schema node's string enum as a tuple, or () when absent."""
    if not isinstance(node, dict):
        return ()
    values = node.get("enum")
    if not isinstance(values, list):
        return ()
    return tuple(str(v) for v in values if isinstance(v, str))


def parse_schema(schema: Any) -> Optional[ModelCapability]:
    """
    Turn one model's additionalModelRequestFieldsSchema into a ModelCapability.

    Args:
        schema: Schema object as published by ListAvailableModels

    Returns:
        The capability, or None when the schema is missing or unusable

    Examples:
        >>> parse_schema({"properties": {"reasoning": {"properties": {
        ...     "effort": {"enum": ["low", "high"], "default": "high"}}}}})
        ModelCapability(effort_path='reasoning', effort_levels=('low', 'high'), default_effort='high', thinking_types=(), display_values=(), max_tokens_bounds=None)
        >>> parse_schema(None) is None
        True
    """
    if not isinstance(schema, dict):
        return None

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return None

    effort_path = None
    for candidate in ("output_config", "reasoning"):
        if isinstance(properties.get(candidate), dict):
            effort_path = candidate
            break

    effort_levels: Tuple[str, ...] = ()
    default_effort: Optional[str] = None
    if effort_path:
        effort_node = properties[effort_path].get("properties", {})
        if isinstance(effort_node, dict):
            effort = effort_node.get("effort")
            effort_levels = _enum_of(effort)
            if isinstance(effort, dict) and isinstance(effort.get("default"), str):
                default_effort = effort["default"]

    thinking_types: Tuple[str, ...] = ()
    display_values: Tuple[str, ...] = ()
    thinking = properties.get("thinking")
    if isinstance(thinking, dict):
        thinking_props = thinking.get("properties")
        if isinstance(thinking_props, dict):
            thinking_types = _enum_of(thinking_props.get("type"))
            display_values = _enum_of(thinking_props.get("display"))

    max_tokens_bounds: Optional[Tuple[int, int]] = None
    max_tokens = properties.get("max_tokens")
    if isinstance(max_tokens, dict):
        minimum = max_tokens.get("minimum")
        maximum = max_tokens.get("maximum")
        if isinstance(minimum, int) and isinstance(maximum, int) and minimum <= maximum:
            max_tokens_bounds = (minimum, maximum)

    if not (effort_path or thinking_types or max_tokens_bounds):
        return None

    return ModelCapability(
        effort_path=effort_path,
        effort_levels=effort_levels,
        default_effort=default_effort,
        thinking_types=thinking_types,
        display_values=display_values,
        max_tokens_bounds=max_tokens_bounds,
    )


def update_from_models(models_data: Any) -> int:
    """
    Learn capabilities from a ListAvailableModels payload.

    A model whose schema is absent leaves any existing entry alone rather than
    marking the model incapable: "auto" ships without a schema yet accepts the
    field, so absent metadata carries no refusal.

    Args:
        models_data: The "models" list from ListAvailableModels

    Returns:
        Number of models whose capabilities were learned

    Examples:
        >>> update_from_models([{"modelId": "m", "additionalModelRequestFieldsSchema": {
        ...     "properties": {"reasoning": {"properties": {"effort": {"enum": ["low"]}}}}}}])
        1
        >>> effort_levels("m")
        ('low',)
        >>> reset_runtime_capabilities()
    """
    if not isinstance(models_data, list):
        return 0

    learned = 0
    for entry in models_data:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("modelId")
        if not isinstance(model_id, str) or not model_id:
            continue
        try:
            capability = parse_schema(entry.get("additionalModelRequestFieldsSchema"))
        except Exception as exc:  # defensive: never let metadata break startup
            logger.warning(f"Could not read capabilities for model '{model_id}': {exc}")
            continue
        if capability is None:
            continue
        _runtime_capabilities[model_id.strip().lower()] = capability
        learned += 1

    if learned:
        logger.debug(f"Learned additionalModelRequestFields capabilities for {learned} model(s)")
    return learned


def reset_runtime_capabilities() -> None:
    """Drop everything learned from the API (used by tests)."""
    _runtime_capabilities.clear()


# ==================================================================================================
# Lookups
# ==================================================================================================

def get(model_id: str) -> Optional[ModelCapability]:
    """
    Return known capabilities for a resolved Kiro model id.

    Args:
        model_id: Resolved Kiro model id

    Returns:
        The capability, or None when the model is unknown to both the live
        metadata and the static table

    Examples:
        >>> get("claude-opus-5").effort_levels
        ('low', 'medium', 'high', 'xhigh', 'max')
        >>> get("some-future-model") is None
        True
    """
    normalized = (model_id or "").strip().lower()
    if not normalized:
        return None
    return _runtime_capabilities.get(normalized) or _STATIC_CAPABILITIES.get(normalized)


def effort_levels(model_id: str) -> Tuple[str, ...]:
    """
    Return the effort values a model allows, or () when unknown.

    Examples:
        >>> effort_levels("claude-sonnet-4.6")
        ('low', 'medium', 'high', 'max')
        >>> effort_levels("some-future-model")
        ()
    """
    capability = get(model_id)
    return capability.effort_levels if capability else ()


def effort_schema_path(model_id: str) -> Optional[str]:
    """
    Return the schema key holding effort for a model, or None when unknown.

    Examples:
        >>> effort_schema_path("gpt-5.6-sol")
        'reasoning'
        >>> effort_schema_path("some-future-model") is None
        True
    """
    capability = get(model_id)
    return capability.effort_path if capability else None


def supports_thinking(model_id: str) -> bool:
    """
    Report whether a model accepts a thinking property.

    Only positive evidence counts: an unknown model answers False, because
    sending an undeclared property fails the whole request with HTTP 400.

    Examples:
        >>> supports_thinking("claude-opus-5")
        True
        >>> supports_thinking("gpt-5.6-sol")
        False
        >>> supports_thinking("some-future-model")
        False
    """
    capability = get(model_id)
    return bool(capability and capability.thinking_types)


def supports_thinking_type(model_id: str, thinking_type: str) -> bool:
    """
    Report whether a model accepts one specific thinking.type value.

    Examples:
        >>> supports_thinking_type("claude-opus-5", "disabled")
        True
        >>> supports_thinking_type("claude-opus-5", "enabled")
        False
    """
    capability = get(model_id)
    if not capability:
        return False
    return thinking_type in capability.thinking_types


def supports_display(model_id: str, display: str) -> bool:
    """
    Report whether a model accepts one specific thinking.display value.

    Examples:
        >>> supports_display("claude-opus-5", "omitted")
        True
        >>> supports_display("gpt-5.6-sol", "omitted")
        False
    """
    capability = get(model_id)
    if not capability:
        return False
    return display in capability.display_values


def adapt_effort(model_id: str, effort: Optional[str]) -> Optional[str]:
    """
    Fit a canonical effort level to what a model's enum actually allows.

    Kiro's enums differ per model: the opus-5 generation has five levels
    including xhigh, the 4.6 generation only four. Sending a level outside the
    enum fails the request, so an unavailable level steps down to the nearest
    one the model does have.

    An unknown model keeps the conservative mapping (xhigh becomes max), since
    four levels are the widest set every schema-bearing model has agreed on.

    Args:
        model_id: Resolved Kiro model id
        effort: Canonical effort level, already normalized

    Returns:
        An effort level safe to send, or None when there is nothing to send

    Examples:
        >>> adapt_effort("claude-opus-5", "xhigh")
        'xhigh'
        >>> adapt_effort("claude-sonnet-4.6", "xhigh")
        'max'
        >>> adapt_effort("some-future-model", "xhigh")
        'max'
        >>> adapt_effort("claude-opus-5", None) is None
        True
    """
    if not effort:
        return None

    levels = effort_levels(model_id)
    if not levels:
        # Unknown model: xhigh is the only level some schemas lack, so fold it.
        return "max" if effort == "xhigh" else effort

    if effort in levels:
        return effort

    # Reach for the next level up before stepping down, so a missing level never
    # quietly gives the client less reasoning than it asked for. xhigh on a
    # four-level model therefore becomes max, which is what the gateway has
    # always sent for it.
    ladder = ("none", "low", "medium", "high", "xhigh", "max")
    if effort not in ladder:
        logger.warning(f"Effort '{effort}' is not a known level; not sending it for '{model_id}'")
        return None

    position = ladder.index(effort)
    neighbours = list(ladder[position + 1:]) + [
        candidate for candidate in reversed(ladder[:position]) if candidate != "none"
    ]
    for candidate in neighbours:
        if candidate in levels:
            logger.debug(
                f"Model '{model_id}' does not allow effort '{effort}'; sending '{candidate}'"
            )
            return candidate

    logger.warning(f"Model '{model_id}' allows none of the known effort levels; not sending effort")
    return None


def clamp_max_tokens(model_id: str, max_tokens: Optional[int]) -> Optional[int]:
    """
    Fit a client max_tokens into a model's declared bounds.

    Args:
        model_id: Resolved Kiro model id
        max_tokens: Value the client asked for

    Returns:
        A value inside the model's bounds, or None when the model declares no
        max_tokens property or the value is unusable

    Examples:
        >>> clamp_max_tokens("claude-opus-5", 64000)
        64000
        >>> clamp_max_tokens("claude-sonnet-4.6", 128000)
        64000
        >>> clamp_max_tokens("claude-opus-5", 64)
        1024
        >>> clamp_max_tokens("gpt-5.6-sol", 64000) is None
        True
    """
    capability = get(model_id)
    if not capability or not capability.max_tokens_bounds:
        return None
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        return None

    minimum, maximum = capability.max_tokens_bounds
    return max(minimum, min(maximum, max_tokens))
