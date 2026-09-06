# -*- coding: utf-8 -*-

"""
Unit tests for model_capabilities and the additionalModelRequestFields assembly.

Every expectation here was measured against the live Kiro API on 2026-09-06, either
from the per-model additionalModelRequestFieldsSchema that ListAvailableModels
publishes or from a request that Kiro accepted or rejected:

- claude-opus-5 accepts effort=xhigh (200); claude-sonnet-4.6 rejects it with
  400 naming enum ["low", "medium", "high", "max"]
- claude-opus-5 with no additionalModelRequestFields emitted 47 reasoning events for
  a request whose client had asked for thinking.type=disabled; with
  thinking.type=disabled forwarded it emitted 0
- gpt-5.6-sol rejects a thinking property with 400 "property 'thinking' is not
  defined in the schema and the schema does not allow additional properties", and
  accepts reasoning.effort=none
- auto is published without a schema yet accepts effort, thinking and max_tokens
"""

import pytest

from kiro import model_capabilities as mc
from kiro.converters_core import ThinkingConfig, build_native_reasoning_fields
from kiro.models_anthropic import AnthropicMessagesRequest
from kiro.converters_anthropic import anthropic_to_kiro


@pytest.fixture(autouse=True)
def clean_runtime_capabilities():
    """Keep learned capabilities from leaking between tests."""
    mc.reset_runtime_capabilities()
    yield
    mc.reset_runtime_capabilities()


# Real schema as published by ListAvailableModels for the opus-5 generation.
OPUS_5_SCHEMA = {
    "type": "object",
    "properties": {
        "thinking": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["adaptive", "disabled"]},
                "display": {"type": "string", "enum": ["summarized", "omitted"]},
            },
            "required": ["type"],
        },
        "output_config": {
            "type": "object",
            "properties": {
                "effort": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "xhigh", "max"],
                    "default": "high",
                }
            },
        },
        "max_tokens": {"type": "integer", "minimum": 1024, "maximum": 128000},
    },
    "additionalProperties": False,
}

GPT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "object",
            "properties": {
                "effort": {
                    "type": "string",
                    "enum": ["none", "low", "medium", "high", "xhigh", "max"],
                    "default": "high",
                }
            },
        }
    },
    "additionalProperties": False,
}


# ==================================================================================================
# Effort levels differ per model
# ==================================================================================================


class TestEffortLevelsPerModel:
    """
    Tests that an effort level is fitted to the enum the model actually has.

    Folding xhigh into max everywhere sent a higher level than the client asked for
    on every model that has a real xhigh level.
    """

    def test_xhigh_survives_on_the_opus_5_generation(self):
        """
        What it does: xhigh stays xhigh on a model whose enum has it.
        Goal: Kiro accepts xhigh there, so replacing it with max overshoots.
        """
        for model in ("claude-opus-5", "claude-sonnet-5", "claude-opus-4.8", "claude-opus-4.7"):
            adapted = mc.adapt_effort(model, "xhigh")
            print(f"{model}: xhigh -> {adapted}")
            assert adapted == "xhigh", f"{model} got {adapted}"

    def test_xhigh_folds_to_max_where_the_enum_lacks_it(self):
        """
        What it does: xhigh becomes max on the 4.6 generation.
        Goal: Its enum is ["low", "medium", "high", "max"] and xhigh answers 400.
        """
        for model in ("claude-sonnet-4.6", "claude-opus-4.6"):
            adapted = mc.adapt_effort(model, "xhigh")
            print(f"{model}: xhigh -> {adapted}")
            assert adapted == "max", f"{model} got {adapted}"

    def test_unknown_model_keeps_the_conservative_fold(self):
        """
        What it does: An unknown model still folds xhigh into max.
        Goal: Four levels are the widest set every schema-bearing model agrees on,
              so this cannot produce an out-of-enum value.
        """
        adapted = mc.adapt_effort("claude-opus-9", "xhigh")
        print(f"unknown model: xhigh -> {adapted}")
        assert adapted == "max"

    def test_supported_level_passes_through_untouched(self):
        """
        What it does: A level the model has is sent as-is.
        Goal: Guard against the fitting logic rewriting valid levels.
        """
        for level in ("low", "medium", "high", "max"):
            assert mc.adapt_effort("claude-opus-5", level) == level

    def test_none_effort_yields_nothing(self):
        """
        What it does: An empty effort produces no value.
        Goal: Callers use the absence to mean "do not send effort".
        """
        assert mc.adapt_effort("claude-opus-5", None) is None
        assert mc.adapt_effort("claude-opus-5", "") is None


# ==================================================================================================
# max_tokens bounds
# ==================================================================================================


class TestMaxTokensClamping:
    """Tests that a client max_tokens is fitted to the model's declared bounds."""

    def test_value_inside_bounds_is_kept(self):
        """
        What it does: Claude Code's 64000 passes through on the opus-5 generation.
        Goal: Its bounds are 1024..128000.
        """
        assert mc.clamp_max_tokens("claude-opus-5", 64000) == 64000

    def test_value_above_the_ceiling_is_lowered(self):
        """
        What it does: 128000 becomes 64000 on the 4.6 generation.
        Goal: Its ceiling is 64000 and Kiro validates the number.
        """
        assert mc.clamp_max_tokens("claude-sonnet-4.6", 128000) == 64000

    def test_value_below_the_floor_is_raised(self):
        """
        What it does: A 64-token request becomes 1024.
        Goal: The schema floor is 1024, so 64 would fail validation. Claude Code
              sends max_tokens=64 for its short side requests.
        """
        assert mc.clamp_max_tokens("claude-opus-5", 64) == 1024

    def test_models_without_the_property_get_nothing(self):
        """
        What it does: The reasoning-schema models yield no max_tokens.
        Goal: Their schema has no max_tokens property and forbids extras.
        """
        assert mc.clamp_max_tokens("gpt-5.6-sol", 64000) is None

    def test_unusable_values_yield_nothing(self):
        """
        What it does: Missing or nonsensical values produce nothing.
        Goal: Never invent a bound from a bad input.
        """
        assert mc.clamp_max_tokens("claude-opus-5", None) is None
        assert mc.clamp_max_tokens("claude-opus-5", 0) is None
        assert mc.clamp_max_tokens("claude-opus-5", -5) is None
        assert mc.clamp_max_tokens("claude-opus-5", True) is None


# ==================================================================================================
# Thinking property support
# ==================================================================================================


class TestThinkingSupport:
    """Tests which models may be sent a thinking property."""

    def test_claude_models_accept_thinking(self):
        """
        What it does: The Claude generations accept adaptive and disabled.
        Goal: Forwarding thinking.type is what makes disabled thinking real.
        """
        assert mc.supports_thinking("claude-opus-5") is True
        assert mc.supports_thinking_type("claude-opus-5", "disabled") is True
        assert mc.supports_thinking_type("claude-opus-5", "adaptive") is True

    def test_anthropic_only_values_are_not_forwarded(self):
        """
        What it does: "enabled" is not a Kiro thinking.type.
        Goal: Kiro's enum is ("adaptive", "disabled"); anything else is a 400.
        """
        assert mc.supports_thinking_type("claude-opus-5", "enabled") is False

    def test_reasoning_schema_models_reject_thinking(self):
        """
        What it does: gpt-5.6 models accept no thinking property.
        Goal: Their schema forbids it outright, failing the whole request.
        """
        assert mc.supports_thinking("gpt-5.6-sol") is False
        assert mc.supports_thinking_type("gpt-5.6-sol", "disabled") is False

    def test_unknown_models_are_treated_as_unsupported(self):
        """
        What it does: An unknown model is sent no thinking property.
        Goal: Only positive evidence justifies a property, since an undeclared one
              fails the request rather than being ignored.
        """
        assert mc.supports_thinking("claude-opus-9") is False
        assert mc.supports_display("claude-opus-9", "omitted") is False

    def test_display_values_follow_the_schema(self):
        """
        What it does: summarized and omitted are accepted, others are not.
        Goal: display is enumerated per model like effort is.
        """
        assert mc.supports_display("claude-opus-5", "summarized") is True
        assert mc.supports_display("claude-opus-5", "omitted") is True
        assert mc.supports_display("claude-opus-5", "updates") is False


# ==================================================================================================
# Learning from ListAvailableModels
# ==================================================================================================


class TestLearningFromMetadata:
    """
    Tests that published schemas drive the lookups, the way Kiro CLI works.

    The static table only covers the window before the first fetch, so a model the
    gateway has never heard of must become usable from metadata alone.
    """

    def test_schema_is_parsed_into_capabilities(self):
        """
        What it does: A published schema yields levels, thinking, display and bounds.
        Goal: This is the single source Kiro CLI trusts.
        """
        learned = mc.update_from_models([
            {"modelId": "claude-opus-7", "additionalModelRequestFieldsSchema": OPUS_5_SCHEMA}
        ])
        print(f"learned={learned}")
        assert learned == 1

        capability = mc.get("claude-opus-7")
        print(f"capability={capability}")
        assert capability.effort_path == "output_config"
        assert capability.effort_levels == ("low", "medium", "high", "xhigh", "max")
        assert capability.default_effort == "high"
        assert capability.thinking_types == ("adaptive", "disabled")
        assert capability.display_values == ("summarized", "omitted")
        assert capability.max_tokens_bounds == (1024, 128000)

    def test_learned_schema_wins_over_the_static_table(self):
        """
        What it does: Metadata overrides a stale static entry.
        Goal: Kiro can add a level to a model without a gateway release.
        """
        assert mc.adapt_effort("claude-sonnet-4.6", "xhigh") == "max"

        widened = dict(OPUS_5_SCHEMA)
        mc.update_from_models([
            {"modelId": "claude-sonnet-4.6", "additionalModelRequestFieldsSchema": widened}
        ])

        adapted = mc.adapt_effort("claude-sonnet-4.6", "xhigh")
        print(f"after learning: xhigh -> {adapted}")
        assert adapted == "xhigh"

    def test_reasoning_schema_is_recognized(self):
        """
        What it does: A reasoning-keyed schema resolves to that path with no thinking.
        Goal: The schema key, not the model name, decides where effort goes.
        """
        mc.update_from_models([
            {"modelId": "gpt-9", "additionalModelRequestFieldsSchema": GPT_SCHEMA}
        ])
        assert mc.effort_schema_path("gpt-9") == "reasoning"
        assert mc.supports_thinking("gpt-9") is False
        assert "none" in mc.effort_levels("gpt-9")

    def test_absent_schema_leaves_a_known_model_alone(self):
        """
        What it does: A model listed without a schema does not overwrite what is known.
        Goal: auto ships without a schema yet accepts the field, so absent metadata
              is not a refusal.
        """
        mc.update_from_models([{"modelId": "auto"}])

        capability = mc.get("auto")
        print(f"auto capability={capability}")
        assert capability is not None
        assert capability.accepts_additional_fields is True

    def test_malformed_metadata_is_ignored(self):
        """
        What it does: Junk in the models list is skipped without raising.
        Goal: Metadata must never break account startup.
        """
        learned = mc.update_from_models([
            None,
            {"noModelId": True},
            {"modelId": "x", "additionalModelRequestFieldsSchema": "not-a-schema"},
            {"modelId": "y", "additionalModelRequestFieldsSchema": {"properties": {}}},
        ])
        print(f"learned={learned}")
        assert learned == 0
        assert mc.get("x") is None
        assert mc.get("y") is None

    def test_non_list_payload_is_ignored(self):
        """
        What it does: A payload that is not a list yields nothing.
        Goal: Defensive against an unexpected response shape.
        """
        assert mc.update_from_models(None) == 0
        assert mc.update_from_models({"models": []}) == 0


# ==================================================================================================
# Field assembly
# ==================================================================================================


class TestNativeFieldAssembly:
    """Tests the object sent as additionalModelRequestFields."""

    def test_full_set_for_a_capable_model(self):
        """
        What it does: effort, thinking.type and max_tokens all travel together.
        Goal: This is the shape Kiro CLI sends and Kiro accepted live.
        """
        fields = build_native_reasoning_fields("claude-opus-5", ThinkingConfig(
            native_effort="xhigh",
            native_thinking_type="adaptive",
            native_max_tokens=64000,
        ))
        print(f"fields={fields}")
        assert fields == {
            "output_config": {"effort": "xhigh"},
            "thinking": {"type": "adaptive"},
            "max_tokens": 64000,
        }

    def test_disabled_thinking_is_forwarded(self):
        """
        What it does: Thinking the client switched off reaches Kiro as disabled.
        Goal: Sending nothing leaves Kiro reasoning by default: claude-opus-5
              emitted 47 reasoning events for a request that asked for none.
        """
        fields = build_native_reasoning_fields("claude-opus-5", ThinkingConfig(
            enabled=False,
            native_thinking_type="disabled",
        ))
        print(f"fields={fields}")
        assert fields == {"thinking": {"type": "disabled"}}

    def test_disabled_thinking_uses_effort_none_where_thinking_is_unsupported(self):
        """
        What it does: A reasoning-schema model expresses "off" as effort=none.
        Goal: It has no thinking property, and its enum does have none.
        """
        fields = build_native_reasoning_fields("gpt-5.6-sol", ThinkingConfig(
            enabled=False,
            native_thinking_type="disabled",
        ))
        print(f"fields={fields}")
        assert fields == {"reasoning": {"effort": "none"}}

    def test_thinking_is_never_sent_to_a_reasoning_schema_model(self):
        """
        What it does: A gpt-5.6 request carries effort only.
        Goal: Kiro answers 400 "property 'thinking' is not defined in the schema"
              and the whole request fails.
        """
        fields = build_native_reasoning_fields("gpt-5.6-sol", ThinkingConfig(
            native_effort="high",
            native_thinking_type="adaptive",
            native_display="omitted",
            native_max_tokens=64000,
        ))
        print(f"fields={fields}")
        assert fields == {"reasoning": {"effort": "high"}}

    def test_display_travels_with_a_type(self):
        """
        What it does: display is accompanied by thinking.type.
        Goal: The schema marks type required, so display cannot travel alone.
        """
        fields = build_native_reasoning_fields("claude-opus-5", ThinkingConfig(
            native_effort="high",
            native_display="omitted",
        ))
        print(f"fields={fields}")
        assert fields["thinking"] == {"type": "adaptive", "display": "omitted"}

    def test_unsupported_display_is_dropped(self):
        """
        What it does: A display value outside the enum is not forwarded.
        Goal: Anthropic's "updates" has no Kiro equivalent.
        """
        fields = build_native_reasoning_fields("claude-opus-5", ThinkingConfig(
            native_effort="high",
            native_thinking_type="adaptive",
            native_display="updates",
        ))
        print(f"fields={fields}")
        assert fields["thinking"] == {"type": "adaptive"}

    def test_model_that_takes_no_fields_gets_none(self):
        """
        What it does: A model without a schema yields an empty object.
        Goal: It answers 400 for the field, so the request falls back to tags.
        """
        fields = build_native_reasoning_fields("claude-sonnet-4.5", ThinkingConfig(
            native_effort="high",
            native_thinking_type="adaptive",
            native_max_tokens=64000,
        ))
        print(f"fields={fields}")
        assert fields == {}

    def test_unknown_model_gets_effort_only(self):
        """
        What it does: An unknown model receives effort and nothing else.
        Goal: Pass-through keeps effort working for models the gateway has not
              heard of, while undeclared properties stay out.
        """
        fields = build_native_reasoning_fields("claude-opus-9", ThinkingConfig(
            native_effort="high",
            native_thinking_type="adaptive",
            native_display="omitted",
            native_max_tokens=64000,
        ))
        print(f"fields={fields}")
        assert fields == {"output_config": {"effort": "high"}}

    def test_max_tokens_alone_is_not_sent(self):
        """
        What it does: A request with nothing to say about reasoning sends no fields.
        Goal: max_tokens rides along with reasoning intent rather than silently
              capping every request that passes through the gateway.
        """
        fields = build_native_reasoning_fields(
            "claude-opus-5", ThinkingConfig(native_max_tokens=64000)
        )
        print(f"fields={fields}")
        assert fields == {}

    def test_explicit_schema_path_override_is_honoured(self):
        """
        What it does: An adapter-supplied schema_path wins.
        Goal: Keeps the override usable for a model the registry misreads.
        """
        fields = build_native_reasoning_fields("claude-opus-5", ThinkingConfig(
            native_effort="high",
            schema_path="reasoning",
        ))
        print(f"fields={fields}")
        assert fields == {"reasoning": {"effort": "high"}}


# ==================================================================================================
# The two shapes Claude Code CLI actually sends
# ==================================================================================================


class TestClaudeCodeRequestShapes:
    """
    Tests the end-to-end payload for the requests Claude Code CLI really sends,
    captured from the live gateway.

    Its main turns use claude-opus-5 with thinking.type=adaptive,
    output_config.effort=high and max_tokens=64000. Its short side requests use
    claude-sonnet-5 with thinking.type=disabled and max_tokens=64.
    """

    def _fields(self, body):
        request = AnthropicMessagesRequest(**body)
        payload = anthropic_to_kiro(request, "conv-claude-code", "arn:aws:test")
        return payload.get("additionalModelRequestFields")

    def test_main_turn_matches_kiro_cli(self):
        """
        What it does: The main shape produces effort, adaptive thinking and max_tokens.
        Goal: This is the request path the gateway serves all day.
        """
        fields = self._fields({
            "model": "claude-opus-5",
            "max_tokens": 64000,
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        })
        print(f"fields={fields}")
        assert fields == {
            "output_config": {"effort": "high"},
            "thinking": {"type": "adaptive"},
            "max_tokens": 64000,
        }

    def test_side_request_switches_thinking_off(self):
        """
        What it does: The short shape forwards disabled thinking.
        Goal: These requests ask for 64 tokens and no thinking; without the
              forward they paid for reasoning anyway.
        """
        fields = self._fields({
            "model": "claude-sonnet-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Summarise"}],
            "thinking": {"type": "disabled"},
        })
        print(f"fields={fields}")
        assert fields == {"thinking": {"type": "disabled"}, "max_tokens": 1024}

    def test_side_request_injects_no_thinking_tags(self):
        """
        What it does: The disabled shape carries no prompt tags either.
        Goal: Emulating thinking in the prompt is what the native field replaces.
        """
        request = AnthropicMessagesRequest(**{
            "model": "claude-sonnet-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Summarise"}],
            "thinking": {"type": "disabled"},
        })
        payload = anthropic_to_kiro(request, "conv-claude-code", "arn:aws:test")
        content = payload["conversationState"]["currentMessage"]["userInputMessage"]["content"]
        print(f"content={content!r}")
        assert "<thinking_mode>" not in content
