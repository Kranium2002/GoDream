"""The skill catalogue and the chat: GET /api/skills and POST /api/chat.

    GET  /api/skills  the bare list of app/backend.py's skill catalogue, each skill saying whether it works
    POST /api/chat    {messages: [{role, content}]} -> {reply, declared_spec | null, problems: [{code, field, message}],
                      missing: [str]}; 503 chat_unavailable naming what to add while the key or Dev B's module is missing

Claude (config.yaml app.chat) answers with JSON in a fixed schema, through structured outputs. Its answer is data: it
is parsed with json.loads, checked against the catalogue and Dev B's check_declaration, and never executed or
evaluated. declared_spec is the declaration only when nothing was found wrong with it. The key is read on the server
for each call (ANTHROPIC_API_KEY in the environment, else in app.chat.env_file) and goes into no response, log line
or file.
"""
import json
import logging
import os
from pathlib import Path

import anthropic
from dotenv import dotenv_values

from app import backend
from app.routes import Refusal, capability, route

ROOT = Path(__file__).resolve().parents[1]  # the repository root, where app.chat.env_file is looked up
FALLBACK_BETA = "server-side-fallback-2026-07-01"  # a refusal is retried server-side on a model chosen by category
log = logging.getLogger(__name__)
logging.getLogger("anthropic").setLevel(logging.WARNING)  # its debug log dumps request options; keep it out

SYSTEM = """You turn a person's request about a simulated Unitree Go2 robot dog into a skill declaration.

Answer with data only, in the JSON format you are given:
- reply: a short plain answer for the person, saying which skill and parameters you chose or why none fits.
- declaration: one declaration of a skill from the catalogue below, or null.
- unsupported: each part of the request that no skill in the catalogue covers, in a few words; empty if none.

Rules:
- Declare only a skill named in the catalogue, with only its listed parameters, each inside its range or choices.
  Leave a parameter out to get its default.
- If the skill the request needs is marked unavailable, still declare it; say in reply that it cannot run yet and
  what it lacks (its "missing" list).
- If no skill fits, set declaration to null and fill unsupported. Never invent a skill, parameter, value or result.
- source_text is the person's own words the declaration was made from.
- Write no code and no instructions for the server; the server only validates your declaration.

Skill catalogue (JSON):
"""


def api_key(settings: dict) -> str | None:
    """ANTHROPIC_API_KEY from the environment, else from app.chat.env_file; None if neither has one."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        file = ROOT / settings["env_file"]
        key = dotenv_values(file).get("ANTHROPIC_API_KEY") if file.is_file() else None
    return key or None


def unavailable_reason(config: dict) -> str | None:
    """One sentence naming everything the chat still needs, or None when it can run."""
    needs = [] if api_key(config["app"]["chat"]) else [
        "ANTHROPIC_API_KEY in .env at the repository root or in the server's environment"]
    needs += backend.declared()[1]
    return f"The chat needs {'; it also needs '.join(needs)}." if needs else None


def check_messages(body: dict, settings: dict) -> list:
    """The conversation from the browser, checked: user and assistant turns of text, starting and ending with the
    user, within app.chat.max_messages and max_message_chars. Refusal 400 invalid_messages otherwise."""
    messages = body.get("messages")
    ok = (set(body) == {"messages"} and isinstance(messages, list) and 0 < len(messages) <= settings["max_messages"]
          and all(isinstance(m, dict) and set(m) == {"role", "content"} and m["role"] in ("user", "assistant")
                  and isinstance(m["content"], str) and m["content"].strip()
                  and len(m["content"]) <= settings["max_message_chars"] for m in messages)
          and messages[0]["role"] == messages[-1]["role"] == "user")
    if not ok:
        raise Refusal(400, "invalid_messages", f"A chat request is {{messages: [{{role, content}}]}} with up to "
                                               f"{settings['max_messages']} user and assistant texts, starting and "
                                               f"ending with the user.")
    return messages


def build_request(messages: list, catalogue: list, schema: dict, settings: dict) -> dict:
    """The keyword arguments of client.beta.messages.create for one chat turn (everything but the key)."""
    answer = {"type": "object", "additionalProperties": False, "required": ["reply", "declaration", "unsupported"],
              "properties": {"reply": {"type": "string"},
                             "declaration": {"anyOf": [schema, {"type": "null"}]},
                             "unsupported": {"type": "array", "items": {"type": "string"}}}}
    return {
        "model": settings["model"],
        "max_tokens": settings["max_tokens"],
        "system": SYSTEM + json.dumps(catalogue, indent=1, sort_keys=True),
        "messages": messages,
        "output_config": {"effort": settings["effort"], "format": {"type": "json_schema", "schema": answer}},
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
    }


def parse_reply(text: str, catalogue: list, check, config: dict) -> dict:
    """The model's JSON answer as the chat response. Only json.loads touches it; the declaration is returned as
    declared_spec only if it names an available catalogue skill and check(declaration, config) finds no problem."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        data = None
    if not (isinstance(data, dict) and set(data) == {"reply", "declaration", "unsupported"}
            and isinstance(data["reply"], str) and isinstance(data["declaration"], dict | None)
            and isinstance(data["unsupported"], list) and all(isinstance(u, str) for u in data["unsupported"])):
        log.warning("The chat model's answer is not in the agreed shape")
        return {"reply": "The answer could not be read; please send the request again.", "declared_spec": None,
                "problems": [{"code": "invalid_model_reply", "field": "",
                              "message": "The model's answer was not in the agreed shape."}], "missing": []}
    declaration, problems = data["declaration"], []
    missing = [f"no registered skill for: {u}" for u in data["unsupported"]]
    if declaration is not None:
        entry = {e["skill"]: e for e in catalogue}.get(declaration.get("skill"))
        if entry is None:
            problems = [{"code": "unknown_skill", "field": "skill",
                         "message": "That skill is not in the registered skill catalogue."}]
        elif not entry["available"]:
            problems = [{"code": "skill_unavailable", "field": "skill",
                         "message": "That skill cannot run yet; missing lists what it lacks."}]
            missing = list(entry["missing"]) + missing
        else:
            problems = list(check(declaration, config))
    return {"reply": data["reply"], "declared_spec": None if problems or declaration is None else declaration,
            "problems": problems, "missing": missing}


def call_model(key: str, request: dict, settings: dict) -> str | None:
    """Claude's answer text for one request, or None if the model declined it."""
    client = anthropic.Anthropic(api_key=key, timeout=settings["timeout_s"], max_retries=settings["max_retries"])
    response = client.beta.messages.create(**request)
    usage = response.usage
    log.info("Chat call: model %s, stop %s, %d input and %d output tokens, request %s", response.model,
             response.stop_reason, usage.input_tokens, usage.output_tokens, response._request_id)
    if response.stop_reason == "refusal":
        return None
    return next((block.text for block in response.content if block.type == "text"), "")


def run_chat(messages: list, config: dict, key: str, declared, call=call_model) -> dict:
    """One chat turn through Dev B's module `declared`. A failed call is Refusal 502 chat_failed; only the error's
    class and status are logged, never its text, which could repeat a request header."""
    catalogue = declared.skill_catalog(config)
    request = build_request(messages, catalogue, declared.declaration_schema(config), config["app"]["chat"])
    try:
        text = call(key, request, config["app"]["chat"])
    except anthropic.AuthenticationError:
        log.warning("Chat call failed: the API did not accept the key (401)")
        raise Refusal(502, "chat_failed", "The Claude API did not accept the key; check ANTHROPIC_API_KEY.") from None
    except anthropic.APIStatusError as e:
        log.warning("Chat call failed: %s, status %s", type(e).__name__, e.status_code)
        raise Refusal(502, "chat_failed", "The Claude API refused the request; try again in a moment.") from None
    except Exception as e:  # connection errors, timeouts and anything else
        log.warning("Chat call failed: %s", type(e).__name__)
        raise Refusal(502, "chat_failed", "The Claude API could not be reached; try again in a moment.") from None
    if text is None:
        return {"reply": "The model declined this request.", "declared_spec": None,
                "problems": [{"code": "model_declined", "field": "", "message": "The model declined this request."}],
                "missing": []}
    return parse_reply(text, catalogue, declared.check_declaration, config)


@capability("chat")
def chat_capability(server) -> dict:
    reason = unavailable_reason(server.config)
    return {"available": reason is None, "reason": reason or ""}


@route("GET", "/api/skills")
def skills(h) -> None:
    h.send_json(200, backend.skill_catalog(h.server.config))


@route("POST", "/api/chat")
def chat(h) -> None:
    config = h.server.config
    reason = unavailable_reason(config)
    if reason:
        raise Refusal(503, "chat_unavailable", reason)
    messages = check_messages(h.read_body(), config["app"]["chat"])
    module, _ = backend.declared()
    h.send_json(200, run_chat(messages, config, api_key(config["app"]["chat"]), module))
