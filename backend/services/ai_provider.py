from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from services.session_credentials import PROVIDERS
from utils.config import settings


def answer_dataset_question(
    question: str,
    profile: dict,
    provider: str | None = None,
    credentials: dict | None = None,
) -> dict:
    selected = (provider or "local").lower()
    if selected == "local":
        return {"answer": _local_answer(question, profile), "provider": "local-profile"}

    if selected not in PROVIDERS:
        raise ValueError(f"Unknown provider. Choose local or one of: {', '.join(PROVIDERS)}.")
    if not credentials:
        raise ValueError(f"Connect {selected} in this browser session before using it.")

    prompt = (
        "Answer the user's question using only this dataset profile. Do not claim to have inspected raw records. "
        "Explain uncertainty clearly and suggest useful next analysis when relevant.\n\n"
        f"Dataset profile JSON:\n{json.dumps(profile, ensure_ascii=True, default=str)}\n\n"
        f"User question: {question}"
    )
    return {"answer": _provider_completion(selected, prompt, credentials), "provider": selected}


def test_provider_connection(provider: str, credentials: dict) -> dict:
    selected = provider.lower()
    if selected not in PROVIDERS:
        raise ValueError(f"Unknown provider. Choose one of: {', '.join(PROVIDERS)}.")
    text = _provider_completion(selected, "Reply with exactly: KATNYO connection ready.", credentials)
    return {"provider": selected, "model": credentials.get("model") or _default_model(selected), "message": text[:180]}


def _provider_completion(provider: str, prompt: str, credentials: dict) -> str:
    api_key = str(credentials.get("api_key") or "").strip()
    if not api_key:
        raise ValueError(f"Connect {provider} in this browser session before using it.")

    model = str(credentials.get("model") or _default_model(provider)).strip()
    headers = {"Content-Type": "application/json"}
    if provider == "openai":
        url = "https://api.openai.com/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    elif provider == "azure-openai":
        endpoint = str(credentials.get("endpoint") or "").rstrip("/")
        deployment = str(credentials.get("deployment") or "").strip()
        if not endpoint or not deployment:
            raise ValueError("Azure OpenAI needs an endpoint and deployment name.")
        url = f"{endpoint}/openai/deployments/{quote(deployment, safe='')}/chat/completions?api-version={quote(settings.azure_openai_api_version)}"
        headers["api-key"] = api_key
        payload = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    elif provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='-')}:generateContent"
        headers["x-goog-api-key"] = api_key
        payload = {"contents": [{"parts": [{"text": prompt}]}]}
    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers.update({"x-api-key": api_key, "anthropic-version": "2023-06-01"})
        payload = {"model": model, "max_tokens": 900, "messages": [{"role": "user", "content": prompt}]}
    elif provider == "huggingface":
        url = "https://router.huggingface.co/v1/chat/completions"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    elif provider == "cohere":
        url = "https://api.cohere.com/v2/chat"
        headers["Authorization"] = f"Bearer {api_key}"
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
    else:
        raise ValueError("Provider response format was not recognized.")

    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=45) as response:
            result = json.loads(response.read().decode("utf-8"))
        return _extract_text(provider, result)
    except HTTPError as exc:
        if exc.code == 401:
            message = f"{provider} rejected the API credential (HTTP 401). Confirm it is active and belongs to the correct provider/project, then re-enter it and choose Save & test. Do not paste the key into chat."
        elif exc.code == 403:
            message = f"{provider} denied access (HTTP 403). Check the account, project permissions, and model access for this credential."
        elif exc.code == 404:
            message = f"{provider} could not find the requested model or endpoint (HTTP 404). Check the model name and provider configuration."
        elif exc.code == 429:
            message = f"{provider} rate limit or quota was reached (HTTP 429). Check provider usage and try again later."
        else:
            message = f"{provider} returned HTTP {exc.code}; check provider availability and configuration."
        raise RuntimeError(message) from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach {provider}; check network access and provider availability.") from exc
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{provider} returned an unexpected response. Check the model and provider settings.") from exc


def _default_model(provider: str) -> str:
    return {
        "openai": settings.openai_model,
        "azure-openai": "",
        "gemini": settings.gemini_model,
        "anthropic": settings.anthropic_model,
        "huggingface": settings.huggingface_model,
        "cohere": settings.cohere_model,
    }[provider]


def _extract_text(provider: str, result: dict) -> str:
    if provider in {"openai", "azure-openai", "huggingface"}:
        return result["choices"][0]["message"]["content"]
    if provider == "gemini":
        return "".join(part.get("text", "") for part in result["candidates"][0]["content"]["parts"])
    if provider == "anthropic":
        return "".join(block.get("text", "") for block in result.get("content", []) if block.get("type") == "text")
    if provider == "cohere":
        return "".join(block.get("text", "") for block in result["message"]["content"] if block.get("type") == "text")
    raise ValueError("Provider response format was not recognized.")


def _local_answer(question: str, profile: dict) -> str:
    normalized = question.lower()
    if any(word in normalized for word in ("target", "predict", "problem", "task")):
        return f"The detected target is '{profile['target']}' and the task looks like {profile['problemType']}."
    if any(word in normalized for word in ("missing", "null", "quality")):
        return f"The profile found {profile['missingCells']} missing cells ({profile['missingPercent']}% of the sampled data)."
    if any(word in normalized for word in ("column", "feature", "variable")):
        kinds = profile["featureTypes"]
        return f"There are {profile['columns']} columns. Numeric: {', '.join(kinds['numeric']) or 'none'}; categorical: {', '.join(kinds['categorical']) or 'none'}; dates: {', '.join(kinds['datetime']) or 'none'}."
    estimated = "estimated " if profile["rowCountEstimated"] else ""
    return f"This dataset has {profile['rows']} {estimated}rows, {profile['columns']} columns, and a detected {profile['problemType']} target '{profile['target']}'. Ask about the target, columns, or missing values for profile-based insights."
