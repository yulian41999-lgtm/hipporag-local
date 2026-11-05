"""Check the supplied chat service independently of model installation."""
import json
import os
import httpx
from settings import ROOT, load_settings


def main():
    cfg = load_settings()
    with httpx.Client(trust_env=False, timeout=60) as client:
        response = client.post(
            cfg["llm_base_url"] + "/chat/completions",
            headers={"Authorization": "Bearer " + os.environ[cfg["llm_api_key_env"]]},
            json={
                "model": cfg["llm_model"],
                "messages": [{"role": "user", "content": "Return only this JSON object: {\"status\": \"ok\"}"}],
                "temperature": 0,
                "max_tokens": 256,
            },
        )
        response.raise_for_status()
        body = response.json()
    message = body["choices"][0]["message"]
    report = {
        "http_status": response.status_code,
        "model": body.get("model"),
        "content": message.get("content"),
        "finish_reason": body["choices"][0].get("finish_reason"),
        "usage": body.get("usage"),
    }
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    (output / "llm_probe.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if not message.get("content") or not body.get("usage"):
        raise RuntimeError("The endpoint must return nonempty content and standard token usage.")


if __name__ == "__main__":
    main()
