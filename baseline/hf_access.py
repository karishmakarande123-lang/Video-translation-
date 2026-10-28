from __future__ import annotations


def format_hf_access_error(model_name: str, exc: Exception | None = None) -> str:
    """Turn a Hugging Face gated-repo error into a clear, actionable message."""
    raw = str(exc) if exc else ""
    lower = raw.lower()
    repo_url = f"https://huggingface.co/{model_name}"

    lines = [
        f"The Hugging Face model '{model_name}' is gated or requires access approval.",
        "",
        "Please do the following before retrying:",
        "1. Open the model page and request access (or accept the access request):",
        f"   {repo_url}",
        "2. Create a read token at https://huggingface.co/settings/tokens",
        "3. Log in from this machine with: hf auth login",
        "   (or: huggingface-cli login)",
        "4. If the model is new, wait a few minutes for approval before retrying.",
    ]

    if raw and (
        "gated" in lower
        or "authenticated" in lower
        or "authorization" in lower
        or "401" in lower
        or "403" in lower
        or "not authorized" in lower
    ):
        lines.append("")
        lines.append(f"Original Hugging Face error: {raw}")
    elif raw:
        lines.append("")
        lines.append(f"Original error: {raw}")

    return "\n".join(lines)
