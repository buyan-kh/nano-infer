"""GPT-2 byte-level BPE via tiktoken, with a prefix-stable incremental decoder."""

from __future__ import annotations

import tiktoken


class Tokenizer:
    def __init__(self, encoding_name: str = "gpt2") -> None:
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.eos_token_id = 50256

    def encode(self, text: str) -> list[int]:
        return list(self.encoding.encode(text))

    def decode(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return self.encoding.decode(token_ids)

    def token_bytes(self, token_id: int) -> bytes:
        return self.encoding.decode_single_token_bytes(token_id)


class IncrementalDecoder:
    """Turn token ids into text without rewriting already-emitted characters.

    GPT-2 tokens are raw bytes. A token can end in the middle of a UTF-8
    character, so the decoder holds incomplete trailing bytes back and only
    commits the longest valid prefix.
    """

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.raw = b""
        self.text = ""

    def push(self, token_id: int, stops: tuple[str, ...] = ()) -> tuple[str, str | None]:
        """Append one token. Return `(delta, stop_reason)`.

        `stop_reason` is `"stop"` when a stop string is found. The delta does
        not include the stop string.
        """
        previous = self.text
        self.raw += self.tokenizer.token_bytes(token_id)
        valid = self.raw
        text = ""
        while True:
            try:
                text = valid.decode("utf-8")
                break
            except UnicodeDecodeError:
                if not valid:
                    text = ""
                    break
                valid = valid[:-1]
        cut = None
        for stop in stops:
            if not stop or stop not in text:
                continue
            index = text.index(stop)
            if cut is None or index < cut:
                cut = index
        if cut is not None:
            visible = text[:cut]
            self.text = visible
            delta = visible[len(previous) :] if visible.startswith(previous) else visible
            return delta, "stop"
        self.text = text
        delta = text[len(previous) :] if text.startswith(previous) else text
        return delta, None


def render_chat(messages: list[dict[str, str]]) -> str:
    """Render a chat transcript for a base GPT-2 model.

    GPT-2 has no chat template. This is a stable, explicit transcript so the
    same messages always produce the same token ids.
    """
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            raise ValueError(f"unsupported role {role!r}")
    lines.append("Assistant:")
    return "\n".join(lines)
