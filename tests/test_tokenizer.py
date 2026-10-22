from nano_infer.tokenizer import IncrementalDecoder, Tokenizer, render_chat


def test_incremental_decode_matches_full_decode():
    tokenizer = Tokenizer()
    text = "Hello, world — café."
    ids = tokenizer.encode(text)
    decoder = IncrementalDecoder(tokenizer)
    built = ""
    for token in ids:
        delta, reason = decoder.push(token)
        assert reason is None
        built += delta
    assert built == text
    assert decoder.text == tokenizer.decode(ids)


def test_stop_string_is_excluded_from_the_delta():
    tokenizer = Tokenizer()
    ids = tokenizer.encode("abcdefghij")
    decoder = IncrementalDecoder(tokenizer)
    built = ""
    reason = None
    for token in ids:
        delta, reason = decoder.push(token, ("def",))
        built += delta
        if reason:
            break
    assert reason == "stop"
    assert built == "abc"
    assert "def" not in built


def test_chat_transcript_is_stable():
    rendered = render_chat(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Ping"},
            {"role": "assistant", "content": "Pong"},
            {"role": "user", "content": "Again"},
        ]
    )
    assert rendered.endswith("Assistant:")
    assert "User: Ping" in rendered
    assert "System: Be brief." in rendered
