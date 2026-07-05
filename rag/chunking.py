from __future__ import annotations


def split_text(text: str, chunk_size: int = 4000, overlap: int = 500) -> list[str]:
    clean_text = text.strip()
    if not clean_text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap must be zero or positive")

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks = []
    start = 0
    text_length = len(clean_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        if end < text_length:
            newline_position = clean_text.rfind("\n", start + chunk_size // 2, end)
            if newline_position > start:
                end = newline_position

        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks
