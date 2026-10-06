def text_validation(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Input must be text!")

    if len(text) > 5000:
        raise ValueError("Input exceeds 5000 characters!")

    clean_text = text.strip()

    if not clean_text:
        raise ValueError("Input should not be empty!")

    return clean_text