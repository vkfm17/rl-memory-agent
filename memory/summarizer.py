import re


def summarize_message(message: str) -> str:
    """
    Compress message while preserving
    key personal information.
    """

    message_lower = message.lower()

    # Birthday extraction
    if "birthday" in message_lower:
        match = re.search(
            r"birthday is ([A-Za-z0-9 ]+)",
            message,
            re.IGNORECASE,
        )

        if match:
            return f"Birthday: {match.group(1)}"

    # Location extraction
    if "live in" in message_lower or "moved to" in message_lower:
        match = re.search(
            r"(live in|moved to) ([A-Za-z ]+)",
            message,
            re.IGNORECASE,
        )

        if match:
            return f"Location: {match.group(2)}"

    # Favorite extraction
    if "favorite" in message_lower:
        return message[:40]

    # Default compression
    return message[:30]
