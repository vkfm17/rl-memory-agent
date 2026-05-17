import random

FACTS = {
    "birthday": [
        "June 9",
        "January 12",
        "March 3",
    ],
    "city": [
        "Hoboken",
        "Seattle",
        "Boston",
    ],
    "movie": [
        "Interstellar",
        "Inception",
        "The Matrix",
    ]
}

DISTRACTORS = [
    "I had coffee today.",
    "The weather is nice.",
    "I enjoy hiking.",
    "I watched TV yesterday.",
    "Pizza is delicious.",
    "I walked my dog.",
]


def generate_conversation(num_distractors: int = 10) -> dict[str, str]:
    """Use the facts and distractors to generate synthetic conversations."""

    fact_type = random.choice(list(FACTS.keys()))
    fact_value = random.choice(FACTS[fact_type])

    # Simple to start
    if fact_type == "birthday":
        fact_message = f"My birthday is {fact_value}."
        question = "What is my birthday?"

    elif fact_type == "city":
        fact_message = f"I live in {fact_value}."
        question = "What city do I live in?"

    else:
        fact_message = f"My favorite movie is {fact_value}."
        question = "What is my favorite movie?"

    conversation: list[dict[str, str | int]] = []

    # Insert fact somewhere random in the conversation
    fact_position = random.randint(0, num_distractors)

    for i in range(num_distractors + 1):

        if i == fact_position:
            conversation.append({
                "message": fact_message,
                "fact_value": fact_value,
            })

        if i < num_distractors:
            conversation.append({
                "message": random.choice(DISTRACTORS),
                "fact_value": None,
            })

    return {
        "conversation": conversation,
        "question": question,
        "answer": fact_value,
    }