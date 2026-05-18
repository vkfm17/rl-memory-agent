import random
from typing import Any

# Hard-coded examples
FACTS = {
    "birthday": ["June 9", "January 12", "March 3", "Jan 8", "April 10"],
    "city": ["Hoboken", "Seattle", "Boston", "Chicago", "New York", "London"],
    "movie": ["Interstellar", "Inception", "The Matrix", "The Godfather"],
}

DISTRACTORS = [
    "I had coffee today.",
    "The weather is nice.",
    "I enjoy hiking.",
    "I watched TV yesterday.",
    "Pizza is delicious.",
    "I walked my dog.",
    "My friend moved away.",
]

CONTRADICTION_FACTS = {
    "city": [
        ("Boston", "Seattle"),
        ("Hoboken", "New York"),
        ("London", "San Francisco"),
    ],
}


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
            conversation.append(
                {
                    "message": fact_message,
                    "fact_value": fact_value,
                    "step": i,
                    # "summarized": False,
                }
            )

        if i < num_distractors:
            conversation.append(
                {
                    "message": random.choice(DISTRACTORS),
                    "fact_value": None,
                    "step": i,
                    # "summarized": False,
                }
            )

    return {
        "conversation": conversation,
        "question": question,
        "answer": fact_value,
    }


def generate_contradiction_conversation(num_distractors: int = 10):
    old_city, new_city = random.choice(CONTRADICTION_FACTS["city"])
    conversation = []

    # old fact
    conversation.append(
        {
            "message": f"I live in {old_city}.",
            "fact_value": old_city,
            "step": len(conversation),
            # "summarized": False,
        }
    )

    # Distractors
    for _ in range(num_distractors // 2):
        conversation.append(
            {
                "message": random.choice(DISTRACTORS),
                "fact_value": None,
                "step": len(conversation),
                # "summarized": False,
            }
        )

    # Contradiction/update
    conversation.append(
        {
            "message": (f"I moved to {new_city}."),
            "fact_value": new_city,
            "step": len(conversation),
            # "summarized": False,
        }
    )

    # More distractors
    for _ in range(num_distractors // 2):
        conversation.append(
            {
                "message": random.choice(DISTRACTORS),
                "fact_value": None,
                "step": len(conversation),
                # "summarized": False,
            }
        )

    question = "What city do I live in?"

    return {
        "conversation": conversation,
        "question": question,
        "answer": new_city,
    }


def generate_contradiction_chain():
    return {
        "conversation": [
            {
                "message": "I live in Boston.",
                "fact_value": "Boston",
                "step": 0,
            },
            {
                "message": "I moved to Seattle.",
                "fact_value": "Seattle",
                "step": 1,
            },
            {
                "message": "I moved to Chicago.",
                "fact_value": "Chicago",
                "step": 2,
            },
            {
                "message": "I used to like pizza but now I prefer sushi.",
                "fact_value": None,
                "step": 3,
            },
            {
                "message": "I moved to Austin last year.",
                "fact_value": "Austin",
                "step": 4,
            },
        ],
        "question": "Where do I live now?",
        "answer": "Austin",
    }


def generate_profile_with_updates():
    return {
        "conversation": [
            {
                "message": "My name is Alice.",
                "fact_value": ("name", "Alice"),
                "step": 0,
            },
            {
                "message": "I live in Boston.",
                "fact_value": ("location", "Boston"),
                "step": 1,
            },
            {
                "message": "My favorite color is blue.",
                "fact_value": ("color", "blue"),
                "step": 2,
            },
            {
                "message": "I moved to Seattle.",
                "fact_value": ("location", "Seattle"),
                "step": 3,
            },
            {
                "message": "My favorite color is green.",
                "fact_value": ("color", "green"),
                "step": 4,
            },
        ],
        "question": "Where does Alice live?",
        "answer": "Seattle",
    }


def generate_distractor_heavy():
    return {
        "conversation": [
            {"message": "I like pizza.", "fact_value": None, "step": 0},
            {"message": "The sky is nice today.", "fact_value": None, "step": 1},
            {"message": "I bought a book yesterday.", "fact_value": None, "step": 2},
            {
                "message": "My favorite movie is Interstellar.",
                "fact_value": "Interstellar",
                "step": 3,
            },
            {"message": "I had coffee this morning.", "fact_value": None, "step": 4},
            {"message": "I enjoy hiking.", "fact_value": None, "step": 5},
            {
                "message": "I changed my favorite movie to Inception.",
                "fact_value": "Inception",
                "step": 6,
            },
        ],
        "question": "What is my favorite movie?",
        "answer": "Inception",
    }


def generate_temporal_updates():
    return {
        "conversation": [
            {"message": "I live in Boston.", "fact_value": "Boston", "step": 0},
            {"message": "I am currently happy.", "fact_value": None, "step": 1},
            {"message": "I moved to Seattle.", "fact_value": "Seattle", "step": 2},
            {"message": "I used to live in Boston.", "fact_value": None, "step": 3},
            {"message": "Now I live in Chicago.", "fact_value": "Chicago", "step": 4},
        ],
        "question": "Where do I live?",
        "answer": "Chicago",
    }


def generate_multi_query():
    return {
        "conversation": [
            {"message": "My name is Bob.", "fact_value": ("name", "Bob"), "step": 0},
            {
                "message": "I live in Miami.",
                "fact_value": ("location", "Miami"),
                "step": 1,
            },
            {
                "message": "My favorite color is red.",
                "fact_value": ("color", "red"),
                "step": 2,
            },
            {
                "message": "I moved to Denver.",
                "fact_value": ("location", "Denver"),
                "step": 3,
            },
        ],
        "question": "Where does Bob live and what is his name?",
        "answer": "Bob, Denver",
    }


def generate_hard_task():
    tasks = [
        generate_contradiction_chain,
        generate_profile_with_updates,
        generate_distractor_heavy,
        generate_temporal_updates,
        generate_multi_query,
    ]
    return random.choice(tasks)()


def sample_task_by_difficulty(level: int, num_distractors: int = 10) -> dict[str, Any]:

    if level == 0:
        return generate_conversation(num_distractors=5)

    elif level == 1:
        return generate_conversation(num_distractors=num_distractors)

    elif level == 2:
        return generate_contradiction_conversation(num_distractors=num_distractors)

    elif level == 3:
        return generate_distractor_heavy()

    elif level == 4:
        return generate_temporal_updates()

    else:
        return generate_hard_task()
