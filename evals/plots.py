import matplotlib.pyplot as plt


def plot_tradeoff(
    memory_tokens,
    accuracies,
):
    plt.scatter(
        memory_tokens,
        accuracies,
    )
    plt.xlabel("Memory Tokens")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Memory Cost")
    plt.show()


def plot_memory_dynamics(tracker):

    steps = list(range(len(tracker.timeline)))

    memory_sizes = tracker.memory_sizes()
    correctness = tracker.correctness_over_time()
    contradictions = tracker.contradiction_over_time()

    plt.figure(figsize=(10, 5))

    # Memory size
    plt.plot(steps, memory_sizes, label="Memory Size")

    # Correctness
    plt.plot(
        steps,
        [int(x) for x in correctness],
        label="Truth in Memory",
    )

    # Contradictions
    plt.plot(
        steps,
        [int(x) for x in contradictions],
        label="Contradiction Present",
    )

    plt.legend()
    plt.title("Memory System Dynamics Over Time")
    plt.xlabel("Step")
    plt.ylabel("Signal")
    plt.show()
