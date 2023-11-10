from random import randint


def generate_query(query_type, range):
    if query_type == "point":
        start = randint(0, range - 1)
        return (start, start)
    elif query_type == "short":
        length_level = 10
    elif query_type == "medium":
        length_level = 1000
    elif query_type == "long":
        length_level = 100000
    else:
        raise ValueError("Unknown query type")

    # Shortest query length is 2
    multi = randint(1, 9)
    length = randint(multi * length_level, (multi + 1) * length_level)
    start = randint(0, range - length)
    return (start, start + length)
