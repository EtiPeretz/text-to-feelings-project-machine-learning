def find_label_from_vector(vector, feelings):
    feeling = ""
    vector_split = vector.split()
    vector_as_list = list(map(float, vector_split))
    for i in range(len(vector_as_list)):
        feeling = feelings[i] if vector_as_list[i] == 1 else None
        if feeling:
            break
    if feeling:
        return feeling.strip()
    else:
        raise Exception("Cannot convert vector")

