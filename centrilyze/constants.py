classes = {
    "Not_Oriented": 0,
    "Oriented": 1,
    "Precieved_Not_Oriented": 2,
    "Precieved_Oriented": 3,
    "Slanted": 4,
    "Unidentified": 5,
    "No_sample": 6,
}
classes_inverse = {
    0: "Not_Oriented",
    1: "Oriented",
    2: "Precieved_Not_Oriented",
    3: "Precieved_Oriented",
    4: "Slanted",
    5: "Unidentified",
    6: "No_sample",
}

classes_reduced = {
    "Not_Oriented": 0,
    "Oriented": 1,
    "Slanted": 2,
    "Unidentified": 3,
    "No_sample": 4,
}
classes_reduced_inverse = {
    0: "Not_Oriented",
    1: "Oriented",
    2: "Slanted",
    3: "Unidentified",
    4: "No_sample",
}
annotation_mapping = {
    0: 0,
    1: 1,
    2: 0,
    3: 1,
    4: 2,
    5: 3,
    6: 4,
}