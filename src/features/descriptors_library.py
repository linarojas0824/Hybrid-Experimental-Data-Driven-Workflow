DESCRIPTORS = {
    "atomic_radius": ["avg","mismatch","range"],
    "metallic_radius": ["avg","mismatch","range"],
    "electronegativity_pauling": ["avg","mismatch","range"],
    "electronegativity_allen": ["avg","mismatch","range"],
    "valence_electrons": ["avg","mismatch","range"],
    "density": ["avg","mismatch","range"],
    "electron_affinity": ["avg","mismatch","range"],
    "ionization_energy_1": ["avg","mismatch","range"],
    "ionization_energy_2": ["avg","mismatch","range"],
    "valence_s": ["avg","mismatch","range"],
    "valence_d": ["avg","mismatch","range"]
}

DESCRIPTORS_ANN = {
    "atomic_radius": ["avg","mismatch"],
    "electronegativity_pauling": ["mismatch"],
    "valence_electrons": ["avg"],
    "cols_names": ['Cu','Ni','Al','r','del_r','del_EN','S','VEC'] # Since the columns names are different
    # The real columns names are saved
}

def select_descriptor_types(descriptor_dict, selection):
    selected = {}

    for descriptor_name, selected_types in selection.items():
        selected[descriptor_name] = [
            dtype for dtype in descriptor_dict[descriptor_name]
            if dtype in selected_types
        ]

    return selected