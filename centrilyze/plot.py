import os
from pathlib import Path

import matplotlib.pyplot as plt

from centrilyze import constants

from loguru import logger

def get_particle_fig(particle_data, particle_key, particle_annotations):

    fig, axs = plt.subplots(nrows=1, ncols=20, figsize=(25, 2))

    axs[0].set_title(f"particle: {[particle_key]}")
    for key, frame in particle_data.items():
        axs[key[2]].imshow(frame["image"])
        axs[key[2]].set_xlabel(particle_annotations[key])

    return fig


def save_particle_fig(particle, path, particle_key, particle_annotations):
    particle_fig = get_particle_fig(particle,particle_key, particle_annotations)
    particle_fig.savefig(str(path))
    plt.close()


def save_state_figs(particles, annotations, state_dir):
    for particle_key, particle in particles.items():
        path = state_dir / f"{str(particle_key)}.png"
        particle_annotations = annotations[particle_key]
        save_particle_fig(particle, path, particle_key, particle_annotations)


def save_all_state_figs(states, testset, output_dir, reannotations,):
    for state, particle_keys in states.items():
        logger.debug(f"Saving figures for state: {state}")

        particles = {particle_key: testset.get_sequence_data(*particle_key) for particle_key in particle_keys}
        annotations = {}
        for particle_key, particle_data in particles.items():

            annotations[particle_key] = {}
            for key, frame in particle_data.items():
                annotations[particle_key][key] =  constants.classes_reduced_inverse[reannotations[key[0]][key[1]][key[
                    2]][
                    "assigned"]]
        state_dir = Path(output_dir) / str(state)
        if not state_dir.exists():
            os.mkdir(state_dir)
        save_state_figs(particles, annotations, state_dir)


