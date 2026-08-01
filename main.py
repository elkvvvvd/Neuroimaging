import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA

# Loading and preparing data
mne.set_log_level('WARNING') #keeping konsole clean (reacts only if sth going wrong)
sample_data_folder = mne.datasets.sample.data_path()
sample_data_raw_file = sample_data_folder / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"

# Loading data to memory, so that all the processes will be faster (preload=True)
raw = mne.io.read_raw_fif(sample_data_raw_file, preload=True)
raw.pick_types(meg=True, eeg=True, stim=True, eog=True)  # Keeping chanel that we will use

# Automated detecting ICA components, for blinks using reference chanel EOG 061 and for heart Cross-Type Phase
# Synchrony method (using this because not every experiment could have ecg electrodes and I want
# my code to be reusable)
print("Looking for ICA")
raw_dirty = raw.copy()  # copy to compare before and after cleaning

ica = ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)

# Looking for blinks and heart components
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="EOG 061", threshold=3.0)
ica.exclude.extend(eog_indices)

try:
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, method='ctps')
    ica.exclude.extend(ecg_indices)
except Exception:
    print("ECG wasn't detected automaticly.")

#cleaning the raw object according to found components
ica.apply(raw)
print(f"Deleted components: {ica.exclude}")

#Visualising Before and After cleaning
if ica.exclude:
    # Retrieve the mixing matrix [n_channels_in_ica x n_components]
    # This matrix represents the spatial distribution of each ICA component.
    comp_matrix = ica.get_components()

    # Extract weights only for the artifact components
    # using absolute values to measure the magnitude of the artifact's impact,
    # regardless of polarity. Summing across rows provides a "pollution score" per channel.
    excluded_weights = np.abs(comp_matrix[:, ica.exclude])
    total_weights = excluded_weights.sum(axis=1)

    # Identify the 10 most polluted channels within the ICA channel subset
    # np.argsort returns indices in ascending order, so we take the last 10.
    top_10_local_idxs = np.argsort(total_weights)[-10:]

    # Data Mapping: Bridge the gap between ICA space and Raw object space
    # Since ICA used a subset of channels, we map local ICA indices back to
    # global channel names, then find their indices in the original Raw object.
    plot_ch_names = [ica.ch_names[i] for i in top_10_local_idxs]
    final_order_indices = [raw.ch_names.index(name) for name in plot_ch_names]

    print(f"Top artifact-impacted channels: {plot_ch_names}")

    # Visualization: Side-by-side Before vs After comparison
    # We use n_channels=10 to display all selected traces in a single view.

    # Contaminated signal (Original)
    raw_dirty.plot(order=final_order_indices, start=12, duration=5,
                   title="BEFORE ICA: Raw signal with artifacts",
                   n_channels=10, block=False)

    # Cleaned signal (Post-ICA)
    # block=True holds the execution to keep the windows open for inspection.
    raw.plot(order=final_order_indices, start=12, duration=5,
             title="AFTER ICA: Cleaned signal",
             n_channels=10, block=True)
# Creating 2 types of epochs (SENSORY vs MOTOR)
# so that we can see them separetely
event_dict = {
    "auditory/left": 1, "auditory/right": 2,
    "visual/left": 3, "visual/right": 4,
    "smiley": 5, "buttonpress": 32,
}
events = mne.find_events(raw, stim_channel="STI 014")
reject_criteria = dict(mag=4000e-15, grad=4000e-13, eeg=150e-6, eog=250e-6)

# Epochs for visual
epochs_stim = mne.Epochs(
    raw, events, event_id=event_dict,
    tmin=-0.2, tmax=0.5,
    baseline=(-0.2, 0),
    reject=reject_criteria, preload=True
)

# Б. Epochs for motion (Readiness potential)
epochs_motor = mne.Epochs(
    raw, events, event_id={'buttonpress': 32},
    tmin=-1.0, tmax=0.2,
    baseline=(-1.0, -0.8),  # Baseline before clicking
    reject=reject_criteria, preload=True
)

del raw

# Analyzing evoked potentials
vis_evoked = epochs_stim["visual"].average()
motor_evoked = epochs_motor.average()

# Visual response (classic N100 peak in the occipital lobe)
vis_evoked.plot(window_title="Visual Evoked Response", time_unit='s')

# Motor response (Readiness Potential / Bereitschaftspotential build-up before the press)
motor_evoked.plot(window_title="Motor Readiness Potential", time_unit='s')


# TIME-FREQUENCY REPRESENTATION (TFR) ---

frequencies = np.arange(7, 30, 3)

# TFR analysis for visual stimuli (focusing on Alpha rhythm changes)
power_vis = epochs_stim["visual"].compute_tfr("morlet", freqs=frequencies, n_cycles=2, average=True)
power_vis.plot(["MEG 1932"], baseline=(-0.2, 0), mode='logratio', title="Visual: Alpha Response")

# TFR analysis for motor activity (focusing on Beta desynchronization during movement planning)
power_motor = epochs_motor.compute_tfr("morlet", freqs=frequencies, n_cycles=3, average=True)
power_motor.plot(["EEG 021"], baseline=(-1.0, -0.8), mode='logratio', title="Motor: Beta Planning")


#  3D source localization (Inverse Solution)
# Load the pre-computed inverse operator
inverse_operator_file = sample_data_folder / "MEG" / "sample" / "sample_audvis-meg-oct-6-meg-inv.fif"
inv_operator = mne.minimum_norm.read_inverse_operator(inverse_operator_file)
subjects_dir = sample_data_folder / "subjects"
lambda2 = 1.0 / 3.0 ** 2

# Localizing the visual signal (expected activity in the primary visual cortex at 100 ms)
stc_vis = mne.minimum_norm.apply_inverse(vis_evoked, inv_operator, lambda2, method="dSPM")
brain_vis = stc_vis.plot(
    initial_time=0.1, hemi="split", views=["lat", "med"],
    subjects_dir=subjects_dir, title="3D: Visual Processing (100ms)"
)

# Localizing the motor planning activity (expected in the motor cortex ~200 ms BEFORE the button press)
stc_motor = mne.minimum_norm.apply_inverse(motor_evoked, inv_operator, lambda2, method="dSPM")
brain_motor = stc_motor.plot(
    initial_time=-0.2, hemi="split", views=["lat", "med"],
    subjects_dir=subjects_dir, title="3D: Motor Planning (-200ms)"
)

plt.show()
