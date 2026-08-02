# Neuroimaging, the Oddball paradigm
My test projects on MNE-Python in field of  Neuroimaging.
This project marks my first journey into MNE-Python, where I explore the
"Oddball" paradigm to analyze how the human brain reacts to stimuli that deviate
from its expectations. In this experiment, the participant’s task was simple: to
monitor a stream of different sounds and images and react by pressing a button
only when a smiley face appeared on the screen. My pipeline traces the full path
of this neural journey, from the moment raw data is captured until it is
transformed into a 3D visualization of signal travel through the cortex. To
ensure the integrity of the data, I conducted an automated ICA to separate
brain-generated signals from artifacts like eye blinks and heartbeats; for the
blinks, I used a reference EOG channel, while for the heart, I applied a
cross-type phase synchrony method to identify rhythmic patterns without needing
chest electrodes. By decomposing the signal into 20 components, I was able to
identify which ones contributed the most "pollution" to each sensor, effectively
cleaning the data-stream by removing these specific artifact weights. After
visualizing the contrast between the contaminated and cleaned signals, I focused
on calculating the average evoked potentials for both the visual input and the
motor output to understand exactly when and where these responses emerge. I also
examined the prevailing frequencies within the signal to see how rhythms like
Alpha and Beta shift during the task. Reconstructing this activity into a 3D
model requires solving the inverse problem, which I like to think of through the
metaphor of a cloud and its shadow: we are trying to guess the real shape of the
"cloud" (internal brain activity) just by looking at its "shadow" (the
electrical current on the surface electrodes). This reconstruction is guided by
several biological principles: focusing on pyramidal neurons as the primary
signal generators, using MRI scans to respect the person's unique brain
structure, and assuming the brain is "lazy," meaning it follows the path of
least energy to transmit information. While the programming side is implemented
in my code, the scientific results are what truly puzzle and fascinate me,
particularly the lateralization of the motion potential. I noticed a much
stronger signal in the left hemisphere of the brain during the planning phase,
which suggests that the participant was right-handed, confirming how distinctly
motion and other senses are lateralized in our biology. Ultimately, this was an
incredible experience that allowed me to transport my existing knowledge of
brain functioning into the dimension of brain imaging, getting to know the
complex data formats and transformations that occur at every step of the
process.

