# LDCT-COVID
<h4>Human-level COVID-19 Diagnosis from Low-dose CT Scans Using a Two-stage Time-distributed Capsule Network.</h4>

LDCT-COVID is a two-stage time-distrubeted capsule network to classify volumetric low-dose CT (LDCT) scans as COVID-19, CAP, and Normal. LDCT scan protocols reduce the radiation exposure close to that of a single X-Ray, while maintaining an acceptable resolution for diagnosis purposes. The proposed farmework takes segmented lung regions as inputs. The segmentation is obtained from a recently developed <a href="https://github.com/JoHof/lungmask"> U-Net based segmentation model</a>.

The first stage capsule network is presented in <a href="https://github.com/ShahinSHH/CT-CAPS"> Here</a> and classifies individual scans as having an evidence of infection or not. 10 most probable slices with infection are then selected as inputs to the second stage, which consists of time-distributed capsule networks, referring to processing slices at the same time through the same model. In this stage, classification probabilities generated from individual slices go through a global max pooling operation to make the final decision. The second stage also takes into account the infection probability calculated in the first stage.

<img src="https://github.com/ParnianA/LDCT-COVID/blob/main/Figures/Framework.png"/>
