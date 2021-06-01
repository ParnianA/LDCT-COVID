# LDCT-COVID
<h4>Human-level COVID-19 Diagnosis from Low-dose CT Scans Using a Two-stage Time-distributed Capsule Network.</h4>

LDCT-COVID is a two-stage time-distributed capsule network to classify volumetric low-dose CT (LDCT) scans as COVID-19, CAP, and Normal. LDCT scan protocols reduce the radiation exposure close to that of a single X-Ray, while maintaining an acceptable resolution for diagnosis purposes. The proposed framework takes segmented lung regions as inputs. The segmentation is obtained from a recently developed <a href="https://github.com/JoHof/lungmask"> U-Net based segmentation model</a>.

The first stage capsule network is presented in <a href="https://github.com/ShahinSHH/CT-CAPS"> Here</a> and classifies individual scans as having an evidence of infection or not. 10 most probable slices with infection are then selected as inputs to the second stage, which consists of time-distributed capsule networks, referring to processing slices at the same time through the same model. In this stage, classification probabilities generated from individual slices go through a global max pooling operation to make the final decision. The second stage also takes into account the infection probability calculated in the first stage.

<img src="https://github.com/ParnianA/LDCT-COVID/blob/main/Figures/Framework.png"/>

It is also possible to merge the output probabilities of the three classes (COVID-19, CAP, normal) with the 8 clinical data (demographic and symptoms,  i.e., sex, age, weight, and presence or absence of 5 symptoms of cough, fever, dyspnea, chest pain, and fatigue) and feed to a multi-layer perceptron (MLP) model for the final decision. 

<img src="https://github.com/ParnianA/LDCT-COVID/blob/main/Figures/MLP.png"/>

<h3>Note : Please don’t use LDCT-COVID as the self-diagnostic model without performing a clinical study and consulting with a medical specialist.</h3>

## Dataset
LDCT scans are publicly available <a href="https://ieee-dataport.org/open-access/covid-19-low-dose-and-ultra-low-dose-ct-scans"> Here</a> and CAP dataset is obtained from <a href="https://springernature.figshare.com/articles/dataset/COVID-CT-MD_COVID-19_Computed_Tomography_Scan_Dataset_Applicable_in_Machine_Learning_and_Deep_Learning/12991592"> Here</a>.

## Requirements
* Tested with tensorflow 1.14.0 and keras 2.2.4<br>
-- Try tensorflow.keras instead of keras if it doesn't work in your system.
* Python 3.7
* Numpy

## Citation
If you found the provided code and the related paper useful in your research, please consider citing:

```
@misc{afshar2021humanlevel,
      title={Human-level COVID-19 Diagnosis from Low-dose CT Scans Using a Two-stage Time-distributed Capsule Network}, 
      author={Parnian Afshar and Moezedin Javad Rafiee and Farnoosh Naderkhani and Shahin Heidarian and Nastaran Enshaei and Anastasia Oikonomou and Faranak Babaki Fard and Reut Anconina and Keyvan Farahani and Konstantinos N. Plataniotis and Arash Mohammadi},
      year={2021},
      eprint={2105.14656},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}



