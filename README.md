# Coastal Cod Re-ID
## Resources for developing individual re-id cod coastal cod (Gadus morhua)
This project contains code, dataset information and links to model weights for our project on deep-learning-based individual re-id of coastal cod.

!! This Repo is a Work in Progress !! Full functionality will be implemented by publishing of the associated paper.

For specific information on related datasets click the icons below:

<table>
  <tr>
    <td align="center" valign="middle"><a href="https://github.com/jdavidpoling/Cod-ID-Dataset"><img src="viz/codid_logov4.png" alt="Cod-ID" height="300"></a></td>
    <td align="center" valign="middle"><a href="https://github.com/jdavidpoling/FjordFish"><img src="viz/fjordfish_logo_v2.png" alt="Trained model" height="300"></a></td>
  </tr>
  <tr>
    <td align="center">Dataset for re-id of individual coastal cod</td>
    <td align="center">Dataset for general North Atlantic fish detection</td>
  </tr>
</table>

## Summary
Deep-learning-based individual re-identification (Re-ID) acts as a non-invasive way to monitor the movement and behavior of individual animals from visual data. This project seeks to develop re-id for coastal cod based on a combination of videos of tagged fish from a semi-natural mesocosm as well as video observations of fish from the wild.

Using video for re-id instead of individual still images allows for more diverse information to be accounted for in matching. We do this by using track-level embeddings based on [Recurrence over Video Frames (RoVF) for Animal Re-identification](https://link.springer.com/content/pdf/10.1007/s11263-025-02709-8.pdf). The figure below shows the diverse information which can be present across a series of frames in a video, allowing for a more informed matching process than using individual images.
<!--
Source - https://stackoverflow.com/a/12118349
Posted by waldyr.ar, modified by community. See post 'Timeline' for change history
Retrieved 2026-09-23, License - CC BY-SA 4.0
-->

<p align="center">
  <img src="https://github.com/jdavidpoling/Coastal-Cod-Individual-Re-Identification/blob/main/viz/track_masks_white.png" width="800" />
</p>


### Models tested:
All Rank-k (Rk) values are on unseen individuals (open-set re-id). Expect noticeably higher performance when testing on IDs seen in training.
"n/a" refers to untested metrics
Data split "Paper" refers to data split used in the associated publication:

| Model | Fine-tuned | Cod-ID version for fine-tuning | Data Split | R1 | R5 | R1 n<=5 | R1 n<=10 |
|-------|------------|--------------------------------|------------|----|----|---------|----------|
|MegaDescriptor-L-384|no|1.0|Paper|39.2|60.0|n/a|n/a|
|MegaDescriptor-L-384|yes|1.0|Paper|65.9|78.6|90.3|88.9|
|MiewID-msv3|no|1.0|Paper|16.1|35.5|n/a|n/a|
|MiewID-msv3|yes|1.0|Paper|37.2|55.2|88.3|79.4|
|DINOv3-vit7b16-pretrain-lvd1689m|no|1.0|Paper|32.1|45.9|n/a|n/a|
|DINOv3-vit7b16-pretrain-lvd1689m|yes|1.0|Paper|48.7|55.2|88.3|81.2|

## Fine-tuned model weights:
Fine-tuned model weights can be found in the linked Huggingface collection: [Cod Re-ID Models HF](https://hf.co/collections/jdpoling/cod-re-id)

## Citation
If you use the FjordFish dataset in your work, please cite the associated paper:
```
TBA
```
