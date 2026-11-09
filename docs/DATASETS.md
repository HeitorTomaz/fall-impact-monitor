# Datasets

This file records the public datasets that Fall Impact Monitor may use for evaluation and training, with their source, terms and status.

**Status: no dataset has been downloaded or evaluated yet.** All entries below are candidates.

## Policy

- **Openly downloadable only.** Any published dataset that can be downloaded openly may be used privately by the installer for evaluation and training, including datasets with non-commercial or research-only terms.
- **No registration or request forms.** Datasets that require registration, a request form, a signed agreement or individual approval are out of scope.
- **Never redistributed.** This project never redistributes datasets or parts of them, even when the license would allow it.
- **Always cited.** Every dataset used is cited with its paper and source as requested by its authors, and its terms are recorded in this file.
- **Nothing committed.** Videos, frames, annotations and derived data (keypoints, features, crops) are never committed to Git or attached to issues and pull requests. Public reports contain aggregate metrics only. CI never uses third-party datasets.
- **No leakage.** Train, validation and test splits are made by person and by scene, never by clip or frame. If a dataset does not identify people or scenes, the split used is documented with the results.
- **Trained weights stay private.** Weights trained on non-commercial or research-only data are never published (see [`MODELS.md`](../MODELS.md)).
- Licenses below were collected during design and must be confirmed on the source page before a dataset is downloaded. Where sources disagree, the most restrictive reading applies.

## Candidate datasets

"Redistributable" describes what the license would allow; this project still never redistributes any dataset. "Private only" means evaluation and training by the installer, with no redistribution and no publication of trained weights.

| Dataset | Source URL | License / terms | Allowed use in this project | Availability notes | Status |
|---|---|---|---|---|---|
| CAUCAFall | <https://data.mendeley.com/datasets/7w7fccy7ky/4> | CC BY 4.0 (as reported by secondary sources; confirm on the Mendeley page) | Evaluation and training; license allows redistribution with attribution | Open download from Mendeley Data | Candidate, not downloaded |
| GMDCSA24 | <https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos> (archive: <https://doi.org/10.5281/zenodo.13354453>) | MIT (repository) / CC BY 4.0 (reported for a copy); confirm at the source | Evaluation and training; license allows redistribution with attribution | Open download | Candidate, not downloaded |
| SAFER-Activities | <https://safer-activities.github.io/> | CC BY-NC-SA 4.0 | Private only | Confirm that access is open; if a request form or registration is required, it is out of scope | Candidate, not downloaded |
| OmniFall | <https://huggingface.co/datasets/simplexsigil2/omnifall> | Non-commercial: CC BY-NC-SA 4.0 in the dataset card, CC BY-NC 4.0 in other metadata; treated as CC BY-NC-SA 4.0 | Private only | Annotations for the staged and in-the-wild parts refer to videos from other datasets, whose own terms and availability apply | Candidate, not downloaded |
| WanFall (OmniFall synthetic part) | <https://huggingface.co/datasets/simplexsigil2/wanfall> | Non-commercial: CC BY-NC 4.0 in the dataset card, CC BY-NC-SA 4.0 in other copies; treated as CC BY-NC-SA 4.0 | Private only | Synthetic videos | Candidate, not downloaded |
| UR Fall Detection | <http://fenix.ur.edu.pl/~mkepski/ds/uf.html> | CC BY-NC-SA 4.0 | Private only | Open download | Candidate, not downloaded |
| Le2i | <https://imvia.ube.fr/?p=346> | Non-commercial (CC BY-NC-SA 3.0 as reported by secondary sources; the source page states no license) | Private only | Availability uncertain: the download link was not available when last checked | Candidate, not downloaded |
| OOPS! | <https://oops.cs.columbia.edu/> (code: <https://github.com/cvlab-columbia/oops>) | CC BY-NC-SA 4.0 | Private only | Availability uncertain; confirm the download still works | Candidate, not downloaded |
| UP-Fall Detection | <https://sites.google.com/up.edu.mx/har-up/> | No license stated | Private only, and only if openly downloadable | Confirm that download is open, without registration or request | Candidate, not downloaded |
| Multiple Cameras Fall Dataset | <http://www.iro.umontreal.ca/~labimage/Dataset/> | No license stated | Private only, and only if openly downloadable | Confirm that download is open, without registration or request | Candidate, not downloaded |

## Out of scope

These datasets require registration, a request form or a signed agreement, so they are not used.

| Dataset | Source URL | Reason |
|---|---|---|
| NTU RGB+D 60 / 120 | <https://rose1.ntu.edu.sg/dataset/actionRecognition/> | Registration and request required |
| Toyota Smarthome | <https://project.inria.fr/toyotasmarthome> | Request required |
| ETRI-Activity3D | <https://ai4robot.github.io/resources/EULA_ETRIActivity3D_en.pdf> (license agreement) | Signed agreement and request required |
| CMDFall | <https://mica.edu.vn/perso/Tran-Thi-Thanh-Hai/CMDFALL.html> | Request required |
