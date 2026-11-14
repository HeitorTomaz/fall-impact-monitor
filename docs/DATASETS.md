# Datasets

This file records the public datasets that Fall Impact Monitor may use for evaluation and training, with their source, terms and status.

**Status:** CAUCAFall, GMDCSA24 and UR Fall Detection were downloaded locally by the maintainer on 2026-11-09 and are used for evaluation (`fim eval`, results in [`evaluation.md`](evaluation.md)). All other entries are candidates. Availability was checked again on 2026-11-12 for datasets with cameras mounted high (ceilings, high walls), the usual installation.

## Policy

- **Openly downloadable only.** Any published dataset that can be downloaded openly may be used privately by the installer for evaluation and training, including datasets with non-commercial or research-only terms.
- **No registration or request forms.** Datasets that require registration, a request form, a signed agreement or individual approval are out of scope.
- **Never redistributed.** This project never redistributes datasets or parts of them, even when the license would allow it.
- **Always cited.** Every dataset used is cited with its paper and source as requested by its authors, and its terms are recorded in this file.
- **Nothing committed.** Videos, frames, annotations and derived data (keypoints, features, crops) are never committed to Git or attached to issues and pull requests. Public reports contain aggregate metrics only. CI never uses third-party datasets.
- **No leakage.** Train, validation and test splits are made by person and by scene, never by clip or frame. If a dataset does not identify people or scenes, the split used is documented with the results.
- **Trained weights stay private.** Weights trained on non-commercial or research-only data are never published (see [`MODELS.md`](../MODELS.md)).
- Licenses below were collected during design and must be confirmed on the source page before a dataset is downloaded. Where sources disagree, the most restrictive reading applies.

## Manifest and download

[`datasets.toml`](../datasets.toml) records each evaluated dataset: source, version,
license and its URL, attribution, status (`private-use` or `redistributable`), the
published SHA-256 of every file to download, and the train / validation / test splits
(by subject; by sequence for UR Fall, which has no subject labels).

```bash
docker compose --profile dev run --rm dev uv run --frozen fim datasets list
docker compose --profile dev run --rm dev uv run --frozen \
  fim datasets fetch gmdcsa24 --dest test_videos/datasets/gmdcsa24
```

- `fetch` writes outside the repository, or under the git-ignored `test_videos/`, and
  refuses any other folder inside the repository.
- Every file is kept only when its SHA-256 matches the manifest. A file already present
  with the right hash is not downloaded again.
- Archives are extracted into `extracted/` only when none of their entries would leave
  that folder (no absolute paths, no `..`, no links). Nothing in an archive is executed.
- **CAUCAFall** is fetched by hand: Mendeley Data serves about 20,000 files one by one.
  Download the videos and the per-frame label files of v4 from the dataset page (skip
  the PNG frames, 8 GB), keep the folder layout `CAUCAFall/Subject.N/<activity>/`, and
  check them against the SHA-256 values that Mendeley publishes for each file.
- **Multiple Cameras Fall Dataset (MCFD)** is downloaded by hand: its site has an
  automatic browser check, which this project never bypasses. From a browser, download
  `dataset.zip`, `technicalReport.pdf` and `Erratum-MCF.pdf` into one folder, then run
  `fim datasets fetch mcfd --dest <that folder>`: it checks the three files against the
  manifest and extracts the archive safely into `extracted/` (nothing is downloaded).
  Evaluate with `fim eval --dataset mcfd --root <folder>/extracted/dataset`.
- The local evaluation folder (`test_videos/datasets/`) is kept under 15 GB; with
  GMDCSA24, UR Fall, CAUCAFall and MCFD extracted it holds about 9.4 GB.

## Candidate datasets

"Redistributable" describes what the license would allow; this project still never redistributes any dataset. "Private only" means evaluation and training by the installer, with no redistribution and no publication of trained weights.

| Dataset | Source URL | License / terms | Allowed use in this project | Availability notes | Status |
|---|---|---|---|---|---|
| CAUCAFall | <https://data.mendeley.com/datasets/7w7fccy7ky/4> (DOI 10.17632/7w7fccy7ky.4) | CC BY 4.0, confirmed in the dataset metadata (v4) | Evaluation and training; license allows redistribution with attribution | Open download from Mendeley Data, no login; a v5 exists | Downloaded (v4, videos and labels), used for evaluation |
| GMDCSA24 | <https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos> (archive: <https://doi.org/10.5281/zenodo.13354453>) | CC BY 4.0 in the Zenodo record, MIT in the repository; treated as CC BY 4.0 | Evaluation and training; license allows redistribution with attribution | Open download | Downloaded (v2.1), used for evaluation |
| OmniFall | <https://huggingface.co/datasets/simplexsigil2/omnifall> | Non-commercial: CC BY-NC-SA 4.0 in the dataset card, CC BY-NC 4.0 in other metadata; treated as CC BY-NC-SA 4.0 | Private only | Annotations for the staged and in-the-wild parts refer to videos from other datasets, whose own terms and availability apply | Candidate, not downloaded |
| WanFall (OmniFall synthetic part) | <https://huggingface.co/datasets/simplexsigil2/wanfall> | Non-commercial: CC BY-NC 4.0 in the dataset card, CC BY-NC-SA 4.0 in other copies; treated as CC BY-NC-SA 4.0 | Private only | Synthetic videos | Candidate, not downloaded |
| UR Fall Detection | <http://fenix.ur.edu.pl/~mkepski/ds/uf.html> | CC BY-NC-SA 4.0, "intended for non-commercial academic use" (source page) | Private only | Open download | Downloaded (cam0 videos and frame labels), used for evaluation |
| Le2i (ImViA Fall Detection Dataset) | <https://search-data.ubfc.fr/imvia/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html> (DOI 10.25666/DATAUBFC-2024-04-09) | CC BY-NC-SA 3.0, stated on the record (2026-11-12) | Private only | Record "available"; the file link (`FallDataset.zip`, 8.95 GB) redirects and was not verified end to end. Single camera, 320×240, 25 fps, four rooms; camera height not stated. Does not fit the local 15 GB evaluation budget next to the four evaluated datasets | Candidate, not downloaded |
| OOPS! | <https://oops.cs.columbia.edu/> (code: <https://github.com/cvlab-columbia/oops>) | CC BY-NC-SA 4.0 | Private only | Not checked again on 2026-11-12; in-the-wild clips, mostly not falls of older adults at home | Candidate, not downloaded |
| UP-Fall Detection | <https://sites.google.com/up.edu.mx/har-up/> | No dataset license (the paper is CC BY 4.0); citation requested | Private only | Google Drive links, no registration seen; two cameras at 1.82 m (lateral and frontal); about 812 GB in full | Candidate, not downloaded |
| Multiple Cameras Fall Dataset (MCFD) | <https://www.iro.umontreal.ca/~labimage/Dataset/> | No license stated; the page asks to cite the technical report (Auvinet et al., DIRO Tech. Report 1350, 2010) | Private only | **Eight wide-angle cameras mounted high in the corners of one room**: the best public match for ceiling and high-wall cameras. 24 scenarios by one actor: 23 contain a fall (one contains two), one only daily activities (sitting, crouching, lying on a sofa, housekeeping); 720×480. The files declare 120 fps (players show them four times too fast) but were captured at 30, as the authors state (Auvinet et al., IEEE TITB 2011: "720 × 480 at 30 fps"; see `evaluation/mcfd.py`). Fall frame ranges come from the technical report and its 2025 erratum; the camera delays of the report match the files' frame counts in 23 of 24 scenarios (the 24th, without falls, differs). Behind an automatic browser check: downloaded by hand (3.78 GB) | Downloaded by the maintainer (2026-11-13), used for evaluation (#69) |
| EDF / OCCU | <https://zenodo.org/records/15494102> | CC BY 4.0 | Evaluation; license allows redistribution with attribution | Open; 26.9 GB; two Kinect v1 views; RGB content not confirmed (described as depth data) | Candidate, not downloaded |

## Out of scope

These datasets require registration, a request form or a signed agreement, so they are not used.

| Dataset | Source URL | Reason |
|---|---|---|
| NTU RGB+D 60 / 120 | <https://rose1.ntu.edu.sg/dataset/actionRecognition/> | Registration and request required |
| Toyota Smarthome | <https://project.inria.fr/toyotasmarthome> | Request required |
| ETRI-Activity3D | <https://ai4robot.github.io/resources/EULA_ETRIActivity3D_en.pdf> (license agreement) | Signed agreement and request required |
| CMDFall | <https://mica.edu.vn/perso/Tran-Thi-Thanh-Hai/CMDFALL.html> | Request required |
| SAFER-Activities | <https://safer-activities.github.io/> | The Hugging Face repository is gated (account and agreement required), checked 2026-11-12 |
| MUVIM | arXiv 2206.12740 | Request by email and a data privacy waiver (ceiling-mounted RGB would otherwise fit well) |
| HQFSD (KU Leuven) | Healthcare Technology Letters, 2016 | The datasets page now requires an institutional login |
