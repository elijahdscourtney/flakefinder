# Flake finder
> **Warning:** currently targets monolayer graphene on SiO2, with specific Leica color settings
This is not ready for general use

### Usage
```bash
python monolayer_finder.py --q <PATH>
```

The input file specified by <PATH> should be a text document containing ONLY Input and Output paths for all imagesets you wish to queue, formatted as below:

```
InputDir: C:\DummyIn\01_19_23_1\Scan 002\TileScan_001\, OutputDir: C:\DummyIn\01_19_23_1\Scanned\
InputDir: C:\DummyIn\01_18_23_1b\Scan 002\TileScan_001\, OutputDir: C:\DummyIn\01_18_23_1b\Scanned\
```

The code expects the TileScan_001 directory to contain all the images from a scan, plus the leicametadata folder with a .xlif inside to position the images

Do not specify directories on the Google Drive or the computer may crash upon execution.
The OutputDir will be created if it does not exist, and a Summary.txt file of performance and parameters will be written to each OutputDir after each InputDir has been analyzed.
Performance ~1-10s per 20 megapixel file at scale=1, k=4 on the Leica computer, depending on how many filter steps the image passes

### Parameters
- `threadsave` — How many logical processors the program does NOT use on your CPU. Can reduce to improve computer usability while program is running at cost of runtime
- `boundflag` — 0/1 to turn off/on a function that calculates flake area and border pixels. Processing intensive but gives useful feedback to both user and developer
- `k` — Scale factor for the image before finding background RGB. Currently set to 4, imscale = (256\*k,171\*k) - for full 20MP resolution, k=21

#### Detection
- `UM_TO_PX` — The conversion factor between microns (μm) and pixels in the scan image.
- `FLAKE_MIN_AREA_UM2` — The minimum area (in μm<sup>2</sup>) a detected contour must have to be considered a valid flake.
- `FLAKE_MIN_EDGE_LENGTH_UM` —  The minimum length (in μm) a detected flake edge must have to be counted.
- `FLAKE_ANGLE_TOLERANCE_RADS` — The tolerance (in radians) an angle between two detected lines can be from a multiple of 30 degrees to still be labelled as interesting.
- `FLAKE_R_CUTOFF` — The maximum ratio of perimeter<sup>2</sup> / area a detection can have to be considered a flake. Increasing this will allow detection of thin, stripey flakes, but may also cause false positives with non-flake noise.

#### Morphology
- `OPEN_MORPH_SIZE` — The size of the `MORPH_OPEN` operation applied to the image mask to reduce noise. Larger = more non-flake
particles removed from the mask, but can possibly lead to more erosion of actual flakes.
- `CLOSE_MORPH_SIZE` — The size of the `MORPH_CLOSE` operation applied to the image mask to fill in small gaps in flakes before
`MORPH_OPEN`. Larger = more gaps filled, but can possibly lead to filling gaps that are actually not part of the flake.
- `OPEN_MORPH_SHAPE` — The structuring element shape of the `MORPH_OPEN` operation.
- `CLOSE_MORPH_SHAPE` — The structuring element shape of the `MORPH_CLOSE` operation.

#### Labelling
- `BOX_OFFSET` — The offset, in pixels, each side of the labelled bounding box should be from the actual bounding box of the flake.
- `BOX_RGB` — The RGB tuple of the labelled bounding box.
- `BOX_THICKNESS` — The thickness, in pixels, of each side of the labelled bounding box.
- `FONT` — The font to use for text labels.

### Scripts
| Name                  | Description                                                                            | Arguments                                                                                                |
|-----------------------|----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| `monolayer_finder.py` | Parses scan images and labels detected monolayer flakes. See [Usage](#usage) for more. | `--q`: path to an input file that specifies input and output directories to parse (see [Usage](#usage)). |
<!-- TODO: parstitch.py -->

`/scripts` also contains helper scripts for development:

| Name                | Description                                                                                                                                                                                      | Arguments                                                                                                                                                                                                                                                                                |
|---------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `gdrive_unzip.py`   | Helper script to unzip and merge chunked zips created when downloading a large folder from Google Drive.                                                                                         | Given a zip file with name `04_03_23_EC_1-20230706T020731Z-006.zip`,<br/><br/>`--i`: path to the directory containing the zips.<br/>`--n`: the name of the original directory (ex. `04_03_23_EC_1`).<br/>`--t`: the download timestamp appended to each folder (ex. `20230706T020731Z`). |
| `threshold_test.py` | Benchmarks and shows each step of the flake finding process on a given list of stage images for testing / tuning.                                                                                | `--s:`: scan stages to test (passed as a list, ex. `--s 246 248 250`)                                                                                                                                                                                                                    |
| `equalize_test.py`  | Benchmarks and shows each step of the flake finding process using the new histogram equalization method on a given list of stage images.                                                         | *(See above)*                                                                                                                                                                                                                                                                            |
| `histogram_test.py` | Displays color histograms of the given stage image(s), showing RGB, HSV, and grayscale channels before and after masking for flake color and where the expected background and flake colors lie. | *(See above)*                                                                                                                                                                                                                                                                            |
| `edge_test.py`      | Displays the results of running Sobel and Laplace edge detection algorithms on given stage image(s).                                                                                             | *(See above)*                                                                                                                                                                                                                                                                            |
