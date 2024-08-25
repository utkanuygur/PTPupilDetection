# PTPupilDetection

Pupil detection using Pytorch implemented with U-Net and Mask R-CNN. Would highly recommend using the U-Net model due to it's superior accuracy and speed.

## Installation

1. Script is designed to run on Google Collab. Make sure you have the images for the dataset (not provided).
   
2. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/PTPupilDetection.git
    cd PTPupilDetection
    ```

3. Install labelme for manual labeling:
```sh
python -r pip install labelme
```
5. Other requirements are installed through the first cells of all the Jupyter Notebooks. 

## Usage

### U-Net

To run the U-Net model, open and execute the Jupyter notebook unet_model.ipynb.

### Mask R-CNN

To run the Mask R-CNN model, open and execute the Jupyter notebook MaskRCNN_model.ipynb.

### Utilities

Various utility scripts are available in the Utils directory:

- `combined_loss.py`: Contains funbction for combined loss calculation that includes circularity and connectivity loss.
- `draw_on_image.py`: Functions to draw on images to manually label them, labelme is also recommended. Can be installed through pip. 
- `extract.ipynb`: Jupyter notebook help generate the masks for the dataset. Is not perfectly accurate for all images.
  Approve the accurate ones to add to the dataset and manually label the edge cases using labelme.
- `approve.py`: Helps approve the extracted masks and add to the dataset.
- `mask.py`: Functions related to mask operations.
- `predict_mask.py`: Script to predict masks.
- `test.py`: Script for testing the models.
- `utils.py`: General utility functions.

## Models

Pre-trained best models can be found here: https://drive.google.com/drive/folders/1ojo2rBHtRu7r0m9UegtAEZXemO8SgQh8

## Contributing

Contributions are welcome! Feel free to reach out, open an issue, or submit a pull request.

## License

This project is licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
