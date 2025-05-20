This repository implements an approximately-supervised learning approach for unpaired MR-to-CT synthesis. Below is the dataset preparation and training process.  
# Dataset Preparation
The required dataset structure is as follows:
```
datasets/
├── trainA/   # Directory containing unpaired MR images for training
├── trainB/   # Directory containing unpaired CT images for training
├── testA/    # Directory containing paired MR images for testing
├── testB/    # Directory containing paired CT images for testing
```
Segment the medical image slices from trainA and trainB into smaller patches, saving them into patchA and patchB respectively. 
```
datasets/
├── patchA/    # Directory containing Source Domain Patch for training
├── patchB/    # Directory containing Target Domain Patch for training
├── trainA/   # Directory containing unpaired Source Domain Slice for training
├── trainB/   # Directory containing unpaired Target Domain Slice for training
├── testA/    # Directory containing paired Source Domain Slice for testing
├── testB/    # Directory containing paired Target Domain Slice for testing
```
Based on an appropriate similarity metric (e.g., HOG similarity, NMI, etc., specific to the task), add patch pairs with high similarity to a DataFrame. The DataFrame will have two columns: 
```
| Source Domain Patch Path        | Target Domain Patch Path        |
|---------------------------------|---------------------------------|
| `patchA/source_slice01_patch001.png` | `patchB/target_slice08_patch056.png` |
| `patchA/source_slice01_patch002.png` | `patchB/target_slice12_patch032.png` |
| `patchA/source_slice02_patch015.png` | `patchB/target_slice03_patch078.png` |
| ...                             | ...                             |
| `patchA/source_sliceX_patchY.png`  | `patchB/target_sliceM_patchN.png`  |
```
the left column for the source domain patch path and the right column for the target domain patch path. Finally, save this DataFrame as a pickle file for training.

# Training
Use train.py to train the model using the generated patch pairs.
Example Command:
```python train.py --batch_size 16 --train_size 64 --num_epochs 100 --learning_rate 2e-4 --dataset_path datasets/unpaired_64patch_train.pkl --save_path weights/```
# Testing
Use test.py to evaluate the trained model on the paired test dataset (testA and testB).
Example Command:
```python test.py --mr_dir datasets/testA --ct_dir datasets/testB --size 256 --model_path weights/as64_attunet_100_epoch.pt --output_dir results/```
