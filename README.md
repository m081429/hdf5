# hdf5

## Process
* compress images to hdf5
```
ls /data2/Naresh/data/BACH/final_train_test_val/
test  train  val

ls /data2/Naresh/data/BACH/final_train_test_val/train/
0  1

ls /data2/Naresh/data/BACH/final_train_test_val/train/0|head -5
Benign_b001.tif_0.jpg
Benign_b001.tif_10.jpg
Benign_b001.tif_11.jpg
Benign_b001.tif_12.jpg
Benign_b001.tif_13.jpg

ls /data2/Naresh/data/BACH/final_train_test_val/train/1|head -5
InSitu_is001.tif_0.jpg
InSitu_is001.tif_10.jpg
InSitu_is001.tif_11.jpg
InSitu_is001.tif_12.jpg
InSitu_is001.tif_13.jpg

python DataPrep_sep.py
```
* Run the model
```
python train_simplekeras_hdf5.py

```
