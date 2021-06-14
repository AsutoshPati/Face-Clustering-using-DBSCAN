# Face Clustering Algorithm

---

This code shows face clustering using [DBSCAN](
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) (Density Based Spatial Clustering of
Applications with Noise) algorithm. Using this codes you can create face database of fine images (by removing blurred
images) and then you can easily apply face clustering on the newly created database.

After clustering unique faces you can also find best pictures from each cluster.

If you find this code helpful please don't forget to give star and follow.

---

## Requirements

> pip install dlib@https://pypi.python.org/packages/da/06/bd3e241c4eb0a662914b3b4875fc52dd176a9db0d4a2c915ac2ad8800e9e/dlib-19.7.0-cp36-cp36m-win_amd64.whl#md5=b7330a5b2d46420343fbed5df69e6a3f

> pip install face-recognition==1.3.0

> pip install opencv-python==4.5.2.52

> pip install scikit-learn==0.24.2

## How to use

* Just put all the cropped face (as this code doesn't have crop face code) images into "face_images" directory.
  (you can add the crop face code in "create_face_database" function at the marked place in "get_encodings.py")
  
* Then run the "example.py" file to stat the face database making procedure. If you want to use it in your code; then 
  you need to put "get_encodings.py" in your project directory and have to use the function "create_face_database".
  ```python
  import get_encodings as enc
  root_path = 'face_images' # path to your
  dir_name, data_len = enc.create_face_database(root_path)
  ```

* Once face database is created it will start doing face cluster immediately after completing the result directory where 
  clustered faces are stored will be displayed. If you want to add "face_cluster.py" to your project directory and have 
  to use "do_cluster" function.
  ```python
  import face_cluster as cluster
  dir_name = '' # path to face database directory
  unq_faces, res_dir = cluster.do_cluster(dir_name)
  ```

* You can find best picture from each class using "get_best_pics" function once clustering is completed, and you have 
  number of clusters and path to result directory.
  ```python
  import face_cluster as cluster
  unq_faces = 0 # num of clusters after clustering
  res_dir = '' # path to stored face cluster result
  unq_fine_faces = cluster.get_best_pics(unq_faces, res_dir)
  ```
