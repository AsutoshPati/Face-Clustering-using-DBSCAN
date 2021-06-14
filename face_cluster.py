"""
    Author              : Asutosh Pati
    Date of Creation    : 15 MAY 2021
    Purpose             : Add on Module to create create face cluster and find the best pictures
                          from each cluster.
    Version             : V 1.0.0
"""

from sklearn.cluster import DBSCAN
import numpy as np
import cv2
import pickle
import os


def do_cluster(src_dir: str, jobs: int = -1):
    """
    This function applies face clustering upon face data exist in a single folder and
    save the unique faces in a result directory with labelling.

    Parameters
    ----------
    src_dir: str
        Path to root directory of generated face database.
    jobs: int
        Number of parallel jobs to run. Default -1.

    Returns
    -------
    num_unique_faces: int
        Number of unique faces found during clustering excluding outliers.
    res_dir: str
        Name of result directory.

    """
    f = open(f'{src_dir}/face_encodings.p', 'rb')
    data = pickle.load(f)
    f.close()

    # you can modify the eps value to fine tune your model
    # in my case 0.41 is working fine.
    clt = DBSCAN(eps=0.41, metric="euclidean", n_jobs=jobs)
    encodings = [d["encoding"] for d in data]
    clt.fit(encodings)
    labels_id = np.unique(clt.labels_)
    num_unique_faces = len(np.where(labels_id > -1)[0])
    print('unique faces found: ', num_unique_faces)
    res_dir = f'result_{src_dir}'
    for labelID in labels_id:
        # print("[INFO] faces for face ID: {}".format(labelID))
        idxs = np.where(clt.labels_ == labelID)[0]
        os.makedirs(f'result_{src_dir}/{labelID}')
        ctr = 0
        for i in idxs:
            image = cv2.imread(data[i]["img_path"])
            cv2.imwrite(f'result_{src_dir}/{labelID}/{ctr}.jpg', image)
            ctr += 1
    print(f'Cluster has been stored in "{res_dir}" directory')
    return num_unique_faces, res_dir


def get_best_pics(num_unique_faces: int, res_dir: str):
    """
    Find the best face from each unique face cluster.

    Parameters
    ----------
    num_unique_faces: int
        Total unique faces found after clustering.
    res_dir: str
        Path of result directory where clustered images are stored.

    Returns
    -------
    fine_images: list
        A list containing best images from each cluster.

    """
    fine_images = []
    for i in range(num_unique_faces):
        files = next(os.walk(f'{res_dir}/{i}'))[-1].copy()
        best_score = 0
        best_img = None
        for file in files:
            img = cv2.imread(f'{res_dir}/{i}/{file}')
            bluriness = cv2.Laplacian(img, cv2.CV_64F).var()
            if bluriness > best_score:
                best_score = bluriness
                best_img = img
        fine_images.append(best_img)
    return fine_images
