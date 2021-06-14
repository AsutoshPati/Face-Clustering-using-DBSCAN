"""
    Author              : Asutosh Pati
    Date of Creation    : 12 MAY 2021
    Purpose             : Add on Module to create face database and finds face encodings.
    Version             : V 1.0.0
"""

import numpy as np
import gc
import cv2
import face_recognition
import os
import uuid
import pickle


def find_encodings(img: np.ndarray):
    """
    Get the 128-D encodings of a face image. if encodings can't be generated then None
    will be returned.

    Parameters
    ----------
    img: numpy.ndarray
        The face image whose encodings needs to be found out.

    Returns
    -------
    encodings: list / None
        128-D face embedding from the image. Else None if could not find encodings.

    """
    encodings = None
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)[0]
    except Exception as err:
        print(err, str(type(err)))
    gc.collect()
    return encodings


def is_print_required(msg: str, req: bool = True):
    """
    Print the given message if print is required.

    Parameters
    ----------
    msg: str
        Message needs to print.
    req: bool
        if True message will be printed. Default True.

    Returns
    -------
        None.

    """
    if req:
        print(msg)


def create_face_database(directory: str, rm_blur_img: bool = False, threshold: float = 100.0,
                         display: bool = True):
    """
    Creates a face database with face images and their encodings as a pickle file. All the faces will
    be taken from a single directory given in the parameter. Using the parameter blurred images can be
    removed with user defined threshold. The progress can also be hidden using parameter.

    Parameters
    ----------
    directory: str
        Path of directory containing the input images.
    rm_blur_img: bool
        Pass True to remove the blurred image. Default False.
    threshold: float
        The threshold score of blurred image below which image will be removed.
    display: bool
        if True the progress message will be displayed. Default True.

    Returns
    -------
    db_dir: str
        Directory name of newly created face database directory.
    num_of_data: int
        Number of data stored in directory after processing.

    """
    encodings = []
    
    if not os.path.exists(directory):
        raise FileNotFoundError("Directory not found")

    files = next(os.walk(directory))[-1].copy()

    num_files = len(files)
    if num_files > 0:
        db_dir = str(uuid.uuid4())
        db_dir = db_dir.replace("-", "_")
        os.makedirs(db_dir)
    else:
        raise FileNotFoundError("No files found in directory")

    ctr = 1
    for i in range(num_files):
        img = cv2.imread(directory + files[i])
        blurriness = cv2.Laplacian(img, cv2.CV_64F).var()

        if rm_blur_img:
            if blurriness >= threshold:
                pass
            else:
                is_print_required(f'{i + 1} / {num_files} -- skipped due to blurriness {blurriness}',
                                  display)
                continue

        # crop face image if required
        # code can be added here
        
        face_encoding = find_encodings(img)
        if face_encoding is not None:
            cv2.imwrite(f'{db_dir}/img_{ctr}.jpg', img)
            encodings.append({'img_path': f'{db_dir}/img_{ctr}.jpg',
                              'encoding': face_encoding})
            is_print_required(f'{i+1} / {num_files} -- used for database', display)
            ctr += 1
        else:
            is_print_required(f'{i+1} / {num_files} -- skipped due to face not found', display)

    num_of_data = ctr - 1
    f = open(f'{db_dir}/face_encodings.p', 'wb')
    pickle.dump(encodings, f)
    f.close()
    gc.collect()
    return db_dir, num_of_data
