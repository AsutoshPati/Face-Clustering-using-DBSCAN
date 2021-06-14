import get_encodings as enc
import face_cluster as cluster

root_path = 'face_images/'

# Create a face database from existing faces
dir_name, data_len = enc.create_face_database(root_path)

# Apply face clustering to retrieve unique faces
unq_faces, res_dir = cluster.do_cluster(dir_name)
# print("Number of unique face detected: ", unq_faces)
# print(f'You can find clustered face in "{res_dir}" directory')
