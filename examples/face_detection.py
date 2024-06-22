import os
import face_recognition
import shutil


def dataset_create():
    if os.path.exists("dataset_photo"):
        pass
        # print("Directory dataset_photo exists")
    else:
        os.mkdir("dataset_photo")
        print("Directory dataset_photo created")

    print("Creating dataset...")
    enc_dataset = []
    images = os.listdir("dataset_photo")

    f = 0
    for (i, img) in enumerate(images):
        print(f"+ Processing {i + 1}/{len(images)} photo...")
        face_img = face_recognition.load_image_file(f"dataset_photo/{img}")
        face_enc = face_recognition.face_encodings(face_img)

        if len(face_enc) > 0:
            for _ in face_enc:
                enc_dataset.append(_)
        else:
            print("++ No faces on photo!")
    if len(enc_dataset) > 0:
        print(f"Dataset crated with {len(enc_dataset)} faces")
    else:
        print("No faces found on photos")
        exit()
    return enc_dataset


def face_rec(dataset):
    if os.path.exists("recognized_photos"):
        pass
        # print("Directory recognized_photos exists")
    else:
        os.mkdir("recognized_photos")
        print("Directory recognized_photos created")

    photos = os.listdir("photos")
    print(f"Found {len(photos)} photos to recognize\n-------")
    f = 0
    for (i, img) in enumerate(photos):
        print(f"+ Processing {i + 1}/{len(photos)} photo...")
        face_img = face_recognition.load_image_file(f"photos/{img}")
        # face_enc = face_recognition.face_encodings(face_img)
        face_enc = face_recognition.face_encodings(face_img, num_jitters=100)  # more accuracy
        print(f"++ Found {len(face_enc)} face on photo")

        if len(face_enc) > 0:
            for (j, face) in enumerate(face_enc):
                print(f"+++ Processing {j + 1}/{len(face_enc)} faces...")
                for enc_data in dataset:
                    # compare = face_recognition.compare_faces([enc_data], face)
                    compare = face_recognition.compare_faces([enc_data], face, tolerance=0.6)  # more accuracy
                    if compare[0]:
                        print(f"++++ We have a match in {img}")
                        f += 1
                        print(f'++++ {f} photos recognized')
                        shutil.copy2(f"photos/{img}", "recognized_photos/")
                        break

        else:
            print("+++ No faces on photo!")


def main():
    dataset = dataset_create()
    face_rec(dataset=dataset)


if __name__ == "__main__":
    main()
