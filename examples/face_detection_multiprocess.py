import os
from typing import List

import face_recognition
import shutil

import time
import multiprocessing
import logging

logging.basicConfig(level=logging.INFO)
logger: logging.Logger = logging.getLogger(__name__)


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


def get_chunks() -> List:
    photos = os.listdir("photos")
    print(f"Found {len(photos)} photos to recognize\n-------")
    chunk_size = len(photos) // 15
    chunks = [photos[i:i + chunk_size] for i in range(0, len(photos), chunk_size)]
    # print(*chunks, sep='\n')
    # print(len(chunks))
    return chunks


def face_rec(dataset, chunk, proc_name):
    if os.path.exists("recognized_photos"):
        pass
        # print("Directory recognized_photos exists")
    else:
        os.mkdir("recognized_photos")
        print("Directory recognized_photos created")

    f = 0
    for (i, img) in enumerate(chunk):
        print(f"{proc_name}+ Processing {i + 1}/{len(chunk)} photo...")
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
    start: float = time.time()

    dataset = dataset_create()
    chunks = get_chunks()

    procs: List[multiprocessing.Process] = []
    for ch in chunks:
        proc = multiprocessing.Process(
            target=face_rec,
            args=(dataset, ch),
        )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    logger.info('Done in {:.4}'.format(time.time() - start))


if __name__ == "__main__":
    main()
