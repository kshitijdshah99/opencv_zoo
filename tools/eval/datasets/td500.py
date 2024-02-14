import os
import cv2

class TD500:
    def __init__(self, root):
        self.root = root

    @property
    def name(self):
        return self.__class__.__name__

    def eval(self, model):
        # folder_path = "./opencv_zoo/tools/eval/MSRA-TD500/test"
        # jpg_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.JPG')]

        # Iterate over each .jpg file in the folder
        jpg_files = ["test.png"]
        for jpg_file in jpg_files:
            # Read image
            original_image = cv2.imread(jpg_file)
            original_w = original_image.shape[1]
            original_h = original_image.shape[0]
            scaleHeight = original_h / 736
            scaleWidth = original_w / 736
            image = cv2.resize(original_image, [736, 736])

            # Inference
            results = model.infer(image)

            # Scale the results bounding box
            for i in range(len(results[0])):
                for j in range(4):
                    box = results[0][i][j]
                    results[0][i][j][0] = box[0] * scaleWidth
                    results[0][i][j][1] = box[1] * scaleHeight

            # Print results
            print(f'Results for image: {jpg_file}')
            print(f'{len(results[0])} texts detected.')
            for idx, (bbox, score) in enumerate(zip(results[0], results[1])):
                print(f'{idx}: {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}, {score}')
