import argparse
import pandas as pd
import os
import funcutils as util
from segmentation import get_green_bounding_box as getbb


def main(route_folder, im_size=(224, 224)):
    # Create DataFrame
    df = pd.DataFrame(columns=['CategoryName', 'FileName', 'CategoryId', 'BBox', 'BBoxYolo'])
    # get categories from route folder
    categories_names = util.get_categories(route_folder)
    for idx, category in enumerate(categories_names):
        # Get images for this category
        category_image_list = util.get_category_images_list(route_folder, category)
        for image in category_image_list:
            bbox = getbb(os.path.join(route_folder, category, image))
            bbox_yolo = util.yolo_format_box(im_size, bbox)
            # Add image information to dataFrame
            df = df.append({'CategoryName': category, 'FileName': image, 'CategoryId': idx, 'BBox': bbox,
                            'BBoxYolo': bbox_yolo}, ignore_index=True)
    df.to_csv('C:/ObjectRecognitionPython/df.csv', encoding='utf-8')


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('PATH')
    args = parse.parse_args()
    main(args.PATH)
