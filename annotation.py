import pandas as pd
import os
import funcutils as util


def main(route_folder):
    # Create DataFrame
    df = pd.DataFrame(columns=['CategoryName', 'FileName', 'CategoryId', 'BBox', 'BBoxYolo'])
    # get categories from route folder
    categories_names = util.get_categories(route_folder)
    for idx, category in enumerate(categories_names):
        # Get images for this category
        category_image_list = util.get_category_images_list(route_folder, category)
        for image in category_image_list:

            # Add image information to dataFrame
            df = df.append({'CategoryName': category, 'FileName': image, 'CategoryId': idx}, ignore_index=True)
    print(df)


# if __name__ == "__main__":
