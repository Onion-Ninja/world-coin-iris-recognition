import iris
import cv2
import os
import matplotlib.pyplot as plt

path = "/home/nishkal/datasets/iris_db/CASIA_iris_thousand/orig/898_L/898_L_4.jpg"
path2 = "/home/nishkal/datasets/iris_db/CASIA_iris_thousand/orig/897_L/897_L_4.jpg"
print(iris.__version__)

img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

iris_pipeline = iris.IRISPipeline(
        env=iris.IRISPipeline.DEBUGGING_ENVIRONMENT
    )
iris_visualizer = iris.visualisation.IRISVisualizer()

output = iris_pipeline(iris.IRImage(img_data=img, image_id="image_id", eye_side="left"))
seg_arr1 = iris_pipeline.call_trace.get('segmentation', None)
canvas = iris_visualizer.plot_segmentation_map(
                    ir_image=iris.IRImage(img_data=img, eye_side='left'),
                    segmap=iris_pipeline.call_trace["segmentation"]
                )
plt.savefig(    
                "./seg1.jpg",
                bbox_inches="tight"
            )
output2 = iris_pipeline(iris.IRImage(img_data=img2, image_id="image_id", eye_side="left"))

print(output.keys())


seg_out1 = output['segmentation_map']
seg_arr2 = iris_pipeline.call_trace.get('segmentation', None)
seg_out2 = output2['segmentation_map']

canvas = iris_visualizer.plot_segmentation_map(
                    ir_image=iris.IRImage(img_data=img2, eye_side='left'),
                    segmap=iris_pipeline.call_trace["segmentation"]
                )
plt.savefig(    
                "./seg2.jpg",
                bbox_inches="tight"
            )

# print(type(seg_arr1))
# print(type(seg_out1))
# print(type(seg_arr1.predictions))
# print(type(seg_out1["predictions"]))

norm_arr1 = iris_pipeline.call_trace.get('normalization', None)
norm_out1 = output2['normalized_iris']

print(type(norm_arr1))
print(type(norm_out1))
print(norm_out1)
# print(type(norm_arr1.predictions))
# print(type(norm_out1["predictions"]))

# norm_arr1 = iris_pipeline.call_trace.get('normalization', None)

# print("===Segmentation Array===")
# print(seg_arr1)
# print("===Segmentation output===")
# print(seg_out1)
# it1 = output['iris_template']
# print(it1)

# print("===2===")

# iris_code = output["iris_template"].iris_codes[0]
# mask_code = output["iris_template"].iris_codes[1]
