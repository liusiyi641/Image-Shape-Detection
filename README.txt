

This is a simple shape detection project using border following algorithm. 

main.py is our main function that does all the computations and detections. (It was first completed in .ipynb and I put them into a .py file so it might seem messy)

Result contains some of the selected results.

shape_train2018 contains all the testing images.

This project still has large room for improvement. We might still try to make it better in the future. 

We first implemented denoising and k-means algorithm with OpenCV to extract the features, then use our own border following algorithm to find the border of the figure, and at last use our shape detection algorithm to determine their shape.

The border following algorithm basically just traverses through the eight neighbors of the first pixel we found that has a certain desired RGB values, and then select one of the neighbors with the same RGB value in an order (basically clockwise). Go to that neighbor and keep traversing until we found the whole border of a certain figure. 

The shape detection algorithm builds upon the idea the the first pixel we find will be the highest vertex of the figure since we are traversing it from top to bottom, left to right. So we take the first first pixel of the border we found and draw two straight lines. One from the pixel to the right and one from the pixel to 60 degrees right bottom. The two lines are like the edges of squares and triangles. So we take the border and compare it to these to lines to find their distance error at each pixel. We take the errors and compute their ratio and determine which shape it is, a triangle, a square, or a circle. 
