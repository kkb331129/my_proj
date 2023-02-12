import cv2
import numpy as np

# Load the cascade classifier for detecting cars
car_cascade = cv2.CascadeClassifier("cars.xml")

# Read the input image
img = cv2.imread("cars.jpg")

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect cars in the image
cars = car_cascade.detectMultiScale(gray_img, 1.1, 3)

# Draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the image with rectangles around the cars
cv2.imshow("Cars", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the number of cars detected
print("Number of cars detected: ", len(cars))