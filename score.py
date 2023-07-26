import cv2
import numpy as np
import matplotlib.pyplot as plt

def dice_score(y_true, y_pred, smooth=1):
    intersection = np.sum(y_true * y_pred, axis=(1, 2))
    union = np.sum(y_true, axis=(1, 2)) + np.sum(y_pred, axis=(1, 2))
    dice = np.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def run(images):
    # Load the two images
    image1 = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(images[1], cv2.IMREAD_GRAYSCALE)
    
    # Resize two images
    image1 = cv2.resize(image1, (128, 128))
    image2 = cv2.resize(image2, (128, 128))

    # Expand dimensions to make images 4-dimensional
    image1 = np.expand_dims(image1, axis=0)
    image2 = np.expand_dims(image2, axis=0)

    # Calculate the Dice score
    score = dice_score(image1, image2)
    print("Dice score:", score)
    
    # Plot the images and Dice score
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(image1[0], cmap='gray')
    ax[0].set_title('Image 1')
    ax[0].axis('off')
    ax[1].imshow(image2[0], cmap='gray')
    ax[1].set_title('Image 2')
    ax[1].axis('off')
    
    # Set the title and display the Dice score
    plt.suptitle("Dice Score : {:.3f}".format(score))
    plt.show()
    plt.savefig("dice.png")

if __name__ == "__main__":
    run(["ISIC_2016/ISBI2016_ISIC_Part1_Test_GroundTruth/ISIC_0000490_Segmentation.png", \
        "ISIC_2016/0000490_output_ens.jpg"])
