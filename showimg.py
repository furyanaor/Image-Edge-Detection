import matplotlib.pyplot as plt

print("showimg function is loaded!")

def showimg(org_img, new_img):
    plt.subplot(1, 2, 1)
    plt.title('Before')
    if len(org_img.shape) == 3:
        plt.imshow(org_img[:, :, ::-1])
    elif len(org_img.shape) == 2:
        plt.imshow(org_img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('After')
    if len(org_img.shape) == 3:
        plt.imshow(new_img[:, :, ::-1])
    elif len(org_img.shape) == 2:
        plt.imshow(new_img, cmap='gray')
    plt.axis('off')

    plt.show()
