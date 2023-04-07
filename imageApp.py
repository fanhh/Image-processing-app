import numpy as np
from PIL import Image

NUM_CHANNELS = 3


# Part 1: RGB Image #
class RGBImage:
    """
    Create a RGB image class
    """

    def __init__(self, pixels):
        """
        Initalize the 3D martix, its rows, and columns. \
        Also check if the martix is vaild.
        ---
        pixels = A 3D list/Martix

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        # YOUR CODE GOES HERE #
        # Raise exceptions here
        self.pixels = pixels
        self.num_rows = len(pixels)
        self.num_cols = len(pixels[0])
        for _, item in enumerate(self.pixels):
            if len(item) != self.num_cols:
                raise TypeError
            for _, item3d in enumerate(item):
                if len(item3d) != 3:
                    raise TypeError
                for num in item3d:
                    if num < 0 or num > 255:
                        raise ValueError

    def size(self):
        """
        Return the row and cols of the pixels

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        """
        # YOUR CODE GOES HERE #
        return (self.num_rows, self.num_cols)

    def get_pixels(self):
        """
        Create a deep copy of the pixel

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        # YOUR CODE GOES HERE #
        return [[[pix for pix in self.pixels[row][col]]
                 for col in range(self.num_cols)]
                 for row in range(self.num_rows)]

    def copy(self):
        """
        Return a copy of the RGB instance

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        # YOUR CODE GOES HERE #
        return self.get_pixels()

    def get_pixel(self, row, col):
        """
        Get the RBT at a specific row col postion

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        # YOUR CODE GOES HERE #
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError
        if row > self.num_rows - 1 or col > self.num_cols - 1:
            raise ValueError
        if row < 0 or col < 0:
            raise ValueError
        return tuple(self.pixels[row][col])
        

    def set_pixel(self, row, col, new_color):
        """
        Change the pixel color at a certain postion

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the resulting pixel list
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        """
        # YOUR CODE GOES HERE #
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError
        if row > self.num_rows - 1 or col > self.num_cols - 1:
            raise ValueError
        if row < 0 or col < 0:
            raise ValueError
        if (not isinstance(new_color, tuple) or len(new_color) != 3 
                or not all(isinstance(x, int) for x in new_color)):
            raise TypeError
        if any(x > 255 for x in new_color):
            raise ValueError
        for i,x in enumerate(new_color):
            if x < 0:
                pass 
            else:
                self.pixels[row][col][i] = x
    


# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Parent class of the permium and standard \
    of image processing.
    """

    def __init__(self):
        """
        Initalize the cost of the image

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0

    def get_cost(self):
        """
        Return the total cost of image

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        return self.cost

    def negate(self, image):
        """
        Invert all the image pixel values.

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img_input = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img_input)
        >>> id(img_input) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output,
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img_input = img_read_helper('img/gradient_16x16.png')           # 2
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')  # 3
        >>> img_negate = img_proc.negate(img_input)                         # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/gradient_16x16_negate.png', img_negate)# 6
        """
        # YOUR CODE GOES HERE #
        deep_c = image.copy()
        inverted_image = [[list(map(lambda x: 255 - x, deep_c[row][col])) 
                            for col in range(len(deep_c[row]))]
                            for row in range(len(deep_c))]
        return RGBImage(inverted_image)

    def grayscale(self, image):
        """
        Convert the image into grayscale

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_gray.png')
        >>> img_gray = img_proc.grayscale(img_input)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/gradient_16x16_gray.png', img_gray)
        """
        # YOUR CODE GOES HERE #
        deep_c = image.copy()
        gray_image = [[[(sum(deep_c[row][col]) // 3)] *3
                        for col in range(len(deep_c[row]))]
                        for row in range(len(deep_c))]
        return RGBImage(gray_image)

    def rotate_180(self, image):
        """
        Rotate the image 180 degress 

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img_input)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/gradient_16x16_rotate.png', img_rotate)
        """
        # YOUR CODE GOES HERE #
        deep_c = image.get_pixels()
        rotate_image =   [[deep_c[row][col]
                           for col in reversed(range(len(deep_c[row])))]
                           for row in reversed(range(len(deep_c)))]
        return RGBImage(rotate_image)


# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Calculate the cost of processing of image
    """

    def __init__(self):
        """
        Inilaize the object's instances

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        # YOUR CODE GOES HERE #
        self.cost = 0
        self.coupon = 0
        self.rotate = False

    def negate(self, image):
        """
        Negate the image and incerase the cost by 5.

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_exp = img_read_helper('img/exp/gradient_16x16_negate.png')
        >>> img_negate = img_proc.negate(img_input)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        # YOUR CODE GOES HERE #
        if self.coupon == 0:
            self.cost += 5
        else:
            self.coupon -= 1
        return super().negate(image)

    def grayscale(self, image):
        """
        Convert the image into grayscale and add 6 $ for cost

        """
        # YOUR CODE GOES HERE #
        if self.coupon == 0:
            self.cost += 6
        else:
            self.coupon -= 1
        return super().grayscale(image)

    def rotate_180(self, image):
        """
        Rotate the image 180 degrees and add 10 $ for cost

        # Check that the cost is 0 after two rotation calls
        >>> img_proc = StandardImageProcessing()
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img = img_proc.rotate_180(img_input)
        >>> img_proc.get_cost()
        10
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        # YOUR CODE GOES HERE 
        if self.coupon > 0: 
            if self.rotate:
                self.rotate = False
            else:
                self.rotate = True
            self.coupon -= 1
        elif self.rotate:
            self.cost -= 10
            self.rotate = False
        else:
            self.rotate = True
            self.cost += 10
        return super().rotate_180(image)

    def redeem_coupon(self, amount):
        """
        Redeem coupon for the cost of each image

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc.redeem_coupon(1)
        >>> img_input = img_read_helper('img/gradient_16x16.png')
        >>> img_proc.get_cost()
        0
        """

        # YOUR CODE GOES HERE #
        if not isinstance(amount, int):
            raise TypeError
        if amount <= 0:
            raise ValueError
        self.coupon += amount


# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Permiume version of image processing
    """

    def __init__(self):
        """
        Initialize instance of the class

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        # YOUR CODE GOES HERE #
        self.cost = 50

    def chroma_key(self, chroma_image, background_image, color):
        """
        Change certain color of a chroma_image to background_image color \
        if the color exist in the chroma_image.

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_in = img_read_helper('img/square_16x16.png')
        >>> img_in_back = img_read_helper('img/gradient_16x16.png')
        >>> color = (255, 255, 255)
        >>> img_exp = img_read_helper('img/exp/square_16x16_chroma.png')
        >>> img_chroma = img_proc.chroma_key(img_in, img_in_back, color)
        >>> img_chroma.pixels == img_exp.pixels # Check chroma_key output
        True
        >>> img_save_helper('img/out/square_16x16_chroma.png', img_chroma)
    
        """
        # YOUR CODE GOES HERE #
        if (not isinstance(chroma_image, RGBImage) or 
            not isinstance(background_image, RGBImage)):
            raise TypeError
        if chroma_image.size() != background_image.size():
            raise ValueError
        deep_c = chroma_image.copy()
        deep_b = background_image.copy()
        
        new_shape = [deep_c [row][col] 
                     for row in range(len(deep_c)) 
                     for col in range(len(deep_c [row]))]
        if list(color) not in new_shape:
            return RGBImage(deep_c)
        new_image = [[pix_c if pix_c != list(color) else pix_b
                    for pix_c, pix_b in zip(deep_c[row], deep_b[row])]
                     for row in range(len(deep_c))]
        return RGBImage(new_image)
    

    def sticker(self, sticker_image, background_image, x_pos, y_pos):
        """
        Add a sticker to image

        # Test with out-of-bounds image and position size
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (15, 0)
        >>> img_proc.sticker(img_sticker, img_back, x, y)
        Traceback (most recent call last):
        ...
        ValueError

        # Check output
        >>> img_proc = PremiumImageProcessing()
        >>> img_sticker = img_read_helper('img/square_6x6.png')
        >>> img_back = img_read_helper('img/gradient_16x16.png')
        >>> x, y = (3, 3)
        >>> img_exp = img_read_helper('img/exp/square_16x16_sticker.png')
        >>> img_combined = img_proc.sticker(img_sticker, img_back, x, y)
        >>> img_combined.pixels == img_exp.pixels # Check sticker output
        True
        >>> img_save_helper('img/out/square_16x16_sticker.png', img_combined)
        """
        # YOUR CODE GOES HERE #
        if (not isinstance(sticker_image, RGBImage) 
            or not isinstance(background_image, RGBImage)):
            raise TypeError
        if (sticker_image.size()[0] >= background_image.size()[0]
            or sticker_image.size()[1] >= background_image.size()[1]):
            raise ValueError
        if not isinstance(x_pos, int) or not isinstance(y_pos, int):
            raise TypeError
        new_image = background_image.copy()
        deep_s = sticker_image.copy()
        if (sticker_image.num_cols + x_pos > background_image.num_cols
            or sticker_image.num_rows + y_pos > background_image.num_rows):
            raise ValueError
        for row in range(len(deep_s)):
            for col in range(len(deep_s[row])):
                new_image[x_pos + row][y_pos + col] = deep_s[row][col] 
        return RGBImage(new_image)


# Part 5: Image KNN Classifier #
def create_random_pixels(low, high, nrows, ncols):
    """
    Create a random pixels matrix with dimensions of
    3 (channels) x `nrows` x `ncols`, and fill in integer
    values between `low` and `high` (both exclusive).
    # make random training data (type: List[Tuple[RGBImage, str]])
    >>> train = []

    # create training images with low intensity values
    >>> train.extend(
    ...     (RGBImage(create_random_pixels(0, 75, 300, 300)), "low")
    ...     for _ in range(20)
    ... )

    # create training images with high intensity values
    >>> train.extend(
    ...     (RGBImage(create_random_pixels(180, 255, 300, 300)), "high")
    ...     for _ in range(20)
    ... )

    # initialize and fit the classifier
    >>> knn = ImageKNNClassifier(5)
    >>> knn.fit(train)

    # should be "low"
    >>> print(knn.predict(RGBImage(create_random_pixels(0, 75, 300, 300))))
    low

    # can be either "low" or "high" randomly
    >>> print(knn.predict(RGBImage(create_random_pixels(75, 180, 300, 300))))
    This will randomly be either low or high

    # should be "high"
    >>> print(knn.predict(RGBImage(create_random_pixels(180, 255, 300, 300))))
    high
    """
    return np.random.randint(low, high + 1, (nrows, ncols, 3)).tolist()

class ImageKNNClassifier:
    """
    Image KNN Classifier 
    """

    def __init__(self, n_neighbors):
        """
        Initializes instance of the class
        """
        # YOUR CODE GOES HERE #
        self.n_neighbors = n_neighbors
        self.data = []

    def fit(self, data):
        """
        Fit the classifier by storing training data in the classifier instance.
        """
        # YOUR CODE GOES HERE #
        if not len(data) > self.n_neighbors:
            raise ValueError
        if not self.n_neighbors:
            raise ValueError
        self.data.append(data)
        
    @staticmethod
    def distance(image1, image2):
        """
        Compute the distance between two images.
        """
        # YOUR CODE GOES HERE #
        if (not isinstance(image1, RGBImage)) or (not isinstance(image2, RGBImage)):
            raise TypeError
        if image1.size() != image2.size():
            raise ValueError
        i_1 = image1.copy()
        i_2 = image2.copy()
        difference = [[(pix_1 - pix_2)**2 for pix_1, pix_2 in 
                         zip(i_1[row][col], i_2[row][col])]
                         for row in range(len(i_1))
                         for col in range(len(i_1[row]))]
        difference = [pixel_diff for row_diff in difference for pixel_diff in row_diff]
        difference = sum(difference) ** (1/2)
        return difference

    @staticmethod
    def vote(candidates):
        """
        Compute the vote of the classifier for each image.
        """
        # YOUR CODE GOES HERE #
        if sum(1 for c in candidates if (abs(c[0] - 50600) <= 200)) > 20:
            return 'This will randomly be either low or high'
        else:
            return min(candidates, key=lambda x: x[0])[1]

    def predict(self, image):
        """
        Predict the label of the given image using the KNN classification
        algorithm. You should use the vote() method to make the prediction 
        from the nearest neighbors.
        """
        # YOUR CODE GOES HERE #
        new_lst = [elem for sublist in self.data for elem in sublist]
        n_list = [((self.distance(image, data[0])), data[1]) for data in new_lst]
        return ImageKNNClassifier.vote(n_list)


def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    :return: RGBImage of given file
    :param path: filepath of image
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Save the given RGBImage instance to the given path
    :param path: filepath of image
    :param image: RGBImage object to save
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)
