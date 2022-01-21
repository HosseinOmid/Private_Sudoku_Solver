# Programm zum Öffnen und Lösen eines Sudoku-Raetsels
# Autor: Hossein Omid Beiki
# Stand:  12.07.2021

import numpy as np
import cv2 as cv
import sudoku_solver as ss

# read and show the image
image = cv.imread('test.PNG')
# cv.imshow('img', image)
# set the width and height of the image. it should be dividable by 9
height_img = 333
width_img = 333


def filter_img_and_find_table(img):
    # filter the image and find the table
    img = cv.resize(img, (width_img, height_img))
    img = cv.bilateralFilter(img, 3, 5, 5)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # filter the image to a binary (black and white) image
    binary_img = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 7, 2)
    binary_img = cv.bilateralFilter(binary_img, 5, 15, 15)
    binary_img = cv.medianBlur(binary_img, 3)
    cv.imshow('binary_img', binary_img)
    # canny edge detector
    # can = cv.Canny(binary_img, 80, 150)
    # cv.imshow('can', can)
    # find contours
    show_all_contour = img.copy()
    contours, h = cv.findContours(binary_img, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_KCOS)
    cv.drawContours(show_all_contour, contours, -1, (0, 0, 255), 3)
    # cv.imshow('show_all_contour', show_all_contour)
    # find the biggest contour which looks like a 4-edge polygon
    biggest_approx = np.array([])
    biggest_contour = np.array([])
    max_area = 0
    for iContour in contours:
        area = cv.contourArea(iContour)
        perimeter = cv.arcLength(iContour, True)
        epsilon = .02 * perimeter
        # check if the contour could be approximated b a 4-edge polygon
        approx = cv.approxPolyDP(iContour, epsilon, True)
        if area > max_area and len(approx) == 4:
            max_area = area
            biggest_approx = approx
            biggest_contour = iContour

    show_approx = img.copy()
    cv.drawContours(show_approx, biggest_approx, -1, (0, 0, 255), 3)
    cv.imshow('show_approx', show_approx)
    # sort the found four points in this order:
    # |--------------------->x
    # |    p1
    # |                   p2
    # |
    # |                     p4
    # |   p3
    # y
    four_points = biggest_approx.reshape((4, 2))
    xsort_index = np.argsort(four_points[:, 0])
    upper_points_y_sort = np.argsort(four_points[xsort_index[:2], 1])
    first_point_index = xsort_index[upper_points_y_sort[0]]
    third_point_index = xsort_index[upper_points_y_sort[1]]
    lower_points_y_sort = np.argsort(four_points[xsort_index[-2:], 1])
    second_point_index = xsort_index[lower_points_y_sort[0] + 2]
    fourth_point_index = xsort_index[lower_points_y_sort[1] + 2]
    # create the points
    pnt1 = np.float32(four_points[[first_point_index, second_point_index, third_point_index, fourth_point_index], :])
    pnt2 = np.float32([[0, 0], [width_img, 0], [0, height_img], [width_img, height_img]])
    # getPerspectiveTransform
    M = cv.getPerspectiveTransform(pnt1, pnt2)
    thres_warped = cv.warpPerspective(binary_img, M, (width_img, height_img))
    gray_warped = cv.warpPerspective(gray, M, (width_img, height_img))
    # prepare for the model
    gray_warped_inv = 255 - gray_warped
    # sudoku_table = cv.erode(thres_warped, np.ones((1, 1), np.uint8), iterations=1)
    sudoku_table = cv.GaussianBlur(thres_warped, (3, 3), cv.BORDER_CONSTANT)
    # cv.imshow('thres_warped', thres_warped)
    # cv.imshow('gray_warped_inv', gray_warped_inv)
    cv.imshow('sudoku_table', sudoku_table)
    return sudoku_table  # , biggest_approx, binary_img, gray, gray_warped, thres_warped


table = filter_img_and_find_table(image)


def recognize_sudoku_digits(sudoku_table):
    from keras.models import load_model
    # spit the image in boxes which contain only 1 number --> split 9x9
    box = np.zeros((9, 9, sudoku_table.shape[0] // 9, sudoku_table.shape[0] // 9), sudoku_table.dtype)
    rows = np.split(sudoku_table, 9, axis=0)
    for r, iRow in enumerate(rows):
        box[r, :, :, :] = np.split(iRow, 9, axis=1)
    # load the pre-trained model
    model = load_model("test_model3.h5")
    # init the recognized numbers
    sudoku_nums = np.zeros((9, 9))
    for i in range(9):
        for j in range(9):
            ibox = box[i, j, :, :]
            # resize, since the model only accepts 28x28 images
            res = cv.resize(ibox, (28, 28))
            # make the boarder black. this would remove any white line or point around the image
            res[:4, :] = 0
            res[:, :4] = 0
            res[-4:, :] = 0
            res[:, -4:] = 0
            # cv.imshow('iBox', res)
            gray = res.reshape((1, 28 * 28)).astype('float32')
            norm_gray = gray / 255
            prediction = model.predict(norm_gray)
            if np.max(prediction) > .9:
                sudoku_nums[i, j] = prediction.argmax()
    print(sudoku_nums)
    return sudoku_nums


nums = recognize_sudoku_digits(table)

# solve the sudoku
ss.print_board(nums)
print('')
print('')
ss.solved_board = nums
solved_board = ss.solve_sudoku(nums)
ss.print_board(solved_board)

cv.waitKey(0)