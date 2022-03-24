#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Values selected to filter skin colour from HSV, use trackbar to adjust these values.
int imaxH = 179, imaxS = 255, imaxV = 255;
int ilowH = 0, ilowS = 0, ilowV = 165;
int ihighH = imaxH, ihighS = imaxS, ihighV = imaxV;

const String windowDetection = "Binary Image with Trackbar";
const String windowDisplay = "Fingerprint Extraction";

Mat palmRemoval(Mat img)
{
    int rows = img.rows;
    int maxContours = 0;
    int minLineIndex = img.rows;
    Mat outImg = img.clone();

    vector <vector <Point> > contours;
    vector<Vec4i> hierarchy;

    for(int i = rows; i > 0; i--) {
        // Draw black line starting from bottom towards top.
        line(img, Point(0, i), Point(img.cols, i), Scalar(0, 0, 0));

        // Find contours in each iteration.
        findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));

        int noContours = 0;
        for (int c = 0; c < contours.size(); c++) {
            // Only consider bigger enough contours.
            if (contours[c].size() > 40) {
                noContours += 1;

                // Update maxContours if the number of contours found great than before.
                if (noContours > maxContours) {
                    maxContours = noContours;
                    minLineIndex = i;
                }
            }
        }
    }

    // Reduce one more row for tolerance
    int finalMinLineIndex = minLineIndex-1;
    // Draw black line until finalMinLineIndex of oupt image.
    for(int j = rows; j >= finalMinLineIndex; j--) {
        line(outImg, Point(0, j), Point(img.cols, j), Scalar(0, 0, 0));
    }

    return outImg;
}

void drawFingerprintROI(Mat& img, vector <vector <Point> > contours)
{
    float scale = 1.6;

    for (int i = 0; i < contours.size(); i++) {
        if (contours[i].size() > 80) {

            RotatedRect oldRect = minAreaRect(Mat(contours[i]));

            RotatedRect tmpRect;
            Point2f tmpRectVertex[4];
            oldRect.points(tmpRectVertex);


            Point2f topPoint;
            if (oldRect.size.height > oldRect.size.width) {
                topPoint = (tmpRectVertex[1] + tmpRectVertex[2]) / 2;
                tmpRect.center.x = oldRect.center.x + floor((topPoint.x - oldRect.center.x) * 2 / 3);
                tmpRect.center.y = oldRect.center.y + floor((topPoint.y - oldRect.center.y) * 2 / 3);
                tmpRect.size.width = oldRect.size.width;
                tmpRect.size.height = (float)scale * (oldRect.size.width);
                tmpRect.angle = oldRect.angle;
            }
            else {
                topPoint = (tmpRectVertex[2] + tmpRectVertex[3]) / 2;
                tmpRect.center.x = oldRect.center.x + floor((topPoint.x - oldRect.center.x) * 2 / 3);
                tmpRect.center.y = oldRect.center.y + floor((topPoint.y - oldRect.center.y) * 2 / 3);
                tmpRect.size.width = (float)scale * (oldRect.size.height);
                tmpRect.size.height = oldRect.size.height;
                tmpRect.angle = oldRect.angle;
            }

            tmpRect.points(tmpRectVertex);
            for (int j = 0; j < 4; j++)
                line(img, tmpRectVertex[j], tmpRectVertex[(j + 1) % 4], Scalar(0, 0, 255), 2, LINE_AA);
        }
    }
}

int main(int argc, char* argv[])
{
    // Capture the vedio from default camara: webcam.
    VideoCapture cap(0);
    // Check if webcam can be opened to capture vedio, else fail.
    if(!cap.isOpened()) {
       return -1;
    }

    namedWindow(windowDetection);
    namedWindow(windowDisplay);

    // Create trackbars to adjust thresholds for HSV values.
    createTrackbar("Low H", windowDetection, &ilowH, imaxH);
    createTrackbar("High H", windowDetection, &ihighH, imaxH);
    createTrackbar("Low S", windowDetection, &ilowS, imaxS);
    createTrackbar("High S", windowDetection, &ihighS, imaxS);
    createTrackbar("Low V", windowDetection, &ilowV, imaxV);
    createTrackbar("High V", windowDetection, &ihighV, imaxV);


    while (true) {

        // Get frame in each itration.
        Mat frameOrg;
        bool captured = cap.read(frameOrg);
        if(!captured)
        {
            break;
        }
//        imshow("BGR Image", frameOrg);

//        frameOrg = imread("/home/sooraj/Desktop/imageprocessing/hand.jpg", CV_LOAD_IMAGE_COLOR);  //BGR image

        // Convert from BGR to HSV color space.
        Mat frameHSV;
        cvtColor(frameOrg, frameHSV, COLOR_BGR2HSV);
//        imshow("HSV Image", frameHSV);

        // Perform thresholding for HSV frame, to obtain binary image.
        Mat frameThresh;
        inRange(frameHSV, Scalar(ilowH, ilowS, ilowV), Scalar(ihighH, ihighS, ihighV), frameThresh);

        // Show the binary frame with trackbar.
        imshow(windowDetection, frameThresh);

        // Smooth the frame using the median filter.
        medianBlur(frameThresh, frameThresh, 7);

        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        // Morphological opening (removes small objects from the foreground)
        erode(frameThresh, frameThresh, kernel);
        dilate( frameThresh, frameThresh, kernel);

        // Morphological closing (removes small holes from the foreground)
        dilate( frameThresh, frameThresh, kernel);
        erode(frameThresh, frameThresh, kernel);

        imshow("Filtered Image", frameThresh);

        line(frameThresh, { 0, frameThresh.rows }, { frameThresh.cols, frameThresh.rows }, Scalar(0, 30, 0), 300, LINE_8);

        // Removing palm portion from the frame.
        frameThresh = palmRemoval(frameThresh);

        // Determine edges in the frame.
        Mat frameDetect;
        int thresh = 100;
        Canny(frameThresh, frameDetect, thresh, thresh * 2, 3);

        // Find the contours of the frame.
        vector <vector <Point> > contours;
        vector<Vec4i> hierarchy;
        findContours(frameDetect, contours, hierarchy, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
        for( size_t i = 0; i< contours.size(); i++ )
        {
            drawContours(frameOrg, contours, (int)i, Scalar(0, 255, 255), 1, LINE_8, hierarchy, 0 );
        }
        imshow("Finger Detection", frameOrg);

        // Draw rectangle around the fingerprint area.
        drawFingerprintROI(frameOrg, contours);

        // Show the frame after processing.
        imshow(windowDisplay, frameOrg);

        // Break the loop when 'esc' or 'q' is pressed.
        char key = (char) waitKey(30);
        if (key == 'q' || key == 27)
        {
            break;
        }
    }

    return 0;
}
