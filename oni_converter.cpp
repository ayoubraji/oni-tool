//------------------------------------------------
// STL Header
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

// OpenNI Header
#include <XnCppWrapper.h>
#include <OpenNI.h>

// OpenCV
//#include "cv.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <highgui.h>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <pcl/console/time.h>
#include <pcl/point_types.h>
using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;

//----- for PCL ----//
typedef pcl::PointXYZRGBA typePoint;
const float bad_point = std::numeric_limits<float>::quiet_NaN();
//Set sensor limits, such as resolution
int limitx_min=0;
int limitx_max=640;
int limity_min=0;
int limity_max=480;
int limitz_min=1;
int limitz_max=3000;
openni::VideoStream depth, color;
/**
  * Define the union structure for the RGB information in the PCD file
 */
union PCD_BGRA
{
    struct
    {
        uchar B; // LSB
        uchar G; // ---
        uchar R; // MSB
        uchar A; //
    };
    float RGB_float;
    uint  RGB_uint;
};
//-----------------//

/**
 * @brief Function similar to printf returning C++ style string
 * @param message
 * @return
 */

inline std::string printfstring(const char *message, ...)
{
    static char buf[8*1024];

    va_list va;
    va_start(va, message);
    vsprintf(buf, message, va);
    va_end(va);

    std::string str(buf);
    return str;
}

void MatToPointXYZRGB(cv::Mat &color, cv::Mat &depth,
                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {

    /*CALIBRATION PARAMETERS
     * //kinect
     const double u0 = 3.3930780975300314e+02;
     const double v0 = 2.4273913761751615e+02;
     const double fx = 5.9421434211923247e+02;
     const double fy = 5.9104053696870778e+02;
     */
    //xtion
     const double u0 = 324.822;
     const double v0 = 221.082;
     const double fx = 582.347;
     const double fy = 579.938;
     int rows = color.rows;
     int cols = color.cols;
     cloud->height = (uint32_t) rows;
     cloud->width = (uint32_t) cols;
     cloud->is_dense = false;
     cloud->points.resize(cloud->width * cloud->height);
     for (unsigned int u = 0; u < rows; ++u) {
       for (unsigned int v = 0; v < cols; ++v) {
         float Xw = 0, Yw = 0, Zw = 0;

         Zw = depth.at<ushort>(u, v);
         Xw = (float) ((v - v0) * Zw / fx);
         Yw = (float) ((u - u0) * Zw / fy);

         cloud->at(v, u).b = color.at<cv::Vec3b>(u, v)[0];
         cloud->at(v, u).g = color.at<cv::Vec3b>(u, v)[1];
         cloud->at(v, u).r = color.at<cv::Vec3b>(u, v)[2];
         cloud->at(v, u).x = Xw;
         cloud->at(v, u).y = Yw;
         cloud->at(v, u).z = Zw;
       }
     }
   }

//Convert pointcloudXYZRGB to PLY
void saveCloud (const std::string filename, const pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, bool binary, bool use_camera)
{
  TicToc tt;
  tt.tic ();

  print_highlight ("Saving "); print_value ("%s ", filename.c_str ());

  pcl::PLYWriter writer;
  writer.write (filename, *cloud, binary, use_camera);
}

//Get pointcloud from cv::Mat image and depth
void get_pcl(cv::Mat& color_mat, cv::Mat& depth_mat, pcl::PointCloud<typePoint>& cloud ){
    float x,y,z;

    for (int j = 0; j< depth_mat.rows; j ++){
        for(int i = 0; i < depth_mat.cols; i++){
            // the RGB data is created
            PCD_BGRA   pcd_BGRA;
                       pcd_BGRA.B  = color_mat.at<cv::Vec3b>(j,i)[0];
                       pcd_BGRA.R  = color_mat.at<cv::Vec3b>(j,i)[2];
                       pcd_BGRA.G  = color_mat.at<cv::Vec3b>(j,i)[1];
                       pcd_BGRA.A  = 0;

            typePoint vertex;
            int depth_value = (int) depth_mat.at<unsigned short>(j,i);
            // find the world coordinates
            openni::CoordinateConverter::convertDepthToWorld(depth, i, j, (openni::DepthPixel) depth_mat.at<unsigned short>(j,i), &x, &y,&z );

            // the point is created with depth and color data
            if ( limitx_min <= i && limitx_max >=i && limity_min <= j && limity_max >= j && depth_value != 0 && depth_value <= limitz_max && depth_value >= limitz_min){
                vertex.x   = (float) x;
                vertex.y   = (float) y;
                vertex.z   = (float) depth_value;
            } else {
                // if the data is outside the boundaries
                vertex.x   = bad_point;
                vertex.y   = bad_point;
                vertex.z   = bad_point;
            }
            vertex.rgb = pcd_BGRA.RGB_float;

            // the point is pushed back in the cloud
            cloud.points.push_back( vertex );
        }
    }
}

/// convert depth map to OpenCV, xn::DepthMetaData to cv::Mat
void xdepth2opencv(xn::DepthMetaData &xDepthMap, cv::Mat &im, int verbose=0)
{
    int h=xDepthMap.YRes();
    int w=xDepthMap.XRes();

    if (verbose)
        printf("xdepth2opencv: w %d, h %d\n", w, h);

    const cv::Mat tmp(h, w, CV_16U, ( void *)xDepthMap.Data());
    tmp.copyTo(im);

}

/// convert image map to OpenCV, xn::ImageMetaData to cv::Mat
void ximage2opencv(xn::ImageMetaData &xImageMap, cv::Mat &im, int verbose=0)
{
    ////////////////////////////////

    //opencv to convert image to png
    //printf("Converting image to png.\n");
    cv::Mat colorArr[3];
    //cv::Mat colorImage;
    const XnRGB24Pixel* pPixel;
    const XnRGB24Pixel* pImageRow;
    pImageRow = xImageMap.RGB24Data();


    colorArr[0] = cv::Mat(xImageMap.YRes(),xImageMap.XRes(),CV_8U);
    colorArr[1] = cv::Mat(xImageMap.YRes(),xImageMap.XRes(),CV_8U);
    colorArr[2] = cv::Mat(xImageMap.YRes(),xImageMap.XRes(),CV_8U);

    for (int y=0; y<xImageMap.YRes(); y++){
        pPixel = pImageRow;
        uchar* Bptr = colorArr[0].ptr<uchar>(y);
        uchar* Gptr = colorArr[1].ptr<uchar>(y);
        uchar* Rptr = colorArr[2].ptr<uchar>(y);

        for(int x=0;x<xImageMap.XRes();++x , ++pPixel){
                Bptr[x] = pPixel->nBlue;
                Gptr[x] = pPixel->nGreen;
                Rptr[x] = pPixel->nRed;
        }

        pImageRow += xImageMap.XRes();
    }

    cv::merge(colorArr,3,im);
}

void createRGBD(cv::Mat& depth_mat, cv::Mat& color_mat, cv::Mat& dst_rgbd, cv::Mat& dst_depth){

    dst_rgbd = cv::Mat::zeros(depth_mat.rows, depth_mat.cols, CV_8UC3);
    dst_depth = cv::Mat::zeros(depth_mat.rows, depth_mat.cols, CV_16UC1);

    for (int j = 0; j< depth_mat.rows; j ++){
        for(int i = 0; i < depth_mat.cols; i++){
            int depth_value = (int) depth_mat.at<unsigned short>(j,i);
            if (depth_value != 0 && depth_value <= limitz_max && depth_value >= limitz_min)
                if ( limitx_min <= i && limitx_max >=i && limity_min <= j && limity_max >= j ){                   
                        dst_rgbd.at<cv::Vec3b>(j,i)  = color_mat.at<cv::Vec3b>(j,i);
                    dst_depth.at<unsigned short>(j,i)  = depth_mat.at<unsigned short>(j,i);
                }
        }
    }
}

//-------------------MAIN-----------------------//
int main( int argc, char** argv )
{
    int verbose=1;
    //--WORK IN PROGRESS--//
    //To add:
    //option: rgb, depth, ply
    if( argc == 1 ) {
        cout << "Please give an ONI file to open" << std::endl;
        return 1;
    }

    //Setting the outputdir
    //--You have to create them before the execution--//
    std::string rgbdir = "./rgb"; //default
    std::string depthdir = "./depth";
    std::string rgbddir = "./rgbd";
    std::string rgbplydir = "./ply_rgb";

    //....//
    //if (argc>2) //if it's taken, the second parameter is the output directory
      //  outputdir = argv[2];

    // Initial OpenNI Context
    xn::Context xContext;
    
    //Catch the status
    XnStatus status = xContext.Init();
    if (status != XN_STATUS_OK)
    {
        printf("Context Initalization Failed: %s\n", xnGetStatusString(status));
        exit(1);
    }

    xn::Player xPlayer;
    //Opening the oni file
    xContext.OpenFileRecording( argv[1], xPlayer ); 
    xPlayer.SetRepeat( false );

    //Create depth generator
    xn::DepthGenerator xDepthGenerator;
    xDepthGenerator.Create( xContext );
    
    //Create image generator
    xn::ImageGenerator xImageGenerator;
    xImageGenerator.Create( xContext );
    //Set pixel format
    xImageGenerator.SetPixelFormat(XN_PIXEL_FORMAT_RGB24 );

    //Retrieve frames number
    XnUInt32 uFrames;
    xPlayer.GetNumFrames( xDepthGenerator.GetName(), uFrames );
    if (verbose)
        cout << "Total " << uFrames << " frames."<< endl;

    //Depth-RGB alignment-> Only for real time from a connected device
  /*  status = xDepthGenerator.GetAlternativeViewPointCap().SetViewPoint(xImageGenerator);
    if (status != XN_STATUS_OK)
    {
    printf("??? Depth view point change failed: %s\n", xnGetStatusString(status));
    xContext.Shutdown();
    exit(1);
    }
    else
    {
    printf("Depth view point changed: status = %s\n", xnGetStatusString(status));
    }
*/
    //Generating data
    xContext.StartGeneratingAll();
	
    //extraction's loop
    for( unsigned int i = 0; i < 10; ++ i )//DEBUG
    //for( unsigned int i = 0; i < uFrames; ++ i )
    {
        //Update frames (frames)
        xDepthGenerator.WaitAndUpdateData();
        xImageGenerator.WaitAndUpdateData();

        if (verbose && (i%10==0))
            cout << i << "/" << uFrames << endl;

        // get image value
        xn::ImageMetaData xImageMap;
        xImageGenerator.GetMetaData( xImageMap );

        //get rgb mat
        cv::Mat im;
        ximage2opencv(xImageMap, im);

        //Set files for rgbply and depth's
        std::string prgboutfile = rgbplydir + "/" + printfstring("pointcloudrgb%06d.ply", i);
        //writePLY(im, poutfile.c_str());

        //Write image rgb png
        std::string outfile = rgbdir + "/" + printfstring("image%06d.png", i);
        cv::imwrite(outfile.c_str(), im );

        // get depth value
        xn::DepthMetaData xDepthMap;
        xDepthGenerator.GetMetaData( xDepthMap );

        //Write depth to image png
        cv::Mat depth_im;
        xdepth2opencv(xDepthMap, depth_im);
        std::string doutfile = depthdir + "/" +  printfstring("depth%06d.png", i);
        cv::imwrite(doutfile.c_str(), depth_im );


        //-------RGBD-------//
        //Write rgbd
        cv::Mat rgbd, depth_thresh;
        createRGBD(depth_im, im, rgbd, depth_thresh);
        std::string rgbdoutfile = rgbddir + "/" + printfstring("rgbd%06d.png", i);
        cv::imwrite(rgbdoutfile.c_str(), rgbd);
        //-------FINE RGBD-----//

        //Write ply,
        if ((i%50==0))//ply too big so I take only 1 every 50 frames
            //ply pesanti e i frame cambiano lentamente quindi ne prendo uno ogni 100
        {
            //------TO PLY-------//
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudxyzrgb(new pcl::PointCloud<pcl::PointXYZRGB>);

            //cv rgb to pointxyzrgb
            MatToPointXYZRGB(im,depth_im, cloudxyzrgb);

            //save the pointcloud in ply
            saveCloud (prgboutfile, cloudxyzrgb, false, false);

            //-----END TO PLY----//
        }

        //option 2
        //writePLY(im, poutfile.c_str());

        //WORK IN PROGRESS
        //-------GET PCD-------//

       /* pcl::PointCloud<typePoint> cloud;
        get_pcl(im, depth_im, cloud );
        cloud.width = depth_im.cols;
        cloud.height  = depth_im.rows;*/
        //pcl::PointCloud<pcl::PointXYZRGB> biscloudxyzrgb=new pcl::PointCloud<pcl::PointXYZRGB>();


        //pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr biscloudxyzrgb;
       // MatToPointXYZRGB(im,depth_im, *biscloudxyzrgb);
        //std::string pcdoutfile = outputdir + "/" + printfstring("pointcloud%06d.pcd", i);
       // pcl::io::savePCDFile( pcdoutfile, *biscloudxyzrgb, false);


        //------END GET PCD------//

    }

    // stop
    xContext.StopGeneratingAll();

    // release resource
    xContext.Release();

    return 0;
}

//------------------------------------------------
